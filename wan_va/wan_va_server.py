# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import os
import sys
import time
from functools import partial
from typing import Any, Callable, Dict
from PIL import Image
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from einops import rearrange
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.fsdp import shard_model
from distributed.util import _configure_model, init_distributed
from modules.utils import (
    WanVAEStreamingWrapper,
    load_text_encoder,
    load_tokenizer,
    load_transformer,
    load_vae,
)
from utils import (
    FlowMatchScheduler,
    data_seq_to_patch,
    get_mesh_id,
    init_logger,
    logger,
    run_async_server_mode,
    save_async,
)


class VA_Server:

    def __init__(self, job_config):
        self.cache_name = 'pos'
        self.job_config = job_config
        self.save_root = job_config.save_root
        self.dtype = job_config.param_dtype
        self.device = torch.device(f"cuda:{job_config.local_rank}")
        self.enable_offload = getattr(job_config, 'enable_offload', True)  # offload vae & text_encoder to save vram
        self.bench_enable = bool(getattr(job_config, "bench_enable", False))
        self.bench_modules = self._parse_bench_modules(getattr(job_config, "bench_modules", "all"))
        self.bench_step_enable = bool(getattr(job_config, "bench_step_enable", False))
        self.bench_step_max_samples = int(getattr(job_config, "bench_step_max_samples", 0))
        self.bench_kv_enable = bool(getattr(job_config, "bench_kv_enable", False))

        self.debug_io = bool(int(os.getenv("DEBUG_IO", "0"))) or bool(getattr(job_config, "debug_io", False))
        self._debug_io_print_limit = int(os.getenv("DEBUG_IO_PRINT_LIMIT", "200"))
        self._debug_io_print_count = 0

        self.scheduler = FlowMatchScheduler(shift=self.job_config.snr_shift,
                                            sigma_min=0.0,
                                            extra_one_step=True)
        self.action_scheduler = FlowMatchScheduler(
            shift=self.job_config.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.action_scheduler.set_timesteps(1000, training=True)

        self.vae = load_vae(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'vae'),
            torch_dtype=self.dtype,
            torch_device='cpu' if self.enable_offload else self.device,
        )
        self.streaming_vae = WanVAEStreamingWrapper(self.vae)

        self.tokenizer = load_tokenizer(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'tokenizer'), )

        self.text_encoder = load_text_encoder(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'text_encoder'),
            torch_dtype=self.dtype,
            torch_device='cpu' if self.enable_offload else self.device,
        )

        self.transformer = load_transformer(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'transformer'),
            torch_dtype=self.dtype,
            torch_device=self.device,
        )
        shard_fn = shard_model
        self.transformer = _configure_model(model=self.transformer,
                                            shard_fn=shard_fn,
                                            param_dtype=self.dtype,
                                            device=self.device,
                                            eval_mode=True,
                                            )
        if hasattr(self.transformer, "set_kv_bench_enabled"):
            self.transformer.set_kv_bench_enabled(self.bench_kv_enable)
            if self.bench_kv_enable and hasattr(self.transformer, "reset_kv_bench_stats"):
                self.transformer.reset_kv_bench_stats()

        self.env_type = job_config.env_type
        self.streaming_vae_half = None
        if self.env_type == 'robotwin_tshape':
            vae_half = load_vae(
                os.path.join(job_config.wan22_pretrained_model_name_or_path,
                             'vae'),
                torch_dtype=self.dtype,
                torch_device='cpu' if self.enable_offload else self.device,
            )
            self.streaming_vae_half = WanVAEStreamingWrapper(vae_half)

    @staticmethod
    def _parse_bench_modules(raw: Any) -> Dict[str, bool]:
        all_keys = [
            "obs_preprocess",
            "vae_to_latent",
            "text_encoder",
            "kv_prefill",
            "video_diffusion",
            "action_diffusion",
            "action_postprocess",
        ]
        if isinstance(raw, str):
            tokens = [x.strip() for x in raw.split(",") if x.strip()]
        elif isinstance(raw, (list, tuple, set)):
            tokens = [str(x).strip() for x in raw if str(x).strip()]
        else:
            tokens = []
        if (not tokens) or ("all" in tokens):
            return {k: True for k in all_keys}
        enabled = {k: False for k in all_keys}
        for k in tokens:
            if k in enabled:
                enabled[k] = True
        return enabled

    def _bench_enabled(self, key: str) -> bool:
        return self.bench_enable and self.bench_modules.get(key, False)

    def _measure_ms(self, key: str, fn: Callable[[], Any]):
        if not self._bench_enabled(key):
            return fn(), None

        # CPU-side preprocessing should be measured on host wall time.
        cpu_only_keys = {"obs_preprocess"}
        if key in cpu_only_keys:
            t0 = time.perf_counter()
            out = fn()
            return out, (time.perf_counter() - t0) * 1000.0

        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            out = fn()
            e1.record()
            torch.cuda.synchronize(self.device)
            return out, float(e0.elapsed_time(e1))

        t0 = time.perf_counter()
        out = fn()
        return out, (time.perf_counter() - t0) * 1000.0


    def _measure_once_ms(self, fn: Callable[[], Any]) -> tuple[Any, float]:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
            e0 = torch.cuda.Event(enable_timing=True)
            e1 = torch.cuda.Event(enable_timing=True)
            e0.record()
            out = fn()
            e1.record()
            torch.cuda.synchronize(self.device)
            return out, float(e0.elapsed_time(e1))

        t0 = time.perf_counter()
        out = fn()
        return out, (time.perf_counter() - t0) * 1000.0

    def _begin_kv_bench_request(self):
        if not self.bench_kv_enable:
            return
        if hasattr(self.transformer, "reset_kv_bench_stats"):
            self.transformer.reset_kv_bench_stats()

    def _collect_kv_timing(self, extra: Dict[str, Any] | None = None) -> Dict[str, Any]:
        if not self.bench_kv_enable:
            return {}
        out = {}
        if hasattr(self.transformer, "pop_kv_bench_summary"):
            summary = self.transformer.pop_kv_bench_summary()
            if summary:
                out["attn_cache"] = summary
        if extra:
            out.update(extra)
        return out

    def _debug_log(self, msg: str):
        if not self.debug_io:
            return
        if self._debug_io_print_count >= self._debug_io_print_limit:
            return
        logger.info(f"[DEBUG_IO] {msg}")
        self._debug_io_print_count += 1

    def _get_t5_prompt_embeds(
        self,
        prompt=None,
        num_videos_per_prompt=1,
        max_sequence_length=512,
        device=None,
        dtype=None,
    ):
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        text_encoder_device = next(self.text_encoder.parameters()).device
        prompt_embeds = self.text_encoder(text_input_ids.to(text_encoder_device),
                                          mask.to(text_encoder_device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack([
            torch.cat(
                [u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
                                    dim=0)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt,
                                           seq_len, -1)

        return prompt_embeds.to(device)

    def encode_prompt(
        self,
        prompt,
        negative_prompt=None,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        max_sequence_length=226,
        device=None,
        dtype=None,
    ):
        r"""
        TODO
        """
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(
                negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(
                    negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}.")
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`.")

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        return prompt_embeds, negative_prompt_embeds

    def normalize_latents(
        self,
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1,
                                         1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1,
                                       1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * latents_std).to(latents)
        return latents

    def preprocess_action(self, action):
        action_model_input = torch.from_numpy(action)
        CA, FA, HA = action_model_input.shape  # C, F, H
        action_model_input_paded = F.pad(action_model_input,
                                         [0, 0, 0, 0, 0, 1],
                                         mode='constant',
                                         value=0)

        action_model_input = action_model_input_paded[
            self.job_config.inverse_used_action_channel_ids]

        if self.action_norm_method == 'quantiles':
            action_model_input = (action_model_input - self.actions_q01) / (
                self.actions_q99 - self.actions_q01 + 1e-6) * 2. - 1.
        else:
            raise NotImplementedError
        return action_model_input.unsqueeze(0).unsqueeze(-1)  # B, C, F, H, W

    def postprocess_action(self, action):
        action = action.cpu()  # B, C, F, H, W

        action = action[0, ..., 0]  #C, F, H
        if self.action_norm_method == 'quantiles':
            action = (action + 1) / 2 * (self.actions_q99 - self.actions_q01 +
                                         1e-6) + self.actions_q01
        else:
            raise NotImplementedError
        action = action.squeeze(0).detach().cpu().numpy()
        return action[self.job_config.used_action_channel_ids]
    
    def _repeat_input_for_cfg(self, input_dict):
        if self.use_cfg:
            input_dict['noisy_latents'] = input_dict['noisy_latents'].repeat(2, 1, 1, 1, 1)
            input_dict['text_emb'] = torch.cat([self.prompt_embeds.to(self.dtype).clone(), self.negative_prompt_embeds.to(self.dtype).clone()], dim=0)
            input_dict['grid_id'] = input_dict['grid_id'][None].repeat(2, 1, 1)
            input_dict['timesteps'] = input_dict['timesteps'][None].repeat(2, 1)
        else:
            input_dict['grid_id'] = input_dict['grid_id'][None]
            input_dict['timesteps'] = input_dict['timesteps'][None]
        return input_dict

    def _prepare_latent_input(self,
                              latent_model_input,
                              action_model_input,
                              latent_t=0,
                              action_t=0,
                              latent_cond=None,
                              action_cond=None,
                              frame_st_id=0,
                              patch_size=(1, 2, 2)):
        logger.info(f"FRAME START ID: {frame_st_id}")
        input_dict = dict()
        if latent_model_input is not None:
            input_dict['latent_res_lst'] = {
                'noisy_latents':
                latent_model_input,
                'timesteps':
                torch.ones([latent_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * latent_t,
                'grid_id':
                get_mesh_id(latent_model_input.shape[-3] // patch_size[0],
                            latent_model_input.shape[-2] // patch_size[1],
                            latent_model_input.shape[-1] // patch_size[2], 0,
                            1, frame_st_id).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }
            if latent_cond is not None:
                input_dict['latent_res_lst'][
                    'noisy_latents'][:, :, 0:1] = latent_cond[:, :, 0:1]
                input_dict['latent_res_lst']['timesteps'][0:1] *= 0

        if action_model_input is not None:
            input_dict['action_res_lst'] = {
                'noisy_latents':
                action_model_input,
                'timesteps':
                torch.ones([action_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * action_t,
                'grid_id':
                get_mesh_id(action_model_input.shape[-3],
                            action_model_input.shape[-2],
                            action_model_input.shape[-1],
                            1,
                            1,
                            frame_st_id,
                            action=True).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }

            if action_cond is not None:
                input_dict['action_res_lst'][
                    'noisy_latents'][:, :, 0:1] = action_cond[:, :, 0:1]
                input_dict['action_res_lst']['timesteps'][0:1] *= 0
            input_dict['action_res_lst']['noisy_latents'][:, ~self.
                                                          action_mask] *= 0
        return input_dict

    def _obs_build_normalized_video_tensors(self, obs):
        """CPU 侧：numpy → resize → 归一化到 [-1,1]。供 `_encode_obs` 与推理耗时拆分测试。"""
        images = obs['obs']
        if not isinstance(images, list):
            images = [images]
        if self.debug_io and len(images) > 0:
            frame0 = images[0]
            cam_shapes = {k: tuple(v.shape) for k, v in frame0.items() if hasattr(v, 'shape')}
            self._debug_log(f"obs frames={len(images)} cam_shapes={cam_shapes}")
        if len(images) < 1:
            return None
        videos = []
        for k_i, k in enumerate(self.job_config.obs_cam_keys):
            if self.env_type == 'robotwin_tshape':
                if k_i == 0:  # camera high
                    height_i, width_i = self.height, self.width
                else:
                    height_i, width_i = self.height // 2, self.width // 2
            else:
                height_i, width_i = self.height, self.width

            history_video_k = torch.from_numpy(
                np.stack([each[k]
                          for each in images])).float().permute(3, 0, 1, 2)
            history_video_k = F.interpolate(history_video_k,
                                            size=(height_i, width_i),
                                            mode='bilinear',
                                            align_corners=False).unsqueeze(0)
            videos.append(history_video_k)

        if self.env_type == 'robotwin_tshape':
            videos_high = videos[0] / 255.0 * 2.0 - 1.0
            videos_left_and_right = torch.cat(videos[1:],
                                              dim=0) / 255.0 * 2.0 - 1.0
            return {
                'mode': 'tshape',
                'videos_high': videos_high,
                'videos_left_and_right': videos_left_and_right,
            }
        videos = torch.cat(videos, dim=0) / 255.0 * 2.0 - 1.0
        return {'mode': 'plain', 'videos': videos}

    def _vae_encode_normalized_to_latent(self, blob):
        """VAE 编码 + latent 归一化；输入来自 `_obs_build_normalized_video_tensors`。"""
        if blob is None:
            return None
        vae_device = next(self.streaming_vae.vae.parameters()).device
        if blob['mode'] == 'tshape':
            enc_out_high = self.streaming_vae.encode_chunk(
                blob['videos_high'].to(vae_device).to(self.dtype))
            enc_out_left_and_right = self.streaming_vae_half.encode_chunk(
                blob['videos_left_and_right'].to(vae_device).to(self.dtype))
            enc_out = torch.cat([
                torch.cat(enc_out_left_and_right.split(1, dim=0), dim=-1),
                enc_out_high
            ],
                                dim=-2)
        else:
            enc_out = self.streaming_vae.encode_chunk(
                blob['videos'].to(vae_device).to(self.dtype))

        mu, logvar = torch.chunk(enc_out, 2, dim=1)
        latents_mean = torch.tensor(self.vae.config.latents_mean).to(mu.device)
        latents_std = torch.tensor(self.vae.config.latents_std).to(mu.device)
        mu_norm = self.normalize_latents(mu, latents_mean, 1.0 / latents_std)
        video_latent = torch.cat(mu_norm.split(1, dim=0), dim=-1)
        video_latent = video_latent.to(self.device)
        if self.debug_io:
            self._debug_log(
                f"latent shape={tuple(video_latent.shape)} dtype={video_latent.dtype} "
                f"min={float(video_latent.min().item()):.4f} max={float(video_latent.max().item()):.4f} "
                f"mean={float(video_latent.mean().item()):.4f}")
        return video_latent

    def _encode_obs(self, obs):
        blob = self._obs_build_normalized_video_tensors(obs)
        return self._vae_encode_normalized_to_latent(blob)

    def _reset(self, prompt=None):
        logger.info('Reset.')
        profile = {}
        self.use_cfg = (self.job_config.guidance_scale > 1) or (self.job_config.action_guidance_scale > 1)
        #### Reset all parameters
        self.frame_st_id = 0
        self.init_latent = None
        #### clean vae and transformer cache
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()

        self.action_per_frame = self.job_config.action_per_frame
        self.height, self.width = self.job_config.height, self.job_config.width

        if self.env_type == 'robotwin_tshape':
            self.latent_height, self.latent_width = (
                (self.height // 16) * 3) // 2, self.width // 16
            self.streaming_vae_half.clear_cache()
        else:
            self.latent_height, self.latent_width = self.height // 16, self.width // 16 * len(
                self.job_config.obs_cam_keys)

        patch_size = self.job_config.patch_size
        latent_token_per_chunk = (self.job_config.frame_chunk_size *
                                  self.latent_height * self.latent_width) // (
                                      patch_size[0] * patch_size[1] *
                                      patch_size[2])
        action_token_per_chunk = self.job_config.frame_chunk_size * self.action_per_frame
        self.transformer.create_empty_cache(self.cache_name,
                                            self.job_config.attn_window,
                                            latent_token_per_chunk,
                                            action_token_per_chunk,
                                            dtype=self.dtype,
                                            device=self.device,
                                            batch_size = 2 if self.use_cfg else 1
                                            )

        self.action_mask = torch.zeros([self.job_config.action_dim]).bool()
        self.action_mask[self.job_config.used_action_channel_ids] = True

        self.actions_q01 = torch.tensor(self.job_config.norm_stat['q01'],
                                        dtype=torch.float32).reshape(-1, 1, 1)
        self.actions_q99 = torch.tensor(self.job_config.norm_stat['q99'],
                                        dtype=torch.float32).reshape(-1, 1, 1)
        self.action_norm_method = self.job_config.action_norm_method

        ##### get prompt
        if prompt is None:
            self.prompt_embeds = self.negative_prompt_embeds = None
        else:
            (self.prompt_embeds, self.negative_prompt_embeds), text_ms = self._measure_ms(
                "text_encoder",
                lambda: self.encode_prompt(
                    prompt=prompt,
                    negative_prompt=None,
                    do_classifier_free_guidance=self.job_config.guidance_scale > 1,
                    num_videos_per_prompt=1,
                    prompt_embeds=None,
                    negative_prompt_embeds=None,
                    max_sequence_length=512,
                    device=self.device,
                    dtype=self.dtype,
                ),
            )
            if text_ms is not None:
                profile["text_encoder"] = text_ms

        self.exp_name = f"{prompt}_{time.strftime('%Y%m%d_%H%M%S')}" if prompt else "default"
        self.exp_save_root = os.path.join(self.save_root, 'real', self.exp_name)
        os.makedirs(self.exp_save_root, exist_ok=True)
        torch.cuda.empty_cache()
        return profile

    def _infer_run_video_phase(self,
                               latents,
                               init_latent,
                               frame_st_id,
                               timesteps,
                               video_step,
                               frame_chunk_size,
                               quiet=False,
                               step_times=None):
        ts_iter = timesteps if quiet else tqdm(timesteps)
        for i, t in enumerate(ts_iter):
            def _run_one_video_step(current_latents):
                last_step = i == len(timesteps) - 1
                latent_cond = init_latent[:, :, 0:1].to(
                    self.dtype) if frame_st_id == 0 else None
                input_dict = self._prepare_latent_input(
                    current_latents,
                    None,
                    t,
                    t,
                    latent_cond,
                    None,
                    frame_st_id=frame_st_id)

                video_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict['latent_res_lst']),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=False,
                    profile_stage="video_diffusion")

                if not last_step or video_step != -1:
                    video_noise_pred = data_seq_to_patch(
                        self.job_config.patch_size, video_noise_pred,
                        frame_chunk_size, self.latent_height,
                        self.latent_width, batch_size=2 if self.use_cfg else 1)
                    if self.job_config.guidance_scale > 1:
                        video_noise_pred = video_noise_pred[1:] + self.job_config.guidance_scale * (video_noise_pred[:1] - video_noise_pred[1:])
                    else:
                        video_noise_pred = video_noise_pred[:1]
                    current_latents = self.scheduler.step(video_noise_pred,
                                                          t,
                                                          current_latents,
                                                          return_dict=False)

                current_latents[:, :, 0:1] = latent_cond if frame_st_id == 0 else current_latents[:, :, 0:1]
                return current_latents

            if step_times is not None:
                latents, step_ms = self._measure_once_ms(lambda: _run_one_video_step(latents))
                step_times.append(step_ms)
            else:
                latents = _run_one_video_step(latents)
        return latents

    def _infer_run_action_phase(self, actions, frame_st_id, action_timesteps,
                                frame_chunk_size, quiet=False,
                                step_times=None):
        ts_iter = action_timesteps if quiet else tqdm(action_timesteps)
        for i, t in enumerate(ts_iter):
            def _run_one_action_step(current_actions):
                last_step = i == len(action_timesteps) - 1
                action_cond = torch.zeros(
                    [
                        1, self.job_config.action_dim, 1,
                        self.action_per_frame, 1
                    ],
                    device=self.device,
                    dtype=self.dtype) if frame_st_id == 0 else None

                input_dict = self._prepare_latent_input(
                    None,
                    current_actions,
                    t,
                    t,
                    None,
                    action_cond,
                    frame_st_id=frame_st_id)
                # breakpoint()
                action_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict['action_res_lst']),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=True,
                    profile_stage="action_diffusion")

                if not last_step:
                    action_noise_pred = rearrange(action_noise_pred,
                                                  'b (f n) c -> b c f n 1',
                                                  f=frame_chunk_size)
                    if self.job_config.action_guidance_scale > 1:
                        action_noise_pred = action_noise_pred[1:] + self.job_config.action_guidance_scale * (action_noise_pred[:1] - action_noise_pred[1:])
                    else:
                        action_noise_pred = action_noise_pred[:1]
                    current_actions = self.action_scheduler.step(action_noise_pred,
                                                                 t,
                                                                 current_actions,
                                                                 return_dict=False)

                current_actions[:, :, 0:1] = action_cond if frame_st_id == 0 else current_actions[:, :, 0:1]
                return current_actions

            if step_times is not None:
                actions, step_ms = self._measure_once_ms(lambda: _run_one_action_step(actions))
                step_times.append(step_ms)
            else:
                actions = _run_one_action_step(actions)
        return actions

    def _infer(self, obs, frame_st_id=0):
        profile = {}
        frame_chunk_size = self.job_config.frame_chunk_size
        init_latent = None
        if frame_st_id == 0:
            blob, obs_ms = self._measure_ms(
                "obs_preprocess",
                lambda: self._obs_build_normalized_video_tensors(obs),
            )
            if obs_ms is not None:
                profile["obs_preprocess"] = obs_ms
            init_latent, vae_ms = self._measure_ms(
                "vae_to_latent",
                lambda: self._vae_encode_normalized_to_latent(blob),
            )
            if vae_ms is not None:
                profile["vae_to_latent"] = vae_ms
            self.init_latent = init_latent

        latents = torch.randn(1,
                              48,
                              frame_chunk_size,
                              self.latent_height,
                              self.latent_width,
                              device=self.device,
                              dtype=self.dtype)
        actions = torch.randn(1,
                              self.job_config.action_dim,
                              frame_chunk_size,
                              self.action_per_frame,
                              1,
                              device=self.device,
                              dtype=self.dtype)

        video_inference_step = self.job_config.num_inference_steps
        action_inference_step = self.job_config.action_num_inference_steps
        video_step = self.job_config.video_exec_step

        self.scheduler.set_timesteps(video_inference_step)
        self.action_scheduler.set_timesteps(action_inference_step)
        timesteps = self.scheduler.timesteps
        action_timesteps = self.action_scheduler.timesteps

        timesteps = F.pad(timesteps, (0, 1), mode='constant', value=0)

        if video_step != -1:
            timesteps = timesteps[:video_step]

        action_timesteps = F.pad(
            action_timesteps,
            (0,
             1),  # pad 1 element at the end (right side) of the last dimension
            mode='constant',
            value=0)

        video_step_ms_list = [] if self.bench_step_enable else None
        action_step_ms_list = [] if self.bench_step_enable else None

        with (
                torch.no_grad(),
        ):
            latents, video_ms = self._measure_ms(
                "video_diffusion",
                lambda: self._infer_run_video_phase(
                    latents,
                    init_latent,
                    frame_st_id,
                    timesteps,
                    video_step,
                    frame_chunk_size,
                    quiet=False,
                    step_times=video_step_ms_list),
            )
            if video_ms is not None:
                profile["video_diffusion"] = video_ms

            actions, action_ms = self._measure_ms(
                "action_diffusion",
                lambda: self._infer_run_action_phase(
                    actions,
                    frame_st_id,
                    action_timesteps,
                    frame_chunk_size,
                    quiet=False,
                    step_times=action_step_ms_list),
            )
            if action_ms is not None:
                profile["action_diffusion"] = action_ms

        actions[:, ~self.action_mask] *= 0

        if not getattr(self, '_benchmark_skip_async_io', False):
            save_async(latents, os.path.join(self.exp_save_root, f'latents_{frame_st_id}.pt'))
            save_async(actions, os.path.join(self.exp_save_root, f'actions_{frame_st_id}.pt'))

        actions, post_ms = self._measure_ms(
            "action_postprocess",
            lambda: self.postprocess_action(actions),
        )
        if post_ms is not None:
            profile["action_postprocess"] = post_ms
        if self.debug_io:
            self._debug_log(
                f"action_out shape={tuple(actions.shape)} min={float(actions.min()):.4f} "
                f"max={float(actions.max()):.4f} mean={float(actions.mean()):.4f}")
        step_profile = {}
        if video_step_ms_list is not None:
            if self.bench_step_max_samples > 0:
                video_step_ms_list = video_step_ms_list[:self.bench_step_max_samples]
            step_profile["video_denoise_step_ms"] = video_step_ms_list
        if action_step_ms_list is not None:
            if self.bench_step_max_samples > 0:
                action_step_ms_list = action_step_ms_list[:self.bench_step_max_samples]
            step_profile["action_denoise_step_ms"] = action_step_ms_list

        torch.cuda.empty_cache()
        return actions, latents, profile, step_profile

    def _compute_kv_cache(self, obs):
        profile = {}
        kv_extra = {}
        ### optional async save obs for debug
        self.transformer.clear_pred_cache(self.cache_name)
        if not getattr(self, '_benchmark_skip_async_io', False):
            save_async(obs['obs'], os.path.join(self.exp_save_root, f'obs_data_{self.frame_st_id}.pt'))

        blob, obs_ms = self._measure_ms(
            "obs_preprocess",
            lambda: self._obs_build_normalized_video_tensors(obs),
        )
        if obs_ms is not None:
            profile["obs_preprocess"] = obs_ms

        latent_model_input, vae_ms = self._measure_ms(
            "vae_to_latent",
            lambda: self._vae_encode_normalized_to_latent(blob),
        )
        if vae_ms is not None:
            profile["vae_to_latent"] = vae_ms

        if self.frame_st_id == 0:
            latent_model_input = torch.cat(
                [self.init_latent, latent_model_input],
                dim=2) if latent_model_input is not None else self.init_latent

        action_model_input = self.preprocess_action(obs['state'])
        action_model_input = action_model_input.to(latent_model_input)
        logger.info(
            f"get KV cache obs: {latent_model_input.shape} {action_model_input.shape}"
        )
        if self.debug_io:
            state_shape = tuple(obs['state'].shape) if hasattr(obs.get('state', None), 'shape') else None
            self._debug_log(
                f"kv state_shape={state_shape} latent_in={tuple(latent_model_input.shape)} action_in={tuple(action_model_input.shape)}")
        if self.bench_kv_enable:
            input_dict, kv_prepare_ms = self._measure_once_ms(
                lambda: self._prepare_latent_input(
                    latent_model_input,
                    action_model_input,
                    frame_st_id=self.frame_st_id,
                )
            )
            kv_extra["kv_prepare_input_ms"] = float(kv_prepare_ms)
        else:
            input_dict = self._prepare_latent_input(latent_model_input,
                                                    action_model_input,
                                                    frame_st_id=self.frame_st_id)

        def _run_kv_prefill_latent():
            with torch.no_grad():
                return self.transformer(
                    self._repeat_input_for_cfg(input_dict['latent_res_lst']),
                    update_cache=2,
                    cache_name=self.cache_name,
                    action_mode=False,
                    profile_stage="kv_prefill_latent",
                )

        def _run_kv_prefill_action():
            with torch.no_grad():
                return self.transformer(
                    self._repeat_input_for_cfg(input_dict['action_res_lst']),
                    update_cache=2,
                    cache_name=self.cache_name,
                    action_mode=True,
                    profile_stage="kv_prefill_action",
                )

        kv_latent_ms = kv_action_ms = None
        if self.bench_kv_enable:
            _, kv_latent_ms = self._measure_once_ms(_run_kv_prefill_latent)
            _, kv_action_ms = self._measure_once_ms(_run_kv_prefill_action)
            kv_extra["kv_prefill_latent_forward_ms"] = float(kv_latent_ms)
            kv_extra["kv_prefill_action_forward_ms"] = float(kv_action_ms)
            kv_extra["kv_prefill_total_forward_ms"] = float(kv_latent_ms + kv_action_ms)
            if self._bench_enabled("kv_prefill"):
                profile["kv_prefill"] = float(kv_latent_ms + kv_action_ms)
        else:
            if self._bench_enabled("kv_prefill"):
                _, kv_ms = self._measure_ms(
                    "kv_prefill",
                    lambda: (_run_kv_prefill_latent(), _run_kv_prefill_action()),
                )
                if kv_ms is not None:
                    profile["kv_prefill"] = kv_ms
            else:
                _run_kv_prefill_latent()
                _run_kv_prefill_action()

        torch.cuda.empty_cache()
        self.frame_st_id += latent_model_input.shape[2]
        return profile, kv_extra

    @torch.no_grad()
    def infer(self, obs):
        reset = obs.get('reset', False)
        prompt = obs.get('prompt', None)
        compute_kv_cache = obs.get('compute_kv_cache', False)

        if reset:
            logger.info(f"******************* Reset server ******************")
            if self.bench_kv_enable and hasattr(self.transformer, "reset_kv_bench_stats"):
                self.transformer.reset_kv_bench_stats()
            profile = self._reset(prompt=prompt)
            return dict(module_timing=profile) if profile else dict()
        elif compute_kv_cache:
            logger.info(
                f"################# Compute KV Cache #################")
            self._begin_kv_bench_request()
            profile, kv_extra = self._compute_kv_cache(obs)
            ret = dict()
            if profile:
                ret["module_timing"] = profile
            kv_timing = self._collect_kv_timing(kv_extra)
            if kv_timing:
                ret["kv_timing"] = kv_timing
            return ret
        else:
            logger.info(f"################# Infer One Chunk #################")
            self._begin_kv_bench_request()
            action, _, profile, step_profile = self._infer(obs, frame_st_id=self.frame_st_id)
            ret = dict(action=action)
            if profile:
                ret["module_timing"] = profile
            if step_profile:
                ret["step_timing"] = step_profile
            kv_timing = self._collect_kv_timing()
            if kv_timing:
                ret["kv_timing"] = kv_timing
            return ret
    
    def decode_one_video(self, latents, output_type):
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)
        return video
    
    def load_init_obs(self):
        imf_dict = {v: np.array(Image.open(os.path.join(self.job_config.input_img_path, f"{v}.png")).convert("RGB")) for v in self.job_config.obs_cam_keys}
        init_obs = {}
        init_obs['obs'] = [imf_dict]
        return init_obs
    
    @torch.no_grad()
    def generate(self):
        self.video_processor = VideoProcessor(vae_scale_factor=1)
        self._reset(self.job_config.prompt)
        init_obs = self.load_init_obs()
        pred_latent_lst = []
        pred_action_lst = []
        for chunk_id in range(self.job_config.num_chunks_to_infer):
            actions, latents, _, _ = self._infer(init_obs, frame_st_id=(chunk_id * self.job_config.frame_chunk_size))
            actions = torch.from_numpy(actions)
            pred_latent_lst.append(latents)
            pred_action_lst.append(actions)
        pred_latent = torch.cat(pred_latent_lst, dim=2)
        pred_action = torch.cat(pred_action_lst, dim=1).flatten(1)
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()
        if self.streaming_vae_half:
            self.streaming_vae_half.clear_cache()
        del self.transformer
        del self.streaming_vae_half
        del self.text_encoder
        torch.cuda.empty_cache()
        
        # Move VAE to GPU for decoding
        if self.enable_offload:
            self.vae = self.vae.to(self.device).to(self.dtype)
        
        decoded_video = self.decode_one_video(pred_latent, 'np')[0]
        export_to_video(decoded_video, os.path.join(self.save_root, "demo.mp4"), fps=10)

def run(args):    
    
    config = VA_CONFIGS[args.config_name]
    port = config.port if args.port is None else args.port
    if args.save_root is not None:
        config.save_root = args.save_root
    if args.attn_window is not None:
        config.attn_window = int(args.attn_window)
    config.bench_enable = args.bench_enable
    config.bench_modules = args.bench_modules
    config.bench_step_enable = args.bench_step_enable
    config.bench_step_max_samples = args.bench_step_max_samples
    config.bench_kv_enable = args.bench_kv_enable
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    model = VA_Server(config)
    if config.infer_mode == 'i2va':
        logger.info(f"******************************USE I2AV mode******************************")
        model.generate()
    elif config.infer_mode == 'server':
        logger.info(f"******************************USE Server mode******************************")
        run_async_server_mode(model, local_rank, config.host, port)
    else:
        raise ValueError(f"Unknown infer mode: {config.infer_mode}")

def main():
    """
    TODO
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name",
        type=str,
        required=False,
        default='robotwin',
        help="config name.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help='(start) port'
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default=None,
        help='save root'
    )
    parser.add_argument(
        "--bench-enable",
        action="store_true",
        help="enable per-module latency timing on real server requests",
    )
    parser.add_argument(
        "--bench-modules",
        type=str,
        default="all",
        help="comma-separated modules: obs_preprocess,vae_to_latent,text_encoder,kv_prefill,video_diffusion,action_diffusion,action_postprocess or all",
    )
    parser.add_argument(
        "--bench-step-enable",
        action="store_true",
        help="enable per-denoise-step latency timing (video/action) on infer requests",
    )
    parser.add_argument(
        "--bench-step-max-samples",
        type=int,
        default=0,
        help="optional cap for returned per-step latency samples per infer call; 0 means no cap",
    )
    parser.add_argument(
        "--bench-kv-enable",
        action="store_true",
        help="enable fine-grained KV cache timing/counter stats (store/read/restore, tokens, evictions)",
    )
    parser.add_argument(
        "--attn-window",
        type=int,
        default=None,
        help="override config attn_window for cache-window sensitivity experiments",
    )
    args = parser.parse_args()
    run(args)
    logger.info("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    init_logger()
    main()
