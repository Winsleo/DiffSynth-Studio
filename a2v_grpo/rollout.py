"""Flow-GRPO rollout on the DiffSynth A2V (VACE) pipeline.

Mirrors `WanVideoPipeline.__call__` (`diffsynth/pipelines/wan_video.py:270-359`) but splits it
into the three pieces an RL loop needs:

    prepare_a2v_inputs      run the preprocessing units ONCE  (wan_video.py:306-307)
    a2v_rollout_with_logprob   own the denoising loop         (wan_video.py:312-336)
    a2v_compute_log_prob    re-score one stored step at update time

Upstream `a2v/` and `diffsynth/` are left untouched on purpose; everything here is additive.

## The one thing you must not break

`a2v_compute_log_prob` has to be fed the *same* conditioning the rollout used
(`vace_context`, and for the 5B config `first_frame_latents`). If it is not, the rollout
samples under one distribution while the update scores under another, the importance ratio is
garbage, and **nothing raises** -- the run just fails to learn. `tests/test_l2_control_wired.py`
asserts that perturbing `vace_context` changes the recomputed log-probability.

## Config-specific behaviour that is easy to miss

* **5B / Wan2.2-TI2V (`first_frame_mode="ti2v_fused"`, what the production A2V model uses)**
  `a2v/infer_a2v.py:181` passes `input_image`, so `WanVideoUnit_ImageEmbedderFused`
  (`wan_video.py:512`) sets `first_frame_latents` and the loop clamps latent frame 0 after
  every step (`wan_video.py:335-336`). That frame is not sampled, so it is masked out of the
  log-probability -- see `logprob_mask` in `a2v_grpo/sde.py`.
* **1.3B / Wan2.1-VACE (`first_frame_mode="vace_reference"`)**
  `vace_reference_image` is passed instead; the reference latent frames sit at the *front* of
  the latent and are stripped before decoding (`wan_video.py:339-344`). No frame-0 clamp.

Both are handled; which one applies is read off the prepared inputs, not hardcoded.
"""

from __future__ import annotations

import contextlib
import inspect
import os
from typing import Any, Optional, Sequence

import torch

from .sde import sde_step_with_logprob

# `__call__` routes these two keys to the positive/negative branches; everything else it
# recognises goes into `inputs_shared`. Deriving the rest from the signature (instead of
# copying the 30-key literal at wan_video.py:284-305) means new upstream params land in
# `inputs_shared` automatically rather than being silently dropped.
_POSI_KEYS = ("prompt", "vap_prompt")
_NEGA_KEYS = ("negative_prompt", "negative_vap_prompt")
_BOTH_KEYS = ("tea_cache_l1_thresh", "tea_cache_model_id", "num_inference_steps")


@contextlib.contextmanager
def disable_lora(*models):
    """Temporarily switch off every injected LoRA adapter -- the KL reference policy.

    DiffSynth injects adapters with `peft.inject_adapter_in_model` into a bare `nn.Module`
    (`diffsynth/diffusion/training_module.py:100`), so there is **no**
    `PeftModel.disable_adapter()` to call. We toggle `BaseTunerLayer.enable_adapters`
    directly (verified present in peft 0.19.1). This avoids keeping a second ~10 GB copy of
    the 5B weights just to score the reference policy.
    """
    from peft.tuners.tuners_utils import BaseTunerLayer

    touched = []
    for model in models:
        if model is None:
            continue
        for module in model.modules():
            if isinstance(module, BaseTunerLayer):
                touched.append(module)
    try:
        for m in touched:
            m.enable_adapters(False)
        yield
    finally:
        for m in touched:
            m.enable_adapters(True)


def _split_inputs(pipe, overrides: dict) -> tuple[dict, dict, dict]:
    """Build (inputs_shared, inputs_posi, inputs_nega) exactly as `__call__` would."""
    sig = inspect.signature(pipe.__call__)
    defaults = {
        name: p.default
        for name, p in sig.parameters.items()
        if p.default is not inspect.Parameter.empty
    }
    unknown = set(overrides) - set(sig.parameters)
    if unknown:
        raise TypeError(f"unknown pipeline kwargs: {sorted(unknown)}")
    merged = {**defaults, **overrides}

    inputs_posi = {k: merged[k] for k in _POSI_KEYS if k in merged}
    inputs_nega = {k: merged[k] for k in _NEGA_KEYS if k in merged}
    for k in _BOTH_KEYS:
        if k in merged:
            inputs_posi[k] = merged[k]
            inputs_nega[k] = merged[k]

    # `_BOTH_KEYS` live ONLY on the posi/nega side upstream (wan_video.py:274-305 keeps them
    # out of inputs_shared). Duplicating them into inputs_shared makes `model_fn(**shared,
    # **side)` raise "got multiple values for keyword argument 'tea_cache_l1_thresh'".
    skip = set(_POSI_KEYS) | set(_NEGA_KEYS) | set(_BOTH_KEYS) | {"progress_bar_cmd", "output_type"}
    inputs_shared = {k: v for k, v in merged.items() if k not in skip}
    return inputs_shared, inputs_posi, inputs_nega, merged


def prepare_a2v_inputs(
    pipe,
    *,
    prompt: str,
    negative_prompt: str = "",
    vace_video: Optional[Sequence] = None,
    vace_reference_image=None,
    input_image=None,
    height: int = 480,
    width: int = 832,
    num_frames: int = 121,
    num_inference_steps: int = 50,
    cfg_scale: float = 5.0,
    seed: Optional[int] = None,
    sigma_shift: Optional[float] = None,
    denoising_strength: float = 1.0,
    **extra,
) -> dict:
    """Run the preprocessing units once; return everything the loop and the update need.

    The returned dict carries `vace_context` (DiffSynth's control latent: `cat(video_latents,
    mask_latents)`, 352 channels for the 5B config -- see `WanVideoUnit_VACE`,
    `wan_video.py:649-707`) and, when applicable, `first_frame_latents`. Those are the tensors
    that must travel with each RL sample.
    """
    pipe.scheduler.set_timesteps(
        num_inference_steps, denoising_strength=denoising_strength, shift=sigma_shift
    )

    overrides = dict(
        prompt=prompt,
        negative_prompt=negative_prompt,
        vace_video=vace_video,
        vace_reference_image=vace_reference_image,
        input_image=input_image,
        height=height,
        width=width,
        num_frames=num_frames,
        num_inference_steps=num_inference_steps,
        cfg_scale=cfg_scale,
        seed=seed,
        sigma_shift=sigma_shift,
        denoising_strength=denoising_strength,
        **extra,
    )
    inputs_shared, inputs_posi, inputs_nega, merged = _split_inputs(pipe, overrides)

    for unit in pipe.units:
        inputs_shared, inputs_posi, inputs_nega = pipe.unit_runner(
            unit, pipe, inputs_shared, inputs_posi, inputs_nega
        )

    if inputs_shared.get("vace_context") is None:
        raise ValueError(
            "vace_context is None after preprocessing -- the VACE unit did not fire. "
            "Check that `vace_video` was passed and that the pipeline has a `vace` model."
        )
    return {
        "inputs_shared": inputs_shared,
        "inputs_posi": inputs_posi,
        "inputs_nega": inputs_nega,
        "merged": merged,
    }


def build_logprob_mask(latents: torch.Tensor, inputs_shared: dict) -> Optional[torch.Tensor]:
    """0 on latent frames that the loop deterministically clamps, 1 elsewhere.

    Only the `ti2v_fused` path clamps (frame 0). Returns None when nothing is clamped so the
    caller keeps the cheaper unmasked average.
    """
    if "first_frame_latents" not in inputs_shared:
        return None
    mask = torch.ones_like(latents)
    mask[:, :, 0:1] = 0.0
    return mask


def _forward(pipe, models: dict, inputs_shared: dict, inputs_side: dict, timestep) -> torch.Tensor:
    # Fail loudly on key collisions: `model_fn(**a, **b)` otherwise raises a bare
    # "got multiple values for keyword argument X" that does not say which dict owns X.
    clash = (set(inputs_shared) & set(inputs_side)) | (set(models) & set(inputs_shared))
    if clash:
        raise TypeError(
            f"input dicts collide on {sorted(clash)}; upstream keeps posi/nega-only keys "
            f"out of inputs_shared (see _split_inputs)."
        )
    return pipe.model_fn(**models, **inputs_shared, **inputs_side, timestep=timestep)


def _predict_velocity(
    pipe, models, inputs_shared, inputs_posi, inputs_nega, timestep, cfg_scale, cfg_merge
) -> torch.Tensor:
    """One CFG-combined velocity prediction; mirrors wan_video.py:323-331."""
    noise_pred_posi = _forward(pipe, models, inputs_shared, inputs_posi, timestep)
    if cfg_scale == 1.0:
        return noise_pred_posi
    if cfg_merge:
        noise_pred_posi, noise_pred_nega = noise_pred_posi.chunk(2, dim=0)
    else:
        noise_pred_nega = _forward(pipe, models, inputs_shared, inputs_nega, timestep)
    return noise_pred_nega + cfg_scale * (noise_pred_posi - noise_pred_nega)


@torch.no_grad()
def a2v_rollout_with_logprob(
    pipe,
    prepared: dict,
    *,
    noise_level: float = 0.7,
    deterministic: bool = False,
    generator: Optional[torch.Generator] = None,
    decode: bool = True,
    switch_DiT_boundary: float = 0.875,
    progress_bar_cmd=lambda x: x,
):
    """Denoise with the SDE, collecting the per-step latents and log-probabilities.

    Returns a dict with:
        latents     (B, T+1, C, F, H, W) -- trajectory including the initial noise
        log_probs   (B, T)
        timesteps   (T,)
        video       decoded frames (when `decode`), else None
        logprob_mask / vace_context / first_frame_latents -- what the update must reuse
    """
    inputs_shared = dict(prepared["inputs_shared"])
    inputs_posi, inputs_nega = prepared["inputs_posi"], prepared["inputs_nega"]
    merged = prepared["merged"]
    cfg_scale = inputs_shared.get("cfg_scale", 1.0)
    cfg_merge = inputs_shared.get("cfg_merge", False)

    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

    latents = inputs_shared["latents"]
    # Prefer a mask the caller cached: the update-time re-score MUST use the same one, or a
    # systematic offset leaks into the importance ratio.
    logprob_mask = prepared.get("logprob_mask")
    if logprob_mask is None:
        logprob_mask = build_logprob_mask(latents, inputs_shared)

    all_latents = [latents.detach().cpu()]
    all_log_probs = []
    timesteps = pipe.scheduler.timesteps

    for progress_id, timestep in enumerate(progress_bar_cmd(timesteps)):
        # dit2 switch (Wan2.2 MoE bases); no-op for the single-DiT 5B TI2V
        if (
            timestep.item() < switch_DiT_boundary * 1000
            and getattr(pipe, "dit2", None) is not None
            and models["dit"] is not pipe.dit2
        ):
            pipe.load_models_to_device(pipe.in_iteration_models_2)
            models["dit"] = pipe.dit2
            models["vace"] = pipe.vace2

        t = timestep.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
        noise_pred = _predict_velocity(
            pipe, models, inputs_shared, inputs_posi, inputs_nega, t, cfg_scale, cfg_merge
        )

        prev, log_prob, _, _, _ = sde_step_with_logprob(
            pipe.scheduler,
            noise_pred,
            timesteps[progress_id],
            inputs_shared["latents"],
            noise_level=noise_level,
            deterministic=deterministic,
            generator=generator,
            logprob_mask=logprob_mask,
            return_dt_and_std_dev_t=True,
        )
        prev = prev.to(dtype=inputs_shared["latents"].dtype)

        # same clamp the stock loop applies (wan_video.py:335-336); masked out of log_prob
        if "first_frame_latents" in inputs_shared:
            prev[:, :, 0:1] = inputs_shared["first_frame_latents"]

        inputs_shared["latents"] = prev
        all_latents.append(prev.detach().cpu())
        all_log_probs.append(log_prob.detach().cpu())

    out = {
        "latents": torch.stack(all_latents, dim=1),
        "log_probs": torch.stack(all_log_probs, dim=1),
        "timesteps": timesteps.detach().cpu(),
        "vace_context": inputs_shared.get("vace_context"),
        "vace_scale": inputs_shared.get("vace_scale", 1.0),
        "first_frame_latents": inputs_shared.get("first_frame_latents"),
        "logprob_mask": logprob_mask,
        "video": None,
    }

    if decode:
        out["video"] = decode_latents(pipe, inputs_shared, inputs_posi, inputs_nega, merged)
    return out


@torch.no_grad()
def decode_latents(pipe, inputs_shared: dict, inputs_posi: dict, inputs_nega: dict, merged: dict):
    """Strip reference frames, run post_units, VAE-decode. Mirrors wan_video.py:338-359."""
    inputs_shared = dict(inputs_shared)
    vace_ref = merged.get("vace_reference_image")
    if vace_ref is not None:
        f = len(vace_ref) if isinstance(vace_ref, list) else 1
        inputs_shared["latents"] = inputs_shared["latents"][:, :, f:]

    for unit in pipe.post_units:
        inputs_shared, _, _ = pipe.unit_runner(unit, pipe, inputs_shared, inputs_posi, inputs_nega)

    pipe.load_models_to_device(["vae"])
    video = pipe.vae.decode(
        inputs_shared["latents"],
        device=pipe.device,
        tiled=merged.get("tiled", False),
        tile_size=merged.get("tile_size", (30, 52)),
        tile_stride=merged.get("tile_stride", (15, 26)),
    )
    if merged.get("output_type", "quantized") == "quantized":
        video = pipe.vae_output_to_video(video)
    pipe.load_models_to_device([])
    return video


def a2v_compute_log_prob(
    pipe,
    prepared_or_inputs: dict,
    latent: torch.Tensor,
    next_latent: torch.Tensor,
    timestep,
    *,
    vace_context: Optional[torch.Tensor] = None,
    first_frame_latents: Optional[torch.Tensor] = None,
    logprob_mask: Optional[torch.Tensor] = None,
    noise_level: float = 0.7,
    cfg_scale: Optional[float] = None,
    models: Optional[dict] = None,
    use_gradient_checkpointing: bool = True,
    use_gradient_checkpointing_offload: bool = False,
):
    """Re-score one stored transition under the *current* policy (training-time forward).

    Unlike the rollout this runs **with** autograd, and that is the memory bottleneck of the
    whole design: a bare 5B forward at 832x480 / 121 frames retains enough activations to fill
    an 80 GB card (measured: OOM at 79.17 GiB for a single sample). `model_fn_wan_video`
    forwards `use_gradient_checkpointing` down into both the DiT and the VACE branch
    (`wan_video.py:1527-1529`), which is how a2v's own SFT fits -- so it defaults to True here.
    Set `use_gradient_checkpointing_offload=True` to also stage activations to CPU if it still
    does not fit.

    `vace_context` / `first_frame_latents` override whatever is in the prepared inputs -- the
    RL loop passes the tensors it stored with the sample, because minibatches are reshuffled
    and no longer line up with any single `prepared` dict.

    Returns `(prev_sample, log_prob, prev_sample_mean, std_dev_t, sqrt(-dt))`, i.e. the same
    5-tuple Embodied-World-R1's `compute_log_prob` unpacks.
    """
    inputs_shared = dict(prepared_or_inputs["inputs_shared"])
    inputs_posi = prepared_or_inputs["inputs_posi"]
    inputs_nega = prepared_or_inputs["inputs_nega"]

    if vace_context is not None:
        inputs_shared["vace_context"] = vace_context
    if first_frame_latents is not None:
        inputs_shared["first_frame_latents"] = first_frame_latents
    if inputs_shared.get("vace_context") is None:
        raise ValueError(
            "a2v_compute_log_prob got no vace_context: the update would score a different "
            "distribution than the rollout sampled from (silently). Pass the tensor stored "
            "with the sample."
        )

    inputs_shared["latents"] = latent
    inputs_shared["use_gradient_checkpointing"] = use_gradient_checkpointing
    inputs_shared["use_gradient_checkpointing_offload"] = use_gradient_checkpointing_offload
    if os.getenv("A2V_GRPO_DEBUG"):
        p = pipe.dit.patch_size
        tok = (latent.shape[2] * (latent.shape[3] // p[1]) * (latent.shape[4] // p[2]))
        print(
            f"[a2v_compute_log_prob] latent={tuple(latent.shape)} tokens={tok} "
            f"grad_ckpt={use_gradient_checkpointing} offload={use_gradient_checkpointing_offload} "
            f"cfg_scale={cfg_scale if cfg_scale is not None else inputs_shared.get('cfg_scale')} "
            f"grad_enabled={torch.is_grad_enabled()}",
            flush=True,
        )
    if cfg_scale is None:
        cfg_scale = inputs_shared.get("cfg_scale", 1.0)
    cfg_merge = inputs_shared.get("cfg_merge", False)

    if models is None:
        models = {name: getattr(pipe, name) for name in pipe.in_iteration_models}

    t = timestep
    if not torch.is_tensor(t):
        t = torch.tensor(float(t))
    t_in = t.reshape(-1)[:1].to(dtype=pipe.torch_dtype, device=pipe.device)

    noise_pred = _predict_velocity(
        pipe, models, inputs_shared, inputs_posi, inputs_nega, t_in, cfg_scale, cfg_merge
    )
    return sde_step_with_logprob(
        pipe.scheduler,
        noise_pred,
        t,
        latent,
        prev_sample=next_latent,
        noise_level=noise_level,
        logprob_mask=logprob_mask,
        return_dt_and_std_dev_t=True,
    )
