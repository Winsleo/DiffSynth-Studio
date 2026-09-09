"""Flow-GRPO rollout for the DiffSynth A2V (VACE) pipeline.

Why this package exists (see Embodied-World-R1/docs/A2V_VACE_INTEGRATION.md §10):
the production A2V model is a **5B** from-DiT full-param VACE branch on Wan2.2-TI2V-5B
(832x480 / 121f, WorldArena EWMScore 62.88). diffusers' WanVACEPipeline cannot host it
(no `expand_timesteps`, Wan2.1-only VAE scaling, and it uses `vace_reference_image`
instead of the 5B model's `ti2v_fused` first frame). So RL rollouts run on the DiffSynth
pipeline, while Embodied-World-R1 keeps owning the GRPO loop / reward / EMA.

Kept as a separate top-level package on purpose: upstream `a2v/` stays untouched.
"""

from .sde import sde_step_with_logprob, scheduler_step_reference
from .rollout import (
    a2v_compute_log_prob,
    a2v_rollout_with_logprob,
    build_logprob_mask,
    decode_latents,
    disable_lora,
    prepare_a2v_inputs,
)

__all__ = [
    "sde_step_with_logprob",
    "scheduler_step_reference",
    "prepare_a2v_inputs",
    "a2v_rollout_with_logprob",
    "a2v_compute_log_prob",
    "decode_latents",
    "build_logprob_mask",
    "disable_lora",
]
