"""SDE step + log-probability for DiffSynth's FlowMatchScheduler.

Ported from Embodied-World-R1 `flow_grpo/diffusers_patch/wan_pipeline_with_logprob.py:159`
(`sde_step_with_logprob`), which targets the diffusers FlowMatch schedulers. The *math* is
unchanged; only the sigma bookkeeping is adapted, because the two schedulers store their
sigma schedule differently:

    diffusers  : sigmas = [1.0, ..., 1/N, 0.0]   <- explicit trailing zero, len = N+1
                 step index via `index_for_timestep`
    DiffSynth  : sigmas = [1.0, ..., 1/N]        <- NO trailing zero, len = N
                 (`diffsynth/diffusion/flow_match.py:315` set_timesteps_wan)
                 step index via `argmin(|timesteps - t|)` on CPU
                 the final step substitutes `sigma_ = 0` in-line (`flow_match.py:332-335`)

So DiffSynth's "`sigma_ = 0` at the last step" *is* diffusers' trailing zero. Treating the
schedule as `[sigmas..., 0]` makes the two exactly equivalent, which is why we use

    sigma_max = sigmas[0]  (= 1.0 for the Wan schedule, measured)
    sigma_min = 0.0        (the implicit trailing zero)
    => std_dev_t = sigma_min + (sigma_max - sigma_min) * sigma == sigma

and `std_dev_t == sigma` is precisely what the diffusers/EWR1 path effectively computes.

Deterministic branch identity (this is what makes L0 hold by construction):

    prev_sample_mean = sample + dt * model_output       # dt = sigma_prev - sigma
    FlowMatchScheduler.step: sample + model_output * (sigma_ - sigma)

...are the same expression. `scheduler_step_reference` below re-implements DiffSynth's
`step` so a test can assert element-wise equality.

## Why rollouts MUST use the SDE here (measured, do not "simplify" this back)

Embodied-World-R1's Wan path samples with the checkpoint's `UniPCMultistepScheduler`
(2nd-order multistep) while modelling the log-probability as a Gaussian around the
*1st-order Euler* mean. Those two disagree systematically, and that gap is what makes
`prev_sample - prev_sample_mean` non-zero, i.e. what produces a policy gradient at all.
Measured on the 1.3B scheduler config, 50 steps:

    step 0      order 1  ->  |UniPC - Euler| = 4.8e-07  (fp32 rounding; no usable gradient)
    steps 1..48 order 2  ->  |UniPC - Euler| = 0.9e-2 .. 3.5e-2  (0.2%-0.8% relative)
    step 49     order 1  ->  exactly 0       (lower_order_final; no gradient)

DiffSynth's `FlowMatchScheduler.step` *is* the Euler step, so `prev_sample` would equal
`prev_sample_mean` **exactly**, the Gaussian would be evaluated at its peak, and
`d log_prob / d model_output` would be identically 0 -- silently no learning. Verified:
with the deterministic branch the gradient is 0/1152 elements non-zero.

Therefore rollouts use the genuine reverse SDE for rectified flow (Flow-GRPO):

    drift : v + (s^2 / 2*sigma) * (x + (1-sigma) * v)
    noise : s * sqrt(-dt) * eps,      s = noise_level * sigma

`noise_level` scales `s` in *both* drift and noise (they are coupled; scaling only the noise
would target the wrong distribution). `noise_level=0` degenerates to plain Euler + no noise,
which is the evaluation/ODE path and is what L0 checks against `FlowMatchScheduler.step`.

(EWR1 spells the flag `determistic`; we use the correct `deterministic` here.)
"""

from __future__ import annotations

import math
from typing import Optional, Union

import torch


def _step_ids(scheduler, timestep: Union[float, torch.Tensor], batch_size: int) -> torch.Tensor:
    """Locate each timestep in `scheduler.timesteps`, mirroring DiffSynth's own lookup.

    `FlowMatchScheduler.step` does `argmin(|timesteps - t|)` after moving `t` to CPU
    (flow_match.py:328-330). We keep that -- indexing by position would silently break
    whenever `denoising_strength < 1` shifts the schedule.
    """
    timesteps = scheduler.timesteps.detach().float().cpu().reshape(1, -1)
    if not torch.is_tensor(timestep):
        timestep = torch.tensor([float(timestep)])
    t = timestep.detach().float().cpu().reshape(-1, 1)
    if t.shape[0] == 1 and batch_size > 1:
        t = t.expand(batch_size, 1)
    return torch.argmin((timesteps - t).abs(), dim=1)


def _native_euler_step(scheduler, sample, model_output, ids, n_steps):
    """`sample + model_output * (sigma_prev - sigma)` in the sample's own dtype.

    Reproduces `FlowMatchScheduler.step` **bit for bit**. Two subtleties, both measured:

    1. Do the step in the native dtype. Computing it in fp32 and casting back differs by
       ~1 bf16 eps (7.8e-03) per step, which amplifies through the chaotic sampler
       (measured: MAE 0.96 / max 233 over 10 steps of real 832x480 generation).

    2. **Leave the sigma delta on the CPU.** Upstream indexes `self.sigmas` (a CPU float32
       tensor) and multiplies the CUDA activation by that 0-dim CPU tensor. PyTorch then
       treats it as a *scalar*: one rounding, `bf16 x f32_scalar -> bf16`. Move the same
       0-dim tensor to CUDA and it goes through the tensor-tensor kernel, which rounds the
       delta to bf16 **first** and multiplies second -- two roundings. Both return bf16, so
       this is invisible in dtypes; it only shows up when the delta is not exactly
       representable in bf16. Measured on CUDA:

           d = -0.0625     (exact)    -> bit-identical
           d = -0.625      (exact)    -> bit-identical
           d = -0.104167   (inexact)  -> differs by 4.9e-04
           d = -0.208333   (inexact)  -> differs by 9.8e-04

       That is why a 4-step schedule diverged at steps 1 and 2 but not 0 and 3: only the
       middle deltas are inexact. **A CPU-only test cannot catch this** -- `.to(x.device)` is
       a no-op there, so both spellings are identical. `test_l0_sde.py` therefore runs its
       bit-exactness check on CUDA too when one is available.

    Returns None when the batch spans multiple timesteps (0-dim sigmas no longer apply). That
    only happens in the training-time recompute, where `prev_sample` is supplied and this
    function is not called.
    """
    uniq = torch.unique(ids)
    if uniq.numel() != 1:
        return None
    i0 = int(uniq.item())
    sig = scheduler.sigmas[i0]
    if i0 + 1 >= n_steps:
        # upstream substitutes the Python int 0 here (flow_match.py:332-335)
        d = 0 - sig
    else:
        d = scheduler.sigmas[i0 + 1] - sig
    return sample + model_output * d  # d stays where the scheduler keeps it: CPU


def scheduler_step_reference(
    scheduler,
    model_output: torch.Tensor,
    timestep: Union[float, torch.Tensor],
    sample: torch.Tensor,
    to_final: bool = False,
) -> torch.Tensor:
    """Re-implementation of `FlowMatchScheduler.step` used only by the L0 equivalence test.

    Kept here (rather than calling the scheduler) so the test compares our sigma bookkeeping
    against the upstream formula rather than against itself.
    """
    if torch.is_tensor(timestep):
        timestep = timestep.detach().cpu()
    timestep_id = torch.argmin((scheduler.timesteps - timestep).abs())
    sigma = scheduler.sigmas[timestep_id]
    if to_final or timestep_id + 1 >= len(scheduler.timesteps):
        sigma_ = 0.0
    else:
        sigma_ = scheduler.sigmas[timestep_id + 1]
    return sample + model_output * (sigma_ - sigma)


def sde_step_with_logprob(
    scheduler,
    model_output: torch.Tensor,
    timestep: Union[float, torch.Tensor],
    sample: torch.Tensor,
    prev_sample: Optional[torch.Tensor] = None,
    generator: Optional[torch.Generator] = None,
    noise_level: float = 0.7,
    deterministic: bool = False,
    logprob_mask: Optional[torch.Tensor] = None,
    return_dt_and_std_dev_t: bool = False,
):
    """One reverse step, plus the Gaussian log-probability of `prev_sample`.

    Args:
        scheduler: a `diffsynth.diffusion.flow_match.FlowMatchScheduler` with `set_timesteps`
            already called (needs `.sigmas` / `.timesteps`).
        model_output: velocity prediction, shape (B, C, T, H, W).
        timestep: scalar or (B,) tensor of timesteps for this step.
        sample: current latent x_t, shape (B, C, T, H, W).
        prev_sample: when given (training-time recompute), the log-prob is evaluated at this
            point instead of sampling a new one. Mutually exclusive with `generator`.
        noise_level: SDE strength. Scales `s` in both the drift correction and the injected
            noise. **0 => plain Euler ODE** (evaluation path; what L0 pins down).
            Non-zero is required for RL -- see the module docstring for why the
            deterministic path has an identically-zero policy gradient here.
        deterministic: force the ODE path regardless of `noise_level` (evaluation).
        logprob_mask: optional 0/1 tensor broadcastable to `sample`, marking which latent
            elements are genuinely *sampled*. Elements set to 0 are excluded from the
            log-probability average. Needed for the Wan2.2 TI2V (`ti2v_fused`) A2V config,
            where `diffsynth/pipelines/wan_video.py:335-336` overwrites latent frame 0 with
            the VAE-encoded first frame after **every** step: that frame is a deterministic
            clamp, not a draw from the transition kernel, yet its `prev_sample_mean` still
            depends on the policy. Including it would add a spurious auxiliary gradient
            pulling the predicted frame-0 mean toward the clamp -- and with a negative
            advantage it would push *away* from the correct first frame.
        return_dt_and_std_dev_t: return `sqrt(-dt)` separately (5-tuple) rather than folded
            into the 4th element. Matches EWR1's flag of the same name so that
            `compute_log_prob` can unpack identically.

    Returns:
        `(prev_sample, log_prob, prev_sample_mean, std_dev_t, sqrt(-dt))` when
        `return_dt_and_std_dev_t`, else `(prev_sample, log_prob, prev_sample_mean,
        std_dev_t * sqrt(-dt))`.
    """
    if prev_sample is not None and generator is not None:
        raise ValueError("Pass either `generator` or `prev_sample`, not both.")
    use_sde = (not deterministic) and noise_level > 0

    # Keep the originals: the ODE branch must step in the *native* dtype to stay bit-exact
    # with FlowMatchScheduler.step (see `_native_euler_step`).
    sample_orig, model_output_orig = sample, model_output

    # fp32 throughout: the log-prob is a difference of large-ish numbers and bf16 loses it.
    model_output = model_output.float()
    sample = sample.float()
    if prev_sample is not None:
        prev_sample = prev_sample.float()

    device = sample.device
    batch_size = sample.shape[0]
    sigmas = scheduler.sigmas.detach().float().to(device)
    n_steps = len(scheduler.timesteps)

    ids = _step_ids(scheduler, timestep, batch_size)
    sigma = sigmas[ids.to(device)]
    # DiffSynth substitutes 0 for the step past the end -- that is the implicit trailing zero.
    next_ids = ids + 1
    is_last = next_ids >= n_steps
    sigma_prev = torch.zeros_like(sigma)
    if (~is_last).any():
        keep = (~is_last).to(device)
        sigma_prev[keep] = sigmas[next_ids[~is_last].to(device)]

    shape = (-1,) + (1,) * (sample.ndim - 1)
    sigma = sigma.view(shape)
    sigma_prev = sigma_prev.view(shape)
    dt = sigma_prev - sigma  # negative: sigma decreases along the schedule

    sigma_max = sigmas[0].item()
    sigma_min = 0.0  # see module docstring: the implicit trailing zero
    # noise_level scales s in BOTH drift and noise -- they are coupled.
    std_dev_t = noise_level * (sigma_min + (sigma_max - sigma_min) * sigma)

    if use_sde:
        prev_sample_mean = sample * (1 + std_dev_t**2 / (2 * sigma) * dt) + model_output * (
            1 + std_dev_t**2 * (1 - sigma) / (2 * sigma)
        ) * dt
    else:
        # identical to FlowMatchScheduler.step -> L0 holds by construction
        prev_sample_mean = sample + dt * model_output

    # The ODE path has std_dev_t == 0, which would make the Gaussian degenerate. Keep a
    # nominal width there so log_prob stays finite; it is unused for optimisation because
    # the ODE path is evaluation-only.
    width = std_dev_t if use_sde else (sigma_min + (sigma_max - sigma_min) * sigma)
    noise_scale = width * torch.sqrt(-dt)

    if prev_sample is None:
        if not use_sde:
            # Bit-exact with FlowMatchScheduler.step. Doing this in fp32 and casting back
            # differs by ~1 bf16 eps (7.8e-03) per step, which amplifies through the chaotic
            # sampler: measured MAE 0.96 / max|diff| 233 over only 10 steps at 832x480.
            prev_sample = _native_euler_step(
                scheduler, sample_orig, model_output_orig, ids, n_steps
            )
            if prev_sample is None:  # mixed timesteps in one batch: eval-only, cast is fine
                prev_sample = prev_sample_mean.to(sample_orig.dtype)
        else:
            noise = torch.randn(
                model_output.shape,
                generator=generator,
                device=device,
                dtype=model_output.dtype,
            )
            prev_sample = prev_sample_mean + noise_scale * noise

    log_prob = (
        -((prev_sample.detach() - prev_sample_mean) ** 2) / (2 * (noise_scale**2 + 1e-12))
        - torch.log(noise_scale + 1e-12)
        - math.log(math.sqrt(2 * math.pi))
    )
    reduce_dims = tuple(range(1, log_prob.ndim))
    if logprob_mask is None:
        log_prob = log_prob.mean(dim=reduce_dims)
    else:
        m = logprob_mask.to(device=log_prob.device, dtype=log_prob.dtype).expand_as(log_prob)
        log_prob = (log_prob * m).sum(dim=reduce_dims) / m.sum(dim=reduce_dims).clamp_min(1.0)

    if return_dt_and_std_dev_t:
        return prev_sample, log_prob, prev_sample_mean, width, torch.sqrt(-dt)
    return prev_sample, log_prob, prev_sample_mean, noise_scale
