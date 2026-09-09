"""L0: SDE step correctness for the DiffSynth FlowMatchScheduler.

Run:  /disk/worldmodel/uv/envs/a2v/bin/python -m a2v_grpo.tests.test_l0_sde

Covers the things that could silently break the port:
  1. sigma bookkeeping (DiffSynth has no trailing zero; the last step substitutes 0)
  2. timestep -> index lookup (argmin, not positional indexing)
  3. ODE path (noise_level=0) == FlowMatchScheduler.step, element-wise
  4. **SDE path has a non-zero policy gradient.** This is the guard for the trap that the
     deterministic/Euler path hits: prev_sample == prev_sample_mean exactly, so the Gaussian
     is evaluated at its peak and d log_prob / d model_output is identically 0 -> RL would
     silently not learn. See the sde.py module docstring for the measured numbers.
"""

from __future__ import annotations

import sys

import torch

from diffsynth.diffusion.flow_match import FlowMatchScheduler
from a2v_grpo.sde import sde_step_with_logprob, scheduler_step_reference


def main() -> int:
    torch.manual_seed(0)
    failures = []

    for num_steps in (50, 20):
        sched = FlowMatchScheduler("Wan")
        sched.set_timesteps(num_steps)

        # latent shape of the 5B A2V config: 832x480 / 121f -> (1, 48, 31, 30, 52).
        # A small stand-in keeps the test fast; the math is shape-agnostic.
        sample = torch.randn(2, 4, 3, 6, 8)
        model_output = torch.randn_like(sample)

        for i, t in enumerate(sched.timesteps):
            # --- our step (deterministic branch), batch of 2 sharing one timestep ---
            ours, log_prob, mean, std_dev_t, sqrt_neg_dt = sde_step_with_logprob(
                sched, model_output, t, sample, noise_level=0.0,
                return_dt_and_std_dev_t=True,
            )
            # --- upstream formula, evaluated per batch element ---
            ref = scheduler_step_reference(sched, model_output, t, sample)
            # --- the real scheduler, as a third independent check ---
            sched_out = sched.step(model_output, t, sample)

            for name, other in (("reference", ref), ("scheduler.step", sched_out)):
                if not torch.equal(ours, other):
                    failures.append(
                        f"N={num_steps} step={i} t={float(t):.2f}: mismatch vs {name}, "
                        f"max|diff|={(ours - other).abs().max().item():.3e}"
                    )

            # prev_sample == mean in the deterministic branch -> log_prob is the peak density
            if not torch.equal(ours, mean):
                failures.append(f"N={num_steps} step={i}: prev_sample != prev_sample_mean")
            if not torch.isfinite(log_prob).all():
                failures.append(f"N={num_steps} step={i}: log_prob not finite")
            if (std_dev_t <= 0).any() or (sqrt_neg_dt <= 0).any():
                failures.append(
                    f"N={num_steps} step={i}: non-positive noise scale "
                    f"(std_dev_t={std_dev_t.flatten()[0].item():.4e}, "
                    f"sqrt(-dt)={sqrt_neg_dt.flatten()[0].item():.4e})"
                )

        # last step must land exactly on sigma_prev = 0 -> dt = -sigma
        t_last = sched.timesteps[-1]
        _, _, _, std_last, sqrt_last = sde_step_with_logprob(
            sched, model_output, t_last, sample, noise_level=0.0,
            return_dt_and_std_dev_t=True,
        )
        sigma_last = sched.sigmas[-1].item()
        got = (sqrt_last.flatten()[0] ** 2).item()
        if abs(got - sigma_last) > 1e-6:
            failures.append(f"N={num_steps} last step: -dt={got:.6f} != sigma_last={sigma_last:.6f}")
        print(f"  N={num_steps}: {len(sched.timesteps)} steps checked; "
              f"sigma[0]={sched.sigmas[0]:.4f} sigma[-1]={sigma_last:.4f} "
              f"std_dev_t(last)={std_last.flatten()[0].item():.4f}")

    # per-element timesteps (each batch row at a different step) must index independently
    sched = FlowMatchScheduler("Wan")
    sched.set_timesteps(50)
    sample = torch.randn(3, 4, 2, 4, 4)
    model_output = torch.randn_like(sample)
    t_vec = torch.stack([sched.timesteps[0], sched.timesteps[10], sched.timesteps[49]])
    ours, _, _, _, _ = sde_step_with_logprob(
        sched, model_output, t_vec, sample, noise_level=0.0, return_dt_and_std_dev_t=True
    )
    for row, t in enumerate(t_vec):
        ref_row = scheduler_step_reference(
            sched, model_output[row: row + 1], t, sample[row: row + 1]
        )
        if not torch.equal(ours[row: row + 1], ref_row):
            failures.append(f"per-element timestep row={row} t={float(t):.2f}: mismatch")
    print(f"  per-element timesteps: 3 rows at steps 0/10/49 checked")

    # ---- bf16 bit-exactness: the real pipeline runs in bf16, not fp32 ----
    # L1 caught this: computing the Euler step in fp32 and casting back differs by ~1 bf16
    # eps per step and amplified to MAE 0.96 / max 233 over 10 steps of real generation.
    # fp32-only testing missed it entirely.
    print()
    # Devices matter here, not just dtypes: the "leave the sigma delta on CPU" requirement is
    # INVISIBLE on CPU (`.to(x.device)` is a no-op), so a CPU-only check passes while real
    # CUDA generation diverges. That is exactly how the bug reached L1. Short schedules are
    # included because whether a sigma delta is exactly representable in bf16 depends on the
    # step count -- a 4-step schedule diverged at steps 1 and 2 while 50 steps looked clean.
    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    if "cuda" not in devices:
        print("  ! CUDA unavailable: skipping the device-dependent bit-exactness check, "
              "which is the only one that can catch the CPU-scalar-vs-CUDA-0dim bug.")
    for device in devices:
        for dtype in (torch.bfloat16, torch.float16):
            for n in (4, 10, 50):
                sched = FlowMatchScheduler("Wan")
                sched.set_timesteps(n)
                xs = torch.randn(1, 48, 4, 6, 8, dtype=dtype, device=device)
                vs = torch.randn_like(xs)
                bad, worst = 0, 0.0
                for t in sched.timesteps:
                    ours, *_ = sde_step_with_logprob(
                        sched, vs, t, xs, noise_level=0.0, return_dt_and_std_dev_t=True
                    )
                    ref = sched.step(vs, t, xs)
                    if ours.dtype != ref.dtype:
                        failures.append(
                            f"{device}/{dtype}: dtype drift {ours.dtype} vs {ref.dtype}")
                    if not torch.equal(ours, ref):
                        bad += 1
                        worst = max(worst, (ours.float() - ref.float()).abs().max().item())
                tag = f"  {device:4s} {str(dtype).replace('torch.',''):9s} N={n:<3d}"
                print(f"{tag}: {n - bad}/{n} bit-exact" + (f"  worst={worst:.3e}" if bad else ""))
                if bad:
                    failures.append(
                        f"{device}/{dtype}/N={n}: {bad} steps not bit-exact (worst {worst:.3e})")

    # ---- the guard that matters: SDE path must yield a usable policy gradient ----
    print()
    sched = FlowMatchScheduler("Wan")
    sched.set_timesteps(50)
    x = torch.randn(2, 4, 3, 6, 8)
    v = torch.randn_like(x)
    t = sched.timesteps[10]

    for noise_level, expect_grad in ((0.0, False), (0.3, True), (0.7, True), (1.0, True)):
        # rollout: sample prev_sample under this noise level
        prev, lp_old, mean, _, _ = sde_step_with_logprob(
            sched, v, t, x, noise_level=noise_level,
            generator=torch.Generator().manual_seed(7), return_dt_and_std_dev_t=True,
        )
        delta = (prev - mean).abs().max().item()
        # update time: same stored prev_sample, gradient w.r.t. the policy output
        vg = v.clone().requires_grad_(True)
        _, lp, _, _, _ = sde_step_with_logprob(
            sched, vg, t, x, prev_sample=prev, noise_level=noise_level,
            return_dt_and_std_dev_t=True,
        )
        lp.sum().backward()
        nz = int((vg.grad != 0).sum())
        gmax = vg.grad.abs().max().item()
        ok = (nz > 0) == expect_grad
        print(f"  noise_level={noise_level:<4}: |prev-mean|={delta:.4e}  grad非零={nz}/{vg.numel()}"
              f"  max|grad|={gmax:.3e}  {'OK' if ok else 'UNEXPECTED'}")
        if not ok:
            failures.append(
                f"noise_level={noise_level}: expected grad non-zero={expect_grad}, got nz={nz}"
            )
        if expect_grad and not torch.isfinite(lp).all():
            failures.append(f"noise_level={noise_level}: log_prob not finite")

    # Importance-sampling premise. The right property is not "the ratio is far from 1" (its
    # size depends on the perturbation and is diluted by averaging over ~1e3 elements) but:
    #   - unchanged policy  -> ratio exactly 1
    #   - larger perturbation -> monotonically larger |log ratio|
    prev, lp_old, _, _, _ = sde_step_with_logprob(
        sched, v, t, x, noise_level=0.7,
        generator=torch.Generator().manual_seed(11), return_dt_and_std_dev_t=True,
    )
    _, lp_same, _, _, _ = sde_step_with_logprob(
        sched, v, t, x, prev_sample=prev, noise_level=0.7, return_dt_and_std_dev_t=True
    )
    r_same = (lp_same - lp_old).exp().mean().item()
    print(f"  importance ratio 同策略 = {r_same:.8f}(应恰为 1)")
    if abs(r_same - 1.0) > 1e-6:
        failures.append(f"ratio at unchanged policy = {r_same}, expected 1")

    torch.manual_seed(3)
    direction = torch.randn_like(v)
    mags = []
    for eps in (0.05, 0.2, 0.8):
        _, lp_moved, _, _, _ = sde_step_with_logprob(
            sched, v + eps * direction, t, x, prev_sample=prev, noise_level=0.7,
            return_dt_and_std_dev_t=True,
        )
        m = (lp_moved - lp_old).abs().mean().item()
        mags.append(m)
        print(f"    扰动 eps={eps:<5}: |Δlog_prob| = {m:.3e}")
    if not (mags[0] < mags[1] < mags[2]):
        failures.append(f"|Δlog_prob| not monotone in perturbation size: {mags}")
    if mags[0] <= 0.0:
        failures.append("log_prob did not respond to a policy change at all")

    print()
    if failures:
        print(f"L0 FAIL ({len(failures)} problems)")
        for f in failures[:10]:
            print(f"  - {f}")
        return 1
    print("L0 PASS: deterministic SDE step == FlowMatchScheduler.step (element-wise), "
          "sigma bookkeeping and argmin lookup verified")
    return 0


if __name__ == "__main__":
    sys.exit(main())
