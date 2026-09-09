"""Locate the exact step where our rollout's latents diverge from the stock loop.

Compares **latents**, not decoded pixels, so the VAE is out of the picture. The stock path's
pre-decode latent is captured by temporarily wrapping `pipe.vae.decode` (no edits to
diffsynth/).

Reads as:
  * step 0 already differs      -> the forward or the step itself
  * identical for k steps, then -> loop-carried state (tea-cache, model swap, in-place reuse)
  * identical throughout        -> the divergence is in decode/post_units, not the loop

Memory-lean on purpose: one `prepare`, one forward per step, frees as it goes (the first
version of this script OOMed at 79 GiB by holding two prepared input sets and two forwards).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from PIL import Image

from a2v.base_spec import REGISTRY
from a2v.infer_a2v import NEG_PROMPT, build_pipe_with_vace, first_frame_kwargs, load_frames
from a2v_grpo.rollout import _predict_velocity, prepare_a2v_inputs
from a2v_grpo.sde import sde_step_with_logprob


def cmp(tag, a, b):
    if a.shape != b.shape:
        print(f"    {tag:26s} SHAPE {tuple(a.shape)} vs {tuple(b.shape)}")
        return False
    same = torch.equal(a, b)
    d = (a.float() - b.float()).abs()
    print(f"    {tag:26s} equal={str(same):5s} max={d.max().item():.4e} mean={d.mean().item():.4e}")
    return same


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_spec", default="wan2.2-ti2v-5b")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--metadata", default="metadata.jsonl")
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--num_frames", type=int, default=121)
    ap.add_argument("--num_inference_steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dataset = Path(args.dataset).resolve()
    with open(dataset / args.metadata, encoding="utf-8") as fh:
        row = json.loads(fh.readline())
    pipe, spec = build_pipe_with_vace(args.base_spec, args.ckpt, 1.0)

    vace_video = load_frames(dataset, row["vace_video"][: args.num_frames])
    ref = Image.open(dataset / row["vace_reference_image"]).convert("RGB")
    common = dict(
        prompt=row.get("prompt", "robot arm manipulation"),
        negative_prompt=NEG_PROMPT,
        vace_video=vace_video,
        height=args.height, width=args.width, num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps, seed=args.seed, tiled=False,
        **first_frame_kwargs(spec, ref),
    )

    # ---- capture the stock path's pre-decode latent ----
    grabbed = {}
    orig_decode = pipe.vae.decode

    def spy(latents, *a, **kw):
        grabbed["latents"] = latents.detach().clone()
        return orig_decode(latents, *a, **kw)

    pipe.vae.decode = spy
    print(f"[diag] stock pipe.__call__ ({args.num_inference_steps} steps) ...")
    _ = pipe(**common)
    pipe.vae.decode = orig_decode
    stock_final = grabbed["latents"]
    print(f"[diag] stock pre-decode latents: {tuple(stock_final.shape)} {stock_final.dtype}")
    torch.cuda.empty_cache()

    # ---- our path, step by step, keeping the per-step latents ----
    print(f"[diag] our rollout, noise_level=0 ...")
    prepared = prepare_a2v_inputs(pipe, **common)
    ish = dict(prepared["inputs_shared"])
    posi, nega = prepared["inputs_posi"], prepared["inputs_nega"]
    cfg_scale, cfg_merge = ish.get("cfg_scale", 1.0), ish.get("cfg_merge", False)

    pipe.load_models_to_device(pipe.in_iteration_models)
    models = {n: getattr(pipe, n) for n in pipe.in_iteration_models}
    timesteps = pipe.scheduler.timesteps
    print(f"[diag] sigmas={[round(float(s),5) for s in pipe.scheduler.sigmas]}")

    ours = ish["latents"]
    # `torch.no_grad()` is NOT optional here: calling the forward bare makes autograd retain
    # the whole 5B graph and OOMs at ~79 GiB. (a2v_rollout_with_logprob carries the decorator
    # itself; this script bypasses it by calling _predict_velocity directly.)
    with torch.no_grad():
        for i, tt in enumerate(timesteps):
            t_in = tt.unsqueeze(0).to(dtype=pipe.torch_dtype, device=pipe.device)
            ish["latents"] = ours
            v = _predict_velocity(pipe, models, ish, posi, nega, t_in, cfg_scale, cfg_merge)

            by_sched = pipe.scheduler.step(v, tt, ours)
            by_ours, *_ = sde_step_with_logprob(
                pipe.scheduler, v, tt, ours, noise_level=0.0, return_dt_and_std_dev_t=True
            )
            print(f"  --- step {i} (t={float(tt):.2f}) ---")
            cmp("sde vs scheduler.step", by_ours, by_sched)

            nxt = by_ours
            if "first_frame_latents" in ish:
                nxt = nxt.clone()
                nxt[:, :, 0:1] = ish["first_frame_latents"]
            ours = nxt
            del v, by_sched, by_ours
            torch.cuda.empty_cache()

    # ---- compare finals (apply the same reference-frame strip the stock tail does) ----
    vace_ref = prepared["merged"].get("vace_reference_image")
    if vace_ref is not None:
        f = len(vace_ref) if isinstance(vace_ref, list) else 1
        ours = ours[:, :, f:]
    print("\n[diag] final pre-decode latents")
    ok = cmp("stock vs ours", stock_final, ours)

    print("\n[diag] verdict")
    if ok:
        print("    latents match -> the pixel difference comes from decode/post_units, "
              "not the denoising loop.")
    else:
        print("    latents differ -> the divergence is inside the loop. The per-step "
              "'sde vs scheduler.step' lines above localise it: if those are all equal, "
              "the difference is in what we FEED the forward (inputs_shared), not the step.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
