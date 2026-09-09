"""L2: the control signal must actually reach the *training-time* forward.

The failure this guards against is the nastiest one in the whole design and it raises nothing:

    rollout samples with `vace_context` (action conditioning)   -> distribution A
    the policy update re-scores log-probs without it            -> distribution B
    => the importance ratio is meaningless, GRPO optimises noise, the job "runs fine"

So we assert that `a2v_compute_log_prob` is *sensitive* to the conditioning:

  1. same conditioning, same policy      -> log_prob identical, ratio exactly 1
  2. zeroed  `vace_context`              -> log_prob must change
  3. another episode's `vace_context`    -> log_prob must change
  4. missing `vace_context`              -> must raise, not silently fall back

Run:
    cd /disk/worldmodel/wangshilong/DiffSynth-Studio
    CUDA_VISIBLE_DEVICES=6 A2V_MODELS_DIR=/disk/worldmodel/public_model \\
      /disk/worldmodel/uv/envs/a2v/bin/python -m a2v_grpo.tests.test_l2_control_wired \\
        --ckpt /disk/worldmodel/ziyang/a2v_khl_v2/checkpoints/a2v_1k_uniform121/step-7400.safetensors \\
        --dataset /disk/worldmodel/ziyang/a2v_khl_v2/dataset_full \\
        --metadata metadata_1k_uniform121.jsonl
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
from a2v_grpo.rollout import a2v_compute_log_prob, a2v_rollout_with_logprob, prepare_a2v_inputs


def pick_two_distinct_episodes(dataset: Path, metadata: str) -> list[dict]:
    """Two rows from *different* episodes.

    The manifest stores each episode 3x, once per instruction variant
    (`__instructions`, `__instructions_1`, `__instructions_2`) sharing the SAME frames -- so
    consecutive rows usually have identical `vace_context` and would make this test vacuous.
    Key on the episode directory, not the row index.
    """
    def episode_of(row: dict) -> str:
        v = (row.get("vace_video") or [""])[0]
        return v.split("/")[0]

    first = None
    with open(dataset / metadata, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if first is None:
                first = row
                continue
            if episode_of(row) != episode_of(first):
                return [first, row]
    return [first] if first else []


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
    ap.add_argument("--noise_level", type=float, default=0.7)
    ap.add_argument("--step_index", type=int, default=1)
    args = ap.parse_args()

    dataset = Path(args.dataset).resolve()
    rows = pick_two_distinct_episodes(dataset, args.metadata)
    if len(rows) < 2:
        print("[L2] could not find two rows from different episodes in the manifest")
        return 1
    spec = REGISTRY[args.base_spec]
    pipe, spec = build_pipe_with_vace(args.base_spec, args.ckpt, 1.0)

    def prep(row):
        frames = load_frames(dataset, row["vace_video"][: args.num_frames])
        ref = Image.open(dataset / row["vace_reference_image"]).convert("RGB")
        return prepare_a2v_inputs(
            pipe,
            prompt=row.get("prompt", "robot arm manipulation"),
            negative_prompt=NEG_PROMPT,
            vace_video=frames,
            height=args.height, width=args.width, num_frames=args.num_frames,
            num_inference_steps=args.num_inference_steps, seed=0, tiled=False,
            **first_frame_kwargs(spec, ref),
        )

    print(f"[L2] preparing 2 episodes ({rows[0].get('id')}, {rows[1].get('id')}) ...")
    pa = prep(rows[0])
    ctx_a = pa["inputs_shared"]["vace_context"]
    pb = prep(rows[1])
    ctx_b = pb["inputs_shared"]["vace_context"]
    d_ctx = (ctx_a.float() - ctx_b.float()).abs().mean().item()
    print(f"[L2] vace_context A vs B: mean|diff| = {d_ctx:.4e} (must be > 0 or the test is void)")
    if d_ctx == 0.0:
        print("[L2] FAIL: the two episodes have identical conditioning; pick different rows.")
        return 1

    # roll out episode A and keep one stored transition
    print(f"[L2] rollout A (noise_level={args.noise_level}) ...")
    out = a2v_rollout_with_logprob(
        pipe, pa, noise_level=args.noise_level, decode=False
    )
    traj, mask = out["latents"], out["logprob_mask"]
    j = min(args.step_index, traj.shape[1] - 2)
    lat, nxt = traj[:, j].to(pipe.device), traj[:, j + 1].to(pipe.device)
    tt = out["timesteps"][j]
    print(f"[L2] scoring step j={j} (t={float(tt):.2f}), mask="
          f"{'frame0 masked' if mask is not None else 'none'}")

    def score(ctx):
        # This test only reads log_prob values, so run without autograd. With the graph
        # retained a single 5B forward at 832x480/121f fills an 80 GB card (measured: OOM at
        # 79.17 GiB). The real training path DOES need gradients -- that is what
        # `use_gradient_checkpointing` is for, see a2v_compute_log_prob.
        with torch.no_grad():
            _, lp, *_ = a2v_compute_log_prob(
                pipe, pa, lat, nxt, tt, vace_context=ctx,
                logprob_mask=mask, noise_level=args.noise_level,
            )
        return lp.detach().float().reshape(-1)[0].item()

    base = score(ctx_a)
    same = score(ctx_a)
    zero = score(torch.zeros_like(ctx_a))
    other = score(ctx_b)

    print()
    print(f"[L2] log_prob | same conditioning : {base:.8f}  /  repeat {same:.8f}")
    print(f"[L2] log_prob | vace_context = 0  : {zero:.8f}   (Δ = {zero - base:+.3e})")
    print(f"[L2] log_prob | other episode ctx : {other:.8f}   (Δ = {other - base:+.3e})")

    failures = []
    if same != base:
        failures.append(f"non-deterministic re-score: {base} vs {same}")
    if zero == base:
        failures.append("zeroing vace_context did NOT change log_prob -> control is NOT wired "
                        "into the training forward")
    if other == base:
        failures.append("swapping in another episode's vace_context did NOT change log_prob "
                        "-> control is NOT wired into the training forward")

    # a missing conditioning must be an error, never a silent fallback
    stripped = {**pa, "inputs_shared": {**pa["inputs_shared"], "vace_context": None}}
    try:
        a2v_compute_log_prob(pipe, stripped, lat, nxt, tt, logprob_mask=mask,
                             noise_level=args.noise_level)
        failures.append("missing vace_context did not raise -- a silent fallback would let a "
                        "broken run look healthy")
    except ValueError as exc:
        print(f"[L2] missing vace_context correctly raised: {str(exc)[:80]}...")

    print()
    if failures:
        print(f"[L2] FAIL ({len(failures)})")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("[L2] PASS: the action conditioning demonstrably drives the training-time log-prob")
    return 0


if __name__ == "__main__":
    sys.exit(main())
