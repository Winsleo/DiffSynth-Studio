"""Does the production 5B A2V model actually read trajectory *content*?

Motivation: the RL plan starts from `a2v_1k_uniform121/step-7400` (WorldArena EWMScore 62.88).
That checkpoint's name matches `dataset_full/metadata_1k_uniform121.jsonl`, and sampling 40
episodes of `dataset_full` found only **28 distinct** trajectory maps -- one shared by 13
episodes. If a large share of its training pairs carried a duplicated control signal, the model
may respond to "a trajectory is present" without following *which* trajectory. Starting RL from
such a policy would optimise the wrong thing.

`real < none` cannot answer this (a model trained on one fixed trajectory would still prefer it
over out-of-distribution gray). The discriminating arm is **swap**: a *valid but wrong*
trajectory from another episode, with the reference frame and prompt held fixed.

    real   ref_A + traj_A   -> baseline
    none   ref_A + gray     -> weak probe ("is a control present")
    swap   ref_A + traj_B   -> strong probe ("is its content used")

Runs through `a2v_grpo.rollout` (proven bit-exact with the stock inference path at
noise_level=0, see test_l1) rather than `a2v/eval_multiep`, for two reasons: the step count is
controllable, and this exercises the exact code path RL will use.

Metrics come from `a2v.causal_metrics` unchanged, so numbers are comparable to the gates in
docs/A2V_VACE_INTEGRATION.md §5.1/§9.

Run:
    cd /disk/worldmodel/wangshilong/DiffSynth-Studio
    CUDA_VISIBLE_DEVICES=6 A2V_MODELS_DIR=/disk/worldmodel/public_model \\
      /disk/worldmodel/uv/envs/a2v/bin/python -m a2v_grpo.tests.gate_swap_5b \\
        --ckpt .../a2v_1k_uniform121/step-7400.safetensors \\
        --pair /disk/worldmodel/wangshilong/a2v_data/gate5b_pair_832x480 \\
        --swap /disk/worldmodel/wangshilong/a2v_data/gate5b_swap_832x480 \\
        --steps 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from a2v.base_spec import REGISTRY
from a2v.causal_metrics import _mae, _motion, _read_pngs
from a2v.infer_a2v import NEG_PROMPT, build_pipe_with_vace, first_frame_kwargs, load_frames
from a2v_grpo.rollout import a2v_rollout_with_logprob, prepare_a2v_inputs


def rows_of(ds: Path) -> list[dict]:
    return [json.loads(l) for l in open(ds / "metadata.jsonl", encoding="utf-8") if l.strip()]


def to_array(video, h: int, w: int) -> np.ndarray:
    if isinstance(video, torch.Tensor):
        v = video.detach().float().cpu()
        if v.ndim == 5:
            v = v[0]
        if v.shape[0] in (3, 4):
            v = v.permute(1, 2, 3, 0)          # C T H W -> T H W C
        v = ((v.clamp(-1, 1) + 1) / 2 * 255).round().numpy().astype(np.uint8)
        return v
    out = []
    for f in video:
        im = f.convert("RGB")
        if im.size != (w, h):
            im = im.resize((w, h), Image.BICUBIC)
        out.append(np.asarray(im, dtype=np.uint8))
    return np.stack(out)


def generate(pipe, spec, ds: Path, row: dict, control: str, args) -> np.ndarray:
    frames = load_frames(ds, row["vace_video"][: args.frames])
    if control == "none":
        frames = [Image.new("RGB", (args.width, args.height), (128, 128, 128)) for _ in frames]
    ref = Image.open(ds / row["vace_reference_image"]).convert("RGB")
    prepared = prepare_a2v_inputs(
        pipe,
        prompt=row.get("prompt", "robot arm manipulation"),
        negative_prompt=NEG_PROMPT,
        vace_video=frames,
        height=args.height, width=args.width, num_frames=args.frames,
        num_inference_steps=args.steps, seed=args.seed, tiled=False,
        **first_frame_kwargs(spec, ref),
    )
    out = a2v_rollout_with_logprob(pipe, prepared, noise_level=0.0, decode=True)
    return to_array(out["video"], args.height, args.width)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_spec", default="wan2.2-ti2v-5b")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--pair", required=True)
    ap.add_argument("--swap", required=True)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--frames", type=int, default=121)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=".cache/a2v_robotwin/gate5b_swap/metrics.json")
    ap.add_argument("--rl_lora", default=None,
                    help="Optional Flow-GRPO LoRA to stack on top of the SFT VACE checkpoint "
                         "(saved by scripts/train_a2v_grpo.py). Use this to check that RL did "
                         "not break the causal behaviour the SFT model already had.")
    ap.add_argument("--rl_lora_rank", type=int, default=32)
    args = ap.parse_args()

    pair_ds, swap_ds = Path(args.pair).resolve(), Path(args.swap).resolve()
    pair_rows, swap_rows = rows_of(pair_ds), rows_of(swap_ds)
    assert len(pair_rows) == len(swap_rows), "pair/swap row counts differ"

    spec = REGISTRY[args.base_spec]
    print(f"[gate5b] spec={args.base_spec} steps={args.steps} rows={len(pair_rows)}")
    pipe, spec = build_pipe_with_vace(args.base_spec, args.ckpt, 1.0)

    if args.rl_lora:
        # Stack the RL adapter on the SFT VACE branch: inject with the same target modules the
        # trainer used (spec-driven), then load the saved tensors. strict=False because the
        # checkpoint holds only `lora_*` keys, not the base weights.
        from peft import LoraConfig, inject_adapter_in_model
        from safetensors.torch import load_file

        targets = [t.strip() for t in spec.lora_target_modules.split(",") if t.strip()]
        pipe.vace = inject_adapter_in_model(
            LoraConfig(r=args.rl_lora_rank, lora_alpha=args.rl_lora_rank, target_modules=targets),
            pipe.vace,
        )
        sd = load_file(args.rl_lora)
        sd = {k: v.to(device=pipe.device, dtype=pipe.torch_dtype) for k, v in sd.items()}
        missing, unexpected = pipe.vace.load_state_dict(sd, strict=False)
        loaded = len(sd) - len(unexpected)
        print(f"[gate5b] RL LoRA: {loaded}/{len(sd)} tensors loaded from {args.rl_lora}")
        if unexpected:
            print(f"[gate5b] ! {len(unexpected)} unexpected keys, e.g. {unexpected[:2]} -- the "
                  f"adapter layout does not match; the gate would silently measure the SFT model")
            return 1
        if loaded == 0:
            print("[gate5b] ! nothing loaded -- refusing to report a result")
            return 1

    records = []
    for i, (pr, sr) in enumerate(zip(pair_rows, swap_rows)):
        print(f"\n=== row {i}: {pr['id']}  (swap uses {sr['id']}) ===")
        gt = _read_pngs(pair_ds, pr["video"], args.frames)
        gt_motion = _motion(gt)

        real = generate(pipe, spec, pair_ds, pr, "real", args)
        none = generate(pipe, spec, pair_ds, pr, "none", args)
        swap = generate(pipe, spec, swap_ds, sr, "real", args)   # swap dataset -> other traj

        rec = {
            "row": i, "id": pr["id"], "swap_id": sr["id"],
            "real_mae": _mae(real, gt), "none_mae": _mae(none, gt), "swap_mae": _mae(swap, gt),
            "real_motion": _motion(real), "none_motion": _motion(none),
            "swap_motion": _motion(swap), "gt_motion": gt_motion,
            "real_vs_swap": _mae(real, swap),
        }
        rec["real_lt_swap"] = rec["real_mae"] < rec["swap_mae"]
        rec["real_lt_none"] = rec["real_mae"] < rec["none_mae"]
        records.append(rec)
        print(f"  real={rec['real_mae']:.2f} none={rec['none_mae']:.2f} swap={rec['swap_mae']:.2f}"
              f" | real<swap={rec['real_lt_swap']} (Δ{rec['swap_mae']-rec['real_mae']:+.2f})"
              f" real<none={rec['real_lt_none']} (Δ{rec['none_mae']-rec['real_mae']:+.2f})")

    n = len(records)
    lt_swap = sum(r["real_lt_swap"] for r in records)
    lt_none = sum(r["real_lt_none"] for r in records)
    margin = float(np.mean([r["swap_mae"] - r["real_mae"] for r in records]))
    summary = {
        "rows": n, "steps": args.steps,
        "real_lt_swap": f"{lt_swap}/{n}", "real_lt_none": f"{lt_none}/{n}",
        "mean_real_mae": float(np.mean([r["real_mae"] for r in records])),
        "mean_none_mae": float(np.mean([r["none_mae"] for r in records])),
        "mean_swap_mae": float(np.mean([r["swap_mae"] for r in records])),
        "mean_swap_minus_real": margin,
        "reads_trajectory_content": lt_swap == n and margin > 0,
    }
    print("\n===== 5B swap gate =====")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print("\n  verdict: " + (
        "PASS -- the 5B model follows WHICH trajectory it is given"
        if summary["reads_trajectory_content"] else
        "FAIL -- real is not consistently better than a wrong-but-valid trajectory, so this "
        "checkpoint responds to the presence of a control signal more than to its content. "
        "RL on it would not learn action following."))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"records": records, "summary": summary}, open(out, "w"), indent=2)
    print(f"\n  written: {out}")
    return 0 if summary["reads_trajectory_content"] else 1


if __name__ == "__main__":
    sys.exit(main())
