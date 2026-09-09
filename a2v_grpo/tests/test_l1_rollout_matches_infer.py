"""L1: our ODE rollout must reproduce the stock `WanVideoPipeline.__call__` output.

This is what proves `a2v_grpo.rollout` really re-implements the *verified* inference path
(the one that produced WorldArena EWMScore 62.88) and did not silently skip a preprocessing
unit. We deliberately reuse `a2v.infer_a2v`'s own helpers so the two calls receive
byte-identical inputs by construction.

Run (5B, needs ~1 free GPU):

    cd /disk/worldmodel/wangshilong/DiffSynth-Studio
    CUDA_VISIBLE_DEVICES=6 A2V_MODELS_DIR=/disk/worldmodel/public_model \\
      /disk/worldmodel/uv/envs/a2v/bin/python -m a2v_grpo.tests.test_l1_rollout_matches_infer \\
        --base_spec wan2.2-ti2v-5b \\
        --ckpt /disk/worldmodel/ziyang/a2v_khl_v2/checkpoints/a2v_1k_uniform121/step-7400.safetensors \\
        --dataset /disk/worldmodel/ziyang/a2v_khl_v2/dataset_full \\
        --metadata metadata_1k_uniform121.jsonl \\
        --height 480 --width 832 --num_frames 121 --num_inference_steps 50
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
from a2v.infer_a2v import NEG_PROMPT, build_pipe_with_vace, first_frame_kwargs, load_frames
from a2v_grpo.rollout import a2v_rollout_with_logprob, prepare_a2v_inputs


def load_row(dataset: Path, metadata: str, row_idx: int) -> dict:
    path = dataset / metadata
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            if i == row_idx:
                return json.loads(line)
    raise IndexError(f"row {row_idx} not found in {path}")


def frames_to_array(video) -> np.ndarray:
    """Pipeline output -> uint8 array (T, H, W, 3), for either PIL frames or a tensor."""
    if isinstance(video, torch.Tensor):
        v = video.detach().float().cpu()
        if v.ndim == 5:
            v = v[0]
        if v.shape[0] in (3, 4) and v.shape[0] < v.shape[1]:  # (C, T, H, W)
            v = v.permute(1, 2, 3, 0)
        arr = (v.clamp(0, 1) * 255).round().numpy().astype(np.uint8)
        return arr
    return np.stack([np.asarray(f.convert("RGB"), dtype=np.uint8) for f in video])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_spec", default="wan2.2-ti2v-5b")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--ckpt_low", default=None)
    ap.add_argument("--lora_alpha", type=float, default=1.0)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--metadata", default="metadata.jsonl")
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--height", type=int, default=480)
    ap.add_argument("--width", type=int, default=832)
    ap.add_argument("--num_frames", type=int, default=121)
    ap.add_argument("--num_inference_steps", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tol_mae", type=float, default=1.0,
                    help="allowed mean abs pixel difference (0-255 scale)")
    ap.add_argument("--self_check", action="store_true",
                    help="also run the stock path TWICE to measure the model's own "
                         "non-determinism floor. Without this number a small-but-nonzero MAE "
                         "cannot be interpreted.")
    args = ap.parse_args()

    dataset = Path(args.dataset).resolve()
    row = load_row(dataset, args.metadata, args.row)
    spec = REGISTRY[args.base_spec]

    print(f"[L1] spec={args.base_spec} dim={spec.dim} first_frame_mode={spec.first_frame_mode}")
    print(f"[L1] row={args.row} id={row.get('id', '?')}")
    pipe, spec = build_pipe_with_vace(args.base_spec, args.ckpt, args.lora_alpha,
                                     ckpt_path_low=args.ckpt_low)

    vace_video = load_frames(dataset, row["vace_video"][: args.num_frames])
    ref = Image.open(dataset / row["vace_reference_image"]).convert("RGB")
    ff = first_frame_kwargs(spec, ref)
    print(f"[L1] first-frame kwarg: {list(ff)}  (ti2v_fused -> input_image, "
          f"vace_reference -> vace_reference_image)")

    common = dict(
        prompt=row.get("prompt", "robot arm manipulation"),
        negative_prompt=NEG_PROMPT,
        vace_video=vace_video,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        seed=args.seed,
        tiled=False,
        **ff,
    )

    # ---------- A: stock pipeline (the verified inference path) ----------
    print("[L1] A: stock pipe.__call__ ...")
    video_a = pipe(**common)
    arr_a = frames_to_array(video_a)

    # ---------- floor: stock vs stock, same seed ----------
    floor_mae = floor_max = None
    if args.self_check:
        print("[L1] A': stock pipe.__call__ again (non-determinism floor) ...")
        arr_a2 = frames_to_array(pipe(**common))
        d0 = np.abs(arr_a.astype(np.int16) - arr_a2.astype(np.int16))
        floor_mae, floor_max = float(d0.mean()), int(d0.max())
        print(f"[L1]    floor: MAE = {floor_mae:.4f}   max|diff| = {floor_max}")

    # ---------- B: our rollout, ODE mode (noise_level=0) ----------
    print("[L1] B: a2v_grpo rollout, noise_level=0 ...")
    prepared = prepare_a2v_inputs(pipe, **common)
    ish = prepared["inputs_shared"]
    print(f"[L1]    vace_context={tuple(ish['vace_context'].shape)} "
          f"vace_scale={ish.get('vace_scale')}  "
          f"first_frame_latents={'yes' if 'first_frame_latents' in ish else 'no'}  "
          f"latents={tuple(ish['latents'].shape)}")
    out = a2v_rollout_with_logprob(pipe, prepared, noise_level=0.0, decode=True)
    arr_b = frames_to_array(out["video"])

    # ---------- compare ----------
    print()
    if arr_a.shape != arr_b.shape:
        print(f"[L1] FAIL: shape mismatch A={arr_a.shape} B={arr_b.shape}")
        return 1
    diff = np.abs(arr_a.astype(np.int16) - arr_b.astype(np.int16))
    mae, mx = float(diff.mean()), int(diff.max())
    print(f"[L1] frames={arr_a.shape[0]} size={arr_a.shape[2]}x{arr_a.shape[1]}")
    print(f"[L1] pixel MAE = {mae:.4f}   max|diff| = {mx}   (tol MAE <= {args.tol_mae})")
    print(f"[L1] log_probs shape = {tuple(out['log_probs'].shape)}  "
          f"latents traj = {tuple(out['latents'].shape)}")
    print(f"[L1] logprob_mask = "
          f"{'frame0 masked' if out['logprob_mask'] is not None else 'none (no clamp)'}")

    print()
    if floor_mae is not None:
        # The only meaningful question: are we within the model's own run-to-run spread?
        ratio = mae / max(floor_mae, 1e-9)
        print(f"[L1] rollout-vs-stock MAE {mae:.4f}  /  stock-vs-stock floor {floor_mae:.4f}"
              f"  = {ratio:.2f}x")
        if floor_mae < 1e-6 and mae > 1e-3:
            print("[L1] FAIL: the stock path is bit-reproducible, so our non-zero MAE is a "
                  "REAL difference, not noise.")
            return 1
        if ratio > 3.0:
            print("[L1] FAIL: MAE is far above the non-determinism floor -> real divergence.")
            return 1
        print("[L1] PASS: rollout matches the stock path to within its own non-determinism")
        return 0

    if mae <= args.tol_mae:
        print("[L1] PASS (absolute tolerance) -- but rerun with --self_check: without the "
              "non-determinism floor a non-zero MAE cannot be interpreted.")
        return 0
    print("[L1] FAIL: rollout diverges from the stock path -- a preprocessing unit or the "
          "loop bookkeeping differs. Compare prepared['inputs_shared'] keys against "
          "wan_video.py:284-307.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
