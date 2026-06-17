#!/usr/bin/env python3
"""Causal-gate metrics for an A2V overfit run (reusable across T1/T3/T4/...).

Compares the generated videos under different controls against the GT episode:
  * MAE-GT      mean |pixel(gen) - pixel(GT)| over all frames (lower = closer to GT).
  * motion      mean |frame[t] - frame[t-1]| (temporal energy; ~0 => static arm).
  * mean        mean pixel value (sanity for global brightness drift).
  * real-vs-none mean |pixel(real) - pixel(none)| (the control's causal footprint).

Gate passes when REAL reproduces the GT trajectory (low MAE-GT, motion ≈ GT) while
NONE does NOT (motion ≈ 0 / high MAE-GT), and real-vs-none is clearly non-zero.

Usage:
    .venv/bin/python -m a2v.causal_metrics \
        --dataset .cache/a2v_robotwin/ep0_dataset_phys \
        --real .cache/a2v_robotwin/gen_i2v_real.mp4 \
        --none .cache/a2v_robotwin/gen_i2v_none.mp4 \
        --num_frames 121
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
import imageio.v2 as imageio


def _read_video(path: str, n: int, hw) -> np.ndarray:
    h, w = hw
    out = []
    reader = imageio.get_reader(path)  # ffmpeg backend
    for i, f in enumerate(reader):
        if i >= n:
            break
        im = Image.fromarray(np.asarray(f)).convert("RGB").resize((w, h))
        out.append(np.asarray(im, dtype=np.uint8))
    reader.close()
    return np.stack(out)


def _read_pngs(dataset: Path, rel_paths, n: int) -> np.ndarray:
    return np.stack([np.asarray(Image.open(dataset / p).convert("RGB"), dtype=np.uint8)
                     for p in rel_paths[:n]])


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    m = min(len(a), len(b))
    return float(np.abs(a[:m].astype(np.int32) - b[:m].astype(np.int32)).mean())


def _motion(a: np.ndarray) -> float:
    if len(a) < 2:
        return 0.0
    return float(np.abs(np.diff(a.astype(np.int32), axis=0)).mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--real", required=True)
    ap.add_argument("--none", required=True)
    ap.add_argument("--row", type=int, default=0)
    ap.add_argument("--num_frames", type=int, default=121)
    ap.add_argument("--height", type=int, default=None,
                    help="GT/compare frame height; default auto-derived from the GT PNGs.")
    ap.add_argument("--width", type=int, default=None,
                    help="GT/compare frame width; default auto-derived from the GT PNGs.")
    args = ap.parse_args()

    dataset = Path(args.dataset).resolve()
    row = json.loads((dataset / "metadata.jsonl").read_text().splitlines()[args.row])

    # Resolution is defined by the GT frames. Read them first, then read the generated
    # videos at the GT's (H, W) so MAE never hits a shape mismatch across tracks (T5 GT is
    # 256x320, earlier tracks 240x320). Explicit --height/--width still override.
    gt = _read_pngs(dataset, row["video"], args.num_frames)
    h = args.height if args.height is not None else gt.shape[1]
    w = args.width if args.width is not None else gt.shape[2]
    hw = (h, w)
    real = _read_video(args.real, args.num_frames, hw)
    none = _read_video(args.none, args.num_frames, hw)

    rows = [
        ("GT",   None,            _motion(gt),   float(gt.mean())),
        ("REAL", _mae(real, gt),  _motion(real), float(real.mean())),
        ("NONE", _mae(none, gt),  _motion(none), float(none.mean())),
    ]
    print(f"{'video':5} {'MAE-GT':>8} {'motion':>8} {'mean':>8}  frames")
    for name, mae, mot, mean in rows:
        mstr = f"{mae:8.2f}" if mae is not None else f"{'--':>8}"
        n = {"GT": len(gt), "REAL": len(real), "NONE": len(none)}[name]
        print(f"{name:5} {mstr} {mot:8.2f} {mean:8.2f}  {n}")
    print(f"\nreal-vs-none MAE = {_mae(real, none):.2f}")

    # Verdict heuristic (informational; final call is visual + these numbers in context).
    real_mae, real_mot = rows[1][1], rows[1][2]
    none_mot = rows[2][2]
    gt_mot = rows[0][2]
    print("\n--- gate read ---")
    print(f"REAL reproduces GT motion? real_motion={real_mot:.2f} vs GT={gt_mot:.2f} "
          f"(ratio {real_mot / gt_mot:.2f})")
    print(f"NONE static?               none_motion={none_mot:.2f} "
          f"({'STATIC' if none_mot < 0.5 * gt_mot else 'MOVING'})")
    print(f"REAL closer to GT than NONE? MAE-GT real={real_mae:.2f} vs none={rows[2][1]:.2f}")


if __name__ == "__main__":
    main()
