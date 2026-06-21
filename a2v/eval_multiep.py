#!/usr/bin/env python3
"""Evaluate A2V multi-episode generalization on a held-out prepared dataset.

Loads the model once, generates requested controls for selected metadata rows, writes
mp4s, and reports per-episode plus aggregate causal metrics.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

from diffsynth.utils.data import save_video

from a2v.causal_metrics import _mae, _motion, _read_pngs
from a2v.infer_a2v import build_pipe_with_vace, generate_one


def _read_rows(dataset: Path) -> list[dict]:
    rows = []
    with (dataset / "metadata.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No rows in {dataset / 'metadata.jsonl'}")
    return rows


def _parse_rows(spec: str | None, total: int) -> list[int]:
    if spec is None or spec == "":
        return list(range(total))
    out: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo_s, hi_s = part.split("-", 1)
            lo, hi = int(lo_s), int(hi_s)
            if hi < lo:
                raise ValueError(f"Invalid row range: {part}")
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    bad = [i for i in out if i < 0 or i >= total]
    if bad:
        raise ValueError(f"Row indices out of range 0..{total - 1}: {bad}")
    return out


def _controls(spec: str) -> list[str]:
    allowed = {"real", "none", "shuffle"}
    controls = [p.strip() for p in spec.split(",") if p.strip()]
    if not controls:
        raise ValueError("--controls must not be empty")
    bad = [c for c in controls if c not in allowed]
    if bad:
        raise ValueError(f"Unknown controls {bad}; allowed={sorted(allowed)}")
    return controls


def _episode_id(row: dict, row_idx: int) -> str:
    for key in ("source_video", "vace_reference_image"):
        value = row.get(key)
        if not value:
            continue
        match = re.search(r"episode(\d+)", str(value))
        if match:
            return f"ep{int(match.group(1))}"
    return f"row{row_idx}"


def _video_to_array(video, height: int, width: int) -> np.ndarray:
    frames = []
    for frame in video:
        if isinstance(frame, Image.Image):
            im = frame.convert("RGB")
        else:
            im = Image.fromarray(np.asarray(frame)).convert("RGB")
        if im.size != (width, height):
            im = im.resize((width, height))
        frames.append(np.asarray(im, dtype=np.uint8))
    if not frames:
        raise ValueError("Generated video has no frames")
    return np.stack(frames)


def _mean(values: Iterable[float]) -> float:
    vals = list(values)
    return float(np.mean(vals)) if vals else float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_spec", required=True)
    parser.add_argument("--lora", required=True)
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--rows", default=None, help="Comma/range row selector, e.g. 0,1,2 or 0-4. Default: all rows.")
    parser.add_argument("--num_frames", type=int, default=105)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--controls", default="real,none")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output_dir", default=".cache/a2v_robotwin/eval_multiep")
    parser.add_argument("--fps", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = Path(args.dataset).resolve()
    rows = _read_rows(dataset)
    row_indices = _parse_rows(args.rows, len(rows))
    controls = _controls(args.controls)
    if "real" not in controls or "none" not in controls:
        raise ValueError("Multi-episode gate requires at least controls real,none")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model once: base_spec={args.base_spec} lora={args.lora}")
    pipe, spec = build_pipe_with_vace(args.base_spec, args.lora, args.lora_alpha)

    records = []
    failures = []
    for row_idx in row_indices:
        row = rows[row_idx]
        ep = _episode_id(row, row_idx)
        print(f"\n=== row={row_idx} episode={ep} prompt={row.get('prompt', '')!r} ===")
        try:
            gt = _read_pngs(dataset, row["video"], args.num_frames)
            gt_motion = _motion(gt)
            generated: dict[str, np.ndarray] = {}

            for control in controls:
                video = generate_one(
                    pipe=pipe,
                    spec=spec,
                    row=row,
                    dataset=dataset,
                    control=control,
                    height=args.height,
                    width=args.width,
                    num_frames=args.num_frames,
                    seed=args.seed,
                )
                video_frames = list(video)
                out = output_dir / f"row{row_idx:03d}_{ep}_{control}.mp4"
                save_video(video_frames, str(out), fps=args.fps, quality=5)
                arr = _video_to_array(video_frames, gt.shape[1], gt.shape[2])
                generated[control] = arr
                print(f"saved {control:7} -> {out}")

            real = generated["real"]
            none = generated["none"]
            rec = {
                "row": row_idx,
                "episode": ep,
                "real_mae": _mae(real, gt),
                "none_mae": _mae(none, gt),
                "real_motion": _motion(real),
                "none_motion": _motion(none),
                "gt_motion": gt_motion,
                "real_vs_none": _mae(real, none),
            }
            rec["real_lt_none"] = rec["real_mae"] < rec["none_mae"]
            rec["real_motion_ratio"] = rec["real_motion"] / gt_motion if gt_motion > 0 else float("nan")
            records.append(rec)
        except Exception as e:  # one bad episode must not abort the whole eval
            print(f"[WARN] row={row_idx} ep={ep} FAILED: {e}")
            failures.append({"row": row_idx, "episode": ep, "error": str(e)})
            continue

    print("\nrow episode real_mae none_mae real<none real_motion none_motion gt_motion motion_ratio real_vs_none")
    for rec in records:
        print(
            f"{rec['row']:3d} {rec['episode']:>7} "
            f"{rec['real_mae']:8.2f} {rec['none_mae']:8.2f} "
            f"{str(rec['real_lt_none']):>9} "
            f"{rec['real_motion']:11.2f} {rec['none_motion']:11.2f} {rec['gt_motion']:9.2f} "
            f"{rec['real_motion_ratio']:12.2f} {rec['real_vs_none']:12.2f}"
        )

    total = len(records)
    real_lt_count = sum(1 for r in records if r["real_lt_none"])
    # motion ratio averaged over FINITE values only (gt_motion==0 -> nan, must not poison the mean)
    finite_ratios = [r["real_motion_ratio"] for r in records if np.isfinite(r["real_motion_ratio"])]
    mean_ratio = _mean(finite_ratios)  # nan if no finite ratio
    summary = {
        "rows": total,
        "requested_rows": len(row_indices),
        "failed": len(failures),
        "mean_real_mae": _mean(r["real_mae"] for r in records),
        "mean_none_mae": _mean(r["none_mae"] for r in records),
        "mean_real_motion": _mean(r["real_motion"] for r in records),
        "mean_none_motion": _mean(r["none_motion"] for r in records),
        "mean_gt_motion": _mean(r["gt_motion"] for r in records),
        "mean_real_vs_none": _mean(r["real_vs_none"] for r in records),
        "real_lt_none_count": real_lt_count,
        "real_lt_none_frac": real_lt_count / total if total else float("nan"),
        "mean_real_motion_ratio": mean_ratio,
    }
    if total == 0:
        # no successful rows -> cannot judge generalization
        pass_gate = False
    else:
        # nan motion ratio (all GT static) skips the motion sub-check; primary criteria stand.
        ratio_ok = (not np.isfinite(mean_ratio)) or (0.5 <= mean_ratio <= 2.0)
        pass_gate = (
            summary["mean_real_mae"] < summary["mean_none_mae"]
            and real_lt_count >= min(total, 8)
            and ratio_ok
            and summary["mean_real_vs_none"] > 1.0
        )
    print(
        "\nSUMMARY "
        f"rows={summary['rows']} "
        f"failed={summary['failed']} "
        f"mean_real_mae={summary['mean_real_mae']:.2f} "
        f"mean_none_mae={summary['mean_none_mae']:.2f} "
        f"real_lt_none={summary['real_lt_none_count']}/{summary['rows']} "
        f"mean_real_motion={summary['mean_real_motion']:.2f} "
        f"mean_none_motion={summary['mean_none_motion']:.2f} "
        f"mean_gt_motion={summary['mean_gt_motion']:.2f} "
        f"mean_motion_ratio={summary['mean_real_motion_ratio']:.2f} "
        f"mean_real_vs_none={summary['mean_real_vs_none']:.2f} "
        f"verdict={'PASS' if pass_gate else 'FAIL'}"
    )
    if failures:
        print(f"[WARN] {len(failures)} row(s) failed: {[f['row'] for f in failures]}")

    metrics_path = output_dir / "metrics.json"
    metrics_path.write_text(
        json.dumps({"records": records, "failures": failures, "summary": summary, "pass": pass_gate}, indent=2),
        encoding="utf-8",
    )
    print(f"metrics -> {metrics_path}")


if __name__ == "__main__":
    main()
