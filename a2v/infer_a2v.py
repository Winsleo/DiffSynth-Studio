#!/usr/bin/env python3
"""A2V overfit-gate inference: reconstruct an episode from its control video.

Loads the base Wan2.1-VACE-1.3B locally, applies a trained VACE LoRA, feeds the
prepared ``vace_video`` (trajectory-map PNGs) + ``vace_reference_image`` + prompt
for one dataset row, and generates a video at the SAME resolution/length used in
training. The gate passes when the generated arm follows the trajectory the
control encodes (compare to the GT ``video/`` frames).

Usage:
    .venv/bin/python -m a2v.infer_a2v \
        --dataset .cache/a2v_robotwin/ep0_dataset \
        --lora models/train/a2v_robotwin_ep0_lora/epoch-9.safetensors \
        --num_frames 121 --height 240 --width 320 --row 0 \
        --output .cache/a2v_robotwin/ep0_gen.mp4

Add ``--control none`` or ``--control shuffle`` for the negative control: with no
/ scrambled trajectory the output should NOT reconstruct the GT, proving the
control signal is what drives generation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.utils.data import save_video

from a2v.base_spec import get_spec
from a2v.provision import provision_a2v

REPO_ROOT = Path(__file__).resolve().parents[1]

NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，"
    "畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def load_row(dataset: Path, row_idx: int) -> dict:
    rows = []
    with (dataset / "metadata.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[row_idx]


def load_frames(dataset: Path, rel_paths: list[str]) -> list[Image.Image]:
    return [Image.open(dataset / p).convert("RGB") for p in rel_paths]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base_spec", default="wan2.1-vace-1.3b", help="WanBaseSpec name (a2v.base_spec.REGISTRY).")
    parser.add_argument("--lora", required=True, help="Trained VACE LoRA safetensors.")
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--control", choices=("real", "none", "shuffle"), default="real",
                        help="real=use prepared vace_video; none/shuffle=negative control.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    dataset = Path(args.dataset).resolve()
    row = load_row(dataset, args.row)

    vace_video = load_frames(dataset, row["vace_video"][: args.num_frames])
    if args.control == "none":
        vace_video = [Image.new("RGB", (args.width, args.height), (128, 128, 128)) for _ in vace_video]
    elif args.control == "shuffle":
        idx = list(range(len(vace_video)))[::-1]  # deterministic scramble: reverse
        vace_video = [vace_video[i] for i in idx]
    ref = Image.open(dataset / row["vace_reference_image"]).convert("RGB")

    spec = get_spec(args.base_spec)
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[ModelConfig(path=p) for p in spec.model_paths()],
        tokenizer_config=ModelConfig(spec.abs_tokenizer_path()),
    )
    # T2 seams: ensure VACE branch (SEAM-1) + install mask_pq-parameterized unit (SEAM-2).
    vace = provision_a2v(pipe, spec)
    pipe.load_lora(vace, args.lora, alpha=args.lora_alpha)

    video = pipe(
        prompt=row.get("prompt", "robot arm manipulation"),
        negative_prompt=NEG_PROMPT,
        vace_video=vace_video,
        vace_reference_image=ref,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        seed=args.seed,
        tiled=False,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_video(video, str(out), fps=args.fps, quality=5)
    print(f"Saved generated video -> {out} (control={args.control}, prompt={row.get('prompt')!r})")


if __name__ == "__main__":
    main()
