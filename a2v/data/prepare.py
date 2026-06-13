#!/usr/bin/env python3
"""Prepare aligned A2V data for stock Wan VACE training.

Input is a JSON/JSONL manifest with one item per source clip. Each item should
contain at least:

    video, action_path, intrinsic_path, extrinsic_path

``original_size`` is optional. When provided, it is checked against the actual
source video size and any mismatch is treated as an alignment error.

The script writes lossless PNG sequences for both the target video and the VACE
trajectory control video. Metadata is emitted as JSONL because DiffSynth CSV
metadata cannot store frame-path lists.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import imageio
import numpy as np
from PIL import Image
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from a2v.render import (  # noqa: E402
    get_vace_traj_maps_with_scaled_intrinsic,
    load_actions_with_quat,
    load_camera_params,
    overlay_rgb,
    select_frame_indices,
    vace_tensor_to_uint8_frames,
)


def load_manifest(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        items = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "items" in data:
        data = data["items"]
    if not isinstance(data, list):
        raise ValueError("Manifest must be a list or a dict with an 'items' list")
    return data


def as_abs(path: str | Path, base: Path) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def parse_size(value: Any, *, field_name: str) -> tuple[int, int]:
    if value is None:
        raise ValueError(f"Missing {field_name}")
    if isinstance(value, str):
        value = json.loads(value)
    if isinstance(value, dict):
        return int(value["height"]), int(value["width"])
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"{field_name} must be [height, width] or an object with height/width")


def safe_name(item: dict[str, Any], index: int) -> str:
    raw = str(item.get("id") or item.get("uid") or item.get("name") or f"sample_{index:06d}")
    keep = []
    for ch in raw:
        keep.append(ch if ch.isalnum() or ch in "._-" else "_")
    return "".join(keep).strip("_") or f"sample_{index:06d}"


def get_reader_num_frames(path: Path) -> int:
    reader = imageio.get_reader(path)
    try:
        return int(reader.count_frames())
    finally:
        reader.close()


def _resize_frame(image: Image.Image, target_size: tuple[int, int], resize_mode: str) -> Image.Image:
    target_h, target_w = target_size
    if resize_mode == "stretch":
        return image.resize((target_w, target_h), resample=Image.BILINEAR)
    if resize_mode == "crop":
        width, height = image.size
        scale = max(target_w / width, target_h / height)
        image = image.resize((round(width * scale), round(height * scale)), resample=Image.BILINEAR)
        left = (image.size[0] - target_w) // 2
        top = (image.size[1] - target_h) // 2
        return image.crop((left, top, left + target_w, top + target_h))
    raise ValueError(f"resize_mode must be crop or stretch, got {resize_mode!r}")


def _read_video_frames_decord(path: Path, frame_indices: list[int], target_size: tuple[int, int], resize_mode: str) -> tuple[list[np.ndarray], tuple[int, int]] | None:
    try:
        from decord import VideoReader, cpu
    except ModuleNotFoundError:
        return None

    reader = VideoReader(str(path), ctx=cpu(0))
    batch = reader.get_batch(frame_indices).asnumpy()
    src_size = None
    frames = []
    for frame in batch:
        image = Image.fromarray(frame).convert("RGB")
        if src_size is None:
            src_size = (image.size[1], image.size[0])
        frames.append(np.array(_resize_frame(image, target_size, resize_mode)))
    if src_size is None:
        raise ValueError(f"No frames loaded from {path}")
    return frames, src_size


def read_video_frames(path: Path, frame_indices: list[int], target_size: tuple[int, int], resize_mode: str) -> tuple[list[np.ndarray], tuple[int, int]]:
    """Read source frames and return resized frames plus actual source ``(H, W)``.

    Source videos should be CFR and reliably random-seekable. ``decord`` is used
    when available; otherwise imageio is used as a fallback.
    """
    decord_result = _read_video_frames_decord(path, frame_indices, target_size, resize_mode)
    if decord_result is not None:
        return decord_result

    reader = imageio.get_reader(path)
    frames = []
    src_size = None
    try:
        for idx in frame_indices:
            frame = reader.get_data(idx)
            image = Image.fromarray(frame).convert("RGB")
            if src_size is None:
                src_size = (image.size[1], image.size[0])
            frames.append(np.array(_resize_frame(image, target_size, resize_mode)))
    finally:
        reader.close()
    if src_size is None:
        raise ValueError(f"No frames loaded from {path}")
    return frames, src_size


def save_png_sequence(frames: Iterable[np.ndarray], out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rel_names = []
    for i, frame in enumerate(frames):
        name = f"{i:05d}.png"
        path = out_dir / name
        Image.fromarray(frame).save(path, compress_level=0)
        rel_names.append(name)
    return rel_names


def save_overlay_sequence(target_frames: list[np.ndarray], control_frames: list[np.ndarray], out_dir: Path, alpha: float) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, (target, control) in enumerate(zip(target_frames, control_frames)):
        Image.fromarray(overlay_rgb(target, control, alpha=alpha)).save(out_dir / f"{i:05d}.png", compress_level=0)


def resolve_prompt(item: dict[str, Any], default_prompt: str) -> str:
    prompt = item.get("prompt")
    if prompt is None or (isinstance(prompt, float) and np.isnan(prompt)):
        return default_prompt
    return str(prompt)


def build_sample(
    item: dict[str, Any],
    index: int,
    args: argparse.Namespace,
    manifest_base: Path,
    output_root: Path,
) -> dict[str, Any]:
    sample_name = safe_name(item, index)
    sample_dir = output_root / sample_name
    target_dir = sample_dir / "video"
    vace_dir = sample_dir / "vace_video"
    overlay_dir = sample_dir / "overlay"
    reference_dir = sample_dir / "reference"

    video_path = as_abs(item["video"], manifest_base)
    action_path = as_abs(item["action_path"], manifest_base)
    intrinsic_path = as_abs(item["intrinsic_path"], manifest_base)
    extrinsic_path = as_abs(item["extrinsic_path"], manifest_base)
    target_size = (args.height, args.width)

    total_frames = int(item.get("total_frames") or get_reader_num_frames(video_path))
    if item.get("frame_indices") is not None:
        frame_indices = [int(i) for i in item["frame_indices"]]
        if len(frame_indices) != args.num_frames:
            raise ValueError(f"{sample_name}: frame_indices has {len(frame_indices)} frames, expected {args.num_frames}")
        if args.num_frames % 4 != 1:
            raise ValueError(f"num_frames must be 4n+1, got {args.num_frames}")
    else:
        start = item.get("start_frame", args.start_frame)
        frame_indices = select_frame_indices(total_frames, args.num_frames, start=start, stride=args.stride)

    target_frames, src_size = read_video_frames(video_path, frame_indices, target_size, args.resize_mode)
    manifest_size = parse_size(item.get("original_size"), field_name="original_size") if item.get("original_size") is not None else None
    if manifest_size is not None and tuple(manifest_size) != tuple(src_size):
        raise ValueError(f"{sample_name}: manifest original_size {manifest_size} != actual video size {src_size}")
    original_size = src_size
    target_rel = save_png_sequence(target_frames, target_dir)

    actions = load_actions_with_quat(action_path, frame_indices)
    intrinsic, c2w_seq = load_camera_params(intrinsic_path, extrinsic_path, frame_indices)
    if c2w_seq.dim() == 3:
        c2w = c2w_seq.unsqueeze(0)
    elif c2w_seq.dim() == 4:
        c2w = c2w_seq
    else:
        raise ValueError(f"{sample_name}: expected c2w shape [T,4,4] or [V,T,4,4], got {tuple(c2w_seq.shape)}")
    if c2w.shape[1] != args.num_frames:
        raise ValueError(f"{sample_name}: c2w has {c2w.shape[1]} frames, expected {args.num_frames}")
    if not torch.allclose(c2w[..., 3, :], torch.tensor([0, 0, 0, 1], dtype=c2w.dtype, device=c2w.device), atol=1e-4):
        raise ValueError(f"{sample_name}: extrinsics must be valid 4x4 c2w transforms with last row [0,0,0,1]")
    # Extrinsics are interpreted as camera-to-world (c2w), matching ABot. If a
    # dataset stores w2c, convert it before preparing A2V metadata.
    w2c = torch.linalg.inv(c2w)

    trajs = get_vace_traj_maps_with_scaled_intrinsic(
        actions,
        w2c,
        c2w,
        intrinsic,
        original_size,
        target_size,
        resize_mode=args.resize_mode,
        gripper_z_offset=args.gripper_z_offset,
        radius_mode=args.radius_mode,
        phys_radius_m=args.phys_radius_m,
        radius_min_px=args.radius_min_px,
        radius_max_frac=args.radius_max_frac,
    )
    vace_frames = vace_tensor_to_uint8_frames(trajs, camera_index=args.camera_index)
    vace_rel = save_png_sequence(vace_frames, vace_dir)

    reference_dir.mkdir(parents=True, exist_ok=True)
    reference_name = "00000.png"
    Image.fromarray(target_frames[0]).save(reference_dir / reference_name, compress_level=0)

    if args.write_overlay:
        save_overlay_sequence(target_frames, vace_frames, overlay_dir, args.overlay_alpha)

    metadata = {
        "video": [str(Path(sample_name) / "video" / name) for name in target_rel],
        "prompt": resolve_prompt(item, args.default_prompt),
        "vace_video": [str(Path(sample_name) / "vace_video" / name) for name in vace_rel],
        "vace_reference_image": str(Path(sample_name) / "reference" / reference_name),
        "frame_indices": frame_indices,
        "source_video": str(video_path),
        "action_path": str(action_path),
        "intrinsic_path": str(intrinsic_path),
        "extrinsic_path": str(extrinsic_path),
        "resize_mode": args.resize_mode,
        "original_size": list(original_size),
        "target_size": list(target_size),
        "gripper_z_offset": args.gripper_z_offset,
        "radius_mode": args.radius_mode,
        "phys_radius_m": args.phys_radius_m,
        "radius_min_px": args.radius_min_px,
        "radius_max_frac": args.radius_max_frac,
    }
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="Input JSON/JSONL manifest.")
    parser.add_argument("--output_dir", required=True, help="Output dataset directory.")
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--num_frames", type=int, default=49, help="Must satisfy 4n+1.")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--start_frame", type=int, default=None)
    parser.add_argument("--resize_mode", choices=("crop", "stretch"), default="crop", help="Use crop to match DiffSynth ImageCropAndResize.")
    parser.add_argument("--camera_index", type=int, default=0)
    parser.add_argument("--gripper_z_offset", type=float, default=0.23,
                        help="EE +z offset (m) before projection. 0.23=ABot/AgiBot wrist->tip; use 0.0 for TCP-frame datasets (e.g. RoboTwin).")
    parser.add_argument("--radius_mode", choices=("physical", "normalized", "constant"), default="physical",
                        help="EE dot sizing: physical=perspective f_eff*R/z_cam (default); normalized=corrected min-max distance ramp; constant=50px.")
    parser.add_argument("--phys_radius_m", type=float, default=0.04,
                        help="Physical EE radius (m) for radius_mode=physical.")
    parser.add_argument("--radius_min_px", type=float, default=3.0,
                        help="Lower clamp on EE dot radius (compression-robust floor).")
    parser.add_argument("--radius_max_frac", type=float, default=1.0 / 6.0,
                        help="Upper clamp on EE dot radius as a fraction of frame height.")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--default_prompt", default="robot arm action trajectory")
    parser.add_argument("--write_overlay", action="store_true")
    parser.add_argument("--overlay_alpha", type=float, default=0.45)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_frames % 4 != 1:
        raise SystemExit(f"--num_frames must satisfy 4n+1, got {args.num_frames}")
    if args.stride <= 0:
        raise SystemExit("--stride must be positive")
    if args.height % 16 != 0 or args.width % 16 != 0:
        raise SystemExit(f"--height/--width must be multiples of 16, got {args.height}x{args.width}")

    manifest_path = Path(args.manifest).resolve()
    manifest_base = manifest_path.parent
    output_root = Path(args.output_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    items = load_manifest(manifest_path)
    if args.max_samples is not None:
        items = items[: args.max_samples]
    if not items:
        raise SystemExit("Manifest is empty")

    metadata_path = output_root / "metadata.jsonl"
    rows = []
    for idx, item in enumerate(items):
        row = build_sample(item, idx, args, manifest_base, output_root)
        rows.append(row)
        print(f"[{idx + 1}/{len(items)}] wrote {row['video'][0].split('/')[0]} with {len(row['frame_indices'])} frames")

    with metadata_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote metadata: {metadata_path}")


if __name__ == "__main__":
    main()
