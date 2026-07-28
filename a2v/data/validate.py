#!/usr/bin/env python3
"""Validate prepared A2V metadata with DiffSynth UnifiedDataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from diffsynth.core import UnifiedDataset

from a2v.base_spec import get_spec
from a2v.data.operators import frame_list_video_operator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_base_path", required=True)
    parser.add_argument("--dataset_metadata_path", required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--num_frames", type=int, required=True)
    parser.add_argument("--max_num_frames", type=int, default=None,
                        help="Variable-length dataset: instead of requiring exactly --num_frames per row, "
                             "accept any 4n+1 length <= this cap (and video/vace_video must match each other).")
    parser.add_argument("--base_spec", default=None, help="Optional WanBaseSpec name; derives spatial/time divisibility factors.")
    parser.add_argument("--max_items", type=int, default=10)
    parser.add_argument("--check_action", action="store_true",
                        help="Scheme A: also verify each row's action .npy exists and is [T,16] frame-locked to video.")
    return parser.parse_args()


def image_size(image) -> tuple[int, int]:
    return image.size[1], image.size[0]


def main() -> None:
    args = parse_args()
    spec = get_spec(args.base_spec) if args.base_spec else None
    height_division_factor = spec.vae_spatial_factor * spec.patch_size[1] if spec is not None else 16
    width_division_factor = spec.vae_spatial_factor * spec.patch_size[2] if spec is not None else 16
    time_division_factor = spec.vae_temporal_factor if spec is not None else 4
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=1,
        data_file_keys=("video", "vace_video", "vace_reference_image"),
        main_data_operator=frame_list_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=None,
            height=args.height,
            width=args.width,
            height_division_factor=height_division_factor,
            width_division_factor=width_division_factor,
            num_frames=args.num_frames,
            time_division_factor=time_division_factor,
            time_division_remainder=1,
        ),
    )
    if len(dataset) == 0:
        raise SystemExit("Dataset has no rows")

    checked = min(len(dataset), args.max_items)
    for idx in range(checked):
        data = dataset[idx]
        for key in ("video", "vace_video"):
            value = data[key]
            if not isinstance(value, list):
                raise AssertionError(f"row {idx}: {key} is {type(value)}, expected list")
            if args.max_num_frames is None:
                if len(value) != args.num_frames:
                    raise AssertionError(f"row {idx}: {key} has {len(value)} frames, expected {args.num_frames}")
            else:
                if len(value) % time_division_factor != 1:
                    raise AssertionError(
                        f"row {idx}: {key} has {len(value)} frames, must be {time_division_factor}n+1")
                if len(value) > args.max_num_frames:
                    raise AssertionError(
                        f"row {idx}: {key} has {len(value)} frames, exceeds cap {args.max_num_frames}")
            for frame_idx, frame in enumerate(value):
                if image_size(frame) != (args.height, args.width):
                    raise AssertionError(
                        f"row {idx}: {key}[{frame_idx}] size {image_size(frame)}, expected {(args.height, args.width)}"
                    )
        if len(data["video"]) != len(data["vace_video"]):
            raise AssertionError(
                f"row {idx}: video has {len(data['video'])} frames but vace_video has "
                f"{len(data['vace_video'])} - the two streams must stay frame-locked")
        ref = data["vace_reference_image"]
        if not isinstance(ref, list) or len(ref) != 1:
            raise AssertionError(f"row {idx}: vace_reference_image should load as one-frame list")
        if image_size(ref[0]) != (args.height, args.width):
            raise AssertionError(f"row {idx}: reference size {image_size(ref[0])}, expected {(args.height, args.width)}")
        if not data.get("prompt"):
            raise AssertionError(f"row {idx}: empty prompt")
        if args.check_action:
            import numpy as np
            action_rel = data.get("action")
            if not action_rel:
                raise AssertionError(f"row {idx}: --check_action set but row has no 'action' key")
            action_arr = np.load(Path(args.dataset_base_path) / action_rel)
            if action_arr.ndim != 2 or action_arr.shape[1] != 16:
                raise AssertionError(f"row {idx}: action shape {action_arr.shape}, expected [T,16]")
            if action_arr.shape[0] != len(data["video"]):
                raise AssertionError(
                    f"row {idx}: action has {action_arr.shape[0]} frames but video has "
                    f"{len(data['video'])} - action must stay frame-locked to video")
        print(
            f"row {idx}: ok video={len(data['video'])} vace={len(data['vace_video'])} "
            f"size={image_size(data['video'][0])} prompt={data['prompt']!r}"
        )
    print(f"Validated {checked} rows from {Path(args.dataset_metadata_path).resolve()}")


if __name__ == "__main__":
    main()
