#!/usr/bin/env python3
"""Validate prepared A2V metadata with DiffSynth UnifiedDataset."""

from __future__ import annotations

import argparse
from pathlib import Path

from diffsynth.core import UnifiedDataset

from a2v.data.operators import frame_list_video_operator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_base_path", required=True)
    parser.add_argument("--dataset_metadata_path", required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--num_frames", type=int, required=True)
    parser.add_argument("--max_items", type=int, default=10)
    return parser.parse_args()


def image_size(image) -> tuple[int, int]:
    return image.size[1], image.size[0]


def main() -> None:
    args = parse_args()
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
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4,
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
            if len(value) != args.num_frames:
                raise AssertionError(f"row {idx}: {key} has {len(value)} frames, expected {args.num_frames}")
            for frame_idx, frame in enumerate(value):
                if image_size(frame) != (args.height, args.width):
                    raise AssertionError(
                        f"row {idx}: {key}[{frame_idx}] size {image_size(frame)}, expected {(args.height, args.width)}"
                    )
        ref = data["vace_reference_image"]
        if not isinstance(ref, list) or len(ref) != 1:
            raise AssertionError(f"row {idx}: vace_reference_image should load as one-frame list")
        if image_size(ref[0]) != (args.height, args.width):
            raise AssertionError(f"row {idx}: reference size {image_size(ref[0])}, expected {(args.height, args.width)}")
        if not data.get("prompt"):
            raise AssertionError(f"row {idx}: empty prompt")
        print(
            f"row {idx}: ok video={len(data['video'])} vace={len(data['vace_video'])} "
            f"size={image_size(data['video'][0])} prompt={data['prompt']!r}"
        )
    print(f"Validated {checked} rows from {Path(args.dataset_metadata_path).resolve()}")


if __name__ == "__main__":
    main()
