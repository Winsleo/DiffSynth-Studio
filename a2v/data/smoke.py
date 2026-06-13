#!/usr/bin/env python3
"""End-to-end smoke tests for offline A2V dataset preparation."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import imageio.v2 as imageio
import numpy as np


def make_synthetic_inputs(root: Path, num_frames: int, height: int, width: int, *, bad_original_size: bool = False) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    video_path = root / "source.mp4"
    frames = []
    for i in range(num_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[..., 0] = np.linspace(20, 180, width, dtype=np.uint8)[None, :]
        frame[..., 1] = np.uint8(30 + i * 7)
        frame[..., 2] = np.linspace(180, 20, height, dtype=np.uint8)[:, None]
        frames.append(frame)
    imageio.mimsave(video_path, frames, fps=8)

    action_path = root / "actions.npy"
    actions = np.zeros((num_frames, 16), dtype=np.float32)
    for i in range(num_frames):
        z = 1.0
        actions[i, 0:3] = [-0.1 + 0.01 * i, 0.0, z]
        actions[i, 3:7] = [0.0, 0.0, 0.0, 1.0]
        actions[i, 7] = 30 + i
        actions[i, 8:11] = [0.1 - 0.01 * i, 0.0, z]
        actions[i, 11:15] = [0.0, 0.0, 0.0, 1.0]
        actions[i, 15] = 90 - i
    np.save(action_path, actions)

    intrinsic_path = root / "intrinsic.json"
    intrinsic_path.write_text(
        json.dumps({"intrinsic": {"fx": 240.0, "fy": 240.0, "ppx": width / 2, "ppy": height / 2}}),
        encoding="utf-8",
    )
    extrinsic_path = root / "extrinsic.json"
    extrinsics = []
    for _ in range(num_frames):
        extrinsics.append(
            {
                "extrinsic": {
                    "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                    "translation_vector": [0, 0, 0],
                }
            }
        )
    extrinsic_path.write_text(json.dumps(extrinsics), encoding="utf-8")

    manifest_path = root / "manifest.jsonl"
    original_size = [height + 16, width] if bad_original_size else [height, width]
    row = {
        "id": root.name,
        "video": str(video_path),
        "action_path": str(action_path),
        "intrinsic_path": str(intrinsic_path),
        "extrinsic_path": str(extrinsic_path),
        "original_size": original_size,
        "prompt": "synthetic robot trajectory",
    }
    manifest_path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return manifest_path


def run(cmd: list[str], cwd: Path) -> None:
    print("+", " ".join(cmd))
    subprocess.run(cmd, cwd=cwd, check=True)


def run_expect_failure(cmd: list[str], cwd: Path, expected: str) -> None:
    print("+ expect-fail", " ".join(cmd))
    result = subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if result.returncode == 0:
        raise AssertionError("Command unexpectedly succeeded")
    if expected not in result.stdout:
        raise AssertionError(f"Expected {expected!r} in failure output, got:\n{result.stdout}")
    print(result.stdout.strip().splitlines()[-1])


def run_case(repo: Path, work: Path, name: str, source_size: tuple[int, int], target_size: tuple[int, int], num_frames: int) -> None:
    source = work / name / "source"
    out = work / name / "prepared"
    manifest = make_synthetic_inputs(source, num_frames=num_frames, height=source_size[0], width=source_size[1])
    run(
        [
            sys.executable,
            "-m",
            "a2v.data.prepare",
            "--manifest",
            str(manifest),
            "--output_dir",
            str(out),
            "--height",
            str(target_size[0]),
            "--width",
            str(target_size[1]),
            "--num_frames",
            str(num_frames),
            "--resize_mode",
            "crop",
            "--write_overlay",
        ],
        repo,
    )
    run(
        [
            sys.executable,
            "-m",
            "a2v.data.validate",
            "--dataset_base_path",
            str(out),
            "--dataset_metadata_path",
            str(out / "metadata.jsonl"),
            "--height",
            str(target_size[0]),
            "--width",
            str(target_size[1]),
            "--num_frames",
            str(num_frames),
        ],
        repo,
    )
    print(f"Smoke case {name} written to {out}")


def main() -> None:
    repo = Path(__file__).resolve().parents[2]
    work = repo / ".cache" / "a2v_dataset_smoke"
    if work.exists():
        shutil.rmtree(work)
    num_frames = 9

    run_case(repo, work, "same_size", source_size=(96, 160), target_size=(96, 160), num_frames=num_frames)
    run_case(repo, work, "scaled_crop", source_size=(128, 224), target_size=(64, 112), num_frames=num_frames)

    bad_source = work / "bad_original_size" / "source"
    bad_out = work / "bad_original_size" / "prepared"
    bad_manifest = make_synthetic_inputs(bad_source, num_frames=num_frames, height=96, width=160, bad_original_size=True)
    run_expect_failure(
        [
            sys.executable,
            "-m",
            "a2v.data.prepare",
            "--manifest",
            str(bad_manifest),
            "--output_dir",
            str(bad_out),
            "--height",
            "96",
            "--width",
            "160",
            "--num_frames",
            str(num_frames),
        ],
        repo,
        "manifest original_size",
    )
    print(f"Smoke data written to {work}")


if __name__ == "__main__":
    main()
