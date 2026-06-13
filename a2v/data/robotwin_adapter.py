#!/usr/bin/env python3
"""Convert RoboTwin2.0 episodes into an A2V manifest for ``a2v.data.prepare``.

RoboTwin stores, per episode, a head-camera ``.mp4`` plus an ``.hdf5`` holding
bimanual end-effector poses and per-frame camera parameters. This adapter reads
the hdf5 and emits, for each episode:

* ``actions.npy``    ``[T, 16]`` in traj_map's
  ``left_xyz, left_xyzw, left_gripper, right_xyz, right_xyzw, right_gripper`` layout
* ``intrinsic.npy``  ``[3, 3]`` OpenCV K (constant in clean mode)
* ``extrinsic.npy``  ``[T, 4, 4]`` camera-to-world (c2w), derived by inverting
  RoboTwin's ``extrinsic_cv`` (which is world-to-camera ``[R|t]`` in OpenCV).

plus a ``manifest.jsonl`` row per episode pointing at the original ``.mp4``.

Conventions (verified against the data):
* ``endpose`` rows are ``[x, y, z, qx, qy, qz, qw]`` — already TCP-framed, so
  prepare.py must run with ``--gripper_z_offset 0`` (the ABot 0.23 wrist->tip
  offset does NOT apply to RoboTwin).
* grippers are in ``[0, 1]`` and rescaled by ``*120`` so traj_map's ``/120``
  colormap spans its full range (purely cosmetic — affects dot colour only).
* ``intrinsic_cv`` / ``extrinsic_cv`` (OpenCV) are mutually consistent and match
  traj_map's projection; the OpenGL ``cam2world_gl`` is intentionally unused.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

GRIPPER_COLOR_SCALE = 120.0  # RoboTwin gripper in [0,1] -> traj_map's /120 colormap range


def _resolve_dataset_dir(root: Path, task: str, robot_mode: str) -> Path:
    path = root / task / robot_mode
    if not path.is_dir():
        raise SystemExit(f"Dataset dir not found: {path}")
    return path


def parse_episodes(spec: str | None, episodes_range: str | None, data_dir: Path) -> list[int]:
    if spec:
        return [int(x) for x in spec.split(",") if x.strip() != ""]
    if episodes_range:
        lo, hi = episodes_range.split("-")
        return list(range(int(lo), int(hi) + 1))
    # default: every episode present in data/
    eps = []
    for p in sorted(data_dir.glob("episode*.hdf5")):
        eps.append(int(p.stem.replace("episode", "")))
    return sorted(eps)


def build_actions(f: h5py.File) -> np.ndarray:
    le = np.asarray(f["endpose/left_endpose"], dtype=np.float32)    # [T,7]
    re = np.asarray(f["endpose/right_endpose"], dtype=np.float32)
    lg = np.asarray(f["endpose/left_gripper"], dtype=np.float32)    # [T]
    rg = np.asarray(f["endpose/right_gripper"], dtype=np.float32)
    T = le.shape[0]
    actions = np.zeros((T, 16), dtype=np.float32)
    actions[:, 0:3] = le[:, 0:3]
    actions[:, 3:7] = le[:, 3:7]                 # qx,qy,qz,qw
    actions[:, 7] = lg * GRIPPER_COLOR_SCALE
    actions[:, 8:11] = re[:, 0:3]
    actions[:, 11:15] = re[:, 3:7]
    actions[:, 15] = rg * GRIPPER_COLOR_SCALE
    return actions


def build_camera(f: h5py.File, camera: str) -> tuple[np.ndarray, np.ndarray]:
    K = np.asarray(f[f"observation/{camera}/intrinsic_cv"][0], dtype=np.float32)  # [3,3]
    ext_cv = np.asarray(f[f"observation/{camera}/extrinsic_cv"], dtype=np.float32)  # [T,3,4] w2c
    T = ext_cv.shape[0]
    w2c = np.broadcast_to(np.eye(4, dtype=np.float32), (T, 4, 4)).copy()
    w2c[:, :3, :] = ext_cv
    c2w = np.linalg.inv(w2c).astype(np.float32)                                   # [T,4,4]
    return K, c2w


def frame_size(f: h5py.File, camera: str) -> tuple[int, int]:
    import io
    from PIL import Image
    raw = bytes(f[f"observation/{camera}/rgb"][0])
    img = Image.open(io.BytesIO(raw))
    return img.size[1], img.size[0]  # (H, W)


def load_prompt(instructions_dir: Path, ep: int, default: str) -> str:
    path = instructions_dir / f"episode{ep}.json"
    if not path.exists():
        return default
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("seen", "unseen", "instructions"):
            if isinstance(data.get(key), list) and data[key]:
                return str(data[key][0])
    if isinstance(data, list) and data:
        return str(data[0])
    return default


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/data/RoboTwin2.0_unpacked")
    parser.add_argument("--task", required=True, help="e.g. beat_block_hammer")
    parser.add_argument("--robot_mode", required=True, help="e.g. aloha-agilex_clean_50")
    parser.add_argument("--episodes", default=None, help="Comma-separated indices, e.g. 0,1,2")
    parser.add_argument("--episodes_range", default=None, help="Inclusive range, e.g. 0-9")
    parser.add_argument("--camera", default="head_camera")
    parser.add_argument("--work_dir", required=True, help="Output dir for npy + manifest.jsonl")
    parser.add_argument("--default_prompt", default="robot arm manipulation")
    args = parser.parse_args()

    root = Path(args.root)
    ds_dir = _resolve_dataset_dir(root, args.task, args.robot_mode)
    data_dir = ds_dir / "data"
    video_dir = ds_dir / "video"
    instr_dir = ds_dir / "instructions"
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)

    episodes = parse_episodes(args.episodes, args.episodes_range, data_dir)
    if not episodes:
        raise SystemExit(f"No episodes found under {data_dir}")

    rows = []
    for ep in episodes:
        hdf5_path = data_dir / f"episode{ep}.hdf5"
        video_path = video_dir / f"episode{ep}.mp4"
        if not hdf5_path.exists() or not video_path.exists():
            print(f"  skip episode{ep}: missing hdf5 or mp4")
            continue
        ep_out = work_dir / f"{args.task}__{args.robot_mode}__episode{ep}"
        ep_out.mkdir(parents=True, exist_ok=True)

        with h5py.File(hdf5_path, "r") as f:
            actions = build_actions(f)
            K, c2w = build_camera(f, args.camera)
            H, W = frame_size(f, args.camera)
        if not (actions.shape[0] == c2w.shape[0]):
            raise ValueError(f"episode{ep}: action frames {actions.shape[0]} != camera frames {c2w.shape[0]}")

        np.save(ep_out / "actions.npy", actions)
        np.save(ep_out / "intrinsic.npy", K)
        np.save(ep_out / "extrinsic.npy", c2w)
        prompt = load_prompt(instr_dir, ep, args.default_prompt)

        rows.append({
            "id": ep_out.name,
            "video": str(video_path),
            "action_path": str(ep_out / "actions.npy"),
            "intrinsic_path": str(ep_out / "intrinsic.npy"),
            "extrinsic_path": str(ep_out / "extrinsic.npy"),
            "original_size": [H, W],
            "prompt": prompt,
            "total_frames": int(actions.shape[0]),
        })
        print(f"  episode{ep}: T={actions.shape[0]} size=({H},{W}) prompt={prompt!r}")

    manifest_path = work_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as fp:
        for row in rows:
            fp.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} rows -> {manifest_path}")


if __name__ == "__main__":
    main()
