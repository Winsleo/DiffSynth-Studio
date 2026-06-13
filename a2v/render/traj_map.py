"""Utilities for rendering ABot action trajectories as 3-channel VACE videos.

This module intentionally stays independent from DiffSynth core code. It
implements the low-risk L0 path from the A2V implementation plan: render action
trajectories offline, store them as lossless PNG frames, and feed them to the
stock Wan VACE training pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from einops import rearrange
from PIL import Image, ImageDraw

try:
    from matplotlib import cm
except ModuleNotFoundError:
    cm = None

SizeHW = Tuple[int, int]


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert quaternions in wxyz order to rotation matrices."""
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)
    out = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return out.reshape(quaternions.shape[:-1] + (3, 3))


def get_transformation_matrix_from_quat(quat: torch.Tensor) -> torch.Tensor:
    """Build 4x4 transforms from xyz + xyzw quaternion rows."""
    rot_quat = quat[:, 3:]
    rot_quat = rot_quat[:, [3, 0, 1, 2]]
    rot = quaternion_to_matrix(rot_quat)
    trans = quat[:, :3]
    output = torch.eye(4, dtype=quat.dtype, device=quat.device).unsqueeze(0).repeat(quat.shape[0], 1, 1)
    output[:, :3, :3] = rot
    output[:, :3, 3] = trans
    return output


def simple_radius_gen_func(
    xyzs: torch.Tensor,
    c_xyzs: torch.Tensor,
    radius_min_px: float = 3.0,
    radius_max_px: float = 40.0,
) -> torch.Tensor:
    """Empirical radius from EE<->camera distance, corrected min-max normalization.

    Fixes the original ABot formula (see ``A2V_action_encoding_redesign.md`` §1),
    which was effectively dead. Two compounding faults made it ~constant:
      * a missing parenthesis left ``dist`` *outside* the normalization, so
        ``(dist - 0.07) / (0.8 - 0.07)`` collapsed to ``0.904 - dist``; and
      * magic near/far constants (0.07 m / 0.8 m) that don't match real camera
        distances (~1 m) -> the clamp pinned the dot to 1 px regardless of depth.

    Here we min-max normalize over the trajectory's *actual* distance range, so the
    dot size always tracks relative depth (closest -> ``radius_max_px``, farthest ->
    ``radius_min_px``) no matter the absolute camera distance. ``xyzs`` is the full
    ``[T, 3]`` EE path and ``c_xyzs`` the ``[T, 3]`` camera centres; normalization
    is over T. For a perspective-calibrated (absolute) size instead of a relative
    ramp, use ``radius_mode='physical'``.
    """
    dist = torch.sqrt(((xyzs - c_xyzs) ** 2).sum(-1))
    span = (dist.max() - dist.min()).clamp(min=1e-6)
    near_big = 1.0 - (dist - dist.min()) / span  # closest frame -> 1, farthest -> 0
    return radius_min_px + near_big * (radius_max_px - radius_min_px)


def physical_projection_radius(
    z_cam: torch.Tensor,
    f_eff: torch.Tensor | float,
    phys_radius_m: float,
    radius_min_px: float,
    radius_max_px: float,
) -> torch.Tensor:
    """Perspective-correct on-screen radius of a physical sphere (redesign §3.1).

    A sphere of real radius ``phys_radius_m`` (e.g. ~0.04 m for a gripper) at
    camera-frame depth ``z_cam`` images, under focal length ``f_eff``, to a pixel
    radius ``r_px = f_eff * phys_radius_m / z_cam`` (true 1/z near-large/far-small,
    calibrated with the *same* intrinsics as the target video). Clamped to
    ``[radius_min_px, radius_max_px]`` so it stays compression-robust (lower) and
    never floods the frame (upper).
    """
    z_cam = z_cam.clamp(min=1e-3)  # guard div-by-zero / points behind the camera
    r_px = f_eff * phys_radius_m / z_cam
    return torch.clamp(r_px, min=radius_min_px, max=radius_max_px)


_GREENS = (
    (0.0, (247, 252, 245)),
    (0.125, (229, 245, 224)),
    (0.25, (199, 233, 192)),
    (0.375, (161, 217, 155)),
    (0.5, (116, 196, 118)),
    (0.625, (65, 171, 93)),
    (0.75, (35, 139, 69)),
    (0.875, (0, 109, 44)),
    (1.0, (0, 68, 27)),
)
_REDS = (
    (0.0, (255, 245, 240)),
    (0.125, (254, 224, 210)),
    (0.25, (252, 187, 161)),
    (0.375, (252, 146, 114)),
    (0.5, (251, 106, 74)),
    (0.625, (239, 59, 44)),
    (0.75, (203, 24, 29)),
    (0.875, (165, 15, 21)),
    (1.0, (103, 0, 13)),
)


def _fallback_colormap(value: float, anchors: tuple[tuple[float, tuple[int, int, int]], ...]) -> tuple[float, float, float, float]:
    """Approximate Matplotlib ColorBrewer maps when matplotlib is absent."""
    value = min(1.0, max(0.0, float(value)))
    for idx in range(1, len(anchors)):
        left_x, left_rgb = anchors[idx - 1]
        right_x, right_rgb = anchors[idx]
        if value <= right_x:
            ratio = (value - left_x) / (right_x - left_x)
            rgb = tuple((left_rgb[i] + (right_rgb[i] - left_rgb[i]) * ratio) / 255.0 for i in range(3))
            return rgb + (1.0,)
    return tuple(channel / 255.0 for channel in anchors[-1][1]) + (1.0,)


def _greens(value: float) -> tuple[float, float, float, float]:
    return cm.Greens(value) if cm is not None else _fallback_colormap(value, _GREENS)


def _reds(value: float) -> tuple[float, float, float, float]:
    return cm.Reds(value) if cm is not None else _fallback_colormap(value, _REDS)


def _draw_circle(draw: ImageDraw.ImageDraw, center: np.ndarray, radius: int, color: tuple[int, int, int]) -> None:
    x, y = int(center[0]), int(center[1])
    draw.ellipse((x - radius, y - radius, x + radius, y + radius), fill=tuple(color))


def _draw_line(draw: ImageDraw.ImageDraw, p0: np.ndarray, p1: np.ndarray, color: tuple[int, int, int], width: int = 8) -> None:
    draw.line((int(p0[0]), int(p0[1]), int(p1[0]), int(p1[1])), fill=tuple(color), width=width)


def get_traj_maps(
    pose: Union[np.ndarray, torch.Tensor],
    w2c: torch.Tensor,
    c2w: torch.Tensor,
    intrinsic: torch.Tensor,
    sample_size: SizeHW,
    radius_gen_func=None,
    gripper_z_offset: float = 0.23,
    radius_mode: Optional[str] = None,
    phys_radius_m: float = 0.04,
    radius_min_px: float = 3.0,
    radius_max_frac: float = 1.0 / 6.0,
) -> torch.Tensor:
    """Render RGB trajectory maps.

    Args:
        pose: ``[T, 16]`` action rows in
            ``left_xyz, left_xyzw, left_gripper, right_xyz, right_xyzw, right_gripper`` format.
        w2c: world-to-camera matrices, ``[V, T, 4, 4]``.
        c2w: camera-to-world matrices, ``[V, T, 4, 4]``.
        intrinsic: camera intrinsics, ``[V, 3, 3]``.
        sample_size: output ``(height, width)``.
        radius_gen_func: optional ``(ee_xyz_world[T,3], cam_center_world[T,3]) ->
            radius_px[T]`` callback overriding the built-in normalized radius.
            Used only in ``radius_mode='normalized'``.
        gripper_z_offset: translation (metres) along the end-effector +z axis
            applied before projection, shifting the marker from the wrist frame to
            the gripper tip. Default ``0.23`` matches ABot/AgiBot; pass ``0.0`` for
            datasets whose end-effector pose already sits at the TCP (e.g. RoboTwin).
        radius_mode: how to size the end-effector dots.
            ``'physical'`` (redesign §3.1) — perspective-correct, absolute
            ``f_eff*R_phys/z_cam``; ``'normalized'`` — corrected min-max distance
            ramp (``simple_radius_gen_func``, or ``radius_gen_func`` if given);
            ``'constant'`` — fixed 50 px. Default resolves to ``'normalized'`` when
            ``radius_gen_func`` is explicitly given, else ``'physical'``.
        phys_radius_m: physical end-effector radius in metres (``'physical'`` mode).
        radius_min_px: lower clamp on dot radius (compression-robust floor).
        radius_max_frac: upper clamp as a fraction of frame height.

    Returns:
        Tensor in ``[3, V, T, H, W]`` with values in ``[0, 1]``.
    """
    if radius_mode is None:
        radius_mode = "normalized" if radius_gen_func is not None else "physical"
    if radius_mode not in ("physical", "normalized", "constant"):
        raise ValueError(f"radius_mode must be 'physical'|'normalized'|'constant', got {radius_mode!r}")
    height, width = sample_size
    axis_colors_l = [(0, 0, 255), (255, 255, 0), (0, 255, 255)]
    axis_colors_r = [(255, 0, 255), (255, 0, 0), (0, 255, 0)]

    if isinstance(pose, np.ndarray):
        pose = torch.tensor(pose, dtype=torch.float32)
    pose = pose.to(dtype=torch.float32)
    device = pose.device

    ee_key_pts = torch.tensor(
        [[0, 0, 0, 1], [0.1, 0, 0, 1], [0, 0.1, 0, 1], [0, 0, 0.1, 1]],
        dtype=torch.float32,
        device=device,
    ).view(1, 1, 4, 4).permute(0, 1, 3, 2)

    pose_l_mat = get_transformation_matrix_from_quat(pose[:, 0:7]).unsqueeze(0)
    pose_r_mat = get_transformation_matrix_from_quat(pose[:, 8:15]).unsqueeze(0)

    ee2cam_l = torch.matmul(w2c, pose_l_mat)
    ee2cam_r = torch.matmul(w2c, pose_r_mat)

    correct_matrix = torch.tensor(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, gripper_z_offset], [0, 0, 0, 1]],
        dtype=torch.float32,
        device=device,
    ).view(1, 1, 4, 4)
    ee2cam_l = torch.matmul(ee2cam_l, correct_matrix)
    ee2cam_r = torch.matmul(ee2cam_r, correct_matrix)

    pts_l = torch.matmul(ee2cam_l, ee_key_pts)
    pts_r = torch.matmul(ee2cam_r, ee_key_pts)

    intrinsic = intrinsic.unsqueeze(1)
    uvs_l0 = torch.matmul(intrinsic, pts_l[:, :, :3, :])
    uvs_l = (uvs_l0 / pts_l[:, :, 2:3, :])[:, :, :2, :].permute(0, 1, 3, 2).to(torch.int64)
    uvs_r0 = torch.matmul(intrinsic, pts_r[:, :, :3, :])
    uvs_r = (uvs_r0 / pts_r[:, :, 2:3, :])[:, :, :2, :].permute(0, 1, 3, 2).to(torch.int64)

    all_img_list = []
    for icam in range(w2c.shape[0]):
        l_xyz = pose[:, 0:3].clone()
        r_xyz = pose[:, 8:11].clone()
        c_xyz = c2w[icam, :, :3, 3].clone()

        radius_max_px = height * radius_max_frac
        if radius_mode == "physical":
            # redesign §3.1: perspective-correct dot size r_px = f_eff * R_phys / z_cam,
            # replacing the dead ABot empirical radius. z_cam is the EE-origin depth in
            # the camera frame (key point 0 of pts_*); f_eff uses the SAME scaled
            # intrinsic as the projected centers, so dot size is calibrated to this view.
            f_eff = (intrinsic[icam, 0, 0, 0] + intrinsic[icam, 0, 1, 1]) / 2.0
            l_dist = physical_projection_radius(pts_l[icam, :, 2, 0], f_eff, phys_radius_m, radius_min_px, radius_max_px)
            r_dist = physical_projection_radius(pts_r[icam, :, 2, 0], f_eff, phys_radius_m, radius_min_px, radius_max_px)
        elif radius_mode == "normalized":
            # corrected min-max distance ramp (shares the same px clamp range as physical).
            if radius_gen_func is not None:
                l_dist = radius_gen_func(l_xyz, c_xyz)
                r_dist = radius_gen_func(r_xyz, c_xyz)
            else:
                l_dist = simple_radius_gen_func(l_xyz, c_xyz, radius_min_px, radius_max_px)
                r_dist = simple_radius_gen_func(r_xyz, c_xyz, radius_min_px, radius_max_px)
        else:  # "constant"
            l_dist = torch.full((pose.shape[0],), 50, device=device)
            r_dist = torch.full((pose.shape[0],), 50, device=device)

        img_list = []
        for i in range(pose.shape[0]):
            image = Image.new("RGB", (width, height), (50, 50, 50))
            draw = ImageDraw.Draw(image)
            normalized_value_l = float(pose[i, 7].item()) / 120
            normalized_value_r = float(pose[i, 15].item()) / 120
            color_l = tuple(int(c * 255) for c in _greens(normalized_value_l)[:3])
            color_r = tuple(int(c * 255) for c in _reds(normalized_value_r)[:3])

            for points, color, radius in zip(
                [uvs_l[icam, i], uvs_r[icam, i]],
                [color_l, color_r],
                [l_dist[i], r_dist[i]],
            ):
                base = np.array(points[0].cpu())
                if base[0] < 0 or base[0] >= width or base[1] < 0 or base[1] >= height:
                    continue
                point = np.array(points[0][:2].cpu(), dtype=np.int64)
                radius_int = max(1, int(radius.item() if torch.is_tensor(radius) else radius))
                _draw_circle(draw, point, radius_int, color)

            for points, colors in zip([uvs_l[icam, i], uvs_r[icam, i]], [axis_colors_l, axis_colors_r]):
                base = np.array(points[0].cpu(), dtype=np.int64)
                if base[0] < 0 or base[0] >= width or base[1] < 0 or base[1] >= height:
                    continue
                for j, point in enumerate(points):
                    if j == 0:
                        continue
                    point = np.array(point[:2].cpu(), dtype=np.int64)
                    _draw_line(draw, base, point, colors[j - 1], width=8)
            img_list.append(np.asarray(image, dtype=np.float32) / 255.0)

        all_img_list.append(np.stack(img_list, axis=0))

    all_img_list = np.stack(all_img_list, axis=0)
    return rearrange(torch.tensor(all_img_list), "v t h w c -> c v t h w").float()


def normalize_angles(radius: np.ndarray) -> np.ndarray:
    return np.mod(radius, 2 * np.pi) - 2 * np.pi * (np.mod(radius, 2 * np.pi) > np.pi)


def load_actions_with_quat(h5_file: Union[str, Path], slices: Optional[Sequence[int]] = None) -> np.ndarray:
    """Load actions in the 16-D trajectory-rendering format.

    Supported formats:
    - ``.h5`` / ``.hdf5`` ABot files with ``state/end/*`` and
      ``state/effector/position`` datasets.
    - ``.npy`` containing an ``[T, 16]`` action array.
    - ``.npz`` containing an ``actions`` array.
    """
    h5_file = Path(h5_file)
    suffix = h5_file.suffix.lower()
    if suffix == ".npy":
        actions = np.load(h5_file).astype(np.float32)
        if actions.ndim != 2 or actions.shape[1] != 16:
            raise ValueError(f"Expected npy actions with shape [T,16], got {actions.shape}")
        return actions[list(slices)] if slices is not None else actions
    if suffix == ".npz":
        data = np.load(h5_file)
        if "actions" not in data:
            raise ValueError("NPZ action file must contain an 'actions' array")
        actions = data["actions"].astype(np.float32)
        if actions.ndim != 2 or actions.shape[1] != 16:
            raise ValueError(f"Expected npz actions with shape [T,16], got {actions.shape}")
        return actions[list(slices)] if slices is not None else actions

    try:
        import h5py
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("h5py is required to load ABot .h5 action files; install h5py or provide .npy/.npz actions") from exc

    with h5py.File(str(h5_file), "r") as fid:
        all_ends_p = np.array(fid["state/end/position"], dtype=np.float32)
        all_ends_o = np.array(fid["state/end/orientation"], dtype=np.float32)
        all_gripper = np.array(fid["state/effector/position"], dtype=np.float32)

    if slices is None:
        slices = list(range(all_ends_p.shape[0]))

    actions = np.zeros((len(slices), 16), dtype=np.float32)
    for i, idx in enumerate(slices):
        actions[i, 0:3] = all_ends_p[idx, 0]
        actions[i, 3:7] = all_ends_o[idx, 0]
        actions[i, 7] = all_gripper[idx, 0]
        actions[i, 8:11] = all_ends_p[idx, 1]
        actions[i, 11:15] = all_ends_o[idx, 1]
        actions[i, 15] = all_gripper[idx, 1]
    return actions


def _intrinsic_from_mapping(mapping: dict) -> torch.Tensor:
    intr = mapping.get("intrinsic", mapping)
    if "matrix" in intr:
        return torch.tensor(intr["matrix"], dtype=torch.float32)
    if "K" in intr:
        return torch.tensor(intr["K"], dtype=torch.float32)
    intrinsic = torch.eye(3, dtype=torch.float32)
    intrinsic[0, 0] = float(intr["fx"])
    intrinsic[1, 1] = float(intr["fy"])
    intrinsic[0, 2] = float(intr.get("ppx", intr.get("cx")))
    intrinsic[1, 2] = float(intr.get("ppy", intr.get("cy")))
    return intrinsic


def _extrinsic_from_mapping(mapping: dict) -> np.ndarray:
    ext = mapping.get("extrinsic", mapping)
    if "matrix" in ext:
        matrix = np.array(ext["matrix"], dtype=np.float32)
    elif "c2w" in ext:
        matrix = np.array(ext["c2w"], dtype=np.float32)
    else:
        matrix = np.eye(4, dtype=np.float32)
        matrix[:3, :3] = np.array(ext["rotation_matrix"], dtype=np.float32)
        matrix[:3, 3] = np.array(ext["translation_vector"], dtype=np.float32)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected 4x4 extrinsic matrix, got {matrix.shape}")
    return matrix


def load_camera_params_from_json(
    intrinsic_path: Union[str, Path],
    extrinsic_path: Union[str, Path],
    frame_indices: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load camera intrinsics and c2w extrinsics from JSON files."""
    with open(intrinsic_path, "r", encoding="utf-8") as f:
        intrinsic_data = json.load(f)
    intrinsic = _intrinsic_from_mapping(intrinsic_data)

    with open(extrinsic_path, "r", encoding="utf-8") as f:
        extrinsic_data = json.load(f)
    if isinstance(extrinsic_data, dict) and "extrinsics" in extrinsic_data:
        extrinsic_data = extrinsic_data["extrinsics"]
    if isinstance(extrinsic_data, dict):
        extrinsic_data = [extrinsic_data]
    extrinsics = torch.tensor(np.stack([_extrinsic_from_mapping(item) for item in extrinsic_data]), dtype=torch.float32)

    if frame_indices is not None:
        extrinsics = extrinsics[list(frame_indices)]
    return intrinsic, extrinsics


def load_camera_params_from_npy(
    intrinsic_path: Union[str, Path],
    extrinsic_path: Union[str, Path],
    frame_indices: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load camera intrinsics and c2w extrinsics from NPY files."""
    intrinsic = torch.tensor(np.load(intrinsic_path), dtype=torch.float32)
    extrinsics = torch.tensor(np.load(extrinsic_path), dtype=torch.float32)
    if intrinsic.dim() == 3 and intrinsic.shape[0] == 1:
        intrinsic = intrinsic[0]
    if frame_indices is not None and extrinsics.dim() > 2:
        extrinsics = extrinsics[list(frame_indices)]
    return intrinsic, extrinsics


def load_camera_params(
    intrinsic_path: Union[str, Path],
    extrinsic_path: Union[str, Path],
    frame_indices: Optional[Sequence[int]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Load camera parameters from JSON or NPY files."""
    intrinsic_path = Path(intrinsic_path)
    extrinsic_path = Path(extrinsic_path)
    if intrinsic_path.suffix.lower() == ".npy" or extrinsic_path.suffix.lower() == ".npy":
        return load_camera_params_from_npy(intrinsic_path, extrinsic_path, frame_indices)
    return load_camera_params_from_json(intrinsic_path, extrinsic_path, frame_indices)


def adjust_intrinsic_for_resize_and_crop(intrinsic: torch.Tensor, original_size: SizeHW, target_size: SizeHW) -> torch.Tensor:
    """Adjust intrinsics for DiffSynth ImageCropAndResize crop mode."""
    orig_h, orig_w = original_size
    tgt_h, tgt_w = target_size
    scale = max(tgt_w / orig_w, tgt_h / orig_h)
    resized_w = round(orig_w * scale)
    resized_h = round(orig_h * scale)
    crop_left = (resized_w - tgt_w) // 2
    crop_top = (resized_h - tgt_h) // 2

    adjusted = intrinsic.clone()
    adjusted[0, 0] = intrinsic[0, 0] * scale
    adjusted[1, 1] = intrinsic[1, 1] * scale
    adjusted[0, 2] = intrinsic[0, 2] * scale - crop_left
    adjusted[1, 2] = intrinsic[1, 2] * scale - crop_top
    return adjusted


def adjust_intrinsic_for_resize_stretch(intrinsic: torch.Tensor, original_size: SizeHW, target_size: SizeHW) -> torch.Tensor:
    """Adjust intrinsics for direct non-aspect-preserving resize."""
    orig_h, orig_w = original_size
    tgt_h, tgt_w = target_size
    adjusted = intrinsic.clone()
    adjusted[0, 0] = intrinsic[0, 0] * (tgt_w / orig_w)
    adjusted[1, 1] = intrinsic[1, 1] * (tgt_h / orig_h)
    adjusted[0, 2] = intrinsic[0, 2] * (tgt_w / orig_w)
    adjusted[1, 2] = intrinsic[1, 2] * (tgt_h / orig_h)
    return adjusted


def get_vace_traj_maps_with_scaled_intrinsic(
    pose: Union[np.ndarray, torch.Tensor],
    w2c: torch.Tensor,
    c2w: torch.Tensor,
    intrinsic: torch.Tensor,
    original_size: SizeHW,
    target_size: SizeHW,
    radius_gen_func=None,
    resize_mode: str = "crop",
    gripper_z_offset: float = 0.23,
    radius_mode: str = "physical",
    phys_radius_m: float = 0.04,
    radius_min_px: float = 3.0,
    radius_max_frac: float = 1.0 / 6.0,
) -> torch.Tensor:
    """Generate 3-channel VACE trajectory maps in ``[-1, 1]``.

    ``resize_mode='crop'`` matches DiffSynth ``ImageCropAndResize``.
    ``resize_mode='stretch'`` is available for datasets preprocessed by direct resize.

    ``radius_mode`` selects EE dot sizing — ``'physical'`` (default,
    perspective-correct ``f_eff*R_phys/z_cam``) or ``'normalized'`` (corrected
    min-max distance ramp). See ``get_traj_maps`` for the full semantics.
    """
    if intrinsic.dim() == 2:
        intrinsic = intrinsic.unsqueeze(0)

    mode = (resize_mode or "crop").lower().strip()
    if mode == "crop":
        adjust_fn = adjust_intrinsic_for_resize_and_crop
    elif mode == "stretch":
        adjust_fn = adjust_intrinsic_for_resize_stretch
    else:
        raise ValueError(f"resize_mode must be 'crop' or 'stretch', got {resize_mode!r}")

    scaled_intrinsic = torch.stack([adjust_fn(intr, original_size, target_size) for intr in intrinsic], dim=0)
    trajs = get_traj_maps(
        pose,
        w2c,
        c2w,
        scaled_intrinsic,
        target_size,
        radius_gen_func=radius_gen_func,
        gripper_z_offset=gripper_z_offset,
        radius_mode=radius_mode,
        phys_radius_m=phys_radius_m,
        radius_min_px=radius_min_px,
        radius_max_frac=radius_max_frac,
    )
    return trajs * 2 - 1


def select_frame_indices(total_frames: int, num_frames: int, start: Optional[int] = None, stride: int = 1) -> list[int]:
    """Select a deterministic 4n+1 frame window."""
    if num_frames % 4 != 1:
        raise ValueError(f"num_frames must satisfy 4n+1, got {num_frames}")
    if total_frames <= 0:
        raise ValueError("total_frames must be positive")
    needed = 1 + (num_frames - 1) * stride
    if total_frames < needed:
        raise ValueError(f"Need {needed} source frames for num_frames={num_frames}, stride={stride}; got {total_frames}")
    if start is None:
        start = max(0, (total_frames - needed) // 2)
    end = start + needed
    if start < 0 or end > total_frames:
        raise ValueError(f"Invalid start/stride: start={start}, end={end}, total={total_frames}")
    return list(range(start, end, stride))


def vace_tensor_to_uint8_frames(trajs: torch.Tensor, camera_index: int = 0) -> list[np.ndarray]:
    """Convert ``[3, V, T, H, W]`` trajectory tensor in ``[-1,1]`` to RGB uint8 frames."""
    if trajs.dim() != 5 or trajs.shape[0] != 3:
        raise ValueError(f"Expected [3,V,T,H,W] trajectory tensor, got {tuple(trajs.shape)}")
    frames = trajs[:, camera_index].permute(1, 2, 3, 0).detach().cpu().numpy()
    frames = ((frames + 1.0) * 127.5).clip(0, 255).astype(np.uint8)
    return [frame for frame in frames]


def overlay_rgb(base: np.ndarray, control: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Blend a control RGB frame over a target RGB frame."""
    if base.shape[:2] != control.shape[:2]:
        control = np.array(Image.fromarray(control).resize((base.shape[1], base.shape[0]), resample=Image.BILINEAR))
    blended = (base.astype(np.float32) * (1.0 - alpha) + control.astype(np.float32) * alpha).clip(0, 255)
    return blended.astype(np.uint8)


__all__ = [
    "adjust_intrinsic_for_resize_and_crop",
    "adjust_intrinsic_for_resize_stretch",
    "get_traj_maps",
    "get_vace_traj_maps_with_scaled_intrinsic",
    "load_actions_with_quat",
    "load_camera_params",
    "load_camera_params_from_json",
    "load_camera_params_from_npy",
    "overlay_rgb",
    "physical_projection_radius",
    "select_frame_indices",
    "simple_radius_gen_func",
    "vace_tensor_to_uint8_frames",
]
