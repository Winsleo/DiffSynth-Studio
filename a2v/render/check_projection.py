"""I1 regression test for intrinsic adjustment under resize/crop."""

from __future__ import annotations

import torch

from a2v.render.traj_map import (
    adjust_intrinsic_for_resize_and_crop,
    adjust_intrinsic_for_resize_stretch,
)


def _project(intrinsic: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    uvw = (intrinsic @ points.unsqueeze(-1)).squeeze(-1)
    return uvw[..., :2] / uvw[..., 2:3]


def main() -> None:
    original_size = (480, 640)
    target_size = (256, 448)
    intrinsic = torch.tensor([[300.0, 0.0, 320.0], [0.0, 300.0, 240.0], [0.0, 0.0, 1.0]])
    points = torch.tensor([[0.05, -0.03, 1.2], [-0.1, 0.08, 0.9], [0.0, 0.0, 1.5]])

    projected = _project(intrinsic, points)
    sx = target_size[1] / original_size[1]
    sy = target_size[0] / original_size[0]
    expected = torch.stack([projected[:, 0] * sx, projected[:, 1] * sy], dim=-1)
    got = _project(adjust_intrinsic_for_resize_stretch(intrinsic, original_size, target_size), points)
    assert torch.allclose(got, expected, atol=1e-3), (got, expected)

    scale = max(target_size[1] / original_size[1], target_size[0] / original_size[0])
    resized_w = round(original_size[1] * scale)
    resized_h = round(original_size[0] * scale)
    crop_left = (resized_w - target_size[1]) // 2
    crop_top = (resized_h - target_size[0]) // 2
    expected = torch.stack([projected[:, 0] * scale - crop_left, projected[:, 1] * scale - crop_top], dim=-1)
    got = _project(adjust_intrinsic_for_resize_and_crop(intrinsic, original_size, target_size), points)
    assert torch.allclose(got, expected, atol=1e-3), (got, expected)

    print("check_projection: OK")


if __name__ == "__main__":
    main()
