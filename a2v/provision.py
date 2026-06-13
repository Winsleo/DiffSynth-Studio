"""SEAM-1 (VACE supply) + the single A2V provisioning entry point.

`provision_a2v(pipe, spec)` is what both training and inference call right after the
WanVideoPipeline is built. It:
  1. ensures the VACE branch exists (SEAM-1), and
  2. installs the mask_pq-parameterized VACE unit (SEAM-2).

For Wan2.1-VACE-1.3B both are no-ops vs the stock T1 path (pretrained vace already loaded;
mask_pq=8 == the hardcoded value) — which `a2v/check_t2_parity.py` proves.
"""

from __future__ import annotations

from a2v.base_spec import WanBaseSpec
from a2v.vace_unit import install_vace_unit


def ensure_vace(pipe, spec: WanBaseSpec, vace_ckpt=None):
    """SEAM-1: return a usable VACE branch on `pipe`.

    VACE-* models ship pretrained weights -> use `pipe.vace` directly.
    Models without VACE weights (T2V-1.3B, TI2V-5B, ...) need `create_vace_from_dit`,
    which is Track T3 and not implemented here.
    """
    if spec.has_pretrained_vace and getattr(pipe, "vace", None) is not None:
        return pipe.vace
    raise NotImplementedError(
        f"Base {spec.name!r} has no pretrained VACE branch; create_vace_from_dit "
        "(zero-init from DiT) is Track T3 and is not implemented yet."
    )


def provision_a2v(pipe, spec: WanBaseSpec):
    """Apply all base-specific seams to a freshly built pipe. Returns the VACE branch."""
    vace = ensure_vace(pipe, spec)
    if not install_vace_unit(pipe, spec.mask_pq):
        raise RuntimeError("No WanVideoUnit_VACE found in pipe.units; cannot install parameterized VACE unit.")
    return vace


__all__ = ["ensure_vace", "provision_a2v"]
