"""SEAM-1 (VACE supply) + the single A2V provisioning entry point.

`provision_a2v(pipe, spec)` is what both training and inference call right after the
WanVideoPipeline is built. It:
  1. ensures the VACE branch exists (SEAM-1), and
  2. installs the mask_pq-parameterized VACE unit (SEAM-2).

SEAM-1 has two cases:
  * VACE-* bases ship pretrained VACE weights -> use `pipe.vace` as loaded (T2).
  * Bases without VACE weights (T2V-1.3B, TI2V-5B, ...) -> `create_vace_from_dit`
    clones the matching DiT blocks and zero-inits the VACE-only projections so the
    fresh branch contributes nothing until trained (T3).

For Wan2.1-VACE-1.3B both `provision_a2v` and `ensure_vace` are no-ops vs the stock
T1 path (pretrained vace already loaded; mask_pq=8 == the hardcoded value) — which
`a2v/check_t2_parity.py` proves.
"""

from __future__ import annotations

import torch

from a2v.base_spec import WanBaseSpec
from a2v.vace_unit import install_vace_unit


def create_vace_from_dit(pipe, spec: WanBaseSpec):
    """SEAM-1 from-DiT path: build a VaceWanModel and warm-start it from `pipe.dit`.

    The VACE context blocks are copies of the DiT transformer blocks at
    `spec.vace_layers` (VACE paper §3.3). For the zero-side-effect-at-init property we
    zero ONLY `after_proj` — the sole layer that couples the VACE branch back into the
    main stream (`x = x + after_proj(c) * scale`). With `after_proj == 0` every hint is
    zero, so the DiT output is bit-identical at init regardless of the control.

    Crucially we do NOT zero `vace_patch_embedding` or `before_proj` (an earlier version
    did, following ABot's init_from_dit, and it deadlocked): the control enters only via
    `patch_embedding`, then block 0 does `c = before_proj(c) + x`. If both
    `patch_embedding` and `before_proj.weight` start at zero, the control is disconnected
    AND its gradient path is dead — `patch_embedding` gets zero grad because
    `before_proj.weight == 0`, and `before_proj.weight` gets zero grad because its input
    (`patch_embedding`'s output) is zero. They stay pinned at zero forever and the
    control never influences the output. Leaving both at their live default init keeps
    the control path differentiable while `after_proj == 0` still gives zero side-effect.

    Shapes come entirely from `spec` (never naive-derived): `vace_in_dim`, `vace_layers`,
    dim, heads, ffn.
    """
    from diffsynth.models.wan_video_vace import VaceWanModel

    dit = pipe.dit
    if dit is None:
        raise RuntimeError(f"create_vace_from_dit({spec.name}): pipe.dit is None; cannot clone.")

    dit_blocks = dit.blocks
    if max(spec.vace_layers) >= len(dit_blocks):
        raise ValueError(
            f"create_vace_from_dit({spec.name}): vace_layers {spec.vace_layers} exceed "
            f"DiT depth {len(dit_blocks)}."
        )

    vace = VaceWanModel(
        vace_layers=tuple(spec.vace_layers),
        vace_in_dim=spec.vace_in_dim,
        patch_size=tuple(spec.patch_size),
        has_image_input=getattr(dit, "has_image_input", False),
        dim=spec.dim,
        num_heads=spec.num_heads,
        ffn_dim=spec.ffn_dim,
    )

    # 1) Copy the matching DiT block sub-modules into each VACE block. named_children()
    #    yields the shared modules (self_attn, cross_attn, norm1/2/3, ffn, gate); the
    #    `modulation` Parameter is not a child and is left at fresh init (as in ABot —
    #    moot at init because after_proj=0, and learned during overfit).
    for vace_idx, layer_id in enumerate(spec.vace_layers):
        src = dit_blocks[layer_id]
        dst = vace.vace_blocks[vace_idx]
        for name, mod in src.named_children():
            dst_mod = getattr(dst, name, None)
            if isinstance(dst_mod, torch.nn.Module):
                dst_mod.load_state_dict(mod.state_dict(), strict=True)

    # 2) Zero-init ONLY after_proj -> zero hint -> zero side-effect at init, while the
    #    control path (patch_embedding, before_proj) stays live so gradients flow there.
    for dst in vace.vace_blocks:
        torch.nn.init.zeros_(dst.after_proj.weight)
        if dst.after_proj.bias is not None:
            torch.nn.init.zeros_(dst.after_proj.bias)

    # Match the DiT's dtype/device so the new branch slots into the pipeline cleanly.
    ref = next(dit.parameters())
    vace = vace.to(dtype=pipe.torch_dtype, device=ref.device)
    return vace


def ensure_vace(pipe, spec: WanBaseSpec, vace_ckpt=None):
    """SEAM-1: return a usable VACE branch on `pipe`. Idempotent.

    * has_pretrained_vace=True  -> expect `pipe.vace` already loaded; return it.
    * has_pretrained_vace=False -> build from DiT once (cached via a marker so repeat
      calls don't clobber a branch that LoRA has already been attached to).
    """
    if spec.has_pretrained_vace:
        if getattr(pipe, "vace", None) is None:
            raise RuntimeError(
                f"Base {spec.name!r} declares has_pretrained_vace=True but pipe.vace is None "
                "(checkpoint missing the VACE branch?)."
            )
        return pipe.vace

    # from-DiT path (T3+). The marker makes this idempotent: the first call builds the
    # branch; later calls (e.g. post-construction provision_a2v) must NOT rebuild it,
    # or they would discard the LoRA already attached by training.
    if getattr(pipe, "_a2v_vace_from_dit", False):
        return pipe.vace
    pipe.vace = create_vace_from_dit(pipe, spec)
    pipe._a2v_vace_from_dit = True
    return pipe.vace


def provision_a2v(pipe, spec: WanBaseSpec):
    """Apply all base-specific seams to a freshly built pipe. Returns the VACE branch."""
    vace = ensure_vace(pipe, spec)
    if not install_vace_unit(pipe, spec.mask_pq):
        raise RuntimeError("No WanVideoUnit_VACE found in pipe.units; cannot install parameterized VACE unit.")
    return vace


__all__ = ["create_vace_from_dit", "ensure_vace", "provision_a2v"]
