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
`a2v/check_parity.py` proves.
"""

from __future__ import annotations

import torch

from a2v.base_spec import WanBaseSpec
from a2v.vace_unit import install_vace_unit


def create_vace_from_dit(pipe, spec: WanBaseSpec, dit=None):
    """SEAM-1 from-DiT path: build a VaceWanModel and warm-start it from a DiT.

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

    `dit` selects the source transformer: default `pipe.dit`. For the SEAM-5 dual-expert
    MoE (A14B) the low-noise branch is cloned from `pipe.dit2` (-> pipe.vace2).
    """
    from diffsynth.models.wan_video_vace import VaceWanModel

    if dit is None:
        dit = pipe.dit
    if dit is None:
        raise RuntimeError(f"create_vace_from_dit({spec.name}): source DiT is None; cannot clone.")

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


def build_bare_vace(spec: WanBaseSpec, has_image_input: bool = False,
                    dtype=None, device="cpu"):
    """Construct a fresh (un-vram-managed) VaceWanModel with the spec's shape.

    Used at inference for a dual pretrained spec (Wan2.2-VACE-Fun-A14B): we want a plain
    GPU-resident branch to load the TRAINED full-vace checkpoint into (the pretrained
    weights from from_pretrained were only the training warm-start). Building bare avoids
    the from_pretrained vram-management wrapping, whose state_dict keys (`.module.`) don't
    match a standard vace checkpoint.
    """
    from diffsynth.models.wan_video_vace import VaceWanModel

    vace = VaceWanModel(
        vace_layers=tuple(spec.vace_layers),
        vace_in_dim=spec.vace_in_dim,
        patch_size=tuple(spec.patch_size),
        has_image_input=has_image_input,
        dim=spec.dim,
        num_heads=spec.num_heads,
        ffn_dim=spec.ffn_dim,
    )
    return vace.to(dtype=dtype or torch.bfloat16, device=device)


def ensure_vace_experts(pipe, spec: WanBaseSpec):
    """SEAM-5: ensure BOTH MoE VACE branches exist (idempotent).

    pipe.vace  <- create_vace_from_dit(pipe.dit)   (high-noise expert)
    pipe.vace2 <- create_vace_from_dit(pipe.dit2)  (low-noise expert)

    Used at inference where both experts are loaded. Training loads one expert per job, so
    it goes through the single-branch `ensure_vace` path instead. Returns (vace, vace2).
    """
    if getattr(pipe, "dit2", None) is None:
        raise RuntimeError(
            f"ensure_vace_experts({spec.name}): pipe.dit2 is None; both experts must be loaded "
            "(model_paths(expert=None))."
        )
    if not getattr(pipe, "_a2v_vace_from_dit", False):
        pipe.vace = create_vace_from_dit(pipe, spec, dit=pipe.dit)
        pipe._a2v_vace_from_dit = True
    if not getattr(pipe, "_a2v_vace2_from_dit", False):
        pipe.vace2 = create_vace_from_dit(pipe, spec, dit=pipe.dit2)
        pipe._a2v_vace2_from_dit = True
    return pipe.vace, pipe.vace2


def provision_a2v(pipe, spec: WanBaseSpec):
    """Apply all base-specific seams to a freshly built pipe. Returns the VACE branch.

    For a dual-expert spec with both DiTs loaded (inference), this provisions vace AND
    vace2 and returns the high-noise `vace`. Otherwise (single expert, or one expert loaded
    for a training job) it provisions the single `vace`.

    Two dual-expert flavors:
      * pretrained (Wan2.2-VACE-Fun-A14B): both pipe.vace/vace2 are loaded by from_pretrained
        (the noise checkpoints bundle DiT+VACE) -> just verify they are present.
      * from-DiT (Wan2.2-I2V-A14B): build both branches from dit/dit2 (ensure_vace_experts).
    """
    if spec.experts and getattr(pipe, "dit2", None) is not None:
        if spec.has_pretrained_vace:
            if getattr(pipe, "vace", None) is None or getattr(pipe, "vace2", None) is None:
                raise RuntimeError(
                    f"{spec.name} declares has_pretrained_vace=True + dual experts, but "
                    f"pipe.vace={pipe.vace is not None}/pipe.vace2={getattr(pipe,'vace2',None) is not None}"
                    " (noise checkpoints should bundle the VACE branches)."
                )
            vace = pipe.vace
        else:
            vace, _ = ensure_vace_experts(pipe, spec)
    else:
        vace = ensure_vace(pipe, spec)
    if not install_vace_unit(pipe, spec.mask_pq):
        raise RuntimeError("No WanVideoUnit_VACE found in pipe.units; cannot install parameterized VACE unit.")
    return vace


__all__ = ["create_vace_from_dit", "build_bare_vace", "ensure_vace", "ensure_vace_experts", "provision_a2v"]
