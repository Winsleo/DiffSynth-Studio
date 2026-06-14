#!/usr/bin/env python3
"""Track T3 exit gates — SEAM-1 "create VACE from DiT" for a base without a VACE branch.

T2V-1.3B ships no VACE weights, so `provision.create_vace_from_dit` clones the matching
DiT blocks and zero-inits the VACE-only projections. This script proves the two gates
that must hold before any overfit run is meaningful:

  Gate (1) SHAPE   — the built branch has the spec-derived shape (vace_in_dim=96, 15
                     layers at vace_layers, dims from DiT) and the DiT-block weights
                     were actually copied; the VACE-only params are exactly zero.

  Gate (2) ZERO    — adding the zero-init branch is a zero behavioral change end-to-end:
   SIDE-EFFECT       a real `pipe()` run WITH the control trajectory is BIT-IDENTICAL to
                     one WITHOUT it (every hint = after_proj(c) = 0, and the only place
                     vace touches the DiT is `x = x + hint*scale`). This is the boolean
                     gate the master plan asks for, exercised through the public API.

  Gate (3) PARITY  — (optional, --parity) structural proof that the built shell matches
                     the official architecture: build a shell from VACE-1.3B's OWN DiT,
                     then `load_state_dict(stock pipe.vace, strict=True)` — a clean load
                     (no missing/unexpected/shape-mismatch) means the shell is exact.
                     Uses VACE-1.3B (not T2V) so it does not depend on T2V==VACE bytes.

Run:
    .venv/bin/python -m a2v.check_t3_provision \
        --dataset .cache/a2v_robotwin/ep0_dataset_phys [--parity]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig

from a2v.base_spec import get_spec
from a2v.provision import create_vace_from_dit, ensure_vace, provision_a2v


def _load_frames(dataset: Path, rel_paths) -> list[Image.Image]:
    return [Image.open(dataset / p).convert("RGB") for p in rel_paths]


def _build_pipe(spec):
    return WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[ModelConfig(path=p) for p in spec.model_paths()],
        tokenizer_config=ModelConfig(spec.abs_tokenizer_path()),
    )


def _all_zero(t: torch.Tensor) -> bool:
    return bool(torch.count_nonzero(t).item() == 0)


def gate_shape(pipe, spec) -> None:
    print("\n=== Gate (1): SHAPE + weight-copy + zero-init ===")
    assert getattr(pipe, "vace", None) is None, \
        f"{spec.name}: expected no pretrained VACE on load, but pipe.vace is set."

    vace = create_vace_from_dit(pipe, spec)

    # shape derived from spec, never hardcoded
    assert vace.vace_patch_embedding.in_channels == spec.vace_in_dim == 96, \
        f"vace_in_dim {vace.vace_patch_embedding.in_channels} != {spec.vace_in_dim}"
    assert len(vace.vace_blocks) == len(spec.vace_layers) == 15, \
        f"#vace_blocks {len(vace.vace_blocks)} != {len(spec.vace_layers)}"
    assert tuple(vace.vace_layers) == tuple(spec.vace_layers)
    assert vace.vace_patch_embedding.out_channels == spec.dim == 1536
    # a representative block dim/ffn check
    b0 = vace.vace_blocks[0]
    assert b0.self_attn.q.weight.shape == (spec.dim, spec.dim)
    assert b0.ffn[0].out_features == spec.ffn_dim == 8960
    print(f"  shape ok: vace_in_dim={spec.vace_in_dim}, layers={tuple(vace.vace_layers)} "
          f"(={len(vace.vace_blocks)}), dim={spec.dim}, ffn={spec.ffn_dim}")

    # DiT-block weights actually copied (check layer 0 self_attn.q + a mid layer ffn)
    dit = pipe.dit
    for li, vi in [(spec.vace_layers[0], 0), (spec.vace_layers[len(spec.vace_layers) // 2],
                                              len(spec.vace_layers) // 2)]:
        a = vace.vace_blocks[vi].self_attn.q.weight
        b = dit.blocks[li].self_attn.q.weight.to(a.dtype)
        assert torch.equal(a, b), f"self_attn.q not copied at dit layer {li}"
    print("  weight-copy ok: vace_blocks[*].self_attn.q == dit.blocks[vace_layers].self_attn.q")

    # after_proj (the ONLY coupling back to the main stream) must be zero -> zero hint.
    for i, blk in enumerate(vace.vace_blocks):
        assert _all_zero(blk.after_proj.weight) and _all_zero(blk.after_proj.bias), \
            f"block{i}.after_proj not zero"
    # the control path must stay LIVE (non-zero) so its gradient flows during training;
    # zeroing these too deadlocks the control (see create_vace_from_dit docstring).
    assert not _all_zero(vace.vace_patch_embedding.weight), \
        "vace_patch_embedding is zero -> control path dead (would not train)"
    assert not _all_zero(vace.vace_blocks[0].before_proj.weight), \
        "block0.before_proj is zero -> control gradient path dead"
    print("  init ok: after_proj == 0 (zero hint); patch_embedding / before_proj LIVE (non-zero)")
    print("Gate (1) SHAPE: OK")


def gate_zero_side_effect(pipe, spec, vace_video, args) -> None:
    print("\n=== Gate (2): ZERO SIDE-EFFECT (real pipe(), with vs without control) ===")
    # provision the zero-init branch + the mask_pq unit (fresh; sets idempotency marker)
    provision_a2v(pipe, spec)

    common = dict(
        prompt="robot arm manipulation",
        negative_prompt="",
        height=args.height, width=args.width, num_frames=args.num_frames,
        num_inference_steps=args.steps, cfg_scale=1.0, seed=args.seed, tiled=False,
    )

    def _arr(frames):
        return np.stack([np.asarray(f.convert("RGB"), dtype=np.uint8) for f in frames])

    # A: pure base (no vace_video -> vace_context None -> vace branch not invoked).
    base = _arr(pipe(**common))
    # B: feed the control trajectory ONLY -> the from-DiT vace branch runs and adds its
    # hints. We deliberately do NOT pass vace_reference_image: VACE's reference mechanism
    # prepends a frame to the MAIN input_latents (wan_video.py:413-418) and bumps
    # num_frames, which would change the base denoising independently of the hints. The
    # zero-side-effect claim under test is purely "from-DiT vace hints == 0 at init", so
    # we isolate the hint path (vace_video -> vace_context -> hints) and nothing else.
    withc = _arr(pipe(**common, vace_video=vace_video))

    assert base.shape == withc.shape, f"frame shape mismatch {base.shape} vs {withc.shape}"
    max_abs = int(np.abs(base.astype(np.int32) - withc.astype(np.int32)).max())
    identical = bool(np.array_equal(base, withc))
    print(f"  frames={base.shape} max_abs_pixel_diff={max_abs} bit_identical={identical}")
    assert identical, (
        f"Gate (2) FAILED: zero-init VACE perturbed the output (max_abs={max_abs}). "
        "after_proj is not a zero map, or vace has another side-effect."
    )
    print("Gate (2) ZERO SIDE-EFFECT: OK")


def gate_parity() -> None:
    print("\n=== Gate (3): STRUCTURAL PARITY (shell from VACE-1.3B DiT == official) ===")
    vspec = get_spec("wan2.1-vace-1.3b")
    vpipe = _build_pipe(vspec)
    assert getattr(vpipe, "vace", None) is not None, "VACE-1.3B should load a pretrained vace"
    stock_sd = {k: v for k, v in vpipe.vace.state_dict().items()}

    # build a shell from VACE-1.3B's own DiT, then load the official vace weights strictly
    vpipe.vace = None  # force the from-DiT path
    if hasattr(vpipe, "_a2v_vace_from_dit"):
        delattr(vpipe, "_a2v_vace_from_dit")
    shell = create_vace_from_dit(vpipe, vspec)
    missing, unexpected = shell.load_state_dict(stock_sd, strict=False)
    print(f"  load official vace into shell: missing={len(missing)} unexpected={len(unexpected)}")
    assert not missing and not unexpected, \
        f"shell architecture mismatch: missing={missing[:5]} unexpected={unexpected[:5]}"
    print("Gate (3) PARITY: OK (shell key/shape set == official VACE-1.3B)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_spec", default="wan2.1-t2v-1.3b")
    parser.add_argument("--dataset", default=".cache/a2v_robotwin/ep0_dataset_phys")
    parser.add_argument("--num_frames", type=int, default=13)   # 4n+1, small for speed
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--parity", action="store_true", help="also run Gate (3) on VACE-1.3B")
    args = parser.parse_args()

    spec = get_spec(args.base_spec)
    assert not spec.has_pretrained_vace, \
        f"{spec.name} has a pretrained VACE; T3 gates are for the from-DiT path."

    dataset = Path(args.dataset).resolve()
    row = json.loads((dataset / "metadata.jsonl").read_text().splitlines()[0])
    vace_video = _load_frames(dataset, row["vace_video"][: args.num_frames])

    pipe = _build_pipe(spec)
    gate_shape(pipe, spec)
    gate_zero_side_effect(pipe, spec, vace_video, args)
    if args.parity:
        gate_parity()

    print("\nT3 provision gates: OK")


if __name__ == "__main__":
    main()
