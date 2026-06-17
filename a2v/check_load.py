#!/usr/bin/env python3
"""A2V load smoke for any registered base model (replaces check_t4_load / check_t5_load).

The cheapest gate before provision/overfit: confirm the base loads fully from local
files and that the A2V VACE branch can be supplied on top of it. Checks are driven by
the spec, with a few first-frame-mode-specific assertions:

  common   DiT blocks == spec.num_layers; VAE z_dim / spatial factor == spec; VACE branch
           has len(spec.vace_layers) blocks, vace_in_dim == spec.vace_in_dim, patch_size
           match, after_proj zero-init / patch_embedding live; ParamWanVideoUnit_VACE
           installed with mask_pq == spec.mask_pq.
  i2v_concat  image (CLIP) encoder loaded; DiT in_dim == 36 (noise16 + y16 + mask4).
  ti2v_fused  DiT in_dim == 48; fused VAE first-frame + separated timestep enabled.

Run:
    .venv/bin/python -m a2v.check_load --base_spec wan2.1-i2v-14b-480p
    .venv/bin/python -m a2v.check_load --base_spec wan2.2-ti2v-5b
"""

from __future__ import annotations

import argparse

import torch

from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

from a2v.base_spec import get_spec
from a2v.provision import create_vace_from_dit, provision_a2v
from a2v.vace_unit import ParamWanVideoUnit_VACE


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base_spec", required=True, help="WanBaseSpec name (a2v.base_spec.REGISTRY).")
    args = ap.parse_args()

    spec = get_spec(args.base_spec)
    print(f"[1] loading {spec.name} from local files ...")
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[ModelConfig(path=p) for p in spec.model_paths()],
        tokenizer_config=ModelConfig(spec.abs_tokenizer_path()),
    )

    # --- DiT ---
    dit = pipe.dit
    assert dit is not None, "DiT failed to load"
    in_dim = getattr(dit, "in_dim", getattr(dit, "in_channels", None))
    print(f"[2] DiT: blocks={len(dit.blocks)} in_dim={in_dim} first_frame={spec.first_frame_mode}")
    assert len(dit.blocks) == spec.num_layers, f"expected {spec.num_layers} blocks, got {len(dit.blocks)}"

    if spec.first_frame_mode == "i2v_concat":
        assert in_dim == 36, f"expected I2V in_dim=36, got {in_dim}"
        assert pipe.image_encoder is not None, "CLIP image encoder did NOT load"
        print(f"    i2v: image_encoder={type(pipe.image_encoder).__name__}")
    elif spec.first_frame_mode == "ti2v_fused":
        assert in_dim == 48, f"expected TI2V in_dim=48, got {in_dim}"
        assert getattr(dit, "fuse_vae_embedding_in_latents", False), "TI2V fused image path is off"
        assert getattr(dit, "seperated_timestep", False), "TI2V separated timestep is off"
        print("    ti2v: fuse_vae_embedding_in_latents + seperated_timestep on")

    # --- VAE (SEAM-2: z_dim / spatial factor define vace_in_dim & mask_pq) ---
    vae = pipe.vae
    print(f"[3] VAE: {type(vae).__name__} z_dim={getattr(vae, 'z_dim', None)} "
          f"upsampling={getattr(vae, 'upsampling_factor', None)}")
    assert getattr(vae, "z_dim", None) == spec.vae_z_dim, "VAE z_dim != spec"
    assert getattr(vae, "upsampling_factor", None) == spec.vae_spatial_factor, "VAE spatial factor != spec"

    # --- VACE branch (SEAM-1) ---
    if spec.has_pretrained_vace:
        vace = pipe.vace
        assert vace is not None, "spec.has_pretrained_vace but pipe.vace is None"
        print(f"[4] pretrained VACE branch: blocks={len(vace.vace_blocks)}")
    else:
        vace = create_vace_from_dit(pipe, spec)
        print(f"[4] create_vace_from_dit OK: blocks={len(vace.vace_blocks)} "
              f"vace_in_dim={vace.vace_patch_embedding.in_channels} mask_pq={spec.mask_pq}")
        assert vace.vace_patch_embedding.in_channels == spec.vace_in_dim, "vace_in_dim != spec"
        assert tuple(vace.vace_patch_embedding.kernel_size) == tuple(spec.patch_size)
        aw = vace.vace_blocks[0].after_proj.weight
        pe = vace.vace_patch_embedding.weight
        print(f"[5] after_proj[0] absmax={aw.abs().max().item():.3e} (expect 0) | "
              f"patch_embedding absmax={pe.abs().max().item():.3e} (expect >0)")
        assert aw.abs().max().item() == 0.0, "after_proj should be zero-init"
        assert pe.abs().max().item() > 0.0, "patch_embedding must stay live (not zero)"
        pipe.vace = vace
        pipe._a2v_vace_from_dit = True
    assert len(vace.vace_blocks) == len(spec.vace_layers), \
        f"expected {len(spec.vace_layers)} vace blocks, got {len(vace.vace_blocks)}"

    # --- parameterized VACE unit (SEAM-2: mask_pq) ---
    provision_a2v(pipe, spec)
    units = [u for u in pipe.units if isinstance(u, ParamWanVideoUnit_VACE)]
    assert units and units[0].mask_pq == spec.mask_pq, \
        f"ParamWanVideoUnit_VACE(mask_pq={spec.mask_pq}) not installed"
    print(f"[6] parameterized VACE unit installed: mask_pq={spec.mask_pq}")

    print(f"\nload smoke ({spec.name}): OK")


if __name__ == "__main__":
    main()
