"""Declarative base-model spec for A2V (Track T2 abstraction).

`WanBaseSpec` collapses the 5 base-model "seams" (see A2V_master_plan.md) into one
frozen dataclass, so adding a new Wan base model is ideally "add one REGISTRY entry".
At T2 only Wan2.1-VACE-1.3B is registered; its spec reproduces the hardcoded T1 path
exactly (`has_pretrained_vace=True`, `mask_pq=8`), which is what the parity gate proves.

Derived, never hardcoded (the three numbers that bite — master plan §1):
    vace_in_dim = 2 * vae_z_dim + vae_spatial_factor ** 2   # 96 (Wan2.1) / 352 (Wan2.2)
    mask_pq     = vae_spatial_factor                        # 8  (Wan2.1) / 16  (Wan2.2)
`vace_layers` is given explicitly (do NOT naively derive — 40-layer models use step 5).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _abs(rel: str) -> str:
    """Resolve a repo-relative model path to an absolute string."""
    p = Path(rel)
    return str(p if p.is_absolute() else (REPO_ROOT / rel))


@dataclass(frozen=True)
class WanBaseSpec:
    name: str
    # --- model files (local, absolute-resolved on access via model_paths()) ---
    dit_path: str                      # DiT (+ VACE branch for VACE-* models)
    vae_path: str
    t5_path: str
    tokenizer_path: str
    # --- DiT structure (SEAM-3) ---
    dim: int
    num_layers: int
    num_heads: int
    ffn_dim: int
    patch_size: tuple = (1, 2, 2)
    # --- VAE (SEAM-2) ---
    vae_z_dim: int = 16
    vae_spatial_factor: int = 8
    vae_temporal_factor: int = 4
    # --- VACE supply (SEAM-1) ---
    has_pretrained_vace: bool = True
    vace_layers: tuple = field(default_factory=lambda: tuple(range(0, 30, 2)))
    vace_remove_prefix: str = "pipe.vace."
    # --- first frame (SEAM-4) ---
    first_frame_mode: str = "vace_reference"   # vace_reference | i2v_concat | ti2v_fused | none
    # --- LoRA defaults ---
    lora_base_model: str = "vace"
    lora_target_modules: str = "q,k,v,o,ffn.0,ffn.2"

    # ---- derived (master plan §1: never hardcode 96/8) ----
    @property
    def vace_in_dim(self) -> int:
        return 2 * self.vae_z_dim + self.vae_spatial_factor ** 2

    @property
    def mask_pq(self) -> int:
        return self.vae_spatial_factor

    def model_paths(self) -> list[str]:
        """Absolute local paths in DiT, T5, VAE order (matches T1 --model_paths)."""
        return [_abs(self.dit_path), _abs(self.t5_path), _abs(self.vae_path)]

    def abs_tokenizer_path(self) -> str:
        return _abs(self.tokenizer_path)


_CONVERTED = "models/DiffSynth-Studio/Wan-Series-Converted-Safetensors"

REGISTRY: dict[str, WanBaseSpec] = {
    "wan2.1-vace-1.3b": WanBaseSpec(
        name="wan2.1-vace-1.3b",
        dit_path="models/Wan-AI/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors",
        vae_path=f"{_CONVERTED}/Wan2.1_VAE.safetensors",
        t5_path=f"{_CONVERTED}/models_t5_umt5-xxl-enc-bf16.safetensors",
        tokenizer_path="models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl",
        dim=1536, num_layers=30, num_heads=12, ffn_dim=8960,
        vae_z_dim=16, vae_spatial_factor=8,
        has_pretrained_vace=True,
        vace_layers=tuple(range(0, 30, 2)),       # 15 layers
        first_frame_mode="vace_reference",
    ),
    # T3: Wan2.1-T2V-1.3B ships NO VACE branch -> SEAM-1 builds one from the DiT
    # (provision.create_vace_from_dit: clone DiT blocks + zero-init). Same 1.3B dims
    # / Wan2.1 VAE as VACE-1.3B; the ONLY new variable vs T2 is has_pretrained_vace=False
    # (+ first_frame_mode="none"). dit_path is the official T2V DiT (no vace_* keys).
    "wan2.1-t2v-1.3b": WanBaseSpec(
        name="wan2.1-t2v-1.3b",
        dit_path="models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors",
        vae_path=f"{_CONVERTED}/Wan2.1_VAE.safetensors",
        t5_path=f"{_CONVERTED}/models_t5_umt5-xxl-enc-bf16.safetensors",
        tokenizer_path="models/Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl",
        dim=1536, num_layers=30, num_heads=12, ffn_dim=8960,
        vae_z_dim=16, vae_spatial_factor=8,
        has_pretrained_vace=False,                # SEAM-1: create_vace_from_dit
        vace_layers=tuple(range(0, 30, 2)),       # 15 layers (same as VACE-1.3B)
        first_frame_mode="none",                  # T2V has no native first-frame path
        # vace_remove_prefix defaults to "pipe.vace." -> LoRA saves as vace_blocks.*
    ),
}


def get_spec(name: str) -> WanBaseSpec:
    if name not in REGISTRY:
        raise KeyError(f"Unknown base spec {name!r}. Known: {sorted(REGISTRY)}")
    return REGISTRY[name]


__all__ = ["WanBaseSpec", "REGISTRY", "get_spec"]
