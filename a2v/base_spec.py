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

import os
from dataclasses import dataclass, field
from glob import glob
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
    # Sharded DiT (14B is split into diffusion_pytorch_model-0000N-of-00007.safetensors):
    # when set, model_paths() expands this repo-relative glob into the sorted shard list
    # and uses it as the (single) DiT entry instead of `dit_path`. ModelConfig.path
    # accepts list[str], so the loader merges the shards into one DiT.
    dit_glob: str = ""
    # I2V bases need a CLIP image encoder (SEAM-4 i2v_concat). When set it is appended as
    # an extra model_paths() entry; the loader auto-detects it as wan_video_image_encoder.
    image_encoder_path: str = ""
    # --- VAE (SEAM-2) ---
    vae_z_dim: int = 16
    vae_spatial_factor: int = 8
    vae_temporal_factor: int = 4
    # --- VACE supply (SEAM-1) ---
    has_pretrained_vace: bool = True
    vace_layers: tuple = field(default_factory=lambda: tuple(range(0, 30, 2)))
    vace_remove_prefix: str = "pipe.vace."
    # --- first frame (SEAM-4) ---
    # vace_reference | i2v_concat | i2v_vae | ti2v_fused | none
    #   i2v_concat : Wan2.1 I2V -> input_image builds CLIP clip_feature + VAE-concat y
    #   i2v_vae    : Wan2.2 I2V (A14B) -> input_image builds ONLY the VAE-concat y (no CLIP;
    #                require_clip_embedding=False), in_dim still 36
    first_frame_mode: str = "vace_reference"
    # --- expert set (SEAM-5: dual-expert MoE) ---
    # experts=() -> single expert (every T1..T5 base). experts=("high","low") -> Wan2.2
    # A14B MoE: a high-noise DiT (pipe.dit + pipe.vace) and a low-noise DiT (pipe.dit2 +
    # pipe.vace2). The two are trained as INDEPENDENT single-expert jobs on disjoint
    # timestep bands and combined at inference via DiffSynth's native switch_DiT_boundary.
    experts: tuple = ()
    # per-expert sharded-DiT globs, e.g. {"high": ".../high_noise_model/...-*.safetensors"}
    expert_dit_globs: dict = field(default_factory=dict)
    # inference expert switch (timestep < switch_boundary*1000 -> low-noise expert)
    switch_boundary: float = 0.875
    # training timestep band per expert: {"high": (min, max), "low": (min, max)} as
    # fractions for --min_timestep_boundary / --max_timestep_boundary (official A14B recipe)
    train_bands: dict = field(default_factory=dict)
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

    def _expert_dit_entry(self, expert: str) -> list:
        """Sorted shard list for one MoE expert (SEAM-5)."""
        pattern = self.expert_dit_globs.get(expert)
        if not pattern:
            raise KeyError(f"{self.name}: no expert_dit_globs entry for {expert!r}.")
        shards = sorted(glob(_abs(pattern)))
        if not shards:
            raise FileNotFoundError(f"{self.name}: expert {expert!r} glob matched no files: {pattern}")
        return shards

    def model_paths(self, expert: str | None = None) -> list:
        """Local model paths for from_pretrained / --model_paths.

        Order is DiT, T5, VAE (matches T1), with the DiT entry being either a single
        path (1.3B) or — for sharded 14B bases — a list of shard paths (ModelConfig.path
        accepts list[str], so the shards merge into one DiT). An I2V CLIP image encoder
        is appended last when `image_encoder_path` is set. Loader type-detects each entry,
        so trailing extras are order-independent.

        Dual-expert (SEAM-5, `self.experts` non-empty):
          * `expert in ("high","low")` -> load ONLY that expert's DiT (training: one job
            per expert, memory == single 14B).
          * `expert is None` -> load BOTH experts (high first, then low) so the loader's
            `fetch_model(..., index=2)` yields `[pipe.dit, pipe.dit2]` (inference).
        """
        if self.experts:
            if expert is not None:
                dit_entries = [self._expert_dit_entry(expert)]
            else:
                # high first -> pipe.dit (high-noise), low -> pipe.dit2 (low-noise)
                dit_entries = [self._expert_dit_entry(e) for e in self.experts]
        elif self.dit_glob:
            shards = sorted(glob(_abs(self.dit_glob)))
            if not shards:
                raise FileNotFoundError(f"{self.name}: dit_glob matched no files: {self.dit_glob}")
            dit_entries = [shards]
        else:
            dit_entries = [_abs(self.dit_path)]
        paths = [*dit_entries, _abs(self.t5_path), _abs(self.vae_path)]
        if self.image_encoder_path:
            paths.append(_abs(self.image_encoder_path))
        return paths

    def abs_tokenizer_path(self) -> str:
        return _abs(self.tokenizer_path)


# All base-model weights resolve under ONE root: $A2V_MODELS_DIR (default: the in-repo
# `models/`). Every path field below is built with _m(<rel>), so there is no hardcoded
# absolute/cluster path. Point A2V_MODELS_DIR at wherever the model repos live; the required
# sub-layout (HF/ModelScope dir names) is documented in A2V_HANDOFF.md (model path layout).
_MODELS = os.environ.get("A2V_MODELS_DIR") or str(REPO_ROOT / "models")


def _m(rel: str) -> str:
    """Absolute path to a model file/dir under the models root ($A2V_MODELS_DIR)."""
    return str(Path(_MODELS) / rel)


_CONVERTED = "DiffSynth-Studio/Wan-Series-Converted-Safetensors"  # converted VAE/T5 dir, under _MODELS

REGISTRY: dict[str, WanBaseSpec] = {
    "wan2.1-vace-1.3b": WanBaseSpec(
        name="wan2.1-vace-1.3b",
        dit_path=_m("Wan-AI/Wan2.1-VACE-1.3B/diffusion_pytorch_model.safetensors"),
        vae_path=_m(f"{_CONVERTED}/Wan2.1_VAE.safetensors"),
        t5_path=_m(f"{_CONVERTED}/models_t5_umt5-xxl-enc-bf16.safetensors"),
        tokenizer_path=_m("Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"),
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
        dit_path=_m("Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors"),
        vae_path=_m(f"{_CONVERTED}/Wan2.1_VAE.safetensors"),
        t5_path=_m(f"{_CONVERTED}/models_t5_umt5-xxl-enc-bf16.safetensors"),
        tokenizer_path=_m("Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"),
        dim=1536, num_layers=30, num_heads=12, ffn_dim=8960,
        vae_z_dim=16, vae_spatial_factor=8,
        has_pretrained_vace=False,                # SEAM-1: create_vace_from_dit
        vace_layers=tuple(range(0, 30, 2)),       # 15 layers (same as VACE-1.3B)
        first_frame_mode="none",                  # T2V has no native first-frame path
        # vace_remove_prefix defaults to "pipe.vace." -> LoRA saves as vace_blocks.*
    ),
    # T4: Wan2.1-I2V-14B-480P. New variables vs T3 (master plan §4.2): SEAM-4 first-frame
    # i2v_concat (input_image -> CLIP clip_feature + VAE-concat y, handled by the stock
    # pipeline's image-embedder units, gated by require_clip/vae_embedding) + 14B scale.
    # No native VACE branch -> SEAM-1 create_vace_from_dit (same as T3; full-param train,
    # only after_proj zero-init). Official Wan2.1-VACE-14B is structurally isomorphic and
    # can warm-start the from-DiT shell (optional; see provision strict-load path).
    # DiT is 7 sharded safetensors -> dit_glob. CLIP encoder via image_encoder_path.
    "wan2.1-i2v-14b-480p": WanBaseSpec(
        name="wan2.1-i2v-14b-480p",
        dit_path="",                              # unused; sharded -> dit_glob
        dit_glob=_m("Wan-AI/Wan2.1-I2V-14B-480P/diffusion_pytorch_model-*.safetensors"),
        vae_path=_m(f"{_CONVERTED}/Wan2.1_VAE.safetensors"),          # Wan2.1 VAE (z=16, s=8)
        t5_path=_m(f"{_CONVERTED}/models_t5_umt5-xxl-enc-bf16.safetensors"),
        tokenizer_path=_m("Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"),
        image_encoder_path=_m("Wan-AI/Wan2.1-I2V-14B-480P/models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth"),
        dim=5120, num_layers=40, num_heads=40, ffn_dim=13824,
        vae_z_dim=16, vae_spatial_factor=8,       # vace_in_dim=96, mask_pq=8 (Wan2.1 VAE)
        has_pretrained_vace=False,                # SEAM-1: create_vace_from_dit (warm-start optional)
        vace_layers=tuple(range(0, 40, 5)),       # 8 layers: (0,5,10,15,20,25,30,35)
        first_frame_mode="i2v_concat",            # SEAM-4: i2v first frame
        # from-DiT vace MUST be trained full-param (T3 lesson); run script uses
        # --trainable_models vace. vace_remove_prefix default "pipe.vace." -> vace_blocks.*
    ),
    # T5: Wan2.2-TI2V-5B. New variable vs T4 is SEAM-2: the Wan2.2 VAE has z=48 and
    # 16x spatial compression, so vace_in_dim=2*48+16^2=352 and mask_pq=16. Its native
    # first-frame mechanism is the fused TI2V path: `input_image` is encoded by the VAE
    # and written into latent frame 0; FlowMatchSFTLoss then excludes that first latent
    # from the training target. No pretrained VACE branch -> SEAM-1 create_vace_from_dit.
    "wan2.2-ti2v-5b": WanBaseSpec(
        name="wan2.2-ti2v-5b",
        dit_path="",                              # robust to one-file vs sharded layout
        dit_glob=_m("Wan-AI/Wan2.2-TI2V-5B/diffusion_pytorch_model*.safetensors"),
        vae_path=_m("Wan-AI/Wan2.2-TI2V-5B/Wan2.2_VAE.pth"),    # Wan2.2 VAE (z=48, s=16); .pth ok
        t5_path=_m("Wan-AI/Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=_m("Wan-AI/Wan2.2-TI2V-5B/google/umt5-xxl"),
        dim=3072, num_layers=30, num_heads=24, ffn_dim=14336,
        vae_z_dim=48, vae_spatial_factor=16,      # vace_in_dim=352, mask_pq=16
        has_pretrained_vace=False,                # SEAM-1: create_vace_from_dit
        vace_layers=tuple(range(0, 30, 2)),       # 15 layers: (0,2,...,28)
        first_frame_mode="ti2v_fused",            # SEAM-4: Wan2.2 fused first frame
    ),
    # T6: Wan2.2-I2V-A14B. New variable vs T4 is SEAM-5: a two-expert MoE. The base ships
    # TWO DiT experts (high_noise_model / low_noise_model, same 5120/40/40/13824 shape as
    # I2V-14B) and NO VACE branch -> SEAM-1 create_vace_from_dit is run per expert, giving
    # pipe.vace (from high DiT) and pipe.vace2 (from low DiT). The two VACE branches are
    # trained as independent single-expert jobs on disjoint timestep bands (official recipe
    # examples/wanvideo/.../full/Wan2.2-I2V-A14B.sh) and combined at inference via
    # DiffSynth's native switch (wan_video.py:314-317), boundary 0.875.
    # SEAM-4 first frame is i2v_vae: Wan2.2 I2V drops CLIP (require_clip_embedding=False,
    # has_image_input=False) and conditions on input_image purely via the VAE-concat y.
    # VAE/T5/tokenizer are reused from the converted assets (Wan2.1 VAE -> 96/8).
    "wan2.2-i2v-a14b": WanBaseSpec(
        name="wan2.2-i2v-a14b",
        dit_path="",                              # unused; experts -> expert_dit_globs
        vae_path=_m(f"{_CONVERTED}/Wan2.1_VAE.safetensors"),         # Wan2.1 VAE (z=16, s=8)
        t5_path=_m(f"{_CONVERTED}/models_t5_umt5-xxl-enc-bf16.safetensors"),
        tokenizer_path=_m("Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"),
        dim=5120, num_layers=40, num_heads=40, ffn_dim=13824,
        vae_z_dim=16, vae_spatial_factor=8,       # vace_in_dim=96, mask_pq=8 (Wan2.1 VAE)
        has_pretrained_vace=False,                # SEAM-1: create_vace_from_dit (per expert)
        vace_layers=tuple(range(0, 40, 5)),       # 8 layers: (0,5,10,15,20,25,30,35)
        first_frame_mode="i2v_vae",               # SEAM-4: Wan2.2 I2V, VAE-concat, no CLIP
        experts=("high", "low"),                  # SEAM-5: dual-expert MoE
        expert_dit_globs={
            "high": _m("Wan-AI/Wan2.2-I2V-A14B/high_noise_model/diffusion_pytorch_model-*.safetensors"),
            "low": _m("Wan-AI/Wan2.2-I2V-A14B/low_noise_model/diffusion_pytorch_model-*.safetensors"),
        },
        switch_boundary=0.875,                    # inference expert switch (timestep)
        train_bands={                             # (min, max) timestep-boundary fractions
            "high": (0.0, 0.358),                 # high-noise expert -> timesteps [900,1000]
            "low": (0.358, 1.0),                  # low-noise expert  -> timesteps [0,900)
        },
    ),
    # Warm-start comparison vs T6: PAI/Wan2.2-VACE-Fun-A14B ships PRETRAINED dual VACE
    # branches (high/low), so has_pretrained_vace=True (no from-DiT build). Its DiT is
    # T2V-style in_dim=16 (config.json: VaceWanModel, dim5120/40L, vace_layers (0,5..35),
    # vace_in_dim96) — control flows entirely through the VACE branch + vace_reference_image
    # (first_frame_mode="vace_reference", NOT i2v). Structurally == Wan2.1-VACE-14B (hash
    # 7a513e1f), so each single-file noise checkpoint loads as DiT + VACE bundled (same
    # mechanism as VACE-1.3B). Reuses Wan2.1 VAE (96/8) + umt5 T5/tokenizer. Trained
    # full-param vace (official recipe) on the same RoboTwin data as T6 for the comparison.
    "wan2.2-vace-fun-a14b": WanBaseSpec(
        name="wan2.2-vace-fun-a14b",
        dit_path="",                              # experts -> expert_dit_globs
        vae_path=_m(f"{_CONVERTED}/Wan2.1_VAE.safetensors"),         # Wan2.1 VAE (z=16, s=8)
        t5_path=_m(f"{_CONVERTED}/models_t5_umt5-xxl-enc-bf16.safetensors"),
        tokenizer_path=_m("Wan-AI/Wan2.1-T2V-1.3B/google/umt5-xxl"),
        dim=5120, num_layers=40, num_heads=40, ffn_dim=13824,
        vae_z_dim=16, vae_spatial_factor=8,       # vace_in_dim=96, mask_pq=8 (Wan2.1 VAE)
        has_pretrained_vace=True,                 # PRETRAINED dual VACE (warm-start)
        vace_layers=tuple(range(0, 40, 5)),       # 8 layers: (0,5,10,15,20,25,30,35)
        first_frame_mode="vace_reference",        # SEAM-4: VACE reference (in_dim=16 DiT, no i2v)
        experts=("high", "low"),                  # SEAM-5: dual-expert MoE (pretrained)
        expert_dit_globs={
            "high": _m("PAI/Wan2.2-VACE-Fun-A14B/high_noise_model/diffusion_pytorch_model*.safetensors"),
            "low": _m("PAI/Wan2.2-VACE-Fun-A14B/low_noise_model/diffusion_pytorch_model*.safetensors"),
        },
        switch_boundary=0.875,
        train_bands={
            "high": (0.0, 0.358),
            "low": (0.358, 1.0),
        },
    ),
}


def get_spec(name: str) -> WanBaseSpec:
    if name not in REGISTRY:
        raise KeyError(f"Unknown base spec {name!r}. Known: {sorted(REGISTRY)}")
    return REGISTRY[name]


__all__ = ["WanBaseSpec", "REGISTRY", "get_spec"]
