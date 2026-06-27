#!/usr/bin/env python3
"""A2V overfit-gate inference: reconstruct an episode from its control video.

Loads the base Wan2.1-VACE-1.3B locally, applies a trained VACE LoRA, feeds the
prepared ``vace_video`` (trajectory-map PNGs) + ``vace_reference_image`` + prompt
for one dataset row, and generates a video at the SAME resolution/length used in
training. The gate passes when the generated arm follows the trajectory the
control encodes (compare to the GT ``video/`` frames).

Usage:
    .venv/bin/python -m a2v.infer_a2v \
        --dataset .cache/a2v_robotwin/ep0_dataset \
        --lora models/train/a2v_robotwin_ep0_lora/epoch-9.safetensors \
        --num_frames 121 --height 240 --width 320 --row 0 \
        --output .cache/a2v_robotwin/ep0_gen.mp4

Add ``--control none`` or ``--control shuffle`` for the negative control: with no
/ scrambled trajectory the output should NOT reconstruct the GT, proving the
control signal is what drives generation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig
from diffsynth.utils.data import save_video

from a2v.base_spec import get_spec
from a2v.provision import build_bare_vace, provision_a2v

REPO_ROOT = Path(__file__).resolve().parents[1]

NEG_PROMPT = (
    "色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，"
    "最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，"
    "畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走"
)


def _load_vace_weights(pipe, vace, ckpt_path: str, lora_alpha: float) -> None:
    """Load a trained VACE checkpoint into `vace`, auto-detecting its kind.

    Two training regimes produce different checkpoints (see a2v/train.sh):
      * LoRA-on-VACE (T1/T2, pretrained vace) -> keys contain ``lora_A``/``lora_B`` ->
        merged via ``pipe.load_lora``.
      * Full-param VACE (T3, from-DiT vace) -> a complete vace state dict
        (``vace_patch_embedding.*`` / ``vace_blocks.*``) -> ``load_state_dict``. The
        from-DiT branch MUST be trained full-param: its zero-init control entry/exit
        (patch_embedding, after_proj) are not LoRA targets, so a LoRA there is inert.
    """
    from safetensors import safe_open

    with safe_open(ckpt_path, "pt") as f:
        keys = list(f.keys())
    is_lora = any("lora_A" in k or "lora_B" in k or ".lora_" in k for k in keys)

    if is_lora:
        pipe.load_lora(vace, ckpt_path, alpha=lora_alpha)
        print(f"Loaded VACE LoRA ({len(keys)} keys) from {ckpt_path}")
        return

    from diffsynth.core import load_state_dict
    sd = load_state_dict(ckpt_path)
    sd = {k: v.to(dtype=vace.vace_patch_embedding.weight.dtype,
                  device=vace.vace_patch_embedding.weight.device) for k, v in sd.items()}
    missing, unexpected = vace.load_state_dict(sd, strict=False)
    print(f"Loaded full VACE state dict ({len(sd)} keys) from {ckpt_path} "
          f"(missing={len(missing)} unexpected={len(unexpected)})")
    if unexpected:
        raise RuntimeError(f"Full-VACE checkpoint has unexpected keys, e.g. {unexpected[:5]}")
    # `missing` is expected to be empty for a full vace save; a non-empty set means the
    # checkpoint did not cover the whole branch (control entry/exit may stay at zero-init).
    if missing:
        print(f"  WARNING: {len(missing)} vace params not in checkpoint, e.g. {missing[:5]}")


def load_row(dataset: Path, row_idx: int) -> dict:
    rows = []
    with (dataset / "metadata.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows[row_idx]


def load_frames(dataset: Path, rel_paths: list[str]) -> list[Image.Image]:
    return [Image.open(dataset / p).convert("RGB") for p in rel_paths]


def build_pipe_with_vace(base_spec: str, ckpt_path: str, lora_alpha: float = 1.0,
                         ckpt_path_low: str | None = None):
    """Build a WanVideoPipeline, provision A2V seams, and load trained VACE weights.

    Single-expert specs: `ckpt_path` is loaded into the one VACE branch.
    Dual-expert specs (SEAM-5 A14B): both DiT experts are loaded, vace + vace2 are built
    from-DiT, the HIGH-noise ckpt (`ckpt_path`) goes into `pipe.vace`, and the LOW-noise
    ckpt (`ckpt_path_low`, required) into `pipe.vace2`. DiffSynth's native expert switch
    (switch_DiT_boundary, default == spec.switch_boundary) routes per timestep at inference.
    """
    spec = get_spec(base_spec)
    # Dual-expert (A14B) loads BOTH 14B DiTs at once; even with the native switch keeping
    # one active, two resident experts + activations exceed 80GB. Enable CPU offload so the
    # inactive expert streams to CPU (active expert + the post-hoc from-DiT VACE branches
    # stay on GPU). Single-expert specs keep the all-on-GPU path (unchanged).
    from_kwargs = {}
    if spec.experts:
        vram_config = {
            "offload_dtype": torch.bfloat16, "offload_device": "cpu",
            "onload_dtype": torch.bfloat16, "onload_device": "cuda",
            "preparing_dtype": torch.bfloat16, "preparing_device": "cuda",
            "computation_dtype": torch.bfloat16, "computation_device": "cuda",
        }
        model_configs = [ModelConfig(path=p, **vram_config) for p in spec.model_paths()]
        # leave headroom for the two unmanaged VACE branches (~12GB) + activations
        from_kwargs["vram_limit"] = torch.cuda.mem_get_info("cuda")[1] / (1024 ** 3) - 16
    else:
        model_configs = [ModelConfig(path=p) for p in spec.model_paths()]
    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=model_configs,
        tokenizer_config=ModelConfig(spec.abs_tokenizer_path()),
        **from_kwargs,
    )
    vace = provision_a2v(pipe, spec)
    if spec.experts:
        if ckpt_path_low is None:
            raise ValueError(
                f"{spec.name} is a dual-expert MoE; pass the low-noise checkpoint via "
                "--lora_low (high-noise -> --lora)."
            )
        dev = pipe.device
        if spec.has_pretrained_vace:
            # Pretrained (Wan2.2-VACE-Fun-A14B): from_pretrained's vace/vace2 are vram-managed
            # (wrapped -> `.module.` state_dict keys), which a standard vace checkpoint can't
            # load into. We only need the TRAINED weights, so build fresh plain GPU-resident
            # branches and load the ckpts (same end-state as the from-DiT path: un-managed
            # vace pinned on GPU while the native switch offloads the inactive DiT).
            hii = bool(getattr(pipe.dit, "has_image_input", False))
            pipe.vace = build_bare_vace(spec, has_image_input=hii, dtype=pipe.torch_dtype, device=dev)
            pipe.vace2 = build_bare_vace(spec, has_image_input=hii, dtype=pipe.torch_dtype, device=dev)
        else:
            # from-DiT VACE branches (Wan2.2-I2V-A14B) are built post-hoc from possibly-
            # offloaded (cpu) DiTs and are NOT vram-managed -> pin them to the compute device.
            pipe.vace = pipe.vace.to(dev)
            pipe.vace2 = pipe.vace2.to(dev)
        vace = pipe.vace
        _load_vace_weights(pipe, pipe.vace, ckpt_path, lora_alpha)
        _load_vace_weights(pipe, pipe.vace2, ckpt_path_low, lora_alpha)
    else:
        _load_vace_weights(pipe, vace, ckpt_path, lora_alpha)
    return pipe, spec


def _control_vace_video(vace_video: list[Image.Image], control: str, width: int, height: int) -> list[Image.Image]:
    if control == "real":
        return vace_video
    if control == "none":
        return [Image.new("RGB", (width, height), (128, 128, 128)) for _ in vace_video]
    if control == "shuffle":
        idx = list(range(len(vace_video)))[::-1]
        return [vace_video[i] for i in idx]
    raise ValueError(f"unknown control: {control}")


def first_frame_kwargs(spec, ref: Image.Image) -> dict:
    # SEAM-4 first frame: route `ref` (= GT frame 0, dataset invariant I4) by base mode.
    #   * i2v_concat (I2V-14B, T4): `input_image` builds CLIP + VAE concat conditioning.
    #   * ti2v_fused (TI2V-5B, T5): `input_image` is VAE-encoded into latent frame 0.
    #   * vace_reference / legacy none: keep the prior VACE reference path.
    # These are alternatives, not additive: native first-frame paths already inject frame 0,
    # so we do NOT also pass vace_reference_image there. The causal control (real vs none)
    # still lives entirely in vace_video.
    if spec.first_frame_mode in ("i2v_concat", "i2v_vae", "ti2v_fused"):
        return {"input_image": ref}
    return {"vace_reference_image": ref}


def generate_one(pipe, spec, row: dict, dataset: Path, control: str, height: int, width: int,
                 num_frames: int, seed: int, num_inference_steps: int | None = None):
    """Generate one dataset row and return the pipeline video frames.

    ``num_inference_steps`` defaults to None (the pipeline's own default, unchanged);
    pass a smaller value (e.g. from in-training periodic sampling) to trade quality for
    speed.
    """
    vace_video = load_frames(dataset, row["vace_video"][:num_frames])
    vace_video = _control_vace_video(vace_video, control, width, height)
    ref = Image.open(dataset / row["vace_reference_image"]).convert("RGB")
    extra = {} if num_inference_steps is None else {"num_inference_steps": num_inference_steps}
    return pipe(
        prompt=row.get("prompt", "robot arm manipulation"),
        negative_prompt=NEG_PROMPT,
        vace_video=vace_video,
        height=height,
        width=width,
        num_frames=num_frames,
        seed=seed,
        tiled=False,
        **first_frame_kwargs(spec, ref),
        **extra,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base_spec", default="wan2.1-vace-1.3b", help="WanBaseSpec name (a2v.base_spec.REGISTRY).")
    parser.add_argument("--lora", required=True,
                        help="Trained VACE checkpoint. For a dual-expert spec this is the "
                             "HIGH-noise expert ckpt (pair with --lora_low).")
    parser.add_argument("--lora_low", default=None,
                        help="Low-noise expert VACE ckpt (required for dual-expert specs).")
    parser.add_argument("--lora_alpha", type=float, default=1.0)
    parser.add_argument("--row", type=int, default=0)
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--control", choices=("real", "none", "shuffle"), default="real",
                        help="real=use prepared vace_video; none/shuffle=negative control.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=int, default=15)
    args = parser.parse_args()

    dataset = Path(args.dataset).resolve()
    row = load_row(dataset, args.row)
    pipe, spec = build_pipe_with_vace(args.base_spec, args.lora, args.lora_alpha,
                                      ckpt_path_low=args.lora_low)
    video = generate_one(
        pipe=pipe,
        spec=spec,
        row=row,
        dataset=dataset,
        control=args.control,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        seed=args.seed,
    )
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    save_video(video, str(out), fps=args.fps, quality=5)
    print(f"Saved generated video -> {out} (control={args.control}, prompt={row.get('prompt')!r})")


if __name__ == "__main__":
    main()
