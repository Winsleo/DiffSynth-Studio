#!/usr/bin/env python3
"""Track T2 对拍 gate — prove the abstraction is a zero behavioral change for VACE-1.3B.

Two seams could in principle alter behavior:
  * SEAM-1 (ensure_vace): must return the *same* pretrained VACE module object.
  * SEAM-2 (parameterized vace_unit): at mask_pq=8 it must produce a bit-identical
    `vace_context` to the stock `WanVideoUnit_VACE` (hardcoded P=Q=8).

We load VACE-1.3B once, take ep0's prepared control video + reference, and run BOTH the
stock unit and `ParamWanVideoUnit_VACE(mask_pq=8)` on identical inputs with the RNG reset to
the same seed before each call (so the VAE-encoded video latents — computed independently in
each unit — match, and the only thing under test is the mask path). Bit-identical `vace_context`
+ `ensure_vace is pipe.vace` ⇒ the whole forward is unchanged ⇒ any future seam bug (T3/T5) is
isolated to the abstraction layer.

Run: .venv/bin/python -m a2v.check_t2_parity [--dataset .cache/a2v_robotwin/ep0_dataset]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from PIL import Image

from diffsynth.pipelines.wan_video import WanVideoPipeline, ModelConfig, WanVideoUnit_VACE

from a2v.base_spec import get_spec
from a2v.provision import ensure_vace
from a2v.vace_unit import ParamWanVideoUnit_VACE


def _load_frames(dataset: Path, rel_paths) -> list[Image.Image]:
    return [Image.open(dataset / p).convert("RGB") for p in rel_paths]


def _run_unit(unit, pipe, kwargs, seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return unit.process(pipe, **kwargs)["vace_context"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default=".cache/a2v_robotwin/ep0_dataset")
    parser.add_argument("--base_spec", default="wan2.1-vace-1.3b")
    parser.add_argument("--num_frames", type=int, default=121)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    spec = get_spec(args.base_spec)
    dataset = Path(args.dataset).resolve()
    row = json.loads((dataset / "metadata.jsonl").read_text().splitlines()[0])
    vace_video = _load_frames(dataset, row["vace_video"][: args.num_frames])
    ref = Image.open(dataset / row["vace_reference_image"]).convert("RGB")
    height, width = ref.size[1], ref.size[0]

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device="cuda",
        model_configs=[ModelConfig(path=p) for p in spec.model_paths()],
        tokenizer_config=ModelConfig(spec.abs_tokenizer_path()),
    )

    # --- SEAM-1: ensure_vace must return the loaded pretrained module, unchanged ---
    assert ensure_vace(pipe, spec) is pipe.vace, "SEAM-1: ensure_vace did not return pipe.vace"
    print("SEAM-1 ok: ensure_vace(pipe, spec) is pipe.vace")

    # --- SEAM-2: stock vs parameterized(mask_pq=8) vace_context, identical seed ---
    kwargs = dict(
        vace_video=vace_video, vace_video_mask=None, vace_reference_image=ref, vace_scale=1.0,
        height=height, width=width, num_frames=args.num_frames,
        tiled=False, tile_size=(30, 52), tile_stride=(15, 26),
    )
    stock_ctx = _run_unit(WanVideoUnit_VACE(), pipe, kwargs, args.seed)
    param_ctx = _run_unit(ParamWanVideoUnit_VACE(mask_pq=spec.mask_pq), pipe, kwargs, args.seed)

    assert spec.mask_pq == 8, f"expected mask_pq=8 for {spec.name}, got {spec.mask_pq}"
    assert stock_ctx.shape == param_ctx.shape, f"shape mismatch {stock_ctx.shape} vs {param_ctx.shape}"
    max_abs = (stock_ctx.float() - param_ctx.float()).abs().max().item()
    identical = torch.equal(stock_ctx, param_ctx)
    print(f"SEAM-2: vace_context shape={tuple(stock_ctx.shape)} "
          f"(vace_in_dim={spec.vace_in_dim}) max_abs_diff={max_abs:g} bit_identical={identical}")
    assert identical, f"SEAM-2 parity FAILED: max_abs_diff={max_abs}"

    print("T2 parity: OK")


if __name__ == "__main__":
    main()
