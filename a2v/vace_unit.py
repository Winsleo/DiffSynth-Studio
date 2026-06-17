"""SEAM-2: parameterized VACE context unit.

The stock `WanVideoUnit_VACE` (`diffsynth/pipelines/wan_video.py:649`) hardcodes the
mask-patchify factor `P=Q=8`, which is correct only for the Wan2.1 VAE (8x spatial).
Wan2.2 (TI2V-5B) needs `P=Q=16`. Per the master plan, this is the ONLY place where the
DiffSynth data flow must vary by base model — and we parameterize it WITHOUT editing the
original file, by subclassing and replacing the unit instance in `pipe.units`.

`process` below is a verbatim copy of the stock body; the SOLE change is
`P=8, Q=8` -> `P=self.mask_pq, Q=self.mask_pq`. At `mask_pq=8` it is provably identical to
stock (proven by `a2v/check_parity.py`).
"""

from __future__ import annotations

import torch
from einops import rearrange

from diffsynth.pipelines.wan_video import WanVideoUnit_VACE


class ParamWanVideoUnit_VACE(WanVideoUnit_VACE):
    def __init__(self, mask_pq: int = 8):
        super().__init__()
        self.mask_pq = int(mask_pq)

    def process(
        self,
        pipe,
        vace_video, vace_video_mask, vace_reference_image, vace_scale,
        height, width, num_frames,
        tiled, tile_size, tile_stride
    ):
        if vace_video is not None or vace_video_mask is not None or vace_reference_image is not None:
            pipe.load_models_to_device(["vae"])
            if vace_video is None:
                vace_video = torch.zeros((1, 3, num_frames, height, width), dtype=pipe.torch_dtype, device=pipe.device)
            else:
                vace_video = pipe.preprocess_video(vace_video)

            if vace_video_mask is None:
                vace_video_mask = torch.ones_like(vace_video)
            else:
                vace_video_mask = pipe.preprocess_video(vace_video_mask, min_value=0, max_value=1)

            inactive = vace_video * (1 - vace_video_mask) + 0 * vace_video_mask
            reactive = vace_video * vace_video_mask + 0 * (1 - vace_video_mask)
            inactive = pipe.vae.encode(inactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            reactive = pipe.vae.encode(reactive, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
            vace_video_latents = torch.concat((inactive, reactive), dim=1)

            # SEAM-2: P=Q parameterized by VAE spatial factor (8 for Wan2.1, 16 for Wan2.2).
            vace_mask_latents = rearrange(vace_video_mask[0, 0], "T (H P) (W Q) -> 1 (P Q) T H W", P=self.mask_pq, Q=self.mask_pq)
            vace_mask_latents = torch.nn.functional.interpolate(vace_mask_latents, size=((vace_mask_latents.shape[2] + 3) // 4, vace_mask_latents.shape[3], vace_mask_latents.shape[4]), mode='nearest-exact')

            if vace_reference_image is None:
                pass
            else:
                if not isinstance(vace_reference_image, list):
                    vace_reference_image = [vace_reference_image]

                vace_reference_image = pipe.preprocess_video(vace_reference_image)

                bs, c, f, h, w = vace_reference_image.shape
                new_vace_ref_images = []
                for j in range(f):
                    new_vace_ref_images.append(vace_reference_image[0, :, j:j+1])
                vace_reference_image = new_vace_ref_images

                vace_reference_latents = pipe.vae.encode(vace_reference_image, device=pipe.device, tiled=tiled, tile_size=tile_size, tile_stride=tile_stride).to(dtype=pipe.torch_dtype, device=pipe.device)
                vace_reference_latents = torch.concat((vace_reference_latents, torch.zeros_like(vace_reference_latents)), dim=1)
                vace_reference_latents = [u.unsqueeze(0) for u in vace_reference_latents]

                vace_video_latents = torch.concat((*vace_reference_latents, vace_video_latents), dim=2)
                vace_mask_latents = torch.concat((torch.zeros_like(vace_mask_latents[:, :, :f]), vace_mask_latents), dim=2)

            vace_context = torch.concat((vace_video_latents, vace_mask_latents), dim=1)
            return {"vace_context": vace_context, "vace_scale": vace_scale}
        else:
            return {"vace_context": None, "vace_scale": vace_scale}


def install_vace_unit(pipe, mask_pq: int) -> bool:
    """Replace the stock WanVideoUnit_VACE in `pipe.units` with a mask_pq-parameterized copy.

    Returns True if a unit was replaced. Idempotent: re-running updates mask_pq in place.
    """
    for i, unit in enumerate(pipe.units):
        if isinstance(unit, WanVideoUnit_VACE):
            pipe.units[i] = ParamWanVideoUnit_VACE(mask_pq=mask_pq)
            return True
    return False


__all__ = ["ParamWanVideoUnit_VACE", "install_vace_unit"]
