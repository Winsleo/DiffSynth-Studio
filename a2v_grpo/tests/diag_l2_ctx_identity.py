"""Why do two different episodes yield identical `vace_context`?

L2 self-invalidated with mean|diff| == 0 between episode231 and episode240, whose trajectories
are measurably different (~30 px mean path distance). Either
  (a) the two prepares alias the same tensor, or
  (b) they really do receive the same frames (a bug in how we index the manifest).

Prints tensor identity, storage pointers, and a fingerprint of the *input frames* so the two
cases are distinguishable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from a2v.base_spec import REGISTRY
from a2v.infer_a2v import NEG_PROMPT, build_pipe_with_vace, first_frame_kwargs, load_frames
from a2v_grpo.rollout import prepare_a2v_inputs


def fp(frames) -> str:
    h = hashlib.sha1()
    for f in frames[:8]:
        h.update(np.asarray(f.convert("RGB"), dtype=np.uint8).tobytes())
    return h.hexdigest()[:12]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_spec", default="wan2.2-ti2v-5b")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--metadata", default="metadata.jsonl")
    ap.add_argument("--num_frames", type=int, default=121)
    args = ap.parse_args()

    dataset = Path(args.dataset).resolve()
    rows, seen = [], set()
    with open(dataset / args.metadata, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ep = (r.get("vace_video") or [""])[0].split("/")[0]
            if ep not in seen:
                seen.add(ep)
                rows.append(r)
            if len(rows) == 2:
                break

    print("[diag] rows chosen")
    for i, r in enumerate(rows):
        print(f"  {i}: id={r.get('id')}")
        print(f"      vace_video[0] = {r['vace_video'][0]}")
        print(f"      vace_video[60]= {r['vace_video'][60]}")
        print(f"      ref           = {r['vace_reference_image']}")

    # (b) do the loaded frames actually differ?
    print("\n[diag] input frame fingerprints (first 8 frames)")
    fa = load_frames(dataset, rows[0]["vace_video"][: args.num_frames])
    fb = load_frames(dataset, rows[1]["vace_video"][: args.num_frames])
    print(f"  A: {fp(fa)}   n={len(fa)} size={fa[0].size}")
    print(f"  B: {fp(fb)}   n={len(fb)} size={fb[0].size}")
    same_frames = fp(fa) == fp(fb)
    print(f"  frames identical? {same_frames}")
    arr_a = np.stack([np.asarray(f.convert("RGB"), np.int16) for f in fa[:20]])
    arr_b = np.stack([np.asarray(f.convert("RGB"), np.int16) for f in fb[:20]])
    print(f"  pixel mean|diff| over first 20 frames = {np.abs(arr_a - arr_b).mean():.4f}")

    spec = REGISTRY[args.base_spec]
    pipe, spec = build_pipe_with_vace(args.base_spec, args.ckpt, 1.0)

    def prep(row, frames):
        ref = Image.open(dataset / row["vace_reference_image"]).convert("RGB")
        return prepare_a2v_inputs(
            pipe,
            prompt=row.get("prompt", "x"),
            negative_prompt=NEG_PROMPT,
            vace_video=frames,
            height=480, width=832, num_frames=args.num_frames,
            num_inference_steps=4, seed=0, tiled=False,
            **first_frame_kwargs(spec, ref),
        )

    pa = prep(rows[0], fa)
    ca = pa["inputs_shared"]["vace_context"]
    ptr_a, sig_a = ca.data_ptr(), ca.float().abs().sum().item()
    pb = prep(rows[1], fb)
    cb = pb["inputs_shared"]["vace_context"]

    print("\n[diag] vace_context identity")
    print(f"  ca is cb            : {ca is cb}")
    print(f"  data_ptr            : A={hex(ptr_a)}  B={hex(cb.data_ptr())}  same={ptr_a == cb.data_ptr()}")
    print(f"  shapes              : {tuple(ca.shape)} / {tuple(cb.shape)}")
    print(f"  abs-sum A (captured before B was built) = {sig_a:.6f}")
    print(f"  abs-sum A (now)                          = {ca.float().abs().sum().item():.6f}"
          f"   <- differs from the captured value => A was mutated in place")
    print(f"  abs-sum B                                = {cb.float().abs().sum().item():.6f}")
    print(f"  mean|A-B| = {(ca.float() - cb.float()).abs().mean().item():.6e}")

    print("\n[diag] verdict")
    if same_frames:
        print("    the two rows feed IDENTICAL frames -> manifest indexing bug in the test")
    elif ca is cb or ptr_a == cb.data_ptr():
        print("    the two prepares ALIAS one tensor -> prepare_a2v_inputs leaks state; the "
              "per-episode conditioning cache in flow_grpo/diffsynth_a2v.py would be corrupt")
    elif abs(sig_a - ca.float().abs().sum().item()) > 1e-6:
        print("    A was mutated in place while building B -> same conclusion as aliasing")
    else:
        print("    frames differ, tensors are distinct and unmutated, yet contexts match: "
              "look at the VACE unit inputs (vace_video may not be reaching it)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
