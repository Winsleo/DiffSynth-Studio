#!/usr/bin/env bash
# A2V Track-T3 去留闸门: single-sample overfit of a VACE branch *built from the DiT*
# (SEAM-1) on top of Wan2.1-T2V-1.3B — the base that ships NO VACE weights.
#
# FULL-PARAM (not LoRA), by necessity. The from-DiT branch is zero-initialized at its
# only control entry/exit points — `vace_patch_embedding` (Conv3d) and each block's
# `after_proj`/`before_proj` — so the control trajectory cannot influence the output
# until those are trained away from zero. LoRA targets only q,k,v,o,ffn, so a LoRA run
# leaves patch_embedding/after_proj frozen at zero and the control has ZERO effect
# (gen(real) == gen(none) bit-for-bit). ABot trains the full vace branch for the same
# reason (training/train_a2v.py: trainable defaults to "vace" when action-conditioned).
# So T3 trains `--trainable_models vace`; the saved checkpoint is the full vace state
# dict (keys vace_patch_embedding.* / vace_blocks.*), loaded back by a2v/infer_a2v.py.
#
# The ONLY new variable vs the passed T1/T2 gate is SEAM-1 (from-DiT vace); the SAME
# physical-radius dataset + recipe are held fixed.
#
# Prereqs:
#   .cache/a2v_robotwin/ep0_dataset_phys                                   (physical dataset)
#   models/Wan-AI/Wan2.1-T2V-1.3B/diffusion_pytorch_model.safetensors      (T2V DiT, no vace)
#
# Run from the DiffSynth-Studio repo root:
#   bash a2v/run_overfit_t2v.sh
#
# Causal gate afterwards (reuses a2v/infer_a2v.py, which auto-detects a full vace ckpt):
#   .venv/bin/python -m a2v.infer_a2v --base_spec wan2.1-t2v-1.3b \
#     --dataset .cache/a2v_robotwin/ep0_dataset_phys \
#     --lora models/train/a2v_robotwin_ep0_vace_t2v/step-1000.safetensors \
#     --num_frames 121 --height 240 --width 320 --control real \
#     --output .cache/a2v_robotwin/gen_t2v_real.mp4
#   (... --control none -> arm should NOT reproduce GT; real should.)
set -euo pipefail
cd "$(dirname "$0")/.."

SPEC=wan2.1-t2v-1.3b
DATASET="${DATASET:-.cache/a2v_robotwin/ep0_dataset_phys}"
OUT="${OUT:-models/train/a2v_robotwin_ep0_vace_t2v}"
ACCEL=.venv/bin/accelerate

# model_paths/tokenizer/remove_prefix derived from --base_spec (a2v/base_spec.py).
# --trainable_models vace: full-param vace branch (no LoRA); train_a2v's subclass builds
# the branch from the DiT (SEAM-1) before freeze_except("vace") unfreezes it.
"$ACCEL" launch --num_processes 1 --mixed_precision bf16 -m a2v.train_a2v \
  --base_spec "$SPEC" \
  --dataset_base_path "$DATASET" \
  --dataset_metadata_path "$DATASET/metadata.jsonl" \
  --data_file_keys "video,vace_video,vace_reference_image" \
  --extra_inputs "vace_video,vace_reference_image" \
  --height 240 --width 320 --num_frames 121 \
  --trainable_models vace \
  --learning_rate 1e-4 \
  --dataset_repeat 100 \
  --num_epochs 10 \
  --save_steps 100 \
  --output_path "$OUT" \
  --use_gradient_checkpointing_offload \
  --enable_tensorboard_log
