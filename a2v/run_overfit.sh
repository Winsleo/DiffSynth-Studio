#!/usr/bin/env bash
# A2V Track-T1 去留闸门: single-sample LoRA overfit of the VACE branch on one
# RoboTwin episode. Pass = the trained LoRA can reconstruct that episode from its
# trajectory-map control video (run a2v/infer_a2v.py afterwards).
#
# Prereq — build the single-episode dataset (run once):
#   PY=.venv/bin/python
#   $PY -m a2v.data.robotwin_adapter \
#     --task beat_block_hammer --robot_mode aloha-agilex_clean_50 --episodes 0 \
#     --work_dir .cache/a2v_robotwin/ep0_work
#   $PY -m a2v.data.prepare \
#     --manifest .cache/a2v_robotwin/ep0_work/manifest.jsonl \
#     --output_dir .cache/a2v_robotwin/ep0_dataset_phys \
#     --height 240 --width 320 --num_frames 121 \
#     --resize_mode stretch --gripper_z_offset 0 \
#     --radius_mode physical --write_overlay
#
# Then run this script from the DiffSynth-Studio repo root:
#   bash a2v/run_overfit.sh
#
# NOTE (2026-06-13): default dataset is now the physical-radius build (§9/§10 of
# A2V_HANDOFF.md). The remove_prefix save bug is fixed in a2v/train_a2v.py, so the
# LoRA saved here has load-ready keys (vace_blocks.*) natively — no manual fixup.
set -euo pipefail
cd "$(dirname "$0")/.."

# Overridable via env (run_overfit_t2v.sh sets SPEC/OUT for the T3 base). Defaults = T1.
DATASET="${DATASET:-.cache/a2v_robotwin/ep0_dataset_phys}"
OUT="${OUT:-models/train/a2v_robotwin_ep0_lora_phys}"
SPEC="${SPEC:-wan2.1-vace-1.3b}"   # WanBaseSpec: supplies model_paths/tokenizer/lora defaults + drives VACE seams

ACCEL=.venv/bin/accelerate

# Model paths, tokenizer, lora_base_model and remove_prefix are all derived from --base_spec
# (a2v/base_spec.py). To override any, pass the explicit flag — it wins over the spec default.
"$ACCEL" launch --num_processes 1 --mixed_precision bf16 -m a2v.train_a2v \
  --base_spec "$SPEC" \
  --dataset_base_path "$DATASET" \
  --dataset_metadata_path "$DATASET/metadata.jsonl" \
  --data_file_keys "video,vace_video,vace_reference_image" \
  --extra_inputs "vace_video,vace_reference_image" \
  --height 240 --width 320 --num_frames 121 \
  --lora_rank 32 \
  --learning_rate 1e-4 \
  --dataset_repeat 100 \
  --num_epochs 10 \
  --save_steps 100 \
  --output_path "$OUT" \
  --use_gradient_checkpointing_offload \
  --enable_tensorboard_log
