#!/usr/bin/env bash
# A2V single-sample overfit ("去留闸门") for ANY registered Wan base model.
#
#   Usage:  bash a2v/run_overfit.sh <base_spec>
#
#   <base_spec> is one of (a2v/base_spec.py REGISTRY):
#     wan2.1-vace-1.3b     T1  pretrained VACE  -> LoRA            (reference first-frame)
#     wan2.1-t2v-1.3b      T3  from-DiT VACE    -> full-param vace (reference first-frame)
#     wan2.1-i2v-14b-480p  T4  from-DiT VACE    -> full-param vace (i2v_concat first-frame)
#     wan2.2-ti2v-5b       T5  from-DiT VACE    -> full-param vace (ti2v_fused first-frame)
#     wan2.2-i2v-a14b      T6  from-DiT VACE    -> full-param vace (i2v_vae first-frame),
#                              SEAM-5 dual-expert MoE: run TWICE with EXPERT=high then
#                              EXPERT=low (each trains one expert on its timestep band).
#     wan2.2-vace-fun-a14b warm-start: PRETRAINED dual VACE (Fun-A14B), vace_reference
#                              first-frame; same EXPERT=high|low dual-job pattern as above.
#
# Why the presets differ (set per base below; override any with env vars):
#   * first frame: reference bases feed `vace_reference_image`; i2v/ti2v feed `input_image`
#     (the stock trainer derives it from video[0]) — these are alternatives, never both.
#   * train mode: a pretrained VACE branch trains via LoRA; a from-DiT branch MUST train
#     full-param (`--trainable_models vace`) — its zero-init control entry/exit
#     (vace_patch_embedding / after_proj) are not LoRA targets, so a LoRA there is inert.
#   * optimizer: 1.3B uses fp32 AdamW + gradient-checkpointing offload; 14B/5B use 8-bit
#     Adam so the optimizer state fits one 80GB GPU (lr 1e-5 — 1e-4 diverged on 14B, §13).
#   * resolution: TI2V's Wan2.2 VAE is 16x spatial x patch 2 = 32x, so H,W must be /32 ->
#     use the 256x320 dataset (240 is not divisible by 32).
#
# Env overrides:  LR=  DATASET=  OUT=  HEIGHT=  WIDTH=  FRAMES=  REPEAT=  EPOCHS=  NPROC=  DRY_RUN=1
#   wan2.2-i2v-a14b / wan2.2-vace-fun-a14b:  EXPERT=high|low (REQUIRED) selects the MoE expert + timestep band.
#
# Causal gate afterwards (real reconstructs GT; none/shuffle do not):
#   for c in real none; do .venv/bin/python -m a2v.infer_a2v --base_spec <base_spec> \
#     --dataset <DATASET> --lora <OUT>/step-1000.safetensors \
#     --height <H> --width <W> --num_frames 121 --control $c \
#     --output .cache/a2v_robotwin/gen_$c.mp4; done
#   .venv/bin/python -m a2v.causal_metrics --dataset <DATASET> \
#     --real .cache/a2v_robotwin/gen_real.mp4 --none .cache/a2v_robotwin/gen_none.mp4
set -euo pipefail
cd "$(dirname "$0")/.."

SPEC="${1:-}"
if [ -z "$SPEC" ]; then
  echo "usage: bash a2v/run_overfit.sh <base_spec>"
  echo "  base_spec: wan2.1-vace-1.3b | wan2.1-t2v-1.3b | wan2.1-i2v-14b-480p | wan2.2-ti2v-5b | wan2.2-i2v-a14b | wan2.2-vace-fun-a14b"
  echo "  (wan2.2-i2v-a14b / wan2.2-vace-fun-a14b require EXPERT=high|low)"
  exit 1
fi

# ---- per-base presets (DATASET_DEFAULT/HEIGHT_DEFAULT shared unless overridden below) ----
DATASET_DEFAULT=.cache/a2v_robotwin/ep0_dataset_phys
HEIGHT_DEFAULT=240
WIDTH_DEFAULT=320
case "$SPEC" in
  wan2.1-vace-1.3b)
    OUT_DEFAULT=models/train/a2v_robotwin_ep0_lora_phys
    LR_DEFAULT=1e-4 ; TRAIN_MODE=lora ; FIRST_FRAME=reference ; OPTIMIZER=fp32 ;;
  wan2.1-t2v-1.3b)
    OUT_DEFAULT=models/train/a2v_robotwin_ep0_vace_t2v
    LR_DEFAULT=1e-4 ; TRAIN_MODE=vace ; FIRST_FRAME=reference ; OPTIMIZER=fp32 ;;
  wan2.1-i2v-14b-480p)
    OUT_DEFAULT=models/train/a2v_robotwin_ep0_vace_i2v
    LR_DEFAULT=1e-5 ; TRAIN_MODE=vace ; FIRST_FRAME=image ; OPTIMIZER=adam8bit ;;
  wan2.2-ti2v-5b)
    DATASET_DEFAULT=.cache/a2v_robotwin/ep0_dataset_phys_256x320
    HEIGHT_DEFAULT=256
    OUT_DEFAULT=models/train/a2v_robotwin_ep0_vace_ti2v
    LR_DEFAULT=1e-5 ; TRAIN_MODE=vace ; FIRST_FRAME=image ; OPTIMIZER=adam8bit ;;
  wan2.2-i2v-a14b)
    # SEAM-5 dual-expert MoE: one job per expert, each on its timestep band (official
    # Wan2.2-I2V-A14B.sh recipe; bands mirror spec.train_bands). first frame = i2v_vae
    # (input_image, VAE-concat, no CLIP). Wan2.1 VAE -> 240x320 (/16) ok, reuse ep0 data.
    EXPERT="${EXPERT:-}"
    case "$EXPERT" in
      high) MIN_TS=0     ; MAX_TS=0.358 ;;   # high-noise expert -> timesteps [900,1000]
      low)  MIN_TS=0.358 ; MAX_TS=1     ;;   # low-noise  expert -> timesteps [0,900)
      *) echo "wan2.2-i2v-a14b requires EXPERT=high|low (got '${EXPERT}')"; exit 1 ;;
    esac
    OUT_DEFAULT=models/train/a2v_robotwin_ep0_vace_a14b_${EXPERT}
    LR_DEFAULT=1e-5 ; TRAIN_MODE=vace ; FIRST_FRAME=image ; OPTIMIZER=adam8bit ;;
  wan2.2-vace-fun-a14b)
    # Warm-start: PRETRAINED dual VACE (Fun-A14B). Same dual-expert band split as I2V-A14B,
    # but first frame = vace_reference (in_dim=16 T2V-style DiT, control via VACE branch).
    EXPERT="${EXPERT:-}"
    case "$EXPERT" in
      high) MIN_TS=0     ; MAX_TS=0.358 ;;
      low)  MIN_TS=0.358 ; MAX_TS=1     ;;
      *) echo "wan2.2-vace-fun-a14b requires EXPERT=high|low (got '${EXPERT}')"; exit 1 ;;
    esac
    OUT_DEFAULT=models/train/a2v_robotwin_ep0_vace_funa14b_${EXPERT}
    LR_DEFAULT=1e-5 ; TRAIN_MODE=vace ; FIRST_FRAME=reference ; OPTIMIZER=adam8bit ;;
  *)
    echo "unknown base_spec: $SPEC"; exit 1 ;;
esac

DATASET="${DATASET:-$DATASET_DEFAULT}"
OUT="${OUT:-$OUT_DEFAULT}"
LR="${LR:-$LR_DEFAULT}"
HEIGHT="${HEIGHT:-$HEIGHT_DEFAULT}"
WIDTH="${WIDTH:-$WIDTH_DEFAULT}"
FRAMES="${FRAMES:-121}"
REPEAT="${REPEAT:-100}"
EPOCHS="${EPOCHS:-10}"
NPROC="${NPROC:-1}"     # data-parallel GPUs (DDP). 1 = single GPU (unchanged default).
DRY_RUN="${DRY_RUN:-0}"

# first-frame seam -> which inputs to feed
if [ "$FIRST_FRAME" = "reference" ]; then
  DATA_FILE_KEYS="video,vace_video,vace_reference_image"
  EXTRA_INPUTS="vace_video,vace_reference_image"
else  # image: i2v_concat / ti2v_fused -> input_image (derived from video[0] by the trainer)
  DATA_FILE_KEYS="video,vace_video"
  EXTRA_INPUTS="vace_video,input_image"
fi

# train mode -> LoRA (pretrained vace) vs full-param vace (from-DiT)
MODE_ARGS=()
if [ "$TRAIN_MODE" = "lora" ]; then
  MODE_ARGS+=(--lora_rank 32)
else
  MODE_ARGS+=(--trainable_models vace)
fi

# optimizer -> fp32 AdamW + offload (1.3B) vs 8-bit Adam (14B/5B fit one 80GB GPU)
OPT_ARGS=()
if [ "$OPTIMIZER" = "adam8bit" ]; then
  OPT_ARGS+=(--customized_optimizer bitsandbytes.optim.Adam8bit)
else
  OPT_ARGS+=(--use_gradient_checkpointing_offload)
fi

# SEAM-5 dual-expert (a14b: I2V-A14B from-DiT or VACE-Fun-A14B warm-start): select the
# expert's DiT + restrict the loss timestep band.
EXPERT_ARGS=()
case "$SPEC" in
  wan2.2-i2v-a14b|wan2.2-vace-fun-a14b)
    EXPERT_ARGS+=(--expert "$EXPERT" --min_timestep_boundary "$MIN_TS" --max_timestep_boundary "$MAX_TS") ;;
esac

echo "[run_overfit] spec=$SPEC dataset=$DATASET out=$OUT ${HEIGHT}x${WIDTH} frames=$FRAMES repeat=$REPEAT epochs=$EPOCHS nproc=$NPROC lr=$LR mode=$TRAIN_MODE opt=$OPTIMIZER first_frame=$FIRST_FRAME${EXPERT_ARGS:+ expert=$EXPERT band=[$MIN_TS,$MAX_TS]}"

CMD=(.venv/bin/accelerate launch --num_processes "$NPROC" --mixed_precision bf16 -m a2v.train_a2v \
  --base_spec "$SPEC" \
  --dataset_base_path "$DATASET" \
  --dataset_metadata_path "$DATASET/metadata.jsonl" \
  --data_file_keys "$DATA_FILE_KEYS" \
  --extra_inputs "$EXTRA_INPUTS" \
  --height "$HEIGHT" --width "$WIDTH" --num_frames "$FRAMES" \
  "${MODE_ARGS[@]}" \
  "${EXPERT_ARGS[@]}" \
  --learning_rate "$LR" \
  --dataset_repeat "$REPEAT" \
  --num_epochs "$EPOCHS" \
  --save_steps 100 \
  --output_path "$OUT" \
  "${OPT_ARGS[@]}" \
  --enable_tensorboard_log
)

printf '[run_overfit] command:'
printf ' %q' "${CMD[@]}"
printf '\n'
if [ "$DRY_RUN" = "1" ]; then
  exit 0
fi

"${CMD[@]}"
