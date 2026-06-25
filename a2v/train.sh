#!/usr/bin/env bash
# A2V training entry. One driver for every registered Wan base model and every scale:
# single-episode overfit, multi-episode, and pre-encoded cache-train. It wraps
# `python -m a2v.train_a2v` (via `accelerate launch`) and fills per-base presets
# (dataset / resolution / lr / optimizer / first-frame / LoRA-vs-full-param) from the spec.
#
#   Usage:  bash a2v/train.sh <base_spec>   [env overrides...]
#
#   <base_spec> (a2v/base_spec.py REGISTRY):
#     wan2.1-vace-1.3b      pretrained VACE  -> LoRA            (vace_reference first frame)
#     wan2.1-t2v-1.3b       from-DiT VACE    -> full-param vace (vace_reference first frame)
#     wan2.1-i2v-14b-480p   from-DiT VACE    -> full-param vace (i2v_concat first frame)
#     wan2.2-ti2v-5b        from-DiT VACE    -> full-param vace (ti2v_fused first frame)
#     wan2.2-i2v-a14b       from-DiT VACE    -> full-param vace (i2v_vae first frame); dual-
#                           expert MoE: run twice, EXPERT=high then EXPERT=low.
#     wan2.2-vace-fun-a14b  pretrained dual VACE (warm-start), vace_reference first frame;
#                           dual-expert MoE, same EXPERT=high|low two-run pattern.
#
# Modes:
#   default       train from the raw PNG dataset (DATASET=); encoders run each step.
#   CACHE_TRAIN=1 train from a pre-encoded cache (CACHE_DIR=); loads only DiT(+VAE), skips
#                 T5/CLIP -> faster + less VRAM. Build the cache once with:
#                   accelerate launch -m a2v.train_a2v --base_spec <spec> --task sft:data_process \
#                     --dataset_base_path <DATA> --dataset_metadata_path <DATA>/metadata.jsonl ...
#
# Env overrides (defaults are per-base presets unless set):
#   SPEC=  DATASET=  OUT=  LR=  HEIGHT=  WIDTH=  FRAMES=  REPEAT=  EPOCHS=  SAVE_STEPS=
#   NPROC=        data-parallel GPUs (DDP); default 1
#   CACHE_TRAIN=1 CACHE_DIR=   train from an encoded cache (see Modes)
#   RESUME=<ckpt>             resume a long run
#   EXPERT=high|low           REQUIRED for the dual-expert A14B specs (selects expert+band)
#   LR_SCHEDULE=constant|cosine   default constant; cosine = warmup -> cosine decay (LR = peak)
#   WARMUP=<steps>            cosine warmup steps (0 -> 3% of total = num_epochs*len(dataset))
#   LR_MIN_RATIO=<frac>       cosine final lr as a fraction of peak (default 0)
#   DRY_RUN=1                 print the resolved command without launching
#
# Why presets differ: from-DiT VACE must train full-param (its zero-init control entry/exit
# are not LoRA targets); 14B/5B use 8-bit Adam to fit one 80GB GPU (lr 1e-5; 1e-4 diverged on
# 14B); TI2V's Wan2.2 VAE is 32x (16x spatial x patch 2) so its H,W must be divisible by 32.
#
# After training, run the causal gate (see README_A2V.md S3): infer real/none/shuffle, then
# `python -m a2v.causal_metrics` (single ep) or `python -m a2v.eval_multiep` (multi ep).
set -euo pipefail
cd "$(dirname "$0")/.."

SPEC="${1:-}"
if [ -z "$SPEC" ]; then
  echo "usage: bash a2v/train.sh <base_spec>"
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
SAVE_STEPS="${SAVE_STEPS:-100}"
CACHE_TRAIN="${CACHE_TRAIN:-0}"   # 1 = train from a pre-encoded cache (CACHE_DIR); loads only the DiT
CACHE_DIR="${CACHE_DIR:-}"        # encoded-cache dir from `--task sft:data_process` (required if CACHE_TRAIN=1)
RESUME="${RESUME:-}"              # checkpoint dir/file to resume from (long production runs)
LR_SCHEDULE="${LR_SCHEDULE:-constant}"  # constant (default) | cosine (warmup -> cosine decay; LR = peak)
WARMUP="${WARMUP:-0}"                    # cosine warmup steps; 0 -> 3% of total (num_epochs*len(dataset))
LR_MIN_RATIO="${LR_MIN_RATIO:-0.0}"     # cosine final lr as a fraction of peak
DRY_RUN="${DRY_RUN:-0}"

# first-frame seam -> which inputs to feed (only used when encoding; cache mode skips encoders)
if [ "$FIRST_FRAME" = "reference" ]; then
  DATA_FILE_KEYS="video,vace_video,vace_reference_image"
  EXTRA_INPUTS="vace_video,vace_reference_image"
else  # image: i2v_concat / ti2v_fused -> input_image (derived from video[0] by the trainer)
  DATA_FILE_KEYS="video,vace_video"
  EXTRA_INPUTS="vace_video,input_image"
fi

# dataset source: encoded cache (only DiT loaded; encoders + their inputs skipped) vs raw PNG dataset
DATA_ARGS=()
if [ "$CACHE_TRAIN" = "1" ]; then
  [ -n "$CACHE_DIR" ] || { echo "CACHE_TRAIN=1 requires CACHE_DIR=<encoded cache dir>"; exit 1; }
  # --task sft:train prunes the encoder units (T5 / VAE-encode / VACE-encode); their outputs
  # are in the cache. The kept NoiseInitializer still reads the VAE config, so train_a2v
  # keeps the (small) VAE loaded alongside the DiT.
  DATA_ARGS+=(--task sft:train --cache_train --dataset_base_path "$CACHE_DIR" --data_file_keys "video,vace_video")
else
  DATA_ARGS+=(--dataset_base_path "$DATASET" --dataset_metadata_path "$DATASET/metadata.jsonl"
              --data_file_keys "$DATA_FILE_KEYS" --extra_inputs "$EXTRA_INPUTS")
fi

RESUME_ARGS=()
[ -n "$RESUME" ] && RESUME_ARGS+=(--resume_from_checkpoint "$RESUME")

# lr schedule (default constant = unchanged stock behavior; cosine = warmup + cosine decay)
SCHED_ARGS=(--lr_schedule "$LR_SCHEDULE" --lr_warmup_steps "$WARMUP" --lr_min_ratio "$LR_MIN_RATIO")

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

SRC_DESC=$([ "$CACHE_TRAIN" = "1" ] && echo "cache=$CACHE_DIR" || echo "dataset=$DATASET")
echo "[train] spec=$SPEC $SRC_DESC out=$OUT ${HEIGHT}x${WIDTH} frames=$FRAMES repeat=$REPEAT epochs=$EPOCHS nproc=$NPROC lr=$LR sched=$LR_SCHEDULE${LR_SCHEDULE:+$([ "$LR_SCHEDULE" = cosine ] && echo " warmup=$WARMUP min_ratio=$LR_MIN_RATIO")} mode=$TRAIN_MODE opt=$OPTIMIZER first_frame=$FIRST_FRAME${EXPERT_ARGS:+ expert=$EXPERT band=[$MIN_TS,$MAX_TS]}${RESUME:+ resume=$RESUME}"

CMD=(.venv/bin/accelerate launch --num_processes "$NPROC" --mixed_precision bf16 -m a2v.train_a2v \
  --base_spec "$SPEC" \
  "${DATA_ARGS[@]}" \
  --height "$HEIGHT" --width "$WIDTH" --num_frames "$FRAMES" \
  "${MODE_ARGS[@]}" \
  "${EXPERT_ARGS[@]}" \
  "${RESUME_ARGS[@]}" \
  "${SCHED_ARGS[@]}" \
  --learning_rate "$LR" \
  --dataset_repeat "$REPEAT" \
  --num_epochs "$EPOCHS" \
  --save_steps "$SAVE_STEPS" \
  --output_path "$OUT" \
  "${OPT_ARGS[@]}" \
  --enable_tensorboard_log
)

printf '[train] command:'
printf ' %q' "${CMD[@]}"
printf '\n'
if [ "$DRY_RUN" = "1" ]; then
  exit 0
fi

"${CMD[@]}"
