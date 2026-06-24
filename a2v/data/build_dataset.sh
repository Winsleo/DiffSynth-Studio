#!/usr/bin/env bash
# Build a multi-task / multi-robot A2V world-model dataset (train + held-out splits).
#
# For each (task, robot) variant it runs robotwin_adapter then prepare.py into its own
# subdir under the split dir, in parallel, then merges the per-variant metadata.jsonl into
# one split-level metadata.jsonl (paths prefixed by the variant subdir so one
# dataset_base_path resolves every row). Episode lengths vary by robot, so --skip_short
# drops the rare episode shorter than num_frames.
#
# Env overrides:
#   TASKS="beat_block_hammer adjust_bottle"   ROBOTS="aloha-agilex franka ur5"
#   SUFFIX=randomized_500   TRAIN_RANGE=0-449   HELDOUT_RANGE=450-499
#   H=480  W=640  FRAMES=49   OUT=.cache/a2v_robotwin   NAME=wm_480   JOBS=12   DRY_RUN=1
set -euo pipefail
cd "$(dirname "$0")/../.."
PY=.venv/bin/python

TASKS="${TASKS:-beat_block_hammer adjust_bottle}"
ROBOTS="${ROBOTS:-aloha-agilex franka ur5}"
SUFFIX="${SUFFIX:-randomized_500}"
TRAIN_RANGE="${TRAIN_RANGE:-0-449}"
HELDOUT_RANGE="${HELDOUT_RANGE:-450-499}"
H="${H:-480}" ; W="${W:-640}" ; FRAMES="${FRAMES:-49}"
OUT="${OUT:-.cache/a2v_robotwin}"
NAME="${NAME:-wm_480}"
JOBS="${JOBS:-12}"
DRY_RUN="${DRY_RUN:-0}"

TRAIN_DIR="$OUT/${NAME}_train"
HELD_DIR="$OUT/${NAME}_heldout"
mkdir -p "$TRAIN_DIR" "$HELD_DIR"

prep_variant() {  # task robot range root
  local task=$1 robot=$2 range=$3 root=$4
  local variant="${task}__${robot}_${SUFFIX}"
  local work="$root/$variant/work"
  "$PY" -m a2v.data.robotwin_adapter --task "$task" --robot_mode "${robot}_${SUFFIX}" \
      --episodes_range "$range" --work_dir "$work"
  "$PY" -m a2v.data.prepare --manifest "$work/manifest.jsonl" --output_dir "$root/$variant" \
      --height "$H" --width "$W" --num_frames "$FRAMES" --resize_mode stretch \
      --gripper_z_offset 0 --radius_mode physical --skip_short
}
export -f prep_variant ; export PY H W FRAMES SUFFIX

merge_split() {  # root  -> writes root/metadata.jsonl from each variant subdir
  "$PY" - "$1" <<'PYEOF'
import json, glob, os, sys
root = sys.argv[1]
parts = sorted(glob.glob(os.path.join(root, "*", "metadata.jsonl")))
out = os.path.join(root, "metadata.jsonl")
n = 0
with open(out, "w", encoding="utf-8") as fo:
    for p in parts:
        variant = os.path.basename(os.path.dirname(p))
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            for k in ("video", "vace_video"):
                r[k] = [f"{variant}/{x}" for x in r[k]]
            if r.get("vace_reference_image"):
                r["vace_reference_image"] = f"{variant}/{r['vace_reference_image']}"
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
print(f"merged {n} rows from {len(parts)} variants -> {out}")
PYEOF
}

echo "[build_dataset] tasks=($TASKS) robots=($ROBOTS) suffix=$SUFFIX ${H}x${W} frames=$FRAMES"
echo "[build_dataset] train=$TRAIN_RANGE -> $TRAIN_DIR ; heldout=$HELDOUT_RANGE -> $HELD_DIR ; jobs=$JOBS"
if [ "$DRY_RUN" = "1" ]; then echo "(dry run)"; exit 0; fi

# fan out all (split, task, robot) jobs with a JOBS-wide throttle
run=0
for split in train heldout; do
  if [ "$split" = "train" ]; then range="$TRAIN_RANGE"; root="$TRAIN_DIR"; else range="$HELDOUT_RANGE"; root="$HELD_DIR"; fi
  for task in $TASKS; do for robot in $ROBOTS; do
    prep_variant "$task" "$robot" "$range" "$root" > "$root/.log_${task}__${robot}.txt" 2>&1 &
    run=$((run + 1))
    if [ "$((run % JOBS))" -eq 0 ]; then wait -n 2>/dev/null || wait; fi
  done; done
done
wait
echo "[build_dataset] all variants rendered; merging metadata ..."
merge_split "$TRAIN_DIR"
merge_split "$HELD_DIR"

echo "[build_dataset] validating ..."
for d in "$TRAIN_DIR" "$HELD_DIR"; do
  "$PY" -m a2v.data.validate --base_spec wan2.2-ti2v-5b \
    --dataset_base_path "$d" --dataset_metadata_path "$d/metadata.jsonl" \
    --height "$H" --width "$W" --num_frames "$FRAMES" --max_items 30
done
echo "[build_dataset] done. train=$TRAIN_DIR heldout=$HELD_DIR"
