#!/usr/bin/env bash
# Build ONE A2V dataset (no train/heldout split) from every RoboTwin variant of a
# given SUFFIX (default clean_50), discovered by globbing the data root so missing
# robot variants (e.g. piper for some tasks) are simply skipped - no cartesian product.
#
# For each <root>/<task>/<robot>_<suffix> it runs robotwin_adapter (ALL episodes) then
# prepare into its own subdir under OUT, in parallel (JOBS throttle), then merges every
# per-variant metadata.jsonl into one OUT/metadata.jsonl (paths prefixed by the variant
# subdir so one dataset_base_path resolves every row), then validates.
#
# Pure CPU (CUDA_VISIBLE_DEVICES="" forced) - does not touch GPUs.
#
# Env overrides (all have defaults):
#   PY=/disk/worldmodel/uv/envs/a2v/bin/python
#   ROOT=/disk/worldmodel/public_data/RoboTwin2.0_unpacked
#   SUFFIX=clean_50            # reuse for randomized_500 later
#   H=480 W=640 FRAMES=121
#   MAX_FRAMES=                # set (e.g. 121) -> variable-length: each episode rendered at the
#                             #   largest 4n+1 <= min(its length, MAX_FRAMES); MIN_FRAMES drops shorter.
#   MIN_FRAMES=49             # cap-mode floor (only used when MAX_FRAMES is set)
#   OUT=.cache/a2v_robotwin/clean50_480x640_f121
#   JOBS=48  DRY_RUN=0
set -uo pipefail
cd "$(dirname "$0")/../.."

PY="${PY:-/disk/worldmodel/uv/envs/a2v/bin/python}"
ROOT="${ROOT:-/disk/worldmodel/public_data/RoboTwin2.0_unpacked}"
SUFFIX="${SUFFIX:-clean_50}"
H="${H:-480}" ; W="${W:-640}" ; FRAMES="${FRAMES:-121}"
MAX_FRAMES="${MAX_FRAMES:-}"
MIN_FRAMES="${MIN_FRAMES:-49}"
OUT="${OUT:-.cache/a2v_robotwin/clean50_480x640_f121}"
JOBS="${JOBS:-48}"
DRY_RUN="${DRY_RUN:-0}"

# Cap mode (variable-length) vs fixed --num_frames.
if [ -n "$MAX_FRAMES" ]; then
  PREP_FRAME_ARGS=(--max_num_frames "$MAX_FRAMES" --min_num_frames "$MIN_FRAMES")
  VAL_FRAME_ARGS=(--num_frames "$MIN_FRAMES" --max_num_frames "$MAX_FRAMES")
else
  PREP_FRAME_ARGS=(--num_frames "$FRAMES")
  VAL_FRAME_ARGS=(--num_frames "$FRAMES")
fi

export CUDA_VISIBLE_DEVICES=""   # rendering is CPU-only; never touch the busy GPUs
mkdir -p "$OUT"

# --- discover variants present on disk -------------------------------------------------
mapfile -t VARIANTS < <(
  for d in "$ROOT"/*/*_"$SUFFIX"; do
    [ -d "$d/data" ] || continue
    [ -d "$d/video" ] || continue
    compgen -G "$d/data/episode*.hdf5" > /dev/null || continue
    task=$(basename "$(dirname "$d")")
    robot_mode=$(basename "$d")
    echo "${task}__${robot_mode}"
  done | sort
)

if [ -n "$MAX_FRAMES" ]; then
  echo "[build_all] suffix=$SUFFIX  ${H}x${W} VARIABLE-LENGTH cap=$MAX_FRAMES floor=$MIN_FRAMES  jobs=$JOBS"
else
  echo "[build_all] suffix=$SUFFIX  ${H}x${W} frames=$FRAMES (fixed)  jobs=$JOBS"
fi
echo "[build_all] root=$ROOT"
echo "[build_all] out=$OUT"
echo "[build_all] discovered ${#VARIANTS[@]} variants present on disk"
if [ "${#VARIANTS[@]}" -eq 0 ]; then echo "[build_all] nothing to do"; exit 1; fi

prep_variant() {  # variant = task__robot_mode
  local variant=$1
  local task="${variant%%__*}"
  local robot_mode="${variant#*__}"
  local vdir="$OUT/$variant"
  local log="$vdir/.log.txt"
  mkdir -p "$vdir"
  {
    echo "=== $(date '+%F %T') START $variant ==="
    "$PY" -m a2v.data.robotwin_adapter --root "$ROOT" --task "$task" \
        --robot_mode "$robot_mode" --work_dir "$vdir/work" \
      && "$PY" -m a2v.data.prepare --manifest "$vdir/work/manifest.jsonl" \
        --output_dir "$vdir" --height "$H" --width "$W" "${PREP_FRAME_ARGS[@]}" \
        --resize_mode stretch --gripper_z_offset 0 --radius_mode physical --skip_short \
      && echo "=== $(date '+%F %T') DONE  $variant ===" \
      || echo "=== $(date '+%F %T') FAIL  $variant (see above) ==="
  } > "$log" 2>&1
}

if [ "$DRY_RUN" = "1" ]; then
  echo "(dry run) would process these variants:"
  printf '  %s\n' "${VARIANTS[@]}"
  v="${VARIANTS[0]}"; t="${v%%__*}"; r="${v#*__}"
  echo
  echo "example per-variant commands:"
  echo "  $PY -m a2v.data.robotwin_adapter --root $ROOT --task $t --robot_mode $r --work_dir $OUT/$v/work"
  echo "  $PY -m a2v.data.prepare --manifest $OUT/$v/work/manifest.jsonl --output_dir $OUT/$v \\"
  echo "      --height $H --width $W ${PREP_FRAME_ARGS[*]} --resize_mode stretch --gripper_z_offset 0 --radius_mode physical --skip_short"
  exit 0
fi

# --- fan out with a JOBS-wide throttle --------------------------------------------------
run=0
for v in "${VARIANTS[@]}"; do
  prep_variant "$v" &
  run=$((run + 1))
  if [ "$((run % JOBS))" -eq 0 ]; then wait -n 2>/dev/null || wait; fi
done
wait
echo "[build_all] all variants processed; merging metadata ..."

# --- merge per-variant metadata into one split-level metadata.jsonl ---------------------
"$PY" - "$OUT" <<'PYEOF'
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
                if k in r and isinstance(r[k], list):
                    r[k] = [f"{variant}/{x}" for x in r[k]]
            if r.get("vace_reference_image"):
                r["vace_reference_image"] = f"{variant}/{r['vace_reference_image']}"
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
print(f"merged {n} rows from {len(parts)} variants -> {out}")
PYEOF

echo "[build_all] validating ..."
"$PY" -m a2v.data.validate --base_spec wan2.2-ti2v-5b \
  --dataset_base_path "$OUT" --dataset_metadata_path "$OUT/metadata.jsonl" \
  --height "$H" --width "$W" "${VAL_FRAME_ARGS[@]}" --max_items 30
echo "[build_all] done. dataset=$OUT"
