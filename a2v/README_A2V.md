# A2V — Action-conditioned video on Wan (VACE)

Turn a robot end-effector trajectory into a **3-channel RGB trajectory map**, feed it as
`vace_video` through the official **VACE** path, freeze the DiT, and train only the VACE
branch. A single declarative `WanBaseSpec` per base model hides the per-model differences, so
the same **prepare → (cache) → train → evaluate** workflow runs on every supported Wan base.

All A2V code lives under `a2v/`; the upstream `diffsynth/` is never modified.

---

## Supported base models

Pass one as `<base_spec>` (defined in `a2v/base_spec.py`):

| `base_spec` | size | VACE branch | first frame | optimizer |
| --- | --- | --- | --- | --- |
| `wan2.1-vace-1.3b` | 1.3B | pretrained → LoRA | `vace_reference` | fp32 AdamW |
| `wan2.1-t2v-1.3b` | 1.3B | from-DiT → full-param | `vace_reference` | fp32 AdamW |
| `wan2.1-i2v-14b-480p` | 14B | from-DiT → full-param | `i2v_concat` (CLIP) | 8-bit Adam |
| `wan2.2-ti2v-5b` | 5B | from-DiT → full-param | `ti2v_fused` | 8-bit Adam |
| `wan2.2-i2v-a14b` | 2×14B MoE | from-DiT → full-param | `i2v_vae` (no CLIP) | 8-bit Adam |
| `wan2.2-vace-fun-a14b` | 2×14B MoE | pretrained dual → full-param | `vace_reference` | 8-bit Adam |

- **Dual-expert MoE** (`*-a14b`): a high-noise and a low-noise expert. Train them as two
  separate jobs (`EXPERT=high`, then `EXPERT=low`); they are combined automatically at inference.
- **Resolution rule**: H and W must be divisible by `vae_spatial_factor × 2` — 16 for Wan2.1
  bases, **32 for Wan2.2** (TI2V / A14B with the Wan2.2 VAE).
- The three derived numbers (never hardcoded): `vace_in_dim = 2·z_dim + spatial²`
  (96 for Wan2.1, **352** for Wan2.2), `mask_pq = spatial` (8 / **16**), and explicit
  `vace_layers` per depth.

---

## Quickstart (single episode)

```bash
cd /vepfs/wangshilong/code/DiffSynth-Studio
PY=.venv/bin/python

# 1. Prepare one RoboTwin episode -> a small A2V dataset
$PY -m a2v.data.robotwin_adapter --task beat_block_hammer \
  --robot_mode aloha-agilex_clean_50 --episodes 0 --work_dir .cache/a2v_robotwin/ep0_work
$PY -m a2v.data.prepare --manifest .cache/a2v_robotwin/ep0_work/manifest.jsonl \
  --output_dir .cache/a2v_robotwin/ep0_dataset_phys \
  --height 240 --width 320 --num_frames 121 \
  --resize_mode stretch --gripper_z_offset 0 --radius_mode physical

# 2. Train (presets come from the spec; see "Training" below)
CUDA_VISIBLE_DEVICES=0 bash a2v/train.sh wan2.1-vace-1.3b

# 3. Causal gate: real control should reconstruct GT, the negative control should not
for c in real none; do $PY -m a2v.infer_a2v --base_spec wan2.1-vace-1.3b \
  --dataset .cache/a2v_robotwin/ep0_dataset_phys \
  --lora models/train/a2v_robotwin_ep0_lora_phys/step-1000.safetensors \
  --num_frames 121 --height 240 --width 320 --control $c \
  --output .cache/a2v_robotwin/gen_$c.mp4; done
$PY -m a2v.causal_metrics --dataset .cache/a2v_robotwin/ep0_dataset_phys \
  --real .cache/a2v_robotwin/gen_real.mp4 --none .cache/a2v_robotwin/gen_none.mp4
```

---

## 1. Prepare data

`robotwin_adapter` reads RoboTwin hdf5 → `actions/intrinsic/extrinsic.npy` + a manifest;
`prepare` renders the trajectory maps and stores `video` and `vace_video` as same-indexed,
lossless PNG lists (the core A2V alignment invariant: both routes see identical frames).

```bash
$PY -m a2v.data.prepare --manifest <manifest.jsonl> --output_dir <dataset_dir> \
  --height H --width W --num_frames N \         # N must be 4n+1; H,W divisible by 16 (Wan2.1) / 32 (Wan2.2)
  --resize_mode stretch --gripper_z_offset 0 --radius_mode physical \
  [--write_overlay] [--skip_short]              # overlay = debug; skip_short = drop too-short episodes
```

Validate before training (no model weights needed):

```bash
$PY -m a2v.data.validate --base_spec <spec> \
  --dataset_base_path <dataset_dir> --dataset_metadata_path <dataset_dir>/metadata.jsonl \
  --height H --width W --num_frames N
```

For multi-task / multi-robot datasets use the batch builder, which fans out adapter+prepare
per variant and merges them into one train + one held-out split:

```bash
TASKS="beat_block_hammer adjust_bottle" ROBOTS="aloha-agilex franka ur5" \
  H=480 W=640 FRAMES=49 bash a2v/data/build_dataset.sh
# -> .cache/a2v_robotwin/wm_480_{train,heldout}
```

---

## 2. Train — `a2v/train.sh`

**`bash a2v/train.sh <base_spec>` is the single training entry** for every base and every
scale. It wraps `python -m a2v.train_a2v` and fills presets (dataset / resolution / lr /
optimizer / first-frame / LoRA-vs-full-param) from the spec. Override any preset with an env var.

```bash
# single-episode overfit (smallest sanity run)
CUDA_VISIBLE_DEVICES=0 bash a2v/train.sh wan2.2-ti2v-5b

# multi-episode, multi-GPU (raw dataset; encoders run each step)
NPROC=8 DATASET=.cache/a2v_robotwin/ep_train40_phys \
  FRAMES=105 REPEAT=4 EPOCHS=40 OUT=models/train/my_run \
  bash a2v/train.sh wan2.1-i2v-14b-480p

# dual-expert A14B: run twice (high then low), combined at inference
EXPERT=high CUDA_VISIBLE_DEVICES=0 bash a2v/train.sh wan2.2-i2v-a14b
EXPERT=low  CUDA_VISIBLE_DEVICES=1 bash a2v/train.sh wan2.2-i2v-a14b

DRY_RUN=1 bash a2v/train.sh wan2.2-ti2v-5b   # print the resolved command, run nothing
```

### Env overrides

| var | meaning |
| --- | --- |
| `DATASET` `OUT` `LR` `HEIGHT` `WIDTH` `FRAMES` `REPEAT` `EPOCHS` `SAVE_STEPS` | override per-base presets |
| `NPROC` | data-parallel GPUs (DDP); default `1` |
| `CACHE_TRAIN=1` `CACHE_DIR=` | train from a pre-encoded cache (see below) |
| `RESUME=<ckpt>` | resume a long run |
| `EXPERT=high\|low` | **required** for `*-a14b` (selects the MoE expert + its timestep band) |
| `DRY_RUN=1` | print the command without launching |

Checkpoints: `models/train/.../step-*.safetensors` (LoRA for `vace-1.3b`, full VACE state
dict otherwise).

### Cache-train (recommended for multi-epoch / large runs)

Pre-encode the dataset once (VAE/T5/CLIP outputs cached), then train loading only the DiT(+VAE)
— faster and lower VRAM. Use a **fresh cache dir whenever resolution or data changes** (cache is
keyed by row index, not content).

```bash
D=.cache/a2v_robotwin/wm_480_train
# encode once (shard across GPUs)
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 .venv/bin/accelerate launch --num_processes 8 \
  --mixed_precision bf16 -m a2v.train_a2v --base_spec wan2.2-ti2v-5b --task sft:data_process \
  --dataset_base_path "$D" --dataset_metadata_path "$D/metadata.jsonl" \
  --data_file_keys video,vace_video --extra_inputs vace_video,input_image \
  --height 480 --width 640 --num_frames 49 --dataset_repeat 1 \
  --output_path .cache/a2v_robotwin/cache_wm_ti2v_480
# train from the cache
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 SPEC=wan2.2-ti2v-5b CACHE_TRAIN=1 \
  CACHE_DIR=.cache/a2v_robotwin/cache_wm_ti2v_480 NPROC=8 HEIGHT=480 WIDTH=640 FRAMES=49 \
  LR=1e-5 REPEAT=1 EPOCHS=30 SAVE_STEPS=500 OUT=models/train/a2v_wm_ti2v_480 \
  bash a2v/train.sh wan2.2-ti2v-5b
```

> Long runs should be detached (`tmux` / `nohup`) so a disconnect does not kill training;
> resume with `RESUME=<ckpt>`.

---

## 3. Evaluate (causal gate)

The model is judged by causal control, not pixel loss alone: generate under the **real**
trajectory and a **negative control** (`none` = blank map, `shuffle` = reversed), and require
the real output to be clearly closer to GT.

**Single episode** — `causal_metrics`:

```bash
for c in real none shuffle; do $PY -m a2v.infer_a2v --base_spec <spec> \
  --dataset <dataset_dir> --lora <ckpt.safetensors> \
  --num_frames N --height H --width W --control $c --output .cache/a2v_robotwin/gen_$c.mp4; done
$PY -m a2v.causal_metrics --dataset <dataset_dir> \
  --real .cache/a2v_robotwin/gen_real.mp4 --none .cache/a2v_robotwin/gen_none.mp4
# dual-expert: add --lora_low <low_ckpt> to infer_a2v
```

**Multi-episode generalization** — `eval_multiep` (loads the model once, writes per-episode
table + `metrics.json` + a `SUMMARY` with verdict):

```bash
$PY -m a2v.eval_multiep --base_spec <spec> --lora <ckpt> [--lora_low <low_ckpt>] \
  --dataset .cache/a2v_robotwin/wm_480_heldout --rows 0,1,50,51 \
  --num_frames N --height H --width W --controls real,none \
  --output_dir .cache/a2v_robotwin/eval_heldout
```

**Pass criteria**: mean REAL MAE-GT well below NONE, `real<none` on (almost) every episode,
REAL motion in the same order as GT. For strong-first-frame bases (i2v / ti2v) the negative
control need not be static — it freely animates from frame 0 — so the signal is the clear
`REAL ≪ NONE` gap, not "NONE is frozen".

---

## 4. Pre-flight checks

```bash
$PY -m a2v.check_load        --base_spec <spec>                 # weights load + VACE branch builds
$PY -m a2v.check_provision   --base_spec <spec> --dataset <ds>  # + zero-side-effect gate (from-DiT bases)
$PY -m a2v.check_parity                                         # VACE-1.3B abstraction parity
$PY -m a2v.render.check_projection                              # intrinsic-scaling regression
$PY -m a2v.data.smoke                                           # synthetic end-to-end data smoke
```

---

## Layout

```
a2v/
  base_spec.py        WanBaseSpec + REGISTRY (one entry per base model)
  provision.py        VACE supply: create_vace_from_dit / ensure_vace / provision_a2v
  vace_unit.py        mask_pq-parameterized VACE unit (no diffsynth edit)
  train_a2v.py        trainer module wrapped by train.sh (frame-list operator, from-DiT vace, cache-train)
  train.sh            >> training entry: bash a2v/train.sh <base_spec> [env overrides]
  infer_a2v.py        inference + causal-control harness (real / none / shuffle)
  eval_multiep.py     multi-episode generalization evaluator (model loaded once)
  causal_metrics.py   MAE-GT / motion / real-vs-none gate metrics
  check_load.py       check_provision.py  check_parity.py   pre-flight gates
  data/               robotwin_adapter, prepare, build_dataset.sh, operators, validate, smoke
  render/             traj_map, action_io, check_projection
  README_A2V.md       this guide
  A2V_HANDOFF.md      current status + handoff checklist + quick reference
  A2V_HISTORY.md      per-track debugging history, decisions, and lessons (appendix)
```

See `A2V_HANDOFF.md` for current status and the latest results; `A2V_HISTORY.md` for the full
per-track history and lessons.
