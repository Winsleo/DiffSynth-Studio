# A2V — Action-conditioned video on Wan (VACE)

Turn a robot end-effector trajectory into a **3-channel RGB trajectory map**, feed it as
`vace_video` through the official **VACE** path, freeze the DiT and train only the VACE
branch. One declarative `WanBaseSpec` per base model collapses the model differences, so
the same prepare → train → infer → gate workflow runs across every Wan base.

All A2V code lives under `a2v/`; the upstream `diffsynth/` is never modified.

## Supported base models

| `--base_spec` | Track | VACE branch | first frame | optimizer | dataset / size |
| --- | --- | --- | --- | --- | --- |
| `wan2.1-vace-1.3b` | T1 | pretrained → LoRA | `vace_reference` | fp32 AdamW | `ep0_dataset_phys` 240×320 |
| `wan2.1-t2v-1.3b` | T3 | from-DiT → full-param | `none`/reference | fp32 AdamW | `ep0_dataset_phys` 240×320 |
| `wan2.1-i2v-14b-480p` | T4 | from-DiT → full-param | `i2v_concat` | 8-bit Adam | `ep0_dataset_phys` 240×320 |
| `wan2.2-ti2v-5b` | T5 | from-DiT → full-param | `ti2v_fused` | 8-bit Adam | `ep0_dataset_phys_256x320` 256×320 |

The three numbers that bite (derived from the spec, never hardcoded): `vace_in_dim =
2·z_dim + spatial²` (96 for Wan2.1, **352** for Wan2.2), `mask_pq = spatial` (8 / **16**),
and `vace_layers` given explicitly per depth. TI2V's Wan2.2 VAE is 16× spatial × patch 2 =
32×, so its H,W must be divisible by 32 (hence 256×320, not 240×320).

## 1. Prepare the dataset (once per episode)

```bash
PY=.venv/bin/python
# RoboTwin hdf5 -> actions/intrinsic/extrinsic.npy + manifest
$PY -m a2v.data.robotwin_adapter \
  --task beat_block_hammer --robot_mode aloha-agilex_clean_50 --episodes 0 \
  --work_dir .cache/a2v_robotwin/ep0_work
# render trajectory maps + lossless PNG frame lists + metadata.jsonl
$PY -m a2v.data.prepare \
  --manifest .cache/a2v_robotwin/ep0_work/manifest.jsonl \
  --output_dir .cache/a2v_robotwin/ep0_dataset_phys \
  --height 240 --width 320 --num_frames 121 \
  --resize_mode stretch --gripper_z_offset 0 --radius_mode physical --write_overlay
```

`video` and `vace_video` are stored as same-indexed PNG lists so both routes see identical
frames (the core A2V alignment invariant). For TI2V use `--height 256` and output dir
`ep0_dataset_phys_256x320` (H,W must be /32).

Validate / synthetic smoke (no weights needed):

```bash
$PY -m a2v.data.validate --base_spec wan2.2-ti2v-5b \
  --dataset_base_path .cache/a2v_robotwin/ep0_dataset_phys_256x320 \
  --dataset_metadata_path .cache/a2v_robotwin/ep0_dataset_phys_256x320/metadata.jsonl \
  --height 256 --width 320 --num_frames 121
$PY -m a2v.data.smoke
```

## 2. Train (single-sample overfit gate)

One script, any base — presets (dataset, resolution, lr, optimizer, first-frame, LoRA vs
full-param) are chosen from the spec:

```bash
bash a2v/run_overfit.sh wan2.2-ti2v-5b
# override any preset via env: LR=1e-5 OUT=... HEIGHT=256 bash a2v/run_overfit.sh <spec>
# multi-episode knobs keep old defaults unless set:
# FRAMES=105 REPEAT=1 EPOCHS=120 OUT=... DATASET=... bash a2v/run_overfit.sh wan2.1-vace-1.3b
# DRY_RUN=1 prints the resolved command without launching training.
```

Checkpoints land in `models/train/.../step-*.safetensors` (LoRA for vace-1.3b, full vace
state dict otherwise).

## 3. Infer + causal gate

```bash
# generate under real control and a negative control (none = blank, shuffle = reversed)
for c in real none; do $PY -m a2v.infer_a2v --base_spec wan2.2-ti2v-5b \
  --dataset .cache/a2v_robotwin/ep0_dataset_phys_256x320 \
  --lora models/train/a2v_robotwin_ep0_vace_ti2v/step-1000.safetensors \
  --height 256 --width 320 --num_frames 121 --control $c \
  --output .cache/a2v_robotwin/gen_$c.mp4; done

# metrics auto-derive H,W from the GT frames — no --height needed
$PY -m a2v.causal_metrics --dataset .cache/a2v_robotwin/ep0_dataset_phys_256x320 \
  --real .cache/a2v_robotwin/gen_real.mp4 --none .cache/a2v_robotwin/gen_none.mp4
```

**Gate passes** when REAL reconstructs the GT trajectory (low MAE-GT, motion ≈ GT) and is
clearly closer to GT than the negative control. For i2v/ti2v bases the negative control is
not expected to be static (they freely animate from frame 0); the signal is `REAL ≪ NONE`
on MAE-GT plus a non-trivial real-vs-none difference.

## 4. Multi-episode train + eval

For the first formal generalization run, train VACE-1.3B LoRA on RoboTwin ep0-39 and evaluate ep40-49:

```bash
PY=.venv/bin/python
FRAMES=105 REPEAT=1 EPOCHS=120 \
OUT=models/train/a2v_robotwin_train40_vace1p3b_lora \
DATASET=.cache/a2v_robotwin/ep_train40_phys \
CUDA_VISIBLE_DEVICES=0 bash a2v/run_overfit.sh wan2.1-vace-1.3b

CK=models/train/a2v_robotwin_train40_vace1p3b_lora/step-4800.safetensors
$PY -m a2v.eval_multiep --base_spec wan2.1-vace-1.3b --lora $CK \
  --dataset .cache/a2v_robotwin/ep_heldout10_phys --num_frames 105 --height 240 --width 320 \
  --controls real,none --output_dir .cache/a2v_robotwin/eval_heldout
$PY -m a2v.eval_multiep --base_spec wan2.1-vace-1.3b --lora $CK \
  --dataset .cache/a2v_robotwin/ep_train40_phys --rows 0,1,2,3,4 \
  --num_frames 105 --height 240 --width 320 --controls real,none \
  --output_dir .cache/a2v_robotwin/eval_train
```

`eval_multiep.py` loads the model once, generates all requested controls, saves mp4s, writes `metrics.json`, and prints a per-episode table plus one `SUMMARY`. The gate is cross-base: mean REAL MAE-GT below NONE, real<none on most episodes, real motion in the same order as GT, and non-trivial real-vs-none difference.

2026-06-18 VACE-1.3B train40 result: held-out ep40-49 PASS (`mean_real_mae=6.25`, `mean_none_mae=26.83`, `real_lt_none=10/10`, `mean_motion_ratio=1.03`); train subset ep0-4 PASS (`6.35` vs `29.89`, `5/5`, ratio `1.02`).

## 5. Pre-flight / provision checks

```bash
$PY -m a2v.check_load          --base_spec wan2.2-ti2v-5b   # loads + builds VACE branch
$PY -m a2v.check_provision  --base_spec wan2.2-ti2v-5b \
  --dataset .cache/a2v_robotwin/ep0_dataset_phys_256x320 --height 256 --width 320
$PY -m a2v.check_parity                                  # VACE-1.3B abstraction parity
$PY -m a2v.render.check_projection                          # intrinsic-scaling regression
```

`check_load` confirms the DiT/VAE/CLIP load and the from-DiT VACE branch builds (shapes,
zero-init, mask_pq). `check_provision` adds the zero-side-effect gate (a real `pipe()`
with vs without control is bit-identical at init).

## Layout

```
a2v/
  base_spec.py        WanBaseSpec + REGISTRY (one entry per base model)
  provision.py        SEAM-1: create_vace_from_dit / ensure_vace / provision_a2v
  vace_unit.py        SEAM-2: mask_pq-parameterized VACE unit (no diffsynth edit)
  train_a2v.py        thin trainer wrapper (frame-list operator + from-DiT vace build)
  infer_a2v.py        inference + causal-control harness
  eval_multiep.py     multi-row generalization evaluator, model loaded once
  causal_metrics.py   MAE-GT / motion / real-vs-none gate metrics
  run_overfit.sh      unified single-sample overfit (takes <base_spec>)
  check_load.py       load smoke (any base)
  check_provision.py / check_parity.py   provision / abstraction gates
  data/               robotwin_adapter, prepare, operators, validate, smoke
  render/             traj_map, action_io, check_projection
  A2V_HANDOFF.md      project handoff: current status + checklist + quick reference
  A2V_HISTORY.md      per-track debugging history, fixes, and lessons (appendix)
```

See `A2V_HANDOFF.md` for current status and the handoff checklist; `A2V_HISTORY.md` for the
full per-track history, decisions, and lessons.
