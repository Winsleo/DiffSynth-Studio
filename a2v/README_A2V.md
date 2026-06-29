# A2V — Action-conditioned video on Wan (VACE)

Turn a robot end-effector trajectory into a **3-channel RGB trajectory map**, feed it as
`vace_video` through the official **VACE** path, freeze the DiT, and train only the VACE branch.
A single declarative `WanBaseSpec` per base model hides the per-model differences, so the **same
workflow runs on every supported Wan base**:

```
Step 1 prepare data  →  Step 2 sanity-check  →  Step 3 train  →  Step 4 evaluate (causal gate)
```

All A2V code lives under `a2v/`; the upstream `diffsynth/` is **never modified**.

> New here? Read **Setup** → **Choose a base model** → run **Quickstart** once to confirm your
> environment, then follow **Step 1–4** to scale up to your own data. Dual-expert A14B and
> bring-your-own-data are in **Advanced** / **Reference** at the bottom.

---

## Setup

```bash
cd /vepfs/wangshilong/code/DiffSynth-Studio
PY=.venv/bin/python            # the venv already has torch/h5py/tensorboard/etc.
```

- **GPU**: A100-80GB (8× available). Single GPU is enough for everything except large multi-epoch runs.
- **Datasets** are written under `.cache/a2v_robotwin/`; **checkpoints** under `models/train/`.
- Long runs should be detached (`tmux` / `nohup`) so a disconnect does not kill training.

---

## Choose a base model

Pass one as `<base_spec>` (defined in `a2v/base_spec.py`):

| `base_spec` | size | VACE branch | first frame | optimizer |
| --- | --- | --- | --- | --- |
| `wan2.1-vace-1.3b` | 1.3B | pretrained → LoRA | `vace_reference` | fp32 AdamW |
| `wan2.1-t2v-1.3b` | 1.3B | from-DiT → full-param | `vace_reference` | fp32 AdamW |
| `wan2.1-i2v-14b-480p` | 14B | from-DiT → full-param | `i2v_concat` (CLIP) | 8-bit Adam |
| `wan2.2-ti2v-5b` | 5B | from-DiT → full-param | `ti2v_fused` | 8-bit Adam |
| `wan2.2-i2v-a14b` | 2×14B MoE | from-DiT → full-param | `i2v_vae` (no CLIP) | 8-bit Adam |
| `wan2.2-vace-fun-a14b` | 2×14B MoE | pretrained dual → full-param | `vace_reference` | 8-bit Adam |

- **Start with `wan2.1-vace-1.3b`** — smallest, fastest sanity loop. `wan2.2-ti2v-5b` is the
  current production world-model base.
- **Resolution rule**: H and W must be divisible by `vae_spatial_factor × 2` — **16** for Wan2.1
  bases, **32** for Wan2.2 (TI2V / A14B). `--num_frames` must be `4n+1`.
- **Dual-expert MoE** (`*-a14b`): trained as two separate jobs and combined at inference — see
  **Advanced** below. The internal derived numbers (`vace_in_dim` / `mask_pq` / `vace_layers`)
  are in **Reference**; you never set them by hand.

---

## Quickstart — the whole pipeline in one block

Copy/paste this to confirm your setup end-to-end on a single episode (smallest base). Each command
is explained in **Step 1–4**.

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

# 2. Train (all presets come from the spec)
CUDA_VISIBLE_DEVICES=0 bash a2v/train.sh wan2.1-vace-1.3b

# 3. Evaluate — causal gate: real control reconstructs GT, the negative control should not
for c in real none; do $PY -m a2v.infer_a2v --base_spec wan2.1-vace-1.3b \
  --dataset .cache/a2v_robotwin/ep0_dataset_phys \
  --lora models/train/a2v_robotwin_ep0_lora_phys/step-1000.safetensors \
  --num_frames 121 --height 240 --width 320 --control $c \
  --output .cache/a2v_robotwin/gen_$c.mp4; done
$PY -m a2v.causal_metrics --dataset .cache/a2v_robotwin/ep0_dataset_phys \
  --real .cache/a2v_robotwin/gen_real.mp4 --none .cache/a2v_robotwin/gen_none.mp4
```

---

## Step 1 · Prepare data

Two stages: `robotwin_adapter` reads RoboTwin hdf5 → `actions/intrinsic/extrinsic.npy` + a
`manifest.jsonl`; `prepare` renders the trajectory maps and stores `video` and `vace_video` as
same-indexed, lossless PNG lists (the core A2V alignment invariant — both routes see identical
frames). For non-RoboTwin data, produce the same manifest yourself (see **Reference**).

```bash
# adapter: RoboTwin episodes -> manifest.jsonl (see episode selection below)
$PY -m a2v.data.robotwin_adapter --task <task> --robot_mode <robot_mode> \
  [--episodes 0,3,7 | --episodes_range 0-49] --work_dir <work_dir>

# prepare: manifest -> a ready A2V dataset (renders EVERY row in the manifest)
$PY -m a2v.data.prepare --manifest <work_dir>/manifest.jsonl --output_dir <dataset_dir> \
  --height H --width W --num_frames N \         # N = 4n+1; H,W divisible by 16 (Wan2.1) / 32 (Wan2.2)
  --resize_mode stretch --gripper_z_offset 0 --radius_mode physical \
  [--max_num_frames C --min_num_frames 49] \    # variable-length: each ep -> largest 4n+1 <= min(its len, C)
  [--write_overlay] [--skip_short]              # overlay = debug; skip_short = drop too-short episodes
```

### One episode, a range, or a whole directory

`robotwin_adapter` reads episodes from `<root>/<task>/<robot_mode>/data/episode*.hdf5`
(default `--root /data/RoboTwin2.0_unpacked`; each needs a matching `video/episodeN.mp4`) and
writes **one manifest row per episode**. `prepare` then renders every row, so the dataset size is
decided entirely at the adapter step:

| `robotwin_adapter` flag | episodes taken |
| --- | --- |
| `--episodes 0` | just episode 0 |
| `--episodes 0,3,7` | those indices |
| `--episodes_range 0-49` | inclusive range 0..49 |
| *(omit both)* | **all `episode*.hdf5` in the dir** (missing hdf5/mp4 are skipped) |

```bash
# whole directory -> one multi-episode dataset (just drop --episodes)
$PY -m a2v.data.robotwin_adapter --task beat_block_hammer \
  --robot_mode aloha-agilex_clean_50 --work_dir .cache/a2v_robotwin/allep_work
$PY -m a2v.data.prepare --manifest .cache/a2v_robotwin/allep_work/manifest.jsonl \
  --output_dir .cache/a2v_robotwin/allep_dataset_phys \
  --height 240 --width 320 --num_frames 105 \
  --resize_mode stretch --gripper_z_offset 0 --radius_mode physical --skip_short
```

> Episodes often vary in length: pass `--skip_short` (without it `prepare` errors on any episode
> shorter than `--num_frames`) and pick an `N` (`4n+1`) the episodes you want all meet.

To split train / held-out within one task, run the adapter twice with complementary ranges
(`--episodes_range 0-39` and `40-49`) into separate `--work_dir`/`--output_dir`.

### Variable-length episodes (`--max_num_frames`)

RoboTwin episodes vary a lot in length (often 50–300+ frames). A fixed `--num_frames N` either
drops every episode shorter than `N` (with `--skip_short`) or wastes the extra frames of longer
ones. Instead, pass `--max_num_frames C` (a cap, `4n+1`): **each episode is rendered at the largest
`4n+1 ≤ min(its length, C)`** — long episodes get the full `C`-frame window, shorter ones keep as
many frames as they have, and only those below `--min_num_frames` (default `49`) are dropped. This
keeps far more data while giving every clip its longest feasible temporal window.

This is safe because A2V stores each stream as a PNG list and training loads the list **as-is**
(batch size 1, no cross-clip stacking), so rows may differ in length; the only hard rule is that
each clip is `4n+1` (VAE temporal factor). When a dataset is variable-length, also pass
`--max_num_frames C` to `a2v.data.validate` (it then checks `4n+1 ≤ C` and that `video`/`vace_video`
stay equal-length, instead of requiring an exact `N`).

### Whole dataset in one command — `build_robotwin_all.sh`

Builds **one** dataset (no split) from every `<task>/<robot>_<SUFFIX>` variant found on disk for a
given `SUFFIX`, running adapter → prepare → merge → validate in parallel. Variants are discovered by
globbing the data root, so robots missing for some tasks are simply skipped. Pure CPU.

```bash
# variable-length, cap 121, all clean_50 variants
JOBS=64 MAX_FRAMES=121 MIN_FRAMES=49 SUFFIX=clean_50 \
  OUT=.cache/a2v_robotwin/clean50_480x640_c121 \
  bash a2v/data/build_robotwin_all.sh
# fixed-length instead: drop MAX_FRAMES, set FRAMES=49 ;  preview only: DRY_RUN=1
```

| env var | meaning |
| --- | --- |
| `SUFFIX` | data subset, e.g. `clean_50` (default) or `randomized_500` |
| `MAX_FRAMES` `MIN_FRAMES` | set `MAX_FRAMES` → variable-length (cap / floor); unset → fixed `FRAMES` |
| `H` `W` `FRAMES` | resolution; `FRAMES` only used in fixed mode |
| `JOBS` | parallel variants (default 48) |
| `PY` `ROOT` `OUT` | python / RoboTwin root / output dataset dir |
| `DRY_RUN=1` | print the variants and example commands, run nothing |

### Multi-task / multi-robot (with automatic train/held-out split)

`build_dataset.sh` fans out adapter+prepare per (task, robot) variant in parallel, then merges
them into one train + one held-out split:

```bash
TASKS="beat_block_hammer adjust_bottle" ROBOTS="aloha-agilex franka ur5" \
  SUFFIX=clean_50 TRAIN_RANGE=0-39 HELDOUT_RANGE=40-49 \
  H=480 W=640 FRAMES=49 bash a2v/data/build_dataset.sh
# -> .cache/a2v_robotwin/wm_480_{train,heldout}
```

- `SUFFIX` is appended to each robot (`<robot>_<SUFFIX>`, e.g. `aloha-agilex_clean_50`).
- `TRAIN_RANGE` / `HELDOUT_RANGE` are inclusive episode ranges per split (defaults `0-449` /
  `450-499` for `randomized_500`); set them to match how many episodes your dir holds.

---

## Step 2 · Sanity checks (recommended before a long run)

Cheap gates that catch data/model problems before you commit GPU hours.

```bash
# data is well-formed for this base (no model weights needed)
$PY -m a2v.data.validate --base_spec <spec> \
  --dataset_base_path <dataset_dir> --dataset_metadata_path <dataset_dir>/metadata.jsonl \
  --height H --width W --num_frames N \
  [--max_num_frames C]                  # add for variable-length datasets (checks 4n+1 <= C)

# weights load + the VACE branch builds correctly for this base
$PY -m a2v.check_load        --base_spec <spec>
# + zero-side-effect provisioning gate (from-DiT bases); needs a dataset
$PY -m a2v.check_provision   --base_spec <spec> --dataset <dataset_dir>

# regression smokes (no dataset / no base needed)
$PY -m a2v.check_parity                  # VACE-1.3B abstraction parity (loads VACE-1.3B)
$PY -m a2v.render.check_projection       # intrinsic-scaling regression
$PY -m a2v.data.smoke                    # synthetic end-to-end data smoke
```

---

## Step 3 · Train — `a2v/train.sh`

**`bash a2v/train.sh <base_spec>` is the single training entry** for every base and every scale.
It wraps `python -m a2v.train_a2v` and fills presets (dataset / resolution / lr / optimizer /
first-frame / LoRA-vs-full-param) from the spec. Override any preset with an env var.

```bash
# single-episode overfit (smallest sanity run)
CUDA_VISIBLE_DEVICES=0 bash a2v/train.sh wan2.1-vace-1.3b

# multi-episode, multi-GPU (raw dataset; encoders run each step)
NPROC=8 DATASET=.cache/a2v_robotwin/ep_train40_phys \
  FRAMES=105 REPEAT=4 EPOCHS=40 OUT=models/train/my_run \
  bash a2v/train.sh wan2.1-i2v-14b-480p

DRY_RUN=1 bash a2v/train.sh wan2.2-ti2v-5b   # print the resolved command, run nothing
```

Checkpoints land in `models/train/.../step-*.safetensors` (a LoRA for `vace-1.3b`, a full VACE
state dict otherwise).

### Env overrides

| var | meaning |
| --- | --- |
| `DATASET` `OUT` `LR` `HEIGHT` `WIDTH` `FRAMES` `REPEAT` `EPOCHS` `SAVE_STEPS` | override per-base presets |
| `NPROC` | data-parallel GPUs (DDP); default `1` |
| `CACHE_TRAIN=1` `CACHE_DIR=` | train from a pre-encoded cache (see below) |
| `RESUME=<ckpt>` | resume a long run |
| `EXPERT=high\|low` | **required** for `*-a14b` (selects the MoE expert + its timestep band) |
| `OPTIMIZER=adamw\|adamw_offload\|adam8bit` | default `adamw` (full fp32 AdamW, no offload — strongest, fits 1.3B/5B). `adamw_offload` = full AdamW + activation offload (default for 14B/A14B). `adam8bit` = last-resort low-VRAM. **None default to 8-bit.** |
| `LR_SCHEDULE=cosine\|constant` | default `cosine` = linear warmup → cosine decay (then `LR` is the **peak**); `constant` = flat lr |
| `WARMUP=<steps>` | cosine warmup steps; `0` → 3% of total (`num_epochs × len(dataset)`) |
| `LR_MIN_RATIO=<frac>` | cosine final lr as a fraction of peak (default `0`) |
| `LOGGER=wandb,…` | logging backends, comma list: `wandb` (default) `tensorboard` `swanlab` `none`. `WANDB_PROJECT`/`SWANLAB_PROJECT` name the project |
| `LOG_EVERY=<n>` | log `lr`/`grad_norm`/`throughput_it_s`/`gpu_mem_gb` every `n` steps (default `10`; `loss` every step) |
| `GRAD_NORM=0\|1` | log grad norm on logging steps, computed before `zero_grad` (default `1`) |
| `SAMPLE_EVERY=<n>` | periodic in-training generation cadence (`0`=off); plus `SAMPLE_ROWS=` `SAMPLE_CONTROLS=` `SAMPLE_FRAMES=` `SAMPLE_STEPS=` |
| `DRY_RUN=1` | print the command without launching |

### Faster / lower-VRAM: cache-train (recommended for multi-epoch / large runs)

Pre-encode the dataset once (VAE/T5/CLIP outputs cached), then train loading only the DiT(+VAE).
Use a **fresh cache dir whenever resolution or data changes** (the cache is keyed by row index,
not content).

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

### LR schedule

The default is constant lr. For longer / larger runs, `LR_SCHEDULE=cosine` warms up then
cosine-decays — it lets you safely use a higher peak `LR` (warmup avoids early divergence) and
settle into a lower final loss. Example:
`LR_SCHEDULE=cosine WARMUP=300 LR=2e-5 LR_MIN_RATIO=0.05 ... bash a2v/train.sh <spec>`.

### Logging & in-training samples (`a2v/train_logging.py`)

Every run logs `loss` (DDP-averaged), `lr`, `grad_norm`, `throughput_it_s`, `gpu_mem_gb` to the
selected backend(s). `SAMPLE_EVERY=N` reuses the *live* pipe to render short clips every `N`
steps → mp4s under `<OUT>/samples/` plus a tensorboard image strip / wandb video. Sampling is
auto-skipped (with a one-time warning) for `CACHE_TRAIN` (encoders are pruned), dual-expert
`*-a14b` (needs both experts), and cpu-offload. Example:
`LOGGER=tensorboard,wandb SAMPLE_EVERY=500 SAMPLE_CONTROLS=real,none SAMPLE_STEPS=20 ... bash a2v/train.sh wan2.2-ti2v-5b`.

---

## Step 4 · Evaluate (causal gate)

The model is judged by **causal control**, not pixel loss alone: generate under the **real**
trajectory and a **negative control** (`none` = blank map, `shuffle` = reversed), and require the
real output to be clearly closer to GT.

**Single episode** — `infer_a2v` then `causal_metrics`:

```bash
for c in real none shuffle; do $PY -m a2v.infer_a2v --base_spec <spec> \
  --dataset <dataset_dir> --lora <ckpt.safetensors> \
  --num_frames N --height H --width W --control $c --output .cache/a2v_robotwin/gen_$c.mp4; done
$PY -m a2v.causal_metrics --dataset <dataset_dir> \
  --real .cache/a2v_robotwin/gen_real.mp4 --none .cache/a2v_robotwin/gen_none.mp4
# dual-expert: add --lora_low <low_ckpt> to infer_a2v
```

**Multi-episode generalization** — `eval_multiep` (loads the model once, writes a per-episode
table + `metrics.json` + a `SUMMARY` with verdict):

```bash
$PY -m a2v.eval_multiep --base_spec <spec> --lora <ckpt> [--lora_low <low_ckpt>] \
  --dataset .cache/a2v_robotwin/wm_480_heldout --rows 0,1,50,51 \
  --num_frames N --height H --width W --controls real,none \
  --output_dir .cache/a2v_robotwin/eval_heldout
```

**Pass criteria**: mean REAL MAE-GT well below NONE, `real<none` on (almost) every episode, REAL
motion in the same order as GT. For strong-first-frame bases (i2v / ti2v) the negative control
need not be static — it freely animates from frame 0 — so the signal is the clear `REAL ≪ NONE`
gap, not "NONE is frozen".

---

## Advanced · Dual-expert A14B (`wan2.2-i2v-a14b`, `wan2.2-vace-fun-a14b`)

A14B bases are a **two-expert MoE**: a high-noise expert and a low-noise expert. Train each as its
own job (`EXPERT=` selects the expert and its timestep band), then pass **both** checkpoints at
inference; DiffSynth switches experts by timestep automatically. Prepare/evaluate are otherwise
the same as Step 1 / Step 4.

```bash
# 1. train the two experts (separate GPUs, or sequentially)
EXPERT=high CUDA_VISIBLE_DEVICES=0 bash a2v/train.sh wan2.2-i2v-a14b   # -> ..._vace_a14b_high/
EXPERT=low  CUDA_VISIBLE_DEVICES=1 bash a2v/train.sh wan2.2-i2v-a14b   # -> ..._vace_a14b_low/

# 2. combined causal gate: high via --lora, low via --lora_low
HI=models/train/a2v_robotwin_ep0_vace_a14b_high/step-1000.safetensors
LO=models/train/a2v_robotwin_ep0_vace_a14b_low/step-1000.safetensors
for c in real none shuffle; do $PY -m a2v.infer_a2v --base_spec wan2.2-i2v-a14b \
  --dataset .cache/a2v_robotwin/ep0_dataset_phys --lora $HI --lora_low $LO \
  --num_frames 121 --height 240 --width 320 --control $c \
  --output .cache/a2v_robotwin/gen_a14b_$c.mp4; done
$PY -m a2v.causal_metrics --dataset .cache/a2v_robotwin/ep0_dataset_phys \
  --real .cache/a2v_robotwin/gen_a14b_real.mp4 --none .cache/a2v_robotwin/gen_a14b_none.mp4
```

- **Two A14B variants**: `wan2.2-i2v-a14b` builds VACE from-DiT with the `i2v_vae` first frame
  (strongest pixel fidelity — best single-episode gate so far); `wan2.2-vace-fun-a14b` warm-starts
  from PAI's pretrained dual VACE with `vace_reference`. Same `EXPERT=high|low` two-run pattern;
  downloads differ (see `A2V_HISTORY.md` §18–§19).
- **Inference loads both 14B experts** → CPU offload is enabled automatically (~45GB GPU); a single
  GPU is enough. Always pass `--lora_low` for A14B (omitting it errors out).
- **Multi-episode + cache**: pre-encode once (the cache is shared by both experts), cache-train
  each expert with its `EXPERT=` band, then evaluate with `--lora_low`:

  ```bash
  # encode once -> shared cache; then per-expert cache-train
  for E in high low; do
    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 SPEC=wan2.2-i2v-a14b CACHE_TRAIN=1 \
      CACHE_DIR=.cache/a2v_robotwin/cache_train40_a14b NPROC=8 EXPERT=$E \
      LR=1e-5 REPEAT=4 EPOCHS=40 SAVE_STEPS=100 \
      OUT=models/train/a2v_robotwin_train40_vace_a14b_${E}_cached \
      bash a2v/train.sh wan2.2-i2v-a14b
  done
  $PY -m a2v.eval_multiep --base_spec wan2.2-i2v-a14b \
    --lora .../a14b_high_cached/step-800.safetensors \
    --lora_low .../a14b_low_cached/step-800.safetensors \
    --dataset .cache/a2v_robotwin/ep_heldout10_phys --num_frames 105 --height 240 --width 320 \
    --controls real,none --output_dir .cache/a2v_robotwin/eval_heldout_a14b
  ```

---

## Reference

### Bring your own (non-RoboTwin) data

`robotwin_adapter` is RoboTwin-specific. For any other source, produce the same inputs `prepare`
consumes — per episode, the **source video** + three `.npy` files, plus one manifest row — then run
`a2v.data.prepare` exactly as in Step 1.

| input | shape / type | meaning |
| --- | --- | --- |
| source video | `.mp4` (or readable frames) | one frame per timestep, **frame-aligned** with the actions |
| `actions.npy` | `(T, 16)` float | both arms: `[Lxyz, Lquat(xyzw), Lgrip, Rxyz, Rquat(xyzw), Rgrip]` (gripper in [0,1], ×120 for the colormap) |
| `intrinsic.npy` | `(3, 3)` float | OpenCV camera intrinsics `K` |
| `extrinsic.npy` | `(T, 4, 4)` float | **camera-to-world (c2w)** per frame |

Manifest (`metadata.jsonl`), one JSON object per line:

```json
{
  "id": "unique_name",                 // becomes the episode output subdir
  "video": "/path/episode.mp4",        // REQUIRED
  "action_path": "/path/actions.npy",  // REQUIRED  (T,16)
  "intrinsic_path": "/path/intrinsic.npy",  // REQUIRED (3,3)
  "extrinsic_path": "/path/extrinsic.npy",  // REQUIRED (T,4,4) c2w
  "original_size": [H, W],             // source video resolution (validated; must match)
  "prompt": "task instruction text",   // optional
  "total_frames": T                    // optional (auto-read from the video if omitted)
}
```

Conventions that must hold (otherwise the projection misaligns):
- `extrinsic` is **c2w**. RoboTwin stores world-to-camera (w2c) and `robotwin_adapter` inverts it;
  if your poses are already c2w, do not invert again.
- `original_size` must equal the real video resolution (intrinsics are scaled to the training
  `--height/--width` from it).
- End-effector poses are **TCP-framed** → `--gripper_z_offset 0`. Quaternions are **xyzw**.
  `video` and `actions` share one timeline (frame `t` ↔ action `t`).

### Base-model internals (derived automatically, never hardcode)

`vace_in_dim = 2·z_dim + spatial²` (96 for Wan2.1, **352** for Wan2.2), `mask_pq = spatial`
(8 / **16**), and explicit `vace_layers` per depth — all derived from the spec in `a2v/base_spec.py`.

### Repo layout

```
a2v/
  base_spec.py        WanBaseSpec + REGISTRY (one entry per base model)
  provision.py        VACE supply: create_vace_from_dit / ensure_vace / provision_a2v
  vace_unit.py        mask_pq-parameterized VACE unit (no diffsynth edit)
  train_a2v.py        trainer module wrapped by train.sh (frame-list operator, from-DiT vace, cache-train)
  train_logging.py    A2VModelLogger + instrumented training loop (lr/grad-norm/throughput/mem) + PeriodicSampler
  train.sh            >> training entry: bash a2v/train.sh <base_spec> [env overrides]
  infer_a2v.py        inference + causal-control harness (real / none / shuffle)
  eval_multiep.py     multi-episode generalization evaluator (model loaded once)
  causal_metrics.py   MAE-GT / motion / real-vs-none gate metrics
  check_load.py       check_provision.py  check_parity.py   pre-flight gates
  data/               robotwin_adapter, prepare (+ --max_num_frames variable-length), operators, validate, smoke
                      build_dataset.sh (multi-task train/heldout split), build_robotwin_all.sh (whole-suffix single split)
  render/             traj_map, action_io, check_projection
  README_A2V.md       this guide
  A2V_HANDOFF.md      current status + handoff checklist + quick reference
  A2V_HISTORY.md      per-track debugging history, decisions, and lessons (appendix)
```

See `A2V_HANDOFF.md` for current status and the latest results; `A2V_HISTORY.md` for the full
per-track history and lessons.
