# A2V environment — reproduce on another machine

The exact environment behind all A2V results. Captured from the working `.venv` at
`/vepfs/wangshilong/code/DiffSynth-Studio/.venv`.

## Target platform

| | value |
| --- | --- |
| OS | Ubuntu 22.04.3 LTS (glibc 2.35) |
| GPU | NVIDIA A100-SXM4-80GB (8×) |
| NVIDIA driver | 535.129.03 (supports CUDA 12.1 runtime) |
| Python | **3.11.9** |
| CUDA (torch wheels) | **12.1** (`torch 2.5.1+cu121`, cuDNN 9.1) |
| venv tool | `uv` 0.5.1 (plain `python -m venv` works too) |

The CUDA toolkit does **not** need to be installed system-wide — the `nvidia-*-cu12` pip wheels
ship the runtime libs. Only a host NVIDIA driver new enough for CUDA 12.1 is required.

## What's pinned

- `a2v/requirements.lock.txt` — full `pip freeze` (112 packages, exact `==` pins) for byte-for-byte
  reproduction. Includes the `--extra-index-url` for the PyTorch cu121 wheels.
- `pyproject.toml` (repo root) — the loose `diffsynth` dependency set; use this for a fresh/relaxed
  install on a different CUDA or Python.

Key versions (for reference): `torch 2.5.1+cu121`, `torchvision 0.20.1+cu121`, `transformers 5.9.0`,
`accelerate 1.13.0`, `peft 0.19.1`, `bitsandbytes 0.49.2`, `safetensors 0.7.0`, `einops 0.8.2`,
`numpy 2.4.4`, `h5py 3.16.0`, `modelscope 1.37.1`.

All three training loggers are installed: `tensorboard 2.20.0` (default), `wandb 0.27.2`,
`swanlab 0.8.3` — switch with `LOGGER=tensorboard|wandb|swanlab` (see `README_A2V.md` Step 3).

## Reproduce — exact (recommended)

```bash
git clone <this-repo> DiffSynth-Studio && cd DiffSynth-Studio
git checkout a2v                       # the A2V branch

# 1. create the venv (uv; or: python3.11 -m venv .venv)
uv venv --python 3.11.9 .venv          #  -> .venv/
#   plain venv alternative:  python3.11 -m venv .venv

# 2. install the exact pinned set
.venv/bin/python -m pip install -U pip
.venv/bin/python -m pip install -r a2v/requirements.lock.txt

# 3. install diffsynth itself (editable, from this repo)
.venv/bin/python -m pip install -e .

# 4. verify
.venv/bin/python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
.venv/bin/python -m a2v.data.smoke               # data path smoke (no GPU)
CUDA_VISIBLE_DEVICES=0 .venv/bin/python -m a2v.check_load --base_spec wan2.1-vace-1.3b
```

> `diffsynth` is installed **editable** (`pip install -e .`) so the repo's `diffsynth/` is what runs
> at import time — and so the A2V code under `a2v/` (which never modifies `diffsynth/`) picks up the
> repo, not a stale site-packages copy. Do not `pip install diffsynth` from PyPI.

## Reproduce — relaxed (different CUDA / Python)

If the target box has a different CUDA, install a matching torch first, then the rest loosely:

```bash
uv venv --python 3.11 .venv
# pick the wheel for your CUDA, e.g. cu124:
.venv/bin/python -m pip install torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu124
.venv/bin/python -m pip install -e .          # pulls deps from pyproject.toml
.venv/bin/python -m pip install bitsandbytes tensorboard wandb swanlab
```

Note `bitsandbytes`, `tensorboard`, `wandb` and `swanlab` are **not** in `pyproject.toml` but are
used by A2V (8-bit Adam for the 14B/5B/A14B bases, and the training loggers), so install them
explicitly. wandb/swanlab pin `protobuf<7`.

## Optional loggers (installed)

`wandb` and `swanlab` are in the lock so you can switch `LOGGER` freely. wandb runs offline unless
you log in — set a key with `wandb login`, or force offline with `WANDB_MODE=offline`; swanlab
writes under `<OUT>/swanlab_log/` and likewise has its own login for cloud sync.

> Installing them downgraded `protobuf` 7.35.0 -> **6.33.6** (wandb/swanlab require `protobuf<7`).
> Core imports (diffsynth / transformers / sentencepiece) were verified to still work at 6.33.6.

## Extras — NOT installed (add only when needed)

| package | needed for |
| --- | --- |
| `torchaudio` `librosa` `av` | audio bases only (not A2V) |
| `matplotlib` `decord` | optional; the code has fallbacks, A2V does not need them |
| `deepspeed` | not used by A2V (see note below) |

## Notes / gotchas

- **deepspeed is intentionally NOT installed.** A2V runs single-GPU 8-bit Adam + DDP and does not
  need it; `diffsynth` imports cleanly without it (`gradient_checkpoint._HAS_DEEPSPEED=False`). If a
  later `pip install deepspeed` reintroduces an `nvcc` probe crash, the simplest fix is to uninstall
  it again.
- **bitsandbytes needs a CUDA GPU** at import (8-bit optimizers). CPU-only boxes can install
  everything else but cannot run the 14B/5B/A14B 8-bit-Adam recipes.
- **numpy 2.x**: TensorBoard's `add_video` breaks under numpy>=2 (`reshape ... newshape`); the A2V
  logger already falls back to an image strip, so this is handled — just don't downpin numpy
  expecting video tiles.
- Model weights are **not** part of this env; they are downloaded/converted separately (see
  `A2V_HANDOFF.md`). This doc covers only the Python environment.

## Refresh this lock

After changing dependencies, regenerate:

```bash
.venv/bin/pip freeze | grep -v -i diffsynth > a2v/requirements.lock.txt
# then re-add the header + --extra-index-url line at the top (see the current file).
```
