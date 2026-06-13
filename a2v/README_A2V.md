# A2V VACE Data Tools

This package contains the low-intrusion A2V preparation path for Wan2.1 VACE.
It renders ABot actions into 3-channel trajectory control videos, writes target
and control frames as lossless PNG sequences, and emits JSONL metadata for
the A2V training wrapper.

## Manifest

Input manifest is JSONL or JSON. Each row needs:

```json
{
  "id": "sample_0001",
  "video": "/path/to/source.mp4",
  "action_path": "/path/to/actions.h5",
  "intrinsic_path": "/path/to/intrinsic.json",
  "extrinsic_path": "/path/to/extrinsic.json",
  "original_size": [480, 832],
  "prompt": "robot arm action trajectory"
}
```

`action_path` supports ABot `.h5` files. For smoke/debugging it also supports
`.npy` or `.npz` arrays with shape `[T, 16]` in:

```text
left_xyz(3), left_xyzw(4), left_gripper(1), right_xyz(3), right_xyzw(4), right_gripper(1)
```

Optional fields:

- `frame_indices`: fixed list of frame ids. If present, it must have `num_frames` entries.
- `start_frame`: deterministic window start when `frame_indices` is absent.
- `total_frames`: skip video frame counting when known.

## Prepare

Use `crop` resize mode to match DiffSynth `ImageCropAndResize`.

```bash
.venv/bin/python -m a2v.data.prepare \
  --manifest /path/to/manifest.jsonl \
  --output_dir data/a2v/my_dataset \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --resize_mode crop \
  --write_overlay
```

Output:

```text
data/a2v/my_dataset/
  metadata.jsonl
  sample_0001/video/00000.png ...
  sample_0001/vace_video/00000.png ...
  sample_0001/reference/00000.png
  sample_0001/overlay/00000.png ...
```

The output metadata stores `video` and `vace_video` as PNG frame-path lists, so
both are fixed to the same frame indices before the DiffSynth dataset sees them.
This is the key alignment invariant for VACE A2V training.

## Rendering Notes

Gripper colors follow ABot's `matplotlib.cm.Greens` and `matplotlib.cm.Reds`
semantics. If `matplotlib` is not installed, A2V falls back to an internal
ColorBrewer approximation so smoke tests remain runnable in minimal envs.

Trajectory drawing uses PIL instead of ABot's cv2 backend. This keeps the first
stage dependency-light, but there can be tiny anti-aliasing or line-width
differences. For strict pixel-level warm-start matching against an ABot
checkpoint, switch the drawing backend to cv2.

## Validate

```bash
.venv/bin/python -m a2v.data.validate \
  --dataset_base_path data/a2v/my_dataset \
  --dataset_metadata_path data/a2v/my_dataset/metadata.jsonl \
  --height 480 \
  --width 832 \
  --num_frames 49
```

## Smoke Test

No real data or model weights are required:

```bash
.venv/bin/python -m a2v.data.smoke
```

## Master Plan Status

Current status against `A2V_master_plan.md`:

- T1 render/data/overlay/synthetic dataset smoke: implemented and verified.
- T1 real-data overlay gate, official baseline, LoRA overfit, full VACE train, ablations, and inference harness: still pending.
- `metadata.jsonl` is used instead of CSV because frame-path lists must remain structured.
- A2V frame-list loading is kept in `a2v.data.frame_list_video_operator()` to avoid modifying DiffSynth core data flow.

## Training Wiring Dependencies

Stock DiffSynth `UnifiedDataset.default_video_operator()` accepts video paths
for the video route, not JSONL frame-path lists. When training from the PNG lists
produced here, inject `a2v.data.operators.frame_list_video_operator` into the
training dataset construction. Do not add list handling to DiffSynth core data
flow for A2V.

`vace_reference_image` is the target video's first frame, aligned with
`video[0]` and `vace_video[0]`. This follows official VACE usage: the pipeline
adds the reference as an extra latent frame in front. Do not insert an additional
first frame into `vace_video`.

## Train With Wan VACE LoRA

The strict master-plan path is to use an A2V training wrapper that passes `a2v.data.frame_list_video_operator()` explicitly. The stock script command below is retained as a reference for model/training flags; it should be wired through `a2v/train_a2v.py` before running on JSONL frame lists.

```bash
accelerate launch examples/wanvideo/model_training/train.py \
  --dataset_base_path data/a2v/my_dataset \
  --dataset_metadata_path data/a2v/my_dataset/metadata.jsonl \
  --data_file_keys "video,vace_video,vace_reference_image" \
  --height 480 \
  --width 832 \
  --num_frames 49 \
  --dataset_repeat 100 \
  --model_id_with_origin_paths "Wan-AI/Wan2.1-VACE-1.3B:diffusion_pytorch_model*.safetensors,Wan-AI/Wan2.1-VACE-1.3B:models_t5_umt5-xxl-enc-bf16.pth,Wan-AI/Wan2.1-VACE-1.3B:Wan2.1_VAE.pth" \
  --learning_rate 1e-4 \
  --num_epochs 5 \
  --remove_prefix_in_ckpt "pipe.vace." \
  --output_path "./models/train/Wan2.1-VACE-1.3B_a2v_lora" \
  --lora_base_model "vace" \
  --lora_target_modules "q,k,v,o,ffn.0,ffn.2" \
  --lora_rank 32 \
  --extra_inputs "vace_video,vace_reference_image" \
  --use_gradient_checkpointing_offload
```
