#!/usr/bin/env python3
"""A2V training entry — stock Wan VACE training with PNG-list data routing.

This is a thin wrapper over ``examples/wanvideo/model_training/train.py``. The
ONLY behavioural difference from the stock entry is the dataset's
``main_data_operator``: A2V prepares ``video`` and ``vace_video`` as aligned PNG
frame *lists* (so the two streams stay frame-locked — invariant I2), but the
stock ``UnifiedDataset.default_video_operator`` only routes ``str`` paths. We
swap in ``a2v.data.operators.frame_list_video_operator``, which adds the
``(list, ...)`` route while keeping the normal image/gif/video routes.

Everything else — model loading, VACE branch selection, LoRA, the loss, the
launcher — is the stock code, imported unchanged.

Run via ``accelerate launch -m a2v.train_a2v ...`` (see ``a2v/run_overfit.sh``).
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import accelerate

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from diffsynth.core import UnifiedDataset  # noqa: E402
from diffsynth.core.data.operators import LoadVideo, LoadAudio, ImageCropAndResize, ToAbsolutePath  # noqa: E402,F401
from diffsynth.diffusion import *  # noqa: E402,F401,F403  (ModelLogger, launch_*_task, ...)

from a2v.base_spec import get_spec  # noqa: E402
from a2v.data.operators import frame_list_video_operator  # noqa: E402
from a2v.provision import ensure_vace, provision_a2v  # noqa: E402


def _load_stock_train_module():
    """Import the stock Wan training module by file path (it is not a package)."""
    stock_path = REPO_ROOT / "examples" / "wanvideo" / "model_training" / "train.py"
    spec = importlib.util.spec_from_file_location("a2v_stock_wan_train", stock_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _make_training_module_cls(stock):
    """Subclass the stock WanTrainingModule so the VACE branch is provisioned BEFORE
    LoRA attaches.

    For a base without a pretrained VACE branch (T3: T2V-1.3B), `pipe.vace` is None
    after `from_pretrained`. The stock constructor then calls
    `switch_pipe_to_training_mode`, whose LoRA step bails out with "No vace models..."
    when `pipe.vace is None` (training_module.py:289-292). We hook that method to
    `ensure_vace` first (create-from-DiT), so LoRA patches the freshly built branch.
    For VACE-1.3B the hook is a no-op (vace already loaded) -> T2 path unchanged.
    """

    class A2VWanTrainingModule(stock.WanTrainingModule):
        def __init__(self, *args, a2v_spec=None, **kwargs):
            # Set before super().__init__ (which calls switch_*). Bypass nn.Module's
            # __setattr__ since _parameters/_modules don't exist yet at this point.
            object.__setattr__(self, "_a2v_spec", a2v_spec)
            super().__init__(*args, **kwargs)

        def switch_pipe_to_training_mode(self, pipe, *args, **kwargs):
            if getattr(self, "_a2v_spec", None) is not None:
                ensure_vace(pipe, self._a2v_spec)  # T3: build vace; T2: idempotent no-op
            return super().switch_pipe_to_training_mode(pipe, *args, **kwargs)

    return A2VWanTrainingModule


def main() -> None:
    stock = _load_stock_train_module()
    parser = stock.wan_parser()
    parser.add_argument("--base_spec", default=None,
                        help="WanBaseSpec name (a2v.base_spec.REGISTRY). Fills model_paths/tokenizer/lora "
                             "defaults and drives the VACE seams. Explicit CLI flags still override.")
    # The stock wan_parser defaults --remove_prefix_in_ckpt to "pipe.dit." (a non-None
    # value), which would shadow the spec default below (`is None` never True) and save
    # the VACE LoRA with a "pipe.vace." prefix that infer's load_lora cannot match.
    # Reset the default to None so "explicit CLI vs unset" is distinguishable; restore
    # the stock "pipe.dit." fallback after the spec branch when no spec is given.
    parser.set_defaults(remove_prefix_in_ckpt=None)
    args = parser.parse_args()

    # --- spec-driven defaults (T2): explicit CLI always wins ---
    spec = get_spec(args.base_spec) if args.base_spec else None
    if spec is not None:
        if args.model_paths is None and args.model_id_with_origin_paths is None:
            args.model_paths = json.dumps(spec.model_paths())
        if args.tokenizer_path is None:
            args.tokenizer_path = spec.abs_tokenizer_path()
        if args.lora_base_model is None and args.trainable_models is None:
            args.lora_base_model = spec.lora_base_model
            args.lora_target_modules = spec.lora_target_modules
        if args.remove_prefix_in_ckpt is None:
            args.remove_prefix_in_ckpt = spec.vace_remove_prefix
    if args.remove_prefix_in_ckpt is None:
        args.remove_prefix_in_ckpt = "pipe.dit."  # stock default (non-spec usage)

    accelerator = accelerate.Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        kwargs_handlers=[accelerate.DistributedDataParallelKwargs(find_unused_parameters=args.find_unused_parameters)],
    )

    # --- the one deviation from stock: PNG-list-aware video operator ---
    dataset = UnifiedDataset(
        base_path=args.dataset_base_path,
        metadata_path=args.dataset_metadata_path,
        repeat=args.dataset_repeat,
        data_file_keys=args.data_file_keys.split(","),
        main_data_operator=frame_list_video_operator(
            base_path=args.dataset_base_path,
            max_pixels=args.max_pixels,
            height=args.height,
            width=args.width,
            height_division_factor=16,
            width_division_factor=16,
            num_frames=args.num_frames,
            time_division_factor=4 if not args.framewise_decoding else 1,
            time_division_remainder=1 if not args.framewise_decoding else 0,
        ),
        special_operator_map={
            "animate_face_video": ToAbsolutePath(args.dataset_base_path) >> LoadVideo(args.num_frames, 4, 1, frame_processor=ImageCropAndResize(512, 512, None, 16, 16)),
            "input_audio": ToAbsolutePath(args.dataset_base_path) >> LoadAudio(sr=16000),
            "wantodance_music_path": ToAbsolutePath(args.dataset_base_path),
        },
    )

    training_module_cls = _make_training_module_cls(stock)
    model = training_module_cls(
        a2v_spec=spec,
        model_paths=args.model_paths,
        model_id_with_origin_paths=args.model_id_with_origin_paths,
        tokenizer_path=args.tokenizer_path,
        audio_processor_path=args.audio_processor_path,
        trainable_models=args.trainable_models,
        lora_base_model=args.lora_base_model,
        lora_target_modules=args.lora_target_modules,
        lora_rank=args.lora_rank,
        lora_checkpoint=args.lora_checkpoint,
        preset_lora_path=args.preset_lora_path,
        preset_lora_model=args.preset_lora_model,
        use_gradient_checkpointing=args.use_gradient_checkpointing,
        use_gradient_checkpointing_offload=args.use_gradient_checkpointing_offload,
        extra_inputs=args.extra_inputs,
        fp8_models=args.fp8_models,
        offload_models=args.offload_models,
        resume_from_checkpoint=args.resume_from_checkpoint,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        task=args.task,
        device="cpu" if (args.initialize_model_on_cpu or args.enable_model_cpu_offload) else accelerator.device,
        max_timestep_boundary=args.max_timestep_boundary,
        min_timestep_boundary=args.min_timestep_boundary,
    )
    # Install the mask_pq-parameterized VACE unit (SEAM-2). ensure_vace here is the
    # idempotent no-op companion to the in-constructor hook above: for T3 the branch was
    # already built (and LoRA-patched) before switch_*, so this only swaps the unit; for
    # VACE-1.3B it is the same no-op as T2 (pretrained vace, mask_pq=8). Required for T5
    # (mask_pq=16).
    if spec is not None:
        provision_a2v(model.pipe, spec)

    model_logger = ModelLogger(  # noqa: F405
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_tensorboard_log=args.enable_tensorboard_log,
        enable_swanlab_log=args.enable_swanlab_log,
        swanlab_project=args.swanlab_project,
        enable_wandb_log=args.enable_wandb_log,
        wandb_project=args.wandb_project,
    )
    launcher_map = {
        "sft:data_process": launch_data_process_task,  # noqa: F405
        "direct_distill:data_process": launch_data_process_task,  # noqa: F405
        "sft": launch_training_task,  # noqa: F405
        "sft:train": launch_training_task,  # noqa: F405
        "direct_distill": launch_training_task,  # noqa: F405
        "direct_distill:train": launch_training_task,  # noqa: F405
    }
    launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
