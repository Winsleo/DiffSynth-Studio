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

Run via ``accelerate launch -m a2v.train_a2v ...`` (see ``a2v/train.sh``).
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
from a2v.train_logging import A2VModelLogger, PeriodicSampler, a2v_launch_training_task  # noqa: E402


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


def _install_cosine_lr_schedule(args, dataset) -> None:
    """Replace the stock trainer's hardcoded ConstantLR with linear-warmup + cosine decay.

    The stock loop (diffsynth/diffusion/runner.py) builds ``ConstantLR(optimizer)`` and steps
    it every optimizer step; we monkeypatch that symbol to a ``LambdaLR`` so no diffsynth edit
    is needed. accelerate (>=1.x) advances a *prepared* scheduler ``num_processes`` steps per
    optimizer step, so over training the scheduler sees ``T = num_epochs * len(dataset)`` steps
    regardless of NPROC (len(dataset) already includes ``dataset_repeat``). LR ramps 0 -> peak
    (= ``--learning_rate``) across ``warmup`` steps, then cosine-decays peak -> ``lr_min_ratio *
    peak`` over the remainder. Assumes ``gradient_accumulation_steps == 1`` (our recipes).
    """
    import math
    import torch

    total_steps = max(1, args.num_epochs * len(dataset))
    warmup = args.lr_warmup_steps if args.lr_warmup_steps > 0 else max(1, round(0.03 * total_steps))
    warmup = min(warmup, total_steps)
    min_ratio = args.lr_min_ratio

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return step / warmup
        progress = min(1.0, (step - warmup) / max(1, total_steps - warmup))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_ratio + (1.0 - min_ratio) * cosine

    def constant_lr_factory(optimizer, *a, **k):
        return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    torch.optim.lr_scheduler.ConstantLR = constant_lr_factory
    print(f"[a2v] lr_schedule=cosine peak_lr={args.learning_rate} warmup={warmup} "
          f"total_steps={total_steps} min_ratio={min_ratio}")


def main() -> None:
    stock = _load_stock_train_module()
    parser = stock.wan_parser()
    parser.add_argument("--base_spec", default=None,
                        help="WanBaseSpec name (a2v.base_spec.REGISTRY). Fills model_paths/tokenizer/lora "
                             "defaults and drives the VACE seams. Explicit CLI flags still override.")
    parser.add_argument("--expert", default=None, choices=("high", "low"),
                        help="SEAM-5 dual-expert MoE (A14B): which expert this job trains. "
                             "Selects that expert's DiT via spec.model_paths(expert=...) and "
                             "should be paired with the matching --min/--max_timestep_boundary "
                             "band (see train.sh). Ignored for single-expert specs.")
    parser.add_argument("--lr_schedule", default="constant", choices=("constant", "cosine"),
                        help="LR schedule. 'constant' (default) = stock behavior. 'cosine' = "
                             "linear warmup then cosine decay to --lr_min_ratio * --learning_rate "
                             "(which becomes the PEAK lr).")
    parser.add_argument("--lr_warmup_steps", type=int, default=0,
                        help="Warmup steps for --lr_schedule cosine; 0 -> 3%% of total steps "
                             "(total = num_epochs * len(dataset)).")
    parser.add_argument("--lr_min_ratio", type=float, default=0.0,
                        help="Final lr as a fraction of peak for --lr_schedule cosine (0 -> to 0).")
    parser.add_argument("--cache_train", action="store_true",
                        help="Train from a pre-encoded cache (built with --task sft:data_process). "
                             "Loads ONLY the DiT (T5/VAE/CLIP outputs come from the cache, so those "
                             "encoders are not loaded -> big VRAM + time saving) and reads cached .pth "
                             "tensors via load_from_cache (dataset_base_path = cache dir).")
    # --- observability (a2v.train_logging): richer scalars + in-training sampling ---
    parser.add_argument("--log_every", type=int, default=10,
                        help="Log lr/grad_norm/throughput/gpu_mem every N steps (loss is logged every step).")
    parser.add_argument("--log_grad_norm", type=int, default=1,
                        help="1 = log grad norm on logging steps (computed before zero_grad); 0 = skip.")
    parser.add_argument("--sample_every", type=int, default=0,
                        help="Periodic in-training generation cadence in steps (0 = off). Reuses the live "
                             "pipe; auto-skipped for cache-train / dual-expert / cpu-offload.")
    parser.add_argument("--sample_rows", type=str, default="0",
                        help="Comma-separated dataset row indices to sample (e.g. '0,1').")
    parser.add_argument("--sample_controls", type=str, default="real",
                        help="Comma-separated causal controls to sample: real|none|shuffle.")
    parser.add_argument("--sample_num_frames", type=int, default=0,
                        help="Frames per sample clip (0 -> use --num_frames).")
    parser.add_argument("--sample_steps", type=int, default=0,
                        help="Denoising steps per sample (0 -> pipeline default, 50). Lower = faster.")
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
            # SEAM-5: --expert selects one MoE expert's DiT (single-DiT training job).
            args.model_paths = json.dumps(spec.model_paths(expert=args.expert))
        if args.tokenizer_path is None:
            args.tokenizer_path = spec.abs_tokenizer_path()
        if args.lora_base_model is None and args.trainable_models is None:
            args.lora_base_model = spec.lora_base_model
            args.lora_target_modules = spec.lora_target_modules
        if args.remove_prefix_in_ckpt is None:
            args.remove_prefix_in_ckpt = spec.vace_remove_prefix
    if args.remove_prefix_in_ckpt is None:
        args.remove_prefix_in_ckpt = "pipe.dit."  # stock default (non-spec usage)

    # --- cache-train: the big encoders (T5/CLIP) are NOT needed (their outputs are cached),
    # but keep the DiT AND the small VAE. The VAE forward never runs (its encode unit is
    # pruned for :train and input_latents come from the cache), yet the kept
    # WanVideoUnit_NoiseInitializer reads the VAE config (z_dim / upsampling_factor) to shape
    # the training noise -- so pipe.vae must exist or it crashes (NoneType.model). ---
    if args.cache_train:
        if spec is not None:
            from a2v.base_spec import _abs
            # [0] is the DiT entry (str, or list of shards). Add the VAE; drop T5/CLIP.
            dit_entry = spec.model_paths(expert=args.expert)[0]
            args.model_paths = json.dumps([dit_entry, _abs(spec.vae_path)])
        # metadata_path=None -> UnifiedDataset.load_from_cache (recursively finds *.pth)
        args.dataset_metadata_path = None

    height_division_factor = spec.vae_spatial_factor * spec.patch_size[1] if spec is not None else 16
    width_division_factor = spec.vae_spatial_factor * spec.patch_size[2] if spec is not None else 16
    time_division_factor = spec.vae_temporal_factor if spec is not None else 4

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
            height_division_factor=height_division_factor,
            width_division_factor=width_division_factor,
            num_frames=args.num_frames,
            time_division_factor=time_division_factor if not args.framewise_decoding else 1,
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
        if args.cache_train:
            # Cache mode: vace_context is already in the cache (encoded with the right mask_pq
            # at data_process time), and the encoder-side WanVideoUnit_VACE is pruned from
            # pipe.units for :train. So we only need the vace MODEL (DiT blocks, the trainable
            # part) — already built by the in-constructor ensure_vace hook; call it again
            # (idempotent) for clarity. Do NOT install_vace_unit (no encoder unit to swap).
            ensure_vace(model.pipe, spec)
        else:
            provision_a2v(model.pipe, spec)

    # A2VModelLogger == stock ModelLogger checkpoint behavior + richer scalar/video logging.
    model_logger = A2VModelLogger(
        args.output_path,
        remove_prefix_in_ckpt=args.remove_prefix_in_ckpt,
        enable_tensorboard_log=args.enable_tensorboard_log,
        enable_swanlab_log=args.enable_swanlab_log,
        swanlab_project=args.swanlab_project,
        enable_wandb_log=args.enable_wandb_log,
        wandb_project=args.wandb_project,
    )
    # Training tasks use the instrumented a2v loop; data_process stays stock (no step loop).
    launcher_map = {
        "sft:data_process": launch_data_process_task,  # noqa: F405
        "direct_distill:data_process": launch_data_process_task,  # noqa: F405
        "sft": a2v_launch_training_task,
        "sft:train": a2v_launch_training_task,
        "direct_distill": a2v_launch_training_task,
        "direct_distill:train": a2v_launch_training_task,
    }
    # Opt-in warmup+cosine LR (default constant = unchanged stock behavior). Training tasks only.
    if args.lr_schedule == "cosine" and not args.task.endswith(":data_process"):
        _install_cosine_lr_schedule(args, dataset)

    is_training = not args.task.endswith(":data_process")
    if is_training:
        # Periodic in-training sampling: only when requested, with a raw PNG dataset (cache-train
        # prunes the encoders pipe(...) needs). PeriodicSampler self-gates dual-expert / cpu-offload.
        sampler = None
        if args.sample_every > 0 and not args.cache_train and spec is not None:
            sampler = PeriodicSampler(
                spec=spec,
                dataset_path=args.dataset_base_path,
                every=args.sample_every,
                rows=[int(r) for r in args.sample_rows.split(",") if r.strip() != ""],
                controls=[c for c in args.sample_controls.split(",") if c.strip() != ""],
                height=args.height,
                width=args.width,
                num_frames=args.sample_num_frames if args.sample_num_frames > 0 else args.num_frames,
                steps=args.sample_steps,
                seed=getattr(args, "seed", 0) or 0,
                model_logger=model_logger,
                enable_model_cpu_offload=args.enable_model_cpu_offload,
            )
        elif args.sample_every > 0 and args.cache_train:
            print("[a2v][sample] disabled: cache-train prunes the encoders pipe(...) needs.")
        launcher_map[args.task](accelerator, dataset, model, model_logger, args=args,
                                sampler=sampler, log_every=args.log_every,
                                log_grad_norm=bool(args.log_grad_norm))
    else:
        launcher_map[args.task](accelerator, dataset, model, model_logger, args=args)


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    main()
