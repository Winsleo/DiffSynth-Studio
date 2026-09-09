#!/usr/bin/env python3
"""A2V training observability — richer logging + in-training sampling.

Stock DiffSynth logs a single scalar (``loss``, main process only) via
``ModelLogger`` (``diffsynth/diffusion/logger.py``) and its training loop
``launch_training_task`` (``diffsynth/diffusion/runner.py``) exposes nothing
else. This module adds, WITHOUT editing ``diffsynth/``:

  * ``A2V*Logger``        — the three stock backends + a ``log_video`` method.
  * ``A2VModelLogger``    — logs lr / grad-norm / throughput / GPU mem alongside
                            loss; identical checkpoint-saving semantics as stock.
  * ``a2v_launch_training_task`` — a faithful mirror of the stock loop (reusing
                            its setup helpers) with the instrumentation inlined,
                            since grad-norm must be read *before* ``zero_grad``.
  * ``PeriodicSampler``   — best-effort in-training generation: every N steps it
                            reuses the live pipe to render a short clip and logs
                            it. Auto-skips the modes where this can't work
                            (cache-train prunes the encoders, dual-expert needs
                            both experts, cpu-offload).

The stock loop this mirrors lives at ``diffsynth/diffusion/runner.py:19-78``; if
that changes upstream, re-sync ``a2v_launch_training_task``.
"""

from __future__ import annotations

import os
import time

import torch
from accelerate import Accelerator

from diffsynth.core import OffloadTrainingManager
from diffsynth.diffusion.logger import (
    ModelLogger,
    TensorBoardLogger,
    SwanLabLogger,
    WandbLogger,
)
from diffsynth.diffusion.runner import (
    get_optimizer_class,
    initialize_deepspeed_gradient_checkpointing,
)
from diffsynth.diffusion.training_module import DiffusionTrainingModule
from diffsynth.utils.data import save_video


# --------------------------------------------------------------------------- #
# media-capable logger backends (stock backends only expose .log(key,value,step))
# --------------------------------------------------------------------------- #
def _frames_to_video_tensor(frames) -> "torch.Tensor":
    """list[PIL.Image] -> float tensor [1, T, C, H, W] in [0,1] for SummaryWriter.add_video."""
    import numpy as np

    arr = np.stack([np.asarray(f.convert("RGB")) for f in frames])  # [T,H,W,C] uint8
    t = torch.from_numpy(arr).float().div_(255.0).permute(0, 3, 1, 2)  # [T,C,H,W]
    return t.unsqueeze(0)  # [1,T,C,H,W]


class A2VTensorBoardLogger(TensorBoardLogger):
    def log_video(self, tag, frames, step, fps=15, video_path=None):
        try:
            self.writer.add_video(tag, _frames_to_video_tensor(frames), global_step=step, fps=fps)
        except Exception:
            # add_video relies on moviepy/numpy internals that break on some numpy>=2 setups
            # ("reshape got an unexpected keyword argument 'newshape'"). Fall back to an image
            # strip of up to 8 evenly-spaced frames (the mp4 is also saved under <OUT>/samples/).
            import numpy as np

            n = len(frames)
            idx = sorted(set(int(i) for i in np.linspace(0, n - 1, min(8, n))))
            grid = _frames_to_video_tensor([frames[i] for i in idx])[0]  # [k,C,H,W]
            self.writer.add_images(tag, grid, global_step=step)


class A2VWandbLogger(WandbLogger):
    def log_video(self, tag, frames, step, fps=15, video_path=None):
        if video_path is None:
            return
        self.wandb.log({tag: self.wandb.Video(video_path, fps=fps, format="mp4")}, step=step)


class A2VSwanLabLogger(SwanLabLogger):
    def log_video(self, tag, frames, step, fps=15, video_path=None):
        # swanlab may not ship a Video media type in every version -> best effort.
        try:
            video = self.swanlab.Video(video_path, caption=tag) if video_path else None
        except Exception:
            return
        if video is not None:
            self.swanlab.log({tag: video}, step=step)


# --------------------------------------------------------------------------- #
# logger: same checkpoint behavior as stock ModelLogger, richer scalar logging
# --------------------------------------------------------------------------- #
class A2VModelLogger(ModelLogger):
    """Drop-in ModelLogger that also logs a ``metrics`` dict and videos.

    Checkpoint saving (num_steps counter + ``% save_steps`` -> ``step-N.safetensors``)
    is byte-for-byte the stock behavior (``diffsynth/diffusion/logger.py:78-107``);
    only the logging side is extended.
    """

    def init_loggers(self):
        if self.enable_tensorboard_log:
            self.loggers.append(A2VTensorBoardLogger(os.path.join(self.output_path, "tensorboard_log")))
        if self.enable_swanlab_log:
            self.loggers.append(A2VSwanLabLogger(project_name=self.swanlab_project,
                                                 log_dir=os.path.join(self.output_path, "swanlab_log")))
        if self.enable_wandb_log:
            self.loggers.append(A2VWandbLogger(project_name=self.wandb_project,
                                               log_dir=os.path.join(self.output_path, "wandb_log")))
        self.loggers_initialized = True

    def on_step_end(self, accelerator: Accelerator, model: torch.nn.Module, save_steps=None,
                    loss=None, metrics=None, **kwargs):
        self.num_steps += 1
        if accelerator.is_main_process:
            if not self.loggers_initialized:
                self.init_loggers()
            for logger in self.loggers:
                if loss is not None:
                    logger.log("loss", loss, self.num_steps)
                if metrics:
                    for key, value in metrics.items():
                        if value is not None:
                            logger.log(key, value, self.num_steps)
        if save_steps is not None and self.num_steps % save_steps == 0:
            self.save_model(accelerator, model, f"step-{self.num_steps}.safetensors")

    def log_video(self, tag, frames, step, fps=15, video_path=None):
        if not self.loggers_initialized:
            self.init_loggers()
        for logger in self.loggers:
            if hasattr(logger, "log_video"):
                try:
                    logger.log_video(tag, frames, step, fps=fps, video_path=video_path)
                except Exception as exc:  # never let a viz failure touch training
                    print(f"[a2v][sample] log_video({tag}) failed on {type(logger).__name__}: {exc}")


# --------------------------------------------------------------------------- #
# periodic in-training sampling (best effort; reuses the live pipe)
# --------------------------------------------------------------------------- #
class PeriodicSampler:
    """Every ``every`` steps, render a short clip from the LIVE training pipe and log it.

    Reuses ``a2v.infer_a2v.generate_one`` against the current weights (no model reload).
    Gated off for the modes where in-process generation can't work faithfully:
      * cache-train  -> the T5/VAE-encode units are pruned, so pipe(...) can't encode.
      * dual-expert  -> each job holds only one expert; a sample needs both + the switch.
      * cpu-offload  -> the live pipe layout is not generation-ready here.
    Any failure is caught and logged; training is never interrupted.
    """

    def __init__(self, spec, dataset_path, every, rows, controls, height, width,
                 num_frames, steps, seed, model_logger, fps=15, enable_model_cpu_offload=False):
        self.spec = spec
        self.dataset_path = dataset_path
        self.every = every
        self.rows = rows
        self.controls = controls
        self.height = height
        self.width = width
        self.num_frames = num_frames
        self.steps = steps           # 0 -> pipe default num_inference_steps
        self.seed = seed
        self.model_logger = model_logger
        self.fps = fps
        self.enable_model_cpu_offload = enable_model_cpu_offload
        self._skip_reason = None     # set + warned once

    def _disabled_reason(self):
        if self.enable_model_cpu_offload:
            return "enable_model_cpu_offload (live pipe not generation-ready)"
        if getattr(self.spec, "experts", None):
            return f"dual-expert spec ({self.spec.name}: needs both experts + switch)"
        return None

    def maybe_sample(self, step, model, accelerator: Accelerator):
        if not accelerator.is_main_process:
            return
        reason = self._disabled_reason()
        if reason is not None:
            if self._skip_reason != reason:
                self._skip_reason = reason
                print(f"[a2v][sample] disabled: {reason}")
            return
        try:
            self._sample(step, model, accelerator)
        except Exception as exc:
            print(f"[a2v][sample] step {step} failed (training continues): {exc}")

    def _sample(self, step, model, accelerator: Accelerator):
        from pathlib import Path

        from a2v.infer_a2v import generate_one, load_row

        pipe = accelerator.unwrap_model(model).pipe
        dataset = Path(self.dataset_path).resolve()
        out_dir = Path(self.model_logger.output_path) / "samples"
        out_dir.mkdir(parents=True, exist_ok=True)
        steps_kw = self.steps if self.steps and self.steps > 0 else None

        was_training = getattr(model, "training", True)
        model.eval()
        try:
            with torch.no_grad():
                for row_idx in self.rows:
                    row = load_row(dataset, row_idx)
                    for control in self.controls:
                        video = generate_one(
                            pipe=pipe, spec=self.spec, row=row, dataset=dataset,
                            control=control, height=self.height, width=self.width,
                            num_frames=self.num_frames, seed=self.seed,
                            num_inference_steps=steps_kw,
                        )
                        mp4 = out_dir / f"step-{step}_row{row_idx}_{control}.mp4"
                        save_video(video, str(mp4), fps=self.fps, quality=5)
                        tag = f"sample/row{row_idx}_{control}"
                        self.model_logger.log_video(tag, video, step, fps=self.fps, video_path=str(mp4))
                        print(f"[a2v][sample] step {step} -> {mp4}")
        finally:
            if was_training:
                model.train()
            # CRITICAL: pipe(...) called scheduler.set_timesteps(steps, training=False), which
            # overwrote the training timesteps/weights AND set scheduler.training=False. Without
            # this, the next training step's InputVideoEmbedder unit takes the inference branch
            # and never emits `input_latents` -> KeyError. Re-arm the exact training scheduler
            # (mirrors diffsynth/diffusion/training_module.py:275).
            try:
                pipe.scheduler.set_timesteps(1000, training=True)
            except Exception as exc:
                print(f"[a2v][sample] WARNING: failed to restore training scheduler: {exc}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# instrumented training loop (mirror of diffsynth.diffusion.runner.launch_training_task)
# --------------------------------------------------------------------------- #
def _grad_norm(model) -> float:
    total = 0.0
    for p in model.parameters():
        if p.requires_grad and p.grad is not None:
            total += float(p.grad.detach().float().pow(2).sum().item())
    return total ** 0.5


def a2v_launch_training_task(
    accelerator: Accelerator,
    dataset: torch.utils.data.Dataset,
    model: DiffusionTrainingModule,
    model_logger: A2VModelLogger,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-2,
    num_workers: int = 1,
    save_steps: int = None,
    num_epochs: int = 1,
    enable_model_cpu_offload: bool = False,
    enable_optimizer_cpu_offload: bool = False,
    cpu_offload_split_threshold: int = None,
    customized_optimizer: str = None,
    args=None,
    sampler: PeriodicSampler = None,
    log_every: int = 10,
    log_grad_norm: bool = True,
    **kwargs,
):
    """Stock training loop + lr/grad-norm/throughput/mem logging + periodic sampling.

    Setup/optimizer/scheduler/prepare are kept identical to the stock launcher (the
    ``ConstantLR`` line is preserved so the cosine-LR monkeypatch in train_a2v still
    applies). Only the inner step body adds instrumentation. Assumes
    ``gradient_accumulation_steps == 1`` (all a2v recipes).
    """
    if args is not None:
        learning_rate = args.learning_rate
        weight_decay = args.weight_decay
        num_workers = args.dataset_num_workers
        save_steps = args.save_steps
        num_epochs = args.num_epochs
        enable_model_cpu_offload = args.enable_model_cpu_offload
        enable_optimizer_cpu_offload = args.enable_optimizer_cpu_offload
        cpu_offload_split_threshold = args.cpu_offload_split_threshold
        customized_optimizer = args.customized_optimizer

    optimizer_class = get_optimizer_class(customized_optimizer)
    optimizer = optimizer_class(model.trainable_modules(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)  # may be monkeypatched -> LambdaLR (cosine)
    dataloader = torch.utils.data.DataLoader(dataset, shuffle=True, collate_fn=lambda x: x[0], num_workers=num_workers)

    if enable_model_cpu_offload:
        optimizer, dataloader, scheduler = accelerator.prepare(optimizer, dataloader, scheduler)
        model.pipe.device = accelerator.device
        offload_manager = OffloadTrainingManager(model, accelerator.device, enable_optimizer_cpu_offload, cpu_offload_split_threshold)
    else:
        model.to(device=accelerator.device)
        model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    initialize_deepspeed_gradient_checkpointing(accelerator)

    step = 0
    last_t = time.perf_counter()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    for epoch_id in range(num_epochs):
        print(f"[a2v_debug] epoch {epoch_id} begin, dataloader has {len(dataloader)} iters", flush=True)
        for data in dataloader:
            with accelerator.accumulate(model):
                if dataset.load_from_cache:
                    loss = model({}, inputs=data)
                else:
                    loss = model(data)
                accelerator.backward(loss)
                if enable_model_cpu_offload:
                    offload_manager.after_backward()

                step += 1
                do_log = (step % log_every == 0)
                if step <= 3:
                    print(f"[a2v_debug] step {step} done, loss={loss.item():.4f}", flush=True)

                # grad norm must be read before zero_grad; gate by log cadence (cost on 14B)
                grad_norm = _grad_norm(model) if (do_log and log_grad_norm) else None
                lr = optimizer.param_groups[0]["lr"]

                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

                # global (DDP-averaged) loss: collective -> all processes must call gather
                loss_g = accelerator.gather(loss.detach().float()).mean().item()

                metrics = None
                if do_log:
                    now = time.perf_counter()
                    metrics = {
                        "lr": lr,
                        "grad_norm": grad_norm,
                        "throughput_it_s": log_every / max(1e-9, now - last_t),
                    }
                    if torch.cuda.is_available():
                        metrics["gpu_mem_gb"] = torch.cuda.max_memory_allocated() / (1024 ** 3)
                        torch.cuda.reset_peak_memory_stats()
                    last_t = now

                model_logger.on_step_end(accelerator, model, save_steps, loss=loss_g, metrics=metrics)

                if sampler is not None and step % sampler.every == 0:
                    sampler.maybe_sample(step, model, accelerator)

        if save_steps is None:
            model_logger.on_epoch_end(accelerator, model, epoch_id)

    model_logger.on_training_end(accelerator, model, save_steps)
