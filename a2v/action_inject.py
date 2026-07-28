"""Scheme A: lightweight numeric-action AdaLN modulation, injected into the
trainable VACE branch WITHOUT editing diffsynth core.

Motivation (see A2V README critique): the trajectory-map control is a lossy,
view-dependent visual proxy for a precise 16-D action. This module adds a
*second, lossless* action channel — a small ``ActionEmbedder`` that turns each
per-latent-frame action chunk into the 6 AdaLN signals
(shift/scale/gate × msa/mlp) and ADDS them onto the VACE blocks' existing
timestep modulation. It is a zero-init gated bypass: at init (and whenever the
action is absent) its contribution is exactly zero, so the pre-existing A2V
behavior is preserved bit-for-bit (parity gate stays green).

No-core-edit strategy (the A2V invariant — never touch ``diffsynth/``):
  * The VACE block (``VaceWanAttentionBlock``) subclasses ``DiTBlock`` and reuses
    its AdaLN. We ``types.MethodType``-swap the forwards of the VACE model and its
    blocks (established precedent: the USP forward swap in ``wan_video.py``) with
    action-aware replicas defined here.
  * ``pipe.model_fn`` is a plain swappable attribute; we wrap it so the ``action``
    kwarg is stashed on the VACE module before delegating to the stock model_fn.

The block/model forward replicas below are VERBATIM copies of the stock bodies
(``wan_video_vace.py:5-74``, ``wan_video_dit.py:229-245``) with the SOLE addition
of the action-modulation term — same maintenance contract as ``vace_unit.py``.
"""

from __future__ import annotations

import os
import types

import torch
from torch import nn

from diffsynth.core.gradient import gradient_checkpoint_forward
from diffsynth.models.wan_video_dit import modulate

_DEBUG = os.environ.get("A2V_ACTION_DEBUG", "0") == "1"


class ActionEmbedder(nn.Module):
    """Per-frame action chunk -> 6*dim AdaLN modulation (zero-init gated).

    ``proj_out`` (weight+bias) is zero-initialized so the initial modulation is
    exactly zero -> the VACE branch behaves identically to the no-action baseline
    until trained. ``mask_token`` (zero-init) provides the null-action embedding
    used when ``action is None`` (negative control / CFG-nega).
    """

    def __init__(self, chunk_input_size: int, model_dim: int):
        super().__init__()
        self.proj_in = nn.Linear(chunk_input_size, model_dim)
        self.act = nn.SiLU()
        self.proj_out = nn.Linear(model_dim, 6 * model_dim)
        self.mask_token = nn.Parameter(torch.zeros(model_dim))
        nn.init.zeros_(self.proj_out.weight)
        nn.init.zeros_(self.proj_out.bias)

    def forward(self, action_chunks: torch.Tensor) -> torch.Tensor:
        # action_chunks: [B, n_chunks, chunk_input_size] -> [B, n_chunks, 6*dim]
        return self.proj_out(self.act(self.proj_in(action_chunks)))

    def null_mod(self, batch_size: int, n_frames: int, device, dtype) -> torch.Tensor:
        null_h = self.mask_token.to(device=device, dtype=dtype).view(1, 1, -1)
        return self.proj_out(null_h.expand(batch_size, n_frames, -1))  # zero at init


def _group_actions(action: torch.Tensor, chunk_size: int, action_dim: int) -> torch.Tensor:
    """[B, T, A] -> [B, n_chunks, chunk_size*A] via VAE-style temporal grouping.

    n_chunks = (T - 1) // chunk_size. Chunk i concatenates raw frames
    [chunk_size*i : chunk_size*i+chunk_size], mirroring worldmodel-wan
    (``[B, 4*(T_latent-1), A] -> [B, T_latent-1, 4A]``); it generates the
    (leading + i)-th latent frame, where the leading latent frames (reference +
    conditioning frame 0) get a zero modulation.
    """
    if action.dim() == 2:
        action = action.unsqueeze(0)
    b, t, a = action.shape
    if a != action_dim:
        raise ValueError(f"action last dim {a} != expected action_dim {action_dim}")
    n_chunks = (t - 1) // chunk_size
    if n_chunks < 1:
        raise ValueError(f"too few action frames ({t}) for chunk_size {chunk_size}")
    grouped = action[:, : n_chunks * chunk_size, :].reshape(b, n_chunks, chunk_size * a)
    return grouped


def action_vace_block_forward(self, c, x, context, t_mod, freqs, action_mod=None):
    """Replica of ``VaceWanAttentionBlock.forward`` (wan_video_vace.py:13-24) with
    the inlined ``DiTBlock.forward`` body (wan_video_dit.py:229-245); the SOLE
    change is adding ``action_mod`` onto the 6 AdaLN chunks.

    ``action_mod`` is passed as an explicit argument (NOT a module attribute) so it
    flows through ``gradient_checkpoint_forward`` as a checkpointed input — the
    recompute in backward then matches the forward (a side-channel attribute that
    the model clears would recompute to a different shape and raise CheckpointError).
    """
    # --- VaceWanAttentionBlock entry (verbatim) ---
    if self.block_id == 0:
        c = self.before_proj(c) + x
        all_c = []
    else:
        all_c = list(torch.unbind(c))
        c = all_c.pop(-1)

    # --- DiTBlock.forward on c (verbatim) + action modulation ---
    has_seq = len(t_mod.shape) == 4
    chunk_dim = 2 if has_seq else 1
    shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
        self.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=chunk_dim)
    if has_seq:
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = (
            shift_msa.squeeze(2), scale_msa.squeeze(2), gate_msa.squeeze(2),
            shift_mlp.squeeze(2), scale_mlp.squeeze(2), gate_mlp.squeeze(2),
        )
    if action_mod is not None:
        a0, a1, a2, a3, a4, a5 = action_mod.to(dtype=shift_msa.dtype).chunk(6, dim=-1)
        shift_msa = shift_msa + a0
        scale_msa = scale_msa + a1
        gate_msa = gate_msa + a2
        shift_mlp = shift_mlp + a3
        scale_mlp = scale_mlp + a4
        gate_mlp = gate_mlp + a5
    input_x = modulate(self.norm1(c), shift_msa, scale_msa)
    c = self.gate(c, gate_msa, self.self_attn(input_x, freqs))
    c = c + self.cross_attn(self.norm3(c), context)
    input_x = modulate(self.norm2(c), shift_mlp, scale_mlp)
    c = self.gate(c, gate_mlp, self.ffn(input_x))

    # --- VaceWanAttentionBlock exit (verbatim) ---
    c_skip = self.after_proj(c)
    all_c += [c_skip, c]
    c = torch.stack(all_c)
    return c


def action_vace_forward(
    self, x, vace_context, context, t_mod, freqs,
    use_gradient_checkpointing: bool = False,
    use_gradient_checkpointing_offload: bool = False,
):
    """Replica of ``VaceWanModel.forward`` (wan_video_vace.py:53-74) that also
    computes the per-token action modulation from ``self._a2v_action`` and writes
    it onto each block as ``_a2v_action_mod`` before running the block loop.

    Alignment (the key correctness point): the VACE conv output has time dim
    ``f'`` (= reference frames + video latent frames). Action chunks map to the
    LAST ``n_chunks`` latent frames; the leading ``f' - n_chunks`` frames
    (reference + conditioning frame 0) receive a zero modulation.
    """
    # verbatim: patch-embed vace_context, capture the latent grid (f', h', w')
    c_grids = [self.vace_patch_embedding(u.unsqueeze(0)) for u in vace_context]
    f_lat, h_lat, w_lat = c_grids[0].shape[2], c_grids[0].shape[3], c_grids[0].shape[4]
    c = [u.flatten(2).transpose(1, 2) for u in c_grids]
    c = torch.cat([
        torch.cat([u, u.new_zeros(1, x.shape[1] - u.size(1), u.size(2))], dim=1) for u in c
    ])

    # --- action modulation (the only addition) ---
    B = c.shape[0]
    dim = self.vace_blocks[0].dim
    dev, dt = c.device, c.dtype
    action = getattr(self, "_a2v_action", None)
    tokens_per_frame = h_lat * w_lat
    if action is not None:
        grouped = _group_actions(
            action.to(device=dev, dtype=dt), self._a2v_chunk_size, self._a2v_action_dim
        )  # [B, n_chunks, chunk*A]
        n_chunks = grouped.shape[1]
        embed = self.action_embedder(grouped)  # [B, n_chunks, 6*dim]
        n_lead = f_lat - n_chunks
        if n_lead < 0:
            # more action chunks than latent frames -> keep the last f_lat chunks
            embed = embed[:, -f_lat:, :]
            n_lead = 0
        lead = torch.zeros(B, n_lead, 6 * dim, device=dev, dtype=embed.dtype)
        per_frame = torch.cat([lead, embed], dim=1)  # [B, f', 6*dim]
        if _DEBUG and not getattr(self, "_a2v_dbg_printed", False):
            print(f"[a2v-action] f'={f_lat} h'={h_lat} w'={w_lat} n_chunks={n_chunks} "
                  f"n_lead={n_lead} tokens/frame={tokens_per_frame} x_seq={x.shape[1]}")
            self._a2v_dbg_printed = True
    else:
        per_frame = self.action_embedder.null_mod(B, f_lat, dev, dt)  # zero at init

    # per-frame -> per-token (frame-major, matching c's flatten(2).transpose layout)
    action_mod = per_frame.unsqueeze(2).expand(B, f_lat, tokens_per_frame, 6 * dim)
    action_mod = action_mod.reshape(B, f_lat * tokens_per_frame, 6 * dim)
    # pad/truncate to c's (possibly zero-padded) sequence length
    L = c.shape[1]
    if action_mod.shape[1] < L:
        pad = torch.zeros(B, L - action_mod.shape[1], 6 * dim, device=dev, dtype=action_mod.dtype)
        action_mod = torch.cat([action_mod, pad], dim=1)
    elif action_mod.shape[1] > L:
        action_mod = action_mod[:, :L, :]

    for block in self.vace_blocks:
        c = gradient_checkpoint_forward(
            block, use_gradient_checkpointing, use_gradient_checkpointing_offload,
            c, x, context, t_mod, freqs, action_mod,
        )

    hints = torch.unbind(c)[:-1]
    return hints


def make_action_model_fn(orig_model_fn):
    """Wrap ``pipe.model_fn`` so the ``action`` kwarg is stashed on the VACE module
    (whichever expert is active) before delegating to the stock model_fn.

    Training routes ``action`` through ``inputs_shared`` -> it arrives as a kwarg
    here every step. Inference has no such route (the stock ``pipe.__call__`` does
    not forward it), so ``infer_a2v`` stashes ``vace._a2v_action`` directly before
    calling the pipe; therefore we ONLY overwrite the stash when ``action`` is
    actually present in kwargs, leaving an inference-time stash intact otherwise.
    """

    def wrapped(**kwargs):
        vace = kwargs.get("vace", None)
        if "action" in kwargs:
            action = kwargs.pop("action")
            if vace is not None and getattr(vace, "_a2v_installed", False):
                vace._a2v_action = action
        return orig_model_fn(**kwargs)

    wrapped._a2v_wrapped = True
    return wrapped


def install_action_injection(pipe, spec, action_dim: int = 16, chunk_size: int = 4):
    """Attach an ``ActionEmbedder`` to each VACE branch, swap the branch/block
    forwards, and wrap ``pipe.model_fn``. Idempotent."""
    vace_modules = [m for m in (getattr(pipe, "vace", None), getattr(pipe, "vace2", None)) if m is not None]
    if not vace_modules:
        raise RuntimeError("install_action_injection: pipe has no VACE branch to attach to.")

    for m in vace_modules:
        if getattr(m, "_a2v_installed", False):
            continue
        dim = m.vace_blocks[0].dim
        ref = next(m.parameters())
        embedder = ActionEmbedder(chunk_size * action_dim, dim).to(dtype=ref.dtype, device=ref.device)
        m.action_embedder = embedder
        m._a2v_action = None
        m._a2v_action_dim = action_dim
        m._a2v_chunk_size = chunk_size
        m.forward = types.MethodType(action_vace_forward, m)
        for block in m.vace_blocks:
            block.forward = types.MethodType(action_vace_block_forward, block)
        m._a2v_installed = True

    if not getattr(pipe.model_fn, "_a2v_wrapped", False):
        pipe.model_fn = make_action_model_fn(pipe.model_fn)
    return True


def stash_action(pipe, action):
    """Inference entry: stash the raw ``[T,16]`` (or ``[B,T,16]``) action on every
    installed VACE branch so the swapped forward picks it up. ``action=None`` ->
    null modulation. Returns the number of branches updated."""
    n = 0
    for m in (getattr(pipe, "vace", None), getattr(pipe, "vace2", None)):
        if m is not None and getattr(m, "_a2v_installed", False):
            m._a2v_action = action
            n += 1
    return n


__all__ = [
    "ActionEmbedder",
    "action_vace_block_forward",
    "action_vace_forward",
    "install_action_injection",
    "make_action_model_fn",
    "stash_action",
]
