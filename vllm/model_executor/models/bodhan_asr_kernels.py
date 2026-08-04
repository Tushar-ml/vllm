"""Fused Bodhan / CohereASR encoder helper kernels (Triton).

Decoder self/cross attn stays FA3 — these only cover encode-path hotspots:
pack padded encoder states into a contiguous buffer without per-sample
``.item()`` GPU syncs in a Python loop.
"""

from __future__ import annotations

from typing import List, Sequence

import torch

try:
    import triton
    import triton.language as tl

    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _pack_encoder_outputs_kernel(
        src_ptr,  # [B, T, D]
        dst_ptr,  # [total, D]
        offsets_ptr,  # [B] int32 — start offset into dst for each sample
        lengths_ptr,  # [B] int32
        T,
        D,
        stride_b,
        stride_t,
        stride_d,
        BLOCK_D: tl.constexpr,
    ):
        pid_b = tl.program_id(0)
        pid_t = tl.program_id(1)

        length = tl.load(lengths_ptr + pid_b)
        if pid_t >= length:
            return

        out_row = tl.load(offsets_ptr + pid_b) + pid_t
        src_row = src_ptr + pid_b * stride_b + pid_t * stride_t
        dst_row = dst_ptr + out_row * D

        offs = tl.arange(0, BLOCK_D)
        for d0 in range(0, D, BLOCK_D):
            idx = d0 + offs
            mask = idx < D
            vals = tl.load(src_row + idx * stride_d, mask=mask, other=0)
            tl.store(dst_row + idx, vals, mask=mask)


def pack_encoder_outputs(
    enc_states: torch.Tensor,
    encoded_len: torch.Tensor,
    out_dtype: torch.dtype | None = None,
) -> List[torch.Tensor]:
    """Pack padded ``[B, T, D]`` encoder states into a list of ``[T_i, D]``.

    Performs a single device→host length copy (via ``.tolist()``), then either
    a Triton fused copy+cast or a torch fallback — never per-sample ``.item()``.
    """
    if enc_states.ndim != 3:
        raise ValueError(f"expected [B,T,D], got {tuple(enc_states.shape)}")
    b, t, d = enc_states.shape
    if b == 0:
        return []

    if encoded_len.device.type != "cpu":
        # One sync for the whole batch.
        lengths_list: list[int] = encoded_len.to(
            dtype=torch.int64, device="cpu"
        ).tolist()
    else:
        lengths_list = encoded_len.to(dtype=torch.int64).tolist()
    lengths_list = [max(0, min(int(x), t)) for x in lengths_list]
    if len(lengths_list) != b:
        raise ValueError(
            f"encoded_len batch {len(lengths_list)} != enc_states batch {b}"
        )

    target_dtype = out_dtype or enc_states.dtype
    src = enc_states if enc_states.dtype == target_dtype else enc_states.to(target_dtype)

    if (
        _HAS_TRITON
        and src.is_cuda
        and b * t > 0
        and sum(lengths_list) > 0
        and os_triton_enabled()
    ):
        return _pack_triton(src, lengths_list, d)

    return [src[i, : lengths_list[i]].contiguous() for i in range(b)]


def os_triton_enabled() -> bool:
    import os

    # Default on (matches serve_bodhan_vllm.sh). Set BODHAN_TRITON_PACK=0 for
    # torch slice pack if debugging short-clip nondeterminism.
    return os.environ.get("BODHAN_TRITON_PACK", "1") != "0"


def _pack_triton(
    src: torch.Tensor, lengths_list: Sequence[int], d: int
) -> List[torch.Tensor]:
    b = src.shape[0]
    offsets = [0]
    for L in lengths_list[:-1]:
        offsets.append(offsets[-1] + L)
    total = offsets[-1] + lengths_list[-1] if lengths_list else 0
    if total == 0:
        return [src.new_empty((0, d)) for _ in range(b)]

    dst = torch.empty((total, d), device=src.device, dtype=src.dtype)
    offsets_t = torch.tensor(offsets, device=src.device, dtype=torch.int32)
    lengths_t = torch.tensor(lengths_list, device=src.device, dtype=torch.int32)

    BLOCK_D = 128 if d >= 128 else 64
    # grid: (B, max_T)
    max_t = max(lengths_list) if lengths_list else 1
    _pack_encoder_outputs_kernel[(b, max_t)](
        src,
        dst,
        offsets_t,
        lengths_t,
        src.shape[1],
        d,
        src.stride(0),
        src.stride(1),
        src.stride(2),
        BLOCK_D=BLOCK_D,
    )
    outs: List[torch.Tensor] = []
    for i, L in enumerate(lengths_list):
        start = offsets[i]
        outs.append(dst[start : start + L])
    return outs


def fused_cross_kv_proj(
    encoder_hidden_states: torch.Tensor,
    weights: torch.Tensor,
    biases: torch.Tensor | None,
) -> torch.Tensor:
    """Fused per-layer cross-attn kv_proj: ``[T,D] x [L,O,D] -> [L,T,O]``.

    Replaces 24 separate GEMMs on the first decoder step when encoder states
    are fresh. Optional; enable with ``BODHAN_FUSED_CROSS_KV=1``.
    """
    # enc: [T, D], weights: [L, O, D]
    out = torch.einsum("td,lod->lto", encoder_hidden_states, weights)
    if biases is not None:
        out = out + biases[:, None, :]
    return out


def warmup_pack_shapes(
    shapes: Sequence[tuple[int, int, int]] | None = None,
    device: str = "cuda",
    dtype: torch.dtype = torch.bfloat16,
) -> None:
    """JIT-warm Triton pack for common Hindi mel→enc shapes."""
    if not _HAS_TRITON or not torch.cuda.is_available():
        return
    if shapes is None:
        # mel_T ∈ {400…1000} → enc_T ≈ mel/8; B ∈ {1,8,32,64}
        enc_ts = [50, 63, 75, 88, 125]
        batches = [1, 8, 32, 64]
        shapes = [(b, t, 1024) for b in batches for t in enc_ts]
    for b, t, d in shapes:
        states = torch.randn(b, t, d, device=device, dtype=dtype)
        lengths = torch.full((b,), t, device=device, dtype=torch.int64)
        _ = pack_encoder_outputs(states, lengths, out_dtype=dtype)
    torch.cuda.synchronize()
