# SPDX-License-Identifier: Apache-2.0
"""Megakernel prefill attention op wrapper with tensor-only ABI."""

from __future__ import annotations

import torch

from vllm.v1.attention.ops.triton_unified_attention import unified_attention


def megakernel_prefill_attention(
    *,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    query_start_loc: torch.Tensor,
    max_query_len: int,
    scale: float,
    alibi_slopes: torch.Tensor | None,
    sliding_window: tuple[int, int],
    logits_soft_cap: float,
) -> torch.Tensor:
    unified_attention(
        q=query,
        k=key,
        v=value,
        out=output,
        cu_seqlens_q=query_start_loc,
        max_seqlen_q=max_query_len,
        cu_seqlens_k=query_start_loc,
        max_seqlen_k=max_query_len,
        softmax_scale=scale,
        causal=True,
        alibi_slopes=alibi_slopes,
        window_size=sliding_window,
        softcap=logits_soft_cap,
    )
    return output
