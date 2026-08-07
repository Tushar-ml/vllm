"""Megakernel decode attention op wrapper.

This wrapper keeps a tensor-only signature so it is safe to use under
piecewise/full CUDA graph capture.
"""

from __future__ import annotations

import torch

from vllm.v1.attention.ops.triton_unified_attention import unified_attention


def megakernel_decode_attention(
    *,
    query: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    output: torch.Tensor,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    max_query_len: int,
    max_seq_len: int,
    block_table: torch.Tensor,
    slot_mapping: torch.Tensor,
    scale: float,
    alibi_slopes: torch.Tensor | None,
    sliding_window: tuple[int, int],
    logits_soft_cap: float,
    k_scale: torch.Tensor,
    v_scale: torch.Tensor,
    seq_threshold_3d: int,
    num_par_softmax_segments: int,
    softmax_segm_output: torch.Tensor,
    softmax_segm_max: torch.Tensor,
    softmax_segm_expsum: torch.Tensor,
    output_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    # slot_mapping is intentionally part of the ABI for paged-KV adapters,
    # even though unified_attention currently consumes block_table/seq_lens.
    del slot_mapping

    unified_attention(
        q=query,
        k=key_cache,
        v=value_cache,
        out=output,
        cu_seqlens_q=query_start_loc,
        max_seqlen_q=max_query_len,
        seqused_k=seq_lens,
        max_seqlen_k=max_seq_len,
        softmax_scale=scale,
        causal=True,
        alibi_slopes=alibi_slopes,
        window_size=sliding_window,
        block_table=block_table,
        softcap=logits_soft_cap,
        q_descale=None,
        k_descale=k_scale,
        v_descale=v_scale,
        seq_threshold_3D=seq_threshold_3d,
        num_par_softmax_segments=num_par_softmax_segments,
        softmax_segm_output=softmax_segm_output,
        softmax_segm_max=softmax_segm_max,
        softmax_segm_expsum=softmax_segm_expsum,
        output_scale=output_scale,
    )
    return output
