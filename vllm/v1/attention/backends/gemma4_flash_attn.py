# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Gemma4-specific FlashAttention backend."""

import torch

from vllm.v1.attention.backend import AttentionType
from .flash_attn import FlashAttentionBackend, FlashAttentionImpl


class Gemma4FlashAttentionBackend(FlashAttentionBackend):
    @classmethod
    def supports_mm_prefix(cls) -> bool:
        # Native MM-prefix masking is handled via hybrid fallback routing.
        return True

    @staticmethod
    def get_name() -> str:
        return "GEMMA4_FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> type["FlashAttentionImpl"]:
        return Gemma4FlashAttentionImpl


class Gemma4FlashAttentionImpl(FlashAttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type=None,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
        *,
        apply_v_norm: bool = True,
        v_norm_eps: float = 1e-6,
        **kwargs,
    ) -> None:
        super().__init__(
            num_heads=num_heads,
            head_size=head_size,
            scale=scale,
            num_kv_heads=num_kv_heads,
            alibi_slopes=alibi_slopes,
            sliding_window=sliding_window,
            kv_cache_dtype=kv_cache_dtype,
            logits_soft_cap=logits_soft_cap,
            attn_type=attn_type,
            kv_sharing_target_layer_name=kv_sharing_target_layer_name,
            sinks=sinks,
            **kwargs,
        )
        self.apply_v_norm = apply_v_norm
        self.v_norm_eps = v_norm_eps

    def _rms_norm_no_weight(self, x: torch.Tensor) -> torch.Tensor:
        inv_rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.v_norm_eps)
        return x * inv_rms

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return super().forward(
            layer=layer,
            query=query,
            key=key,
            value=value,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
            output=output,
            output_scale=output_scale,
            output_block_scale=output_block_scale,
        )

    def do_kv_cache_update(
        self,
        layer: torch.nn.Module,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> None:
        if self.attn_type in (AttentionType.ENCODER_ONLY, AttentionType.ENCODER):
            return

        # Apply Gemma4 V-norm before caching; delegate KV layout/unbind to
        # FlashAttentionImpl (see FlashAttentionBackend.get_kv_cache_shape).
        if self.apply_v_norm:
            value = self._rms_norm_no_weight(value)

        super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)
