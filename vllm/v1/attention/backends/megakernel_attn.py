"""Megakernel-backed attention backend wrappers for decode and prefill."""

from __future__ import annotations

from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    QuantKey,
    kFp8StaticTensorSym,
)
from vllm.platforms import current_platform
from vllm.v1.attention.backend import AttentionType, MultipleOf
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionBackend,
    TritonAttentionImpl,
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.attention.ops.megakernel_decode_attention import (
    megakernel_decode_attention,
)
from vllm.v1.attention.ops.megakernel_prefill_attention import (
    megakernel_prefill_attention,
)


class MegakernelAttentionBackend(TritonAttentionBackend):
    """Backend shell that reuses Triton metadata/builder."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.bfloat16,
        torch.float16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "bfloat16",
        "float16",
        "fp8",
        "fp8_e4m3",
        "fp8_e5m2",
    ]

    @staticmethod
    def get_name() -> str:
        # Keep enum compatibility in Attention layer wiring.
        return "TRITON_ATTN"

    @staticmethod
    def get_impl_cls() -> type["MegakernelAttentionImpl"]:
        return MegakernelAttentionImpl

    @staticmethod
    def get_builder_cls() -> type["TritonAttentionMetadataBuilder"]:
        return TritonAttentionMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]


class MegakernelAttentionImpl(TritonAttentionImpl):
    """Megakernel routing for decode and prefill op wrappers."""

    def fused_output_quant_supported(self, quant_key: QuantKey):
        return quant_key == kFp8StaticTensorSym

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if output is None:
            raise ValueError("Output tensor must be provided.")
        if output_block_scale is not None:
            raise NotImplementedError("Block output quant is not supported.")
        if attn_metadata is None:
            return output.fill_(0)

        if attn_metadata.max_query_len != 1:
            num_actual_tokens = attn_metadata.num_actual_tokens
            megakernel_prefill_attention(
                query=query[:num_actual_tokens],
                key=key[:num_actual_tokens],
                value=value[:num_actual_tokens],
                output=output[:num_actual_tokens],
                query_start_loc=attn_metadata.query_start_loc,
                max_query_len=attn_metadata.max_query_len,
                scale=self.scale,
                alibi_slopes=self.alibi_slopes,
                sliding_window=self.sliding_window,
                logits_soft_cap=self.logits_soft_cap,
            )
            return output

        key_cache, value_cache = kv_cache.unbind(1)
        if self.kv_cache_dtype.startswith("fp8"):
            fp8_dtype = current_platform.fp8_dtype()
            if key_cache.dtype != fp8_dtype:
                key_cache = key_cache.view(fp8_dtype)
                value_cache = value_cache.view(fp8_dtype)

        num_actual_tokens = attn_metadata.num_actual_tokens
        descale_shape = (attn_metadata.query_start_loc.shape[0] - 1, key_cache.shape[2])
        megakernel_decode_attention(
            query=query[:num_actual_tokens],
            key_cache=key_cache,
            value_cache=value_cache,
            output=output[:num_actual_tokens],
            query_start_loc=attn_metadata.query_start_loc,
            seq_lens=attn_metadata.seq_lens,
            max_query_len=attn_metadata.max_query_len,
            max_seq_len=attn_metadata.max_seq_len,
            block_table=attn_metadata.block_table,
            slot_mapping=attn_metadata.slot_mapping,
            scale=self.scale,
            alibi_slopes=self.alibi_slopes,
            sliding_window=self.sliding_window,
            logits_soft_cap=self.logits_soft_cap,
            k_scale=layer._k_scale.expand(descale_shape),
            v_scale=layer._v_scale.expand(descale_shape),
            seq_threshold_3d=attn_metadata.seq_threshold_3D,
            num_par_softmax_segments=attn_metadata.num_par_softmax_segments,
            softmax_segm_output=attn_metadata.softmax_segm_output,
            softmax_segm_max=attn_metadata.softmax_segm_max,
            softmax_segm_expsum=attn_metadata.softmax_segm_expsum,
            output_scale=output_scale,
        )
        return output
