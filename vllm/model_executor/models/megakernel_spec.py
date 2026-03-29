# SPDX-License-Identifier: Apache-2.0
"""Compile-time shape checks for Megakernels `llama_1b_globals`."""

from __future__ import annotations

from dataclasses import dataclass

from transformers import LlamaConfig

from vllm.config import ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig


@dataclass(frozen=True)
class MegakernelLlama1BSpec:
    num_hidden_layers: int = 16
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 64
    vocab_size: int = 128256


SPEC = MegakernelLlama1BSpec()


def _hf_config_matches_llama_1b(hf: LlamaConfig) -> bool:
    hd = getattr(hf, "head_dim", None) or (hf.hidden_size // hf.num_attention_heads)
    return (
        hf.num_hidden_layers == SPEC.num_hidden_layers
        and hf.hidden_size == SPEC.hidden_size
        and hf.intermediate_size == SPEC.intermediate_size
        and hf.num_attention_heads == SPEC.num_attention_heads
        and hf.num_key_value_heads == SPEC.num_key_value_heads
        and hd == SPEC.head_dim
        and hf.vocab_size == SPEC.vocab_size
    )


def validate_megakernel_llama_hf_only_or_raise(model_config: ModelConfig) -> None:
    hf = model_config.hf_config
    if not isinstance(hf, LlamaConfig):
        raise ValueError("Megakernel path requires a LlamaConfig model.")
    if not _hf_config_matches_llama_1b(hf):
        raise ValueError(
            "VLLM_MEGAKERNEL_ON requires Llama 3.2 1B-compatible config "
            f"(layers={SPEC.num_hidden_layers}, hidden={SPEC.hidden_size}, "
            f"intermediate={SPEC.intermediate_size}, heads={SPEC.num_attention_heads}, "
            f"kv_heads={SPEC.num_key_value_heads}, head_dim={SPEC.head_dim}, "
            f"vocab={SPEC.vocab_size})."
        )
    if model_config.quantization is not None:
        raise ValueError("Megakernel path does not support quantization in this PoC.")


def validate_megakernel_llama_runtime_or_raise(vllm_config: VllmConfig) -> None:
    validate_megakernel_llama_hf_only_or_raise(vllm_config.model_config)
    parallel_config: ParallelConfig = vllm_config.parallel_config
    scheduler_config: SchedulerConfig = vllm_config.scheduler_config
    if parallel_config.tensor_parallel_size != 1:
        raise ValueError("Megakernel path requires tensor_parallel_size=1.")
    if parallel_config.pipeline_parallel_size != 1:
        raise ValueError("Megakernel path requires pipeline_parallel_size=1.")
    if scheduler_config.max_num_seqs < 1:
        raise ValueError("Megakernel path requires max_num_seqs >= 1.")
