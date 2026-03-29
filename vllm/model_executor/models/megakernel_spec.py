# SPDX-License-Identifier: Apache-2.0
"""Compile-time shape checks for Megakernel small-model families."""

from __future__ import annotations

from dataclasses import dataclass

from transformers import LlamaConfig, MistralConfig, Qwen2Config

from vllm.config import ModelConfig, ParallelConfig, SchedulerConfig, VllmConfig


@dataclass(frozen=True)
class MegakernelShapeSpec:
    max_num_hidden_layers: int
    max_hidden_size: int
    max_intermediate_size: int
    max_num_attention_heads: int
    max_num_key_value_heads: int
    max_head_dim: int


@dataclass(frozen=True)
class MegakernelLlama1BSpec(MegakernelShapeSpec):
    num_hidden_layers: int = 16
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 64
    vocab_size: int = 128256


LLAMA_1B_SPEC = MegakernelLlama1BSpec(
    max_num_hidden_layers=16,
    max_hidden_size=2048,
    max_intermediate_size=8192,
    max_num_attention_heads=32,
    max_num_key_value_heads=8,
    max_head_dim=64,
)


SMALL_FAMILY_SPECS: dict[str, MegakernelShapeSpec] = {
    # Current production path: strict Llama 3.2 1B-compatible shape.
    "llama_small": LLAMA_1B_SPEC,
    # Phase-1 families: allow only "small" envelopes; runtime path is shared.
    "qwen_small": MegakernelShapeSpec(
        max_num_hidden_layers=36,
        max_hidden_size=3072,
        max_intermediate_size=12288,
        max_num_attention_heads=32,
        max_num_key_value_heads=16,
        max_head_dim=128,
    ),
    "mistral_small": MegakernelShapeSpec(
        max_num_hidden_layers=36,
        max_hidden_size=4096,
        max_intermediate_size=14336,
        max_num_attention_heads=32,
        max_num_key_value_heads=16,
        max_head_dim=128,
    ),
}


def _hf_config_matches_llama_1b(hf: LlamaConfig) -> bool:
    hd = getattr(hf, "head_dim", None) or (hf.hidden_size // hf.num_attention_heads)
    return (
        hf.num_hidden_layers == LLAMA_1B_SPEC.num_hidden_layers
        and hf.hidden_size == LLAMA_1B_SPEC.hidden_size
        and hf.intermediate_size == LLAMA_1B_SPEC.intermediate_size
        and hf.num_attention_heads == LLAMA_1B_SPEC.num_attention_heads
        and hf.num_key_value_heads == LLAMA_1B_SPEC.num_key_value_heads
        and hd == LLAMA_1B_SPEC.head_dim
        and hf.vocab_size == LLAMA_1B_SPEC.vocab_size
    )


def validate_megakernel_llama_hf_only_or_raise(model_config: ModelConfig) -> None:
    hf = model_config.hf_config
    if not isinstance(hf, LlamaConfig):
        raise ValueError("Megakernel path requires a LlamaConfig model.")
    if not _hf_config_matches_llama_1b(hf):
        raise ValueError(
            "VLLM_MEGAKERNEL_ON requires Llama 3.2 1B-compatible config "
            f"(layers={LLAMA_1B_SPEC.num_hidden_layers}, hidden={LLAMA_1B_SPEC.hidden_size}, "
            f"intermediate={LLAMA_1B_SPEC.intermediate_size}, heads={LLAMA_1B_SPEC.num_attention_heads}, "
            f"kv_heads={LLAMA_1B_SPEC.num_key_value_heads}, head_dim={LLAMA_1B_SPEC.head_dim}, "
            f"vocab={LLAMA_1B_SPEC.vocab_size})."
        )
    if model_config.quantization is not None:
        raise ValueError("Megakernel path does not support quantization in this PoC.")


def _validate_small_shape_or_raise(
    *,
    family_id: str,
    num_hidden_layers: int,
    hidden_size: int,
    intermediate_size: int,
    num_attention_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> None:
    spec = SMALL_FAMILY_SPECS[family_id]
    if num_hidden_layers > spec.max_num_hidden_layers:
        raise ValueError(
            f"{family_id} exceeds max_num_hidden_layers={spec.max_num_hidden_layers}."
        )
    if hidden_size > spec.max_hidden_size:
        raise ValueError(f"{family_id} exceeds max_hidden_size={spec.max_hidden_size}.")
    if intermediate_size > spec.max_intermediate_size:
        raise ValueError(
            f"{family_id} exceeds max_intermediate_size={spec.max_intermediate_size}."
        )
    if num_attention_heads > spec.max_num_attention_heads:
        raise ValueError(
            f"{family_id} exceeds max_num_attention_heads={spec.max_num_attention_heads}."
        )
    if num_key_value_heads > spec.max_num_key_value_heads:
        raise ValueError(
            f"{family_id} exceeds max_num_key_value_heads={spec.max_num_key_value_heads}."
        )
    if head_dim > spec.max_head_dim:
        raise ValueError(f"{family_id} exceeds max_head_dim={spec.max_head_dim}.")


def validate_megakernel_family_hf_only_or_raise(
    family_id: str, model_config: ModelConfig
) -> None:
    if family_id == "llama_small":
        validate_megakernel_llama_hf_only_or_raise(model_config)
        return
    if model_config.quantization is not None:
        raise ValueError("Megakernel path does not support quantization in this PoC.")
    hf = model_config.hf_config
    if family_id == "qwen_small":
        if not isinstance(hf, Qwen2Config):
            raise ValueError("Megakernel qwen_small path requires a Qwen2Config model.")
        hd = getattr(hf, "head_dim", None) or (hf.hidden_size // hf.num_attention_heads)
        _validate_small_shape_or_raise(
            family_id=family_id,
            num_hidden_layers=hf.num_hidden_layers,
            hidden_size=hf.hidden_size,
            intermediate_size=hf.intermediate_size,
            num_attention_heads=hf.num_attention_heads,
            num_key_value_heads=hf.num_key_value_heads,
            head_dim=hd,
        )
        return
    if family_id == "mistral_small":
        if not isinstance(hf, MistralConfig):
            raise ValueError("Megakernel mistral_small path requires a MistralConfig model.")
        hd = getattr(hf, "head_dim", None) or (hf.hidden_size // hf.num_attention_heads)
        _validate_small_shape_or_raise(
            family_id=family_id,
            num_hidden_layers=hf.num_hidden_layers,
            hidden_size=hf.hidden_size,
            intermediate_size=hf.intermediate_size,
            num_attention_heads=hf.num_attention_heads,
            num_key_value_heads=hf.num_key_value_heads,
            head_dim=hd,
        )
        return
    raise ValueError(f"Unknown Megakernel family '{family_id}'.")


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


def _infer_family_from_hf_config(model_config: ModelConfig) -> str:
    hf = model_config.hf_config
    if isinstance(hf, LlamaConfig):
        return "llama_small"
    if isinstance(hf, Qwen2Config):
        return "qwen_small"
    if isinstance(hf, MistralConfig):
        return "mistral_small"
    raise ValueError(f"Unsupported Megakernel model config type: {type(hf).__name__}.")


def get_megakernel_family_id(model_config: ModelConfig) -> str | None:
    try:
        return _infer_family_from_hf_config(model_config)
    except Exception:
        return None


def validate_megakernel_runtime_or_raise(vllm_config: VllmConfig) -> None:
    family_id = _infer_family_from_hf_config(vllm_config.model_config)
    validate_megakernel_family_hf_only_or_raise(family_id, vllm_config.model_config)
    parallel_config: ParallelConfig = vllm_config.parallel_config
    scheduler_config: SchedulerConfig = vllm_config.scheduler_config
    if parallel_config.tensor_parallel_size != 1:
        raise ValueError("Megakernel path requires tensor_parallel_size=1.")
    if parallel_config.pipeline_parallel_size != 1:
        raise ValueError("Megakernel path requires pipeline_parallel_size=1.")
    if scheduler_config.max_num_seqs < 1:
        raise ValueError("Megakernel path requires max_num_seqs >= 1.")
