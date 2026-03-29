# SPDX-License-Identifier: Apache-2.0
"""Registry hooks for Megakernel Llama 1B."""

from __future__ import annotations

import torch.nn as nn

import vllm.envs as envs
from vllm.config import ModelConfig
from vllm.model_executor.models.registry import _ModelInfo


def apply_megakernel_llama_model_cls(
    model_cls: type[nn.Module],
    arch: str,
    model_config: ModelConfig,
) -> type[nn.Module]:
    if not envs.VLLM_MEGAKERNEL_ON:
        return model_cls
    if arch != "LlamaForCausalLM" or model_cls.__name__ != "LlamaForCausalLM":
        raise ValueError(
            "VLLM_MEGAKERNEL_ON is set but model is not LlamaForCausalLM."
        )

    from vllm.model_executor.models.megakernel_llama import MegakernelLlamaForCausalLM
    from vllm.model_executor.models.megakernel_spec import (
        validate_megakernel_llama_hf_only_or_raise,
    )

    validate_megakernel_llama_hf_only_or_raise(model_config)
    return MegakernelLlamaForCausalLM


def apply_megakernel_llama_inspection(
    model_info: _ModelInfo,
    arch: str,
    model_config: ModelConfig,
) -> _ModelInfo:
    if not envs.VLLM_MEGAKERNEL_ON:
        return model_info
    if arch != "LlamaForCausalLM":
        raise ValueError("VLLM_MEGAKERNEL_ON is set but architecture is not Llama.")

    from vllm.model_executor.models.megakernel_llama import MegakernelLlamaForCausalLM
    from vllm.model_executor.models.megakernel_spec import (
        validate_megakernel_llama_hf_only_or_raise,
    )

    validate_megakernel_llama_hf_only_or_raise(model_config)
    return _ModelInfo.from_model_cls(MegakernelLlamaForCausalLM)
