# SPDX-License-Identifier: Apache-2.0
"""Megakernel model-family plugin registry."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch.nn as nn

from vllm.config import ModelConfig
from vllm.model_executor.models.megakernel_llama import MegakernelLlamaForCausalLM
from vllm.model_executor.models.megakernel_spec import (
    validate_megakernel_family_hf_only_or_raise,
)


Validator = Callable[[ModelConfig], None]


@dataclass(frozen=True)
class MegakernelPlugin:
    family_id: str
    supported_arches: tuple[str, ...]
    wrapper_cls: type[nn.Module]
    validator: Validator


def _family_validator(family_id: str) -> Validator:
    def _validate(model_config: ModelConfig) -> None:
        validate_megakernel_family_hf_only_or_raise(family_id, model_config)

    return _validate


MEGAKERNEL_PLUGINS: tuple[MegakernelPlugin, ...] = (
    MegakernelPlugin(
        family_id="llama_small",
        supported_arches=("LlamaForCausalLM",),
        wrapper_cls=MegakernelLlamaForCausalLM,
        validator=_family_validator("llama_small"),
    ),
    MegakernelPlugin(
        family_id="qwen_small",
        supported_arches=("Qwen2ForCausalLM",),
        wrapper_cls=MegakernelLlamaForCausalLM,
        validator=_family_validator("qwen_small"),
    ),
    MegakernelPlugin(
        family_id="mistral_small",
        supported_arches=("MistralForCausalLM",),
        wrapper_cls=MegakernelLlamaForCausalLM,
        validator=_family_validator("mistral_small"),
    ),
)


def get_plugin_for_arch(arch: str, enabled_families: set[str]) -> MegakernelPlugin | None:
    for plugin in MEGAKERNEL_PLUGINS:
        if plugin.family_id not in enabled_families:
            continue
        if arch in plugin.supported_arches:
            return plugin
    return None
