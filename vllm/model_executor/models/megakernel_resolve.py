# SPDX-License-Identifier: Apache-2.0
"""Registry hooks for Megakernel small-model plugin dispatch."""

from __future__ import annotations

import torch.nn as nn

import vllm.envs as envs
from vllm.config import ModelConfig
from vllm.model_executor.models.megakernel_plugins import get_plugin_for_arch
from vllm.model_executor.models.registry import _ModelInfo


def apply_megakernel_model_cls(
    model_cls: type[nn.Module],
    arch: str,
    model_config: ModelConfig,
) -> type[nn.Module]:
    if not envs.VLLM_MEGAKERNEL_ON:
        return model_cls
    enabled_families = set(envs.VLLM_MEGAKERNEL_FAMILIES)
    plugin = get_plugin_for_arch(arch, enabled_families)
    if plugin is None:
        if envs.VLLM_MEGAKERNEL_STRICT:
            raise ValueError(
                "VLLM_MEGAKERNEL_ON is set but architecture is not enabled for "
                f"Megakernel plugins (arch={arch}, families={sorted(enabled_families)})."
            )
        return model_cls
    plugin.validator(model_config)
    return plugin.wrapper_cls


def apply_megakernel_inspection(
    model_info: _ModelInfo,
    arch: str,
    model_config: ModelConfig,
) -> _ModelInfo:
    if not envs.VLLM_MEGAKERNEL_ON:
        return model_info
    enabled_families = set(envs.VLLM_MEGAKERNEL_FAMILIES)
    plugin = get_plugin_for_arch(arch, enabled_families)
    if plugin is None:
        if envs.VLLM_MEGAKERNEL_STRICT:
            raise ValueError(
                "VLLM_MEGAKERNEL_ON is set but architecture is not enabled for "
                f"Megakernel plugins (arch={arch}, families={sorted(enabled_families)})."
            )
        return model_info
    plugin.validator(model_config)
    return _ModelInfo.from_model_cls(plugin.wrapper_cls)
