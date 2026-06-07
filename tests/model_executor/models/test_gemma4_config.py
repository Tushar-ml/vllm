# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.config import AttentionConfig
from vllm.model_executor.models.config import Gemma4Config
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def _heterogeneous_gemma4_vllm_config(
    backend: AttentionBackendEnum | None,
) -> SimpleNamespace:
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(
                head_dim=256,
                global_head_dim=512,
            ),
        ),
        attention_config=AttentionConfig(backend=backend),
    )


@patch(
    "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend.supports_head_size",
    return_value=False,
)
def test_gemma4_config_forces_triton_when_backend_unset(_mock_supports: object) -> None:
    vllm_config = _heterogeneous_gemma4_vllm_config(backend=None)
    Gemma4Config.verify_and_update_config(vllm_config)
    assert (
        vllm_config.attention_config.backend == AttentionBackendEnum.TRITON_ATTN
    )


@patch(
    "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend.supports_head_size",
    return_value=False,
)
def test_gemma4_config_raises_when_explicit_flash_unsupported(
    _mock_supports: object,
) -> None:
    vllm_config = _heterogeneous_gemma4_vllm_config(
        backend=AttentionBackendEnum.GEMMA4_FLASH_ATTN
    )
    with pytest.raises(ValueError, match="cannot run full-attention layers"):
        Gemma4Config.verify_and_update_config(vllm_config)
    assert (
        vllm_config.attention_config.backend
        == AttentionBackendEnum.GEMMA4_FLASH_ATTN
    )


@patch(
    "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend.supports_head_size",
    return_value=True,
)
def test_gemma4_config_keeps_explicit_flash_when_supported(
    _mock_supports: object,
) -> None:
    vllm_config = _heterogeneous_gemma4_vllm_config(
        backend=AttentionBackendEnum.GEMMA4_FLASH_ATTN
    )
    Gemma4Config.verify_and_update_config(vllm_config)
    assert (
        vllm_config.attention_config.backend
        == AttentionBackendEnum.GEMMA4_FLASH_ATTN
    )


@patch(
    "vllm.v1.attention.backends.flash_attn.FlashAttentionBackend.supports_head_size",
    return_value=False,
)
def test_gemma4_config_keeps_explicit_triton(_mock_supports: object) -> None:
    vllm_config = _heterogeneous_gemma4_vllm_config(
        backend=AttentionBackendEnum.TRITON_ATTN
    )
    Gemma4Config.verify_and_update_config(vllm_config)
    assert vllm_config.attention_config.backend == AttentionBackendEnum.TRITON_ATTN
