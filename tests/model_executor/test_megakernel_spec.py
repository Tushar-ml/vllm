# SPDX-License-Identifier: Apache-2.0
"""Tests for Megakernel Llama 1B config validation."""

import importlib
import importlib.util
from unittest.mock import Mock

import pytest  # type: ignore[import-not-found]
from transformers import LlamaConfig

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("vllm._C") is None,
    reason="vLLM extension vllm._C not built (pip install -e .)",
)

from vllm.config import ModelConfig
from vllm.model_executor.models.megakernel_spec import (
    SPEC,
    _hf_config_matches_llama_1b,
    validate_megakernel_llama_hf_only_or_raise,
)


def test_llama_1b_spec_matches_megakernel_constants():
    c = LlamaConfig(
        vocab_size=SPEC.vocab_size,
        hidden_size=SPEC.hidden_size,
        intermediate_size=SPEC.intermediate_size,
        num_hidden_layers=SPEC.num_hidden_layers,
        num_attention_heads=SPEC.num_attention_heads,
        num_key_value_heads=SPEC.num_key_value_heads,
        head_dim=SPEC.head_dim,
    )
    assert _hf_config_matches_llama_1b(c)


def test_wrong_hidden_fails():
    c = LlamaConfig(
        vocab_size=SPEC.vocab_size,
        hidden_size=SPEC.hidden_size + 1,
        intermediate_size=SPEC.intermediate_size,
        num_hidden_layers=SPEC.num_hidden_layers,
        num_attention_heads=SPEC.num_attention_heads,
        num_key_value_heads=SPEC.num_key_value_heads,
        head_dim=SPEC.head_dim,
    )
    assert not _hf_config_matches_llama_1b(c)


def test_validate_hf_only_raises_on_mismatch():
    mc = Mock(spec=ModelConfig)
    mc.hf_config = LlamaConfig(
        vocab_size=128256,
        hidden_size=4096,
        intermediate_size=8192,
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=128,
    )
    mc.quantization = None
    with pytest.raises(ValueError, match="VLLM_MEGAKERNEL_ON"):
        validate_megakernel_llama_hf_only_or_raise(mc)
