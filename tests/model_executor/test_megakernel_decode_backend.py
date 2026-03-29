# SPDX-License-Identifier: Apache-2.0

import importlib.util
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig

pytestmark = pytest.mark.skipif(
    importlib.util.find_spec("vllm._C") is None,
    reason="vLLM extension vllm._C not built (pip install -e .)",
)

from vllm.v1.attention.backend import AttentionType
from vllm.v1.attention.selector import AttentionSelectorConfig, _cached_get_attn_backend


def test_selector_uses_megakernel_backend_when_enabled(monkeypatch: pytest.MonkeyPatch):
    import vllm.envs as envs
    from vllm import config as config_mod
    from vllm.v1 import attention as attention_pkg

    monkeypatch.setattr(envs, "VLLM_MEGAKERNEL_ON", True)

    llama_cfg = LlamaConfig(
        vocab_size=128256,
        hidden_size=2048,
        intermediate_size=8192,
        num_hidden_layers=16,
        num_attention_heads=32,
        num_key_value_heads=8,
        head_dim=64,
    )
    model_cfg = SimpleNamespace(hf_config=llama_cfg, quantization=None)
    vllm_cfg = SimpleNamespace(model_config=model_cfg)
    monkeypatch.setattr(config_mod, "get_current_vllm_config", lambda: vllm_cfg)

    from vllm.platforms import current_platform

    monkeypatch.setattr(
        current_platform,
        "get_attn_backend_cls",
        lambda backend, attn_selector_config, num_heads: (
            "vllm.v1.attention.backends.triton_attn.TritonAttentionBackend"
        ),
    )

    _cached_get_attn_backend.cache_clear()
    cfg = AttentionSelectorConfig(
        head_size=64,
        dtype=torch.bfloat16,
        kv_cache_dtype="auto",
        block_size=16,
        use_mla=False,
        has_sink=False,
        use_sparse=False,
        use_mm_prefix=False,
        use_per_head_quant_scales=False,
        attn_type=AttentionType.DECODER,
    )
    cls = _cached_get_attn_backend(backend=None, attn_selector_config=cfg, num_heads=32)
    assert cls.__name__ == "MegakernelAttentionBackend"


def test_megakernel_model_wrapper_has_no_custom_forward():
    from vllm.model_executor.models.llama import LlamaForCausalLM
    from vllm.model_executor.models.megakernel_llama import MegakernelLlamaForCausalLM

    assert MegakernelLlamaForCausalLM.forward is LlamaForCausalLM.forward
