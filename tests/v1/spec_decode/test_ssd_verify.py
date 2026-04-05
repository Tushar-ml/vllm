# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.spec_decode.ssd.algo import effective_ssd_cache_hits_for_verify
from vllm.v1.spec_decode.ssd.protocol import ssd_async_layout_valid
from vllm.v1.spec_decode.ssd.verify_ops import apply_sampler_x_rescaling, ssd_verify


def test_ssd_verify_greedy_full_accept():
    B, K, V = 1, 3, 8
    # preds_p positions 0..K argmax -> [1,2,3,4] for rows; draft_tokens should match preds_p[:,:-1]
    logits_p = torch.zeros(B, K + 1, V)
    for i in range(K + 1):
        logits_p[0, i, i + 1] = 10.0
    logits_q = torch.zeros(B, K, V)
    # speculations: col0 recovery (unused for match test), cols 1..K draft tokens
    speculations = torch.tensor([[0, 1, 2, 3]], dtype=torch.long)
    temps = torch.zeros(B, dtype=torch.float32)
    suffixes, rec = ssd_verify(
        logits_p,
        logits_q,
        speculations,
        temps,
        temps,
        cache_hits=None,
        jit_speculate=False,
    )
    assert suffixes == [[0, 1, 2, 3]]
    assert rec == [4]


def test_ssd_verify_greedy_partial_accept():
    B, K, V = 1, 3, 8
    logits_p = torch.zeros(B, K + 1, V)
    logits_p[0, 0, 1] = 10.0
    logits_p[0, 1, 2] = 10.0
    logits_p[0, 2, 2] = 10.0  # mismatch: draft token is 3 at position 2
    logits_p[0, 3, 4] = 10.0
    logits_q = torch.zeros(B, K, V)
    speculations = torch.tensor([[5, 1, 2, 3]], dtype=torch.long)
    temps = torch.zeros(B, dtype=torch.float32)
    suffixes, rec = ssd_verify(
        logits_p,
        logits_q,
        speculations,
        temps,
        temps,
    )
    assert suffixes == [[5, 1]]
    assert rec == [2]  # argmax at verify position 2


def test_apply_sampler_x_rescaling_renormalizes():
    probs = torch.tensor([[[0.5, 0.3, 0.2]]])
    out = apply_sampler_x_rescaling(probs, sampler_x=2.0, fan_out=2)
    assert torch.allclose(out.sum(dim=-1), torch.ones(1, 1))
    assert out.shape == probs.shape


def test_ssd_async_layout_valid():
    assert ssd_async_layout_valid(4, 5) is True
    assert ssd_async_layout_valid(4, 4) is False


def test_effective_ssd_cache_hits_none_without_ipc():
    class _MiniSpec:
        method = "ssd"
        ssd_async = True

    hits = effective_ssd_cache_hits_for_verify(
        _MiniSpec(), batch_size=2, device=torch.device("cpu")
    )
    assert hits is None


def test_ssd_verify_ratio_with_cache_hits():
    """Ratio acceptance applies only on cache-hit rows when jit_speculate=False."""
    torch.manual_seed(0)
    B, K, V = 2, 1, 4
    logits_p = torch.zeros(B, K + 1, V)
    logits_p[:, :, 1] = 5.0
    logits_p[:, :, 2] = 3.0
    logits_q = torch.zeros(B, K, V)
    logits_q[:, :, 1] = 4.0
    logits_q[:, :, 2] = 5.0
    speculations = torch.tensor([[0, 1], [0, 1]], dtype=torch.long)
    temps_t = torch.tensor([1.0, 1.0])
    temps_q = torch.tensor([1.0, 1.0])
    cache_hits = torch.tensor([1, 0], dtype=torch.int64)
    suffixes, _rec = ssd_verify(
        logits_p,
        logits_q,
        speculations,
        temps_t,
        temps_q,
        cache_hits=cache_hits,
        jit_speculate=False,
    )
    assert len(suffixes) == 2


def test_create_ssd_async_process_group_raises():
    from vllm.v1.spec_decode.ssd.async_ipc import create_ssd_async_process_group

    with pytest.raises(NotImplementedError):
        create_ssd_async_process_group()
