# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for FLy (Training-Free Loosely Speculative Decoding) verification."""

import math
from types import SimpleNamespace

import pytest
import torch

from vllm.platforms import current_platform
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import (
    PLACEHOLDER_TOKEN_ID,
    _fly_argmax_and_h_norm_full,
    _fly_argmax_and_h_norm_top_k,
    _fly_compute_stop_indices_tensor,
    _fly_h_norm_full_vocab_ref,
    fly_greedy_rejection_sample,
    rejection_sample,
)
from vllm.v1.sample.logits_processor import LogitsProcessors

DEVICE = current_platform.device_type


def _sampling_metadata_greedy() -> SamplingMetadata:
    return SamplingMetadata(
        temperature=None,
        all_greedy=True,
        all_random=False,
        top_p=None,
        top_k=None,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.tensor([]),
        presence_penalties=torch.tensor([]),
        repetition_penalties=torch.tensor([]),
        output_token_ids=[],
        spec_token_ids=None,
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


def test_fly_strict_mismatch_low_entropy():
    """Strict gate: low entropy mismatch rejects at first position (standard SPD)."""
    V = 8
    K = 3
    draft = torch.tensor([5, 1, 2], dtype=torch.int32, device=DEVICE)
    logits = torch.full((K, V), -100.0, device=DEVICE)
    # Position 0: target argmax = 0, draft 5 -> mismatch, very peaked -> low h
    logits[0, 0] = 10.0
    # Positions 1,2 match draft
    logits[1, 1] = 10.0
    logits[2, 2] = 10.0

    out = torch.full((1, K + 1), PLACEHOLDER_TOKEN_ID, dtype=torch.int32, device=DEVICE)
    bonus = torch.tensor([[99]], dtype=torch.int32, device=DEVICE)
    fly_greedy_rejection_sample(
        out,
        draft,
        [K],
        K,
        logits,
        bonus,
        theta=0.3,
        defer_window=6,
    )
    ta0 = int(logits[0].argmax().item())
    assert ta0 == 0
    assert int(out[0, 0].item()) == 0
    assert int(out[0, 1].item()) == PLACEHOLDER_TOKEN_ID


def test_fly_loose_accept_high_entropy():
    """Defer: high-entropy mismatch, no further mismatches in window -> keep draft."""
    V = 4
    K = 3
    W = 2
    # draft [3,1,2]; target argmax pos0 = 0 (uniform), pos1=1, pos2=2
    draft = torch.tensor([3, 1, 2], dtype=torch.int32, device=DEVICE)
    logits = torch.zeros((K, V), device=DEVICE)
    logits[1, 1] = 10.0
    logits[2, 2] = 10.0

    out = torch.full((1, K + 1), PLACEHOLDER_TOKEN_ID, dtype=torch.int32, device=DEVICE)
    bonus = torch.tensor([[42]], dtype=torch.int32, device=DEVICE)
    fly_greedy_rejection_sample(
        out,
        draft,
        [K],
        K,
        logits,
        bonus,
        theta=0.3,
        defer_window=W,
    )
    assert int(out[0, 0].item()) == 3
    assert int(out[0, 1].item()) == 1
    assert int(out[0, 2].item()) == 2
    assert int(out[0, 3].item()) == 42


def test_fly_defer_reject_second_mismatch_in_window():
    """Another mismatch inside the lookahead window rejects the first loose slot."""
    V = 4
    K = 4
    W = 3
    # Mismatch at 0 (high h), mismatch at 1 as well -> N_W(0) > 0
    draft = torch.tensor([3, 3, 2, 2], dtype=torch.int32, device=DEVICE)
    logits = torch.zeros((K, V), device=DEVICE)
    # pos0 uniform -> argmax 0; pos1 uniform -> argmax 0; pos2 target 2; pos3 target 2
    logits[2, 2] = 10.0
    logits[3, 2] = 10.0

    out = torch.full((1, K + 1), PLACEHOLDER_TOKEN_ID, dtype=torch.int32, device=DEVICE)
    bonus = torch.tensor([[7]], dtype=torch.int32, device=DEVICE)
    fly_greedy_rejection_sample(
        out,
        draft,
        [K],
        K,
        logits,
        bonus,
        theta=0.3,
        defer_window=W,
    )
    # First rejection at paper position 1 (defer reject at mismatch 0)
    assert int(out[0, 0].item()) == 0  # target argmax at 0
    assert int(out[0, 1].item()) == PLACEHOLDER_TOKEN_ID


def test_fly_window_boundary_insufficient_lookahead():
    """j + W > K - 1 -> conservative defer reject (paper Eq. 9)."""
    V = 4
    K = 2
    W = 6
    draft = torch.tensor([3, 1], dtype=torch.int32, device=DEVICE)
    logits = torch.zeros((K, V), device=DEVICE)
    logits[1, 1] = 10.0

    out = torch.full((1, K + 1), PLACEHOLDER_TOKEN_ID, dtype=torch.int32, device=DEVICE)
    bonus = torch.tensor([[9]], dtype=torch.int32, device=DEVICE)
    fly_greedy_rejection_sample(
        out,
        draft,
        [K],
        K,
        logits,
        bonus,
        theta=0.3,
        defer_window=W,
    )
    assert int(out[0, 0].item()) == 0
    assert int(out[0, 1].item()) == PLACEHOLDER_TOKEN_ID


def test_rejection_sample_fly_end_to_end():
    """rejection_sample with a fly config object delegates to FLy path."""
    V = 4
    K = 3
    draft = torch.tensor([3, 1, 2], dtype=torch.int32, device=DEVICE)
    logits = torch.zeros((K, V), device=DEVICE)
    logits[1, 1] = 10.0
    logits[2, 2] = 10.0

    cfg = SimpleNamespace(
        rejection_sample_method="fly",
        fly_entropy_threshold=0.3,
        fly_defer_window=2,
        fly_entropy_mode="full",
        fly_entropy_top_k=1024,
        fly_use_triton_entropy=False,
    )
    out = rejection_sample(
        draft,
        [K],
        K,
        torch.tensor([K], dtype=torch.int32, device=DEVICE),
        None,
        logits,
        torch.tensor([[99]], dtype=torch.int32, device=DEVICE),
        _sampling_metadata_greedy(),
        speculative_config=cfg,
    )
    assert int(out[0, 0].item()) == 3
    assert int(out[0, 3].item()) == 99


def test_top_k_entropy_matches_full_when_k_covers_vocab():
    logits = torch.randn(6, 64, device=DEVICE)
    _, hf = _fly_argmax_and_h_norm_full(logits)
    _, ht = _fly_argmax_and_h_norm_top_k(logits, 64)
    assert torch.allclose(hf, ht, rtol=1e-4, atol=1e-4)


def test_fly_full_entropy_argmax_parity_vs_reference():
    """Single log-softmax path matches softmax+entropy and logits argmax."""
    torch.manual_seed(0)
    logits = torch.randn(4, 32, device=DEVICE)
    ta_new, h_new = _fly_argmax_and_h_norm_full(logits)
    h_ref = _fly_h_norm_full_vocab_ref(logits)
    ta_ref = logits.argmax(dim=-1)
    assert torch.equal(ta_new, ta_ref)
    assert torch.allclose(h_new, h_ref, rtol=1e-5, atol=1e-5)


def test_fly_top_k_argmax_entropy_parity_vs_two_pass():
    """top_k path: argmax from top-1 index and h_norm match two-pass reference."""
    torch.manual_seed(1)
    V, k = 128, 16
    logits = torch.randn(5, V, device=DEVICE)
    ta_new, h_new = _fly_argmax_and_h_norm_top_k(logits, k)
    vals, idx = torch.topk(logits, k, dim=-1)
    ta_ref = idx[:, 0]
    sub = torch.softmax(vals.to(torch.float32), dim=-1)
    clamped = sub.clamp(min=1e-20)
    entropy = -(sub * torch.log(clamped)).sum(dim=-1)
    h_ref = entropy / math.log(float(k))
    assert torch.equal(ta_new, ta_ref.to(torch.long))
    assert torch.allclose(h_new, h_ref, rtol=1e-5, atol=1e-5)


def test_fly_stop_indices_tensor_matches_loop_reference():
    """Tensorized stop indices match sequential reference (CPU gold)."""
    torch.manual_seed(2)
    for _ in range(32):
        K = int(torch.randint(1, 9, (1,)).item())
        W = int(torch.randint(0, 8, (1,)).item())
        theta = 0.35
        delta = torch.rand(K, device=DEVICE) > 0.5
        h = torch.rand(K, device=DEVICE)

        def ref_s_gate():
            strict = (~delta) & (h < theta)
            if not bool(strict.any().item()):
                return K + 1
            j = int(torch.nonzero(strict, as_tuple=False)[0, 0].item())
            return j + 1

        def ref_s_defer():
            s_defer = K + 1
            for j in range(K):
                if bool(delta[j].item()):
                    continue
                if bool((h[j] < theta).item()):
                    continue
                if j + W > K - 1:
                    s_defer = min(s_defer, j + 1)
                else:
                    seg = delta[j + 1 : j + W + 1]
                    if bool((~seg).any().item()):
                        s_defer = min(s_defer, j + 1)
            return s_defer

        sg_e = ref_s_gate()
        sd_e = ref_s_defer()
        sg_t, sd_t = _fly_compute_stop_indices_tensor(
            delta, h, theta, K, W
        )
        assert int(sg_t.item()) == sg_e
        assert int(sd_t.item()) == sd_e


def test_fly_use_triton_entropy_raises():
    out = torch.zeros(1, 2, dtype=torch.int32, device=DEVICE)
    with pytest.raises(NotImplementedError):
        fly_greedy_rejection_sample(
            out,
            torch.zeros(1, dtype=torch.int32, device=DEVICE),
            [1],
            1,
            torch.zeros(1, 4, device=DEVICE),
            torch.tensor([[0]], dtype=torch.int32, device=DEVICE),
            theta=0.3,
            defer_window=1,
            use_triton_entropy=True,
        )


def test_fly_loose_accept_top_k_mode_agrees_with_full():
    """With V=4, top_k=4 matches full entropy; FLy output should match."""
    K = 3
    W = 2
    draft = torch.tensor([3, 1, 2], dtype=torch.int32, device=DEVICE)
    logits = torch.zeros((K, 4), device=DEVICE)
    logits[1, 1] = 10.0
    logits[2, 2] = 10.0

    out_full = torch.full(
        (1, K + 1), PLACEHOLDER_TOKEN_ID, dtype=torch.int32, device=DEVICE
    )
    out_topk = torch.full(
        (1, K + 1), PLACEHOLDER_TOKEN_ID, dtype=torch.int32, device=DEVICE
    )
    bonus = torch.tensor([[42]], dtype=torch.int32, device=DEVICE)
    fly_greedy_rejection_sample(
        out_full,
        draft,
        [K],
        K,
        logits,
        bonus,
        theta=0.3,
        defer_window=W,
        entropy_mode="full",
    )
    fly_greedy_rejection_sample(
        out_topk,
        draft,
        [K],
        K,
        logits,
        bonus,
        theta=0.3,
        defer_window=W,
        entropy_mode="top_k",
        entropy_top_k=4,
    )
    assert torch.equal(out_full, out_topk)

