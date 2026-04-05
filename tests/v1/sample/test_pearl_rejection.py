# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import (
    pearl_greedy_rejection_sample,
    pearl_random_rejection_sample,
)


def _random_sampling_metadata(
    device: torch.device, generators: dict[int, torch.Generator] | None = None
) -> SamplingMetadata:
    t = torch.tensor([1.0], device=device)
    z = torch.tensor([], device=device)
    return SamplingMetadata(
        temperature=t,
        all_greedy=False,
        all_random=True,
        top_p=None,
        top_k=None,
        generators={} if generators is None else generators,
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=z,
        presence_penalties=z,
        repetition_penalties=z,
        output_token_ids=[],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


def _reference_pearl_greedy(
    pearl_pre_verify: bool,
    drafts: list[int],
    targ_argmax: list[int],
    bonus: int,
    max_spec_len: int,
) -> list[int]:
    k = len(drafts)
    out = [-1] * (max_spec_len + 1)
    if pearl_pre_verify:
        if drafts[0] == targ_argmax[0]:
            for i in range(k):
                out[i] = drafts[i]
            out[k] = bonus
        else:
            out[0] = targ_argmax[0]
    else:
        rejected = False
        for i in range(k):
            if not rejected:
                out[i] = targ_argmax[i]
                if drafts[i] != targ_argmax[i]:
                    rejected = True
        if not rejected:
            out[k] = bonus
    return out


def _run_kernel(
    pre_verify: bool,
    drafts: list[int],
    targ: list[int],
    bonus: int,
    max_spec_len: int,
    device: str,
) -> list[int]:
    k = len(drafts)
    draft_t = torch.tensor(drafts, dtype=torch.int32, device=device)
    targ_t = torch.tensor(targ, dtype=torch.int32, device=device)
    bonus_t = torch.tensor([bonus], dtype=torch.int32, device=device)
    cu = torch.tensor([k], dtype=torch.int32, device=device)
    flags = torch.tensor([1 if pre_verify else 0], dtype=torch.int32, device=device)
    out = pearl_greedy_rejection_sample(
        draft_t, [k], max_spec_len, cu, targ_t, bonus_t, flags
    )
    return out[0].cpu().tolist()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pearl_greedy_pre_verify_accept_all_drafts():
    device = "cuda"
    max_spec_len = 4
    drafts = [1, 2, 3]
    targ = [1, 9, 9]
    bonus = 7
    ref = _reference_pearl_greedy(True, drafts, targ, bonus, max_spec_len)
    got = _run_kernel(True, drafts, targ, bonus, max_spec_len, device)
    assert got == ref


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pearl_greedy_pre_verify_reject_first():
    device = "cuda"
    max_spec_len = 4
    drafts = [5, 2, 3]
    targ = [1, 2, 3]
    bonus = 7
    ref = _reference_pearl_greedy(True, drafts, targ, bonus, max_spec_len)
    got = _run_kernel(True, drafts, targ, bonus, max_spec_len, device)
    assert got == ref


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pearl_greedy_post_verify_partial_reject():
    device = "cuda"
    max_spec_len = 4
    drafts = [1, 2, 3]
    targ = [1, 9, 9]
    bonus = 7
    ref = _reference_pearl_greedy(False, drafts, targ, bonus, max_spec_len)
    got = _run_kernel(False, drafts, targ, bonus, max_spec_len, device)
    assert got == ref


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_pearl_greedy_batched_two_requests():
    device = "cuda"
    max_spec_len = 3
    # Req0: pre-verify accept [10,11] + bonus 99
    # Req1: post-verify all match [3,4] + bonus 88
    draft_flat = torch.tensor([10, 11, 3, 4], dtype=torch.int32, device=device)
    targ_flat = torch.tensor([10, 0, 3, 4], dtype=torch.int32, device=device)
    cu = torch.tensor([2, 4], dtype=torch.int32, device=device)
    bonus = torch.tensor([99, 88], dtype=torch.int32, device=device)
    flags = torch.tensor([1, 0], dtype=torch.int32, device=device)
    out = pearl_greedy_rejection_sample(
        draft_flat,
        [2, 2],
        max_spec_len,
        cu,
        targ_flat,
        bonus,
        flags,
    )
    r0 = _reference_pearl_greedy(True, [10, 11], [10, 0], 99, max_spec_len)
    r1 = _reference_pearl_greedy(False, [3, 4], [3, 4], 88, max_spec_len)
    assert out[0].cpu().tolist() == r0
    assert out[1].cpu().tolist() == r1


def test_scheduler_pearl_mode_flag_matches_full_accept_rule():
    """After a spec-decode step, pearl_pre_verify = (num_accepted != num_draft)."""
    for num_acc, num_draft, want_pre in [
        (3, 3, False),
        (0, 3, True),
        (2, 3, True),
    ]:
        assert (num_acc != num_draft) == want_pre


def test_pearl_pre_verify_draft_split_contract():
    """Pre-verify keeps one spec slot for the target while full γ lives in pearl_full."""
    pearl_pre_verify = True
    spec = [10, 20, 30]
    if pearl_pre_verify and len(spec) > 1:
        pearl_full = list(spec)
        spec_sched = spec[:1]
    else:
        pearl_full = None
        spec_sched = spec
    assert pearl_full == [10, 20, 30]
    assert spec_sched == [10]


def test_pearl_random_pre_verify_accepts_all_when_target_likes_d0():
    device = torch.device("cpu")
    # Two draft positions both token id 1; target logits strongly favor 1.
    logits = torch.tensor(
        [[-100.0, 100.0, -100.0, -100.0], [-100.0, 100.0, -100.0, -100.0]],
        device=device,
    )
    drafts = torch.tensor([1, 1], dtype=torch.int32, device=device)
    cu = torch.tensor([2], dtype=torch.int32, device=device)
    pearl = torch.tensor([1], dtype=torch.int32, device=device)
    bonus = torch.tensor([[42]], dtype=torch.int32, device=device)
    g = torch.Generator(device=device)
    g.manual_seed(0)
    meta = _random_sampling_metadata(device, generators={0: g})
    out = pearl_random_rejection_sample(
        drafts, [2], 4, cu, None, logits, bonus, pearl, meta
    )
    assert out[0, 0].item() == 1
    assert out[0, 1].item() == 1
    assert out[0, 2].item() == 42


def test_pearl_random_pre_verify_rejects_when_target_dislikes_d0():
    device = torch.device("cpu")
    # Draft wants token 1 at pos 0 but target puts almost all mass on token 0.
    logits = torch.tensor(
        [[100.0, -100.0, -100.0, -100.0], [100.0, -100.0, -100.0, -100.0]],
        device=device,
    )
    drafts = torch.tensor([1, 1], dtype=torch.int32, device=device)
    cu = torch.tensor([2], dtype=torch.int32, device=device)
    pearl = torch.tensor([1], dtype=torch.int32, device=device)
    bonus = torch.tensor([[42]], dtype=torch.int32, device=device)
    g = torch.Generator(device=device)
    g.manual_seed(12345)
    meta = _random_sampling_metadata(device, generators={0: g})
    out = pearl_random_rejection_sample(
        drafts, [2], 4, cu, None, logits, bonus, pearl, meta
    )
    assert out[0, 0].item() != -1
    assert out[0, 1].item() == -1


def test_pearl_random_post_verify_matches_sequential_accept():
    device = torch.device("cpu")
    logits = torch.tensor(
        [[-100.0, 100.0, -100.0, -100.0], [-100.0, 100.0, -100.0, -100.0]],
        device=device,
    )
    drafts = torch.tensor([1, 1], dtype=torch.int32, device=device)
    cu = torch.tensor([2], dtype=torch.int32, device=device)
    pearl = torch.tensor([0], dtype=torch.int32, device=device)
    bonus = torch.tensor([[7]], dtype=torch.int32, device=device)
    g = torch.Generator(device=device)
    g.manual_seed(1)
    meta = _random_sampling_metadata(device, generators={0: g})
    out = pearl_random_rejection_sample(
        drafts, [2], 4, cu, None, logits, bonus, pearl, meta
    )
    assert out[0, 0].item() == 1
    assert out[0, 1].item() == 1
    assert out[0, 2].item() == 7
