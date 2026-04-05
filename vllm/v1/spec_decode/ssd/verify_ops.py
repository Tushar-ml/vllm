# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SSD verification math (ported from ``ssd/utils/verify.py``).

This module is self-contained PyTorch—no dependency on the external ``ssd``
package. It implements the acceptance and recovery-token logic used by the
reference ``Verifier`` in https://github.com/tanishqkumar/ssd .
"""

from __future__ import annotations

import torch


def apply_sampler_x_rescaling(
    probs: torch.Tensor, sampler_x: float, fan_out: int
) -> torch.Tensor:
    """Rescale top-(F+1) mass by ``sampler_x`` and renormalize (SSD reference).

    Args:
        probs: ``[B, S, V]`` probabilities.
        sampler_x: Multiplicative factor on selected entries.
        fan_out: ``F`` in the reference code; topk uses ``F + 1`` indices.
    """
    _, topk_indices = torch.topk(probs, fan_out + 1, dim=-1)
    topf_mask = torch.zeros_like(probs, dtype=torch.bool)
    topf_mask.scatter_(dim=-1, index=topk_indices, value=True)
    probs = torch.where(topf_mask, probs * sampler_x, probs)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs


def ssd_verify(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    speculations: torch.Tensor,
    temperatures_target: torch.Tensor,
    temperatures_draft: torch.Tensor,
    cache_hits: torch.Tensor | None = None,
    sampler_x: float | None = None,
    async_fan_out: int | None = None,
    jit_speculate: bool = False,
) -> tuple[list[list[int]], list[int]]:
    """Run SSD speculative verification (reference semantics).

    Args:
        logits_p: Target logits ``[B, K+1, V]``.
        logits_q: Draft logits ``[B, K, V]`` aligned with speculative tokens.
        speculations: Long tensor ``[B, K+1]``; column 0 is the recovery/prefix
            token per row, columns ``1:`` are draft proposals.
        temperatures_target: Float ``[B]`` per-sequence target temperature
            (0 => greedy).
        temperatures_draft: Float ``[B]`` draft temperature (may differ).
        cache_hits: Optional ``[B]`` int/bool; when not ``jit_speculate``,
            ratio acceptance applies only where cache hits are true.
        sampler_x: Optional draft rescaling (requires ``async_fan_out``).
        async_fan_out: Fan-out ``F`` for ``apply_sampler_x_rescaling``.
        jit_speculate: If True, ratio rows ignore cache hit mask (reference).

    Returns:
        ``(accepted_suffixes, recovery_tokens)`` where each suffix is
        ``[recovery, draft_0, ..., draft_{n-1}]`` accepted tokens and
        recovery is the next sampled token from the target.
    """
    device = logits_p.device
    B, Kp1, _V = logits_p.shape
    K = Kp1 - 1

    draft_tokens = speculations[:, 1:]
    preds_p = logits_p.argmax(dim=-1)
    matches = draft_tokens == preds_p[:, :-1]
    any_mismatch = (~matches).any(dim=1)
    first_mismatch = (~matches).int().argmax(dim=1)
    accept_greedy = torch.where(
        any_mismatch,
        first_mismatch,
        torch.full_like(first_mismatch, K),
    )
    batch_idx = torch.arange(B, device=device)
    rec_greedy = preds_p[batch_idx, accept_greedy]

    temps_t = temperatures_target
    temps_q = temperatures_draft
    base_ratio_rows = (temps_t > 0) | (temps_q > 0)
    if jit_speculate:
        ratio_rows = base_ratio_rows
    else:
        if cache_hits is not None:
            ratio_rows = base_ratio_rows & cache_hits.to(torch.bool)
        else:
            ratio_rows = base_ratio_rows & torch.zeros(
                B, dtype=torch.bool, device=device
            )

    do_any_ratio = ratio_rows.any().item()
    need_p_probs = (temps_t > 0).any().item() or do_any_ratio

    probs_p: torch.Tensor | None = None
    if need_p_probs:
        probs_p = torch.zeros(B, Kp1, _V, device=device, dtype=torch.float32)
        nz_p = temps_t > 0
        if nz_p.any():
            t = temps_t[nz_p].unsqueeze(1).unsqueeze(2).clamp(min=1e-8)
            probs_p[nz_p] = torch.softmax(
                (logits_p[nz_p] / t).to(torch.float32), dim=-1
            )
        z_p = ~nz_p
        if z_p.any():
            argmax_p = logits_p[z_p].argmax(dim=-1)
            one_hot_p = torch.zeros_like(logits_p[z_p], dtype=torch.float32)
            one_hot_p.scatter_(2, argmax_p.unsqueeze(-1), 1.0)
            probs_p[z_p] = one_hot_p

    if do_any_ratio:
        probs_q = torch.zeros(B, K, _V, device=device, dtype=torch.float32)
        nz_q = temps_q > 0
        if nz_q.any():
            tq = temps_q[nz_q].unsqueeze(1).unsqueeze(2).clamp(min=1e-8)
            probs_q[nz_q] = torch.softmax(
                (logits_q[nz_q] / tq).to(torch.float32), dim=-1
            )
        z_q = ~nz_q
        if z_q.any():
            argmax_q = logits_q[z_q].argmax(dim=-1)
            one_hot_q = torch.zeros_like(logits_q[z_q], dtype=torch.float32)
            one_hot_q.scatter_(2, argmax_q.unsqueeze(-1), 1.0)
            probs_q[z_q] = one_hot_q
        if sampler_x is not None:
            assert async_fan_out is not None, (
                "async_fan_out must be provided if sampler_x is provided"
            )
            probs_q = apply_sampler_x_rescaling(
                probs_q, sampler_x, async_fan_out
            )
        p_all = (
            probs_p[:, :K, :]
            if probs_p is not None
            else torch.zeros(B, K, _V, device=device, dtype=torch.float32)
        )
        q_all = probs_q
        gather_idx = draft_tokens.unsqueeze(2)
        p_vals = p_all.gather(2, gather_idx).squeeze(2)
        q_vals = q_all.gather(2, gather_idx).squeeze(2)
        accept_probs = (p_vals / (q_vals + 1e-10)).clamp(max=1.0)
        rand = torch.rand_like(accept_probs)
        accepts = rand <= accept_probs
        rej_any = (~accepts).any(dim=1)
        first_rej = (~accepts).int().argmax(dim=1)
        accept_ratio = torch.where(
            rej_any,
            first_rej,
            torch.full_like(first_rej, K),
        )
        accept_until = torch.where(ratio_rows, accept_ratio, accept_greedy)
    else:
        accept_until = accept_greedy

    batch_idx = torch.arange(B, device=device)
    if probs_p is None:
        rec_ratio = rec_greedy
    else:
        p_fallback = probs_p[batch_idx, accept_until]
        p_sum = p_fallback.sum(dim=1, keepdim=True)
        fallback_dist = p_fallback / p_sum
        if do_any_ratio:
            assert probs_p is not None
            q_idx_safe = accept_until.clamp(max=K - 1)
            q_slice = q_all[batch_idx, q_idx_safe]
            mask_adjust = (temps_t > 0) & (accept_until < K) & ratio_rows
            adj = (p_fallback - q_slice).clamp(min=0.0)
            sums = adj.sum(dim=1, keepdim=True)
            adj_norm = torch.where(sums > 0, adj / sums, fallback_dist)
            rec_ratio_adjusted = torch.multinomial(adj_norm, 1).squeeze(1)
            rec_from_p = torch.multinomial(fallback_dist, 1).squeeze(1)
            rec_ratio = torch.where(
                mask_adjust, rec_ratio_adjusted, rec_from_p
            )
        else:
            rec_from_p = torch.multinomial(fallback_dist, 1).squeeze(1)
            rec_ratio = rec_from_p

    rec_final = torch.where(temps_t > 0, rec_ratio, rec_greedy)

    starts = speculations[:, 0].tolist()
    counts = accept_until.tolist()
    accepted_suffixes: list[list[int]] = []
    for b in range(B):
        n = counts[b]
        suffix = [starts[b]] + draft_tokens[b, :n].tolist()
        accepted_suffixes.append(suffix)

    return accepted_suffixes, rec_final.tolist()
