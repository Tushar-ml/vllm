# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Train-free speculative drafting from a per-request adjacency matrix (Token Recycling).

This implements the *matrix + online top-k update* core of arXiv:2408.08696 and uses
vLLM's standard chain speculative verification (same path as ngram), not the paper's
BFS draft tree + custom tree attention (which would require the EAGLE tree backend).
"""

from __future__ import annotations

import os
import numpy as np
import torch

from vllm.config import VllmConfig


def _valid_candidates(row: np.ndarray, k_take: int) -> list[int]:
    """Skip non-positive entries (0 = no candidate in paper-style matrices)."""
    out: list[int] = []
    for x in row.tolist():
        if len(out) >= k_take:
            break
        if x <= 0:
            continue
        out.append(int(x))
    return out


class TokenRecyclingProposer:
    def __init__(self, vllm_config: VllmConfig):
        assert vllm_config.speculative_config is not None
        spec = vllm_config.speculative_config
        self.k = spec.token_recycling_k
        self.num_spec_tokens = spec.num_speculative_tokens
        assert self.num_spec_tokens is not None
        tc = vllm_config.model_config.hf_text_config
        self.vocab_size = int(tc.vocab_size)
        self.max_model_len = vllm_config.model_config.max_model_len
        # req_id -> int32 (vocab, k) on CPU
        self._matrices: dict[str, torch.Tensor] = {}
        path = spec.token_recycling_matrix_path
        if path:
            if not os.path.isfile(path):
                raise FileNotFoundError(
                    f"token_recycling_matrix_path={path!r} does not exist"
                )
            shared = torch.load(path, map_location="cpu", weights_only=False)
            if tuple(shared.shape) != (self.vocab_size, self.k):
                raise ValueError(
                    f"Expected matrix {(self.vocab_size, self.k)}, got {tuple(shared.shape)}"
                )
            self._shared_bootstrap = shared.to(dtype=torch.int32)
        else:
            self._shared_bootstrap = None

    def _ensure_matrix(self, req_id: str) -> torch.Tensor:
        m = self._matrices.get(req_id)
        if m is None:
            if self._shared_bootstrap is not None:
                m = self._shared_bootstrap.clone()
            else:
                m = torch.zeros((self.vocab_size, self.k), dtype=torch.int32)
            self._matrices[req_id] = m
        return m

    def free_request(self, req_id: str) -> None:
        self._matrices.pop(req_id, None)

    def update_rows_gpu(
        self,
        parent_token_ids: torch.Tensor,
        req_indices: torch.Tensor,
        topk_indices: torch.Tensor,
        req_ids: list[str],
    ) -> None:
        """Scatter top-k rows into CPU matrices (caller provides GPU top-k)."""
        parent_token_ids_cpu = parent_token_ids.cpu()
        req_indices_cpu = req_indices.cpu()
        topk_cpu = topk_indices.cpu()
        for i in range(parent_token_ids_cpu.shape[0]):
            rid = req_ids[int(req_indices_cpu[i].item())]
            tid = int(parent_token_ids_cpu[i].item())
            if not (0 <= tid < self.vocab_size):
                continue
            m = self._ensure_matrix(rid)
            m[tid] = topk_cpu[i].to(dtype=torch.int32)

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
        req_ids: list[str],
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> list[list[int]]:
        del slot_mappings
        draft_token_ids: list[list[int]] = []
        for i, sampled_ids in enumerate(sampled_token_ids):
            if not sampled_ids:
                draft_token_ids.append([])
                continue
            num_tokens = int(num_tokens_no_spec[i])
            if num_tokens >= self.max_model_len:
                draft_token_ids.append([])
                continue
            row = token_ids_cpu[i]
            last_tok = int(row[num_tokens - 1])
            if not (0 <= last_tok < self.vocab_size):
                draft_token_ids.append([])
                continue
            m = self._ensure_matrix(req_ids[i]).numpy()
            chain: list[int] = []
            cur = last_tok
            for _ in range(self.num_spec_tokens):
                cand = _valid_candidates(m[cur], self.k)
                if not cand:
                    break
                nxt = cand[0]
                chain.append(nxt)
                cur = nxt
            draft_token_ids.append(chain)
        return draft_token_ids
