# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Train-free speculative drafting from a per-request adjacency matrix (Token Recycling).

This implements the *matrix + online top-k update* core of arXiv:2408.08696 and uses
vLLM's standard chain speculative verification (same path as ngram), not the paper's
BFS draft tree + custom tree attention (which would require the EAGLE tree backend).

Future (tree parity with the paper): integrate BFS draft construction with
``TreeAttentionMetadata`` / ``vllm.v1.attention.backends.tree_attn`` and the EAGLE
speculative path, reusing draft tree layout and verify kernels—see upstream EAGLE
proposer and ``SpecDecodeMetadata`` tree extensions.
"""

from __future__ import annotations

import os
from collections import OrderedDict

import numpy as np
import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


def _dedupe_key_last(key: np.ndarray) -> np.ndarray:
    """Row indices to keep (last occurrence per key) for deduplication."""
    if len(key) == 0:
        return np.array([], dtype=np.int64)
    order = np.argsort(key, kind="mergesort")
    ks = key[order]
    is_start = np.ones(len(ks), dtype=bool)
    is_start[1:] = ks[1:] != ks[:-1]
    starts = np.flatnonzero(is_start)
    ends = np.concatenate([starts[1:], [len(ks)]]) - 1
    return order[ends].astype(np.int64)


def _first_positive_token_id(row: np.ndarray) -> int | None:
    """First positive entry in a row (0 = empty)."""
    mask = row > 0
    if not np.any(mask):
        return None
    return int(row[np.argmax(mask)])


class _SparseRows:
    """LRU-capped dict[token_id] -> (k,) int32 row."""

    def __init__(
        self,
        k: int,
        max_rows: int,
        fallback: np.ndarray | None = None,
    ):
        self.k = k
        self.max_rows = max_rows
        self.fallback = fallback
        self.rows: OrderedDict[int, np.ndarray] = OrderedDict()

    def set_row(self, tid: int, values: np.ndarray) -> None:
        if tid in self.rows:
            self.rows.move_to_end(tid)
        self.rows[tid] = values.astype(np.int32, copy=False)
        while len(self.rows) > self.max_rows:
            self.rows.popitem(last=False)

    def get_row_view(self, tid: int) -> np.ndarray:
        r = self.rows.get(tid)
        if r is not None:
            self.rows.move_to_end(tid)
            return r
        if self.fallback is not None and 0 <= tid < len(self.fallback):
            fr = self.fallback[tid]
            if np.any(fr > 0):
                r = fr.astype(np.int32, copy=True)
                self.set_row(tid, r)
                return r
        return np.zeros((self.k,), dtype=np.int32)


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
        self._use_sparse = spec.token_recycling_use_sparse_matrix
        smax = spec.token_recycling_sparse_max_rows
        self._sparse_max_rows = smax if smax is not None else (65536 if self._use_sparse else 0)

        # req_id -> dense CPU tensor (vocab, k) OR sparse table
        self._dense_matrices: dict[str, torch.Tensor] = {}
        self._sparse_matrices: dict[str, _SparseRows] = {}
        # Shared numpy view of dense tensor (invalidated only on new tensor)
        self._dense_numpy: dict[str, np.ndarray] = {}

        # Optional read-only bootstrap (numpy), shared across requests; deltas overlay
        self._shared_bootstrap_np: np.ndarray | None = None
        self._per_req_delta: dict[str, dict[int, np.ndarray]] = {}

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
            self._shared_bootstrap_np = (
                shared.to(dtype=torch.int32).contiguous().numpy()
            )
        else:
            self._shared_bootstrap_np = None

        if self._use_sparse and path:
            logger.warning_once(
                "token_recycling_matrix_path with token_recycling_use_sparse_matrix: "
                "sparse tables use the file as a lazy fallback per token id.",
                scope="local",
            )

    def _ensure_dense(self, req_id: str) -> torch.Tensor:
        m = self._dense_matrices.get(req_id)
        if m is None:
            if self._shared_bootstrap_np is not None:
                m = torch.from_numpy(self._shared_bootstrap_np.copy())
            else:
                m = torch.zeros((self.vocab_size, self.k), dtype=torch.int32)
            self._dense_matrices[req_id] = m
            self._dense_numpy[req_id] = m.numpy()
        return m

    def _ensure_sparse(self, req_id: str) -> _SparseRows:
        s = self._sparse_matrices.get(req_id)
        if s is None:
            s = _SparseRows(
                self.k,
                self._sparse_max_rows,
                fallback=self._shared_bootstrap_np,
            )
            self._sparse_matrices[req_id] = s
        return s

    def free_request(self, req_id: str) -> None:
        self._dense_matrices.pop(req_id, None)
        self._dense_numpy.pop(req_id, None)
        self._sparse_matrices.pop(req_id, None)
        self._per_req_delta.pop(req_id, None)

    def update_rows_batched(
        self,
        parent_token_ids: np.ndarray,
        req_indices: np.ndarray,
        topk: np.ndarray,
        req_ids: list[str],
    ) -> None:
        """Vectorized CPU scatter: parent_token_ids [N], req_indices [N], topk [N, k]."""
        if parent_token_ids.shape[0] == 0:
            return
        if self._use_sparse:
            self._update_sparse_batched(parent_token_ids, req_indices, topk, req_ids)
            return
        if self._shared_bootstrap_np is not None:
            self._update_shared_overlay_batched(
                parent_token_ids, req_indices, topk, req_ids
            )
            return
        # Dense path: group by req index
        for rid_int in np.unique(req_indices):
            mask = req_indices == rid_int
            pts = parent_token_ids[mask]
            tk = topk[mask]
            rid = req_ids[int(rid_int)]
            m = self._ensure_dense(rid)
            m_np = self._dense_numpy[rid]
            valid = (pts >= 0) & (pts < self.vocab_size)
            if not np.any(valid):
                continue
            m_np[pts[valid]] = tk[valid]

    def _update_sparse_batched(
        self,
        parent_token_ids: np.ndarray,
        req_indices: np.ndarray,
        topk: np.ndarray,
        req_ids: list[str],
    ) -> None:
        for rid_int in np.unique(req_indices):
            mask = req_indices == rid_int
            pts = parent_token_ids[mask]
            tk = topk[mask]
            rid = req_ids[int(rid_int)]
            sp = self._ensure_sparse(rid)
            for j in range(len(pts)):
                tid = int(pts[j])
                if 0 <= tid < self.vocab_size:
                    sp.set_row(tid, tk[j])

    def _update_shared_overlay_batched(
        self,
        parent_token_ids: np.ndarray,
        req_indices: np.ndarray,
        topk: np.ndarray,
        req_ids: list[str],
    ) -> None:
        assert self._shared_bootstrap_np is not None
        for rid_int in np.unique(req_indices):
            mask = req_indices == rid_int
            pts = parent_token_ids[mask]
            tk = topk[mask]
            rid = req_ids[int(rid_int)]
            delta = self._per_req_delta.setdefault(rid, {})
            for j in range(len(pts)):
                tid = int(pts[j])
                if 0 <= tid < self.vocab_size:
                    delta[tid] = tk[j].astype(np.int32, copy=True)

    def _row_dense(self, req_id: str, tid: int) -> np.ndarray:
        m = self._ensure_dense(req_id)
        return self._dense_numpy[req_id][tid]

    def _row_shared_overlay(self, req_id: str, tid: int) -> np.ndarray:
        assert self._shared_bootstrap_np is not None
        delta = self._per_req_delta.get(req_id)
        if delta is not None and tid in delta:
            return delta[tid]
        return self._shared_bootstrap_np[tid]

    def _row_sparse(self, req_id: str, tid: int) -> np.ndarray:
        return self._ensure_sparse(req_id).get_row_view(tid)

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
            rid = req_ids[i]
            chain: list[int] = []
            cur = last_tok
            for _ in range(self.num_spec_tokens):
                if self._use_sparse:
                    r = self._row_sparse(rid, cur)
                elif self._shared_bootstrap_np is not None:
                    r = self._row_shared_overlay(rid, cur)
                else:
                    r = self._row_dense(rid, cur)
                nxt = _first_positive_token_id(r)
                if nxt is None:
                    break
                chain.append(nxt)
                cur = nxt
            draft_token_ids.append(chain)
        return draft_token_ids

    # Backwards-compatible name for gpu_model_runner (stages batched copy then update_rows_batched)
    def stage_gpu_rows_then_update(
        self,
        parent_token_ids: torch.Tensor,
        req_indices: torch.Tensor,
        topk: torch.Tensor,
        req_ids: list[str],
    ) -> None:
        """Single D2H sync: stack (parent, req_idx, topk) then vectorized scatter."""
        if parent_token_ids.shape[0] == 0:
            return
        n = parent_token_ids.shape[0]
        k = topk.shape[-1]
        buf = torch.empty((n, 2 + k), dtype=torch.int64, device=parent_token_ids.device)
        buf[:, 0] = parent_token_ids.long()
        buf[:, 1] = req_indices.long()
        buf[:, 2:] = topk.long()
        arr = buf.cpu().numpy()
        pts = arr[:, 0].astype(np.int64, copy=False)
        rqi = arr[:, 1].astype(np.int64, copy=False)
        tk = arr[:, 2:].astype(np.int32, copy=False)
        self.update_rows_batched(pts, rqi, tk, req_ids)
