# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SSD async protocol specification (reference: ``SpeculatorAsync`` / ``DraftRunner``).

The external engine uses ``torch.distributed`` P2P sends between target ranks
and a dedicated draft rank. This module documents buffer shapes and provides
layout validation helpers for a future v1 multi-rank draft worker.

Handshake payload (target → draft), per batch of size ``B`` (see
``ssd/engine/speculator_async.py``):

* ``cmd``: ``int64[1]`` command code.
* ``meta``: ``int64[3]`` = ``[B, K, async_fan_out]``.
* ``cache_keys``: ``int64[B, 3]`` — ``(seq_id, last_spec_step_accepted_len-1,
  recovery_token_id)``.
* ``num_tokens``: ``int64[B]`` sequence lengths on draft side.
* ``block_tables``: ``int64[B, max_blocks]`` paged KV indices (-1 padding).
* ``temperatures``: ``int32`` reinterpreted as ``int64`` burst (draft temps).
* Optional EAGLE: recovery hidden states ``[B, hidden]``, then
  ``extend_counts``, ``extend_eagle_acts``, ``extend_token_ids``.

Response (draft → target):

* ``fused_response``: ``int64[B + B*K]`` — first ``B`` entries cache hit flags,
  remainder draft token proposals ``[B, K]``.
* ``logits_q``: draft dtype ``[B, K, vocab_size]``.

v1 does not yet run a separate draft process; use
:func:`ssd_async_layout_valid` to detect when the cluster could match the
reference layout.
"""

from __future__ import annotations

from dataclasses import dataclass

from vllm.logger import init_logger

logger = init_logger(__name__)


def ssd_async_layout_valid(
    tensor_parallel_size: int, distributed_world_size: int
) -> bool:
    """Return True if ``world_size == TP + 1`` (one rank reserved for draft)."""
    return distributed_world_size == tensor_parallel_size + 1


def log_ssd_async_layout_once(
    *,
    tensor_parallel_size: int,
    distributed_world_size: int,
    ssd_async_requested: bool,
) -> None:
    """Log how SSD async relates to the current distributed layout."""
    if not ssd_async_requested:
        return
    if ssd_async_layout_valid(tensor_parallel_size, distributed_world_size):
        logger.info(
            "SSD async: distributed world_size=%s matches tensor_parallel_size=%s + 1; "
            "dedicated-draft rank layout is eligible (IPC wiring may still be incomplete).",
            distributed_world_size,
            tensor_parallel_size,
        )
    else:
        logger.warning(
            "SSD async requested but world_size=%s != tensor_parallel_size=%s + 1; "
            "using colocated (sync) draft behavior. For paper-faithful SSD, launch "
            "one extra process/GPU for the draft worker.",
            distributed_world_size,
            tensor_parallel_size,
        )


@dataclass(frozen=True)
class SsdAsyncTensorShapes:
    """Expected tensor ranks for one speculate round (batch ``B``)."""

    batch_size: int
    k: int
    vocab_size: int
    max_blocks: int
    hidden_size: int | None = None

    def logits_q_shape(self) -> tuple[int, int, int]:
        return (self.batch_size, self.k, self.vocab_size)

    def fused_response_numel(self) -> int:
        return self.batch_size + self.batch_size * self.k
