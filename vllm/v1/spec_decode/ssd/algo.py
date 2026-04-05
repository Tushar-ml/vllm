# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SSD algorithm integration points for the v1 execution loop.

Greedy (temperature 0) draft-model speculative decoding in v1 already matches
:class:`~vllm.v1.worker.gpu.spec_decode.rejection_sampler.strict_rejection_sample`
semantics, which align with ``ssd_verify`` when all temperatures are zero.

When :attr:`~vllm.config.SpeculativeConfig.ssd_async` is enabled and async IPC
is implemented, :func:`effective_ssd_cache_hits_for_verify` should return a
``[B]`` boolean/int tensor so the rejection stage can call :func:`ssd_verify`
with the same cache-hit gating as the reference ``Verifier``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from vllm.v1.spec_decode.ssd.async_ipc import ssd_async_ipc_available

if TYPE_CHECKING:
    from vllm.config import SpeculativeConfig


def effective_ssd_cache_hits_for_verify(
    speculative_config: "SpeculativeConfig | None",
    batch_size: int,
    device: torch.device,
) -> torch.Tensor | None:
    """Per-request cache-hit flags for SSD ratio acceptance, or ``None`` if N/A."""
    if speculative_config is None or speculative_config.method != "ssd":
        return None
    if not speculative_config.ssd_async or not ssd_async_ipc_available():
        return None
    return torch.ones(batch_size, dtype=torch.int64, device=device)
