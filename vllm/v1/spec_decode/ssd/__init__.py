# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Speculative Speculative Decoding (SSD) hooks for v1.

See :mod:`vllm.v1.spec_decode.ssd.reference` for bibliography and protocol
notes. Use ``speculative_config.method == "ssd"`` with a draft ``model`` id.
"""

from vllm.v1.spec_decode.ssd.algo import effective_ssd_cache_hits_for_verify
from vllm.v1.spec_decode.ssd.async_ipc import (
    create_ssd_async_process_group,
    ssd_async_ipc_available,
)
from vllm.v1.spec_decode.ssd.proposer import SsdDraftModelProposer
from vllm.v1.spec_decode.ssd.protocol import (
    SsdAsyncTensorShapes,
    log_ssd_async_layout_once,
    ssd_async_layout_valid,
)
from vllm.v1.spec_decode.ssd.verify_ops import apply_sampler_x_rescaling, ssd_verify

__all__ = [
    "SsdDraftModelProposer",
    "SsdAsyncTensorShapes",
    "apply_sampler_x_rescaling",
    "create_ssd_async_process_group",
    "effective_ssd_cache_hits_for_verify",
    "log_ssd_async_layout_once",
    "ssd_async_ipc_available",
    "ssd_async_layout_valid",
    "ssd_verify",
]
