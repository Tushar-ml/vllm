# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""SSD async inter-process communication (placeholder).

The reference engine uses ``torch.distributed`` P2P between target tensor-parallel
ranks and a ``DraftRunner`` on the final GPU. A future v1 integration should:

1. Create a process group spanning target ranks plus the draft rank when
   :func:`~vllm.v1.spec_decode.ssd.protocol.ssd_async_layout_valid` is true.
2. Mirror the send/recv ordering in ``ssd/engine/speculator_async.py`` and
   ``ssd/engine/draft_runner.py``.

Colocated execution (default) does not use this module.
"""

from __future__ import annotations


def ssd_async_ipc_available() -> bool:
    """Always false until NCCL P2P draft channel is implemented in v1."""
    return False


def create_ssd_async_process_group() -> None:
    """Reserved: initialize target↔draft P2P group matching ``tanishqkumar/ssd``."""
    raise NotImplementedError(
        "SSD async NCCL P2P is not implemented in vLLM v1 yet; "
        "use ssd_async=false for colocated draft, or run the reference "
        "https://github.com/tanishqkumar/ssd engine."
    )
