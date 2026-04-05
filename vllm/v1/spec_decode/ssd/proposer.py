# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from typing_extensions import override

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.spec_decode.draft_model import DraftModelProposer
from vllm.v1.spec_decode.ssd.protocol import log_ssd_async_layout_once

logger = init_logger(__name__)


class SsdDraftModelProposer(DraftModelProposer):
    """Draft proposer for :attr:`~vllm.config.SpeculativeConfig.method` ``\"ssd\"``.

    The colocated configuration (default) matches standard draft-model
    speculative decoding. ``ssd_async=true`` requests the dedicated-draft-GPU
    layout from the SSD paper; when the distributed world size does not equal
    ``tensor_parallel_size + 1``, vLLM logs a warning and continues with the
    colocated path until async IPC is fully integrated.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device,
        runner=None,
    ):
        spec = vllm_config.speculative_config
        assert spec is not None
        if spec.ssd_async:
            try:
                import torch.distributed as dist

                if dist.is_available() and dist.is_initialized():
                    log_ssd_async_layout_once(
                        tensor_parallel_size=vllm_config.parallel_config.tensor_parallel_size,
                        distributed_world_size=dist.get_world_size(),
                        ssd_async_requested=True,
                    )
                else:
                    logger.warning_once(
                        "SSD async is set but torch.distributed is not initialized; "
                        "using colocated draft execution.",
                        scope="local",
                    )
            except Exception:
                logger.warning_once(
                    "SSD async layout check failed; using colocated draft execution.",
                    scope="local",
                )
        super().__init__(vllm_config=vllm_config, device=device, runner=runner)

    @override
    def _raise_if_draft_tp_mismatch(self):
        spec_cfg = self.speculative_config
        tgt_tp = spec_cfg.target_parallel_config.tensor_parallel_size
        draft_tp = spec_cfg.draft_parallel_config.tensor_parallel_size
        if draft_tp == tgt_tp:
            return
        if spec_cfg.ssd_async and draft_tp == 1:
            # Dedicated draft GPU in the reference engine uses draft TP=1 while
            # target uses TP=N. v1 still runs the draft on the speculative PP
            # rank only; full correctness requires the async worker layout.
            logger.warning_once(
                "SSD is using draft_tensor_parallel_size=1 with target TP=%s. "
                "Ensure dedicated-draft multi-rank support is active for your "
                "deployment; otherwise set draft_tensor_parallel_size equal to "
                "tensor_parallel_size.",
                tgt_tp,
                scope="local",
            )
            return
        super()._raise_if_draft_tp_mismatch()
