# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


class AsyncScheduler(Scheduler):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # reusable read-only placeholder list for speculative decoding.
        self._spec_token_placeholders: list[int] = [-1] * self.num_spec_tokens
        self.pp_size = self.parallel_config.pipeline_parallel_size

    def _update_after_schedule(self, scheduler_output: SchedulerOutput) -> None:
        super()._update_after_schedule(scheduler_output)
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        for req_id in scheduler_output.num_scheduled_tokens:
            request = self.requests[req_id]
            if request.is_prefill_chunk:
                continue

            scheduler_output.pending_structured_output_tokens |= (
                request.use_structured_output and request.num_output_placeholders > 0
            )
            # The request will generate a new token plus num_spec_tokens
            # in this scheduling step.
            cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
            request.num_output_placeholders += 1 + cur_num_spec_tokens
            # Add placeholders for the new draft/spec tokens.
            # We will update the actual spec token ids in the worker process.
            request.spec_token_ids = self._spec_token_placeholders

    def _release_scheduled_async_placeholders(
        self,
        request: Request,
        scheduler_output: SchedulerOutput,
        req_id: str,
    ) -> None:
        if request.is_prefill_chunk:
            return
        num_scheduled = scheduler_output.num_scheduled_tokens.get(req_id, 0)
        if request.num_output_placeholders <= 0 and num_scheduled <= 0:
            return
        spec_decode_tokens = scheduler_output.scheduled_spec_decode_tokens
        cur_num_spec_tokens = len(spec_decode_tokens.get(req_id, ()))
        if request.num_output_placeholders > 0:
            release = min(request.num_output_placeholders, 1 + cur_num_spec_tokens)
            request.num_output_placeholders -= release
        if num_scheduled > 0:
            request.num_computed_tokens = max(
                0, request.num_computed_tokens - num_scheduled
            )

    def _handle_discarded_async_step(
        self,
        request: Request,
        scheduler_output: SchedulerOutput,
        req_id: str,
    ) -> None:
        self._release_scheduled_async_placeholders(
            request, scheduler_output, req_id
        )

    def _reconcile_async_placeholders(self, request: Request) -> bool:
        if (
            request.is_prefill_chunk
            or request.is_finished()
            or request.num_output_placeholders <= 0
        ):
            return False
        if request.num_computed_tokens >= request.num_tokens:
            logger.debug(
                "Reconciling async placeholders for request %s: "
                "placeholders %d -> 0, computed %d -> %d",
                request.request_id,
                request.num_output_placeholders,
                request.num_computed_tokens,
                request.num_tokens,
            )
            request.num_output_placeholders = 0
            if request.num_computed_tokens > request.num_tokens:
                request.num_computed_tokens = request.num_tokens
            return True
        return False

    def _update_request_with_output(
        self,
        request: Request,
        new_token_ids: list[int],
        scheduler_output: SchedulerOutput | None = None,
        req_id: str | None = None,
    ) -> tuple[list[int], bool]:
        if request.discard_latest_async_tokens:
            # If the request is force preempted in reset_prefix_cache, we
            # should discard the latest async token.
            request.discard_latest_async_tokens = False
            if scheduler_output is not None and req_id is not None:
                self._release_scheduled_async_placeholders(
                    request, scheduler_output, req_id
                )
            return [], False

        status_before_update = request.status
        new_token_ids, stopped = super()._update_request_with_output(
            request, new_token_ids
        )

        # Update the number of output placeholders.
        request.num_output_placeholders -= len(new_token_ids)
        assert request.num_output_placeholders >= 0

        # Cache the new tokens. Preempted requests should be skipped.
        if status_before_update == RequestStatus.RUNNING:
            self.kv_cache_manager.cache_blocks(
                request, request.num_computed_tokens - request.num_output_placeholders
            )
        return new_token_ids, stopped
