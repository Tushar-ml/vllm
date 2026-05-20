# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Sequence
from typing import TYPE_CHECKING

from vllm.entrypoints.openai.engine.protocol import DeltaMessage
from vllm.reasoning.basic_parsers import BaseThinkingReasoningParser
from vllm.tokenizers import TokenizerLike
from vllm.tool_parsers.gemma4_format import (
    diff_reasoning_streaming_snapshots,
    extract_reasoning_non_streaming,
    strip_trailing_incomplete_token,
)

if TYPE_CHECKING:
    from vllm.entrypoints.openai.chat_completion.protocol import (
        ChatCompletionRequest,
    )
    from vllm.entrypoints.openai.responses.protocol import ResponsesRequest


class Gemma4ReasoningParser(BaseThinkingReasoningParser):
    """
    Reasoning parser for Google Gemma4 thinking models.

    Gemma4 uses <|channel>...<channel|> tokens to delimit reasoning/thinking
    content. The ``thought\\n`` role label after ``<|channel>`` is stripped.

    Parsing semantics align with SGLang's ``Gemma4Detector`` (including
    streaming-safe handling of incomplete trailing markers).
    """

    def __init__(self, tokenizer: TokenizerLike, *args, **kwargs):
        super().__init__(tokenizer, *args, **kwargs)
        self.new_turn_token_id = self.vocab["<|turn>"]
        self.tool_call_token_id = self.vocab["<|tool_call>"]
        self.tool_response_token_id = self.vocab["<|tool_response>"]
        # Streaming cache (valid only while previous_text == last current_text).
        self._stream_safe_prev: str | None = None
        self._stream_prev_snapshot: tuple[str | None, str | None] | None = None
        self._last_current_text: str | None = None

    def adjust_request(
        self, request: "ChatCompletionRequest | ResponsesRequest"
    ) -> "ChatCompletionRequest | ResponsesRequest":
        """Disable special-token stripping to preserve boundary tokens."""
        request.skip_special_tokens = False
        return request

    @property
    def start_token(self) -> str:
        return "<|channel>"

    @property
    def end_token(self) -> str:
        return "<channel|>"

    def is_reasoning_end(self, input_ids: Sequence[int]) -> bool:
        start_token_id = self.start_token_id
        end_token_id = self.end_token_id
        new_turn_token_id = self.new_turn_token_id
        tool_call_token_id = self.tool_call_token_id
        tool_response_token_id = self.tool_response_token_id

        for i in range(len(input_ids) - 1, -1, -1):
            if input_ids[i] == start_token_id:
                return False
            if input_ids[i] == tool_call_token_id:
                return True
            if input_ids[i] in (new_turn_token_id, tool_response_token_id):
                return False
            if input_ids[i] == end_token_id:
                return True
        return False

    def extract_reasoning(
        self,
        model_output: str,
        request: "ChatCompletionRequest | ResponsesRequest",
    ) -> tuple[str | None, str | None]:
        """Extract reasoning using SGLang-aligned rules (requires ``<|channel>``)."""
        return extract_reasoning_non_streaming(model_output)

    def extract_reasoning_streaming(
        self,
        previous_text: str,
        current_text: str,
        delta_text: str,
        previous_token_ids: Sequence[int],
        current_token_ids: Sequence[int],
        delta_token_ids: Sequence[int],
    ) -> DeltaMessage | None:
        """Streaming deltas via snapshots on incomplete-marker-safe prefixes."""
        del delta_text, previous_token_ids, current_token_ids, delta_token_ids

        safe_curr = strip_trailing_incomplete_token(current_text)
        # Only reuse cache for the same stream (not another choice / request).
        same_stream = (
            self._last_current_text is not None
            and previous_text == self._last_current_text
        )
        if same_stream and safe_curr == self._stream_safe_prev:
            return None

        safe_prev = strip_trailing_incomplete_token(previous_text)
        if (
            same_stream
            and self._stream_safe_prev is not None
            and safe_prev == self._stream_safe_prev
            and self._stream_prev_snapshot is not None
        ):
            prev_snapshot = self._stream_prev_snapshot
        else:
            prev_snapshot = extract_reasoning_non_streaming(safe_prev)

        curr_snapshot = extract_reasoning_non_streaming(safe_curr)
        self._stream_safe_prev = safe_curr
        self._stream_prev_snapshot = curr_snapshot
        self._last_current_text = current_text

        dr, dc = diff_reasoning_streaming_snapshots(curr_snapshot, prev_snapshot)

        # Prefer reasoning deltas first — never emit both in one message.
        if dr:
            return DeltaMessage(reasoning=dr)
        if dc:
            return DeltaMessage(content=dc if dc else None)
        return None
