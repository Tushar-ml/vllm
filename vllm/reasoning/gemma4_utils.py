# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.

"""Gemma4 thinking/reasoning output parsing utilities for offline inference.

Standalone functions that parse decoded model text to extract structured
thinking content from Gemma4 models. These are pure-Python utilities with
zero heavy dependencies — they work on raw decoded strings from any inference
backend (vLLM, HuggingFace, TGI, etc.).

For the OpenAI-compatible API reasoning parser (streaming +
non-streaming), see ``vllm.reasoning.gemma4_reasoning_parser``.
For tool call parsing, see ``vllm.tool_parsers.gemma4_utils``.

Usage with vLLM offline inference::

    from vllm import LLM, SamplingParams
    from vllm.reasoning.gemma4_utils import parse_thinking_output

    llm = LLM(model="google/gemma-4-it")
    outputs = llm.generate(prompt, SamplingParams(...))
    text = tokenizer.decode(outputs[0].outputs[0].token_ids, skip_special_tokens=False)

    # Extract thinking / answer (works with or without enable_thinking)
    result = parse_thinking_output(text)
    print(result["thinking"])  # chain-of-thought or None
    print(result["answer"])  # final answer

Aligned with ``vllm.tool_parsers.gemma4_format.extract_reasoning_non_streaming``.
"""

from vllm.tool_parsers.gemma4_format import (
    extract_reasoning_non_streaming,
    strip_thought_label,
)

# Sentinel tokens that may appear in decoded output.
_TURN_END_TAG = "<turn|>"


def parse_thinking_output(text: str) -> dict[str, str | None]:
    """Parse decoded Gemma4 model output into thinking + answer."""
    reasoning, answer = extract_reasoning_non_streaming(text)
    if reasoning is None:
        ans = strip_thought_label(text)
        cleaned = _clean_answer(ans)
        return {"thinking": None, "answer": cleaned if cleaned else None}

    thinking = reasoning.strip()
    cleaned_ans = _clean_answer(answer or "")
    return {
        "thinking": thinking if thinking else "",
        "answer": cleaned_ans if cleaned_ans else None,
    }


def _clean_answer(text: str) -> str:
    """Clean trailing sentinel tokens from the answer text."""
    text = text.strip()
    if text.endswith(_TURN_END_TAG):
        text = text[: -len(_TURN_END_TAG)].rstrip()
    if text.endswith("<eos>"):
        text = text[:-5].rstrip()
    return text
