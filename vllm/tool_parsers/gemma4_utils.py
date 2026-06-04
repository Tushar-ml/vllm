# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.

"""Gemma4 tool call parsing utilities for offline inference.

Standalone functions that parse decoded model text to extract tool calls
from Gemma4 models. These are pure-Python utilities with zero heavy
dependencies — they work on raw decoded strings from any inference
backend (vLLM, HuggingFace, TGI, etc.).

For the OpenAI-compatible API server tool parser (streaming +
non-streaming), see ``vllm.tool_parsers.gemma4_tool_parser``.
For thinking/reasoning output parsing, see
``vllm.reasoning.gemma4_utils``.

Usage with vLLM offline inference::

    from vllm import LLM, SamplingParams
    from vllm.tool_parsers.gemma4_utils import (
        parse_tool_calls,
        has_tool_response_tag,
    )

    llm = LLM(model="google/gemma-4-it")
    outputs = llm.generate(prompt, SamplingParams(...))
    text = tokenizer.decode(outputs[0].outputs[0].token_ids, skip_special_tokens=False)

    # Extract tool calls
    tool_calls = parse_tool_calls(text)
    for tc in tool_calls:
        print(f"{tc['name']}({tc['arguments']})")
"""

import regex as re

from vllm.tool_parsers.gemma4_format import (
    _parse_gemma4_args,
    extract_tool_calls as extract_tool_calls_balanced,
)

_TOOL_RESPONSE_START_TAG = "<|tool_response>"


def parse_tool_calls(text: str, *, strict: bool = False) -> list[dict]:
    """Parse tool calls from decoded Gemma4 model output.

    Tier 1 uses brace-balanced extraction (same as the OpenAI tool parser).

    Tier 2 (when ``strict=False``): regex fallback for alternate terminators
    or bare ``call:name{args}`` patterns from multimodal / fragmented outputs.
    """
    results: list[dict] = []

    for name, args_str in extract_tool_calls_balanced(text):
        results.append(
            {
                "name": name,
                "arguments": _parse_gemma4_args(args_str),
            }
        )

    if not results and not strict:
        standard_pattern = (
            r"<\|tool_call\>call:([\w\-\.]+)\{(.*?)\}(?:<tool_call\|>|<turn\|>)"
        )
        for match in re.finditer(standard_pattern, text, re.DOTALL):
            name, args_str = match.group(1), match.group(2)
            results.append({"name": name, "arguments": _parse_gemma4_args(args_str)})

    if results or strict:
        return results

    fallback_pattern = r"(?:<call>|(?:^|\s)call:)([\w\-\.]+)\{(.*?)\}"
    for match in re.finditer(fallback_pattern, text, re.DOTALL):
        name, args_str = match.group(1), match.group(2)
        results.append({"name": name, "arguments": _parse_gemma4_args(args_str)})

    return results


def has_tool_response_tag(text: str) -> bool:
    """Return True if decoded output ends with ``<|tool_response>``."""
    stripped = text.rstrip()
    return stripped.endswith(_TOOL_RESPONSE_START_TAG)
