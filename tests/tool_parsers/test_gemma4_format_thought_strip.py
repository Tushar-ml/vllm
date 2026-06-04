# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Standalone tests for Gemma ``thought`` glued-shard stripping.

Loads ``gemma4_format.py`` directly (no ``import vllm``).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_format():
    path = Path(__file__).resolve().parents[2] / "vllm" / "tool_parsers" / "gemma4_format.py"
    spec = importlib.util.spec_from_file_location("gemma4_format_under_test", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


_mod = None


def _fmt():
    global _mod  # noqa: PLW0603
    if _mod is None:
        _mod = _load_format()
    return _mod


def strip(s: str) -> str:
    return _fmt().strip_leaked_empty_thinking(s)


def test_many_glued_odd_and_truncated_suffix():
    core = "".join(("thought",) * 11) + "tho"
    assert strip(core).strip() == ""


def test_pairs_then_partial_only():
    assert strip("thoughtthoughtthoughttho").strip() == ""


def test_glued_then_real_text():
    assert strip("thoughtthoughtHello") == "thoughtthoughtHello"


def test_thoughtthoughtful_untouched():
    assert strip("thoughtthoughtful") == "thoughtthoughtful"


def test_mid_sentence_thoughtful():
    s = "That was thoughtful of you."
    assert strip(s) == s


def test_multiline_only_garbage_lines():
    lines = ("\n".join(("thoughtthought",) * 3)) + "\n"
    assert strip(lines).strip() == ""
