# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared Gemma4 structured-output parsing.

Aligned with SGLang's Gemma4 detectors for reasoning channels and tool-call blocks.
Used by ``gemma4_tool_parser``, ``gemma4_utils``, and ``gemma4_reasoning_parser``.
"""

from __future__ import annotations

# Reasoning / thinking channel (asymmetric open/close tags).
CHANNEL_START = "<|channel>"
CHANNEL_END = "<channel|>"
# Role label inside the channel after <|channel> (matches HF / SGLang).
THOUGHT_PREFIX = "thought\n"

# Tool calls
TOOL_CALL_START = "<|tool_call>"
TOOL_CALL_END = "<tool_call|>"
STRING_DELIM = '<|"|>'

_TH_WORD = "thought"
_LEN_TH = len(_TH_WORD)
_TH_PAIR = _TH_WORD * 2
_LEN_PAIR = len(_TH_PAIR)

# Longest-first incomplete marker suffixes for streaming (module-level, built once).
_INCOMPLETE_SUFFIXES: tuple[str, ...] = tuple(
    sorted(
        {
            tok[:i]
            for tok in (
                CHANNEL_START,
                CHANNEL_END,
                CHANNEL_START + THOUGHT_PREFIX,
                TOOL_CALL_START,
                TOOL_CALL_END,
                STRING_DELIM,
            )
            for i in range(1, len(tok))
        },
        key=len,
        reverse=True,
    )
)

_EMPTY_THINKING_PATTERNS = (
    CHANNEL_START + THOUGHT_PREFIX + CHANNEL_END,
    CHANNEL_START + "thought\r\n" + CHANNEL_END,
)


def tool_call_markup_start(text: str) -> int:
    """Index where Gemma4 tool-call markup begins, or ``-1``.

    Matches ``<|tool_call>`` or bare ``call:name{`` after the last ``<channel|>``.
    """
    i = text.find(TOOL_CALL_START)
    if i != -1:
        return i
    search_from = 0
    channel_end = text.rfind(CHANNEL_END)
    if channel_end != -1:
        search_from = channel_end + len(CHANNEL_END)
    slice_text = text[search_from:]
    call_rel = slice_text.find("call:")
    if call_rel == -1:
        return -1
    call_abs = search_from + call_rel
    brace = text.find("{", call_abs + 5)
    if brace == -1:
        return -1
    return call_abs


def has_tool_call_markup(text: str) -> bool:
    """True if *text* contains Gemma4 tool-call delimiters or a bare ``call:fn{`` block."""
    if TOOL_CALL_START in text or TOOL_CALL_END in text or STRING_DELIM in text:
        return True
    return tool_call_markup_start(text) != -1


def strip_tool_call_suffix(text: str) -> str:
    """Remove trailing tool-call markup so it is not emitted as client ``content``."""
    idx = tool_call_markup_start(text)
    if idx != -1:
        text = text[:idx]
    return text.rstrip()


def extract_tool_handoff_text(text: str) -> str:
    """Text from the first tool-call marker onward (for reasoning→tool handoff)."""
    idx = tool_call_markup_start(text)
    return text[idx:] if idx != -1 else ""


def _compact_cf_no_ws(core: str, cf_core: str | None = None) -> str:
    """Fold case and drop **all** whitespace (matches ``str.isspace()`` semantics via split)."""
    if cf_core is None:
        cf_core = core.casefold()
    return "".join(cf_core.split())


def _strip_one_thought_shard_line(core: str, nl: str, *, cf_core: str | None = None) -> str:
    compact = _compact_cf_no_ws(core, cf_core)
    if not compact:
        return core + nl

    limit = len(compact)
    i = 0
    while i + _LEN_PAIR <= limit and compact.startswith(_TH_PAIR, i):
        i += _LEN_PAIR
    c = compact[i:]
    lc = len(c)

    if lc == 0:
        return nl

    if c == _TH_WORD:
        return nl

    if lc < _LEN_TH:
        return nl if _TH_WORD.startswith(c) else core + nl

    if lc <= 5 and _TH_WORD.endswith(c) and c != _TH_WORD:
        return core + nl

    tail = c[_LEN_TH:]
    if (
        lc > _LEN_TH
        and tail
        and tail != _TH_WORD
        and c.startswith(_TH_WORD)
        and _TH_WORD.startswith(tail)
    ):
        return nl

    return core + nl


def strip_thought_shard_echoes(text: str) -> str:
    """Strip glued ``thought`` shards (sometimes truncated mid-token at stream cut).

    Models occasionally emit ``thoughtthoughtthought...`` or ``...thoughttho`` as
    plain ``content``. Peel **pairs** of ``thought`` so odd genuine prefixes stay
    tractable, then strip a lone tail ``thought`` or a streaming prefix thereof.

    ``thoughtthoughtful`` remains untouched (after removing pairs the shard ``ful``
    is a suffix of the English word ``thoughtful``, not ``thought`` garbage).
    """
    if not text:
        return text
    cf_full = text.casefold()
    if "thought" not in cf_full:
        return text

    # Hot path: streaming deltas are usually single-line — avoid splitlines copies.
    if "\n" not in text and "\r" not in text:
        return _strip_one_thought_shard_line(text, "", cf_core=cf_full)

    rebuilt: list[str] = []
    for raw_line in text.splitlines(keepends=True):
        nl = ""
        core = raw_line
        if raw_line.endswith("\n"):
            nl = "\n"
            core = raw_line[:-1]

        rebuilt.append(_strip_one_thought_shard_line(core, nl))

    return "".join(rebuilt)


def strip_leaked_empty_thinking(text: str) -> str:
    """Remove echoed empty thinking channels (Gemma4 \"suppress CoT\" pattern).

    The model sometimes emits one or more copies of
    ``<|channel>thought\\n<channel|>`` even with ``enable_thinking: false``.
    Those snippets must not appear in customer-visible ``content``.

    After removing those blocks, **orphaned** ``<|channel>`` / ``<channel|>``
    fragments are stripped as well — the model may emit an extra opening tag
    before ``<|tool_call>``, and streaming used to surface one chunk at a time.
    """
    if not text:
        return text
    if (
        CHANNEL_START not in text
        and CHANNEL_END not in text
        and "thought" not in text.casefold()
    ):
        return text
    s = text
    for p in _EMPTY_THINKING_PATTERNS:
        if p in s:
            s = s.replace(p, "")
    # Remove any remaining channel delimiters (customer-visible text must not
    # contain these control tokens; they only appear in raw model format).
    if CHANNEL_START in s or CHANNEL_END in s:
        while True:
            old = s
            s = s.replace(CHANNEL_START, "").replace(CHANNEL_END, "")
            s = s.strip()
            if s == old:
                break
    return strip_thought_shard_echoes(s)


def _finalize_client_content(text: str | None) -> str | None:
    after = strip_leaked_empty_thinking(text) if text is not None else None
    if not after:
        return None
    after = strip_tool_call_suffix(after)
    return after if after else None


def strip_thought_label(text: str) -> str:
    """Strip ``thought\\n`` from the start of reasoning body text."""
    if text.startswith(THOUGHT_PREFIX):
        return text[len(THOUGHT_PREFIX) :]
    return text


def extract_reasoning_non_streaming(model_output: str) -> tuple[str | None, str | None]:
    """Split Gemma4 assistant output into reasoning and visible content.

    Matches SGLang ``Gemma4Detector.detect_and_parse`` semantics:

    - Without ``<|channel>``, the full string is content (no reasoning).
    - Optional ``thought\\n`` immediately after ``<|channel>`` is stripped.
    - Text before the first ``<|channel>`` is prepended to content after ``<channel|>``.
    - If ``<|tool_call>`` appears before ``<channel|>``, reasoning ends at the tool token.
    """
    if CHANNEL_START not in model_output:
        return None, _finalize_client_content(model_output)

    before_channel, _, after_channel = model_output.partition(CHANNEL_START)
    rest = after_channel
    if rest.startswith(THOUGHT_PREFIX):
        rest = rest[len(THOUGHT_PREFIX) :]

    if CHANNEL_END not in rest:
        if TOOL_CALL_START in rest:
            tool_idx = rest.find(TOOL_CALL_START)
            reasoning_text = rest[:tool_idx].strip()
            normal_text = (before_channel + rest[tool_idx:]).strip()
            return (
                reasoning_text if reasoning_text else None,
                _finalize_client_content(normal_text),
            )
        # Still inside reasoning — no visible answer yet (matches SGLang streaming).
        reasoning_text = rest.strip()
        return reasoning_text if reasoning_text else None, None

    reason_part, _, content_part = rest.partition(CHANNEL_END)
    reasoning_text = reason_part.strip()
    merged_content = before_channel + content_part
    merged_content = merged_content.strip()
    final_content = merged_content if merged_content else None
    final_content = _finalize_client_content(final_content)
    # Mirror tests: empty reasoning block still yields reasoning=""
    return reasoning_text if reasoning_text else "", final_content


def _find_matching_brace(text: str) -> int:
    """Find index of matching ``}`` for args substring starting after ``{``.

    Respects ``STRING_DELIM`` and nesting. Returns ``-1`` if incomplete.
    """
    depth = 1
    i = 0
    n = len(text)
    delim_len = len(STRING_DELIM)
    while i < n and depth > 0:
        if text[i : i + delim_len] == STRING_DELIM:
            i += delim_len
            next_delim = text.find(STRING_DELIM, i)
            if next_delim == -1:
                return -1
            i = next_delim + delim_len
            continue
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    return (i - 1) if depth == 0 else -1


def extract_tool_calls(text: str) -> list[tuple[str, str]]:
    """Extract ``(func_name, raw_args_str)`` pairs using brace-balanced parsing."""
    results: list[tuple[str, str]] = []
    search_from = 0
    while True:
        start = text.find(TOOL_CALL_START, search_from)
        if start == -1:
            break
        end = text.find(TOOL_CALL_END, start)
        if end == -1:
            break
        inner = text[start + len(TOOL_CALL_START) : end]
        if inner.startswith("call:"):
            brace = inner.find("{")
            if brace != -1:
                func_name = inner[5:brace].strip()
                args_content = inner[brace + 1 :]
                match_idx = _find_matching_brace(args_content)
                args_str = (
                    args_content[:match_idx] if match_idx != -1 else args_content
                )
                results.append((func_name, args_str))
        search_from = end + len(TOOL_CALL_END)

    if results:
        return results

    # Bare ``call:name{...}<tool_call|>`` without ``<|tool_call>`` (handoff loss).
    search_from = 0
    while True:
        call_idx = text.find("call:", search_from)
        if call_idx == -1:
            break
        end = text.find(TOOL_CALL_END, call_idx)
        if end == -1:
            break
        func_name, args_str = _parse_call_region(text[call_idx:end])
        if func_name:
            results.append((func_name, args_str))
        search_from = end + len(TOOL_CALL_END)
    return results


def _parse_call_region(partial_call: str) -> tuple[str | None, str]:
    """Parse ``call:name{args...}`` (optional surrounding tool tags already removed)."""
    if TOOL_CALL_END in partial_call:
        partial_call = partial_call.split(TOOL_CALL_END, 1)[0]

    if not partial_call.startswith("call:"):
        return None, ""

    func_part = partial_call[5:]
    if "{" not in func_part:
        return None, ""

    func_name, _, args_tail = func_part.partition("{")
    func_name = func_name.strip()
    match_idx = _find_matching_brace(args_tail)
    if match_idx != -1:
        args_str = args_tail[:match_idx]
    else:
        args_str = args_tail
    return func_name, args_str


def extract_partial_tool_call(current_text: str) -> tuple[str | None, str]:
    """Parse active (possibly incomplete) tool call after last ``TOOL_CALL_START``.

    Returns ``(func_name, raw_args_str)``. ``func_name`` is ``None`` until ``call:name{``
    is complete. ``raw_args_str`` is the interior of ``{...}`` (balanced when complete).

    Falls back to bare ``call:name{`` when ``<|tool_call>`` was dropped on the
    reasoning→tool handoff chunk (``enable_thinking``).
    """
    last_start = current_text.rfind(TOOL_CALL_START)
    if last_start != -1:
        partial_call = current_text[last_start + len(TOOL_CALL_START) :]
        return _parse_call_region(partial_call)

    call_idx = tool_call_markup_start(current_text)
    if call_idx == -1:
        return None, ""

    return _parse_call_region(current_text[call_idx:])
def _parse_gemma4_value(value_str: str) -> object:
    value_str = value_str.strip()
    if not value_str:
        return value_str

    if value_str == "true":
        return True
    if value_str == "false":
        return False

    if value_str.lower() in ("null", "none", "nil"):
        return None

    try:
        if "." in value_str:
            return float(value_str)
        return int(value_str)
    except ValueError:
        pass

    return value_str


def _parse_gemma4_args(args_str: str, *, partial: bool = False) -> dict:
    if not args_str or not args_str.strip():
        return {}

    result: dict = {}
    i = 0
    n = len(args_str)

    while i < n:
        while i < n and args_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break

        key_start = i
        while i < n and args_str[i] != ":":
            i += 1
        if i >= n:
            break
        key = args_str[key_start:i].strip()
        i += 1

        if i >= n:
            if not partial:
                result[key] = ""
            break

        while i < n and args_str[i] in (" ", "\n", "\t"):
            i += 1
        if i >= n:
            if not partial:
                result[key] = ""
            break

        if args_str[i:].startswith(STRING_DELIM):
            i += len(STRING_DELIM)
            val_start = i
            end_pos = args_str.find(STRING_DELIM, i)
            if end_pos == -1:
                result[key] = args_str[val_start:]
                break
            result[key] = args_str[val_start:end_pos]
            i = end_pos + len(STRING_DELIM)

        elif args_str[i] == "{":
            depth = 1
            obj_start = i + 1
            i += 1
            while i < n and depth > 0:
                if args_str[i:].startswith(STRING_DELIM):
                    i += len(STRING_DELIM)
                    next_delim = args_str.find(STRING_DELIM, i)
                    i = n if next_delim == -1 else next_delim + len(STRING_DELIM)
                    continue
                if args_str[i] == "{":
                    depth += 1
                elif args_str[i] == "}":
                    depth -= 1
                i += 1
            if depth > 0:
                result[key] = _parse_gemma4_args(args_str[obj_start:i], partial=True)
            else:
                result[key] = _parse_gemma4_args(args_str[obj_start : i - 1])

        elif args_str[i] == "[":
            depth = 1
            arr_start = i + 1
            i += 1
            while i < n and depth > 0:
                if args_str[i:].startswith(STRING_DELIM):
                    i += len(STRING_DELIM)
                    next_delim = args_str.find(STRING_DELIM, i)
                    i = n if next_delim == -1 else next_delim + len(STRING_DELIM)
                    continue
                if args_str[i] == "[":
                    depth += 1
                elif args_str[i] == "]":
                    depth -= 1
                i += 1
            if depth > 0:
                result[key] = _parse_gemma4_array(args_str[arr_start:i], partial=True)
            else:
                result[key] = _parse_gemma4_array(args_str[arr_start : i - 1])

        else:
            val_start = i
            while i < n and args_str[i] not in (",", "}", "]"):
                i += 1
            if partial and i >= n:
                break
            result[key] = _parse_gemma4_value(args_str[val_start:i])

    return result


def _parse_gemma4_array(arr_str: str, *, partial: bool = False) -> list:
    items: list = []
    i = 0
    n = len(arr_str)

    while i < n:
        while i < n and arr_str[i] in (" ", ",", "\n", "\t"):
            i += 1
        if i >= n:
            break

        if arr_str[i:].startswith(STRING_DELIM):
            i += len(STRING_DELIM)
            end_pos = arr_str.find(STRING_DELIM, i)
            if end_pos == -1:
                items.append(arr_str[i:])
                break
            items.append(arr_str[i:end_pos])
            i = end_pos + len(STRING_DELIM)

        elif arr_str[i] == "{":
            depth = 1
            obj_start = i + 1
            i += 1
            while i < n and depth > 0:
                if arr_str[i:].startswith(STRING_DELIM):
                    i += len(STRING_DELIM)
                    nd = arr_str.find(STRING_DELIM, i)
                    i = nd + len(STRING_DELIM) if nd != -1 else n
                    continue
                if arr_str[i] == "{":
                    depth += 1
                elif arr_str[i] == "}":
                    depth -= 1
                i += 1
            if depth > 0:
                items.append(_parse_gemma4_args(arr_str[obj_start:i], partial=True))
            else:
                items.append(_parse_gemma4_args(arr_str[obj_start : i - 1]))

        elif arr_str[i] == "[":
            depth = 1
            sub_start = i + 1
            i += 1
            while i < n and depth > 0:
                if arr_str[i:].startswith(STRING_DELIM):
                    i += len(STRING_DELIM)
                    next_delim = arr_str.find(STRING_DELIM, i)
                    i = next_delim + len(STRING_DELIM) if next_delim != -1 else n
                    continue
                if arr_str[i] == "[":
                    depth += 1
                elif arr_str[i] == "]":
                    depth -= 1
                i += 1
            if depth > 0:
                items.append(_parse_gemma4_array(arr_str[sub_start:i], partial=True))
            else:
                items.append(_parse_gemma4_array(arr_str[sub_start : i - 1]))

        else:
            val_start = i
            while i < n and arr_str[i] not in (",", "]"):
                i += 1
            if partial and i >= n:
                break
            items.append(_parse_gemma4_value(arr_str[val_start:i]))

    return items


def strip_trailing_incomplete_token(text: str) -> str:
    """Trim suffix that may be an incomplete Gemma4 marker (streaming-safe)."""
    if not text:
        return text
    longest = 0
    for pref in _INCOMPLETE_SUFFIXES:
        if text.endswith(pref) and len(pref) > longest:
            longest = len(pref)
    return text[:-longest] if longest else text


def _norm_snapshot_part(x: str | None) -> str:
    return "" if x is None else x


def diff_reasoning_streaming_snapshots(
    curr: tuple[str | None, str | None],
    prev: tuple[str | None, str | None],
) -> tuple[str, str]:
    """Return ``(reasoning_delta, content_delta)`` from two parse snapshots."""
    nr, nc = _norm_snapshot_part(curr[0]), _norm_snapshot_part(curr[1])
    nr_prev, nc_prev = _norm_snapshot_part(prev[0]), _norm_snapshot_part(prev[1])
    dr = nr[len(nr_prev) :] if nr.startswith(nr_prev) else nr
    dc = nc[len(nc_prev) :] if nc.startswith(nc_prev) else nc
    return dr, dc

