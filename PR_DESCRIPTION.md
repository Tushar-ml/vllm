<!-- markdownlint-disable -->
PLEASE FILL IN THE PR DESCRIPTION HERE ENSURING ALL CHECKLIST ITEMS (AT THE BOTTOM) HAVE BEEN CONSIDERED.

## Purpose

Fix Gemma4 attention, MTP speculative decoding, and `GEMMA4_FLASH_ATTN` backend selection so sliding (head_dim=256) and full-attention (global_head_dim=512) layers always share a **single** attention backend and KV cache layout.

**Problems addressed**

- **Mixed backends:** `Gemma4Attention` always registered `Gemma4FlashAttentionBackend`, while full-attention layers could fall back to TRITON when FlashAttention could not run `head_size=512`. Per-layer backend mismatch caused numerical divergence and corrupted generations.
- **Missing V RMS norm on TRITON:** Value normalization was only applied in `Gemma4FlashAttentionImpl.do_kv_cache_update`. When config forced TRITON (e.g. Blackwell + FA4 TMEM limits on large head dims), V was never normalized → garbage output.
- **MTP / draft mismatch:** Draft MTP layers did not mirror the target model’s attention backend; the proposer cleared `attention_config.backend` for the draft model.
- **Incorrect FA head-size checks:** `FlashAttentionBackend.supports_head_size()` did not respect runtime FA version fallbacks (e.g. FA4 → FA2 on Blackwell for `head_size > 128`).

**Changes**

- `Gemma4Config`: Force `TRITON_ATTN` only when no attention backend is set and Flash cannot support `max_head_dim`; raise if user explicitly requests `FLASH_ATTN` / `GEMMA4_FLASH_ATTN` on an incompatible build (no silent fallback). H100 + FA4 keeps the requested Flash backend.
- `Gemma4Attention` / `Gemma4MTPAttention`: Use `Gemma4FlashAttentionBackend` only when a non-Triton backend is active; apply `v_norm` in `forward` (aligned with Gemma3n).
- `Gemma4FlashAttentionImpl`: Remove duplicate V-norm in KV cache update (handled in model forward).
- `Gemma4Proposer`: Propagate target attention backend to draft config for KV-shared MTP layers.
- `flash_attn.py`, `fa_utils.py`, `flash_attn_interface.py`: Correct FA version selection and `supports_head_size()`; enable FA4 on SM120+.

## Test Plan

**Unit test**

```bash
pytest tests/v1/attention/test_attention_backends.py::test_flash_attn_supports_head_size_matches_runtime_fa_version -q
pytest tests/model_executor/models/test_gemma4_config.py -q
```

**Server (Gemma4 + MTP speculative decoding)**

```bash
vllm serve google/gemma-4-31B-it \
  --port 8000 \
  --host 0.0.0.0 \
  --speculative-config='{"model": "google/gemma-4-31B-it-assistant", "num_speculative_tokens": 4}' \
  --attention-backend=GEMMA4_FLASH_ATTN \
  --language-model-only \
  --max-model-len=32000 \
  --attention-config.flash_attn_version 4
```

**Client smoke tests**

```bash
# Non-streaming
python -c "
from openai import OpenAI
c = OpenAI(base_url='http://localhost:8000/v1', api_key='empty')
r = c.chat.completions.create(
    model='google/gemma-4-31B-it',
    messages=[{'role': 'user', 'content': 'Say hello in one sentence.'}],
    max_tokens=32, temperature=0.0)
print(r.choices[0].message.content)
"

# Streaming (longer prompt)
python test.py
```

**Platform note:** On H100 (SM90) with FA4, `GEMMA4_FLASH_ATTN` is kept when `supports_head_size(512)` is true. On Blackwell (SM12.0), the same command fails at startup with a clear error; use `--attention-backend=TRITON_ATTN` or omit `--attention-backend` to auto-select TRITON.

## Test Result

**Unit tests:** `5 passed` (head-size + `test_gemma4_config.py`: no silent fallback when backend is explicit)

**Environment:** NVIDIA RTX PRO 6000 Blackwell Server Edition (compute capability 12.0)

**Before (mixed backend / missing V norm on TRITON fallback)**

- Short completion: `' {s}\n { la.'` or repetitive garbage (`s's's's...`)
- Long streaming prompt: incoherent repeated tokens

**After**

- Blackwell + explicit `GEMMA4_FLASH_ATTN`: startup `ValueError` (no silent TRITON fallback).
- Blackwell with `--attention-backend=TRITON_ATTN` or default backend: `INFO ... Using TRITON_ATTN backend.` and coherent output.
- H100 + `GEMMA4_FLASH_ATTN`: uses Flash backend (no TRITON override).

- Short completion (`Say hello in one sentence.`, `max_tokens=32`, `temperature=0.0`):

  ```
  Hello!
  ```

- Streaming article prompt (`test.py`): coherent multi-paragraph output on AI in the workplace with speculative MTP (`num_speculative_tokens=4`).

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x] The test plan, such as providing test command.
- [x] The test results, such as pasting the results comparison before and after, or e2e results
- [ ] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
</details>

**BEFORE SUBMITTING, PLEASE READ <https://docs.vllm.ai/en/latest/contributing>** (anything written below this line will be removed by GitHub Actions)
