<!-- markdownlint-disable -->
PLEASE FILL IN THE PR DESCRIPTION HERE ENSURING ALL CHECKLIST ITEMS (AT THE BOTTOM) HAVE BEEN CONSIDERED.

## Purpose

This PR generalizes Megakernel integration from a Llama-only path to a modular small-model architecture and keeps decode latency as the primary optimization target. It introduces plugin-based family routing (Llama/Qwen/Mistral), centralizes Megakernel configuration, and updates decode/prefill backend wiring to reduce fallback coupling while preserving fail-closed behavior.

### What changed

- Added Megakernel plugin registry and generic resolver dispatch:
  - `llama_small`
  - `qwen_small`
  - `mistral_small`
- Refactored spec checks into shared family-aware validation with strict fail-closed semantics.
- Centralized Megakernel env/config controls in `vllm/envs.py`:
  - `VLLM_MEGAKERNEL_STRICT`
  - `VLLM_MEGAKERNEL_BACKEND` (`mk_llama` or `mirage_mpk`)
  - `VLLM_MEGAKERNEL_FAMILIES`
  - `VLLM_MEGAKERNEL_MAX_LEN`
  - `VLLM_MEGAKERNEL_MAX_BATCH_SIZE`
  - `VLLM_MEGAKERNEL_PARITY_TOKENS`
- Split runtime builder into backend-specific constructors:
  - `mk_llama` (active)
  - `mirage_mpk` (opt-in seam with fail-loud `NotImplementedError` until runtime integration lands)
- Optimized direct `mk_llama` decode hot path:
  - reduced per-step tensor churn
  - reused static buffers
  - reduced host-scalar overhead in token position handling
- Reworked attention backend wrappers:
  - decode path via Megakernel wrapper op
  - prefill path via dedicated Megakernel prefill wrapper op
- Kept/validated CUDA graph mode coverage including `FULL_AND_PIECEWISE`.
- Updated design docs with modular architecture and family support matrix:
  - `docs/design/megakernel_integration_architecture.md`
- Added benchmark harness for small-model matrix runs:
  - `benchmarks/megakernel_small_model_matrix.py`

## Test Plan

1. **Server bring-up and modular routing checks**
   - Start OpenAI-compatible server with:
     - `--model meta-llama/Llama-3.2-1B-Instruct`
     - `--max-model-len 4096`
     - `--max-num-seqs 1`
   - Toggle `VLLM_MEGAKERNEL_ON=1/0`.
   - Validate plugin/family routing with:
     - `VLLM_MEGAKERNEL_FAMILIES=llama_small,qwen_small,mistral_small`
     - `VLLM_MEGAKERNEL_STRICT=1`

2. **Eager benchmark comparison**
   - Run with `--enforce-eager`.
   - Same prompt/token limits for ON vs OFF.
   - Warmup requests before timing.
   - 15 measured `/v1/completions` requests per mode.

3. **Graph benchmark comparison**
   - Run with `-cc.cudagraph_mode=FULL_AND_PIECEWISE`.
   - Same prompt/token limits for ON vs OFF.
   - Warmup requests before timing.
   - 15 measured `/v1/completions` requests per mode.

4. **Functional sanity and backend behavior**
   - Verify non-garbled decode text under both eager and graph modes.
   - Confirm no decode corruption under `FULL_AND_PIECEWISE`.
   - Verify Megakernel family validator pass/fail behavior for supported small-model configs.

5. **Syntax/runtime sanity**
   - `python -m compileall` on changed Megakernel/attention/env files.
   - Validator sanity test for representative Llama/Qwen/Mistral config objects.
   - Mirage opt-in smoke:
     - run with `VLLM_MEGAKERNEL_BACKEND=mirage_mpk`
     - confirm explicit startup failure with clear `NotImplementedError` (expected phase-1 behavior)

## Test Result

### Eager mode (`--enforce-eager`)

- **Megakernel ON**
  - avg: `57.403 ms`
  - p50: `57.895 ms`
  - p95: `58.385 ms`
- **Megakernel OFF**
  - avg: `208.068 ms`
  - p50: `207.813 ms`
  - p95: `210.677 ms`

**Improvement:** `3.625x` speedup, `72.41%` latency reduction.

### CUDA graph mode (`FULL_AND_PIECEWISE`)

- **Megakernel ON**
  - avg: `50.027 ms`
  - p50: `50.037 ms`
  - p95: `50.278 ms`
- **Megakernel OFF**
  - avg: `58.918 ms`
  - p50: `58.750 ms`
  - p95: `59.717 ms`

**Improvement:** `1.178x` speedup, `15.09%` latency reduction.

Graph benchmark artifact: `/tmp/megakernel_full_piecewise_bench.json`.

### Modularization validation results

- `python -m compileall` passed for all modified files.
- Family validator sanity checks passed for representative:
  - `LlamaConfig` (`llama_small`)
  - `Qwen2Config` (`qwen_small`)
  - `MistralConfig` (`mistral_small`)
- Added reusable matrix benchmark script:
  - `benchmarks/megakernel_small_model_matrix.py`

### Mirage MPK opt-in smoke result (phase 1)

- With `VLLM_MEGAKERNEL_BACKEND=mirage_mpk`, engine startup fails loudly by design with:
  - `NotImplementedError: ... mirage_mpk ... runtime integration is not available in this build yet.`
- This verifies:
  - opt-in gating is wired
  - failure visibility is explicit (not silent fallback)
  - default `mk_llama` path behavior remains unchanged.

---
<details>
<summary> Essential Elements of an Effective PR Description Checklist </summary>

- [x] The purpose of the PR, such as "Fix some issue (link existing issues this PR will resolve)".
- [x] The test plan, such as providing test command.
- [x] The test results, such as pasting the results comparison before and after, or e2e results
- [x] (Optional) The necessary documentation update, such as updating `supported_models.md` and `examples` for a new model.
- [ ] (Optional) Release notes update. If your change is user facing, please update the release notes draft in the [Google Doc](https://docs.google.com/document/d/1YyVqrgX4gHTtrstbq8oWUImOyPCKSGnJ7xtTpmXzlRs/edit?tab=t.0).
</details>

**BEFORE SUBMITTING, PLEASE READ <https://docs.vllm.ai/en/latest/contributing>** (anything written below this line will be removed by GitHub Actions)
