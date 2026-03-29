# Megakernel Integration Architecture (Llama 3.2 1B)

## Scope

This document records the architectural changes made to integrate Megakernels into vLLM for `meta-llama/Llama-3.2-1B-Instruct`, plus the latest eager-mode latency benchmark.

The current target is:

- Llama 3.2 1B only (spec-gated).
- Megakernel enabled only when `VLLM_MEGAKERNEL_ON=1`.
- Direct `mk_llama` execution path integrated into vLLM model execution.

## Architecture Changes

### 1) Model registration and gating

Megakernel activation is hooked into vLLM model resolution and inspection so the Megakernel model wrapper is selected only when the runtime and model config match the supported 1B spec.

Key files:

- `vllm/model_executor/models/registry.py`
- `vllm/model_executor/models/megakernel_resolve.py`
- `vllm/model_executor/models/megakernel_spec.py`

What changed:

- Added conditional model-class substitution for Llama architecture when Megakernel is on.
- Added strict spec validation for supported architecture/runtime combinations.
- Kept fail-closed behavior for unsupported combinations.

### 2) Environment-variable plumbing

Megakernel runtime controls were added to vLLM environment handling.

Key file:

- `vllm/envs.py`

Variables used:

- `VLLM_MEGAKERNEL_ON`
- `VLLM_MEGAKERNEL_ROOT`
- `VLLM_MEGAKERNEL_MK_LLAMA_PATH`
- `VLLM_MEGAKERNEL_ALLOW_TRITON_FALLBACK` (for legacy backend fallback testing)

### 3) Direct `mk_llama` model path in vLLM

The core architecture moved from a temporary attention-backend fallback to a direct model path for Megakernel.

Key files:

- `vllm/model_executor/models/megakernel_llama.py`
- `vllm/model_executor/models/megakernel_runtime.py`

What changed:

- `MegakernelLlamaForCausalLM` executes `mk_llama` interpreter directly.
- Runtime bootstraps Megakernels path and schedule/interpreter construction.
- Added explicit logs to confirm backend enablement and first kernel launch.
- Added spec/runtime checks in constructor.

### 4) Cudagraph safety hardening

Multiple capture-safety fixes were applied for decode stability:

- Position IDs switched to tensor-driven flow (no Python scalar control in hot path).
- Internal position cursor logic removed in favor of vLLM `positions` inputs.
- Output/hidden-state tensors converted to static preallocated buffers to keep graph-replay pointers stable.
- Decode behavior validated under both:
  - `PIECEWISE`
  - `FULL_AND_PIECEWISE`

### 5) Decode/prefill backend work (parallel track)

A separate attention-backend path was built to support future true Megakernel attention ops while preserving vLLM scheduling integration.

Key files:

- `vllm/v1/attention/backends/megakernel_attn.py`
- `vllm/v1/attention/ops/megakernel_decode_attention.py`
- `vllm/v1/attention/selector.py`

Current status:

- Direct `mk_llama` path is the active performance path for this milestone.
- Attention-op route remains for future decode/prefill kernelization and generalized batching.

## Operational Configuration

Representative launch configuration used for decode graph + piecewise capture:

- `--max-model-len 4096`
- `--max-num-seqs 1`
- `-cc.cudagraph_mode=FULL_AND_PIECEWISE`

Representative eager benchmarking configuration:

- `--max-model-len 4096`
- `--max-num-seqs 1`
- `--enforce-eager`

## Eager Latency Benchmark (with and without Megakernel)

Date: 2026-03-29

Method:

- Same model/prompt/token limits for both runs.
- vLLM in eager mode (`--enforce-eager`), same `max_model_len=4096`, same `max_num_seqs=1`.
- Warmup requests executed before timed runs.
- 15 measured requests per mode.
- Metric: end-to-end `/v1/completions` request latency in milliseconds.

Prompt:

- `Write one short sentence about Paris.`

Results:

- Megakernel ON (eager):
  - avg: `57.403 ms`
  - p50: `57.895 ms`
  - p95: `58.385 ms`
- Megakernel OFF (eager):
  - avg: `208.068 ms`
  - p50: `207.813 ms`
  - p95: `210.677 ms`

Computed improvement:

- Speedup: `3.625x`
- Latency reduction: `72.41%`

Notes:

- These numbers are from a controlled local run on this machine and this model.
- Absolute values can vary with load, networking, and thermal/GPU state.
- Relative gain is the key signal and matches the expected large improvement trend.

## Remaining Work

- Remove `max_num_seqs=1` constraint safely for robust multi-request decode on direct `mk_llama`.
- Continue moving toward true Megakernel decode/prefill ops in the vLLM attention backend path for broader generalization.
- Expand benchmark suite to include throughput (QPS), TTFT, and multi-concurrency stress scenarios.
