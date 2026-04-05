# PEARL scheduling (draft model)

[PEARL](https://arxiv.org/abs/2408.11850) (Parallel spEculative decoding with Adaptive dRaft Length, ICLR 2025) alternates **pre-verify** and **post-verify** steps when accepting draft tokens. In vLLM this is exposed as optional scheduling on top of the usual draft-model speculative decoding forward pass (same target verification pass as standard spec decode; no multi-process draft/target overlap from the reference repo).

## Requirements

- `speculative_config.method` must be `"draft_model"`.
- `parallel_drafting` must be `False` (PEARL is unrelated to PARD / parallel draft tokens).
- Greedy batches (`SamplingParams` greedy for all requests in the batch) use the PEARL greedy acceptance rules.
- Batches where **every** request uses random sampling (`all_random`) use the **PEARL random** path when `pearl_scheduling` is enabled: pre-verify applies a single ratio test on the first draft token; post-verify uses the same sequential rejection loop as standard speculative decoding.
- Set `rejection_sample_method` to `"strict"` or `"probabilistic"` in `speculative_config`. For the PEARL random path with **non-`None` draft token probabilities**, `rejection_sample_method` must be `"probabilistic"` (same requirement as for ratio-based acceptance with explicit draft probs elsewhere). When draft probabilities are not passed through (current default in the draft-model worker), the random PEARL path still runs under either method by treating draft probability as 1.0 at each position.

## Draft probabilities (`draft_probs`)

The target always supplies logits used in the ratio test. Draft probabilities matter for the factor \(p_T(x) / p_D(x)\):

- If `draft_probs` is **available** (future plumbing or tests), use `rejection_sample_method="probabilistic"` so the acceptance math matches the standard probabilistic rejection sampler.
- If `draft_probs` is **`None`** (typical today), the implementation uses **`p_D = 1`** at each tested position—the same convention as the Triton path with `NO_DRAFT_PROBS`. Acceptance reduces to comparing the target mass on the draft token to a uniform draw. This does **not** reproduce the paper’s ratios when the draft was produced by a stochastic policy; it is still useful when drafts are effectively deterministic (e.g. greedy draft) while the target is sampled.

## Configuration

Add `"pearl_scheduling": true` to `speculative_config` alongside your draft model and `num_speculative_tokens` (this is the PEARL parameter γ in the paper).

```python
from vllm import LLM, SamplingParams

prompts = ["The future of AI is"]
sampling_params = SamplingParams(temperature=0.0)

llm = LLM(
    model="Qwen/Qwen3-8B",
    tensor_parallel_size=1,
    speculative_config={
        "model": "path-or-id-to-draft",
        "method": "draft_model",
        "num_speculative_tokens": 5,
        "pearl_scheduling": True,
        # Optional: second CUDA stream for draft proposal (scaffold only).
        # "pearl_overlap_streams": True,
    },
)
outputs = llm.generate(prompts, sampling_params)
```

## Limitations

- **Pre-verify target-1:** When `pearl_scheduling` is on and a request is in **pre-verify** mode, the target forward schedules **one** speculative slot; full γ draft token ids are carried in `pearl_full_spec_decode_tokens` / internal request state so acceptance can still emit all γ drafts on success. **Post-verify** still uses γ target slots as before.
- **Overlap:** Same-process wall-clock overlap of target vs draft forwards (reference Accelerate layout) is **not** implemented by default. See `pearl_overlap_streams` (scaffold) and [pearl_overlap_rfc.md](pearl_overlap_rfc.md).
- For the authoritative reproduction code, see the [ParallelSpeculativeDecoding](https://github.com/smart-lty/ParallelSpeculativeDecoding) repository.

## Non-goals and residual gap vs the reference repo

vLLM intentionally does **not** aim for a clone of the HuggingFace + Accelerate reference stack. In particular:

- **No two-process engine**: There is no separate draft process gathered with the target via Accelerate; scheduling stays inside the vLLM engine.
- **No bit-identical parity**: Random streams, float width, kernel order, and the logits-processor pipeline can differ from the reference; do not expect token-for-token equality with [ParallelSpeculativeDecoding](https://github.com/smart-lty/ParallelSpeculativeDecoding).
- **Default scope**: PEARL scheduling is validated for **`draft_model`** speculative decoding only; other speculative methods are out of scope unless explicitly extended later.
- **Overlap for latency only**: Draft/target **overlap** (below) is optional product work and not required for **lossless** speculative semantics relative to the target model.

## Pre-verify target-1 scheduling (implemented)

When `pearl_scheduling` is enabled and `request.pearl_pre_verify` is true after
`update_draft_token_ids`, the scheduler keeps `spec_token_ids` length **1** for
token-budget / KV scheduling while storing the full γ list for metadata. The
worker builds `SpecDecodeMetadata` so **target logits** use the single-slot
layout and `target_logits_indices` repeats that row for all γ PEARL kernel
positions. CUDA graphs are forced to **eager** (`force_eager`) for steps that
use `pearl_full_spec_decode_tokens`.

## Same-GPU stream scaffold (`pearl_overlap_streams`)

`pearl_overlap_streams: true` runs draft proposal on a dedicated CUDA stream and
synchronizes before bookkeeping when any request is in PEARL pre-verify. This
does **not** overlap the target forward with draft work; it only prepares a
second stream for future extensions or minor overlap with host work.

## RFC: Multi-GPU / dual-engine overlap (Phase 2)

See [pearl_overlap_rfc.md](pearl_overlap_rfc.md) for options (split worker
groups, dual EngineCore, executor API changes), invariants, and risks. This
remains **out of scope** for the default product until a dedicated design is
approved.
