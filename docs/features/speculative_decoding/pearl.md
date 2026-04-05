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
    },
)
outputs = llm.generate(prompts, sampling_params)
```

## Limitations

- This port applies PEARL **acceptance policy** on the existing vLLM batched verify pass; it does **not** implement the reference implementation’s separate target 1-step vs draft γ-step scheduling or cross-GPU overlap.
- For the authoritative reproduction code, see the [ParallelSpeculativeDecoding](https://github.com/smart-lty/ParallelSpeculativeDecoding) repository.

## Non-goals and residual gap vs the reference repo

vLLM intentionally does **not** aim for a clone of the HuggingFace + Accelerate reference stack. In particular:

- **No two-process engine**: There is no separate draft process gathered with the target via Accelerate; scheduling stays inside the vLLM engine.
- **No bit-identical parity**: Random streams, float width, kernel order, and the logits-processor pipeline can differ from the reference; do not expect token-for-token equality with [ParallelSpeculativeDecoding](https://github.com/smart-lty/ParallelSpeculativeDecoding).
- **Default scope**: PEARL scheduling is validated for **`draft_model`** speculative decoding only; other speculative methods are out of scope unless explicitly extended later.
- **Overlap for latency only**: Draft/target **overlap** (below) is optional product work and not required for **lossless** speculative semantics relative to the target model.

## RFC: Pre-verify “target-1 forward” scheduling

**Problem (gap vs paper):** In the reference, the target performs **one** forward from the shared prefix during a **pre-verify** step while the draft proposes γ tokens internally. vLLM today schedules **γ** target query positions per speculative step whenever drafts are present.

**Proposed direction (design only):**

1. **Scheduler**: When `pearl_scheduling` and `request.pearl_pre_verify`, set the effective number of **target** speculative positions to **1** for that step while the draft model still produces γ proposals. This implies **split** bookkeeping: full γ draft token storage vs a truncated schedule for the target forward.
2. **Worker**: Adjust `_prepare_inputs` / `_calc_spec_decode_metadata` in `gpu_model_runner.py` (and related spec-decode metadata) so target logits rows align with a single slot in pre-verify, without breaking post-verify γ-wide steps.
3. **KV / correctness**: Rely on existing `num_computed_tokens` rollback on rejection; add tests where pre-verify rejects after **one** target slot is written.
4. **CUDA graphs / padding**: Expect to **disable or specialize** captured graphs for this mode because batch shapes and query lengths differ from the uniform-γ path.

**Intentional remaining gap:** Even with target-1 scheduling, vLLM still does not reproduce the reference’s wall-clock **parallelism** between draft and target processes.

## RFC (optional): Draft/target overlap

**Reference behavior:** Accelerate-style `gather` overlapping draft and target processes.

**Scope:** Engine-level or multi-worker coordination (`EngineCore`, possible multi-GPU draft+target KV invariants). This is **explicitly out of scope** for the default product: it targets paper-scale latency, not semantic equivalence.

**Follow-on only:** Treat as a dedicated project with its own design review; keep disabled unless approved.
