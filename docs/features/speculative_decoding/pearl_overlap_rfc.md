# RFC: Multi-GPU draft/target overlap (PEARL reference parity)

This document captures **Phase 2** design space for reproducing the
[ParallelSpeculativeDecoding](https://github.com/smart-lty/ParallelSpeculativeDecoding)
style **process-level** overlap: draft and target run on different devices with
minimal blocking until logits and draft ids are ready.

## Current vLLM constraints

- `DraftModelProposer` requires `draft_tensor_parallel_size` to match target
  `tensor_parallel_size` so both models compile and run on the **same** ranks.
- `EngineCore.step` runs `execute_model` to completion before `sample_tokens`;
  draft proposal runs **after** target verification in the same step.
- `pearl_overlap_streams` only moves draft proposal onto a second CUDA stream
  **after** the target forward; it does **not** overlap target and draft forwards.

## Goals

- Optional mode where draft and target occupy **disjoint** GPU groups (or
  processes) and communicate draft token ids and target logits (or rejection
  summaries) with explicit collectives or IPC.
- Preserve continuous batching invariants: per-request PEARL phase (pre vs post
  verify), KV rollback on reject, and correct `num_computed_tokens` accounting.

## Architectural options

1. **Dedicated draft worker group**  
   Subset of GPUs runs only the draft model with its own TP group; another
   subset runs the target. Cross-group `broadcast` / `gather` of draft ids and
   target logits rows needed for rejection. Scheduler emits two coordinated
   waves or a split `SchedulerOutput`.

2. **Dual EngineCore / dual process**  
   Closest to the reference Accelerate layout. A thin orchestrator merges
   outputs and advances requests. Highest integration cost with vLLM’s
   single-engine assumptions.

3. **Executor API extension**  
   Extend the executor so `execute_model` can return a second future for
   overlapped draft work, or split one logical step into two RPCs that join
   before `RejectionSampler`. Requires proof that CUDA graph capture and KV
   connectors remain sound.

## Invariants and risks

- **KV**: Draft KV and target KV stay disjoint; on pre-verify reject with
  target-1 scheduling, only one target slot is committed before rejection
  handling.
- **Numerics**: No bit-identical guarantee vs the reference; only distribution
  parity under the existing vLLM sampling pipeline.
- **Collectives**: Separate TP groups imply separate NCCL communicators;
  cross-group traffic must not deadlock with existing vLLM collectives.

## Recommendation

Treat this as a **standalone project** after target-1 scheduling and optional
same-GPU stream scaffolding are stable in production. Start with a design
review choosing option 1 vs 2 and a small single-request prototype before
batching.
