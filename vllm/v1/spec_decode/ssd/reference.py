# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reference material for Speculative Speculative Decoding (SSD).

SSD is described in *Speculative Speculative Decoding* (Kumar, Dao, May;
`arXiv:2603.03251 <https://arxiv.org/abs/2603.03251>`_, ICLR 2026). The
reference implementation lives at https://github.com/tanishqkumar/ssd .

**Core idea (async / paper setting).** Unlike standard speculative decoding
where draft and target verification run back-to-back on the same device, SSD
overlaps them across hardware: a small draft model may run on a dedicated GPU
while the target verifies another batch. The draft **speculates ahead of
verification outcomes** (branching / cache keyed by recovery tokens and
prefix state) so that when the cache hits, proposals already match the
distribution the verifier would have conditioned on—hiding draft latency.

**Exactness.** Verification uses the same acceptance math as classical
speculative sampling for greedy (argmax) decoding, and softmax + p/q ratio tests
with a recovery distribution adjustment for temperature > 0. Async cache
misses must **not** use ratio acceptance on tokens that were not actually
sampled from the draft; see :mod:`vllm.v1.spec_decode.ssd.verify_ops`.

**Reference engine layout (``tanishqkumar/ssd``).**

* **Sync SD:** ``SpeculatorSync`` + ``DraftRunner`` colocated (rank 0).
* **Async SSD:** Target tensor-parallel ranks ``0 .. num_tp_gpus-1``; draft on
  rank ``num_gpus - 1``. ``SpeculatorAsync`` exchanges, over a dedicated
  process group, command/metadata, per-sequence cache keys ``(seq_id,
  last_spec_step_accepted_len - 1, recovery_token_id)``, token counts, draft
  block tables, temperatures, optional EAGLE hidden states, then receives a
  fused buffer ``[cache_hits; speculations_flat]`` and draft logits
  ``logits_q`` of shape ``[B, K, V]`` for verification.

vLLM integration in this tree starts from the **colocated** path
(``ssd_async=false``), which matches standard draft-model speculative decoding
while exposing SSD-specific config and a portable ``ssd_verify`` used for
tests and future sampler wiring. **Dedicated-draft-GPU** mode
(``ssd_async=true``) requires a launcher where
``distributed_world_size == tensor_parallel_size + 1``; otherwise vLLM logs a
warning and behaves as sync SSD until multi-rank draft workers are wired.

**EAGLE + SSD.** The reference repo can condition the draft on target hidden
states (``use_eagle``). vLLM exposes ``ssd_use_eagle`` as a reserved flag for
future connection to the EAGLE proposer path; it is not active yet.
"""
