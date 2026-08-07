# SPDX-License-Identifier: Apache-2.0
"""Llama 3.2 1B with direct Megakernels mk_llama execution."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

import torch
from torch import nn

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.models.interfaces import IsAttentionFree, SupportsPP
from vllm.model_executor.models.megakernel_runtime import (
    build_megakernel_runtime,
    ensure_megakernel_sys_path,
)
from vllm.model_executor.models.megakernel_spec import (
    validate_megakernel_runtime_or_raise,
)
from vllm.model_executor.models.utils import make_empty_intermediate_tensors_factory
from vllm.sequence import IntermediateTensors

logger = init_logger(__name__)


class MegakernelLlamaForCausalLM(nn.Module, IsAttentionFree, SupportsPP):
    """Direct mk_llama path for Llama 1B integration."""

    is_attention_free: bool = True
    supports_pp: bool = True
    attn_type: str = "attention_free"
    packed_modules_mapping: dict[str, list[str]] = {}

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        del prefix
        super().__init__()
        validate_megakernel_runtime_or_raise(vllm_config)
        if not torch.cuda.is_available():
            raise ValueError("Megakernel Llama requires a CUDA device.")
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.parallel_config = vllm_config.parallel_config

        self._mk_model: Any = None
        self._schedule: Any = None
        self._interpreter: Any = None
        self._mk_dir: Path | None = None

        self._token_logits: torch.Tensor | None = None
        self._logged_first_launch = False
        self._serving_ready = True
        self._use_megakernel = True
        self._parity_checked_tokens = 0
        self._max_parity_tokens = envs.VLLM_MEGAKERNEL_PARITY_TOKENS

        logit_scale = getattr(self.config, "logit_scale", 1.0)
        self.logits_processor = LogitsProcessor(
            self.config.vocab_size,
            scale=logit_scale,
            logits_as_input=True,
        )
        self.lm_head = nn.Identity()
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], self.config.hidden_size
        )
        max_tokens = max(1, self.vllm_config.scheduler_config.max_num_batched_tokens)
        self._hidden_states_buf = torch.zeros(
            max_tokens, self.config.hidden_size, device="cuda", dtype=torch.float32
        )
        self._token_logits_buf = torch.empty(
            max_tokens, self.config.vocab_size, device="cuda", dtype=torch.bfloat16
        )
        self._token_index_buf = torch.arange(max_tokens, device="cuda", dtype=torch.float32)

    def embed_input_ids(self, input_ids: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        del kwargs
        assert self._mk_model is not None, "Model weights are not loaded yet."
        from megakernels.model_types import BatchState  # type: ignore[import-not-found]

        inp = input_ids.unsqueeze(0) if input_ids.dim() == 1 else input_ids
        bs = BatchState(input_ids=inp)
        out = self._mk_model.model.embed_tokens(bs)
        assert out.hidden_states is not None
        return out.hidden_states

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | IntermediateTensors:
        del kwargs, intermediate_tensors
        if not get_pp_group().is_first_rank:
            raise NotImplementedError("Megakernel Llama only supports single PP rank.")
        if inputs_embeds is not None:
            raise NotImplementedError("Megakernel Llama does not support inputs_embeds.")
        assert input_ids is not None
        assert self._mk_model is not None and self._schedule is not None
        assert self._interpreter is not None

        device = input_ids.device
        num_t = input_ids.shape[0]
        if num_t > self._hidden_states_buf.shape[0]:
            raise RuntimeError(
                f"Scheduled tokens ({num_t}) exceed static buffer size "
                f"({self._hidden_states_buf.shape[0]})."
            )
        hs = self._hidden_states_buf[:num_t]
        hs[:, 0] = self._token_index_buf[:num_t]
        self._token_logits = self._token_logits_buf[:num_t]

        from megakernels.model_types import BatchState  # type: ignore[import-not-found]

        is_capturing = torch.cuda.is_current_stream_capturing()
        pos_i32 = positions.to(dtype=torch.int32)
        globs = self._schedule.globs
        for t in range(num_t):
            if not self._logged_first_launch:
                logger.info(
                    "Megakernel launch confirmed: executing mk_llama interpreter "
                    "(first token in this worker)."
                )
                self._logged_first_launch = True
            tok = input_ids[t : t + 1].view(1, 1)
            pos_id = pos_i32[t : t + 1]
            bs = BatchState(input_ids=tok)
            emb = self._mk_model.model.embed_tokens(bs)
            assert emb.hidden_states is not None
            hidden_vec = emb.hidden_states.squeeze(0).squeeze(0).contiguous()
            if self._use_megakernel:
                globs.hidden_states.copy_(hidden_vec)
                globs.barriers.zero_()
                globs.pos_id = pos_id
                self._interpreter.interpret(globs)
                mk_logits = globs.logits
                self._token_logits[t].copy_(mk_logits)
                if (
                    self._max_parity_tokens > 0
                    and not is_capturing
                    and self._parity_checked_tokens < self._max_parity_tokens
                ):
                    baseline_logits = self._baseline_logits(
                        input_ids=tok,
                            position_id=pos_id,
                            seq_len=int(pos_id.item()) + 1,
                        preserve_kv=True,
                    )
                    mk_token = int(torch.argmax(mk_logits).item())
                    baseline_token = int(torch.argmax(baseline_logits).item())
                    self._parity_checked_tokens += 1
                    if mk_token != baseline_token:
                        logger.error(
                            "Megakernel parity mismatch at token %d "
                            "(mk=%d baseline=%d). Falling back to baseline path.",
                            self._parity_checked_tokens,
                            mk_token,
                            baseline_token,
                        )
                        self._use_megakernel = False
                        self._token_logits[t].copy_(baseline_logits.to(torch.bfloat16))
            else:
                baseline_logits = self._baseline_logits(
                    input_ids=tok,
                    position_id=pos_id,
                    seq_len=int(pos_id.item()) + 1,
                    preserve_kv=False,
                )
                self._token_logits[t].copy_(baseline_logits.to(torch.bfloat16))

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hs, "residual": None})
        return hs

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor | None:
        assert self._token_logits is not None
        idx = hidden_states[:, 0].long().clamp(0, self._token_logits.shape[0] - 1)
        logits = self._token_logits[idx]
        return self.logits_processor(self.lm_head, logits)

    @torch.inference_mode()
    def _baseline_logits(
        self,
        input_ids: torch.Tensor,
        position_id: torch.Tensor,
        seq_len: int,
        preserve_kv: bool,
    ) -> torch.Tensor:
        from megakernels.model_types import BatchState  # type: ignore[import-not-found]

        pos = position_id.view(1, 1).to(torch.long)
        k_backup = None
        v_backup = None
        if preserve_kv:
            pos_i = int(pos[0, 0].item())
            k_cache, v_cache = self._mk_model.stacked_kv_cache
            k_backup = k_cache[:, 0, pos_i].clone()
            v_backup = v_cache[:, 0, pos_i].clone()
        bs = BatchState(input_ids=input_ids, position_ids=pos, seq_len=seq_len)
        out = self._mk_model.model(bs)
        if preserve_kv:
            k_cache, v_cache = self._mk_model.stacked_kv_cache
            k_cache[:, 0, pos_i].copy_(k_backup)
            v_cache[:, 0, pos_i].copy_(v_backup)
        assert out.hidden_states is not None
        hidden = out.hidden_states
        x = hidden.to(torch.float32)
        var = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self._mk_model.config.rms_norm_eps)
        x = x.to(hidden.dtype) * self._mk_model.lm_head.input_norm.weight
        logits = torch.matmul(x, self._mk_model.lm_head.lm_head.weight.transpose(0, 1))
        return logits.squeeze(0).squeeze(0)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        ensure_megakernel_sys_path()
        from megakernels.llama import LlamaForCausalLM  # type: ignore[import-not-found]
        from megakernels.model_types import ExtraModelConfig  # type: ignore[import-not-found]

        drained: list[str] = []
        for name, _ in weights:
            drained.append(name)

        model_id = self.vllm_config.model_config.model
        configured_max_len = self.vllm_config.model_config.max_model_len
        mk_max_len = envs.VLLM_MEGAKERNEL_MAX_LEN
        max_len = min(configured_max_len, mk_max_len)
        dev = self.vllm_config.device_config.device
        if isinstance(dev, str):
            device: str | torch.device = "cuda:0" if dev == "cuda" else dev
        elif dev is None:
            device = "cuda:0"
        else:
            device = f"cuda:{0 if dev.index is None else dev.index}" if dev.type == "cuda" else dev
        dtype = self.vllm_config.model_config.dtype
        if not isinstance(dtype, torch.dtype):
            dtype = torch.bfloat16

        extra = ExtraModelConfig(
            interleave_rope=True,
            max_len_override=max_len,
            max_batch_size=max(1, min(envs.VLLM_MEGAKERNEL_MAX_BATCH_SIZE, self.vllm_config.scheduler_config.max_num_seqs)),
        )
        self._mk_model = LlamaForCausalLM.from_pretrained(
            model_id, device=device, dtype=dtype, extra_config=extra
        )

        mk_dir = envs.VLLM_MEGAKERNEL_MK_LLAMA_PATH
        if not mk_dir:
            raise ValueError(
                "Set VLLM_MEGAKERNEL_MK_LLAMA_PATH to the directory containing "
                "the compiled mk_llama extension."
            )
        self._mk_dir = Path(mk_dir).resolve()
        self._schedule, self._interpreter, _ = build_megakernel_runtime(
            self._mk_model, self._mk_dir
        )
        logger.info(
            "Megakernel backend enabled for model %s (mk_dir=%s, max_len=%d).",
            model_id,
            self._mk_dir,
            max_len,
        )
        return set(drained)
