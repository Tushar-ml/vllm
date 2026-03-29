# SPDX-License-Identifier: Apache-2.0
"""Megakernels runtime setup helpers."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import vllm.envs as envs


def ensure_megakernel_sys_path() -> None:
    root = envs.VLLM_MEGAKERNEL_ROOT
    if root:
        p = str(Path(root).resolve())
        if p not in sys.path:
            sys.path.insert(0, p)
    mk_dir = envs.VLLM_MEGAKERNEL_MK_LLAMA_PATH
    if mk_dir:
        p = str(Path(mk_dir).resolve())
        if p not in sys.path:
            sys.path.insert(0, p)


def build_megakernel_runtime(
    mk_model: Any,
    mk_dir: Path,
    sched_mode: str = "rr",
) -> tuple[Any, Any, Any]:
    ensure_megakernel_sys_path()
    from megakernels.dispatch import make_mk_interpreter, make_schedule_builder
    from megakernels.scheduler import assign_to_sms, tensorize_instructions

    schedule_builder = make_schedule_builder("latency")
    schedule = schedule_builder.build(mk_model)
    assigned = assign_to_sms(sched_mode, schedule=schedule, memory_fraction=None)
    tensorize_instructions(schedule.globs, assigned)
    interpreter = make_mk_interpreter("latency", mk_dir)
    return schedule, interpreter, mk_dir
