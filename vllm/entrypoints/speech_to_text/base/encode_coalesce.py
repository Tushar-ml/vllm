# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""API-side encode coalesce for Bodhan / ASR transcription.

Increments inflight before mel FE. After FE, ``gate()`` holds
``engine.generate`` until a wave forms:

- ready >= inflight → all concurrent FEs done (C1: immediate)
- ready >= TARGET → flush full encode batch
- max_total from first ready → avoid hangs

This favors large dynamic TRT B over waiting on quiet partial flushes.
"""

from __future__ import annotations

import asyncio
import os


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, str(default)))
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)))
    except ValueError:
        return default


class EncodeCoalesceBarrier:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._inflight = 0
        self._waiters: list[asyncio.Future[None]] = []
        self._first_ready_ts: float | None = None
        self._flush_handle: asyncio.TimerHandle | None = None
        self._ms = max(0.0, _env_float("BODHAN_ENCODER_COALESCE_MS", 0.0))
        self._target = max(0, _env_int("BODHAN_ENCODER_COALESCE_TARGET", 0))
        self._max_total_s = max(
            0.05, _env_float("BODHAN_ENCODER_COALESCE_MAX_S", 0.100)
        )

    @property
    def enabled(self) -> bool:
        return self._ms > 0 or self._target > 0

    @property
    def inflight(self) -> int:
        return self._inflight

    async def begin(self) -> None:
        if not self.enabled:
            return
        async with self._lock:
            self._inflight += 1

    async def end(self) -> None:
        if not self.enabled:
            return
        async with self._lock:
            self._inflight = max(0, self._inflight - 1)
            if self._waiters:
                self._maybe_flush_locked()

    async def gate(self) -> None:
        if not self.enabled:
            return
        loop = asyncio.get_running_loop()
        fut: asyncio.Future[None] = loop.create_future()
        async with self._lock:
            if self._first_ready_ts is None:
                self._first_ready_ts = loop.time()
            self._waiters.append(fut)
            self._maybe_flush_locked()
            if not fut.done():
                self._arm_deadline_locked(loop)
        await fut

    def _maybe_flush_locked(self) -> None:
        n = len(self._waiters)
        if n == 0:
            return
        if self._target > 0 and n >= self._target:
            self._flush_locked()
            return
        if n >= self._inflight:
            self._flush_locked()
            return
        loop = asyncio.get_running_loop()
        if (
            self._first_ready_ts is not None
            and (loop.time() - self._first_ready_ts) >= self._max_total_s
        ):
            self._flush_locked()

    def _flush_locked(self) -> None:
        if self._flush_handle is not None:
            self._flush_handle.cancel()
            self._flush_handle = None
        waiters = self._waiters
        self._waiters = []
        self._first_ready_ts = None
        for fut in waiters:
            if not fut.done():
                fut.set_result(None)

    def _arm_deadline_locked(self, loop: asyncio.AbstractEventLoop) -> None:
        if self._flush_handle is not None:
            self._flush_handle.cancel()
        if self._first_ready_ts is None:
            return
        remaining = self._max_total_s - (loop.time() - self._first_ready_ts)
        if remaining <= 0:
            self._flush_locked()
            return

        def _on_deadline() -> None:
            async def _flush() -> None:
                async with self._lock:
                    self._flush_handle = None
                    if self._waiters:
                        self._flush_locked()

            loop.create_task(_flush())

        self._flush_handle = loop.call_later(remaining, _on_deadline)


_BARRIER: EncodeCoalesceBarrier | None = None


def get_encode_coalesce_barrier() -> EncodeCoalesceBarrier:
    global _BARRIER
    if _BARRIER is None:
        _BARRIER = EncodeCoalesceBarrier()
    return _BARRIER
