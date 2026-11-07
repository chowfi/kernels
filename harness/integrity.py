"""Timing integrity: the kernel module must not rebind the functions the harness times with.

snapshot() is taken before the kernel is imported; verify() is called after import, after the
correctness pass and after timing. Identity (`is`) comparison, so any monkeypatch is caught.
"""
from __future__ import annotations

import torch
import triton.testing

import harness.timing


def _refs() -> dict[str, object]:
    return {
        "triton.testing.do_bench": triton.testing.do_bench,
        "torch.cuda.Event": torch.cuda.Event,
        "torch.cuda.Event.elapsed_time": torch.cuda.Event.elapsed_time,
        "torch.cuda.Event.record": torch.cuda.Event.record,
        "torch.cuda.synchronize": torch.cuda.synchronize,
        "harness.timing.time_fn": harness.timing.time_fn,
    }


def snapshot() -> dict[str, object]:
    return _refs()


def verify(snap: dict[str, object]) -> list[str]:
    """Names whose current object differs from the snapshot (empty list = intact)."""
    now = _refs()
    return [k for k, v in snap.items() if now.get(k) is not v]
