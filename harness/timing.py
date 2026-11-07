"""Timing via triton do_bench (flushes a 256 MB buffer between reps, > the 96 MB L2)."""
import statistics
from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class TimingResult:
    median_ms: float
    stdev_ms: float
    trial_ms: list[float]


def time_fn(fn: Callable[[], Any], trials: int = 5) -> TimingResult:
    from triton.testing import do_bench

    ts = [float(do_bench(fn, warmup=25, rep=100, return_mode="median")) for _ in range(trials)]
    sd = statistics.stdev(ts) if len(ts) > 1 else 0.0
    return TimingResult(statistics.median(ts), sd, ts)
