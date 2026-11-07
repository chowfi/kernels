"""Keep/revert scoring."""
import math
import statistics

from harness.shapes import BENCH_SHAPES


def _ratios(ours_ms: dict[str, float], fa2_ms: dict[str, float]) -> list[float]:
    return [fa2_ms[s.name] / ours_ms[s.name] for s in BENCH_SHAPES]


def score(ours_ms: dict[str, float], fa2_ms: dict[str, float]) -> float:
    r = _ratios(ours_ms, fa2_ms)
    return math.exp(sum(math.log(x) for x in r) / len(r))


def min_ratio(ours_ms: dict[str, float], fa2_ms: dict[str, float]) -> float:
    return min(_ratios(ours_ms, fa2_ms))


def noise_threshold(scores: list[float]) -> float:
    if len(scores) < 2:
        return 0.005
    return max(0.005, 2 * statistics.stdev(scores) / statistics.mean(scores))


def is_improvement(candidate: float, best: float, threshold: float) -> bool:
    return candidate > best * (1 + threshold)
