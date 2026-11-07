import pytest

from harness.score import is_improvement, min_ratio, noise_threshold, score
from harness.shapes import BENCH_SHAPES


def test_score_geomean():
    fa2 = {s.name: 2.0 for s in BENCH_SHAPES}
    ours = {s.name: 1.0 for s in BENCH_SHAPES}
    assert score(ours, fa2) == pytest.approx(2.0)
    ours[BENCH_SHAPES[0].name] = 4.0
    assert min_ratio(ours, fa2) == pytest.approx(0.5)


def test_noise_threshold_floor_and_scaling():
    assert noise_threshold([1.0, 1.0, 1.0]) == 0.005
    assert noise_threshold([0.9, 1.0, 1.1]) == pytest.approx(0.2, rel=0.01)


def test_is_improvement():
    assert is_improvement(1.02, 1.0, 0.01)
    assert not is_improvement(1.005, 1.0, 0.01)
