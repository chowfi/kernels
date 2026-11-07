import json

import pytest
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel

from harness import correctness
from harness.baselines import fa2_attention, sdpa_attention
from harness.correctness import check_kernel
from harness.inputs import make_inputs, to_sdpa_layout
from harness.reference import reference_attention
from harness.shapes import BENCH_SHAPES, Shape


@pytest.mark.gpu
def test_reference_matches_sdpa_math_small():
    s = Shape("prefill", 512, 512, causal=True)
    q, k, v = make_inputs(s)
    ref = reference_attention(q, k, v, True)
    assert ref.dtype == torch.float32
    with sdpa_kernel(SDPBackend.MATH):
        m = sdpa_attention(*(to_sdpa_layout(t.float()) for t in (q, k, v)), True).transpose(1, 2)
    torch.testing.assert_close(ref, m, atol=1e-4, rtol=1e-4)


@pytest.mark.gpu
def test_gate_passes_fa2_and_rejects_broken():
    shapes = [Shape("prefill", 1024, 1024, causal=True), Shape("decode", 1, 3000, causal=False)]
    assert all(r.ok for r in check_kernel(fa2_attention, shapes))
    broken = lambda q, k, v, c: fa2_attention(q, k, v, c) * 1.01
    assert not any(r.ok for r in check_kernel(broken, shapes))
    raising = lambda q, k, v, c: 1 / 0
    r = check_kernel(raising, shapes[:1])[0]
    assert not r.ok and "ZeroDivisionError" in r.error


@pytest.mark.gpu
def test_gate_rejects_wrong_dtype_shape_nan():
    s = Shape("prefill", 256, 256, causal=True)
    bad = {
        "dtype": lambda q, k, v, c: fa2_attention(q, k, v, c).float(),
        "shape": lambda q, k, v, c: fa2_attention(q, k, v, c)[:, :-1],
        "nan": lambda q, k, v, c: fa2_attention(q, k, v, c) * float("nan"),
    }
    for name, fn in bad.items():
        r = check_kernel(fn, [s])[0]
        assert not r.ok, name


@pytest.mark.gpu
def test_tolerance_not_written_by_check_and_builder_writes(tmp_path, monkeypatch):
    path = tmp_path / "tolerances.json"
    monkeypatch.setattr(correctness, "TOL_PATH", path)
    s = Shape("decode", 1, 2048)
    r1 = check_kernel(fa2_attention, [s])[0]
    assert r1.ok and not path.exists()
    correctness.build_tolerances([s])
    text = path.read_text()
    assert text.endswith("\n")
    entry = json.loads(text)[f"{s.name}|seed0"]
    assert set(entry) == {"abs_tol", "rel_tol"}
    r2 = check_kernel(fa2_attention, [s])[0]
    assert r2.tol == entry["abs_tol"] == r1.tol and r2.rel_tol == entry["rel_tol"]


@pytest.mark.gpu
def test_tolerance_validation_rejects_bad_entry(tmp_path, monkeypatch):
    path = tmp_path / "tolerances.json"
    s = Shape("decode", 1, 2048)
    path.write_text(json.dumps({f"{s.name}|seed0": {"abs_tol": 0.5, "rel_tol": 0.01}}))
    monkeypatch.setattr(correctness, "TOL_PATH", path)
    r = check_kernel(fa2_attention, [s])[0]
    assert not r.ok and "out of range" in r.error


@pytest.mark.gpu
def test_gate_rejects_late_row_corruption():
    s = Shape("prefill", 4096, 4096, causal=True)
    assert check_kernel(fa2_attention, [s])[0].ok

    def late_zero(q, k, v, c):
        o = fa2_attention(q, k, v, c).clone()
        o[:, -64:] = 0
        return o

    r = check_kernel(late_zero, [s])[0]
    assert not r.ok and r.max_row_rel > r.rel_tol


@pytest.mark.gpu
def test_reference_handles_32k_prefill():
    s = BENCH_SHAPES[4]
    q, k, v = make_inputs(s)
    assert reference_attention(q, k, v, True).shape == q.shape
