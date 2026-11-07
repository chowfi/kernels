import pytest
import torch

from harness.baselines import fa2_attention, sdpa_attention, sdpa_backend_name
from harness.inputs import make_inputs, to_sdpa_layout
from harness.shapes import BENCH_SHAPES, HELDOUT_SHAPES, Shape


def test_shape_lists():
    assert [s.seq_kv for s in BENCH_SHAPES] == [2048, 4096, 8192, 16384, 32768, 2048, 4096, 8192, 16384, 32768, 65536]
    assert all(s.causal for s in BENCH_SHAPES if s.workload == "prefill")
    assert all(s.seq_q == 1 and not s.causal for s in BENCH_SHAPES if s.workload == "decode")
    assert len(HELDOUT_SHAPES) == 4 and not set(HELDOUT_SHAPES) & set(BENCH_SHAPES)


def test_shape_names():
    assert BENCH_SHAPES[0].name == "prefill_2048"
    assert BENCH_SHAPES[-1].name == "decode_65536"
    names = [s.name for s in HELDOUT_SHAPES]
    assert "heldout_prefill_3000" in names
    assert "heldout_prefill_4096_g8" in names
    assert "heldout_decode_5001" in names
    assert "heldout_decode_8192_g1" in names


@pytest.mark.gpu
@pytest.mark.parametrize("shape", [Shape("prefill", 1024, 1024, causal=True), Shape("decode", 1, 4096, causal=False)])
def test_fa2_matches_sdpa(shape):
    q, k, v = make_inputs(shape)
    o_fa2 = fa2_attention(q, k, v, shape.causal)
    o_sdpa = sdpa_attention(*map(to_sdpa_layout, (q, k, v)), shape.causal).transpose(1, 2)
    assert o_fa2.shape == q.shape
    torch.testing.assert_close(o_fa2, o_sdpa, atol=2e-2, rtol=2e-2)


@pytest.mark.gpu
def test_sdpa_backend_reported():
    assert sdpa_backend_name(BENCH_SHAPES[0]) != ""
