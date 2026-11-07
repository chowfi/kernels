import pytest
import torch

from harness.correctness import check_kernel
from harness.shapes import BENCH_SHAPES, HELDOUT_SHAPES
from kernel.attention import attention


@pytest.mark.gpu
def test_seed_kernel_correct_everywhere():
    results = check_kernel(attention, BENCH_SHAPES + HELDOUT_SHAPES)
    bad = [r for r in results if not r.ok]
    assert not bad, bad


@pytest.mark.gpu
def test_seed_kernel_rejects_bad_head_dim():
    q = torch.empty(1, 8, 32, 96, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError):
        attention(q, q[:, :, :8], q[:, :, :8], False)
