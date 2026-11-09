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


@pytest.mark.gpu
def test_fp32acc_variant_correct_everywhere():
    from kernel.attention_fp32acc import attention as fair
    results = check_kernel(fair, BENCH_SHAPES + HELDOUT_SHAPES)
    bad = [r for r in results if not r.ok]
    assert not bad, bad


def test_fp32acc_variant_has_no_fp16_and_passes_guard():
    from pathlib import Path
    from driver.guard import kernel_violation
    src = (Path(__file__).parent.parent / "kernel" / "attention_fp32acc.py").read_text()
    import re
    assert not re.search(r"(?<!b)float16", src) and kernel_violation(src) is None
