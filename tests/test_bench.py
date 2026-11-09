import json
import os
import subprocess
import sys

import pytest
from pathlib import Path

from harness.shapes import BENCH_SHAPES


@pytest.mark.gpu
def test_check_cli_prints_no_timing(tmp_path):
    env = {**os.environ, "HARNESS_RUNS_LOG": str(tmp_path / "runs.log")}
    out = subprocess.run([sys.executable, "-m", "harness.check"], capture_output=True, text=True, env=env)
    assert out.returncode == 0, out.stdout + out.stderr
    assert "ms" not in out.stdout.lower()


@pytest.mark.gpu
def test_bench_ours_and_baselines(tmp_path):
    p = tmp_path / "sub" / "o.json"
    subprocess.run([sys.executable, "-m", "harness.bench", "--ours", "--trials", "2", "--out", str(p)], check=True)
    o = json.loads(p.read_text())
    assert o["correct"] and set(o["ours_ms"]) == {s.name for s in BENCH_SHAPES}
    assert all(v > 0 for v in o["ours_ms"].values())
    assert "utc" in o and o["trials"] == 2
    assert set(o["post_timing_checks"]) == {s.name for s in BENCH_SHAPES}
    # every launched kernel is one of the Triton kernels defined in kernel/attention.py
    src = (Path(__file__).parent.parent / "kernel" / "attention.py").read_text()
    assert o["triton_kernels"] and all(f"def {n}(" in src for n in o["triton_kernels"])
    assert o["fresh_seed"] > 0


def test_check_logs_import_failure(tmp_path):
    (tmp_path / "broken_kernel.py").write_text("def attention(:\n")
    log = tmp_path / "runs.log"
    env = {**os.environ, "HARNESS_KERNEL_MODULE": "broken_kernel", "HARNESS_RUNS_LOG": str(log),
           "PYTHONPATH": str(tmp_path)}
    out = subprocess.run([sys.executable, "-m", "harness.check"], capture_output=True, text=True, env=env)
    assert out.returncode == 1
    assert "IMPORT FAIL" in out.stdout and "SyntaxError" in out.stdout
    assert "IMPORT FAIL" in log.read_text()


_PATCH_AT_IMPORT = """import torch
torch.cuda.synchronize = lambda *a, **k: None
from kernel.attention import attention
"""

_PATCH_AT_CALL = """import triton.testing
from kernel.attention import attention as _real


def attention(q, k, v, causal):
    triton.testing.do_bench = lambda fn, **kw: 0.001
    return _real(q, k, v, causal)
"""


def _env_with_kernel(tmp_path, name, src):
    (tmp_path / f"{name}.py").write_text(src)
    return {**os.environ, "HARNESS_KERNEL_MODULE": name, "PYTHONPATH": str(tmp_path),
            "HARNESS_RUNS_LOG": str(tmp_path / "runs.log")}


def test_bench_detects_timing_patch_at_import(tmp_path):
    env = _env_with_kernel(tmp_path, "patch_import_kernel", _PATCH_AT_IMPORT)
    p = tmp_path / "o.json"
    r = subprocess.run([sys.executable, "-m", "harness.bench", "--ours", "--out", str(p)],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 2, r.stdout + r.stderr
    o = json.loads(p.read_text())
    assert o["correct"] is False and "timing integrity violated" in o["error"]
    assert "torch.cuda.synchronize" in o["error"] and "ours_ms" not in o


def test_check_detects_timing_patch_at_import(tmp_path):
    env = _env_with_kernel(tmp_path, "patch_import_kernel2", _PATCH_AT_IMPORT)
    r = subprocess.run([sys.executable, "-m", "harness.check"], capture_output=True, text=True, env=env)
    assert r.returncode == 1 and "INTEGRITY FAIL" in r.stdout


@pytest.mark.gpu
def test_bench_detects_timing_patch_at_call(tmp_path):
    env = _env_with_kernel(tmp_path, "patch_call_kernel", _PATCH_AT_CALL)
    p = tmp_path / "o.json"
    r = subprocess.run([sys.executable, "-m", "harness.bench", "--ours", "--trials", "1", "--out", str(p)],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 2, r.stdout + r.stderr
    o = json.loads(p.read_text())
    assert o["correct"] is False and "triton.testing.do_bench" in o["error"]


# ---------------------------------------------------------------- I1: freshness + Triton-only

def test_non_triton_kernel_filter():
    from harness.bench import non_triton_kernels
    tri = {"_attn_fwd", "_decode_split"}
    names = ["_attn_fwd", "_decode_split",
             "void at::native::vectorized_elementwise_kernel<4, at::native::FillFunctor<float>, ...>",
             "void at::native::elementwise_kernel<128, 2, at::native::direct_copy_kernel_cuda(...)>",
             "Memset (Device)", "Memcpy DtoD (Device -> Device)"]
    assert non_triton_kernels(names, tri) == []
    bad = ["void pytorch_flash::flash_fwd_kernel<Flash_fwd_kernel_traits<128>, cutlass::bfloat16_t>",
           "void (anonymous namespace)::softmax_warp_forward<float, float, float, 11>",
           "sm80_xmma_gemm_bf16bf16_bf16f32_f32_tn_n_tilesize96x128x32",
           "void at::native::elementwise_kernel<128, 2, at::native::exp_kernel_cuda(...)>",
           "_attn_fwd_flash_fwd"]  # unknown name (not launched through Triton)
    assert non_triton_kernels(names + bad, tri) == bad


_CACHING_KERNEL = """from kernel.attention import attention as _real
_OUT = {}


def attention(q, k, v, causal):
    key = (q.data_ptr(), k.data_ptr(), tuple(q.shape), tuple(k.shape), causal)
    if key not in _OUT:
        _OUT[key] = _real(q, k, v, causal)
    return _OUT[key]
"""

_SDPA_KERNEL = """import torch.nn.functional as F


def attention(q, k, v, causal):
    o = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                       is_causal=causal, enable_gqa=True)
    return o.transpose(1, 2).contiguous()
"""


@pytest.mark.gpu
def test_bench_rejects_data_keyed_output_cache(tmp_path):
    env = _env_with_kernel(tmp_path, "caching_kernel", _CACHING_KERNEL)
    p = tmp_path / "o.json"
    r = subprocess.run([sys.executable, "-m", "harness.bench", "--ours", "--trials", "1", "--out", str(p)],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 2, r.stdout + r.stderr
    o = json.loads(p.read_text())
    assert o["correct"] is False and "post-timing freshness check failed" in o["error"]
    assert "ours_ms" not in o


@pytest.mark.gpu
def test_bench_rejects_non_triton_kernels(tmp_path):
    env = _env_with_kernel(tmp_path, "sdpa_kernel", _SDPA_KERNEL)
    p = tmp_path / "o.json"
    r = subprocess.run([sys.executable, "-m", "harness.bench", "--ours", "--trials", "1", "--out", str(p)],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 2, r.stdout + r.stderr
    o = json.loads(p.read_text())
    assert o["correct"] is False and "non-Triton CUDA kernel" in o["error"]
    assert "flash_fwd" in o["error"] and "ours_ms" not in o


@pytest.mark.gpu
def test_bench_accepts_smoke_exp1_kernel(tmp_path):
    src = (Path(__file__).parent / "fixtures" / "kernels" / "smoke_exp1.py").read_text()
    env = _env_with_kernel(tmp_path, "smoke_exp1_kernel", src)
    p = tmp_path / "o.json"
    r = subprocess.run([sys.executable, "-m", "harness.bench", "--ours", "--trials", "1", "--out", str(p)],
                       capture_output=True, text=True, env=env)
    assert r.returncode == 0, r.stdout + r.stderr
    o = json.loads(p.read_text())
    assert o["correct"] is True and set(o["ours_ms"]) == {s.name for s in BENCH_SHAPES}
    assert set(o["post_timing_checks"]) == {s.name for s in BENCH_SHAPES}
    assert {"_decode_split", "_decode_combine", "_attn_fwd"} <= set(o["triton_kernels"])


_CHATTY_KERNEL = """import os
import sys
import torch
print("SECRET_IMPORT_STDOUT")
print("SECRET_IMPORT_STDERR", file=sys.stderr)
os.write(1, b"SECRET_IMPORT_FD1\\n")


def attention(q, k, v, causal):
    print("SECRET_CALL_STDOUT 0.123 ms")
    os.write(2, b"SECRET_CALL_FD2\\n")
    return torch.zeros_like(q)
"""


def test_check_silences_kernel_output(tmp_path):
    """M1: a kernel cannot print (e.g. self-timings) to the agent through harness.check."""
    env = _env_with_kernel(tmp_path, "chatty_kernel", _CHATTY_KERNEL)
    r = subprocess.run([sys.executable, "-m", "harness.check"], capture_output=True, text=True, env=env)
    assert "SECRET" not in r.stdout + r.stderr, r.stdout + r.stderr
    assert "prefill_2048 FAIL" in r.stdout and "heldout_decode_8192_g1 FAIL" in r.stdout
    assert r.returncode == 1
