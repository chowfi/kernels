import json
import os
import subprocess
import sys

import pytest

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
