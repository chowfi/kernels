import os
import re
import subprocess
import sys

import pytest


def _run(*args, tmp_path, extra_env=None):
    env = {**os.environ, "HARNESS_RUNS_LOG": str(tmp_path / "runs.log"), **(extra_env or {})}
    return subprocess.run([sys.executable, "-m", "harness.profile", *args],
                          capture_output=True, text=True, env=env)


@pytest.mark.gpu
def test_profile_uses_head_kernel(tmp_path):
    out = _run("--shape", "decode_8192", tmp_path=tmp_path)
    assert out.returncode == 0, out.stderr
    first = out.stdout.splitlines()[0]
    assert "HEAD" in first and re.search(r"\b[0-9a-f]{7,}\b", first)
    assert " ms" not in out.stdout
    assert not re.search(r"\d\s*(ms|us|ns|usecond|msecond|nsecond)\b", out.stdout)
    assert "gpu__time" not in out.stdout
    log = (tmp_path / "runs.log").read_text()
    assert "harness.profile" in log and "gpu__time" not in log
    assert "registers" in out.stdout.lower()


def test_profile_unknown_shape(tmp_path):
    out = _run("--shape", "nope", tmp_path=tmp_path)
    assert out.returncode == 2
    assert "decode_8192" in out.stdout + out.stderr


_TWO_KERNELS = '''
import torch, triton, triton.language as tl

@triton.jit
def weird_first_k(x, n, BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    tl.store(x + o, tl.load(x + o, mask=o < n) + 1, mask=o < n)

@triton.jit
def weird_second_k(x, n, BLOCK: tl.constexpr):
    o = tl.arange(0, BLOCK)
    tl.store(x + o, tl.load(x + o, mask=o < n) * 2, mask=o < n)

def attention(q, k, v, causal):
    x = torch.zeros(128, device="cuda")
    weird_first_k[(1,)](x, 128, BLOCK=128)
    weird_second_k[(1,)](x, 128, BLOCK=128)
    return q
'''


@pytest.mark.gpu
def test_profile_covers_all_kernels_any_name(tmp_path):
    mod = tmp_path / "fake_attn.py"
    mod.write_text(_TWO_KERNELS)
    out = _run("--shape", "decode_2048", tmp_path=tmp_path, extra_env={"HARNESS_PROFILE_TEST_MODULE": str(mod)})
    assert out.returncode == 0, out.stderr
    assert "weird_first_k" in out.stdout and "weird_second_k" in out.stdout
    assert not re.search(r"\d\s*(ms|us|ns)\b", out.stdout)


def test_module_path_flag_rejected(tmp_path):
    out = _run("--shape", "decode_2048", "--module-path", "kernel/attention.py", tmp_path=tmp_path)
    assert out.returncode == 2 and "unrecognized" in out.stderr


def test_target_helper_not_publicly_runnable(tmp_path):
    out = _run("--_target", "--shape", "decode_2048", "--_module", "/tmp/x.py", tmp_path=tmp_path)
    assert out.returncode == 2


def test_profiling_unavailable(tmp_path):
    out = _run("--shape", "decode_2048", tmp_path=tmp_path, extra_env={"NCU_PATH": "/nonexistent/ncu"})
    assert out.returncode == 0 and "profiling unavailable" in out.stdout
