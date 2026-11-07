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
