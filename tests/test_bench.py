import json
import subprocess
import sys

import pytest

from harness.shapes import BENCH_SHAPES


@pytest.mark.gpu
def test_check_cli_prints_no_timing(tmp_path):
    out = subprocess.run([sys.executable, "-m", "harness.check"], capture_output=True, text=True)
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
