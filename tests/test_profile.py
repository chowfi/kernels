import re
import subprocess
import sys

import pytest


def _run(*args, tmp_path):
    env = {**__import__("os").environ, "HARNESS_RUNS_LOG": str(tmp_path / "runs.log")}
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
