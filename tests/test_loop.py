"""End-to-end tests of driver.loop in a throwaway git repo.

The loop runs in-process (driver.loop.main(argv, root=, now=, sleep=)) so time can be faked:
FakeTime.now() = real time + offset, FakeTime.sleep(s) just adds s to the offset, so back-offs
and usage-limit waits are instant but still show up as paused time in RunClock.
Claude is tests/fake_claude.py (CLAUDE_BIN wrapper, FAKE_MODE); the bench is tests/fake_bench.py
(HARNESS_BENCH_CMD, canned scores from a queue file; fake FA2 = 1.0 ms so score == queued value).
The seed kernel benches at 0.5 (below FA2, so the all-shapes-beat-FA2 stop does not fire).
"""
import csv
import json
import os
import shutil
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from driver import loop as driver_loop

REPO = Path(__file__).resolve().parent.parent
SEED = [0.5, 0.5, 0.5]


class FakeTime:
    def __init__(self):
        self.offset = 0.0
        self.slept = []

    def now(self) -> float:
        return time.time() + self.offset

    def sleep(self, s: float) -> None:
        self.slept.append(s)
        self.offset += s


def _git(root, *a):
    return subprocess.run(["git", *a], cwd=root, check=True, capture_output=True, text=True).stdout


@pytest.fixture
def loop_env(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    files = _git(REPO, "ls-files", "-co", "--exclude-standard").split("\n")
    for f in filter(None, files):
        src = REPO / f
        if src.is_file():
            (root / f).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, root / f)
    for k, v in {"GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@t", "GIT_COMMITTER_NAME": "t",
                 "GIT_COMMITTER_EMAIL": "t@t"}.items():
        monkeypatch.setenv(k, v)
    _git(root, "init", "-q", "-b", "main")
    _git(root, "add", "-A")
    _git(root, "commit", "-qm", "init")

    state_dir = tmp_path / "state"
    state_dir.mkdir()
    wrapper = tmp_path / "claude"
    wrapper.write_text(f"#!/bin/sh\nexec {sys.executable} {REPO / 'tests' / 'fake_claude.py'} \"$@\"\n")
    wrapper.chmod(wrapper.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("CLAUDE_BIN", str(wrapper))
    monkeypatch.setenv("FAKE_STATE_DIR", str(state_dir))
    queue = tmp_path / "bench_queue.json"
    monkeypatch.setenv("FAKE_BENCH_QUEUE", str(queue))
    monkeypatch.setenv("HARNESS_BENCH_CMD",
                       f"{sys.executable} {REPO / 'tests' / 'fake_bench.py'} {{mode}} --out {{out}}")
    gpu_apps = tmp_path / "gpu_apps.txt"  # lines "pid, name" served by the fake nvidia-smi query
    gpu_apps.write_text("")
    monkeypatch.setenv("FAKE_GPU_APPS", str(gpu_apps))
    monkeypatch.setenv("HARNESS_GPU_QUERY_CMD", f"{sys.executable} {REPO / 'tests' / 'fake_gpu_query.py'}")
    env = {"root": root, "queue": queue, "time": FakeTime(), "monkeypatch": monkeypatch,
           "gpu_apps": gpu_apps}
    global _ENV
    _ENV = env
    return env


_ENV: dict = {}


def run(env, mode, bench_scores=(), bench_exit=None, bench_crash=False, max_exp=1,
        agent_timeout_s=None, extra=()):
    root = env["root"]
    resumed = subprocess.run(["git", "cat-file", "-e", "HEAD:results/run_state.json"], cwd=root,
                             capture_output=True).returncode == 0
    q = [] if resumed else list(SEED)
    if bench_exit == 2:
        q.append("fail")
    elif bench_crash:
        q.append("crash")
    q += list(bench_scores)
    env["queue"].write_text(json.dumps(q))
    env["monkeypatch"].setenv("FAKE_MODE", mode)
    argv = ["--max-experiments", str(max_exp), *extra]
    if agent_timeout_s is not None:
        argv += ["--agent-timeout-s", str(agent_timeout_s)]
    t = env["time"]
    return driver_loop.main(argv, root=root, now=t.now, sleep=t.sleep)


def rows():
    with open(_ENV["root"] / "results.tsv") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def last_row():
    return rows()[-1]


def exp_ids():
    return [int(r["exp_id"]) for r in rows()]


def clean_tree():
    """No uncommitted changes except the driver's own live state files (run_state.json gets
    best_commit/stop_reason after the last commit; clock.json gets heartbeats)."""
    lines = _git(_ENV["root"], "status", "--porcelain", "-uall").splitlines()
    return [l for l in lines if l[3:] not in {"results/run_state.json", "results/clock.json"}] == []


def head_msg():
    return _git(_ENV["root"], "log", "-1", "--format=%s").strip()


def state():
    r = _ENV["root"] / "results"
    return {**json.loads((r / "run_state.json").read_text()),
            **json.loads((r / "clock.json").read_text())}


def dirty_kernel():
    k = _ENV["root"] / "kernel" / "attention.py"
    k.write_text(k.read_text() + "\n# half-finished edit from a killed run\n")


def kernel_src():
    return (_ENV["root"] / "kernel" / "attention.py").read_text()


# ---------------------------------------------------------------- brief's tests

def test_violation_resets(loop_env):
    run(loop_env, "touch_harness")
    assert last_row()["status"] == "violation"
    assert clean_tree()
    assert "tampered" not in (loop_env["root"] / "harness" / "score.py").read_text()


def test_failed_correctness(loop_env):
    run(loop_env, "break", bench_exit=2)
    assert last_row()["status"] == "failed_correctness"
    assert "1.01" not in kernel_src()
    assert clean_tree()


def test_kept_needs_two_wins(loop_env):
    run(loop_env, "improve", bench_scores=[0.6, 0.5])
    r = last_row()
    assert r["status"] == "reverted"
    assert "BLOCK_N=64," in kernel_src()
    # notes survive a revert, committed in the log commit
    assert "BLOCK_N=128" in (loop_env["root"] / "notes.md").read_text()
    assert head_msg() == "exp 1 log (reverted)"
    assert clean_tree()


def test_kept(loop_env):
    run(loop_env, "improve", bench_scores=[0.6, 0.6])
    r = last_row()
    assert r["status"] == "kept"
    assert head_msg().startswith("exp 1:")
    assert "(score 0.600)" in head_msg()
    assert "BLOCK_N=128," in kernel_src()
    assert float(r["best_score"]) == pytest.approx(0.6)
    assert r["hypothesis"] == "larger KV tile BLOCK_N=128 for fewer loop iterations"
    assert int(r["input_tokens"]) == 10 and int(r["cache_read_tokens"]) == 300
    assert r["session_id"] == "fake-session-0001"
    st = state()
    assert st["best_score"] == pytest.approx(0.6) and st["next_exp_id"] == 2
    assert st["best_commit"] == _git(loop_env["root"], "log", "-1", "--format=%H", "--grep=^exp 1:").strip()
    assert clean_tree()


def test_usage_limit_pauses(loop_env):
    run(loop_env, "limit_once", bench_scores=[0.6, 0.6], max_exp=1)
    st = state()
    assert st["paused_seconds"] > 0
    assert last_row()["status"] == "kept"
    rs = rows()
    assert [r["status"] for r in rs] == ["baseline", "usage_limit", "kept"]
    assert [r["exp_id"] for r in rs] == ["0", "1", "1"]  # same exp_id retried
    assert int(rs[1]["input_tokens"]) == 1  # limit row still carries its tokens
    assert any(s >= 120 for s in loop_env["time"].slept) or sum(loop_env["time"].slept) >= 120


def test_timeout(loop_env):
    run(loop_env, "hang", agent_timeout_s=2)
    assert last_row()["status"] == "timeout"
    assert clean_tree()


def test_bench_crash(loop_env):
    run(loop_env, "improve", bench_crash=True)
    assert last_row()["status"] == "crashed"
    assert "BLOCK_N=64," in kernel_src()
    assert clean_tree()


def test_resume_after_kill(loop_env):
    run(loop_env, "improve", bench_scores=[0.6, 0.6])
    dirty_kernel()
    with open(loop_env["root"] / "runs.log", "a") as f:  # agent check output of the killed exp
        f.write("[killed-exp] harness.check\nprefill_2048 OK\n")
    run(loop_env, "improve", bench_scores=[0.5, 0.5])
    assert exp_ids() == [0, 1, 2]
    assert "[killed-exp]" in (loop_env["root"] / "runs.log").read_text()
    assert "[killed-exp]" in _git(loop_env["root"], "show", "HEAD:runs.log")
    assert "half-finished" not in kernel_src()
    assert last_row()["status"] == "reverted"


def test_three_errors_back_off(loop_env):
    run(loop_env, "error", max_exp=3)
    assert state()["paused_seconds"] > 0
    assert [r["status"] for r in rows()] == ["baseline"] + ["agent_error"] * 3
    assert state()["consecutive_errors"] == 0
    assert loop_env["time"].slept and sum(loop_env["time"].slept) >= 290


# ---------------------------------------------------------------- extra guardrail tests

def test_untracked_file_is_violation_and_removed(loop_env):
    run(loop_env, "untracked")
    assert last_row()["status"] == "violation"
    assert not (loop_env["root"] / "scratch").exists()
    assert clean_tree()


def test_tolerances_edit_is_violation(loop_env):
    before = (loop_env["root"] / "results" / "tolerances.json").read_text()
    run(loop_env, "touch_tolerances")
    assert last_row()["status"] == "violation"
    assert (loop_env["root"] / "results" / "tolerances.json").read_text() == before
    assert "results/tolerances.json" in last_row()["hypothesis"]
    assert clean_tree()


def test_exp0_baseline_row_and_commit(loop_env):
    run(loop_env, "error", max_exp=0)
    rs = rows()
    assert len(rs) == 1 and rs[0]["status"] == "baseline" and rs[0]["exp_id"] == "0"
    assert float(rs[0]["score"]) == pytest.approx(0.5)
    log = _git(loop_env["root"], "log", "--format=%s").split("\n")
    assert "exp 0: baseline (score 0.500)" in log
    st = state()
    assert st["threshold"] == pytest.approx(0.005)
    assert st["stop_reason"] == "max_experiments"


def test_refuses_dirty_tree_when_fresh(loop_env):
    (loop_env["root"] / "stray.txt").write_text("x")
    with pytest.raises(SystemExit):
        run(loop_env, "improve")
    assert not (loop_env["root"] / "results.tsv").exists()


def test_stop_when_all_shapes_beat_fa2(loop_env):
    run(loop_env, "improve", bench_scores=[1.2, 1.2], max_exp=5)
    assert exp_ids() == [0, 1]
    assert state()["stop_reason"] == "beat_fa2_everywhere"


def test_baseline_retime_after_3h(loop_env):
    run(loop_env, "improve", bench_scores=[0.6, 0.6])
    loop_env["time"].offset += 3.1 * 3600  # restart 3.1 h later (counted as offline pause)
    calls = Path(str(loop_env["queue"]) + ".calls")
    calls.write_text("")
    run(loop_env, "improve", bench_scores=[0.6])
    assert calls.read_text().split()[0] == "--baselines"
    assert "drift" in (loop_env["root"] / "progress.log").read_text()


def test_harness_change_aborts(loop_env):
    run(loop_env, "improve", bench_scores=[0.6, 0.6])
    p = loop_env["root"] / "harness" / "score.py"
    p.write_text(p.read_text() + "\n# human edit\n")
    _git(loop_env["root"], "commit", "-qam", "human edits harness")
    with pytest.raises(SystemExit):
        run(loop_env, "improve")
    assert exp_ids() == [0, 1]


def test_kill_mid_pause_stays_paused_on_restart(loop_env):
    t = loop_env["time"]

    def killing_sleep(s):
        t.sleep(30)  # 30 s into the 3 min usage-limit wait, the driver is killed
        raise KeyboardInterrupt

    loop_env["monkeypatch"].setenv("FAKE_MODE", "limit_once")
    loop_env["queue"].write_text(json.dumps(SEED))
    with pytest.raises(KeyboardInterrupt):
        driver_loop.main(["--max-experiments", "1"], root=loop_env["root"], now=t.now,
                         sleep=killing_sleep)
    assert state()["paused_since"] is not None and head_msg() == "exp 1 log (usage_limit)"
    t.slept.clear()
    run(loop_env, "limit_once", bench_scores=[0.6, 0.6])
    assert 100 <= sum(t.slept) <= 160  # only the remaining ~150 s is waited out
    assert state()["paused_seconds"] >= 170
    assert [r["status"] for r in rows()] == ["baseline", "usage_limit", "kept"]


def test_pause_clamped_to_wall_ceiling(loop_env):
    loop_env["monkeypatch"].setenv("FAKE_RESET_IN", str(3 * 86400))
    run(loop_env, "limit_once", bench_scores=[0.6, 0.6], max_exp=5, extra=["--wall-hours", "1"])
    assert state()["stop_reason"] == "wall_hours"
    assert sum(loop_env["time"].slept) <= 3600
    assert [r["status"] for r in rows()] == ["baseline", "usage_limit"]


def test_fresh_start_recovers_from_crashed_startup(loop_env):
    root = loop_env["root"]
    # a startup that died after timing baselines and writing state, before the exp 0 commit
    (root / "results" / "baselines.json").write_text('{"partial": true}')
    (root / "results" / "run_state.json").write_text("{}")
    (root / "results" / "clock.json").write_text('{"start": 1}')
    (root / "results.tsv").write_text("exp_id\n")
    run(loop_env, "improve", bench_scores=[0.6, 0.6])
    assert exp_ids() == [0, 1] and last_row()["status"] == "kept"
    log = _git(root, "log", "--format=%s").split("\n")
    assert "exp 0: baseline (score 0.500)" in log


def test_fresh_start_still_refuses_other_dirty_files(loop_env):
    root = loop_env["root"]
    (root / "results" / "run_state.json").write_text("{}")
    k = root / "kernel" / "attention.py"
    k.write_text(k.read_text() + "# wip\n")
    with pytest.raises(SystemExit):
        run(loop_env, "improve")


def test_sigterm_kills_agent_and_stops(loop_env, tmp_path):
    import signal
    pidf = tmp_path / "claude.pid"
    loop_env["monkeypatch"].setenv("FAKE_PID_FILE", str(pidf))

    def send_term(*_):
        os.kill(os.getpid(), signal.SIGTERM)

    old = signal.signal(signal.SIGALRM, send_term)
    signal.setitimer(signal.ITIMER_REAL, 1.5)
    try:
        with pytest.raises(KeyboardInterrupt):
            run(loop_env, "hang", agent_timeout_s=60)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
    assert signal.getsignal(signal.SIGTERM) is signal.SIG_DFL or callable(signal.getsignal(signal.SIGTERM))
    pid = int(pidf.read_text())
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)


def test_driver_commits_skip_hooks(loop_env):
    hook = loop_env["root"] / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\nexit 1\n")
    hook.chmod(0o755)
    run(loop_env, "improve", bench_scores=[0.6, 0.6])
    assert last_row()["status"] == "kept" and head_msg().startswith("exp 1:")


# ---------------------------------------------------------------- real bench on the GPU

@pytest.mark.gpu
def test_one_real_experiment_gpu(loop_env, monkeypatch):
    # Earlier GPU tests leave ~23 GB in this process's caching allocator; the bench subprocess
    # then OOMs on prefill_32768. (The real driver process never touches CUDA.)
    import gc

    import torch
    gc.collect()
    torch.cuda.empty_cache()
    monkeypatch.setenv("HARNESS_BENCH_CMD", f"{sys.executable} -m harness.bench {{mode}} --out {{out}}")
    root = loop_env["root"]
    run(loop_env, "improve")
    rs = rows()
    assert [r["exp_id"] for r in rs] == ["0", "1"]
    assert rs[0]["status"] == "baseline" and float(rs[0]["score"]) > 0
    assert rs[1]["status"] in {"kept", "reverted"}
    assert float(rs[1]["score"]) > 0 and float(rs[1]["ms_prefill_2048"]) > 0
    assert clean_tree()
    out = REPO / ".superpowers" / "sdd" / "2026-11-06-gqa-flash-attn-autoresearch" / "task-8-gpu-results.tsv"
    if out.parent.exists():
        shutil.copy(root / "results.tsv", out)


# ---------------------------------------------------------------- unit: guard / results

def test_kernel_violation_rules():
    from driver.guard import kernel_violation
    ok = "import math\nimport torch\nimport triton\nimport triton.language as tl\nfrom typing import Any\n"
    assert kernel_violation(ok) is None
    assert "os" in kernel_violation(ok + "import os\n")
    assert kernel_violation(ok + "from numpy import x\n")
    assert kernel_violation(ok + "y = tl.inline_asm_elementwise('x')\n")
    assert kernel_violation(ok + "torch.utils.cpp_extension.load_inline('a', 'b')\n")
    assert kernel_violation(ok + "m = __import__('os')\n")


def test_append_result_sanitizes(tmp_path):
    from driver.results import RESULTS_COLUMNS, append_result, read_results
    p = tmp_path / "r.tsv"
    append_result(p, {"exp_id": 1, "status": "kept", "hypothesis": "a\tb\nc", "score": 1.23456789})
    append_result(p, {"exp_id": 2, "status": "reverted"})
    lines = p.read_text().splitlines()
    assert lines[0].split("\t") == RESULTS_COLUMNS and len(lines) == 3
    r = read_results(p)
    assert r[0]["hypothesis"] == "a b c" and r[0]["score"] == "1.23457" and r[1]["score"] == ""
    with pytest.raises(ValueError):
        append_result(p, {"exp_id": 3, "status": "nope"})


def test_changed_files_untracked_and_renames(loop_env):
    from driver.guard import changed_files
    root = loop_env["root"]
    (root / "newdir").mkdir()
    (root / "newdir" / "x.py").write_text("x")
    _git(root, "mv", "README.md", "README2.md")
    assert {"newdir/x.py", "README.md", "README2.md"} <= changed_files(root)


SEED_KERNEL = (REPO / "kernel" / "attention.py").read_text()


@pytest.mark.parametrize("snippet", [
    "torch.cuda.synchronize = lambda: None\n",
    "tl.dot = None\n",
    "triton.testing.do_bench = None\n",
    "torch.cuda.Event.elapsed_time = None\n",
    "math.pi += 1\n",
    "del torch.cuda.synchronize\n",
    "torch.__dict__['x'] = 1\n",
    "f = getattr(tl, 'inline_' + 'as' + 'm_elementwise')\n",
    "n = 'x'\nf = getattr(tl, n)\n",
    "hasattr(tl, n)\n",
    "setattr(torch, 'x', 1)\n",
    "delattr(torch, 'x')\n",
    "open('/tmp/x')\n",
    "compile('1', 'x', 'eval')\n",
    "g = globals()\n",
    "v = vars(torch)\n",
    "s = 'asm'\n",
    "s = 'ASM volatile'\n",
    "s = 'tl.inline_' 'asm'\n",
    "ev = torch.cuda.Event\nev.elapsed_time = None\n",
    "tb = triton.testing\nd = tb.__dict__\nd['do_bench'] = None\n",
    "torch.cuda.__dict__.update({})\n",
    "torch.cuda.__setattr__('x', 1)\n",
    "c = torch.cuda\n(a, c.synchronize) = (1, 2)\n",
    "from torch import cuda\ncuda.synchronize = None\n",
    "from triton import testing as tt\ntt.do_bench = None\n",
    "E = getattr(torch.cuda, 'Event')\nE.elapsed_time = None\n",
    "for torch.x in range(2):\n    pass\n",
    "def f(q):\n    q.__class__.x = 1\n",
    "def f(q):\n    return q.__class__\n",
    "g = attention.__globals__\n",
    "f = getattr(tl, '__dict__')\n",
    # round 3: reviewer's bypasses of alias tracking
    "def f(m):\n    m.cuda.synchronize = None\nf(torch)\n",
    "def f(t=torch.cuda):\n    t.synchronize = None\n",
    "(t,) = (torch,)\nt.x = 1\n",
    "c = True\nt = torch if c else None\nt.x = 1\n",
    "for m in (torch.cuda,):\n    m.x = 1\n",
    "with open_ctx() as m:\n    m.x = 1\n",
    "def g():\n    return torch.cuda\ng().synchronize = None\n",
    "[torch.cuda][0].x = 1\n",
    "type(torch.cuda.Event()).elapsed_time = None\n",
    "e = exec\n",
    "g = getattr\nn = 'x'\ng(torch, n)\n",
    "b = __builtins__\n",
    "fs = [setattr]\n",
    "def h(x=breakpoint):\n    pass\n",
    "def attention2():\n    pass\nattention2 = torch.cuda\nattention2.synchronize = None\n",
    "class K:\n    pass\ndef f(K):\n    K.x = 1\n",
    "_C = {}\n_C = torch.cuda.__dict__\n_C['synchronize'] = None\n",
    "_C = {}\n_C[0] = torch.cuda\n_C[0].synchronize = None\n",
    "def f(q):\n    q[0] = 1\n",
    # round 4: self/ctx/cls binding and one-level writes
    "self = torch\nself.cuda.synchronize = None\n",
    "def f():\n    self = torch\n    self.cuda.synchronize = None\n",
    "cls = torch\ncls.cuda.synchronize = None\n",
    "class H:\n    def __init__(self):\n        self.m = torch.cuda\n        self.m.synchronize = None\n",
    "def a():\n    pass\na.m = torch.cuda\na.m.synchronize = None\n",
    "def f(q, self):\n    self.x = 1\n",
    "def f(ctx):\n    for ctx in (torch,):\n        ctx.x = 1\n",
    "attention._cache = {}\nattention._cache['k'] = 1\n",
])
def test_guard_rejects_tampering_patterns(snippet):
    from driver.guard import kernel_violation
    assert kernel_violation(SEED_KERNEL + "\n" + snippet) is not None


@pytest.mark.parametrize("snippet", [
    "f = getattr(tl, 'dot')\n",
    "class _Attn(torch.autograd.Function):\n    @staticmethod\n    def forward(ctx, q, s):\n"
    "        ctx.sm_scale = s\n        return q\n",
    "class Holder:\n    def __init__(self):\n        self.x = 1\n",
    "attention._cache = {}\n",
    "class Cfg:\n    pass\nCfg.block = 64\n",
    "msg = 'enthusiasm for plasma physics'\n",
    "def g(cls):\n    cls.n += 1\n",
    "_CACHE = {}\ndef get(k, v):\n    _CACHE[k] = v\n    return _CACHE[k]\n",
    "def f(k):\n    seen = set()\n    memo = dict()\n    memo[k] = 1\n    return memo\n",
    "class H:\n    def put(self, k, v):\n        self[k] = v\n",
])
def test_guard_accepts_legit_code(snippet):
    from driver.guard import kernel_violation
    assert kernel_violation(SEED_KERNEL) is None
    assert kernel_violation(SEED_KERNEL + "\n" + snippet) is None


def test_guard_rules_text():
    from driver.guard import GUARD_RULES
    assert "kernel/attention.py" in GUARD_RULES and "inline_asm" in GUARD_RULES
    assert "breakpoint" in GUARD_RULES and "input" in GUARD_RULES


def test_program_md_guard_section_matches_guard_rules():
    from driver.guard import GUARD_RULES
    assert GUARD_RULES.strip() in (REPO / "program.md").read_text()


def test_integrity_error_surfaces_in_progress_log(loop_env):
    q = list(SEED) + ["integrity"]
    loop_env["queue"].write_text(json.dumps(q))
    loop_env["monkeypatch"].setenv("FAKE_MODE", "improve")
    t = loop_env["time"]
    driver_loop.main(["--max-experiments", "1"], root=loop_env["root"], now=t.now, sleep=t.sleep)
    assert last_row()["status"] == "failed_correctness"
    assert "timing integrity violated" in (loop_env["root"] / "progress.log").read_text()


# ---------------------------------------------------------------- final review fixes

def notes_text():
    return (_ENV["root"] / "notes.md").read_text()


def test_agent_error_without_work_is_not_an_experiment(loop_env):
    """I4: an agent_error with no hypothesis and zero output tokens is retried, not counted."""
    run(loop_env, "error_silent_once", bench_scores=[0.6, 0.6], max_exp=1)
    rs = rows()
    assert [r["status"] for r in rs] == ["baseline", "agent_error", "kept"]
    assert [r["exp_id"] for r in rs] == ["0", "1", "1"]
    assert rs[1]["hypothesis"] == "(none recorded)" and rs[1]["output_tokens"] == "0"
    assert state()["next_exp_id"] == 2


def test_agent_error_with_work_still_counts(loop_env):
    run(loop_env, "error", max_exp=1)
    assert [r["status"] for r in rows()] == ["baseline", "agent_error"]
    assert state()["next_exp_id"] == 2


@pytest.mark.parametrize("mode,kw,status,line", [
    ("notes_hang", {"agent_timeout_s": 2}, "timeout", "- timeout-run note"),
    ("error", {}, "agent_error", "- error-run note"),
    ("touch_harness", {}, "violation", "- violation attempt note"),
])
def test_notes_kept_on_timeout_error_violation(loop_env, mode, kw, status, line):
    """M4: the agent's notes.md lines survive timeout / agent_error / violation reverts."""
    run(loop_env, mode, **kw)
    assert last_row()["status"] == status
    assert line in notes_text()
    assert line in _git(loop_env["root"], "show", "HEAD:notes.md")
    assert clean_tree()


def test_rows_record_active_and_wall_seconds(loop_env):
    """M8: every row carries active_s / wall_s at append time."""
    run(loop_env, "improve", bench_scores=[0.6, 0.6])
    for r in rows():
        assert float(r["active_s"]) >= 0 and float(r["wall_s"]) >= float(r["active_s"])


def test_progress_log_written_once_per_line(loop_env):
    """I6: the driver is the only writer of progress.log (run_loop.sh no longer tees into it)."""
    script = (REPO / "run_loop.sh").read_text()
    assert "tee" not in script and "exec uv run --frozen python -m driver.loop" in script
    run(loop_env, "improve", bench_scores=[0.6, 0.6])
    lines = (loop_env["root"] / "progress.log").read_text().splitlines()
    assert len(lines) == len(set(lines))


def test_bench_commands_use_frozen():
    """M3: never let uv re-sync (flash-attn is built from source)."""
    from scripts import final_report
    assert driver_loop.DEFAULT_BENCH_CMD.startswith("uv run --frozen python -m harness.bench")
    assert final_report.DEFAULT_BENCH_CMD[:4] == ["uv", "run", "--frozen", "python"]


def _spawn(*argv):
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(600)", *argv],
                            start_new_session=True)


def test_stray_agent_gpu_process_is_killed_before_bench(loop_env):
    """I5: a leftover harness.check from the agent is killed before the official bench."""
    stray = _spawn("-m", "harness.check")
    try:
        loop_env["gpu_apps"].write_text(f"{stray.pid}, python\n")
        run(loop_env, "improve", bench_scores=[0.6, 0.6])
        assert stray.wait(timeout=10) != 0
        assert last_row()["status"] == "kept"
        log = (loop_env["root"] / "progress.log").read_text()
        assert f"killed stray GPU process {stray.pid}" in log
    finally:
        stray.kill()


def test_foreign_gpu_process_makes_bench_crash_gpu_busy(loop_env):
    """I5: another compute app that does not go away within 60 s -> crashed, 'gpu busy', no timing."""
    other = _spawn("--some-other-job")
    try:
        run(loop_env, "error", max_exp=0)  # fresh start (benches run with an idle GPU)
        loop_env["gpu_apps"].write_text(f"{other.pid}, python\n")
        calls = Path(str(loop_env["queue"]) + ".calls")
        calls.write_text("")
        run(loop_env, "improve", max_exp=1)
        assert other.poll() is None  # not ours to kill
        r = last_row()
        assert r["status"] == "crashed" and r["score"] == ""
        assert "--ours" not in calls.read_text()
        log = (loop_env["root"] / "progress.log").read_text()
        assert "gpu busy" in log
        assert sum(loop_env["time"].slept) >= 60
    finally:
        other.kill()


def test_is_stray_matches_tokens_not_substrings():
    from driver.loop import _is_stray
    assert _is_stray("/usr/bin/python3 -m harness.check")
    assert _is_stray("uv run --frozen python -m harness.profile --x")
    assert _is_stray("/usr/local/cuda/bin/ncu --nvtx python -m harness.profile")
    assert not _is_stray("python my_ncurses_app.py")
    assert not _is_stray("python uncut.py")
    assert not _is_stray("python -m other.harness.check_tool")


def test_no_goal_stop_resumes_past_goal(loop_env):
    run(loop_env, "improve", bench_scores=[1.2, 1.2], max_exp=5)
    assert state()["stop_reason"] == "beat_fa2_everywhere"
    run(loop_env, "improve", bench_scores=[1.4, 1.4], max_exp=1, extra=["--no-goal-stop"])
    assert exp_ids() == [0, 1, 2]
    assert state()["stop_reason"] == "max_experiments"
