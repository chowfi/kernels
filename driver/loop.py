"""Autoresearch main loop: one headless Claude invocation (= one idea) per experiment, then the
official benchmark, then keep (commit) or revert, logging everything.

    python -m driver.loop [--max-experiments N] [--active-hours 6] [--wall-hours 36]
                          [--agent-timeout-min 20] [--agent-timeout-s S]

Git model: history is linear. Every experiment ends in exactly one commit that holds its
results.tsv row together with the run state (results/run_state.json, clock.json, tokens.json)
and runs.log, so HEAD is always a consistent checkpoint and the kernel at HEAD is always the
best kernel so far. "Revert" therefore means: throw away the agent's working-tree changes back
to the HEAD the experiment started from. A restart after a kill resets to HEAD (keeping only the
clock and the token ledger, which record real elapsed time / real spend) and resumes.

Test hooks: HARNESS_BENCH_CMD (bench command template with {mode} and {out}), CLAUDE_BIN (read
at call time), and main(argv, root=, now=, sleep=) for injecting a fake clock.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shlex
import signal
import statistics
import subprocess
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from driver.claude_stream import run_claude
from driver.clock import RunClock
from driver.guard import DRIVER_FILES, changed_files, check_agent_changes, harness_hash, revert_to
from driver.ledger import TokenLedger
from driver.results import NO_HYPOTHESIS, append_result, is_idle_agent_error, read_results
from harness.score import is_improvement, min_ratio, noise_threshold, score
from harness.shapes import BENCH_SHAPES

PROMPT = ("You are running experiment {exp_id} of an autoresearch loop. Read program.md and "
          "follow it exactly. Write your hypothesis to experiments/current.json before editing "
          "the kernel.")

DEFAULT_BENCH_CMD = "uv run --frozen python -m harness.bench {mode} --out {out}"
# Compute apps on the GPU (graphics apps such as Xorg are not listed). Test hook: HARNESS_GPU_QUERY_CMD.
GPU_QUERY_CMD = "nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader"
GPU_IDLE_WAIT_S = 60.0
GPU_POLL_S = 5.0
# Leftovers of the agent's own tools, safe to kill before an official bench (own user only).
STRAY_MARKERS = ("harness.check", "harness.profile", "ncu")
BENCH_TIMEOUT_S = 15 * 60
RETIME_S = 3 * 3600
BACKOFF_MIN = [5, 10, 20, 30]  # then 30, 30, ...
MAX_CONSECUTIVE_ERRORS = 3
SLEEP_CHUNK_S = 60.0
SEED_RUNS = 3
SHAPES = [s.name for s in BENCH_SHAPES]

RUN_STATE = "results/run_state.json"
CLOCK = "results/clock.json"
TOKENS = "results/tokens.json"
RESULTS = "results.tsv"
BASELINES = "results/baselines.json"
RUNS_LOG = "runs.log"
NOTES = "notes.md"
CURRENT = "experiments/current.json"
KERNEL = "kernel/attention.py"
PROGRESS = "progress.log"  # gitignored
BENCH_DIR = "results/bench"  # gitignored


def _iso_to_ts(s: str) -> float:
    return datetime.fromisoformat(s).timestamp()


def _killpg(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    proc.wait()


def _rate_limit_note(info: dict | None) -> str:
    """'rate limit: status=allowed five_hour=0.30 seven_day=0.11' from a rate_limit_info dict."""
    if not isinstance(info, dict):
        return ""
    parts = [f"status={info.get('status')}"]
    for name, w in (info.get("unifiedWindows") or {}).items():
        if isinstance(w, dict) and "utilization" in w:
            parts.append(f"{name}={w['utilization']}")
    return "rate limit: " + " ".join(parts)


def _one_line(s: str, n: int = 120) -> str:
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 3] + "..."


class Loop:
    def __init__(self, root: Path, args: argparse.Namespace, now: Callable[[], float],
                 sleep: Callable[[float], None]):
        self.root = Path(root).resolve()
        self.args = args
        self.now = now
        self.sleep = sleep
        self.agent_timeout_s = (args.agent_timeout_s if args.agent_timeout_s is not None
                                else args.agent_timeout_min * 60)
        self.st: dict = {}
        self.experiments_this_run = 0

    # ------------------------------------------------------------------ helpers
    def p(self, rel: str) -> Path:
        return self.root / rel

    def git(self, *a: str) -> str:
        return subprocess.run(["git", *a], cwd=self.root, check=True, capture_output=True,
                              text=True).stdout

    def head(self) -> str:
        return self.git("rev-parse", "HEAD").strip()

    def utc(self) -> str:
        return datetime.fromtimestamp(self.now(), timezone.utc).isoformat(timespec="seconds")

    def log(self, msg: str) -> None:
        line = f"{self.utc()} {msg}"
        print(line, flush=True)
        with open(self.p(PROGRESS), "a") as f:
            f.write(line + "\n")

    def save_state(self) -> None:
        path = self.p(RUN_STATE)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(path.name + ".tmp")
        tmp.write_text(json.dumps(self.st, indent=2, sort_keys=True) + "\n")
        os.replace(tmp, path)

    def fa2(self) -> dict[str, float]:
        return json.loads(self.p(BASELINES).read_text())["fa2_ms"]

    def commit(self, msg: str, paths: list[str]) -> None:
        tracked = set(self.git("ls-files").split("\n"))
        add = [x for x in dict.fromkeys(paths) if self.p(x).exists() or x in tracked]
        if add:
            self.git("add", "-A", "--", *add)
        if subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=self.root).returncode == 0:
            return
        self.git("-c", "core.hooksPath=/dev/null", "commit", "--no-verify", "-q", "-m", msg)

    def state_paths(self) -> list[str]:
        return [RESULTS, RUN_STATE, CLOCK, TOKENS, RUNS_LOG]

    def restore(self, commit: str, keep: tuple[str, ...] = ()) -> None:
        """revert_to(commit), but driver-owned files (and `keep`) keep their current bytes."""
        saved = {}
        for rel in [*DRIVER_FILES, RUNS_LOG, *keep]:
            if self.p(rel).is_file():
                saved[rel] = self.p(rel).read_bytes()
        revert_to(self.root, commit)
        for rel, b in saved.items():
            self.p(rel).parent.mkdir(parents=True, exist_ok=True)
            self.p(rel).write_bytes(b)

    # protected files: anything the agent must not change, incl. gitignored progress.log
    def protected(self) -> list[str]:
        files = [RESULTS, PROGRESS]
        if self.p("results").is_dir():
            files += [f"results/{q.name}" for q in sorted(self.p("results").glob("*.json"))]
        return files

    def snapshot(self) -> dict[str, bytes | None]:
        return {rel: (self.p(rel).read_bytes() if self.p(rel).is_file() else None)
                for rel in self.protected()}

    def tampered(self, snap: dict[str, bytes | None]) -> list[str]:
        changed = []
        for rel in sorted(set(snap) | set(self.protected())):
            now_b = self.p(rel).read_bytes() if self.p(rel).is_file() else None
            if now_b != snap.get(rel):
                changed.append(rel)
                if snap.get(rel) is None:
                    self.p(rel).unlink()
                else:
                    self.p(rel).write_bytes(snap[rel])  # undo immediately
        return changed

    # ------------------------------------------------------------------ GPU idle (I5)
    def gpu_apps(self) -> list[tuple[int, str]] | None:
        """Compute apps on the GPU other than this process; None if the query fails."""
        cmd = shlex.split(os.environ.get("HARNESS_GPU_QUERY_CMD") or GPU_QUERY_CMD)
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        except (OSError, subprocess.TimeoutExpired) as e:
            self.log(f"gpu idle check: query failed ({e!r}); proceeding")
            return None
        if r.returncode != 0:
            self.log(f"gpu idle check: query exit {r.returncode} ({_one_line(r.stderr)}); proceeding")
            return None
        apps = []
        for line in r.stdout.splitlines():
            pid_s, _, name = line.partition(",")
            try:
                pid = int(pid_s.strip())
            except ValueError:
                continue
            if pid != os.getpid():
                apps.append((pid, name.strip()))
        return apps

    @staticmethod
    def _cmdline(pid: int) -> str:
        try:
            return Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            return ""

    @staticmethod
    def _own(pid: int) -> bool:
        try:
            return os.stat(f"/proc/{pid}").st_uid == os.getuid()
        except OSError:
            return False

    def ensure_gpu_idle(self) -> bool:
        """Kill own leftover harness.check / harness.profile / ncu processes, then wait up to
        60 s for any other compute app to leave. Returns False if the GPU is still busy."""
        apps = self.gpu_apps()
        if not apps:
            return True
        for pid, name in apps:
            cmd = self._cmdline(pid)
            if self._own(pid) and any(m in cmd for m in STRAY_MARKERS):
                try:
                    os.kill(pid, signal.SIGKILL)
                    self.log(f"gpu idle check: killed stray GPU process {pid} ({_one_line(cmd, 160)})")
                except (ProcessLookupError, PermissionError) as e:
                    self.log(f"gpu idle check: could not kill {pid}: {e!r}")
            else:
                self.log(f"gpu idle check: GPU in use by pid {pid} {name} ({_one_line(cmd, 160)}); waiting")
        deadline = self.now() + GPU_IDLE_WAIT_S
        while True:
            apps = self.gpu_apps()
            if not apps:
                self.log("gpu idle check: GPU idle")
                return True
            if self.now() >= deadline:
                self.log("gpu idle check: gpu busy after "
                         f"{GPU_IDLE_WAIT_S:.0f}s: {', '.join(f'{p} {n}' for p, n in apps)}")
                return False
            self.sleep(GPU_POLL_S)

    # ------------------------------------------------------------------ bench
    def bench(self, mode: str, out_rel: str) -> tuple[str, dict | None, str]:
        """Returns (ok|failed_correctness|crashed, data, detail)."""
        if not self.ensure_gpu_idle():
            return "crashed", None, "gpu busy"
        out = self.p(out_rel)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.unlink(missing_ok=True)
        tmpl = os.environ.get("HARNESS_BENCH_CMD") or DEFAULT_BENCH_CMD
        argv = [t.replace("{mode}", mode).replace("{out}", str(out)) for t in shlex.split(tmpl)]
        with open(out.with_suffix(".log"), "w") as logf:
            proc = subprocess.Popen(argv, cwd=self.root, stdin=subprocess.DEVNULL, stdout=logf,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            try:
                rc = proc.wait(timeout=BENCH_TIMEOUT_S)
            except subprocess.TimeoutExpired:
                _killpg(proc)
                return "crashed", None, f"bench timeout after {BENCH_TIMEOUT_S}s"
            except BaseException:  # KeyboardInterrupt / SIGTERM: don't leave it on the GPU
                _killpg(proc)
                raise
        data = None
        if out.is_file():
            try:
                data = json.loads(out.read_text())
            except ValueError:
                data = None
        if not isinstance(data, dict):
            data = None
        if rc == 2 and data is not None and data.get("correct") is False:
            if data.get("error"):  # e.g. "timing integrity violated (...): ..."
                return "failed_correctness", data, str(data["error"])
            bad = [c.get("shape_name") for c in data.get("correctness", []) if not c.get("ok")]
            return "failed_correctness", data, f"correctness failed: {bad}"
        if rc != 0 or data is None:
            return "crashed", data, f"bench exit {rc}, json {'ok' if data else 'missing/unparseable'}"
        key = "ours_ms" if mode == "--ours" else "fa2_ms"
        if mode == "--ours" and data.get("correct") is not True:
            return "crashed", data, "bench json lacks correct=true"
        ms = data.get(key)
        if not isinstance(ms, dict) or not all(
                isinstance(ms.get(s), (int, float)) and math.isfinite(ms[s]) and ms[s] > 0
                for s in SHAPES):
            return "crashed", data, f"bench json has missing/invalid {key}"
        return "ok", data, ""

    def bench_baselines(self) -> dict | None:
        out = f"{BENCH_DIR}/baselines_{int(self.now())}.json"
        kind, data, detail = self.bench("--baselines", out)
        if kind != "ok":
            self.log(f"baselines bench failed: {detail}")
            return None
        self.p(BASELINES).write_text(json.dumps(data, indent=2) + "\n")
        return data

    # ------------------------------------------------------------------ startup
    def tracked_at_head(self, rel: str) -> bool:
        return subprocess.run(["git", "cat-file", "-e", f"HEAD:{rel}"], cwd=self.root,
                              capture_output=True).returncode == 0

    def startup(self) -> None:
        # Exp 0's commit is the first to contain run_state.json; until then the run is fresh.
        if not self.tracked_at_head(RUN_STATE):
            self.fresh_start()
        else:
            self.resume()

    def fresh_start(self) -> None:
        # Files a startup that died before the exp 0 commit may have left behind.
        leftovers = {RESULTS, RUN_STATE, CLOCK, TOKENS, BASELINES}
        dirty = sorted(changed_files(self.root))
        other = [f for f in dirty if f not in leftovers]
        if other:
            raise SystemExit("driver.loop: working tree is not clean; commit or stash first:\n  "
                             + "\n  ".join(other))
        for rel in dirty:
            if self.tracked_at_head(rel):
                self.git("checkout", "-q", "HEAD", "--", rel)
            else:
                self.p(rel).unlink(missing_ok=True)
        if dirty:
            self.log(f"fresh start: discarded leftovers of an earlier startup: {', '.join(dirty)}")
        self.clock = RunClock(self.p(CLOCK), now=self.now)
        self.ledger = TokenLedger(self.p(TOKENS))
        self.clock.start()
        self.log("fresh start: timing baselines")
        self.st = {"harness_hash": harness_hash(self.root), "next_exp_id": 1,
                   "consecutive_errors": 0, "backoff_idx": 0, "pause_until": None,
                   "pause_reason": None, "stop_reason": None}
        if self.bench_baselines() is None:
            raise SystemExit("driver.loop: baselines bench failed; see results/bench/*.log")
        fa2 = self.fa2()
        runs = []
        for n in range(1, SEED_RUNS + 1):
            kind, data, detail = self.bench("--ours", f"{BENCH_DIR}/exp0_{n}.json")
            self.clock.heartbeat()
            if kind != "ok":
                raise SystemExit(f"driver.loop: seed kernel bench {n} {kind}: {detail}")
            runs.append(data["ours_ms"])
        scores = [score(ms, fa2) for ms in runs]
        med = statistics.median(scores)
        ours = runs[scores.index(med)] if med in scores else runs[0]
        self.st.update(threshold=noise_threshold(scores), best_score=med,
                       best_ours_ms={s: ours[s] for s in SHAPES})
        self.log(f"seed scores {[round(x, 4) for x in scores]} threshold "
                 f"{self.st['threshold']:.4f}")
        head = self.head()
        row = {"exp_id": 0, "utc": self.utc(), "hypothesis": "seed kernel (baseline)",
               "status": "baseline", "score": med, "best_score": med,
               "min_vs_fa2": min_ratio(ours, fa2), **{f"ms_{s}": ours[s] for s in SHAPES},
               "input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
               "cache_write_tokens": 0, "agent_seconds": 0,
               "active_s": round(self.clock.active_seconds(), 1),
               "wall_s": round(self.clock.wall_seconds(), 1), "session_id": "", "commit": head}
        append_result(self.p(RESULTS), row)
        self.save_state()
        self.commit(f"exp 0: baseline (score {med:.3f})", [*self.state_paths(), BASELINES])
        self.st["best_commit"] = self.head()
        self.save_state()
        self.progress(0, "baseline", med)

    def resume(self) -> None:
        self.clock = RunClock(self.p(CLOCK), now=self.now)  # may record a driver_offline gap
        # Back to the last checkpoint. Clock + ledger keep their bytes (real time / real spend),
        # runs.log keeps the killed experiment's harness.check output (committed with the next exp).
        saved = {rel: self.p(rel).read_bytes() for rel in (CLOCK, TOKENS, RUNS_LOG)
                 if self.p(rel).is_file()}
        revert_to(self.root, "HEAD")
        for rel, b in saved.items():
            self.p(rel).write_bytes(b)
        self.ledger = TokenLedger(self.p(TOKENS))
        self.st = json.loads(self.p(RUN_STATE).read_text())
        self.st["best_commit"] = self.git("log", "-1", "--format=%H", "-E",
                                          "--grep=^exp [0-9]+: ").strip() or self.head()
        self.st["stop_reason"] = None
        self.clock.start()  # no-op if already started
        self.log(f"resumed at exp {self.st['next_exp_id']} (best {self.st['best_score']:.4f}, "
                 f"best_commit {self.st['best_commit'][:10]})")
        if self.clock.is_paused or self.st.get("pause_until"):
            if not self.clock.is_paused:
                self.clock.pause(self.st.get("pause_reason") or "resume")
            self.wait_pause()

    # ------------------------------------------------------------------ pauses
    def backoff_s(self) -> float:
        i = self.st.get("backoff_idx", 0)
        self.st["backoff_idx"] = i + 1
        return 60.0 * BACKOFF_MIN[min(i, len(BACKOFF_MIN) - 1)]

    def wall_remaining(self) -> float:
        return self.args.wall_hours * 3600 - self.clock.wall_seconds()

    def begin_pause(self, reason: str, wait_s: float) -> float:
        # Never wait past the hard wall-clock ceiling (small margin so the stop check fires).
        wait_s = max(0.0, min(wait_s, self.wall_remaining() + 1e-3))
        self.st["pause_until"] = self.now() + wait_s
        self.st["pause_reason"] = reason
        self.clock.pause(reason)
        return wait_s

    def wait_pause(self) -> None:
        until = self.st.get("pause_until") or self.now()
        until = min(until, self.now() + max(0.0, self.wall_remaining()) + 1e-3)
        self.log(f"paused ({self.st.get('pause_reason')}) for {max(0, until - self.now()) / 60:.1f}"
                 f" min, until {datetime.fromtimestamp(until, timezone.utc).isoformat()}")
        while (rem := until - self.now()) > 0:
            if self.wall_remaining() <= 0:
                self.log("wall-clock ceiling reached during pause")
                break
            self.sleep(min(rem, SLEEP_CHUNK_S))
        self.clock.resume()
        self.st["pause_until"] = None
        self.st["pause_reason"] = None
        self.save_state()
        self.log("resumed after pause")

    # ------------------------------------------------------------------ loop
    def stop_reason(self) -> str | None:
        a = self.args
        if self.clock.active_seconds() >= a.active_hours * 3600:
            return "active_hours"
        if self.clock.wall_seconds() >= a.wall_hours * 3600:
            return "wall_hours"
        if a.max_experiments is not None and self.experiments_this_run >= a.max_experiments:
            return "max_experiments"
        if min_ratio(self.st["best_ours_ms"], self.fa2()) >= 1.0:
            return "beat_fa2_everywhere"
        return None

    def progress(self, exp_id: int, status: str, s: float | None, note: str = "") -> None:
        self.log(f"exp {exp_id} {status} score={'-' if s is None else f'{s:.4f}'} "
                 f"best={self.st['best_score']:.4f} active={self.clock.active_seconds() / 3600:.2f}h "
                 f"wall={self.clock.wall_seconds() / 3600:.2f}h "
                 f"tokens={self.ledger.totals().total}" + (f" | {note}" if note else ""))

    def maybe_retime(self) -> None:
        b = json.loads(self.p(BASELINES).read_text())
        ts = b.get("timestamp") or b.get("utc")
        if ts and self.now() - _iso_to_ts(ts) < RETIME_S:
            return
        old = b["fa2_ms"]
        self.log("re-timing baselines (>= 3 h old)")
        new = self.bench_baselines()
        self.clock.heartbeat()
        if new is None:
            return  # keep the old baselines; retried next iteration
        drift = " ".join(f"{s}={100 * (new['fa2_ms'][s] - old[s]) / old[s]:+.2f}%" for s in SHAPES)
        self.log(f"baseline fa2 drift: {drift}")
        prev = self.st["best_score"]
        self.st["best_score"] = score(self.st["best_ours_ms"], new["fa2_ms"])
        self.log(f"best_score re-based {prev:.4f} -> {self.st['best_score']:.4f}; min_vs_fa2 "
                 f"{min_ratio(self.st['best_ours_ms'], new['fa2_ms']):.4f}")
        self.save_state()
        self.commit("baselines re-timed", [BASELINES, RUN_STATE, CLOCK, TOKENS])

    def stop(self, reason: str) -> None:
        self.st["stop_reason"] = reason
        self.save_state()
        self.log(f"stopping: {reason} (active {self.clock.active_seconds() / 3600:.2f}h, wall "
                 f"{self.clock.wall_seconds() / 3600:.2f}h, best {self.st['best_score']:.4f})")
        # No commit: HEAD stays the last experiment's commit. run_state.json (stop_reason,
        # best_commit) and clock.json are left modified; a restart resets to HEAD anyway.

    def run(self) -> int:
        try:
            return self._run()
        except KeyboardInterrupt:
            self.log("interrupted (SIGINT/SIGTERM); agent/bench killed; resume continues from HEAD")
            raise

    def _run(self) -> int:
        self.startup()
        while True:
            self.clock.heartbeat()
            reason = self.stop_reason()
            if reason:
                self.stop(reason)
                return 0
            if harness_hash(self.root) != self.st["harness_hash"]:
                self.st["stop_reason"] = "harness_changed"
                self.save_state()
                self.log("ABORT: harness hash changed (harness/*.py, program.md, pyproject.toml "
                         "or uv.lock edited mid-run)")
                raise SystemExit("driver.loop: harness hash changed mid-run; refusing to continue")
            self.maybe_retime()
            if self.experiment(self.st["next_exp_id"]):
                self.experiments_this_run += 1

    def read_hypothesis(self) -> str:
        try:
            h = json.loads(self.p(CURRENT).read_text()).get("hypothesis")
        except (OSError, ValueError, AttributeError):
            h = None
        return _one_line(h, 500) if isinstance(h, str) and h.strip() else NO_HYPOTHESIS

    def experiment(self, exp_id: int) -> bool:
        """Runs one invocation. Returns True if it counts as an experiment (not usage_limit)."""
        head = self.head()
        self.p(CURRENT).parent.mkdir(parents=True, exist_ok=True)
        self.p(CURRENT).write_text("{}\n")
        snap = self.snapshot()
        out = run_claude(PROMPT.format(exp_id=exp_id), self.root, self.agent_timeout_s,
                         claude_bin=os.environ.get("CLAUDE_BIN", "claude"),
                         stream_log=self.p(f"{BENCH_DIR}/agent_exp{exp_id}_{int(self.now())}.jsonl"))
        tampered = self.tampered(snap)  # restores protected files before anything else
        self.clock.heartbeat()
        hyp = self.read_hypothesis()
        self.ledger.add(exp_id, out.usage, out.session_id)
        row = {"exp_id": exp_id, "utc": self.utc(), "hypothesis": hyp,
               "input_tokens": out.usage.input_tokens, "output_tokens": out.usage.output_tokens,
               "cache_read_tokens": out.usage.cache_read_tokens,
               "cache_write_tokens": out.usage.cache_write_tokens,
               "agent_seconds": round(out.duration_s, 1), "session_id": out.session_id or "",
               "commit": head}
        rl = _rate_limit_note(out.rate_limit)
        if rl:
            self.log(f"exp {exp_id} agent {out.status}; {rl}")
        note = ""
        if out.status in ("ok", "timeout") and not tampered:
            self.st["consecutive_errors"] = 0

        if tampered:
            self.restore(head, keep=(NOTES,))
            v = f"protected files modified: {', '.join(tampered)}"
            row["hypothesis"] = f"{hyp} [violation: {v}]"
            return self.finish(row, "violation", v)

        if out.status == "usage_limit":
            self.restore(head)
            row["hypothesis"] = "usage limit hit"
            if out.reset_at and out.reset_at + 60 - self.now() > 60:
                wait = out.reset_at + 60 - self.now()
            else:
                wait = self.backoff_s()
            wait = self.begin_pause("usage_limit", wait)
            self.finish(row, "usage_limit", f"{out.result_text[:200]!r}; waiting {wait / 60:.1f} min",
                        counted=False)
            self.wait_pause()
            return False

        if out.status == "error":
            self.restore(head, keep=(NOTES,))
            self.st["consecutive_errors"] = self.st.get("consecutive_errors", 0) + 1
            note = _one_line(out.result_text, 200)
            # I4: an error before the agent did anything is retried under the same exp_id.
            counted = not is_idle_agent_error({**row, "status": "agent_error"})
            if not counted:
                note += " (no work done: not counted as an experiment)"
            if self.st["consecutive_errors"] >= MAX_CONSECUTIVE_ERRORS:
                self.st["consecutive_errors"] = 0
                wait = self.backoff_s()
                wait = self.begin_pause("agent_errors", wait)
                self.finish(row, "agent_error", f"{note}; {MAX_CONSECUTIVE_ERRORS} in a row, "
                            f"backing off {wait / 60:.1f} min", counted=counted)
                self.wait_pause()
                return counted
            return self.finish(row, "agent_error", note, counted=counted)

        if out.status == "timeout":
            self.restore(head, keep=(NOTES,))
            return self.finish(row, "timeout", f"agent exceeded {self.agent_timeout_s:.0f}s")

        self.st["backoff_idx"] = 0
        v = check_agent_changes(self.root, head)
        if v:
            # M4: keep the agent's notes unless notes.md itself is what broke the rules.
            self.restore(head, keep=() if NOTES in v else (NOTES,))
            row["hypothesis"] = f"{hyp} [violation: {v}]"
            return self.finish(row, "violation", v)

        fa2 = self.fa2()
        kind, data, detail = self.bench("--ours", f"{BENCH_DIR}/exp{exp_id}_1.json")
        self.clock.heartbeat()
        if kind != "ok":
            self.restore(head, keep=(NOTES,))
            return self.finish(row, kind, detail)
        ms1 = data["ours_ms"]
        s1 = score(ms1, fa2)
        row.update(score=s1, min_vs_fa2=min_ratio(ms1, fa2), **{f"ms_{s}": ms1[s] for s in SHAPES})
        best, thr = self.st["best_score"], self.st["threshold"]
        if not is_improvement(s1, best, thr):
            self.restore(head, keep=(NOTES,))
            return self.finish(row, "reverted", f"{s1:.4f} <= best {best:.4f} * (1+{thr:.4f})")

        kind2, data2, detail2 = self.bench("--ours", f"{BENCH_DIR}/exp{exp_id}_2.json")
        self.clock.heartbeat()
        if kind2 != "ok":
            self.restore(head, keep=(NOTES,))
            return self.finish(row, kind2, f"confirmation run: {detail2}")
        ms2 = data2["ours_ms"]
        s2 = score(ms2, fa2)
        if not is_improvement(s2, best, thr):
            self.restore(head, keep=(NOTES,))
            return self.finish(row, "reverted", f"first run {s1:.4f} won, confirmation {s2:.4f} "
                               f"did not beat {best:.4f} * (1+{thr:.4f})")
        self.st["best_score"] = (s1 + s2) / 2
        self.st["best_ours_ms"] = {s: (ms1[s] + ms2[s]) / 2 for s in SHAPES}
        return self.finish(row, "kept", f"scores {s1:.4f}, {s2:.4f}",
                           msg=f"exp {exp_id}: {_one_line(hyp)} (score {s1:.3f})",
                           extra=[KERNEL, NOTES, CURRENT])

    def finish(self, row: dict, status: str, note: str = "", counted: bool = True,
               msg: str | None = None, extra: tuple | list = (NOTES,)) -> bool:
        exp_id = row["exp_id"]
        row["status"] = status
        row["best_score"] = self.st["best_score"]
        row["active_s"] = round(self.clock.active_seconds(), 1)
        row["wall_s"] = round(self.clock.wall_seconds(), 1)
        append_result(self.p(RESULTS), row)
        if counted:
            self.st["next_exp_id"] = exp_id + 1
        self.save_state()
        paths = [*self.state_paths(), *extra]
        self.commit(msg or f"exp {exp_id} log ({status})", paths)
        if status == "kept":
            self.st["best_commit"] = self.head()
            self.save_state()
        self.progress(exp_id, status, row.get("score"), note)
        return counted


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(prog="python -m driver.loop")
    ap.add_argument("--max-experiments", type=int, default=None,
                    help="stop after N experiments in this process (usage-limit retries and agent errors "
                         "before any work excluded)")
    ap.add_argument("--active-hours", type=float, default=6.0)
    ap.add_argument("--wall-hours", type=float, default=36.0)
    ap.add_argument("--agent-timeout-min", type=float, default=20.0)
    ap.add_argument("--agent-timeout-s", type=float, default=None, help="overrides --agent-timeout-min")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None, root: Path | None = None,
         now: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep) -> int:
    args = parse_args(argv)
    root = Path(root) if root is not None else Path.cwd()

    def _on_term(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    old = None
    if threading.current_thread() is threading.main_thread():
        old = signal.signal(signal.SIGTERM, _on_term)
    try:
        return Loop(root, args, now, sleep).run()
    finally:
        if old is not None:
            signal.signal(signal.SIGTERM, old)


if __name__ == "__main__":
    raise SystemExit(main())
