"""Run headless Claude Code and parse its stream-json output."""
from __future__ import annotations

import datetime as dt
import json
import os
import queue
import re
import signal
import subprocess
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal

from driver.ledger import Usage

# Path-scoped rules (gitignore-style, relative to cwd = repo root). `claude --help`
# says --allowedTools accepts comma or space separated values; we pass them as
# separate argv entries (space-separated form).
AGENT_TOOLS = [
    "Read", "Glob", "Grep",
    "Edit(kernel/attention.py)", "Write(kernel/attention.py)",
    "Edit(notes.md)",
    "Write(experiments/current.json)", "Edit(experiments/current.json)",
    "Bash(uv run --frozen python -m harness.check:*)",
    "Bash(uv run --frozen python -m harness.profile:*)",
    "Bash(git log:*)", "Bash(git diff:*)", "Bash(git show:*)",
]
# M2: git log/diff/show can write a file with --output=<file> (git rejects abbreviations like
# --outp, and these commands have no -o form). Deny rules win over allow rules; verified with a
# real dontAsk scratch run (all four --output forms denied, plain git log/diff/show allowed).
DISALLOWED_TOOLS = ["Bash(git log --output*)", "Bash(git diff --output*)",
                    "Bash(git show --output*)", "Bash(git * --output*)"]

# acceptEdits auto-approves ANY edit under cwd, which defeats the path-scoped
# Edit/Write rules (verified: other.txt got written). dontAsk denies everything
# not in --allowedTools, so the scoping is real (verified).
PERMISSION_MODE = "dontAsk"

# "limit reached" alone is ambiguous: "context limit reached" / "context window limit reached" is a
# prompt-size error, not a usage limit (M7).
LIMIT_RE = re.compile(r"usage limit|rate limit|\b429\b|hit your (?:[\w-]+ )?limit"
                      r"|(?<!context )(?<!window )limit reached", re.I)
MAX_TEXT_RESET_S = 6 * 3600  # a reset parsed from text further away than this is not trusted (I3)


@dataclass
class ClaudeOutcome:
    status: Literal["ok", "usage_limit", "error", "timeout"]
    usage: Usage
    session_id: str | None
    result_text: str
    reset_at: float | None
    duration_s: float = 0.0
    rate_limit: dict | None = None  # last rate_limit_event.rate_limit_info seen in the stream
    has_result: bool = False


def _usage_from(u: dict) -> Usage:
    return Usage(int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0),
                 int(u.get("cache_read_input_tokens") or 0),
                 int(u.get("cache_creation_input_tokens") or 0))


def _epoch_s(x) -> float | None:
    """resetsAt may be epoch seconds or milliseconds."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if v <= 0:
        return None
    return v / 1000.0 if v > 1e11 else v


def _parse_reset(text: str, now: float | None = None) -> float | None:
    """Reset time from a limit message. `...|<epoch>` is trusted as is; a wall-clock time like
    "resets 3pm" (local time, today) is used only if it is in the future and at most 6 h away;
    otherwise None, so the caller falls back to its back-off schedule (I3)."""
    m = re.search(r"\|\s*(\d{9,})", text)
    if m:
        return _epoch_s(m.group(1))
    m = re.search(r"resets?\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", text, re.I)
    if m:
        h, mi, ap = int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "").lower()
        if ap == "pm" and h < 12:
            h += 12
        elif ap == "am" and h == 12:
            h = 0
        if h > 23 or mi > 59:
            return None
        now = now if now is not None else time.time()
        t = dt.datetime.fromtimestamp(now).replace(hour=h, minute=mi, second=0, microsecond=0)
        ts = t.timestamp()
        if ts <= now or ts - now > MAX_TEXT_RESET_S:
            return None
        return ts
    return None


def _usage_from_model_usage(mu) -> Usage:
    """Sum result.modelUsage (camelCase, per model): used when result.usage is all zero (M6)."""
    total = Usage()
    if isinstance(mu, dict):
        for m in mu.values():
            if isinstance(m, dict):
                total = total + Usage(int(m.get("inputTokens") or 0), int(m.get("outputTokens") or 0),
                                      int(m.get("cacheReadInputTokens") or 0),
                                      int(m.get("cacheCreationInputTokens") or 0))
    return total


def parse_stream(lines: Iterable[str]) -> ClaudeOutcome:
    session_id = None
    result = None
    rate_limit = None
    seen: dict[str, Usage] = {}
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        if not isinstance(ev, dict):
            continue
        t = ev.get("type")
        if t == "system" and ev.get("subtype") == "init":
            session_id = ev.get("session_id") or session_id
        elif t == "assistant":
            msg = ev.get("message") or {}
            if msg.get("id") and isinstance(msg.get("usage"), dict):
                seen[msg["id"]] = _usage_from(msg["usage"])  # last write per id wins
            session_id = session_id or ev.get("session_id")
        elif t == "result":
            result = ev
            session_id = ev.get("session_id") or session_id
        elif t == "rate_limit_event" and isinstance(ev.get("rate_limit_info"), dict):
            rate_limit = ev["rate_limit_info"]

    rejected = bool(rate_limit) and rate_limit.get("status") == "rejected"
    limit_reset = _epoch_s(rate_limit.get("resetsAt")) if rejected else None

    if result is None:
        usage = Usage()
        for u in seen.values():
            usage = usage + u
        if rejected:  # I2: the stream died on a rejected rate-limit window
            return ClaudeOutcome("usage_limit", usage, session_id, "rate_limit_event rejected",
                                 limit_reset, rate_limit=rate_limit)
        return ClaudeOutcome("error", usage, session_id, "", None, rate_limit=rate_limit)

    # Evidence (tests/fixtures/stream_multiturn.jsonl, 3 turns, 3 distinct ids): the
    # result.usage is cumulative over all turns and matches modelUsage exactly
    # (in 6, cache_read 54741, cache_write 15485, out 369). The per-message sum agrees
    # on input/cache fields but UNDERCOUNTS output (173 vs 369): assistant events carry
    # mid-stream output_tokens snapshots. So result.usage is preferred; the per-message
    # sum (last usage per id) is only a lower-bound fallback for timeouts/crashes.
    if isinstance(result.get("usage"), dict):
        usage = _usage_from(result["usage"])
    else:
        usage = Usage()
        for u in seen.values():
            usage = usage + u
    if usage.total == 0:  # e.g. usage-limit results report zeros in `usage` but not in modelUsage
        usage = _usage_from_model_usage(result.get("modelUsage"))
    text = result.get("result") or ""
    if not isinstance(text, str):
        text = str(text)
    status: str = "ok"
    reset = None
    if result.get("is_error"):
        status = "error"
        if LIMIT_RE.search(text) or rejected:
            status = "usage_limit"
    if result.get("api_error_status") == 429:
        status = "usage_limit"
    if status == "usage_limit":
        reset = limit_reset if limit_reset else _parse_reset(text)  # prefer resetsAt (I3)
    return ClaudeOutcome(status, usage, session_id, text, reset,  # type: ignore[arg-type]
                         rate_limit=rate_limit, has_result=True)


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def run_claude(prompt: str, cwd: Path, timeout_s: int,
               claude_bin: str = os.environ.get("CLAUDE_BIN", "claude"),
               stream_log: Path | None = None) -> ClaudeOutcome:
    """stream_log: if given, the raw stream-json lines are written there (debugging/audit)."""
    cmd = [claude_bin, "-p", prompt, "--model", "claude-opus-5-5", "--effort", "high",
           "--output-format", "stream-json", "--verbose", "--permission-mode", PERMISSION_MODE,
           "--disallowedTools", *DISALLOWED_TOOLS, "--allowedTools", *AGENT_TOOLS]
    t0 = time.monotonic()
    deadline = t0 + timeout_s
    lines: list[str] = []
    q: queue.Queue = queue.Queue()
    with tempfile.TemporaryFile("w+") as errf:
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=errf, text=True, errors="replace",
                                start_new_session=True)

        def reader():
            try:
                for ln in proc.stdout:  # type: ignore[union-attr]
                    q.put(ln)
            finally:
                q.put(None)

        th = threading.Thread(target=reader, daemon=True)
        th.start()
        timed_out = False
        try:
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break
                try:
                    ln = q.get(timeout=min(remaining, 1.0))
                except queue.Empty:
                    continue
                if ln is None:
                    break
                lines.append(ln)
        except BaseException:
            # KeyboardInterrupt / SIGTERM (driver raises KeyboardInterrupt): claude runs in its
            # own session, so it would survive the driver unless we kill its group here.
            _kill_group(proc)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                pass
            raise
        if timed_out:
            _kill_group(proc)
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        _kill_group(proc)  # reap stray children / stuck process
        if proc.poll() is None:
            proc.wait()
        th.join(timeout=5)
        while True:  # drain anything read but not yet consumed
            try:
                ln = q.get_nowait()
            except queue.Empty:
                break
            if ln is not None:
                lines.append(ln)
        errf.seek(0)
        err_tail = errf.read()[-2000:]
    if stream_log is not None:
        try:
            Path(stream_log).parent.mkdir(parents=True, exist_ok=True)
            Path(stream_log).write_text("".join(lines))
        except OSError:
            pass
    out = parse_stream(lines)
    out.duration_s = time.monotonic() - t0
    if not timed_out and not out.has_result and out.status == "error" and LIMIT_RE.search(err_tail):
        out.status = "usage_limit"  # I2: limit reported only on stderr (no result event)
        out.result_text = err_tail.strip()[-500:]
        out.reset_at = _parse_reset(err_tail)
        return out
    if timed_out:
        out.status = "timeout"
    elif out.status != "usage_limit" and proc.returncode not in (0, None):
        out.status = "error"
        if err_tail and not out.result_text:
            out.result_text = err_tail
    elif out.status == "error" and err_tail and not out.result_text:
        out.result_text = err_tail
    return out
