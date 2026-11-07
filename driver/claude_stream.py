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
    "Bash(uv run python -m harness.check:*)",
    "Bash(uv run python -m harness.profile:*)",
    "Bash(git log:*)", "Bash(git diff:*)", "Bash(git show:*)",
]

# acceptEdits auto-approves ANY edit under cwd, which defeats the path-scoped
# Edit/Write rules (verified: other.txt got written). dontAsk denies everything
# not in --allowedTools, so the scoping is real (verified).
PERMISSION_MODE = "dontAsk"

LIMIT_RE = re.compile(r"usage limit|rate limit|429|limit reached", re.I)


@dataclass
class ClaudeOutcome:
    status: Literal["ok", "usage_limit", "error", "timeout"]
    usage: Usage
    session_id: str | None
    result_text: str
    reset_at: float | None
    duration_s: float = 0.0


def _usage_from(u: dict) -> Usage:
    return Usage(int(u.get("input_tokens") or 0), int(u.get("output_tokens") or 0),
                 int(u.get("cache_read_input_tokens") or 0),
                 int(u.get("cache_creation_input_tokens") or 0))


def _parse_reset(text: str, now: float | None = None) -> float | None:
    m = re.search(r"\|\s*(\d{9,})", text)
    if m:
        return float(m.group(1))
    m = re.search(r"resets?\s+(?:at\s+)?(\d{1,2})(?::(\d{2}))?\s*(am|pm)?", text, re.I)
    if m:
        h, mi, ap = int(m.group(1)), int(m.group(2) or 0), (m.group(3) or "").lower()
        if ap == "pm" and h < 12:
            h += 12
        elif ap == "am" and h == 12:
            h = 0
        if h > 23 or mi > 59:
            return None
        n = dt.datetime.fromtimestamp(now if now is not None else time.time())
        t = n.replace(hour=h, minute=mi, second=0, microsecond=0)
        if t <= n:
            t += dt.timedelta(days=1)
        return t.timestamp()
    return None


def parse_stream(lines: Iterable[str]) -> ClaudeOutcome:
    session_id = None
    result = None
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

    if result is None:
        usage = Usage()
        for u in seen.values():
            usage = usage + u
        return ClaudeOutcome("error", usage, session_id, "", None)

    if isinstance(result.get("usage"), dict):
        usage = _usage_from(result["usage"])
    else:
        usage = Usage()
        for u in seen.values():
            usage = usage + u
    text = result.get("result") or ""
    if not isinstance(text, str):
        text = str(text)
    status: str = "ok"
    reset = None
    if result.get("is_error"):
        status = "error"
        if result.get("api_error_status") == 429 or LIMIT_RE.search(text):
            status = "usage_limit"
            reset = _parse_reset(text)
    return ClaudeOutcome(status, usage, session_id, text, reset)  # type: ignore[arg-type]


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def run_claude(prompt: str, cwd: Path, timeout_s: int,
               claude_bin: str = os.environ.get("CLAUDE_BIN", "claude")) -> ClaudeOutcome:
    cmd = [claude_bin, "-p", prompt, "--model", "claude-opus-5-5", "--effort", "high",
           "--output-format", "stream-json", "--verbose", "--permission-mode", PERMISSION_MODE,
           "--allowedTools", *AGENT_TOOLS]
    t0 = time.monotonic()
    deadline = t0 + timeout_s
    lines: list[str] = []
    q: queue.Queue = queue.Queue()
    with tempfile.TemporaryFile("w+") as errf:
        proc = subprocess.Popen(cmd, cwd=str(cwd), stdin=subprocess.DEVNULL,
                                stdout=subprocess.PIPE, stderr=errf, text=True,
                                start_new_session=True)

        def reader():
            for ln in proc.stdout:  # type: ignore[union-attr]
                q.put(ln)
            q.put(None)

        th = threading.Thread(target=reader, daemon=True)
        th.start()
        timed_out = False
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
    out = parse_stream(lines)
    out.duration_s = time.monotonic() - t0
    if timed_out:
        out.status = "timeout"
    elif out.status != "usage_limit" and proc.returncode not in (0, None):
        out.status = "error"
        if err_tail and not out.result_text:
            out.result_text = err_tail
    elif out.status == "error" and err_tail and not out.result_text:
        out.result_text = err_tail
    return out
