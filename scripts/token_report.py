"""Split Claude Code token usage into loop vs build sessions from on-disk transcripts.

Transcripts live in ~/.claude/projects/<project-slug>/<session>.jsonl; subagent transcripts live in
<project-slug>/<session>/subagents/agent-*.jsonl (attributed to their parent session).
Assistant entries repeat per message.id with mid-stream usage snapshots, so each usage
field is the MAX over all entries sharing a message.id (taking the last undercounts output ~2x).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

DEFAULT_PROJECTS = Path.home() / ".claude" / "projects" / "-home-fchow-Documents-kernels"
FIELDS = {"input_tokens": "input_tokens", "output_tokens": "output_tokens",
          "cache_read_tokens": "cache_read_input_tokens", "cache_write_tokens": "cache_creation_input_tokens"}


def session_of(path: Path, root: Path) -> str:
    rel = path.relative_to(root)
    return rel.parts[0] if len(rel.parts) > 1 else path.stem


def scan(projects_dir: Path) -> dict:
    """-> {message_id: {"session", "usage": {field: max}}} plus per-session timestamps."""
    root = Path(projects_dir)
    msgs: dict[str, dict] = {}
    ts: dict[str, list[str]] = {}
    for p in sorted(root.rglob("*.jsonl")):
        sid = session_of(p, root)
        lo_hi = ts.setdefault(sid, ["", ""])
        for line in p.read_text().splitlines():
            try:
                d = json.loads(line)
            except ValueError:
                continue
            t = d.get("timestamp")
            if isinstance(t, str):
                if not lo_hi[0] or t < lo_hi[0]:
                    lo_hi[0] = t
                if t > lo_hi[1]:
                    lo_hi[1] = t
            m = d.get("message")
            if d.get("type") != "assistant" or not isinstance(m, dict) or not m.get("id"):
                continue
            u = m.get("usage") or {}
            rec = msgs.setdefault(m["id"], {"session": sid, "usage": dict.fromkeys(FIELDS, 0)})
            for k, src in FIELDS.items():
                rec["usage"][k] = max(rec["usage"][k], int(u.get(src) or 0))
    return {"messages": msgs, "ts": ts}


def split_tokens(projects_dir: Path, loop_sessions: set[str]) -> dict:
    s = scan(projects_dir)
    out = {g: {**dict.fromkeys(FIELDS, 0), "messages": 0, "sessions": 0, "first_ts": None, "last_ts": None}
           for g in ("loop", "build", "project")}
    for rec in s["messages"].values():
        for g in ("project", "loop" if rec["session"] in loop_sessions else "build"):
            out[g]["messages"] += 1
            for k in FIELDS:
                out[g][k] += rec["usage"][k]
    for sid, (lo, hi) in s["ts"].items():
        if not lo:
            continue
        for g in ("project", "loop" if sid in loop_sessions else "build"):
            o = out[g]
            o["sessions"] += 1
            o["first_ts"] = lo if o["first_ts"] is None or lo < o["first_ts"] else o["first_ts"]
            o["last_ts"] = hi if o["last_ts"] is None or hi > o["last_ts"] else o["last_ts"]
    for o in out.values():
        o["total_tokens"] = sum(o[k] for k in FIELDS)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--projects-dir", type=Path, default=DEFAULT_PROJECTS)
    ap.add_argument("--tokens-json", type=Path, default=Path("results/tokens.json"))
    ap.add_argument("--out", type=Path, default=Path("results/token_report.json"))
    a = ap.parse_args()
    from driver.ledger import TokenLedger
    loop_sessions = TokenLedger(a.tokens_json).session_ids()
    r = split_tokens(a.projects_dir, loop_sessions)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(r, indent=2) + "\n")
    for g in ("loop", "build", "project"):
        o = r[g]
        print(f"{g:8s} in={o['input_tokens']:,} out={o['output_tokens']:,} cache_read={o['cache_read_tokens']:,} "
              f"cache_write={o['cache_write_tokens']:,} total={o['total_tokens']:,} "
              f"sessions={o['sessions']} span={o['first_ts']} -> {o['last_ts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
