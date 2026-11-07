"""Stand-in for `claude -p ... --output-format stream-json` used by tests/test_loop.py.

Behaviour is chosen by env FAKE_MODE:
  improve          hypothesis + BLOCK_N=64 -> 128 (or a harmless comment if already 128) + notes.md
  break            hypothesis + scale the output by 1.01 (wrong answer)
  touch_harness    appends a comment to harness/score.py (guard violation)
  untracked        writes scratch/evil.py (untracked file outside the allowed set)
  touch_tolerances edits results/tolerances.json (protected file)
  limit_once       usage-limit stream the first time (marker in $FAKE_STATE_DIR), then `improve`
  error            error result, exit 1
  hang             sleeps forever (writes its pid to $FAKE_PID_FILE if set)
FAKE_RESET_IN: seconds until the reset time reported by limit_once (default 120).
The cwd is the repo root (run_claude sets it). Writes nothing outside the repo except the
limit_once marker in FAKE_STATE_DIR.
"""
import json
import os
import sys
import time
from pathlib import Path

SID = "fake-session-0001"
ROOT = Path.cwd()


def emit(ev: dict) -> None:
    sys.stdout.write(json.dumps(ev) + "\n")
    sys.stdout.flush()


def usage(i=10, o=20, cr=300, cw=40) -> dict:
    return {"input_tokens": i, "output_tokens": o, "cache_read_input_tokens": cr,
            "cache_creation_input_tokens": cw}


def ok_stream(text: str = "done") -> None:
    emit({"type": "system", "subtype": "init", "session_id": SID})
    emit({"type": "assistant", "session_id": SID,
          "message": {"id": "msg_1", "content": [{"type": "text", "text": text}], "usage": usage()}})
    emit({"type": "result", "subtype": "success", "is_error": False, "result": text,
          "session_id": SID, "usage": usage()})


def hypothesis(h: str) -> None:
    p = ROOT / "experiments" / "current.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"hypothesis": h}))


def improve() -> None:
    hypothesis("larger KV tile\tBLOCK_N=128\nfor fewer loop iterations")
    k = ROOT / "kernel" / "attention.py"
    src = k.read_text()
    if "BLOCK_N=64," in src:
        src = src.replace("BLOCK_N=64,", "BLOCK_N=128,")
    else:
        src += "\n# fake tweak\n"
    k.write_text(src)
    with open(ROOT / "notes.md", "a") as f:
        f.write("- tried BLOCK_N=128\n")
    ok_stream()


def main() -> int:
    mode = os.environ.get("FAKE_MODE", "improve")
    if mode == "improve":
        improve()
    elif mode == "break":
        hypothesis("scale output")
        k = ROOT / "kernel" / "attention.py"
        k.write_text(k.read_text().replace("acc = acc / l_i[:, None]",
                                           "acc = acc / l_i[:, None] * 1.01"))
        ok_stream()
    elif mode == "touch_harness":
        hypothesis("loosen the scorer")
        with open(ROOT / "harness" / "score.py", "a") as f:
            f.write("\n# tampered\n")
        ok_stream()
    elif mode == "untracked":
        hypothesis("sneaky helper")
        (ROOT / "scratch").mkdir(exist_ok=True)
        (ROOT / "scratch" / "evil.py").write_text("print('hi')\n")
        ok_stream()
    elif mode == "touch_tolerances":
        hypothesis("relax tolerances")
        p = ROOT / "results" / "tolerances.json"
        d = json.loads(p.read_text())
        d["_tampered"] = True
        p.write_text(json.dumps(d))
        ok_stream()
    elif mode == "limit_once":
        marker = Path(os.environ["FAKE_STATE_DIR"]) / "limit_hit"
        if marker.exists():
            improve()
        else:
            marker.write_text("1")
            reset_in = int(os.environ.get("FAKE_RESET_IN", "120"))
            text = f"Claude AI usage limit reached|{int(time.time()) + reset_in}"
            emit({"type": "system", "subtype": "init", "session_id": SID})
            emit({"type": "assistant", "session_id": SID,
                  "message": {"id": "msg_l", "content": [{"type": "text", "text": text}],
                              "usage": usage(1, 2, 3, 4)}})
            emit({"type": "result", "is_error": True, "result": text, "session_id": SID,
                  "usage": usage(1, 2, 3, 4)})
    elif mode == "error":
        emit({"type": "system", "subtype": "init", "session_id": SID})
        emit({"type": "result", "is_error": True, "result": "boom", "api_error_status": 500,
              "session_id": SID, "usage": usage(5, 1, 0, 0)})
        return 1
    elif mode == "hang":
        if os.environ.get("FAKE_PID_FILE"):
            Path(os.environ["FAKE_PID_FILE"]).write_text(str(os.getpid()))
        emit({"type": "system", "subtype": "init", "session_id": SID})
        while True:
            time.sleep(60)
    else:
        print(f"unknown FAKE_MODE {mode}", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main())
