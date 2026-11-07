import pytest
import json
import stat
import sys
from pathlib import Path

from driver.claude_stream import AGENT_TOOLS, parse_stream, run_claude
from driver.ledger import TokenLedger, Usage

FIX = Path(__file__).parent / "fixtures"


def test_parse_ok_fixture():
    o = parse_stream(open(FIX / "stream_ok.jsonl"))
    assert o.status == "ok" and o.session_id and o.usage.output_tokens > 0
    assert o.usage.cache_read_tokens == 9075 and o.usage.cache_write_tokens == 12072
    assert o.result_text == "ok"


def test_parse_usage_limit():
    o = parse_stream(open(FIX / "stream_limit.jsonl"))
    assert o.status == "usage_limit" and o.reset_at == 1762480800


def test_parse_truncated_stream_sums_assistant_usage():
    lines = [l for l in open(FIX / "stream_ok.jsonl") if '"type":"result"' not in l.replace(" ", "")]
    o = parse_stream(lines)
    assert o.usage.output_tokens > 0
    assert o.usage.cache_write_tokens == 12072


def test_distinct_message_ids_only():
    def a(i, out):
        return json.dumps({"type": "assistant", "message": {"id": i, "usage": {"output_tokens": out}}})
    o = parse_stream([a("m1", 5), a("m1", 5), a("m2", 7), "garbage"])
    assert o.usage.output_tokens == 12


def test_error_without_limit():
    r = json.dumps({"type": "result", "is_error": True, "result": "boom", "api_error_status": 500,
                    "usage": {"output_tokens": 1}})
    o = parse_stream([r])
    assert o.status == "error"


def test_429_status_is_limit():
    r = json.dumps({"type": "result", "is_error": True, "result": "whatever", "api_error_status": 429})
    assert parse_stream([r]).status == "usage_limit"


def test_agent_tools_scoped():
    assert AGENT_TOOLS == [
        "Read", "Glob", "Grep", "Edit(kernel/attention.py)", "Write(kernel/attention.py)",
        "Edit(notes.md)", "Write(experiments/current.json)", "Edit(experiments/current.json)",
        "Bash(uv run python -m harness.check:*)", "Bash(uv run python -m harness.profile:*)",
        "Bash(git log:*)", "Bash(git diff:*)", "Bash(git show:*)"]


def test_429_status_without_is_error():
    r = json.dumps({"type": "result", "is_error": False, "result": "x", "api_error_status": 429})
    assert parse_stream([r]).status == "usage_limit"


def test_bare_429_anchored():
    r = json.dumps({"type": "result", "is_error": True, "result": "request id 14290 failed"})
    assert parse_stream([r]).status == "error"


def test_multiturn_uses_cumulative_result_usage():
    o = parse_stream(open(FIX / "stream_multiturn.jsonl"))
    assert o.status == "ok"
    assert o.usage == Usage(6, 369, 54741, 15485)


def test_multiturn_truncated_fallback_dedupes_ids():
    lines = [l for l in open(FIX / "stream_multiturn.jsonl") if '"type":"result"' not in l.replace(" ", "")]
    o = parse_stream(lines)
    assert o.usage.input_tokens == 6 and o.usage.cache_read_tokens == 54741
    assert o.usage.cache_write_tokens == 15485


def test_command_line(tmp_path):
    out = tmp_path / "argv.json"
    b = _fake(tmp_path, f"import sys, json\njson.dump(sys.argv[1:], open({str(out)!r}, 'w'))\n")
    run_claude("my prompt", tmp_path, 30, claude_bin=b)
    argv = json.loads(out.read_text())
    assert argv[:2] == ["-p", "my prompt"]
    def val(flag):
        return argv[argv.index(flag) + 1]
    assert val("--model") == "claude-opus-5-5" and val("--effort") == "high"
    assert val("--output-format") == "stream-json" and "--verbose" in argv
    assert val("--permission-mode") == "dontAsk"
    i = argv.index("--allowedTools")
    assert argv[i + 1:] == AGENT_TOOLS


def _fake(tmp_path, body):
    p = tmp_path / "fakeclaude"
    p.write_text(f"#!{sys.executable}\n" + body)
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def test_run_claude_ok(tmp_path):
    fx = FIX / "stream_ok.jsonl"
    b = _fake(tmp_path, f"import sys\nsys.stdout.write(open({str(fx)!r}).read())\n")
    o = run_claude("hi", tmp_path, 30, claude_bin=b)
    assert o.status == "ok" and o.usage.output_tokens > 0 and o.session_id


def test_run_claude_nonzero_exit_is_error(tmp_path):
    b = _fake(tmp_path, "import sys\nprint('not json')\nsys.exit(3)\n")
    assert run_claude("hi", tmp_path, 30, claude_bin=b).status == "error"


def test_run_claude_timeout_partial_usage(tmp_path):
    fx = FIX / "stream_ok.jsonl"
    body = (
        "import sys, time, subprocess\n"
        f"for l in open({str(fx)!r}):\n"
        "    if '\"type\":\"result\"' not in l.replace(' ', ''):\n"
        "        sys.stdout.write(l)\n"
        "sys.stdout.flush()\n"
        "subprocess.Popen(['sleep', '60'])\n"
        "time.sleep(60)\n"
    )
    b = _fake(tmp_path, body)
    o = run_claude("hi", tmp_path, 2, claude_bin=b)
    assert o.status == "timeout" and o.usage.output_tokens > 0
    assert o.duration_s < 20


def test_run_claude_silent_hang_times_out(tmp_path):
    b = _fake(tmp_path, "import time\ntime.sleep(60)\n")
    o = run_claude("hi", tmp_path, 1, claude_bin=b)
    assert o.status == "timeout"


def test_ledger(tmp_path):
    p = tmp_path / "l.json"
    led = TokenLedger(p)
    led.add(1, Usage(1, 2, 3, 4), "s1")
    led.add(2, Usage(10, 20, 30, 40), None)
    led2 = TokenLedger(p)
    assert led2.totals() == Usage(11, 22, 33, 44)
    assert led2.totals().total == 110
    assert led2.session_ids() == {"s1"}


def test_run_claude_kills_child_on_keyboard_interrupt(tmp_path):
    import os
    import signal
    import time
    pidf = tmp_path / "pid"
    b = _fake(tmp_path, f"import os, time\nopen({str(pidf)!r}, 'w').write(str(os.getpid()))\ntime.sleep(60)\n")

    def boom(*_):
        raise KeyboardInterrupt

    old = signal.signal(signal.SIGALRM, boom)
    signal.setitimer(signal.ITIMER_REAL, 1.0)
    try:
        with pytest.raises(KeyboardInterrupt):
            run_claude("hi", tmp_path, 30, claude_bin=b)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)
    pid = int(pidf.read_text())
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(pid, 0)
