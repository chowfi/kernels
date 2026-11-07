import pytest
import json
import stat
import sys
from pathlib import Path

from driver.claude_stream import AGENT_TOOLS, DISALLOWED_TOOLS, _parse_reset, parse_stream, run_claude
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
        "Bash(uv run --frozen python -m harness.check:*)",
        "Bash(uv run --frozen python -m harness.profile:*)",
        "Bash(git log:*)", "Bash(git diff:*)", "Bash(git show:*)"]


def test_agent_git_output_flags_denied():
    """M2: git log/diff/show could write files with --output=<file>; deny rules block it
    (verified with a real `claude -p --permission-mode dontAsk` scratch run, see final-fix-report)."""
    assert DISALLOWED_TOOLS == ["Bash(git log --output*)", "Bash(git diff --output*)",
                                "Bash(git show --output*)", "Bash(git * --output*)"]


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
    j = argv.index("--disallowedTools")
    assert argv[j + 1:i] == DISALLOWED_TOOLS


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


# ---------------------------------------------------------------- I2 / I3 / M6 / M7

def _rle(status, resets_at=1794025200, five=0.3, seven=0.11):
    return json.dumps({"type": "rate_limit_event", "rate_limit_info": {
        "status": status, "resetsAt": resets_at, "rateLimitType": "five_hour",
        "unifiedWindows": {"five_hour": {"utilization": five, "resetsAt": resets_at, "status": status},
                           "seven_day": {"utilization": seven, "resetsAt": 1794243600,
                                         "status": "allowed"}}}})


def _res(text, is_error=True, **kw):
    return json.dumps({"type": "result", "is_error": is_error, "result": text, **kw})


def test_rate_limit_event_rejected_is_usage_limit():
    o = parse_stream([_rle("rejected"), _res("API Error: something went wrong")])
    assert o.status == "usage_limit" and o.reset_at == 1794025200


def test_rate_limit_event_rejected_without_result():
    o = parse_stream([_rle("rejected", resets_at=1794025200000)])  # milliseconds
    assert o.status == "usage_limit" and o.reset_at == 1794025200


def test_rate_limit_event_allowed_is_not_a_limit_and_is_recorded():
    o = parse_stream([_rle("allowed", five=0.42, seven=0.17), _res("ok", is_error=False)])
    assert o.status == "ok" and o.reset_at is None
    assert o.rate_limit["unifiedWindows"]["five_hour"]["utilization"] == 0.42


def test_real_fixture_rate_limit_recorded():
    o = parse_stream(open(FIX / "stream_ok.jsonl"))
    assert o.rate_limit is not None and "five_hour" in o.rate_limit["unifiedWindows"]


@pytest.mark.parametrize("text", ["You've hit your limit · resets 3pm",
                                  "Claude AI usage limit reached|1762480800",
                                  "Rate limit exceeded"])
def test_limit_texts(text):
    assert parse_stream([_res(text)]).status == "usage_limit"


@pytest.mark.parametrize("text", ["Context limit reached", "context window limit reached",
                                  "Prompt is too long: context limit reached"])
def test_context_limit_is_not_usage_limit(text):
    assert parse_stream([_res(text)]).status == "error"


def _local(h, m=0):
    import datetime as dt
    return dt.datetime(2026, 11, 6, h, m).timestamp()


def test_parse_reset_text_rules():
    now = _local(13)
    assert _parse_reset("You've hit your limit · resets 3pm", now) == _local(15)
    assert _parse_reset("resets 2:30pm", now) == _local(14, 30)
    assert _parse_reset("resets 11am", now) is None          # already passed: no +1 day
    assert _parse_reset("resets 11pm", now) is None          # > 6 h away: use back-off
    assert _parse_reset("resets 1pm", now) is None           # == now


def test_usage_limit_tokens_fall_back_to_model_usage():
    o = parse_stream(open(FIX / "stream_limit.jsonl"))
    assert o.status == "usage_limit"
    assert o.usage == Usage(2, 4, 9075, 12072)


def test_run_claude_scans_stderr_for_limit(tmp_path):
    b = _fake(tmp_path, "import sys\nsys.stderr.write(\"You've hit your limit \\u00b7 resets 3pm\\n\")\nsys.exit(1)\n")
    o = run_claude("hi", tmp_path, 30, claude_bin=b)
    assert o.status == "usage_limit", o


def test_run_claude_stderr_context_limit_is_error(tmp_path):
    b = _fake(tmp_path, "import sys\nsys.stderr.write('context limit reached\\n')\nsys.exit(1)\n")
    assert run_claude("hi", tmp_path, 30, claude_bin=b).status == "error"


def test_run_claude_writes_stream_log(tmp_path):
    fx = FIX / "stream_ok.jsonl"
    b = _fake(tmp_path, f"import sys\nsys.stdout.write(open({str(fx)!r}).read())\n")
    log = tmp_path / "logs" / "agent.jsonl"
    o = run_claude("hi", tmp_path, 30, claude_bin=b, stream_log=log)
    assert o.status == "ok" and log.read_text() == fx.read_text()


def test_program_md_states_exact_allowed_commands():
    """M3: program.md names the exact allowed commands (with --frozen), never a bare `uv run python`."""
    text = (Path(__file__).resolve().parent.parent / "program.md").read_text()
    for rule in AGENT_TOOLS:
        if rule.startswith("Bash("):
            assert f"`{rule[5:].removesuffix(':*)')}" in text, rule
    assert "uv run python" not in text
    assert "--output" in text  # the denied git flag is mentioned


def test_real_five_hour_limit_stream():
    """Captured from smoke/run2 exp 1 (claude 2.1.319): rejected rate_limit_event + 429 result."""
    o = parse_stream(open(FIX / "stream_limit_real.jsonl"))
    assert o.status == "usage_limit" and o.reset_at == 1794025200
    assert o.rate_limit["unifiedWindows"]["five_hour"]["utilization"] == 1


@pytest.mark.parametrize("text", ["You've hit your 5-hour limit · resets 11:20pm (America/New_York)",
                                  "You've hit your weekly limit · resets Nov 10"])
def test_named_window_limit_texts_match_without_429(text):
    assert parse_stream([_res(text)]).status == "usage_limit"
