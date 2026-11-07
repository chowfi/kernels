import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.final_report import (count_experiments, render_accounting, render_table,
                                  update_readme, headline)
from scripts.token_report import split_tokens

FIX = Path(__file__).parent / "fixtures"


def sample_final_json():
    return json.loads((FIX / "final_sample.json").read_text())


def fixture_transcripts(tmp_path):
    d = tmp_path / "projects"
    shutil.copytree(FIX / "transcripts", d)
    return d


def test_table_format(tmp_path):
    md = render_table(sample_final_json())
    assert md.splitlines()[0] == "| Workload | seq_kv | ours (ms) | FA2 (ms) | SDPA (ms) | vs FA2 | vs SDPA |"
    assert "| Prefill | 2048 |" in md and "| Decode | 65536 |" in md
    assert "| **1.03×** |" in md          # >= 1.00 bold
    assert "| 0.90× |" in md              # < 1.00 plain
    assert len(md.splitlines()) == 2 + 11


def test_headline_ranges():
    h = headline(sample_final_json())
    assert "0.90×" in h and "1.03×" in h and "FA2" in h and "SDPA" in h


def test_token_split(tmp_path):
    r = split_tokens(fixture_transcripts(tmp_path), loop_sessions={"loop-1"})
    assert r["loop"]["output_tokens"] == 7 and r["build"]["output_tokens"] == 5   # duplicates by message.id counted once
    assert r["build"]["input_tokens"] == 2          # subagent transcript included
    assert r["build"]["first_ts"].startswith("2026-11-06T10:00:00")
    assert r["build"]["last_ts"].startswith("2026-11-06T12:00:00")
    assert r["project"]["first_ts"].startswith("2026-11-06T10:00:00")
    assert r["project"]["last_ts"].startswith("2026-11-07T01:00:05")


def test_experiment_counting_excludes_baseline_and_usage_limit():
    from driver.results import read_results
    c = count_experiments(read_results(FIX / "results_sample.tsv"))
    assert c == {"total": 7, "kept": 2, "reverted": 2, "failed_correctness": 1, "other": 2,
                 "usage_limit_pauses": 1, "seed_score": 0.27, "best_score": 0.90}


def test_update_readme_between_markers(tmp_path):
    p = tmp_path / "README.md"
    p.write_text("a\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\nb\n")
    update_readme(p, "NEW")
    assert p.read_text() == "a\n<!-- RESULTS:START -->\nNEW\n<!-- RESULTS:END -->\nb\n"


def test_final_report_refuses_incorrect(tmp_path):
    readme = tmp_path / "README.md"
    orig = "<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\n"
    readme.write_text(orig)
    r = subprocess.run([sys.executable, "-m", "scripts.final_report", "--from-json",
                        str(FIX / "final_incorrect.json"), "--readme", str(readme)],
                       capture_output=True, text=True, cwd=Path(__file__).parent.parent)
    assert r.returncode != 0 and readme.read_text() == orig


def test_final_report_end_to_end(tmp_path):
    res = tmp_path / "results"
    res.mkdir()
    (res / "tokens.json").write_text(json.dumps({"entries": [], "totals": {
        "input_tokens": 1, "output_tokens": 2, "cache_read_tokens": 3, "cache_write_tokens": 4}}))
    (res / "clock.json").write_text(json.dumps({"start": 1000.0, "pauses": [], "paused_seconds": 600.0,
                                                "paused_since": None, "paused_reason": None,
                                                "last_heartbeat": 1000.0 + 7200}))
    (res / "run_state.json").write_text(json.dumps({"stop_reason": "budget"}))
    readme = tmp_path / "README.md"
    readme.write_text("x\n<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\n")
    r = subprocess.run([sys.executable, "-m", "scripts.final_report", "--from-json", str(FIX / "final_sample.json"),
                        "--readme", str(readme), "--results-tsv", str(FIX / "results_sample.tsv"),
                        "--tokens-json", str(res / "tokens.json"), "--clock-json", str(res / "clock.json"),
                        "--run-state", str(res / "run_state.json"),
                        "--token-report", str(tmp_path / "none.json")],
                       capture_output=True, text=True, cwd=Path(__file__).parent.parent)
    assert r.returncode == 0, r.stderr
    t = readme.read_text()
    assert "| Prefill | 2048 |" in t and "budget" in t and "1h 50m" in t and "2h 00m" in t
    assert "old" not in t


def test_accounting_block():
    from driver.ledger import Usage
    s = render_accounting(counts={"total": 7, "kept": 2, "reverted": 2, "failed_correctness": 1, "other": 2,
                                  "usage_limit_pauses": 1, "seed_score": 0.27, "best_score": 0.9},
                          loop=Usage(1, 2, 3, 4), build=None, active_s=3600, wall_s=7200,
                          paused_s=3600, stop_reason="budget")
    assert "7 total" in s and "stop reason" in s.lower() and "lower bound" in s


def test_plot_progress(tmp_path):
    out = tmp_path / "p.png"
    r = subprocess.run([sys.executable, "-m", "scripts.plot_progress", "--results-tsv",
                        str(FIX / "results_sample.tsv"), "--out", str(out)],
                       capture_output=True, text=True, cwd=Path(__file__).parent.parent)
    assert r.returncode == 0, r.stderr
    assert out.stat().st_size > 5000


def test_bold_uses_unrounded_ratio():
    from scripts.final_report import _ratio
    assert _ratio(0.996) == "1.00×" and _ratio(1.0) == "**1.00×**"


def test_read_clock_open_pause_not_double_counted(tmp_path):
    from scripts.final_report import read_clock
    p = tmp_path / "clock.json"
    # start 0, closed pause 600s, open pause since 5000, no heartbeat after
    p.write_text(json.dumps({"start": 0.0, "pauses": [], "paused_seconds": 600.0, "paused_since": 5000.0,
                             "paused_reason": "usage_limit", "last_heartbeat": 5000.0}))
    active, wall, paused = read_clock(p)
    assert (active, wall, paused) == (4400.0, 5000.0, 600.0)


def test_final_report_bench_crash_does_not_publish_stale(tmp_path):
    readme = tmp_path / "README.md"
    orig = "<!-- RESULTS:START -->\nold\n<!-- RESULTS:END -->\n"
    readme.write_text(orig)
    stale = tmp_path / "final.json"
    shutil.copy(FIX / "final_sample.json", stale)
    r = subprocess.run([sys.executable, "-m", "scripts.final_report", "--final-json", str(stale),
                        "--bench-cmd", "false", "--readme", str(readme)],
                       capture_output=True, text=True, cwd=Path(__file__).parent.parent)
    assert r.returncode != 0 and readme.read_text() == orig and not stale.exists()
