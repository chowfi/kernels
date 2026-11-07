"""Run the final benchmark and rewrite the README results block (between RESULTS markers)."""
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from driver.clock import RunClock
from driver.ledger import TokenLedger, Usage
from driver.results import read_results
from harness.shapes import BENCH_SHAPES

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"


def _ratio(x: float) -> str:
    s = f"{x:.2f}×"
    return f"**{s}**" if round(x, 2) >= 1.00 else s


def ratios(final: dict) -> list[tuple]:
    out = []
    for s in BENCH_SHAPES:
        o, f, d = final["ours_ms"][s.name], final["fa2_ms"][s.name], final["sdpa_ms"][s.name]
        out.append((s, o, f, d, f / o, d / o))
    return out


def render_table(final: dict) -> str:
    lines = ["| Workload | seq_kv | ours (ms) | FA2 (ms) | SDPA (ms) | vs FA2 | vs SDPA |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for s, o, f, d, rf, rd in ratios(final):
        lines.append(f"| {s.workload.capitalize()} | {s.seq_kv} | {o:.3f} | {f:.3f} | {d:.3f} | "
                     f"{_ratio(rf)} | {_ratio(rd)} |")
    return "\n".join(lines)


def headline(final: dict) -> str:
    r = ratios(final)
    fa, sd = [x[4] for x in r], [x[5] for x in r]
    geo = math.exp(sum(math.log(x) for x in fa) / len(fa))
    return (f"**Ours vs FA2: {min(fa):.2f}×–{max(fa):.2f}× (geomean {geo:.2f}×); "
            f"vs PyTorch SDPA: {min(sd):.2f}×–{max(sd):.2f}×** across the 11 table shapes "
            f"(>1 means ours is faster).")


def sdpa_note(final: dict) -> str:
    labels = sorted({v.split(" (")[0] for v in (final.get("sdpa_backend") or {}).values()})
    if not labels:
        return "SDPA backend: not recorded."
    return (f"SDPA backend: on this card SDPA's default dispatch runs the `{'`, `'.join(labels)}` backend "
            f"({'PyTorch built-in flash kernel' if labels == ['flash'] else 'see per-shape labels'}), recorded per shape in `results/final.json`. "
            f"SDPA is called with `enable_gqa=True`.")


def count_experiments(rows: list[dict]) -> dict:
    seed = next((r for r in rows if r["status"] == "baseline"), None)
    exps = [r for r in rows if r["status"] not in ("baseline", "usage_limit")]
    n = lambda *st: sum(r["status"] in st for r in exps)  # noqa: E731
    scored = [r for r in rows if r["status"] != "usage_limit" and r["best_score"]]
    return {"total": len(exps), "kept": n("kept"), "reverted": n("reverted"),
            "failed_correctness": n("failed_correctness"),
            "other": n("crashed", "violation", "timeout", "agent_error"),
            "usage_limit_pauses": sum(r["status"] == "usage_limit" for r in rows),
            "seed_score": float(seed["score"]) if seed and seed["score"] else None,
            "best_score": float(scored[-1]["best_score"]) if scored else None}


def fmt_dur(s: float) -> str:
    m = int(round(s / 60))
    return f"{m // 60}h {m % 60:02d}m"


def read_clock(path: Path) -> tuple[float, float, float]:
    """(active, wall, paused) seconds as of the last recorded heartbeat/pause (never writes the original)."""
    path = Path(path)
    if not path.exists():
        return 0.0, 0.0, 0.0
    st = json.loads(path.read_text())
    end = st.get("paused_since") or st.get("last_heartbeat") or st.get("start") or 0.0
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d) / "clock.json"
        shutil.copy(path, tmp)
        c = RunClock(tmp, now=lambda: end)
        return c.active_seconds(), c.wall_seconds(), st.get("paused_seconds", 0.0) + (
            end - st["paused_since"] if st.get("paused_since") else 0.0)


def _tok(u) -> str:
    return (f"input {u.input_tokens:,}, output {u.output_tokens:,}, cache read {u.cache_read_tokens:,}, "
            f"cache write {u.cache_write_tokens:,}; total {u.total:,}")


def render_accounting(counts: dict, loop: Usage, build: dict | None, active_s: float, wall_s: float,
                      paused_s: float, stop_reason: str | None) -> str:
    L = ["**Accounting**", "",
         f"- Experiments: {counts['total']} total ({counts['kept']} kept, {counts['reverted']} reverted, "
         f"{counts['failed_correctness']} failed correctness, {counts['other']} other: crash/violation/timeout/agent error). "
         f"The seed (exp 0) and usage-limit pauses are not counted as experiments.",
         f"- Score (geomean FA2 ms / ours ms): seed {counts['seed_score']:.3f}, final best {counts['best_score']:.3f}"
         if counts["seed_score"] is not None and counts["best_score"] is not None else "- Score: n/a",
         f"- Active time: {fmt_dur(active_s)}; wall time: {fmt_dur(wall_s)}; "
         f"{counts['usage_limit_pauses']} usage-limit pause(s), {fmt_dur(paused_s)} paused in total",
         f"- Stop reason: {stop_reason or 'n/a'}",
         f"- Loop tokens (agent invocations, exact per-session totals from Claude result events): {_tok(loop)}. "
         f"Experiments that hit the 20-minute timeout report no result event, so loop tokens are a lower bound.",
         ]
    if build:
        b = Usage(build["input_tokens"], build["output_tokens"], build["cache_read_tokens"], build["cache_write_tokens"])
        L.append(f"- Build tokens (interactive sessions plus subagents that built the harness and driver, "
                 f"from local transcripts): {_tok(b)}; span {build['first_ts'][:16]}Z to {build['last_ts'][:16]}Z")
    else:
        L.append("- Build tokens: run `uv run python -m scripts.token_report` first.")
    return "\n".join(L)


def update_readme(path: Path, block: str) -> None:
    t = Path(path).read_text()
    i, j = t.find(START), t.find(END)
    if i < 0 or j < i:
        raise SystemExit(f"{path}: RESULTS markers not found")
    Path(path).write_text(t[: i + len(START)] + "\n" + block + "\n" + t[j:])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-json", type=Path)
    ap.add_argument("--final-json", type=Path, default=Path("results/final.json"))
    ap.add_argument("--readme", type=Path, default=Path("README.md"))
    ap.add_argument("--results-tsv", type=Path, default=Path("results.tsv"))
    ap.add_argument("--tokens-json", type=Path, default=Path("results/tokens.json"))
    ap.add_argument("--clock-json", type=Path, default=Path("results/clock.json"))
    ap.add_argument("--run-state", type=Path, default=Path("results/run_state.json"))
    ap.add_argument("--token-report", type=Path, default=Path("results/token_report.json"))
    a = ap.parse_args()
    if a.from_json:
        final_path = a.from_json
    else:
        final_path = a.final_json
        rc = subprocess.run(["uv", "run", "python", "-m", "harness.bench", "--all", "--out", str(final_path)]).returncode
        if rc != 0 and not final_path.exists():
            print("bench failed", file=sys.stderr)
            return rc or 1
    final = json.loads(final_path.read_text())
    if not final.get("correct"):
        print(f"REFUSING to publish: kernel is not correct ({final.get('error', 'correctness failed')})", file=sys.stderr)
        return 1
    counts = count_experiments(read_results(a.results_tsv))
    active, wall, paused = read_clock(a.clock_json)
    stop = json.loads(a.run_state.read_text()).get("stop_reason") if a.run_state.exists() else None
    build = json.loads(a.token_report.read_text())["build"] if a.token_report.exists() else None
    block = "\n\n".join([headline(final), render_table(final), sdpa_note(final),
                         render_accounting(counts, TokenLedger(a.tokens_json).totals(), build, active, wall, paused, stop)])
    update_readme(a.readme, block)
    print(f"updated {a.readme}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
