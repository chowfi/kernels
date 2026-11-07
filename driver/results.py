"""results.tsv: one row per experiment (plus usage-limit rows, which are not experiments)."""
from __future__ import annotations

import csv
from pathlib import Path

from harness.shapes import BENCH_SHAPES

RESULTS_COLUMNS = ["exp_id", "utc", "hypothesis", "status", "score", "best_score", "min_vs_fa2",
                   *[f"ms_{s.name}" for s in BENCH_SHAPES],
                   "input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens",
                   "agent_seconds", "active_s", "wall_s", "session_id", "commit"]

STATUSES = {"kept", "reverted", "failed_correctness", "crashed", "violation", "timeout",
            "agent_error", "baseline", "usage_limit"}


NO_HYPOTHESIS = "(none recorded)"


def is_experiment(row: dict) -> bool:
    """Experiments are every row except the seed baseline, usage-limit pauses, and agent errors
    where the agent did nothing (no hypothesis recorded and zero output tokens), e.g. an API error
    on the first request. Those are retried under the same exp_id and reported separately (I4)."""
    st = row.get("status")
    if st in ("baseline", "usage_limit"):
        return False
    return not is_idle_agent_error(row)


def is_idle_agent_error(row: dict) -> bool:
    try:
        out = int(float(row.get("output_tokens") or 0))
    except (TypeError, ValueError):
        out = 0
    return row.get("status") == "agent_error" and row.get("hypothesis") == NO_HYPOTHESIS and out == 0


def _cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return f"{v:.6g}"
    return " ".join(str(v).replace("\t", " ").replace("\r", " ").replace("\n", " ").split()) \
        if isinstance(v, str) else str(v)


def append_result(path: Path, row: dict) -> None:
    path = Path(path)
    unknown = set(row) - set(RESULTS_COLUMNS)
    if unknown:
        raise ValueError(f"unknown result columns {sorted(unknown)}")
    if row.get("status") not in STATUSES:
        raise ValueError(f"bad status {row.get('status')!r}")
    new = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="") as f:
        if new:
            f.write("\t".join(RESULTS_COLUMNS) + "\n")
        f.write("\t".join(_cell(row.get(c)) for c in RESULTS_COLUMNS) + "\n")


def read_results(path: Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE))
