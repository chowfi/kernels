"""CPU stand-in for `python -m harness.bench` used by tests/test_loop.py (via HARNESS_BENCH_CMD).

Usage: fake_bench.py (--ours|--baselines) --out PATH
--baselines writes fa2_ms = 1.0 for every shape, so score == the queued value.
--ours pops the first entry of the JSON list in $FAKE_BENCH_QUEUE:
  number -> correct run with ours_ms = 1/number on every shape (score == number)
  "fail" -> correctness failure (writes correct=false, exit 2)
  "crash" -> exit 1 with no output
An empty queue means score 1.0. Every call is appended to $FAKE_BENCH_QUEUE.calls.
"""
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harness.shapes import BENCH_SHAPES  # noqa: E402


def main() -> int:
    mode, out = sys.argv[1], Path(sys.argv[sys.argv.index("--out") + 1])
    qp = Path(os.environ["FAKE_BENCH_QUEUE"])
    with open(str(qp) + ".calls", "a") as f:
        f.write(mode + "\n")
    utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    out.parent.mkdir(parents=True, exist_ok=True)
    if mode == "--baselines":
        ms = {s.name: 1.0 for s in BENCH_SHAPES}
        out.write_text(json.dumps({"utc": utc, "timestamp": utc, "trials": 1, "fa2_ms": ms,
                                   "sdpa_ms": ms, "sdpa_backend": {}, "fa2_source": "fake"}))
        return 0
    q = json.loads(qp.read_text()) if qp.exists() else []
    item = q.pop(0) if q else 1.0
    qp.write_text(json.dumps(q))
    if item == "crash":
        return 1
    if item == "fail":
        out.write_text(json.dumps({"utc": utc, "trials": 1, "correct": False, "correctness": []}))
        return 2
    ms = {s.name: 1.0 / float(item) for s in BENCH_SHAPES}
    out.write_text(json.dumps({"utc": utc, "trials": 1, "correct": True, "correctness": [],
                               "ours_ms": ms, "ours_stdev_ms": {k: 0.0 for k in ms}}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
