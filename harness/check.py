"""Agent-facing correctness CLI. Prints NO timing. Appends a block to runs.log."""
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RUNS_LOG = ROOT / "runs.log"


def append_runs_log(command: str, lines: list[str]) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(RUNS_LOG, "a") as f:
        f.write(f"[{ts}] {command}\n" + "".join(l + "\n" for l in lines) + "\n")


def format_result(r) -> str:
    s = f"{r.shape_name} {'OK' if r.ok else 'FAIL'} max_abs_err={r.max_abs_err:.4g} tol={r.tol:.4g}"
    s += f" max_row_rel={r.max_row_rel:.4g} rel_tol={r.rel_tol:.4g}"
    if r.error:
        s += f" error={r.error}"
    return s


def load_kernel():
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from kernel.attention import attention

    return attention


def main() -> int:
    from harness.correctness import check_kernel
    from harness.shapes import BENCH_SHAPES, HELDOUT_SHAPES

    results = check_kernel(load_kernel(), BENCH_SHAPES + HELDOUT_SHAPES)
    lines = [format_result(r) for r in results]
    for l in lines:
        print(l)
    append_runs_log("harness.check", lines)
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
