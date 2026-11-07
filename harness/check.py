"""Agent-facing correctness CLI. Prints NO timing. Appends a block to runs.log."""
import importlib
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _runs_log() -> Path:
    return Path(os.environ.get("HARNESS_RUNS_LOG") or ROOT / "runs.log")


def append_runs_log(command: str, lines: list[str]) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(_runs_log(), "a") as f:
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
    mod = importlib.import_module(os.environ.get("HARNESS_KERNEL_MODULE", "kernel.attention"))
    return mod.attention


def main() -> int:
    from harness import integrity
    from harness.correctness import check_kernel
    from harness.shapes import BENCH_SHAPES, HELDOUT_SHAPES

    snap = integrity.snapshot()

    def integrity_fail(stage: str) -> bool:
        bad = integrity.verify(snap)
        if bad:
            line = f"INTEGRITY FAIL ({stage}): kernel rebinds {', '.join(bad)}"
            print(line)
            append_runs_log("harness.check", [line])
        return bool(bad)

    try:
        kernel = load_kernel()
    except Exception as e:  # noqa: BLE001 - import failure is a gate failure
        line = f"IMPORT FAIL error={e!r}"
        print(line)
        append_runs_log("harness.check", [line])
        return 1
    if integrity_fail("after kernel import"):
        return 1
    results = check_kernel(kernel, BENCH_SHAPES + HELDOUT_SHAPES)
    if integrity_fail("after correctness"):
        return 1
    lines = [format_result(r) for r in results]
    for l in lines:
        print(l)
    append_runs_log("harness.check", lines)
    return 0 if all(r.ok for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
