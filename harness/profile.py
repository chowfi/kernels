"""Agent-facing profiling diagnostic for the LAST KEPT kernel (git HEAD).

Runs ncu on `git show HEAD:kernel/attention.py` (never the working tree) and prints
structural/utilisation metrics only. Prints NO durations (any time-unit metric is dropped).
"""
import argparse
import csv
import importlib.util
import io
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NCU = os.environ.get("NCU_PATH") or shutil.which("ncu") or "/usr/local/cuda-12.8/bin/ncu"
KERNEL_REGEX = "regex:_attn_fwd"

# (label, ncu metric). Order = print order. None of these are time metrics.
METRICS = [
    ("achieved occupancy (active warps, % of peak)", "sm__warps_active.avg.pct_of_peak_sustained_active"),
    ("theoretical occupancy (%)", "sm__maximum_warps_per_active_cycle_pct"),
    ("occupancy limit: registers (blocks/SM)", "launch__occupancy_limit_registers"),
    ("occupancy limit: shared mem (blocks/SM)", "launch__occupancy_limit_shared_mem"),
    ("SM throughput (% of peak)", "sm__throughput.avg.pct_of_peak_sustained_elapsed"),
    ("compute+memory throughput (% of peak)", "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed"),
    ("DRAM throughput (% of peak)", "gpu__dram_throughput.avg.pct_of_peak_sustained_elapsed"),
    ("L2 hit rate (%)", "lts__t_sector_hit_rate.pct"),
    ("registers per thread", "launch__registers_per_thread"),
    ("shared memory per block, dynamic (bytes)", "launch__shared_mem_per_block_dynamic"),
    ("shared memory per block, static (bytes)", "launch__shared_mem_per_block_static"),
    ("grid size (blocks)", "launch__grid_size"),
    ("block size (threads)", "launch__block_size"),
    ("waves per SM", "launch__waves_per_multiprocessor"),
]
_TIME_UNIT = re.compile(r"^(n|u|m|µ)?s(econd)?$|^(n|u|m)?second$", re.I)


def _all_shapes():
    from harness.shapes import BENCH_SHAPES, HELDOUT_SHAPES
    return {s.name: s for s in BENCH_SHAPES + HELDOUT_SHAPES}


def _head_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()


def parse_ncu_csv(text: str) -> dict[str, tuple[str, str]]:
    """Return {metric: (value, unit)} for the first profiled kernel in `--csv --page raw` output."""
    i = text.find('"ID"')
    if i < 0:
        return {}
    rows = list(csv.reader(io.StringIO(text[i:])))
    if len(rows) < 3:
        return {}
    return {h: (v, u) for h, u, v in zip(rows[0], rows[1], rows[2])}


def format_metrics(m: dict[str, tuple[str, str]]) -> list[str]:
    out = []
    for label, key in METRICS:
        if key not in m:
            continue
        val, unit = m[key]
        if _TIME_UNIT.match(unit.strip()) or "time" in key or "duration" in key:
            continue  # never leak latency
        out.append(f"  {label}: {val}")
    return out


def _target(shape_name: str, module_path: str) -> int:
    import torch
    from harness.inputs import make_inputs

    sys.path.insert(0, str(ROOT))
    spec = importlib.util.spec_from_file_location("head_attention", module_path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["head_attention"] = mod
    spec.loader.exec_module(mod)
    s = _all_shapes()[shape_name]
    q, k, v = make_inputs(s)
    for _ in range(3):  # launch 0 compiles/warms; ncu skips it
        mod.attention(q, k, v, s.causal)
    torch.cuda.synchronize()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shape", required=True)
    ap.add_argument("--_target", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--module-path", help=argparse.SUPPRESS)
    a = ap.parse_args()

    shapes = _all_shapes()
    if a.shape not in shapes:
        print(f"unknown shape {a.shape!r}; valid names: {', '.join(shapes)}", file=sys.stderr)
        print(f"valid shapes: {', '.join(shapes)}")
        return 2
    if a._target:
        return _target(a.shape, a.module_path)

    from harness.check import append_runs_log

    sha = _head_sha()
    header = f"Profile of HEAD kernel ({sha}) on {a.shape}"
    with tempfile.TemporaryDirectory() as tmp:
        mod_path = Path(tmp) / "head_attention.py"
        mod_path.write_text(subprocess.check_output(["git", "show", "HEAD:kernel/attention.py"], cwd=ROOT, text=True))
        cmd = [NCU, "--set", "full", "--kernel-name", KERNEL_REGEX, "--launch-skip", "1", "--launch-count", "1",
               "--csv", "--page", "raw", sys.executable, "-m", "harness.profile",
               "--_target", "--shape", a.shape, "--module-path", str(mod_path)]
        try:
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600)
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            lines = [header, f"profiling unavailable ({type(e).__name__})"]
            r = None
    if r is not None:
        blob = r.stdout + r.stderr
        metrics = parse_ncu_csv(r.stdout)
        if "ERR_NVGPUCTRPERM" in blob:
            lines = [header, "profiling unavailable (ERR_NVGPUCTRPERM: GPU counters need admin enablement)"]
        elif not metrics:
            lines = [header, "profiling failed: no metrics captured"] + [l for l in r.stderr.splitlines()[-5:]]
            print("\n".join(lines))
            append_runs_log(f"harness.profile --shape {a.shape}", lines)
            return 1
        else:
            lines = [header] + format_metrics(metrics)
    print("\n".join(lines))
    append_runs_log(f"harness.profile --shape {a.shape}", lines)
    return 0


if __name__ == "__main__":
    sys.exit(main())
