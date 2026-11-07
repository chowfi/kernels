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
NVTX_RANGE = "attn_profile"

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
_TIME_RE = re.compile(r"\d\s*(ms|us|ns|usecond|msecond|nsecond|s)\b", re.I)
_TIME_UNIT = re.compile(r"^(n|u|m|µ)?s(econd)?$|^(n|u|m)?second$", re.I)


def _all_shapes():
    from harness.shapes import BENCH_SHAPES, HELDOUT_SHAPES
    return {s.name: s for s in BENCH_SHAPES + HELDOUT_SHAPES}


def _head_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True).strip()


def parse_ncu_csv(text: str) -> list[tuple[str, dict[str, tuple[str, str]]]]:
    """Return [(kernel_name, {metric: (value, unit)})] for each profiled kernel in `--csv --page raw` output."""
    i = text.find('"ID"')
    if i < 0:
        return []
    rows = list(csv.reader(io.StringIO(text[i:])))
    if len(rows) < 3:
        return []
    h, units = rows[0], rows[1]
    out = []
    for r in rows[2:]:
        m = {k: (v, u) for k, u, v in zip(h, units, r)}
        out.append((m.get("Kernel Name", ("?", ""))[0], m))
    return out


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
    for _ in range(2):  # compile/warm outside the profiled range
        mod.attention(q, k, v, s.causal)
    torch.cuda.synchronize()
    torch.cuda.nvtx.range_push(NVTX_RANGE)
    mod.attention(q, k, v, s.causal)
    torch.cuda.nvtx.range_pop()
    torch.cuda.synchronize()
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(allow_abbrev=False)
    ap.add_argument("--shape", required=True)
    ap.add_argument("--_target", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--_module", help=argparse.SUPPRESS)
    a = ap.parse_args()

    shapes = _all_shapes()
    if a.shape not in shapes:
        print(f"unknown shape {a.shape!r}; valid names: {', '.join(shapes)}")
        return 2
    if a._target:
        # Internal: only the parent process (which sets this env var) may run the target helper,
        # and it must never load a module from the working-tree kernel/ dir.
        if os.environ.get("HARNESS_PROFILE_TARGET") != "1" or not a._module:
            print("--_target is internal", file=sys.stderr)
            return 2
        mp = Path(a._module).resolve()
        if (ROOT / "kernel") in mp.parents:
            print("refusing to profile working-tree kernel", file=sys.stderr)
            return 2
        return _target(a.shape, str(mp))

    from harness.check import append_runs_log

    try:
        sha = _head_sha()
        head_src = None
        test_mod = os.environ.get("HARNESS_PROFILE_TEST_MODULE")  # test hook (env only)
        if not test_mod:
            head_src = subprocess.check_output(["git", "show", "HEAD:kernel/attention.py"], cwd=ROOT,
                                               text=True, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("cannot read HEAD:kernel/attention.py (no committed kernel?)")
        return 1
    header = f"Profile of HEAD kernel ({sha}) on {a.shape}"
    with tempfile.TemporaryDirectory() as tmp:
        mod_path = Path(tmp) / "head_attention.py"
        if test_mod:
            shutil.copy(test_mod, mod_path)
        else:
            mod_path.write_text(head_src)
        cmd = [NCU, "--set", "full", "--nvtx", "--nvtx-include", f"{NVTX_RANGE}/",
               "--csv", "--page", "raw", sys.executable, "-m", "harness.profile",
               "--_target", "--shape", a.shape, "--_module", str(mod_path)]
        try:
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=600,
                               env={**os.environ, "HARNESS_PROFILE_TARGET": "1"})
        except (FileNotFoundError, subprocess.TimeoutExpired) as e:
            lines = [header, f"profiling unavailable ({type(e).__name__})"]
            r = None
    if r is not None:
        blob = r.stdout + r.stderr
        metrics = parse_ncu_csv(r.stdout)
        if "ERR_NVGPUCTRPERM" in blob:
            lines = [header, "profiling unavailable (ERR_NVGPUCTRPERM: GPU counters need admin enablement)"]
        elif not metrics:
            lines = [header, "profiling failed: no metrics captured"] + [l for l in r.stderr.splitlines()[-5:] if not _TIME_RE.search(l)]
            print("\n".join(lines))
            append_runs_log(f"harness.profile --shape {a.shape}", lines)
            return 1
        else:
            lines = [header]
            for n, (name, m) in enumerate(metrics):
                lines += [f"kernel {n}: {name}"] + format_metrics(m)
    print("\n".join(lines))
    append_runs_log(f"harness.profile --shape {a.shape}", lines)
    return 0


if __name__ == "__main__":
    sys.exit(main())
