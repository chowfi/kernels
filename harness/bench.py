"""Driver-only benchmark CLI (official keep/revert numbers). Not for the agent.

For `ours`, after timing each shape the bench refills q, k, v IN PLACE with new random data (a
fresh random seed per run), calls the kernel once more under torch.profiler and requires (a) the
output to match a fresh fp32 reference within the correctness tolerances and (b) every CUDA kernel
launched by that call to have been launched through Triton (torch fill/copy housekeeping kernels
from allocation are allowed). Either failure -> correct=false, exit 2, with an `error`.
"""
import argparse
import json
import math
import re
import secrets
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

from harness import integrity
from harness.baselines import fa2_attention, sdpa_attention, sdpa_backend_name
from harness.check import load_kernel
from harness.inputs import make_inputs, to_sdpa_layout
from harness.shapes import BENCH_SHAPES, HELDOUT_SHAPES
from harness.timing import time_fn


def _clean(o):
    if isinstance(o, float) and not math.isfinite(o):
        return None
    if isinstance(o, dict):
        return {k: _clean(v) for k, v in o.items()}
    if isinstance(o, list):
        return [_clean(v) for v in o]
    return o


def _write(path: str, data: dict) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(_clean(data), indent=2, allow_nan=False) + "\n")


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# Kernels torch itself may launch while a Triton kernel allocates its output/workspace
# (torch.zeros / zero_() / copy_()). Everything else must have been launched through Triton.
HOUSEKEEPING_RE = re.compile(r"^(Memset|Memcpy)\b|elementwise_kernel.*(FillFunctor|direct_copy_kernel)", re.S)


def non_triton_kernels(names: list[str], triton_names: set[str]) -> list[str]:
    """CUDA kernel names that were neither launched through Triton nor torch fill/copy housekeeping."""
    return [n for n in names if n not in triton_names and not HOUSEKEEPING_RE.search(n)]


def profile_call(fn):
    """Run fn() once under torch.profiler. Returns (result, CUDA kernel names in launch order,
    names of the kernels launched through Triton during the call)."""
    import triton.knobs as knobs
    from torch.profiler import ProfilerActivity, profile

    launched: set[str] = set()

    def hook(meta):
        d = meta if isinstance(meta, dict) else meta.get()  # triton LazyDict
        launched.add(d["name"])

    prev = knobs.runtime.launch_enter_hook
    knobs.runtime.launch_enter_hook = hook
    try:
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            res = fn()
            torch.cuda.synchronize()
    finally:
        knobs.runtime.launch_enter_hook = prev
    names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    return res, names, launched


def post_timing_check(attn, shape, q, k, v, seed: int) -> tuple[str | None, dict]:
    """I1: refill the SAME tensor objects with new random data, call the kernel once (profiled) and
    check it against a fresh fp32 reference. Catches outputs cached on tensor identity and work
    done by non-Triton kernels (torch/cuBLAS/FA/SDPA)."""
    from harness.correctness import check_output, tolerance
    from harness.reference import reference_attention

    q2, k2, v2 = make_inputs(shape, seed)
    for dst, src in ((q, q2), (k, k2), (v, v2)):
        dst.copy_(src)
    del q2, k2, v2
    ref = reference_attention(q, k, v, shape.causal)  # before the kernel runs
    tols = tolerance(shape, seed, q, k, v, ref)
    torch.cuda.synchronize()
    try:
        out, names, launched = profile_call(lambda: attn(q, k, v, shape.causal))
    except Exception as e:  # noqa: BLE001
        return f"post-timing freshness check failed on {shape.name}: kernel raised {e!r}", {}
    info = {"kernels": sorted(set(names)), "triton_kernels": sorted(launched)}
    bad = non_triton_kernels(names, launched)
    if bad:
        return (f"non-Triton CUDA kernel launched on {shape.name}: "
                f"{'; '.join(sorted(set(bad)))[:600]}"), info
    r = check_output(shape, out, ref, tols)
    info.update(max_abs_err=r.max_abs_err, max_row_rel=r.max_row_rel)
    del out, ref
    if not r.ok:
        return (f"post-timing freshness check failed on {shape.name}: max_abs_err={r.max_abs_err:.4g} "
                f"(tol {r.tol:.4g}), max_row_rel={r.max_row_rel:.4g} (tol {r.rel_tol:.4g})"
                + (f", error={r.error}" if r.error else "")), info
    return None, info


def _time_all(kind: str, trials: int, fresh_seed: int | None = None):
    """Returns (median_ms, stdev_ms, post-timing checks, error). Post-timing checks run for `ours`."""
    med, sd, checks = {}, {}, {}
    attn = load_kernel() if kind == "ours" else None
    for shape in BENCH_SHAPES:
        q, k, v = make_inputs(shape)
        c = shape.causal
        if kind == "ours":
            fn = lambda: attn(q, k, v, c)  # noqa: E731
        elif kind == "fa2":
            fn = lambda: fa2_attention(q, k, v, c)  # noqa: E731
        else:
            qs, ks, vs = (to_sdpa_layout(t) for t in (q, k, v))
            fn = lambda: sdpa_attention(qs, ks, vs, c)  # noqa: E731
        r = time_fn(fn, trials)
        med[shape.name], sd[shape.name] = r.median_ms, r.stdev_ms
        print(f"{kind} {shape.name} {r.median_ms:.4f} ms", file=sys.stderr)
        if kind == "ours":
            err, checks[shape.name] = post_timing_check(attn, shape, q, k, v, fresh_seed)
            if err:
                print(err, file=sys.stderr)
                return med, sd, checks, err
        del q, k, v, fn
        torch.cuda.empty_cache()
    return med, sd, checks, None


def _device_fa2_source() -> str:
    return json.loads((Path(__file__).resolve().parent.parent / "results" / "device.json").read_text())["fa2_source"]


def main() -> int:
    ap = argparse.ArgumentParser()
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--ours", action="store_true")
    g.add_argument("--baselines", action="store_true")
    g.add_argument("--all", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--trials", type=int, default=5)
    a = ap.parse_args()
    out = {"utc": _utc(), "trials": a.trials}
    snap = integrity.snapshot()  # before the kernel module is imported

    def tampered(stage: str) -> bool:
        bad = integrity.verify(snap)
        if bad:
            _write(a.out, {**out, "correct": False,
                           "error": f"timing integrity violated ({stage}): {', '.join(bad)}"})
        return bool(bad)

    if a.ours or a.all:
        from harness.correctness import check_kernel

        kernel = load_kernel()
        if tampered("after kernel import"):
            return 2
        corr = [asdict(r) for r in check_kernel(kernel, BENCH_SHAPES + HELDOUT_SHAPES)]
        if tampered("after correctness"):
            return 2
        if not all(c["ok"] for c in corr):
            _write(a.out, {**out, "correct": False, "correctness": corr})
            return 2
        fresh_seed = 1 + secrets.randbelow(2**31 - 2)  # never 0 (the correctness seed)
        med, sd, checks, err = _time_all("ours", a.trials, fresh_seed)
        if tampered("after timing"):
            return 2
        if err:
            _write(a.out, {**out, "correct": False, "correctness": corr, "error": err,
                           "fresh_seed": fresh_seed, "post_timing_checks": checks})
            return 2
        out.update(correct=True, correctness=corr, ours_ms=med, ours_stdev_ms=sd,
                   fresh_seed=fresh_seed, post_timing_checks=checks,
                   triton_kernels=sorted({n for c in checks.values() for n in c["triton_kernels"]}))
    if a.baselines or a.all:
        out["fa2_ms"], out["fa2_stdev_ms"], _, _ = _time_all("fa2", a.trials)
        out["sdpa_ms"], out["sdpa_stdev_ms"], _, _ = _time_all("sdpa", a.trials)
        out["sdpa_backend"] = {s.name: sdpa_backend_name(s) for s in BENCH_SHAPES}
        out["fa2_source"] = _device_fa2_source()
    if a.baselines:
        out["timestamp"] = out["utc"]
    _write(a.out, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
