"""Driver-only benchmark CLI (official keep/revert numbers). Not for the agent."""
import argparse
import json
import math
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

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


def _time_all(kind: str, trials: int):
    med, sd = {}, {}
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
        del q, k, v, fn
        torch.cuda.empty_cache()
    return med, sd


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

    if a.ours or a.all:
        from harness.correctness import check_kernel

        corr = [asdict(r) for r in check_kernel(load_kernel(), BENCH_SHAPES + HELDOUT_SHAPES)]
        if not all(c["ok"] for c in corr):
            _write(a.out, {**out, "correct": False, "correctness": corr})
            return 2
        out.update(correct=True, correctness=corr)
        out["ours_ms"], out["ours_stdev_ms"] = _time_all("ours", a.trials)
    if a.baselines or a.all:
        out["fa2_ms"], out["fa2_stdev_ms"] = _time_all("fa2", a.trials)
        out["sdpa_ms"], out["sdpa_stdev_ms"] = _time_all("sdpa", a.trials)
        out["sdpa_backend"] = {s.name: sdpa_backend_name(s) for s in BENCH_SHAPES}
        out["fa2_source"] = _device_fa2_source()
    if a.baselines:
        out["timestamp"] = out["utc"]
    _write(a.out, out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
