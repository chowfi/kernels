"""Correctness gate: kernel output vs fp32 reference, FlashAttention-style tolerance."""
import argparse
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from harness.inputs import make_inputs
from harness.reference import bf16_math_attention, reference_attention
from harness.shapes import BENCH_SHAPES, HELDOUT_SHAPES, Shape

TOL_PATH = Path(__file__).resolve().parent.parent / "results" / "tolerances.json"


@dataclass
class CorrectnessResult:
    shape_name: str
    ok: bool
    max_abs_err: float
    tol: float  # abs tol
    error: str | None = None
    max_row_rel: float = float("nan")
    rel_tol: float = float("nan")


ABS_TOL_CAP = 0.1


def _row_rel(out_f32, ref) -> float:
    """max over (query row, head) of ||out_row - ref_row||_2 / ||ref_row||_2 (over head_dim)."""
    num = torch.linalg.vector_norm(out_f32 - ref, dim=-1)
    den = torch.linalg.vector_norm(ref, dim=-1).clamp_min(1e-12)
    return (num / den).max().item()


def _compute_tols(shape: Shape, q, k, v, ref) -> dict:
    """abs_tol = 2*max|e - ref| + 1e-4 ; rel_tol = 2*max_row_rel(e, ref) + 1e-3.

    e is chunked FA-style bf16 attention (P rounded to bf16 before P.V; see harness/reference.py).
    The small abs floor keeps decode (outputs ~0.1) sensitive to ~1% bugs; the per-row relative
    check catches errors confined to late rows, which a global max-abs would miss.
    """
    e = bf16_math_attention(q, k, v, shape.causal).float()
    abs_tol = 2 * (e - ref).abs().max().item() + 1e-4
    rel_tol = 2 * _row_rel(e, ref) + 1e-3
    return {"abs_tol": abs_tol, "rel_tol": rel_tol}


def _validate(key: str, entry) -> dict:
    try:
        a, r = float(entry["abs_tol"]), float(entry["rel_tol"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"malformed tolerance entry {key}: {entry!r}") from e
    if not (math.isfinite(a) and math.isfinite(r) and 0 < a < ABS_TOL_CAP and 0 < r < 1):
        raise ValueError(f"tolerance entry {key} out of range: abs_tol={a}, rel_tol={r}")
    return {"abs_tol": a, "rel_tol": r}


def tolerance(shape: Shape, seed: int, q, k, v, ref) -> dict:
    """Committed results/tolerances.json is a trusted artifact: read-only here.

    A missing entry (or file) is computed in memory and NOT written; use --build-tolerances.
    """
    key = f"{shape.name}|seed{seed}"
    try:
        data = json.loads(Path(TOL_PATH).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        data = {}
    if key in data:
        return _validate(key, data[key])
    return _validate(key, _compute_tols(shape, q, k, v, ref))


def build_tolerances(shapes: list[Shape] | None = None, seed: int = 0) -> dict:
    """Rebuild all entries (BENCH + HELDOUT by default) and atomically write TOL_PATH."""
    shapes = shapes if shapes is not None else BENCH_SHAPES + HELDOUT_SHAPES
    data = {}
    for shape in shapes:
        q, k, v = make_inputs(shape, seed)
        ref = reference_attention(q, k, v, shape.causal)
        key = f"{shape.name}|seed{seed}"
        data[key] = _validate(key, _compute_tols(shape, q, k, v, ref))
        del q, k, v, ref
        torch.cuda.empty_cache()
    path = Path(TOL_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp, path)
    return data


def check_output(shape: Shape, out, ref, tols: dict) -> CorrectnessResult:
    """Compare a kernel output against `ref` (fp32 reference computed from the inputs BEFORE the
    kernel ran, so a kernel that mutates its inputs cannot move the target) with tolerances `tols`."""
    of = None
    try:
        if not isinstance(out, torch.Tensor):
            raise TypeError(f"output is {type(out).__name__}, not a Tensor")
        if out.shape != ref.shape:
            raise ValueError(f"wrong shape {tuple(out.shape)}, expected {tuple(ref.shape)}")
        if out.dtype != torch.bfloat16:
            raise ValueError(f"wrong dtype {out.dtype}, expected torch.bfloat16")
        of = out.float()
        if torch.isnan(of).any() or torch.isinf(of).any():
            raise ValueError("output contains NaN/Inf")
        err = (of - ref).abs().max().item()
        rel = _row_rel(of, ref)
        ok = err <= tols["abs_tol"] and rel <= tols["rel_tol"]
        return CorrectnessResult(shape.name, ok, err, tols["abs_tol"], None, rel, tols["rel_tol"])
    except Exception as e:  # noqa: BLE001 - any kernel failure is a gate failure
        return CorrectnessResult(shape.name, False, float("nan"), tols["abs_tol"], repr(e))
    finally:
        del of


def check_kernel(fn: Callable, shapes: list[Shape], seed: int = 0) -> list[CorrectnessResult]:
    results = []
    for shape in shapes:
        q = k = v = ref = out = None
        tol = float("nan")
        try:
            q, k, v = make_inputs(shape, seed)
            ref = reference_attention(q, k, v, shape.causal)
            tols = tolerance(shape, seed, q, k, v, ref)
            tol = tols["abs_tol"]
            out = fn(q, k, v, shape.causal)
            results.append(check_output(shape, out, ref, tols))
        except Exception as e:  # noqa: BLE001 - any kernel failure is a gate failure
            results.append(CorrectnessResult(shape.name, False, float("nan"), tol, repr(e)))
        finally:
            del q, k, v, ref, out
            torch.cuda.empty_cache()
    return results

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--build-tolerances", action="store_true")
    if ap.parse_args().build_tolerances:
        for k_, v_ in build_tolerances().items():
            print(k_, v_)
