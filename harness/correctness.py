"""Correctness gate: kernel output vs fp32 reference, FlashAttention-style tolerance."""
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch

from harness.inputs import make_inputs
from harness.reference import bf16_math_attention, reference_attention
from harness.shapes import Shape

TOL_PATH = Path(__file__).resolve().parent.parent / "results" / "tolerances.json"


@dataclass
class CorrectnessResult:
    shape_name: str
    ok: bool
    max_abs_err: float
    tol: float
    error: str | None = None


def _load() -> dict:
    try:
        return json.loads(Path(TOL_PATH).read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def _store(key: str, value: float) -> None:
    path = Path(TOL_PATH)
    data = _load()
    data[key] = value
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    os.replace(tmp, path)


def tolerance(shape: Shape, seed: int, q, k, v, ref) -> float:
    """tol = 2 * max|bf16_math - ref| + 1e-3, cached per (shape, seed).

    bf16_math is a chunked emulation of SDPA MATH on bf16 (see harness/reference.py).
    """
    key = f"{shape.name}|seed{seed}"
    data = _load()
    if key in data:
        return data[key]
    m = bf16_math_attention(q, k, v, shape.causal)
    err = (m.float() - ref).abs().max().item()
    del m
    tol = 2 * err + 1e-3
    _store(key, tol)
    return tol


def check_kernel(fn: Callable, shapes: list[Shape], seed: int = 0) -> list[CorrectnessResult]:
    results = []
    for shape in shapes:
        q = k = v = ref = out = None
        tol = float("nan")
        try:
            q, k, v = make_inputs(shape, seed)
            ref = reference_attention(q, k, v, shape.causal)
            tol = tolerance(shape, seed, q, k, v, ref)
            out = fn(q, k, v, shape.causal)
            if not isinstance(out, torch.Tensor):
                raise TypeError(f"output is {type(out).__name__}, not a Tensor")
            if out.shape != q.shape:
                raise ValueError(f"wrong shape {tuple(out.shape)}, expected {tuple(q.shape)}")
            if out.dtype != torch.bfloat16:
                raise ValueError(f"wrong dtype {out.dtype}, expected torch.bfloat16")
            of = out.float()
            if torch.isnan(of).any() or torch.isinf(of).any():
                raise ValueError("output contains NaN/Inf")
            err = (of - ref).abs().max().item()
            results.append(CorrectnessResult(shape.name, err <= tol, err, tol, None))
        except Exception as e:  # noqa: BLE001 - any kernel failure is a gate failure
            results.append(CorrectnessResult(shape.name, False, float("nan"), tol, repr(e)))
        finally:
            del q, k, v, ref, out
            torch.cuda.empty_cache()
    return results
