"""Large-value stress test: where does fp16 accumulation break?

The official harness checks correctness on random normal inputs (values about +-4). Real
activations can be larger, and fp16 overflows above 65504. This script feeds larger values to
three kernels and compares each to the fp32 reference:
  - kernel.attention          the agent's final kernel (fp16-accumulate prefill matmuls)
  - kernel.attention_fp32acc  the precision-matched variant (bf16 in, fp32 accumulate, like FA2)
  - FA2
Prefill only (the decode path never used fp16). Writes results/precision_stress.json.

Run: uv run --frozen python -m scripts.precision_stress
"""
import importlib
import json
import math
from pathlib import Path

import torch

from harness.baselines import fa2_attention
from harness.reference import reference_attention

SEQ, HQ, HKV, D = 4096, 32, 8, 128

# (case name, description, q/k scale, v scale, k outlier-channel scale)
CASES = [
    ("normal", "harness inputs: randn", 1, 1, 1),
    ("qk_x8", "q and k x8", 8, 1, 1),
    ("qk_x16", "q and k x16", 16, 1, 1),
    ("v_x100", "v x100", 1, 100, 1),
    ("v_x1000", "v x1000", 1, 1000, 1),
    ("k_outlier_x100", "one k channel x100 (LLM-style outlier)", 1, 1, 100),
    ("k_outlier_x1000", "one k channel x1000", 1, 1, 1000),
]


def make(qk_s, v_s, out_s, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q = torch.randn(1, SEQ, HQ, D, device="cuda", generator=g) * qk_s
    k = torch.randn(1, SEQ, HKV, D, device="cuda", generator=g) * qk_s
    v = torch.randn(1, SEQ, HKV, D, device="cuda", generator=g) * v_s
    k[..., 7] *= out_s
    return q.bfloat16(), k.bfloat16(), v.bfloat16()


def err(o, ref):
    o = o.float()
    bad = int((~torch.isfinite(o)).sum())
    if bad:
        return {"rel_err": math.inf, "nonfinite": bad}
    return {"rel_err": float((o - ref).abs().max() / ref.abs().max()), "nonfinite": 0}


def main() -> int:
    kernels = {
        "ours_fp16acc": importlib.import_module("kernel.attention").attention,
        "ours_fp32acc": importlib.import_module("kernel.attention_fp32acc").attention,
        "fa2": fa2_attention,
    }
    rows = []
    for name, desc, qk_s, v_s, out_s in CASES:
        q, k, v = make(qk_s, v_s, out_s)
        ref = reference_attention(q, k, v, causal=True).float()
        row = {"case": name, "desc": desc}
        for kn, fn in kernels.items():
            row[kn] = err(fn(q, k, v, True), ref)
        rows.append(row)
        print(f"{name:18s} " + "  ".join(
            f"{kn} {r['rel_err']:.2e}" + (f" ({r['nonfinite']} inf/nan)" if r["nonfinite"] else "")
            for kn, r in ((kn, row[kn]) for kn in kernels)))
    out = Path("results/precision_stress.json")
    out.write_text(json.dumps({"shape": f"prefill causal seq {SEQ}, {HQ}q/{HKV}kv heads, d{D}, bf16",
                               "metric": "max |out - fp32 ref| / max |fp32 ref|", "rows": rows},
                              indent=2) + "\n")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
