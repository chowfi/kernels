"""Benchmark and held-out shapes (GQA: 32 q heads, 8 kv heads, D=128, bf16, batch 1)."""
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class Shape:
    workload: Literal["prefill", "decode"]
    seq_q: int
    seq_kv: int
    n_heads: int = 32
    n_kv_heads: int = 8
    head_dim: int = 128
    causal: bool = False
    heldout: bool = False

    @property
    def name(self) -> str:
        base = f"{self.workload}_{self.seq_kv}"
        if self.n_heads // self.n_kv_heads != 4:
            base += f"_g{self.n_heads // self.n_kv_heads}"
        return f"heldout_{base}" if self.heldout else base


BENCH_SHAPES: list[Shape] = [
    Shape("prefill", n, n, causal=True) for n in (2048, 4096, 8192, 16384, 32768)
] + [Shape("decode", 1, n, causal=False) for n in (2048, 4096, 8192, 16384, 32768, 65536)]

HELDOUT_SHAPES: list[Shape] = [
    Shape("prefill", 3000, 3000, causal=True, heldout=True),
    Shape("decode", 1, 5001, causal=False, heldout=True),
    Shape("prefill", 4096, 4096, n_kv_heads=4, causal=True, heldout=True),
    Shape("decode", 1, 8192, n_kv_heads=32, causal=False, heldout=True),
]
