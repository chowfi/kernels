import torch

from harness.shapes import Shape


def make_inputs(shape: Shape, seed: int = 0):
    """Return (q, k, v) in FA2 layout [B, S, H, D], bf16, on cuda."""
    g = torch.Generator(device="cuda").manual_seed(seed)

    def r(s, h):
        return torch.randn(1, s, h, shape.head_dim, device="cuda", dtype=torch.bfloat16, generator=g)

    return r(shape.seq_q, shape.n_heads), r(shape.seq_kv, shape.n_kv_heads), r(shape.seq_kv, shape.n_kv_heads)


def to_sdpa_layout(t: torch.Tensor) -> torch.Tensor:
    """[B, S, H, D] -> [B, H, S, D] contiguous."""
    return t.transpose(1, 2).contiguous()
