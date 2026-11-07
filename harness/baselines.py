"""FA2 and SDPA reference attention implementations."""
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from harness.inputs import make_inputs, to_sdpa_layout
from harness.shapes import Shape


def fa2_attention(q, k, v, causal: bool):
    """FA2 layout in/out. Decode (sq==1, non-causal) uses flash_attn_with_kvcache."""
    from flash_attn import flash_attn_func, flash_attn_with_kvcache

    if q.shape[1] == 1 and not causal:
        return flash_attn_with_kvcache(q, k, v, causal=False)
    return flash_attn_func(q, k, v, causal=causal)


def sdpa_attention(q, k, v, causal: bool):
    """SDPA layout [B, H, S, D] in/out, GQA via enable_gqa."""
    return F.scaled_dot_product_attention(q, k, v, is_causal=causal, enable_gqa=True)


_PRIORITY = [
    ("cudnn", SDPBackend.CUDNN_ATTENTION),
    ("flash", SDPBackend.FLASH_ATTENTION),
    ("efficient", SDPBackend.EFFICIENT_ATTENTION),
    ("math", SDPBackend.MATH),
]


def sdpa_backend_name(shape: Shape) -> str:
    """Probe one backend at a time in priority order; first that runs is what default dispatch ~picks."""
    q, k, v = (to_sdpa_layout(t) for t in make_inputs(shape))
    for name, backend in _PRIORITY:
        try:
            with sdpa_kernel(backend):
                sdpa_attention(q, k, v, shape.causal)
            torch.cuda.synchronize()
            return f"default (≈{name})"
        except Exception:
            continue
    return "default (≈unknown)"
