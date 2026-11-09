"""FA2 and SDPA reference attention implementations."""
import torch
import torch.nn.functional as F

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


def _classify(name: str) -> str | None:
    n = name.lower()
    if "cudnn" in n:
        return "cudnn"
    if "flash" in n:
        return "flash"
    if "fmha_cutlass" in n or "efficient_attention" in n or "attention_kernel" in n:
        return "efficient"
    return None


def sdpa_backend_name(shape: Shape) -> str:
    """Profile one default-dispatch SDPA call at the real shape; map launched kernel names to a backend."""
    from torch.profiler import ProfilerActivity, profile

    q, k, v = (to_sdpa_layout(t) for t in make_inputs(shape))
    sdpa_attention(q, k, v, shape.causal)  # warm up (cudnn plan build, autotune)
    torch.cuda.synchronize()
    # Right after another profiler session (bench.py's Triton-only check), the first sessions can
    # capture no CUDA activity at all; retry so the label is not a spurious "math (kernel: none)".
    kernels = []
    for _ in range(5):
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            sdpa_attention(q, k, v, shape.causal)
            torch.cuda.synchronize()
        kernels = [e for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
        if kernels:
            break
    for e in sorted(kernels, key=lambda e: -e.device_time_total):
        b = _classify(e.name)
        if b:
            return f"{b} (kernel: {e.name[:60]})"
    return "math (kernel: " + (kernels[0].name[:60] if kernels else "none") + ")"
