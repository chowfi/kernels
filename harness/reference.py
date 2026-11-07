"""Chunked reference attentions (FA2 layout in/out: q [B,Sq,H,D], k/v [B,Skv,Hkv,D])."""
import math

import torch

CHUNK = 1024


def _chunked(q, k, v, causal: bool, bf16_pv: bool = False):
    B, Sq, H, D = q.shape
    Skv, Hkv = k.shape[1], k.shape[2]
    G = H // Hkv
    scale = 1.0 / math.sqrt(D)
    out = torch.empty(B, Sq, H, D, device=q.device, dtype=torch.float32)
    off = Skv - Sq  # bottom-right alignment; equals top-left when Sq == Skv (the only causal case used)
    for b in range(B):
        # [Hkv, 1, Skv, D] so the G query heads of a group broadcast against one KV head.
        kb = k[b].transpose(0, 1).unsqueeze(1)
        vb = v[b].transpose(0, 1).unsqueeze(1)
        kb, vb = kb.float(), vb.float()
        for s0 in range(0, Sq, CHUNK):
            s1 = min(s0 + CHUNK, Sq)
            kv_end = min(Skv, s1 + off) if causal else Skv
            qc = q[b, s0:s1].transpose(0, 1).reshape(Hkv, G, s1 - s0, D)  # head h = kvh*G + g
            kc, vc = kb[:, :, :kv_end], vb[:, :, :kv_end]
            sc = torch.matmul(qc.float(), kc.transpose(-1, -2)) * scale
            if causal:
                qi = torch.arange(s0, s1, device=q.device)[:, None] + off
                kj = torch.arange(kv_end, device=q.device)[None, :]
                sc.masked_fill_(kj > qi, float("-inf"))
            p = torch.softmax(sc, dim=-1)
            del sc
            if bf16_pv:
                o = torch.matmul(p.to(torch.bfloat16), vc.to(torch.bfloat16)).float()
            else:
                o = torch.matmul(p, vc)
            out[b, s0:s1] = o.reshape(H, s1 - s0, D).transpose(0, 1)
    return out


def reference_attention(q, k, v, causal: bool):
    """fp32 ground truth, FA2 layout, GQA via broadcasting; query rows processed in chunks of <=1024."""
    return _chunked(q, k, v, causal)


def bf16_math_attention(q, k, v, causal: bool):
    """bf16 output with FlashAttention-style precision, chunked so it fits at 32K prefill.

    Mirrors FA's own test convention (a lower-precision PyTorch reference): Q.K^T from bf16 inputs
    with fp32 accumulate (scores kept fp32), fp32 softmax, P rounded to bf16, P.V in bf16 with fp32
    accumulate, output rounded to bf16. Its max error vs the fp32 reference sets the gate tolerance.
    """
    return _chunked(q, k, v, causal, bf16_pv=True).to(torch.bfloat16)
