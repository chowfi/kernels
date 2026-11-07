"""Untuned experiment-0 seed: plain Triton GQA flash attention (forward only)."""
import math

import torch
import triton
import triton.language as tl


@triton.jit
def _attn_fwd(
    Q, K, V, O,
    sq_b, sq_s, sq_h,
    sk_b, sk_s, sk_h,
    sv_b, sv_s, sv_h,
    so_b, so_s, so_h,
    H, SEQ_Q, SEQ_KV, GROUP, scale,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
):
    start_m = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    kvh = h // GROUP

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)

    q_ptrs = Q + b * sq_b + h * sq_h + offs_m[:, None] * sq_s + offs_d[None, :]
    q = tl.load(q_ptrs, mask=offs_m[:, None] < SEQ_Q, other=0.0)

    k_base = K + b * sk_b + kvh * sk_h
    v_base = V + b * sv_b + kvh * sv_h

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    for start_n in range(0, SEQ_KV, BLOCK_N):
        cur_n = start_n + offs_n
        k = tl.load(k_base + cur_n[None, :] * sk_s + offs_d[:, None],
                    mask=cur_n[None, :] < SEQ_KV, other=0.0)
        s = tl.dot(q, k) * scale
        mask = cur_n[None, :] < SEQ_KV
        if CAUSAL:
            mask = mask & (offs_m[:, None] >= cur_n[None, :])
        s = tl.where(mask, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v = tl.load(v_base + cur_n[:, None] * sv_s + offs_d[None, :],
                    mask=cur_n[:, None] < SEQ_KV, other=0.0)
        acc += tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    o_ptrs = O + b * so_b + h * so_h + offs_m[:, None] * so_s + offs_d[None, :]
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=offs_m[:, None] < SEQ_Q)


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, causal: bool) -> torch.Tensor:
    """q [B,Sq,H,D], k/v [B,Skv,Hkv,D] bf16 -> [B,Sq,H,D] bf16."""
    B, Sq, H, D = q.shape
    _, Skv, Hkv, _ = k.shape
    if D not in (64, 128):
        raise ValueError(f"head_dim must be 64 or 128, got {D}")
    if Hkv == 0 or H % Hkv != 0:
        raise ValueError(f"n_heads {H} not divisible by n_kv_heads {Hkv}")
    if causal and Sq != Skv:
        raise ValueError(f"causal requires seq_q == seq_kv, got {Sq} vs {Skv}")

    o = torch.empty_like(q)
    block_m = 16 if Sq == 1 else 64
    grid = (triton.cdiv(Sq, block_m), H, B)
    _attn_fwd[grid](
        q, k, v, o,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        H, Sq, Skv, H // Hkv, 1.0 / math.sqrt(D),
        HEAD_DIM=D, BLOCK_M=block_m, BLOCK_N=64, CAUSAL=causal,
        num_warps=4, num_stages=2,
    )
    return o
