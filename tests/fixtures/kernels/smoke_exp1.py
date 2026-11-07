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


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_N": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 32}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_N": 32}, num_warps=2, num_stages=4),
    ],
    key=["SEQ_KV", "CHUNK", "GROUP", "HEAD_DIM"],
)
@triton.jit
def _decode_split(
    Q, K, V, OP, MP, LP,
    sq_b, sq_h,
    sk_b, sk_s, sk_h,
    sv_b, sv_s, sv_h,
    H, SEQ_KV, GROUP, CHUNK, NUM_SPLITS, scale,
    HEAD_DIM: tl.constexpr, BLOCK_G: tl.constexpr, BLOCK_N: tl.constexpr,
):
    """One program = one KV split of one KV head; its M rows are the GROUP query heads of that KV head."""
    split = tl.program_id(0)
    kvh = tl.program_id(1)
    b = tl.program_id(2)

    offs_g = tl.arange(0, BLOCK_G)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    g_mask = offs_g < GROUP
    heads = kvh * GROUP + offs_g

    q = tl.load(Q + b * sq_b + heads[:, None] * sq_h + offs_d[None, :],
                mask=g_mask[:, None], other=0.0)

    k_base = K + b * sk_b + kvh * sk_h
    v_base = V + b * sv_b + kvh * sv_h

    start = split * CHUNK
    end = tl.minimum(start + CHUNK, SEQ_KV)

    m_i = tl.full([BLOCK_G], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_G], dtype=tl.float32)
    acc = tl.zeros([BLOCK_G, HEAD_DIM], dtype=tl.float32)

    for start_n in range(start, end, BLOCK_N):
        cur_n = start_n + offs_n
        valid = cur_n < end
        k = tl.load(k_base + cur_n[None, :] * sk_s + offs_d[:, None],
                    mask=valid[None, :], other=0.0)
        s = tl.dot(q, k) * scale
        s = tl.where(valid[None, :], s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.exp(s - m_new[:, None])
        alpha = tl.exp(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v = tl.load(v_base + cur_n[:, None] * sv_s + offs_d[None, :],
                    mask=valid[:, None], other=0.0)
        acc += tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new

    row = (b * H + heads) * NUM_SPLITS + split
    tl.store(OP + row[:, None] * HEAD_DIM + offs_d[None, :], acc, mask=g_mask[:, None])
    tl.store(MP + row, m_i, mask=g_mask)
    tl.store(LP + row, l_i, mask=g_mask)


@triton.jit
def _decode_combine(
    OP, MP, LP, O,
    so_b, so_h,
    H, NUM_SPLITS,
    HEAD_DIM: tl.constexpr, SPLITS_P2: tl.constexpr,
):
    h = tl.program_id(0)
    b = tl.program_id(1)
    offs_s = tl.arange(0, SPLITS_P2)
    offs_d = tl.arange(0, HEAD_DIM)
    s_mask = offs_s < NUM_SPLITS
    row = (b * H + h) * NUM_SPLITS + offs_s

    m = tl.load(MP + row, mask=s_mask, other=float("-inf"))
    l = tl.load(LP + row, mask=s_mask, other=0.0)
    m_max = tl.max(m, 0)
    w = tl.exp(m - m_max)
    l_tot = tl.sum(w * l, 0)
    acc = tl.load(OP + row[:, None] * HEAD_DIM + offs_d[None, :], mask=s_mask[:, None], other=0.0)
    o = tl.sum(acc * w[:, None], 0) / l_tot
    tl.store(O + b * so_b + h * so_h + offs_d, o.to(O.dtype.element_ty))


_SM_COUNT = {}


def _num_sms(device: torch.device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _SM_COUNT:
        _SM_COUNT[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _SM_COUNT[idx]


def _decode(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    B, _, H, D = q.shape
    _, Skv, Hkv, _ = k.shape
    group = H // Hkv

    # Split KV so the grid holds at most ~2 programs per SM; chunks are multiples of 64 (max BLOCK_N).
    target = min(64, max(1, (2 * _num_sms(q.device)) // (Hkv * B)))
    chunk = triton.cdiv(triton.cdiv(Skv, target), 64) * 64
    num_splits = triton.cdiv(Skv, chunk)

    op = torch.empty((B, H, num_splits, D), device=q.device, dtype=torch.float32)
    mp = torch.empty((B, H, num_splits), device=q.device, dtype=torch.float32)
    lp = torch.empty((B, H, num_splits), device=q.device, dtype=torch.float32)
    o = torch.empty_like(q)

    _decode_split[(num_splits, Hkv, B)](
        q, k, v, op, mp, lp,
        q.stride(0), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        H, Skv, group, chunk, num_splits, 1.0 / math.sqrt(D),
        HEAD_DIM=D, BLOCK_G=max(16, triton.next_power_of_2(group)),
    )
    splits_p2 = triton.next_power_of_2(num_splits)
    _decode_combine[(H, B)](
        op, mp, lp, o,
        o.stride(0), o.stride(2),
        H, num_splits,
        HEAD_DIM=D, SPLITS_P2=splits_p2,
        num_warps=8 if splits_p2 * D >= 8192 else 4,
    )
    return o


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

    if Sq == 1:
        # Causal with Sq == 1 implies Skv == 1, which is identical to non-causal.
        return _decode(q, k, v)

    o = torch.empty_like(q)
    block_m = 64
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
