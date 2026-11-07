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
    # grid (H, num_m_blocks, B): heads vary fastest. For causal, launch the
    # heaviest m-blocks (longest KV prefix) first so cheap blocks fill the tail.
    h = tl.program_id(0)
    if CAUSAL:
        start_m = tl.num_programs(1) - 1 - tl.program_id(1)
    else:
        start_m = tl.program_id(1)
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

    # exp2 softmax: log2(e) folded into the scale, m_i kept in the log2 domain
    qk_scale = scale * 1.4426950408889634

    if CAUSAL:
        # causal requires SEQ_Q == SEQ_KV. KV blocks entirely below the diagonal
        # (cur_n < start_m*BLOCK_M <= every offs_m, and < SEQ_KV) need no mask.
        full_hi = ((start_m * BLOCK_M) // BLOCK_N) * BLOCK_N
        for start_n in range(0, full_hi, BLOCK_N):
            cur_n = start_n + offs_n
            k = tl.load(k_base + cur_n[None, :] * sk_s + offs_d[:, None])
            qk = tl.dot(q, k)

            m_new = tl.maximum(m_i, tl.max(qk, 1) * qk_scale)
            p = tl.math.exp2(qk * qk_scale - m_new[:, None])
            alpha = tl.math.exp2(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)

            v = tl.load(v_base + cur_n[:, None] * sv_s + offs_d[None, :])
            # fp16-accumulate mma (2x the fp32-acc rate on GeForce); tile sum folded into fp32 acc
            pv = tl.dot(p.to(tl.float16), v.to(tl.float16), out_dtype=tl.float16)
            acc = acc * alpha[:, None] + pv.to(tl.float32)
            m_i = m_new
        lo = full_hi
        hi = tl.minimum((start_m + 1) * BLOCK_M, SEQ_KV)
    else:
        lo = 0
        hi = SEQ_KV

    for start_n in range(lo, hi, BLOCK_N):
        cur_n = start_n + offs_n
        k = tl.load(k_base + cur_n[None, :] * sk_s + offs_d[:, None],
                    mask=cur_n[None, :] < SEQ_KV, other=0.0)
        s = tl.dot(q, k) * qk_scale
        mask = cur_n[None, :] < SEQ_KV
        if CAUSAL:
            mask = mask & (offs_m[:, None] >= cur_n[None, :])
        s = tl.where(mask, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.math.exp2(s - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)

        v = tl.load(v_base + cur_n[:, None] * sv_s + offs_d[None, :],
                    mask=cur_n[:, None] < SEQ_KV, other=0.0)
        pv = tl.dot(p.to(tl.float16), v.to(tl.float16), out_dtype=tl.float16)
        acc = acc * alpha[:, None] + pv.to(tl.float32)
        m_i = m_new

    acc = acc / l_i[:, None]
    o_ptrs = O + b * so_b + h * so_h + offs_m[:, None] * so_s + offs_d[None, :]
    tl.store(o_ptrs, acc.to(O.dtype.element_ty), mask=offs_m[:, None] < SEQ_Q)


_PREFILL_CONFIGS = [
    triton.Config({"BLOCK_M": bm, "BLOCK_N": bn}, num_warps=w, num_stages=s)
    for bm, bn, w, s in [
        (64, 64, 4, 2),
        (64, 64, 4, 1),
        (64, 64, 8, 2),
        (64, 32, 4, 3),
        (128, 64, 8, 2),
        (128, 64, 8, 1),
        (128, 32, 8, 2),
        (64, 128, 8, 1),
    ]
]

# Autotuned prefill launch (writes only O, so re-running candidates is safe).
_attn_fwd_tuned = triton.autotune(
    configs=_PREFILL_CONFIGS, key=["SEQ_Q", "SEQ_KV", "CAUSAL", "HEAD_DIM"],
)(_attn_fwd)


@triton.jit
def _attn_fwd_split(
    Q, K, V, MID_O, MID_LSE,
    sq_b, sq_s, sq_h,
    sk_b, sk_s, sk_h,
    sv_b, sv_s, sv_h,
    H, SEQ_Q, SEQ_KV, GROUP, NUM_SPLITS, CHUNK, scale,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
):
    # Non-causal split-KV pass with GQA packing: each program handles one
    # (packed m-block, KV head, split). Packed row r covers query position
    # r // GROUP and query head kvh*GROUP + r % GROUP, so the GROUP heads that
    # share a KV head read each K/V tile once. Writes a normalized fp32 partial
    # output plus its log-sum-exp.
    split = tl.program_id(0)
    pid_mh = tl.program_id(1)
    b = tl.program_id(2)
    HKV = H // GROUP
    start_m = pid_mh // HKV
    kvh = pid_mh % HKV

    offs_r = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_m = offs_r // GROUP
    offs_h = kvh * GROUP + offs_r % GROUP
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, HEAD_DIM)
    row_mask = offs_r < SEQ_Q * GROUP

    q_ptrs = Q + b * sq_b + offs_h[:, None] * sq_h + offs_m[:, None] * sq_s + offs_d[None, :]
    q = tl.load(q_ptrs, mask=row_mask[:, None], other=0.0)

    k_base = K + b * sk_b + kvh * sk_h
    v_base = V + b * sv_b + kvh * sv_h

    m_i = tl.full([BLOCK_M], float("-inf"), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    qk_scale = scale * 1.4426950408889634  # log2 domain throughout

    lo = split * CHUNK
    hi = tl.minimum(lo + CHUNK, SEQ_KV)
    for start_n in range(lo, hi, BLOCK_N):
        cur_n = start_n + offs_n
        kv_mask = cur_n < hi
        k = tl.load(k_base + cur_n[None, :] * sk_s + offs_d[:, None],
                    mask=kv_mask[None, :], other=0.0)
        s = tl.dot(q, k) * qk_scale
        s = tl.where(kv_mask[None, :], s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.math.exp2(s - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v = tl.load(v_base + cur_n[:, None] * sv_s + offs_d[None, :],
                    mask=kv_mask[:, None], other=0.0)
        acc += tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    lse = m_i + tl.math.log2(l_i)  # log2-domain LSE; combine uses exp2

    # workspace layout: MID_O [B, H, SEQ_Q, NUM_SPLITS, D], MID_LSE [B, H, SEQ_Q, NUM_SPLITS]
    row = (b * H + offs_h) * SEQ_Q + offs_m
    o_ptrs = MID_O + (row[:, None] * NUM_SPLITS + split) * HEAD_DIM + offs_d[None, :]
    tl.store(o_ptrs, acc, mask=row_mask[:, None])
    tl.store(MID_LSE + row * NUM_SPLITS + split, lse, mask=row_mask)


@triton.jit
def _attn_combine(
    MID_O, MID_LSE, O,
    so_b, so_s, so_h,
    H, SEQ_Q, NUM_SPLITS,
    HEAD_DIM: tl.constexpr, BLOCK_S: tl.constexpr,
):
    m = tl.program_id(0)
    h = tl.program_id(1)
    b = tl.program_id(2)
    offs_s = tl.arange(0, BLOCK_S)
    offs_d = tl.arange(0, HEAD_DIM)
    s_mask = offs_s < NUM_SPLITS

    row = (b * H + h) * SEQ_Q + m
    lse = tl.load(MID_LSE + row * NUM_SPLITS + offs_s, mask=s_mask, other=float("-inf"))
    m_max = tl.max(lse, 0)
    w = tl.math.exp2(lse - m_max)
    w = tl.where(s_mask, w, 0.0)
    parts = tl.load(MID_O + (row * NUM_SPLITS + offs_s[:, None]) * HEAD_DIM + offs_d[None, :],
                    mask=s_mask[:, None], other=0.0)
    out = tl.sum(parts * w[:, None], 0) / tl.sum(w, 0)
    tl.store(O + b * so_b + m * so_s + h * so_h + offs_d, out.to(O.dtype.element_ty))


_SM_COUNT = {}


def _sm_count(device: torch.device) -> int:
    idx = device.index if device.index is not None else torch.cuda.current_device()
    if idx not in _SM_COUNT:
        _SM_COUNT[idx] = torch.cuda.get_device_properties(idx).multi_processor_count
    return _SM_COUNT[idx]


def _attention_split(q, k, v, o, block_m, block_n, base_ctas, sm_count):
    B, Sq, H, D = q.shape
    _, Skv, Hkv, _ = k.shape
    target_ctas = 4 * sm_count
    min_chunk = 2 * block_n
    want = max(1, min(triton.cdiv(target_ctas, base_ctas), triton.cdiv(Skv, min_chunk)))
    chunk = triton.cdiv(triton.cdiv(Skv, want), block_n) * block_n
    num_splits = triton.cdiv(Skv, chunk)  # every split is non-empty

    mid_o = torch.empty((B, H, Sq, num_splits, D), device=q.device, dtype=torch.float32)
    mid_lse = torch.empty((B, H, Sq, num_splits), device=q.device, dtype=torch.float32)
    grid = (num_splits, triton.cdiv(Sq * (H // Hkv), block_m) * Hkv, B)
    _attn_fwd_split[grid](
        q, k, v, mid_o, mid_lse,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        H, Sq, Skv, H // Hkv, num_splits, chunk, 1.0 / math.sqrt(D),
        HEAD_DIM=D, BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=4, num_stages=2,
    )
    _attn_combine[(Sq, H, B)](
        mid_o, mid_lse, o,
        o.stride(0), o.stride(1), o.stride(2),
        H, Sq, num_splits,
        HEAD_DIM=D, BLOCK_S=max(2, triton.next_power_of_2(num_splits)),
        num_warps=4,
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

    o = torch.empty_like(q)
    block_m = 16 if Sq <= 16 else 64
    block_n = 64
    if not causal and Skv >= 4 * block_n:
        # split path packs the GROUP q-heads sharing a KV head into M
        rows = Sq * (H // Hkv)
        pack_m = 16 if rows <= 16 else 64
        base_ctas = triton.cdiv(rows, pack_m) * Hkv * B
        sm_count = _sm_count(q.device)
        if base_ctas < sm_count:
            return _attention_split(q, k, v, o, pack_m, block_n, base_ctas, sm_count)
    if Sq > 16:
        grid = lambda meta: (H, triton.cdiv(Sq, meta["BLOCK_M"]), B)  # noqa: E731
        _attn_fwd_tuned[grid](
            q, k, v, o,
            q.stride(0), q.stride(1), q.stride(2),
            k.stride(0), k.stride(1), k.stride(2),
            v.stride(0), v.stride(1), v.stride(2),
            o.stride(0), o.stride(1), o.stride(2),
            H, Sq, Skv, H // Hkv, 1.0 / math.sqrt(D),
            HEAD_DIM=D, CAUSAL=causal,
        )
        return o
    grid = (H, triton.cdiv(Sq, block_m), B)
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
