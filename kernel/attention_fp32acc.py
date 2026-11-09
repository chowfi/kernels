"""Precision-matched variant of the agent's final kernel (exp 48): the fp16-accumulate changes of
exps 12, 13 and 15 removed by hand, so both matmuls take bf16 inputs and accumulate in fp32 like FA2."""
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

    # exp2 softmax: log2(e) folded into the scale, m_i kept in the log2 domain.
    # Precision-matched to FA2: bf16 inputs, fp32 accumulation in both matmuls.
    qk_scale = scale * 1.4426950408889634

    if CAUSAL:
        # causal requires SEQ_Q == SEQ_KV. KV blocks entirely below the diagonal
        # (cur_n < start_m*BLOCK_M <= every offs_m, and < SEQ_KV) need no mask.
        full_hi = ((start_m * BLOCK_M) // BLOCK_N) * BLOCK_N
        for start_n in range(0, full_hi, BLOCK_N):
            cur_n = start_n + offs_n
            k = tl.load(k_base + cur_n[None, :] * sk_s + offs_d[:, None])
            qk = tl.dot(q, k) * qk_scale

            m_new = tl.maximum(m_i, tl.max(qk, 1))
            p = tl.math.exp2(qk - m_new[:, None])
            alpha = tl.math.exp2(m_i - m_new)
            l_i = l_i * alpha + tl.sum(p, 1)

            v = tl.load(v_base + cur_n[:, None] * sv_s + offs_d[None, :])
            acc = acc * alpha[:, None]
            acc = tl.dot(p.to(tl.bfloat16), v, acc)
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
        acc = acc * alpha[:, None]
        acc = tl.dot(p.to(tl.bfloat16), v, acc)
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
        # wide KV tile: fold/alpha/q-reload/barrier paid once per 128 columns;
        # s2 single-buffers K/V here, so ~32 KB q + 32 KB K + 32 KB V = 96 KB smem
        (128, 128, 8, 2),
    ]
]

# Autotuned prefill launch (writes only O, so re-running candidates is safe).
_attn_fwd_tuned = triton.autotune(
    configs=_PREFILL_CONFIGS, key=["SEQ_Q", "SEQ_KV", "CAUSAL", "HEAD_DIM"],
)(_attn_fwd)


@triton.jit
def _attn_fwd_split(
    Q, K, V, MID_O, MID_LSE, O, CNT,
    sq_b, sq_s, sq_h,
    sk_b, sk_s, sk_h,
    sv_b, sv_s, sv_h,
    so_b, so_s, so_h,
    H, SEQ_Q, SEQ_KV, GROUP, NUM_SPLITS, CHUNK, scale,
    HEAD_DIM: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    FUSE: tl.constexpr, BLOCK_S: tl.constexpr, CR: tl.constexpr,
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
        # K/V are streamed exactly once: evict-first keeps them from displacing
        # (and forcing writeback of) other L2-resident lines.
        k = tl.load(k_base + cur_n[None, :] * sk_s + offs_d[:, None],
                    mask=kv_mask[None, :], other=0.0, eviction_policy="evict_first")
        s = tl.dot(q, k) * qk_scale
        s = tl.where(kv_mask[None, :], s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, 1))
        p = tl.math.exp2(s - m_new[:, None])
        alpha = tl.math.exp2(m_i - m_new)
        l_i = l_i * alpha + tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v = tl.load(v_base + cur_n[:, None] * sv_s + offs_d[None, :],
                    mask=kv_mask[:, None], other=0.0, eviction_policy="evict_first")
        acc += tl.dot(p.to(tl.bfloat16), v)
        m_i = m_new

    acc = acc / l_i[:, None]
    lse = m_i + tl.math.log2(l_i)  # log2-domain LSE; combine uses exp2

    # workspace layout: MID_O [B, H, SEQ_Q, NUM_SPLITS, D], MID_LSE [B, H, SEQ_Q, NUM_SPLITS]
    row = (b * H + offs_h) * SEQ_Q + offs_m
    o_ptrs = MID_O + (row[:, None] * NUM_SPLITS + split) * HEAD_DIM + offs_d[None, :]
    tl.store(o_ptrs, acc, mask=row_mask[:, None])
    tl.store(MID_LSE + row * NUM_SPLITS + split, lse, mask=row_mask)

    if FUSE:
        # Short decodes only: the last split CTA for these packed rows combines
        # all splits in-kernel (one L2 round trip), so no combine launch is needed.
        tl.debug_barrier()  # every thread's partial stores precede the release
        cnt_ptr = CNT + b * tl.num_programs(1) + pid_mh
        old = tl.atomic_add(cnt_ptr, 1, sem="acq_rel", scope="gpu")
        if old == NUM_SPLITS - 1:
            tl.atomic_xchg(cnt_ptr, 0, sem="relaxed", scope="gpu")  # ready for next call
            offs_s = tl.arange(0, BLOCK_S)
            s_mask = offs_s < NUM_SPLITS
            offs_cr = tl.arange(0, CR)
            n_rows = tl.minimum(SEQ_Q * GROUP - start_m * BLOCK_M, BLOCK_M)
            for r0 in range(0, n_rows, CR):
                r_ok = (r0 + offs_cr) < n_rows
                rr = start_m * BLOCK_M + r0 + offs_cr
                rm = rr // GROUP
                rh = kvh * GROUP + rr % GROUP
                crow = (b * H + rh) * SEQ_Q + rm
                lmask = r_ok[:, None] & s_mask[None, :]
                lse_c = tl.load(MID_LSE + crow[:, None] * NUM_SPLITS + offs_s[None, :],
                                mask=lmask, other=float("-inf"), cache_modifier=".cg")
                m_max = tl.where(r_ok, tl.max(lse_c, 1), 0.0)
                w = tl.math.exp2(lse_c - m_max[:, None])
                parts = tl.load(
                    MID_O + (crow[:, None, None] * NUM_SPLITS + offs_s[None, :, None]) * HEAD_DIM
                    + offs_d[None, None, :],
                    mask=lmask[:, :, None], other=0.0, cache_modifier=".cg")
                num = tl.sum(parts * w[:, :, None], 1)
                den = tl.where(r_ok, tl.sum(w, 1), 1.0)
                out = num / den[:, None]
                tl.store(O + b * so_b + rm[:, None] * so_s + rh[:, None] * so_h + offs_d[None, :],
                         out.to(O.dtype.element_ty), mask=r_ok[:, None])


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

# Zeroed int32 arrival counters for the fused short-decode combine, keyed on
# (device, count); the last CTA of each group resets its counter to 0.
_COUNTERS = {}
_FUSE_MAX_SPLITS = 32

# Sequence length from which prefill uses the wide 128x128 tile without autotuning
# (diagonal waste and pipeline prologue are small relative to per-tile overheads).
_WIDE_TILE_MIN_SEQ = 8192


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
    # Fused in-kernel combine only when it is one cheap round trip for the last
    # CTA (few splits) and the grid is small enough that its extra registers
    # cannot cost residency.
    fuse = num_splits <= _FUSE_MAX_SPLITS and num_splits * grid[1] * B <= 2 * sm_count
    n_cnt = grid[1] * B
    ckey = (q.device.index, n_cnt)
    if fuse and ckey not in _COUNTERS:
        _COUNTERS[ckey] = torch.zeros(n_cnt, device=q.device, dtype=torch.int32)
    cnt = _COUNTERS[ckey] if fuse else mid_lse
    _attn_fwd_split[grid](
        q, k, v, mid_o, mid_lse, o, cnt,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        H, Sq, Skv, H // Hkv, num_splits, chunk, 1.0 / math.sqrt(D),
        HEAD_DIM=D, BLOCK_M=block_m, BLOCK_N=block_n,
        FUSE=fuse, BLOCK_S=max(2, triton.next_power_of_2(num_splits)), CR=4,
        # num_stages=1: plain ld.global keeps the L2 evict_first policy (the
        # cp.async path of the pipeliner drops it).
        num_warps=4, num_stages=1,
    )
    if fuse:
        return o
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
        if min(Sq, Skv) >= _WIDE_TILE_MIN_SEQ:
            # Long prefill: the 128x128 w8 s2 tile is consistently the fastest here,
            # but the one-shot autotuner picks it only about half the time (timing noise
            # vs 128x64), so choose it deterministically.
            _attn_fwd[(H, triton.cdiv(Sq, 128), B)](
                q, k, v, o,
                q.stride(0), q.stride(1), q.stride(2),
                k.stride(0), k.stride(1), k.stride(2),
                v.stride(0), v.stride(1), v.stride(2),
                o.stride(0), o.stride(1), o.stride(2),
                H, Sq, Skv, H // Hkv, 1.0 / math.sqrt(D),
                HEAD_DIM=D, BLOCK_M=128, BLOCK_N=128, CAUSAL=causal,
                num_warps=8, num_stages=2,
            )
            return o
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
