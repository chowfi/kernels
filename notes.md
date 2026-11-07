# Research notes

Experiment 0: untuned seed kernel.

Experiment 1: split-KV (flash-decoding) for non-causal launches with base grid < SM count (decode: 32 CTAs
on 170 SMs). Splits sized for ~4x SM count CTAs (min chunk 2*BLOCK_N, BLOCK_N-aligned so no empty split),
fp32 normalized partials + LSE to workspace, separate Triton combine kernel. Prefill untouched. Passed check.
Next: GQA packing in decode (4 q-heads into M so each K/V tile is read once, BLOCK_M=16 currently wastes
15/16 rows), causal block skipping for prefill (~2x on prefill rows), then exp2 and tile sweeps.

Experiment 2: causal block skipping in _attn_fwd. Causal m-blocks loop KV only to min((start_m+1)*BLOCK_M, SEQ_KV):
an unmasked phase over blocks fully below the diagonal (no mask math, unmasked K/V loads) then a masked phase
for the diagonal block(s). Expect ~2x on prefill rows. Passed check (incl. heldout_prefill_3000).
Next: reverse start_m order so the longest causal blocks launch first (tail effect), exp2 softmax with
log2e folded into scale, then prefill tile sweep (BLOCK_M 128, num_warps 8, num_stages 2-3); GQA packing for decode.

Experiment 3: prefill tile sweep via triton.autotune on _attn_fwd (Sq > 16 only; key SEQ_Q/SEQ_KV/CAUSAL/HEAD_DIM),
8 configs over BLOCK_M {64,128} x BLOCK_N {32,64,128} x warps {4,8} x stages {1,2,3}. Motivation: HEAD profile on
prefill_8192 showed 255 regs/thread (spills), 57 KB smem -> 1 block/SM, 8.3% occupancy, 25.7% SM throughput.
Passed check. Next: exp2 softmax (log2e in scale), reverse start_m order for causal tail, GQA packing in decode
(decode_16384+ still < FA2), and re-profile prefill to see which config won and whether regs still spill.

Experiment 4: GQA packing in the split-KV (decode) kernel. Each program now owns one KV head; packed row r ->
(seq r//GROUP, head kvh*GROUP + r%GROUP), so K/V is streamed once per KV head instead of once per q-head (4x less
traffic; decode_65536 KV = 268 MB, beyond L2). Split count recomputed from packed grid (8 base CTAs for decode ->
~80 splits at 64k). Workspace/combine unchanged. Passed check incl. g1/g8 held-outs. Next: decode BLOCK_N/stages/warps
sweep for the split kernel (BLOCK_N 128, num_stages 3-4), split-count target tuning, exp2 softmax, causal reverse order.

Experiment 5: exp2 softmax with log2(e) folded into qk_scale in all kernels (unmasked causal loop: raw qk, m in log2 domain,
p = exp2(qk*qk_scale - m) -> one FFMA + ex2 per element); split kernel stores log2-domain LSE, combine uses exp2.
Motivation: HEAD prefill_8192 profile = BLOCK_M 128 w8, 223 regs, 1 block/SM (16.7% occ), SM throughput 37.6%, L2 hit 96%
-> prefill is issue-limited in-CTA and is the weakest row group (~0.93x). Passed check. Next: reverse causal start_m order
(heaviest blocks first, tail), configs that fit 2 blocks/SM (BLOCK_M 64 w4 with <=128 regs / <=50 KB smem), decode split sweep.

Experiment 6: LPT launch order for causal prefill. _attn_fwd grid is now (H, num_m, B) (heads fastest) and causal maps
start_m = num_m - 1 - pid1, so every head's heaviest m-blocks launch first and diagonal-only blocks fill the tail; GROUP
q-heads sharing a KV head are adjacent CTAs streaming the same K/V in lockstep (L2 reuse). Aimed at prefill_2048/4096
(~3-12 waves at 1 block/SM). Passed check. Next: 2-blocks/SM prefill configs, decode split sweep (BLOCK_N 128, stages 3-4,
split target) for decode_32768 (~1.015x), persistent prefill grid if tail still matters.

Experiment 7: decode split-kernel config sweep. _attn_fwd_split wrapped in triton.autotune over (BLOCK_N, warps, stages)
{64w4s2 (old), 64w4s3, 64w4s4, 64w2s3, 64w8s3, 32w4s4, 32w2s4, 128w4s2, 128w8s2}, key SEQ_Q/SEQ_KV/GROUP/NUM_SPLITS/BLOCK_M/D.
Motivation: HEAD decode_32768 profile = 74% DRAM peak, L2 hit 0.7%, 2 blocks/SM (38.9 KB smem), 1.74 waves; config never tuned.
Chunk sizing stays 64-aligned (kv_mask covers BLOCK_N=128). Passed check. Next: combine kernel (150 regs, 32 CTAs, one row
each; could pack/parallelize over D), split-count target sized to whole waves of resident CTAs, unmasked full-block loop in split.

Experiment 8: wave-exact, block-balanced split-KV for decode. num_splits = min(2*SMs // base_ctas, kv_blocks // 2) (2 = smem-limited
blocks/SM of _attn_fwd_split, constant _SPLIT_BLOCKS_PER_SM) so the split grid is one full wave (336 CTAs vs 592 = 1.74 waves at 32k);
split s covers KV blocks [s*NB//S, (s+1)*NB//S) so the count is exact and splits differ by <= 1 block (CHUNK arg replaced by NUM_KV_BLOCKS).
Motivation: decode_32768 profile 74% DRAM peak with a partial trailing wave. Passed check. If neutral, the tail is not the limiter: next try
more bytes in flight per SM (3 blocks/SM via BLOCK_N 32 s3 + 1 wave sizing), unmasked inner loop in split, or a cheaper combine (150 regs).

Experiment 9: single-kernel decode. _attn_fwd_split now fuses the combine: after storing its partial, each CTA does debug_barrier +
acq_rel atomic_add on a per-(b, pid_mh) int32 counter (cached zeroed workspace, self-reset by the last CTA); the CTA seeing NUM_SPLITS-1
merges all splits for its rows as [COMB_ROWS, COMB_S, D] chunks (online log2 max) and writes bf16 O. _attn_combine launch removed.
Reasoning: prefill large-S is ~97% of est. mma.sync peak (~223 TFLOPS); decode carries 7-17 us fixed overhead, part of it the kernel
boundary + 0.06-wave combine. Passed check (incl. g1/g8/odd held-outs). If neutral, retry exp 8's 1-wave sizing on top (fewer splits ->
fewer combine chunks); if worse, the serial last-CTA combine (up to 3 chunk round trips) costs more than the launch gap it removes.

Experiment 10: one-wave split sizing for all decode lengths. This is exp 8's balanced partition (split s = KV blocks [s*NB//S, (s+1)*NB//S),
NUM_KV_BLOCKS arg) with num_splits = min(2*SMs // base_ctas, NB): the floor is 1 block per split instead of 2, so decode_2048 gets 32 splits
(256 single-tile CTAs, not 128 two-tile ones) and decode_4096 gets 42 (336 CTAs, not 256). Exp 8 won the first run (1.1581) and missed the
confirmation (1.1538 vs 1.1559). The idea is to keep its 16k-64k gain and add a lift on short decodes. Passed check. Next: decode grid with kvh
fastest (whole 2 KB token rows streamed together for DRAM locality), a 3-blocks/SM split config with 1-wave sizing, then the prefill_2048 tail/prologue.

Experiment 11: KV-head-fastest launch order for _attn_fwd_split. Grid changed from (splits, m*Hkv, B) to (m*Hkv, splits, B), so the 8 KV-head CTAs for a
token range are adjacent. Their 256 B slices of each 2 KB [Hkv, D] token row are then requested together. Before, the ~340-CTA first wave covered heads 0-4 for every
token and heads 5-7 came a wave later. Only the order changed, nothing else. Targets decode_32768/65536 (thinnest rows, DRAM-bound, 74% of peak). Passed check.
Next: if this helps, add exp 8's one-wave sizing on top (min(2*SMs//base, NB//2), and keep the 2-block floor so decode_2048 does not regress as it did in exp 10).
If it is neutral, DRAM order is not the limiter: try an unmasked inner loop in the split kernel, or 3 blocks/SM.

Experiment 12: P.V uses FP16 accumulation in _attn_fwd (both loops). pv = tl.dot(p.to(fp16), v.to(fp16), out_dtype=fp16), then acc = acc*alpha + pv.to(fp32) as one fused rescale-add.
Q.K^T, m_i/l_i and acc stay fp32. Why: prefill_32768 runs at 223 TFLOPS, which is the 5090's FP32-accumulate mma ceiling (209.5 spec, half the 419 FP16-acc rate). FA2 is capped there too.
Passed check with margin: prefill max_row_rel ~0.0023 vs tol ~0.009, the same as before. fp16 P is finer than bf16 P. Caveat: fp16 V overflows if |v| > 65504.
Next: if it wins, retune prefill tiles (lower reg/acc pressure may favour 128x128 or 2 blocks/SM), then try fp16-acc Q.K^T. That one is riskier: scores have a wider range, and q/k need a bf16->fp16 cast.
If it is neutral or slower, Triton probably converted V before the smem stage (no cp.async pipelining). Try converting after the load instead, or keep bf16 V and change only P.
