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

Experiment 13: Q.K^T also uses FP16 accumulation in _attn_fwd (both loops). q is pre-scaled by scale*log2e in fp32 and cast to fp16 once; each K tile is cast to fp16 and
qk = tl.dot(q16, k16, out_dtype=fp16).to(fp32) comes out in the log2 domain (the per-element scale multiply is gone). Now both prefill mmas run at the 419 TFLOPS fp16-acc rate (ceiling ~21 ms at 32k vs ~31.5 before).
Passed check, but the margin is thinner: prefill max_row_rel 0.0039-0.0064 vs tol ~0.009-0.010 (was ~0.0023). It grows with seq length, so do NOT stack more precision cuts on prefill (no fp16 m/l or acc).
Next: if it wins, retune prefill tiles (128x128 w8, 2-blocks/SM configs), since mma is no longer the limiter and softmax/exp2 issue rate is. If it loses, the fp16->fp32 cvt of the scores costs more than the mma saving: keep exp 12 only.

Experiment 14: prefill autotune sweep retuned for the fp16-acc kernel. I kept the 8 old configs and added 128x64 w8 s3 (96 KB smem), 128x32 w8 s3/s4, 128x128 w8 s1, 64x64 w4 s3 and 64x32 w4 s4.
HEAD profile on prefill_16384: the winner is 128x64 w8 s2, with 255 regs (probably spilling), 64 KB smem, 1 block/SM, 16.7% occupancy and SM throughput 64%. Now that the mmas run at 2x, the bet is that more K/V stages in flight hide L2 latency. Passed check (prefill row_rel up to 0.0064 vs tol ~0.0099, unchanged).
Next: if it is neutral, register pressure (255 regs at 128x64 w8) is the limiter, not latency. Try shrinking live state, e.g. reload q16 from smem each iteration instead of holding it in registers, or add a maxnreg hint on the 128-row configs. A persistent prefill grid is another option, for the 24-wave tail.

Experiment 15: pre-convert K/V bf16->fp16 once per prefill call. A new streaming kernel, _kv_to_fp16, writes contiguous fp16 K/V into torch.empty workspaces, and the autotuned _attn_fwd then reads those, so its in-loop .to(fp16) calls become no-ops.
Why: with fp16-acc mmas (exps 12/13), every warp of every CTA was converting the full K and V tiles each iteration, about 512 cvts per thread per iteration, which also adds register pressure (255 regs). The pre-pass costs one K/V read+write (~5 us at 2048, ~170 us at 32k).
Passed check with errors identical to HEAD (same rounding, just done earlier). Small-Sq and decode paths are unchanged.
Next: if it wins, re-profile prefill_16384 for regs/SM throughput and retry the 128x64 s3 / 128x128 configs (smem per stage is the same, but registers are freed). If it is neutral, Triton was already converting cheaply. Then try a persistent prefill grid, or decode one-wave sizing (exp 8) with the 2-block floor.

Experiment 16: software-pipelined the unmasked causal loop of _attn_fwd, FA3 intra-warp overlap style. Iteration j issues S_{j+1} = Q.K_{j+1}^T (fp16 tile carried across iterations) before softmax(S_j) and P_j.V_j. A prologue computes S_0 and an epilogue finishes the last tile, all inside `if full_hi > 0`. The masked diagonal loop and the non-causal path are unchanged.
Why: HEAD prefill_16384 runs 128x64 w8 at 255 regs, 1 block/SM, SM throughput 74.6%, ~285 of ~419 TFLOPS. The per-iteration barrier keeps the warps in lockstep, so the tensor pipe idles during softmax. Passed check with errors equal to HEAD (row_rel up to 0.0064 at 32k).
Next: if it is neutral or slower, the extra score tile probably spilled (255 regs already), so retry on 64-row / BLOCK_N=32 configs only. Also untried: 128-row w4 configs (32 rows per warp halves smem ldmatrix traffic per mma, the FA2 sm86 choice), and a faster _kv_to_fp16 pre-pass (74% DRAM, ~4% of prefill_2048).

Experiment 17: residency-aware decode tile. If the 64-token split grid needs more than one wave at 2 CTAs/SM, _attention_split switches to BLOCK_N=32 (32-token chunk alignment, same 128-token min chunk and 4xSM target). At 32 that is ~21 KB smem -> 4 CTAs/SM, so decode 8k-64k (512-656 CTAs) run in one wave with the same 64 KB in flight per SM. 2k/4k keep BLOCK_N=64.
Findings: the cached TTGIR shows num_stages=2 is SINGLE-buffered here (1x K/V buffer; the next tile's cp.async is issued after the V local_load). Decode: 16 KB K + 16 KB V + 4 KB q + 2 KB p = 38.9 KB. Prefill 128x64: 32 KB K/V + 32 KB q kept in SMEM (reloaded each iteration, FA2 style) = 64 KB.
Decode fits T = a + bytes/1.64 TB/s (streaming is at DRAM peak). The overhead a grows from ~7 us (<1 wave) to ~16 us (1.7-1.9 waves), so the wave boundary is the target. Passed check, errors unchanged.
Next: if neutral, a is launches+combine, not waves (try a D-split combine with more CTAs). Prefill: 25% tensor-pipe idle is lockstep softmax; q already sits in smem, so 128-row w4 (32 rows/warp) costs registers only for acc+scores.

Experiment 18: fp16 score arithmetic in the unmasked causal loop of _attn_fwd. qk stays fp16 out of the fp16-acc mma, the row max and qk - m_new are done in fp16, and only the difference is cvt'd to fp32 for exp2. That saves ~32 of ~176 score-path ops per thread per iteration. The plan was ex2 in fp16 too (2x MUFU rate), but tl.math.exp2 rejects fp16 ("Expected dtype fp32/fp64") and inline ex2.f16x2 is banned, so that half of the idea cannot be done here.
HEAD prefill_16384 re-profile (post exp 15): still 255 regs, 64 KB smem, 1 block/SM, SM throughput 74.6%, so exp 15 did not free registers. Passed check, errors equal to HEAD (prefill row_rel 0.0039-0.0064).
Next: if neutral, ALU in the score path is not the limiter (rough count: ~1-1.5k ALU cycles vs ~8k mma cycles per SMSP per iteration). Then suspect L2->smem latency with single-buffered K/V (128-row tiles pull ~2.2 TB/s from L2 at 285 TFLOPS) or the ldmatrix/barrier cadence. Try freeing smem by keeping q in registers so K/V can double-buffer within 99 KB, or 128x128 tiles to halve barriers per flop.

Experiment 19: prefill autotune now also tries 128-row 4-warp tiles (32 rows per warp, the FA2 sm8x layout): 128x32 w4 s2, 128x32 w4 s1, 128x64 w4 s2. All 8 old configs are kept, and nothing else changed.
Why: at 128x64 w8 each warp ldmatrix-reads the whole K/V tile every iteration, and both warps on an SMSP are in one CTA, so barriers keep softmax and mma in lockstep. 128x32 w4 s2 is ~48 KB smem, so 2 independent CTAs fit per SM. That gives the same 8 warps/SM, but the CTAs can overlap one's softmax with the other's mma. Passed check, errors equal to HEAD.
Next: if it is neutral, profile to see whether 128x32 w4 was picked. If it was not, the acc (128 regs) + fp16 pv temp probably spills. Then split P.V along D into two [BLOCK_M, 64] accumulators so the pv temp halves, and retry 32-rows-per-warp. Decode (thinnest rows) still carries ~7-17 us of fixed overhead per call.

Experiment 20: prefill autotune gets register-capped 128-row 8-warp configs, so two CTAs fit per SM (16 warps/SM): 128x32 w8 s2, 128x32 w8 s1 and 128x16 w8 s2, all with maxnreg=128 (triton 3.6 Config supports it). All 8 old configs are kept. Smem is 40-48 KB (q 32 KB + K/V), under the ~49 KB/CTA limit for 2 CTAs.
Why: the lockstep softmax in the single 255-reg CTA leaves the tensor pipe ~25% idle. A second CTA overlaps it, and 128 rows keeps L2 traffic at ~2.2 TB/s, where 64-row tiles (already 2/SM) need ~4.3 TB/s. Decode profile (decode_32768): split kernel 74% DRAM, 78 regs, 2 CTAs/SM, 1.74 waves. The combine kernel is launch-bound (~0.5 us of work), so a D-split combine is not worth an experiment. Passed check, errors equal to HEAD.
Next: if neutral, profile prefill_16384 to see if a maxnreg config won. If one did not, spills at 128 are the likely cause: split P.V along D (two [M,64] accs, half the fp16 pv temp, ~16 regs saved) and retry. Triton's cache JSON has no spill count.

Experiment 21: L2 evict_first on the decode split kernel's K/V loads. do_bench zero-fills 256 MB before each rep, so L2 starts full of dirty lines, and normal-priority KV streaming evicts them and forces writebacks in the timed window. A fit T = a + (read + 0.16*min(read, 96 MiB))/1.68 TB/s gives a constant a ~7.3 us across 2k-64k decode.
FINDING: Triton drops eviction_policy on the cp.async path (num_stages>=2 PTX is plain cp.async.cg). Only plain loads get ld.global.L2::cache_hint + createpolicy.fractional.L2::evict_first, so the split launch is now num_stages=1. Side effect: smem drops 38.9 KB -> 22.5 KB, i.e. 4 CTAs/SM (32k grid = 1 wave). Passed check, errors identical.
Next: if it wins, try evict_first (num_stages=1) on _kv_to_fp16's bf16 reads, and evict_last on the fp16 K/V workspace writes so prefill reads them from L2. If neutral, writebacks are not in the window: go back to split sizing for 4 CTAs/SM (one-wave target = 4*SMs).

Experiment 22: 32-token KV tiles in the evict-first split kernel when the chunk is >= 256 tokens (16k/32k/64k decode). Chunk sizing, grid and combine are unchanged, and 2k-8k keep 64.
HEAD decode_32768 profile: 102 regs, 22.5 KB smem, 592 CTAs at 4/SM (0.87 waves), 74.6% DRAM. About 9.5 MB is in flight, roughly 6x what Little's law needs, so each load waits ~6 us in the DRAM queue and more in-flight bytes cannot help. The bet is that halving the stream's in-flight L2 footprint cuts the dirty flush-line writebacks. Passed check, errors unchanged.
Rejected PDL (launch_pdl + gdc_wait to hide the split->combine gap): Triton's gdc helpers are inline-asm wrappers, against the no-asm rule.
Next: if neutral, the writebacks do not scale with in-flight bytes, and the decode floor (~7 us) is launch + combine. If it wins, try 32-token tiles for 8k (chunk 128) too, or 16-token tiles at 64k.

Experiment 23: contiguous all-KV-heads decode. When Sq*H <= 64, the new _attn_fwd_split_flat packs all Sq*H query rows (BLOCK_M=32 for decode) and streams flattened KV rows j = tok*Hkv + kvh, so each 64-row tile is one contiguous 16 KB block (8 tokens x 8 heads) instead of 64 scattered 256 B slices at 2 KB stride. GQA is a block-diagonal mask ((r%H)//GROUP == j%Hkv) with -inf-safe max and l==0 guards. Splits are ~2*SM CTAs (min 2 tiles); still evict_first + num_stages=1.
Since each split now holds partials for all heads (~300 splits at 16k-64k, ~5 MB workspace), a new _attn_combine_chunked uses grid (Sq, H, B*D/32) and loops over splits in chunks of 64. The old split kernel and combine remain for Sq*H > 64. Why: HEAD decode_32768 is at 74.4% DRAM peak vs ~89% for plain streaming, which looks like the random-256B-access regime (each 2 KB DRAM page opened by 8 CTAs at unrelated times). Passed check (incl. g1/g8/odd held-outs), errors equal to HEAD.
Next: if it wins, try 4*SM CTAs (more in flight) or BLOCK_J=128. If neutral, page locality is not the limiter. If slower, suspect the workspace/combine (4x more partial bytes) or the drop from ~4 to ~2 CTAs/SM.

Experiment 24: split P.V along head_dim in _attn_fwd (both loops). There are two [M, D/2] fp32 accs, and each half's fp16-acc mma is folded in right away, so only a [M, D/2] pv temp is live. On top of that, exp 20's capped 2-CTA configs are back (128x32 / 128x16, w8 s2, maxnreg=128); all 8 old configs are kept.
Why: exp 20's capped configs never won (prefill identical to HEAD). My estimate puts the P.V-phase peak at ~130 regs (acc 64 + pv 32 + p 8 + frags/addr), which spills at 128. The D-split brings it to ~115, so two independent CTAs/SM can overlap softmax with mma. Passed check, errors identical to HEAD.
HEAD decode_16384 profile: split kernel at 67.9% DRAM peak, 0.75 waves (512 of 680 slots), 102 regs. Decode effective BW rises with tiles per CTA (16k ~1.45, 64k ~1.6 TB/s after ~7 us fixed), so ramp/drain or per-CTA cost dominates, not the access pattern.
Next: if neutral, profile prefill_16384 to see whether a capped config won. If none did, the 2-CTA route is dead in Triton. Then for decode, try co-issuing the K and V loads per tile (V hoisted above the QK dot, maxnreg=128 to hold 4 CTAs/SM), which halves the serialized DRAM round trips per CTA.

Experiment 25: co-issued K/V loads in _attn_fwd_split (one DRAM round trip per tile instead of two) plus maxnreg=128 on the split launch. FINDING: Triton (num_stages=1) puts each load's smem staging (st.shared for the dot operand) right after the load. So a naive hoist (K then V before the dot) gives 36 KB smem (separate K and V buffers -> 2 CTAs/SM), and the V lds still wait behind K's st.shared stall.
Fix: issue V first, then K, and defer V's staging with a redundant guard v = where(cur_n < SEQ_KV, v, 0) after the softmax (a different mask than the load's, so Triton's select(load) fold does not apply). PTX: 8 V ld, 8 K ld, bar, K st.shared, mma, softmax, V st.shared into K's buffer. shared = 22528 B (4 CTAs/SM kept). Passed check, errors identical to HEAD.
Next: if it wins, apply the same trick for software prefetch of the next tile's K (loop-carried registers). If neutral, the per-CTA latency chain is not the limiter, and the ~7 us decode floor is launch + combine. If slower, suspect spills at maxnreg=128 (try dropping maxnreg, which gives 3 CTAs/SM, with one-wave split sizing).

Experiment 26: overlapped the q and first-K round trips in _attn_fwd_split. The first KV tile is peeled out of the loop. Its K load is issued before the q load, and its smem staging is deferred with exp 25's redundant-guard trick (k = where(cur_n < SEQ_KV, k, 0) after the q load). The loop starts at lo + BLOCK_N and its body is unchanged; no maxnreg.
Why: every decode CTA used to wait one full DRAM round trip for q (flushed out of L2, tiny) before even issuing K0. All CTAs start together, so DRAM sat nearly idle for ~1 us at the start of every decode. Verified in the cached PTX: 8 K0 ld.global (evict_first), 2 q ld.global, q st.shared + bar, then K0 st.shared + bar, then mma. shared = 22528 B, so 4 CTAs/SM is kept. Passed check, errors unchanged.
HEAD prefill_2048 profile: _kv_to_fp16 is 0.30 waves at 48% DRAM (latency-bound, ~5% of the row). _attn_fwd is 128x64 w8, 3.01 waves, 1 CTA/SM. Its efficiency (~281 TFLOPS incl. diagonal waste) matches 32k, so the 2048 deficit is the pre-pass plus diagonal waste.
Next: if it wins, extend the overlap to V0 (issue V0 right after K0 in the prologue only, staged after the first softmax). If neutral, decode's fixed cost is launch + combine, not the q round trip.
