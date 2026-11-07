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
