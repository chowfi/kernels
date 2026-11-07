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
