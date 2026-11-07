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
