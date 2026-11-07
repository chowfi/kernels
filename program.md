# program.md: instructions for the research agent

You are one step of an autoresearch loop (in the spirit of karpathy/autoresearch). Each time you are
invoked you run exactly ONE experiment on `kernel/attention.py`, a Triton forward-only GQA flash-attention
kernel, then stop. A driver outside your control benchmarks the result and keeps or reverts it with git.
You never see timings. You improve the kernel by reasoning about the hardware, the code, and the history.

## Goal

Beat FlashAttention-2 on every row of the benchmark table. The score is the geometric mean over the 11
table shapes of `fa2_ms / ours_ms` (above 1.0 means faster than FA2 on average). The loop stops early
when every row is at least 1.0x FA2. Correctness failures on any shape disqualify an experiment.

Workload: bf16, batch 1, 32 query heads, 8 KV heads (group size 4), head_dim 128, forward pass only.
- Prefill: causal, seq_q = seq_kv in {2048, 4096, 8192, 16384, 32768}.
- Decode: seq_q = 1, non-causal, seq_kv in {2048, 4096, 8192, 16384, 32768, 65536}.
- Layout is FA2's: Q `[B, Sq, H, D]`, K/V `[B, Skv, Hkv, D]`, output `[B, Sq, H, D]`.

Baselines (ms, from `results/baselines.json`; FA2 = flash-attn 2.8.3 built for sm_120; SDPA's default
dispatch on this card is PyTorch's built-in flash kernel, so it tracks FA2 closely):

| shape | FA2 ms | SDPA ms |
|---|---|---|
| prefill_2048 | 0.244 | 0.249 |
| prefill_4096 | 0.785 | 0.805 |
| prefill_8192 | 2.79 | 2.84 |
| prefill_16384 | 10.48 | 10.64 |
| prefill_32768 | 40.53 | 41.39 |
| decode_2048 | 0.0187 | 0.0187 |
| decode_4096 | 0.0269 | 0.0269 |
| decode_8192 | 0.0372 | 0.0372 |
| decode_16384 | 0.0577 | 0.0576 |
| decode_32768 | 0.0998 | 0.0986 |
| decode_65536 | 0.185 | 0.185 |

Decode is memory-bound (it must stream the whole KV cache once); prefill is compute-bound.

## Hardware facts (RTX 5090, from `results/device.json`)

- sm_120 (compute capability 12.0), 170 SMs, 32 GB.
- L2 cache: 100,663,296 bytes (96 MiB). Measured L2-resident bandwidth is about 4.0-4.7 TB/s for working
  sets up to 32-64 MB and falls to about 1.6 TB/s (DRAM) beyond 96 MiB. A K/V pair for one KV head at
  32k tokens is 16 MB, so one head's K/V fits in L2 many times over; all 8 KV heads at 32k (128 MB) do not.
- Shared memory: 101,376 bytes (99 KB) maximum per block (opt-in), 102,400 bytes per SM.
- sm_120 has NO wgmma and NO tcgen05 (no Hopper/Blackwell-datacenter warpgroup or TMEM paths). Tensor
  cores are the mma.sync class (m16n8k16 for bf16), as on Ampere/Ada. Triton `tl.dot` lowers to these.
- Reference FA2 on this card is therefore also an mma.sync kernel; you can win by better scheduling,
  packing, and occupancy, not by a new instruction set.

## The one-idea rule

Exactly one hypothesis per experiment. It may be a single coherent change or a config sweep of ONE
parameter family (e.g. BLOCK_M/BLOCK_N/num_warps/num_stages candidates for the autotuner). Do not bundle
unrelated changes: if the score moves we must be able to say why.

## Workflow

1. Read `results.tsv` (what was tried, statuses, scores), `notes.md`, and `git log --oneline -20`.
   Use `git show <commit>` or `git diff` to see what earlier kept kernels did.
2. Optionally run `uv run python -m harness.profile --shape NAME` (for example `--shape prefill_8192`).
   It profiles the LAST KEPT kernel (git HEAD), never your working copy, and prints occupancy,
   throughput and memory metrics (no durations).
3. Write `experiments/current.json` BEFORE touching the kernel:
   `{"exp_id": <your experiment number>, "hypothesis": "...", "rationale": "...", "expected_effect": "..."}`
   Your experiment number is given in the prompt. Use the Write tool.
4. Edit `kernel/attention.py` to implement the hypothesis.
5. Run `uv run python -m harness.check` until it passes. It checks correctness on all 15 shapes
   (the 11 table shapes plus 4 held-out shapes you cannot see or special-case: odd lengths and other
   GQA group sizes, so your kernel must be general) and prints no timing. A shape passes when both the
   max absolute error and the per-row relative error are within tolerances calibrated from bf16 rounding.
   Fix and re-run. If you cannot get it to pass, give up: edit `kernel/attention.py` back to the version
   in `git show HEAD:kernel/attention.py` (you have no `git checkout`), or just leave it; the driver
   will record `failed_correctness` and revert. A failed experiment is fine; a broken one is not hidden.
6. Append 2-5 lines to `notes.md` (use Edit to add after the last line) saying what you tried and why,
   and what you would try next. If it failed, say why you think so.
7. Stop. Do not run anything further and do not summarize at length.

## Rules

- You may edit ONLY `kernel/attention.py`, `notes.md` and `experiments/current.json`. Everything else is
  denied by permissions; a change to any other file is a violation.
- Commands allowed: `uv run python -m harness.check`, `uv run python -m harness.profile --shape NAME`,
  `git log`, `git diff`, `git show`. Everything else is denied.
- You cannot benchmark and must not try to time anything. Decide from reasoning, profile output
  and the history.
- No hard-coding per benchmark shape (no "if seq == 8192" special values tuned to the table). Shape-keyed
  autotune configs and a prefill/decode dispatch are fine, because those are general mechanisms.
- Triton + Python only. No inline asm, no CUDA/C++.
- Keep the signature `attention(q, k, v, causal)` and the FA2 layout `[B, S, H, D]`.
- Do not touch git state (no commit, no checkout, no branch).

Guard rules enforced by `driver/guard.py` (a breach reverts the whole experiment as `violation`):
- `kernel/attention.py` may import only torch, triton, math and typing. No relative or star imports.
- No inline PTX or assembly: nothing containing `inline_asm`, and no string literal containing the
  standalone word "asm" (never write it in a string; avoid it in comments too). Also banned:
  cpp_extension, load_inline, load_library, jiterator, ctypes.
- No dynamic code or introspection: no `exec`, `eval`, `compile`, `open`, `globals`, `vars`, `locals`,
  `__import__`, `importlib`, `setattr`, `delattr`. `getattr`/`hasattr` only with a literal attribute name.
- Never assign to, augment, or delete an attribute or item of an imported module (torch, triton, tl,
  math, typing) or of any name bound from one (for example `ev = torch.cuda.Event; ev.x = 1`).
  Attributes of your own objects (`self.x`, `ctx.sm_scale`, `attention._cache`) are allowed.
- No access to object internals: `__dict__`, `__setattr__`, `__delattr__`, `__class__`, `__builtins__`,
  `__globals__`, `__code__`, `__closure__`, `__subclasses__`, `__getattribute__`, `__mro__`, `__bases__`.
- The official benchmark verifies that `do_bench`, `torch.cuda.Event`, `torch.cuda.synchronize` and the
  harness timer are untouched after your kernel runs; tampering fails the experiment.

## How an experiment is judged

The driver benchmarks the kernel on the 11 table shapes twice back to back. It keeps the change only if
the whole-table geomean of `fa2_ms / ours_ms` beats the best so far by more than a noise threshold
(at least 0.5%) on both runs. Otherwise it reverts. So a clear win on some shapes that costs others is
judged on the total; consider whether your idea helps the geomean, not one row.

## Research strategy seed

Starting ideas (you may go beyond them):

1. **Fill the 170 SMs:** split-KV for decode sized to SM count; persistent grid for prefill.
2. **GQA group packing:** pack the 4 query heads sharing a KV head into the M dimension so each K/V tile is loaded once (biggest win in the 3060 prior art).
3. **L2-aware tile ordering:** schedule so K/V reuse stays resident in L2.
4. **Causal block skipping** for fully masked tiles in prefill.
5. **exp2 softmax** with log2(e) folded into the scale.
6. **Tile/stage/warp sweeps** within the ~99 KB shared-memory limit.
7. **Better decode split-combine** (reduction step).

## Using the log

`results.tsv` lists every past experiment with its status: `kept`, `reverted` (correct but not faster),
`failed_correctness`, `crashed`, `timeout`, `violation`, and others. Learn from them:
- Do not retry a reverted idea unchanged. If you revisit one, say in `rationale` what is different
  (different tile sizes, combined with a now-kept change, a bug fixed).
- A `failed_correctness` idea may still be right with a bug fixed; say what you fixed.
- Prefer ideas that build on the current kept kernel; check `git log` and `notes.md` for what it contains.
- Larger structural wins (packing, split-KV, scheduling) usually beat micro-tuning, but sweeps are cheap
  and safe once the structure is right.
