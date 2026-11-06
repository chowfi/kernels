# Autoresearch for a GQA Flash Attention Kernel on RTX 5090 — Design

Date: 2026-11-06
Author: Fiona Chow (with Claude Code)

## Intent

A public portfolio project for a hiring manager looking for a research engineer
who can **use agents to optimize infrastructure**. The artifact demonstrates:

1. A Karpathy-style autoresearch loop (https://github.com/karpathy/autoresearch),
   adapted from "train a small LLM" to "optimize a GPU kernel", that runs
   autonomously for up to 12 hours.
2. The result of that loop: a Triton GQA flash-attention forward kernel tuned
   for the user's RTX 5090, benchmarked against FlashAttention-2 and PyTorch SDPA.
3. Honest accounting: Claude tokens used, wall-clock time, number of experiments.

**Success** = credible, reproducible comparison vs FA2 and SDPA on a real 5090,
plus a transparent experiment log showing how the agent got there. Getting close
to FA2 (not necessarily beating it everywhere) with a clear story **is a win**.

**Anti-goals** (failure even if numbers look good):
- Numbers that don't hold up: wrong outputs, rigged timing, unfair baselines.
- The agent editing the benchmark harness or correctness check.
- A log that hides failed experiments.

## Prior art (cite in README)

- `tonbistudio/gqa-flash-attn-rtx3060` — same idea on an RTX 3060, matched FA2
  at 0.94–1.03×, beat SDPA 1.5–7.4×. **Its results table is the template.**
  It has no agent loop or token/time accounting; that is our differentiator.
- `fastflash-attention` (PyPI) — a Triton FA tuned on RTX 5090 claiming to beat
  FA2 forward. Cite it; do not hide it.
- Karpathy's `autoresearch` — the loop design (fixed harness, one editable file,
  instructions file, results log, keep/discard via git).

## Hardware facts (RTX 5090, this machine)

- Compute capability 12.0 (sm_120, consumer Blackwell), 3150 MHz max SM clock, 600 W, 32 GB.
- 170 SMs. L2 reported ~96 MB (sources say 88–98 MB) — **measure it** in setup.
- 128 KB L1/shared per SM; ~99 KB max shared memory per block (verify via
  device query).
- ~1.8 TB/s DRAM bandwidth, ~209 TFLOP/s dense BF16 tensor.
- No wgmma/tcgen05 (those are Hopper / datacenter Blackwell). FA3 is
  Hopper-only; FA4 reportedly does not run on sm_120. **FA2 is therefore the
  correct "best official library" baseline** for this card.
- Local tooling: nvcc 12.8 at `/usr/local/cuda-12.8`, driver CUDA 13.0, `uv`
  available, no torch installed yet.

## Decided (by the user)

- **No Modal.** Everything runs locally on the RTX 5090.
- **Triton**, GQA flash attention, **forward pass only**.
- **No model weights.** Random tensors in real model shapes. No end-to-end LLM demo.
- **Batch 1.**
- **Budget: 12 hours** of autonomous loop time.
- **Metrics to log:** Claude tokens used, wall-clock time to reach the goal,
  number of experiments. **No compute cost.**
- **Deliverable:** public GitHub repo; the README is what the hiring manager reads.
- Results table mirrors the tonbistudio repo format.

## Benchmark specification

- Shapes: Llama-3.1-8B attention — 32 query heads, 8 KV heads (group size 4),
  head_dim 128, bf16, batch 1.
- **Prefill:** causal, seq_q = seq_kv ∈ {2048, 4096, 8192, 16384, 32768}.
- **Decode:** seq_q = 1, non-causal over KV cache, seq_kv ∈ {2048, 4096, 8192, 16384, 32768, 65536}.
- Baselines:
  - **FA2:** official `flash-attn` built for sm_120 (CUDA 12.8). Prefill via
    `flash_attn_func(causal=True)`; decode via `flash_attn_with_kvcache`. If the
    official build fails, fall back to a community sm_120 build and note it.
  - **SDPA:** `torch.nn.functional.scaled_dot_product_attention(..., enable_gqa=True)`,
    default backend selection; record and report which backend it chose.
- Each kernel receives inputs pre-converted to its native layout; layout
  conversion excluded from timing (same as prior art).
- Timing: warmup, then CUDA-event / `triton.testing.do_bench`-style timing,
  median over N trials (builder picks N; ≥5). L2 flushed between reps so
  results aren't inflated by a warm cache — builder's call on exact method, but
  apply identically to all three kernels.
- **Correctness gate:** every shape is compared to an fp32 reference (or SDPA
  math backend) with bf16-appropriate tolerance. A failing shape disqualifies
  the experiment.

### Results table (README)

| Workload | seq_kv | ours (ms) | FA2 (ms) | SDPA (ms) | vs FA2 | vs SDPA |
|---|---|---|---|---|---|---|

Headline: min–max "vs FA2" and "vs SDPA" ranges. Below the table:
Claude tokens used, wall-clock time, experiments run / kept / reverted, and a
short "what worked / what didn't" list.

## The autoresearch loop

Modeled on Karpathy's structure:

- **Locked harness** (`bench/` or similar): correctness check, timing, baseline
  runs, scoring. The agent must not modify it. Enforce mechanically (e.g. the
  driver verifies the harness hash before scoring / rejects commits touching it).
- **One editable file:** the Triton kernel (plus its launch wrapper/config).
- **`program.md`:** instructions for the agent — goal, rules, research strategy,
  how to log.
- **`results.tsv`:** one row per experiment: id, timestamp, hypothesis,
  per-shape timings, score, kept/reverted, tokens used, short note.
- **Git:** each kept experiment is a commit; reverted ones are logged (TSV +
  notes) but code is reset. Git history + log = the research record.
- **Notes file:** running memory of insights across experiments.

### Driver

A script that invokes Claude Code headlessly (`claude -p`, JSON output) **once
per experiment**. My call, for two reasons: each invocation reports exact token
usage (precise per-experiment accounting), and fresh context per experiment
avoids context degradation over 12 hours. Cross-experiment memory lives in
files (results.tsv, notes, git log).

The driver records start/end wall-clock and active time (see below), sums tokens (input, output, cache
read, cache write reported separately), and stops when **either** 12 hours of active time
elapse **or** every row in the table is ≥ 1.0× FA2. It writes a progress log the
user can tail. The user starts it and walks away.

Scoring for keep/discard: builder's call, but it must reflect the whole table
(e.g. geometric mean of speedup vs FA2 across all shapes), must require all
shapes correct, and should guard against noise (e.g. require improvement beyond
measured run-to-run variance).

### Starting research strategy (seed for program.md; agent may go beyond)

1. **Fill the 170 SMs:** split-KV for decode sized to SM count; persistent
   grid for prefill.
2. **GQA group packing:** pack the 4 query heads sharing a KV head into the M
   dimension so each K/V tile is loaded once (biggest win in the 3060 prior art).
3. **L2-aware tile ordering:** schedule so K/V reuse stays resident in L2.
4. **Causal block skipping** for fully masked tiles in prefill.
5. **exp2 softmax** with log2(e) folded into the scale.
6. **Tile/stage/warp sweeps** within the ~99 KB shared-memory limit.
7. **Better decode split-combine** (reduction step).

Further ideas to suggest in the README as next steps: FP8 K/V, TMA loads on
sm_120 if Triton exposes them, warp specialization, autotune caching per shape.

## Accounting

- **Claude tokens:** from headless JSON usage per invocation, summed by the
  driver. Also include tokens from the setup/building session (this
  conversation and the implementation), reported separately from loop tokens.
- **Wall-clock and active time:** loop start → goal reached or 12h active-time cap (pauses on usage limits excluded from active time); also report total
  project time including setup.
- **Experiments:** total / kept / reverted / failed correctness.

## Loop model and usage limits (decided by the user)

- The loop runs on **Claude Opus 5.5 (`claude-opus-5-5`) at high effort** for
  every invocation, through the user's subscription plan.
- **Handle pauses:** the driver detects usage-limit / rate-limit errors from
  `claude -p`, waits with back-off until it can continue (probing
  periodically), and resumes without losing state.
- **Report active time:** the driver tracks wall-clock and active time
  separately (active = wall-clock minus time paused on limits). The README
  reports both. The **12-hour budget is measured in active time**, with a hard
  wall-clock ceiling the builder picks (e.g. 36 h) so it can't run forever.
  (The active-time budget and the ceiling are Claude's calls.)

## Rules settled after builder review (Claude's calls; user may override)

- **What an experiment is (user-confirmed): 1 experiment = 1 idea,
  implemented and benchmarked once.** In one `claude -p` invocation the agent:
  (1) reads results.tsv, notes and git log, and writes one hypothesis;
  (2) implements it in the kernel file; (3) may run the **correctness-only**
  check (no timing) to fix bugs in its implementation, with every run logged
  to `runs.log`; (4) hands off. The **driver** then runs **one** official
  benchmark, and that result alone decides keep/revert. The agent gets no timing
  access during the experiment, so the experiment count is an honest count of
  ideas tried. Profiling (`ncu`/`nsys`) is allowed only as a separate, logged
  diagnostic step to inform the *next* idea, never to tune the current one.
  Each invocation gets a timeout (~20 min, builder may tune).
- **What the official benchmark covers (user-confirmed):** our kernel on the
  full table (all 5 prefill + 6 decode lengths) plus correctness on those and
  the held-out shapes. **FA2 and SDPA are timed once at loop start and cached**,
  since their code never changes. Each experiment's vs-FA2 / vs-SDPA ratios use
  the cached baselines. Safeguards: baselines are **re-timed periodically**
  (every few hours, builder picks the interval) to catch GPU/thermal drift, and
  the **README table comes from one fresh final run** timing all three kernels
  side by side.
- **Generality:** the kernel must be general, not fit to the 11 benchmark
  shapes. The locked harness also checks correctness on held-out shapes that
  aren't on the table (e.g. non-tile-multiple seq lengths like 3000 and 5001,
  plus another GQA group size). `program.md` bans hard-coding per benchmark shape.
  Autotuning keyed on shape (as Triton's `autotune` does) and a
  prefill/decode dispatch are allowed.
- **Layout:** our kernel takes FA2's layout (`[B, S, H, D]` for Q/K/V and the
  KV cache), so no work can move into an untimed conversion step. FA2 gets its
  native layout and SDPA gets `[B, H, S, D]`, converted outside the timed region.
- **What counts as Triton:** Triton kernels plus Python launch code only. No
  inline PTX/asm, no CUDA/C++ extensions. Pin the Triton/PyTorch versions in
  the harness; the agent may not change them.

## Left to the builder

- Repo layout, file names, Python environment setup (uv suggested).
- Exact timing method details, trial count, tolerances.
- Exact scoring formula and noise threshold (within constraints above).
- How the harness lock is enforced.
- Whether to add plots in the README (nice to have).
