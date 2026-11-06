# GQA Flash Attention Autoresearch (RTX 5090) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Karpathy-style autoresearch loop in which headless Claude Code (Opus 5.5, high effort) iteratively optimizes a Triton GQA flash-attention forward kernel on the local RTX 5090 for up to 12 active hours, then publish a README results table vs FA2 and SDPA with token, time, and experiment accounting.

**Architecture:** A locked `harness/` package owns shapes, inputs, baselines, the fp32 reference, correctness, timing, and scoring. The agent may edit only `kernel/attention.py` (plus `notes.md` and `experiments/current.json`). A `driver/` package runs one `claude -p` invocation per experiment (one idea), then does the single official benchmark, keeps or reverts via git, appends `results.tsv`, and tracks tokens and active vs wall time through usage-limit pauses. `scripts/` produce the final fresh benchmark, README table, progress plot, and token report.

**Tech Stack:** Python 3.12 via `uv`, PyTorch (CUDA 12.8+ wheels), Triton (the version bundled with that PyTorch), `flash-attn` 2.x built for sm_120 with `/usr/local/cuda-12.8`, pytest, matplotlib, Claude Code CLI 2.1.x (`claude -p --output-format stream-json --verbose --model claude-opus-5-5 --effort high`).

**Spec:** `docs/superpowers/specs/2026-11-06-gqa-flash-attn-autoresearch-design.md`

## Global Constraints

- Everything runs locally on the RTX 5090 (sm_120, 170 SMs). No Modal. No compute-cost reporting.
- Triton forward pass only. No inline PTX/asm (`tl.inline_asm_elementwise` banned), no CUDA/C++ extensions in `kernel/`.
- Shapes: 32 query heads, 8 KV heads, head_dim 128, bf16, batch 1.
- Prefill: causal, seq_q = seq_kv ∈ {2048, 4096, 8192, 16384, 32768}. Decode: seq_q = 1, non-causal, seq_kv ∈ {2048, 4096, 8192, 16384, 32768, 65536}.
- Held-out correctness-only shapes (never on the table): prefill causal 3000; decode 5001; prefill causal 4096 with 32 q heads / 4 KV heads (group 8); decode 8192 with 32 q / 32 KV heads (group 1).
- Our kernel's layout is FA2's: Q `[B, Sq, H, D]`, K/V `[B, Skv, Hkv, D]`, output `[B, Sq, H, D]`. SDPA receives `[B, H, S, D]`; conversion outside the timed region.
- FA2: `flash_attn_func(causal=True)` for prefill, `flash_attn_with_kvcache` for decode. SDPA: `F.scaled_dot_product_attention(..., enable_gqa=True)` with default backend; record the chosen backend.
- Loop model: `claude-opus-5-5`, effort `high`, every invocation.
- Budget: 12 hours **active** time; hard wall-clock ceiling 36 hours. Stop early when every table row is ≥ 1.0× FA2.
- 1 experiment = 1 idea = 1 `claude -p` invocation (timeout 20 min) + 1 official driver benchmark. The agent has no timing access.
- FA2/SDPA baselines timed at loop start, cached in `results/baselines.json`, re-timed every 3 hours; README table comes from one fresh final run of all three kernels.
- Keep/revert score: geometric mean over the 11 table shapes of `fa2_ms / ours_ms`; any correctness failure on any shape (table or held-out) disqualifies.
- Accounting: loop tokens (input, output, cache read, cache write separately) and build-session tokens reported separately; wall-clock and active time; experiments total / kept / reverted / failed correctness.
- Public GitHub repo; README cites `tonbistudio/gqa-flash-attn-rtx3060`, `fastflash-attention` (PyPI), and `karpathy/autoresearch`.

## Review Focus

1. **The Claude CLI fails for a reason that isn't a usage limit** (network drop, non-JSON output, crash): log the experiment as `agent_error`, revert, continue. After 3 consecutive `agent_error`s, treat it as a pause with back-off so a dead network doesn't burn the active-time budget. Test in Task 8.
2. **A kernel hangs or triggers an illegal memory access** (poisons the CUDA context): `harness.bench` and `harness.check` run as subprocesses with timeouts. The experiment is logged `crashed`, the kernel reverted, and the loop continues. Test in Task 8.
3. **The driver process dies or the machine reboots mid-run:** on restart the driver resumes from `results/run_state.json`, reverts any dirty kernel to the last kept commit, counts the downtime as paused (not active), and continues the experiment numbering. Test in Task 7 (clock) and Task 8 (resume).
4. **The agent touches anything outside its allowed files, or moves HEAD** (edits the harness, `results.tsv`, `program.md`, or commits): log `violation` and hard-reset to the last kept commit. Test in Task 8.
5. **A lucky noisy run looks like an improvement:** a candidate is kept only if its score beats best × (1 + threshold) on two back-to-back official benchmark runs, where threshold = max(0.5%, 2 × score coefficient of variation measured at loop start). Test in Task 5 (`is_improvement`) and Task 8 (confirm run).

---

## File Structure

```
pyproject.toml, uv.lock            env (Task 1)
harness/                           LOCKED — agent may not edit
  __init__.py
  device_info.py                   device props + L2 bandwidth sweep (Task 1)
  shapes.py                        Shape dataclass, BENCH_SHAPES, HELDOUT_SHAPES (Task 2)
  inputs.py                        seeded input construction per layout (Task 2)
  baselines.py                     fa2 / sdpa callables, sdpa backend detection (Task 2)
  reference.py                     chunked fp32 reference attention (Task 3)
  correctness.py                   correctness gate (Task 3)
  timing.py                        L2-flushed median timing (Task 5)
  score.py                         geomean score, noise threshold, improvement test (Task 5)
  check.py                         `python -m harness.check` — correctness only, agent-callable (Task 5)
  bench.py                         `python -m harness.bench` — official, driver-only (Task 5)
  profile.py                       `python -m harness.profile` — ncu on HEAD kernel (Task 6)
kernel/attention.py                EDITABLE — seed Triton kernel (Task 4)
driver/
  clock.py                         active/wall clock with persisted state (Task 7)
  claude_stream.py                 run claude, parse stream-json, classify errors (Task 7)
  ledger.py                        token totals (Task 7)
  guard.py                         harness hash + allowed-change checks (Task 8)
  results.py                       results.tsv schema + append (Task 8)
  loop.py                          `python -m driver.loop` main loop (Task 8)
program.md, notes.md               agent instructions + running memory (Task 9)
experiments/current.json           agent's hypothesis for the current experiment (Task 9)
results/                           baselines.json, run_state.json, device.json, bench/*.json, final.json
results.tsv, runs.log, progress.log
scripts/token_report.py, scripts/final_report.py, scripts/plot_progress.py (Task 10)
README.md                          (Task 10, finalized Task 11)
tests/                             pytest; GPU tests marked `@pytest.mark.gpu`
```

---

### Task 1: Environment and device facts

**Files:**
- Create: `pyproject.toml`, `harness/__init__.py`, `harness/device_info.py`, `tests/test_device_info.py`, `tests/conftest.py`, `.gitignore`

**Interfaces:**
- Produces: `get_device_facts() -> dict` with keys `name, compute_capability (tuple), sm_count, l2_bytes, smem_per_block_optin_bytes, smem_per_sm_bytes, total_mem_bytes`; `l2_bandwidth_sweep(sizes_mb: list[int]) -> list[tuple[int, float]]` returning (MB, GB/s); `python -m harness.device_info` writes `results/device.json`.

- [ ] **Step 1:** Create a `uv` project pinned to Python 3.12. Add torch from the PyTorch CUDA 12.8 index (newest stable at install time) plus pytest, numpy, matplotlib. Register the pytest marker `gpu`. `.gitignore` covers `.venv/`, `__pycache__/`, `results/bench/`.
- [ ] **Step 2: Write the failing test** `tests/test_device_info.py`:

```python
@pytest.mark.gpu
def test_device_is_5090():
    f = get_device_facts()
    assert f["compute_capability"] == (12, 0)
    assert f["sm_count"] == 170
    assert 80 * 2**20 <= f["l2_bytes"] <= 130 * 2**20
    assert f["smem_per_block_optin_bytes"] >= 96 * 1024

@pytest.mark.gpu
def test_l2_sweep_shows_cliff():
    pts = dict(l2_bandwidth_sweep([16, 512]))
    assert pts[16] > 1.3 * pts[512]   # in-L2 reads beat DRAM reads
```

- [ ] **Step 3:** Run `uv run pytest tests/test_device_info.py -v`. Expected: FAIL (module missing).
- [ ] **Step 4:** Implement `harness/device_info.py`. Use `torch.cuda.get_device_properties` (`multi_processor_count`, `L2_cache_size`, `shared_memory_per_block_optin`, `shared_memory_per_multiprocessor`). The sweep repeatedly sums a bf16 buffer of each size and reports GB/s from CUDA events. The `__main__` sweeps 4…512 MB in powers of two and writes `results/device.json`.
- [ ] **Step 5:** Run the tests and `uv run python -m harness.device_info`. Expected: PASS, and `results/device.json` exists. Use its `l2_bytes` wherever docs say "~96 MB".
- [ ] **Step 6:** Commit `pyproject.toml uv.lock .gitignore harness/ tests/ results/device.json` with message `feat: env setup and 5090 device facts`.

### Task 2: Shapes, inputs, and baselines (FA2 + SDPA)

**Files:**
- Create: `harness/shapes.py`, `harness/inputs.py`, `harness/baselines.py`, `tests/test_baselines.py`
- Modify: `pyproject.toml` (add flash-attn)

**Interfaces:**
- Produces:
  - `@dataclass(frozen=True) Shape(workload: Literal["prefill","decode"], seq_q: int, seq_kv: int, n_heads: int = 32, n_kv_heads: int = 8, head_dim: int = 128, causal: bool)` with property `name -> str` (e.g. `"prefill_8192"`, held-out names like `"heldout_prefill_3000"`, `"heldout_prefill_4096_g8"`).
  - `BENCH_SHAPES: list[Shape]` (11, in table order: prefill ascending, then decode ascending), `HELDOUT_SHAPES: list[Shape]` (4, per Global Constraints).
  - `make_inputs(shape: Shape, seed: int = 0) -> tuple[Tensor, Tensor, Tensor]` in FA2 layout, bf16, `torch.randn` on cuda.
  - `to_sdpa_layout(t: Tensor) -> Tensor` (`transpose(1,2).contiguous()`).
  - `fa2_attention(q, k, v, causal: bool) -> Tensor` (FA2 layout in/out; dispatches to `flash_attn_with_kvcache` when `q.shape[1] == 1` and not causal, else `flash_attn_func`).
  - `sdpa_attention(q, k, v, causal: bool) -> Tensor` (SDPA layout in/out, `enable_gqa=True`).
  - `sdpa_backend_name(shape: Shape) -> str` (which backend the default dispatch picks: probe by enabling one backend at a time with `torch.nn.attention.sdpa_kernel` in priority order cuDNN, flash, efficient, math, and return the first that runs; label the result "default (≈<name>)").

- [ ] **Step 1:** Install flash-attn: `MAX_JOBS=8 CUDA_HOME=/usr/local/cuda-12.8 TORCH_CUDA_ARCH_LIST=12.0 uv pip install flash-attn --no-build-isolation`, pinning the newest 2.x release that lists sm_120. If the build fails, fall back to `kernels-community/flash-attn2` via the `kernels` package, and record which one was used in `results/device.json["fa2_source"]`. The compile can take over 30 minutes; run it in the background.
- [ ] **Step 2: Write the failing tests** `tests/test_baselines.py`:

```python
def test_shape_lists():
    assert [s.seq_kv for s in BENCH_SHAPES] == [2048,4096,8192,16384,32768, 2048,4096,8192,16384,32768,65536]
    assert all(s.causal for s in BENCH_SHAPES if s.workload == "prefill")
    assert all(s.seq_q == 1 and not s.causal for s in BENCH_SHAPES if s.workload == "decode")
    assert len(HELDOUT_SHAPES) == 4 and not set(HELDOUT_SHAPES) & set(BENCH_SHAPES)

@pytest.mark.gpu
@pytest.mark.parametrize("shape", [Shape("prefill",1024,1024,causal=True), Shape("decode",1,4096,causal=False)])
def test_fa2_matches_sdpa(shape):
    q,k,v = make_inputs(shape)
    o_fa2 = fa2_attention(q,k,v,shape.causal)
    o_sdpa = sdpa_attention(*map(to_sdpa_layout,(q,k,v)), shape.causal).transpose(1,2)
    assert o_fa2.shape == q.shape
    torch.testing.assert_close(o_fa2, o_sdpa, atol=2e-2, rtol=2e-2)

@pytest.mark.gpu
def test_sdpa_backend_reported():
    assert sdpa_backend_name(BENCH_SHAPES[0]) != ""
```

- [ ] **Step 3:** Run `uv run pytest tests/test_baselines.py -v`. Expected: FAIL.
- [ ] **Step 4:** Implement the three modules per the Interfaces.
- [ ] **Step 5:** Run the tests. Expected: PASS.
- [ ] **Step 6:** Commit with message `feat: shapes, inputs, FA2 and SDPA baselines`.

### Task 3: fp32 reference and correctness gate

**Files:**
- Create: `harness/reference.py`, `harness/correctness.py`, `tests/test_correctness.py`

**Interfaces:**
- Consumes: `Shape`, `make_inputs`, `sdpa_attention`, `to_sdpa_layout` (Task 2).
- Produces:
  - `reference_attention(q, k, v, causal: bool) -> Tensor` (fp32 output, FA2 layout; processes query rows in chunks of ≤1024 per KV head group so 32K prefill fits in memory; GQA by repeating KV heads per chunk).
  - `@dataclass CorrectnessResult(shape_name: str, ok: bool, max_abs_err: float, tol: float, error: str | None)`.
  - `check_kernel(fn: Callable[[Tensor,Tensor,Tensor,bool],Tensor], shapes: list[Shape], seed: int = 0) -> list[CorrectnessResult]`. Catches exceptions per shape (`ok=False, error=repr`). Wrong shape, wrong dtype, or NaN means fail.
  - Tolerance (FlashAttention's convention): `tol = 2 * max_abs_err(sdpa_math_bf16, ref) + 1e-3`. Here `sdpa_math_bf16` is SDPA under `sdpa_kernel(SDPBackend.MATH)` on the bf16 inputs. Cache it per (shape, seed) in `results/tolerances.json` so later checks are fast.

- [ ] **Step 1: Write the failing tests**:

```python
@pytest.mark.gpu
def test_reference_matches_sdpa_math_small():
    s = Shape("prefill",512,512,causal=True); q,k,v = make_inputs(s)
    ref = reference_attention(q,k,v,True)
    assert ref.dtype == torch.float32
    with sdpa_kernel(SDPBackend.MATH):
        m = sdpa_attention(*(to_sdpa_layout(t.float()) for t in (q,k,v)), True).transpose(1,2)
    torch.testing.assert_close(ref, m, atol=1e-4, rtol=1e-4)

@pytest.mark.gpu
def test_gate_passes_fa2_and_rejects_broken():
    shapes = [Shape("prefill",1024,1024,causal=True), Shape("decode",1,3000,causal=False)]
    assert all(r.ok for r in check_kernel(fa2_attention, shapes))
    broken = lambda q,k,v,c: fa2_attention(q,k,v,c) * 1.01
    assert not any(r.ok for r in check_kernel(broken, shapes))
    raising = lambda q,k,v,c: 1/0
    r = check_kernel(raising, shapes[:1])[0]
    assert not r.ok and "ZeroDivisionError" in r.error

@pytest.mark.gpu
def test_reference_handles_32k_prefill():
    s = BENCH_SHAPES[4]; q,k,v = make_inputs(s)
    assert reference_attention(q,k,v,True).shape == q.shape
```

- [ ] **Step 2:** Run `uv run pytest tests/test_correctness.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `reference.py` and `correctness.py` per the Interfaces.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit with message `feat: chunked fp32 reference and correctness gate`.

### Task 4: Seed kernel (experiment 0)

**Files:**
- Create: `kernel/__init__.py`, `kernel/attention.py`, `tests/test_seed_kernel.py`

**Interfaces:**
- Consumes: `check_kernel`, `BENCH_SHAPES`, `HELDOUT_SHAPES`.
- Produces: `attention(q: Tensor, k: Tensor, v: Tensor, causal: bool) -> Tensor` in FA2 layout, bf16. It raises `ValueError` for head_dim ∉ {64, 128}, for `n_heads % n_kv_heads != 0`, or for causal with seq_q ≠ seq_kv. This is the only symbol the harness imports from `kernel/`.

The seed is deliberately **plain and untuned** so the agent has the known wins left to find. It must be correct, not fast:
- One Triton program per (query block, query head). Each query head maps to its KV head via `h // (H // Hkv)`.
- Loop over K/V blocks with online softmax (natural `exp`, fp32 accumulator).
- Fixed `BLOCK_M=64, BLOCK_N=64, num_warps=4, num_stages=2`. Decode uses the same kernel with `BLOCK_M=16`.
- Masking for non-multiple sequence lengths and causal. No causal block skipping, no split-KV, no GQA packing, no autotune.

- [ ] **Step 1: Write the failing test**:

```python
@pytest.mark.gpu
def test_seed_kernel_correct_everywhere():
    results = check_kernel(attention, BENCH_SHAPES + HELDOUT_SHAPES)
    bad = [r for r in results if not r.ok]
    assert not bad, bad

def test_seed_kernel_rejects_bad_head_dim():
    q = torch.empty(1,8,32,96, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(ValueError):
        attention(q, q[:,:,:8], q[:,:,:8], False)
```

- [ ] **Step 2:** Run `uv run pytest tests/test_seed_kernel.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement `kernel/attention.py` as described.
- [ ] **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Commit with message `feat: untuned seed Triton GQA attention kernel (experiment 0)`.

### Task 5: Timing, scoring, `harness.check` and `harness.bench`

**Files:**
- Create: `harness/timing.py`, `harness/score.py`, `harness/check.py`, `harness/bench.py`, `tests/test_score.py`, `tests/test_bench.py`

**Interfaces:**
- Consumes: Tasks 2–4.
- Produces:
  - `time_fn(fn: Callable[[], Any], trials: int = 5) -> TimingResult(median_ms: float, stdev_ms: float, trial_ms: list[float])`. Each trial is `triton.testing.do_bench(fn, warmup=25, rep=100, return_mode="median")`; do_bench flushes L2 between reps. Report the median and stdev of the trial medians.
  - `score(ours_ms: dict[str,float], fa2_ms: dict[str,float]) -> float` (geomean of `fa2_ms[s]/ours_ms[s]` over the 11 bench names).
  - `min_ratio(ours_ms, fa2_ms) -> float`.
  - `noise_threshold(scores: list[float]) -> float` = `max(0.005, 2 * stdev/mean)`.
  - `is_improvement(candidate: float, best: float, threshold: float) -> bool` = `candidate > best * (1 + threshold)`.
  - `python -m harness.check` loads `kernel.attention.attention` from the working tree, runs `check_kernel` on bench + held-out shapes, and prints one line per shape (`name OK/FAIL max_abs_err tol [error]`). **It prints no timing.** It appends a block to `runs.log` and exits 0 iff all OK.
  - `python -m harness.bench --ours --out PATH` runs correctness on all 15 shapes. If any fail, it writes `{"correct": false, "correctness": [...]}` and exits 2. Otherwise it times ours on the 11 bench shapes and writes `{"correct": true, "ours_ms": {...}, "ours_stdev_ms": {...}, "correctness": [...]}`.
  - `python -m harness.bench --baselines --out results/baselines.json` times FA2 and SDPA on the 11 shapes and writes `{"timestamp", "fa2_ms", "sdpa_ms", "sdpa_backend": {...}, "fa2_source"}`.
  - `python -m harness.bench --all --out PATH` times all three side by side (used for the final report).
  - Input construction and layout conversion happen before timing. Each timed callable closes over prebuilt tensors.

- [ ] **Step 1: Write the failing tests** `tests/test_score.py` (CPU):

```python
def test_score_geomean():
    fa2 = {s.name: 2.0 for s in BENCH_SHAPES}
    ours = {s.name: 1.0 for s in BENCH_SHAPES}
    assert score(ours, fa2) == pytest.approx(2.0)
    ours[BENCH_SHAPES[0].name] = 4.0
    assert min_ratio(ours, fa2) == pytest.approx(0.5)

def test_noise_threshold_floor_and_scaling():
    assert noise_threshold([1.0, 1.0, 1.0]) == 0.005
    assert noise_threshold([0.9, 1.0, 1.1]) == pytest.approx(0.2, rel=0.01)

def test_is_improvement():
    assert is_improvement(1.02, 1.0, 0.01)
    assert not is_improvement(1.005, 1.0, 0.01)
```

and `tests/test_bench.py` (GPU, subprocess):

```python
@pytest.mark.gpu
def test_check_cli_prints_no_timing(tmp_path):
    out = subprocess.run([sys.executable,"-m","harness.check"],capture_output=True,text=True)
    assert out.returncode == 0
    assert "ms" not in out.stdout.lower()

@pytest.mark.gpu
def test_bench_ours_and_baselines(tmp_path):
    subprocess.run([sys.executable,"-m","harness.bench","--ours","--out",tmp_path/"o.json"],check=True)
    o = json.loads((tmp_path/"o.json").read_text())
    assert o["correct"] and set(o["ours_ms"]) == {s.name for s in BENCH_SHAPES}
    assert all(v > 0 for v in o["ours_ms"].values())
```

- [ ] **Step 2:** Run `uv run pytest tests/test_score.py tests/test_bench.py -v`. Expected: FAIL.
- [ ] **Step 3:** Implement the four modules per the Interfaces.
- [ ] **Step 4:** Run the tests. Expected: PASS. Then run `uv run python -m harness.bench --baselines --out results/baselines.json` once and eyeball that FA2 beats SDPA on decode, as in prior art. Record anomalies in the commit message.
- [ ] **Step 5:** Commit with message `feat: timing, scoring, check and bench CLIs`.

### Task 6: Profiling diagnostic on the HEAD kernel

**Files:**
- Create: `harness/profile.py`, `tests/test_profile.py`

**Interfaces:**
- Produces: `python -m harness.profile --shape NAME`. It exports `git show HEAD:kernel/attention.py` to a temp module (so it **profiles the last kept kernel, never the working tree**). It runs `ncu --set full` limited to the attention kernel launch on that shape, prints the key metrics (achieved occupancy, SM throughput %, DRAM throughput %, L2 hit rate, registers per thread, shared memory per block), and appends them to `runs.log`. It prints no end-to-end latency.

- [ ] **Step 1:** Check that `ncu` exists (`/usr/local/cuda-12.8/bin/ncu`) and can read counters as this user. If you get `ERR_NVGPUCTRPERM`, stop and tell the user: enabling counters needs a root modprobe option they must choose to set. Until then, make `harness.profile` print "profiling unavailable" and exit 0, and mark this task partially done.
- [ ] **Step 2: Write the failing test**:

```python
@pytest.mark.gpu
def test_profile_uses_head_kernel(tmp_path, monkeypatch):
    out = subprocess.run([sys.executable,"-m","harness.profile","--shape","decode_8192"],capture_output=True,text=True)
    assert out.returncode == 0
    assert "HEAD" in out.stdout and " ms" not in out.stdout
```

- [ ] **Step 3:** Run it. Expected: FAIL. **Step 4:** Implement. **Step 5:** Run it. Expected: PASS.
- [ ] **Step 6:** Commit with message `feat: ncu profiling of last kept kernel`.

### Task 7: Driver accounting — clock, Claude stream parsing, token ledger

**Files:**
- Create: `driver/__init__.py`, `driver/clock.py`, `driver/claude_stream.py`, `driver/ledger.py`, `tests/test_clock.py`, `tests/test_claude_stream.py`, `tests/fixtures/stream_ok.jsonl`, `tests/fixtures/stream_limit.jsonl`

**Interfaces:**
- Produces:
  - `RunClock(state_path: Path, now: Callable[[], float] = time.time)` with `start()`, `pause(reason: str)`, `resume()`, `heartbeat()`, `active_seconds() -> float`, `wall_seconds() -> float`, `is_paused -> bool`. Every call persists state as JSON. On load, if the last heartbeat is older than 120 s, the gap (last heartbeat → now) is recorded as paused time (reason `"driver_offline"`).
  - `@dataclass Usage(input_tokens:int=0, output_tokens:int=0, cache_read_tokens:int=0, cache_write_tokens:int=0)` with `__add__` and `total`.
  - `@dataclass ClaudeOutcome(status: Literal["ok","usage_limit","error","timeout"], usage: Usage, session_id: str | None, result_text: str, reset_at: float | None, duration_s: float)`.
  - `run_claude(prompt: str, cwd: Path, timeout_s: int, claude_bin: str = os.environ.get("CLAUDE_BIN","claude")) -> ClaudeOutcome`. Command: `[claude_bin, "-p", prompt, "--model", "claude-opus-5-5", "--effort", "high", "--output-format", "stream-json", "--verbose", "--permission-mode", "acceptEdits", "--allowedTools", *AGENT_TOOLS]`.
  - `AGENT_TOOLS = ["Read","Edit","Write","Glob","Grep","Bash(uv run python -m harness.check:*)","Bash(uv run python -m harness.profile:*)","Bash(git log:*)","Bash(git diff:*)","Bash(git show:*)"]`.
  - `parse_stream(lines: Iterable[str]) -> ClaudeOutcome` (pure, testable). Take `session_id` from the `system/init` event. Usage comes from the final `result` event's `usage` when present. Otherwise (timeout or crash) sum the `message.usage` of each distinct assistant `message.id`. Classify `usage_limit` when the result event has `is_error` and its text, or any `api_error_status == 429`, matches `/usage limit|rate limit|429|limit reached/i`. Parse a reset epoch if the text contains `|<digits>` or "resets at" with a parseable time. Non-JSON lines are ignored.
  - `TokenLedger(path: Path)` with `add(exp_id: int, u: Usage, session_id: str | None)`, `totals() -> Usage`, `session_ids() -> set[str]`, persisted as JSON.

- [ ] **Step 1:** Capture real fixtures. Run `claude -p "say ok" --model claude-opus-5-5 --effort high --output-format stream-json --verbose > tests/fixtures/stream_ok.jsonl`. Hand-write `stream_limit.jsonl` from that file's schema, with a `result` event where `is_error: true` and text `"Claude AI usage limit reached|1762480800"`. Before relying on the field names, check them against the real fixture.
- [ ] **Step 2: Write the failing tests**:

```python
def test_clock_excludes_pauses(tmp_path):
    t = [1000.0]; c = RunClock(tmp_path/"s.json", now=lambda: t[0]); c.start()
    t[0] += 100; c.pause("usage_limit"); t[0] += 50; c.resume(); t[0] += 10
    assert c.active_seconds() == 110 and c.wall_seconds() == 160

def test_clock_counts_offline_gap_as_paused(tmp_path):
    t = [0.0]; c = RunClock(tmp_path/"s.json", now=lambda: t[0]); c.start()
    t[0] = 60; c.heartbeat(); t[0] = 4000
    c2 = RunClock(tmp_path/"s.json", now=lambda: t[0])
    assert c2.active_seconds() == 60 and c2.wall_seconds() == 4000

def test_parse_ok_fixture():
    o = parse_stream(open("tests/fixtures/stream_ok.jsonl"))
    assert o.status == "ok" and o.session_id and o.usage.output_tokens > 0

def test_parse_usage_limit():
    o = parse_stream(open("tests/fixtures/stream_limit.jsonl"))
    assert o.status == "usage_limit" and o.reset_at == 1762480800

def test_parse_truncated_stream_sums_assistant_usage():
    lines = [l for l in open("tests/fixtures/stream_ok.jsonl") if '"type":"result"' not in l.replace(" ","")]
    o = parse_stream(lines)
    assert o.usage.output_tokens > 0
```

- [ ] **Step 3:** Run `uv run pytest tests/test_clock.py tests/test_claude_stream.py -v`. Expected: FAIL.
- [ ] **Step 4:** Implement the modules. `run_claude` streams stdout line by line, enforces `timeout_s` by killing the process group (status `timeout`, usage from the lines read so far), and maps a non-zero exit without a usage limit to `error`.
- [ ] **Step 5:** Run the tests. Expected: PASS.
- [ ] **Step 6:** Commit with message `feat: driver clock, claude stream parsing, token ledger`.

### Task 8: Guardrails, results log, and the main loop

**Files:**
- Create: `driver/guard.py`, `driver/results.py`, `driver/loop.py`, `tests/fake_claude.py`, `tests/test_loop.py`

**Interfaces:**
- Consumes: Tasks 5 and 7.
- Produces:
  - `harness_hash(root: Path) -> str` (sha256 over sorted `harness/**/*.py`, `program.md`, `pyproject.toml`, `uv.lock`).
  - `changed_files(root) -> set[str]` (`git status --porcelain`, including untracked).
  - `ALLOWED_AGENT_FILES = {"kernel/attention.py", "notes.md", "experiments/current.json"}`.
  - `check_agent_changes(root, head_before: str) -> str | None`. Returns a violation reason if HEAD moved, if any changed file is outside the allowed set, or if `kernel/attention.py` contains `inline_asm` or imports anything outside `torch`, `triton`, `math`, `typing`. Returns None if clean.
  - `revert_to(root, commit: str)` (`git reset --hard commit` + `git clean -fd kernel experiments`).
  - `RESULTS_COLUMNS = ["exp_id","utc","hypothesis","status","score","best_score","min_vs_fa2", *[f"ms_{s.name}" for s in BENCH_SHAPES], "input_tokens","output_tokens","cache_read_tokens","cache_write_tokens","agent_seconds","session_id","commit"]`.
  - `append_result(path: Path, row: dict)` (TSV; writes the header if the file is new; tabs and newlines in the hypothesis are replaced with spaces).
  - Status values: `kept | reverted | failed_correctness | crashed | violation | timeout | agent_error | baseline` (exp 0).
  - `python -m driver.loop [--max-experiments N] [--active-hours 12] [--wall-hours 36] [--agent-timeout-min 20]`.

**Loop algorithm** (decided here; the implementer follows it exactly):

```
startup:
  load or create RunClock(results/run_state.json), TokenLedger(results/tokens.json)
  if fresh: record harness_hash; bench baselines -> results/baselines.json;
            bench seed kernel 3x -> threshold = noise_threshold(scores);
            exp 0 row (status baseline); best = median seed score; best_commit = HEAD
  else: restore best, best_commit, threshold, next exp_id from run_state; revert_to(best_commit)
loop:
  clock.heartbeat()
  stop if active >= 12h, wall >= 36h, max-experiments reached,
       or min_vs_fa2(best) >= 1.0
  if harness_hash changed: abort loudly (human tampering or bug)
  if now - baselines.timestamp >= 3h: re-bench baselines (log drift % per shape to progress.log)
  outcome = run_claude(PROMPT.format(exp_id), timeout=20m)
  ledger.add(exp_id, outcome.usage, outcome.session_id)
  usage_limit -> revert_to(best_commit); clock.pause("usage_limit");
                 sleep until reset_at + 60s, or back off 5,10,20,30,30… min if unknown;
                 clock.resume(); retry the same exp_id (tokens still logged on a row with status agent_error, hypothesis "usage limit hit")
  error -> row agent_error; revert; after 3 consecutive -> pause/back-off like usage_limit
  timeout -> row timeout; revert
  ok -> v = check_agent_changes(); if v: row violation (note v); revert; continue
        run `harness.bench --ours` in a subprocess (timeout 15m)
          timeout or non-JSON or nonzero exit other than 2 -> crashed; revert
          exit 2 -> failed_correctness; revert
          s = score(...); if is_improvement(s,best,threshold): re-run bench once;
              keep only if the second score also is_improvement -> commit kernel+notes+current.json
              "exp {id}: {hypothesis} (score {s:.3f})", best = mean of the two runs, best_commit = HEAD -> kept
          else -> reverted (keep notes.md: restore kernel only, then commit notes in "exp {id} log")
  append_result; write progress.log line: exp, status, score, best, active h, wall h, tokens so far
  persist run_state
```

`PROMPT` = `"You are running experiment {exp_id} of an autoresearch loop. Read program.md and follow it exactly. Write your hypothesis to experiments/current.json before editing the kernel."`

- [ ] **Step 1:** Write `tests/fake_claude.py`. It emits stream-json mimicking the fixtures, and its behavior is chosen by env `FAKE_MODE` ∈ {`improve`, `break`, `touch_harness`, `limit_once`, `error`, `hang`}. `improve` swaps `BLOCK_N=64→128` in the kernel (or any change that's correct). `break` multiplies the output by 1.01. `touch_harness` appends a comment to `harness/score.py`. `limit_once` emits a usage limit the first time (a marker file tracks this) and then `improve`. `hang` sleeps forever.
- [ ] **Step 2: Write the failing tests** `tests/test_loop.py`. Each runs the loop in a temp git clone of the repo with `CLAUDE_BIN=tests/fake_claude.py`, `--max-experiments 1`, a tiny agent timeout where relevant, and a monkeypatched sleep and clock. Bench is stubbed through an env hook `HARNESS_BENCH_CMD` so CPU tests can supply canned JSON. Add one GPU-marked test with the real bench.

```python
def test_violation_resets(loop_env):    run(loop_env, "touch_harness"); assert last_row()["status"]=="violation"; assert clean_tree()
def test_failed_correctness(loop_env):  run(loop_env, "break", bench_exit=2); assert last_row()["status"]=="failed_correctness"
def test_kept_needs_two_wins(loop_env): run(loop_env, "improve", bench_scores=[1.2,1.0]); assert last_row()["status"]=="reverted"
def test_kept(loop_env):                run(loop_env, "improve", bench_scores=[1.2,1.2]); assert last_row()["status"]=="kept"; assert head_msg().startswith("exp 1:")
def test_usage_limit_pauses(loop_env):  run(loop_env, "limit_once", max_exp=1); st=state(); assert st["paused_seconds"]>0; assert last_row()["status"]=="kept"
def test_timeout(loop_env):             run(loop_env, "hang", agent_timeout_s=2); assert last_row()["status"]=="timeout"
def test_bench_crash(loop_env):         run(loop_env, "improve", bench_crash=True); assert last_row()["status"]=="crashed"
def test_resume_after_kill(loop_env):   run(loop_env, "improve", bench_scores=[1.2,1.2]); dirty_kernel(); run(loop_env,"improve",bench_scores=[1.0,1.0]); assert exp_ids()==[0,1,2]
def test_three_errors_back_off(loop_env): run(loop_env, "error", max_exp=3); assert state()["paused_seconds"]>0
```

- [ ] **Step 3:** Run `uv run pytest tests/test_loop.py -v`. Expected: FAIL.
- [ ] **Step 4:** Implement `guard.py`, `results.py`, `loop.py` per the Interfaces and algorithm.
- [ ] **Step 5:** Run the tests. Expected: PASS. Then run the full suite: `uv run pytest -v`. Expected: all PASS.
- [ ] **Step 6:** Commit with message `feat: autoresearch driver loop with guardrails and pause handling`.

### Task 9: `program.md`, notes, and a real one-experiment smoke run

**Files:**
- Create: `program.md`, `notes.md`, `experiments/current.json`, `run_loop.sh`

**Interfaces:**
- Consumes: everything above. `program.md` is part of `harness_hash`, so it is locked once the loop starts.

- [ ] **Step 1:** Write `program.md` with these sections:
  - **Goal:** beat FA2 on every row; the score is the geomean of fa2/ours.
  - **Hardware facts:** pasted from `results/device.json` (170 SMs, measured L2, smem limits, sm_120 has no wgmma/tcgen05).
  - **The one-idea rule:** exactly one hypothesis per experiment, which can be a single coherent change or a config sweep of one parameter family.
  - **Workflow:**
    1. Read `results.tsv`, `notes.md`, and `git log --oneline -20`.
    2. Optionally run `uv run python -m harness.profile --shape NAME` on the last kept kernel.
    3. Write `experiments/current.json` as `{"exp_id", "hypothesis", "rationale", "expected_effect"}`.
    4. Edit `kernel/attention.py`.
    5. Run `uv run python -m harness.check` until it passes, or give up and restore the kernel with `git diff`/your own edits.
    6. Append 2–5 lines to `notes.md` on what you tried and why.
    7. Stop.
  - **Rules:**
    - Edit only the three allowed files.
    - No timing; you cannot benchmark.
    - No hard-coding per benchmark shape (shape-keyed autotune and a prefill/decode dispatch are fine).
    - Triton + Python only, with no inline asm.
    - Keep the `attention(q,k,v,causal)` signature and FA2 layout.
    - Don't touch git state.
  - **Research strategy seed:** the 7 items from the spec, verbatim.
  - **Using the log:** read `results.tsv` statuses. Don't retry a reverted idea unchanged; explain what's different.
- [ ] **Step 2:** Create `notes.md` with a heading and the line "Experiment 0: untuned seed kernel." Create `experiments/current.json` as `{}`. Write `run_loop.sh`, which runs `uv run python -m driver.loop "$@" 2>&1 | tee -a progress.log` under `nohup`.
- [ ] **Step 3:** Smoke run with the real Claude: `./run_loop.sh --max-experiments 1`. Expected: `results.tsv` has a row 0 (`baseline`) and a row 1 with a real hypothesis, non-zero tokens, and a status from the allowed set. `git log` shows either an `exp 1:` commit or an `exp 1 log` commit. `runs.log` shows the agent's check runs.
- [ ] **Step 4:** Read the smoke run's session transcript for permission denials or confusion, fix `program.md` or `AGENT_TOOLS`, and repeat until clean. Then reset results (`results.tsv`, `results/run_state.json`, `results/tokens.json`, `runs.log`, `progress.log`) so the real run starts fresh. Keep the smoke tokens and add them to the build-session total.
- [ ] **Step 5:** Commit with message `feat: program.md and loop launch script`.

### Task 10: Reporting scripts and README skeleton

**Files:**
- Create: `scripts/token_report.py`, `scripts/final_report.py`, `scripts/plot_progress.py`, `README.md`, `tests/test_reports.py`

**Interfaces:**
- Produces:
  - `scripts/token_report.py` reads all transcripts under `~/.claude/projects/-home-fchow-Documents-kernels/` (`*.jsonl` and subagent transcripts). It dedupes assistant messages by `message.id` and splits usage into **loop** (session ids in `results/tokens.json`) and **build** (everything else). It also reports the build span from the first to the last transcript timestamp, and writes `results/token_report.json`.
  - `scripts/final_report.py` runs `harness.bench --all --out results/final.json`, then rewrites the README between `<!-- RESULTS:START -->` and `<!-- RESULTS:END -->` with:
    - the table in the exact column format `| Workload | seq_kv | ours (ms) | FA2 (ms) | SDPA (ms) | vs FA2 | vs SDPA |` (ms to 3 decimals, ratios as `1.03×`, bold when ≥ 1.00)
    - a headline line with the min–max vs-FA2 and vs-SDPA ranges
    - the SDPA backend note
    - an accounting block: loop tokens by category and total, build tokens, active time, wall time, experiments total/kept/reverted/failed_correctness/other
  - `scripts/plot_progress.py` writes `results/progress.png`: score per experiment as grey dots, kept experiments as a highlighted running-best step line, and a y=1.0 "FA2 parity" reference line. Load the `dataviz` skill before writing this chart code.
- [ ] **Step 1: Write the failing tests** (CPU, using fixtures):

```python
def test_table_format(tmp_path):
    md = render_table(sample_final_json())
    assert md.splitlines()[0] == "| Workload | seq_kv | ours (ms) | FA2 (ms) | SDPA (ms) | vs FA2 | vs SDPA |"
    assert "| Prefill | 2048 |" in md and "| Decode | 65536 |" in md

def test_token_split(tmp_path):
    r = split_tokens(fixture_transcripts(tmp_path), loop_sessions={"loop-1"})
    assert r["loop"]["output_tokens"] == 7 and r["build"]["output_tokens"] == 5   # duplicates by message.id counted once
```

- [ ] **Step 2:** Run the tests. Expected: FAIL. **Step 3:** Implement the scripts. **Step 4:** Run the tests. Expected: PASS.
- [ ] **Step 5:** Write the README skeleton:
  - what this is (agent-driven kernel optimization)
  - hardware facts
  - how the loop works, including a diagram of driver → claude → harness → keep/revert
  - the results markers
  - the progress plot
  - "What worked / what didn't" (filled in Task 11)
  - prior art with all three citations
  - reproduce commands
  - next steps: FP8 K/V, TMA on sm_120 if Triton exposes it, warp specialization, per-shape autotune caching
- [ ] **Step 6:** Commit with message `feat: reporting scripts and README skeleton`.

### Task 11: The 12-hour run, final report, publish

**Files:**
- Modify: `README.md`, `results.tsv`, `results/*`

- [ ] **Step 1:** Make sure the GPU is otherwise idle: `nvidia-smi --query-compute-apps=pid,name --format=csv` shows nothing. Then start with `./run_loop.sh`. Tell the user they can follow progress with `tail -f progress.log`.
- [ ] **Step 2:** The run ends on its own (12 h active, 36 h wall, or every row ≥ 1.0× FA2). Confirm the stop reason in `progress.log`.
- [ ] **Step 3:** Run `uv run python scripts/final_report.py`, `scripts/plot_progress.py`, and `scripts/token_report.py`. Expected: the README table is filled in and `results/progress.png` exists.
- [ ] **Step 4:** Fill in "What worked / what didn't" from `results.tsv` and `notes.md`, with each claim tied to an experiment id. Run `uv run pytest -v` one last time. Expected: all PASS.
- [ ] **Step 5:** Commit with message `docs: final results`. **Ask the user before** creating the public GitHub repo and pushing, since publishing is outward-facing.
