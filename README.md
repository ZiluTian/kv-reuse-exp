# Reuse-Aware KV Placement & Routing

**Research question:** When is retaining KV state worth its memory cost?
**Hypothesis (H):** Reuse-aware KV placement and routing can reduce repeated prefill under a fixed GPU-memory budget.

Created: 2026-10-28

## 1. Testable form of the hypothesis

Split H into two independent interventions and name the variables:

- **IV1 (routing):** steer a request to the GPU that already holds the KV for its
  prefix (max-prefix-overlap / LPM routing) vs. naive load-balanced routing
  (round-robin / shortest-queue).
- **IV2 (placement):** decide *which* KV blocks live in *which* GPU's HBM (and
  which tier, HBM/DRAM) based on predicted future reuse, vs. default local
  LRU eviction.
- **DV (dependent variable):** repeated prefill — recompute-token ratio
  (prompt tokens actually prefill-computed ÷ total prompt tokens) or
  prefill FLOPs.
- **CVs (held constant):** total KV memory budget, scheduler/admission policy,
  model, batch policy.

### 2×2 factorial (core of the design)

| | Naive routing (RR) | Reuse-aware routing (LPM/overlap) |
|---|---|---|
| Default placement (local LRU, all HBM) | **B0** baseline | **B1** routing-only |
| Reuse-aware placement (hot-block score, HBM/DRAM tiering) | **B2** placement-only | **S** full system |

- Implement both interventions as thin plugins on a single engine
  (SGLang RadixAttention or vLLM prefix-caching paged blocks are the natural
  base) so the *only* delta between cells is the policy logic.
- If B1 ≈ S, placement adds nothing; if B2 ≈ S, routing adds nothing.
  H holds only if S clearly exceeds all three baselines.
- Baseline strength: B0 must be tuned (LRU with the same budget), and routing
  baselines should include a *prefix-aware* variant (not only round-robin) —
  this is the reviewer attack point given prior art (SGLang RadixAttention,
  vLLM prefix caching, Mooncake, Splitwise/DistServe).

## 2. Workloads

A trace generator with knobs (one fixed workload is not enough — reuse-awareness
only pays off for specific workload shapes). Implemented model
(`workloads/tracegen.py`, v2): **multi-turn conversations** — each family is a
conversation with a fixed prefix (system prompt / RAG context) plus a sequence
of Poisson-arriving turns; turn *t*'s prompt is the whole accumulated context
(prefix + turns 1..t), so prompts strictly extend one another (leading-prefix
reuse, exactly like real multi-turn serving). Conversation lifetimes are
random windows inside the horizon; popularity is Zipf(α) over expected turn
counts. Knobs map 1:1 to the sweep matrix:

| Knob | Meaning / default |
|---|---|
| `n_requests` | expected total turns (Poisson truncation at the horizon means actual ≤ requested; the actual count is logged per trace) |
| `n_families` | conversations, default 200 |
| `alpha` | Zipf popularity, {0.3 ≈ uniform … 2.5 hot head} |
| `horizon` | trace length (s), default 3600 |
| `prefix_len_range` | prefix blocks per conversation, default (64, 512) = 1–8K tokens |
| `lifetime_scale` | conversation lifetime window scale (the knob that decides whether LRU survives) |
| `turn_len_range` | fresh blocks per turn, default (1, 8) = 16–128 tokens |
| `out_range` | decode length per request, default (16, 64) |

Sim-side knobs (`sim/run_sweep.py`): GPU count {1, 2, 4, 8}, per-GPU budget as
a fraction of the total distinct blocks {0.2, 0.5, 1.0}, policy cells,
lpm escape-k (load-balance hatch), score-policy recency window.

Realistic anchors (later): multi-turn chat traces (Chatbot-Arena-style),
long-context agent workloads, Azure/Alibaba-style LLM serving traces if
obtainable ("deployment-realistic" arm).

## 3. Metrics

- **Primary:** recompute-token ratio; total prefill FLOPs. Measures H directly
  and is immune to decode-side noise.
- **Constraints (must not regress):** TTFT p50/p95/p99, ITL, SLO attainment
  (e.g. "≤2× median TTFT"). Reuse-aware routing *can* create hotspots — TTFT
  tails are where H can fail.
- **Resource audit:** peak KV HBM per GPU (proves the budget was held constant),
  hit-tier breakdown (HBM/DRAM/SSD), load balance (max/mean queue length),
  eviction counts, routing-decision overhead.
- **Efficiency metric:** prefill tokens saved **per GB of KV memory** — the fair
  comparison at fixed budget.

## 4. Pre-registered success criteria (decide before running)

1. **S beats B0** by ≥20% recompute-token reduction at equal memory, with TTFT
   p95 within 15% of B0. (Tune thresholds to the platform; freeze them before
   the main sweep.)
2. **Disambiguation claim (pre-registered):** routing dominates at high α /
   short lifetimes; placement dominates at tight budgets. If the factorial
   cells do not separate, that is a result in itself (it identifies which
   intervention is doing the work).

## 5. Oracles and ablations

- **Clairvoyant upper bound:** placement with perfect future-reuse knowledge
  (trivial in simulation). Report the fraction of the oracle gap that S closes
  — quantifies whether the reuse *signal* or the *policy* is the bottleneck.
- **Ablate the signal (implemented):** `score_v1` = past-hit-based,
  windowed count, last-anchored expected-next (the LFU-like arm) vs.
  `score` = predicted inter-turn gap, now-anchored (v3). Driven by
  `bench/policy_ablate.py`; v1 measured worse than LRU on the smoke grid
  and is kept as the documented ablation arm.
- **Ablate routing:** pure LPM vs. LPM with a load-balance escape hatch
  (deviate from the overlap-best GPU when its queue exceeds k× the mean).
  Main practical failure mode; worth its own knob.
- **Ablate placement:** LRU / LFU / score-based eviction; block granularity;
  tier boundaries.

## 6. Threats to validity

1. **Budget not actually fixed** — hard-cap the KV pool; assert and report
   peak usage every run.
2. **Scheduler confound** — identical admission/ordering across all cells
   (FIFO and SJF, reported separately).
3. **Tiering vs. placement conflation** — placement = *which GPU* + *which
   tier*; report the two effects separately so DRAM offloading cannot
   masquerade as reuse-aware placement.
4. **Weak baselines** — tuned LRU at equal budget; ideally also best static
   allocation.
5. **Statistics** — ≥3 seeds, report CIs; measure *compute* (tokens/FLOPs),
   not just wall-clock.
6. **Hit accounting** — define reuse at block-hash level; count only tokens
   actually served from cache, not merely present.

## 7. Execution plan (cheap → expensive)

1. **Simulator sweep first:** discrete-event model (or trace replay through the
   real engine with a latency model) over the full knob space — 100+ configs
   are cheap here. Find which workload regimes separate the cells.
2. **Real-system verification** of the top 3–5 configs + B0 on actual GPUs
   (2× or 8× SGLang node) to prove simulated savings survive real
   scheduling/memory overheads.
3. **Report the Pareto frontier** (prefill savings vs. TTFT p95) per memory
   budget, plus a workload-characterization map: "reuse-awareness pays when
   α > X and working set > budget."

### Falsification path

If S ≈ B0 across the sweep, H is wrong *for those workloads*; the contribution
becomes the characterization of where reuse is too diffuse (low α, short
lifetimes) for KV retention to amortize — a negative result that guides where
to point the system (e.g. prefix compression instead).

## 8. Repository layout

```
kv-reuse-exp/
├── README.md          # this document (design + protocol)
├── sim/               # discrete-event simulator: simulator.py (cells + oracle),
│                      #   run_sweep.py (factorial/ablation sweep runner)
├── workloads/         # trace generator (multi-turn conversation model, knobs above)
├── tests/             # invariant tests (trace invariants, sim invariants,
│                      #   score-policy tight-budget stress, oracle dominance)
├── bench/             # benchmark scripts: pre-registered full sweep + ablations
├── system/            # policy plugins (routing, placement) on top of the engine
└── results/           # raw runs, sweeps, Pareto plots, final tables
```

## 9. Status / how to run

Smoke tests (fast, no GPU needed):

```bash
python3 tests/test_invariants.py     # invariants + oracle dominance
python3 sim/run_sweep.py --n-requests 200 --n-families 40 \
    --gpus 1,2 --alpha 1.0 --lifetime 1.0 --budget 0.25 --out results/smoke
```

Benchmark scripts (see `bench/README.md`): `bench/full_sweep.sh` runs the
pre-registered factorial; `bench/ablations.sh` runs the escape-k and
recency-window ablations; `bench/policy_ablate.py` is the fast
eviction-policy iteration loop.

**Results so far (2026-10-28, smoke grid only — not the pre-registered run):**
- LPM routing beats the default (≈ SGLang-as-shipped) with the stock LRU
  cache: B1 vs B0 = +0.086 mean reuse ratio (artifact:
  `results/smoke-20261028-score-v1/`).
- Score v1 (windowed past-hit, last-anchored) measured ~0.14 WORSE than LRU
  (B2: 0.571 vs 0.712) — failure modes documented in the `sim/simulator.py`
  docstring; superseded by score v3 (now-anchored expected inter-turn gap),
  currently under ablation.
- Clairvoyant oracle on the same grid: 0.931 → +0.22 headroom over B0.
- Known cost: LPM routing pays +24% TTFT p95 and 1.65× load imbalance on the
  smoke grid; the escape-k hatch (`bench/ablations.sh`) is the mitigation.

- [x] Pin the success-criteria thresholds (20% / 15%) — encoded in
      `sim/run_sweep.py` summary verdict (tune after the full sweep).
- [x] Build the trace generator (`workloads/`) — multi-turn Zipf + lifetime.
- [x] Build the simulator (`sim/`) — FIFO single-server per GPU, leading-
      prefix hit accounting, Belady global oracle.
- [ ] Implement B0–B2 and S as engine plugins (`system/`) — real-engine
      verification phase (top configs only, per section 7).
- [ ] Run the factorial sweep; log to `results/`.
