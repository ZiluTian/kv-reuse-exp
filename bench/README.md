# bench/ — benchmark scripts

Launchers for the pre-registered experiments. Both are pure-CPU (the
simulator is Python; "GPUs" are simulated FIFO servers) and can be run
from anywhere — they `cd` to the repo root themselves.

| Script | What it runs | Runs | Wall time (1 core) |
|---|---|---|---|
| `full_sweep.sh` | Pre-registered 2×2 factorial + oracle: 5 seeds × α {0.3, 1.0, 2.5} × lifetime {0.5, 1.0, 3.0} × GPUs {1, 2, 4, 8} × budget {0.2, 0.5, 1.0} × cells {B0, B1, B2, S, S_oracle} | 2700 | ~8–12 h |
| `ablations.sh` | (A) LPM escape-k ∈ {1, 2, 4} on cells B1, S; (B) score recency window ∈ {180, 720} s on cells B2, S — same workload grid | 4500 | ~12–20 h |
| `policy_ablate.py` | Eviction-policy iteration loop: lru vs score_v1 vs score on a compact 81-config grid (rr routing only, isolates placement); knobs `--death-k`, `--shrink-k`, `--rec-c`, `--clip` | 243 | ~2.5 min |

(Baseline for the estimates: 405 runs at n_requests=2000 / n_families=100
took 15.5 min on one core; the full grid is 5000 / 200, so per-run cost is
roughly 3–4× higher.)

**Parallelizing:** the sweep is embarrassingly parallel across seeds. Run
one process per seed (disjoint `--seeds`, distinct `--out` dirs), then
`cat results/*/sweep.csv` into one file for analysis — e.g. 5 processes
brings the full sweep to ~2 h.

Each run writes `results/<slug>-<timestamp>/{sweep.csv, summary.md}`;
`summary.md` carries the pre-registered verdict (S vs B0: mean reuse gain
≥ 0.20 AND TTFT p95 cost ≤ 15% of B0) and, for the full sweep, the share of
the oracle gap captured by S.

Quick smoke test before a long run:

```bash
python3 tests/test_invariants.py
python3 sim/run_sweep.py --n-requests 200 --n-families 40 \
    --gpus 1,2 --alpha 1.0 --lifetime 1.0 --budget 0.25 --out results/smoke
```
