# bench/ — benchmark scripts

Launchers for the pre-registered experiments. Both are pure-CPU (the
simulator is Python; "GPUs" are simulated FIFO servers) and can be run
from anywhere — they `cd` to the repo root themselves.

| Script | What it runs | Runs | Notes |
|---|---|---|---|
| `full_sweep.sh` | Pre-registered 2×2 factorial + oracle: 5 seeds × α {0.3, 1.0, 2.5} × lifetime {0.5, 1.0, 3.0} × GPUs {1, 2, 4, 8} × budget {0.2, 0.5, 1.0} × cells {B0, B1, B2, S, S_oracle} | 2700 | The primary evidence for the hypothesis (README §4) |
| `ablations.sh` | (A) LPM escape-k ∈ {1, 2, 4} on cells B1, S; (B) score recency window ∈ {180, 720} s on cells B2, S — same workload grid | 3 × 900 + 2 × 900 = 4500 | README §5; baseline points (k=0, window=360 s) come from the full sweep |

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
