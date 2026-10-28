# Sweep summary (2026-10-28 23:24:57)

- 405 runs = 81 workload configs (seeds x alpha x lifetime x n_gpus x budget) x 5 cells; n_requests=2000, n_families=100, horizon=3600.0s
- CSV: `results/smoke-20261028-score-v5/sweep.csv`

| cell | routing | eviction | mean reuse ratio | mean TTFT p95 (s) | mean load max/mean |
|---|---|---|---|---|---|
| B0 | rr | lru | 0.7119 | 209.743 | 1.01 |
| B1 | lpm | lru | 0.7981 | 259.028 | 1.65 |
| B2 | rr | score | 0.7623 | 94.819 | 1.01 |
| S | lpm | score | 0.8472 | 328.421 | 1.74 |
| S_oracle | oracle | belady-global | 0.9310 | n/a | n/a |

- **S vs B0 (primary test, README section 4):** mean reuse gain +0.1353 (fraction of prompt blocks), mean TTFT p95 cost +118.679 s.
- pre-registered threshold: gain >= +0.20 AND TTFT p95 cost <= 15% of B0 -> **FAIL** (gain +0.1353 < +0.20; TTFT p95 cost +118.679 s > 15% of B0 (31.461 s)) (see threshold tuning, README section 9).
- oracle gap (S_oracle - S, mean reuse): 0.0838; share of oracle gain captured by S: 61.76%
