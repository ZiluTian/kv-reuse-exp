# Sweep summary (2026-10-28 21:59:37)

- 405 runs = 81 workload configs (seeds x alpha x lifetime x n_gpus x budget) x 5 cells; n_requests=2000, n_families=100, horizon=3600.0s
- CSV: `/tmp/kv-smoke-v2/sweep.csv`

| cell | routing | eviction | mean reuse ratio | mean TTFT p95 (s) | mean load max/mean |
|---|---|---|---|---|---|
| B0 | rr | lru | 0.7119 | 209.743 | 1.01 |
| B1 | lpm | lru | 0.7981 | 259.028 | 1.65 |
| B2 | rr | score | 0.5710 | 123.936 | 1.02 |
| S | lpm | score | 0.6943 | 393.763 | 1.63 |
| S_oracle | oracle | belady-global | 0.9310 | n/a | n/a |

- **S vs B0 (primary test, README section 4):** mean reuse gain -0.0176 (fraction of prompt blocks), mean TTFT p95 cost +184.020 s.
- pre-registered threshold: gain >= +0.20 AND TTFT p95 cost <= 15% of B0 -> **FAIL** (gain -0.0176 < +0.20; TTFT p95 cost +184.020 s > 15% of B0 (31.461 s)) (see threshold tuning, README section 9).
- oracle gap (S_oracle - S, mean reuse): 0.2367; share of oracle gain captured by S: -8.04%
