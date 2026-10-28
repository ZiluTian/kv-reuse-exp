#!/usr/bin/env bash
# Pre-registered main factorial sweep (README sections 1, 4, 7).
#
#   5 seeds x alpha {0.3, 1.0, 2.5} x lifetime {0.5, 1.0, 3.0}
#   x gpus {1, 2, 4, 8} x budget {0.2, 0.5, 1.0}
#   x cells {B0, B1, B2, S, S_oracle}  =  2700 runs
#
# Pure-CPU simulator (no GPU access needed). Outputs:
#   results/full-sweep-<timestamp>/{sweep.csv, summary.md}
# summary.md carries the pre-registered verdict (S vs B0: reuse gain >= 0.20
# AND TTFT p95 cost <= 15% of B0) plus the oracle-gap share.
set -euo pipefail
cd "$(dirname "$0")/.."
OUT="results/full-sweep-$(date +%Y%m%d-%H%M%S)"
python3 sim/run_sweep.py \
  --n-requests 5000 \
  --n-families 200 \
  --horizon 3600 \
  --seeds 0,1,2,3,4 \
  --gpus 1,2,4,8 \
  --alpha 0.3,1.0,2.5 \
  --lifetime 0.5,1.0,3.0 \
  --budget 0.2,0.5,1.0 \
  --cells B0,B1,B2,S,S_oracle \
  --out "$OUT"
echo "wrote $OUT"
