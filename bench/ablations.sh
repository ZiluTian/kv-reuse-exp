#!/usr/bin/env bash
# Ablations (README section 5), same workload grid as bench/full_sweep.sh:
#   5 seeds x alpha {0.3, 1.0, 2.5} x lifetime {0.5, 1.0, 3.0}
#   x gpus {1, 2, 4, 8} x budget {0.2, 0.5, 1.0}
#
# A) routing: LPM load-balance escape hatch, k in {1, 2, 4}
#    (k=0 is pure LPM = cell B1 of the full sweep, not re-run here).
#    Cells: B1 (lpm+lru) and S (lpm+score); the rr cells are unaffected.
# B) placement: score-policy recency window in {180, 720} s
#    (360 s = horizon/10 is the default used by the full sweep, not re-run).
#    Cells: B2 (rr+score) and S (lpm+score); the lru cells are unaffected.
set -euo pipefail
cd "$(dirname "$0")/.."
COMMON=(--n-requests 5000 --n-families 200 --horizon 3600
        --seeds 0,1,2,3,4 --gpus 1,2,4,8
        --alpha 0.3,1.0,2.5 --lifetime 0.5,1.0,3.0 --budget 0.2,0.5,1.0)

for K in 1 2 4; do
  python3 sim/run_sweep.py "${COMMON[@]}" \
    --cells B1,S --escape-k "$K" \
    --out "results/ablation-escape-k${K}-$(date +%Y%m%d-%H%M%S)"
  echo "wrote escape-k=$K ablation"
done

for W in 180 720; do
  python3 sim/run_sweep.py "${COMMON[@]}" \
    --cells B2,S --window "$W" \
    --out "results/ablation-window-${W}s-$(date +%Y%m%d-%H%M%S)"
  echo "wrote window=${W}s ablation"
done
