#!/usr/bin/env python3
"""Fast policy ablation: eviction policies vs LRU on a compact config grid.

Goal: iterate on the score (reuse-aware placement) policy until it at least
matches LRU, before spending the full-sweep budget.

Grid (seed 0, n_requests=2000, n_families=100):
    alpha {0.3, 1.0, 2.5} x lifetime {0.5, 1.0, 3.0}
    x n_gpus {1, 2, 4} x budget {0.2, 0.5, 1.0}   =  81 configs
Cells (all rr routing, to isolate the EVICTION policy -- the placement arm):
    lru (B0) | score_v1 (last-anchored, ablation) | score (v5 composite key)

Outputs a per-config table, per-regime means, and win counts. v5 knobs:
    --death-k F   death threshold, in units of the family's own gap (default 3.0)
    --shrink-k F  empirical-Bayes shrinkage: equivalent prior observations
                  (default 5.0; 0 = un-shrunken raw gap, the pure v3 arm)
    --rec-c F     composite-key silence weight, K = gap_eff + rec_c x silence
                  (default 1.0; 0.0 = v4, gap only)
    --clip        clip the per-family raw gap to [window/10, window]

Example:
    python3 bench/policy_ablate.py
    python3 bench/policy_ablate.py --death-k 4.0 --rec-c 0 --clip
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from workloads.tracegen import generate as generate_trace  # noqa: E402
from sim.simulator import simulate  # noqa: E402

CELLS = [
    ("lru", dict()),
    ("score_v1", dict()),
    ("score", dict()),
]
ALPHAS = [0.3, 1.0, 2.5]
LIFES = [0.5, 1.0, 3.0]
GPUS = [1, 2, 4]
BUDGETS = [0.2, 0.5, 1.0]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-requests", type=int, default=2000)
    p.add_argument("--n-families", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--death-k", type=float, default=3.0,
                   help="v3 death threshold, in units of the family's own "
                        "effective gap (default 3.0)")
    p.add_argument("--shrink-k", type=float, default=5.0,
                   help="v5 empirical-Bayes shrinkage: equivalent prior "
                        "observations (default 5.0; 0 = un-shrunken)")
    p.add_argument("--rec-c", type=float, default=1.0,
                   help="v5 composite-key weight on the silence (liveness) "
                        "term, K = gap_eff + rec_c x silence "
                        "(default 1.0; 0.0 = v4, gap only)")
    p.add_argument("--clip", action="store_true",
                   help="v3: clip per-family raw gap to [window/10, window]")
    p.add_argument("--out", default=None, help="optional CSV dump")
    a = p.parse_args()

    t0 = time.time()
    rows = []
    traces = {}
    for alpha in ALPHAS:
        for life in LIFES:
            tr = generate_trace(seed=a.seed, n_requests=a.n_requests,
                                n_families=a.n_families, alpha=alpha,
                                lifetime_scale=life)
            traces[(alpha, life)] = tr
            for ng in GPUS:
                for bud in BUDGETS:
                    pool = max(16, int(bud * tr.unique_prefix_blocks / ng))
                    for cell, kw in CELLS:
                        kw2 = dict(kw)
                        if cell == "score":
                            kw2["death_k"] = a.death_k
                            kw2["shrink_k"] = a.shrink_k
                            kw2["rec_c"] = a.rec_c
                            kw2["gap_clip"] = a.clip
                        r = simulate(tr, ng, pool, routing="rr", eviction=cell,
                                     **kw2, cell=cell)
                        rows.append(dict(alpha=alpha, lifetime=life,
                                         n_gpus=ng, budget=bud, cell=cell,
                                         reuse=r["reuse_ratio"],
                                         t95=r["ttft_p95"]))

    # per-config table
    combos = {}
    for r in rows:
        combos.setdefault((r["alpha"], r["lifetime"], r["n_gpus"],
                           r["budget"]), {})[r["cell"]] = r["reuse"]
    print(f"{'alpha':>5} {'life':>4} {'ng':>2} {'bud':>4} | "
          f"{'lru':>6} {'v1':>6} {'v3':>6} | {'v3-lru':>7} {'v1-lru':>7}")
    wins3 = wins1 = tot = 0
    for k in sorted(combos):
        v = combos[k]
        lru, v1, v3 = v["lru"], v["score_v1"], v["score"]
        tot += 1
        wins3 += v3 >= lru
        wins1 += v1 >= lru
        mark = "  <-- v3 loses" if v3 < lru else ""
        print(f"{k[0]:>5} {k[1]:>4} {k[2]:>2} {k[3]:>4} | "
              f"{lru:6.3f} {v1:6.3f} {v3:6.3f} | "
              f"{v3 - lru:+7.3f} {v1 - lru:+7.3f}{mark}")

    # per-regime means
    print("\nper-regime means (lru / v3):")
    for alpha in ALPHAS:
        for life in LIFES:
            sel = [r for r in rows if r["alpha"] == alpha
                   and r["lifetime"] == life]
            m = {c: sum(r["reuse"] for r in sel if r["cell"] == c) /
                 sum(1 for r in sel if r["cell"] == c) for c, _ in CELLS}
            print(f"  alpha={alpha:>4} life={life:>4}: "
                  f"lru {m['lru']:.3f} | v1 {m['score_v1']:.3f} | "
                  f"v3 {m['score']:.3f}  (v3-lru {m['score'] - m['lru']:+.3f})")
    print(f"\nv3 >= lru in {wins3}/{tot} configs; "
          f"v1 >= lru in {wins1}/{tot} configs")
    print(f"({time.time() - t0:.0f}s)")

    if a.out:
        import csv
        with open(a.out, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["alpha", "lifetime", "n_gpus",
                                              "budget", "cell", "reuse",
                                              "t95"])
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
