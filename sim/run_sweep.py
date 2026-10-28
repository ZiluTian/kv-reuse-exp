#!/usr/bin/env python3
"""Run the 2x2 factorial sweep (README sections 1, 2, 7) and log results.

For each workload config (alpha, lifetime, seed) the trace is generated once
and shared across all GPU counts, budget ratios and policy cells, so cell
comparisons are on identical workloads.

Outputs:
    <out>/sweep.csv     one row per (config x cell)
    <out>/summary.md    cross-cell comparison (mean reuse ratio, S vs B0 gap,
                        load-balance and TTFT cost)

Example (fast smoke):
    python3 sim/run_sweep.py --n-requests 400 --n-families 60 \
        --gpus 2,4 --alpha 1.0 --lifetime 1.0 --budget 0.25 \
        --out results/smoke
"""
import argparse
import csv
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from workloads.tracegen import generate as generate_trace  # noqa: E402
from sim.simulator import simulate, simulate_oracle_global  # noqa: E402

CELLS = {
    "B0": dict(routing="rr",  eviction="lru"),
    "B1": dict(routing="lpm", eviction="lru"),
    "B2": dict(routing="rr",  eviction="score"),
    "S":  dict(routing="lpm", eviction="score"),
}
ORACLE = "S_oracle"  # global Belady upper bound (see simulator docstring)


def run_cell(tr, cell, n_gpus, pool, escape_k=0.0, window=None):
    """Run one policy cell on trace tr (pool = per-GPU block budget).

    escape_k / window only affect the lpm routing and score eviction
    respectively; the oracle ignores both.
    """
    if cell == ORACLE:
        return simulate_oracle_global(tr, n_gpus * pool, n_gpus=n_gpus,
                                      pool_blocks=pool)
    return simulate(tr, n_gpus, pool, escape_k=escape_k, window=window,
                    **CELLS[cell], cell=cell)

CSV_FIELDS = ["seed", "alpha", "lifetime", "n_gpus", "budget_ratio",
              "n_requests_actual", "unique_prefix_blocks",
              "cell", "routing", "eviction", "pool_blocks",
              "reuse_ratio", "prefill_blocks", "prompt_blocks",
              "ttft_p50", "ttft_p95", "ttft_p99", "load_max_over_mean",
              "avg_occupancy", "makespan"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-requests", type=int, default=5000)
    p.add_argument("--n-families", type=int, default=200)
    p.add_argument("--horizon", type=float, default=3600.0)
    p.add_argument("--seeds", default="0")
    p.add_argument("--gpus", default="1,2,4")
    p.add_argument("--alpha", default="0.3,1.0,2.5")
    p.add_argument("--lifetime", default="0.5,1.0,3.0")
    p.add_argument("--budget", default="0.2,0.5,1.0")
    p.add_argument("--cells", default=",".join(CELLS) + "," + ORACLE)
    p.add_argument("--escape-k", type=float, default=0.0,
                   help="lpm load-balance escape hatch: k x mean queue "
                        "(0 = off; ablation, README section 5)")
    p.add_argument("--window", type=float, default=None,
                   help="score-policy recency window in s (default "
                        "horizon/10; ablation, README section 5)")
    p.add_argument("--out", default=None)
    a = p.parse_args()

    seeds = [int(x) for x in a.seeds.split(",")]
    gpus = [int(x) for x in a.gpus.split(",")]
    alphas = [float(x) for x in a.alpha.split(",")]
    lifes = [float(x) for x in a.lifetime.split(",")]
    budgets = [float(x) for x in a.budget.split(",")]
    cells = a.cells.split(",")
    unknown = [c for c in cells if c not in CELLS and c != ORACLE]
    if unknown:
        p.error(f"unknown cells {unknown}; choose from {list(CELLS) + [ORACLE]}")

    out = a.out or f"results/sweep-{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(out, exist_ok=True)

    traces = {}
    rows = []
    t0 = time.time()
    n_configs = len(seeds) * len(alphas) * len(lifes) * len(gpus) * len(budgets)
    n_runs = n_configs * len(cells)
    done = 0
    for seed in seeds:
        for alpha in alphas:
            for life in lifes:
                tr = generate_trace(seed=seed, n_requests=a.n_requests,
                                    n_families=a.n_families, alpha=alpha,
                                    horizon=a.horizon, lifetime_scale=life)
                traces[(seed, alpha, life)] = tr
                for ng in gpus:
                    for bud in budgets:
                        pool = max(16, int(bud * tr.unique_prefix_blocks / ng))
                        for cell in cells:
                            r = run_cell(tr, cell, ng, pool,
                                         escape_k=a.escape_k,
                                         window=a.window)
                            rows.append({
                                "seed": seed, "alpha": alpha, "lifetime": life,
                                "n_gpus": ng, "budget_ratio": bud,
                                "n_requests_actual": len(tr.requests),
                                "unique_prefix_blocks":
                                    tr.unique_prefix_blocks,
                                **{k: r[k] for k in (
                                    "cell", "routing", "eviction",
                                    "pool_blocks", "reuse_ratio",
                                    "prefill_blocks", "prompt_blocks",
                                    "ttft_p50", "ttft_p95", "ttft_p99",
                                    "load_max_over_mean", "avg_occupancy",
                                    "makespan")},
                            })
                            done += 1
                            if done % max(1, n_runs // 10) == 0:
                                el = time.time() - t0
                                print(f"  ... {done}/{n_runs} runs "
                                      f"({el:.0f}s elapsed)", file=sys.stderr)

    csv_path = os.path.join(out, "sweep.csv")
    with open(csv_path, "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        wcsv.writeheader()
        for r in rows:
            wcsv.writerow({k: r[k] for k in CSV_FIELDS})

    # ---- summary: cross-cell comparison ----
    agg = {}
    for r in rows:
        k = r["cell"]
        d = agg.setdefault(k, {"n": 0, "rr": 0.0, "t95": 0.0, "lb": 0.0})
        d["n"] += 1
        d["rr"] += r["reuse_ratio"]
        d["t95"] += r["ttft_p95"]
        d["lb"] += r["load_max_over_mean"]
    cell_label = {}
    for r in rows:
        cell_label[r["cell"]] = (r["routing"], r["eviction"])
    lines = [f"# Sweep summary ({time.strftime('%Y-%m-%d %H:%M:%S')})", "",
             f"- {n_runs} runs = {n_configs} workload configs "
             f"(seeds x alpha x lifetime x n_gpus x budget) x {len(cells)} "
             f"cells; n_requests={a.n_requests}, "
             f"n_families={a.n_families}, horizon={a.horizon}s",
             f"- CSV: `{csv_path}`", "",
             "| cell | routing | eviction | mean reuse ratio | "
             "mean TTFT p95 (s) | mean load max/mean |",
             "|---|---|---|---|---|---|"]
    for c in cells:
        d = agg[c]
        routing, eviction = cell_label.get(c, ("?", "?"))
        t95 = d["t95"] / d["n"]
        lb = d["lb"] / d["n"]
        t95_s = "n/a" if t95 != t95 else f"{t95:.3f}"  # NaN (oracle) check
        lb_s = "n/a" if lb != lb else f"{lb:.2f}"
        lines.append(f"| {c} | {routing} | {eviction} "
                     f"| {d['rr']/d['n']:.4f} | {t95_s} | {lb_s} |")
    if "S" in agg and "B0" in agg:
        gain = (agg["S"]["rr"] - agg["B0"]["rr"]) / agg["B0"]["n"]
        ttft_cost = (agg["S"]["t95"] - agg["B0"]["t95"]) / agg["B0"]["n"]
        oracle = ""
        if "S_oracle" in agg:
            oracle_gain = (agg["S_oracle"]["rr"] - agg["B0"]["rr"]) / agg["B0"]["n"]
            gap = (agg["S_oracle"]["rr"] - agg["S"]["rr"]) / agg["S"]["n"]
            share = (gain / oracle_gain) if oracle_gain > 0 else float("nan")
            oracle = (f"- oracle gap (S_oracle - S, mean reuse): {gap:.4f}; "
                      f"share of oracle gain captured by S: {share:.2%}")
        b0_t95 = agg["B0"]["t95"] / agg["B0"]["n"]
        gain_ok = gain >= 0.20
        ttft_ok = ttft_cost <= 0.15 * b0_t95
        verdict = "PASS" if (gain_ok and ttft_ok) else "FAIL"
        why = []
        if not gain_ok:
            why.append(f"gain {gain:+.4f} < +0.20")
        if not ttft_ok:
            why.append(f"TTFT p95 cost {ttft_cost:+.3f} s > 15% of B0 "
                       f"({0.15 * b0_t95:.3f} s)")
        lines += ["", f"- **S vs B0 (primary test, README section 4):** mean "
                     f"reuse gain {gain:+.4f} (fraction of prompt blocks), "
                     f"mean TTFT p95 cost {ttft_cost:+.3f} s.",
                  f"- pre-registered threshold: gain >= +0.20 AND TTFT p95 "
                  f"cost <= 15% of B0 -> **{verdict}**"
                  + (f" ({'; '.join(why)})" if why else "")
                  + " (see threshold tuning, README section 9)."]
        if oracle:
            lines.append(oracle)
    summary_path = os.path.join(out, "summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"wrote {csv_path}\nwrote {summary_path}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
