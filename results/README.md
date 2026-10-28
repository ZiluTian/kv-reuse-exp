# Results

One subdirectory per sweep or run: `results/<date>_<config-slug>/` containing
raw output, per-config metrics (recompute-token ratio, TTFT p50/p95/p99, peak
KV HBM, hit-tier breakdown), and the derived tables.

Pareto frontiers (prefill savings vs. TTFT p95 per memory budget) and the
workload-characterization map live here as final artifacts.
