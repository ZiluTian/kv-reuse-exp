# sim/ — discrete-event simulator

Trace replay or discrete-event model of the KV-serving system:
- per-block prefill/decode latency model (from real engine calibration)
- block-hash hit accounting (only tokens actually served from cache count)
- memory-budget enforcement (hard cap per GPU pool; report peak usage)
- runs the 2×2 factorial cells (B0/B1/B2/S) against generated traces

Goal: 100+ configs cheaply to find which workload regimes separate the cells.
