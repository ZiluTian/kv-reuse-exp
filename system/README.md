# system/ — policy plugins

Thin plugins on a single engine base (SGLang RadixAttention or vLLM
prefix-caching paged blocks) so the only delta between factorial cells is the
policy logic:

- routing: naive (round-robin / shortest-queue) vs. reuse-aware
  (max-prefix-overlap / LPM), with optional load-balance escape hatch
  (deviate when overlap-best queue > k× mean)
- placement: default local LRU vs. reuse-aware hot-block scoring with
  HBM/DRAM tiering; LRU/LFU/score eviction ablations
