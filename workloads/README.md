# workloads/ — trace generator

Knobs (see README §2):
- prefix structure: shared system prompt (agent) / shared RAG docs / multi-turn
- popularity: Zipf(α), α ∈ {0.3, 1.0, 1.5, 2.5}
- prefix length 1K–32K tokens, per-turn length
- Poisson arrivals at a range of rates
- prefix lifetime (first→last use gap)

Realistic anchors: Chatbot-Arena-style multi-turn traces; long-context agent
workload; Azure/Alibaba-style traces if obtainable.
