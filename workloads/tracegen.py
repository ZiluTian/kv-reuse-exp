"""Synthetic LLM-serving trace generator for the kv-reuse-exp experiment.

Workload model (v2, multi-turn conversations)
---------------------------------------------
A "family" is a CONVERSATION (an agent session / multi-turn chat):
  - a fixed prefix P_f blocks (the system prompt / shared RAG context),
  - a sequence of turns; turn t appends K_t fresh blocks (the user's new
    content) and its prompt is the WHOLE accumulated context:
        prompt_t = prefix + turn_1 + ... + turn_t
    so the prompt strictly extends the previous turn's prompt (leading-prefix
    reuse, exactly like real multi-turn serving).
  - turn arrivals follow a Poisson process (exponential inter-turn gaps)
    over the conversation's lifetime window; turns landing past the horizon
    are dropped (the conversation is truncated).
  - popularity: Zipf(alpha) over conversations (expected turn counts).

This models the workloads of the hypothesis: long-lived shared prefixes
(system prompts, RAG docs, agent state) plus per-conversation context that
must be retained while the conversation is active and is dead weight once it
finishes. "When is retaining state worth its memory cost?" is answered by
how long a conversation's context must survive between turns (the turn gap)
versus how much memory it occupies meanwhile.

Knobs map 1:1 to the workload matrix in README section 2.
"""
import random
from dataclasses import dataclass


@dataclass
class Request:
    t: float          # arrival time (s)
    family: int       # conversation id
    blocks: tuple     # ordered block ids = the full prompt (prefix + turns)
    out_tokens: int   # decode length

    @property
    def prompt_blocks(self) -> int:
        return len(self.blocks)


@dataclass
class Trace:
    requests: list
    n_families: int
    family_lengths: list            # total block span per family
    unique_prefix_blocks: int       # total distinct blocks
    horizon: float
    family_times: list              # per family: sorted turn arrival times
    prefix_lengths: list            # prefix (system-prompt) blocks per family


def generate(seed=0,
             n_requests=5000,
             n_families=200,
             alpha=1.0,
             horizon=3600.0,
             prefix_len_range=(64, 512),    # blocks: 1K .. 8K tokens (agent/RAG)
             lifetime_scale=1.0,
             turn_len_range=(1, 8),        # fresh blocks per turn (16..128 tok)
             out_range=(16, 64)):
    """Generate a trace. See README section 2 for the knob semantics."""
    rng = random.Random(seed)

    # expected turn counts from Zipf popularity
    w = [1.0 / (i + 1) ** alpha for i in range(n_families)]
    total_w = sum(w)
    counts = [max(1, int(round(n_requests * wi / total_w))) for wi in w]
    while sum(counts) != n_requests:
        i = rng.randrange(n_families)
        if sum(counts) > n_requests:
            if counts[i] > 1:
                counts[i] -= 1
        else:
            counts[i] += 1

    prefix_lengths = [rng.randint(*prefix_len_range) for _ in range(n_families)]

    reqs = []
    family_lengths = []
    family_times = []
    block_id = 0
    for i in range(n_families):
        P = prefix_lengths[i]
        n_turns = counts[i]
        # conversation lifetime (active window), then turn rate
        base = min(1.0, max(0.05, rng.lognormvariate(-1.0, 0.6)))  # med ~0.37
        life = min(horizon, lifetime_scale * base * horizon)
        start = rng.uniform(0.0, max(0.0, horizon - 1.0))
        lam = n_turns / max(life, 1e-9)
        # Poisson turn arrivals, truncated at the horizon
        t = start
        turns = []
        while len(turns) < n_turns and t < horizon:
            t += rng.expovariate(lam)
            if t >= horizon:
                break
            turns.append(t)
        if not turns:  # ensure every conversation has at least one turn
            turns = [min(start, horizon - 1e-6)]
        # block layout: prefix, then per-turn fresh blocks
        turn_blocks = [rng.randint(*turn_len_range) for _ in turns]
        total = P + sum(turn_blocks)
        base_id = block_id
        block_id += total
        family_lengths.append(total)
        ctx = P  # blocks generated so far (after the prefix)
        for (tt, K) in zip(turns, turn_blocks):
            ctx += K
            reqs.append(Request(t=tt, family=i,
                                blocks=tuple(range(base_id, base_id + ctx)),
                                out_tokens=rng.randint(*out_range)))
        family_times.append(turns)

    reqs.sort(key=lambda r: r.t)
    return Trace(requests=reqs,
                 n_families=n_families,
                 family_lengths=family_lengths,
                 unique_prefix_blocks=block_id,
                 horizon=horizon,
                 family_times=family_times,
                 prefix_lengths=prefix_lengths)
