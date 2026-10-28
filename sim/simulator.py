"""One-pass online discrete-event simulator of a multi-GPU KV serving system.

Implements the 2x2 factorial from README section 1:
    routing  in {rr, lpm}     (naive load-balanced vs reuse-aware)
    eviction in {lru, score}  (default vs reuse-aware placement)

Cells:
    B0 = rr    + lru     (baseline: per-GPU block-level LRU radix cache,
                          naive routing -- the status-quo SGLang/vLLM setup)
    B1 = lpm   + lru     (routing-only)
    B2 = rr    + score   (placement-only)
    S  = lpm   + score   (full system)
    S_oracle = simulate_oracle_global()
        A single *shared* cache of the total budget (n_gpus * pool_blocks)
        with full future knowledge and Belady (MIN-VF) eviction. A shared
        cache with perfect knowledge and no routing penalty is an upper bound
        on every per-GPU partitioning at the same total budget, so its reuse
        ratio bounds all cells from above (verified empirically across many
        workloads; Belady's theorem is exact for total faults, and "keep the
        soonest next use" preferentially retains leading-prefix blocks, which
        are referenced first when their request arrives).

Eviction policies
-----------------
lru:
    Block-level LRU (status quo). Oldest block evicted first. Optimal for
    homogeneous inter-arrival rates (recency IS the signal), which is why it
    is a strong baseline: it cannot make O(1) "family is dead" errors.

score (the reuse-aware placement of the hypothesis, v5):
    Prefix-family-granular, NOW-ANCHORED expected-gap eviction, with a
    COMPOSITE eviction key that combines the rate signal and the liveness
    signal (see the recency_c bullet below).
    Key workload fact: in multi-turn serving every future request for a
    family references ALL of its blocks so far (context only grows), so the
    placement decision is purely per-family -- which family's next turn is
    closest -- and within a family only the LEADING blocks matter for hits.
    - Per (GPU, family): a deque of arrival times within `window` s (the
      observed inter-turn gaps), the last arrival; plus a per-GPU EMA of
      observed inter-arrivals ("gap prior") for families with < 2
      observations in the window.
    - gap(f) = (last - first_in_window)/(n_in_window - 1) when >= 2
      observations, else the gap prior.
      Expected return = NOW + gap_eff(f): the Poisson hazard is memoryless, so
      the correct conditional expectation of the next turn is anchored at
      decision time. (Anchoring at last use -- v1/v2 below -- puts families
      that merely had one long gap at a return time in the PAST, inverting
      the eviction order; a brand-new family also has the MOST remaining
      turns, not the fewest.)
    - EMPIRICAL-BAYES SHRINKAGE of the per-family gap toward the
      population prior (mean per-family gap, EMA-updated), with equivalent
      prior sample size shrink_k (default 5):
          lambda_eff = (n/gap_raw + k/prior) / (n + k),  gap_eff = 1/lambda_eff
      This is the fix that makes the policy span the popularity range:
      at low Zipf alpha all rates are ~equal, so a family's raw gap estimate
      is pure noise and the policy must be NEUTRAL among live families (the
      shrunken gap ~= the common prior, and the heap tie-break then acts as
      recency, matching LRU's optimal behavior); at high alpha the hot head
      has many observations (n >> k) and keeps its full signal. Measured
      without shrinkage (k=0): the noisy raw-gap ranking cost ~0.1-0.15
      reuse vs LRU at alpha <= 1.0.
    - COMPOSITE KEY, K(f) = gap_eff(f) + rec_c x silence(f) (default
      rec_c = 1; rec_c = 0 reproduces v4). Why both terms:
      * gap_eff is the RATE signal -- among live families it ranks by how
        soon each is expected back (memoryless hazard: for a LIVE family the
        expected return is now + gap_eff, independent of how long it has
        been quiet, so recency adds no rate information);
      * silence is the LIVENESS signal -- for a Poisson process, being quiet
        is evidence of death, and the hard death test below only acts at
        silence >= death_k x gap_eff. Before that cliff, a dead family is
        ranked by its STALE gap_eff, which can put a corpse ahead of live
        families for the eviction slot (the main way v4 lost to LRU at
        alpha <= 1.0, where the pool is crowded with corpses whose
        recycling LRU does instantly via recency). Adding rec_c x silence
        makes corpses sink continuously, without waiting for the cliff.
      * Net effect: at low Zipf alpha (homogeneous rates) gap_eff ~= prior
        for everyone, the key is dominated by silence, and the policy
        degenerates to recency -- LRU's optimal behavior there; at high
        alpha the hot head stays recently used AND small-gapped (kept),
        the cold tail is both slow and quiet (evicted first).
      The key is implemented as (gap_eff - rec_c x last_use) in the heap:
      the common +rec_c x now term is order-invariant, and last_use only
      changes on arrival, so the HEAP_TOL validity check keeps working.
    - dead: silence >= death_k x gap_eff (default death_k = 3) -> gap = INF
      (evicted first). The silence tolerance is PACE-RELATIVE: a family's
      own estimated gap sets how long a silent stretch is evidence of death
      (a uniform ~5% false-death rate, exp(-death_k), for live families
      regardless of pace). A fixed wall-clock window is wrong: a family with
      30 s turns must be declared dead within ~90 s of dying, not after 360
      s of looking indistinguishable from an active family. With rec_c > 0
      the cliff matters less (near-dead families are already ranked near
      the top by the silence term), but the INF mark still guarantees
      corpses rank above finite-key families.
    - On pool overflow, evict one block from the family with the LARGEST
      composite key, taking that family's HIGHEST-position block first, so a
      partially-resident family keeps its leading prefix (a hit requires
      contiguity from block 0).

score_v1 (ablation, kept for the record -- see bench/policy_ablate.py):
    Last-anchored expected-next with windowed count: en = last +
    min(elapsed, window)/count (count >= 2), en = last + window for
    count == 1, INF for an empty window. Measured ~0.14 reuse WORSE than
    LRU on the smoke grid (B2: 0.571 vs 0.712); failure modes (i) the
    count==1 rule protected brand-new families, and (ii) last-anchoring
    misranked quiet families. Superseded by `score`.

Modeling assumptions (kept simple on purpose, identical for all cells):
  - each GPU runs a FIFO queue; a request occupies the GPU from prefill start
    to last decode token (no continuous batching, no preemption);
  - the routing decision at arrival time uses the *current* resident block
    set (slightly stale cache view; same staleness for every cell);
  - a prefill hit is a contiguous leading prefix (paged prefix-cache
    semantics): the first cache-miss block stops the match;
  - blocks of a prompt are (re)inserted into the destination GPU's cache at
    the request's arrival time (approximation of insertion at prefill start);
  - latency model: prefill of b blocks = b * PREBLOCK_S, decode of o tokens =
    o * DECTOK_S. Calibrate both against a real engine before trusting wall
    numbers; the compute-side metrics (reuse ratio) are model-free.
"""
import bisect
import collections
import heapq
import itertools

BLOCK_TOKENS = 16
PREBLOCK_S = 0.0016   # s per 16-token block: ~100 us/token batch-1 prefill
DECTOK_S = 0.025      # s per decode token: ~40 tok/s batch-1
INF = float("inf")
MIN_ELAPSED = 1e-3    # s, guard for the v1 mean-gap estimate
PRIOR_BETA = 0.1      # EMA weight for the per-family-mean gap prior
DEFAULT_SHRINK_K = 5.0  # equivalent prior observations (0 = no shrinkage)
HEAP_TOL = 1e-2       # relative validity tolerance for score heap entries


def _leading_hits(blocks, resident):
    hits = 0
    for b in blocks:
        if b in resident:
            hits += 1
        else:
            break
    return hits


def _pct(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    k = min(len(xs) - 1, int(round(q * (len(xs) - 1))))
    return xs[k]


def _fam_of(b, unique, cum_len):
    """Family of a block: the family whose contiguous span contains it."""
    return bisect.bisect_right(cum_len, b)


class _Gpu:
    __slots__ = ("i", "pool", "eviction", "window", "death_k", "shrink_k",
                 "rec_c", "gap_clip", "resident", "lru_order", "score_heap",
                 "fam_first", "fam_deques", "fam_last", "fam_blocks",
                 "fam_tail", "g_prior", "last_end", "completions", "req_count",
                 "occ_sum", "n_occ")

    def __init__(self, i, pool, eviction, window, death_k=3.0,
                 shrink_k=DEFAULT_SHRINK_K, rec_c=1.0, gap_clip=False):
        self.i = i
        self.pool = pool
        self.eviction = eviction
        self.window = window
        self.death_k = death_k
        self.shrink_k = shrink_k
        self.rec_c = rec_c
        self.gap_clip = gap_clip
        self.resident = {}            # block -> None (membership only)
        self.lru_order = collections.OrderedDict()
        # score-family heaps (see _evict for entry shapes):
        #   v5 ("score"):    (-(gap_eff - rec_c*last_use), t_push, fam)
        #                     == -(composite key) up to an order-invariant
        #                     constant (see module docstring)
        #   v1 ("score_v1"): (-en, last_p, cnt_p, first, fam)
        # Min-heap => top = family with the LARGEST composite key (INF gap
        # = dead family) = the one to evict.
        self.score_heap = []
        self.fam_first = {}           # family -> first-seen time (v1 + debug)
        self.fam_deques = {}          # family -> deque of arrivals within window
        self.fam_last = {}            # family -> last arrival time
        self.fam_blocks = {}          # family -> set of resident block ids
        # max-heap (of -block-id) of each family's resident blocks, with
        # LAZY DELETION (stale tops are dropped in _evict). Eviction always
        # takes the family's highest-position block, so the heap top IS the
        # answer in amortized O(log n) -- a bare max() over the set would
        # re-scan O(family size) per eviction, which is quadratic-ish once a
        # hot family dies with thousands of resident blocks (the alpha=2.5
        # tight-budget path).
        self.fam_tail = {}            # family -> list heap of -block ids
        self.g_prior = window / 2.0   # prior: mean PER-FAMILY gap (EMA)
        self.last_end = 0.0
        self.completions = collections.deque()  # end times, for queue length
        self.req_count = 0
        self.occ_sum = 0.0
        self.n_occ = 0

    # ---------- v1 (ablation) signal ----------
    def _expected_next_v1(self, f, now, count):
        """Last-anchored expected next reuse (see module docstring)."""
        if count == 0:
            return INF
        last = self.fam_last[f]
        if count == 1:
            return last + self.window
        elapsed = max(now - self.fam_first[f], MIN_ELAPSED)
        return last + min(elapsed, self.window) / count

    # ---------- v3 (score) signal ----------
    def _gap_v3(self, f, now):
        """Effective inter-turn gap of family f at time `now` (v3/v4).

        Dead families (silence >= death_k x their own effective gap) get
        INF. The raw per-family gap is shrunk toward the per-family-mean
        population prior (empirical Bayes, equivalent sample size
        shrink_k): families with few in-window observations have an
        estimate that is mostly noise, and shrinking keeps them neutral
        (which is the right behavior when rates are homogeneous).
        """
        dq = self.fam_deques.get(f)
        n = max(len(dq) - 1, 0) if dq else 0
        if n >= 1:
            gap_raw = (dq[-1] - dq[0]) / n
        else:
            gap_raw = self.g_prior
        if self.gap_clip:
            gap_raw = min(max(gap_raw, self.window / 10.0), self.window)
        if self.shrink_k > 0:
            lam_eff = (n / gap_raw + self.shrink_k / self.g_prior) \
                / (n + self.shrink_k)
            gap_eff = 1.0 / lam_eff
        else:
            gap_eff = gap_raw
        if now - self.fam_last[f] >= self.death_k * gap_eff:
            return INF
        return gap_eff

    def _note_arrival(self, f, t):
        """Record a request for family f at time t (score-policy signal)."""
        if f not in self.fam_first:
            self.fam_first[f] = t
        dq = self.fam_deques.setdefault(f, collections.deque())
        while dq and dq[0] < t - self.window:
            dq.popleft()
        dq.append(t)
        if len(dq) >= 2:
            # update the prior from this family's current gap estimate
            # (weighted per FAMILY, not per arrival, so it targets the
            # mean per-family rate -- the right shrinkage target)
            gap_obs = (dq[-1] - dq[0]) / (len(dq) - 1)
            self.g_prior = (1.0 - PRIOR_BETA) * self.g_prior \
                + PRIOR_BETA * gap_obs
        self.fam_last[f] = t
        self.fam_blocks.setdefault(f, set())
        if self.eviction == "score_v1":
            heapq.heappush(self.score_heap,
                           (-self._expected_next_v1(f, t, len(dq)), t, len(dq),
                            self.fam_first[f], f))
        else:
            # composite key: gap_eff + rec_c * silence; the common
            # rec_c * t term is order-invariant, so store the offset form
            # (gap_eff - rec_c * last_use), which only changes on arrival
            # and on prior drift (keeps the HEAP_TOL validity check cheap)
            heapq.heappush(self.score_heap,
                           (-(self._gap_v3(f, t) -
                              self.rec_c * self.fam_last[f]), t, f))

    # ---------- eviction ----------
    def _finish_evict(self, f, reheap):
        """Post-evict bookkeeping: drop all state if the family is gone,
        otherwise keep a live heap entry (the valid pop consumed the only
        one)."""
        blocks = self.fam_blocks.get(f)
        if not blocks:
            self.fam_first.pop(f, None)
            self.fam_deques.pop(f, None)
            self.fam_last.pop(f, None)
            self.fam_blocks.pop(f, None)
            self.fam_tail.pop(f, None)
        else:
            heapq.heappush(self.score_heap, reheap)

    def _evict(self, now, unique, cum_len):
        """Evict one block per the active policy."""
        if self.eviction == "lru":
            b, _ = self.lru_order.popitem(last=False)
            del self.resident[b]
            self.fam_blocks.get(_fam_of(b, unique, cum_len), set()).discard(b)
            return

        while self.score_heap:
            entry = heapq.heappop(self.score_heap)
            if self.eviction == "score_v1":
                neg_en, last_p, cnt_p, first, f = entry
            else:
                neg_k, t_push, f = entry
            blocks = self.fam_blocks.get(f)
            if not blocks:
                continue  # fully evicted -> stale entry
            dq = self.fam_deques[f]
            while dq and dq[0] < now - self.window:
                dq.popleft()
            if self.eviction == "score_v1":
                cnt = len(dq)
                if self.fam_last[f] != last_p or cnt != cnt_p:
                    heapq.heappush(self.score_heap,
                                   (-self._expected_next_v1(f, now, cnt),
                                    self.fam_last[f], cnt, first, f))
                    continue
                key = (-self._expected_next_v1(f, now, cnt),
                       self.fam_last[f], cnt, first)
            else:
                k_now = self._gap_v3(f, now) - self.rec_c * self.fam_last[f]
                # tolerance-based validity: last_use only changes on
                # arrival and gap_eff only drifts with the prior, so a
                # stale entry is detected by a key move beyond the
                # tolerance (a hard equality would force an
                # O(n_families) re-push churn per eviction). The offset
                # form is immune to the common rec_c x now growth of the
                # composite key. (Dead flip: INF - finite -> INF move,
                # re-push; two INFs -> NaN > tol is False, entry stays
                # valid, key = -INF as intended.)
                if abs(k_now - (-neg_k)) > HEAP_TOL * max(1.0, abs(k_now),
                                                         abs(-neg_k)):
                    heapq.heappush(self.score_heap, (-k_now, now, f))
                    continue
                key = (-k_now, now)
            # highest-position block first: least useful for leading hits
            # (lazy-delete the family's tail heap: stale entries are blocks
            # already evicted from this family by an earlier pass)
            tail = self.fam_tail[f]
            while True:
                b = -heapq.heappop(tail)
                if b in blocks:
                    break
            del self.resident[b]
            self.lru_order.pop(b, None)
            blocks.discard(b)
            self._finish_evict(f, key + (f,))
            return
        # fallback (should not happen: each resident family has a live entry)
        b = next(iter(self.resident))
        del self.resident[b]
        self.lru_order.pop(b, None)
        self.fam_blocks.get(_fam_of(b, unique, cum_len), set()).discard(b)


def simulate(trace, n_gpus, pool_blocks, routing="rr", eviction="lru",
             escape_k=0.0, window=None, death_k=3.0, shrink_k=DEFAULT_SHRINK_K,
             rec_c=1.0, gap_clip=False, cell=""):
    """Run one cell. pool_blocks = KV pool size PER GPU (the fixed budget).

    routing in {"rr", "lpm"}; eviction in {"lru", "score", "score_v1"}.
    escape_k: for lpm routing, if the overlap-best GPU's queue exceeds
    escape_k * mean queue (and escape_k > 0), fall back to the min-queue GPU
    (load-balance escape hatch, README section 5 ablation).
    window: recency window (s) for the score policies' gap estimate.
            Default: horizon / 10.
    death_k: v5 death threshold, in units of the family's own effective gap
             (silence >= death_k x gap => dead; default 3.0).
    shrink_k: v5 empirical-Bayes shrinkage, equivalent prior observations
              (default 5.0; 0 = raw un-shrunken gap, the pure v3 arm).
    rec_c: v5 composite-key weight on the silence (liveness) term:
           K = gap_eff + rec_c x silence (default 1.0; 0.0 = v4, gap only).
    gap_clip: clip the v5 per-family raw gap to [window/10, window].
    """
    unique = trace.unique_prefix_blocks
    cum_len = list(itertools.accumulate(trace.family_lengths))
    window = window if window is not None else trace.horizon / 10.0
    gpus = [_Gpu(i, pool_blocks, eviction, window, death_k=death_k,
                 shrink_k=shrink_k, rec_c=rec_c, gap_clip=gap_clip)
            for i in range(n_gpus)]
    rr_idx = 0
    prompt_blocks = hit_blocks = 0
    ttfts = []

    for req in trace.requests:
        t = req.t
        for g in gpus:
            while g.completions and g.completions[0] <= t:
                g.completions.popleft()

        # ---- routing decision (at arrival, current-state view) ----
        if routing == "rr":
            qlens = {g.i: len(g.completions) for g in gpus}
            m = min(qlens.values())
            cands = [g for g in gpus if qlens[g.i] == m]
            g = cands[rr_idx % len(cands)]
            rr_idx += 1
        else:  # lpm
            best = max(gpus, key=lambda g: (_leading_hits(req.blocks, g.resident),
                                            -len(g.completions)))
            if escape_k > 0:
                mean_q = sum(len(x.completions) for x in gpus) / n_gpus
                if len(best.completions) > escape_k * max(mean_q, 1.0):
                    best = min(gpus, key=lambda g: len(g.completions))
            g = best
        hits = _leading_hits(req.blocks, g.resident)

        # ---- schedule on the chosen GPU (FIFO single-server) ----
        prefill_blocks = req.prompt_blocks - hits
        service = prefill_blocks * PREBLOCK_S + req.out_tokens * DECTOK_S
        start = max(t, g.last_end)
        end = start + service
        g.last_end = end
        g.completions.append(end)
        g.req_count += 1

        # ---- family-level reuse signal (once per request) ----
        g._note_arrival(req.family, t)

        # ---- cache insertions (approximated at arrival time) ----
        for b in req.blocks:
            if b not in g.resident:
                g.resident[b] = None
                g.lru_order[b] = None
                fb = _fam_of(b, unique, cum_len)
                # the family's signal state may have been dropped mid-request
                # (pool smaller than the family: its blocks evicted as this
                # same request inserts them) -- recreate it before adding the
                # block so fam_blocks and the signal state stay in lockstep
                if fb not in g.fam_first:
                    g._note_arrival(fb, t)
                g.fam_blocks.setdefault(fb, set()).add(b)
                heapq.heappush(g.fam_tail.setdefault(fb, []), -b)
            else:
                g.lru_order.move_to_end(b)
            while len(g.resident) > pool_blocks:
                g._evict(t, unique, cum_len)

        prompt_blocks += req.prompt_blocks
        hit_blocks += hits
        g.occ_sum += len(g.resident)
        g.n_occ += 1
        ttfts.append(start + prefill_blocks * PREBLOCK_S - t)

    req_counts = [g.req_count for g in gpus]
    mean_c = sum(req_counts) / n_gpus
    return {
        "cell": cell,
        "n_gpus": n_gpus,
        "pool_blocks": pool_blocks,
        "routing": routing,
        "eviction": eviction,
        "prompt_blocks": prompt_blocks,
        "hit_blocks": hit_blocks,
        "reuse_ratio": hit_blocks / prompt_blocks if prompt_blocks else 0.0,
        "prefill_blocks": prompt_blocks - hit_blocks,
        "ttft_p50": _pct(ttfts, 0.50),
        "ttft_p95": _pct(ttfts, 0.95),
        "ttft_p99": _pct(ttfts, 0.99),
        "load_max_over_mean": max(req_counts) / mean_c if mean_c else 0.0,
        "avg_occupancy": (sum(g.occ_sum for g in gpus) /
                          sum(g.n_occ for g in gpus)) / pool_blocks,
        "makespan": max(g.last_end for g in gpus),
    }


def simulate_oracle_global(trace, total_pool, n_gpus=1, pool_blocks=None,
                           cell="S_oracle"):
    """Clairvoyant upper bound: ONE shared cache of `total_pool` blocks, full
    future knowledge, Belady (MIN-VF: evict the farthest-future use) eviction.

    This bounds every per-GPU cell from above: a shared cache holds the union
    of all GPUs' caches at the same total budget, has perfect knowledge, and
    pays no routing penalty, so no partitioned system can exceed its reuse.

    Reference stream: for each request (in order), each of its blocks. A
    request's *leading* blocks present in the cache at its arrival are counted
    as hits (paged prefix-cache semantics), matching the other cells.
    """
    from collections import defaultdict
    # build the flat reference stream + per-block reference positions
    ref_list = []
    per_block = defaultdict(list)
    start_pos = []
    pos = 0
    for req in trace.requests:
        start_pos.append(pos)
        for b in req.blocks:
            per_block[b].append(pos)
            ref_list.append(b)
            pos += 1

    cache = set()
    cur_idx = {}  # block -> index (within per_block[b]) of NEXT unprocessed ref

    def next_ref(b):
        lst = per_block[b]
        i = cur_idx.get(b, 0)
        return lst[i] if i < len(lst) else INF

    heap = []  # entries (-next_ref_pos, block); heappop => farthest use first
    hit_total = 0
    prompt_total = 0
    for ri, req in enumerate(trace.requests):
        # leading hits at request arrival (before this request's insertions)
        h = 0
        for b in req.blocks:
            if b in cache:
                h += 1
            else:
                break
        hit_total += h
        prompt_total += len(req.blocks)
        # consume this request's references, in order
        s = start_pos[ri]
        for k in range(len(req.blocks)):
            b = req.blocks[k]
            lst = per_block[b]
            j = cur_idx.get(b, 0)
            assert lst[j] == s + k, "reference-stream misalignment"
            cur_idx[b] = j + 1          # advance past the reference just used
            if b not in cache:
                if len(cache) >= total_pool:
                    # Belady: evict the cached block with the farthest next use
                    while heap:
                        negp, eb = heapq.heappop(heap)
                        if eb in cache and -negp == next_ref(eb):
                            cache.discard(eb)
                            break
                cache.add(b)
            heapq.heappush(heap, (-next_ref(b), b))
    reuse = hit_total / prompt_total if prompt_total else 0.0
    return {
        "cell": cell,
        "n_gpus": n_gpus,
        "pool_blocks": pool_blocks,
        "routing": "oracle",
        "eviction": "belady-global",
        "prompt_blocks": prompt_total,
        "hit_blocks": hit_total,
        "reuse_ratio": reuse,
        "prefill_blocks": prompt_total - hit_total,
        "ttft_p50": float("nan"),
        "ttft_p95": float("nan"),
        "ttft_p99": float("nan"),
        "load_max_over_mean": float("nan"),
        "avg_occupancy": float("nan"),
        "makespan": float("nan"),
    }
