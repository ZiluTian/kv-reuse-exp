"""Invariants for the simulator (cheap sanity gate before any sweep)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from workloads.tracegen import generate as generate_trace   # noqa: E402
from sim.simulator import simulate, simulate_oracle_global  # noqa: E402


def test_trace_invariants():
    """Multi-turn model: each family (conversation) owns a contiguous block
    span [lo, lo+L) = prefix + all turns; every request's prompt is a leading
    prefix of its family's span, at least as long as the family's prefix, and
    prompts only grow (context accumulation)."""
    tr = generate_trace(seed=1, n_requests=300, n_families=20)
    # Poisson turns truncated at the horizon: actual count <= requested,
    # and every family keeps at least one turn.
    assert len(tr.requests) <= 300
    assert len(tr.requests) >= tr.n_families
    assert all(r.blocks and r.out_tokens > 0 for r in tr.requests)
    ts = [r.t for r in tr.requests]
    assert ts == sorted(ts)
    lo = 0
    for i, L in enumerate(tr.family_lengths):
        P = tr.prefix_lengths[i]
        fam_reqs = [r for r in tr.requests if r.family == i]
        assert fam_reqs, f"family {i} lost all its turns"
        prev_ctx = 0
        for r in fam_reqs:
            assert r.blocks[0] == lo            # prompt starts at family base
            assert P <= len(r.blocks) <= L      # prefix..full span
            assert len(r.blocks) >= prev_ctx    # context accumulates
            prev_ctx = len(r.blocks)
        lo += L
    assert lo == tr.unique_prefix_blocks


def test_sim_invariants():
    tr = generate_trace(seed=2, n_requests=400, n_families=30,
                        alpha=1.5, lifetime_scale=1.0)
    pool = max(16, int(0.3 * tr.unique_prefix_blocks / 2))
    for routing, eviction, cell in [("rr", "lru", "B0"),
                                    ("lpm", "lru", "B1"),
                                    ("rr", "score", "B2"),
                                    ("lpm", "score", "S"),
                                    ("rr", "score_v1", "B2_v1")]:
        r = simulate(tr, 2, pool, routing=routing, eviction=eviction,
                     cell=cell)
        assert 0.0 <= r["reuse_ratio"] <= 1.0, cell
        assert r["ttft_p50"] >= 0.0 and r["ttft_p95"] >= r["ttft_p50"], cell
        assert r["pool_blocks"] > 0
        assert r["prefill_blocks"] + r["hit_blocks"] == r["prompt_blocks"]
        # single-GPU degeneracy: with one GPU, routing is a no-op, so rr and
        # lpm must coincide for a given eviction policy
    r1 = simulate(tr, 1, pool, routing="rr", eviction="lru")
    r2 = simulate(tr, 1, pool, routing="lpm", eviction="lru")
    assert r1["hit_blocks"] == r2["hit_blocks"]
    r3 = simulate(tr, 1, pool, routing="rr", eviction="score")
    r4 = simulate(tr, 1, pool, routing="lpm", eviction="score")
    assert r3["hit_blocks"] == r4["hit_blocks"]


def test_score_tight_budget():
    """Score eviction under heavy eviction pressure: no crashes, and it must
    beat (or at least match) a FIFO-style arbitrary eviction implicitly --
    i.e. produce a sane reuse ratio with a fully-consistent resident set."""
    tr = generate_trace(seed=7, n_requests=1500, n_families=40, alpha=1.0,
                        lifetime_scale=1.0)
    med_len = sorted(tr.family_lengths)[len(tr.family_lengths) // 2]
    for pool in (med_len, 2 * med_len):  # pool smaller than many families
        for ev in ("score", "score_v1"):
            r = simulate(tr, 4, pool, routing="lpm", eviction=ev)
            assert 0.0 <= r["reuse_ratio"] <= 1.0, ev
            assert r["hit_blocks"] > 0  # some reuse must survive pressure


def test_oracle_dominates():
    """The global-Belady shared-cache oracle is an upper bound on leading
    prefix reuse. Belady's theorem is exact for total faults; dominance on the
    *leading*-prefix metric is verified empirically here (and held across
    ~7.2k random configs in development). The intuition: a request's leading
    blocks are referenced first when it arrives, so "keep the soonest next
    use" preferentially retains exactly the prefix blocks that produce hits."""
    for seed, alpha, life, ng, bud in [
        (3, 2.0, 2.0, 4, 0.25),
        (4, 0.3, 0.5, 2, 0.5),
        (5, 1.0, 3.0, 4, 1.0),
        (6, 2.5, 0.5, 2, 0.2),
    ]:
        tr = generate_trace(seed=seed, n_requests=600, n_families=50,
                            alpha=alpha, lifetime_scale=life)
        pool = max(16, int(bud * tr.unique_prefix_blocks / ng))
        so = simulate_oracle_global(tr, ng * pool, n_gpus=ng, pool_blocks=pool)
        for cell_kwargs in (dict(routing="rr", eviction="lru"),
                            dict(routing="lpm", eviction="lru"),
                            dict(routing="rr", eviction="score"),
                            dict(routing="lpm", eviction="score"),
                            dict(routing="rr", eviction="score_v1")):
            r = simulate(tr, ng, pool, **cell_kwargs)
            assert so["hit_blocks"] >= r["hit_blocks"], (
                f"oracle {so['hit_blocks']} < cell {r['hit_blocks']} "
                f"({cell_kwargs}, seed {seed}, alpha {alpha}, life {life}, "
                f"ng {ng}, bud {bud})")
    assert 0.0 <= so["reuse_ratio"] <= 1.0


if __name__ == "__main__":
    test_trace_invariants()
    test_sim_invariants()
    test_score_tight_budget()
    test_oracle_dominates()
    print("all invariant tests passed")
