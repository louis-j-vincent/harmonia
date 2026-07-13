#!/usr/bin/env python3
"""Mission 5 — does an LLM warm-start help a Bayesian chord solver converge?

WHAT THIS MEASURES (and what it does NOT)
-----------------------------------------
The shipped decoder (``joint_decode.joint_decode``) is an EXACT segment Viterbi —
it has no random init and nothing to "converge", so "LLM init converges faster"
is not literally testable against it (documented honestly in
``docs/mission_5_bayesian_integration.md`` §"the convergence framing"). What
*is* testable, and what the mission's premise actually reduces to, is: when a
Bayesian labeler must solve a coupled factor graph by ITERATION (coordinate
ascent / Gibbs — the shape any per-song *learned* transition matrix or EM
refinement would take), does seeding it with LLM priors reach the right answer
in fewer sweeps and/or at higher final accuracy than an uninformed start?

This script builds that controlled experiment with the iReal chart as ground
truth and a SYNTHETIC emission model. It is a simulation — clearly labelled as
such — because the real end-to-end audio benchmark is gated on Mission 1
(``data/real_audio_benchmark/``, known_issues #20/#28). The emission noise is
not arbitrary: it reproduces the two confusions the corpus study found dominate
real audio — 5th-apart root swaps (#19: 46–51% of root errors) and maj↔dom
quality blur (#19: dom→maj is the failure mode) — so the simulation stresses the
solver where real audio actually hurts.

Two arms, same emission, same graph:
  * UNINFORMED : uniform transition prior, no quality prior, no pooling,
                 init = per-segment emission argmax.
  * LLM-GUIDED : LLM tonic + P(q|root) quality bonus + P(root|prev) transition
                 bias + pooled repeated spans (tied), init = LLM MAP labels.
                 Prior strength scaled by the LLM's self-reported confidence.

Reports, over many noise seeds: sweeps-to-convergence, final root accuracy, and
final (root,q5) accuracy vs the chart.

Usage:
    python scripts/eval_llm_priors.py --song "Autumn Leaves"
    python scripts/eval_llm_priors.py --song "Autumn Leaves" --sigma 1.3 --seeds 200
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.llm_chord_priors import (  # noqa: E402
    Q5_NAMES, load_chart, analyze, to_bayesian_factors, chart_from_tune,
    offline_analyze,
)

N_ROOT, N_Q5 = 12, 5
N_STATE = N_ROOT * N_Q5


def sidx(root: int, q5: int) -> int:
    return root * N_Q5 + q5


def unpack(s: int) -> tuple[int, int]:
    return divmod(s, N_Q5)


def ground_truth_states(chart) -> list[int]:
    """One state per bar: the bar's primary (first) chord, forward-filled."""
    gt: list[int] = []
    last = None
    for _label, chords in chart.sections:
        if chords:
            last = sidx(chords[0][0], Q5_NAMES.index(chords[0][1]))
        gt.append(last if last is not None else sidx(0, 0))
    return gt


def make_emission(gt: list[int], sigma: float, rng: np.random.Generator) -> np.ndarray:
    """(T, 60) log-emission. Truth gets a bump; noise concentrates on the two
    confusions real audio actually makes (known_issues #19)."""
    T = len(gt)
    E = rng.normal(0.0, sigma, size=(T, N_STATE))
    for t, g in enumerate(gt):
        gr, gq = unpack(g)
        E[t, g] += 3.0                                   # true state
        E[t, sidx((gr + 7) % 12, gq)] += 1.6 * sigma     # 5th-up confusion
        E[t, sidx((gr - 7) % 12, gq)] += 1.6 * sigma     # 5th-down (== 4th) confusion
        if gq == 2:                                       # dom often blurs to maj
            E[t, sidx(gr, 0)] += 1.4 * sigma
        if gq == 0:
            E[t, sidx(gr, 2)] += 0.8 * sigma
    return E


def build_priors(factors, T: int):
    """Return (trans[60,60], qbonus[60], pool_parent list) in log space."""
    trans = np.zeros((N_STATE, N_STATE))
    for pr, dist in factors.root_transition_bias.items():
        for nx, nats in dist.items():
            for qp in range(N_Q5):
                for qn in range(N_Q5):
                    trans[sidx(pr, qp), sidx(nx, qn)] += nats
    qbonus = np.zeros(N_STATE)
    for root, dist in factors.quality_bonus.items():
        for q5, nats in dist.items():
            qbonus[sidx(root, q5)] += nats
    return trans, qbonus


def pool_parent_map(factors, chart, T: int) -> list[int]:
    """Map each bar to a 'parent' bar it is tied to (pooled). Bars in the k-th
    slot of parallel spans share a parent (the first span's slot)."""
    parent = list(range(T))
    for spans in factors.pool_group_bars:
        base_s, base_e = spans[0]
        base_len = base_e - base_s + 1
        for (s, e) in spans[1:]:
            if e - s + 1 != base_len:
                continue
            for k in range(base_len):
                child = (s - 1) + k
                root = (base_s - 1) + k
                if child < T and root < T:
                    parent[child] = root
    return parent


def coordinate_ascent(E, trans, qbonus, parent, init, max_sweeps=100):
    """Iterated conditional modes over the segment chain with tied (pooled)
    groups. Returns (labels, sweeps_to_convergence)."""
    T = E.shape[0]
    # group bars by shared parent (pooled tie): pooled emission is summed.
    groups: dict[int, list[int]] = {}
    for t in range(T):
        groups.setdefault(parent[t], []).append(t)
    labels = init.copy()
    for sweep in range(1, max_sweeps + 1):
        changed = False
        for p, members in groups.items():
            # pooled emission over the tied slot(s): √N denoising (known_issues #28)
            emis = np.zeros(N_STATE)
            for t in members:
                emis += E[t]
            score = emis + len(members) * qbonus
            for t in members:
                if t > 0:
                    score = score + trans[labels[t - 1]]
                if t < T - 1:
                    score = score + trans[:, labels[t + 1]]
            best = int(np.argmax(score))
            if any(labels[t] != best for t in members):
                changed = True
                for t in members:
                    labels[t] = best
        if not changed:
            return labels, sweep
    return labels, max_sweeps


def run_arm(E, trans, qbonus, parent, init):
    return coordinate_ascent(E, trans, qbonus, parent, init)


def accuracy(labels, gt):
    root_ok = np.mean([unpack(l)[0] == unpack(g)[0] for l, g in zip(labels, gt)])
    full_ok = np.mean([l == g for l, g in zip(labels, gt)])
    return root_ok, full_ok


# ── Part B1: NON-CIRCULAR cross-source eval ───────────────────────────────────
# The sim above is circular: prior and GT are both the SAME chart, so the
# quality prior is an aggregate of the answer key (audit §2–3). The honest test
# derives the prior from iReal source A and scores it against a DIFFERENT
# lead-sheet B of the same tune. Where A and B DISAGREE (inversions, tritone
# subs, passing chords, key choice) is exactly where an analyst must earn its
# keep — and it cannot cheat, because it never saw B's labels. Expected: a much
# smaller Δ than the circular sim.

def _key_pc(key: str) -> int:
    """Root pitch class of an iReal key string ('D-', 'Bb', 'F#') — for the
    same-key validity filter (transposed 'versions' are different songs or a
    homonym, not a lead-sheet variant of the SAME tune)."""
    from scripts.llm_chord_priors import note_to_pc
    return note_to_pc(key.strip().rstrip("-mM ") or "C") or 0


def _find_cross_source_tunes(playlists: list[Path], min_bars: int = 8,
                             max_len_diff: int = 4, disagree_ceiling: float = 0.5):
    """Group tunes by title across playlists; return the VALID same-tune,
    multi-source pairs plus a census of why the rest are unusable.

    A valid B1 pair is the SAME tune transcribed by two sources with GENUINE
    but bounded chord-level disagreement:
      * same key (a transposed 'version' is a homonym/different arrangement, and
        every root then disagrees — a measurement artifact, not analyst signal);
      * comparable length (|nA − nB| ≤ ``max_len_diff`` — a form/length mismatch
        misaligns the per-bar GT and inflates disagreement spuriously);
      * ``0 < disagree ≤ disagree_ceiling`` — >0 (else trivially circular:
        byte-identical transcription), ≤ceiling (above it the two are different
        compositions sharing a title, e.g. Adele vs Lionel Richie 'Hello').

    Returns (pairs, census) where pairs is a list of
    (title, chartA, chartB, gtA, gtB, T, disagree) and census counts the reject
    reasons."""
    import contextlib
    import io
    from collections import Counter, defaultdict

    from harmonia.data.ireal_corpus import load_playlist

    by_title: dict[str, list] = defaultdict(list)
    for pl in playlists:
        if not pl.exists():
            continue
        with contextlib.redirect_stdout(io.StringIO()):
            tunes = load_playlist(pl)
        for t in tunes:
            by_title[t.title.strip().lower()].append((pl.stem, t))

    out = []
    census: Counter = Counter()
    for title, versions in sorted(by_title.items()):
        seen_src, uniq = set(), []
        for src, t in versions:
            if src not in seen_src:
                seen_src.add(src)
                uniq.append((src, t))
        if len(uniq) < 2:
            continue
        census["multi_source_titles"] += 1
        chartA, chartB = chart_from_tune(uniq[0][1]), chart_from_tune(uniq[1][1])
        gtA, gtB = ground_truth_states(chartA), ground_truth_states(chartB)
        if min(len(gtA), len(gtB)) < min_bars:
            census["too_short"] += 1
            continue
        if _key_pc(chartA.key) != _key_pc(chartB.key):
            census["key_mismatch(homonym/transposed)"] += 1
            continue
        if abs(len(gtA) - len(gtB)) > max_len_diff:
            census["length_mismatch"] += 1
            continue
        T = min(len(gtA), len(gtB))
        gtA, gtB = gtA[:T], gtB[:T]
        disagree = float(np.mean([a != b for a, b in zip(gtA, gtB)]))
        if disagree == 0.0:
            census["identical_transcription"] += 1
            continue
        if disagree > disagree_ceiling:
            census["disagree_too_high(diff_song)"] += 1
            continue
        census["VALID"] += 1
        out.append((title, chartA, chartB, gtA, gtB, T, disagree))
    return out, census


def _delta_full(gt, factors, sigma, seeds):
    """Δ (root,q5) accuracy pp, guided(factors)−uninformed, over `seeds` seeds,
    scoring against `gt`. Guided uses the factors' quality+transition+pooling."""
    T = len(gt)
    trans, qbonus = build_priors(factors, T)
    # pool parent needs a chart-shaped object; reuse pool_group_bars directly.
    parent = list(range(T))
    for spans in factors.pool_group_bars:
        base_s, base_e = spans[0]
        base_len = base_e - base_s + 1
        for (s, e) in spans[1:]:
            if e - s + 1 != base_len:
                continue
            for k in range(base_len):
                child, root = (s - 1) + k, (base_s - 1) + k
                if child < T and root < T:
                    parent[child] = root
    uniform_parent = list(range(T))
    zero_trans, zero_q = np.zeros((N_STATE, N_STATE)), np.zeros(N_STATE)
    u_full, g_full = [], []
    for seed in range(seeds):
        rng = np.random.default_rng(seed)
        E = make_emission(gt, sigma, rng)
        u_lab, _ = run_arm(E, zero_trans, zero_q, uniform_parent, E.argmax(1).copy())
        warm = (E + qbonus).argmax(1)
        g_lab, _ = run_arm(E, trans, qbonus, parent, warm.copy())
        u_full.append(accuracy(u_lab, gt)[1])
        g_full.append(accuracy(g_lab, gt)[1])
    return (np.mean(g_full) - np.mean(u_full)) * 100, np.mean(u_full) * 100, np.mean(g_full) * 100


def run_cross_source(sigma: float, seeds: int, max_tunes: int) -> None:
    playlists = [REPO / "data" / "ireal" / f"{n}.txt"
                 for n in ("jazz1460", "pop400", "brazilian220", "blues50",
                           "latin_salsa50", "country", "dixieland1")]
    tunes, census = _find_cross_source_tunes(playlists)
    print(f"# Corpus census (why the non-circular symbolic set is small):")
    for k, v in census.most_common():
        print(f"#   {k:<38}{v:>4}")
    print()
    if not tunes:
        print("no VALID cross-source tune pairs (same key, comparable length, "
              "0<disagree≤0.5) — the symbolic non-circular test set is empty on "
              "this corpus; route to Part B2 (real audio, Mission 1).")
        return
    tunes = tunes[:max_tunes]

    print(f"# Mission 5 Part B1 — NON-CIRCULAR cross-source eval "
          f"(σ={sigma}, seeds={seeds}, offline analyst)")
    print(f"# prior from source A, GT from source B; {len(tunes)} VALID tune pair(s)\n")
    print(f"{'tune':<34}{'T':>4}{'dis%':>6}{'Δcross':>8}{'Δcirc':>8}")
    print("-" * 60)

    cross_deltas, circ_deltas = [], []
    for title, chartA, chartB, gtA, gtB, T, disagree in tunes:
        # NON-circular: prior from A, GT from B.
        fA = to_bayesian_factors(offline_analyze(chartA))
        d_cross, _, _ = _delta_full(gtB, fA, sigma, seeds)
        # circular control: prior from B, GT from B (the sim's setup).
        fB = to_bayesian_factors(offline_analyze(chartB))
        d_circ, _, _ = _delta_full(gtB, fB, sigma, seeds)
        cross_deltas.append(d_cross)
        circ_deltas.append(d_circ)
        name = title if len(title) <= 33 else title[:30] + "..."
        print(f"{name:<34}{T:>4}{disagree * 100:>5.0f}%{d_cross:>+8.1f}{d_circ:>+8.1f}")

    print("-" * 60)
    mc, mz = float(np.mean(cross_deltas)), float(np.mean(circ_deltas))
    mean_dis = float(np.mean([t[6] for t in tunes]))
    print(f"{'MEAN':<34}{'':>4}{'':>6}{mc:>+8.1f}{mz:>+8.1f}")

    # Power/validity guard (CLAUDE.md #5): the +2pp gate assumes a real
    # non-circular set. If the set is tiny AND the sources barely differ, Δcross
    # ≈ Δcirc *because A≈B*, not because the analyst generalizes — so a numeric
    # "PASS" here is not evidence of transfer. Require enough pairs and enough
    # genuine disagreement before trusting the sign.
    n_valid = len(tunes)
    powered = n_valid >= 5 and mean_dis >= 0.10
    numeric = "PASS" if mc >= 2.0 else "FAIL"
    print(f"\nStopping criterion (Δcross ≥ +2pp): numeric {numeric}  "
          f"(mean cross-source Δ = {mc:+.1f}pp; circular control = {mz:+.1f}pp)")
    print(f"Test power: {n_valid} valid pair(s), mean disagreement "
          f"{mean_dis * 100:.0f}%  →  {'ADEQUATE' if powered else 'INADEQUATE'}")
    if not powered:
        print("VERDICT: INCONCLUSIVE. The symbolic corpus has no usable "
              "non-circular set — the multi-source 'versions' are byte-identical "
              "transcriptions, homonyms, or transpositions (see census). With "
              f"{n_valid} near-identical pairs Δcross≈Δcirc by construction; this "
              "does NOT measure analyst transfer. Do not proceed to Part C on "
              "this basis — the real gate is Part B2 (real audio, Mission 1).")
    elif numeric == "FAIL":
        print("VERDICT: FAIL — prior mechanism too weak on non-circular data; "
              "STOP (no Part C).")
    else:
        print("VERDICT: PASS — prior mechanism transfers; proceed to Part C "
              "(section-conditional quality).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--song", default="Autumn Leaves")
    ap.add_argument("--playlist", default=str(REPO / "data" / "ireal" / "jazz1460.txt"))
    ap.add_argument("--offline", action="store_true", help="force rule-based analyst")
    ap.add_argument("--sigma", type=float, default=1.2, help="emission noise level")
    ap.add_argument("--seeds", type=int, default=100)
    ap.add_argument("--cross-source", action="store_true",
                    help="Part B1: non-circular eval (prior from A, GT from B)")
    ap.add_argument("--max-tunes", type=int, default=20,
                    help="cap on cross-source tune pairs")
    args = ap.parse_args()

    if args.cross_source:
        run_cross_source(args.sigma, args.seeds, args.max_tunes)
        return

    chart = load_chart(args.song, Path(args.playlist))
    analysis, path = analyze(chart, offline=args.offline)
    factors = to_bayesian_factors(analysis)
    gt = ground_truth_states(chart)
    T = len(gt)
    trans, qbonus = build_priors(factors, T)
    parent = pool_parent_map(factors, chart, T)
    uniform_parent = list(range(T))
    zero_trans = np.zeros((N_STATE, N_STATE))
    zero_q = np.zeros(N_STATE)

    # LLM MAP init: quality-prior argmax per bar's known root region is cheating
    # (uses the truth root), so instead init from emission argmax + the priors
    # applied once — a fair warm start the LLM could actually produce.
    res = {"uninformed": [], "llm": []}
    for seed in range(args.seeds):
        rng = np.random.default_rng(seed)
        E = make_emission(gt, args.sigma, rng)
        emis_argmax = E.argmax(1)

        u_lab, u_sw = run_arm(E, zero_trans, zero_q, uniform_parent, emis_argmax.copy())
        # LLM warm start: one prior-informed relabel pass for the init
        warm = (E + qbonus).argmax(1)
        l_lab, l_sw = run_arm(E, trans, qbonus, parent, warm.copy())

        res["uninformed"].append((u_sw, *accuracy(u_lab, gt)))
        res["llm"].append((l_sw, *accuracy(l_lab, gt)))

    def agg(rows):
        a = np.array(rows)
        return a[:, 0].mean(), a[:, 1].mean() * 100, a[:, 2].mean() * 100

    u_sw, u_root, u_full = agg(res["uninformed"])
    l_sw, l_root, l_full = agg(res["llm"])

    print(f"# Mission 5 convergence eval — {chart.title}  (analyst: {path}, "
          f"σ={args.sigma}, seeds={args.seeds})")
    print(f"# {T} bars, {len(factors.pool_group_bars)} pooled span-group(s), "
          f"prior strength {factors.strength:.1f} nats (conf {factors.confidence})")
    print(f"{'arm':<12}{'sweeps→conv':>13}{'root acc %':>12}{'(root,q5) %':>13}")
    print(f"{'uninformed':<12}{u_sw:>13.2f}{u_root:>12.1f}{u_full:>13.1f}")
    print(f"{'LLM-guided':<12}{l_sw:>13.2f}{l_root:>12.1f}{l_full:>13.1f}")
    print(f"{'Δ':<12}{u_sw - l_sw:>+13.2f}{l_root - u_root:>+12.1f}{l_full - u_full:>+13.1f}")
    print("\nInterpretation: positive sweeps-Δ = LLM warm-start converges faster; "
          "positive acc-Δ = higher final accuracy. This is a controlled "
          "simulation (synthetic emission); the real-audio number is gated on "
          "the Mission 1 benchmark.")


if __name__ == "__main__":
    main()
