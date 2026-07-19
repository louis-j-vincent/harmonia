"""bar_merge_candidates.py — 2026-07-18 chord-robustness reframe, Step 2.

Generate candidate bar-merge GROUPS (>=2 harmonically-near-identical bars)
from the 1-bar raw-chroma SSM (scratchpad/rawchroma.py's bt_concat variant,
already validated tonight as the untrained real-audio SSM input), for
presentation in the confirm/reject "pairs game" UI (Step 3).

UPDATED 2026-07-18 (continuation call): the multi-algorithm bakeoff flagged
as unfinished scope below has now been run — `scratchpad/clustering_bakeoff.py`,
corpus-scale iReal (900 tunes, grain=8 blocks, 5 seeds), comparing
threshold+pairs, k-NN+connected-components, agglomerative (single/complete/
average linkage), DBSCAN, and spectral+eigengap at a matched FPR<=0.05
operating point (results: `scratchpad/clustering_bakeoff_results.json`,
full writeup in docs/known_issues.md "★ CHORD-ROBUSTNESS / BAR-MERGE" and
docs/research_sessions/structure_realaudio_2026_07_18.md).

**Verdict: k-NN (k=1, floor~0.9) is a small, consistent, multi-seed-
confirmed winner** (mean recall 0.217 vs 0.187-0.192 for threshold/
agglomerative/DBSCAN, which are all numerically near-identical to each
other in this small-block-count regime; precision comparable-to-better;
~13% relative recall gain at matched FPR) — NOT a decisive knockout, a
modest edge. **Spectral+eigengap is a clear loser for this project's low-FPR
priority** (can't get below FPR~0.34-0.39 even at its best operating point;
eigengap on tiny per-tune graphs, m~4-15 blocks, is too noisy to pick a
trustworthy k). DBSCAN's "noise rejection" feature never activated
(min_samples=1 always won on val — at this operating point DBSCAN
degenerates to plain single-linkage threshold closure).

**Real-audio check (mandatory per the brief) found the SAME over-merge
collapse this docstring already warned about, reproduced quantitatively**:
plain threshold + FULL transitive closure at tau=0.93 on the 3 real songs'
1-bar SSM gives components of 71/117/183 bars (aretha/autumn_leaves/abba)
— i.e. confirms the original union-find failure mode is not
song-specific or a one-off. k-NN(k=1) + connected-components is dramatically
more collapse-resistant at the SAME thresholds (largest components 7-21
bars, not 71-183) because bounding each node to 1 outgoing edge caps
component growth — but k-NN's connected-components groups (up to 21 bars)
still don't fit this UI's pairs-only card format and are a bigger trust
ask per suggestion than a 2-bar pair. **Decision: ship k-NN's per-bar
top-1-neighbor EDGE SELECTION (the actual source of its corpus-validated
recall/precision edge) WITHOUT the connected-components closure step** —
i.e. each bar contributes at most one candidate edge (its best
above-floor match) to the ranked pool, then the existing rank+dedup+cap
pipeline below (unchanged) turns that into the same pairs-only candidate
list format as before. This keeps the corpus-validated selection
criterion, avoids the collapse risk of shipping transitive groups, and
causes zero candidate-JSON format drift for the UI or any other consumer.
Old global-threshold-only mode kept via `--algo threshold` for comparison.

UPDATED 2026-07-18 (SCOPE-GUARDED continuation call, two-tier auto/suggest
split): added a `tier` field to each candidate ("auto" vs "suggest"),
computed from a NEW similarity threshold TAU_AUTO=0.96, found via
`scratchpad/tau_auto_search.py` -- corpus-scale (full 1989/2399-tune iReal
corpus, bar-level pairs, GT redefined as same-chord-identity not
same-section, see that script's docstring for why), nested train/val/test
song-level selection to avoid a diagnosed single-split overfitting failure
mode (naive single-split selection put tau as low as 0.933 and then blew
past the target 2%-error band on a genuinely held-out fold in most seeds,
up to 10.6% observed). tau=0.96 is the lowest round-number threshold that
stayed reliably <=2% (max observed 1.54% Clopper-Pearson upper bound) across
ALL 5 independently-held-out blind test folds -- see known_issues.md
"★ CHORD-ROBUSTNESS / BAR-MERGE" for the full writeup and the honest
"never truly zero" finding (even at sim==1.0 there's a real, small
~0.1-0.3% floor from feature-representation aliasing between chords that
happen to produce identical proxy vectors; literal "never a false
positive" is not achievable with this feature space at ANY threshold, so
the user's request was relaxed to a 1-2%-band target instead).
TAU_SUGGEST=DEFAULT_TAU=0.93 remains the existing suggestion-tier floor,
unchanged. Every candidate this module emits already has sim>=DEFAULT_TAU
by construction, so tier is exactly: "auto" if confidence>=TAU_AUTO else
"suggest". Auto-APPLYING these (calling /api/reinfer without a human tap)
is explicitly NOT implemented here -- server/UI files are locked by a
parallel agent this call; see known_issues.md's "NEXT STEP (blocked on
parallel agent)" note for the exact handoff.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from rawchroma import per_bar_rawchroma

REPO = Path(__file__).resolve().parent.parent

# 2026-07-18 (dual-matrix cross-validation continuation call): JOINT
# audio+symbolic auto-tier gate. See docs/known_issues.md "★ CHORD-ROBUSTNESS
# / BAR-MERGE" and scratchpad/joint_threshold_search.py for the full
# derivation. Premise (scratchpad/dual_matrix_correlation_results.json's 2
# known false positives vs 2 known true positives): audio_sim alone shows NO
# gap between good/bad merges (0.978-0.994 for BOTH classes -- this is WHY
# tau_auto=0.96 fails on real audio). A SECOND, independently-sourced signal
# -- chord_distance.py's V1_binary chord-tone cosine similarity between the
# two bars' MAJORITY-VOTE label from the model's own unconstrained baseline
# decode -- DOES show a gap (false positives 0.67/0.87, true positives
# 1.0/1.0 exactly). Corpus-scale check (all 3 songs' full candidate census,
# pseudo-GT = baseline-decode root+quality-family bucket match, same
# methodology as realaudio_threshold_check.py): requiring BOTH
# audio_sim>=TAU_AUTO(0.96) AND symbolic_sim>=TAU_SYMBOLIC(0.90) raises
# pooled agreement from audio-alone's 39.4% to 89.6% (n=77/180 pairs
# retained) -- a real, corpus-validated improvement, NOT a single-song
# artifact (per-song: aretha 24.2%->54.5%, autumn_leaves 38.9%->100%, abba
# 44.1%->94.2%). HONEST CAVEAT, not glossed over: 89.6% pooled (and
# aretha's 54.5% alone) is STILL far short of the ~98-99% "never a false
# positive" bar tau_auto=0.96 was originally designed to meet -- this gate
# is a meaningful precision improvement, NOT a green light to wire silent
# auto-apply. See known_issues.md for the explicit "do not ship" call.
# Also note the audio_sim>=0.96 anchor here is a DEPLOYED-GATE convenience,
# not re-derived from scratch -- joint_threshold_search.py's full 2D sweep
# shows tau_audio itself contributes comparatively little once tau_symbolic
# is applied (e.g. tau_audio=0.90+tau_sym=0.90 gives precision=0.894,
# barely below tau_audio=0.96+tau_sym=0.90's 0.896) -- tau_symbolic is doing
# most of the discriminating work in this joint criterion.
TAU_SYMBOLIC = 0.90
MIN_GAP = 4          # exclude merge candidates closer than 4 bars apart
# 2026-07-18 (overnight autonomous call, task 1 of 3, ROC/AUC re-tune): was
# 0.93 (precision-first, FPR<=0.05-tuned against the grain=8 block bakeoff --
# see docs/known_issues.md "Multi-algorithm bar-merge candidate-generation
# BAKEOFF"). Re-derived around a LOW false-negative-rate target instead, per
# the user's explicit reasoning: SUGGEST-tier candidates are human-reviewed
# before anything happens, so a false positive costs one wasted tap while a
# false negative silently hides a real merge opportunity forever -- FPR<=0.05
# was optimizing the wrong side of that tradeoff. `scratchpad/roc_suggest_tier
# .py`, reusing tau_auto_search.py's corpus (full iReal, 2399 tunes, bar-level
# pairs, min_gap=4, CORRECTED same-chord-identity GT label) and nested
# train/val/test song-split selection (5 seeds, tau chosen on train+val,
# reported on blind test only): recall/FPR at a dense threshold grid, 5-fold
# mean+-std ROC-AUC=0.9885+-0.0016, PR-AUC(AP)=0.9590+-0.0047. Recall-target
# table (blind-test, mean across 5 folds): 60%/75% recall need only tau=1.0
# exact-duplicate pairs (FPR~0.02-0.03%, the corpus has a lot of literal
# exact-match same-chord bar pairs); 85% recall needs tau~=0.80 (FPR~3.5-3.9%
# mean, precision~84-85%); 90% recall needs tau~=0.75 (FPR~6.0%, precision
# ~78%) -- FPR/precision degrade noticeably faster past 85% than up to it, so
# 0.80 (85% recall target) is the shipped choice, matching the coordinator's
# stated heuristic ("FPR cost still low at 85%, rises sharply toward 90%+").
# Directly verified AT tau=0.80 (not just the fold-selected values, which
# ranged 0.7887-0.8100): blind-test recall mean=0.841 (worst fold 0.819),
# FPR mean=0.0388 (worst fold 0.0400), precision mean=0.836 -- consistent
# with the recall-target table. HONEST CAVEAT (do not silently gloss over):
# this recall/FPR is for the underlying similarity-THRESHOLD's discriminative
# power over the FULL bar-pair census, not the deployed k-NN top-1-edge-per-
# bar candidate GENERATOR (candidate_groups(algo="knn") below) -- the k-NN
# cap additionally restricts which pairs ever become candidates regardless of
# threshold, so realized suggest-tier recall on real audio will be lower than
# this table implies; also NOT directly comparable to the earlier grain=8
# block-level bakeoff's 0.217+-0.031 recall number (different grain, feature,
# and candidate-generation algorithm, not a like-for-like re-measurement).
DEFAULT_TAU = 0.80
DEFAULT_K = 1          # bakeoff-winning k-NN neighbor count (2026-07-18)
TAU_AUTO = 0.96        # 2026-07-18 continuation: two-tier auto/suggest split
                       # threshold, see module docstring "UPDATED" section
                       # and scratchpad/tau_auto_search.py for the corpus-
                       # scale, nested-cross-validated derivation.


def candidate_groups(audio_path: Path, tau: float = DEFAULT_TAU, min_gap: int = MIN_GAP,
                      beats_per_bar: int = 4, max_candidates: int = 60,
                      max_pairs_per_bar: int = 2, algo: str = "knn",
                      k: int = DEFAULT_K, tau_auto: float = TAU_AUTO):
    """PAIRS-first candidate generation (method (a) from the brief, but
    WITHOUT transitive union-find closure into groups — see rationale
    below). Each candidate is a single (bar_i, bar_j) pair, ranked by
    cosine similarity, deduplicated so no bar appears in more than
    `max_pairs_per_bar` candidates (keeps the confirm-game list short and
    each item genuinely inspectable, rather than one bar's neighborhood
    dominating the whole list).

    SCOPE DECISION (logged, not silently narrowed): the brief asked for a
    comparison of >=3 clustering strategies (threshold+union-find,
    k-NN+connected-components, hierarchical/DBSCAN/spectral). Threshold+
    union-find's OWN failure mode — transitive collapse into one giant
    low-precision blob — is already extensively characterized in
    docs/known_issues.md's structure-detection thread tonight (e.g. "Step
    2... logreg does NOT beat a simple threshold", multiple "over-merge
    collapse" entries) and reproduced immediately here (see session log: a
    first pass at grain=1 real-audio bt_concat gave a 329/330-bar single
    component at tau=0.93 before a normalization bug was even fixed, and
    still gave 70+ and 183-bar components after the fix). Given the time
    budget, re-running that already-known failure mode through 2 more
    clustering variants to re-confirm the same collapse was deprioritized
    in favor of shipping a working, precision-first candidate generator:
    PAIRS avoid the collapse entirely (no transitivity), match the user's
    own "pairs game" framing more directly than n-way groups would, and are
    trivially precision-checkable (one similarity number per candidate, no
    within-group averaging to hide a bad member). A real k-NN/hierarchical
    comparison remains legitimate future work, not attempted here.
    """
    variants, bar_times, tempo, tonic = per_bar_rawchroma(audio_path, beats_per_bar)
    v = variants["bt_concat"]
    n = len(v)
    # bt_concat rows have norm sqrt(2) (bass half and treble half each
    # independently L2-normalized before concatenation, per rawchroma.py's
    # own docstring) — NOT unit norm. A raw v @ v.T here is a bug: it yields
    # dot products up to 2.0, not cosine similarity, silently halving the
    # effective threshold and massively over-merging (caught by inspection:
    # first run gave one ~330-bar "group" covering nearly a whole song, and
    # confidence values >1.0, which is impossible for a real cosine sim).
    # Explicit row-normalize before the Gram product fixes it regardless of
    # each half's exact norm.
    row_norm = np.linalg.norm(v, axis=1, keepdims=True)
    v_unit = v / np.clip(row_norm, 1e-9, None)
    sim = v_unit @ v_unit.T

    if algo == "knn":
        # 2026-07-18 bakeoff winner's edge-selection rule, WITHOUT the
        # connected-components closure (see module docstring): each bar
        # contributes at most its own top-`k` above-floor matches (respecting
        # min_gap) to the candidate pool, rather than every globally-above-
        # threshold pair. `tau` is reused as the floor here.
        edge_set = {}
        for i in range(n):
            cand = [(float(sim[i, j]), j) for j in range(n)
                    if abs(j - i) >= min_gap and sim[i, j] >= tau]
            cand.sort(reverse=True)
            for s, j in cand[:k]:
                key = (min(i, j), max(i, j))
                if key not in edge_set or s > edge_set[key]:
                    edge_set[key] = s
        edges = [(s, i, j) for (i, j), s in edge_set.items()]
    elif algo == "threshold":
        edges = []
        for i in range(n):
            for j in range(i + min_gap, n):
                if sim[i, j] >= tau:
                    edges.append((float(sim[i, j]), i, j))
    else:
        raise ValueError("unknown algo %r (expected 'knn' or 'threshold')" % algo)
    edges.sort(reverse=True)

    used_count = {}
    out = []
    for s, i, j in edges:
        if used_count.get(i, 0) >= max_pairs_per_bar or used_count.get(j, 0) >= max_pairs_per_bar:
            continue
        used_count[i] = used_count.get(i, 0) + 1
        used_count[j] = used_count.get(j, 0) + 1
        spans = [[float(bar_times[i]), float(bar_times[i + 1])],
                 [float(bar_times[j]), float(bar_times[j + 1])]]
        tier = "auto" if s >= tau_auto else "suggest"
        out.append({"bars": [i, j], "spans": spans, "confidence": s, "n_bars": 2, "tier": tier})
        if len(out) >= max_candidates:
            break

    return out, {"n_bars_total": n, "tempo_bpm": tempo, "tonic_pc": tonic,
                 "tau": tau, "min_gap": min_gap, "algo": algo, "k": k if algo == "knn" else None,
                 "tau_auto": tau_auto, "tau_suggest": tau,
                 "n_auto": sum(1 for c in out if c["tier"] == "auto"),
                 "n_suggest": sum(1 for c in out if c["tier"] == "suggest")}


def apply_symbolic_gate(candidates, base_chords, tau_symbolic: float = TAU_SYMBOLIC):
    """Post-process an already-generated candidate list (unchanged output of
    candidate_groups()): for every candidate currently tier=="auto", also
    require chord_distance V1_binary symbolic chord-tone cosine similarity
    (computed from `base_chords` -- the model's own unconstrained baseline
    decode, majority-vote per bar span) to be >= tau_symbolic; candidates
    that fail this second gate are DEMOTED to tier=="suggest" (they already
    passed the audio-only DEFAULT_TAU floor by construction, so demotion,
    not deletion, is correct -- a human still sees them). tier=="suggest"
    candidates are left untouched (per the brief: audio-similarity-ranked,
    human-reviewed, this gate is specifically an AUTO-tier precision fix).

    PURELY ADDITIVE / OPT-IN: candidate_groups() itself is not modified;
    callers that don't call this function see zero behavior change (rule #6
    caution -- this needs `base_chords`, an unconstrained infer_chords_v1
    decode, which candidate_groups() does not compute and should not be
    made to compute unconditionally, since most callers only need the
    rawchroma SSM, not a full chord decode).

    Adds two fields to each candidate for transparency/auditing:
    `symbolic_sim` (the computed V1_binary similarity, or None if a bar had
    no decodable baseline chord) and `tier_reason` ("audio_only" if tier
    was decided without ever reaching the symbolic gate, i.e. it was never
    tier=="auto" to begin with; "joint_pass"/"joint_fail" for auto-tier
    candidates that were checked).
    """
    from dual_matrix_correlation import label_to_root_qual, bar_chord_majority
    from chord_distance import chord_vector_binary, cosine

    out = []
    for c in candidates:
        c = dict(c)
        if c["tier"] != "auto":
            c["symbolic_sim"] = None
            c["tier_reason"] = "audio_only"
            out.append(c)
            continue
        (t0a, t1a), (t0b, t1b) = c["spans"]
        label_a = bar_chord_majority(base_chords, t0a, t1a)
        label_b = bar_chord_majority(base_chords, t0b, t1b)
        pc_a, q_a = label_to_root_qual(label_a)
        pc_b, q_b = label_to_root_qual(label_b)
        if pc_a is None or pc_b is None:
            # no decodable chord at one of the spans -- can't confirm the
            # symbolic side; fail closed (demote), don't silently pass.
            c["symbolic_sim"] = None
            c["tier"] = "suggest"
            c["tier_reason"] = "joint_fail"
            out.append(c)
            continue
        va, vb = chord_vector_binary(pc_a, q_a), chord_vector_binary(pc_b, q_b)
        sim = cosine(va, vb)
        c["symbolic_sim"] = float(sim)
        if sim >= tau_symbolic:
            c["tier_reason"] = "joint_pass"
        else:
            c["tier"] = "suggest"
            c["tier_reason"] = "joint_fail"
        out.append(c)
    return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    ap.add_argument("--tau", type=float, default=DEFAULT_TAU)
    ap.add_argument("--algo", choices=["knn", "threshold"], default="knn")
    ap.add_argument("--k", type=int, default=DEFAULT_K)
    ap.add_argument("--tau-auto", type=float, default=TAU_AUTO)
    args = ap.parse_args()
    groups, meta = candidate_groups(args.audio, tau=args.tau, algo=args.algo, k=args.k,
                                     tau_auto=args.tau_auto)
    print(json.dumps({"groups": groups, "meta": meta}, indent=2))
