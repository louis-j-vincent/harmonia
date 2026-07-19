"""Unit tests for the user-constraint bookkeeping (Mission 3).

Pure index/geometry mapping (seconds → beats → segments), no model state.
"""
import numpy as np
import pytest

from harmonia.models.user_constraints import (
    ChordConfirm,
    SectionMerge,
    _span_to_beats,
    build_pool_groups,
    build_segment_constraints,
    confirm_cut_beats,
    force_boundaries,
    pool_beat_evidence,
)

# beat grid: 9 beats at 0.5 s each → times 0.0, 0.5, ..., 4.0
BT = np.arange(0, 4.5, 0.5)


def test_force_boundaries_splits_and_is_idempotent():
    segs = [(0, 4), (4, 8)]
    cut = force_boundaries(segs, [2])
    assert cut == [(0, 2), (2, 4), (4, 8)]
    # idempotent: cutting again at the same beat changes nothing
    assert force_boundaries(cut, [2]) == cut
    # cut on an existing boundary is a no-op
    assert force_boundaries(segs, [4]) == segs


def test_confirm_cut_beats_maps_to_nearest_beat():
    confirms = [ChordConfirm(t0=1.0, t1=2.0, root=0)]  # → beats 2 and 4
    assert confirm_cut_beats(confirms, BT) == [2, 4]


def test_segment_constraints_land_on_overlapping_segments():
    # segs cover beats; confirm span [1.0,2.0) → beats [2,4)
    segs = [(0, 2), (2, 4), (4, 6)]
    confirms = [ChordConfirm(t0=1.0, t1=2.0, root=9, q5=1)]
    cons = build_segment_constraints(confirms, segs, BT)
    assert cons[0] is None
    assert cons[1] == {"root": 9, "q5": 1, "bonus": pytest.approx(40.0)}
    assert cons[2] is None


def test_pool_groups_equal_length_ties_corresponding_segments():
    # span A beats [0,2) → seg 0; span B beats [2,4) → seg 1 (equal length: 1 seg each)
    segs = [(0, 2), (2, 4), (4, 6)]
    merges = [SectionMerge(spans=[(0.0, 1.0), (1.0, 2.0)])]
    groups = build_pool_groups(merges, segs, BT)
    assert groups == [[0, 1]]


def test_pool_groups_multi_segment_spans_tie_slotwise():
    # span A = beats [0,4) → segs 0,1 ; span B = beats [4,8) → segs 2,3
    segs = [(0, 2), (2, 4), (4, 6), (6, 8)]
    bt = np.arange(0, 4.5, 0.5)
    merges = [SectionMerge(spans=[(0.0, 2.0), (2.0, 4.0)])]
    groups = build_pool_groups(merges, segs, bt)
    assert groups == [[0, 2], [1, 3]]   # slot 0 tied, slot 1 tied


def test_pool_beat_evidence_sums_and_renormalises():
    # 4 beats @ 0.5s. Merge span A = beats [0,2), span B = beats [2,4).
    bt = np.arange(0, 2.5, 0.5)
    # posterior rows (sum to 1): A leans root 0, B leans root 5
    bp = np.array([[0.6, 0.4], [0.6, 0.4], [0.4, 0.6], [0.4, 0.6]])
    merges = [SectionMerge(spans=[(0.0, 1.0), (1.0, 2.0)])]
    (out,) = pool_beat_evidence(merges, bt, bp)
    # beat 0 (span A off 0) pools with beat 2 (span B off 0): [0.6,0.4]+[0.4,0.6]
    # = [1.0,1.0] renormalised → [0.5,0.5]; both tied beats get it
    assert np.allclose(out[0], [0.5, 0.5])
    assert np.allclose(out[2], [0.5, 0.5])
    assert np.allclose(out[1], [0.5, 0.5])
    assert np.allclose(out[3], [0.5, 0.5])
    # rows still sum to 1 (still a posterior)
    assert np.allclose(out.sum(1), 1.0)


def test_pool_beat_evidence_denoises_toward_agreement():
    """Pooling a decisive observation with a noisy one sharpens the noisy one."""
    bt = np.arange(0, 2.5, 0.5)
    bp = np.array([[0.55, 0.45],   # span A beat: barely root 0
                   [0.9, 0.1],
                   [0.95, 0.05],   # span B beat: decisively root 0
                   [0.9, 0.1]])
    merges = [SectionMerge(spans=[(0.0, 0.5), (1.0, 1.5)])]  # 1-beat spans
    (out,) = pool_beat_evidence(merges, bt, bp)
    # beat 0 (0.55) pooled with beat 2 (0.95) → both lean much harder to root 0
    assert out[0][0] > 0.55
    assert out[0][0] == pytest.approx(out[2][0])


def test_pool_beat_evidence_merges_read_original_evidence_not_each_others_output():
    """2026-07-18 real bug (found via multi-merge-per-request production
    testing, ★ CHORD-ROBUSTNESS / BAR-MERGE): two merge groups sharing a beat
    index (beat 2 below, in both merge A-B and merge B-C) must each pool
    from beat 2's ORIGINAL evidence, not from whatever the previous merge in
    the list already wrote there — otherwise the same set of merges in a
    different list order silently produces different confidences (reproduced
    on real audio via /api/reinfer with 2 real songs before this fix: same
    2 merges, reversed list order, gave different confidence at a beat that
    only ONE of the two merges actually touches).

    Post-fix invariant: a beat exclusive to one merge is fully ORDER-
    INDEPENDENT (that merge always reads the original evidence of its
    partner beat, never a previous merge's pooled output). A beat the two
    merges genuinely CONTEST (beat 2/3 here) still follows last-write-wins
    — same convention as overlapping chord-confirms
    (``build_segment_constraints``) — but must equal exactly the LAST
    merge's own pooled value, not some contaminated blend of both.
    """
    bt = np.arange(0, 3.5, 0.5)   # 7 beats
    bp = np.array([[0.9, 0.1],    # beat 0 (span A, exclusive to merge_ab)
                   [0.9, 0.1],
                   [0.5, 0.5],    # beat 2 (span B) — shared/contested beat
                   [0.5, 0.5],
                   [0.1, 0.9],    # beat 4 (span C, exclusive to merge_bc)
                   [0.1, 0.9],
                   [0.5, 0.5]])
    merge_ab = SectionMerge(spans=[(0.0, 1.0), (1.0, 2.0)])   # beats {0,1}<->{2,3}
    merge_bc = SectionMerge(spans=[(1.0, 2.0), (2.0, 3.0)])   # beats {2,3}<->{4,5}

    (out_order1,) = pool_beat_evidence([merge_ab, merge_bc], bt, bp)
    (out_order2,) = pool_beat_evidence([merge_bc, merge_ab], bt, bp)

    # exclusive beats: fully order-independent
    assert np.allclose(out_order1[0], out_order2[0])   # beat 0 (merge_ab only)
    assert np.allclose(out_order1[4], out_order2[4])   # beat 4 (merge_bc only)

    # contested beat 2: last-write-wins, and must match that merge's OWN
    # solo pooled value exactly (no compounding contamination)
    (solo_ab,) = pool_beat_evidence([merge_ab], bt, bp)
    (solo_bc,) = pool_beat_evidence([merge_bc], bt, bp)
    assert np.allclose(out_order1[2], solo_bc[2])   # bc ran last in order1
    assert np.allclose(out_order2[2], solo_ab[2])   # ab ran last in order2


def test_pool_beat_evidence_rejects_unequal_beats():
    bt = np.arange(0, 3.5, 0.5)
    bp = np.ones((7, 2)) * 0.5
    # span A = [0,1.0) → 2 beats, span B = [1.0,1.5) → 1 beat
    merges = [SectionMerge(spans=[(0.0, 1.0), (1.0, 1.5)])]
    with pytest.raises(ValueError, match="equal"):
        pool_beat_evidence(merges, bt, bp)


def test_pool_beat_evidence_partial_batch_skips_only_the_bad_merge():
    """2026-07-18 (overnight autonomous call, auto-apply task): real bug found
    running MANY auto-tier merge groups in a single request (the exact
    use case the multi-merge order-independence fix earlier tonight was
    FOR) — a single malformed merge (unequal beat count, e.g. from small
    tempo-grid drift between the candidate generator's own bar boundaries
    and the pipeline's independently-estimated beat grid `bt`) used to
    raise ValueError and silently reject the ENTIRE batch, including every
    other perfectly valid merge group in the same request. On the 3 real
    songs' auto-tier batches (17-54 groups each) this meant 0/many groups
    ever actually applied, every single time, because at least one
    malformed group was essentially always present.

    Desired behaviour: a merge with unequal beat count is skipped (and
    reported via the `rejected` out-list), while every OTHER, valid merge
    in the same call still pools normally — matching the "graceful
    per-item degradation" convention `build_segment_constraints` already
    uses for overlapping confirms. Only when EVERY merge in the batch is
    malformed does the function still raise (preserves
    test_pool_beat_evidence_rejects_unequal_beats's single-merge contract
    above, unmodified)."""
    bt = np.arange(0, 3.5, 0.5)   # 7 beats
    bp = np.array([[0.6, 0.4], [0.6, 0.4], [0.4, 0.6], [0.4, 0.6],
                   [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    good = SectionMerge(spans=[(0.0, 1.0), (1.0, 2.0)])       # 2 beats each — valid
    bad = SectionMerge(spans=[(0.0, 1.0), (2.0, 2.5)])        # 2 beats vs 1 beat — invalid
    rejected: list = []
    (out,) = pool_beat_evidence([good, bad], bt, bp, rejected=rejected)
    # the good merge still pooled beat 0 with beat 2 exactly as its solo case would
    (solo,) = pool_beat_evidence([good], bt, bp)
    assert np.allclose(out[0], solo[0])
    assert np.allclose(out[2], solo[2])
    # the bad merge was recorded as rejected, not silently dropped with no trace
    assert len(rejected) == 1
    assert rejected[0]["beat_lens"] == [2, 1]


def test_pool_beat_evidence_all_bad_still_raises():
    """The batch-tolerant fix above must not silently swallow a batch where
    EVERY merge is malformed — that case still needs to surface as a hard
    rejection (api_reinfer's caller relies on the exception to report
    'that correction couldn't be applied' rather than a silent no-op)."""
    bt = np.arange(0, 3.5, 0.5)
    bp = np.ones((7, 2)) * 0.5
    bad1 = SectionMerge(spans=[(0.0, 1.0), (1.0, 1.5)])
    bad2 = SectionMerge(spans=[(0.0, 1.5), (2.0, 2.5)])
    with pytest.raises(ValueError, match="equal"):
        pool_beat_evidence([bad1, bad2], bt, bp)


def test_pool_beat_evidence_partial_pool_within_group_excludes_only_the_weak_link():
    """2026-07-19 (★ CHORD-ROBUSTNESS / BAR-MERGE): the graceful-degradation
    fix. BEFORE this fix, a merge group whose spans did NOT all share the
    exact same beat count was rejected WHOLESALE — one weak link (a single
    span quantised +/-1 beat off the mode by beat-grid drift) broke the
    entire group, and if it was the only group in the request the whole
    thing raised. This is the failure that took down real N-way section-
    cluster pooling (aretha 0/2, abba 0/3, autumn_leaves 1/5 groups applied)
    until a caller-side per-bar-offset workaround was built INSTEAD of fixing
    the function.

    Desired behaviour: pool the MAJORITY/MODE beat count's spans, EXCLUDE the
    mismatched span(s) explicitly (reported, never silently dropped and never
    force-aligned/truncated), and still succeed. The surviving pooled spans
    are equal-length among themselves by construction.
    """
    bt = np.arange(0, 3.5, 0.5)   # 7 beats @ 0.5s
    # 3 spans: A=[0,1.0)->2 beats {0,1}, B=[1.0,2.0)->2 beats {2,3},
    #          C=[2.0,2.5)->1 beat {4} (the weak link, off-by-one)
    bp = np.array([[0.6, 0.4], [0.6, 0.4], [0.4, 0.6], [0.4, 0.6],
                   [0.1, 0.9], [0.5, 0.5], [0.5, 0.5]])
    merge = SectionMerge(spans=[(0.0, 1.0), (1.0, 2.0), (2.0, 2.5)])
    report: list = []
    (out,) = pool_beat_evidence([merge], bt, bp, pooled_report=report)

    # the two mode (2-beat) spans pooled exactly as a clean 2-span merge of
    # just A and B would (the weak link C never contaminates them)
    clean = SectionMerge(spans=[(0.0, 1.0), (1.0, 2.0)])
    (solo,) = pool_beat_evidence([clean], bt, bp)
    assert np.allclose(out[0], solo[0])
    assert np.allclose(out[2], solo[2])
    # C's beat (4) was NOT pooled — left at its original value
    assert np.allclose(out[4], bp[4])

    # the exclusion is reported explicitly, with the magnitude of the miss
    assert len(report) == 1
    r = report[0]
    assert r["status"] == "partial"
    assert r["mode_beats"] == 2
    assert len(r["pooled_spans"]) == 2
    assert len(r["excluded"]) == 1
    exc = r["excluded"][0]
    assert exc["expected_beats"] == 2
    assert exc["got_beats"] == 1
    assert exc["span"] == [2.0, 2.5]


def test_pool_beat_evidence_partial_pool_does_not_raise_as_lone_group():
    """RED on pre-fix code: a lone merge with one mismatched span used to
    raise ValueError (all-or-nothing). After the fix it partially pools and
    returns normally."""
    bt = np.arange(0, 3.5, 0.5)
    bp = np.array([[0.6, 0.4], [0.6, 0.4], [0.4, 0.6], [0.4, 0.6],
                   [0.5, 0.5], [0.5, 0.5], [0.5, 0.5]])
    merge = SectionMerge(spans=[(0.0, 1.0), (1.0, 2.0), (2.0, 2.5)])
    # must NOT raise
    (out,) = pool_beat_evidence([merge], bt, bp)
    assert out.shape == bp.shape


def test_pool_beat_evidence_surviving_pooled_spans_are_equal_length_among_themselves():
    """A far-miss span (off by many beats) is excluded, not force-aligned —
    the survivors remain genuinely equal musical length among themselves."""
    bt = np.arange(0, 5.5, 0.5)   # 11 beats
    bp = np.tile(np.array([0.7, 0.3]), (11, 1))
    # A=[0,1.5)->3 beats {0,1,2}, B=[1.5,3.0)->3 beats {3,4,5},
    # C=[3.0,5.0)->4 beats {6,7,8,9} (far miss, +1 over a 3-beat mode)
    merge = SectionMerge(spans=[(0.0, 1.5), (1.5, 3.0), (3.0, 5.0)])
    report: list = []
    (out,) = pool_beat_evidence([merge], bt, bp, pooled_report=report)
    r = report[0]
    assert r["status"] == "partial"
    assert r["mode_beats"] == 3
    # every pooled span is exactly 3 beats (equal among themselves)
    assert all(sp["got_beats"] == 3 for sp in r["pooled_spans"])
    assert r["excluded"][0]["got_beats"] == 4


def test_pool_beat_evidence_no_majority_two_way_tie_pools_larger_beatcount():
    """No clear majority (2-2 split): tiebreak deterministically toward the
    larger beat count (more musical evidence per pooled beat). Both 2-span
    subgroups are internally consistent, so either is safe; we pick
    deterministically and report which."""
    bt = np.arange(0, 6.5, 0.5)   # 13 beats
    bp = np.tile(np.array([0.6, 0.4]), (13, 1))
    # A=[0,1.0)->2 beats, B=[1.0,2.0)->2 beats, C=[2.0,3.5)->3 beats, D=[3.5,5.0)->3 beats
    merge = SectionMerge(spans=[(0.0, 1.0), (1.0, 2.0), (2.0, 3.5), (3.5, 5.0)])
    report: list = []
    pool_beat_evidence([merge], bt, bp, pooled_report=report)
    r = report[0]
    assert r["status"] == "partial"
    assert r["mode_beats"] == 3           # tie broken toward larger count
    assert len(r["pooled_spans"]) == 2
    assert len(r["excluded"]) == 2        # the two 2-beat spans excluded


def test_pool_beat_evidence_unpoolable_when_fewer_than_two_survive():
    """If the mode has <2 spans (e.g. all spans a different length), the
    group can't be pooled — reported as unpoolable via `rejected`, NOT a
    silent no-op success. As a lone group it still raises (preserves the
    single-merge hard-reject contract)."""
    bt = np.arange(0, 5.5, 0.5)   # 11 beats
    bp = np.ones((11, 2)) * 0.5
    # 3 spans of 2, 3, 4 beats — every length unique, no mode >=2
    merge = SectionMerge(spans=[(0.0, 1.0), (1.0, 2.5), (2.5, 4.5)])
    with pytest.raises(ValueError, match="equal"):
        pool_beat_evidence([merge], bt, bp)
    # but alongside a good group it degrades instead of raising, and is
    # reported as unpoolable in `rejected`
    good = SectionMerge(spans=[(0.0, 1.0), (1.0, 2.0)])
    rejected: list = []
    report: list = []
    pool_beat_evidence([good, merge], bt, bp, rejected=rejected, pooled_report=report)
    assert len(rejected) == 1
    assert sorted(rejected[0]["beat_lens"]) == [2, 3, 4]
    statuses = sorted(r["status"] for r in report)
    assert statuses == ["pooled", "unpoolable"]


def test_pool_beat_evidence_partial_pool_is_order_independent():
    """The existing multi-merge order-independence guarantee must still hold
    when one group needs internal exclusion: a beat exclusive to one merge is
    order-independent even if the OTHER merge partially pools."""
    bt = np.arange(0, 4.5, 0.5)   # 9 beats
    bp = np.array([[0.9, 0.1], [0.9, 0.1], [0.5, 0.5], [0.5, 0.5],
                   [0.1, 0.9], [0.1, 0.9], [0.5, 0.5], [0.3, 0.7], [0.3, 0.7]])
    # merge_ab: clean 2-span (beats {0,1}<->{2,3}); exclusive beats 0,1
    merge_ab = SectionMerge(spans=[(0.0, 1.0), (1.0, 2.0)])
    # merge_partial: 3 spans, one weak link — {2,3}, {4,5}, and a 1-beat {6}
    merge_partial = SectionMerge(spans=[(1.0, 2.0), (2.0, 3.0), (3.0, 3.5)])
    (o1,) = pool_beat_evidence([merge_ab, merge_partial], bt, bp)
    (o2,) = pool_beat_evidence([merge_partial, merge_ab], bt, bp)
    # exclusive-to-merge_ab beat 0 is fully order-independent
    assert np.allclose(o1[0], o2[0])
    # exclusive-to-merge_partial beat 4 is fully order-independent
    assert np.allclose(o1[4], o2[4])


# --- REAL-AUDIO fixture: aretha "Chain Of Fools" production beat grid -------
# Reconstructed EXACTLY as chord_pipeline_v1 builds it (constant-tempo uniform
# arange grid with 0.0 prepended and the duration appended), from the real
# measured tempo/phase/duration of the actual recording. Cluster spans are the
# real 8-bar-block time ranges from scratchpad/dual_matrix_grain8_results.json.
# Source of every number here: scratchpad/drift_rootcause_check_results.json +
# extract of the real audio (Part A's own measurement path) — NOT synthetic.
_ARETHA_PERIOD = 0.5108390022675736
_ARETHA_PHASE = 0.028521176874107997
_ARETHA_DUR = 168.9813605442177


def _aretha_bt() -> np.ndarray:
    bt = np.arange(_ARETHA_PHASE, _ARETHA_DUR + _ARETHA_PERIOD, _ARETHA_PERIOD)
    return np.unique(np.concatenate([[0.0], bt, [_ARETHA_DUR]]))


# Real cluster "B" — five repeats of one static Cm7 16-bar vamp. Under the OLD
# double-quantized count one repeat mis-reads as 33 beats (endpoint rounding),
# so the whole group would lose a span to graceful exclusion; single-quantized
# duration reads all five as exactly 32.
_ARETHA_B_SPANS = [
    (16.082192567782954, 32.466192567782954),
    (32.466192567782954, 48.850192567782955),
    (48.850192567782955, 65.23419256778296),
    (65.23419256778296, 81.61819256778296),
    (114.38619256778296, 130.77019256778294),
]
# Real cluster "A" — the last span (147.15–169.17s) is a GENUINE 43-beat
# length-outlier the fix must NOT paper over (Part B still excludes it).
_ARETHA_A_SPANS = [
    (0.0, 16.082192567782954),
    (98.00219256778296, 114.38619256778296),
    (130.77019256778294, 147.15419256778296),
    (147.15419256778296, 169.17019256778295),
]


def test_span_to_beats_single_quantized_recovers_equal_counts_real_audio():
    """RED on pre-fix (double-quantized) code — Part A's root-cause fix.

    On aretha "Chain Of Fools" cluster B (five repeats of one static 16-bar
    vamp on the real production beat grid), the OLD ``argmin(t1)-argmin(t0)``
    count mis-reads one repeat as 33 beats; the NEW ``round((t1-t0)/period)``
    count reads all five as exactly 32. Concrete before/after from
    drift_rootcause_check_results.json: OLD [32,33,32,32,32] → NEW
    [32,32,32,32,32]."""
    bt = _aretha_bt()
    counts = [b1 - b0 for (b0, b1) in
              (_span_to_beats(t0, t1, bt) for (t0, t1) in _ARETHA_B_SPANS)]
    assert counts == [32, 32, 32, 32, 32]      # RED on old code: [32,33,32,32,32]
    # b0 is still the true START anchor (nearest beat to t0) — the fix changes
    # only the COUNT, never where a span begins, so no over/under-run into a
    # neighbour's beats at the start (the subtlety Part A flagged).
    for (t0, t1) in _ARETHA_B_SPANS:
        b0, _b1 = _span_to_beats(t0, t1, bt)
        assert b0 == int(np.clip(np.abs(bt - t0).argmin(), 0, len(bt) - 1))


def test_pool_beat_evidence_fully_pools_real_vamp_cluster_after_fix():
    """End-to-end: the single-quantized count makes the real aretha-B vamp
    cluster pool FULLY (status 'pooled', zero exclusions) through
    pool_beat_evidence. On pre-fix code it would be 'partial' with one span
    excluded — i.e. exclusions become RARE (genuine outliers only), not the
    routine ±1-beat-noise norm they were before Part A's fix."""
    bt = _aretha_bt()
    arr = np.tile(np.array([0.5, 0.5]), (len(bt), 1))
    merge = SectionMerge(spans=list(_ARETHA_B_SPANS))
    report: list = []
    pool_beat_evidence([merge], bt, arr, pooled_report=report)
    assert len(report) == 1
    assert report[0]["status"] == "pooled"          # RED on old: "partial"
    assert len(report[0]["excluded"]) == 0          # RED on old: 1 excluded
    assert len(report[0]["pooled_spans"]) == 5


def test_pool_beat_evidence_still_excludes_genuine_length_outlier_after_fix():
    """The fix must NOT paper over a REAL length-outlier: aretha cluster A's
    last block genuinely spans ~43 beats vs the other three's ~31-32. Even
    with single-quantized counting it is excluded (Part B's weak-link
    safety-net), confirming the two fixes are complementary — Part A removes
    routine ±1 noise, Part B still catches true outliers."""
    bt = _aretha_bt()
    arr = np.tile(np.array([0.5, 0.5]), (len(bt), 1))
    counts = [b1 - b0 for (b0, b1) in
              (_span_to_beats(t0, t1, bt) for (t0, t1) in _ARETHA_A_SPANS)]
    assert counts == [31, 32, 32, 43]               # genuine outlier survives the fix
    merge = SectionMerge(spans=list(_ARETHA_A_SPANS))
    report: list = []
    pool_beat_evidence([merge], bt, arr, pooled_report=report)
    r = report[0]
    assert r["status"] == "partial"
    assert r["mode_beats"] == 32
    got = sorted(e["got_beats"] for e in r["excluded"])
    assert got == [31, 43]                           # the 43-beat outlier is dropped


def test_pool_groups_rejects_unequal_length():
    # span A → 2 segs, span B → 1 seg: must raise (v1 rejects unequal lengths)
    segs = [(0, 2), (2, 4), (4, 8)]
    bt = np.arange(0, 4.5, 0.5)
    merges = [SectionMerge(spans=[(0.0, 2.0), (2.0, 4.0)])]
    with pytest.raises(ValueError, match="equal musical length"):
        build_pool_groups(merges, segs, bt)
