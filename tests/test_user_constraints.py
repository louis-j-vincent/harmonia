"""Unit tests for the user-constraint bookkeeping (Mission 3).

Pure index/geometry mapping (seconds → beats → segments), no model state.
"""
import numpy as np
import pytest

from harmonia.models.user_constraints import (
    ChordConfirm,
    SectionMerge,
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


def test_pool_beat_evidence_rejects_unequal_beats():
    bt = np.arange(0, 3.5, 0.5)
    bp = np.ones((7, 2)) * 0.5
    # span A = [0,1.0) → 2 beats, span B = [1.0,1.5) → 1 beat
    merges = [SectionMerge(spans=[(0.0, 1.0), (1.0, 1.5)])]
    with pytest.raises(ValueError, match="equal"):
        pool_beat_evidence(merges, bt, bp)


def test_pool_groups_rejects_unequal_length():
    # span A → 2 segs, span B → 1 seg: must raise (v1 rejects unequal lengths)
    segs = [(0, 2), (2, 4), (4, 8)]
    bt = np.arange(0, 4.5, 0.5)
    merges = [SectionMerge(spans=[(0.0, 2.0), (2.0, 4.0)])]
    with pytest.raises(ValueError, match="equal musical length"):
        build_pool_groups(merges, segs, bt)
