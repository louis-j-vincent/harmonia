"""apply_rigid_grid re-bins chords onto a corrected bar grid (the receiving end
for the per-section repetition-period grid corrector, 2026-07-29)."""
from harmonia.models.rigid_grid import apply_rigid_grid, rigid_grid_for


def test_one_chord_per_bar():
    # four chords on their bar downbeats -> each its own bar, beat 0
    chords = [{"t0": 0.0, "t1": 1.0, "root": 0}, {"t0": 1.0, "t1": 2.0, "root": 5},
              {"t0": 2.0, "t1": 3.0, "root": 7}, {"t0": 3.0, "t1": 4.0, "root": 2}]
    bounds = [0.0, 1.0, 2.0, 3.0, 4.0]
    out, n = apply_rigid_grid(chords, bounds)
    assert n == 4
    assert [(c["bar"], c["beat"]) for c in out] == [(0, 0), (1, 0), (2, 0), (3, 0)]
    # t0/t1 are the ground-truth onsets — never rewritten, only the labels change
    assert [c["t0"] for c in out] == [0.0, 1.0, 2.0, 3.0]


def test_split_bar_gets_distinct_beats():
    # two chords inside one 4-beat bar -> distinct beats (a split bar)
    chords = [{"t0": 0.0, "t1": 0.5, "root": 0}, {"t0": 0.5, "t1": 1.0, "root": 7}]
    out, n = apply_rigid_grid(chords, [0.0, 1.0], beats_per_bar=4)
    assert n == 1
    assert out[0]["bar"] == 0 and out[0]["beat"] == 0
    assert out[1]["bar"] == 0 and out[1]["beat"] == 2   # halfway -> beat 2 of 4


def test_out_of_range_onsets_clamp():
    chords = [{"t0": -1.0, "root": 0}, {"t0": 99.0, "root": 1}]
    out, n = apply_rigid_grid(chords, [0.0, 1.0, 2.0])
    assert out[0]["bar"] == 0        # before the grid -> bar 0
    assert out[1]["bar"] == n - 1    # after the grid -> last bar


def test_stub_defers_by_default():
    # the finder is a stub until the subagent's algorithm lands: never regrids yet
    assert rigid_grid_for([{"t0": 1.08, "root": 7}], tonic_pc=0) is None
