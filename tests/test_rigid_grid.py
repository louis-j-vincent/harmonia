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


# ── the 2x metrical-octave defect (Louis, 2026-07-30) ─────────────────────────
# "Fix the 2x octave issue it's a real problem!"  Norah Jones, "Don't Know Why":
# 88 BPM 4/4, bar 2.72 s, 67 bars.  The chart rendered 134 bars of 1.36 s —
# every displayed "bar" was half a bar.  Cause: `rigid_grid_for` recovers the
# CHORD-CHANGE period, and this tune changes chord twice per bar, so the finder
# locks the half-bar.  From onsets alone that is genuinely ambiguous; the cue
# that breaks it is beat_this's NATIVE downbeat spacing, passed in as
# `bar_ref_sec`.

def _dont_know_why_chords(n_phrases: int = 8) -> list[dict]:
    """The tune's A phrase, at its real timings: | Bb | Eb^7 D | Gm7 C7 | F7 |.

    Bar = 2.72 s (88 BPM, 4/4).  Bars 1 and 3 hold one chord for the whole bar;
    bars 2 and 3 split into two half-bar chords — so the MODAL chord-change
    interval is 1.36 s, exactly half the bar.  That is the trap.
    """
    bar = 2.72
    phrase = [(0.0, 2.72, 10), (2.72, 1.36, 3), (4.08, 1.36, 2),
              (5.44, 1.36, 7), (6.80, 1.36, 0), (8.16, 2.72, 5)]
    out = []
    for p in range(n_phrases):
        base = p * 4 * bar
        for off, dur, root in phrase:
            out.append({"t0": round(base + off, 4), "t1": round(base + off + dur, 4),
                        "root": root, "q": ""})
    return out


def _bar_len(bounds: list[float]) -> float:
    import statistics
    return statistics.median(b - a for a, b in zip(bounds, bounds[1:]))


def test_dont_know_why_halves_the_bar_without_the_cue():
    """RED-FIRST: pins the defect.  With no external cue the finder returns the
    half-bar (1.36 s), i.e. twice as many bars as the song has."""
    grid = rigid_grid_for(_dont_know_why_chords())
    assert grid is not None
    assert abs(_bar_len(grid) - 1.36) < 0.10, _bar_len(grid)


def test_downbeat_cue_breaks_the_octave():
    """beat_this measures 2.72 s between downbeats; the grid must follow it."""
    grid = rigid_grid_for(_dont_know_why_chords(), bar_ref_sec=2.72)
    assert grid is not None
    assert abs(_bar_len(grid) - 2.72) < 0.10, _bar_len(grid)


def test_cue_does_not_move_an_already_correct_grid():
    """This Love-shaped: one chord per 2.52 s bar plus a mid-bar passing chord.
    The finder already gets this right — the cue must agree, not perturb it."""
    bar = 2.52
    chords = []
    for p in range(10):
        base = p * 4 * bar
        for i, root in enumerate((7, 0, 5, 2)):
            chords.append({"t0": round(base + i * bar, 4),
                           "t1": round(base + (i + 1) * bar, 4), "root": root})
    before = rigid_grid_for(chords)
    after = rigid_grid_for(chords, bar_ref_sec=bar)
    assert before is not None and after is not None
    assert abs(_bar_len(before) - bar) < 0.10
    assert abs(_bar_len(after) - bar) < 0.10


def test_unreliable_cue_is_ignored():
    """A cue that matches no metrical octave of the recovered grid, and is not
    itself plausible, must not silently replace a good grid: the finder falls
    back to its own answer rather than inventing one."""
    chords = _dont_know_why_chords()
    assert rigid_grid_for(chords, bar_ref_sec=None) is not None
    assert rigid_grid_for(chords, bar_ref_sec=0.0) is not None
