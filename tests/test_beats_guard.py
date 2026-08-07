"""harmonia_min.beats.check_grid — the metre guard after the waltz opening.

2026-08-07 (Louis: quart ET tiers de barre affichables) : a detected metre of
3 is a legitimate waltz and passes; 2 stays refused — it is the half-tempo /
octave-error signature the guard was built for (Georgia On My Mind); low
consistency stays refused whatever the metre.
"""
from __future__ import annotations

import numpy as np
import pytest

from harmonia_min.beats import (BeatTrackingError, check_grid, repair_grid)


# ── 2026-08-07 second pass: the guard was refusing the MEASURE ───────────────
# Red-first tests for the over-marked-downbeat repair. Each one FAILS against
# the pre-repair guard, which read an extra bar-start on beat 3 as a broken
# grid (Close to You 64%, Land of 1000 Dances 62%, Georgia 45% "metre 2").

def _split_bars(metre: int, n_bars: int, step: float = 0.5, every: int = 2):
    """A perfect metre-beat grid where the tracker ALSO marks the mid-bar beat
    of every `every`-th bar — Beat This!'s real failure on this corpus."""
    beats = np.arange(n_bars * metre + 1) * step
    db = []
    for k in range(n_bars):
        db.append(beats[k * metre])
        if k % every == 0:
            db.append(beats[k * metre + metre // 2])
    return beats, np.asarray(sorted(db))


def test_mid_bar_downbeats_no_longer_refused():
    """Close to You {4:63, 2:35} / Land of 1000 Dances {4:64, 2:40}:
    metronomic beats, extra bar starts on beat 3 in a MINORITY of bars. The
    raw statistic reads ~0.6 consistency; the grid is fine."""
    beats, db = _split_bars(4, 60, every=2)          # every 2nd bar split 2+2
    q = check_grid(beats, db, "close_to_you.m4a")
    assert q["metre"] == 4
    assert q["coverage"] > 0.99
    assert q["raw_consistency"] < 0.80               # what used to refuse it


def test_mid_bar_downbeats_in_the_majority_still_repaired():
    """Georgia / Blue Bossa: MORE than half the bars split, so the raw mode
    flips to 2 and the old guard called it a half-tempo error. It is not —
    4-beat bars are still directly visible in the tracker's own gaps."""
    beats, db = _split_bars(4, 60, every=1)          # every bar split 2+2 …
    unsplit = {float(beats[k * 4 + 2]) for k in range(0, 60, 3)}
    db = np.asarray([t for t in db if float(t) not in unsplit])   # … but 1 in 3
    from collections import Counter
    ib = [int(np.argmin(np.abs(beats - t))) for t in db]
    assert Counter(np.diff(ib).tolist()).most_common(1)[0][0] == 2   # mode IS 2
    q = check_grid(beats, db, "georgia.m4a")
    assert q["metre"] == 4 and q["coverage"] > 0.99
    assert q["raw_metre"] == 2                       # what used to refuse it


def test_repair_returns_a_clean_subset_of_the_tracker_downbeats():
    beats, db = _split_bars(4, 60, every=2)
    r = repair_grid(beats, db)
    assert set(r["downbeats"]) <= {round(float(t), 4) for t in db}
    gaps = np.diff([int(np.argmin(np.abs(beats - t))) for t in r["downbeats"]])
    assert set(gaps.tolist()) == {4}
    assert r["kept"] < 0.85                          # the 2+2 splits dropped


def test_repair_never_moves_a_healthy_grid():
    beats, downbeats = _grid(4, n_bars=60)
    r = repair_grid(beats, downbeats)
    assert r["downbeats"] == [round(float(t), 4) for t in downbeats]


def test_a_real_half_tempo_lock_is_still_refused():
    """The half-tempo protection, and the trap in the repair itself: a
    downbeat every 2 beats tiles PERFECTLY at 4 (coverage 1.0), so coverage
    alone would wave the octave error straight through. What refuses it is
    `direct` — not one bar of 4 beats is visible in the tracker's own gaps."""
    beats, downbeats = _grid(2, n_bars=60)
    r = repair_grid(beats, downbeats)
    assert r["coverage"] == 1.0        # coverage does NOT catch this
    assert r["direct"] == 0.0          # this does
    with pytest.raises(BeatTrackingError, match="2 beats per bar"):
        check_grid(beats, downbeats, "georgia.m4a")


def _grid(metre: int, n_bars: int = 40, step: float = 0.5):
    beats = np.arange(n_bars * metre + 1) * step
    downbeats = beats[::metre]
    return beats, downbeats


def test_waltz_metre_3_passes():
    beats, downbeats = _grid(3)
    q = check_grid(beats, downbeats, "waltz.m4a")
    assert q["metre"] == 3 and q["consistency"] > 0.95


def test_metre_4_still_passes():
    beats, downbeats = _grid(4)
    assert check_grid(beats, downbeats, "pop.m4a")["metre"] == 4


def test_metre_2_still_refused():
    beats, downbeats = _grid(2)
    with pytest.raises(BeatTrackingError, match="2 beats per bar"):
        check_grid(beats, downbeats, "georgia.m4a")


def test_unvalidated_metres_still_refused():
    for metre in (5, 6, 7):
        beats, downbeats = _grid(metre)
        with pytest.raises(BeatTrackingError):
            check_grid(beats, downbeats, f"odd{metre}.m4a")


def test_loose_grid_still_refused_even_in_3():
    beats, downbeats = _grid(3, n_bars=40)
    # corrupt 30% of the downbeats so bars stop tiling in 3s
    db = list(downbeats)
    for i in range(3, len(db), 3):
        db[i] += 0.5   # one beat late
    with pytest.raises(BeatTrackingError, match="too loose or rubato"):
        check_grid(beats, np.asarray(db), "loose.m4a")


def test_short_song_lets_through():
    beats, downbeats = _grid(2, n_bars=10)   # under GRID_MIN_BARS: no verdict
    assert check_grid(beats, downbeats, "snippet.m4a")["n_bars"] < 30
