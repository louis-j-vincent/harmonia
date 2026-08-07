"""tests/test_musx_suggestions.py — the annotation editor's candidate list
must be musx's own ranking (Louis, 2026-08-07).

Pure-numpy: probs are hand-built so the expected posterior is computable by
hand. Layout reminders (span_rescore.py): triad col 0 = N, col i>=1 has root
(i-1)%12 FAST, type (i-1)//12 SLOW over ("maj","min","sus4","sus2","dim",
"aug"); s7 = (none, maj7, b7, bb7); C:maj x b7 folds to QUAL5 "dom".
"""
from __future__ import annotations

import numpy as np
import pytest

from harmonia_min.span_rescore import Q5_TAIL, musx_suggestions

T = 100  # frames (~2.3 s)


def _probs():
    """C:maj 0.60, A:min 0.30, N 0.05 on every frame; s7 mostly bare."""
    triad = np.full((T, 73), 0.05 / 70)
    triad[:, 0] = 0.05                # N
    triad[:, 1] = 0.60                # root C, type maj  -> col 1+0*12+0
    triad[:, 1 + 12 + 9] = 0.30       # root A, type min  -> col 22
    s7 = np.tile([0.90, 0.02, 0.08, 0.0], (T, 1))
    dummy = np.zeros((T, 3))
    return [triad, np.zeros((T, 13)), s7, np.zeros((T, 4)), dummy, dummy]


# Hand-computed posterior over the 60 (root, QUAL5) cells, given a chord:
#   C maj:  0.60*(0.90+0.02+0.0)/0.95 = 0.581
#   A min:  0.30*1.0/0.95            = 0.316
#   C dom:  0.60*0.08/0.95           = 0.0505
C_MAJ, A_MIN, C_DOM = 0.581, 0.316, 0.0505


def test_top3_is_musx_ranking():
    chords = [{"root": 0, "q": "", "nc": False, "t0": 0.5, "t1": 1.5}]
    assert musx_suggestions(_probs(), chords) == 1
    sug = chords[0]["sug"]
    assert [(s["root"], s["q"]) for s in sug] == [(0, ""), (9, "-"), (0, "7")]
    assert sug[0]["c"] == pytest.approx(C_MAJ, abs=0.02)
    assert sug[1]["c"] == pytest.approx(A_MIN, abs=0.02)
    assert sug[2]["c"] == pytest.approx(C_DOM, abs=0.02)
    # real posteriors, descending, never renormalised over the 3
    assert sug[0]["c"] > sug[1]["c"] > sug[2]["c"]
    assert sum(s["c"] for s in sug) < 1.0


def test_written_tail_survives_qual5_fold():
    """A written A-7 whose own (root, family) cell ranks keeps its '-7' tail:
    tapping that orb re-picks the same chord, it must not strip the 7th."""
    chords = [{"root": 9, "q": "-7", "nc": False, "t0": 0.0, "t1": 2.0}]
    musx_suggestions(_probs(), chords)
    a_cell = [s for s in chords[0]["sug"] if s["root"] == 9][0]
    assert a_cell["q"] == "-7"


def test_nc_and_malformed_are_skipped():
    chords = [
        {"root": 0, "q": "", "nc": True, "t0": 0.0, "t1": 1.0},   # N.C.
        {"root": 0, "q": ""},                                     # no span
        {"root": 5, "q": "", "nc": False, "t0": 1.0, "t1": 2.0},
    ]
    assert musx_suggestions(_probs(), chords) == 1
    assert "sug" not in chords[0] and "sug" not in chords[1]
    assert len(chords[2]["sug"]) == 3


def test_tail_vocabulary_is_the_shells():
    """Alternates use exactly the wire tails the shell renders (the same
    mapping /api/context_rescore already sends it)."""
    assert Q5_TAIL == {0: "", 1: "-", 2: "7", 3: "-7b5", 4: "o"}
