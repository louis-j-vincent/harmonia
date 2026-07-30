"""Transposition unit check for the generic colour tracker.

Rotate This Love's NNLS chroma (both halves) up K semitones, shift the
chart's chord roots/basses by K, set tonic 0 -> K, and verify the decode
is IDENTICAL chord-for-chord (unfolded and structure-folded). Validates
the tonic-relative rotation path with no new audio involved
(CLAUDE.md error-pattern #1: test the load-bearing assumption first).

Usage: .venv/bin/python scratchpad/colour_transpose_check.py [K=3]
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratchpad"))

from colour_hmm_song import (  # noqa: E402
    Song, PC_SHARP, PC_FLAT, SHARP_TONICS, decode, decode_folded, load_song,
)
from harmonia.models import nnls_features as nf  # noqa: E402

K = int(sys.argv[1]) if len(sys.argv) > 1 else 3
SLUG = "maroon_5_this_love"


def main() -> None:
    song = load_song(SLUG)
    assert song.tonic == 0
    arr, times = nf.extract_bothchroma(REPO / "docs/audio" / f"{SLUG}.m4a")

    # transpose audio chroma up K semitones (A-first blocks: pc p -> p+K)
    arr2 = np.concatenate(
        [np.roll(arr[:, :12], K, axis=1), np.roll(arr[:, 12:], K, axis=1)], axis=1
    )
    chords2 = copy.deepcopy(song.chords)
    for c in chords2:
        if "root" in c:
            c["root"] = (c["root"] + K) % 12
        if c.get("bass", -1) >= 0:
            c["bass"] = (c["bass"] + K) % 12
    t2 = (song.tonic + K) % 12
    song2 = Song(slug=f"{SLUG}+{K}", chords=chords2, tonic=t2, mode="minor",
                 key_name=f"transposed+{K}", bpb=song.bpb,
                 pcn=PC_SHARP if t2 in SHARP_TONICS else PC_FLAT)

    p1, *_ = decode(arr, times, song.chords, song)
    p2, *_ = decode(arr2, times, chords2, song2)
    diffs = [i for i in range(len(p1)) if p1[i] != p2[i]]
    print(f"unfolded decode: {len(diffs)} diffs / {len(p1)} chords "
          f"(K={K})  ->  {'PASS' if not diffs else 'FAIL ' + str(diffs[:10])}")

    f1 = decode_folded(arr, times, song.chords, song)
    f2 = decode_folded(arr2, times, chords2, song2)
    fdiffs = [i for i in range(len(f1[0])) if f1[0][i] != f2[0][i]]
    print(f"folded decode:   {len(fdiffs)} diffs / {len(f1[0])} chords "
          f"(form1='{f1[6]}'  form2='{f2[6]}')  ->  "
          f"{'PASS' if not fdiffs and f1[6] == f2[6] else 'FAIL'}")


if __name__ == "__main__":
    main()
