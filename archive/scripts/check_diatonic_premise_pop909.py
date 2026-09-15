"""
Premise check pour issue #20 (re-scope POP909) : quelle fraction des GT chords
sont diatoniques dans la clé du morceau ? Corpus : POP909 songs 001–005.

Motivation : jazz1460 n'a que 49.4% de diatonicité (seuil 60% → FAIL). La
question est si POP909 (pop chinois, progressions plus tonales) passe le seuil.

Méthode :
  - Clé GT : key_audio.txt de POP909 (KeyEvent par segment, mode maj/min).
  - Pour chaque ChordEvent dont la clé de la section est connue, on calcule
    le degré (root − tonic) mod 12 et on vérifie si la qualité est dans la
    table diatonique.
  - Table diatonique identique à check_diatonic_premise.py (5 classes :
    major, minor, dom7, maj7, dim) avec maj7/min7 tolérés comme variants.

Seuil de décision (spec nightly) : >= 60% diatonique → PASS.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.data.pop909_parser import POP909Parser, POP909Song, ChordEvent, KeyEvent
from harmonia.theory.chord_vocabulary import ChordQuality

POP909_DIR = REPO / "data" / "pop909" / "POP909"

# ---------------------------------------------------------------------------
# Quality → 5-class mapping (same as jazz script)
# ---------------------------------------------------------------------------

_Q5_MAP: dict[ChordQuality, str] = {
    ChordQuality.MAJOR:      "major",
    ChordQuality.MAJ7:       "maj7",
    ChordQuality.MINOR:      "minor",
    ChordQuality.MIN7:       "minor",   # min7 tolerated as minor
    ChordQuality.DOM7:       "dom7",
    ChordQuality.DOM7SUS4:   "dom7",
    ChordQuality.DIMINISHED: "dim",
    ChordQuality.DIM7:       "dim",
    ChordQuality.HALF_DIM7:  "dim",
}

def quality5(q: ChordQuality) -> str | None:
    """Map ChordQuality to 5-class name, or None for ignored qualities."""
    return _Q5_MAP.get(q)


# ---------------------------------------------------------------------------
# Diatonic tables (degree semitone offset from tonic → allowed q5 names)
# ---------------------------------------------------------------------------

DIATONIC_MAJOR: dict[int, set[str]] = {
    0:  {"major", "maj7"},   # I
    2:  {"minor"},           # ii
    4:  {"minor"},           # iii
    5:  {"major", "maj7"},   # IV
    7:  {"dom7", "major"},   # V (triad or dom7)
    9:  {"minor"},           # vi
    11: {"dim"},             # vii°
}
DIATONIC_MINOR: dict[int, set[str]] = {
    0:  {"minor"},                    # i
    2:  {"dim"},                      # ii°
    3:  {"major", "maj7"},            # bIII
    5:  {"minor"},                    # iv
    7:  {"minor", "dom7", "major"},   # v (natural) / V (harmonic)
    8:  {"major", "maj7"},            # bVI
    10: {"dom7", "major"},            # bVII
    11: {"dim"},                      # vii° (harmonic-minor leading tone)
}


def is_diatonic(root_pc: int, q5: str, tonic: int, mode: str) -> bool:
    deg = (root_pc - tonic) % 12
    table = DIATONIC_MAJOR if mode == "major" else DIATONIC_MINOR
    allowed = table.get(deg)
    if allowed is None:
        return False
    return q5 in allowed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

NOTE = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def score_song(song: POP909Song) -> dict:
    """Return per-song diatonic stats."""
    total = 0
    diatonic = 0
    skipped_no_key = 0
    skipped_no_q5 = 0
    rows = []

    for ev in song.chord_events:
        if ev.root == -1:
            continue  # N (no chord)
        q5 = quality5(ev.quality)
        if q5 is None:
            skipped_no_q5 += 1
            continue

        # Use midpoint of chord event to look up key
        mid = (ev.start_beat + ev.end_beat) / 2.0
        kev = song.key_at_time(mid)
        if kev is None:
            skipped_no_key += 1
            continue

        total += 1
        dia = is_diatonic(ev.root, q5, kev.tonic, kev.mode)
        if dia:
            diatonic += 1
        rows.append((ev.label, kev.label, dia))

    return {
        "song_id": song.song_id,
        "total": total,
        "diatonic": diatonic,
        "skipped_no_key": skipped_no_key,
        "skipped_no_q5": skipped_no_q5,
        "pct": diatonic / total if total else 0.0,
        "rows": rows,
    }


def main() -> None:
    parser = POP909Parser(POP909_DIR)
    song_ids = [f"{i:03d}" for i in range(1, 6)]  # 001–005

    songs = [parser.parse_song(sid) for sid in song_ids]
    songs = [s for s in songs if s is not None]
    print(f"Loaded {len(songs)} POP909 songs: {[s.song_id for s in songs]}\n")

    per_song_stats = [score_song(s) for s in songs]

    # Header
    print(f"{'song_id':<10} {'n_events':>9} {'diatonic':>9} {'pct':>8}  key_events")
    print("-" * 60)
    for st in per_song_stats:
        song = next(s for s in songs if s.song_id == st["song_id"])
        key_info = ", ".join(
            f"{kev.label}[{kev.start_s:.0f}s-{kev.end_s:.0f}s]"
            for kev in song.key_events
        )
        print(
            f"{st['song_id']:<10} {st['total']:>9} {st['diatonic']:>9} "
            f"{st['pct']:>7.1%}  {key_info}"
        )
        if st["skipped_no_key"]:
            print(f"           (skipped {st['skipped_no_key']} events without key coverage)")
        if st["skipped_no_q5"]:
            print(f"           (skipped {st['skipped_no_q5']} events with unmapped quality)")

    total_n = sum(st["total"] for st in per_song_stats)
    total_dia = sum(st["diatonic"] for st in per_song_stats)
    global_pct = total_dia / total_n if total_n else 0.0

    print(f"\n=== PREMISE CHECK — POP909 (issue #20 re-scope) ===")
    print(f"songs scored    : {len(per_song_stats)}")
    print(f"GT chord events : {total_n}")
    print(f"diatonic count  : {total_dia}")
    print(f"diatonic %%      : {global_pct:.1%}")
    verdict = "PASS → implement diatonic prior" if global_pct >= 0.60 else "FAIL → STOP, prior gain too small"
    print(f"\nDECISION (>=60%% → implement): {verdict}")


if __name__ == "__main__":
    main()
