"""Screen BEFORE building: does an extended chord vocabulary still have data?

Louis's proposal (2026-08-02): teach the context prior 7ths/6ths/9ths instead
of collapsing everything to 5 families. The cheap falsifier is coverage — the
trigram key is (q_candidate, dp, q_prev, dn, q_next), so multiplying the
quality alphabet multiplies the key space by ~|Q|^3. If most occurrences land
in keys seen too rarely to be usable, backoff is mandatory, not optional.

Control: the accepted LABEL set is identical across arms (same songs, same
parsed chords) — only the granularity of the quality class changes.

Usage: .venv/bin/python <this>
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO))

import harmonia_min.chord_context_prior as cp  # noqa: E402

# ── the three arms (same label domain, finer classes) ───────────────────────
Q5 = cp.HARTE_QUAL_TO_QUAL5   # 5 classes, what ships today

Q8 = {  # + sevenths and sus, the "cheap win" arm
    "maj": "maj", "maj6": "maj", "maj7": "maj7", "maj9": "maj7",
    "maj13": "maj7", "aug": "maj", "augmaj7": "maj7",
    "sus2": "sus", "sus4": "sus",
    "min": "min", "min6": "min", "min7": "min7", "min9": "min7",
    "min11": "min7", "min13": "min7", "minmaj7": "min7",
    "7": "dom7", "9": "dom7", "11": "dom7", "13": "dom7", "aug7": "dom7",
    "hdim7": "hdim7", "dim": "dim", "dim7": "dim7",
}

Q16 = {  # full extended vocabulary: 6ths and 9ths split out too
    "maj": "maj", "maj6": "maj6", "maj7": "maj7", "maj9": "maj9",
    "maj13": "maj9", "aug": "aug", "augmaj7": "aug",
    "sus2": "sus", "sus4": "sus",
    "min": "min", "min6": "min6", "min7": "min7", "min9": "min9",
    "min11": "min9", "min13": "min9", "minmaj7": "minmaj7",
    "7": "dom7", "9": "dom9", "11": "dom9", "13": "dom13", "aug7": "aug7",
    "hdim7": "hdim7", "dim": "dim", "dim7": "dim7",
}

ARMS = [("QUAL5 (today)", Q5), ("Q8 (+7ths, sus)", Q8), ("Q16 (+6ths, 9ths)", Q16)]
_orig_parse = cp.parse_harte_lite


def make_parser(table: dict[str, str]):
    names = sorted(set(table.values()))
    idx = {n: i for i, n in enumerate(names)}

    def parse(label: str):
        label = label.strip()
        if label in ("N", "n", "NC", "X", "x", "?", ""):
            return None
        pc = cp.NOTE_TO_PC.get(label[0].upper())
        if pc is None:
            return None
        i = 1
        while i < len(label) and label[i] in "#b":
            pc += 1 if label[i] == "#" else -1
            i += 1
        rest = label[i:]
        if rest.startswith(":"):
            q = rest[1:].split("/")[0].split("(")[0]
            if q == "":
                return None
        else:
            q = "maj"
        name = table.get(q)
        if name is None:
            return None
        return pc % 12, idx[name]

    return parse, names


def coverage(songs: dict) -> tuple[Counter, int]:
    """Trigram key counts over all songs; key = (q_c, dp, q_p, dn, q_n)."""
    keys = Counter()
    total = 0
    for seqs in songs.values():
        for seq in seqs:
            for a, b, c in zip(seq, seq[1:], seq[2:]):
                (rp, qp), (rc, qc), (rn, qn) = a, b, c
                keys[(qc, (rp - rc) % 12, qp, (rn - rc) % 12, qn)] += 1
                total += 1
    return keys, total


def main() -> None:
    print(f"{'arm':20s} {'classes':>7s} {'trigrams':>10s} {'distinct':>9s} "
          f"{'>=5 obs':>8s} {'>=10':>7s} {'>=20':>7s}")
    for label, table in ARMS:
        parser, names = make_parser(table)
        cp.parse_harte_lite = parser
        try:
            songs, _stats = cp.load_all_corpus_sequences()
        finally:
            cp.parse_harte_lite = _orig_parse
        keys, total = coverage(songs)
        if not total:
            print(f"{label:20s} NO DATA")
            continue
        def usable(k: int) -> float:
            return sum(c for c in keys.values() if c >= k) / total
        print(f"{label:20s} {len(names):7d} {total:10d} {len(keys):9d} "
              f"{usable(5):7.1%} {usable(10):6.1%} {usable(20):6.1%}")


if __name__ == "__main__":
    main()
