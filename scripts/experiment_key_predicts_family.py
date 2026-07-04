"""How well does the KEY predict a chord's family (major/minor/dim), given the root?

User's hypothesis (2026-07-04): the strongest lever for the third — the note we
can't hear well from audio — is the key. If we know the song is in C major and
the bass says the root is D, music theory says the third is F (minor) → D minor,
without needing to hear the third at all.

This measures the ceiling of that idea, symbolically (db.jsonl only, no audio):
for every chord in the jazz corpus, predict its family purely from
(scale-degree-of-root, key), and compare to the true family. Baseline to beat:
family recovered from real audio alone was ~81% (docs/chord_tree_2026-07-04.md).

Usage: .venv/bin/python scripts/experiment_key_predicts_family.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from analyze_accomp_emission import parse_chord  # noqa: E402
from analyze_accomp_priors import parse_key  # noqa: E402

DB = REPO / "data" / "accomp_db" / "db.jsonl"

# MMA quality bucket → chord family
BUCKET_FAMILY = {
    "maj": "major", "maj7": "major", "6": "major", "dom7": "major", "dom7alt": "major",
    "min": "minor", "min7": "minor", "m6": "minor", "minmaj7": "minor",
    "dim": "diminished", "dim7": "diminished", "m7b5": "diminished",
    "aug": "augmented", "aug7": "augmented", "augmaj7": "augmented",
    "sus2": "suspended", "sus4": "suspended", "7sus4": "suspended",
}

# Diatonic triad family by scale degree (semitones above the tonic).
# Major key: I ii iii IV V vi vii°
MAJOR_DEGREE_FAMILY = {0: "major", 2: "minor", 4: "minor", 5: "major",
                       7: "major", 9: "minor", 11: "diminished"}
# Minor key: use harmonic-minor V (jazz almost always makes the v a dominant),
# else natural minor. i ii° bIII iv V bVI bVII
MINOR_DEGREE_FAMILY = {0: "minor", 2: "diminished", 3: "major", 5: "minor",
                       7: "major", 8: "major", 10: "major"}


def main() -> None:
    records = [json.loads(line) for line in open(DB)]
    jazz = [r for r in records if r["corpus"] == "jazz1460"]

    total = diatonic = 0
    key_correct = 0                 # key-only prediction correct (diatonic roots)
    non_diatonic = 0
    fam_counts = Counter()
    # confusion when key is wrong
    key_wrong = Counter()
    # how often the audio-hard distinction (major vs minor family) is settled by key
    third_settled = third_total = 0

    for rec in jazz:
        k = parse_key(rec["key"])
        if k is None:
            continue
        tonic, mode = k
        table = MAJOR_DEGREE_FAMILY if mode == "major" else MINOR_DEGREE_FAMILY
        seen = set()
        for ev in rec["chord_timeline"]:
            parsed = parse_chord(ev["mma"])
            if parsed is None:
                continue
            root, bucket = parsed
            true_fam = BUCKET_FAMILY.get(bucket)
            if true_fam is None:
                continue
            key0 = (ev["bar"], ev["beat"], ev["mma"])
            if key0 in seen:
                continue
            seen.add(key0)
            total += 1
            fam_counts[true_fam] += 1
            deg = (root - tonic) % 12
            pred = table.get(deg)
            if pred is None:
                non_diatonic += 1
                continue
            diatonic += 1
            if pred == true_fam:
                key_correct += 1
            else:
                key_wrong[(true_fam, pred)] += 1
            # "third settled": for chords whose family is major or minor (the
            # audio-hard third distinction), does the key pin it correctly?
            if true_fam in ("major", "minor"):
                third_total += 1
                if pred == true_fam:
                    third_settled += 1

    print(f"Jazz corpus: {total} chord instances\n")
    print(f"True family mix: " + ", ".join(
        f"{f} {n/total:.0%}" for f, n in fam_counts.most_common()))
    print()
    print(f"Root is diatonic to the key: {diatonic/total:.1%} of chords "
          f"({non_diatonic/total:.1%} non-diatonic, key can't predict → defer to audio)")
    print()
    print("KEY-ONLY family prediction (on diatonic roots, zero audio):")
    print(f"    accuracy: {key_correct/diatonic:.1%}")
    print(f"    coverage-weighted over ALL chords "
          f"(non-diatonic counted as 'defer'): {key_correct/total:.1%} decided correctly, "
          f"{non_diatonic/total:.1%} deferred to audio")
    print()
    print(f"The audio-hard distinction (major-vs-minor family): key settles "
          f"{third_settled/third_total:.1%} of them correctly, on diatonic roots.")
    print()
    print("When the hand diatonic table mispredicts (true → guess), the informative cases:")
    for (t, p), n in key_wrong.most_common(6):
        print(f"    {t} chord on a scale degree that's usually {p}: {n} ({n/diatonic:.1%})")

    # ── learned key→family table (style-adapted): most common family per degree ─
    from collections import defaultdict
    deg_fam = defaultdict(Counter)   # (mode, degree) → family counts
    for rec in jazz:
        k = parse_key(rec["key"])
        if k is None:
            continue
        tonic, mode = k
        seen = set()
        for ev in rec["chord_timeline"]:
            parsed = parse_chord(ev["mma"])
            if parsed is None:
                continue
            root, bucket = parsed
            fam = BUCKET_FAMILY.get(bucket)
            if fam is None:
                continue
            key0 = (ev["bar"], ev["beat"], ev["mma"])
            if key0 in seen:
                continue
            seen.add(key0)
            deg_fam[(mode, (root - tonic) % 12)][fam] += 1
    # accuracy of predicting the corpus-modal family for each (mode, degree)
    learned_correct = learned_total = 0
    for (mode, deg), counts in deg_fam.items():
        learned_correct += counts.most_common(1)[0][1]
        learned_total += sum(counts.values())
    print()
    print("LEARNED key→family table (corpus-modal family per scale degree, "
          "style-adapted, still zero audio):")
    print(f"    accuracy over ALL chords: {learned_correct/learned_total:.1%}")
    print("    (the jump over the hand table = jazz's secondary dominants, "
          "learned rather than assumed)")
    # show the learned major-key table
    print("\n    Learned major-key family by degree (share of the winning family):")
    names = {0: "I", 1: "bII", 2: "II", 3: "bIII", 4: "III", 5: "IV", 6: "#IV",
             7: "V", 8: "bVI", 9: "VI", 10: "bVII", 11: "VII"}
    for deg in range(12):
        c = deg_fam.get(("major", deg))
        if c and sum(c.values()) > 200:
            fam, n = c.most_common(1)[0]
            print(f"      {names[deg]:>4} (deg {deg:>2}): {fam:<11} {n/sum(c.values()):.0%}")


if __name__ == "__main__":
    main()
