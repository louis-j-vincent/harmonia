"""Learn note→chord emission weights from the accompaniment DB's MIDI voicings.

H1 — the documented bottleneck (docs/known_issues.md #1): the emission model
cannot discriminate similar chord qualities, and the hand templates in
chord_vocabulary.py (root 1.0 / 3rd 0.85 / 5th 0.3 / 7th 0.85) are guesses.
Here the MIDI is noise-free and the chord label is ground truth by construction,
so we can measure:

  1. What accompaniment voicings *actually* weight, per quality, per register
     (comping tracks vs bass track).
  2. How separable confusable qualities are with PERFECT note evidence
     (oracle ceiling for quality classification, hand vs learned templates).

Caveat stated up front: MMA's voicings are algorithmic groove libraries, not
human performances — treat learned weights as "what a competent accompanist
pattern voices", cleaner than reality but directionally informative.

Usage: .venv/bin/python scripts/analyze_accomp_emission.py [--max-songs N]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pretty_midi

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

DB = REPO / "data" / "accomp_db" / "db.jsonl"

NOTE_TO_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# MMA chord-name quality → canonical bucket used for the emission analysis.
# Extensions collapse onto their 7th-chord base; '6' stays its own bucket
# (deliberately: C6 is pitch-identical to Am7, a known confusable).
QUALITY_MAP = {
    "": "maj", "M": "maj", "M7": "maj7", "M9": "maj7", "M13": "maj7", "M7#11": "maj7",
    "M6": "6", "6": "6", "69": "6", "6(add9)": "6",
    "m": "min", "m6": "m6", "m69": "m6", "m(add9)": "min",
    "m7": "min7", "m9": "min7", "m11": "min7", "m13": "min7",
    "mM7": "minmaj7", "mM7(add9)": "minmaj7",
    "m7b5": "m7b5", "m9b5": "m7b5", "m11b5": "m7b5",
    "dim": "dim", "dim7": "dim7", "dim7(addM7)": "dim7",
    "aug": "aug", "7#5": "aug7", "aug7": "aug7", "M7#5": "augmaj7",
    "7": "dom7", "9": "dom7", "13": "dom7", "11": "dom7",
    "7b9": "dom7alt", "7#9": "dom7alt", "7alt": "dom7alt", "7b5": "dom7alt",
    "7#11": "dom7alt", "7b13": "dom7alt", "7b9#5": "dom7alt", "7b5b9": "dom7alt",
    "7#5#9": "dom7alt", "7b9b13": "dom7alt", "7b9#11": "dom7alt", "7#9#11": "dom7alt",
    "13b9": "dom7alt", "13#9": "dom7alt", "13#11": "dom7alt", "9#11": "dom7alt",
    "9b5": "dom7alt", "9#5": "dom7alt",
    "sus4": "sus4", "sus": "sus4", "sus2": "sus2", "(add9)": "maj",
    "7sus4": "7sus4", "7sus": "7sus4", "9sus4": "7sus4", "13sus4": "7sus4",
    "7susb9": "7sus4", "7b9sus": "7sus4",
    "5": "maj",
}

# Interval sets for the oracle-classification templates (hand version) — mirrors
# chord_vocabulary.py CHORD_TEMPLATES weights for the qualities present here.
HAND_TEMPLATES = {
    "maj": {0: 1.0, 4: 0.85, 7: 0.35},
    "min": {0: 1.0, 3: 0.85, 7: 0.35},
    "dim": {0: 1.0, 3: 0.85, 6: 0.85},
    "aug": {0: 1.0, 4: 0.85, 8: 0.85},
    "sus2": {0: 1.0, 2: 0.85, 7: 0.35},
    "sus4": {0: 1.0, 5: 0.85, 7: 0.35},
    "maj7": {0: 1.0, 4: 0.85, 7: 0.3, 11: 0.85},
    "min7": {0: 1.0, 3: 0.85, 7: 0.3, 10: 0.85},
    "dom7": {0: 1.0, 4: 0.85, 7: 0.3, 10: 0.85},
    "minmaj7": {0: 1.0, 3: 0.85, 7: 0.3, 11: 0.85},
    "m7b5": {0: 1.0, 3: 0.85, 6: 0.85, 10: 0.85},
    "dim7": {0: 1.0, 3: 0.85, 6: 0.85, 9: 0.85},
    "augmaj7": {0: 1.0, 4: 0.85, 8: 0.85, 11: 0.85},
    "aug7": {0: 1.0, 4: 0.85, 8: 0.85, 10: 0.85},
    "7sus4": {0: 1.0, 5: 0.85, 7: 0.3, 10: 0.85},
    # no hand template exists for these two — nearest phase-1 stand-ins
    "6": {0: 1.0, 4: 0.85, 7: 0.35, 9: 0.6},
    "m6": {0: 1.0, 3: 0.85, 7: 0.35, 9: 0.6},
    "dom7alt": {0: 1.0, 4: 0.85, 10: 0.85, 1: 0.4, 3: 0.4},
}

INTERVAL_NAMES = ["R", "b2", "2", "b3", "3", "4", "b5", "5", "#5", "6", "b7", "7"]


def parse_chord(chord: str) -> tuple[int, str] | None:
    """MMA chord name → (root pc, quality bucket)."""
    if not chord or chord == "z":
        return None
    pc = NOTE_TO_PC.get(chord[0])
    if pc is None:
        return None
    body = chord[1:]
    if body[:1] in ("#", "b"):
        pc += 1 if body[0] == "#" else -1
        body = body[1:]
    body = body.split("/")[0]
    bucket = QUALITY_MAP.get(body)
    if bucket is None:
        return None
    return pc % 12, bucket


def song_chord_spans(rec: dict) -> list[tuple[float, float, int, str]]:
    """Merged chord events as (t0, t1, root_pc, quality)."""
    bpb = rec["beats_per_bar"]
    spb = 60.0 / rec["tempo"]
    slots = sorted(
        ((ev["bar"] - 1) * bpb + ev["beat"], ev["mma"]) for ev in rec["chord_timeline"]
    )
    n_beats = rec["n_bars"] * bpb
    spans = []
    for i, (beat, chord) in enumerate(slots):
        end_beat = slots[i + 1][0] if i + 1 < len(slots) else n_beats
        parsed = parse_chord(chord)
        if parsed is None:
            continue
        # merge with previous span if identical chord
        if spans and spans[-1][2:] == parsed and abs(spans[-1][1] - beat * spb) < 1e-6:
            spans[-1] = (spans[-1][0], end_beat * spb, *parsed)
        else:
            spans.append((beat * spb, end_beat * spb, *parsed))
    return spans


def pc_vector(notes: list, t0: float, t1: float) -> np.ndarray:
    """Duration-weighted pitch-class salience of notes overlapping [t0, t1)."""
    v = np.zeros(12)
    for n in notes:
        ov = min(n.end, t1) - max(n.start, t0)
        if ov > 0:
            v[n.pitch % 12] += ov
    return v


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-songs", type=int, default=None)
    ap.add_argument("--corpus", default="jazz1460")
    args = ap.parse_args()

    records = [json.loads(line) for line in open(DB)]
    songs = [r for r in records if r["corpus"] == args.corpus]
    if args.max_songs:
        songs = songs[: args.max_songs]

    # accumulators: quality → summed interval vector (comping / bass separately)
    comp_acc = defaultdict(lambda: np.zeros(12))
    bass_acc = defaultdict(lambda: np.zeros(12))
    presence = defaultdict(lambda: np.zeros(12))  # instance-level presence count
    counts = Counter()
    # per-instance vectors for the oracle classification (subsampled)
    instances: list[tuple[str, np.ndarray, int]] = []  # (quality, comp interval vec, song_idx)

    for si, rec in enumerate(songs):
        mid = REPO / rec["midi_path"]
        if not mid.exists():
            continue
        try:
            pm = pretty_midi.PrettyMIDI(str(mid))
        except Exception:
            continue
        comp_notes = [n for i in pm.instruments
                      if not i.is_drum and "bass" not in i.name.lower()
                      for n in i.notes]
        bass_notes = [n for i in pm.instruments
                      if not i.is_drum and "bass" in i.name.lower()
                      for n in i.notes]
        comp_notes.sort(key=lambda n: n.start)
        for t0, t1, root, qual in song_chord_spans(rec):
            cv = pc_vector(comp_notes, t0, t1)
            bv = pc_vector(bass_notes, t0, t1)
            if cv.sum() < 1e-9:
                continue
            cv_rel = np.roll(cv, -root)
            bv_rel = np.roll(bv, -root)
            comp_acc[qual] += cv_rel
            bass_acc[qual] += bv_rel
            presence[qual] += (cv_rel / cv_rel.sum()) > 0.05
            counts[qual] += 1
            instances.append((qual, cv_rel / cv_rel.sum(), si))

    top_quals = [q for q, _ in counts.most_common(14)]
    print(f"{len(songs)} songs, {sum(counts.values())} chord instances; "
          f"top qualities: {counts.most_common(10)}\n")

    # ── 1. empirical comping weights vs hand templates ─────────────────────────
    print("H1a — Empirical comping-voicing interval weights (root-normalized), "
          "vs hand template in brackets:")
    header = "    qual      n     " + " ".join(f"{n:>5}" for n in INTERVAL_NAMES)
    print(header)
    emp_templates = {}
    for q in top_quals:
        v = comp_acc[q] / max(comp_acc[q][0], 1e-9)
        emp_templates[q] = comp_acc[q] / comp_acc[q].sum()
        hand = HAND_TEMPLATES.get(q, {})
        cells = []
        for i in range(12):
            e = f"{v[i]:.2f}" if v[i] >= 0.05 else "  ."
            h = f"({hand[i]:.2f})" if i in hand else ""
            cells.append(f"{e:>5}{h:<6}")
        print(f"    {q:<9}{counts[q]:>6} " + "".join(cells))
    print()

    print("H1b — Interval presence rate (fraction of instances where the interval "
          "carries >5% of comping mass):")
    for q in top_quals[:8]:
        p = presence[q] / counts[q]
        marks = " ".join(f"{INTERVAL_NAMES[i]}:{p[i]:.0%}" for i in range(12) if p[i] > 0.10)
        print(f"    {q:<9} {marks}")
    print()

    # ── 2. register question: where does root evidence live? ──────────────────
    tot_comp = sum(comp_acc[q] for q in top_quals)
    tot_bass = sum(bass_acc[q] for q in top_quals)
    print("H1c — Register: share of each interval's total evidence that comes from "
          "the bass track:")
    for i in [0, 3, 4, 7, 10, 11]:
        b, c = tot_bass[i], tot_comp[i]
        print(f"    {INTERVAL_NAMES[i]:>3}: bass {b/(b+c):5.1%}   "
              f"(root share within bass track: {tot_bass[i]/tot_bass.sum():.1%})"
              if i == 0 else
              f"    {INTERVAL_NAMES[i]:>3}: bass {b/(b+c):5.1%}")
    print()

    # ── 3. separability of confusable qualities ────────────────────────────────
    def cos(a, b):
        return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

    def tvec(d):
        v = np.zeros(12)
        for i, w in d.items():
            v[i] = w
        return v

    pairs = [("dom7", "7sus4"), ("sus4", "7sus4"), ("dom7", "sus4"),
             ("maj", "maj7"), ("maj", "6"), ("maj7", "6"),
             ("min", "min7"), ("min7", "m7b5"), ("dom7", "dom7alt"),
             ("min7", "6"), ("maj", "min")]
    print("H1d — Cosine similarity of confusable quality pairs "
          "(hand templates vs learned voicing templates):")
    print(f"    {'pair':<18} {'hand':>6} {'learned':>8}")
    for a, b in pairs:
        if a in emp_templates and b in emp_templates:
            h = cos(tvec(HAND_TEMPLATES[a]), tvec(HAND_TEMPLATES[b]))
            e = cos(emp_templates[a], emp_templates[b])
            print(f"    {a+' vs '+b:<18} {h:>6.3f} {e:>8.3f}")
    print()

    # ── 4. oracle quality classification (perfect notes, known root) ──────────
    rng = np.random.default_rng(0)
    split = {si: rng.random() < 0.5 for si in range(len(songs))}
    train = [(q, v) for q, v, si in instances if split[si]]
    test = [(q, v, si) for q, v, si in instances if not split[si]]
    # learned templates from train half
    learned = defaultdict(lambda: np.zeros(12))
    tn = Counter()
    for q, v in train:
        learned[q] += v
        tn[q] += 1
    learned = {q: learned[q] / tn[q] for q in top_quals if tn[q] >= 20}
    quals = sorted(learned.keys())
    hand_m = np.stack([tvec(HAND_TEMPLATES[q]) for q in quals])
    hand_m /= np.linalg.norm(hand_m, axis=1, keepdims=True)
    learn_m = np.stack([learned[q] for q in quals])
    learn_m /= np.linalg.norm(learn_m, axis=1, keepdims=True)

    conf_hand = Counter()
    conf_learn = Counter()
    n_ok_h = n_ok_l = n_test = 0
    for q, v, _ in test:
        if q not in learned:
            continue
        vn = v / (np.linalg.norm(v) + 1e-12)
        ph = quals[int(np.argmax(hand_m @ vn))]
        pl = quals[int(np.argmax(learn_m @ vn))]
        n_test += 1
        n_ok_h += ph == q
        n_ok_l += pl == q
        if ph != q:
            conf_hand[(q, ph)] += 1
        if pl != q:
            conf_learn[(q, pl)] += 1
    print(f"H1e — Oracle quality classification ceiling ({n_test} test instances, "
          f"{len(quals)} qualities, TRUE root given, perfect MIDI notes):")
    print(f"    hand templates   : {n_ok_h/n_test:.1%}")
    print(f"    learned templates: {n_ok_l/n_test:.1%}")
    print("    top hand-template confusions:   ",
          [(f"{a}→{b}", n) for (a, b), n in conf_hand.most_common(6)])
    print("    top learned-template confusions:",
          [(f"{a}→{b}", n) for (a, b), n in conf_learn.most_common(6)])

    # collapsed to majmin-style buckets (what MIREX majmin actually scores)
    def collapse(q):
        return {"maj": "maj", "maj7": "maj", "6": "maj", "aug": "maj", "augmaj7": "maj",
                "min": "min", "min7": "min", "m6": "min", "minmaj7": "min",
                "dom7": "dom", "dom7alt": "dom", "aug7": "dom", "7sus4": "dom",
                "sus4": "other", "sus2": "other", "m7b5": "dim", "dim": "dim",
                "dim7": "dim"}.get(q, "other")

    ok_h = ok_l = 0
    for q, v, _ in test:
        if q not in learned:
            continue
        vn = v / (np.linalg.norm(v) + 1e-12)
        ph = quals[int(np.argmax(hand_m @ vn))]
        pl = quals[int(np.argmax(learn_m @ vn))]
        ok_h += collapse(ph) == collapse(q)
        ok_l += collapse(pl) == collapse(q)
    print(f"    collapsed to maj/min/dom/dim/other: hand {ok_h/n_test:.1%}, "
          f"learned {ok_l/n_test:.1%}")


if __name__ == "__main__":
    main()
