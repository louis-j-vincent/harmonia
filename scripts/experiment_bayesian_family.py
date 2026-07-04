"""Combine every clue for the chord family, measure each one's weight + overlap.

This is the skeleton of the Bayesian chord model, tested on the family (the
third — major/min/dim/aug/sus) because that's the bottleneck. For each chord we
build one probability distribution over families from each independent clue:

  AUDIO  — P(family | notes heard)          : softmax of template match (likelihood)
  KEY    — P(family | scale-degree, key)    : learned per-degree table (the third lever)
  PROG   — P(family | root motion prev→now) : learned, the ii-V-I / progression signal
  FOLD   — AUDIO, but averaged over the repeats of this slot in the song's form
           (multiple observations of the same chord → a cleaner likelihood)

They are combined as a weighted sum of log-probabilities (a log-linear / Bayesian
pool): score(f) = Σ_clue w_clue · log P_clue(f). The weights w are FIT on training
songs — that fitting IS the estimate of how much to trust each clue. We then report:

  * each clue alone,
  * the fitted weights and combined accuracy,
  * leave-one-clue-out ablation (how much unique information each adds),
  * accuracy split by diatonic vs out-of-scale chords (where KEY must defer to AUDIO).

Everything uses cached Basic Pitch activations, ground-truth key/root/structure.

Usage: .venv/bin/python scripts/experiment_bayesian_family.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from analyze_accomp_emission import parse_chord, song_chord_spans  # noqa: E402
from analyze_accomp_priors import parse_key  # noqa: E402
from learn_stage1_mapping import pool_beats, to_chroma  # noqa: E402
from harmonia.models.stage1_pitch import PitchExtractor  # noqa: E402

DB = REPO / "data" / "accomp_db" / "db.jsonl"
MANIFEST = REPO / "data" / "accomp_db" / "audio" / "manifest.jsonl"

BUCKET_FAMILY = {
    "maj": "major", "maj7": "major", "6": "major", "dom7": "major", "dom7alt": "major",
    "min": "minor", "min7": "minor", "m6": "minor", "minmaj7": "minor",
    "dim": "diminished", "dim7": "diminished", "m7b5": "diminished",
    "aug": "augmented", "aug7": "augmented", "augmaj7": "augmented",
    "sus2": "suspended", "sus4": "suspended", "7sus4": "suspended",
}
FAMILIES = ["major", "minor", "diminished", "augmented", "suspended"]
FI = {f: i for i, f in enumerate(FAMILIES)}
DIATONIC = {0, 2, 4, 5, 7, 9, 11}   # major-scale degrees (rough diatonic test)


def normed(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def section_runs(section_per_bar):
    """List of (start_bar, end_bar, label) for each contiguous section run."""
    runs = []
    i = 0
    while i < len(section_per_bar):
        j = i
        while j < len(section_per_bar) and section_per_bar[j] == section_per_bar[i]:
            j += 1
        runs.append((i, j, section_per_bar[i]))
        i = j
    return runs


class Inst:
    __slots__ = ("fam", "chroma", "mode", "deg", "prev_deg", "song", "wav",
                 "sec", "pos")

    def __init__(self, **kw):
        for k, v in kw.items():
            setattr(self, k, v)


def build_instances():
    records = {r["song_id"]: r for r in map(json.loads, open(DB))}
    manifest = [json.loads(line) for line in open(MANIFEST)]
    extractor = PitchExtractor(cache_dir=REPO / "data" / "cache" / "accomp")
    out = []
    for m in manifest:
        wav = REPO / m["wav"]
        if not wav.exists():
            continue
        rec = records[m["song_id"]]
        k = parse_key(rec["key"])
        if k is None:
            continue
        tonic, mode = k
        spb = 60.0 / m["tempo"]
        bpb = m["beats_per_bar"]
        n_beats = m["n_bars"] * bpb
        try:
            acts = extractor.extract(wav)
        except Exception:
            continue
        au_c = to_chroma(pool_beats(acts.frame_times, acts.onset_probs, n_beats, spb))
        chord_at_beat = {(ev["bar"] - 1) * bpb + ev["beat"]: ev["mma"]
                         for ev in rec["chord_timeline"]}
        # per-bar section run start, for position-in-section
        sec_start = {}
        for s, e, lab in section_runs(rec["section_per_bar"]):
            for b in range(s, e):
                sec_start[b] = (lab, s)
        prev_deg = None
        for t0, t1, root, _q in song_chord_spans(rec):
            b0, b1 = int(round(t0 / spb)), min(int(round(t1 / spb)), n_beats)
            mma = chord_at_beat.get(b0)
            parsed = parse_chord(mma) if mma else None
            fam = BUCKET_FAMILY.get(parsed[1]) if parsed else None
            if fam is None or b1 <= b0:
                continue
            root_t = (root + m["transpose"]) % 12
            chroma = np.roll(au_c[b0:b1].sum(axis=0), -root_t)
            if chroma.sum() < 1e-9:
                prev_deg = (root - tonic) % 12
                continue
            bar = b0 // bpb
            lab, s = sec_start.get(bar, ("?", bar))
            out.append(Inst(fam=fam, chroma=chroma, mode=mode,
                            deg=(root - tonic) % 12, prev_deg=prev_deg,
                            song=m["song_id"], wav=m["wav"], sec=lab,
                            pos=b0 - s * bpb))
            prev_deg = (root - tonic) % 12
    return out


def main():
    inst = build_instances()
    rng = np.random.default_rng(0)
    songs = sorted({x.song for x in inst})
    train_ids = {s for s in songs if rng.random() < 0.5}
    train = [x for x in inst if x.song in train_ids]
    test = [x for x in inst if x.song not in train_ids]
    print(f"{len(inst)} chords, {len(train)} train / {len(test)} test\n")

    # ── fit the clue distributions on train ────────────────────────────────────
    # AUDIO templates
    tacc = defaultdict(lambda: np.zeros(12)); tn = Counter()
    for x in train:
        tacc[x.fam] += normed(x.chroma); tn[x.fam] += 1
    T = np.stack([normed(tacc[f] / tn[f]) if tn[f] >= 15 else np.zeros(12) for f in FAMILIES])

    # KEY table  P(fam | deg, mode)
    key_c = defaultdict(lambda: np.zeros(5))
    for x in train:
        key_c[(x.mode, x.deg)][FI[x.fam]] += 1
    # PROG table P(fam | prev_deg, deg)
    prog_c = defaultdict(lambda: np.zeros(5))
    for x in train:
        prog_c[(x.prev_deg, x.deg)][FI[x.fam]] += 1

    def audio_logp(chroma):
        cos = T @ normed(chroma)
        e = np.exp(cos / 0.15)
        return np.log(e / e.sum() + 1e-9)

    def table_logp(counts):
        p = counts + 0.5
        return np.log(p / p.sum())

    # FOLD: average chroma over repeats of (wav, sec, pos)
    groups = defaultdict(list)
    for x in inst:
        groups[(x.wav, x.sec, x.pos)].append(x)
    fold_chroma = {}
    for g in groups.values():
        avg = np.mean([y.chroma for y in g], axis=0)
        for y in g:
            fold_chroma[id(y)] = avg

    # precompute per-instance log-prob vectors for each clue
    def features(x):
        return {
            "AUDIO": audio_logp(x.chroma),
            "KEY": table_logp(key_c[(x.mode, x.deg)]),
            "PROG": table_logp(prog_c[(x.prev_deg, x.deg)]),
            "FOLD": audio_logp(fold_chroma[id(x)]),
        }

    train_f = [(x.fam, features(x)) for x in train]
    test_f = [(x.fam, features(x), x.deg) for x in test]
    clues = ["AUDIO", "KEY", "PROG", "FOLD"]

    def acc(data, weights):
        ok = 0
        for item in data:
            fam, f = item[0], item[1]
            score = sum(weights[c] * f[c] for c in clues)
            ok += FAMILIES[int(np.argmax(score))] == fam
        return ok / len(data)

    # single-clue test accuracy
    print("Each clue alone (test accuracy at deciding the family):")
    for c in clues:
        w = {k: (1.0 if k == c else 0.0) for k in clues}
        print(f"    {c:<6} {acc(test_f, w):.1%}")

    # ── fit weights by coordinate ascent on train accuracy ────────────────────
    grid = [0.0, 0.2, 0.35, 0.5, 0.75, 1.0, 1.5, 2.0]
    weights = {c: 1.0 for c in clues}
    for _ in range(3):
        for c in clues:
            best_w, best_a = weights[c], -1
            for w in grid:
                weights[c] = w
                a = acc(train_f, weights)
                if a > best_a:
                    best_a, best_w = a, w
            weights[c] = best_w
    tot = sum(weights.values()) or 1
    print("\nFitted clue weights (how much to trust each; normalized share):")
    for c in clues:
        print(f"    {c:<6} weight {weights[c]:.2f}   ({weights[c]/tot:.0%} of total)")
    print(f"\n  COMBINED (all clues, fitted weights): {acc(test_f, weights):.1%}")

    # ── ablation: leave one clue out ──────────────────────────────────────────
    full = acc(test_f, weights)
    print("\nLeave-one-clue-out (drop in test accuracy = unique info that clue adds):")
    for c in clues:
        w = dict(weights); w[c] = 0.0
        print(f"    without {c:<6} {acc(test_f, w):.1%}   (Δ {acc(test_f, w)-full:+.1%})")

    # ── diatonic vs out-of-scale ──────────────────────────────────────────────
    dia = [t for t in test_f if t[2] in DIATONIC]
    chrom = [t for t in test_f if t[2] not in DIATONIC]
    w_audio = {k: (1.0 if k == "AUDIO" else 0.0) for k in clues}
    print(f"\nDiatonic vs out-of-scale (chromatic) chords "
          f"({len(dia)} vs {len(chrom)} test):")
    print(f"    diatonic   : audio {acc(dia, w_audio):.1%}  →  combined {acc(dia, weights):.1%}")
    print(f"    chromatic  : audio {acc(chrom, w_audio):.1%}  →  combined {acc(chrom, weights):.1%}")
    print("    (on out-of-scale chords the key prior is deliberately wrong — the")
    print("     combined model must lean on audio there; that gap IS the modulation signal)")


if __name__ == "__main__":
    main()
