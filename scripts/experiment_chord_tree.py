"""Hierarchical chord detection: how deep can we reliably name a chord?

The idea (user's): organize chords as a tree by how much they specify.
  Level 1 — the FAMILY, decided by the third + fifth:
            major / minor / diminished / augmented / suspended
  Level 2 — the SEVENTH added on top of that family:
            none(triad/6) / dominant-7th(b7) / major-7th / diminished-7th
  Level 3 — the exact chord incl. color notes (9ths, alterations, 6ths).

A plain C major triad is the PARENT of Cmaj7, C7 and C6 (they all contain
C-E-G). C7 is in turn the parent of C9 and C7b9 (they all contain C-E-G-Bb).
Detecting at level 1 asks only the easy, loud question (major-ish vs
minor-ish); each deeper level asks a quieter question.

This experiment measures classification accuracy AT EACH LEVEL, for
  (a) perfect notes  — the ceiling, from the ground-truth MIDI, and
  (b) real audio     — Basic Pitch listening to the rendered audio,
plus a "descend only while confident" procedure that outputs the deepest
level the evidence actually supports, and reports how deep it typically gets.

Reuses cached Basic Pitch activations from scripts/build_accomp_audio.py, so
it is cheap to re-run.

Usage: .venv/bin/python scripts/experiment_chord_tree.py
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pretty_midi

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from analyze_accomp_emission import parse_chord, song_chord_spans  # noqa: E402
from learn_stage1_mapping import gt_beat_roll, pool_beats, to_chroma  # noqa: E402
from harmonia.models.stage1_pitch import PitchExtractor  # noqa: E402

DB = REPO / "data" / "accomp_db" / "db.jsonl"
MANIFEST = REPO / "data" / "accomp_db" / "audio" / "manifest.jsonl"

# ── the chord tree: fine quality → (level-1 family, level-2 seventh) ───────────
# level-3 is the fine quality itself.
TREE: dict[str, tuple[str, str]] = {
    # major-third family
    "maj":      ("major",      "none"),
    "6":        ("major",      "none"),
    "maj7":     ("major",      "maj7"),
    "dom7":     ("major",      "dom7"),
    "dom7alt":  ("major",      "dom7"),
    # minor-third family
    "min":      ("minor",      "none"),
    "m6":       ("minor",      "none"),
    "min7":     ("minor",      "dom7"),   # min7 = minor triad + b7
    "minmaj7":  ("minor",      "maj7"),
    # diminished family (minor third + flat fifth)
    "dim":      ("diminished", "none"),
    "dim7":     ("diminished", "dim7"),
    "m7b5":     ("diminished", "dom7"),   # half-diminished = dim triad + b7
    # augmented family (major third + sharp fifth)
    "aug":      ("augmented",  "none"),
    "aug7":     ("augmented",  "dom7"),
    "augmaj7":  ("augmented",  "maj7"),
    # suspended (no third at all)
    "sus2":     ("suspended",  "none"),
    "sus4":     ("suspended",  "none"),
    "7sus4":    ("suspended",  "dom7"),
}


def levels(fine: str) -> tuple[str, str, str]:
    fam, sev = TREE[fine]
    return fam, f"{fam}/{sev}", fine  # L1, L2 (family+seventh), L3 (exact)


def normed(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def build_templates(train: list[tuple[str, np.ndarray]], level: int) -> dict[str, np.ndarray]:
    """Average training chroma per label at the requested tree level (1/2/3)."""
    acc: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(12))
    cnt: Counter = Counter()
    for fine, v in train:
        lab = levels(fine)[level - 1]
        acc[lab] += normed(v)
        cnt[lab] += 1
    return {lab: normed(acc[lab] / cnt[lab]) for lab in acc if cnt[lab] >= 15}


def classify(v: np.ndarray, templates: dict[str, np.ndarray]) -> tuple[str, float]:
    """Nearest-template label + confidence margin (best minus second-best cosine)."""
    vn = normed(v)
    scored = sorted(((float(vn @ t), lab) for lab, t in templates.items()), reverse=True)
    if len(scored) == 1:
        return scored[0][1], 1.0
    return scored[0][1], scored[0][0] - scored[1][0]


def main() -> None:
    records = {r["song_id"]: r for r in map(json.loads, open(DB))}
    manifest = [json.loads(line) for line in open(MANIFEST)]
    extractor = PitchExtractor(cache_dir=REPO / "data" / "cache" / "accomp")

    # collect per-instance (fine quality, perfect-note chroma, real-audio chroma, song)
    perfect: list[tuple[str, np.ndarray, str]] = []
    audio: list[tuple[str, np.ndarray, str]] = []

    for m in manifest:
        wav = REPO / m["wav"]
        if not wav.exists():
            continue
        rec = records[m["song_id"]]
        spb = 60.0 / m["tempo"]
        n_beats = m["n_bars"] * m["beats_per_bar"]
        try:
            acts = extractor.extract(wav)   # cached
        except Exception:
            continue
        onset_b = pool_beats(acts.frame_times, acts.onset_probs, n_beats, spb)
        pm = pretty_midi.PrettyMIDI(str(REPO / m["midi_path"]))
        gt = gt_beat_roll(pm, n_beats, spb, m["transpose"])
        gt_c = to_chroma(gt)
        au_c = to_chroma(onset_b)
        for t0, t1, root, qual in song_chord_spans(rec):
            if qual not in TREE:
                continue
            b0, b1 = int(t0 / spb), min(int(t1 / spb), n_beats)
            if b1 <= b0:
                continue
            root_t = (root + m["transpose"]) % 12
            pv = np.roll(gt_c[b0:b1].sum(axis=0), -root_t)
            av = np.roll(au_c[b0:b1].sum(axis=0), -root_t)
            if pv.sum() > 1e-9:
                perfect.append((qual, pv, m["song_id"]))
            if av.sum() > 1e-9:
                audio.append((qual, av, m["song_id"]))

    rng = np.random.default_rng(0)
    song_ids = sorted({s for *_, s in perfect})
    train_ids = {s for s in song_ids if rng.random() < 0.5}

    def run(data: list[tuple[str, np.ndarray, str]], name: str) -> None:
        train = [(q, v) for q, v, s in data if s in train_ids]
        test = [(q, v) for q, v, s in data if s not in train_ids]
        tmpl = {lv: build_templates(train, lv) for lv in (1, 2, 3)}

        # ── unconditional accuracy at each fixed level ─────────────────────
        hits = {1: 0, 2: 0, 3: 0}
        for q, v in test:
            true = levels(q)
            for lv in (1, 2, 3):
                pred, _ = classify(v, tmpl[lv])
                hits[lv] += pred == true[lv - 1]
        n = len(test)
        print(f"\n  {name}  ({n} test chords)")
        print(f"    Level 1  (major / minor / dim / aug / sus) : {hits[1]/n:5.1%}")
        print(f"    Level 2  (+ which 7th)                     : {hits[2]/n:5.1%}")
        print(f"    Level 3  (exact chord, ~15 types)          : {hits[3]/n:5.1%}")

        # ── confidence-gated tree walk ─────────────────────────────────────
        # Proper walk: pick a family at level 1; then choose the seventh only
        # among that family's children; then the exact chord only among that
        # node's children. Descend while the winner clearly beats the runner-up.
        # Output = the label at the level we stop on. "Precise" = every level we
        # committed to matched the truth (we never output a wrong label, though
        # we're allowed to stop shallow and just say the family).
        children2 = {lab: {c for c in tmpl[2] if c.startswith(lab + "/")} for lab in tmpl[1]}
        children3 = {
            lab2: {c for c in tmpl[3] if levels(c)[1] == lab2} for lab2 in tmpl[2]
        }
        for margin_min in (0.02, 0.05, 0.10):
            precise = 0
            depth_sum = 0
            for q, v in test:
                true = levels(q)
                # level 1
                fam, m1 = classify(v, tmpl[1])
                committed = [fam]
                if m1 >= margin_min:
                    kids2 = {c: tmpl[2][c] for c in children2.get(fam, ())}
                    if kids2:
                        l2, m2 = classify(v, kids2)
                        committed.append(l2)
                        if m2 >= margin_min:
                            kids3 = {c: tmpl[3][c] for c in children3.get(l2, ())}
                            if len(kids3) > 1:
                                l3, _ = classify(v, kids3)
                                committed.append(l3)
                depth_sum += len(committed)
                precise += all(committed[i] == true[i] for i in range(len(committed)))
            print(f"    gated walk (margin≥{margin_min:.2f}): never-wrong "
                  f"{precise/n:5.1%}, avg depth {depth_sum/n:.2f}/3")

    run(perfect, "PERFECT NOTES (ceiling)")
    run(audio, "REAL AUDIO (Basic Pitch)")

    # ── where does level-1 actually fail on real audio? ───────────────────────
    train = [(q, v) for q, v, s in audio if s in train_ids]
    test = [(q, v, s) for q, v, s in audio if s not in train_ids]
    t1 = build_templates(train, 1)
    conf = Counter()
    for q, v, _ in test:
        pred, _ = classify(v, t1)
        true = levels(q)[0]
        if pred != true:
            conf[(true, pred)] += 1
    print("\n  Level-1 confusions on real audio (true → predicted):",
          [(f"{a}→{b}", n) for (a, b), n in conf.most_common(6)])


if __name__ == "__main__":
    main()
