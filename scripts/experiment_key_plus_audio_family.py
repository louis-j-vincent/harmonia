"""Does KEY + AUDIO beat audio-alone at deciding the chord family (the third)?

User's hypothesis: the key is the strongest lever for the third (major vs minor),
which we can't hear well. Key-alone tops out ~72% (secondary dominants), audio-
alone ~81%. This tests the combination on rendered audio: blend the audio family
evidence with a key-conditioned family prior (given the root's scale degree) and
see if it beats audio-alone.

Blend is log P(family | audio) + w · log P(family | key, scale-degree) — the same
soft-prior shape as the rest of the pipeline. Scale degree is transpose-invariant,
so augmented renders are handled for free.

Usage: .venv/bin/python scripts/experiment_key_plus_audio_family.py
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


def normed(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def main() -> None:
    records = {r["song_id"]: r for r in map(json.loads, open(DB))}
    manifest = [json.loads(line) for line in open(MANIFEST)]
    extractor = PitchExtractor(cache_dir=REPO / "data" / "cache" / "accomp")

    # instances: (family, root-relative audio chroma, mode, scale-degree, song_id)
    inst = []
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
        n_beats = m["n_bars"] * m["beats_per_bar"]
        try:
            acts = extractor.extract(wav)   # cached
        except Exception:
            continue
        onset_b = pool_beats(acts.frame_times, acts.onset_probs, n_beats, spb)
        au_c = to_chroma(onset_b)
        # exact chord per span-start beat, for the true family label
        bpb = m["beats_per_bar"]
        chord_at_beat = {(ev["bar"] - 1) * bpb + ev["beat"]: ev["mma"]
                         for ev in rec["chord_timeline"]}
        for t0, t1, root, _q in song_chord_spans(rec):
            b0, b1 = int(round(t0 / spb)), min(int(round(t1 / spb)), n_beats)
            mma = chord_at_beat.get(b0)
            parsed = parse_chord(mma) if mma else None
            fam = BUCKET_FAMILY.get(parsed[1]) if parsed else None
            if fam is None or b1 <= b0:
                continue
            root_t = (root + m["transpose"]) % 12
            v = np.roll(au_c[b0:b1].sum(axis=0), -root_t)
            if v.sum() < 1e-9:
                continue
            deg = (root - tonic) % 12   # transpose-invariant
            inst.append((fam, v, mode, deg, m["song_id"]))

    rng = np.random.default_rng(0)
    song_ids = sorted({s for *_, s in inst})
    train_ids = {s for s in song_ids if rng.random() < 0.5}

    # audio family templates (train)
    tmpl_acc = defaultdict(lambda: np.zeros(12))
    tmpl_n = Counter()
    for fam, v, *_ , sid in [(f, v, m, d, s) for f, v, m, d, s in inst]:
        if sid in train_ids:
            tmpl_acc[fam] += normed(v)
            tmpl_n[fam] += 1
    templates = {f: normed(tmpl_acc[f] / tmpl_n[f]) for f in tmpl_acc if tmpl_n[f] >= 15}
    fams = [f for f in FAMILIES if f in templates]
    T = np.stack([templates[f] for f in fams])

    # key→family soft prior (train), P(family | mode, degree), Laplace-smoothed
    key_counts = defaultdict(lambda: Counter())
    for fam, v, mode, deg, sid in inst:
        if sid in train_ids:
            key_counts[(mode, deg)][fam] += 1

    def key_prior(mode, deg):
        c = key_counts.get((mode, deg), Counter())
        p = np.array([c[f] + 0.5 for f in fams])
        return p / p.sum()

    test = [(f, v, mode, deg) for f, v, mode, deg, s in inst if s not in train_ids]

    def evaluate(weight, temp=0.15):
        hits_a = hits_k = hits_b = 0
        hard_a = hard_b = hard_n = 0
        for fam, v, mode, deg in test:
            vn = normed(v)
            cos = T @ vn
            audio_lp = np.log(np.exp(cos / temp) / np.exp(cos / temp).sum() + 1e-12)
            kp = np.log(key_prior(mode, deg) + 1e-12)
            pa = fams[int(np.argmax(cos))]
            pk = fams[int(np.argmax(kp))]
            pb = fams[int(np.argmax(audio_lp + weight * kp))]
            true = fam if fam in fams else None
            if true is None:
                continue
            hits_a += pa == true
            hits_k += pk == true
            hits_b += pb == true
            # "hard" = audio-alone got the major/minor family wrong
            if true in ("major", "minor") and pa != true:
                hard_n += 1
                hard_b += pb == true
        n = sum(1 for f, *_ in test if f in fams)
        return hits_a / n, hits_k / n, hits_b / n, (hard_b / hard_n if hard_n else 0), hard_n, n

    a, k, _, _, _, n = evaluate(0.0)
    print(f"{n} test chords\n")
    print(f"  audio-only family accuracy : {a:.1%}")
    print(f"  key-only  family accuracy  : {k:.1%}")
    print("\n  audio + key blend, by key-prior weight:")
    best = (0, 0)
    for w in (0.3, 0.6, 1.0, 1.5, 2.0, 3.0):
        _, _, b, hard_recover, hard_n, _ = evaluate(w)
        marker = "  ←" if b > best[1] else ""
        if b > best[1]:
            best = (w, b)
        print(f"    w={w:>3}:  {b:.1%}   (of {hard_n} chords audio got wrong, "
              f"key rescued {hard_recover:.0%}){marker}")
    print(f"\n  best blend: weight {best[0]} → {best[1]:.1%} "
          f"(audio-alone {a:.1%}, so +{best[1]-a:+.1%})")


if __name__ == "__main__":
    main()
