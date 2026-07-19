"""Extract per-chord BP48 features in ABSOLUTE frame (no root rotation) from the
accomp_db synthetic audio, reusing cached Basic-Pitch activations.

Stores, per chord instance:
  onset12, note12, bass12, treble12  — absolute-pitch-class chroma blocks
  root (pc 0-11), tonic (pc 0-11), mode (0 maj / 1 min)
  fam5 (major/minor/diminished/augmented/suspended), base7 (idx), dom-vs-maj7 etc.

All rotation (root-relative / key-relative) is done offline from this, so the
12-vs-48 and normalization A/B/C experiments need no re-extraction.

Writes data/cache/bp48_absolute.npz (one small file; disk-safe).
"""
from __future__ import annotations
import json, sys
from collections import Counter
from pathlib import Path
import numpy as np
import pretty_midi

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from analyze_accomp_emission import parse_chord, song_chord_spans  # noqa
from analyze_accomp_priors import parse_key  # noqa
from learn_stage1_mapping import pool_beats  # noqa
from harmonia.models.stage1_pitch import PitchExtractor  # noqa

DB = REPO / "data/accomp_db/db.jsonl"
MANIFEST = REPO / "data/accomp_db/audio/manifest.jsonl"
OUT = REPO / "data/cache/bp48_absolute.npz"
MIDI_START = 21

BUCKET_FAMILY = {
    "maj": "major", "maj7": "major", "6": "major", "dom7": "major", "dom7alt": "major",
    "min": "minor", "min7": "minor", "m6": "minor", "minmaj7": "minor",
    "dim": "diminished", "dim7": "diminished", "m7b5": "diminished",
    "aug": "augmented", "aug7": "augmented", "augmaj7": "augmented",
    "sus2": "suspended", "sus4": "suspended", "7sus4": "suspended",
}
BUCKET_BASE7 = {
    "maj": "majT", "6": "majT", "maj7": "maj7", "dom7": "dom7", "dom7alt": "dom7",
    "min": "minT", "m6": "minT", "min7": "min7", "minmaj7": "minmaj7",
    "dim": "dimT", "dim7": "dim7", "m7b5": "m7b5",
    "aug": "augT", "aug7": "aug7", "augmaj7": "augmaj7",
    "sus2": "susT", "sus4": "susT", "7sus4": "7sus4",
}
# 5-way quality aligned to the NNLS Billboard head: maj/min/dom/hdim/dim
Q5 = {  # base7 bucket -> 5-way quality index (maj0 min1 dom2 hdim3 dim4), or None to drop
    "majT": 0, "maj7": 0, "min7": 1, "minT": 1, "minmaj7": 1,
    "dom7": 2, "m7b5": 3, "dimT": 4, "dim7": 4,
}
FAMILIES = ["major", "minor", "diminished", "augmented", "suspended"]
FAM_IDX = {f: i for i, f in enumerate(FAMILIES)}
BASE7 = sorted(set(BUCKET_BASE7.values()))
BASE7_IDX = {v: i for i, v in enumerate(BASE7)}


def reg_chroma(v88_pooled, lo, hi):
    c = np.zeros(12)
    total = v88_pooled.sum(axis=0)
    for k in range(88):
        midi = k + MIDI_START
        if lo <= midi < hi:
            c[midi % 12] += total[k]
    return c


def full_chroma(v88_total):
    c = np.zeros(12)
    for k in range(88):
        c[(k + MIDI_START) % 12] += v88_total[k]
    return c


def main():
    records = {r["song_id"]: r for r in map(json.loads, open(DB))}
    manifest = [json.loads(l) for l in open(MANIFEST)]
    extractor = PitchExtractor(cache_dir=REPO / "data/cache/accomp")
    rows = []
    n_no_cache = 0
    for m in manifest:
        wav = REPO / m["wav"]
        if not wav.exists():
            continue
        rec = records[m["song_id"]]
        k = parse_key(rec["key"])
        if k is None:
            continue
        tonic, mode = k
        spb = 60.0 / m["tempo"]; bpb = m["beats_per_bar"]; n_beats = m["n_bars"] * bpb
        try:
            acts = extractor.extract(wav)  # uses cache
        except Exception:
            n_no_cache += 1; continue
        onset = pool_beats(acts.frame_times, acts.onset_probs, n_beats, spb)
        note = pool_beats(acts.frame_times, acts.note_probs, n_beats, spb)
        for t0, t1, root, _q in song_chord_spans(rec):
            b0, b1 = int(round(t0 / spb)), min(int(round(t1 / spb)), n_beats)
            if b1 <= b0:
                continue
            # find mma quality at this span
            chord_at_beat = {(ev["bar"] - 1) * bpb + ev["beat"]: ev["mma"]
                             for ev in rec["chord_timeline"]}
            mma = chord_at_beat.get(b0)
            parsed = parse_chord(mma) if mma else None
            if parsed is None or parsed[1] not in BUCKET_FAMILY:
                continue
            bucket = parsed[1]
            base7 = BUCKET_BASE7[bucket]
            root_t = (root + m["transpose"]) % 12
            tonic_t = (tonic + m["transpose"]) % 12
            on_c = full_chroma(onset[b0:b1].sum(axis=0))
            if on_c.sum() < 1e-9:
                continue
            nt_c = full_chroma(note[b0:b1].sum(axis=0))
            bass_c = reg_chroma(onset[b0:b1], 0, 52)
            treb_c = reg_chroma(onset[b0:b1], 60, 200)
            rows.append({
                "onset": on_c, "note": nt_c, "bass": bass_c, "treble": treb_c,
                "root": root_t, "tonic": tonic_t, "mode": 0 if mode == "major" else 1,
                "fam5": FAM_IDX[BUCKET_FAMILY[bucket]],
                "base7": BASE7_IDX[base7],
                "q5": Q5.get(base7, -1),
                "song": m["song_id"],
            })
    def st(k): return np.stack([r[k] for r in rows]).astype(np.float32)
    np.savez_compressed(
        OUT,
        onset=st("onset"), note=st("note"), bass=st("bass"), treble=st("treble"),
        root=np.array([r["root"] for r in rows]),
        tonic=np.array([r["tonic"] for r in rows]),
        mode=np.array([r["mode"] for r in rows]),
        fam5=np.array([r["fam5"] for r in rows]),
        base7=np.array([r["base7"] for r in rows]),
        q5=np.array([r["q5"] for r in rows]),
        song=np.array([r["song"] for r in rows]),
        q5_labels=np.array(["maj", "min", "dom", "hdim", "dim"], dtype=object),
    )
    print(f"Wrote {len(rows)} chords -> {OUT}  (songs={len(set(r['song'] for r in rows))})")
    print("q5 mix:", Counter(r["q5"] for r in rows))
    print("fam5 mix:", Counter(r["fam5"] for r in rows))


if __name__ == "__main__":
    main()
