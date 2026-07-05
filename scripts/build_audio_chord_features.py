"""Extract (audio-evidence, perfect-MIDI-label) features for the audio→chord model.

One row per chord instance across all rendered pilot audio. Features are the
real Basic Pitch evidence; labels are the ground-truth chord from the chart.
Cached to an .npz so model training can iterate without re-pooling.

Feature blocks (all root-relative, so the model learns quality independent of key):
  onset_chroma   (12)  — the pipeline's default observation
  note_chroma    (12)  — the sustain channel (may carry held 3rds/7ths better)
  bass_chroma    (12)  — onset activity in the bass register (MIDI < 52)
  treble_chroma  (12)  — onset activity above middle C (MIDI >= 60)
  perfect_chroma (12)  — ground-truth MIDI chroma (for the ceiling model)
Context (not audio):
  degree (0-11), mode (0/1), prev_base7 (index), key_prior (5, P(fam|deg,mode))

Labels: family (5), base_seventh (index into a fixed list), exact (index).

Usage: .venv/bin/python scripts/build_audio_chord_features.py
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
from learn_stage1_mapping import pool_beats  # noqa: E402
from harmonia.models.stage1_pitch import PitchExtractor  # noqa: E402
DB = REPO / "data" / "accomp_db" / "db.jsonl"
MANIFEST = REPO / "data" / "accomp_db" / "audio" / "manifest.jsonl"
OUT = REPO / "data" / "cache" / "audio_chord_features.npz"
MIDI_START = 21

# Three genuinely distinct tree levels, keyed on the MMA quality bucket:
#   EXACT (18)   — the full quality incl. 6ths and altered dominants
#   BASE7 (14)   — collapse to the base triad-or-seventh (6→majT, dom7alt→dom7)
#   FAMILY (5)   — major/minor/dim/aug/sus
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
FAMILIES = ["major", "minor", "diminished", "augmented", "suspended"]
FAM_IDX = {f: i for i, f in enumerate(FAMILIES)}
BASE7 = sorted(set(BUCKET_BASE7.values()))
BASE7_IDX = {v: i for i, v in enumerate(BASE7)}
EXACT = sorted(BUCKET_FAMILY.keys())
EXACT_IDX = {v: i for i, v in enumerate(EXACT)}


def reg_chroma(v88_pooled: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Chroma from a pooled (n_beats,88) slice, keys in [lo,hi) MIDI, summed over beats."""
    c = np.zeros(12)
    total = v88_pooled.sum(axis=0)
    for k in range(88):
        midi = k + MIDI_START
        if lo <= midi < hi:
            c[(midi) % 12] += total[k]
    return c


def full_chroma(v88_total: np.ndarray) -> np.ndarray:
    c = np.zeros(12)
    for k in range(88):
        c[(k + MIDI_START) % 12] += v88_total[k]
    return c


def main() -> None:
    records = {r["song_id"]: r for r in map(json.loads, open(DB))}
    manifest = [json.loads(line) for line in open(MANIFEST)]
    extractor = PitchExtractor(cache_dir=REPO / "data" / "cache" / "accomp")

    # learn key prior P(fam|deg,mode) once (from all data; it's a soft context feature)
    key_c = defaultdict(lambda: np.zeros(5))

    rows = []
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
        onset = pool_beats(acts.frame_times, acts.onset_probs, n_beats, spb)
        note = pool_beats(acts.frame_times, acts.note_probs, n_beats, spb)
        pm = pretty_midi.PrettyMIDI(str(REPO / m["midi_path"]))
        # perfect pooled roll (transposed)
        perf = np.zeros((n_beats, 88), dtype=np.float32)
        for inst in pm.instruments:
            if inst.is_drum:
                continue
            for nt in inst.notes:
                key = nt.pitch + m["transpose"] - MIDI_START
                if 0 <= key < 88:
                    b0, b1 = int(nt.start / spb), int(np.ceil(nt.end / spb))
                    for b in range(max(b0, 0), min(b1, n_beats)):
                        ov = min(nt.end, (b + 1) * spb) - max(nt.start, b * spb)
                        if ov > 0:
                            perf[b, key] += ov
        chord_at_beat = {(ev["bar"] - 1) * bpb + ev["beat"]: ev["mma"]
                         for ev in rec["chord_timeline"]}
        prev_b7 = -1
        prev_deg = -1
        _prev_root = 0
        for t0, t1, root, _q in song_chord_spans(rec):
            b0, b1 = int(round(t0 / spb)), min(int(round(t1 / spb)), n_beats)
            mma = chord_at_beat.get(b0)
            parsed = parse_chord(mma) if mma else None
            if parsed is None or b1 <= b0:
                continue
            bucket = parsed[1]
            if bucket not in BUCKET_FAMILY:
                continue
            root_t = (root + m["transpose"]) % 12

            def rr(c):  # root-relative
                return np.roll(c, -root_t)

            on_c = rr(full_chroma(onset[b0:b1].sum(axis=0)))
            deg = (root - tonic) % 12
            this_b7 = BASE7_IDX[BUCKET_BASE7[bucket]]
            if on_c.sum() < 1e-9:
                prev_b7, prev_deg, _prev_root = this_b7, deg, root
                continue
            nt_c = rr(full_chroma(note[b0:b1].sum(axis=0)))
            bass_c = rr(reg_chroma(onset[b0:b1], 0, 52))
            treb_c = rr(reg_chroma(onset[b0:b1], 60, 200))
            perf_c = rr(full_chroma(perf[b0:b1].sum(axis=0)))

            fam = BUCKET_FAMILY[bucket]
            key_c[(mode, deg)][FAM_IDX[fam]] += 1
            # root motion from previous chord (the real progression signal), 0-11 or 12=none
            root_interval = ((root - _prev_root) % 12) if prev_deg >= 0 else 12

            rows.append({
                "onset": on_c, "note": nt_c, "bass": bass_c, "treble": treb_c,
                "perfect": perf_c,
                "degree": deg, "mode": 0 if mode == "major" else 1,
                "prev_b7": prev_b7, "prev_deg": prev_deg, "root_interval": root_interval,
                "family": FAM_IDX[fam],
                "base7": this_b7,
                "exact": EXACT_IDX[bucket],
                "song": m["song_id"],
            })
            prev_b7, prev_deg = this_b7, deg
            _prev_root = root

    # materialize arrays
    def stack(key):
        return np.stack([r[key] for r in rows])

    key_prior = np.zeros((len(rows), 5))
    for i, r in enumerate(rows):
        c = key_c[(("major" if r["mode"] == 0 else "minor"), r["degree"])] + 0.5
        key_prior[i] = c / c.sum()

    songs = np.array([r["song"] for r in rows])
    np.savez_compressed(
        OUT,
        onset=stack("onset"), note=stack("note"), bass=stack("bass"),
        treble=stack("treble"), perfect=stack("perfect"),
        degree=np.array([r["degree"] for r in rows]),
        mode=np.array([r["mode"] for r in rows]),
        prev_b7=np.array([r["prev_b7"] for r in rows]),
        prev_deg=np.array([r["prev_deg"] for r in rows]),
        root_interval=np.array([r["root_interval"] for r in rows]),
        key_prior=key_prior,
        family=np.array([r["family"] for r in rows]),
        base7=np.array([r["base7"] for r in rows]),
        exact=np.array([r["exact"] for r in rows]),
        song=songs,
        base7_labels=np.array(BASE7), exact_labels=np.array(EXACT),
        family_labels=np.array(FAMILIES),
    )
    print(f"Wrote {len(rows)} instances → {OUT}")
    print(f"  family mix: {Counter(r['family'] for r in rows)}")
    print(f"  base7 classes: {BASE7}")
    print(f"  exact classes: {len(EXACT)}")


if __name__ == "__main__":
    main()
