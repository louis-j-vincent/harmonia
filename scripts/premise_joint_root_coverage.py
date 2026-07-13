"""Premise check for the joint decode (CLAUDE.md rule #2).

The joint decode's candidate set is the top-K roots of each segment's summed
beat posterior. That is only worth building if the GT root is actually IN that
top-K for the real segmentation. This measures, on the FIT split (jazz1460 idx
20..25 — NOT the 70..95 eval set), the % of segments whose GT root is in the
top-1/2/3 of beat_proba, replicating the production front-end exactly (same beat
grid, Basic Pitch features, beat-seq model, gmerge segmentation).

If top-3 coverage is well under ~85%, K=3 is too small / the premise is weak.

Usage: .venv/bin/python scripts/premise_joint_root_coverage.py --start 20 --n 5
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import warnings
from collections import Counter
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import librosa
import soundfile as sf

from analyze_accomp_emission import song_chord_spans
from build_audio_chord_features import BUCKET_FAMILY
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig
from harmonia.models import chord_pipeline_v1 as P

DB = REPO / "data" / "accomp_db" / "db.jsonl"


def front_end(wav: Path, cache_dir: Path):
    """Reproduce infer_chords_v1 up through segmentation → (bt, beat_proba, segs)."""
    y, sr = sf.read(wav)
    y = (y.mean(1) if y.ndim > 1 else y).astype("float32")
    duration_s = len(y) / sr
    tempo_arr, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    tempo_bpm = float(np.atleast_1d(tempo_arr)[0])
    beat_times_raw = librosa.frames_to_time(beat_frames, sr=sr)
    period = 60.0 / max(tempo_bpm, 1.0)
    ang = 2 * np.pi * (beat_times_raw % period) / period
    phase = (np.angle(np.mean(np.exp(1j * ang))) % (2 * np.pi)) * period / (2 * np.pi)
    bt = np.arange(phase, duration_s + period, period)
    bt = np.unique(np.concatenate([[0.0], bt, [duration_s]]))

    ex = P.PitchExtractor(cache_dir=cache_dir)
    acts = ex.extract(wav)
    onset_b = P._pool_beats(acts.frame_times, acts.onset_probs, bt)
    note_b = P._pool_beats(acts.frame_times, acts.note_probs, bt)
    n_beats = len(onset_b)

    beat_seq = P._get_beat_seq()
    beat_proba = beat_seq.predict_proba(onset_b, note_b)

    qual_proba = None
    bsv3 = P._get_beat_seq_v3()
    if bsv3 is not None:
        qual_proba = bsv3.qual_proba(onset_b, note_b)

    mean_conf = float(beat_proba.max(1).mean())
    if mean_conf < 0.30:
        segs = P._coarse_segments(onset_b, theta=0.08, cell=2)
    else:
        grid = P._fit_harmonic_grid(beat_proba)
        segs = P._make_grid_segs(n_beats, grid)
        segs = P._merge_grid_by_root_and_bass(segs, beat_proba, onset_b, qual_proba)
    return bt, beat_proba, segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=20)
    ap.add_argument("--n", type=int, default=5)
    args = ap.parse_args()

    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2 = renderer._find_soundfont("MuseScore_General.sf2")
    recs = [json.loads(l) for l in open(DB)]
    jz = [r for r in recs if r.get("corpus") == "jazz1460"
          and r["beats_per_bar"] == 4 and (REPO / r["midi_path"]).exists()]
    held = jz[args.start:args.start + args.n]
    print(f"fit-split jazz songs: {len(held)} (idx {args.start}..{args.start + args.n})")

    hits = Counter()   # K -> segments with GT root in top-K
    total = 0
    with tempfile.TemporaryDirectory() as cd:
        cache = Path(cd)
        for i, rec in enumerate(held):
            spans = [(t0, t1, r % 12, q) for t0, t1, r, q in song_chord_spans(rec)
                     if t1 > t0 and q in BUCKET_FAMILY]
            if not spans:
                continue
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
                tmp = Path(wf.name)
            try:
                renderer.render(REPO / rec["midi_path"], tmp,
                                RenderConfig(soundfont_path=sf2))
                bt, beat_proba, segs = front_end(tmp, cache)
            finally:
                tmp.unlink(missing_ok=True)

            def gt_root_at(t):
                for t0, t1, r, q in spans:
                    if t0 <= t < t1:
                        return r
                return None

            song_total = 0
            for (s, e) in segs:
                tmid = 0.5 * (bt[s] + bt[min(e, len(bt) - 1)])
                gt_r = gt_root_at(tmid)
                if gt_r is None:
                    continue
                order = list(np.argsort(beat_proba[s:e].sum(0))[::-1])
                total += 1
                song_total += 1
                for K in (1, 2, 3):
                    if gt_r in order[:K]:
                        hits[K] += 1
            print(f"  [{i+1}/{len(held)}] {rec['song_id']}: {song_total} segs "
                  f"(top1={hits[1]} top2={hits[2]} top3={hits[3]} cum)", flush=True)

    print(f"\n=== GT-root candidate coverage on {total} segments ===")
    for K in (1, 2, 3):
        print(f"  top-{K}: {hits[K] / total:.1%}" if total else "  (no segments)")


if __name__ == "__main__":
    main()
