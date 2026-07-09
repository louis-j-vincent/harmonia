"""Premise check on the CANONICAL corpus (irealb/jazz1460): is the root error
5th-apart, and is it rescuable by harmonic context — or is it an acoustic wall?

Renders N odd jazz songs (MMA MIDI → WAV → Basic Pitch), exact metronomic tempo
grid (matches training; no librosa). Runs the v2 beat-seq root model per beat.

Q1  What fraction of root errors are +5/+7 (P4/P5, root↔5th)?  [confirm 46% on jazz]
Q2  Of those 5th-apart errors, are they RESCUABLE by a context/progression prior?
      - true root in v2 top-2 / top-3 of the soft proba  → a mild prior can flip it
      - OR either neighbour beat's argmax == true root    → local context carries it
    If rescuable → build the ctx/progression root prior. If not (5th dominates,
    true root absent from the neighbourhood) → the wall is Basic Pitch bass
    transcription, not the model.

Caveat: the shipped v2 was trained on a strided 50-song subset, so there may be
mild train/eval leak — it inflates ACCURACY but the error STRUCTURE (interval dist
+ rescuability among the errors) is what we read here, and that's robust to leak.
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from analyze_accomp_emission import song_chord_spans
from build_audio_chord_features import BUCKET_FAMILY
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig
from harmonia.models.stage1_pitch import PitchExtractor
from harmonia.models.chord_pipeline_v1 import _BeatSeqModel, _pool_beats, MODELS

DB = REPO / "data" / "accomp_db" / "db.jsonl"


def collect(rec, renderer, sf2, ex):
    spb = 60.0 / rec["tempo"]
    bpb = rec["beats_per_bar"]
    n_beats = rec["n_bars"] * bpb
    spans = [(t0, t1, r % 12) for t0, t1, r, q in song_chord_spans(rec)
             if t1 > t0 and q in BUCKET_FAMILY]
    if not spans:
        return None

    def gtroot(t):
        for t0, t1, r in spans:
            if t0 <= t < t1:
                return r
        return None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
        tmp = Path(wf.name)
    try:
        renderer.render(REPO / rec["midi_path"], tmp, RenderConfig(soundfont_path=sf2))
        acts = ex.extract(tmp, use_cache=False)
    finally:
        tmp.unlink(missing_ok=True)

    bt = np.arange(n_beats + 1) * spb
    onset_b = _pool_beats(acts.frame_times, acts.onset_probs, bt)
    note_b = _pool_beats(acts.frame_times, acts.note_probs, bt)
    gt = np.array([gtroot((b + 0.5) * spb) for b in range(n_beats)], dtype=object)
    return onset_b, note_b, gt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25, help="n odd jazz songs")
    args = ap.parse_args()

    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2 = renderer._find_soundfont("MuseScore_General.sf2")
    ex = PitchExtractor(cache_dir=None)  # constants-cache caveat: no cache
    v2 = _BeatSeqModel(MODELS / "beat_seq_model_v2.npz")

    recs = [json.loads(l) for l in open(DB)]
    songs = [r for r in recs if r.get("corpus") == "jazz1460"
             and r["beats_per_bar"] == 4 and (REPO / r["midi_path"]).exists()]
    odd = songs[1::2][: args.n]  # odd-indexed songs
    print(f"jazz1460 odd songs: {len(odd)}")

    GT = []; P1 = []; TOP = []; NBR = []
    for i, rec in enumerate(odd):
        print(f"  [{i+1}/{len(odd)}] {rec['song_id']}", end="\r", flush=True)
        try:
            res = collect(rec, renderer, sf2, ex)
        except Exception as e:
            print(f"\n  SKIP {rec['song_id']}: {e}"); continue
        if res is None:
            continue
        onset_b, note_b, gt = res
        proba = v2.predict_proba(onset_b, note_b)  # (n,12)
        args = np.argsort(proba, axis=1)[:, ::-1]  # descending
        pred = args[:, 0]
        n = len(gt)
        for b in range(n):
            if gt[b] is None:
                continue
            g = int(gt[b])
            GT.append(g); P1.append(int(pred[b]))
            # rank of true root in this beat's proba
            TOP.append(int(np.where(args[b] == g)[0][0]))
            # is true root the argmax of either neighbour?
            nb = False
            for d in (-1, 1):
                bb = b + d
                if 0 <= bb < n and int(pred[bb]) == g:
                    nb = True
            NBR.append(nb)

    GT = np.array(GT); P1 = np.array(P1); TOP = np.array(TOP); NBR = np.array(NBR)
    ok = GT == P1
    print(f"\n\n=== jazz1460 v2 per-beat root (n={len(GT)}) ===")
    print(f"  overall root acc: {ok.mean():.1%}")

    err = ~ok
    iv = (P1[err] - GT[err]) % 12
    counts = np.bincount(iv, minlength=12)
    print(f"\n  Q1 error-interval distribution (semitones up from true root):")
    for s in np.argsort(counts)[::-1][:5]:
        print(f"    +{s:2d}: {counts[s]/err.sum():5.1%}")
    p57 = (counts[5] + counts[7]) / err.sum()
    print(f"  +5/+7 (P4/P5, root↔5th) share of errors: {p57:.1%}")

    # rescuability among the 5th-apart errors specifically
    fifth = err & (((P1 - GT) % 12 == 5) | ((P1 - GT) % 12 == 7))
    m = fifth
    print(f"\n  Q2 rescuability of the {m.sum()} 5th-apart errors:")
    print(f"    true root in top-2 of soft proba: {(TOP[m] <= 1).mean():.1%}")
    print(f"    true root in top-3 of soft proba: {(TOP[m] <= 2).mean():.1%}")
    print(f"    true root = a neighbour's argmax:  {NBR[m].mean():.1%}")
    print(f"    rescuable (top-2 OR neighbour):    {((TOP[m] <= 1) | NBR[m]).mean():.1%}")
    print(f"\n  [reference] among ALL errors, true root in top-2: {(TOP[err] <= 1).mean():.1%}")


if __name__ == "__main__":
    main()
