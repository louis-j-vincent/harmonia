"""Premise check for the per-track zoom (rule #2, before building it): at BEAT
resolution, does an isolated-track cue beat the mixed signal (AUC 0.643) at telling
change from hold? Landmine: jazz walking bass changes the bass note nearly every
beat regardless of the chord — if so, bass-PC-motion is a bad cue.

Renders per-instrument stems free from the MIDI. For each interior beat, compare
vs GT chord change:
  bass_pc_change   isolated bass root differs from previous beat (walking-bass risk)
  bass_novelty     isolated bass CHROMA cosine distance (softer than a hard PC flip)
  chord_novelty    isolated comping (piano/guitar) chroma cosine distance
  mixed_novelty    full-mix chroma+bass distance (the 0.643 baseline)

Usage: .venv/bin/python scripts/pertrack_zoom_probe.py --n-songs 12 [--degrade]
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

from build_accomp_audio_hard import time_varying_degrade  # noqa: E402
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig  # noqa: E402
from harmonia.models.stage1_pitch import PitchExtractor  # noqa: E402
from harmonic_rhythm_probe import gt_chord_per_beat, pool_beats  # noqa: E402
from stem_benefit import stem_midi  # noqa: E402


def chroma(v88, lo=0, hi=200):
    c = np.zeros(12)
    for k in range(88):
        m = 21 + k
        if lo <= m < hi:
            c[m % 12] += v88[k]
    n = np.linalg.norm(c)
    return c / n if n > 1e-9 else c


def cos_d(a, b):
    return 1 - float(a @ b)


def auc(score, label):
    score = np.asarray(score, float); label = np.asarray(label)
    pos, neg = score[label == 1], score[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    return (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-songs", type=int, default=12)
    ap.add_argument("--degrade", action="store_true")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(REPO / "data/accomp_db/db.jsonl")]
    songs = [r for r in recs if r["corpus"] == "jazz1460" and r["beats_per_bar"] == 4
             and (REPO / r["midi_path"]).exists() and len(set(r["section_per_bar"])) > 1]
    songs = songs[:: max(len(songs) // args.n_songs, 1)][: args.n_songs]

    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2 = renderer._find_soundfont("MuseScore_General.sf2")
    ex = PitchExtractor(cache_dir=None)
    rng = np.random.default_rng(3)

    stems = {"bass": dict(keep_drums=False, keep_harmonic=False, keep_bass_only=True),
             "chord": dict(keep_drums=False, keep_harmonic=True),
             "full": dict(keep_drums=True, keep_harmonic=True)}

    cues = {"bass_pc_change": [], "bass_novelty": [], "chord_novelty": [], "mixed_novelty": []}
    labels = []
    bass_flip_rate = []

    for rec in songs:
        src = REPO / rec["midi_path"]
        spb = 60.0 / rec["tempo"]; bpb = rec["beats_per_bar"]; nb = rec["n_bars"]
        n_beats = nb * bpb
        bt = np.arange(n_beats + 1) * spb
        gtc = gt_chord_per_beat(rec, n_beats, spb)

        pooled = {}
        for name, kw in stems.items():
            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as mf:
                mid = Path(mf.name)
            stem_midi(src, mid, **kw)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
                tmp = Path(wf.name)
            try:
                renderer.render(mid, tmp, RenderConfig(soundfont_path=sf2))
                y, sr = sf.read(tmp); y = (y.mean(1) if y.ndim > 1 else y).astype("float32")
                if args.degrade:
                    y = time_varying_degrade(y, sr, rng); sf.write(tmp, y, sr)
                acts = ex.extract(tmp, use_cache=False)
            finally:
                mid.unlink(missing_ok=True); tmp.unlink(missing_ok=True)
            pooled[name] = pool_beats(acts.frame_times, acts.onset_probs, bt)

        bass_ch = np.stack([chroma(pooled["bass"][b]) for b in range(n_beats)])
        bass_pc = [int(bass_ch[b].argmax()) if bass_ch[b].sum() > 1e-9 else -1 for b in range(n_beats)]
        chord_ch = np.stack([chroma(pooled["chord"][b]) for b in range(n_beats)])
        full_ch = np.stack([chroma(pooled["full"][b]) for b in range(n_beats)])
        full_ba = np.stack([chroma(pooled["full"][b], 0, 52) for b in range(n_beats)])

        flips = 0; nb_int = 0
        for b in range(1, n_beats):
            if gtc[b] is None:
                continue
            labels.append(int(gtc[b] != gtc[b - 1]))
            cues["bass_pc_change"].append(float(bass_pc[b] != bass_pc[b - 1] and bass_pc[b] >= 0))
            cues["bass_novelty"].append(cos_d(bass_ch[b], bass_ch[b - 1]))
            cues["chord_novelty"].append(cos_d(chord_ch[b], chord_ch[b - 1]))
            cues["mixed_novelty"].append(0.5 * cos_d(full_ch[b], full_ch[b - 1])
                                         + 0.5 * cos_d(full_ba[b], full_ba[b - 1]))
            flips += int(bass_pc[b] != bass_pc[b - 1]); nb_int += 1
        bass_flip_rate.append(flips / max(nb_int, 1))

    cond = "DEGRADED" if args.degrade else "clean"
    base = np.mean(labels)
    print(f"\n=== per-track zoom premise, {len(songs)} {cond} songs ===")
    print(f"beats: {len(labels)}, GT changes: {base:.0%}   |   isolated-bass PC flips on "
          f"{np.mean(bass_flip_rate):.0%} of beats (walking-bass check)\n")
    print("beat-level change-vs-hold separability (AUC; mixed 2-beat block was 0.962):")
    for k in ("bass_pc_change", "bass_novelty", "chord_novelty", "mixed_novelty"):
        print(f"    {k:<15} AUC={auc(cues[k], labels):.3f}")
    print("\nA per-track cue is worth zooming with only if it clearly beats mixed_novelty.")


if __name__ == "__main__":
    main()
