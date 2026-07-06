"""Beat tracking is the suspected end-to-end bottleneck (oracle boundaries → 86.8%
root, detected beats → ~67%). Measure it DIRECTLY against the known MMA tempo grid
(rule #1: unit-test the load-bearing assumption against an external reference),
and test whether beat-tracking on an isolated percussion stem beats the full mix.

For our synthetic audio the drum stem is free (filter the MIDI to the drum track);
for real audio we'd get it from HDemucs (torchaudio, MIT) — this checks the premise
before we pay for separation.

Variants scored (mir_eval beat F-measure, 70 ms tol):
  full      full mix (what pipeline_v0 uses today)
  drums     MIDI drum track only (oracle stem)
  no_drums  everything except drums (harmonic content only)
  demucs    HDemucs 'drums' stem separated FROM the full mix (the real-audio path)

Usage: .venv/bin/python scripts/beat_tracking_experiment.py --n-songs 12 [--degrade] [--demucs]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import warnings
from pathlib import Path

import librosa
import mir_eval
import numpy as np
import pretty_midi
import soundfile as sf

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from build_accomp_audio_hard import time_varying_degrade  # noqa: E402
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig  # noqa: E402

DB = REPO / "data" / "accomp_db" / "db.jsonl"


def stem_midi(src: Path, out: Path, keep_drums: bool, keep_harmonic: bool):
    """Write a filtered copy of `src` keeping only selected tracks."""
    pm = pretty_midi.PrettyMIDI(str(src))
    pm.instruments = [i for i in pm.instruments
                      if (i.is_drum and keep_drums) or (not i.is_drum and keep_harmonic)]
    pm.write(str(out))


def gt_beats(rec) -> np.ndarray:
    spb = 60.0 / rec["tempo"]
    n = rec["n_bars"] * rec["beats_per_bar"]
    return np.arange(n) * spb


def track_f(y, sr, ref):
    if len(y) < sr:
        return 0.0
    _, bf = librosa.beat.beat_track(y=y, sr=sr)
    est = librosa.frames_to_time(bf, sr=sr)
    if len(est) < 2:
        return 0.0
    return mir_eval.beat.f_measure(ref, est)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-songs", type=int, default=12)
    ap.add_argument("--degrade", action="store_true")
    ap.add_argument("--demucs", action="store_true", help="also test HDemucs drum separation")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(DB)]
    songs = [r for r in recs if r["corpus"] == "jazz1460" and r["beats_per_bar"] == 4
             and (REPO / r["midi_path"]).exists()]
    songs = songs[:: max(len(songs) // args.n_songs, 1)][: args.n_songs]

    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2 = renderer._find_soundfont("MuseScore_General.sf2")
    rng = np.random.default_rng(3)

    separator = None
    if args.demucs:
        import torch
        from torchaudio.pipelines import HDEMUCS_HIGH_MUSDB_PLUS as BUNDLE
        separator = BUNDLE.get_model().eval()
        demucs_sr = BUNDLE.sample_rate
        drum_idx = list(BUNDLE.sources).index("drums")

    variants = {"full": (True, True), "drums": (True, False), "no_drums": (False, True)}
    scores = {k: [] for k in variants}
    if args.demucs:
        scores["demucs"] = []

    def render_and_score(midi_path, ref, degrade):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
            tmp = Path(wf.name)
        try:
            renderer.render(midi_path, tmp, RenderConfig(soundfont_path=sf2))
            y, sr = sf.read(tmp)
            y = (y.mean(1) if y.ndim > 1 else y).astype("float32")
            if degrade:
                y = time_varying_degrade(y, sr, rng)
            return track_f(y, sr, ref), y, sr
        finally:
            tmp.unlink(missing_ok=True)

    for rec in songs:
        ref = gt_beats(rec)
        src = REPO / rec["midi_path"]
        full_y = full_sr = None
        for name, (kd, kh) in variants.items():
            with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as mf:
                mid = Path(mf.name)
            try:
                stem_midi(src, mid, kd, kh)
                f, y, sr = render_and_score(mid, ref, args.degrade)
            finally:
                mid.unlink(missing_ok=True)
            scores[name].append(f)
            if name == "full":
                full_y, full_sr = y, sr

        if args.demucs and full_y is not None:
            import torch
            wav = torch.from_numpy(full_y)[None].repeat(2, 1)          # mono→stereo
            wav = torchaudio_resample(wav, full_sr, demucs_sr)
            ref_mean, ref_std = wav.mean(), wav.std() + 1e-8
            with torch.no_grad():
                out = separator((wav[None] - ref_mean) / ref_std)[0]
            drums = (out[drum_idx] * ref_std + ref_mean).mean(0).numpy()
            scores["demucs"].append(track_f(drums, demucs_sr, ref))

    cond = "DEGRADED" if args.degrade else "clean"
    print(f"\nBeat-tracking F-measure vs known MMA grid, {len(scores['full'])} {cond} songs "
          f"(70 ms tol, 1.0 = perfect):")
    for name in scores:
        v = scores[name]
        print(f"    {name:<9} {np.mean(v):.3f}   (per-song min {np.min(v):.2f} max {np.max(v):.2f})")
    print("\nIf 'drums' >> 'full', an isolated percussion stem is the beat-tracking lever;")
    print("'demucs' shows whether we can recover that stem from real audio (torchaudio, MIT).")


def torchaudio_resample(wav, sr, target):
    import torchaudio.functional as AF
    return wav if sr == target else AF.resample(wav, sr, target)


if __name__ == "__main__":
    main()
