"""Render accompaniment-DB MIDIs to audio with controlled variation + noise.

Purpose: create (audio, perfect-MIDI-ground-truth) pairs for learning the
audio→notes mapping — the Stage-1 evidence-quality bottleneck identified in
docs/accomp_db_signal_analysis_2026-07-03.md.

Variation axes (per variant, drawn deterministically from the song index):
  - transposition  ∈ {-5..+6} semitones  (MIDI notes AND ground-truth chords
    shift together — free key coverage)
  - soundfont      ∈ {MuseScore_General, GeneralUser(VintageDreams)}
  - reverb         ∈ {on, off}
  - additive pink noise ∈ {clean, 15 dB SNR, 8 dB SNR}

Each sampled song gets one CLEAN canonical render (no transpose, good
soundfont, no noise) plus `--variants-per-song` varied renders.

Output: data/accomp_db/audio/<song_id>_<variant>.wav + manifest.jsonl
(one line per render: song_id, wav path, transpose, soundfont, snr, …).

Usage:
    .venv/bin/python scripts/build_accomp_audio.py --n-songs 60 --variants-per-song 1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig  # noqa: E402

DB = REPO / "data" / "accomp_db" / "db.jsonl"
AUDIO_DIR = REPO / "data" / "accomp_db" / "audio"
SOUNDFONTS = ["MuseScore_General.sf2", "GeneralUser.sf2"]


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    """Pink (1/f) noise via FFT shaping, unit variance."""
    white = rng.standard_normal(n)
    spec = np.fft.rfft(white)
    freqs = np.fft.rfftfreq(n)
    freqs[0] = freqs[1]
    spec /= np.sqrt(freqs)
    pink = np.fft.irfft(spec, n)
    return pink / (pink.std() + 1e-12)


def add_noise(wav_path: Path, snr_db: float, rng: np.random.Generator) -> None:
    """Add pink noise at the given SNR, in place."""
    audio, sr = sf.read(wav_path)
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    sig_power = float(np.mean(audio**2)) + 1e-12
    noise = pink_noise(len(audio), rng)
    noise_power = sig_power / (10 ** (snr_db / 10))
    audio = audio + noise * np.sqrt(noise_power)
    peak = np.abs(audio).max()
    if peak > 0.99:
        audio = audio * 0.99 / peak
    sf.write(wav_path, audio, sr)


def transpose_midi(src: Path, dst: Path, semitones: int) -> None:
    pm = pretty_midi.PrettyMIDI(str(src))
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            n.pitch = int(np.clip(n.pitch + semitones, 0, 127))
    pm.write(str(dst))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-songs", type=int, default=60)
    ap.add_argument("--variants-per-song", type=int, default=1)
    ap.add_argument("--corpus", default="jazz1460")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    records = [json.loads(line) for line in open(DB)]
    songs = [r for r in records if r["corpus"] == args.corpus and r["beats_per_bar"] == 4]
    # stratify: sort by groove then stride-sample so all grooves are represented
    songs.sort(key=lambda r: (r["groove"], r["song_id"]))
    stride = max(len(songs) // args.n_songs, 1)
    songs = songs[::stride][: args.n_songs]

    rng = np.random.default_rng(args.seed)
    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    tmp_midi_dir = AUDIO_DIR / "midi_variants"
    tmp_midi_dir.mkdir(exist_ok=True)

    manifest_path = AUDIO_DIR / "manifest.jsonl"
    n_done = 0
    with open(manifest_path, "w") as mf:
        for rec in songs:
            src_mid = REPO / rec["midi_path"]
            if not src_mid.exists():
                continue
            variants = [dict(transpose=0, sf=SOUNDFONTS[0], reverb=True, snr=None)]
            for _ in range(args.variants_per_song):
                variants.append(
                    dict(
                        transpose=int(rng.integers(-5, 7)),
                        sf=SOUNDFONTS[int(rng.integers(0, len(SOUNDFONTS)))],
                        reverb=bool(rng.integers(0, 2)),
                        snr=[None, 15.0, 8.0][int(rng.integers(0, 3))],
                    )
                )
            for vi, v in enumerate(variants):
                tag = (f"v{vi}_t{v['transpose']:+d}_{Path(v['sf']).stem[:8]}"
                       f"_{'rev' if v['reverb'] else 'dry'}"
                       f"_{'clean' if v['snr'] is None else f'snr{int(v['snr'])}'}")
                wav_path = AUDIO_DIR / f"{rec['song_id']}_{tag}.wav"
                if not wav_path.exists():
                    if v["transpose"] != 0:
                        mid_path = tmp_midi_dir / f"{rec['song_id']}_t{v['transpose']:+d}.mid"
                        if not mid_path.exists():
                            transpose_midi(src_mid, mid_path, v["transpose"])
                    else:
                        mid_path = src_mid
                    config = RenderConfig(
                        soundfont_path=renderer._find_soundfont(v["sf"]),
                        reverb=v["reverb"],
                    )
                    renderer.render(mid_path, wav_path, config)
                    if v["snr"] is not None:
                        add_noise(wav_path, v["snr"], rng)
                mf.write(json.dumps({
                    "song_id": rec["song_id"],
                    "wav": str(wav_path.relative_to(REPO)),
                    "variant": tag,
                    "transpose": v["transpose"],
                    "soundfont": v["sf"],
                    "reverb": v["reverb"],
                    "snr_db": v["snr"],
                    "tempo": rec["tempo"],
                    "beats_per_bar": rec["beats_per_bar"],
                    "n_bars": rec["n_bars"],
                    "midi_path": rec["midi_path"],
                }) + "\n")
                n_done += 1
                if n_done % 20 == 0:
                    print(f"  … {n_done} renders")
    print(f"Done: {n_done} renders → {AUDIO_DIR}\nManifest: {manifest_path}")


if __name__ == "__main__":
    main()
