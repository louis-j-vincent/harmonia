"""A/B: beat_period_mode librosa vs bestfit through the FULL live path.

CLAUDE.md rule #6 (component swaps change more than the target metric): before
any default flip, diff the intermediates — tempo, beat count, chord count,
label agreement, duration histogram — on the songs the madmom cross-reference
flagged as worst drifters (abba, commodores, leo_sayer, let_it_be) plus one
low-drift control (ronettes).

Run: .venv/bin/python scratchpad/beatgrid_ab_livepath.py
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SONGS = [
    "abba_chiquitita_official_lyric_video",
    "the_commodores_easy_1977",
    "leo_sayer_you_make_me_feel_like_dancing_official_hd_music_vi",
    "let_it_be_remastered_2009",
    "the_ronettes_be_my_baby_music_video",  # low-drift control
]
OUT = Path(__file__).with_suffix(".json")


def label_agreement(a, b, dur):
    """Time-weighted fraction of the song where both charts show the same label."""
    ts = np.linspace(0, dur, 2000, endpoint=False)

    def at(chords, t):
        for c in chords:
            if c["start_s"] <= t < c["end_s"]:
                return c["label"]
        return None

    same = sum(at(a, t) == at(b, t) for t in ts)
    return same / len(ts)


def main():
    from harmonia.models.chord_pipeline_v1 import infer_chords_v1

    rows = []
    for song in SONGS:
        m4a = REPO / "docs" / "audio" / f"{song}.m4a"
        with tempfile.TemporaryDirectory() as td:
            # CRITICAL: musx/nnls caches are keyed on the file STEM — a shared
            # temp name silently reuses song 1's analysis for every song (this
            # exact bug invalidated the first run of this script, 2026-07-19).
            wav = Path(td) / f"abtest_{song[:40]}.wav"
            subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(m4a),
                            "-ac", "1", "-ar", "22050", str(wav)], check=True)
            charts = {}
            for mode in ("librosa", "bestfit"):
                charts[mode] = infer_chords_v1(
                    wav, feature_frontend="nnls24", bass_frontend="musx",
                    quality_frontend="musx", beat_period_mode=mode)
        a, b = charts["librosa"], charts["bestfit"]
        agree = label_agreement(a.chords, b.chords, a.duration_s)
        row = {
            "song": song,
            "tempo": [a.tempo_bpm, b.tempo_bpm],
            "n_chords": [len(a.chords), len(b.chords)],
            "n_beats_total": [int(sum(c["duration_beats"] for c in a.chords)),
                              int(sum(c["duration_beats"] for c in b.chords))],
            "mean_conf": [round(float(np.mean([c["confidence"] for c in a.chords])), 3),
                          round(float(np.mean([c["confidence"] for c in b.chords])), 3)],
            "key": [a.global_key, b.global_key],
            "timeline_label_agreement": round(agree, 3),
        }
        rows.append(row)
        print(json.dumps(row))
    OUT.write_text(json.dumps(rows, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
