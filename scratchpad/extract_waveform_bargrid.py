"""extract_waveform_bargrid.py — for the waveform+bar-grid debug player: a
downsampled amplitude envelope (for rendering) + the EXACT bar-grid
timestamps the production pipeline uses (scratchpad/real_root_proba.py's
beat_grid(), same construction chord_pipeline_v1.py uses — see the
'GRID PHASE MISALIGNMENT' / 'Part A drift root-cause' known_issues.md
entries for why this specific grid is under scrutiny tonight)."""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import librosa
from real_root_proba import beat_grid

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent

SONGS = {
    "aretha_chain_of_fools": "aretha_franklin_chain_of_fools_official_lyric_video.m4a",
    "autumn_leaves": "autumn_leaves.m4a",
    "abba_chiquitita": "abba_chiquitita_official_lyric_video.m4a",
}

N_ENVELOPE_POINTS = 2000  # downsampled amplitude points for rendering


def extract(name, fname):
    path = REPO / "docs" / "audio" / fname
    # NATIVE sample rate (sr=None), matching beat_grid()'s own default path
    # exactly (no y/sr pre-supplied) -- this is what the production chart
    # actually uses. An earlier pass of this script pre-loaded at sr=22050,
    # which reproduces the ALREADY-FLAGGED-BUT-UNINVESTIGATED sample-rate-
    # dependent tempo-octave lock (known_issues.md "★ CHART / BAR-GRID":
    # 184.57 BPM at native 44.1kHz vs 92.29 BPM at 22.05kHz on the SAME
    # audio) -- autumn_leaves flipped from 187.5 BPM/330 bars to 92.3 BPM/
    # 163 bars purely from that resampling choice, confirming the bug is
    # real and still live. Using sr=None here so this tool shows the SAME
    # grid the real chart uses, not a third, different one.
    y, sr = librosa.load(str(path), sr=None, mono=True)
    duration = len(y) / sr

    # downsampled abs-amplitude envelope for waveform rendering
    hop = max(1, len(y) // N_ENVELOPE_POINTS)
    env = np.array([np.abs(y[i:i + hop]).max() for i in range(0, len(y), hop)])
    env = env / (env.max() + 1e-9)

    bt, tempo_bpm, dur = beat_grid(path, y=y, sr=sr)
    bar_times = bt[::4].tolist()  # bars = every 4 beats, matches production bpb=4

    return {
        "song": name, "duration_s": float(duration), "tempo_bpm": float(tempo_bpm),
        "envelope": [round(float(v), 4) for v in env],
        "n_envelope_points": len(env),
        "beat_times_s": [round(float(t), 4) for t in bt],
        "bar_times_s": [round(float(t), 4) for t in bar_times],
        "n_bars": len(bar_times) - 1,
        "audio_file": fname,
    }


def main():
    out = {}
    for name, fname in SONGS.items():
        print("=== %s ===" % name)
        d = extract(name, fname)
        print("  duration=%.1fs tempo=%.1f bpm n_bars=%d envelope_pts=%d" %
              (d["duration_s"], d["tempo_bpm"], d["n_bars"], d["n_envelope_points"]))
        out[name] = d
    (OUT_DIR / "waveform_bargrid_data.json").write_text(json.dumps(out))
    print("wrote scratchpad/waveform_bargrid_data.json")


if __name__ == "__main__":
    main()
