"""real_root_proba.py — per-bar 12-d root softmax from a real audio file, end
to end, using the EXISTING nnls24 heads (no re-derivation of the model).

Reuses exactly the pieces chord_pipeline_v1.infer_chords_v1(feature_frontend
="nnls24") uses internally (nf.extract_bothchroma, nf.pool_beats,
heads.root_proba), plus the SAME beat-grid construction
(librosa.beat.beat_track + circular-mean phase correction) that
infer_chords_v1 does before handing off to _infer_nnls24 — duplicated here
only because bt isn't exposed as a return value, not because the beat
tracker itself needed re-deriving. Grid is quarter-note beats; bars = groups
of 4 beats starting at beat 0 (same fixed-phase-0 assumption already flagged
in docs/known_issues.md's GRID PHASE MISALIGNMENT entry — not fixed here,
inherited as-is).

Stage B of docs/research_sessions/structure_realaudio_2026_07_18.md.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import librosa

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harmonia.models import nnls_features as nf


def beat_grid(audio_path: Path, y=None, sr=None):
    """Same construction as chord_pipeline_v1.infer_chords_v1's librosa path:
    tempo/beat_track -> circular-mean phase correction -> uniform quarter-
    note grid from phase to duration. Returns (bt, tempo_bpm, duration_s)."""
    if y is None:
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
    duration_s = len(y) / sr
    tempo_arr, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    tempo_bpm = float(np.atleast_1d(tempo_arr)[0])
    beat_times_raw = librosa.frames_to_time(beat_frames, sr=sr)
    period = 60.0 / max(tempo_bpm, 1.0)
    ang = 2 * np.pi * (beat_times_raw % period) / period
    phase = (np.angle(np.mean(np.exp(1j * ang))) % (2 * np.pi)) * period / (2 * np.pi)
    bt = np.arange(phase, duration_s + period, period)
    bt = np.unique(np.concatenate([[0.0], bt, [duration_s]]))
    return bt, tempo_bpm, duration_s


def per_beat_root_proba(audio_path: Path):
    """-> (beat_proba (n_beats,12), bt (n_beats+1,), tempo_bpm)."""
    heads = nf.get_heads()
    if heads is None:
        raise RuntimeError("nnls24 heads unavailable — run scripts/train_nnls24_heads.py")
    bt, tempo_bpm, duration_s = beat_grid(audio_path)
    arr, times = nf.extract_bothchroma(audio_path)
    feat = nf.pool_beats(arr, times, bt)          # (n_beats, 24)
    beat_proba = heads.root_proba(feat)            # (n_beats, 12)
    return beat_proba, bt, tempo_bpm


def per_bar_root_proba(audio_path: Path, beats_per_bar: int = 4):
    """Aggregate per-beat root softmax to per-BAR 13-d vectors (12 pc + a
    'no-chord' mass, always 0 here since heads.root_proba has no NC class —
    kept as a 13th dim so this is a drop-in match for BlockEncoder's
    root_mode='proba' input, which was trained with an NC slot for symbolic
    'no chord' bars). Bars = fixed groups of `beats_per_bar` beats starting
    at beat 0 (inherits the known fixed-phase-0 bar-grid limitation — no
    downbeat detection here, matches the existing pipeline's own assumption).

    Aggregation: arithmetic mean of the per-beat softmaxes within the bar,
    renormalized to sum to 1 over the 12 real pcs (NC mass stays 0).
    Returns (bar_proba (n_bars,13), bar_times (n_bars+1,) in seconds)."""
    beat_proba, bt, tempo_bpm = per_beat_root_proba(audio_path)
    n_beats = len(beat_proba)
    n_bars = max(1, n_beats // beats_per_bar)
    bar_proba = np.zeros((n_bars, 13), np.float32)
    bar_times = []
    for b in range(n_bars):
        s, e = b * beats_per_bar, min((b + 1) * beats_per_bar, n_beats)
        v = beat_proba[s:e].mean(0)
        v = v / max(v.sum(), 1e-9)
        bar_proba[b, :12] = v
        bar_times.append(float(bt[s]))
    bar_times.append(float(bt[min(n_bars * beats_per_bar, len(bt) - 1)]))
    return bar_proba, np.array(bar_times), tempo_bpm


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    args = ap.parse_args()
    bar_proba, bar_times, tempo = per_bar_root_proba(args.audio)
    print("tempo=%.1f bpm, n_bars=%d" % (tempo, len(bar_proba)))
    for i, row in enumerate(bar_proba[:16]):
        top = int(row[:12].argmax())
        print("bar %3d  t=%6.2fs  top_pc=%2d (p=%.2f)  entropy=%.2f"
              % (i, bar_times[i], top, row[top],
                 float(-(row[:12] * np.log(row[:12] + 1e-9)).sum())))
