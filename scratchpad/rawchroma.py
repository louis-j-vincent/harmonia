"""rawchroma.py — genuinely UNTRAINED real-audio chord vector: raw NNLS
bothchroma (bass|treble), no trained root/quality head anywhere in the path.
Corrects the earlier V4 mislabeling (docs/known_issues.md CORRECTION,
2026-07-18): that version went through heads.root_proba, a pre-trained
classifier — this one does not.

Pipeline: nf.extract_bothchroma (raw VAMP NNLS-Chroma) -> nf.pool_beats
(per-beat 24-d bass|treble, L2-per-half, C-frame) -> per-bar mean -> whole-
song key-normalization (rigid rotation to a chroma-energy-argmax tonic, same
"whole-song rigid shift" convention as every other keynorm in this project,
just estimated from raw chroma energy instead of a trained root posterior).

Three vector variants per bar, for comparison:
  bt_concat  — 24-d, bass 12 + treble 12 concatenated (both registers)
  bass_only  — 12-d, bass half only
  treble_only— 12-d, treble half only
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from harmonia.models import nnls_features as nf
from real_root_proba import beat_grid


def per_bar_rawchroma(audio_path: Path, beats_per_bar: int = 4):
    """-> dict of {variant: (n_bars, d) np.float64 array}, bar_times, tempo."""
    bt, tempo_bpm, duration_s = beat_grid(audio_path)
    arr, times = nf.extract_bothchroma(audio_path)
    beat24 = nf.pool_beats(arr, times, bt)          # (n_beats,24) bass|treble
    n_beats = len(beat24)
    n_bars = max(1, n_beats // beats_per_bar)
    bar24 = np.zeros((n_bars, 24), np.float64)
    bar_times = []
    for b in range(n_bars):
        s, e = b * beats_per_bar, min((b + 1) * beats_per_bar, n_beats)
        bar24[b] = beat24[s:e].mean(0)
        bar_times.append(float(bt[s]))
    bar_times.append(float(bt[min(n_bars * beats_per_bar, len(bt) - 1)]))

    # whole-song key-norm: tonic = pc with highest total (bass+treble) energy,
    # same "rigid whole-song shift" convention as everywhere else tonight —
    # estimated here from raw chroma energy, not a trained posterior.
    total_energy = bar24[:, :12].sum(0) + bar24[:, 12:].sum(0)
    tonic = int(total_energy.argmax())
    shift = (-tonic) % 12
    bass_kn = np.roll(bar24[:, :12], shift, axis=1)
    treb_kn = np.roll(bar24[:, 12:], shift, axis=1)

    # per-bar mean over per-beat unit vectors does NOT stay unit-norm, and
    # bass/treble can end up at different relative magnitudes after
    # averaging — re-normalize EACH half independently, PER BAR, before
    # concatenating, so bt_concat's dot product weighs the two registers by
    # their harmonic content, not by an averaging artifact (user's fix,
    # 2026-07-18).
    def _l2_rows(m):
        n = np.linalg.norm(m, axis=1, keepdims=True)
        return m / np.clip(n, 1e-9, None)

    bass_n = _l2_rows(bass_kn)
    treb_n = _l2_rows(treb_kn)
    bt_concat = np.concatenate([bass_n, treb_n], axis=1)

    variants = {
        "bt_concat": bt_concat,
        "bass_only": bass_n,
        "treble_only": treb_n,
    }
    return variants, np.array(bar_times), tempo_bpm, tonic


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("audio", type=Path)
    args = ap.parse_args()
    variants, bar_times, tempo, tonic = per_bar_rawchroma(args.audio)
    print("tempo=%.1f bpm  n_bars=%d  tonic_pc=%d" %
          (tempo, len(variants["bt_concat"]), tonic))
    for name, v in variants.items():
        print("  %-11s shape=%s  row0 norm=%.3f" % (name, v.shape, np.linalg.norm(v[0])))
