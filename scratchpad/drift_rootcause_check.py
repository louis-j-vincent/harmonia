"""Part A root-cause: is the span->beat-count for pool_beat_evidence's
precondition computed via CONSTANT-TEMPO ARITHMETIC on an idealized uniform
grid, or by counting REAL DETECTED beat timestamps?

This reproduces chord_pipeline_v1's EXACT beat-grid construction (lines
~2907-2927) and compares, for the 3 real songs and their actual failing
cluster spans:

  (a) the uniform grid `bt` = np.arange(phase, dur+period, period)  [production]
  (b) the raw librosa/madmom detected beat times `beat_times_raw`

and measures:
  1. cumulative drift between (a) and (b) as a function of absolute time /
     bar index -- when does it exceed half a beat (=> +-1 quantization likely)?
  2. for each real cluster's whole-block spans, the beat COUNT each span maps
     to under `_span_to_beats` (uniform argmin) vs. counting real beats in
     [t0,t1) -- which encoding makes a nominally-equal cluster equal-count?
"""
from __future__ import annotations
import sys, json, tempfile, shutil, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
import librosa
import soundfile as sf

from harmonia.models.user_constraints import _span_to_beats

REPO = Path(__file__).resolve().parent.parent
AUDIO = REPO / "docs" / "audio"
OUT = Path(__file__).resolve().parent

SONGS = {
    "aretha_franklin_chain_of_fools_official_lyric_video": {"audio": "aretha_franklin_chain_of_fools_official_lyric_video.m4a", "k": 3},
    "abba_chiquitita_official_lyric_video": {"audio": "abba_chiquitita_official_lyric_video.m4a", "k": 4},
    "autumn_leaves": {"audio": "autumn_leaves.m4a", "k": 5},
}


def transcode(src, dst):
    subprocess.run(["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", "22050", str(dst)],
                   check=True, capture_output=True, timeout=180)


def build_production_bt(wav):
    """EXACT copy of chord_pipeline_v1 beat-grid construction."""
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
    return bt, beat_times_raw, tempo_bpm, period, duration_s


def count_real_beats_in_span(t0, t1, beat_times_raw):
    return int(np.sum((beat_times_raw >= t0) & (beat_times_raw < t1)))


def main():
    dual = json.loads((OUT / "dual_matrix_grain8_results.json").read_text())
    clusters = json.loads((OUT / "section_structure_clusters_grain8.json").read_text())
    report = {}

    for slug, cfg in SONGS.items():
        print(f"\n{'='*70}\n{slug}\n{'='*70}")
        tmp = Path(tempfile.mkdtemp(prefix="drift_"))
        try:
            wav = tmp / "a.wav"
            transcode(AUDIO / cfg["audio"], wav)
            bt, raw, tempo, period, dur = build_production_bt(wav)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

        print(f"  tempo={tempo:.2f} bpm  period={period:.4f}s  dur={dur:.1f}s  "
              f"uniform bt: {len(bt)} pts  raw beats: {len(raw)}")

        # --- (1) cumulative drift: uniform grid vs real detected beats -------
        # For each real detected beat, find the nearest uniform grid point and
        # record |offset| (in fractions of a beat). Drift = how far the idealized
        # constant-tempo grid has walked away from the real beat by that time.
        drift_frac = []
        for tb in raw:
            j = int(np.abs(bt - tb).argmin())
            drift_frac.append(abs(bt[j] - tb) / period)
        drift_frac = np.array(drift_frac)
        # find first real beat whose drift exceeds 0.5 beat (=> the uniform
        # grid would quantize a nearby span endpoint to the WRONG beat)
        over = np.where(drift_frac > 0.5)[0]
        first_over_beat = int(over[0]) if len(over) else None
        first_over_time = float(raw[over[0]]) if len(over) else None
        print(f"  drift |uniform - real| / period: mean={drift_frac.mean():.3f} "
              f"max={drift_frac.max():.3f}")
        if first_over_beat is not None:
            print(f"  FIRST >0.5-beat drift at real-beat #{first_over_beat} "
                  f"(t={first_over_time:.1f}s, ~bar {first_over_beat//4})")
        else:
            print("  drift never exceeds 0.5 beat")

        # also: net signed cumulative drift = (bt uniform count up to t) vs
        # (real beats up to t) -- the accumulation the brief asks about
        n_grid = np.searchsorted(bt, raw)       # uniform grid pts before each real beat
        n_real = np.arange(1, len(raw) + 1)
        net_drift = n_grid - n_real
        print(f"  net (uniform_count - real_count) across song: "
              f"start={net_drift[0]}, end={net_drift[-1]}, range=[{net_drift.min()},{net_drift.max()}]")

        # --- (2) real cluster whole-block spans: uniform vs real-beat count ---
        d = dual[slug]
        block_times = d["block_times_s"]
        cl = clusters[slug]
        letters = {}
        for b in cl["blocks"]:
            letters.setdefault(b["section"], []).append(b["block"])

        song_rep = {"tempo_bpm": tempo, "period_s": period, "duration_s": dur,
                    "n_uniform_grid_pts": len(bt), "n_raw_beats": len(raw),
                    "drift_frac_mean": float(drift_frac.mean()),
                    "drift_frac_max": float(drift_frac.max()),
                    "first_over_half_beat_at_time_s": first_over_time,
                    "net_drift_end": int(net_drift[-1]),
                    "clusters": {}}

        from collections import Counter
        for letter, members in sorted(letters.items()):
            if len(members) < 2:
                continue
            members = sorted(members)
            uni_counts, real_counts, dur_counts, starts = [], [], [], []
            for m in members:
                t0, t1 = block_times[m]
                b0, b1 = _span_to_beats(t0, t1, bt)
                uni_counts.append(b1 - b0)                          # CURRENT: argmin(t1)-argmin(t0) (double-quantized)
                real_counts.append(count_real_beats_in_span(t0, t1, raw))  # real detected beats in span
                dur_counts.append(int(round((t1 - t0) / period)))   # DURATION-based: round(dur/period) (single-quantized)
                starts.append(t0)
            uni_equal = len(set(uni_counts)) == 1
            real_equal = len(set(real_counts)) == 1
            dur_equal = len(set(dur_counts)) == 1
            # accumulation test: does |count - mode| grow with block start time?
            mode = Counter(uni_counts).most_common(1)[0][0]
            dev = [abs(c - mode) for c in uni_counts]
            pos_corr = float(np.corrcoef(starts, dev)[0, 1]) if len(set(dev)) > 1 else 0.0
            print(f"  cluster {letter} (n={len(members)}): "
                  f"CURRENT(argmin)={uni_counts} eq={uni_equal} | "
                  f"real-beats={real_counts} eq={real_equal} | "
                  f"DURATION(round dur/period)={dur_counts} eq={dur_equal} | "
                  f"pos-corr(dev,time)={pos_corr:+.2f}")
            song_rep["clusters"][letter] = {
                "members": members,
                "current_argmin_beat_counts": uni_counts, "current_equal": uni_equal,
                "real_beat_counts": real_counts, "real_equal": real_equal,
                "duration_based_beat_counts": dur_counts, "duration_equal": dur_equal,
                "dev_vs_time_correlation": pos_corr,
            }
        report[slug] = song_rep

    (OUT / "drift_rootcause_check_results.json").write_text(json.dumps(report, indent=2))
    print(f"\nwrote {OUT / 'drift_rootcause_check_results.json'}")

    # ---- headline summary ----
    print(f"\n{'='*70}\nHEADLINE\n{'='*70}")
    for slug, r in report.items():
        n_cur = sum(1 for c in r["clusters"].values() if c["current_equal"])
        n_real = sum(1 for c in r["clusters"].values() if c["real_equal"])
        n_dur = sum(1 for c in r["clusters"].values() if c["duration_equal"])
        n_tot = len(r["clusters"])
        print(f"{slug}: equal-count clusters -> CURRENT(argmin) {n_cur}/{n_tot}, "
              f"real-beat {n_real}/{n_tot}, DURATION {n_dur}/{n_tot}; "
              f"first >0.5-beat drift @ {r['first_over_half_beat_at_time_s']}s")


if __name__ == "__main__":
    main()
