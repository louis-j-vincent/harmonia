"""HCDF (Harte harmonic-change) boundary detection vs the rigid beat grid.

Premise: on expressive audio the beat grid is a suggestion, not law.  Detect chord
boundaries from FRAME-level harmonic novelty first (before beat pooling), then snap
loosely to the grid as post-processing.

Pipeline:
  frame chroma (Basic Pitch note activations, 88→12) → librosa tonnetz (6D tonal
  centroid) → Gaussian smooth → HCDF ξ(t)=||c(t+1)-c(t-1)|| → peak-pick → boundary
  times → (optional) snap to nearest beat.

Measures, per corpus:
  (1) boundary F vs GT chord-change times (±½ beat tol) for HCDF / HCDF-snapped /
      fixed grids / oracle.
  (2) DOWNSTREAM root accuracy: label each segmentation's segments with beat_seq v4
      (sum per-beat proba → argmax), score per-beat (duration-weighted) vs GT root.
      This is the real test — can HCDF replace ORACLE segmentation?

Usage:
    .venv/bin/python scripts/hcdf_boundary_probe.py --corpus pop909
    .venv/bin/python scripts/hcdf_boundary_probe.py --corpus jazz --n 20
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

import librosa
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from analyze_accomp_emission import song_chord_spans
from build_audio_chord_features import BUCKET_FAMILY
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig
from harmonia.data.pop909_parser import POP909Parser
from harmonia.models.stage1_pitch import PitchExtractor
from harmonia.models.chord_pipeline_v1 import _BeatSeqModelV4, _pool_beats, MODELS
from train_beat_seq_model_v3 import _tempo_grid_beats, HARTE_TO_PC

DB = REPO / "data" / "accomp_db" / "db.jsonl"
POP = REPO / "data" / "pop909" / "POP909"

FOLD = np.zeros((88, 12), np.float32)
for _k in range(88):
    FOLD[_k, (21 + _k) % 12] = 1.0


# ── HCDF ──────────────────────────────────────────────────────────────────────

def frame_chroma(note_probs: np.ndarray) -> np.ndarray:
    c = note_probs @ FOLD                      # (T,12) raw
    n = np.linalg.norm(c, axis=1, keepdims=True)
    return c / (n + 1e-9)


def hcdf(chroma: np.ndarray, dt: float, sigma_sec: float = 0.15) -> np.ndarray:
    """chroma (T,12) → detection function ξ (T,) via smoothed tonal-centroid novelty."""
    tc = librosa.feature.tonnetz(chroma=chroma.T)          # (6,T)
    tc = gaussian_filter1d(tc, sigma=max(sigma_sec / dt, 0.5), axis=1)
    xi = np.zeros(tc.shape[1], np.float32)
    xi[1:-1] = np.linalg.norm(tc[:, 2:] - tc[:, :-2], axis=0)
    xi[0] = xi[1]; xi[-1] = xi[-2]
    return xi


def pick_boundaries(xi, frame_times, min_gap_sec, k_std):
    dt = np.median(np.diff(frame_times))
    dist = max(int(min_gap_sec / dt), 1)
    thr = xi.mean() + k_std * xi.std()
    pk, _ = find_peaks(xi, distance=dist, height=thr)
    return frame_times[pk]


def snap_to_grid(times, beat_times, tol):
    out = []
    for t in times:
        j = int(np.argmin(np.abs(beat_times - t)))
        out.append(beat_times[j] if abs(beat_times[j] - t) <= tol else t)
    return np.unique(out)


# ── metrics ───────────────────────────────────────────────────────────────────

def boundary_f(est, gt, tol):
    if len(gt) == 0:
        return (1.0, 1.0, 1.0) if len(est) == 0 else (0.0, 0.0, 0.0)
    if len(est) == 0:
        return 0.0, 0.0, 0.0
    gt_used = np.zeros(len(gt), bool)
    tp = 0
    for t in est:
        d = np.abs(gt - t); j = int(np.argmin(d))
        if d[j] <= tol and not gt_used[j]:
            gt_used[j] = True; tp += 1
    p = tp / len(est); r = tp / len(gt)
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return f, p, r


def segs_from_boundaries(bnd_times, bt):
    """boundary times → list of (b0,b1) beat-index segments over the beat grid bt."""
    n = len(bt) - 1
    cuts = sorted(set([0] + [int(np.argmin(np.abs(bt - t))) for t in bnd_times] + [n]))
    cuts = [c for c in cuts if 0 <= c <= n]
    segs = [(cuts[i], cuts[i+1]) for i in range(len(cuts)-1) if cuts[i+1] > cuts[i]]
    return segs or [(0, n)]


def grid_merge_boundaries(v4, onset_b, note_b, bt):
    """Model-driven segmentation: boundary wherever v4's per-beat root argmax changes.
    No grid law beyond the beat, no oracle — the honest non-oracle baseline."""
    pred = v4.predict_proba(onset_b, note_b).argmax(1)
    return np.array([bt[b] for b in range(1, len(pred)) if pred[b] != pred[b-1]])


def root_acc_under(segs, v4, onset_b, note_b, gt_root, valid):
    """Label each seg by v4 (sum per-beat proba→argmax), score per-beat vs GT."""
    proba = v4.predict_proba(onset_b, note_b)     # (n,12)
    pred = np.zeros(len(gt_root), int)
    for b0, b1 in segs:
        r = int(proba[b0:b1].sum(0).argmax())
        pred[b0:b1] = r
    m = valid
    return (pred[m] == gt_root[m]).mean()


# ── data loaders (frame acts + beat grid + GT) ────────────────────────────────

def load_pop(sid, ex):
    wav = REPO / "data" / "renders" / "pop909" / sid / f"{sid}_v005_musescoregeneral.wav"
    if not wav.exists():
        return None
    y, sr = sf.read(wav); y = (y.mean(1) if y.ndim > 1 else y).astype("float32")
    acts = ex.extract(wav)
    bt, _ = _tempo_grid_beats(y, sr)
    parser = POP909Parser(POP); song = parser.parse_song(sid)
    n = len(bt) - 1
    gt_root = np.full(n, -1, int); valid = np.zeros(n, bool)
    prev = None; gt_change = []
    for b in range(n):
        ev = song.chord_at_time(0.5 * (bt[b] + bt[b+1]))
        lbl = ev.label if ev else "N"
        rs = lbl.split(":")[0].split("/")[0]
        if rs in HARTE_TO_PC:
            gt_root[b] = HARTE_TO_PC[rs]; valid[b] = True
        if lbl != prev:
            gt_change.append(bt[b]); prev = lbl
    return acts, bt, gt_root, valid, np.array(gt_change[1:])  # drop t0


def load_jazz(rec, renderer, sf2, ex):
    spb = 60.0 / rec["tempo"]; n_beats = rec["n_bars"] * rec["beats_per_bar"]
    spans = [(t0, t1, r % 12) for t0, t1, r, q in song_chord_spans(rec)
             if t1 > t0 and q in BUCKET_FAMILY]
    if not spans:
        return None
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
        tmp = Path(wf.name)
    try:
        renderer.render(REPO / rec["midi_path"], tmp, RenderConfig(soundfont_path=sf2))
        acts = ex.extract(tmp, use_cache=False)
    finally:
        tmp.unlink(missing_ok=True)
    bt = np.arange(n_beats + 1) * spb

    def gtr(t):
        for t0, t1, r in spans:
            if t0 <= t < t1:
                return r
        return None
    gt_root = np.full(n_beats, -1, int); valid = np.zeros(n_beats, bool)
    for b in range(n_beats):
        r = gtr((b + 0.5) * spb)
        if r is not None:
            gt_root[b] = r; valid[b] = True
    gt_change = np.array([t0 for t0, _, _ in spans[1:]])
    return acts, bt, gt_root, valid, gt_change


# ── run ───────────────────────────────────────────────────────────────────────

def run_song(acts, bt, gt_root, valid, gt_change, v4, k_std):
    dt = float(np.median(np.diff(acts.frame_times)))
    beat_int = float(np.median(np.diff(bt)))
    tol = 0.5 * beat_int
    ch = frame_chroma(acts.note_probs)
    xi = hcdf(ch, dt)
    hb = pick_boundaries(xi, acts.frame_times, min_gap_sec=beat_int, k_std=k_std)
    hb_snap = snap_to_grid(hb, bt, tol)

    onset_b = _pool_beats(acts.frame_times, acts.onset_probs, bt)
    note_b = _pool_beats(acts.frame_times, acts.note_probs, bt)

    # boundary sets
    grid1 = bt[1:-1]                              # every beat
    grid2 = bt[2:-1:2]                            # every 2 beats
    gmerge = grid_merge_boundaries(v4, onset_b, note_b, bt)  # model-driven
    hybrid = np.unique(np.concatenate([hb_snap, gmerge]))    # HCDF ∪ model-merge
    sets = {"hcdf": hb, "hcdf_snap": hb_snap, "grid1": grid1, "grid2": grid2,
            "gmerge": gmerge, "hybrid": hybrid}
    bf = {k: boundary_f(v, gt_change, tol)[0] for k, v in sets.items()}

    # downstream root accuracy
    oracle_segs = segs_from_boundaries(gt_change, bt)
    racc = {"oracle": root_acc_under(oracle_segs, v4, onset_b, note_b, gt_root, valid)}
    for k, v in sets.items():
        racc[k] = root_acc_under(segs_from_boundaries(v, bt), v4, onset_b, note_b, gt_root, valid)
    nseg = {k: len(segs_from_boundaries(v, bt)) for k, v in sets.items()}
    nseg["oracle"] = len(oracle_segs)
    return bf, racc, nseg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["pop909", "jazz"], default="pop909")
    ap.add_argument("--n", type=int, default=20, help="jazz songs")
    ap.add_argument("--k-std", type=float, default=1.0, help="peak threshold = mean + k*std")
    args = ap.parse_args()

    ex = PitchExtractor(cache_dir=REPO / "data" / "cache" if args.corpus == "pop909" else None)
    v4 = _BeatSeqModelV4(MODELS / "beat_seq_model_v4.npz")

    songs = []
    if args.corpus == "pop909":
        for sid in ("001", "002", "003", "004", "005"):
            r = load_pop(sid, ex)
            if r: songs.append((sid, r))
    else:
        renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
        sf2 = renderer._find_soundfont("MuseScore_General.sf2")
        recs = [json.loads(l) for l in open(DB)]
        jz = [r for r in recs if r.get("corpus") == "jazz1460"
              and r["beats_per_bar"] == 4 and (REPO / r["midi_path"]).exists()][1::2][:args.n]
        for i, rec in enumerate(jz):
            print(f"  render [{i+1}/{len(jz)}] {rec['song_id']}", end="\r", flush=True)
            r = load_jazz(rec, renderer, sf2, ex)
            if r: songs.append((rec["song_id"], r))
        print()

    keys = ("hcdf", "hcdf_snap", "grid1", "grid2", "gmerge", "hybrid")
    BF = {k: [] for k in keys}
    RA = {k: [] for k in ("oracle",) + keys}
    NS = {k: [] for k in RA}
    for sid, (acts, bt, gt_root, valid, gt_change) in songs:
        bf, racc, nseg = run_song(acts, bt, gt_root, valid, gt_change, v4, args.k_std)
        for k in BF: BF[k].append(bf[k])
        for k in RA: RA[k].append(racc[k]); NS[k].append(nseg[k])

    print(f"\n=== {args.corpus}  ({len(songs)} songs, k_std={args.k_std}, tol=½ beat) ===")
    print(f"{'segmentation':<12} {'boundF':>7} {'rootAcc':>8} {'segs/song':>10}")
    print("-" * 42)
    print(f"{'oracle':<12} {'1.000':>7} {np.mean(RA['oracle']):>8.1%} {np.mean(NS['oracle']):>10.1f}")
    for k in ("hcdf", "hcdf_snap", "grid1", "grid2", "gmerge", "hybrid"):
        print(f"{k:<12} {np.mean(BF[k]):>7.3f} {np.mean(RA[k]):>8.1%} {np.mean(NS[k]):>10.1f}")


if __name__ == "__main__":
    main()
