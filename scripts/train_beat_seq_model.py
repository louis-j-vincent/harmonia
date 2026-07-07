"""Train the beat-sequence root model: for every beat in the corpus, extract 48d
chroma features and concatenate ±window neighbours → windowed LR that gets 88.9%
per-beat root accuracy (validated in per_beat_context_experiment.py).

Saves harmonia/models/beat_seq_model.npz with keys:
  mean, scale, coef, intercept, classes  — the windowed LR root model (12-way)
  window                                 — context half-window (default 2)

Why train separately from root_model_experiment.py:
  - root_model is trained on SEGMENT-level pooled features (oracle chord boundaries)
    and works well for labeling known segments.
  - beat_seq_model is trained on BEAT-level features so it's correctly calibrated
    when we want per-beat root probabilities for (a) within-cell split decisions
    and (b) probabilistic segment labeling by pooling per-beat soft probs.
  - The ±window context (neighbours concatenated) is what recovers the per-beat gap
    from 85.5% (no context) → 88.9% (±2), validating the BiLSTM direction without
    needing a recurrent model yet.

Usage:
    .venv/bin/python scripts/train_beat_seq_model.py [--n-songs 50] [--window 2]
                                                     [--augment] [--parity 0|1]
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

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import GroupKFold       # noqa: E402
from sklearn.preprocessing import StandardScaler     # noqa: E402

from analyze_accomp_emission import song_chord_spans  # noqa: E402
from build_accomp_audio_hard import time_varying_degrade  # noqa: E402
from build_audio_chord_features import BUCKET_FAMILY  # noqa: E402
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig  # noqa: E402
from harmonia.models.stage1_pitch import PitchExtractor  # noqa: E402
from harmonic_rhythm_probe import pool_beats  # noqa: E402
from root_model_experiment import chroma88  # noqa: E402

DB = REPO / "data" / "accomp_db" / "db.jsonl"
OUT = REPO / "harmonia" / "models" / "beat_seq_model.npz"


def beat_features(onset_b: np.ndarray, note_b: np.ndarray) -> np.ndarray:
    """48d per-beat feature: chroma88 on onset (full, bass, treble) + note (full).
    Same as per_beat_context_experiment.py to reproduce the validated 88.9%."""
    n = len(onset_b)
    F = np.zeros((n, 48))
    for b in range(n):
        F[b] = np.concatenate([
            chroma88(onset_b[b]),            # full onset chroma  (12d)
            chroma88(note_b[b]),             # full note chroma   (12d)
            chroma88(onset_b[b], 0, 52),     # bass register      (12d)
            chroma88(onset_b[b], 60, 200),   # treble register    (12d)
        ])
    return F


def windowed_features(F: np.ndarray, window: int) -> np.ndarray:
    """Concatenate ±window neighbours for each beat (zero-pad edges).
    Input: (n_beats, 48).  Output: (n_beats, 48*(2*window+1))."""
    n, d = F.shape
    W = 2 * window + 1
    out = np.zeros((n, d * W))
    for b in range(n):
        row = []
        for delta in range(-window, window + 1):
            nb = b + delta
            row.append(F[nb] if 0 <= nb < n else np.zeros(d))
        out[b] = np.concatenate(row)
    return out


def collect_song(rec, renderer, sf2, ex, rng, degrade):
    """Render one song, extract per-beat features and GT roots.
    Returns (features (n_beats, 48), gt_roots (n_beats,), valid_mask (n_beats,)).
    GT root from song_chord_spans mid-beat lookup (authoritative single source)."""
    spb = 60.0 / rec["tempo"]
    bpb = rec["beats_per_bar"]
    n_beats = rec["n_bars"] * bpb

    spans = [(t0, t1, r % 12) for t0, t1, r, _q in song_chord_spans(rec)
             if t1 > t0 and _q in BUCKET_FAMILY]

    def gtroot(t):
        for t0, t1, root in spans:
            if t0 <= t < t1:
                return root
        return None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
        tmp = Path(wf.name)
    try:
        renderer.render(REPO / rec["midi_path"], tmp, RenderConfig(soundfont_path=sf2))
        y, sr = sf.read(tmp)
        y = (y.mean(1) if y.ndim > 1 else y).astype("float32")
        if degrade:
            y = time_varying_degrade(y, sr, rng)
            sf.write(tmp, y, sr)
        acts = ex.extract(tmp, use_cache=False)
    finally:
        tmp.unlink(missing_ok=True)

    bt = np.arange(n_beats + 1) * spb
    onset_b = pool_beats(acts.frame_times, acts.onset_probs, bt)
    note_b  = pool_beats(acts.frame_times, acts.note_probs, bt)

    F = beat_features(onset_b, note_b)
    gt = np.array([gtroot((b + 0.5) * spb) for b in range(n_beats)], dtype=object)
    valid = np.array([g is not None for g in gt])
    roots = np.zeros(n_beats, int)
    for b in range(n_beats):
        if valid[b]:
            roots[b] = int(gt[b])
    return F, roots, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-songs", type=int, default=50,
                    help="number of songs to train on (default 50, ~full corpus)")
    ap.add_argument("--window", type=int, default=2,
                    help="context half-window in beats (default 2 → 48*5=240d features)")
    ap.add_argument("--augment", action="store_true",
                    help="train on BOTH clean and degraded renders for noise robustness")
    ap.add_argument("--parity", type=int, default=None,
                    help="keep only songs whose number %% 2 == parity (disjoint split)")
    ap.add_argument("--eval", action="store_true",
                    help="5-fold CV eval instead of saving the model (like the experiment)")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(DB)]
    songs = [r for r in recs if r["corpus"] == "jazz1460"
             and r["beats_per_bar"] == 4
             and (REPO / r["midi_path"]).exists()]
    if args.parity is not None:
        songs = [r for r in songs if int(r["song_id"].split("_")[1]) % 2 == args.parity]
    songs = songs[:: max(len(songs) // args.n_songs, 1)][: args.n_songs]
    print(f"Training on {len(songs)} songs (window=±{args.window}, "
          f"augment={args.augment}, parity={args.parity})")

    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2 = renderer._find_soundfont("MuseScore_General.sf2")
    ex = PitchExtractor(cache_dir=None)
    rng = np.random.default_rng(7)

    all_feats, all_roots, all_songs = [], [], []
    for i, rec in enumerate(songs):
        sid = rec["song_id"]
        print(f"  [{i+1}/{len(songs)}] {sid} ...", end="\r", flush=True)
        try:
            F_clean, roots, valid = collect_song(rec, renderer, sf2, ex, rng, degrade=False)
        except Exception as e:
            print(f"\n  SKIP {sid}: {e}")
            continue
        n_beats = int(valid.sum())
        if n_beats == 0:
            continue
        all_feats.append(F_clean[valid])
        all_roots.append(roots[valid])
        all_songs.extend([sid] * n_beats)

        if args.augment:
            try:
                F_deg, roots_d, valid_d = collect_song(rec, renderer, sf2, ex, rng, degrade=True)
            except Exception:
                pass
            else:
                # GT is the same as clean; use clean roots for degraded beats
                # (same chord, degraded audio surface)
                all_feats.append(F_deg[valid])
                all_roots.append(roots[valid])
                all_songs.extend([sid + "_deg"] * int(valid.sum()))

    print(f"\nCollected {sum(len(f) for f in all_feats)} beat samples")
    all_feats_raw = np.vstack(all_feats)
    all_roots_arr = np.concatenate(all_roots)
    all_songs_arr = np.array(all_songs)

    # Apply windowed context.  NOTE: within-song window is already implicit in
    # all_feats_raw because we collected beats sequentially per song.  We need
    # to reconstruct the within-song structure for the context window.
    # Build (song_id → row indices) map, apply windowed_features per song, then cat.
    # This preserves the "zero-pad at song edges" boundary from the experiment.
    song_ids_unique = list(dict.fromkeys(all_songs_arr))  # preserve order
    windowed_parts = []
    roots_parts = []
    group_parts = []
    for sid in song_ids_unique:
        mask = all_songs_arr == sid
        F_song = all_feats_raw[mask]
        r_song = all_roots_arr[mask]
        W_song = windowed_features(F_song, args.window)
        windowed_parts.append(W_song)
        roots_parts.append(r_song)
        group_parts.append(np.full(len(r_song), sid))

    X = np.vstack(windowed_parts)
    y = np.concatenate(roots_parts)
    groups = np.concatenate(group_parts)
    print(f"Feature shape: {X.shape}  (48 × {2*args.window+1} = {X.shape[1]})")

    if args.eval:
        pred = np.zeros(len(y), int)
        for tr, te in GroupKFold(5).split(X, y, groups):
            sc = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
            pred[te] = clf.predict(sc.transform(X[te]))
        acc = (pred == y).mean()
        print(f"5-fold CV per-beat root accuracy: {acc:.1%}")
        print("(run without --eval to fit on all data and save)")
        return

    # Fit on all data and save
    sc = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X), y)
    train_acc = (clf.predict(sc.transform(X)) == y).mean()
    print(f"Train accuracy: {train_acc:.1%}  (n={len(y)})")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez(OUT,
             mean=sc.mean_.astype("float32"),
             scale=sc.scale_.astype("float32"),
             coef=clf.coef_.astype("float32"),
             intercept=clf.intercept_.astype("float32"),
             classes=clf.classes_.astype(int),
             window=np.array([args.window], dtype=int))
    print(f"Saved → {OUT}")


if __name__ == "__main__":
    main()
