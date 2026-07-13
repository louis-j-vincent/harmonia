"""Train beat_seq_model_v2: key-agnostic root classifier via chroma-rotation augmentation.

Key improvements over v1:
  1. Uniform key augmentation: every song's per-beat chroma features are rotated
     through all 12 transpositions at training time (no re-render needed — L2-normed
     chroma rolls are exact). This eliminates the iReal corpus bias toward C/F/Bb/Eb.
  2. POP909 piano data: adds piano-voiced chords (5th-in-bass common) alongside
     iReal jazz (walking-bass dominant). Songs 001-005 are held out for eval.

Architecture unchanged: windowed LR on 48*(2*window+1)-dim features.

Usage:
    .venv/bin/python scripts/train_beat_seq_model_v2.py [--n-jazz 50] [--n-pop 50] [--window 2] [--eval]

Saves: harmonia/models/beat_seq_model_v2.npz
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

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler

from analyze_accomp_emission import song_chord_spans
from build_audio_chord_features import BUCKET_FAMILY
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig
from harmonia.models.stage1_pitch import PitchExtractor
from harmonic_rhythm_probe import pool_beats
from root_model_experiment import chroma88

DB    = REPO / "data" / "accomp_db" / "db.jsonl"
POP   = REPO / "data" / "pop909" / "POP909"
OUT   = REPO / "harmonia" / "models" / "beat_seq_model_v2.npz"
NOTE  = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]

HARTE_TO_PC = {n: i for i, n in enumerate(NOTE)}
HARTE_TO_PC.update({"Db":1,"Eb":3,"Gb":6,"Ab":8,"Bb":10})


# ── feature helpers ───────────────────────────────────────────────────────────

def beat_features(onset_b: np.ndarray, note_b: np.ndarray) -> np.ndarray:
    """48d per-beat feature (same definition as v1)."""
    n = len(onset_b)
    F = np.zeros((n, 48), np.float32)
    for b in range(n):
        F[b] = np.concatenate([
            chroma88(onset_b[b]),
            chroma88(note_b[b]),
            chroma88(onset_b[b], 0, 52),
            chroma88(onset_b[b], 60, 200),
        ])
    return F


def rotate_features(F: np.ndarray, shift: int) -> np.ndarray:
    """Rotate all 12d chroma blocks in F by `shift` semitones.

    F has shape (n_beats, 48) with four consecutive 12d L2-normed chroma blocks.
    Rolling each block by `shift` is exactly equivalent to pitch-transposing the song.
    Cost: zero extra renders.
    """
    out = F.copy()
    for start in range(0, 48, 12):
        out[:, start:start+12] = np.roll(F[:, start:start+12], shift, axis=1)
    return out


def windowed_features(F: np.ndarray, window: int) -> np.ndarray:
    """Concatenate ±window neighbours → (n_beats, 48*(2w+1))."""
    n, d = F.shape
    W = 2 * window + 1
    out = np.zeros((n, d * W), np.float32)
    for b in range(n):
        row = []
        for delta in range(-window, window + 1):
            nb = b + delta
            row.append(F[nb] if 0 <= nb < n else np.zeros(d, np.float32))
        out[b] = np.concatenate(row)
    return out


# ── data collection ───────────────────────────────────────────────────────────

def collect_ireal_song(rec, renderer, sf2, ex):
    """Render one iReal song, return (F_raw, roots, valid) before augmentation."""
    spb  = 60.0 / rec["tempo"]
    bpb  = rec["beats_per_bar"]
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
        acts = ex.extract(tmp, use_cache=False)
    finally:
        tmp.unlink(missing_ok=True)

    bt     = np.arange(n_beats + 1) * spb
    onset_b = pool_beats(acts.frame_times, acts.onset_probs, bt)
    note_b  = pool_beats(acts.frame_times, acts.note_probs,  bt)
    F       = beat_features(onset_b, note_b)

    gt    = np.array([gtroot((b + 0.5) * spb) for b in range(n_beats)], dtype=object)
    valid = np.array([g is not None for g in gt])
    roots = np.zeros(n_beats, int)
    for b in range(n_beats):
        if valid[b]:
            roots[b] = int(gt[b])
    return F, roots, valid


def collect_pop909_song(sid: str, renderer, sf2, ex):
    """Render one POP909 song from MIDI + MuseScore, return (F_raw, roots, valid).

    GT roots from chord_midi.txt (Harte-style labels, seconds-aligned).
    Beat times from beat_midi.txt.
    """
    song_dir  = POP / sid
    midi_path = song_dir / f"{sid}.mid"
    chord_txt = song_dir / "chord_midi.txt"
    beat_txt  = song_dir / "beat_midi.txt"

    if not midi_path.exists() or not chord_txt.exists():
        return None

    # parse chord spans (seconds → root PC)
    spans = []
    for line in chord_txt.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 3:
            continue
        t0, t1, label = float(parts[0]), float(parts[1]), parts[2]
        if label == "N":
            continue
        root_str = label.split(":")[0]
        if root_str in HARTE_TO_PC:
            spans.append((t0, t1, HARTE_TO_PC[root_str]))

    if not spans:
        return None

    # parse beat times (column 1 = beat time in seconds)
    beat_times = []
    for line in beat_txt.read_text().splitlines():
        parts = line.strip().split()
        if parts:
            try:
                beat_times.append(float(parts[0]))
            except ValueError:
                pass
    if len(beat_times) < 2:
        return None
    bt = np.array(beat_times + [beat_times[-1] + (beat_times[-1] - beat_times[-2])])

    def gtroot(t):
        for t0, t1, root in spans:
            if t0 <= t < t1:
                return root
        return None

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
        tmp = Path(wf.name)
    try:
        renderer.render(midi_path, tmp, RenderConfig(soundfont_path=sf2))
        y, sr = sf.read(tmp)
        y = (y.mean(1) if y.ndim > 1 else y).astype("float32")
        acts = ex.extract(tmp, use_cache=False)
    finally:
        tmp.unlink(missing_ok=True)

    n_beats = len(bt) - 1
    onset_b = pool_beats(acts.frame_times, acts.onset_probs, bt)
    note_b  = pool_beats(acts.frame_times, acts.note_probs,  bt)
    F       = beat_features(onset_b, note_b)

    gt    = np.array([gtroot(0.5 * (bt[b] + bt[b+1])) for b in range(n_beats)], dtype=object)
    valid = np.array([g is not None for g in gt])
    roots = np.zeros(n_beats, int)
    for b in range(n_beats):
        if valid[b]:
            roots[b] = int(gt[b])
    return F, roots, valid


def augment_song(F: np.ndarray, roots: np.ndarray, valid: np.ndarray,
                 window: int, sid: str, n_keys: int = 12):
    """Apply chroma rotation for n_keys transpositions.

    Returns list of (windowed_F, roots_shifted, group_labels).
    """
    parts = []
    F_v = F[valid]; r_v = roots[valid]
    # rebuild windowed features within-song (preserving neighbour context)
    W_full = windowed_features(F, window)
    W_v    = W_full[valid]

    for shift in range(n_keys):
        W_rot = np.zeros_like(W_v)
        for start in range(0, W_v.shape[1], 12):
            W_rot[:, start:start+12] = np.roll(W_v[:, start:start+12], shift, axis=1)
        r_rot = (r_v + shift) % 12
        grp   = np.full(len(r_rot), f"{sid}_k{shift}")
        parts.append((W_rot, r_rot, grp))
    return parts


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-jazz",   type=int, default=50, help="iReal songs (default 50)")
    ap.add_argument("--n-pop",    type=int, default=60, help="POP909 songs to add (default 60, hold out 001-005)")
    ap.add_argument("--window",   type=int, default=2,  help="beat context half-window")
    ap.add_argument("--n-keys",   type=int, default=12, help="transpositions per song (default 12 = all keys)")
    ap.add_argument("--eval",     action="store_true",  help="5-fold CV on combined data instead of saving")
    ap.add_argument("--out",      default=str(OUT))
    args = ap.parse_args()

    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2      = renderer._find_soundfont("MuseScore_General.sf2")
    ex       = PitchExtractor(cache_dir=None)

    all_W, all_roots, all_groups = [], [], []

    # ── iReal jazz ────────────────────────────────────────────────────────────
    recs  = [json.loads(l) for l in open(DB)]
    songs = [r for r in recs if r.get("corpus") == "jazz1460"
             and r["beats_per_bar"] == 4
             and (REPO / r["midi_path"]).exists()]
    songs = songs[:: max(len(songs) // args.n_jazz, 1)][: args.n_jazz]
    print(f"iReal: {len(songs)} songs  ×{args.n_keys} keys = {len(songs)*args.n_keys} virtual songs")

    for i, rec in enumerate(songs):
        sid = rec["song_id"]
        print(f"  [{i+1}/{len(songs)}] {sid}", end="\r", flush=True)
        try:
            F, roots, valid = collect_ireal_song(rec, renderer, sf2, ex)
        except Exception as e:
            print(f"\n  SKIP {sid}: {e}")
            continue
        if valid.sum() == 0:
            continue
        for W_rot, r_rot, grp in augment_song(F, roots, valid, args.window, sid, args.n_keys):
            all_W.append(W_rot); all_roots.append(r_rot); all_groups.append(grp)

    print(f"\niReal collected: {sum(len(r) for r in all_roots)} beat-samples")

    # ── POP909 piano ──────────────────────────────────────────────────────────
    HOLD_OUT = {"001", "002", "003", "004", "005"}
    pop_sids = sorted(d.name for d in POP.iterdir()
                      if d.is_dir() and d.name not in HOLD_OUT and (d / f"{d.name}.mid").exists())
    pop_sids = pop_sids[: args.n_pop]
    print(f"POP909: {len(pop_sids)} songs  ×{args.n_keys} keys")

    for i, sid in enumerate(pop_sids):
        print(f"  [{i+1}/{len(pop_sids)}] {sid}", end="\r", flush=True)
        try:
            result = collect_pop909_song(sid, renderer, sf2, ex)
        except Exception as e:
            print(f"\n  SKIP {sid}: {e}")
            continue
        if result is None:
            continue
        F, roots, valid = result
        if valid.sum() == 0:
            continue
        for W_rot, r_rot, grp in augment_song(F, roots, valid, args.window,
                                               f"pop_{sid}", args.n_keys):
            all_W.append(W_rot); all_roots.append(r_rot); all_groups.append(grp)

    print(f"\nPOP909 + iReal total: {sum(len(r) for r in all_roots)} beat-samples")

    X      = np.vstack(all_W)
    y      = np.concatenate(all_roots)
    groups = np.concatenate(all_groups)
    print(f"Feature shape: {X.shape}")

    # key distribution after augmentation (should be flat)
    pc_counts = np.bincount(y, minlength=12)
    print("Key distribution after augmentation:")
    for i in range(12):
        print(f"  {NOTE[i]:>3}: {pc_counts[i]:6d} ({pc_counts[i]/len(y):.1%})")

    if args.eval:
        pred = np.zeros(len(y), int)
        for tr, te in GroupKFold(5).split(X, y, groups):
            sc  = StandardScaler().fit(X[tr])
            clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
            pred[te] = clf.predict(sc.transform(X[te]))
        acc = (pred == y).mean()
        print(f"5-fold CV per-beat root accuracy: {acc:.1%}")
        return

    sc  = StandardScaler().fit(X)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X), y)
    train_acc = (clf.predict(sc.transform(X)) == y).mean()
    print(f"Train accuracy: {train_acc:.1%}  (n={len(y)})")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out,
             mean      = sc.mean_.astype("float32"),
             scale     = sc.scale_.astype("float32"),
             coef      = clf.coef_.astype("float32"),
             intercept = clf.intercept_.astype("float32"),
             classes   = clf.classes_.astype(int),
             window    = np.array([args.window], int))
    print(f"Saved → {out}")


if __name__ == "__main__":
    main()
