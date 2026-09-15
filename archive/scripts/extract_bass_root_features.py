#!/usr/bin/env python3
"""Extract 24-dim bothchroma (bass+treble) per-chord features from McGill Billboard.

Unlike the existing billboard_training_corpus_full.npz (which collapses bothchroma
to a single 12-dim vector), this preserves BOTH 12-dim halves of Chordino's
bothchroma output so a bass/root detector can exploit the bass register.

Per chord segment (from full.lab), we average all bothchroma frames whose time
falls inside [start, end). We also record prev/next functional root within the
song for context features.

Output: data/cache/bass_root_features.npz
"""
import sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from harmonia.data.billboard_translator import parse_billboard_chord

MCGILL = Path.home() / "mir_datasets" / "billboard" / "McGill-Billboard"
OUT = REPO / "data" / "cache" / "bass_root_features.npz"


def load_chroma(song_dir: Path):
    """Return (times[T], chroma[T,24]) from bothchroma.csv."""
    times, rows = [], []
    with open(song_dir / "bothchroma.csv") as f:
        for line in f:
            parts = line.strip().split(",")
            if len(parts) < 26:
                continue
            times.append(float(parts[1]))
            rows.append([float(x) for x in parts[2:26]])
    if not rows:
        return None, None
    return np.asarray(times), np.asarray(rows, dtype=np.float32)


def load_segments(song_dir: Path):
    """Return list of (start, end, label) from full.lab."""
    segs = []
    with open(song_dir / "full.lab") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                parts = line.strip().split()
            if len(parts) < 3:
                continue
            segs.append((float(parts[0]), float(parts[1]), parts[2]))
    return segs


def main():
    song_dirs = sorted(d for d in MCGILL.iterdir()
                       if d.is_dir() and d.name.isdigit())
    feats, roots, quals, song_ids = [], [], [], []
    prev_roots, next_roots, durations = [], [], []

    QUAL = {"maj": 0, "min": 1, "dom": 2, "hdim": 3, "dim": 4}

    for sd in song_dirs:
        if not (sd / "bothchroma.csv").exists() or not (sd / "full.lab").exists():
            continue
        times, chroma = load_chroma(sd)
        if times is None:
            continue
        segs = load_segments(sd)

        # First pass: parse all segments to sequence of (root, qual, feat, dur)
        parsed = []
        for (s, e, lab) in segs:
            root, q = parse_billboard_chord(lab)
            if root is None or q is None or q not in QUAL:
                parsed.append(None)  # placeholder to keep N-context boundaries clean
                continue
            mask = (times >= s) & (times < e)
            if mask.sum() == 0:
                parsed.append(None)
                continue
            fv = chroma[mask].mean(axis=0)
            parsed.append((root, QUAL[q], fv, e - s))

        # Build context using only valid neighbors (skip N/None)
        valid_idx = [i for i, p in enumerate(parsed) if p is not None]
        for pos, i in enumerate(valid_idx):
            root, qi, fv, dur = parsed[i]
            pr = parsed[valid_idx[pos - 1]][0] if pos > 0 else -1
            nr = parsed[valid_idx[pos + 1]][0] if pos < len(valid_idx) - 1 else -1
            feats.append(fv)
            roots.append(root)
            quals.append(qi)
            song_ids.append(sd.name)
            prev_roots.append(pr)
            next_roots.append(nr)
            durations.append(dur)

    feats = np.asarray(feats, dtype=np.float32)
    np.savez_compressed(
        OUT,
        feats=feats,
        roots=np.asarray(roots, dtype=np.int64),
        quals=np.asarray(quals, dtype=np.int64),
        song_id=np.asarray(song_ids),
        prev_root=np.asarray(prev_roots, dtype=np.int64),
        next_root=np.asarray(next_roots, dtype=np.int64),
        duration=np.asarray(durations, dtype=np.float32),
        qualities=np.asarray(["maj", "min", "dom", "hdim", "dim"], dtype=object),
    )
    print(f"Saved {feats.shape[0]} chords, feat dim {feats.shape[1]}, "
          f"{len(set(song_ids))} songs -> {OUT}")


if __name__ == "__main__":
    main()
