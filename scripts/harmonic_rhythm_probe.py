"""Foundation check for the coarse-to-fine chord-change method: when we MERGE beats
into g-beat blocks, is a true chord change separable from a held chord using
chroma+bass? Everything downstream (per-section harmonic rhythm, same-or-different
fill, zoom) rests on this. Validate against GT, before building the machinery.

Scaffold = GT section_per_bar + exact MMA beat grid (structure detection is a
separable, still-open problem). Two premise checks:

  A. Is the harmonic rhythm even regular? Histogram GT chord-change spacings (beats),
     overall and per section — is it really 1/2/4?
  B. Block-diff separability: pool chroma+bass into g-beat blocks (within a section),
     measure ROC-AUC of adjacent-block cosine distance for "true change" vs "hold",
     for g in {1,2,4}. High AUC (esp. vs g=1) => merging gives a reliable
     same-or-different signal; ~0.5 => the method has no foundation.

Usage: .venv/bin/python scripts/harmonic_rhythm_probe.py --n-songs 12 [--degrade]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from analyze_accomp_emission import parse_chord  # noqa: E402
from build_accomp_audio_hard import time_varying_degrade  # noqa: E402
from build_audio_chord_features import BUCKET_FAMILY  # noqa: E402
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig  # noqa: E402
from harmonia.models.stage1_pitch import PitchExtractor  # noqa: E402

DB = REPO / "data" / "accomp_db" / "db.jsonl"
NOTE_TO_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


def gt_chord_per_beat(rec, n_beats, spb):
    """(root, family) active at each beat, from the chart."""
    tl = sorted(rec["chord_timeline"], key=lambda e: e["time"])
    out = []
    for b in range(n_beats):
        t = b * spb
        cur = None
        for ev in tl:
            if ev["time"] <= t + 1e-6:
                s = ev["ireal"] or ev["mma"]
                if not s or s[0] not in NOTE_TO_PC:
                    continue
                pc = NOTE_TO_PC[s[0]]
                if len(s) > 1 and s[1] == "#": pc = (pc + 1) % 12
                elif len(s) > 1 and s[1] == "b": pc = (pc - 1) % 12
                p = parse_chord(s)
                fam = BUCKET_FAMILY.get(p[1]) if (p and p[1] in BUCKET_FAMILY) else "major"
                cur = (pc, fam)
            else:
                break
        out.append(cur)
    return out


def pool_beats(frame_times, probs, beat_times):
    out = np.zeros((len(beat_times) - 1, probs.shape[1]), dtype=np.float32)
    for b in range(len(beat_times) - 1):
        m = (frame_times >= beat_times[b]) & (frame_times < beat_times[b + 1])
        if m.any():
            out[b] = probs[m].mean(0)
    return out


def beat_feats(onset_b):
    """Per-beat [chroma(12) ⊕ bass(12)], each L2-normalized."""
    feats = []
    for b in range(len(onset_b)):
        ch = np.zeros(12); ba = np.zeros(12)
        for k in range(88):
            m = 21 + k
            ch[m % 12] += onset_b[b, k]
            if m < 52:
                ba[m % 12] += onset_b[b, k]
        ch /= (np.linalg.norm(ch) + 1e-9); ba /= (np.linalg.norm(ba) + 1e-9)
        feats.append(np.concatenate([ch, ba]))
    return np.array(feats)


def auc(dist, label):
    """ROC-AUC via rank statistic; label 1 = true change."""
    dist = np.asarray(dist); label = np.asarray(label)
    pos, neg = dist[label == 1], dist[label == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty(len(order)); ranks[order] = np.arange(1, len(order) + 1)
    r_pos = ranks[:len(pos)].sum()
    return (r_pos - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-songs", type=int, default=12)
    ap.add_argument("--degrade", action="store_true")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(DB)]
    songs = [r for r in recs if r["corpus"] == "jazz1460" and r["beats_per_bar"] == 4
             and (REPO / r["midi_path"]).exists() and len(set(r["section_per_bar"])) > 1]
    songs = songs[:: max(len(songs) // args.n_songs, 1)][: args.n_songs]

    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2 = renderer._find_soundfont("MuseScore_General.sf2")
    ex = PitchExtractor(cache_dir=None)
    rng = np.random.default_rng(3)

    spacing = Counter()                         # A: GT change spacings (beats)
    dists = {1: ([], []), 2: ([], []), 4: ([], [])}   # B: g -> (distance, label)

    for rec in songs:
        spb = 60.0 / rec["tempo"]; bpb = rec["beats_per_bar"]; nb = rec["n_bars"]
        n_beats = nb * bpb
        sec = [rec["section_per_bar"][b // bpb] for b in range(n_beats)]
        gtc = gt_chord_per_beat(rec, n_beats, spb)

        # A. GT change spacings (consecutive changes, within a section)
        last = None
        for b in range(n_beats):
            if gtc[b] is not None and gtc[b] != gtc[b - 1] if b else False:
                if last is not None and sec[b] == sec[last]:
                    spacing[b - last] += 1
                last = b

        # render + BP
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
            tmp = Path(wf.name)
        try:
            renderer.render(REPO / rec["midi_path"], tmp, RenderConfig(soundfont_path=sf2))
            y, sr = sf.read(tmp); y = (y.mean(1) if y.ndim > 1 else y).astype("float32")
            if args.degrade:
                y = time_varying_degrade(y, sr, rng); sf.write(tmp, y, sr)
            acts = ex.extract(tmp, use_cache=False)
        finally:
            tmp.unlink(missing_ok=True)
        beat_times = np.arange(n_beats + 1) * spb
        feats = beat_feats(pool_beats(acts.frame_times, acts.onset_probs, beat_times))

        # B. block-diff separability for each candidate grid g
        for g in (1, 2, 4):
            for start in range(0, n_beats - 2 * g + 1, g):
                a0, a1, b1 = start, start + g, start + 2 * g
                if sec[a0] != sec[b1 - 1]:            # don't cross a section boundary
                    continue
                fa, fb = feats[a0:a1].mean(0), feats[a1:b1].mean(0)
                d = 1 - float(fa @ fb / (np.linalg.norm(fa) * np.linalg.norm(fb) + 1e-9))
                ca, cb = gtc[a1 - 1], gtc[b1 - 1]
                if ca is None or cb is None:
                    continue
                dists[g][0].append(d); dists[g][1].append(int(ca != cb))

    cond = "DEGRADED" if args.degrade else "clean"
    print(f"\n=== harmonic-rhythm foundation check, {len(songs)} {cond} songs ===\n")
    tot = sum(spacing.values())
    print("A. GT chord-change spacing within a section (beats → share):")
    for k in sorted(spacing):
        print(f"     {k:>2} beat(s): {spacing[k] / tot:5.1%}  ({spacing[k]})")
    print("\nB. Block-diff separability of change-vs-hold (chroma+bass, ROC-AUC; 0.5=useless):")
    for g in (1, 2, 4):
        d, l = dists[g]
        base = np.mean(l) if l else float("nan")
        print(f"     g={g} beat block: AUC={auc(d, l):.3f}   (n={len(l)}, {base:.0%} are true changes)")
    print("\nHigh AUC — esp. g=2/4 above g=1 — means merging gives a reliable "
          "same-or-different\nsignal, and the coarse-to-fine method has a foundation.")


if __name__ == "__main__":
    main()
