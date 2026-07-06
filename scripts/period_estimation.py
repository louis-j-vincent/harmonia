"""Chord-change engine, step 1 (the user's refinement): estimate the harmonic-rhythm
PERIOD before merging — per song and per SECTION — and check whether it holds
throughout or shifts. Then we merge at the estimated period, not a fixed 2.

Cue (unsupervised): at the true period g, changes land ON the g-grid, so beats
INSIDE a block (off-grid) stay harmonically still → low beat-to-beat novelty. So
the right g is the COARSEST grid whose best-phase off-grid novelty stays quiet.

We have GT here, so we validate the estimator: GT period of a section = coarsest
g in {4,2,1} on which ≥80% of that section's real chord changes fall on one phase.

This first pass DUMPS the raw off/on-grid novelties per section so the decision
rule is chosen from data, not guessed, then scores a first rule vs GT.

Usage: .venv/bin/python scripts/period_estimation.py --n-songs 15 [--degrade]
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

from build_accomp_audio_hard import time_varying_degrade  # noqa: E402
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig  # noqa: E402
from harmonia.models.stage1_pitch import PitchExtractor  # noqa: E402
from harmonic_rhythm_probe import beat_feats, gt_chord_per_beat, pool_beats  # noqa: E402

DB = REPO / "data" / "accomp_db" / "db.jsonl"
CANDS = (1, 2, 4)


def gt_period(change_beats, s, e, min_frac=0.8):
    """Coarsest g in {4,2,1} on which >=min_frac of the section's changes share a phase."""
    ch = [b - s for b in change_beats if s < b < e]      # section-relative
    if len(ch) < 2:
        return None
    for g in (4, 2):
        best = max(sum((c % g) == phi for c in ch) for phi in range(g))
        if best / len(ch) >= min_frac:
            return g
    return 1


def grid_novelty(nov, s, e, g):
    """Best-phase (min off-grid) → (on_mean, off_mean) novelty for a g-grid in [s,e)."""
    idx = np.arange(s + 1, e)                              # skip section's first beat
    if len(idx) < g + 1:
        return None
    best = None
    for phi in range(g):
        on = idx[(idx - s) % g == phi]
        off = idx[(idx - s) % g != phi]
        if len(off) == 0:
            on_m, off_m = nov[on].mean(), 0.0
        else:
            on_m, off_m = nov[on].mean(), nov[off].mean()
        if best is None or (on_m - off_m) > best[0]:
            best = (on_m - off_m, on_m, off_m)
    return best[1], best[2]


def estimate_period(nov, s, e, tau_ratio=0.6):
    """Coarsest g whose off-grid novelty < tau_ratio * its on-grid novelty; else finer."""
    for g in (4, 2):
        gn = grid_novelty(nov, s, e, g)
        if gn and gn[1] <= tau_ratio * gn[0] + 1e-9:
            return g
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-songs", type=int, default=15)
    ap.add_argument("--degrade", action="store_true")
    ap.add_argument("--dump", type=int, default=2, help="print per-section table for first N songs")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(DB)]
    songs = [r for r in recs if r["corpus"] == "jazz1460" and r["beats_per_bar"] == 4
             and (REPO / r["midi_path"]).exists() and len(set(r["section_per_bar"])) > 1]
    songs = songs[:: max(len(songs) // args.n_songs, 1)][: args.n_songs]

    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2 = renderer._find_soundfont("MuseScore_General.sf2")
    ex = PitchExtractor(cache_dir=None)
    rng = np.random.default_rng(3)

    gt_dist = Counter(); est_dist = Counter()
    confusion = Counter()                                  # (gt, est) -> n
    const_song = []                                        # is GT period constant across sections?
    rows = []                                              # (off2/on2, off4/on4, gt) for rule design

    for si, rec in enumerate(songs):
        spb = 60.0 / rec["tempo"]; bpb = rec["beats_per_bar"]; nb = rec["n_bars"]
        n_beats = nb * bpb
        sec = [rec["section_per_bar"][b // bpb] for b in range(n_beats)]
        gtc = gt_chord_per_beat(rec, n_beats, spb)
        changes = [b for b in range(1, n_beats) if gtc[b] is not None and gtc[b] != gtc[b - 1]]

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
        bt = np.arange(n_beats + 1) * spb
        feats = beat_feats(pool_beats(acts.frame_times, acts.onset_probs, bt))
        nov = np.zeros(n_beats)
        for b in range(1, n_beats):
            fa, fb = feats[b - 1], feats[b]
            nov[b] = 1 - float(fa @ fb / (np.linalg.norm(fa) * np.linalg.norm(fb) + 1e-9))

        # section spans
        spans = []
        b0 = 0
        for b in range(1, n_beats + 1):
            if b == n_beats or sec[b] != sec[b0]:
                spans.append((b0, b)); b0 = b
        song_gt = []
        if si < args.dump:
            print(f"\n  {rec['song_id']} form={rec['form']}")
        for (s, e) in spans:
            gp = gt_period(changes, s, e)
            ep = estimate_period(nov, s, e)
            if gp is None:
                continue
            gt_dist[gp] += 1; est_dist[ep] += 1; confusion[(gp, ep)] += 1
            song_gt.append(gp)
            g2 = grid_novelty(nov, s, e, 2); g4 = grid_novelty(nov, s, e, 4)
            r2 = g2[1] / (g2[0] + 1e-9) if g2 else float("nan")
            r4 = g4[1] / (g4[0] + 1e-9) if g4 else float("nan")
            rows.append((r2, r4, gp))
            if si < args.dump:
                print(f"    sec {sec[s]} beats[{s:3d}:{e:3d}]  GT_period={gp}  est={ep}  "
                      f"off/on(g2)={r2:.2f} off/on(g4)={r4:.2f}")
        if song_gt:
            const_song.append(len(set(song_gt)) == 1)

    cond = "DEGRADED" if args.degrade else "clean"
    n = sum(confusion.values())
    acc = sum(v for (g, e), v in confusion.items() if g == e) / (n + 1e-9)
    print(f"\n=== period estimation, {len(songs)} {cond} songs, {n} sections with a defined period ===")
    print("GT per-section period dist :", {g: gt_dist[g] for g in CANDS})
    print("Est per-section period dist:", {g: est_dist[g] for g in CANDS})
    print(f"Estimator exact accuracy   : {acc:.1%}")
    print(f"Period constant across sections within a song: "
          f"{np.mean(const_song):.0%} of songs")
    print("\nConfusion (GT→est):")
    for g in CANDS:
        print(f"    GT={g}: " + "  ".join(f"est{e}={confusion[(g,e)]}" for e in CANDS))
    # rule-design view: mean off/on ratio by GT period
    print("\noff/on novelty ratio by GT period (rule-design view):")
    for g in CANDS:
        rr = [(r2, r4) for r2, r4, gp in rows if gp == g]
        if rr:
            a = np.nanmean([x[0] for x in rr]); b = np.nanmean([x[1] for x in rr])
            print(f"    GT={g} (n={len(rr)}): mean off/on(g2)={a:.2f}  off/on(g4)={b:.2f}")


if __name__ == "__main__":
    main()
