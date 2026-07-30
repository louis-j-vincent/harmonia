#!/usr/bin/env python3
"""Does 4 x 2 s of a chord count like 1 x 8 s? (Louis, 2026-07-30)

Duration turned out to be the best per-chord confidence signal (mean per-song
AUC 0.768; see docs/known_issues.md). That leaves the mechanism open, and the
two possibilities predict opposite things:

  (a) EVIDENCE — a longer span is more frames, so a better estimate. Then four
      separate 2 s observations should be worth about as much as one 8 s
      observation (minus whatever the extra boundaries cost).
  (b) STRUCTURE — long spans are a different kind of musical object. A chord
      that sits for 8 s is a stable harmonic zone; 2 s chords are passing,
      transitional, or decoder flicker. Then repetition should NOT substitute
      for contiguous length.

Measured three ways, on the same 603 spans / 7 verified Brick-0 songs:
  1. AUC of contiguous duration vs total duration of that chord across the song
     vs number of occurrences.
  2. Accuracy in a 2-D table: contiguous duration x repeat count. This is the
     decisive one — it holds one axis fixed and asks whether the other still
     moves the number.
  3. Whether POOLING music-x-lab's posterior across every occurrence (the same
     averaging idea as the posterior fold, applied to scoring) discriminates
     better than the per-span posterior.

"The same chord" = same (root pitch class, parent family) as DISPLAYED, which is
what a reader means by it.

Run: .venv/bin/python scratchpad/conf_duration_vs_repeats.py
"""
from __future__ import annotations

import json
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harmonia.eval.accuracy_score import (SHIPPED_CONFIG, _decode_to_wav,  # noqa: E402
                                          chord_family, chord_from_label,
                                          load_frozen_gt)
from harmonia.models.chord_pipeline_v1 import infer_chords_v1  # noqa: E402
from harmonia.models.musx_redecode import FRAME_DT, frame_posteriors  # noqa: E402

CACHE = REPO / "data" / "cache"
GOLDEN = REPO / "golden" / "brick0"
OUT = Path(__file__).with_suffix(".json")

TRIM = 0.15
MIN_FRAMES = 4
_FAMILY_TO_TYPES = {"maj": (0,), "dom": (0,), "min": (1,), "sus": (2, 3),
                    "dim": (4,), "hdim": (4,), "aug": (5,)}


def gather(gt_path: Path):
    gt = load_frozen_gt(gt_path)
    audio = gt.resolved_audio_path
    with tempfile.TemporaryDirectory() as tmp:
        wav = _decode_to_wav(audio, Path(tmp))
        chart = infer_chords_v1(wav, cache_dir=CACHE, **SHIPPED_CONFIG)
    triad = frame_posteriors(audio)[0]
    rows = []
    for c in chart.chords:
        t0, t1 = float(c["start_s"]), float(c["end_s"])
        pred = chord_from_label(t0, t1, str(c["label"]))
        if t1 <= t0 or pred.is_nc or pred.root_pc is None:
            continue
        best, best_ov = None, 0.0
        for g in gt.gt_chords:
            ov = min(t1, g.t1) - max(t0, g.t0)
            if ov > best_ov:
                best, best_ov = g, ov
        if best is None or best_ov <= 0 or best.is_nc:
            continue
        fam = chord_family(pred.quality)
        a = int(round((t0 + TRIM * (t1 - t0)) / FRAME_DT))
        b = int(round((t1 - TRIM * (t1 - t0)) / FRAME_DT))
        a, b = max(0, a), min(len(triad), b)
        if b - a < MIN_FRAMES:
            continue
        P = triad[a:b]
        n_chord = P[:, 1:].reshape(len(P), 6, 12)
        types = _FAMILY_TO_TYPES.get(fam, (0,))
        rows.append({
            "song": gt.song_id,
            "key": (int(pred.root_pc), fam),
            "dur": t1 - t0,
            "p_chord": float(n_chord[:, types, int(pred.root_pc)].sum(1).mean()),
            "n_frames": int(b - a),
            "ok": bool(pred.root_pc == best.root_pc
                       and fam == chord_family(best.quality)),
        })

    # per-song aggregation over every occurrence of the same displayed chord
    tot = defaultdict(float)
    cnt = defaultdict(int)
    pool_num, pool_den = defaultdict(float), defaultdict(float)
    for r in rows:
        tot[r["key"]] += r["dur"]
        cnt[r["key"]] += 1
        pool_num[r["key"]] += r["p_chord"] * r["n_frames"]
        pool_den[r["key"]] += r["n_frames"]
    for r in rows:
        k = r["key"]
        r["tot_dur"] = tot[k]
        r["n_occ"] = cnt[k]
        r["p_chord_pooled"] = pool_num[k] / max(pool_den[k], 1e-9)
    return rows


def main():
    from sklearn.metrics import roc_auc_score

    rows = []
    for p in sorted(GOLDEN.glob("*.gt.json")):
        gt = load_frozen_gt(p)
        if not gt.verified or not gt.resolved_audio_path.exists():
            continue
        r = gather(p)
        if r:
            rows += r
            print(f"  {gt.song_id:24s} {len(r):4d} spans, "
                  f"{len({x['key'] for x in r}):2d} distinct chords")

    Y = np.array([r["ok"] for r in rows], float)
    W = np.array([r["dur"] for r in rows])
    G = np.array([r["song"] for r in rows])
    songs = list(dict.fromkeys(G.tolist()))

    def per_song_auc(x):
        v = []
        for s in songs:
            m = G == s
            v.append(roc_auc_score(Y[m], x[m], sample_weight=W[m])
                     if len(set(Y[m])) > 1 else np.nan)
        return np.array(v)

    print(f"\n{len(rows)} spans, {len(songs)} songs, "
          f"acc={np.average(Y, weights=W):.3f}\n")
    print(f"{'score':34s} {'mean per-song AUC':>18s} {'worst':>7s}")
    for name, key in [("contiguous duration of THIS span", "dur"),
                      ("total duration of that chord in song", "tot_dur"),
                      ("number of occurrences", "n_occ"),
                      ("musx posterior, this span only", "p_chord"),
                      ("musx posterior, POOLED over repeats", "p_chord_pooled")]:
        x = np.array([r[key] for r in rows], float)
        v = per_song_auc(x)
        print(f"{name:34s} {np.nanmean(v):18.3f} {np.nanmin(v):7.2f}")

    # ── the decisive table: hold one axis fixed, move the other ─────────────
    dur = np.array([r["dur"] for r in rows])
    occ = np.array([r["n_occ"] for r in rows])
    dbins = [(0, 1.5, "< 1.5 s"), (1.5, 3.0, "1.5 - 3 s"), (3.0, 1e9, "> 3 s")]
    obins = [(1, 2, "1"), (2, 4, "2-3"), (4, 8, "4-7"), (8, 10**9, "8+")]
    print("\naccuracy by CONTIGUOUS duration (rows) x TIMES THAT CHORD REPEATS (cols)")
    print(f"{'':12s}" + "".join(f"{o[2]:>12s}" for o in obins) + f"{'all':>12s}")
    for lo, hi, dn in dbins:
        line = f"{dn:12s}"
        for olo, ohi, _ in obins:
            m = (dur >= lo) & (dur < hi) & (occ >= olo) & (occ < ohi)
            line += (f"{np.average(Y[m], weights=W[m]):8.2f}({m.sum():3d})"
                     if m.sum() >= 8 else f"{'-':>12s}")
        m = (dur >= lo) & (dur < hi)
        line += f"{np.average(Y[m], weights=W[m]):8.2f}({m.sum():3d})"
        print(line)
    line = f"{'all':12s}"
    for olo, ohi, _ in obins:
        m = (occ >= olo) & (occ < ohi)
        line += (f"{np.average(Y[m], weights=W[m]):8.2f}({m.sum():3d})"
                 if m.sum() >= 8 else f"{'-':>12s}")
    print(line)

    # ── the direct answer to the question as asked ──────────────────────────
    print("\nDIRECT COMPARISON — same TOTAL evidence, delivered differently:")
    for tlo, thi, tn in [(6, 12, "6-12 s total"), (12, 25, "12-25 s total")]:
        m_all = (np.array([r["tot_dur"] for r in rows]) >= tlo) & \
                (np.array([r["tot_dur"] for r in rows]) < thi)
        few = m_all & (occ <= 2)
        many = m_all & (occ >= 4)
        if few.sum() >= 8 and many.sum() >= 8:
            print(f"  {tn}:  in 1-2 long spans -> "
                  f"{np.average(Y[few], weights=W[few]):.2f} (n={few.sum()})   |   "
                  f"in 4+ short spans -> "
                  f"{np.average(Y[many], weights=W[many]):.2f} (n={many.sum()})")

    OUT.write_text(json.dumps({"n": len(rows), "songs": songs}, indent=1))
    np.savez(Path(__file__).with_name("dur_repeat_rows.npz"),
             y=Y, w=W, g=G, dur=dur, occ=occ,
             tot=np.array([r["tot_dur"] for r in rows]),
             p=np.array([r["p_chord"] for r in rows]),
             pp=np.array([r["p_chord_pooled"] for r in rows]))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
