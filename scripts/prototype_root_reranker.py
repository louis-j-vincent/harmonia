#!/usr/bin/env python3
"""Prototype root re-rankers over musx's top-2 root candidates, measured vs the
top-1 baseline and the top-2 ceiling (0.763), on clean JAAH.

The premise-check showed the GT root is musx's rank-2 candidate 15.7% of the
time (top-2 recall 0.763 vs top-1 0.607). A re-ranker only has to choose the
right one of the two. We try cheap tie-breakers:

  bass     : between the top-2 marginal roots, pick the one the musx BASS head
             gives more mass over the span. (Different from "bass AS root", which
             was refuted; here bass only breaks a 2-way tie.)
  selfcon  : pick the top-2 root that carries more total musx root-marginal mass
             across the WHOLE song (self-consistency — a real chord root recurs).
  key      : pick the top-2 root more consistent with a song key estimated from
             the musx root-marginal aggregated over the song (Krumhansl-lite:
             prefer the root in the more populated diatonic set).
  bass+key : bass tie-break, fall back to key when bass is ~tied.

Reports duration-weighted root accuracy for each, so we see how much of the
+15.7pp ceiling each cheap signal actually captures. Read-only, cached probs.

Usage:
    .venv/bin/python scripts/prototype_root_reranker.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.models.musx_redecode import frame_posteriors, FRAME_DT  # noqa: E402
from scripts.build_jaah_corpus import parse_jaah, load_lab, LABS_DIR   # noqa: E402

RS = REPO / "docs" / "research_sessions"
JAAH_AUD = REPO / "data" / "cache" / "jaah" / "audio"
PROB_CACHE = REPO / "data" / "cache" / "musx_probs"
MUSX_LAT = 0.113

# major-scale diatonic pitch classes relative to tonic
_MAJ = [0, 2, 4, 5, 7, 9, 11]


def root_marg(triad_span):
    m = triad_span.mean(0)
    r = np.zeros(12)
    for i in range(1, 73):
        r[(i - 1) % 12] += m[i]
    return r


def bass_marg(bass_span):
    m = bass_span.mean(0)                       # (13,)
    r = np.zeros(12)
    for i in range(1, min(13, len(m))):
        r[i - 1] += m[i]
    return r


def key_scores(song_root_hist):
    """For each of 12 major keys, the mass of the song root-hist falling on that
    key's diatonic set. Returns (12,) — higher = more likely tonic."""
    out = np.zeros(12)
    for tonic in range(12):
        dia = {(tonic + d) % 12 for d in _MAJ}
        out[tonic] = sum(song_root_hist[p] for p in dia)
    return out


def main():
    pins = json.loads((RS / "jaah_source_pins.json").read_text())
    scores = json.loads((RS / "jaah_benchmark_scores.json").read_text())["scores"]

    dur = 0.0
    acc = {k: 0.0 for k in ("top1", "top2ceil", "bass", "selfcon", "key", "bass_key")}
    n_songs = 0
    for s in scores:
        slug = s["slug"]; vid = pins.get(slug)
        if not vid or not (PROB_CACHE / f"{vid}.npz").exists():
            continue
        P = frame_posteriors(JAAH_AUD / f"{vid}.wav")
        triad, bass = P[0], P[1]
        am = triad.argmax(1); nf = len(am)
        rows = load_lab(LABS_DIR / f"{slug}.lab")
        # gather spans + song-level root histogram
        spans = []
        song_hist = np.zeros(12)
        d_all = rc = 0.0
        for t0, t1, lab in rows:
            gr, gf, _ = parse_jaah(lab.split("/")[0])
            if gr is None:
                continue
            a = max(0, int(round((t0 + MUSX_LAT) / FRAME_DT)))
            b = min(nf, int(round((t1 + MUSX_LAT) / FRAME_DT)))
            if b <= a:
                continue
            rr = [((int(x) - 1) % 12) for x in am[a:b] if x != 0]
            if not rr:
                continue
            rm = root_marg(triad[a:b]); bm = bass_marg(bass[a:b])
            song_hist += rm * (t1 - t0)
            spans.append((t0, t1, gr, rm, bm))
            d_all += (t1 - t0); rc += (t1 - t0) * (int(np.bincount(rr).argmax()) == gr)
        if d_all == 0 or rc / d_all < 0.35:
            continue
        n_songs += 1
        ks = key_scores(song_hist)
        best_key = int(ks.argmax())
        dia = {(best_key + dd) % 12 for dd in _MAJ}
        for t0, t1, gr, rm, bm in spans:
            d = t1 - t0; dur += d
            order = np.argsort(-rm)
            r1, r2 = int(order[0]), int(order[1])
            acc["top1"] += d * (r1 == gr)
            acc["top2ceil"] += d * (gr in (r1, r2))
            # bass tie-break
            pick_b = r1 if bm[r1] >= bm[r2] else r2
            acc["bass"] += d * (pick_b == gr)
            # self-consistency (song-level marginal mass)
            pick_s = r1 if song_hist[r1] >= song_hist[r2] else r2
            acc["selfcon"] += d * (pick_s == gr)
            # key diatonic preference
            r1d, r2d = (r1 in dia), (r2 in dia)
            pick_k = r1 if r1d == r2d else (r1 if r1d else r2)
            acc["key"] += d * (pick_k == gr)
            # bass, fall back to key on near-tie
            if abs(bm[r1] - bm[r2]) < 0.02:
                pick_bk = pick_k
            else:
                pick_bk = pick_b
            acc["bass_key"] += d * (pick_bk == gr)

    print(f"=== root re-ranker prototypes on clean JAAH (n={n_songs}, {dur:.0f}s) ===\n")
    order = ["top1", "bass", "selfcon", "key", "bass_key", "top2ceil"]
    label = {"top1": "musx top-1 (baseline)", "bass": "bass tie-break",
             "selfcon": "self-consistency", "key": "key-diatonic",
             "bass_key": "bass + key fallback", "top2ceil": "top-2 CEILING"}
    base = acc["top1"] / dur
    for k in order:
        v = acc[k] / dur
        tag = "" if k in ("top1", "top2ceil") else f"  ({v-base:+.3f} vs top-1)"
        print(f"  {label[k]:<24} {v:.3f}{tag}")
    print("\nRead: how much of the +15.7pp top-2 ceiling each cheap signal captures. "
          "A signal near the ceiling is a free re-ranker; all near baseline means "
          "the tie-break needs real harmonic context (progression HMM).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
