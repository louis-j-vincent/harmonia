"""H4 — Can the existing structure machinery recover KNOWN song forms?

The accompaniment DB gives per-bar section labels (ground truth POP909 never
had). MIDI-derived beat chroma is fed through the *existing* pipeline pieces
(structure.build_ssm → periodicity.score_periods → greedy nearest-centroid
clustering, as in plot_structure_proposal_illustrations.illustrate_form_clustering)
and scored against the true form.

Design note: MMA uses one groove per song, so accompaniment rhythm/timbre are
constant throughout — the SSM can only see HARMONY. This isolates exactly the
signal Candidate C conflated with rhythm repetition on real audio.

Usage: .venv/bin/python scripts/analyze_accomp_structure.py [--max-songs N]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pretty_midi

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.models.periodicity import score_periods  # noqa: E402
from harmonia.models.structure import build_ssm  # noqa: E402

DB = REPO / "data" / "accomp_db" / "db.jsonl"


def beat_probs_from_midi(pm: pretty_midi.PrettyMIDI, n_beats: int, spb: float) -> np.ndarray:
    """(n_beats, 88) duration-weighted note activity, harmonic tracks only."""
    bp = np.zeros((n_beats, 88), dtype=np.float32)
    for inst in pm.instruments:
        if inst.is_drum:
            continue
        for n in inst.notes:
            key = n.pitch - 21
            if not (0 <= key < 88):
                continue
            b0 = int(n.start / spb)
            b1 = int(np.ceil(n.end / spb))
            for b in range(max(b0, 0), min(b1, n_beats)):
                ov = min(n.end, (b + 1) * spb) - max(n.start, b * spb)
                if ov > 0:
                    bp[b, key] += ov
    return bp


def greedy_cluster(windows: np.ndarray, threshold: float) -> list[int]:
    """Nearest-centroid greedy clustering of window chroma summaries."""
    centroids: list[np.ndarray] = []
    members: list[list[int]] = []
    labels = []
    for i, w in enumerate(windows):
        wn = w / (np.linalg.norm(w) + 1e-12)
        best, best_sim = -1, -1.0
        for ci, c in enumerate(centroids):
            sim = float(wn @ (c / (np.linalg.norm(c) + 1e-12)))
            if sim > best_sim:
                best, best_sim = ci, sim
        if best >= 0 and best_sim >= threshold:
            labels.append(best)
            centroids[best] = centroids[best] + wn
            members[best].append(i)
        else:
            labels.append(len(centroids))
            centroids.append(wn.copy())
            members.append([i])
    return labels


def adjusted_rand(a: list, b: list) -> float:
    """Adjusted Rand Index via pair counting (no sklearn dependency)."""
    n = len(a)
    if n < 2:
        return 0.0
    from math import comb

    ct: dict[tuple, int] = Counter(zip(a, b))
    rows = Counter(a)
    cols = Counter(b)
    sum_ct = sum(comb(v, 2) for v in ct.values())
    sum_r = sum(comb(v, 2) for v in rows.values())
    sum_c = sum(comb(v, 2) for v in cols.values())
    total = comb(n, 2)
    expected = sum_r * sum_c / total
    max_idx = (sum_r + sum_c) / 2
    if max_idx == expected:
        return 0.0
    return (sum_ct - expected) / (max_idx - expected)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-songs", type=int, default=200)
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--mode", choices=["pooled", "sequence"], default="pooled",
                    help="window summary: pooled mean chroma vs position-wise "
                         "concatenated per-bar chroma")
    args = ap.parse_args()

    records = [json.loads(line) for line in open(DB)]
    jazz = [r for r in records if r["corpus"] == "jazz1460" and r["beats_per_bar"] == 4]
    # only songs whose form has ≥2 distinct sections — recovery is trivial otherwise
    jazz = [r for r in jazz if len(set(r["section_per_bar"])) >= 2]
    jazz = jazz[: args.max_songs]

    period_hits_top1 = period_hits_top3 = 0
    aris = []
    aris_gtperiod = []
    n_done = 0
    for rec in jazz:
        mid = REPO / rec["midi_path"]
        if not mid.exists():
            continue
        try:
            pm = pretty_midi.PrettyMIDI(str(mid))
        except Exception:
            continue
        spb = 60.0 / rec["tempo"]
        n_beats = rec["n_bars"] * 4
        bp = beat_probs_from_midi(pm, n_beats, spb)
        if bp.sum() < 1e-6 or n_beats < 32:
            continue

        # ── period detection vs GT dominant section length ──────────────
        sections = rec["section_per_bar"]
        runs = []
        i = 0
        while i < len(sections):
            j = i
            while j < len(sections) and sections[j] == sections[i]:
                j += 1
            runs.append(j - i)
            i = j
        gt_period_beats = Counter(r * 4 for r in runs).most_common(1)[0][0]
        cands = score_periods(bp, beats_per_bar=4, max_period_bars=16, top_k=3)
        top = sorted(cands, key=cands.get, reverse=True)
        if top and top[0] == gt_period_beats:
            period_hits_top1 += 1
        if gt_period_beats in top:
            period_hits_top3 += 1

        # ── form clustering at detected period (phase 0 = true by construction)
        def cluster_ari(period_beats: int) -> float:
            n_win = n_beats // period_beats
            if n_win < 2:
                return float("nan")
            ssm_chroma = bp[: n_win * period_beats].reshape(n_win, period_beats, 88)
            n_bars_win = period_beats // 4
            if args.mode == "pooled":
                # summarize each window by its mean chroma (fold 88→12)
                win = np.zeros((n_win, 12))
                for w in range(n_win):
                    v = ssm_chroma[w].sum(axis=0)
                    for k in range(88):
                        win[w, (k + 21) % 12] += v[k]
            else:
                # position-wise: per-bar chroma, L2'd per bar, concatenated —
                # cosine then respects the chord SEQUENCE, not just its content
                win = np.zeros((n_win, n_bars_win * 12))
                for w in range(n_win):
                    for bar in range(n_bars_win):
                        v = ssm_chroma[w, bar * 4 : (bar + 1) * 4].sum(axis=0)
                        c = np.zeros(12)
                        for k in range(88):
                            c[(k + 21) % 12] += v[k]
                        c /= np.linalg.norm(c) + 1e-12
                        win[w, bar * 12 : (bar + 1) * 12] = c
            labels = greedy_cluster(win, args.threshold)
            # per-bar predicted labels
            pred_bar = []
            for bar in range(rec["n_bars"]):
                w = min(bar * 4 // period_beats, n_win - 1)
                pred_bar.append(labels[w])
            return adjusted_rand(sections[: len(pred_bar)], pred_bar)

        if top:
            a = cluster_ari(top[0])
            if not np.isnan(a):
                aris.append(a)
        a = cluster_ari(gt_period_beats)
        if not np.isnan(a):
            aris_gtperiod.append(a)
        n_done += 1

    aris = np.array(aris)
    aris_gt = np.array(aris_gtperiod)
    print(f"H4 — Structure recovery on {n_done} multi-section jazz songs "
          f"(threshold={args.threshold}):")
    print(f"    period detection: top-1 hit {period_hits_top1/n_done:.0%}, "
          f"top-3 hit {period_hits_top3/n_done:.0%} (vs GT dominant section length)")
    print(f"    form-clustering ARI @ detected period: mean {aris.mean():.2f}, "
          f"median {np.median(aris):.2f}, ≥0.5 in {(aris >= 0.5).mean():.0%} of songs")
    print(f"    form-clustering ARI @ GT period      : mean {aris_gt.mean():.2f}, "
          f"median {np.median(aris_gt):.2f}, ≥0.5 in {(aris_gt >= 0.5).mean():.0%} of songs")
    print("    (ARI=1 perfect form recovery, 0 = chance)")


if __name__ == "__main__":
    main()
