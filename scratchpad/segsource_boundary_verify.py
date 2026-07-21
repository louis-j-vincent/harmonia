"""Proper verification of the segment_source="musx" lever on REAL audio
(2026-07-21, coordinator-directed follow-up to the earlier premise-check in
production_audit_root_quality_bass.py --segment-source musx).

That earlier check only compared root/quality/bass IDENTITY accuracy at GT
chord-span midpoints across two SEPARATE downloads (nnls run vs musx run) --
confounded by download variance (a different video/take could be pulled each
time) and never actually measured boundary quality (the RWC number this
lever is based on, F1 0.87-0.90, is a BOUNDARY metric, not an identity
metric). This script fixes both:

1. PAIRED same-download comparison: each held-out song is downloaded ONCE,
   then infer_chords_v1 is run TWICE on the identical audio file (once per
   segment_source) -- eliminates the download-variance confound entirely.
2. Real boundary-F1: aligned_corpus's per-row t0 values ARE real GT chord-
   change timestamps (iReal chart chord onsets, warped to audio time via the
   validated per-section alignment -- see build_aligned_corpus.py), not just
   a scoring midpoint. Restricting to the time range actually covered by
   ACCEPTED sections (outside that range there is no GT at all), this script
   computes MIREX-style greedy one-to-one boundary matching (same method as
   the RWC citation in docs/known_issues.md "Measurement" entry, generalized
   to real non-circular audio instead of RWC) at tolerances {0.5, 1.0}s (a
   looser tolerance than RWC's {0.25,0.5} because this corpus's own
   section-alignment has a higher noise floor than RWC's clean GT --
   align_tune_sections_to_audio's own accept gate reports median timing
   errors in the tens-hundreds of ms).

Multi-seed = multiple independent random song samples (there's no training
stochasticity here, just sampling variance), for an honest accept/reject
verdict rather than one directional number.

Usage:
  .venv/bin/python scratchpad/segsource_boundary_verify.py --seeds 0 1 2 --songs-per-seed 6
"""
from __future__ import annotations

import argparse
import io
import contextlib
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from build_aligned_corpus import _download_audio  # noqa: E402
from harmonia.models.chord_pipeline_v1 import infer_chords_v1  # noqa: E402
from harmonia.data.ireal_corpus import load_playlist  # noqa: E402


def greedy_match_f1(pred_times, gt_times, tol):
    """MIREX-style greedy one-to-one boundary matching. Returns (P, R, F1, nPred, nGT)."""
    pred_times = sorted(pred_times)
    gt_times = sorted(gt_times)
    used_gt = [False] * len(gt_times)
    hits = 0
    for pt in pred_times:
        best_j, best_d = -1, tol
        for j, gt in enumerate(gt_times):
            if used_gt[j]:
                continue
            d = abs(pt - gt)
            if d <= best_d:
                best_d, best_j = d, j
        if best_j >= 0:
            used_gt[best_j] = True
            hits += 1
    p = hits / len(pred_times) if pred_times else 0.0
    r = hits / len(gt_times) if gt_times else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return p, r, f1, len(pred_times), len(gt_times)


def boundaries_for_song(chart_chords, sections, tmin, tmax):
    """Predicted chord onset times (start_s) that fall inside [tmin, tmax)."""
    return [c["start_s"] for c in chart_chords if tmin <= c["start_s"] < tmax]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--songs-per-seed", type=int, default=6)
    a = ap.parse_args()

    d = np.load(REPO / "data/cache/aligned_corpus/aligned_corpus.npz", allow_pickle=True)
    sid_all, t0_all, t1_all, sec_all = d["song_id"], d["t0"], d["t1"], d["section"]
    all_songs = sorted(set(sid_all.tolist()))

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        tunes_pop = load_playlist(str(REPO / "data/ireal/pop400.txt"))
        tunes_jazz = load_playlist(str(REPO / "data/ireal/jazz1460.txt"))
    composer_of = {t.title: (t.composer or "") for t in tunes_pop + tunes_jazz}

    # per-seed, per-tolerance accumulators: {seed: {"nnls": [(P,R,F1,nP,nG),...], "musx": [...]}}
    tol_results = {0.5: {"nnls": [], "musx": []}, 1.0: {"nnls": [], "musx": []}}
    per_seed_summary = []

    done_songs = set()
    song_idx_counter = 0
    for seed in a.seeds:
        print(f"\n{'#'*70}\n# SEED {seed}\n{'#'*70}")
        rng = np.random.RandomState(seed)
        candidates = [s for s in all_songs if s not in done_songs]
        rng.shuffle(candidates)
        songs = candidates[: a.songs_per_seed]
        done_songs.update(songs)  # avoid re-testing the same song across seeds (independence)

        seed_rows = {0.5: {"nnls": [], "musx": []}, 1.0: {"nnls": [], "musx": []}}
        for title in songs:
            m = sid_all == title
            print(f"\n=== {title} ({int(m.sum())} rows) ===", flush=True)
            with tempfile.TemporaryDirectory() as td:
                wav = Path(td) / f"song{song_idx_counter}.wav"
                song_idx_counter += 1
                query = f"ytsearch1:{title} {composer_of.get(title, '')}".strip()
                if not _download_audio(query, wav):
                    print("  download failed, skipping"); continue
                import librosa
                dur = librosa.get_duration(path=str(wav))
                span_max = float(t1_all[m].max())
                if dur < span_max * 0.7:
                    print("  SUSPECT MISMATCH -- skipping"); continue

                charts = {}
                for seg_src in ("nnls", "musx"):
                    try:
                        charts[seg_src] = infer_chords_v1(
                            wav, cache_dir=Path(td) / f"cache_{seg_src}",
                            feature_frontend="nnls24", bass_frontend="musx",
                            quality_frontend="musx", segment_source=seg_src,
                            beat_period_mode="bestfit",
                        )
                    except Exception as exc:
                        print(f"  {seg_src} pipeline failed: {exc}")
                        charts[seg_src] = None
                if charts["nnls"] is None or charts["musx"] is None:
                    continue

                # per accepted-section GT boundary matching
                sections_here = sorted(set(sec_all[m].tolist()))
                for sec in sections_here:
                    ms = m & (sec_all == sec)
                    gt_times = sorted(set(t0_all[ms].tolist()))
                    if len(gt_times) < 3:
                        continue  # too few GT points in this section to be meaningful
                    tmin, tmax = min(gt_times), float(t1_all[ms].max())
                    for seg_src in ("nnls", "musx"):
                        pred_times = boundaries_for_song(charts[seg_src].chords, sections_here, tmin, tmax)
                        for tol in (0.5, 1.0):
                            p, r, f1, npred, ngt = greedy_match_f1(pred_times, gt_times, tol)
                            tol_results[tol][seg_src].append((p, r, f1, npred, ngt))
                            seed_rows[tol][seg_src].append((p, r, f1, npred, ngt))
                    print(f"  section {sec}: {len(gt_times)} GT boundaries, "
                          f"nnls_pred={len(boundaries_for_song(charts['nnls'].chords, sections_here, tmin, tmax))}, "
                          f"musx_pred={len(boundaries_for_song(charts['musx'].chords, sections_here, tmin, tmax))}")

        # per-seed micro summary (pool hits/preds/gts across sections in this seed)
        seed_summary = {"seed": seed}
        for tol in (0.5, 1.0):
            for seg_src in ("nnls", "musx"):
                rows = seed_rows[tol][seg_src]
                if not rows:
                    continue
                tot_hits_p = sum(p * npred for p, r, f1, npred, ngt in rows)
                tot_pred = sum(npred for p, r, f1, npred, ngt in rows)
                tot_gt = sum(ngt for p, r, f1, npred, ngt in rows)
                tot_hits_r = sum(r * ngt for p, r, f1, npred, ngt in rows)
                P = tot_hits_p / tot_pred if tot_pred else 0.0
                R = tot_hits_r / tot_gt if tot_gt else 0.0
                F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
                seed_summary[f"{seg_src}_tol{tol}"] = dict(P=P, R=R, F1=F1, n_sections=len(rows))
                print(f"[seed {seed}] {seg_src} @ tol={tol}s: P={P:.3f} R={R:.3f} F1={F1:.3f} "
                      f"(n_sections={len(rows)}, pred/gt ratio={tot_pred/tot_gt if tot_gt else float('nan'):.2f})")
        per_seed_summary.append(seed_summary)
        # incremental save after EVERY seed (2026-07-21: a prior run got cut off
        # mid-seed-2 by the tool's foreground timeout with zero results persisted
        # -- seeds 0/1 had already printed real numbers to stdout but nothing was
        # written to disk. Save after each seed so a later interruption never
        # loses completed work.)
        import json
        (REPO / "scratchpad/segsource_boundary_verify_results.json").write_text(
            json.dumps(per_seed_summary, indent=1))

    print(f"\n\n{'='*70}\nFINAL VERDICT across {len(a.seeds)} seeds\n{'='*70}")
    for tol in (0.5, 1.0):
        for seg_src in ("nnls", "musx"):
            rows = tol_results[tol][seg_src]
            if not rows:
                continue
            tot_pred = sum(npred for p, r, f1, npred, ngt in rows)
            tot_gt = sum(ngt for p, r, f1, npred, ngt in rows)
            tot_hits_p = sum(p * npred for p, r, f1, npred, ngt in rows)
            tot_hits_r = sum(r * ngt for p, r, f1, npred, ngt in rows)
            P = tot_hits_p / tot_pred if tot_pred else 0.0
            R = tot_hits_r / tot_gt if tot_gt else 0.0
            F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
            print(f"POOLED {seg_src} @ tol={tol}s: P={P:.3f} R={R:.3f} F1={F1:.3f} "
                  f"(n_sections={len(rows)}, pred/gt ratio={tot_pred/tot_gt if tot_gt else float('nan'):.2f})")

    import json
    (REPO / "scratchpad/segsource_boundary_verify_results.json").write_text(
        json.dumps(per_seed_summary, indent=1))
    print("\nsaved scratchpad/segsource_boundary_verify_results.json")


if __name__ == "__main__":
    main()
