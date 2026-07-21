"""Diagnostic audit (2026-07-21, user-directed pivot): root / quality / bass
accuracy of the ACTUAL LIVE PRODUCTION analyze path
(scripts/harmonia_server.py's nnls24-frontend call, confirmed via `ps eww` on
every running harmonia_server.py instance -- no HARMONIA_ANALYZE_BASS/QUALITY/
FRONTEND overrides are set anywhere, so the code defaults ARE what's live:
feature_frontend=nnls24, bass_frontend=musx, quality_frontend=musx,
segment_source=nnls, beat_period_mode=bestfit), run end-to-end on real audio
and scored against aligned_corpus.npz's non-circular iReal GT (root, quality,
bass -- the sounding-bass target).

This is DIFFERENT from the prior session's musx_vs_inhouse comparison: that
called musx_bass.musx_labels()/root_quality_per_segment() directly (bypassing
the full pipeline -- no chord-change segmentation, no HMM/joint-decode, no
Occam post-pass). This calls infer_chords_v1() itself, i.e. what a real user
actually gets from /api/analyze.

Usage:
  .venv/bin/python scratchpad/production_audit_root_quality_bass.py --max-songs 15
"""
from __future__ import annotations

import argparse
import sys
import tempfile
from pathlib import Path

import numpy as np

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from build_aligned_corpus import _family, _download_audio  # noqa: E402
from harmonia.models.chord_pipeline_v1 import infer_chords_v1, _parse_harte_label  # noqa: E402
from harmonia.data.ireal_corpus import load_playlist  # noqa: E402
import io as _io
import contextlib as _contextlib

QUALITIES = ["maj", "min", "dom", "hdim", "dim", "aug", "sus"]
FAMILY_OF_Q7 = np.array([0, 1, 0, 2, 2, 0, 0])


def chord_at(chart_chords, t):
    for c in chart_chords:
        if c["start_s"] <= t < c["end_s"]:
            return c["label"]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-songs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--segment-source", choices=["nnls", "musx"], default="nnls",
                     help="2026-07-21 premise-check: live default is 'nnls' (unmeasured "
                          "boundary-F1); 'musx' measures F1 0.87-0.90 on RWC but is opt-in/off. "
                          "Pass musx here to test on real audio before considering a default flip.")
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    if a.out is None:
        a.out = REPO / f"scratchpad/production_audit_results_{a.segment_source}.npz"

    d = np.load(REPO / "data/cache/aligned_corpus/aligned_corpus.npz", allow_pickle=True)
    root_all, q_all = d["root"], d["quality_idx"]
    bass_all = d["bass"] if "bass" in d else d["root"]
    sid_all, t0_all, t1_all = d["song_id"], d["t0"], d["t1"]

    songs = sorted(set(sid_all.tolist()))
    rng = np.random.RandomState(a.seed)
    rng.shuffle(songs)
    songs = songs[: a.max_songs]
    print(f"auditing live production pipeline on {len(songs)} aligned_corpus songs")

    # Title-only ytsearch1 (what the prior session's musx_vs_inhouse script used)
    # is a real hazard for ambiguous/common titles ("Again" pulled a wrong/
    # different recording entirely in a smoke test here -- predicted roots
    # stuck in one key, true roots scattered with no transposition pattern,
    # not the aligned_corpus's own alignment bug). Match build_aligned_corpus.py's
    # OWN query convention (title + composer) so we redownload the SAME video
    # family the corpus was built from.
    buf = _io.StringIO()
    with _contextlib.redirect_stdout(buf):
        tunes_pop = load_playlist(str(REPO / "data/ireal/pop400.txt"))
        tunes_jazz = load_playlist(str(REPO / "data/ireal/jazz1460.txt"))
    composer_of = {t.title: (t.composer or "") for t in tunes_pop + tunes_jazz}

    rows = []  # (title, pred_root, pred_qual_idx, pred_bass, true_root, true_qual, true_bass)
    skipped = []
    n_chords_emitted = []
    for song_idx, title in enumerate(songs):
        m = sid_all == title
        n = int(m.sum())
        print(f"\n=== {title} ({n} rows) ===", flush=True)
        with tempfile.TemporaryDirectory() as td:
            # CRITICAL: nnls_features.extract_bothchroma / musx_bass cache by
            # audio_path.stem ONLY (data/cache/nnls_infer/<stem>.npz,
            # data/cache/musx_infer/<stem>_submission.lab) -- NOT by content or
            # mtime. A literal "audio.wav" name (what a naive per-song
            # tempfile.mkdtemp()/audio.wav pattern gives) collides across every
            # song in this loop, silently reusing song #1's cached features/
            # labels for every subsequent song. Caught by a smoke test here
            # (2026-07-21): 3 different songs all logged identical
            # "key=A minor" / identical OCCAM bar-decision traces, and overall
            # root_acc collapsed to 0.094 -- not a real production number.
            # Cleaned the polluted data/cache/{nnls_infer,musx_infer}/audio*
            # files and fixed by giving each song a UNIQUE stem (its index),
            # matching production's own video-id-stem convention that avoids
            # this collision class by construction.
            wav = Path(td) / f"song{song_idx}.wav"
            query = f"ytsearch1:{title} {composer_of.get(title, '')}".strip()
            ok = _download_audio(query, wav)
            if not ok:
                print("  download failed, skipping"); skipped.append((title, "download_failed")); continue
            import librosa
            dur = librosa.get_duration(path=str(wav))
            span_max = float(t1_all[m].max())
            if dur < span_max * 0.7:
                print("  SUSPECT MISMATCH -- skipping"); skipped.append((title, "duration_mismatch")); continue
            try:
                chart = infer_chords_v1(
                    wav,
                    cache_dir=Path(td) / "cache",
                    feature_frontend="nnls24",
                    bass_frontend="musx",
                    quality_frontend="musx",
                    segment_source=a.segment_source,
                    beat_period_mode="bestfit",
                )
            except Exception as exc:
                print(f"  pipeline failed: {exc}"); skipped.append((title, f"pipeline_error:{exc}")); continue

            n_scored = 0
            n_chords_emitted.append(len(chart.chords))
            for i in np.where(m)[0]:
                tmid = 0.5 * (float(t0_all[i]) + float(t1_all[i]))
                label = chord_at(chart.chords, tmid)
                if label is None:
                    continue
                pr, psev, pbass = _parse_harte_label(label)
                if pr is None:
                    continue
                pq = QUALITIES.index(_family(psev))
                rows.append((title, int(pr), pq, pbass if pbass is not None else int(pr),
                             int(root_all[i]), int(q_all[i]), int(bass_all[i])))
                n_scored += 1
            print(f"  scored {n_scored}/{n} rows")

    if not rows:
        print("\nNo songs successfully audited."); return

    arr = np.array([(r[1], r[2], r[3], r[4], r[5], r[6]) for r in rows])
    pr, pq, pb, tr, tq, tb = arr.T
    n_songs = len(set(r[0] for r in rows))
    print(f"\n{'='*70}\nLIVE PRODUCTION PIPELINE AUDIT: {len(arr)} rows / {n_songs} songs "
          f"(skipped: {skipped})\n{'='*70}")
    print(f"ROOT accuracy          : {(pr == tr).mean():.3f}")
    print(f"QUALITY accuracy (7way): {(pq == tq).mean():.3f}")
    print(f"QUALITY family (partial): {(FAMILY_OF_Q7[pq] == FAMILY_OF_Q7[tq]).mean():.3f}")
    print(f"BASS (sounding) accuracy: {(pb == tb).mean():.3f}")
    print(f"JOINT root+quality      : {((pr == tr) & (pq == tq)).mean():.3f}")
    print(f"chords emitted per song (segmentation density): mean={np.mean(n_chords_emitted):.1f} "
          f"median={np.median(n_chords_emitted):.1f} (n={len(n_chords_emitted)} songs)")

    np.savez(a.out, rows=np.array(rows, dtype=object), skipped=np.array(skipped, dtype=object),
              n_chords_emitted=np.array(n_chords_emitted))
    print(f"\nsaved {a.out}")


if __name__ == "__main__":
    main()
