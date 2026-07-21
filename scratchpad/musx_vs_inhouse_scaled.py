"""Scaled-up re-run of the musx (production) vs in-house nnls24-heads
held-out comparison from the 2026-07-21 aligned_corpus session, generalized
to run on an ARBITRARY list of held-out song titles (originally hardcoded to
6 songs) so it can track the +6pp in-house-root lead as the corpus grows.

Re-downloads audio for each held-out song (aligned_corpus.npz keeps only
pooled features, not raw audio -- disk hygiene, by design, temp dir deleted
per song same as build_aligned_corpus.py). Time- and network-bounded.

Usage:
  .venv/bin/python scratchpad/musx_vs_inhouse_scaled.py --songs "Song A" "Song B" ...
  # or, more commonly, called from a driver script with test_songs from
  # train_aligned_corpus_heads.song_level_split.
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
sys.path.insert(0, str(REPO / "scratchpad"))

from build_aligned_corpus import _family, _download_audio  # noqa: E402
from harmonia.models import musx_bass as mxb  # noqa: E402

QUALITIES = ["maj", "min", "dom", "hdim", "dim", "aug", "sus"]
FAMILY_OF_Q7 = np.array([0, 1, 0, 2, 2, 0, 0])


def run_musx_compare(songs: list[str], out_npz: Path | None = None):
    """Returns (all_rows, summary_dict). all_rows: list of
    (title, musx_root, musx_qual_idx, true_root, true_qual_idx)."""
    d = np.load(REPO / "data/cache/aligned_corpus/aligned_corpus.npz", allow_pickle=True)
    root_all, q_all, sid_all, t0_all, t1_all = d["root"], d["quality_idx"], d["song_id"], d["t0"], d["t1"]

    all_rows = []
    skipped = []
    for title in songs:
        m = sid_all == title
        n = int(m.sum())
        if n == 0:
            print(f"SKIP {title}: not in corpus")
            skipped.append((title, "not_in_corpus"))
            continue
        print(f"\n=== {title} ({n} rows) ===", flush=True)
        with tempfile.TemporaryDirectory() as td:
            wav = Path(td) / "audio.wav"
            query = f"ytsearch1:{title}"
            ok = _download_audio(query, wav)
            if not ok:
                print(f"  download failed, skipping")
                skipped.append((title, "download_failed"))
                continue
            import librosa
            dur = librosa.get_duration(path=str(wav))
            span_max = float(t1_all[m].max())
            print(f"  downloaded duration={dur:.1f}s, corpus max t1={span_max:.1f}s")
            if dur < span_max * 0.7:
                print(f"  SUSPECT MISMATCH (audio shorter than corpus's own aligned span) -- skipping")
                skipped.append((title, "duration_mismatch"))
                continue
            try:
                labels = mxb.musx_labels(wav, use_cache=False)
            except Exception as exc:
                print(f"  musx failed: {exc}")
                skipped.append((title, f"musx_error:{exc}"))
                continue
            seg_bounds = list(zip(t0_all[m].tolist(), t1_all[m].tolist()))
            rq = mxb.root_quality_per_segment(labels, seg_bounds)
            for (mr, msev), tr, tq in zip(rq, root_all[m], q_all[m]):
                mfam = QUALITIES.index(_family(msev)) if msev is not None else -1
                all_rows.append((title, mr, mfam, int(tr), int(tq)))
            n_valid = sum(1 for r in rq if r[0] != -1)
            print(f"  musx labeled {n_valid}/{n} rows")

    if not all_rows:
        print("\nNo songs successfully compared.")
        return all_rows, {}

    arr = np.array([(mr, mfam, tr, tq) for _, mr, mfam, tr, tq in all_rows])
    valid = arr[:, 0] != -1
    mr, mfam, tr, tq = arr[valid, 0], arr[valid, 1], arr[valid, 2], arr[valid, 3]
    n_songs = len(set(r[0] for r in all_rows))
    summary = dict(
        n_rows=int(valid.sum()), n_rows_total=len(arr), n_songs=n_songs,
        root_acc=float((mr == tr).mean()),
        qual_acc=float((mfam == tq).mean()),
        qual_family_acc=float((FAMILY_OF_Q7[mfam] == FAMILY_OF_Q7[tq]).mean()),
    )
    print(f"\n=== MUSX (current production) on {valid.sum()}/{len(arr)} comparable rows across "
          f"{n_songs} songs ===")
    print(f"  root_acc  = {summary['root_acc']:.3f}")
    print(f"  qual_acc  = {summary['qual_acc']:.3f}")
    print(f"  qual_fam  = {summary['qual_family_acc']:.3f}")
    print(f"  skipped: {skipped}")

    if out_npz is not None:
        np.savez(out_npz, rows=np.array(all_rows, dtype=object),
                  skipped=np.array(skipped, dtype=object))
        print(f"\nsaved {out_npz}")
    return all_rows, summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=REPO / "scratchpad/musx_held_out_compare_scaled.npz")
    a = ap.parse_args()
    run_musx_compare(a.songs, a.out)
