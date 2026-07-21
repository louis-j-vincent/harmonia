#!/usr/bin/env python3
"""Build a TRUSTWORTHY, non-circular training corpus from iReal charts
aligned per-SECTION to real YouTube audio (2026-07-21).

Why this exists: the old `corpus_50.npz` (used by earlier retraining work,
see docs/known_issues.md #19/#31) has iReal GT time-aligned via DTW against
this project's OWN model predictions — when the model is wrong, GT slides
with it (circular measurement, exactly what Mission 1,
docs/mission_1_real_audio_benchmark_design.md, was built to fix). Mission 1's
own DTW approach then failed its own ±150ms gate on all 3 pilots. This
script uses the method that superseded it: Beat This! (ISMIR 2024) real
downbeat anchors + a discrete per-SECTION search (not one rigid whole-song
walk), scored by a DTW-independent chord-change-point validator — see
`harmonia/data/ireal_youtube_align.py` for the full method and its honest,
measured yield (52% of individual sections on a 6-song spot-check, vs ~10-
20% of whole songs with the earlier whole-song-only version).

Only ACCEPTED sections (align_tune_sections_to_audio's own quality gate)
contribute rows. A row is one iReal chord span mapped to real audio time,
labeled with (root pitch-class, 7-way quality family) from the iReal
chart's OWN raw token (NOT the MMA-converted string — the two notations
disagree on e.g. half-diminished, see _mma_chart_to_chords_for_bars's
docstring) — never from a model prediction, so this corpus can be used to
TRAIN or EVALUATE the model without circularity.

Resumable: progress (which tunes have been attempted, row count so far) is
persisted to data/cache/aligned_corpus/progress.json; re-running the same
command skips already-attempted tunes and appends to the existing corpus.
Downloaded audio is deleted after each tune (disk hygiene) — only the
extracted 24-dim NNLS features + labels are kept.

NOT safe to run concurrently: --corpus pop400 and --corpus jazz1460 both
read/write the SAME progress.json + output npz (by design, so they build
one shared corpus) — two processes racing on the same files WILL corrupt
one or both writes. Run corpora sequentially, one at a time.

Usage:
    .venv/bin/python scripts/build_aligned_corpus.py --corpus pop400 --max-songs 40
    .venv/bin/python scripts/build_aligned_corpus.py --corpus jazz1460 --max-songs 40
    .venv/bin/python scripts/build_aligned_corpus.py --corpus pop400 --max-songs 40 --resume
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import logging
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

QUALITIES = ["maj", "min", "dom", "hdim", "dim", "aug", "sus"]
CACHE_DIR = REPO / "data" / "cache" / "aligned_corpus"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
PROGRESS_JSON = CACHE_DIR / "progress.json"


def _family(sev: str) -> str:
    """iReal-token quality string (min7, maj9, hdim7, dom13, 7b9, ...) -> the
    7-way family this project's trained heads use (maj/min/dom/hdim/dim/aug/
    sus). A bare "7"/"9"/"11"/"13"/"7b9" etc. (no maj/min/dim/aug/sus prefix)
    is a plain dominant chord in this notation."""
    if sev.startswith("hdim"):
        return "hdim"
    if sev.startswith("dim"):
        return "dim"
    if sev.startswith("aug"):
        return "aug"
    if "sus" in sev:
        return "sus"
    if sev.startswith("min"):
        return "min"
    if sev.startswith("maj"):
        return "maj"
    return "dom"


def _load_progress() -> dict:
    if PROGRESS_JSON.exists():
        return json.loads(PROGRESS_JSON.read_text())
    return {"done": [], "n_rows": 0, "n_accepted_sections": 0, "n_attempted": 0}


def _save_progress(p: dict) -> None:
    PROGRESS_JSON.write_text(json.dumps(p, indent=1))


def _download_audio(query: str, dst_wav: Path) -> bool:
    """yt-dlp search+download the first hit, transcoded straight to wav."""
    import yt_dlp

    opts = {
        "format": "bestaudio/best",
        "outtmpl": str(dst_wav.with_suffix("")) + ".%(ext)s",
        "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "wav"}],
        "quiet": True, "no_warnings": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            ydl.extract_info(query, download=True)
    except Exception as exc:  # noqa: BLE001 — one bad search must not kill the run
        logger.warning("  download failed for %r: %s", query, exc)
        return False
    # yt-dlp names the postprocessed file <stem>.wav; find and normalise it.
    for f in dst_wav.parent.glob(dst_wav.stem + ".*"):
        if f.suffix == ".wav":
            if f != dst_wav:
                f.rename(dst_wav)
            return True
    return False


def process_tune(tune, tmp_dir: Path) -> list[dict]:
    """Download + align + extract one tune. Returns [] on any failure or
    zero accepted sections — never raises (one bad tune must not kill the
    run)."""
    from harmonia.data.ireal_youtube_align import align_tune_sections_to_audio
    from harmonia.irealb_fetcher import _parse_ireal_chord_token
    from harmonia.models import nnls_features as nf

    audio_path = tmp_dir / "audio.wav"
    query = f"ytsearch1:{tune.title} {tune.composer or ''}".strip()
    if not _download_audio(query, audio_path):
        return []

    try:
        results = align_tune_sections_to_audio(tune, audio_path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("  alignment failed for %s: %s", tune.title, exc)
        return []

    accepted = [r for r in results if r.get("accepted")]
    if not accepted:
        return []

    try:
        # use_cache=False: every tune gets a FRESH tmp_dir but the SAME
        # literal filename ("audio.wav") — a stem-keyed cache would collide
        # across tunes exactly like the /api/record-analyze bug this session
        # already found and fixed (known_issues.md, 2026-07-21).
        arr, times = nf.extract_bothchroma(audio_path, use_cache=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("  feature extraction failed for %s: %s", tune.title, exc)
        return []

    rows: list[dict] = []
    for r in accepted:
        warp = r["warp"]
        for c in r["chords"]:
            tok = c.get("tok")
            if not tok or tok.strip() in ("z", ""):
                continue
            parsed = _parse_ireal_chord_token(tok)
            if parsed is None:
                continue
            root, sev = parsed
            t0, t1 = warp(c["start_s"]), warp(c["end_s"])
            if t1 <= t0:
                continue
            mask = (times >= t0) & (times < t1)
            if not mask.any():
                continue
            rows.append({
                "feat24": arr[mask].mean(0).astype(np.float32),
                "root": int(root) % 12, "quality_idx": QUALITIES.index(_family(sev)),
                "t0": float(t0), "t1": float(t1),
                "song": tune.title, "section": r["label"],
            })
    logger.info("  %s: %d/%d sections accepted -> %d chord rows",
               tune.title, len(accepted), len(results), len(rows))
    return rows


def _append_to_corpus(out_npz: Path, rows: list[dict]) -> int:
    """Merge new rows into the existing corpus npz (if any), save, return
    the new total row count."""
    new = {
        "feat24": np.stack([r["feat24"] for r in rows]),
        "root": np.array([r["root"] for r in rows], dtype=np.int64),
        "quality_idx": np.array([r["quality_idx"] for r in rows], dtype=np.int64),
        "t0": np.array([r["t0"] for r in rows], dtype=np.float64),
        "t1": np.array([r["t1"] for r in rows], dtype=np.float64),
        "song_id": np.array([r["song"] for r in rows]),
        "section": np.array([r["section"] for r in rows]),
    }
    if out_npz.exists():
        old = np.load(out_npz, allow_pickle=True)
        merged = {k: np.concatenate([old[k], new[k]]) for k in new}
    else:
        merged = new
    merged["qualities"] = np.array(QUALITIES)
    np.savez(out_npz, **merged)
    return len(merged["root"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", choices=["pop400", "jazz1460"], required=True)
    ap.add_argument("--max-songs", type=int, default=40)
    ap.add_argument("--out", type=Path, default=CACHE_DIR / "aligned_corpus.npz")
    a = ap.parse_args()

    from harmonia.data.ireal_corpus import load_playlist

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):   # load_playlist prints "Parsed X" per tune
        tunes = load_playlist(str(REPO / "data" / "ireal" / f"{a.corpus}.txt"))

    progress = _load_progress()
    done = set(progress["done"])
    todo = [t for t in tunes if t.title not in done][: a.max_songs]
    logger.info("corpus=%s: %d tunes total, %d already done, processing %d",
               a.corpus, len(tunes), len(done), len(todo))

    t_start = time.time()
    for i, tune in enumerate(todo):
        tmp_dir = Path(tempfile.mkdtemp(prefix="aligned_corpus_"))
        try:
            rows = process_tune(tune, tmp_dir)
        except Exception as exc:  # noqa: BLE001 — never let one tune kill the run
            logger.warning("  UNEXPECTED failure on %s: %s", tune.title, exc)
            rows = []
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

        progress["done"].append(tune.title)
        progress["n_attempted"] += 1
        if rows:
            n_total = _append_to_corpus(a.out, rows)
            progress["n_rows"] = n_total
            progress["n_accepted_sections"] += len({(r["song"], r["section"]) for r in rows})
        _save_progress(progress)
        logger.info("[%d/%d] elapsed=%.0fs total_rows=%d",
                   i + 1, len(todo), time.time() - t_start, progress["n_rows"])

    logger.info("DONE. %d tunes attempted this run, %d total rows in %s",
               len(todo), progress["n_rows"], a.out)


if __name__ == "__main__":
    main()
