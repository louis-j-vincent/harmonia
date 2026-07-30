"""scripts/backfill_musx_probs_app_charts.py — bounded musx_probs backfill for
the app's own YouTube charts (task 4 of the lock-propagation final-integration
brief). Only 3/35 app charts have a musx_probs cache entry (verified: the
cache key is the YOUTUBE VIDEO ID, not the audio file's slug -- app audio is
re-transcoded to docs/audio/<slug>.m4a for playback, a DIFFERENT stem than
the video-id-named file the original analysis decoded; see
harmonia/serving/api.py's api_context_rescore stem-resolution comment). Without
this, /api/context_rescore falls back to the slower nnls_heads backend for
every app chart except those 3.

Capped at 15 songs / 20 minutes wall-clock total (whichever hits first), and
gated on free disk >= 2 GB (checked here AND before each song, since a
disk-full crisis has happened on this machine before -- CLAUDE.md).

Mechanism: musx_redecode.frame_posteriors(path) caches under
data/cache/musx_probs/<path.stem>.npz and decodes whatever real audio is AT
that path -- it does not require the path's name to match anything. So a
temp HARD LINK named <video_id>.<ext> pointing at the REAL slug-named audio
file gives frame_posteriors both a correct decode (real audio content) and
the correct cache key (video_id) the live endpoint will look up. A SYMLINK
does NOT work here (tried first, verified broken): frame_posteriors does
``Path(audio_path).resolve()`` before taking ``.stem``, and ``resolve()``
follows symlinks back to the real slug-named file -- silently caching under
the WRONG (slug) stem again. A hard link is its own directory entry, so
``.resolve()`` returns the hard link's own path.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.models.musx_redecode import _PROB_CACHE, frame_posteriors  # noqa: E402

AUDIO_DIR = REPO / "docs" / "audio"
YT_AUDIO_META = REPO / "docs" / "plots" / ".yt_audio_meta.json"
YT_VIDEO_IDS = REPO / "docs" / "plots" / ".yt_video_ids.json"
SYMLINK_DIR = REPO / "data" / "cache" / "_backfill_tmp"

MAX_SONGS = 15
MAX_SECONDS = 20 * 60
MIN_FREE_GB = 2.0


def free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / (1024 ** 3)


def main() -> None:
    g = free_gb(REPO)
    print(f"free disk: {g:.1f} GB")
    if g < MIN_FREE_GB:
        print(f"SKIP: free disk {g:.1f} GB < {MIN_FREE_GB} GB floor -- backfill not attempted.")
        return

    audio_meta = json.loads(YT_AUDIO_META.read_text())
    video_ids = json.loads(YT_VIDEO_IDS.read_text()) if YT_VIDEO_IDS.exists() else {}

    candidates = []
    for chart, meta in audio_meta.items():
        vid = video_ids.get(chart, "")
        if not vid:
            continue
        cache_f = _PROB_CACHE / f"{vid}.npz"
        if cache_f.exists():
            continue  # already cached
        audio_path = AUDIO_DIR / Path(meta["audio"]).name
        if not audio_path.exists():
            continue  # per brief: only charts whose audio resolves on disk
        candidates.append((chart, vid, audio_path))

    # de-dup by video_id (several chart variants can share one video_id, e.g.
    # "..._npattern.html" / "..._barlocked.html" siblings) -- backfill each
    # underlying song once.
    seen_vid = set()
    deduped = []
    for chart, vid, audio_path in candidates:
        if vid in seen_vid:
            continue
        seen_vid.add(vid)
        deduped.append((chart, vid, audio_path))

    print(f"{len(audio_meta)} app charts total, {len(candidates)} missing musx_probs "
         f"with resolvable on-disk audio, {len(deduped)} unique video_ids to backfill "
         f"(capped at {MAX_SONGS})")

    SYMLINK_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    results = []
    for i, (chart, vid, audio_path) in enumerate(deduped):
        if len(results) >= MAX_SONGS:
            print(f"STOP: reached {MAX_SONGS}-song cap.")
            break
        elapsed_total = time.time() - t_start
        if elapsed_total > MAX_SECONDS:
            print(f"STOP: reached {MAX_SECONDS/60:.0f}-minute time cap "
                 f"({elapsed_total:.0f}s elapsed).")
            break
        g = free_gb(REPO)
        if g < MIN_FREE_GB:
            print(f"STOP: free disk dropped to {g:.1f} GB < {MIN_FREE_GB} GB floor mid-run.")
            break

        link = SYMLINK_DIR / f"{vid}{audio_path.suffix}"
        if link.exists() or link.is_symlink():
            link.unlink()
        os.link(audio_path.resolve(), link)

        t0 = time.time()
        try:
            frame_posteriors(link)
            dt = time.time() - t0
            cache_f = _PROB_CACHE / f"{vid}.npz"
            size_mb = cache_f.stat().st_size / 1e6 if cache_f.exists() else 0.0
            print(f"  [{i+1}] {chart} (vid={vid}, audio={audio_path.name}): "
                 f"{dt:.1f}s, cache {size_mb:.1f} MB")
            results.append({"chart": chart, "video_id": vid, "audio": audio_path.name,
                           "seconds": round(dt, 1), "cache_mb": round(size_mb, 1),
                           "ok": True})
        except Exception as e:  # noqa: BLE001 -- one bad song must not kill the batch
            dt = time.time() - t0
            print(f"  [{i+1}] {chart} (vid={vid}): FAILED after {dt:.1f}s ({e})")
            results.append({"chart": chart, "video_id": vid, "audio": audio_path.name,
                           "seconds": round(dt, 1), "cache_mb": 0.0, "ok": False,
                           "error": str(e)})
        finally:
            if link.exists() or link.is_symlink():
                link.unlink()

    total_elapsed = time.time() - t_start
    n_ok = sum(1 for r in results if r["ok"])
    cache_total_mb = sum(f.stat().st_size for f in _PROB_CACHE.glob("*.npz")) / 1e6
    print(f"\nDone: {n_ok}/{len(results)} songs backfilled OK in {total_elapsed:.1f}s "
         f"total. musx_probs cache dir now {cache_total_mb:.0f} MB "
         f"({len(list(_PROB_CACHE.glob('*.npz')))} files).")

    out = {"results": results, "total_elapsed_s": round(total_elapsed, 1),
          "cache_dir_mb": round(cache_total_mb, 1),
          "n_files_total": len(list(_PROB_CACHE.glob("*.npz")))}
    (REPO / "data" / "cache" / "tune_lock_propagation" / "backfill_report.json").write_text(
        json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
