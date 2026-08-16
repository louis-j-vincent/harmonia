"""Ingest a Spotify playlist's audio via the existing dataset lane.

Two-step flow (Spotify side is done by spotdl, download side by harmonia):

  1. ``spotdl save <playlist_url> --save-file playlist.spotdl --preload``
     → per-track metadata (artist/title/album) + matched YouTube URL.
  2. ``python scripts/ingest_spotify_playlist.py playlist.spotdl``
     → downloads each match through
     :class:`harmonia.dataset.ingest.YouTubeFetcher` (reuses ``docs/audio/``
     and ``data/dataset_audio/`` caches, files keyed by video id) and appends
     one JSON line per track to ``data/dataset_audio/manifest_spotify.jsonl``.

The manifest is the annotation layer: artist / title / album / year / ISRC /
Spotify ids + the video id and local file. Idempotent: tracks whose video id
is already in the manifest are skipped, so re-running after failures is safe.

A track whose YouTube duration differs from Spotify's by more than
``--duration-tol`` seconds is still downloaded but flagged
``duration_mismatch`` — those matches deserve an ear-check before use.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harmonia.dataset.ingest import DATASET_AUDIO, YouTubeFetcher, slugify, youtube_id  # noqa: E402

DEFAULT_MANIFEST = DATASET_AUDIO / "manifest_spotify.jsonl"


def _clean_name(s: str) -> str:
    s = re.sub(r"[/:\\\x00-\x1f]", "-", s).strip().rstrip(".")
    return s[:140] or "untitled"


def finalize_file(path: Path, t: dict, vid: str | None) -> Path:
    """Rename a fresh dataset download to '<Title>.m4a' (collision ->
    '<Title> (<Artist>).m4a', then '[vid]') and embed MP4 tags + album cover.

    Files reused from other dirs (docs/audio) are untouched. Note: title-based
    names no longer carry the video id, so re-download protection comes from
    the manifest (spotify_id/video_id skip), not YouTubeFetcher's cache scan.
    """
    if path.parent != DATASET_AUDIO:
        return path
    title, artist = t.get("name") or path.stem, t.get("artist") or ""
    if vid and path.stem == vid:
        base = _clean_name(title)
        for cand in (base, f"{base} ({_clean_name(artist)})" if artist else f"{base} 2",
                     f"{base} [{vid}]"):
            new = path.with_name(f"{cand}{path.suffix}")
            if not new.exists():
                path = path.rename(new) or new
                break
    if path.suffix.lower() == ".m4a":
        try:
            from mutagen.mp4 import MP4, MP4Cover
            m = MP4(path)
            for key, val in (("\xa9nam", title), ("\xa9ART", artist),
                             ("\xa9alb", t.get("album_name")), ("aART", t.get("album_artist")),
                             ("\xa9day", str(t["year"]) if t.get("year") else None)):
                if val:
                    m[key] = [val]
            if t.get("cover_url"):
                try:
                    import urllib.request
                    with urllib.request.urlopen(t["cover_url"], timeout=30) as r:
                        m["covr"] = [MP4Cover(r.read(), imageformat=MP4Cover.FORMAT_JPEG)]
                except OSError:
                    pass
            m.save()
        except Exception as e:  # noqa: BLE001 — tags are best-effort
            print(f"  (tagging failed: {e})")
    return path


def probe_duration(path: Path) -> float | None:
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(path)],
            capture_output=True, text=True, timeout=30,
        ).stdout.strip()
        return float(out) if out else None
    except (OSError, ValueError, subprocess.TimeoutExpired):
        return None


def load_done(manifest: Path) -> tuple[set[str], set[str]]:
    """(video ids, spotify ids) already recorded — both used to skip work."""
    vids: set[str] = set()
    sids: set[str] = set()
    if manifest.exists():
        for line in manifest.read_text().splitlines():
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("video_id"):
                vids.add(row["video_id"])
            if row.get("spotify_id"):
                sids.add(row["spotify_id"])
    return vids, sids


# Title words that usually mean "not the album take" — penalized unless the
# Spotify title itself contains them.
_VARIANT_WORDS = ("1 hour", "sped up", "slowed", "reverb", "nightcore", "8d",
                  "live", "remix", "cover", "karaoke", "instrumental", "loop",
                  "reaction", "extended")


def resolve_youtube(t: dict, n: int = 3) -> list[str]:
    """Match one Spotify track to YouTube candidates with yt-dlp search:
    ranked by duration gap to Spotify's, penalizing variant uploads."""
    import yt_dlp
    artist, title = t.get("artist") or "", t.get("name") or ""
    target = float(t.get("duration") or 0)
    spotify_title = f"{artist} {title}".lower()
    opts = {"quiet": True, "no_warnings": True, "extract_flat": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"ytsearch6:{artist} {title}", download=False)
        entries = [e for e in (info.get("entries") or []) if e and e.get("id")]
    except Exception:  # noqa: BLE001 — treat any search failure as no match
        return []

    def score(e: dict) -> float:
        dur = float(e.get("duration") or 0)
        gap = abs(dur - target) if (dur and target) else 60.0
        name = (e.get("title") or "").lower()
        penalty = sum(30.0 for w in _VARIANT_WORDS
                      if w in name and w not in spotify_title)
        return gap + penalty

    entries.sort(key=score)
    return [f"https://www.youtube.com/watch?v={e['id']}" for e in entries[:n]]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("spotdl_file", type=Path, help=".spotdl JSON from `spotdl save --preload`")
    ap.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    ap.add_argument("--playlist", default=None, help="playlist name/url recorded in each row")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--duration-tol", type=float, default=10.0)
    ap.add_argument("--sleep", type=float, default=1.0,
                    help="pause between downloads (politeness, seconds)")
    ap.add_argument("--min-free-gb", type=float, default=5.0,
                    help="stop cleanly if free disk drops below this")
    ap.add_argument("--resolve", action="store_true",
                    help="resolve missing YouTube matches via yt-dlp search")
    ap.add_argument("--backoff", type=float, default=120.0,
                    help="pause after 3 consecutive search misses (rate-limit ride-out)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    tracks = json.loads(args.spotdl_file.read_text())
    done, done_sids = load_done(args.manifest)
    fetcher = YouTubeFetcher()
    ok = skipped = failed = flagged = misses = 0

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.manifest.open("a") as mf:
        for t in tracks[: args.limit]:
            if not t:  # spotdl writes null for tracks whose preload failed
                failed += 1
                continue
            free_gb = shutil.disk_usage(args.manifest.parent).free / 1e9
            if free_gb < args.min_free_gb:
                print(f"STOP: only {free_gb:.1f} GB free (< {args.min_free_gb}) — "
                      f"resume with the same command after freeing space")
                break
            artist, title, album = t.get("artist"), t.get("name"), t.get("album_name")
            label = f"{artist} - {title}"
            if t.get("song_id") and t["song_id"] in done_sids:
                skipped += 1
                continue
            yt = t.get("download_url")
            cands = [yt] if yt else []
            if not cands and args.resolve and not args.dry_run:
                cands = resolve_youtube(t)
                if not cands and misses >= 2 and args.backoff > 0:
                    print(f"  ({misses + 1} consecutive search misses — "
                          f"pausing {args.backoff:.0f}s)")
                    time.sleep(args.backoff)
                    cands = resolve_youtube(t)
                misses = 0 if cands else misses + 1
            if not cands:
                print(f"SKIP (no YouTube match): {label}")
                failed += 1
                continue
            vid = youtube_id(cands[0])
            if vid and vid in done:
                skipped += 1
                continue
            if args.dry_run:
                print(f"would fetch {label}  <- {cands[0]}")
                continue
            path = None
            for yt in cands:  # best candidate first; fall through on dead videos
                vid = youtube_id(yt)
                try:
                    path = fetcher.fetch(yt, prefer_stem=slugify(f"{artist} {title}"))
                    break
                except Exception as e:  # noqa: BLE001
                    print(f"  candidate {vid} failed: {str(e).splitlines()[-1][:110]}")
            if path is None:
                print(f"FAIL: {label}: all candidates failed")
                failed += 1
                time.sleep(args.sleep)
                continue
            path = finalize_file(path, t, vid)
            time.sleep(args.sleep)
            dur = probe_duration(path)
            spotify_dur = t.get("duration")
            mismatch = (
                dur is not None and spotify_dur
                and abs(dur - float(spotify_dur)) > args.duration_tol
            )
            row = {
                "artist": artist,
                "title": title,
                "album": album,
                "album_artist": t.get("album_artist"),
                "year": t.get("year"),
                "isrc": t.get("isrc"),
                "spotify_id": t.get("song_id"),
                "spotify_url": t.get("url"),
                "youtube_url": yt,
                "video_id": vid,
                "file": str(path.relative_to(REPO) if path.is_relative_to(REPO) else path),
                "duration_s": round(dur, 2) if dur is not None else None,
                "spotify_duration_s": spotify_dur,
                "duration_mismatch": bool(mismatch),
                "playlist": args.playlist,
            }
            mf.write(json.dumps(row, ensure_ascii=False) + "\n")
            mf.flush()
            if vid:
                done.add(vid)
            ok += 1
            if mismatch:
                flagged += 1
                print(f"OK (duration mismatch {dur:.0f}s vs {spotify_dur}s): {label}")
            else:
                print(f"OK: {label}")

    print(f"\n{ok} downloaded, {skipped} already present, {failed} failed, "
          f"{flagged} flagged duration_mismatch -> {args.manifest}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
