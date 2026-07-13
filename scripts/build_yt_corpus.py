"""build_yt_corpus.py — build a YouTube+iReal corpus for training.

Two modes:
  1. Pilot (default): a hardcoded list of songs for which we already know the
     YouTube video IDs.  Run this first to validate the pipeline end-to-end.

  2. Auto-search (--search): for each song in jazz1460.txt, search YouTube
     via yt-dlp and try to download + align.  Slow but scales to 1000+ songs.

Output:
  data/cache/yt_corpus/corpus.npz   — feature arrays for training

Usage:
    # Pilot run (known video IDs)
    .venv/bin/python scripts/build_yt_corpus.py --pilot

    # Auto-search first N songs from jazz1460
    .venv/bin/python scripts/build_yt_corpus.py --search --max-songs 50

    # Add more songs by specifying a JSON file
    .venv/bin/python scripts/build_yt_corpus.py --entries my_entries.json
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CACHE_DIR  = REPO / "data" / "cache" / "yt_corpus"
CORPUS_NPZ = CACHE_DIR / "corpus.npz"
JAZZ1460   = REPO / "data" / "ireal" / "jazz1460.txt"

# ── pilot song list ──────────────────────────────────────────────────────────
# Manually curated: video IDs for well-known jazz standards.
# Add more entries here as you identify clean recordings.
PILOT_ENTRIES = [
    # Standard/pop recordings — vocals + orchestra or piano trio, minimal extensions.
    # Prefer artists known for straight-ahead changes: Sinatra, Nat Cole, Ella+Louis,
    # Chet Baker (vocal), Diana Krall, Johnny Mathis, Tony Bennett, Duke Ellington.
    {"video_id": "YVedK1VUfLM", "title": "Autumn Leaves",                   "irealb_song": "Autumn Leaves"},
    # Nat King Cole, Capitol Records remastered — standard ballad, very clean.
    {"video_id": "zutQlPaQ6V4", "title": "Bye Bye Blackbird",               "irealb_song": "Bye Bye Blackbird"},
    # Diana Krall — Public Enemies soundtrack (2009) — piano trio, straight changes.
    {"video_id": "ASNSS_G62zw", "title": "All The Things You Are",          "irealb_song": "All The Things You Are"},
    # Ella Fitzgerald + Nelson Riddle Orchestra, Verve 1963 — standard arrangement.
    {"video_id": "ytHMBYLwgVU", "title": "There Will Never Be Another You", "irealb_song": "There Will Never Be Another You"},
    # Chet Baker (vocal version), "Chet Baker Sings" 1954, Universal — clean standard.
    {"video_id": "DkC9bCuahC8", "title": "Misty",                           "irealb_song": "Misty"},
    # Johnny Mathis — Official Audio, standard ballad arrangement.
    {"video_id": "ZEMCeymW1Ow", "title": "Fly Me To The Moon",              "irealb_song": "Fly Me To The Moon"},
    # Frank Sinatra + Count Basie Orchestra, Remastered — the canonical pop-jazz version.
    {"video_id": "2HJCN3upMHE", "title": "Summertime",                      "irealb_song": "Summertime"},
    # Ella Fitzgerald & Louis Armstrong — Porgy and Bess album, Verve 1958.
    {"video_id": "wTFPV1pk654", "title": "Satin Doll",                      "irealb_song": "Satin Doll"},
    # Duke Ellington and his Orchestra, 1962 official video.
    {"video_id": "AUTUYBJBiJ4", "title": "Body And Soul",                   "irealb_song": "Body And Soul"},
    # Tony Bennett + Amy Winehouse, acoustic duet — very clean, standard changes.
    {"video_id": "D6mFGy4g_n8", "title": "Take The A Train",                "irealb_song": "Take The A Train"},
    # Duke Ellington and his Orchestra, 1962 official video, 1.9M views.
]


# ── jazz1460 iReal URL lookup ─────────────────────────────────────────────────

def _load_jazz1460() -> str:
    if not JAZZ1460.exists():
        raise FileNotFoundError(f"jazz1460.txt not found at {JAZZ1460}")
    return JAZZ1460.read_text()


def find_irealb_url(song_name: str, corpus_raw: str | None = None) -> str | None:
    """Find the irealb:// URL for a song by title in the jazz1460 corpus.

    Uses the same approach as the session's compare_autumn_leaves script:
    scan the URL-encoded corpus for the title, then extract the entry.
    """
    import urllib.parse

    if corpus_raw is None:
        corpus_raw = _load_jazz1460()

    # URL-encode the song name to search in the encoded corpus
    encoded = urllib.parse.quote(song_name, safe="")
    # Try exact match first, then partial
    for query in [encoded, urllib.parse.quote(song_name[:8], safe="")]:
        pos = corpus_raw.find(query)
        if pos == -1:
            continue
        # find the song entry boundaries
        entry_start = corpus_raw.rfind("===", 0, pos)
        entry_start = (entry_start + 3) if entry_start != -1 else 0
        entry_end   = corpus_raw.find("===", pos)
        entry = corpus_raw[entry_start: entry_end if entry_end != -1 else None].rstrip("=")
        if entry:
            return "irealb://" + entry

    return None


# ── yt-dlp search ────────────────────────────────────────────────────────────

def search_youtube_id(title: str, n: int = 1) -> str | None:
    """Use yt-dlp to search YouTube and return the top video ID."""
    import shutil, sys
    ytdlp = shutil.which("yt-dlp") or str(Path(sys.executable).parent / "yt-dlp")
    query = f"ytsearch{n}:{title} jazz piano"
    cmd = [ytdlp, "--get-id", "--no-warnings", query]
    try:
        out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
        ids = out.strip().splitlines()
        return ids[0] if ids else None
    except Exception:
        return None


# ── auto-search entries from jazz1460 ─────────────────────────────────────────

def make_search_entries(max_songs: int = 50) -> list[dict]:
    """Build entries from the first max_songs titles in jazz1460 via yt-dlp search."""
    from pyRealParser import Tune
    import urllib.parse

    corpus_raw = _load_jazz1460()
    # Split on === to get individual irealb URL fragments
    parts = corpus_raw.split("===")
    entries = []
    for part in parts:
        part = part.strip().lstrip("=")
        if not part:
            continue
        url = "irealb://" + part
        try:
            tunes = Tune.parse_ireal_url(urllib.parse.unquote(url))
        except Exception:
            continue
        if not tunes:
            continue
        try:
            tune = tunes[0]
        except (IndexError, Exception):
            continue
        title = tune.title or ""
        if not title:
            continue

        vid = search_youtube_id(f"{title} {tune.composer or ''}")
        if not vid:
            print(f"  [skip] no YouTube video found for {title!r}", flush=True)
            continue

        entries.append({"video_id": vid, "title": title, "irealb_url": url})
        print(f"  [found] {title!r} → {vid}", flush=True)

        if len(entries) >= max_songs:
            break

    return entries


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--pilot",   action="store_true",
                     help="Use hardcoded pilot video list")
    grp.add_argument("--search",  action="store_true",
                     help="Search YouTube automatically for jazz1460 songs")
    grp.add_argument("--entries", type=Path,
                     help="JSON file with [{video_id, title, irealb_url}] entries")
    ap.add_argument("--max-songs",   default=50, type=int,
                    help="(--search mode) max songs to try")
    ap.add_argument("--skip-mismatches", action="store_true")
    ap.add_argument("--out",  default=CORPUS_NPZ, type=Path,
                    help="Output .npz path")
    ap.add_argument("--force-download", action="store_true")
    args = ap.parse_args()

    from harmonia.data.yt_chord_corpus import build_corpus, pack_arrays

    if args.entries:
        entries = json.loads(args.entries.read_text())
    elif args.search:
        print("Auto-searching YouTube for jazz1460 songs...", flush=True)
        entries = make_search_entries(args.max_songs)
    else:
        # pilot mode: resolve iReal URLs from jazz1460
        print("Building pilot entry list...")
        corpus_raw = _load_jazz1460()
        entries = []
        for e in PILOT_ENTRIES:
            url = find_irealb_url(e["irealb_song"], corpus_raw)
            if url is None:
                print(f"  [skip] {e['title']!r} not found in jazz1460")
                continue
            entries.append({
                "video_id":  e["video_id"],
                "title":     e["title"],
                "irealb_url": url,
            })
            print(f"  [queued] {e['title']!r} → {e['video_id']}")

    print(f"\nBuilding corpus from {len(entries)} entries → {args.out}")
    records = build_corpus(
        entries, CACHE_DIR,
        skip_mismatches=args.skip_mismatches,
        force_download=args.force_download,
    )

    if not records:
        print("ERROR: no records collected.")
        sys.exit(1)

    arrays = pack_arrays(records)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, **arrays)
    print(f"\nCorpus saved: {len(records)} records → {args.out}")

    # quick summary
    from collections import Counter
    from harmonia.data.yt_chord_corpus import QUALITIES
    qcounts = Counter(int(x) for x in arrays["quality_idx"])
    print("Quality distribution:")
    for qi, q in enumerate(QUALITIES):
        n = qcounts.get(qi, 0)
        print(f"  {q:6s}  {n:5d}  ({100*n/len(records):.1f}%)")


if __name__ == "__main__":
    main()
