"""
Ultimate Guitar tab fetcher.

Searches for a song by title + artist, returns the top results sorted by
(rating * log(votes+1)) — a score that balances quality and popularity —
and fetches the chord sequence from the highest-ranked tab.

Requires curl_cffi (pip install curl_cffi) to bypass Cloudflare.
"""

from __future__ import annotations

import html as htmlmod
import json
import logging
import math
import re
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)

_UG_SEARCH = "https://www.ultimate-guitar.com/search.php"
_HEADERS = {
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}


def _scraper():
    try:
        from curl_cffi import requests as cffi_requests
        return cffi_requests
    except ImportError:
        raise ImportError(
            "curl_cffi is required for tab fetching.\n"
            "Install with:  pip install curl_cffi"
        )


@dataclass
class TabResult:
    id: int
    song_name: str
    artist_name: str
    tab_type: str          # "Chords", "Tab", "Guitar Pro"
    rating: float
    votes: int
    tonality: str          # e.g. "Bm"
    difficulty: str
    tab_url: str
    score: float           # ranking score = rating * log2(votes+2)


@dataclass
class TabChords:
    result: TabResult
    raw_content: str       # raw [ch]...[/ch] markup
    chords: list[str]      # ordered unique chord sequence (deduplicated)
    chord_occurrences: dict[str, int]  # chord → count in the tab


def _ug_score(rating: float, votes: int) -> float:
    """Rank by a Wilson-score-inspired heuristic: rating × log2(votes+2)."""
    return rating * math.log2(votes + 2)


def search_tabs(
    title: str,
    artist: str = "",
    tab_types: tuple[str, ...] = ("Chords", "Guitar Pro"),
    max_results: int = 10,
) -> list[TabResult]:
    """Search Ultimate Guitar and return up to max_results tabs, best-first.

    Filters to tab_types and sorts by _ug_score (rating × log votes).
    Returns [] on failure (logs the error).
    """
    cffi = _scraper()
    query = f"{title} {artist}".strip()
    logger.info("tab_fetcher: searching UG for %r", query)

    try:
        r = cffi.get(
            _UG_SEARCH,
            params={"search_type": "title", "value": query},
            headers=_HEADERS,
            impersonate="chrome124",
            timeout=20,
        )
        r.raise_for_status()
    except Exception as e:
        logger.warning("tab_fetcher: search request failed: %s", e)
        return []

    m = re.search(r'data-content="([^"]+)"', r.text)
    if not m:
        logger.warning("tab_fetcher: no data-content in UG search page")
        return []

    try:
        data = json.loads(htmlmod.unescape(m.group(1)))
        raw = data["store"]["page"]["data"]["results"]
    except (KeyError, json.JSONDecodeError) as e:
        logger.warning("tab_fetcher: failed to parse UG data: %s", e)
        return []

    results = []
    for item in raw:
        t = item.get("type", "")
        if t not in tab_types:
            continue
        rating = float(item.get("rating") or 0)
        votes  = int(item.get("votes") or 0)
        results.append(TabResult(
            id=item.get("id", 0),
            song_name=item.get("song_name", ""),
            artist_name=item.get("artist_name", ""),
            tab_type=t,
            rating=rating,
            votes=votes,
            tonality=item.get("tonality_name", ""),
            difficulty=item.get("difficulty", ""),
            tab_url=item.get("tab_url", ""),
            score=_ug_score(rating, votes),
        ))

    results.sort(key=lambda r: r.score, reverse=True)
    logger.info("tab_fetcher: found %d matching tabs", len(results))
    return results[:max_results]


def fetch_tab_chords(result: TabResult) -> Optional[TabChords]:
    """Fetch the chord content from a TabResult. Returns None on failure."""
    if not result.tab_url:
        return None

    cffi = _scraper()
    logger.info("tab_fetcher: fetching %s", result.tab_url)

    try:
        r = cffi.get(
            result.tab_url,
            headers=_HEADERS,
            impersonate="chrome124",
            timeout=20,
        )
        r.raise_for_status()
    except Exception as e:
        logger.warning("tab_fetcher: fetch failed: %s", e)
        return None

    m = re.search(r'data-content="([^"]+)"', r.text)
    if not m:
        return None

    try:
        data = json.loads(htmlmod.unescape(m.group(1)))
        content = data["store"]["page"]["data"]["tab_view"]["wiki_tab"]["content"]
    except (KeyError, json.JSONDecodeError):
        return None

    # Extract all [ch]...[/ch] tokens in order
    raw_chords = re.findall(r'\[ch\](.*?)\[/ch\]', content)

    # Deduplicate while preserving first-occurrence order
    seen: set[str] = set()
    unique: list[str] = []
    for ch in raw_chords:
        ch = ch.strip()
        if ch and ch not in seen:
            seen.add(ch)
            unique.append(ch)

    counts: dict[str, int] = {}
    for ch in raw_chords:
        ch = ch.strip()
        if ch:
            counts[ch] = counts.get(ch, 0) + 1

    return TabChords(
        result=result,
        raw_content=content,
        chords=unique,
        chord_occurrences=counts,
    )


def fetch_best_tab(title: str, artist: str = "") -> Optional[TabChords]:
    """Convenience: search + fetch the highest-ranked tab. Returns None on failure."""
    results = search_tabs(title, artist)
    if not results:
        return None
    return fetch_tab_chords(results[0])
