"""Chercher un morceau dans la communauté iReal Pro.

Un seul point d'entrée : `search_community(query)` — les ~2 200 standards de
`/main-playlists/` d'abord, le forum ensuite. Renvoie des dicts
{title, composer, key, style, time_sig, irealb_url, source}.

Porté depuis `harmonia/irealb_fetcher.py` (legacy) le 2026-09-14, sans sa
seconde moitié : la conversion irealb:// → ChordChart et le rendu HTML
servaient l'ancienne app (:7771) et son import iReal, que l'app n'a pas.
"""

from __future__ import annotations

import http.cookiejar
import json
import re
import time
import urllib.parse
import urllib.request

from pyRealParser import Tune


_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Harmonia/1.0"


# ── Community search ──────────────────────────────────────────────────────────

_MAIN_PLAYLISTS_URL = "https://www.irealpro.com/main-playlists/"
_FORUM_BASE = "https://forums.irealpro.com"


def _iter_tunes_in_blob(raw_blob: str):
    """Yield ``(segment, Tune)`` for each song in one ``irealb://`` blob.

    A blob is a whole playlist: MANY songs joined by ``===``. ``segment`` is
    the raw slice for a SINGLE song, re-encodable into a single-tune URL (so
    importing song #7 of a playlist renders song #7, not tunes[0]).
    """
    decoded = urllib.parse.unquote(raw_blob)
    m = re.match(r"irealb://(.+)", decoded)
    if not m:
        return
    for seg in re.split("===", m.group(1)):
        if not seg:
            continue
        try:
            yield seg, Tune(seg)
        except Exception:
            continue


def _result_dict(seg: str, tune: Tune, source: str) -> dict:
    ts = tune.time_signature or (4, 4)
    return {
        "title":      tune.title,
        "composer":   tune.composer or "",
        "key":        tune.key or "",
        "style":      tune.style or "",
        "time_sig":   f"{ts[0]}/{ts[1]}",
        # SINGLE-tune URL (re-encoded from just this song's segment) — NOT the
        # multi-song playlist blob; render/import always take tunes[0].
        "irealb_url": "irealb://" + urllib.parse.quote(seg, safe="="),
        "source":     source,
    }


def _search_main_playlists(query: str, max_results: int) -> list[dict]:
    """The ~2200 jazz-standard corpus on ``/main-playlists/`` (fast, one page).

    irealpro.com restructured after this module was first written — the old
    ``/music/?s=<query>`` search endpoint 404s now (confirmed 2026-07-20).
    The standards content moved to ``/main-playlists/``: 6 big playlist blobs
    that decode to ~2236 tunes total (jazz standards, indexed by COMPOSER in
    "Last First" order — NOT performer). No server-side filter param remains,
    so fetch once and filter locally by word-overlap on title+composer.
    """
    req = urllib.request.Request(_MAIN_PLAYLISTS_URL, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        raise RuntimeError(f"iReal main-playlists search failed: {exc}") from exc

    html = re.sub(r"&#0*38;|&amp;", "&", html)
    raw_blobs = re.findall(r"irealb://[^\s\"'<>]+", html)

    # Word-overlap match (2026-07-20): ALL query words appearing somewhere in
    # title+composer — forgiving of an artist name pasted after the title,
    # still precise at ~2200 candidates.
    q_words = [w for w in re.split(r"\s+", query.strip().lower()) if w]
    results: list[dict] = []
    seen: set[str] = set()
    for raw_blob in raw_blobs:
        for seg, tune in _iter_tunes_in_blob(raw_blob):
            if len(results) >= max_results:
                return results
            haystack = f"{tune.title} {tune.composer or ''}".lower()
            if tune.title in seen or (q_words and not all(w in haystack for w in q_words)):
                continue
            seen.add(tune.title)
            results.append(_result_dict(seg, tune, "standards"))
    return results


def _forum_opener() -> urllib.request.OpenerDirector:
    cj = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    op.addheaders = [("User-Agent", _UA)]
    return op


def _search_forum(query: str, max_results: int, *,
                  max_threads: int = 6, pace_s: float = 0.35) -> list[dict]:
    """Search the iReal Pro XenForo community forum (forums.irealpro.com).

    This is where the REAL breadth lives. ``/main-playlists/`` is only the
    ~2200 jazz standards; the forum's genre subforums (jazz, pop-rock-blues,
    brazilian-latin, country-folk, holiday/film/worship, …) hold thousands of
    user-submitted charts for specific songs/artists that are NOT standards —
    e.g. Nina Simone, "Feeling Good" (Newley/Bricusse), Bob Dylan. Those
    return zero from main-playlists no matter how good the local filter is,
    because the tune simply isn't in that corpus (confirmed 2026-07-21).

    Guests can search without logging in via XenForo's stored-search flow:
      1. GET the search form for the ``_xfToken`` CSRF token,
      2. POST ``keywords`` to ``/search/search`` (302 → a results page),
      3. fetch the top matching THREADS and pull their ``irealb://`` blobs.
    The forum's own relevance ranking does the hard part — a thread for a
    Nina Simone cover often has no "Nina Simone" in the tune's composer field
    (that's the songwriter), so we trust the search over a re-applied title
    filter for focused (single-song) threads and only word-filter the big
    multi-song playlist threads.

    Paced: at most ``max_threads`` thread fetches, ``pace_s`` apart — a
    polite read-only scraper, same posture as the rest of this module. This
    is a SUPPLEMENT; callers should let its failure fall back to standards
    rather than blank the whole search.
    """
    op = _forum_opener()
    try:
        form = op.open(_FORUM_BASE + "/search/?type=post", timeout=20).read().decode("utf-8", "replace")
    except Exception as exc:
        raise RuntimeError(f"iReal forum search failed (form fetch): {exc}") from exc
    m = re.search(r'name="_xfToken"\s+value="([^"]*)"', form)
    token = m.group(1) if m else ""

    data = urllib.parse.urlencode({
        "keywords": query, "order": "relevance",
        "_xfToken": token, "search_type": "post",
    }).encode()
    req = urllib.request.Request(_FORUM_BASE + "/search/search", data=data,
                                 headers={"User-Agent": _UA,
                                          "Referer": _FORUM_BASE + "/search/"})
    try:
        page = op.open(req, timeout=20).read().decode("utf-8", "replace")
    except Exception as exc:
        raise RuntimeError(f"iReal forum search failed (post): {exc}") from exc

    # thread (slug, id) in relevance order, de-duped by id
    threads: list[tuple[str, str]] = []
    seen_ids: set[str] = set()
    for slug, tid in re.findall(r'/threads/([a-z0-9\-]+)\.(\d+)', page):
        if tid in seen_ids:
            continue
        seen_ids.add(tid)
        threads.append((slug, tid))

    q_words = [w for w in re.split(r"\s+", query.strip().lower()) if w]
    sig_words = [w for w in q_words if len(w) >= 4]  # drop "in"/"the" noise
    out: list[dict] = []
    seen_titles: set[str] = set()
    for i, (slug, tid) in enumerate(threads[:max_threads]):
        if len(out) >= max_results:
            break
        if i:
            time.sleep(pace_s)
        try:
            th = op.open(f"{_FORUM_BASE}/threads/{slug}.{tid}/", timeout=20).read().decode("utf-8", "replace")
        except Exception:
            continue
        th = re.sub(r"&#0*38;|&amp;", "&", th)
        tunes = [pair for b in re.findall(r"irealb://[^\s\"'<>]+", th)
                 for pair in _iter_tunes_in_blob(b)]
        big = len(tunes) > 8  # a shared multi-song playlist, not a single song
        for seg, tune in tunes:
            if len(out) >= max_results:
                break
            title = (tune.title or "").strip()
            if not title or title.lower() in seen_titles:
                continue
            if big and sig_words:
                hay = f"{title} {tune.composer or ''}".lower()
                if not any(w in hay for w in sig_words):
                    continue
            seen_titles.add(title.lower())
            out.append(_result_dict(seg, tune, "forum"))
    return out


def search_community(query: str, max_results: int = 8) -> list[dict]:
    """Search the iReal Pro community for songs matching ``query``.

    Two sources, standards first then forum as a supplement:
      * ``_search_main_playlists`` — the ~2200 jazz standards on
        ``/main-playlists/`` (one fast page fetch).
      * ``_search_forum`` — forums.irealpro.com, thousands of user-submitted
        charts for non-standard songs/artists across every genre subforum.

    Merged and de-duped by title. Returns dicts with the shape the UI expects
    plus a ``source`` key ("standards" | "forum") for transparency:
    {title, composer, key, style, time_sig, irealb_url, source}.

    Why two sources (2026-07-21): main-playlists is COMPLETE only for jazz
    standards — a search for "Nina Simone" or "feeling good" returns 0 there
    no matter the filter, because the songs aren't in that corpus. Almost all
    of iReal Pro's actual community breadth lives on the forum, reachable by
    guest search. This does NOT cover 100% of iReal charts in existence
    (there is no single public index of all of them, and forum coverage is
    whatever users have posted), but it takes common non-standard queries
    from "0 results" to real, importable charts. See docs/known_issues.md.
    """
    try:
        results = _search_main_playlists(query, max_results)
    except Exception:
        results = []  # forum below may still save the search

    if len(results) < max_results:
        seen = {r["title"].lower() for r in results}
        try:
            for r in _search_forum(query, max_results - len(results)):
                if r["title"].lower() in seen:
                    continue
                seen.add(r["title"].lower())
                results.append(r)
                if len(results) >= max_results:
                    break
        except Exception:
            # Forum is a supplement — never let its failure blank out the
            # standards results we already have.
            pass

    return results
