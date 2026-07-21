"""Regression tests for the iRealb -> Harte chord-token parser.

2026-07-21 user report on a real iReal import: "Bb6 devient B" (sixth chords
lose their extension) and "les slash chords se perdent" (slash-bass chords
lose their bass note). Confirmed pre-fix: _parse_ireal_chord_token("Bb6")
returned (10, "maj") and _parse_ireal_chord_token("Bb6/D") returned the same
(10, "maj") — bass silently dropped, quality collapsed to a bare triad.
"""
import socket

import pytest

from harmonia.irealb_fetcher import (
    _parse_ireal_chord_token,
    _search_forum,
    _search_main_playlists,
    search_community,
)
from scripts.render_youtube_chart import label_to_ireal


def _online() -> bool:
    try:
        socket.create_connection(("forums.irealpro.com", 443), timeout=5).close()
        return True
    except OSError:
        return False


needs_net = pytest.mark.skipif(not _online(), reason="needs network to irealpro.com")


def test_bare_sixth_keeps_extension():
    assert _parse_ireal_chord_token("Bb6") == (10, "6", None)


def test_minor_sixth_keeps_extension():
    assert _parse_ireal_chord_token("C-6") == (0, "min6", None)


def test_six_nine_does_not_get_misread_as_slash_bass():
    # "9" is not a valid note letter -> the "/" here belongs to the quality
    # token itself, not a slash-bass split.
    pc, sev, bass = _parse_ireal_chord_token("F6/9")
    assert bass is None
    assert pc == 5


def test_slash_chord_keeps_bass():
    assert _parse_ireal_chord_token("Bb6/D") == (10, "6", 2)
    assert _parse_ireal_chord_token("G7/B") == (7, "7", 11)


def test_no_chord_token_still_returns_none():
    assert _parse_ireal_chord_token("NC") is None


def test_label_to_ireal_round_trips_bass():
    assert label_to_ireal("A#:6/D", "exact") == "A#6/D"
    assert label_to_ireal("G:7/B", "exact") == "G7/B"
    # family-level collapse still keeps the bass -- it's a real sounding
    # note, not an extension to hide.
    assert label_to_ireal("G:7/B", "family") == "G/B"


def test_label_to_ireal_sixth_chords():
    assert label_to_ireal("A#:6", "exact") == "A#6"
    assert label_to_ireal("A#:min6", "exact") == "A#-6"


# ── Community-search completeness (2026-07-21) ────────────────────────────────
# Red-first against the completeness bug: before the forum supplement, a search
# for a famous NON-standard song/artist returned an empty list because
# /main-playlists/ only holds ~2200 jazz standards. These assert the fix —
# real, importable results now come back for such queries.

@needs_net
def test_main_playlists_alone_misses_non_standards():
    # Characterizes the ROOT cause: the standards corpus genuinely lacks these
    # (not a filter bug). This is what used to make search_community empty.
    assert _search_main_playlists("feeling good", 8) == []
    assert _search_main_playlists("nina simone", 8) == []


@needs_net
def test_forum_supplies_famous_non_standards():
    results = search_community("feeling good", max_results=8)
    titles = [r["title"].lower() for r in results]
    assert results, "search_community returned nothing for 'feeling good'"
    assert any("feeling good" in t for t in titles)
    assert all("irealb://" in r["irealb_url"] for r in results)


@needs_net
def test_forum_supplies_artist_query():
    results = search_community("nina simone", max_results=8)
    assert results, "search_community returned nothing for 'nina simone'"
    # Every result must be a real single-tune importable URL.
    assert all(r["irealb_url"].startswith("irealb://") for r in results)


@needs_net
def test_standards_still_found_directly():
    results = search_community("autumn leaves", max_results=8)
    assert any("autumn leaves" in r["title"].lower() for r in results)
    assert any(r["source"] == "standards" for r in results)


@needs_net
def test_forum_search_direct_returns_threads():
    out = _search_forum("nina simone", 8)
    assert out and all(o["source"] == "forum" for o in out)
