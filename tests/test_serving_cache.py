"""Pin the behaviour of harmonia.serving.cache (Stage 6a port).

The gate for this stage is BYTE-IDENTICAL behaviour vs the inline regex forms
that lived in scripts/harmonia_server.py. The parity tests below re-express each
literal inline expression and assert the extracted function equals it over a
representative corpus. The collision tests pin the invariant that motivated
extracting analysis_stem (docs/known_issues.md: the audio.wav / input.wav
stem-collision bugs).
"""

from __future__ import annotations

import re

import pytest

from harmonia.serving.cache import analysis_stem, chart_slug, lookup_slug


# Representative corpus: unicode/accents, spaces, punctuation, apostrophes,
# CJK, already-slugged, all-caps, all-punctuation, and titles > 60 chars.
CORPUS = [
    "Autumn Leaves",
    "I Can't Help It",
    "I'm a Fool to Want You",
    "Café del Mar",
    "Naïve Café — Résumé",
    "Blue Bossa (150bpm) Backing Track",
    "Michael Jackson - Billie Jean (Official Video)",
    "señor_blues",
    "already_slugged_title",
    "ALL CAPS TITLE",
    "  leading and trailing spaces  ",
    "___underscores___",
    "!!!",
    "track#3 @ 120bpm!",
    "日本語のタイトル",
    "A" * 80,
    "Someone Like You (Adele) - a very very very long descriptive youtube "
    "title that far exceeds sixty characters in length",
    "",
]


# ---------------------------------------------------------------------------
# (d) byte-identical parity vs the literal inline regex expressions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("song", CORPUS)
def test_lookup_slug_byte_identical(song):
    # Literal inline form from the server's "lookup / sanitize" call sites.
    expected = re.sub(r"[^A-Za-z0-9_]", "", song)
    assert lookup_slug(song) == expected


@pytest.mark.parametrize("title", CORPUS)
def test_chart_slug_byte_identical(title):
    # Literal inline form from the server's "create / slugify" call sites.
    expected = re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:60]
    assert chart_slug(title) == expected


def test_chart_slug_truncates_to_60():
    long = "A" * 200
    assert len(chart_slug(long)) == 60
    assert chart_slug(long) == "a" * 60


# ---------------------------------------------------------------------------
# (a) chart_slug output is idempotent under lookup_slug
#     (chart_slug emits only [a-z0-9_], which lookup_slug preserves verbatim)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("title", CORPUS)
def test_chart_slug_idempotent_under_lookup_slug(title):
    s = chart_slug(title)
    assert lookup_slug(s) == s


@pytest.mark.parametrize("song", CORPUS)
def test_lookup_slug_is_a_fixed_point_of_itself(song):
    once = lookup_slug(song)
    assert lookup_slug(once) == once


# ---------------------------------------------------------------------------
# (b) two distinct upload job_ids -> distinct stems (no-collision invariant)
# ---------------------------------------------------------------------------

def test_distinct_job_ids_give_distinct_stems():
    j1 = "1721600000000"
    j2 = "1721600000001"
    assert analysis_stem("upload", job_id=j1) != analysis_stem("upload", job_id=j2)


def test_many_distinct_job_ids_never_collide():
    job_ids = [str(1721600000000 + i) for i in range(50)]
    stems = {analysis_stem("upload", job_id=j) for j in job_ids}
    assert len(stems) == len(job_ids)


# ---------------------------------------------------------------------------
# (c) same video_id -> same stem (cache-reuse invariant)
# ---------------------------------------------------------------------------

def test_same_video_id_gives_same_stem():
    vid = "dQw4w9WgXcQ"
    assert analysis_stem("youtube", video_id=vid) == analysis_stem("youtube", video_id=vid)
    assert analysis_stem("youtube", video_id=vid) == vid


def test_youtube_returns_video_id_even_if_job_id_present():
    # A YouTube source is keyed by its stable video_id, never a per-run job_id.
    assert analysis_stem("youtube", video_id="abc12345678", job_id="9999") == "abc12345678"


# ---------------------------------------------------------------------------
# (e) analysis_stem raises rather than returning a constant when ids missing
# ---------------------------------------------------------------------------

def test_raises_when_neither_id_given():
    with pytest.raises(ValueError):
        analysis_stem("youtube")
    with pytest.raises(ValueError):
        analysis_stem("upload")
    with pytest.raises(ValueError):
        analysis_stem("something-unrecognised")


def test_youtube_source_requires_video_id():
    with pytest.raises(ValueError):
        analysis_stem("youtube", job_id="1721600000000")


def test_upload_source_requires_job_id():
    with pytest.raises(ValueError):
        analysis_stem("upload", video_id="dQw4w9WgXcQ")


def test_never_returns_empty_or_constant_stem():
    # Whatever it returns, it is the caller-supplied id, never a literal.
    assert analysis_stem("youtube", video_id="v") == "v"
    assert analysis_stem("upload", job_id="j") == "j"
