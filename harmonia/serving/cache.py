"""Pure slug / cache-stem derivations for the Harmonia serving layer.

Extracted from ``scripts/harmonia_server.py``, where the same three rules were
inlined dozens of times. These are the single source of truth for how a song
name / chart title / analysis input maps to an on-disk cache key or filename
stem. They are pure (no I/O, no globals) and unit-tested, so the mapping can be
reasoned about and tested in one place.

Two of the three encode a *cache-collision* invariant that has bitten this
project twice — see ``docs/known_issues.md``:

  * "production-audit ... self-inflicted stem-keyed cache collision gave a fake
    root_acc=0.094" — every re-downloaded song reused the literal stem
    ``audio.wav``, so every song after the first was scored against the FIRST
    song's cached NNLS/musx features.
  * "/api/record-analyze cache collision truncated a re-analysis to ~45s" —
    every upload was written to a literal ``input.wav``; the second upload
    collided with the first's cache entry.

The NNLS/musx feature caches key on the audio file's *stem alone*
(``data/cache/nnls_infer/<stem>.npz``,
``data/cache/musx_infer/<stem>_submission.lab``) by deliberate design; a
*constant* stem therefore silently serves one input's features for another.
``analysis_stem`` exists to make that impossible to express by accident.
"""

from __future__ import annotations

import re


def lookup_slug(song: str) -> str:
    """Read-path slug: strip every char that is not ``[A-Za-z0-9_]``.

    Byte-identical to the inline form ``re.sub(r"[^A-Za-z0-9_]", "", song)``
    used at ~19 "lookup / sanitize" call sites in the server. Preserves case
    and underscores; does NOT lowercase, collapse runs, or truncate. Callers
    that require a non-empty result still apply their own ``or "..."`` fallback,
    exactly as before (this function never substitutes a fallback of its own).

    >>> lookup_slug("Autumn Leaves")
    'AutumnLeaves'
    >>> lookup_slug("I'm a Fool")
    'ImaFool'
    """
    return re.sub(r"[^A-Za-z0-9_]", "", song)


def chart_slug(title: str) -> str:
    """Write-path slug: lowercase, collapse non-``[a-z0-9]`` runs to ``_``,
    strip leading/trailing ``_``, then truncate to 60 chars.

    Byte-identical to
    ``re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:60]`` (the
    "create / slugify" rule). Operation order is load-bearing: the strip runs
    *before* the 60-char truncation, and the truncation is part of THIS
    expression — call sites that instead apply ``[:60]`` later (at the
    filename, or after their own ``or`` fallback) are intentionally NOT rerouted
    to this function, because that reordering is not byte-identical. Callers
    apply their own ``or "irealb"`` / ``or "yt"`` fallback.

    >>> chart_slug("Blue Bossa (150bpm) Backing Track")
    'blue_bossa_150bpm_backing_track'
    >>> chart_slug("!!!")
    ''
    """
    return re.sub(r"[^a-z0-9]+", "_", title.lower()).strip("_")[:60]


# Source strings recognised by ``analysis_stem``. Matching is lenient (a leading
# constant literal is never returned regardless of the source), so unrecognised
# sources still fall through to the id-or-raise logic below.
_YOUTUBE_SOURCES = frozenset({"youtube", "yt", "url", "youtube_url"})
_UPLOAD_SOURCES = frozenset({"upload", "record", "record-analyze", "recording", "file"})


def analysis_stem(
    source: str,
    *,
    video_id: str | None = None,
    job_id: str | None = None,
) -> str:
    """Collision-safe cache stem for an analysis input — NEVER a constant.

    The NNLS/musx feature caches key on the audio file stem alone (see module
    docstring and ``docs/known_issues.md``). For that keying to be *correct* the
    stem must be:

      * STABLE for the same input  — so re-analysing the same input reuses its
        cached features rather than recomputing;
      * UNIQUE across different inputs — so two inputs that must NOT share
        features can never collide.

    A constant literal (``audio.wav`` / ``input.wav``) satisfies the first and
    violates the second, which is exactly the twice-root-caused bug this
    function prevents. The invariant is encoded in the choice of id:

      * YouTube source -> the yt-dlp ``video_id`` (stable per video; the server
        downloads with outtmpl ``%(id)s``), so a re-analysis of the same video
        correctly hits the cache.
      * upload source  -> the ``job_id`` millisecond timestamp (generated per
        upload before the file is written), so two distinct uploads never
        collide.

    There is deliberately no constant fallback: if the required id is missing —
    or if neither id is supplied for an unrecognised source — this raises
    ``ValueError`` rather than returning a colliding literal.

    Args:
        source: origin of the input, e.g. ``"youtube"`` or ``"upload"``.
        video_id: yt-dlp video id; required (and returned) for a YouTube source.
        job_id: per-upload millisecond-timestamp id; required (and returned)
            for an upload source.

    Returns:
        The per-input stem (``video_id`` for YouTube, ``job_id`` for uploads).

    Raises:
        ValueError: if the id required by ``source`` is missing, or if neither
            id is supplied.
    """
    src = (source or "").strip().lower()

    if src in _YOUTUBE_SOURCES:
        if not video_id:
            raise ValueError(
                "analysis_stem: youtube source requires a non-empty video_id "
                "(refusing a constant stem — see docs/known_issues.md cache-"
                "collision entries)"
            )
        return video_id

    if src in _UPLOAD_SOURCES:
        if not job_id:
            raise ValueError(
                "analysis_stem: upload source requires a non-empty job_id "
                "(refusing a constant stem — see docs/known_issues.md cache-"
                "collision entries)"
            )
        return job_id

    # Unrecognised source string: still enforce the no-constant invariant by
    # returning whichever id was supplied, else raise.
    if video_id:
        return video_id
    if job_id:
        return job_id
    raise ValueError(
        "analysis_stem: neither video_id nor job_id supplied — refusing to "
        "return a constant stem (would reintroduce the cache collision "
        "documented in docs/known_issues.md)"
    )
