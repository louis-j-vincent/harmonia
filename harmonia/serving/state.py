"""harmonia/serving/state.py — persistent shared serving state (registries + offsets).

Extracted verbatim out of ``scripts/harmonia_server.py`` (Phase 6d of the
serving refactor, following ``cache.py``/``render.py``/``config.py``). This is a
behavior-preserving MOVE: same disk-backed JSON files, same import-time load of
each registry, same in-place mutation semantics.

The three registries (``_yt_video_ids``, ``_yt_audio_meta``, ``_ireal_urls``)
are the only MUTABLE module-level objects moved here. They are safe to move
because the server only ever mutates them IN PLACE (``d[k] = v``, ``del d[k]``)
— never by reassignment — so ``scripts.harmonia_server`` re-imports each object
and shares the single live instance; a mutation through either module's name is
visible through the other.

The offset stores are disk-only: ``_load_gt_offsets`` / ``_load_bar1_offsets``
re-read their JSON file on every call, so there is no in-memory dict to keep in
sync — only the file-path constants and their load/save helpers move here.
``_save_gt_offset`` deliberately stays in the server: it also clears the
server-local in-memory ``_billboard_gt_cache``, which is out of scope for this
round; it uses the ``_load_gt_offsets`` / ``_GT_OFFSETS_FILE`` re-imported from
here.

File paths come from ``harmonia.serving.config`` (identical values).
"""

from __future__ import annotations

import json
import logging

from harmonia.serving.config import PLOTS_DIR, REPO

log = logging.getLogger(__name__)


# ── YouTube video ID registry: {html_filename → video_id} — disk-backed so
# it survives server restarts (the app got restarted a lot during dev, and
# every restart used to silently drop the video link for every prior chart).
_YT_IDS_FILE = PLOTS_DIR / ".yt_video_ids.json"


def _load_yt_video_ids() -> dict[str, str]:
    try:
        return json.loads(_YT_IDS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _remember_video_id(filename: str, vid: str) -> None:
    _yt_video_ids[filename] = vid
    try:
        _YT_IDS_FILE.write_text(json.dumps(_yt_video_ids), encoding="utf-8")
    except OSError:
        log.warning("Could not persist YouTube video id for %s", filename)


_yt_video_ids: dict[str, str] = _load_yt_video_ids()

# ── Downloaded-audio registry: {html_filename → {"audio": "/audio/x.m4a",
# "thumb": "https://i.ytimg.com/..."}} — we already download the source
# audio to run inference; instead of throwing it away, we keep it and the
# docked player plays it back locally. Sidesteps the entire class of
# YouTube-iframe problems (origin/CORS, playsinline-forced-fullscreen,
# embedding-disabled videos, duplicate-player collisions) the same way
# other chord-from-YouTube apps (e.g. Chord AI) do it.
_YT_AUDIO_FILE = PLOTS_DIR / ".yt_audio_meta.json"


def _load_yt_audio_meta() -> dict[str, dict[str, str]]:
    try:
        return json.loads(_YT_AUDIO_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _remember_audio(filename: str, audio_url: str, thumb_url: str) -> None:
    _yt_audio_meta[filename] = {"audio": audio_url, "thumb": thumb_url}
    try:
        _YT_AUDIO_FILE.write_text(json.dumps(_yt_audio_meta), encoding="utf-8")
    except OSError:
        log.warning("Could not persist audio link for %s", filename)


_yt_audio_meta: dict[str, dict[str, str]] = _load_yt_audio_meta()

# ── iReal URL registry: {html_filename → irealb_url} — disk-backed so we can
# re-render or re-align the chart later without re-searching iReal ─────────────
_IREAL_URLS_FILE = PLOTS_DIR / ".ireal_urls.json"


def _load_ireal_urls() -> dict[str, str]:
    try:
        return json.loads(_IREAL_URLS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _remember_ireal_url(inferred_filename: str, irealb_url: str) -> None:
    _ireal_urls[inferred_filename] = irealb_url
    try:
        _IREAL_URLS_FILE.write_text(json.dumps(_ireal_urls), encoding="utf-8")
    except OSError:
        log.warning("Could not persist iReal URL for %s", inferred_filename)


_ireal_urls: dict[str, str] = _load_ireal_urls()


# Per-song GT-offset corrections (see docs/known_issues.md "DATA bug, not
# display bug" — Billboard's chords_full timestamps are relative to
# McGill's original master, but this corpus uses a different, duration-
# matched YouTube audio file per song; offsets are per-song, not a global
# constant). Keyed by McGill Billboard track_id (stable across which
# YouTube video happens to be matched), value {offset_s, source, updated}.
# offset_s convention: corrected_time = raw_time + offset_s (matches
# scratchpad/offset_final.py's "+: audio later than GT; shift GT +offset").
_GT_OFFSETS_FILE = REPO / "data" / "cache" / "billboard_gt_offsets.json"


def _load_gt_offsets() -> dict[str, dict]:
    try:
        return json.loads(_GT_OFFSETS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


_BAR1_OFFSETS_FILE = REPO / "data" / "cache" / "chart_bar1_offsets.json"


def _load_bar1_offsets() -> dict[str, dict]:
    try:
        return json.loads(_BAR1_OFFSETS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _save_bar1_offset(slug: str, offset_beats: int) -> None:
    """Persist a song's bar-1 phase offset (see chart_to_interactive_inputs's
    bar1_offset_beats docstring — this is the GRID PHASE, distinct from the
    step-size fix already applied via start_beat_idx). Only takes effect on
    the NEXT analysis of this song (/api/analyze re-reads the store when it
    calls chart_to_interactive_inputs) — it does not retroactively edit an
    already-baked chart HTML file, same caveat as chart_interactive.py
    template edits."""
    import datetime as _dt
    offsets = _load_bar1_offsets()
    offsets[slug] = {
        "offset_beats": int(offset_beats),
        "updated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    _BAR1_OFFSETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _BAR1_OFFSETS_FILE.write_text(json.dumps(offsets, indent=2), encoding="utf-8")
