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
``_save_gt_offset`` is NOT here: because it also clears the in-memory
``_billboard_gt_cache``, a later round moved it with the rest of the billboard-GT
cluster into ``harmonia.serving.billboard_gt``, which imports ``_load_gt_offsets``
/ ``_GT_OFFSETS_FILE`` from here.

File paths come from ``harmonia.serving.config`` (identical values).
"""

from __future__ import annotations

import json
import logging
import time

from harmonia.serving.cache import lookup_slug
from harmonia.serving.config import ANNOT_DIR, PLOTS_DIR, REPO, TRAINING_LOGS_DIR
from harmonia.serving.loaders import _annot_path

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


def _bar1_offset_bounds(bpb: int, n_bars: int) -> tuple[int, int]:
    """Safe [lo, hi] range (in beats) for a saved bar-1 offset, given this
    chart's bpb and current bar count.

    2026-07-17 redesign: offset_beats is no longer just a SUB-BAR phase.
    Beyond one bar's worth of phase, whole multiples of bpb are a legitimate,
    different operation — "exclude N whole bars from the front of the chart
    as intro/pickup material" (see docs/known_issues.md "Yesterday align-tool
    ... intro exclusion" entry: a real case needed +8 beats = 2 full bars of
    intro skipped, which the old mod-bpb reduction wrongly treated as a
    same-as-0 no-op). _apply_bar1_offset_to_payload now DROPS chords that
    fall before the offset instead of merging them into bar 0, so any
    magnitude is representable safely — the only real constraint is not
    emptying the whole chart. Cap at whichever is smaller of (n_bars - 1)
    bars [must leave >=1 bar] and a sane absolute ceiling (16 bars) so a
    typo/garbage value can't wipe a short chart or blow up a huge one.
    Symmetric negative bound: a negative offset only ever INSERTS pickup
    beats before bar 1 (nBars grows, nothing is dropped), so it can't
    corrupt data, but is capped the same way to keep the range sane.

    Pure arithmetic leaf (serving refactor, Phase 6c batch-2): MOVED VERBATIM
    out of scripts/harmonia_server.py so the extracted POST /api/bar1-offset
    route (in harmonia.serving.api) and the still-inline /bar1-offset-fix page
    route can both import it from here. The _apply_bar1_offset_to_payload
    referenced above stays server-owned (render.py reaches it via the
    documented lazy back-import); only this bound-computing leaf moved."""
    cap_bars = max(0, min(max(n_bars - 1, 0), 16))
    hi = bpb * cap_bars
    return -hi, hi


# ---------------------------------------------------------------------------
# Writer helpers for the annotation / correction-log / section-label sidecars
# (serving refactor, mutating-routes round). MOVED VERBATIM out of
# scripts/harmonia_server.py: same bodies, same guards. Their POST routes moved
# to harmonia.serving.api alongside; the server re-imports these so every
# existing call site (incl. the GET-side _load_section_labels that stays there)
# is unchanged and server.X is state.X.
# ---------------------------------------------------------------------------


def _training_log_dir(song: str) -> Path:
    safe = lookup_slug(song or "") or "unknown"
    return TRAINING_LOGS_DIR / safe

def _remember_annotation(filename: str, doc: dict) -> dict:
    doc["schema"] = 1
    doc["chart"] = filename
    doc["modified"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    doc.setdefault("chords", [])
    doc.setdefault("merges", [])
    try:
        ANNOT_DIR.mkdir(parents=True, exist_ok=True)
        _annot_path(filename).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    except OSError:
        log.warning("Could not persist annotation for %s", filename)
    return doc

def _section_labels_path(filename: str) -> Path:
    return ANNOT_DIR / f"{filename}.sections.json"

def _save_section_labels(filename: str, labels: dict) -> dict:
    import datetime as _dt
    doc = {
        "labels": labels,
        "updated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    ANNOT_DIR.mkdir(parents=True, exist_ok=True)
    _section_labels_path(filename).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return doc


# ---------------------------------------------------------------------------
# _waveform_peaks / _beat_grid_for (audio-envelope + beat-grid compute-cache
# leaves, briefly parked here in Phase 6c batch-3) MOVED to the dedicated
# harmonia.serving.audio module in the Phase 6c architectural pass, alongside
# _raw_beat_times_cached. Their callers (serving/api.py routes + the staying
# server page routes) now import them from harmonia.serving.audio.
# ---------------------------------------------------------------------------
