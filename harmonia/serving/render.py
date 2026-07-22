"""harmonia/serving/render.py — chart-model / chart-render helpers (Phase 6b PORT).

Moved verbatim out of ``scripts/harmonia_server.py`` (Phase 6b of the serving
refactor). This is a behavior-preserving MOVE: the function body is byte-for-byte
the original ``_chart_model_for``; only a lazy dependency-binding preamble was
added at the top so the helper can live here without an import cycle.

``_chart_model_for`` is the single non-route helper in the server that wraps
``harmonia.output.chart_model`` (``payload_from_chart_html`` / ``to_chart_model``).
The persistent registries/offsets it needs (``_yt_audio_meta``, ``_yt_video_ids``,
``_load_bar1_offsets``) now live in ``harmonia.serving.state`` (Phase 6d) and are
imported at module top — state is a leaf module (no server import), so this is
safe and no longer needs the lazy back-import. It still depends on server-owned
state out of scope for this round (sidecar stores, the bar-1 offset transform,
GT lookup, ``_raw_beat_times_cached``); those remain bound lazily from
``scripts.harmonia_server`` at call time. ``scripts/harmonia_server.py`` imports
``_chart_model_for`` back from here, so all existing call sites are unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

from harmonia.serving.config import AUDIO_DIR, PLOTS_DIR
from harmonia.serving.state import _load_bar1_offsets, _yt_audio_meta, _yt_video_ids


def _chart_model_for(filename: str, include_gt: bool = True) -> dict:
    """ChartModel for a rendered chart — payload + sidecar + audio/video links.

    ``include_gt``: attach McGill Billboard ground-truth chords (training-mode
    songs only — see _gt_chords_for_video) as model["gt"]. Skipped for the
    /api/library summary loop (chart_summary never reads it, and a mirdata
    lookup per song on every library load is needless overhead)."""
    from harmonia.output.chart_model import payload_from_chart_html, to_chart_model

    # Phase 6b PORT: this helper still leans on stateful, server-owned deps
    # that are NOT in this round's move scope (the sidecar stores, the bar-1
    # offset transform, the McGill-GT lookup, and _raw_beat_times_cached).
    # Bind those from the server module lazily (at call time, never at import
    # time) so this move introduces no import cycle and reuses the SAME live
    # objects. The body below is byte-for-byte the original _chart_model_for.
    #
    # PLOTS_DIR / AUDIO_DIR come from harmonia.serving.config; the persistent
    # registries/offsets (_yt_audio_meta, _yt_video_ids, _load_bar1_offsets)
    # come from harmonia.serving.state — both imported at module top (state is
    # a leaf module, so no import cycle), no longer back-imported via _srv.
    import scripts.harmonia_server as _srv
    _apply_bar1_offset_to_payload = _srv._apply_bar1_offset_to_payload
    _load_annotation = _srv._load_annotation
    _gt_chords_for_video = _srv._gt_chords_for_video
    _raw_beat_times_cached = _srv._raw_beat_times_cached

    p = PLOTS_DIR / filename
    payload = payload_from_chart_html(p)
    slug = filename.removeprefix("inferred_").removesuffix(".html")
    saved_offset = _load_bar1_offsets().get(slug, {}).get("offset_beats", 0)
    if saved_offset:
        payload = _apply_bar1_offset_to_payload(payload, int(saved_offset))
    meta = _yt_audio_meta.get(filename) or {}
    audio_url = meta.get("audio", "")
    if not audio_url:
        # Variant/demo charts (_npattern, _barlocked, _bestfit…) are copies of
        # a base chart and share its audio, but _yt_audio_meta is keyed by the
        # exact chart filename so copies had no play button (user report
        # 2026-07-19). Strip trailing _suffix tokens until an audio file
        # matches the base slug.
        s = slug
        while s:
            if (AUDIO_DIR / f"{s}.m4a").exists():
                audio_url = f"/audio/{s}.m4a"
                break
            if "_" not in s:
                break
            s = s.rsplit("_", 1)[0]
    if audio_url and not (AUDIO_DIR / Path(audio_url).name).exists():
        audio_url = ""
    video_id = _yt_video_ids.get(filename, "")
    model = to_chart_model(
        payload,
        filename=filename,
        video_id=video_id,
        audio_url=audio_url,
        annotation=_load_annotation(filename),
    )
    if include_gt and video_id:
        gt = _gt_chords_for_video(video_id)
        if gt:
            model["gt"] = gt
    # Default ON 2026-07-20 (validated->prod): corpus mean 84->27ms (-68%),
    # improves all 7 matched songs, zero regression (only chord/section
    # DISPLAY TIMING moves — labels, sections, folds are untouched, verified
    # byte-identical). Rollback: HARMONIA_BOUNDARY_SNAP=0.
    if os.environ.get("HARMONIA_BOUNDARY_SNAP", "1") == "1" and audio_url:
        bt = _raw_beat_times_cached(Path(audio_url).stem)
        if bt:
            model["beatTimes"] = bt
    return model
