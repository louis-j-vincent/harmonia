"""harmonia/serving/billboard_gt.py — McGill Billboard ground-truth chord lookup.

Behavior-preserving MOVE out of ``scripts/harmonia_server.py`` (serving refactor,
billboard-GT round — following cache/config/state/render/templates/api/runtime/
loaders). The whole self-contained billboard-GT stateful cluster moves together
so its shared in-memory state is not fragmented across two modules.

Two stateful module-level objects live here:

  * ``_billboard_ds`` — the lazily-initialised mirdata Billboard dataset handle.
    It is a REASSIGNED module global: ``_gt_chords_for_video_raw`` does
    ``global _billboard_ds; ... _billboard_ds = mirdata.initialize("billboard")``
    on first use, so the rebind targets ``billboard_gt._billboard_ds``. It is
    read/written only through THIS module's namespace; the server does NOT
    re-import it (nothing in the server reads it), so there is no stale-None
    local copy to freeze. Verify first-use via ``billboard_gt._billboard_ds``.
  * ``_billboard_gt_cache`` — an in-memory ``{cache_key: gt}`` dict, mutated IN
    PLACE only (``d[k] = v`` in ``_gt_chords_for_video_raw``; ``.clear()`` in
    ``_save_gt_offset``), never reassigned. The server re-imports the SAME live
    object, so a mutation through either module's name is visible through the
    other and ``server._billboard_gt_cache is billboard_gt._billboard_gt_cache``.

The GT-offset store (``_GT_OFFSETS_FILE`` / ``_load_gt_offsets``) comes from the
leaf module ``harmonia.serving.state`` and the corpus file list
(``_BILLBOARD_CORPUS_FILES``) from ``harmonia.serving.config`` — no server
import, no cycle. ``scripts/harmonia_server.py`` re-imports every public name
back, so existing call sites (``gt_playalong_training`` → ``_gt_chords_for_
video``, ``gt_offset_fix`` → ``_gt_chords_for_video_raw``,
``_load_billboard_corpus``) resolve unchanged; ``harmonia.serving.render``
imports ``_gt_chords_for_video`` and ``harmonia.serving.api`` imports
``_save_gt_offset`` straight from here.

Function bodies below are BYTE-FOR-BYTE the HEAD originals (extracted via
``ast.get_source_segment``); only their home changed. ``_estimate_gt_offset``
deliberately STAYS in the server — a librosa onset-alignment heuristic used only
by the ``/gt-offset-fix`` page route, not part of this GT-lookup/cache cluster.
The two ``/api/gt-offset/<track_id>`` routes moved to ``harmonia.serving.api``.
"""

from __future__ import annotations

import json
import logging

from harmonia.serving.config import _BILLBOARD_CORPUS_FILES
from harmonia.serving.state import _GT_OFFSETS_FILE, _load_gt_offsets

log = logging.getLogger(__name__)


def _billboard_video_to_track_id() -> dict[str, str]:
    """video_id -> McGill Billboard track_id, for the ~60 training-corpus
    songs (reads the same JSON files as _load_billboard_corpus — cheap,
    small files, no caching needed)."""
    merged: dict[str, dict] = {}
    for p in _BILLBOARD_CORPUS_FILES:
        try:
            merged.update(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass
    out = {}
    for track_id, v in merged.items():
        best = v.get("best") or []
        if best:
            out[best[0]] = track_id
    return out


_billboard_ds = None
_billboard_gt_cache: dict[str, list] = {}


def _save_gt_offset(track_id: str, offset_s: float, source: str = "manual") -> None:
    import datetime as _dt
    offsets = _load_gt_offsets()
    offsets[track_id] = {
        "offset_s": float(offset_s),
        "source": source,
        "updated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    _GT_OFFSETS_FILE.parent.mkdir(parents=True, exist_ok=True)
    _GT_OFFSETS_FILE.write_text(json.dumps(offsets, indent=2), encoding="utf-8")
    _billboard_gt_cache.clear()  # any cached (offset-applied) GT is now stale


def _gt_chords_for_video_raw(video_id: str) -> tuple[str | None, list[dict] | None]:
    """(track_id, raw GT chords) with no offset applied. None GT if this
    video isn't a training-corpus song."""
    global _billboard_ds
    track_id = _billboard_video_to_track_id().get(video_id)
    if not track_id:
        return None, None
    cache_key = f"raw:{track_id}"
    if cache_key in _billboard_gt_cache:
        return track_id, _billboard_gt_cache[cache_key]
    try:
        import mirdata
        if _billboard_ds is None:
            _billboard_ds = mirdata.initialize("billboard")
        cf = _billboard_ds.track(track_id).chords_full
        gt = [
            {"t0": float(t0), "t1": float(t1), "label": str(lbl)}
            for (t0, t1), lbl in zip(cf.intervals, cf.labels)
        ]
    except Exception as e:
        log.warning("billboard GT: could not load chords_full for %s (%s)", track_id, e)
        gt = []
    _billboard_gt_cache[cache_key] = gt
    return track_id, gt


def _gt_chords_for_video(video_id: str) -> list[dict] | None:
    """Ground-truth chord intervals (McGill Billboard hand annotations) for a
    training-corpus video, as [{t0, t1, label}] with ``label`` left in raw
    Harte notation ("C:min7") — the app UI's own parseLabel() (app_shell.html)
    already turns Harte into the same {root, q} shape it renders inferred
    chords with, so the display code is shared rather than reimplemented here.
    Returns None if this video isn't a training-corpus song (arbitrary pasted
    YouTube links must not show a GT row — they have none).

    Applies this song's saved GT-offset correction (see
    data/cache/billboard_gt_offsets.json / /gt-offset-fix), if any, so every
    view that calls this function (training-mode chart, gt-playalong*)
    automatically reflects a hand-corrected offset without further plumbing."""
    track_id, gt_raw = _gt_chords_for_video_raw(video_id)
    if gt_raw is None:
        return None
    offset = _load_gt_offsets().get(track_id or "", {}).get("offset_s", 0.0)
    if not offset:
        return gt_raw
    return [
        {"t0": max(0.0, c["t0"] + offset), "t1": max(0.0, c["t1"] + offset), "label": c["label"]}
        for c in gt_raw
    ]
