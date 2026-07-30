"""harmonia/serving/config.py — shared filesystem paths for the serving layer.

Extracted verbatim out of ``scripts/harmonia_server.py`` (serving refactor,
constants round). This is a behavior-preserving MOVE: same values, same
import-time ``.mkdir(...)`` side effects. Route/render extraction can import
these paths from here instead of back-importing the server monolith.

``REPO`` resolves to the repo root just as it did in ``harmonia_server.py``.
That module computed it as ``Path(__file__).resolve().parent.parent`` from
``scripts/harmonia_server.py``; this file lives one level deeper
(``harmonia/serving/config.py``), so it uses ``.parents[2]`` to reach the
identical directory.
"""

from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

PLOTS_DIR = REPO / "docs" / "plots"
PWA_DIR = REPO / "docs" / "pwa"
AUDIO_DIR = REPO / "docs" / "audio"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Chroma/pitch (Basic Pitch) activations, keyed by song slug — stable and
# directly addressable, unlike PitchExtractor's own internal cache (keyed by
# the *downloaded temp file's* path+mtime, which stops existing the moment
# _run_analysis deletes tmp_dir, making that cache practically unreachable
# after the fact even though the .npz blob itself never gets evicted).
# Lets a later "re-score these bars against pooled chroma" pass (annotator
# tool, docs/architecture_extensions.md §13) reload activations for a song
# without re-running Basic Pitch — same slug as docs/audio/<slug>.m4a and
# the inferred_<slug>.html chart, so no separate manifest is needed.
PITCH_CACHE_DIR = REPO / "data" / "cache" / "pitch"
PITCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)

# Annotator-tool sidecars — one JSON file per chart, per docs/annotation_
# sidecar_schema.md. Per-song files (not one aggregate dict like
# _yt_video_ids) so a sidecar can travel with its chart and corruption in
# one doesn't touch another; see that doc's §5.4 for the rationale.
ANNOT_DIR = PLOTS_DIR / "annotations"
ANNOT_DIR.mkdir(parents=True, exist_ok=True)

# Human-correction training logs — one JSON file per corrected chord, grouped
# by song slug (data/training_logs/<song>/<ts>_<user>_bar<N>.json). Each record
# pairs the model's original reading with the human fix and the /api/reinfer
# diff it produced, so we can later mine systematic model errors and retrain the
# quality head. Written best-effort: a logging failure must never break a save.
TRAINING_LOGS_DIR = REPO / "data" / "training_logs"

# Real detected beat times per slug, disk-cached (see _raw_beat_times_cached in
# the server). v2 dir so the 2026-07-21 backend-mismatch fix can't be masked by
# a stale v1 entry matching on slug alone.
# v3 (2026-07-30): v2's 48 entries were written by the LIBROSA fallback, because
# Beat This! silently could not decode .m4a on this box. A new directory rather
# than an in-place invalidation, so a poisoned entry cannot survive the fix by
# matching on slug alone — same precedent as the v1->v2 move.
_BEAT_TIMES_CACHE = REPO / "data" / "cache" / "raw_beat_times_v3"

# Bar-grid and server-side waveform-peaks caches (see _waveform_peaks).
BEATGRID_CACHE = REPO / "data" / "cache" / "beat_grid"
WAVEFORM_CACHE = REPO / "data" / "cache" / "waveform_peaks"

# Bar LENGTH per slug measured from Beat This!'s native downbeats — the accent
# cue that breaks the rigid grid's 2x metrical octave (see bar_ref_for_slug in
# harmonia.serving.audio and harmonia.models.rigid_grid). Disk-cached because it
# costs a full beat-tracker pass; keyed by slug, holds ``None`` results too so an
# abstention is not recomputed on every chart load.
BAR_REF_CACHE = REPO / "data" / "cache" / "bar_ref_downbeats"

# The ~58-60 Billboard real-audio training-corpus search results (two disjoint
# keyed-by-track_id JSON dumps produced by an earlier YouTube-match pass; union
# = full corpus). Shared path constant: read read-only by both the server's
# _load_billboard_corpus and harmonia.serving.billboard_gt's
# _billboard_video_to_track_id, so it lives here (a leaf) to keep either
# consumer from importing the other.
_BILLBOARD_CORPUS_FILES = [
    REPO / "scratchpad" / "billboard_search_results_60.json",
    REPO / "scratchpad" / "billboard_search_results.json",
]
