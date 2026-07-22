"""
Harmonia local web server.

Serves the existing HTML chart files with a floating "Analyze YouTube" button
injected into each page.  When you paste a YouTube URL and click Analyze, the
server downloads the audio, runs chord_pipeline_v1 (Gen-2), and redirects you
to the freshly-generated interactive chart.

Usage:
    .venv/bin/python scripts/harmonia_server.py
    → opens http://localhost:7771 in the browser

Options:
    --port PORT      (default 7771)
    --no-open        don't open the browser automatically
    --phase N        chord vocabulary phase (1-4, default 1)
    --cache-dir DIR  Basic Pitch cache dir (default data/cache)
    --no-madmom      use librosa beat tracker instead of madmom
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import tempfile
import socket
import threading
import time
import webbrowser
from pathlib import Path
from urllib.parse import quote

# Bootstrap REPO onto sys.path BEFORE any ``harmonia.`` import can resolve
# (a direct ``python scripts/harmonia_server.py`` adds scripts/, not the repo
# root). harmonia.serving.config re-exports REPO (identical value) below.
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from flask import Flask, Response, jsonify, redirect, render_template_string, request, send_from_directory

from harmonia.serving.cache import chart_slug, lookup_slug
# Shared filesystem paths (and their import-time mkdir side effects) now live in
# harmonia.serving.config; re-bound here so every existing reference is unchanged.
from harmonia.serving.config import (
    REPO,
    PLOTS_DIR,
    PWA_DIR,
    AUDIO_DIR,
    PITCH_CACHE_DIR,
    ANNOT_DIR,
    TRAINING_LOGS_DIR,
    _BEAT_TIMES_CACHE,
    BEATGRID_CACHE,
    WAVEFORM_CACHE,
    _BILLBOARD_CORPUS_FILES,
)
# Persistent shared serving state (disk-backed registries + offset stores) now
# lives in harmonia.serving.state (Phase 6d). Re-bound here so every existing
# reference (_yt_video_ids, _remember_*, _YT_IDS_FILE, _load_gt_offsets, …) is
# unchanged; the moved registry dicts are the SAME live objects — mutated in
# place here and visible in state, and vice-versa (they are only ever mutated
# via d[k]=v / del d[k], never reassigned, which is what makes the move safe).
# render (imported below) pulls the registries straight from state, so state
# must load first.
from harmonia.serving.state import (
    _YT_IDS_FILE,
    _load_yt_video_ids,
    _remember_video_id,
    _yt_video_ids,
    _YT_AUDIO_FILE,
    _load_yt_audio_meta,
    _remember_audio,
    _yt_audio_meta,
    _IREAL_URLS_FILE,
    _load_ireal_urls,
    _remember_ireal_url,
    _ireal_urls,
    _load_gt_offsets,
    _BAR1_OFFSETS_FILE,
    _load_bar1_offsets,
    _save_bar1_offset,
    # Writer helpers MOVED to state.py this round (mutating-routes serving
    # refactor); re-imported so server.X is state.X and _load_section_labels
    # (still here) resolves _section_labels_path.  noqa: F401 re-export.
    _remember_annotation,
    _training_log_dir,
    _section_labels_path,
    _save_section_labels,
)
from harmonia.serving.render import (
    _chart_model_for,
    # Presentation/injection helpers + their dedicated constants now live in
    # harmonia.serving.render (serving refactor, render round). Pure string
    # manipulation (html in → html out), no beat/grid/model deps; imported back
    # here so all existing call sites are unchanged.
    _inject_overlay,
    _inject_back_button,
    _PWA_HEAD,
    _BACK_BUTTON_HTML,
    _INJECT_MARKER,
)
# Pure read-only data-loaders (serving refactor, loaders round). Byte-identical
# MOVE out of this module: _annot_path/_load_annotation (annotation sidecar) and
# _load_ireal_alignment (iReal chart payload). loaders is a leaf (config-only, no
# server import); imported back here so every existing call site — including the
# annotation write helpers and the grid-lane iReal-alignment routes — is unchanged.
from harmonia.serving.loaders import (
    _annot_path,
    _load_annotation,
    _load_ireal_alignment,
)
# McGill Billboard ground-truth chord lookup — the self-contained billboard-GT
# stateful cluster (_billboard_ds handle, _billboard_gt_cache, corpus->track_id
# reader, raw/offset GT lookups, _save_gt_offset) now lives in
# harmonia.serving.billboard_gt. Re-imported here so server.X is billboard_gt.X
# and existing call sites (gt_playalong_training -> _gt_chords_for_video,
# gt_offset_fix -> _gt_chords_for_video_raw) resolve unchanged. NOTE: _billboard_ds
# is a lazily-REASSIGNED module global and is deliberately NOT re-bound here —
# nothing in the server reads it, and re-binding it as a local would freeze a
# stale None while billboard_gt's own global gets populated on first use. The
# _billboard_gt_cache dict IS re-bound: it is mutated in place only (never
# reassigned), so it is the SAME live object in both modules. The two
# /api/gt-offset/<track_id> routes moved to harmonia.serving.api.  noqa: F401.
from harmonia.serving.billboard_gt import (
    _billboard_gt_cache,
    _billboard_video_to_track_id,
    _gt_chords_for_video,
    _gt_chords_for_video_raw,
    _save_gt_offset,
)
from harmonia.serving.templates import (
    ANNOTATOR_SIMPLE_TEMPLATE,
    ANNOTATOR_TEMPLATE,
    ANNOTATOR_V3_TEMPLATE,
    ANNOTATOR_V4_TEMPLATE,
    HOME_TEMPLATE,
    LIBRARY_TEMPLATE,
    _OVERLAY_HTML_TOOLS,
    _OVERLAY_HTML_YT,
    _SWIPE_NAV_JS,  # noqa: F401  re-exported for identity/back-compat; currently unused
)
# Runtime settings + shared in-memory state now live in harmonia.serving.runtime.
# ``ARGS`` is reassigned at startup, so it is NOT re-bound as a local name (that
# would freeze at the pre-startup None) — it is read live as ``runtime.ARGS.<x>``
# and set by main() as ``runtime.ARGS = parse_args()``. The mutated-in-place
# state (_jobs/_jam_sessions + locks) and the read-only _ANALYZE_* env config are
# re-bound below; the dicts are the SAME live objects as runtime.jobs /
# runtime.jam_sessions (mutated in place here, visible there, and vice-versa).
import harmonia.serving.runtime as runtime
from harmonia.serving.runtime import (
    jobs as _jobs,
    jobs_lock as _jobs_lock,
    jam_sessions as _jam_sessions,
    jam_sessions_lock as _jam_sessions_lock,
    _ANALYZE_FEATURE_FRONTEND,
    _ANALYZE_BASS_FRONTEND,
    _ANALYZE_QUALITY_FRONTEND,
    _ANALYZE_SEGMENT_SOURCE,
    _ANALYZE_BEAT_PERIOD_MODE,
)

log = logging.getLogger(__name__)


# Bump this to force every installed client to drop its old cache on next visit.
# Cache version was a hardcoded "harmonia-v1" — the installed PWA never saw a
# "new version", so the update banner never fired and phones kept running a
# stale shell (user report 2026-07-19: shipped playhead fix invisible on
# iPhone). Now derived at request time from app_shell.html's mtime: every UI
# deploy auto-bumps it.
_SW_CACHE_VERSION = "harmonia-v1"  # fallback only; see service_worker()

_SERVICE_WORKER_JS = """const CACHE = "%%VERSION%%";
self.addEventListener("install", () => self.skipWaiting());
self.addEventListener("activate", e => {
  e.waitUntil(
    caches.keys().then(keys => Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k))))
  );
  self.clients.claim();
});
// Network-first, cache fallback — so a chart you've already opened once still
// opens with no signal (dead spot in a venue, subway, etc). Job-status polling
// under /api/ is never cached: those responses go stale in seconds.
self.addEventListener("fetch", e => {
  const req = e.request;
  const url = new URL(req.url);
  // /audio/ uses Range requests for seeking — let the browser's own HTTP
  // cache handle partial-content responses natively instead of us caching
  // a byte-range slice as if it were the whole file.
  if (req.method !== "GET" || url.origin !== self.location.origin
      || url.pathname.startsWith("/api/") || url.pathname.startsWith("/audio/")) {
    return;
  }
  e.respondWith(
    fetch(req).then(res => {
      if (res.ok) { const copy = res.clone(); caches.open(CACHE).then(c => c.put(req, copy)); }
      return res;
    }).catch(() => caches.match(req).then(cached => cached || (req.mode === "navigate" ? caches.match("/") : undefined)))
  );
});
""".replace("%%VERSION%%", _SW_CACHE_VERSION)

app = Flask(__name__, static_folder=None)

# ── Serving Blueprint (Phase 6e): the first batch of cleanly-movable,
# read-only SAFE-GET routes now live in harmonia.serving.api. Registered with
# name="" so the blueprint does NOT namespace endpoints — Flask computes each
# endpoint as f"{name_prefix}.{name}.{endpoint}".lstrip("."), so an empty name
# yields the ORIGINAL bare endpoint (serve_audio, api_library, …). The app
# url_map is therefore identical (same rules/endpoints/methods) and any
# url_for() still resolves. See harmonia/serving/api.py for the full contract.
from harmonia.serving.api import api as api_bp
app.register_blueprint(api_bp, name="")

# ── CLI args, the mutated-in-place job/jam registries + locks, and the read-only
# _ANALYZE_* env config now live in harmonia.serving.runtime (serving refactor,
# runtime round); imported and re-bound at module top. CLI args are the special
# case: reassigned at startup, so they are read live as ``runtime.ARGS.<attr>``
# (see the import block above and main()) rather than re-bound as a local name.
# The registry dicts (_jobs / _jam_sessions) are the SAME live objects as
# runtime.jobs / runtime.jam_sessions — mutated in place here (d[k]=v / .pop /
# .update), never reassigned, which is what makes the re-bind safe.

# ── Disk-backed registries (YouTube video ids, retained-audio meta, iReal
# URLs) + their load/remember closures now live in harmonia.serving.state
# (Phase 6d); imported and re-bound at module top. The dicts are the SAME live
# objects (mutated in place here via _remember_* and the delete route). ────────


def _lan_ip() -> str:
    """Best-effort local network IP (for phones on the same Wi-Fi)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except OSError:
        return ""
    finally:
        s.close()


def _extract_video_id(url: str) -> str:
    """Extract YouTube video ID from a URL. Returns '' if not found."""
    m = re.search(r'(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})', url)
    return m.group(1) if m else ""

# ── Inject snippet ────────────────────────────────────────────────────────────
# _INJECT_MARKER / _BACK_BUTTON_HTML + _inject_overlay / _inject_back_button
# moved to harmonia.serving.render (serving refactor, render round) and imported
# back at module top. Pure string injectors (html in → html out), no beat/grid/
# model deps; call sites unchanged.


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/sw.js")
def service_worker():
    """Offline cache — network-first, falls back to cache when there's no signal."""
    try:
        ver = f"harmonia-{int(_APP_SHELL.stat().st_mtime)}"
    except OSError:
        ver = _SW_CACHE_VERSION
    return Response(_SERVICE_WORKER_JS.replace(_SW_CACHE_VERSION, ver),
                    mimetype="application/javascript")


_STRUCTURE_DEBUG_JSON = REPO / "scratchpad" / "real_structure_results.json"
_STRUCTURE_MULTILEVEL_JSON = REPO / "scratchpad" / "real_structure_multilevel.json"

# NEW debug route (2026-07-18, structure-detection real-audio checkpoint —
# explicitly authorized by the user for THIS purpose; does not touch any
# existing chart-serving path). Renders the Stage B qualitative real-audio
# structure segmentation from docs/research_sessions/
# structure_realaudio_2026_07_18.md: root-only, probabilistic-input, learned
# key-normalized encoder fed the real pipeline's per-bar root softmax
# (scratchpad/symstruct_proba.py + scratchpad/run_real_structure.py). No GT
# section labels exist for real audio in this repo, so this is a QUALITATIVE
# inspection page, not a scored metric — no V-measure number is shown here.
#
# 2026-07-18 Call 2 update: now renders THREE nested levels (phrase/section/
# form) from scratchpad/run_real_structure_multilevel.py (falls back to the
# old single-level JSON if the multilevel one hasn't been generated yet) —
# see docs/known_issues.md "Task 2" entries for what's validated vs not:
# the section level is statistically tied with flat block8 on clean symbolic
# data (multi-seed re-audit), kept here for its demonstrated real-audio noise
# robustness (Call 1 Stage B3a), not for a flat-V_F win.
_STRUCT_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
                  "#46f0f0", "#f032e6", "#bcf60c", "#fabebe", "#008080",
                  "#e6beff", "#9a6324", "#800000", "#aaffc3", "#808000",
                  "#ffd8b1", "#000075", "#808080"]


def _seg_divs_from_runs(runs, total_bars):
    seg_divs = []
    for r in runs:
        width_pct = 100.0 * (r["bar_end"] - r["bar_start"]) / max(1, total_bars)
        digits = "".join(ch for ch in r["label"] if ch.isdigit())
        lab_num = int(digits) if digits else 0
        color = _STRUCT_COLORS[lab_num % len(_STRUCT_COLORS)]
        seg_divs.append(
            '<div class="seg" style="width:%.3f%%;background:%s" '
            'title="%s  bars %d-%d  t=%.1fs-%.1fs">%s</div>' % (
                width_pct, color, r["label"], r["bar_start"], r["bar_end"],
                r["t_start"], r["t_end"], r["label"]))
    return "".join(seg_divs)


@app.route("/debug/structure")
def debug_structure():
    """Qualitative checkpoint page: predicted section labels laid over the
    bar timeline for 2-3 real songs, plus the docked audio player, so the
    user can listen/look and judge it directly (per the task brief — no
    fabricated metric)."""
    audio_map = {
        "autumn_leaves": "autumn_leaves.m4a",
        "abba_chiquitita": "abba_chiquitita_official_lyric_video.m4a",
        "aretha_chain_of_fools": "aretha_franklin_chain_of_fools_official_lyric_video.m4a",
    }
    blocks_html = []

    if _STRUCTURE_MULTILEVEL_JSON.exists():
        data = json.loads(_STRUCTURE_MULTILEVEL_JSON.read_text())
        level_meta = [("phrase", "PHRASE (2-bar nuclear)"),
                      ("section", "SECTION (8-bar, deployed)"),
                      ("form", "FORM (coarse regrouping)")]
        for name, res in data.items():
            total_bars = max(1, res["n_bars"])
            audio_file = audio_map.get(name, "")
            audio_tag = (
                '<audio controls preload="none" src="/audio/%s"></audio>' % audio_file
                if audio_file else "<em>(no audio mapped)</em>")
            level_rows = []
            for key, title in level_meta:
                lvl = res["levels"][key]
                level_rows.append(
                    '<div class="level-label">%s &middot; %d distinct</div>'
                    '<div class="timeline">%s</div>' % (
                        title, lvl["n_distinct"],
                        _seg_divs_from_runs(lvl["runs"], total_bars)))
            blocks_html.append("""
            <section class="song">
              <h2>%s</h2>
              <div class="meta">tempo=%.1f bpm &middot; n_bars=%d &middot;
                est_tonic_pc=%d</div>
              %s
              %s
            </section>
            """ % (name, res["tempo_bpm"], res["n_bars"], res["est_tonic_pc"],
                  audio_tag, "".join(level_rows)))
        note = ("""Call 2 (2026-07-18): THREE nested levels per song, one
          probabilistic-root variable-span encoder (keynorm_proba_varspan.pt)
          shared across all levels &mdash; PHRASE (2-bar, mandated nuclear
          default), SECTION (8-bar, the deployed level), FORM (coarse
          re-clustering of section labels at a lower similarity threshold).
          No section ground truth exists for real audio in this repo &mdash;
          NOT scored (no V-measure), inspect by eye/ear. Same color anywhere
          within one level's row = predicted same group at that level; colors
          are NOT comparable across levels. Known caveat: level granularity
          isn't always monotonic in distinct-label count (each level's
          threshold is independently tuned) &mdash; see
          docs/known_issues.md "Task 2" entries. Full writeup:
          docs/research_sessions/structure_realaudio_2026_07_18.md.""")
    elif _STRUCTURE_DEBUG_JSON.exists():
        data = json.loads(_STRUCTURE_DEBUG_JSON.read_text())
        for name, res in data.items():
            total_bars = max(1, res["n_bars"])
            audio_file = audio_map.get(name, "")
            audio_tag = (
                '<audio controls preload="none" src="/audio/%s"></audio>' % audio_file
                if audio_file else "<em>(no audio mapped)</em>")
            blocks_html.append("""
            <section class="song">
              <h2>%s</h2>
              <div class="meta">tempo=%.1f bpm &middot; n_bars=%d &middot;
                est_tonic_pc=%d &middot; n_sections=%d &middot; tau=%.2f</div>
              %s
              <div class="timeline">%s</div>
            </section>
            """ % (name, res["tempo_bpm"], res["n_bars"], res["est_tonic_pc"],
                  res["n_sections"], res["tau"], audio_tag,
                  _seg_divs_from_runs(res["runs"], total_bars)))
        note = ("""Stage B (Call 1, 2026-07-17): single-level, root-only +
          probabilistic-input learned key-normalized encoder. No section
          ground truth exists for real audio in this repo &mdash; NOT scored.
          See docs/research_sessions/structure_realaudio_2026_07_18.md.""")
    else:
        return ("No results yet — run scratchpad/run_real_structure_multilevel.py "
                "(or run_real_structure.py) first"), 404

    html = """<!doctype html><html><head><meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>Structure debug — real audio</title>
    <style>
      body { font-family: -apple-system, sans-serif; margin: 0; padding: 16px;
             background: #111; color: #eee; }
      h1 { font-size: 1.1rem; }
      .note { color: #aaa; font-size: 0.85rem; margin-bottom: 20px; }
      .song { margin-bottom: 28px; }
      .meta { color: #999; font-size: 0.8rem; margin-bottom: 6px; }
      audio { width: 100%; margin-bottom: 8px; }
      .level-label { color: #888; font-size: 0.7rem; margin: 6px 0 2px; }
      .timeline { display: flex; width: 100%; height: 30px; border-radius: 4px;
                  overflow: hidden; margin-bottom: 4px; }
      .seg { display: flex; align-items: center; justify-content: center;
             font-size: 0.6rem; color: #000; overflow: hidden;
             white-space: nowrap; border-right: 1px solid rgba(0,0,0,0.3); }
    </style></head><body>
    <h1>Real-audio structure segmentation — qualitative checkpoint</h1>
    <div class="note">__NOTE__</div>
    __BLOCKS__
    </body></html>""".replace("__NOTE__", note).replace("__BLOCKS__", "".join(blocks_html))
    return Response(html, mimetype="text/html")


_APP_SHELL = REPO / "harmonia" / "output" / "app_shell.html"


@app.route("/")
def index():
    """The app (design handoff 2): search → analyse → chart → annotate, one
    page. It reads /api/library, /api/chart-model, /api/analyze, /api/reinfer;
    the ChartModel adapter (harmonia/output/chart_model.py) is the only place
    the raw inference payload gets normalised.

    The pre-app pages are still live: /classic is the old search home and
    /chart/<file> the baked per-song chart, which remains the source the app
    reads its ChartModel out of."""
    page = _APP_SHELL.read_text(encoding="utf-8")
    return Response(page.replace("</head>", _PWA_HEAD + "</head>", 1), mimetype="text/html")


@app.route("/classic")
def classic_index():
    """The previous search-first home page, kept reachable."""
    n_charts = len(list(PLOTS_DIR.glob("inferred_*.html")))
    page = render_template_string(HOME_TEMPLATE, n_charts=n_charts)
    return Response(page.replace("</head>", _PWA_HEAD + "</head>", 1), mimetype="text/html")


def _raw_beat_times_cached(slug: str) -> list | None:
    """Real detected beat times for <slug>, disk-cached — SAME backend
    (Beat This!, falling back to librosa) as the production decode's default
    ``beat_backend="beatthis"`` (chord_pipeline_v1.infer_chords_v1).

    2026-07-21 bug found and fixed: this function used to hard-code
    ``librosa.beat.beat_track`` regardless of which backend actually built
    the chart's bar-grid — a genuine two-different-clocks bug (CLAUDE.md
    rule #6, "component swaps change more than the target metric"), not
    just a display nicety. The DISPLAY SNAP (below) is supposed to correct
    the playhead onto the real beat the chart's own grid is built from; with
    a mismatched backend it was instead correcting onto an UNRELATED
    tracker's beats, which can disagree by more than the wobble it was
    trying to fix (librosa is validated worse on tempo-octave: 65% vs Beat
    This!'s 78%, see docs/known_issues.md). User report, confirmed live on
    "Happy": a bar between two "real" (old, librosa) beats was only 3 beats
    long where the surrounding tempo says 4 — a genuine missed detection.
    Cache moved to a new directory (``raw_beat_times_v2``) rather than
    invalidating the old one in place, so a stale entry can never silently
    survive this fix by matching on slug alone.

    Production consumer for the boundary-placement fix (2026-07-20,
    docs/research_sessions/boundary_snap_2026-07-20.md): a folded A×N
    section's reconstructed onsets phase-wobble vs the real beat (corpus
    mean 84ms wrapped-std) because one representative phrase is offset onto
    every repeat's span, and repeats aren't identically timed (intra-phrase
    rubato). Snapping each reconstructed onset to the nearest of THESE
    beats (app_shell.html's loadModel, gated on ``m.beatTimes``) measured
    84->27ms (-68%) on the matched set. Opt-in (HARMONIA_BOUNDARY_SNAP=1,
    default off — zero cost/risk when unset, matches every other flag in
    this file)."""
    audio_path = AUDIO_DIR / f"{slug}.m4a"
    if not audio_path.exists():
        return None
    _BEAT_TIMES_CACHE.mkdir(parents=True, exist_ok=True)
    cache = _BEAT_TIMES_CACHE / f"{slug}.json"
    if cache.exists():
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except ValueError:
            pass
    try:
        import librosa
        beat_times_raw = None
        try:
            from harmonia.models.chord_pipeline_v1 import _get_beatthis
            _f2b = _get_beatthis()
            if _f2b is not None:
                _bts, _dbs = _f2b(str(audio_path))
                _bts = [float(t) for t in _bts]
                if len(_bts) >= 4:
                    beat_times_raw = _bts
        except Exception as exc:  # noqa: BLE001 — never break the snap over an opt-in
            log.warning("raw-beat-times beatthis backend failed for %s (%s); librosa fallback", slug, exc)
        if beat_times_raw is None:
            import librosa.beat
            y, sr = librosa.load(str(audio_path), mono=True, sr=None)
            _tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            beat_times_raw = librosa.frames_to_time(beat_frames, sr=sr)
        times = [round(float(t), 4) for t in beat_times_raw]
    except Exception as e:
        log.warning("raw-beat-times extraction failed for %s (%s)", slug, e)
        return None
    try:
        cache.write_text(json.dumps(times), encoding="utf-8")
    except OSError:
        pass
    return times


# _BILLBOARD_CORPUS_FILES now lives in harmonia.serving.config (imported at
# module top); _load_billboard_corpus below and _billboard_video_to_track_id
# (moved to harmonia.serving.billboard_gt) both read it from there.
def _load_billboard_corpus() -> list[dict]:
    """The ~58-60 Billboard songs in the real-audio training corpus, each
    already duration-matched to a verified YouTube video (see
    docs/known_issues.md "Ship model to prod" thread — this is the exact
    corpus billboard_bp48_60_rollaug_v1 was trained on). Read-only: the two
    JSON files are disjoint keyed-by-track_id dicts produced by an earlier
    search pass and union to the full corpus — nothing is re-searched here."""
    merged: dict[str, dict] = {}
    for p in _BILLBOARD_CORPUS_FILES:
        try:
            merged.update(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError) as e:
            log.warning("billboard-corpus: could not read %s (%s)", p, e)

    # video_id -> chart filename, so we can flag songs already analysed/
    # corrected without re-running anything.
    vid_to_file: dict[str, str] = {}
    for fname, vid in _yt_video_ids.items():
        vid_to_file.setdefault(vid, fname)

    out = []
    for track_id, v in merged.items():
        best = v.get("best") or []
        if not best:
            continue
        vid = best[0]
        fname = vid_to_file.get(vid)
        status = "new"
        if fname:
            ann = _load_annotation(fname)
            status = "corrected" if ann.get("chords") else "analyzed"
        out.append({
            "track_id": track_id, "artist": v.get("artist", ""),
            "title": v.get("title", ""), "video_id": vid,
            "gt_dur": v.get("gt_dur"), "status": status,
            "file": fname or "",
        })
    out.sort(key=lambda r: (r["artist"] or "").lower())
    return out


@app.route("/api/billboard-corpus")
def api_billboard_corpus():
    """List the Billboard training-corpus songs for 'training mode' — the
    human-correction loop the /api/reinfer work above feeds. Each entry's
    video_id is a duration-verified YouTube match, ready to hand straight to
    /api/analyze (see app_shell.html's Training tab)."""
    return jsonify(songs=_load_billboard_corpus())


# ── Billboard ground-truth chord lookup cluster MOVED to
# harmonia.serving.billboard_gt (serving refactor, billboard-GT round):
# _billboard_video_to_track_id, _billboard_ds, _billboard_gt_cache,
# _save_gt_offset, _gt_chords_for_video_raw, _gt_chords_for_video — all
# re-imported at module top. _estimate_gt_offset (below) stays: it is a librosa
# onset heuristic used only by the /gt-offset-fix page route.


def _estimate_gt_offset(audio_path: Path, gt_raw: list[dict]) -> float:
    """First-strong-onset alignment guess for a starting offset (same
    heuristic as scratchpad/offset_final.py's diagnosis run): first onset
    above 40% of the first-30s max envelope, vs GT's first non-N/X chord
    onset. Cheap (single song, ~2-5s) but NOT reliable alone — confirmed
    wrong on intro-flourish songs in the original diagnosis (1/5 songs); it
    is only ever a pre-seeded starting point for human correction in
    /gt-offset-fix, never applied automatically."""
    import numpy as np
    import librosa
    y, sr = librosa.load(str(audio_path), sr=22050, mono=True)
    hop = 512
    oenv = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop)
    onsets = librosa.onset.onset_detect(onset_envelope=oenv, sr=sr, hop_length=hop, units="time", backtrack=True)
    if len(onsets) == 0:
        return 0.0
    ot = librosa.times_like(oenv, sr=sr, hop_length=hop)
    strengths = np.interp(onsets, ot, oenv)
    head = oenv[: int(30 * sr / hop)]
    thr = 0.4 * float(np.max(head)) if len(head) else 0.0
    strong = onsets[strengths > thr]
    first_strong = float(strong[0]) if len(strong) else float(onsets[0])
    real = [c for c in gt_raw if c["label"] not in ("N", "X")]
    gt_first = real[0]["t0"] if real else 0.0
    return round(first_strong - gt_first, 3)


# GET + POST /api/gt-offset/<track_id> MOVED to the harmonia.serving.api
# blueprint (serving refactor, billboard-GT round) — now that _save_gt_offset is
# extracted to harmonia.serving.billboard_gt, the POST route's last server-owned
# dep is gone. Both routes keep their bare endpoints via the name="" blueprint.


# _BAR1_OFFSETS_FILE + _load_bar1_offsets + _save_bar1_offset now live in
# harmonia.serving.state (Phase 6d), imported at module top.
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
    corrupt data, but is capped the same way to keep the range sane."""
    cap_bars = max(0, min(max(n_bars - 1, 0), 16))
    hi = bpb * cap_bars
    return -hi, hi


def _apply_bar1_offset_to_payload(payload: dict, offset_beats: int) -> dict:
    """Re-derive a chart payload's bar/beat numbering under a saved bar-1
    offset, WITHOUT re-baking the chart HTML.

    Fixes the gap where saving via /bar1-offset-fix only took effect on a
    song's *next* /api/analyze run — the main app chart view (served from
    _chart_model_for, which reads the already-baked HTML via
    payload_from_chart_html) never saw the correction until then. Every
    chart today was baked with offset_beats=0 (see bar1_offset_fix's
    docstring), so the baked ``bar``/``beat`` fields ARE ``abs_beat`` in
    disguise: abs_beat = bar*bpb + beat. Re-deriving from that and
    reapplying eff_beat = abs_beat - offset_beats keeps this a single
    source of truth for the shift math. No-op when offset_beats == 0.

    offset_beats has TWO distinct effects depending on magnitude, both
    handled by the same eff_beat computation:
    - |offset_beats| < bpb (sub-bar): a pure PHASE correction — which
      detected beat counts as beat 1 of bar 1. No chords are dropped, only
      renumbered.
    - |offset_beats| >= bpb (whole bars): an explicit INTRO-EXCLUSION. Any
      chord whose eff_beat < 0 (i.e. it sits before the new bar 1) is now
      DROPPED from the numbered chart rather than clamped into bar 0. This
      was the original 2026-07-17 bug: clamping via ``max(0, eff_beat //
      bpb)`` silently merged whatever fell before the offset onto bar 0,
      shrinking nBars by exactly the number of skipped bars while garbling
      bar 0's contents. Dropping instead of merging makes the "skip N bars
      of intro" case an explicit, visible, lossless-at-the-source operation
      (the underlying baked chart HTML is untouched — this transform is
      re-run fresh from it on every request, so the excluded bars are never
      actually deleted from disk, only hidden from THIS numbered view).
    Range is bounded by the caller via _bar1_offset_bounds() before this is
    invoked from a persisted value, so eff_beat<0 for EVERY chord (fully
    emptying the chart) should not happen in practice, but is handled
    gracefully here too (n_bars becomes 0, empty chart).
    """
    from scripts.render_youtube_chart import rebalance_near_boundary_onsets
    bpb = payload.get("bpb") or 4
    if not offset_beats:
        # No phase shift requested, but the baked (offset=0) bar assignment
        # can itself hit the near-boundary onset-crowding bug (see
        # rebalance_near_boundary_onsets's docstring — confirmed present even
        # at offset=0 on autumn_leaves, 11/329 bars) — fix it here too so
        # every song benefits, not only ones with a saved offset.
        chords = payload.get("chords") or []
        moved = rebalance_near_boundary_onsets(chords, bpb)
        if moved:
            n_bars = max((c["bar"] for c in chords), default=-1) + 1
            payload = {**payload, "chords": chords,
                       "nBars": max(int(payload.get("nBars") or 0), n_bars)}
        return payload
    old_sections = payload.get("sections") or []
    chords = payload.get("chords") or []
    new_chords = []
    max_bar = -1
    carry = None  # (chord, abs_beat) of the last dropped chord — its harmony
    # may still be sounding at the cut point if it was a HELD chord spanning
    # across the boundary (e.g. one long intro chord). Chords here are a
    # sparse "start of each change" list (held bars have no entry of their
    # own — see app_shell.html's loadModel / the 2026-07-17 held-bar bug), so
    # naively dropping every chord with eff_beat<0 can leave the new bar 0
    # with NO chord at all if the boundary lands mid-hold. Re-anchor that
    # last dropped chord at the new bar 0 instead, so its label survives —
    # otherwise this reintroduces the exact "silently blank cell" defect
    # class already fixed once in app_shell.html.
    first_kept_abs_beat = None
    for c in chords:
        abs_beat = int(c.get("bar", 0)) * bpb + int(c.get("beat", 0))
        eff_beat = abs_beat - offset_beats
        if eff_beat < 0:
            carry = (c, abs_beat)
            continue  # part of the excluded intro/pickup region — drop, don't merge into bar 0
        if first_kept_abs_beat is None:
            first_kept_abs_beat = abs_beat
        bar = eff_beat // bpb
        beat = eff_beat % bpb
        c = {**c, "bar": bar, "beat": beat}
        new_chords.append(c)
        max_bar = max(max_bar, bar)
    if carry is not None and (first_kept_abs_beat is None or first_kept_abs_beat > offset_beats):
        # The cut landed inside carry's hold — synthesize its continuation at
        # the new bar 0 beat 0. Estimate the cut's real time by linear
        # interpolation between carry's own t0 and the next surviving
        # chord's t0 (no per-beat tempo array available at this layer); with
        # nothing to interpolate against, fall back to carry's own t0.
        carry_chord, carry_abs_beat = carry
        t0 = float(carry_chord.get("t0", 0.0))
        if first_kept_abs_beat is not None:
            t1 = float(carry_chord.get("t1", t0))
            span = first_kept_abs_beat - carry_abs_beat
            frac = (offset_beats - carry_abs_beat) / span if span > 0 else 0.0
            t0 = t0 + frac * (t1 - t0)
        synth = {**carry_chord, "bar": 0, "beat": 0, "t0": t0}
        new_chords.insert(0, synth)
        max_bar = max(max_bar, 0)
    n_bars = max_bar + 1 if new_chords else 0
    # Shift the per-bar section-label array the same way: bar b's old label
    # moves to whatever bar its own abs_beat (b*bpb) now lands on; labels
    # whose bar fell in the excluded region are dropped along with it.
    new_sections = [""] * n_bars
    for old_bar, label in enumerate(old_sections):
        abs_beat = old_bar * bpb
        eff_beat = abs_beat - offset_beats
        if eff_beat < 0:
            continue
        bar = eff_beat // bpb
        if 0 <= bar < n_bars:
            new_sections[bar] = label
    # Same near-boundary onset-crowding fix as the offset==0 branch above —
    # a global phase shift that fixes the song's intro can (and on
    # autumn_leaves, does — 17/328 bars vs 11/329 at offset=0) make this
    # WORSE for mid-song passages, so it must be re-applied after shifting,
    # not just once at bake time.
    moved = rebalance_near_boundary_onsets(new_chords, bpb)
    if moved:
        n_bars = max((c["bar"] for c in new_chords), default=-1) + 1
        new_sections = (new_sections + [""] * n_bars)[:n_bars] if n_bars > len(new_sections) else new_sections
    payload = {**payload, "chords": new_chords, "nBars": n_bars, "sections": new_sections}
    return payload


@app.route("/api/bar1-offset/<slug>", methods=["GET"])
def api_bar1_offset_get(slug):
    """Current saved bar-1 phase offset (in beats) for a chart slug, if any."""
    return jsonify(_load_bar1_offsets().get(slug, {}))


@app.route("/api/bar1-offset/<slug>", methods=["POST"])
def api_bar1_offset_save(slug):
    """Persist a hand-set bar-1 offset. Body: {"offset_beats": int}.
    Takes effect the next time this song is analysed via /api/analyze (or
    re-rendered from a baked pipeline_chart) — see _save_bar1_offset.

    offset_beats is NOT limited to a sub-bar phase: whole multiples of bpb
    are the legitimate "exclude N bars of intro/pickup" operation (see
    _apply_bar1_offset_to_payload's docstring and docs/known_issues.md
    2026-07-17 "Yesterday align-tool" entry — a real case needed +8 beats
    = 2 bars to skip an instrumental intro that the beat grid had wrongly
    numbered as bars 1-2 of the song).

    2026-07-17, first pass of this endpoint reduced any offset mod bpb,
    which was WRONG for that case (silently coerced a deliberate 2-bar skip
    back to a no-op). Replaced with a defense-in-depth CLAMP instead of a
    modulo: get this chart's real bpb and current nBars, then clamp
    offset_beats into the safe range from _bar1_offset_bounds() (which caps
    at n_bars-1 bars so the chart can never be fully emptied, plus a sane
    absolute ceiling against typos). Unlike the old mod-bpb reduction, whole
    multiples of bpb within that range now persist unchanged — the safety
    net moved from "can only be a phase" to "_apply_bar1_offset_to_payload
    drops excluded chords instead of merging them into bar 0," which is
    what actually made large offsets dangerous in the first place."""
    data = request.get_json(force=True, silent=True) or {}
    try:
        offset_beats = int(data.get("offset_beats"))
    except (TypeError, ValueError):
        return jsonify(error="offset_beats must be an integer"), 400
    bpb, n_bars = 4, 1
    try:
        from harmonia.output.chart_model import payload_from_chart_html
        chart_path = PLOTS_DIR / f"inferred_{slug}.html"
        if chart_path.exists():
            chart_payload = payload_from_chart_html(chart_path)
            bpb = chart_payload.get("bpb") or 4
            n_bars = chart_payload.get("nBars") or 1
    except Exception:
        pass
    lo, hi = _bar1_offset_bounds(bpb, n_bars)
    clamped = max(lo, min(hi, offset_beats))
    _save_bar1_offset(slug, clamped)
    return jsonify(ok=True, slug=slug, offset_beats=clamped, requested=offset_beats, bounds=[lo, hi])


# ── User-drawn song-structure section labels (2026-07-17) ────────────────────
# The auto SSM sections (P.sections / P.sectionChips) are the MODEL's guess; this
# is an independent, hand-drawn layer where the user marks "this is A, this is B"
# on the chart. Persisted as its own sidecar so it never collides with the
# annotation doc's last-write-wins /api/annotations POST (which posts the whole
# {annotator,chords,merges} on every chord edit and would otherwise clobber it).
# Same small-file GET/POST shape as /api/bar1-offset. Doc: {"labels": {"<bar>":
# "<label>", ...}, "updated": iso}. A label at bar b starts a named section that
# runs until the next labeled bar. Purely additive; render-only on the client.
def _load_section_labels(filename: str) -> dict:
    try:
        doc = json.loads(_section_labels_path(filename).read_text(encoding="utf-8"))
        if isinstance(doc, dict) and isinstance(doc.get("labels"), dict):
            return doc
    except (OSError, ValueError):
        pass
    return {"labels": {}}


@app.route("/api/section-labels/<filename>", methods=["GET"])
def api_section_labels_get(filename):
    """Current hand-drawn section labels for a chart (empty {labels:{}} if none)."""
    return jsonify(_load_section_labels(filename))


# ── Chord-audio snippet: serve the EXACT [t0,t1) span of a song's downloaded
# audio so the Annotate tab can play the real recording of the chord being
# corrected (not the synthesized preview, and not a bar-snapped approximation).
# Reuses harmonia.models.audio_snippet (the bleed-fixed frame-clip convention,
# ffmpeg sample-accurate, zero padding). The audio is the one already retained
# at docs/audio/<slug>.m4a from analysis — nothing new is cached to disk, the
# WAV is streamed from memory and never written. Additive; GET-only.
def _audio_path_for_chart(filename: str):
    """Resolve the retained downloaded audio for an inferred_<slug>.html chart.
    Prefers the audio registry's exact filename; falls back to the slug path
    (docs/audio/<slug>.m4a), same mapping analysis writes."""
    meta = _yt_audio_meta.get(filename)
    if meta and meta.get("audio"):
        p = AUDIO_DIR / Path(meta["audio"]).name
        if p.exists():
            return p
    slug = filename.removeprefix("inferred_").removesuffix(".html")
    p = AUDIO_DIR / Path(f"{slug}.m4a").name
    return p if p.parent == AUDIO_DIR and p.exists() else None


@app.route("/api/chord-snippet/<filename>", methods=["GET"])
def api_chord_snippet(filename):
    """Exact [t0,t1) audio clip (WAV) of a chord span from the song's audio.
    Query params t0,t1 in seconds. Streamed from memory, zero padding, duration
    == t1-t0 to sub-ms (same standard as docs/bleed_verification_2026_07_16)."""
    from harmonia.models.audio_snippet import extract_snippet_wav
    audio_path = _audio_path_for_chart(filename)
    if audio_path is None:
        return jsonify(error=f"no downloaded audio for '{filename}'"), 404
    try:
        t0 = float(request.args.get("t0", ""))
        t1 = float(request.args.get("t1", ""))
    except (TypeError, ValueError):
        return jsonify(error="t0 and t1 (seconds) are required"), 400
    # Guard against a pathologically long span (whole-song download); a chord is
    # at most a few seconds. Cap at 30s and keep t0>=0 (handled in the helper).
    if t1 - t0 > 30.0:
        t1 = t0 + 30.0
    try:
        wav = extract_snippet_wav(audio_path, t0, t1)
    except ValueError as e:
        return jsonify(error=str(e)), 400
    except RuntimeError as e:
        log.warning("chord-snippet extraction failed for %s [%s,%s): %s",
                    filename, t0, t1, e)
        return jsonify(error="snippet extraction failed"), 500
    return Response(wav, mimetype="audio/wav",
                    headers={"Cache-Control": "no-store"})


@app.route("/library")
def library():
    """Your already-analyzed charts — a deliberately separate page from the
    search-first home, reached via the "Your charts" pill."""
    charts = sorted(PLOTS_DIR.glob("inferred_*.html"))
    items = [{"name": p.stem.replace("inferred_", "").replace("_", " ").title(),
              "file": p.name} for p in charts]
    page = render_template_string(LIBRARY_TEMPLATE, charts=items)
    return Response(page.replace("</head>", _PWA_HEAD + "</head>", 1), mimetype="text/html")


# 2026-07-18: no longer referenced — its only call site (serve_chart's
# baked-HTML swipe-nav injection) was removed when /chart/<file> became an
# unconditional redirect to the SPA (see serve_chart's docstring). Left in
# place rather than deleted since it's inert and reviving the baked-HTML
# path later (if ever) would want it back verbatim.


@app.route("/gt-align")
def gt_align():
    """GT alignment corrector: 4-bar focused waveform view with draggable chord
    markers, edge-gutter hit areas, continuous auto-pan on edge drag, timeline
    scrubbing, click-to-seek, and keyboard nudging.

    ?song=<slug>  →  drag iReal chords onto the audio to build GT alignment.
    """
    from html import escape

    slug = lookup_slug(request.args.get("song") or "autumn_leaves")

    chords, tempo = _load_ireal_alignment(slug)
    if not chords:
        return f"<p>No iReal chart for {slug}</p>", 404

    audio_path = AUDIO_DIR / f"{slug}.m4a"
    if not audio_path.exists():
        return f"<p>No audio for {slug}</p>", 404

    peaks_data = _waveform_peaks(slug)
    peaks = peaks_data.get("peaks", []) if peaks_data else []
    total_duration = max((c["t1"] for c in chords), default=30.0)

    # 4-bar focused window. Prefer tempo (iReal 4/4 assumption); fall back to 8 s.
    try:
        bpm = float(tempo) if tempo else 0.0
    except (TypeError, ValueError):
        bpm = 0.0
    window_duration = round(4 * 4 * 60.0 / bpm, 3) if bpm > 0 else 8.0
    window_duration = max(3.0, min(window_duration, total_duration or 8.0))

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no,viewport-fit=cover">
<title>GT Align: {escape(slug)}</title>
<style>
  * {{ box-sizing:border-box; -webkit-tap-highlight-color:transparent; }}
  html,body {{ margin:0; background:#0e1116; color:#e8edf4;
    font-family:system-ui,-apple-system,sans-serif; overflow:hidden; height:100%; }}
  #container {{ display:flex; flex-direction:column; height:100vh; }}
  header {{ padding:12px 16px; padding-right:108px; padding-left:76px;
    padding-top:calc(12px + env(safe-area-inset-top, 0px));
    background:#171c24; border-bottom:1px solid #2a3340; flex:0 0 auto; }}
  h1 {{ margin:0; font-size:15px; font-weight:700; }}
  header p {{ margin:6px 0 0; font-size:11.5px; color:#8b97a8; line-height:1.4; }}
  kbd {{ background:#0e1116; border:1px solid #2a3340; border-radius:3px;
    padding:0 4px; font-size:10px; font-family:inherit; color:#c3ccd8; }}

  #waveContainer {{ flex:1 1 auto; position:relative; background:#12161d;
    overflow:hidden; min-height:180px; touch-action:none; user-select:none; }}
  canvas {{ display:block; width:100%; height:100%; position:absolute; top:0; left:0; }}
  #markers {{ position:absolute; inset:0; z-index:10; pointer-events:none; }}

  /* Marker: wide invisible hit area, narrow visible stick centered inside it. */
  .chordMarker {{ position:absolute; top:0; height:100%; width:44px; margin-left:-22px;
    background:transparent; cursor:grab; z-index:20; pointer-events:auto;
    touch-action:none; }}
  .chordMarker::before {{ content:''; position:absolute; top:0; left:50%; height:100%;
    width:3px; margin-left:-1.5px; background:#00c9a7;
    box-shadow:0 0 0 1px rgba(0,201,167,0.25); transition:width .08s, background .08s; }}
  .chordMarker::after {{ content:''; position:absolute; top:50%; left:50%; width:13px;
    height:13px; margin:-6.5px 0 0 -6.5px; border-radius:50%; background:#00c9a7;
    box-shadow:0 1px 4px rgba(0,0,0,.5); transition:transform .08s, background .08s; }}
  .chordMarker:hover::before {{ width:5px; margin-left:-2.5px; }}
  .chordMarker:hover::after {{ transform:scale(1.25); }}
  .chordMarker.dragging {{ cursor:grabbing; }}
  .chordMarker.dragging::before, .chordMarker.selected::before {{ background:#6ef0d4; width:5px; margin-left:-2.5px; }}
  .chordMarker.dragging::after, .chordMarker.selected::after {{ background:#6ef0d4; transform:scale(1.35); }}
  .chordLabel {{ position:absolute; top:8px; left:50%; transform:translateX(-50%);
    background:#0e1116e6; padding:3px 7px; border-radius:4px; font-size:11px;
    font-weight:600; white-space:nowrap; border:1px solid #00c9a7; color:#00c9a7;
    pointer-events:none; }}
  .chordMarker.selected .chordLabel, .chordMarker.dragging .chordLabel {{
    border-color:#6ef0d4; color:#6ef0d4; }}
  .chordTime {{ display:block; font-size:9px; color:#8b97a8; font-weight:400; text-align:center; }}

  /* Edge-zone tint that appears while auto-panning */
  .edgeGlow {{ position:absolute; top:0; height:100%; width:70px; z-index:15; opacity:0;
    pointer-events:none; transition:opacity .12s; }}
  #edgeGlowL {{ left:0; background:linear-gradient(90deg,rgba(110,168,255,.30),transparent); }}
  #edgeGlowR {{ right:0; background:linear-gradient(270deg,rgba(110,168,255,.30),transparent); }}
  .edgeGlow.on {{ opacity:1; }}

  #timeline {{ flex:0 0 auto; position:relative; height:46px; background:#12161d;
    border-top:1px solid #2a3340; cursor:pointer; touch-action:none; user-select:none; }}
  #tlChords {{ position:absolute; inset:0; }}
  .tlChord {{ position:absolute; top:8px; width:1px; height:30px; background:rgba(0,201,167,.55); }}
  #tlWindow {{ position:absolute; top:0; height:100%; background:rgba(110,168,255,.14);
    border-left:2px solid #6ea8ff; border-right:2px solid #6ea8ff; pointer-events:none; }}
  #tlPlayhead {{ position:absolute; top:0; width:2px; height:100%; background:#ffd166; pointer-events:none; }}

  #controls {{ display:flex; gap:8px; padding:10px 12px; background:#171c24;
    border-top:1px solid #2a3340; flex-wrap:wrap; flex:0 0 auto; }}
  button {{ padding:8px 14px; background:#1e2530; border:1px solid #2a3340; color:#e8edf4;
    border-radius:6px; font:600 13px system-ui; cursor:pointer; transition:background .1s; }}
  button:hover {{ background:#252d3a; }}
  button:active {{ background:#00c9a7; color:#0e1116; }}
  button:disabled {{ opacity:.5; cursor:not-allowed; }}
  .spacer {{ flex:1; }}
  audio {{ width:100%; padding:8px 12px; background:#171c24; }}
  #info {{ padding:8px 12px 12px; background:#171c24; font-size:12px; color:#8b97a8;
    flex:0 0 auto; display:flex; gap:16px; align-items:center; }}
  .teal {{ color:#00c9a7; font-weight:600; }}
  .amber {{ color:#ffd166; font-weight:600; }}

  /* Floating Save button — anchored in the header strip (top-right), OUTSIDE
     #controls so it survives the mobile `#controls{{display:none}}` rule.
     A position:fixed descendant of a display:none ancestor is NOT rendered,
     which is why the previous in-#controls floating button never appeared. */
  #saveBtn {{ position:fixed; z-index:100;
    top:calc(env(safe-area-inset-top, 0px) + 10px);
    right:calc(env(safe-area-inset-right, 0px) + 12px);
    min-height:44px; padding:11px 18px;
    background:#00c9a7; color:#0e1116; border:none; border-radius:10px;
    font:700 15px system-ui; box-shadow:0 3px 10px rgba(0,0,0,.5); }}
  #saveBtn:hover {{ background:#1fd4b4; }}
  #saveBtn:active {{ background:#6ef0d4; }}
  #saveBtn:disabled {{ opacity:.5; }}

  /* Play/Pause transport — fixed top-left, mirror of #saveBtn. Lives OUTSIDE
     #controls so it survives the mobile `#controls{{display:none}}` rule and is
     always reachable (the native <audio> widget gets pushed below the fold on
     mobile Safari, so it can't be the only play affordance). */
  #playBtn {{ position:fixed; z-index:100;
    top:calc(env(safe-area-inset-top, 0px) + 10px);
    left:calc(env(safe-area-inset-left, 0px) + 12px);
    min-width:52px; min-height:44px; padding:11px 14px;
    background:#1e2530; color:#e8edf4; border:1px solid #2a3340; border-radius:10px;
    font:700 17px system-ui; box-shadow:0 3px 10px rgba(0,0,0,.5); cursor:pointer; }}
  #playBtn:hover {{ background:#252d3a; }}
  #playBtn:active {{ background:#00c9a7; color:#0e1116; }}
  #playBtn.playing {{ background:#00c9a7; color:#0e1116; border-color:#00c9a7; }}

  /* Hide non-essential controls on mobile (<600px). #saveBtn lives outside
     #controls, so it stays visible. */
  @media (max-width:600px) {{
    #prevBtn, #nextBtn, #resetBtn {{ display:none; }}
    #controls {{ display:none; }}
  }}
</style>
</head><body>
<div id="container">
  <header>
    <h1>🎼 GT Alignment · {escape(slug)}</h1>
    <p><kbd>▶</kbd> or <kbd>Space</kbd> to play. Drag the teal markers onto the audio
       onset; drag toward an edge to auto-pan. Click the waveform to seek. Click a
       marker then <kbd>←</kbd>/<kbd>→</kbd> to nudge ±100&nbsp;ms
       (<kbd>Shift</kbd> = ±10&nbsp;ms).</p>
  </header>

  <div id="waveContainer">
    <canvas id="canvas"></canvas>
    <div id="edgeGlowL" class="edgeGlow"></div>
    <div id="edgeGlowR" class="edgeGlow"></div>
    <div id="markers"></div>
  </div>

  <div id="timeline">
    <div id="tlChords"></div>
    <div id="tlWindow"></div>
    <div id="tlPlayhead"></div>
  </div>

  <audio id="audio" crossOrigin="anonymous" controls src="/audio/{escape(slug)}.m4a"></audio>

  <!-- Play/Pause and Save are direct children of #container (fixed-positioned,
       float over the header) so neither is hidden by the mobile
       #controls{{display:none}} rule. The native <audio controls> widget below
       is kept for fine scrubbing but is NOT the only play affordance. -->
  <button id="playBtn" aria-label="Play/Pause">▶</button>
  <button id="saveBtn">💾 Save</button>

  <div id="controls">
    <button id="prevBtn">◀ Prev</button>
    <button id="nextBtn">Next ▶</button>
    <span class="spacer"></span>
    <button id="resetBtn">↻ Reset</button>
  </div>
  <div id="info">
    <span>Window <span id="winLabel" class="teal">0:00–0:00</span></span>
    <span><span id="count">0</span> chords in view</span>
    <span id="status"></span>
  </div>
</div>

<script>
const CHORDS = {json.dumps(chords)};
const TOTAL = {total_duration};
const PEAKS = {json.dumps(peaks)};
const SLUG = '{slug}';
const WIN = {window_duration};          // window duration (seconds, ~4 bars)
const PAD = 30;                          // px gutter each side of the time axis
const EDGE = 70;                         // px edge zone that triggers auto-pan
const MAX_PAN = 9;                       // px/frame max auto-pan speed

const wave = document.getElementById('waveContainer');
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const audio = document.getElementById('audio');
const markersDiv = document.getElementById('markers');
const glowL = document.getElementById('edgeGlowL');
const glowR = document.getElementById('edgeGlowR');

let viewStart = 0;
let chordsDisplay = structuredClone(CHORDS);
let selectedIdx = -1;
let dirty = false;
let drag = null;          // {{ idx, el, pointerX }}
let raf = null;

const maxViewStart = () => Math.max(0, TOTAL - WIN);
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
function fmt(s) {{ const m = Math.floor(s/60), ss = Math.floor(s%60); return m+':'+(ss<10?'0':'')+ss; }}

// ---- Shared time <-> pixel mapping (gutter-inset so t=0 is never flush) ----
function axisW() {{ return Math.max(1, wave.clientWidth - 2*PAD); }}
function timeToX(t) {{ return PAD + ((t - viewStart) / WIN) * axisW(); }}
function xToTime(x) {{ return viewStart + ((x - PAD) / axisW()) * WIN; }}

// ---------------------------------------------------------------- rendering
function draw() {{
  const cssW = wave.clientWidth, cssH = wave.clientHeight;
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== cssW*dpr || canvas.height !== cssH*dpr) {{
    canvas.width = cssW*dpr; canvas.height = cssH*dpr;
  }}
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssW, cssH);

  // gutters (dim) so the active axis reads as inset
  ctx.fillStyle = '#0e1116';
  ctx.fillRect(0, 0, PAD, cssH);
  ctx.fillRect(cssW - PAD, 0, PAD, cssH);

  const mid = cssH/2, viewEnd = viewStart + WIN, n = PEAKS.length;
  if (n) {{
    ctx.fillStyle = '#3d4a5c';
    const x0 = Math.round(PAD), x1 = Math.round(cssW - PAD);
    for (let x = x0; x < x1; x++) {{
      const t = xToTime(x);
      const idx = clamp(Math.floor(t / TOTAL * n), 0, n-1);
      const h = Math.max(1, (PEAKS[idx]||0) * (cssH*0.44) * 2);
      ctx.fillRect(x, mid - h/2, 1, h);
    }}
  }}
  ctx.strokeStyle = '#2a3340'; ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(PAD, mid); ctx.lineTo(cssW-PAD, mid); ctx.stroke();

  // playhead
  const t = audio.currentTime || 0;
  if (t >= viewStart && t <= viewEnd) {{
    const x = timeToX(t);
    ctx.strokeStyle = '#ffd166'; ctx.lineWidth = 2;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, cssH); ctx.stroke();
  }}
}}

function layoutMarkers() {{
  const viewEnd = viewStart + WIN;
  let visible = 0;
  markersDiv.innerHTML = '';
  chordsDisplay.forEach((c, i) => {{
    if (c.t0 < viewStart - 0.001 || c.t0 > viewEnd + 0.001) return;
    visible++;
    const el = document.createElement('div');
    el.className = 'chordMarker' + (i === selectedIdx ? ' selected' : '') +
                   (drag && drag.idx === i ? ' dragging' : '');
    el.style.left = timeToX(c.t0) + 'px';
    el.dataset.idx = i;
    const label = document.createElement('div');
    label.className = 'chordLabel';
    label.innerHTML = c.label + '<span class="chordTime">' + c.t0.toFixed(2) + 's</span>';
    el.appendChild(label);
    el.addEventListener('pointerdown', startDrag);
    markersDiv.appendChild(el);
  }});
  document.getElementById('count').textContent = visible;
  document.getElementById('winLabel').textContent =
    fmt(viewStart) + '–' + fmt(Math.min(TOTAL, viewEnd));
}}

function updateTimeline() {{
  const tl = document.getElementById('timeline');
  const W = tl.clientWidth;
  const tlChords = document.getElementById('tlChords');
  tlChords.innerHTML = '';
  chordsDisplay.forEach(c => {{
    const el = document.createElement('div');
    el.className = 'tlChord';
    el.style.left = (c.t0 / TOTAL) * W + 'px';
    tlChords.appendChild(el);
  }});
  const win = document.getElementById('tlWindow');
  win.style.left = (viewStart / TOTAL) * W + 'px';
  win.style.width = (WIN / TOTAL) * W + 'px';
  const ph = document.getElementById('tlPlayhead');
  ph.style.left = ((audio.currentTime||0) / TOTAL) * W + 'px';
}}

function renderAll() {{ draw(); layoutMarkers(); updateTimeline(); }}

function setView(v) {{
  const nv = clamp(v, 0, maxViewStart());
  if (nv !== viewStart) {{ viewStart = nv; return true; }}
  return false;
}}

// ---------------------------------------------------------------- dragging
function startDrag(e) {{
  e.preventDefault();
  const idx = parseInt(e.currentTarget.dataset.idx);
  selectedIdx = idx;
  drag = {{ idx, el: e.currentTarget, pointerX: e.clientX }};
  e.currentTarget.classList.add('dragging');
  e.currentTarget.setPointerCapture?.(e.pointerId);
  document.getElementById('status').innerHTML = '<span class="amber">dragging…</span>';
  if (!raf) raf = requestAnimationFrame(tick);
}}

function tick() {{
  raf = null;
  if (!drag) {{ glowL.classList.remove('on'); glowR.classList.remove('on'); return; }}
  const rect = wave.getBoundingClientRect();
  const x = drag.pointerX - rect.left;

  // continuous auto-pan when the pointer sits inside an edge zone
  let pan = 0;
  const canL = viewStart > 0, canR = viewStart < maxViewStart();
  if (x < PAD + EDGE && canL) {{
    const depth = (PAD + EDGE - x) / EDGE;              // 0..1+
    pan = -MAX_PAN * clamp(depth, 0, 1.5);
  }} else if (x > rect.width - PAD - EDGE && canR) {{
    const depth = (x - (rect.width - PAD - EDGE)) / EDGE;
    pan = MAX_PAN * clamp(depth, 0, 1.5);
  }}
  glowL.classList.toggle('on', pan < 0);
  glowR.classList.toggle('on', pan > 0);
  if (pan) setView(viewStart + pan * (WIN / axisW()));   // px/frame -> seconds

  // marker follows the pointer's absolute time (works across pans)
  const t = clamp(xToTime(x), 0, TOTAL);
  if (t !== chordsDisplay[drag.idx].t0) {{ chordsDisplay[drag.idx].t0 = t; dirty = true; }}
  renderAll();
  raf = requestAnimationFrame(tick);
}}

document.addEventListener('pointermove', e => {{ if (drag) {{ drag.pointerX = e.clientX; }} }}, {{ passive:true }});

function endDrag() {{
  if (!drag) return;
  drag.el.classList.remove('dragging');
  drag = null;
  glowL.classList.remove('on'); glowR.classList.remove('on');
  if (raf) {{ cancelAnimationFrame(raf); raf = null; }}
  document.getElementById('status').textContent = dirty ? '● unsaved changes' : '';
  layoutMarkers();
}}
document.addEventListener('pointerup', endDrag);
document.addEventListener('pointercancel', endDrag);

// ---------------------------------------------------------- click-to-seek
wave.addEventListener('pointerdown', e => {{
  if (e.target.closest('.chordMarker')) return;   // marker handles its own drag
  const rect = wave.getBoundingClientRect();
  const t = clamp(xToTime(e.clientX - rect.left), 0, TOTAL);
  audio.currentTime = t;
  selectedIdx = -1;
  renderAll();
}});

// ------------------------------------------------------ timeline scrubbing
const tl = document.getElementById('timeline');
let tlDrag = false;
function tlSeek(clientX) {{
  const rect = tl.getBoundingClientRect();
  const frac = clamp((clientX - rect.left) / rect.width, 0, 1);
  setView(frac * TOTAL - WIN/2);
  renderAll();
}}
tl.addEventListener('pointerdown', e => {{ tlDrag = true; tl.setPointerCapture?.(e.pointerId); tlSeek(e.clientX); }});
tl.addEventListener('pointermove', e => {{ if (tlDrag) tlSeek(e.clientX); }});
tl.addEventListener('pointerup', () => {{ tlDrag = false; }});
tl.addEventListener('pointercancel', () => {{ tlDrag = false; }});

// ------------------------------------------------------------- navigation
function centerOn(t) {{ setView(t - WIN/2); }}
document.getElementById('prevBtn').addEventListener('click', () => {{ setView(viewStart - WIN*0.9); renderAll(); }});
document.getElementById('nextBtn').addEventListener('click', () => {{ setView(viewStart + WIN*0.9); renderAll(); }});

// ------------------------------------------------ keyboard nudge (±100/±10 ms)
document.addEventListener('keydown', e => {{
  if (selectedIdx < 0) return;
  if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
  e.preventDefault();
  const step = (e.shiftKey ? 0.01 : 0.1) * (e.key === 'ArrowLeft' ? -1 : 1);
  const c = chordsDisplay[selectedIdx];
  c.t0 = clamp(parseFloat((c.t0 + step).toFixed(3)), 0, TOTAL);
  dirty = true;
  if (c.t0 < viewStart + 0.2 || c.t0 > viewStart + WIN - 0.2) centerOn(c.t0);
  document.getElementById('status').textContent = '● unsaved changes';
  renderAll();
}});

// --------------------------------------------------------- play/pause transport
// The native <audio controls> widget sits at the bottom of a 100vh flex column
// and gets pushed below the fold on mobile Safari, so this always-visible button
// (and Space on desktop) is the primary way to start/stop playback.
const playBtn = document.getElementById('playBtn');
function togglePlay() {{
  if (audio.paused) {{
    audio.play().catch(err => {{
      document.getElementById('status').innerHTML =
        '<span style="color:#ff6b6b">✕ playback: ' + err.message + '</span>';
    }});
  }} else {{
    audio.pause();
  }}
}}
playBtn.addEventListener('click', togglePlay);
function syncPlayBtn() {{
  const playing = !audio.paused && !audio.ended;
  playBtn.textContent = playing ? '❚❚' : '▶';
  playBtn.classList.toggle('playing', playing);
}}
audio.addEventListener('play', syncPlayBtn);
audio.addEventListener('pause', syncPlayBtn);
audio.addEventListener('ended', syncPlayBtn);

// Space toggles playback (desktop). ArrowKeys stay reserved for marker nudging.
document.addEventListener('keydown', e => {{
  if (e.code !== 'Space' && e.key !== ' ') return;
  const t = e.target;
  if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
  e.preventDefault();
  togglePlay();
}});

// ------------------------------------------------- playback auto-follow view
audio.addEventListener('timeupdate', () => {{
  const t = audio.currentTime;
  if (!drag && !tlDrag && (t < viewStart || t > viewStart + WIN)) setView(t - WIN*0.3);
  draw(); updateTimeline();
}});

// -------------------------------------------------------------- save / reset
document.getElementById('saveBtn').addEventListener('click', async () => {{
  const btn = document.getElementById('saveBtn');
  const st = document.getElementById('status');
  // Monotonicity guard: t0 must be non-decreasing across chords. A swapped
  // pair (e.g. the bar-1/bar-2 regression) would silently corrupt the sidecar,
  // so refuse to save and point at the offender.
  for (let i = 1; i < chordsDisplay.length; i++) {{
    if (chordsDisplay[i].t0 < chordsDisplay[i - 1].t0) {{
      alert('Cannot save: chord ' + i + ' (' + chordsDisplay[i].label + ' @ ' +
            chordsDisplay[i].t0.toFixed(3) + 's) starts before chord ' + (i - 1) +
            ' (' + chordsDisplay[i - 1].label + ' @ ' +
            chordsDisplay[i - 1].t0.toFixed(3) + 's). Fix the ordering first.');
      return;
    }}
  }}
  btn.disabled = true;
  try {{
    const now = new Date().toISOString();
    // t1 of each chord is the next chord's t0 (song end for the last), so the
    // annotation stays gap-free even when a t0 was dragged.
    const body = {{
      annotator: 'gt-align',
      chords: chordsDisplay.map((c, i) => ({{
        bar: c.bar, beat: c.beat, section: c.section, label: c.label,
        t0: parseFloat(c.t0.toFixed(3)),
        t1: parseFloat((i + 1 < chordsDisplay.length ? chordsDisplay[i + 1].t0 : TOTAL).toFixed(3)),
        ts: now
      }})),
      merges: []
    }};
    const r = await fetch('/api/annotations/' + encodeURIComponent('irealb_' + SLUG + '.html'), {{
      method: 'POST', headers: {{ 'Content-Type': 'application/json' }}, body: JSON.stringify(body)
    }});
    if (!r.ok) throw new Error(r.statusText || r.status);
    dirty = false;
    st.innerHTML = '<span class="teal">✓ Saved</span>';
  }} catch (err) {{
    st.innerHTML = '<span style="color:#ff6b6b">✕ ' + err.message + '</span>';
  }} finally {{
    btn.disabled = false;
  }}
}});

document.getElementById('resetBtn').addEventListener('click', () => {{
  if (dirty && !confirm('Discard all edits and reset to the original iReal timing?')) return;
  chordsDisplay = structuredClone(CHORDS);
  selectedIdx = -1; dirty = false;
  document.getElementById('status').textContent = '';
  renderAll();
}});

window.addEventListener('beforeunload', e => {{ if (dirty) {{ e.preventDefault(); e.returnValue = ''; }} }});
window.addEventListener('resize', renderAll);

// ------------------------------------------------------------------- init
renderAll();
</script>
</body></html>"""

    return Response(page, mimetype="text/html")


@app.route("/api/yt-search", methods=["POST"])
def api_yt_search():
    """Search YouTube for songs to analyze — via yt-dlp's search extractor,
    no API key needed. Metadata-only (extract_flat), so this is a couple of
    seconds, not a download."""
    data = request.get_json(silent=True) or {}
    q = (data.get("q") or "").strip()
    if not q:
        return jsonify(error="Type something to search for.")

    try:
        import yt_dlp
    except ImportError:
        return jsonify(error="yt-dlp not installed in venv"), 500

    try:
        opts = {"quiet": True, "no_warnings": True,
                "extract_flat": "in_playlist", "skip_download": True}
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(f"ytsearch12:{q}", download=False)
        results = [
            {
                "id": e["id"],
                "title": e.get("title") or "Untitled",
                "uploader": e.get("uploader") or e.get("channel") or "",
                "duration": int(e.get("duration") or 0),
                "thumb": f"https://i.ytimg.com/vi/{e['id']}/mqdefault.jpg",
            }
            for e in (info.get("entries") or []) if e.get("id")
        ]
        return jsonify(results=results)
    except Exception as e:
        log.exception("YouTube search failed for %r", q)
        return jsonify(error=f"Search failed: {e}"), 500


@app.route("/api/annotations/<filename>", methods=["GET"])
def get_annotations(filename):
    """Current annotation sidecar for a chart (empty skeleton if none yet)."""
    return jsonify(_load_annotation(filename))


def _chart_audio_path(filename: str) -> Path | None:
    """Locate the cached local audio for a chart, or None."""
    meta = _yt_audio_meta.get(filename)
    if not meta:
        return None
    p = AUDIO_DIR / Path(meta["audio"]).name
    return p if p.exists() else None


def _chord_at(chords: list[dict], t: float) -> dict | None:
    for c in chords:
        if c["start_s"] <= t < c["end_s"]:
            return c
    return None


@app.route("/api/bar-merge-candidates/<filename>")
def api_bar_merge_candidates(filename):
    """2026-07-18 chord-robustness reframe, on-chart suggestions overlay.

    Serves precomputed bar-merge candidates (scratchpad/bar_merge_candidates
    .py's threshold+pairs output on the untrained 1-bar raw-chroma SSM — see
    docs/known_issues.md "REFRAME: bar-merge SSM pooling") for the new
    #suggest-mode-btn overlay in chart_interactive.py's chart page, which
    renders them as badges directly on the chart (distinct from, and
    additive to, the existing free-select "Merge sections" tool).

    Deliberately a THIN passthrough, not a live computation: reads whichever
    scratchpad/bar_merge_candidates_<stem>.json already exists (same file
    /debug/bar-merge-game consumes via bar_merge_game_data.json) rather than
    recomputing the SSM per request. This is the data-contract seam the
    candidate SOURCE can be swapped behind later (e.g. the parallel
    clustering-algorithm bake-off running tonight) without any client change,
    as long as the replacement keeps emitting the same
    {candidates:[{bars,spans,confidence,n_bars}]} shape.

    Returns 200 with an EMPTY candidate list (not 404) when no file exists
    for this song — scoped to one song for now (aretha_chain_of_fools), and
    an empty-but-valid response lets the UI say "no suggestions yet" instead
    of treating a not-yet-generated song as an error."""
    stem = filename[:-5] if filename.endswith(".html") else filename
    path = REPO / "scratchpad" / f"bar_merge_candidates_{stem}.json"
    if not path.exists():
        return jsonify(chart_file=filename, candidates=[], meta={})
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        log.warning("bar-merge-candidates: failed to read %s (%s)", path, e)
        return jsonify(chart_file=filename, candidates=[], meta={})
    return jsonify(data)


@app.route("/api/section-merge-candidates/<filename>")
def api_section_merge_candidates(filename):
    """2026-07-18, section-level (8-bar) analog of api_bar_merge_candidates
    above — same thin-passthrough contract, over the section-scale candidate
    files from `scratchpad/section_merge_candidates.py` (see
    docs/known_issues.md "SECTION-level (8-bar) repeat-detection suggestion
    tool"). Deliberately reuses the same conventions as the bar-level route
    (200+empty on missing file, no live recomputation) rather than
    introducing a new contract.

    Filename→stem differs from the bar-level route: those candidate files
    were generated keyed by the bare song slug (e.g. "autumn_leaves"), NOT
    the "inferred_" chart-file prefix the bar-level files happen to carry —
    so the "inferred_" prefix is stripped here if present, in addition to
    the ".html" suffix.

    `?grain=4` serves the 4-bar comparison file instead of the 8-bar
    default (the user's stated standard grain, see known_issues.md); any
    other value falls back to grain=8."""
    stem = filename[:-5] if filename.endswith(".html") else filename
    if stem.startswith("inferred_"):
        stem = stem[len("inferred_"):]
    grain = request.args.get("grain", "8")
    if grain not in ("4", "8"):
        grain = "8"
    path = REPO / "scratchpad" / f"section_merge_candidates_{stem}_grain{grain}.json"
    if not path.exists():
        return jsonify(chart_file=filename, candidates=[], meta={})
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        log.warning("section-merge-candidates: failed to read %s (%s)", path, e)
        return jsonify(chart_file=filename, candidates=[], meta={})
    return jsonify(data)


# iReal quality tail (as stored in the annotation sidecar's `q`) → the model's
# 5-way q5 family index (maj/min/dom/hdim/dim). Mirrors the chart's qualBucket().
def _ireal_q_to_q5(q: str | None) -> int:
    if not q:
        return 0
    if q.startswith("-7b5") or q.startswith("h"):
        return 3                                    # half-diminished
    if q.startswith("-") or q.startswith("m"):
        return 1                                    # minor (any)
    if q.startswith("o") or q.startswith("dim"):
        return 4                                    # diminished
    if q.startswith("^") or "maj7" in q or "M7" in q:
        return 0                                    # major (with maj7)
    if any(t in q for t in ("7", "9", "13", "alt")):
        return 2                                    # dominant
    return 0                                        # plain major / 6


@app.route("/api/reinfer/<filename>", methods=["POST"])
def api_reinfer(filename):
    """Re-run inference with the user's corrections as constraint factors
    (Mission 3, handoff §8). The client posts TIME-based constraints built from
    the payload it already holds:

        { "confirms": [{t0,t1,root,q5}, ...],          # chord-confirm / edit
          "merges":   [{"spans": [[t0,t1], ...]}, ...] # section-merge (P3) }

    Returns the re-decoded chart plus, for each chord, whether it CHANGED vs the
    same-config unconstrained decode — so the UI can highlight exactly what the
    user's corrections propagated to (not the whole chart). Re-decode is a
    PitchExtractor cache hit (stage-1 activations reused) so it's ~seconds."""
    data = request.get_json(silent=True) or {}
    raw_confirms = data.get("confirms") or []
    merges = data.get("merges") or []
    if not raw_confirms and not merges:
        return jsonify(error="No corrections to apply."), 400

    # Normalise confirms: each needs {t0, t1, root, q5}. The UI may send q5
    # directly (int 0..4) or the iReal quality tail `q` from the sidecar.
    confirms = []
    for c in raw_confirms:
        if "t0" not in c or "t1" not in c or "root" not in c:
            continue
        q5 = c.get("q5")
        if q5 is None:
            q5 = _ireal_q_to_q5(c.get("q"))
        confirms.append({"t0": float(c["t0"]), "t1": float(c["t1"]),
                         "root": int(c["root"]) % 12, "q5": int(q5)})

    audio = _chart_audio_path(filename)
    if audio is None:
        return jsonify(error="No cached audio for this chart — re-inference "
                             "needs the local audio (only analyzed songs have it)."), 404

    # Confirms open the propagation channel (progression transition factor);
    # merges are beat-level pooling and need no transition. See eval_user_*.py.
    tw = 2.0 if confirms else 0.0
    constraints = {"confirms": confirms, "merges": merges}

    import subprocess as _sp

    from harmonia.models.chord_pipeline_v1 import (
        NOTE, _BB_FAMILY_TO_SEV, _Q5_NAMES, infer_chords_billboard_v1, infer_chords_v1,
    )

    tmp_dir = Path(tempfile.mkdtemp(prefix="harmonia_reinfer_"))
    try:
        wav = tmp_dir / "a.wav"
        try:
            _sp.run(["ffmpeg", "-y", "-i", str(audio), "-ac", "1", "-ar", "22050",
                     str(wav)], check=True, capture_output=True, timeout=120)
        except (OSError, _sp.CalledProcessError, _sp.TimeoutExpired) as e:
            return jsonify(error=f"Audio transcode failed: {e}"), 500

        cache = tmp_dir            # shared cache_dir → 2nd infer is a stage-1 cache hit
        warnings: list[str] = []

        # Acoustic backend (2026-07-15, mirrors _run_analysis's /api/analyze
        # choice): prefer the Billboard real-audio checkpoint so a chart that
        # was FIRST analyzed with billboard_v1 doesn't silently switch to the
        # old POP909/jazz1460 ensemble the moment the user corrects a chord
        # (see docs/known_issues.md "Billboard model shipped to prod" — this
        # was the explicitly flagged gap). infer_chords_billboard_v1 has no
        # user_constraints/joint-decode machinery (its module comment: no
        # joint decode at all), so confirms are applied as direct label
        # overrides on the decoded chart instead of biasing the decoder —
        # cruder than the old joint_transition_weight propagation, but it's
        # exactly the correction the user just made, degrades gracefully, and
        # keeps the acoustic backend consistent with the original analysis.
        # Section-merges have no billboard equivalent (no pooling in this
        # backend). 2026-07-18 chord-robustness reframe: previously `merges`
        # would silently fall into the confirms-only billboard branch below
        # and land in `rejected` — a real bug (found via code-read, not a
        # user report) that made `pool_beat_evidence` unreachable from this
        # endpoint for EVERY real-audio song, since the Billboard checkpoint
        # is always present in prod and this branch was always taken first.
        # Fix: when merges are present, skip billboard and go straight to
        # the infer_chords_v1 branch below, which is the only backend with
        # working beat-pooling — same "degrade to the capability that
        # actually exists" principle the fallback branch already documents
        # for its own unequal-beat-count case. Confirms-only requests are
        # unaffected (still prefer billboard, unchanged).
        try:
            if merges:
                raise RuntimeError(
                    "merges present — routing to infer_chords_v1 for pool_beat_evidence")
            base = infer_chords_billboard_v1(wav, cache_dir=cache)
            backend_used = "billboard_bp48_60_rollaug_v1"

            cons_chords = [dict(c) for c in base.chords]
            for cf in confirms:
                mid = 0.5 * (cf["t0"] + cf["t1"])
                for c in cons_chords:
                    if c["start_s"] <= mid < c["end_s"]:
                        fam = _Q5_NAMES[cf["q5"]]
                        sev = _BB_FAMILY_TO_SEV.get(fam, fam)
                        c["label"] = f"{NOTE[cf['root'] % 12]}:{sev}"
                        c["confidence"] = 1.0
                        c["confidence_raw"] = 1.0
                        break
            if merges:
                warnings.append(
                    "billboard backend: section-merge not supported (no beat "
                    "pooling in this backend) — rejected, decode unpooled")

            class _Cons:
                pass
            cons = _Cons()
            cons.chords = cons_chords
            cons.global_key = base.global_key
            cons.tempo_bpm = base.tempo_bpm
        except RuntimeError as e:
            log.warning("reinfer: using infer_chords_v1 instead of billboard (%s)", e)
            backend_used = "infer_chords_v1 (fallback)"
            base = infer_chords_v1(wav, cache_dir=cache, joint_transition_weight=tw)
            # The pipeline DEGRADES GRACEFULLY when a constraint can't be applied —
            # e.g. pool_beat_evidence rejects a section-merge whose spans differ in
            # beat count ("equal musical length" is a v1 precondition). It logs a
            # warning and decodes unconstrained, so without this the endpoint would
            # answer 200 / n_changed=0 and the UI would report "Merged — one shared
            # reading" when nothing was pooled at all. Capture the warning and hand
            # it back so the client can say what actually happened.
            class _CatchRejections(logging.Handler):
                def emit(self, record):
                    if record.levelno >= logging.WARNING:
                        warnings.append(record.getMessage())

            pipe_log = logging.getLogger("harmonia.models.chord_pipeline_v1")
            handler = _CatchRejections()
            pipe_log.addHandler(handler)
            try:
                cons = infer_chords_v1(wav, cache_dir=cache, joint_transition_weight=tw,
                                       user_constraints=constraints)
            finally:
                pipe_log.removeHandler(handler)
        log.info("reinfer %s: acoustic backend = %s", filename, backend_used)
        base_ch = [c for c in base.chords if c["end_s"] > c["start_s"]]
        out = []
        diff = []
        for i, c in enumerate(cons.chords):
            mid = 0.5 * (c["start_s"] + c["end_s"])
            b = _chord_at(base_ch, mid)          # the same-config UNCONSTRAINED decode
            old_label = b["label"] if b else None
            changed = old_label != c["label"]
            entry = {"index": i, "label": c["label"], "start_s": c["start_s"],
                     "end_s": c["end_s"], "duration_beats": c.get("duration_beats", 1),
                     "confidence": c.get("confidence", 0.0),
                     "confidence_raw": c.get("confidence_raw", 0.0),
                     "changed": bool(changed)}
            out.append(entry)
            if changed:
                diff.append({
                    "index": i, "start_s": c["start_s"], "end_s": c["end_s"],
                    "old_label": old_label, "new_label": c["label"],
                    "old_confidence": (b.get("confidence") if b else None),
                    "new_confidence": c.get("confidence", 0.0),
                })
        rejected = [w for w in warnings if "rejected" in w.lower()]
        # 2026-07-19 (★ CHORD-ROBUSTNESS / BAR-MERGE, graceful per-GROUP
        # degradation): a merge group whose spans disagree on beat count now
        # pools its majority-length spans and EXCLUDES the mismatched ones
        # rather than dying wholesale. Surface those partial pools to the
        # client as their OWN field (distinct from `rejected`, which still
        # means "did not apply at all") so the UI can say "merged — N span(s)
        # left out for a beat-grid mismatch" honestly, instead of silently
        # pretending the whole group merged cleanly.
        partial = [w for w in warnings if "partially applied" in w.lower()]
        log.info("reinfer %s: %d confirms, %d merges, %d/%d chords changed%s%s",
                 filename, len(confirms), len(merges), len(diff), len(out),
                 f" (REJECTED: {rejected})" if rejected else "",
                 f" (PARTIAL: {partial})" if partial else "")
        return jsonify(chords=out, diff=diff, n_changed=len(diff),
                       key=cons.global_key, tempo_bpm=cons.tempo_bpm,
                       rejected=rejected, partial=partial)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    """Accept a YouTube URL, start a background analysis job, return job_id.

    Optional per-request override of the boundary-segmentation source, for
    interactive A/B testing without a server restart (2026-07-17): JSON field
    `seg_source` or query string `?seg_source=` / `?seg=`, either "nnls" or
    "musx". Anything else (missing, typo, other value) is ignored and falls
    back to the server-wide _ANALYZE_SEGMENT_SOURCE default — fails closed.
    """
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify(error="No URL provided"), 400
    if "youtube.com" not in url and "youtu.be" not in url:
        return jsonify(error="Please provide a YouTube URL"), 400

    seg_source_override = (data.get("seg_source") or request.args.get("seg_source")
                            or request.args.get("seg") or "").strip().lower()
    if seg_source_override not in ("nnls", "musx"):
        seg_source_override = None

    job_id = f"job_{int(time.time() * 1000)}"
    with _jobs_lock:
        _jobs[job_id] = {"status": "queued", "url": url, "message": "Queued"}

    t = threading.Thread(target=_run_analysis, args=(job_id, url),
                          kwargs={"seg_source_override": seg_source_override}, daemon=True)
    t.start()
    return jsonify(job_id=job_id)


@app.route("/api/job/<job_id>")
def api_job(job_id):
    with _jobs_lock:
        job = dict(_jobs.get(job_id, {"status": "error", "error": "Unknown job"}))
    return jsonify(job)


@app.route("/api/record-analyze", methods=["POST"])
def api_record_analyze():
    """Mic-recording analysis (2026-07-20): same job/analysing-screen path as
    YouTube, just with the audio already local instead of yt-dlp'd. Accepts a
    multipart upload (field "audio" — whatever MIME MediaRecorder produced,
    typically webm/opus on Chrome or mp4/aac on Safari) + optional "title".
    """
    f = request.files.get("audio")
    if f is None:
        return jsonify(error="No audio uploaded"), 400
    title = (request.form.get("title") or "").strip()

    # Filename STEM must be unique per upload — nnls_features.extract_bothchroma
    # AND musx_bass.musx_labels both cache keyed on the audio path's stem alone
    # (by design, see their own docstrings: it's what makes a re-analysed
    # YouTube video_id hit the cache). A literal "input.wav" reused across
    # requests from different fresh tmp_dirs still collides on that SAME stem
    # — confirmed live, 2026-07-21: a second recording silently served the
    # FIRST recording's cached features (a 45s clip's features on a 283s
    # song), truncating the whole analysis to ~45s of bogus bars. job_id is
    # already a millisecond timestamp — reuse it as the stem.
    job_id = f"job_{int(time.time() * 1000)}"
    tmp_dir = Path(tempfile.mkdtemp(prefix="harmonia_rec_"))
    raw = tmp_dir / f"{job_id}.upload"
    f.save(raw)
    wav = tmp_dir / f"{job_id}.wav"
    try:
        subprocess.run(
            ["ffmpeg", "-y", "-i", str(raw), "-ar", "44100", "-ac", "1", str(wav)],
            check=True, capture_output=True, timeout=60,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
        log.warning("record-analyze: transcode failed: %s", e)
        return jsonify(error="Could not decode the recorded audio"), 400

    default_title = title or f"Recording {time.strftime('%Y-%m-%d %H:%M:%S')}"

    def _run_then_cleanup():
        # `tmp_dir` here (harmonia_rec_*) is OURS, separate from _run_analysis's
        # own internal tmp_dir (which it already cleans up itself) — it holds
        # the uploaded blob + transcoded wav, and nothing deletes it unless we
        # do it here, after _run_analysis is done reading audio_path from it.
        try:
            _run_analysis(job_id, "local-recording",
                          local_audio_path=wav, local_title=default_title)
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)

    with _jobs_lock:
        _jobs[job_id] = {"status": "queued", "url": "local-recording", "message": "Queued"}
    t = threading.Thread(target=_run_then_cleanup, daemon=True)
    t.start()
    return jsonify(job_id=job_id)


@app.route("/api/jam/start", methods=["POST"])
def api_jam_start():
    """Start a new Jam Mode session (2026-07-20) — see harmonia/models/jam_mode.py
    for the loop-detection design. Returns a session_id the client attaches
    every mic chunk to."""
    from harmonia.models.jam_mode import JamSession
    session_id = f"jam_{int(time.time() * 1000)}"
    with _jam_sessions_lock:
        _jam_sessions[session_id] = JamSession(sr=44100)
    return jsonify(session_id=session_id)


@app.route("/api/jam/chunk", methods=["POST"])
def api_jam_chunk():
    """Append one mic-recorded chunk to a Jam session and return the freshly
    redecoded state in the SAME response — collapses upload+analyze+poll into
    one round-trip per chunk, so the client just awaits each chunk upload."""
    session_id = request.form.get("session_id") or ""
    with _jam_sessions_lock:
        sess = _jam_sessions.get(session_id)
    if sess is None:
        return jsonify(error="Unknown or expired jam session"), 404
    f = request.files.get("audio")
    if f is None:
        return jsonify(error="No audio uploaded"), 400

    tmp_dir = Path(tempfile.mkdtemp(prefix="harmonia_jam_"))
    try:
        raw = tmp_dir / "chunk.upload"
        f.save(raw)
        wav = tmp_dir / "chunk.wav"
        try:
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(raw), "-ar", "44100", "-ac", "1", str(wav)],
                check=True, capture_output=True, timeout=30,
            )
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            log.warning("jam/chunk: transcode failed: %s", e)
            return jsonify(error="Could not decode chunk"), 400
        import soundfile as sf
        y, sr = sf.read(wav)
        y = y.mean(1) if y.ndim > 1 else y
        sess.append(y)
        try:
            state = sess.update(tmp_dir / "session_buf.wav")
        except Exception as e:  # noqa: BLE001 — one bad chunk must never kill the session
            log.warning("jam/chunk: update failed: %s", e)
            state = sess.state()
        return jsonify(state)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@app.route("/api/jam/stop", methods=["POST"])
def api_jam_stop():
    """End a Jam session, freeing its buffer. Does NOT save anything to the
    library yet — a natural next step (bake the best-fit loop into a normal
    chart the way iReal import does), flagged but not built this pass."""
    session_id = (request.get_json(silent=True) or {}).get("session_id") or ""
    with _jam_sessions_lock:
        _jam_sessions.pop(session_id, None)
    return jsonify(ok=True)


@app.route("/api/tab-search", methods=["POST"])
def api_tab_search():
    """Search Ultimate Guitar by title + artist. Returns ranked results."""
    data = request.get_json(silent=True) or {}
    title  = (data.get("title")  or "").strip()
    artist = (data.get("artist") or "").strip()
    if not title:
        return jsonify(error="No title provided"), 400
    try:
        from harmonia.tab_fetcher import search_tabs
        results = search_tabs(title, artist, max_results=8)
    except ImportError as e:
        return jsonify(error=str(e)), 500
    except Exception as e:
        log.exception("tab-search failed")
        return jsonify(error=str(e)), 500

    return jsonify(results=[
        {
            "id":          r.id,
            "song_name":   r.song_name,
            "artist_name": r.artist_name,
            "tab_type":    r.tab_type,
            "rating":      round(r.rating, 3),
            "votes":       r.votes,
            "tonality":    r.tonality,
            "difficulty":  r.difficulty,
            "score":       round(r.score, 2),
            "tab_url":     r.tab_url,
        }
        for r in results
    ])


@app.route("/api/tab-fetch", methods=["POST"])
def api_tab_fetch():
    """Fetch chord content from a UG tab URL and render a chord-list page."""
    data = request.get_json(silent=True) or {}
    tab_url = (data.get("tab_url") or "").strip()
    if not tab_url:
        return jsonify(error="No tab_url provided"), 400
    # Accept optional metadata from the search result so the rendered page has
    # the correct title, rating, etc.
    song_name   = (data.get("song_name")   or "").strip()
    artist_name = (data.get("artist_name") or "").strip()
    rating      = float(data.get("rating") or 0)
    votes       = int(data.get("votes")    or 0)
    tonality    = (data.get("tonality")    or "").strip()

    try:
        from harmonia.tab_fetcher import TabResult, fetch_tab_chords
        stub = TabResult(id=0, song_name=song_name, artist_name=artist_name,
                         tab_type="Chords", rating=rating, votes=votes,
                         tonality=tonality, difficulty="", tab_url=tab_url, score=0)
        tab = fetch_tab_chords(stub)
    except ImportError as e:
        return jsonify(error=str(e)), 500
    except Exception as e:
        log.exception("tab-fetch failed")
        return jsonify(error=str(e)), 500

    if tab is None:
        return jsonify(error="Could not fetch tab content"), 502

    # Render a simple chord-sheet page and save it under docs/plots/
    import html as htmlmod
    slug = re.sub(r"[^a-z0-9]+", "_",
                  f"{tab.result.artist_name} {tab.result.song_name}".lower()).strip("_") or "tab"
    out = PLOTS_DIR / f"tab_{slug[:60]}.html"
    out.write_text(_render_tab_page(tab), encoding="utf-8")
    return jsonify(url=f"/chart/{out.name}")


@app.route("/api/render-tab", methods=["POST"])
def api_render_tab():
    """Fetch a UG tab and render it as an interactive HTML chord chart.

    Body: {tab_url, song_name, artist_name, tempo (optional, default 120),
           duration_s (optional, song duration in seconds for repeat expansion),
           video_id (optional, for YT sync)}
    Returns: {url: "/chart/<filename>"}
    """
    data = request.get_json(silent=True) or {}
    tab_url     = (data.get("tab_url") or "").strip()
    song_name   = (data.get("song_name") or "").strip()
    artist_name = (data.get("artist_name") or "").strip()
    tempo       = int(data.get("tempo") or 120)
    duration_s  = float(data.get("duration_s") or 0)
    vid         = (data.get("video_id") or "").strip()

    if not tab_url:
        return jsonify(error="No tab_url provided"), 400

    try:
        from harmonia.tab_fetcher import TabResult, fetch_tab_chords
        stub = TabResult(id=0, song_name=song_name, artist_name=artist_name,
                         tab_type="Chords", rating=0, votes=0, tonality="",
                         difficulty="", tab_url=tab_url, score=0)
        tab = fetch_tab_chords(stub)
        if tab is None:
            return jsonify(error="Could not fetch tab content"), 502

        from harmonia.tab_renderer import render_tab_chart
        slug = re.sub(r"[^a-z0-9]+", "_",
                      f"{artist_name}_{song_name}".lower()).strip("_") or "tab"
        out = PLOTS_DIR / f"tab_{slug[:60]}.html"
        render_tab_chart(tab.raw_content, title=song_name, artist=artist_name,
                         tempo=tempo, duration_s=duration_s, out_path=out)
    except ImportError as e:
        return jsonify(error=str(e)), 500
    except Exception as e:
        log.exception("render-tab failed")
        return jsonify(error=str(e)), 500

    if vid:
        _remember_video_id(out.name, vid)
    return jsonify(url=f"/chart/{out.name}")


def _render_tab_page(tab) -> str:
    """Render the raw UG tab content as a standalone HTML page."""
    import html as htmlmod, re as re_

    title = f"{tab.result.artist_name} — {tab.result.song_name}"
    key   = tab.result.tonality
    votes = tab.result.votes
    rating = tab.result.rating

    # Convert [ch]X[/ch] to styled spans and [tab]...[/tab] to <pre> blocks
    content = tab.raw_content
    content = htmlmod.escape(content)
    content = re_.sub(r'\[ch\](.*?)\[/ch\]',
                      r'<span class="ch">\1</span>', content)
    content = re_.sub(r'\[tab\](.*?)\[/tab\]',
                      r'<pre class="tab-block">\1</pre>',
                      content, flags=re_.DOTALL)
    content = re_.sub(r'\[(Verse[^\]]*|Chorus[^\]]*|Bridge[^\]]*|Intro[^\]]*|Outro[^\]]*|Pre[^\]]*|Hook[^\]]*)\]',
                      r'<div class="section-hd">[\1]</div>', content)
    content = content.replace('\n', '<br>')

    stars = '★' * round(rating) + '☆' * (5 - round(rating))
    chords_unique = ', '.join(tab.chords) if tab.chords else '—'

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{htmlmod.escape(title)} — Chords</title>
<style>
  :root{{--paper:#f7f3e9;--ink:#1c1c1c;--rule:#b9b09a;--accent:#8a2b2b;--faint:#8a8371;}}
  body{{background:var(--paper);color:var(--ink);margin:0;font-family:Georgia,'Times New Roman',serif;}}
  .sheet{{max-width:860px;margin:0 auto;padding:28px 32px 60px;}}
  h1{{text-align:center;font-size:26px;margin:0 0 4px;}}
  .meta{{text-align:center;color:var(--faint);font-style:italic;font-size:14px;margin-bottom:6px;}}
  .chord-summary{{background:#efe9d9;border:1px solid #e2dac4;border-radius:8px;padding:10px 16px;
    font-family:system-ui,sans-serif;font-size:13px;color:#4a4636;margin-bottom:18px;}}
  .chord-summary b{{color:var(--ink);}}
  .content{{font-family:monospace;font-size:14px;line-height:1.9;white-space:pre-wrap;word-break:break-word;}}
  .ch{{color:var(--accent);font-weight:700;font-size:15px;font-family:system-ui,sans-serif;}}
  .tab-block{{background:#f0ece0;border-left:3px solid var(--rule);padding:6px 12px;
    margin:4px 0;border-radius:0 6px 6px 0;overflow-x:auto;display:block;}}
  .section-hd{{font-family:system-ui,sans-serif;font-weight:700;font-size:13px;
    color:#5a4030;margin:14px 0 2px;}}
  .back{{display:inline-block;margin-bottom:18px;font-family:system-ui,sans-serif;
    font-size:13px;color:var(--accent);text-decoration:none;}}
  .back:hover{{text-decoration:underline;}}
  .stars{{color:#c07a20;}}
</style>
</head><body>
<div class="sheet">
  <a class="back" href="javascript:history.back()">← Back</a>
  <h1>{htmlmod.escape(title)}</h1>
  <p class="meta">
    <span class="stars">{stars}</span> {rating:.2f} ({votes} votes)
    {f'· Key: {htmlmod.escape(key)}' if key else ''}
    · <a href="{htmlmod.escape(tab.result.tab_url)}" target="_blank" style="color:var(--faint)">Ultimate Guitar ↗</a>
  </p>
  <div class="chord-summary">
    <b>Chords used:</b> {htmlmod.escape(chords_unique)}
  </div>
  <div class="content">{content}</div>
</div>
</body></html>"""


@app.route("/api/irealb-align", methods=["POST"])
def api_irealb_align():
    """Align an irealb:// chart to an inferred P.chords array and render.

    Body: {irealb_url, p_chords: [...], bpm (optional), video_id (optional)}
    Returns: {url: "/chart/<filename>", transpose, dtw_cost, exact_frac, ...}
    """
    data       = request.get_json(silent=True) or {}
    irealb_url = (data.get("irealb_url") or "").strip()
    p_chords   = data.get("p_chords") or []
    bpm        = data.get("bpm")
    bpm        = float(bpm) if bpm else None
    vid        = (data.get("video_id") or "").strip()

    if not irealb_url:
        return jsonify(error="No irealb_url provided"), 400
    if not p_chords:
        return jsonify(error="No p_chords provided"), 400

    try:
        import urllib.parse as _up
        from pyRealParser import Tune
        from harmonia.data.ireal_corpus import tune_to_mma
        from harmonia.irealb_aligner import align_irealb_to_inferred
        from harmonia.irealb_fetcher import render_irealb_chart, _esc

        decoded = _up.unquote(irealb_url)
        tunes = Tune.parse_ireal_url(decoded)
        if not tunes:
            return jsonify(error="No tunes found in irealb URL"), 400
        tune = tunes[0]
        mma = tune_to_mma(tune, tempo=int(bpm) if bpm else None)

        result = align_irealb_to_inferred(mma, p_chords, bpm_override=bpm)

        # Mission 6: structural QA gate — is this alignment coherent? which section
        # slipped?  Display-only (banner colour + suspect sections); never blocks.
        validation = None
        try:
            from harmonia.models.alignment_validator import validate_alignment
            validation = validate_alignment(result, p_chords)
        except Exception:
            log.exception("alignment validation failed (non-fatal)")

        # Render the iReal page with aligned timestamps replacing BPM-derived ones
        import json as _json
        p_json = _json.dumps({"chords": result.chords, "tempo": mma.tempo})

        html = render_irealb_chart(irealb_url,
                                   chart_offset_s=result.chords[0]["t0"] or 0.0
                                   if result.chords else 0.0,
                                   tempo_override=int(bpm) if bpm else None)

        # Patch P with aligned timestamps.
        # render_irealb_chart emits exactly one: <script>window.P = {...};</script>
        # Use a sentinel-based replace: find the marker and cut to the next </script>.
        import re as _re
        html = _re.sub(
            r"<script>window\.P\s*=\s*\{[^<]*\};</script>",
            f"<script>window.P = {p_json};</script>",
            html,
        )
        if p_json not in html:
            # Fallback if JSON had characters that confused the regex (rare)
            html = html + f"\n<script>window.P = {p_json};</script>"

        # Inject alignment stats banner
        stats = (f'<div style="font-family:system-ui,sans-serif;font-size:12px;'
                 f'color:#6b6050;text-align:center;margin:8px 0;padding:6px 12px;'
                 f'background:#efe9d9;border-radius:6px;">'
                 f'DTW aligned · +{result.transpose_semitones} semitones · '
                 f'{result.n_repeats}× form · '
                 f'exact {result.exact_frac:.0%} · family {result.family_frac:.0%} · '
                 f'mismatch {result.mismatch_frac:.0%}'
                 f'</div>')
        html = html.replace('<div class="ir-grid">', stats + '<div class="ir-grid">', 1)

        # Mission-6 verdict banner (green OK / yellow SUSPECT / red MISALIGNED /
        # gray UNVERIFIABLE).  Purely additive; names the suspect section(s).
        if validation is not None:
            _vc = {"OK": ("#1a7f37", "#dcffe4"), "SUSPECT": ("#8a6d00", "#fff4c2"),
                   "MISALIGNED": ("#b0202a", "#ffe0e0"),
                   "UNVERIFIABLE": ("#555", "#e8e8e8")}
            _fg, _bg = _vc.get(validation.verdict, ("#555", "#e8e8e8"))
            _sus = (" · slip: " + ", ".join(validation.suspect_sections)
                    if validation.suspect_sections else "")
            _sc = ("" if validation.align_score != validation.align_score
                   else f" · coherence {validation.align_score:.0%}")
            vbanner = (f'<div style="font-family:system-ui,sans-serif;font-size:13px;'
                       f'font-weight:600;color:{_fg};text-align:center;margin:8px 0;'
                       f'padding:6px 12px;background:{_bg};border-radius:6px;">'
                       f'alignment: {validation.verdict}{_sc}{_sus}</div>')
            html = html.replace('<div class="ir-grid">', vbanner + '<div class="ir-grid">', 1)

    except Exception as e:
        log.exception("irealb-align failed")
        return jsonify(error=str(e)), 500

    import urllib.parse as _up2
    try:
        slug_raw = _up2.unquote(irealb_url).split("=")[0].replace("irealb://", "")
        slug = re.sub(r"[^a-z0-9]+", "_", slug_raw.lower()).strip("_") or "irealb"
    except Exception:
        slug = "irealb"

    out = PLOTS_DIR / f"irealb_{slug[:60]}.html"
    out.write_text(html, encoding="utf-8")
    if vid:
        _remember_video_id(out.name, vid)

    return jsonify(
        url=f"/chart/{out.name}",
        transpose_semitones=result.transpose_semitones,
        dtw_cost=result.dtw_cost,
        n_repeats=result.n_repeats,
        exact_frac=result.exact_frac,
        family_frac=result.family_frac,
        mismatch_frac=result.mismatch_frac,
        validation=None if validation is None else {
            "verdict": validation.verdict,
            "align_score": (None if validation.align_score != validation.align_score
                            else round(validation.align_score, 3)),
            "suspect_sections": validation.suspect_sections,
            "repeat_consistency": (None if validation.repeat_consistency != validation.repeat_consistency
                                   else round(validation.repeat_consistency, 4)),
            "notes": validation.notes,
        },
    )


@app.route("/api/irealb-render", methods=["POST"])
def api_irealb_render():
    """Render an irealb:// URL as an interactive HTML chord chart.

    Body: {irealb_url, chart_offset_s (default 0), tempo (optional), video_id (optional)}
    Returns: {url: "/chart/<filename>"}
    """
    data           = request.get_json(silent=True) or {}
    irealb_url     = (data.get("irealb_url") or "").strip()
    chart_offset_s = float(data.get("chart_offset_s") or 0)
    tempo          = data.get("tempo")
    tempo          = int(tempo) if tempo else None
    vid            = (data.get("video_id") or "").strip()

    if not irealb_url:
        return jsonify(error="No irealb_url provided"), 400

    try:
        from harmonia.irealb_fetcher import render_irealb_chart
        html = render_irealb_chart(
            irealb_url,
            chart_offset_s=chart_offset_s,
            tempo_override=tempo,
        )
    except Exception as e:
        log.exception("irealb-render failed")
        return jsonify(error=str(e)), 500

    # Derive a slug from the irealb URL title field (first segment after irealb://)
    import urllib.parse as _up
    try:
        decoded = _up.unquote(irealb_url)
        slug_raw = decoded.split("=")[0].replace("irealb://", "")
        slug = re.sub(r"[^a-z0-9]+", "_", slug_raw.lower()).strip("_") or "irealb"
    except Exception:
        slug = "irealb"

    out = PLOTS_DIR / f"irealb_{slug[:60]}.html"
    out.write_text(html, encoding="utf-8")
    if vid:
        _remember_video_id(out.name, vid)
    return jsonify(url=f"/chart/{out.name}")


@app.route("/api/irealb-import", methods=["POST"])
def api_irealb_import():
    """Import an iReal community chart into the library proper (SPA "Import"
    button, 2026-07-20) — a DIFFERENT route from /api/irealb-render above on
    purpose: that one feeds the older YouTube-alignment overlay tool and
    writes the older window.P.chords shape those pages still expect; this one
    builds a real ChordChart (harmonia.irealb_fetcher.irealb_tune_to_chord_
    chart) and renders it through the EXACT SAME chart_to_interactive_inputs
    / render_interactive pipeline a real analysis uses, so the result is a
    normal inferred_*.html — opens in the SPA, sorts/deletes/exports like any
    other chart. (The old route's output is NOT SPA-compatible: /chart/<file>
    now unconditionally redirects into the SPA's ChartModel adapter, which
    chokes on that older shape — every import silently failed to open until
    this route existed.)

    Body: {irealb_url}. Returns {url: "/chart/<filename>"}.
    """
    data = request.get_json(silent=True) or {}
    irealb_url = (data.get("irealb_url") or "").strip()
    if not irealb_url:
        return jsonify(error="No irealb_url provided"), 400
    try:
        from harmonia.irealb_fetcher import irealb_tune_to_chord_chart
        from scripts.render_youtube_chart import chart_to_interactive_inputs
        from harmonia.output.chart_interactive import render_interactive

        chart = irealb_tune_to_chord_chart(irealb_url)
        title = chart.source_path.removeprefix("irealb:") or "Imported chart"
        slug = chart_slug(title) or "irealb"
        out = PLOTS_DIR / f"inferred_ireal_{slug}.html"
        chart_obj, chord_dicts = chart_to_interactive_inputs(chart, title, "imported from iReal Pro")
        render_interactive(chart_obj, chord_dicts, out, bars_per_row=4, sections=chart.sections)
        # Mark this payload's sections as ground-truth (came straight from
        # iReal's own *A/*B/*C markers) so to_chart_model's loop-fold
        # heuristic — built to recover structure from UNTRUSTED barlocked
        # audio-decode boundaries — doesn't run on it. Confirmed 2026-07-20:
        # without this it mis-detected a legitimate A(x2)/B/C form as "3 reps
        # of one loop" (an exactly-cyclic import is the one case that
        # heuristic, tuned for noisy real-audio decodes, wasn't built for).
        html = out.read_text(encoding="utf-8")
        html = re.sub(r'^(const P = \{)', r'\1"sections_trusted":true,', html, count=1, flags=re.M)
        out.write_text(html, encoding="utf-8")
        return jsonify(url=f"/chart/{out.name}")
    except Exception as e:
        log.exception("irealb-import failed")
        return jsonify(error=str(e)), 500


def _run_analysis(job_id: str, url: str, seg_source_override: str | None = None,
                   local_audio_path: "Path | None" = None,
                   local_title: str | None = None) -> None:
    """``local_audio_path`` (2026-07-20, mic recording): analyze an already-
    local audio file instead of downloading one — skips the yt-dlp block
    entirely, everything downstream (inference, baking, audio transcode,
    library registration) is unchanged since it already only depends on
    having *some* ``audio_path`` + ``video_title``, not on the source being
    YouTube. ``url`` is then a synthetic, non-YouTube string purely for
    ``_extract_video_id`` (which safely returns "" for it, same as any other
    non-YouTube URL) — no video id / thumbnail gets attached."""
    def update(status, message="", **kw):
        with _jobs_lock:
            _jobs[job_id].update(status=status, message=message, **kw)

    # `stage` + `results` drive the app's Analysing screen. They are the real
    # steps, not a scripted animation: stage 1 is one call into infer_chords_v1
    # (Basic Pitch → beats → sections → key → chord HMM), which reports nothing
    # intermediate, so it stays lit for as long as the decode actually takes and
    # its result chip is filled from what the decode returned.
    results = (["preparing your recording"] if local_audio_path is not None
               else ["downloading from YouTube"]) + [
               "notes, beats, sections, key, chords", "laying out the lead sheet"]

    def stage(i: int, result: str | None = None, **kw):
        if result is not None and i > 0:
            results[i - 1] = result
        update("running", "", stage=i, results=list(results), **kw)

    # Progressive-analysis wiring (2026-07-20): infer_chords_v1's progress_cb
    # fires mid-decode, well before the job would otherwise next touch the job
    # record — see docs/inference_pipeline_timing_and_animation_scope.md. Each
    # event just merges its fields into the job dict; the client
    # (app_shell.html) computes its own bar grid from tempo+chords rather than
    # requiring the full chart_model pipeline mid-flight, so this preview never
    # needs sections. Best-effort by construction (infer_chords_v1 already
    # swallows progress_cb exceptions) — a failure here can't break analysis.
    def on_progress(kind: str, data: dict) -> None:
        field = {
            "beats": lambda: {"tempo_bpm": data["tempo_bpm"],
                               "time_signature": data["time_signature"]},
            "key": lambda: {"key_name": data["key"]},
            "draft": lambda: {"draft_chords": data["chords"]},
            # "fold"/"n_folds" present = an intermediate music-x-lab ensemble
            # fold (2026-07-20, fold-by-fold reveal); absent = the true final
            # result. Always set both so the client can tell them apart.
            "chords": lambda: {"final_chords": data["chords"],
                                "musx_fold": data.get("fold"),
                                "musx_n_folds": data.get("n_folds")},
            "sections": lambda: {"n_sections": data["n_sections"]},
        }.get(kind)
        if field is None:
            return
        with _jobs_lock:
            _jobs[job_id].update(field())

    tmp_dir = Path(tempfile.mkdtemp(prefix="harmonia_yt_"))
    _bg_owns_tmp = False  # set True once the post-"done" finalizer thread (which
                          # reads audio_path from tmp_dir) takes over cleanup.
    try:
        stage(0)

        if local_audio_path is not None:
            audio_path = local_audio_path
            video_title = local_title or "Recording"
            duration = 0
            try:
                import soundfile as _sf
                _info = _sf.info(str(audio_path))
                duration = int(_info.frames / _info.samplerate)
            except Exception:
                duration = 0
        else:
            # Download via yt-dlp Python API (no subprocess, no shell quoting issues)
            try:
                import yt_dlp
            except ImportError:
                update("error", error="yt-dlp not installed in venv")
                return

            audio_path: Path | None = None
            video_title = url

            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": str(tmp_dir / "%(id)s.%(ext)s"),
                # unchanged from before — the inference pipeline's audio loader
                # expects whatever this has always produced. A separate m4a copy
                # is transcoded below, only for browser playback.
                "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "best"}],
                "quiet": True,
                "no_warnings": True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_title = info.get("title", url)
                # Find downloaded file
                audio_exts = {".opus", ".m4a", ".mp3", ".webm", ".ogg", ".flac", ".wav", ".aac"}
                files = sorted(tmp_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
                for f in files:
                    if f.suffix in audio_exts:
                        audio_path = f
                        break

            if audio_path is None:
                update("error", error="yt-dlp did not produce an audio file")
                return

            duration = 0
            try:
                duration = int(info.get("duration") or 0)
            except (TypeError, ValueError):
                duration = 0
        stage(1, result=(f"{duration // 60}:{duration % 60:02d} of audio" if duration
                         else "audio fetched"), title=video_title, duration_s=duration)

        # Acoustic backend (2026-07-15): prefer the Billboard real-audio
        # checkpoint (data/models/billboard_bp48_60_rollaug_v1.pt — 58 real
        # Billboard songs, zero alignment error, root 54.3%/quality balanced
        # ~19.7% held-out) for freshly-analyzed real audio, since this is the
        # only model in the repo actually validated on real (non-synthetic)
        # recordings — see docs/known_issues.md 2026-07-15 "Billboard root
        # accuracy campaign". Falls back to the Gen-2 ensemble (infer_chords_v1,
        # tuned on POP909/jazz1460) if the checkpoint is missing so this route
        # never hard-fails.
        from harmonia.models.chord_pipeline_v1 import (
            infer_chords_billboard_v1, infer_chords_v1,
        )

        pipeline_chart = None
        backend_used = None

        # New default (2026-07-17 deploy): NNLS-24 features + music-x-lab routed
        # bass. Additive + reversible — controlled by _ANALYZE_FEATURE_FRONTEND
        # (env HARMONIA_ANALYZE_FRONTEND). Any failure here falls through to the
        # exact prior Billboard→infer_chords_v1 chain below, so this can never
        # hard-break analysis for users.
        if _ANALYZE_FEATURE_FRONTEND == "nnls24":
            # Per-request A/B override (2026-07-17) beats the env-var default
            # for this one job only; server-wide default is untouched.
            segment_source_used = seg_source_override or _ANALYZE_SEGMENT_SOURCE
            try:
                pipeline_chart = infer_chords_v1(
                    audio_path,
                    cache_dir=Path(runtime.ARGS.cache_dir),
                    feature_frontend="nnls24",
                    bass_frontend=_ANALYZE_BASS_FRONTEND,
                    quality_frontend=_ANALYZE_QUALITY_FRONTEND,
                    segment_source=segment_source_used,
                    beat_period_mode=_ANALYZE_BEAT_PERIOD_MODE,
                    progress_cb=on_progress,
                )
                backend_used = (f"infer_chords_v1(nnls24, bass={_ANALYZE_BASS_FRONTEND}, "
                                f"quality={_ANALYZE_QUALITY_FRONTEND}, "
                                f"seg={segment_source_used}"
                                f"{' [override]' if seg_source_override else ''})")
            except Exception as e:  # noqa: BLE001 — defensive: never break analyze
                log.warning("analysis %s: nnls24 front-end failed (%s) — falling "
                            "back to Billboard/BP48 chain", job_id, e)
                pipeline_chart = None

        # Prior behaviour (unchanged) — also the fallback if nnls24 is disabled or
        # raised above: Billboard real-audio checkpoint, then Gen-2 ensemble.
        if pipeline_chart is None:
            try:
                pipeline_chart = infer_chords_billboard_v1(
                    audio_path, cache_dir=Path(runtime.ARGS.cache_dir),
                )
                backend_used = "billboard_bp48_60_rollaug_v1"
            except RuntimeError as e:
                log.warning("billboard backend unavailable (%s) — falling back to infer_chords_v1", e)
                pipeline_chart = infer_chords_v1(
                    audio_path,
                    cache_dir=Path(runtime.ARGS.cache_dir),
                )
                backend_used = "infer_chords_v1 (fallback)"
        log.info("analysis %s: acoustic backend = %s", job_id, backend_used)

        stage(2, result=(f"{pipeline_chart.global_key} · {pipeline_chart.tempo_bpm:.0f} bpm"
                         f" · {len(pipeline_chart.chords)} chords"))

        from scripts.render_youtube_chart import chart_to_interactive_inputs
        from harmonia.output.chart_interactive import render_interactive

        source_desc = f"inferred from YouTube · {url}"
        slug = re.sub(r"[^a-z0-9]+", "_", video_title.lower()).strip("_") or "yt"
        slug = slug[:60]
        bar1_offset = _load_bar1_offsets().get(slug, {}).get("offset_beats", 0)
        # The bar-locked section pass OWNS the bar-1 phase: it detects the intro
        # and locks every section boundary to a 4-bar phrase on the beat-0 grid.
        # A saved bar1 offset (typically set on an earlier acoustic chart, and
        # commonly a non-multiple of 4) would shift that grid and knock the
        # phrase-locked boundaries off the 4-bar lattice (user report 2026-07-19:
        # offset=9 dropped the intro + first A). So when barlocked is active and
        # actually produced an intro/section structure, ignore the stale offset.
        if (os.environ.get("HARMONIA_SECTION_MODE", "barlocked") == "barlocked"
                and getattr(pipeline_chart, "sections", None)):
            # barlocked OWNS the bar-1 phase.  With HARMONIA_GRID_ANCHOR=structure
            # it computed a structure-anchored downbeat phase; use THAT as the
            # renderer's bar1_offset so chords and sections share the anchored grid
            # (else use 0 — the beat-0 grid barlocked pooled on).  Either way a
            # stale saved offset must NOT survive to the serve path.
            anchor = int(getattr(pipeline_chart, "grid_anchor_beats", 0) or 0)
            if anchor != bar1_offset:
                log.warning("analysis %s: barlocked owns bar-1 phase — baking "
                            "bar1_offset=%d (structure anchor), was saved %d",
                            job_id, anchor, bar1_offset)
            bar1_offset = anchor
            # The chart is BAKED with this offset; the serve path
            # (_apply_bar1_offset_to_payload) must NOT re-apply it — persist 0 so
            # it does not double-shift the already-anchored chart.
            _save_bar1_offset(slug, 0)
        chart_obj, chord_dicts = chart_to_interactive_inputs(
            pipeline_chart, video_title, source_desc, bar1_offset_beats=bar1_offset,
        )

        out = PLOTS_DIR / f"inferred_{slug}.html"
        render_interactive(chart_obj, chord_dicts, out, bars_per_row=4,
                           sections=pipeline_chart.sections)

        vid = _extract_video_id(url)
        if vid:
            _remember_video_id(out.name, vid)

        # Keep the audio we already downloaded instead of throwing it away —
        # the docked player plays this back locally rather than re-embedding
        # a YouTube iframe. Transcode to AAC/m4a: whatever format the
        # inference pipeline's loader wants (the download above, unchanged)
        # isn't necessarily one iOS Safari's <audio> can play natively.
        try:
            audio_dest = AUDIO_DIR / f"{slug[:60]}.m4a"
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(audio_path), "-vn",
                 "-acodec", "aac", "-b:a", "128k",
                 # +faststart moves the moov atom to the FRONT of the file.
                 # Without it ffmpeg writes moov after mdat (audio data first,
                 # index last) — iOS Safari's <audio> then gets duration from
                 # a metadata probe but never manages to actually fetch the
                 # sample data via Range requests: readyState stalls at
                 # HAVE_METADATA, networkState goes idle, buffered stays
                 # empty, forever (confirmed on-device 2026-07-20 — this is
                 # THE bug behind "button toggles, no sound, time stuck at
                 # 0:00"). Desktop/Chromium players are lenient about moov
                 # position and played the same file fine, which masked this.
                 "-movflags", "+faststart", str(audio_dest)],
                check=True, capture_output=True, timeout=120,
            )
            thumb_url = f"https://i.ytimg.com/vi/{vid}/mqdefault.jpg" if vid else ""
            _remember_audio(out.name, f"/audio/{audio_dest.name}", thumb_url)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            log.warning("Could not persist/transcode audio for %s: %s", out.name, e)

        # Mark the job done NOW — the chart the user is waiting on is fully
        # rendered and the playback audio is transcoded. Everything below
        # (Basic Pitch persist + iReal fetch) produces *annotator-facing*
        # sidecar artifacts that the freshly-opened chart never reads, so it
        # is moved to a background thread AFTER "done" (2026-07-17 perf fix,
        # docs/known_issues.md). Previously the Basic Pitch persist ran an
        # ~11–18 s COLD Basic Pitch pass here, inside "Drawing the chart", on
        # every analyze: its "cache hit" comment was WRONG — the live nnls24
        # front-end (_infer_nnls24) never calls PitchExtractor, and the
        # extract is temp-path-keyed so it never warms — pure user-visible
        # wait for a .npz that only a future annotator-reload surface (arch-
        # extensions §13) will read.
        results[2] = f"{chart_obj.n_bars} bars"
        update("done", url=f"/chart/{out.name}", stage=3,
               results=list(results), title=video_title)

        # Background post-processing. Owns tmp_dir cleanup (Basic Pitch reads
        # audio_path, which lives inside tmp_dir) so the outer finally must NOT
        # rmtree once this thread is launched. Fully guarded: any failure here
        # is logged and swallowed — the user already has their chart, so a
        # background hiccup must never surface as a user-facing error.
        def _finalize_bg(job_id=job_id, tmp_dir=tmp_dir, audio_path=audio_path,
                         slug=slug, out=out, video_title=video_title,
                         pipeline_chart=pipeline_chart):
            try:
                # Persist chroma/pitch activations, addressable by slug — for a
                # later "re-score bars against pooled chroma" annotator surface
                # (arch-extensions §13). This is a full Basic Pitch run (the
                # nnls24 path never populated PitchExtractor's cache), hence why
                # it belongs off the hot path.
                try:
                    from harmonia.models.stage1_pitch import PitchExtractor
                    activations = PitchExtractor(cache_dir=Path(runtime.ARGS.cache_dir)).extract(audio_path)
                    activations.save(PITCH_CACHE_DIR / f"{slug[:60]}.npz")
                except Exception as e:  # noqa: BLE001 — best-effort, never user-facing
                    log.warning("bg: could not persist pitch/chroma cache for %s: %s", out.name, e)

                # Fetch iReal chart from community if available, so the annotator
                # tool (which needs docs/plots/irealb_<slug>.html) doesn't 404.
                try:
                    from harmonia.irealb_fetcher import search_community, render_irealb_chart
                    results_ir = search_community(video_title, max_results=1)
                    if results_ir:
                        irealb_url = results_ir[0]["irealb_url"]
                        html_ir = render_irealb_chart(irealb_url, chart_offset_s=0.0,
                                                      tempo_override=int(round(pipeline_chart.tempo_bpm)))
                        ir_out = PLOTS_DIR / f"irealb_{slug[:60]}.html"
                        ir_out.write_text(html_ir, encoding="utf-8")
                        _remember_ireal_url(out.name, irealb_url)
                        log.info("bg: saved iReal chart for %s (%s)", out.name, irealb_url)
                except Exception as e:  # noqa: BLE001 — best-effort, never user-facing
                    log.warning("bg: could not fetch/render iReal chart for %s: %s", video_title, e)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        threading.Thread(target=_finalize_bg, daemon=True,
                          name=f"harmonia-finalize-{job_id}").start()
        _bg_owns_tmp = True

    except Exception as e:
        log.exception("Analysis failed for %s", url)
        update("error", error=str(e))
    finally:
        # The background finalizer owns tmp_dir once launched; only clean up
        # here if we never got that far (early error, or nnls24 disabled path).
        if not _bg_owns_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Index template ─────────────────────────────────────────────────────────────







# ── Rebuilt annotator (simple linear flow, iPhone Safari first) ─────────────────


# ── Manual chord-alignment annotator (iPhone-first) ─────────────────────────────
#
# GET /annotator?song=<slug>
#   Loads the iReal GT chords + their Mission-1 initial alignment (DTW t0/t1
#   from docs/plots/irealb_<slug>.html) and the song audio, and serves a
#   touch-optimised page where the user drags/snaps each chord boundary to the
#   audio and saves the corrected times. Save reuses POST /api/annotations/
#   (non-destructive merge — existing quality corrections are preserved).


def _waveform_peaks(slug: str, n_cols: int = 1800) -> dict | None:
    """Server-side waveform peaks for <slug> as a normalised RMS array.

    Decoding the audio in the browser (Web Audio `decodeAudioData` on an m4a/
    AAC blob) is the load-bearing fragility of the v1/v2 annotators on iPhone:
    iOS Safari decodes AAC unreliably and a multi-minute buffer + a >16k-px
    canvas can silently paint nothing. Decoding once here with librosa and
    shipping ~1.8k floats sidesteps all of that — the client just draws bars.
    Cached to disk keyed by slug (+n_cols); librosa load of an m4a is ~1–2 s.
    """
    audio_path = AUDIO_DIR / f"{slug}.m4a"
    if not audio_path.exists():
        return None
    WAVEFORM_CACHE.mkdir(parents=True, exist_ok=True)
    cache = WAVEFORM_CACHE / f"{slug}.{n_cols}.json"
    if cache.exists():
        try:
            d = json.loads(cache.read_text(encoding="utf-8"))
            if d.get("peaks"):
                return d
        except ValueError:
            pass
    try:
        import numpy as np
        import librosa
        # 8 kHz mono is plenty for an amplitude envelope and keeps decode fast.
        y, sr = librosa.load(str(audio_path), sr=8000, mono=True)
        dur = float(len(y) / sr) if sr else 0.0
        if len(y) == 0:
            return None
        edges = np.linspace(0, len(y), n_cols + 1).astype(int)
        peaks = np.empty(n_cols, dtype=np.float64)
        for i in range(n_cols):
            seg = y[edges[i]:edges[i + 1]]
            peaks[i] = np.sqrt(np.mean(seg * seg)) if seg.size else 0.0
        pmax = float(peaks.max()) or 1e-6
        # gentle gamma so quiet intros/outros stay visible
        peaks = np.power(np.clip(peaks / pmax, 0.0, 1.0), 0.7)
        result = {"peaks": [round(float(p), 3) for p in peaks],
                  "duration": round(dur, 3), "n": n_cols}
    except Exception as e:
        log.warning("waveform peak extraction failed for %s (%s)", slug, e)
        return None
    try:
        cache.write_text(json.dumps(result), encoding="utf-8")
    except OSError:
        pass
    return result


@app.route("/api/waveform-peaks/<song>")
def api_waveform_peaks(song):
    """Normalised RMS waveform envelope for <song> (see _waveform_peaks)."""
    slug = lookup_slug(song or "")
    data = _waveform_peaks(slug)
    if data is None:
        return jsonify(error=f"no audio for '{slug}'"), 404
    return jsonify(data)


@app.route("/api/grid-align-data/<song>")
def api_grid_align_data(song):
    """Diagnostic bundle for /debug/grid-align (2026-07-20, user request: "il
    faut que tu arrives à t'auto évaluer sur l'alignement par grilles...
    proposes moi une interface visuelle").

    Returns every candidate grid the project has tried, so the drift-vs-
    constant-tempo hypothesis is visually falsifiable rather than
    self-reported:
      - raw_beat_times: librosa's onset-following beat detections, UN-
        de-jittered — the closest thing to ground truth this project has
        (no model assumption, just onset tracking).
      - uniform_grid_times: the ORIGINAL stock grid (circular-mean phase,
        librosa tempo scalar, no bestfit correction) — the pre-2026-07-19
        baseline.
      - bestfit_grid_times: the current PRODUCTION decode grid
        (chord_pipeline_v1._bestfit_beat_period, default since commit
        eb11d26).
      - displayed_chords: if this slug has a baked chart, its ACTUAL live
        /api/chart-model chord boundaries — the real-beat-snapped times
        (render_youtube_chart.py's `_snap`, 2026-07-20) users see today.
    """
    import librosa
    import librosa.beat
    import numpy as np

    from harmonia.models.chord_pipeline_v1 import _bestfit_beat_period

    slug = lookup_slug(song or "")
    audio_path = AUDIO_DIR / f"{slug}.m4a"
    if not audio_path.exists():
        return jsonify(error=f"no audio for '{slug}'"), 404

    try:
        y, sr = librosa.load(str(audio_path), mono=True, sr=None)
        duration_s = float(len(y) / sr)
        tempo_arr, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        tempo_bpm = float(np.atleast_1d(tempo_arr)[0])
        raw_beats = librosa.frames_to_time(beat_frames, sr=sr)

        period_stock = 60.0 / max(tempo_bpm, 1.0)

        def _grid(period):
            ang = 2 * np.pi * (raw_beats % period) / period
            phase = float((np.angle(np.mean(np.exp(1j * ang))) % (2 * np.pi))
                          * period / (2 * np.pi))
            bt = np.arange(phase, duration_s + period, period)
            return np.unique(np.concatenate([[0.0], bt, [duration_s]])).tolist()

        uniform_grid = _grid(period_stock)
        period_best = _bestfit_beat_period(raw_beats, period_stock)
        bestfit_grid = _grid(period_best)

        _NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
        displayed = None
        downbeats = None
        for candidate in (f"inferred_{slug}.html",):
            p = PLOTS_DIR / candidate
            if p.exists():
                try:
                    from harmonia.output.chart_model import payload_from_chart_html, to_chart_model
                    payload = payload_from_chart_html(p)
                    model = to_chart_model(payload, filename=candidate)
                    # chords are nested sections[*].bars[*] (each bar a list of
                    # chord dicts, ONE representative pass) + sections[*].spans
                    # (every repeat's [t0,t1] window, ×N for a folded section) —
                    # offset the representative bars onto EVERY span to get the
                    # full song timeline (mirrors app_shell.html's own
                    # spans.map(sp=>c.t0+(sp[0]-base)) reconstruction), else a
                    # folded ×18 section would only contribute 1 pass's worth of
                    # onsets to the diagnostic.
                    displayed = []
                    downbeats = []
                    for sec in model.get("sections", []):
                        spans = sec.get("spans") or []
                        if not spans:
                            continue
                        base = spans[0][0]
                        for sp0, sp1 in spans:
                            off = sp0 - base
                            for bar in sec.get("bars", []):
                                # bar-1 (the first chord of each BAR, i.e. the
                                # downbeat slot) — user request 2026-07-20: "le
                                # premier accord est toujours juste avant le
                                # premier temps" needs its own layer + stats,
                                # not lumped in with every chord change.
                                if bar:
                                    downbeats.append(round(bar[0]["t0"] + off, 4))
                                for i, c in enumerate(bar):
                                    displayed.append({
                                        "t0": round(c["t0"] + off, 4),
                                        "t1": round(c["t1"] + off, 4),
                                        "label": _NOTE[c["root"] % 12] + (c.get("q") or ""),
                                        "barFirst": i == 0,
                                    })
                    displayed.sort(key=lambda x: x["t0"])
                    downbeats.sort()
                except Exception as exc:  # noqa: BLE001 - best-effort overlay
                    log.warning("grid-align-data: no chart-model overlay for %s (%s)", slug, exc)
                break

        # ── boundary-snap before/after (2026-07-20) ────────────────────────────
        # A FOLDED section replays one representative phrase offset onto every
        # repeat's span; the repeats are NOT identically timed (rubato within the
        # phrase), so the reconstructed bar-first onsets phase-wobble vs the real
        # beats (measured: Let It Be 111 ms wrapped-std, corpus mean 84 ms). The
        # fix that WORKS is snapping each RECONSTRUCTED onset to the nearest raw
        # beat within +-1 beat — corpus mean 84->27 ms (-68%). (DeepChroma peak-
        # snap was tested and REJECTED: it moves onsets OFF the beat to harmonic-
        # change points -> 84->118 ms WORSE; the metric is offset-vs-beats. See
        # docs/research_sessions/boundary_snap_2026-07-20.md.) These overlays let
        # the tool SHOW the tightened alignment; the production consumer is the
        # app_shell fold reconstruction (opt-in HARMONIA_BOUNDARY_SNAP).
        _rb = np.asarray(sorted(raw_beats)) if len(raw_beats) else np.array([])
        _period = float(period_best) if period_best else 0.5

        def _snap_beat(t):
            if not len(_rb):
                return t
            i = int(np.searchsorted(_rb, t))
            c = [_rb[j] for j in (i - 1, i) if 0 <= j < len(_rb) and abs(_rb[j] - t) <= _period]
            return float(min(c, key=lambda b: abs(b - t))) if c else float(t)

        def _wrapped_std_ms(ts):
            if not ts or not len(_rb):
                return None
            offs = []
            for t in ts:
                i = int(np.searchsorted(_rb, t))
                c = [_rb[j] for j in (i - 1, i) if 0 <= j < len(_rb)]
                if c:
                    d = min(c, key=lambda b: abs(b - t)) - t
                    offs.append(((d + _period / 2) % _period) - _period / 2)
            if not offs:
                return None
            return {"std_ms": round(1000 * float(np.std(offs)), 1),
                    "mean_ms": round(1000 * float(np.mean(offs)), 1), "n": len(offs)}

        downbeats_snapped = [round(_snap_beat(t), 4) for t in (downbeats or [])]
        displayed_snapped = None
        if displayed is not None:
            displayed_snapped = [{**c, "t0": round(_snap_beat(c["t0"]), 4),
                                  "t1": round(_snap_beat(c["t1"]), 4)} for c in displayed]

        return jsonify({
            "song": slug, "duration_s": duration_s,
            "tempo_bpm_stock": tempo_bpm, "tempo_bpm_bestfit": 60.0 / period_best,
            "raw_beat_times": [round(float(t), 4) for t in raw_beats],
            "uniform_grid_times": [round(float(t), 4) for t in uniform_grid],
            "bestfit_grid_times": [round(float(t), 4) for t in bestfit_grid],
            "displayed_chords": displayed,
            "downbeat_times": downbeats,
            "displayed_chords_snapped": displayed_snapped,
            "downbeat_times_snapped": downbeats_snapped,
            "boundary_offset_stats": {
                "off": _wrapped_std_ms(downbeats),
                "beat_snapped": _wrapped_std_ms(downbeats_snapped),
            },
            "audio_url": f"/audio/{slug}.m4a",
        })
    except Exception as e:
        log.exception(f"grid-align-data error for {slug}")
        return jsonify(error=str(e)), 500


@app.route("/debug/section-align")
def debug_section_align():
    """Visual/audible check for the per-section iReal<->audio alignment
    (2026-07-21, user's own request: "montres moi des exemples... pour que
    je voie si ça marche"). Runs harmonia.data.ireal_youtube_align.
    align_tune_sections_to_audio LIVE against a local audio file already in
    docs/audio/, and overlays every chord it placed — colour-coded ACCEPTED
    (green) vs rejected (red, dashed) — on a waveform with a synced,
    click-to-seek <audio> player, so the alignment can be confirmed or
    refuted by ear, not by a self-reported metric alone.

    ?song=<docs/audio stem, no extension>&title=<exact iReal tune title>
    &corpus=pop400|jazz1460 (default pop400)
    """
    from html import escape

    slug = lookup_slug(request.args.get("song") or "")
    title = (request.args.get("title") or "").strip()
    corpus = request.args.get("corpus") or "pop400"
    if corpus not in ("pop400", "jazz1460"):
        corpus = "pop400"
    if not slug or not title:
        return ("<p>Pass ?song=&lt;docs/audio stem&gt;&amp;title=&lt;exact iReal title&gt;"
                "&amp;corpus=pop400|jazz1460</p>"), 400

    audio_path = AUDIO_DIR / f"{slug}.m4a"
    if not audio_path.exists():
        return f"<p>No audio at docs/audio/{escape(slug)}.m4a</p>", 404

    import contextlib
    import io as _io
    buf = _io.StringIO()
    with contextlib.redirect_stdout(buf):
        from harmonia.data.ireal_corpus import load_playlist
        tunes = load_playlist(str(REPO / "data" / "ireal" / f"{corpus}.txt"))
    tune = next((t for t in tunes if t.title == title), None)
    if tune is None:
        return f"<p>No tune titled {escape(title)!r} in {corpus}.</p>", 404

    from harmonia.data.ireal_youtube_align import align_tune_sections_to_audio, DEFAULT_MIN_SHAPE_AGREEMENT
    # Run ONCE with the median-gate only (min_shape_agreement=0.0) — this
    # still COMPUTES shape_agreement for every section that clears the
    # median gate (see align_tune_sections_to_audio), so both the "without
    # model" and "with model" acceptance decisions can be derived from this
    # single pass without re-running the (expensive) search twice. Where the
    # placement lands is IDENTICAL either way — the model shape check only
    # ever gates ACCEPTANCE after the position is already chosen by the
    # chroma score, it never moves anything (confirmed 2026-07-21 A/B).
    with_model_gate = 0.6
    tmp_dir = Path(tempfile.mkdtemp(prefix="section_align_"))
    try:
        wav = tmp_dir / "audio.wav"
        subprocess.run(["ffmpeg", "-y", "-i", str(audio_path), "-ar", "22050", "-ac", "1", str(wav)],
                       check=True, capture_output=True, timeout=120)
        try:
            results = align_tune_sections_to_audio(tune, wav, min_shape_agreement=0.0)
        except Exception as exc:  # noqa: BLE001 — show the page with an error, don't 500
            log.exception("section-align failed for %s / %s", title, slug)
            return f"<p>Alignment failed: {escape(str(exc))}</p>", 500
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    for r in results:
        sa = r.get("shape_agreement")
        r["accepted_without_model"] = bool(r.get("accepted"))
        r["accepted_with_model"] = bool(r.get("accepted")) and (sa is None or sa >= with_model_gate)

    markers = []
    sections = []
    for si, r in enumerate(results):
        warp = r.get("warp")
        section_markers = []
        for c in (r.get("chords") or []):
            t0 = warp(c["start_s"]) if warp else None
            t1 = warp(c["end_s"]) if warp else None
            if t0 is None or t1 is None or t1 <= t0:
                continue
            m = {"t0": round(t0, 3), "t1": round(t1, 3), "mma": c["mma"], "sectionIdx": si}
            markers.append(m)
            section_markers.append(m)
        sections.append({
            "idx": si, "label": r["label"], "bar0": r["bar0"], "bar1": r["bar1"],
            "acceptedWithoutModel": r["accepted_without_model"],
            "acceptedWithModel": r["accepted_with_model"],
            "t0": section_markers[0]["t0"] if section_markers else None,
            "t1": section_markers[-1]["t1"] if section_markers else None,
            "medianMs": (r.get("error") or {}).get("median_ms"),
            "shapeAgreement": r.get("shape_agreement"),
            "reason": r.get("reason"),
        })
    n_without = sum(1 for s in sections if s["acceptedWithoutModel"])
    n_with = sum(1 for s in sections if s["acceptedWithModel"])

    peaks = _waveform_peaks(slug)
    duration = peaks["duration"] if peaks else (markers[-1]["t1"] + 5 if markers else 60)

    page_data = {
        "title": tune.title, "slug": slug, "audioUrl": f"/audio/{slug}.m4a",
        "duration": duration, "markers": markers, "sections": sections,
        "nWithoutModel": n_without, "nWithModel": n_with, "nTotal": len(results),
    }

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>Section align: {escape(tune.title)}</title>
<style>
  :root {{ --paper:#f7f3e9; --card:#fffdf6; --ink:#1c1c1c; --rule:#b9b09a; --faint:#8a8371; --accent:#8a2b2b; --line:#e5dcc6; --green:#1f8a5b; }}
  * {{ box-sizing:border-box; -webkit-tap-highlight-color:transparent; }}
  html, body {{ margin:0; background:var(--paper); color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif; }}
  header {{ padding:14px 16px; background:var(--card); border-bottom:1px solid var(--line);
    display:flex; align-items:center; gap:10px; }}
  header a {{ font:700 13px system-ui,sans-serif; color:#fff; text-decoration:none;
    background:var(--accent); border:none; border-radius:20px; padding:7px 13px; flex:0 0 auto; }}
  h1 {{ margin:0; font:italic 600 17px Georgia,'Times New Roman',serif; flex:1; min-width:0;
    overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  #summary {{ padding:8px 16px; background:#fbf6e6; border-bottom:1px solid var(--line); font:600 12px system-ui; color:var(--faint); }}
  #controls {{ padding:10px 16px; background:var(--card); display:flex; align-items:center; gap:12px;
    border-bottom:1px solid var(--line); }}
  audio {{ flex:1; min-width:220px; height:34px; }}
  #waveWrap {{ position:relative; overflow-x:auto; background:var(--card); }}
  canvas {{ display:block; }}
  .rowLabel {{ position:absolute; left:4px; top:2px; font:700 10px system-ui; letter-spacing:.04em;
    color:var(--faint); text-transform:uppercase; z-index:5; pointer-events:none; }}
  .sectionRow {{ position:relative; height:34px; border-top:1px solid var(--line); background:#f2ecd8; }}
  .sectionBlock {{ position:absolute; top:14px; bottom:2px; border-radius:4px; opacity:.85;
    display:flex; align-items:center; justify-content:center; overflow:hidden; }}
  .sectionBlock.rejected {{ opacity:.32; background-image:repeating-linear-gradient(45deg,rgba(0,0,0,.15) 0 4px,transparent 4px 8px); }}
  .sectionBlock span {{ font:700 11px Georgia,serif; color:#fff; text-shadow:0 1px 2px rgba(0,0,0,.4); white-space:nowrap; padding:0 4px; }}
  .sectionDivider {{ position:absolute; top:0; bottom:0; width:1px; background:var(--ink); opacity:.15; }}
  .marker {{ position:absolute; top:0; bottom:0; border-left:1px solid #ffffff77; }}
  .mlabel {{ position:absolute; bottom:1px; font:600 8px Georgia,serif; color:var(--ink); opacity:.65;
    white-space:nowrap; transform:translateX(2px); }}
  .playhead {{ position:absolute; top:0; bottom:0; width:2px; background:#1c1c1c; z-index:10; pointer-events:none; }}
  #hint {{ padding:10px 16px; font:italic 13px Georgia,serif; color:var(--faint); line-height:1.5; max-width:680px; }}
  #hint b {{ color:var(--ink); }}
</style>
</head><body>
<div id="container">
  <header>
    <a href="/library">&larr; library</a>
    <h1>Section align — {escape(tune.title)}</h1>
  </header>
  <div id="summary">without model: {n_without}/{len(results)} accepted &nbsp;·&nbsp; with model (shape-agreement gate ≥{with_model_gate:g}): {n_with}/{len(results)} accepted</div>
  <div id="controls">
    <audio id="audio" controls preload="metadata" src="{escape(page_data['audioUrl'])}"></audio>
    <span id="timeLbl" style="font:600 12px system-ui;color:var(--faint);white-space:nowrap;">0:00 / 0:00</span>
  </div>
  <div id="waveWrap">
    <canvas id="canvas" height="70"></canvas>
    <div id="rowWithout" class="sectionRow"><div class="rowLabel">without model</div></div>
    <div id="rowWith" class="sectionRow"><div class="rowLabel">with model</div></div>
    <div id="markerLayer" style="position:relative;height:14px;"></div>
    <div id="playhead" class="playhead"></div>
  </div>
  <p id="hint">Each coloured block is one iReal chart SECTION (i/A/B/C…), one colour per
    label. <b>Solid</b> = accepted (trustworthy real timing); <b>hatched/faded</b> = rejected
    at that gate. Compare the two rows: any block that's solid in "without model" but hatched
    in "with model" is one the model's shape check specifically vetoed. Thin ticks below are
    individual chord onsets (iReal's own tokens, never a model prediction). Click the wave to seek.</p>
</div>
<script>
const D = {json.dumps(page_data)};
const scale = 60; // px/sec
const w = Math.max(320, D.duration * scale);
const canvas = document.getElementById('canvas');
canvas.width = w; canvas.height = 70; canvas.style.width = w+'px';
const ctx = canvas.getContext('2d');
const rowWithout = document.getElementById('rowWithout');
const rowWith = document.getElementById('rowWith');
const markerLayer = document.getElementById('markerLayer');
[rowWithout, rowWith, markerLayer].forEach(el => el.style.width = w+'px');
const playhead = document.getElementById('playhead');
const audio = document.getElementById('audio');

const PALETTE = ['#8a2b2b','#2b5f8a','#1f8a5b','#8a6b1f','#5f2b8a','#8a2b6b','#2b8a7a','#6b8a2b'];
const colorFor = (() => {{
  const seen = new Map();
  return (label) => {{
    if (!seen.has(label)) seen.set(label, PALETTE[seen.size % PALETTE.length]);
    return seen.get(label);
  }};
}})();

function drawWave(peaks) {{
  ctx.clearRect(0,0,w,70);
  ctx.fillStyle = '#efe8d6'; ctx.fillRect(0,0,w,70);
  if (peaks && peaks.length) {{
    ctx.fillStyle = '#b9b09a';
    const n = peaks.length;
    for (let x=0; x<w; x++) {{
      const idx = Math.floor(x/w*n);
      const h = Math.max(1,(peaks[idx]||0)*56);
      ctx.fillRect(x, 35-h/2, 1, h);
    }}
  }}
}}
fetch('/api/waveform-peaks/'+encodeURIComponent(D.slug)).then(r=>r.ok?r.json():null)
  .then(d=>drawWave(d&&d.peaks)).catch(()=>drawWave(null));

function renderSectionRow(row, acceptedKey) {{
  D.sections.forEach(s => {{
    if (s.t0 == null || s.t1 == null) return;   // no placement at all — can't draw
    const x0 = s.t0*scale, x1 = s.t1*scale;
    const block = document.createElement('div');
    block.className = 'sectionBlock' + (s[acceptedKey] ? '' : ' rejected');
    block.style.position = 'absolute'; block.style.left = x0+'px'; block.style.width = Math.max(2,x1-x0)+'px';
    block.style.background = colorFor(s.label);
    const lbl = document.createElement('span');
    lbl.textContent = `${{s.label}} [${{s.bar0}}-${{s.bar1}}]`;
    block.appendChild(lbl);
    block.title = `${{s.label}} bars ${{s.bar0}}-${{s.bar1}} — median=${{s.medianMs!=null?Math.round(s.medianMs)+'ms':'n/a'}}` +
                 (s.shapeAgreement!=null ? ` shape=${{s.shapeAgreement.toFixed(2)}}` : '');
    row.appendChild(block);
    const div = document.createElement('div');
    div.className = 'sectionDivider'; div.style.left = x0+'px';
    row.appendChild(div);
  }});
}}
renderSectionRow(rowWithout, 'acceptedWithoutModel');
renderSectionRow(rowWith, 'acceptedWithModel');

D.markers.forEach(m => {{
  const x0 = m.t0*scale;
  const el = document.createElement('div');
  el.className = 'marker';
  el.style.left = x0+'px'; el.style.position='absolute'; el.style.top='0'; el.style.bottom='0';
  markerLayer.appendChild(el);
  const lbl = document.createElement('div');
  lbl.className = 'mlabel';
  lbl.style.left = x0+'px'; lbl.style.position='absolute'; lbl.style.bottom='1px';
  lbl.textContent = m.mma;
  markerLayer.appendChild(lbl);
}});

function fmt(t) {{ t=Math.max(0,t|0); return `${{(t/60)|0}}:${{String(t%60).padStart(2,'0')}}`; }}
audio.addEventListener('timeupdate', () => {{
  const t = audio.currentTime;
  playhead.style.left = (t*scale)+'px';
  document.getElementById('timeLbl').textContent = fmt(t)+' / '+fmt(audio.duration||D.duration);
}});
document.getElementById('waveWrap').addEventListener('click', (e) => {{
  const rect = canvas.getBoundingClientRect();
  const x = e.clientX - rect.left + document.getElementById('waveWrap').scrollLeft;
  audio.currentTime = Math.max(0, x/scale);
}});
</script>
</body></html>"""
    return Response(page, mimetype="text/html")


@app.route("/api/beat-grid-audio/<song>")
def api_beat_grid_audio(song):
    """Detected beat times + tempo from audio for waveform V4 beat-grid editor.

    Returns {beat_times: [...], tempo_bpm: X, duration_s: Y, n_bars: Z}
    """
    import librosa
    import librosa.beat
    import numpy as np

    slug = lookup_slug(song or "")
    audio_path = AUDIO_DIR / f"{slug}.m4a"
    if not audio_path.exists():
        return jsonify(error=f"no audio for '{slug}'"), 404

    try:
        # Load audio using librosa (falls back to audioread for .m4a)
        y, sr = librosa.load(str(audio_path), mono=True, sr=None)
        duration_s = float(len(y) / sr)

        # Detect beats
        tempo_arr, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        tempo_bpm = float(np.atleast_1d(tempo_arr)[0])

        # De-jitter beat times using uniform grid (same as pipeline)
        beat_times_raw = librosa.frames_to_time(beat_frames, sr=sr)
        period = 60.0 / max(tempo_bpm, 1.0)
        ang = 2 * np.pi * (beat_times_raw % period) / period
        phase = float((np.angle(np.mean(np.exp(1j * ang))) % (2 * np.pi)) * period / (2 * np.pi))
        beat_times = np.arange(phase, duration_s + period, period)
        beat_times = np.unique(np.concatenate([[0.0], beat_times, [duration_s]]))

        n_bars = max(1, len(beat_times) // 4)
        return jsonify({
            "beat_times": beat_times.tolist(),
            "tempo_bpm": tempo_bpm,
            "duration_s": duration_s,
            "n_bars": n_bars,
        })
    except Exception as e:
        log.exception(f"beat-grid-audio error for {slug}")
        return jsonify(error=str(e)), 500


@app.route("/api/reinfer-from-beats/<song>", methods=["POST"])
def api_reinfer_from_beats(song):
    """Re-infer chords with corrected beat grid.

    Request body:
      {
        "corrected_beat_times": [...],  # beat times for first N bars (seconds)
        "n_locked_beats": N,             # number of beats that were manually corrected
        "tempo_bpm": X                   # detected tempo (used to extrapolate)
      }

    Returns: {chords: [...], beat_times: [...]}
    """
    from harmonia.models.chord_pipeline_v1 import infer_chords_v1

    slug = lookup_slug(song or "")
    audio_path = AUDIO_DIR / f"{slug}.m4a"
    if not audio_path.exists():
        return jsonify(error=f"no audio for '{slug}'"), 404

    try:
        data = request.get_json() or {}
        corrected_beats = np.array(data.get("corrected_beat_times", []), dtype=float)
        n_locked = int(data.get("n_locked_beats", len(corrected_beats)))
        tempo_bpm = float(data.get("tempo_bpm", 120.0))

        if n_locked < 1:
            return jsonify(error="n_locked_beats must be >= 1"), 400

        # Extrapolate beat grid from corrected beats
        if len(corrected_beats) < 2:
            return jsonify(error="need at least 2 corrected beat times"), 400

        beat_period = 60.0 / max(tempo_bpm, 1.0)

        # Estimate the beat offset from first two corrected beats
        if len(corrected_beats) >= 2:
            actual_period = corrected_beats[1] - corrected_beats[0]
            # Fine-tune tempo estimate if the corrected period differs significantly
            if abs(actual_period - beat_period) < 0.1:
                beat_period = actual_period

        # Extrapolate forward: beat_times[n_locked:] = beat_times[n_locked-1] + k*period
        last_corrected = corrected_beats[-1]

        # Import librosa to get total duration
        import librosa
        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
        total_duration = librosa.get_duration(y=y, sr=sr)

        # Build extrapolated beat times
        all_beats = list(corrected_beats)
        beat_idx = len(corrected_beats)
        while True:
            next_beat = last_corrected + (beat_idx - n_locked + 1) * beat_period
            if next_beat > total_duration + 1.0:  # 1s tolerance
                break
            all_beats.append(next_beat)
            beat_idx += 1

        beat_times_arr = np.array(all_beats, dtype=float)

        # Re-infer chords with corrected beat times
        # This requires modifying chord_pipeline_v1 to accept pre-computed beat times
        # For now, we'll just return the corrected beat times and let the front-end
        # know that it should re-load the inference. In practice, we'd need a variant
        # that doesn't re-detect beats.
        # Workaround: store corrected beats in a temp file, then re-infer normally

        # Load and write corrected beat grid to a temporary pickle
        import tempfile
        import pickle

        temp_beats_file = Path(tempfile.gettempdir()) / f"beats_{slug}.pkl"
        pickle.dump(beat_times_arr, temp_beats_file.open("wb"))

        # For now, just return the corrected beats and note that full re-inference
        # would require deeper integration
        return jsonify({
            "beat_times": beat_times_arr.tolist(),
            "tempo_bpm": tempo_bpm,
            "status": "beats_corrected",
            "note": "Full chord re-inference pending integration"
        })

    except Exception as e:
        log.exception(f"reinfer-from-beats error for {slug}")
        return jsonify(error=str(e)), 500


# ── Music-aware waveform annotator v4 (beat-grid editor + chord events) ──
#
# Two-stage interface:
#   Stage 1: Beat-grid editor — correct beat phase for first N bars, tap "Lock & Infer"
#   Stage 2: Chord event editor — add/delete/relabel chord boundaries (point markers, not intervals)
#
# Chord model: chords[i] = {t: time_s, label: "C", dirty: bool}
#   — represents a point event "new chord starts here"
#   — no t1; duration is implicit (until next chord or end of song)
#   — add/delete by adding/removing events
#   — left boundary is locked (can't delete or move)


# ── Music-aware waveform annotator v3 (iPhone-first, server-decoded waveform) ──
#
# GET /annotator-v3?song=<slug>
#   The v1/v2 annotators decode the audio in-browser (Web Audio); that is the
#   part that fails on iPhone. v3 draws a server-computed peak envelope
#   (/api/waveform-peaks) and plays a plain <audio> element (HTTP Range 206 is
#   already supported), so nothing music-critical depends on iOS Web Audio.
#
#   Music model held explicitly and kept in sync:
#     • beats / downbeats  → the temporal grid (bar lines = downbeats)
#     • chords[i].t0/t1     → harmonic spans; adjacent spans share one boundary
#     • dragging boundary k rewrites chords[k-1].t1 = chords[k].t0 together
#       (this IS the chord-sheet sync), snapping to the nearest beat when close
#   Save reuses POST /api/annotations/<saveFile> — same contract as v1/v2.



@app.route("/api/beat-0-shift/<song>", methods=["POST"])
def api_beat_0_shift(song):
    """Shift beat 0 by delta_ms, extrapolate entire grid, re-infer chords.

    Request: {delta_ms: int}  (positive = beat too early, shift forward)
    Response: {beat_times: [...], chords: [...], note: "..."}
    """
    import librosa
    import numpy as np

    slug = lookup_slug(song or "")
    audio_path = AUDIO_DIR / f"{slug}.m4a"
    if not audio_path.exists():
        return jsonify(error=f"no audio for '{slug}'"), 404

    try:
        data = request.get_json() or {}
        delta_ms = float(data.get("delta_ms", 0))
        delta_s = delta_ms / 1000.0

        # Load audio and extract beat times
        y, sr = librosa.load(str(audio_path), mono=True, sr=None)
        duration_s = float(len(y) / sr)

        tempo_arr, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        tempo_bpm = float(np.atleast_1d(tempo_arr)[0])

        # De-jitter beat times
        beat_times_raw = librosa.frames_to_time(beat_frames, sr=sr)
        period = 60.0 / max(tempo_bpm, 1.0)
        ang = 2 * np.pi * (beat_times_raw % period) / period
        phase = float((np.angle(np.mean(np.exp(1j * ang))) % (2 * np.pi)) * period / (2 * np.pi))

        # Shift beat 0
        phase_new = phase + delta_s
        beat_times = np.arange(phase_new, duration_s + period, period)
        beat_times = np.unique(np.concatenate([[0.0], beat_times, [duration_s]]))

        # For now, just return the corrected beat times
        # (full re-inference with beat override requires deeper pipeline refactoring)
        return jsonify({
            "beat_times": beat_times.tolist(),
            "tempo_bpm": tempo_bpm,
            "delta_ms": delta_ms,
            "duration_s": duration_s,
            "note": f"Beat 0 shifted by {delta_ms:+.0f}ms, grid extrapolated. Use this beat grid for re-inference.",
            "next_step": "Call /api/reinfer with this beat grid"
        })
    except Exception as e:
        log.exception(f"beat-0-shift error for {slug}")
        return jsonify(error=str(e)), 500


@app.route("/gt-playalong")
def gt_playalong():
    """Ground truth play-along: waveform + iReal chords synced to audio.

    Helps user verify GT alignment is correct before evaluating model.
    ?song=<slug>
    """
    from html import escape

    slug = lookup_slug(request.args.get("song") or "autumn_leaves")

    # Load iReal chart
    chords, tempo = _load_ireal_alignment(slug)
    if not chords:
        return f"<p>No iReal chart for {slug}. Expected docs/plots/irealb_{slug}.html</p>", 404

    audio_path = AUDIO_DIR / f"{slug}.m4a"
    have_audio = audio_path.exists()

    if not have_audio:
        return f"<p>No audio for {slug}</p>", 404

    # Build beat grid
    duration = max((c["t1"] for c in chords), default=0.0)
    grid = _beat_grid_for(slug, audio_path, float(tempo or 120), duration)

    # Prepare chart data
    chart_data = {
        "title": slug.replace("_", " ").title(),
        "chords": chords,
        "beats": grid["beats"],
        "downbeats": grid["downbeats"],
        "audioUrl": f"/audio/{slug}.m4a" if have_audio else "",
        "duration": duration,
        "tempo": tempo or 120,
    }

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>GT Play-Along: {slug}</title>
<style>
  * {{ box-sizing:border-box; }}
  html, body {{ margin:0; background:#0e1116; color:#e8edf4; font-family:system-ui,sans-serif; }}
  header {{ padding:16px; background:#171c24; border-bottom:1px solid #2a3340; }}
  h1 {{ margin:0; font-size:18px; }}
  #container {{ display:flex; flex-direction:column; height:100vh; }}
  #waveContainer {{ flex:1; position:relative; background:#171c24; border-bottom:1px solid #2a3340;
    overflow:hidden; }}
  canvas {{ display:block; width:100%; height:100%; }}
  #chordLabels {{ position:absolute; top:8px; left:0; right:0; font-size:13px; color:#8b97a8;
    pointer-events:none; }}
  .chordLabel {{ position:absolute; padding:4px 8px; background:rgba(0,201,167,0.2);
    border-radius:3px; white-space:nowrap; }}
  audio {{ width:100%; padding:12px; background:#171c24; border-top:1px solid #2a3340; }}
  #info {{ padding:12px; background:#1e2530; font-size:12px; color:#8b97a8; }}
  .playhead {{ position:absolute; width:2px; height:100%; background:#6ea8ff; z-index:100; }}
</style>
</head><body>

<div id="container">
  <header>
    <h1>🎵 GT Play-Along: {escape(slug)}</h1>
    <p style="margin:8px 0 0; font-size:12px; color:#8b97a8;">
      iReal chart synced to audio. Tap play and watch the chord timeline.
      Verify alignment is correct — beat grid + chord changes should line up with the music.
    </p>
  </header>

  <div id="waveContainer">
    <canvas id="canvas"></canvas>
    <div id="chordLabels"></div>
    <div id="playhead" class="playhead"></div>
  </div>

  <audio id="audio" crossOrigin="anonymous" controls>
    <source src="{escape(chart_data['audioUrl'])}" type="audio/mpeg">
  </audio>

  <div id="info">
    <div>⏱️ <span id="curTime">0:00</span> / <span id="durTime">0:00</span></div>
    <div>🎼 <span id="curChord">—</span> (click to verify)</div>
  </div>
</div>

<script>
const data = {json.dumps(chart_data)};
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');
const audio = document.getElementById('audio');
const playhead = document.getElementById('playhead');

let scale = 100; // px/sec
let peaks = null;

// Format time
function fmt(s) {{
  const m = Math.floor(s / 60), ss = Math.floor(s % 60);
  return m + ':' + (ss < 10 ? '0' : '') + ss;
}}

// Load waveform peaks
async function loadPeaks() {{
  try {{
    const r = await fetch('/api/waveform-peaks/' + encodeURIComponent(data.title.replace(/ /g, '_')));
    if (r.ok) {{
      const d = await r.json();
      peaks = d.peaks || [];
    }}
  }} catch (e) {{ console.warn('peaks fetch failed', e); }}
  draw();
}}

// Draw waveform + beat grid + chord spans
function draw() {{
  const w = Math.max(300, data.duration * scale);
  canvas.width = w;
  canvas.height = 200;

  // Waveform background
  const grad = ctx.createLinearGradient(0, 0, w, 0);
  grad.addColorStop(0, '#2a3340');
  grad.addColorStop(0.5, '#4a5a70');
  grad.addColorStop(1, '#2a3340');
  ctx.fillStyle = grad;
  ctx.fillRect(0, 0, w, 200);

  // Draw peaks
  if (peaks && peaks.length) {{
    ctx.fillStyle = '#8b97a8';
    const n = peaks.length;
    for (let x = 0; x < w; x++) {{
      const idx = Math.floor(x / w * n);
      const h = Math.max(1, (peaks[idx] || 0) * 80);
      ctx.fillRect(x, 100 - h / 2, 1, h);
    }}
  }}

  // Beat grid
  ctx.strokeStyle = 'rgba(255,180,84,0.2)';
  ctx.lineWidth = 1;
  data.beats.forEach(t => {{
    const x = t * scale;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, 200);
    ctx.stroke();
  }});

  // Downbeats (thicker)
  ctx.strokeStyle = 'rgba(255,180,84,0.5)';
  ctx.lineWidth = 2;
  data.downbeats.forEach(t => {{
    const x = t * scale;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, 200);
    ctx.stroke();
  }});

  // Chord spans
  ctx.fillStyle = 'rgba(0,201,167,0.15)';
  for (let i = 0; i < data.chords.length; i++) {{
    const c = data.chords[i];
    const cn = data.chords[i + 1];
    const x0 = c.t0 * scale;
    const x1 = (cn ? cn.t0 : data.duration) * scale;
    ctx.fillRect(x0, 0, x1 - x0, 200);
  }}

  // Chord labels
  const labels = document.getElementById('chordLabels');
  labels.innerHTML = '';
  data.chords.forEach((c, i) => {{
    const x = c.t0 * scale;
    const el = document.createElement('div');
    el.className = 'chordLabel';
    el.style.left = x + 'px';
    el.textContent = c.label;
    el.addEventListener('click', () => {{
      document.getElementById('curChord').textContent = c.label + ' @ ' + fmt(c.t0);
    }});
    labels.appendChild(el);
  }});

  // Playhead
  const t = audio.currentTime || 0;
  playhead.style.left = Math.max(0, t * scale) + 'px';

  // Current time
  document.getElementById('curTime').textContent = fmt(t);
  document.getElementById('durTime').textContent = fmt(data.duration);

  // Current chord
  let cur = '—';
  for (let i = data.chords.length - 1; i >= 0; i--) {{
    if (data.chords[i].t0 <= t) {{
      cur = data.chords[i].label + ' @ ' + fmt(data.chords[i].t0);
      break;
    }}
  }}
  document.getElementById('curChord').textContent = cur;
}}

// Sync playhead
audio.addEventListener('timeupdate', draw);
audio.addEventListener('play', () => {{ draw(); }});
audio.addEventListener('pause', () => {{ draw(); }});

// Zoom
canvas.addEventListener('wheel', e => {{
  e.preventDefault();
  scale *= (e.deltaY < 0 ? 1.2 : 0.8);
  scale = Math.max(20, Math.min(500, scale));
  draw();
}});

// Load
loadPeaks();
</script>
</body></html>"""

    return Response(page, mimetype="text/html")


@app.route("/gt-playalong-training")
def gt_playalong_training():
    """Ground-truth play-along for a Billboard training-corpus song: real
    audio + the actual McGill Billboard chords_full boundaries (not the
    inferred-chord-cell-collapsed GT row the training-mode chart shows —
    see app_shell.html's gtForSpan(), which snaps GT to the model's own
    segmentation). This route renders GT's own boundaries as an independent
    timeline strip so a listener can judge, by ear, whether a GT chord
    change is early/late/right relative to what they actually hear.

    ?song=<inferred_*.html chart filename> — reuses _yt_video_ids /
    _yt_audio_meta (already populated when the song was analysed) and
    _gt_chords_for_video (mirdata Billboard chords_full) rather than adding
    any new backend plumbing.
    """
    from html import escape

    filename = request.args.get("song") or ""
    filename = re.sub(r"[^A-Za-z0-9_.\-]", "", filename)
    if not filename.startswith("inferred_") or not filename.endswith(".html"):
        return "<p>Pass ?song=inferred_&lt;slug&gt;.html (a training-mode chart).</p>", 400
    if not (PLOTS_DIR / filename).exists():
        return f"<p>No chart {escape(filename)}.</p>", 404

    video_id = _yt_video_ids.get(filename, "")
    if not video_id:
        return f"<p>{escape(filename)} has no YouTube video id.</p>", 404
    gt = _gt_chords_for_video(video_id)
    if not gt:
        return (f"<p>{escape(filename)} is not a training-corpus song (no McGill "
                f"Billboard chords_full for video {escape(video_id)}).</p>"), 404

    audio_meta = _yt_audio_meta.get(filename) or {}
    audio_url = audio_meta.get("audio", "")
    audio_path = AUDIO_DIR / Path(audio_url).name if audio_url else None
    if not audio_url or not audio_path or not audio_path.exists():
        return f"<p>No downloaded audio for {escape(filename)}.</p>", 404
    slug = audio_path.stem

    title = filename.removeprefix("inferred_").removesuffix(".html").replace("_", " ").title()
    duration = max((c["t1"] for c in gt), default=0.0)

    chart_data = {
        "title": title,
        "gt": gt,
        "audioUrl": f"/audio/{audio_path.name}",
        "peaksSlug": slug,
        "duration": duration,
        "backHref": f"/chart/{filename}",
    }

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>GT play-along: {escape(title)}</title>
<style>
  :root {{ --paper:#f7f3e9; --card:#fffdf6; --ink:#1c1c1c; --rule:#b9b09a; --faint:#8a8371; --accent:#8a2b2b; --line:#e5dcc6; --green:#1f8a5b; }}
  * {{ box-sizing:border-box; -webkit-tap-highlight-color:transparent; }}
  html, body {{ margin:0; background:var(--paper); color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif; }}
  header {{ padding:14px 16px; background:var(--card); border-bottom:1px solid var(--line);
    display:flex; align-items:center; gap:10px; }}
  header a {{ font:600 12px system-ui; color:var(--faint); text-decoration:none; border:1px solid var(--rule);
    border-radius:20px; padding:5px 10px; flex:0 0 auto; }}
  h1 {{ margin:0; font:italic 600 17px Georgia,'Times New Roman',serif; flex:1; min-width:0;
    overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  #container {{ display:flex; flex-direction:column; }}
  #waveWrap {{ position:relative; overflow-x:auto; background:var(--card); border-bottom:1px solid var(--line); }}
  canvas {{ display:block; }}
  #gtStrip {{ position:relative; height:56px; }}
  .gtBlock {{ position:absolute; top:6px; bottom:6px; border-radius:6px; display:flex; align-items:center;
    justify-content:center; font:600 12.5px Georgia,serif; color:#fff; overflow:hidden; white-space:nowrap;
    cursor:pointer; transition:transform .08s, filter .08s; border:1px solid rgba(0,0,0,.15); }}
  .gtBlock.active {{ transform:scaleY(1.12); filter:brightness(1.12); box-shadow:0 0 0 2px var(--ink); z-index:5; }}
  .playhead {{ position:absolute; top:0; bottom:0; width:2px; background:var(--accent); z-index:10;
    pointer-events:none; box-shadow:0 0 4px var(--accent); }}
  #controls {{ padding:12px 16px; background:var(--card); display:flex; align-items:center; gap:14px;
    border-bottom:1px solid var(--line); }}
  audio {{ flex:1; min-width:0; height:34px; }}
  #curChordCard {{ padding:16px; text-align:center; }}
  #curChordLabel {{ font:italic 700 44px Georgia,'Times New Roman',serif; color:var(--ink); }}
  #curChordMeta {{ font:500 13px system-ui; color:var(--faint); margin-top:4px; }}
  #hint {{ padding:0 16px 16px; font:italic 13px Georgia,serif; color:var(--faint); line-height:1.5; max-width:640px; }}
</style>
</head><body>
<div id="container">
  <header>
    <a href="{escape(chart_data['backHref'])}">&larr; chart</a>
    <h1>GT play-along — {escape(title)}</h1>
  </header>
  <div id="controls">
    <audio id="audio" controls preload="metadata" src="{escape(chart_data['audioUrl'])}"></audio>
    <span id="timeLbl" style="font:600 12px system-ui;color:var(--faint);white-space:nowrap;">0:00 / 0:00</span>
  </div>
  <div id="curChordCard">
    <div id="curChordLabel">&mdash;</div>
    <div id="curChordMeta">ground truth (McGill Billboard) &middot; tap a block below to seek</div>
  </div>
  <div id="waveWrap">
    <canvas id="canvas" height="80"></canvas>
    <div id="gtStrip"></div>
    <div id="playhead" class="playhead"></div>
  </div>
  <p id="hint">Press play and listen. Each block below is one ground-truth chord span from the
    McGill Billboard hand annotation (not the model's inferred segmentation). The highlighted block
    tracks playback in real time &mdash; if the highlight changes before/after you actually hear the
    harmony change, that GT boundary is mistimed. Scroll horizontally to see the whole song; click any
    block or the waveform to seek there.</p>
</div>
<script>
const D = {json.dumps(chart_data)};
const NOTE=["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"];
const QTXT={{"":"","maj":"","maj7":"maj7","min":"m","min7":"m7","7":"7","hdim7":"m7♭5","dim":"dim","dim7":"dim7","6":"6","min6":"m6","maj6":"6","9":"9","min9":"m9","sus4":"sus4","sus2":"sus2","aug":"aug"}};
function parseHarte(l){{
  if(!l || l==="N" || l==="X") return {{text:"N.C.", color:"#8a8371"}};
  const p=String(l).split(":");
  const pc={{"C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,"E":4,"Fb":4,"F":5,"E#":5,"F#":6,"Gb":6,"G":7,"G#":8,"Ab":8,"A":9,"A#":10,"Bb":10,"B":11,"Cb":11}}[p[0]];
  const root = pc==null?0:pc;
  let q=(p[1]||"maj").split("(")[0];
  const text = NOTE[root] + (QTXT[q]!=null?QTXT[q]:q);
  const hue = (root*30)%360;
  return {{text, color:`hsl(${{hue}},42%,38%)`}};
}}
const gt = D.gt.map(g=>Object.assign({{}}, g, parseHarte(g.label)));

const scale = 90; // px/sec
const canvas = document.getElementById('canvas');
const gtStrip = document.getElementById('gtStrip');
const playhead = document.getElementById('playhead');
const audio = document.getElementById('audio');
const w = Math.max(320, D.duration * scale);
canvas.width = w; canvas.height = 80;
canvas.style.width = w+'px';
gtStrip.style.width = w+'px';
const ctx = canvas.getContext('2d');

function drawWave(peaks){{
  ctx.clearRect(0,0,w,80);
  ctx.fillStyle = '#efe8d6';
  ctx.fillRect(0,0,w,80);
  if(peaks && peaks.length){{
    ctx.fillStyle = '#b9b09a';
    const n = peaks.length;
    for(let x=0; x<w; x++){{
      const idx = Math.floor(x/w*n);
      const h = Math.max(1,(peaks[idx]||0)*64);
      ctx.fillRect(x, 40-h/2, 1, h);
    }}
  }}
}}
fetch('/api/waveform-peaks/'+encodeURIComponent(D.peaksSlug)).then(r=>r.ok?r.json():null)
  .then(d=>drawWave(d&&d.peaks)).catch(()=>drawWave(null));

gt.forEach((g,i)=>{{
  const b=document.createElement('div');
  b.className='gtBlock';
  b.style.left=(g.t0*scale)+'px';
  b.style.width=Math.max(2,(g.t1-g.t0)*scale-1)+'px';
  b.style.background=g.color;
  b.dataset.i=i;
  if((g.t1-g.t0)*scale > 26) b.textContent=g.text;
  b.title=g.text+'  '+g.t0.toFixed(2)+'s → '+g.t1.toFixed(2)+'s';
  b.onclick=()=>{{ audio.currentTime=g.t0; audio.play(); }};
  gtStrip.appendChild(b);
}});

function fmt(s){{ s=Math.max(0,Math.floor(s||0)); return Math.floor(s/60)+':'+String(s%60).padStart(2,'0'); }}
let lastActive=-1;
function tick(){{
  const t=audio.currentTime||0;
  playhead.style.left=Math.max(0,t*scale)+'px';
  document.getElementById('timeLbl').textContent=fmt(t)+' / '+fmt(D.duration);
  let idx=-1;
  for(let i=0;i<gt.length;i++){{ if(gt[i].t0<=t && t<gt[i].t1){{ idx=i; break; }} }}
  if(idx===-1){{ for(let i=gt.length-1;i>=0;i--){{ if(gt[i].t0<=t){{ idx=i; break; }} }} }}
  if(idx!==lastActive){{
    if(lastActive>=0){{ const prev=gtStrip.children[lastActive]; if(prev) prev.classList.remove('active'); }}
    if(idx>=0){{
      const cur=gtStrip.children[idx];
      if(cur){{ cur.classList.add('active');
        const wrap=document.getElementById('waveWrap');
        const bx=cur.offsetLeft;
        if(bx < wrap.scrollLeft+40 || bx > wrap.scrollLeft+wrap.clientWidth-80){{
          wrap.scrollTo({{left: Math.max(0,bx-120), behavior:'smooth'}});
        }}
      }}
      document.getElementById('curChordLabel').textContent = gt[idx].text;
      document.getElementById('curChordMeta').textContent =
        gt[idx].t0.toFixed(2)+'s → '+gt[idx].t1.toFixed(2)+'s  (span '+(gt[idx].t1-gt[idx].t0).toFixed(2)+'s)  ·  ground truth (McGill Billboard)';
    }}
    lastActive=idx;
  }}
}}
audio.addEventListener('timeupdate', tick);
audio.addEventListener('play', ()=>requestAnimationFrame(function loop(){{ tick(); if(!audio.paused) requestAnimationFrame(loop); }}));
document.getElementById('waveWrap').addEventListener('click', e=>{{
  if(e.target.classList.contains('gtBlock')) return;
  const rect=canvas.getBoundingClientRect();
  const x = e.clientX - rect.left + document.getElementById('waveWrap').scrollLeft;
  audio.currentTime = Math.max(0, x/scale);
}});
tick();
</script>
</body></html>"""

    return Response(page, mimetype="text/html")


_RWC_CHORD_BASE = ("https://raw.githubusercontent.com/rwc-music/rwc-annotations/"
                    "main/01_annotations_preprocessed/chords/RWC-P")
_RWC_UA = "harmonia-research/1.0 (louisjvincent@gmail.com)"


def _fetch_rwc_chords(rwcid: str) -> list[dict] | None:
    """Pull one RWC-Popular song's Cho-Bello chord CSV straight from the
    rwc-annotations GitHub repo (same source + same absolute-second-timestamp
    format as scripts/build_rwc_corpus.py::fetch_chords — duplicated here
    rather than importing that module, since it pulls in the full
    chord_pipeline_v1/remotezip feature-extraction stack which is unwanted
    weight for a long-running Flask process). Returns [{t0,t1,label}, ...].
    """
    import csv, io, urllib.request

    url = f"{_RWC_CHORD_BASE}/{rwcid}.csv"
    req = urllib.request.Request(url, headers={"User-Agent": _RWC_UA})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            text = r.read().decode("utf-8")
    except Exception:
        return None
    rows = []
    rd = csv.reader(io.StringIO(text), delimiter=";")
    next(rd, None)  # header
    for row in rd:
        if len(row) != 3:
            continue
        try:
            rows.append({"t0": float(row[0]), "t1": float(row[1]), "label": row[2].strip()})
        except ValueError:
            continue
    return rows


@app.route("/rwc-playalong")
def rwc_playalong():
    """Ground-truth play-along for the RWC-Popular real-audio corpus (the
    project's new primary real-audio training source as of 2026-07-16 — see
    docs/known_issues.md "RWC-Popular... BUNDLED-AUDIO winner"). Same
    mechanism as /gt-playalong-training (waveform + audio element + a synced
    GT chord-block strip so alignment can be judged by ear), but for RWC:
    RWC ships audio and Cho-Bello chord annotations as a matched 1:1 pair
    (no separate YouTube-sourcing/duration-matching step, unlike Billboard),
    so this is expected to check out cleanly — verify by ear before trusting.

    ?song=RWC_Pnnn — audio must already be cached locally as
    docs/audio/rwc_<rwcid-lower>.m4a (RWC's own audio isn't downloaded by
    default; it's streamed from Zenodo via remotezip only by
    scripts/build_rwc_corpus.py during corpus builds. For this demo one
    song's audio was fetched once and converted to m4a — see
    scratchpad/fetch_rwc_demo_song.py).
    """
    from html import escape

    rwcid = lookup_slug(request.args.get("song") or "")
    if not re.fullmatch(r"RWC_P\d{3}", rwcid):
        return "<p>Pass ?song=RWC_Pnnn (e.g. RWC_P001).</p>", 400

    slug = f"rwc_{rwcid.lower()}"
    audio_path = AUDIO_DIR / f"{slug}.m4a"
    if not audio_path.exists():
        return (f"<p>No cached audio for {escape(rwcid)} "
                f"({escape(str(audio_path))} missing). Only a demo subset of "
                f"RWC songs has locally-cached audio; the full 100-song corpus "
                f"is fetched on demand by scripts/build_rwc_corpus.py.</p>"), 404

    gt = _fetch_rwc_chords(rwcid)
    if not gt:
        return f"<p>Could not fetch Cho-Bello chords for {escape(rwcid)}.</p>", 404

    title = f"RWC-Popular {rwcid}"
    duration = max((c["t1"] for c in gt), default=0.0)

    chart_data = {
        "title": title,
        "gt": gt,
        "audioUrl": f"/audio/{audio_path.name}",
        "peaksSlug": slug,
        "duration": duration,
        "backHref": "/",
    }

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>GT play-along: {escape(title)}</title>
<style>
  :root {{ --paper:#f7f3e9; --card:#fffdf6; --ink:#1c1c1c; --rule:#b9b09a; --faint:#8a8371; --accent:#8a2b2b; --line:#e5dcc6; --green:#1f8a5b; }}
  * {{ box-sizing:border-box; -webkit-tap-highlight-color:transparent; }}
  html, body {{ margin:0; background:var(--paper); color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif; }}
  header {{ padding:14px 16px; background:var(--card); border-bottom:1px solid var(--line);
    display:flex; align-items:center; gap:10px; }}
  header a {{ font:600 12px system-ui; color:var(--faint); text-decoration:none; border:1px solid var(--rule);
    border-radius:20px; padding:5px 10px; flex:0 0 auto; }}
  h1 {{ margin:0; font:italic 600 17px Georgia,'Times New Roman',serif; flex:1; min-width:0;
    overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  #container {{ display:flex; flex-direction:column; }}
  #waveWrap {{ position:relative; overflow-x:auto; background:var(--card); border-bottom:1px solid var(--line); }}
  canvas {{ display:block; }}
  #gtStrip {{ position:relative; height:56px; }}
  .gtBlock {{ position:absolute; top:6px; bottom:6px; border-radius:6px; display:flex; align-items:center;
    justify-content:center; font:600 12.5px Georgia,serif; color:#fff; overflow:hidden; white-space:nowrap;
    cursor:pointer; transition:transform .08s, filter .08s; border:1px solid rgba(0,0,0,.15); }}
  .gtBlock.active {{ transform:scaleY(1.12); filter:brightness(1.12); box-shadow:0 0 0 2px var(--ink); z-index:5; }}
  .playhead {{ position:absolute; top:0; bottom:0; width:2px; background:var(--accent); z-index:10;
    pointer-events:none; box-shadow:0 0 4px var(--accent); }}
  #controls {{ padding:12px 16px; background:var(--card); display:flex; align-items:center; gap:14px;
    border-bottom:1px solid var(--line); }}
  audio {{ flex:1; min-width:0; height:34px; }}
  #curChordCard {{ padding:16px; text-align:center; }}
  #curChordLabel {{ font:italic 700 44px Georgia,'Times New Roman',serif; color:var(--ink); }}
  #curChordMeta {{ font:500 13px system-ui; color:var(--faint); margin-top:4px; }}
  #hint {{ padding:0 16px 16px; font:italic 13px Georgia,serif; color:var(--faint); line-height:1.5; max-width:640px; }}
</style>
</head><body>
<div id="container">
  <header>
    <a href="{escape(chart_data['backHref'])}">&larr; home</a>
    <h1>GT play-along — {escape(title)}</h1>
  </header>
  <div id="controls">
    <audio id="audio" controls preload="metadata" src="{escape(chart_data['audioUrl'])}"></audio>
    <span id="timeLbl" style="font:600 12px system-ui;color:var(--faint);white-space:nowrap;">0:00 / 0:00</span>
  </div>
  <div id="curChordCard">
    <div id="curChordLabel">&mdash;</div>
    <div id="curChordMeta">ground truth (RWC-Popular / Cho-Bello annotations) &middot; tap a block below to seek</div>
  </div>
  <div id="waveWrap">
    <canvas id="canvas" height="80"></canvas>
    <div id="gtStrip"></div>
    <div id="playhead" class="playhead"></div>
  </div>
  <p id="hint">Press play and listen. Each block below is one ground-truth chord span from the
    RWC-Popular Cho-Bello hand annotation (bundled 1:1 with this exact audio file — no separate
    YouTube-sourcing/duration-matching step, unlike the Billboard corpus). The highlighted block
    tracks playback in real time &mdash; if the highlight changes before/after you actually hear the
    harmony change, that's a real alignment bug, not a sourcing artifact. Scroll horizontally to see
    the whole song; click any block or the waveform to seek there.</p>
</div>
<script>
const D = {json.dumps(chart_data)};
const NOTE=["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"];
const QTXT={{"":"","maj":"","maj7":"maj7","min":"m","min7":"m7","7":"7","hdim7":"m7♭5","dim":"dim","dim7":"dim7","6":"6","min6":"m6","maj6":"6","9":"9","min9":"m9","sus4":"sus4","sus2":"sus2","aug":"aug"}};
function parseHarte(l){{
  if(!l || l==="N" || l==="X") return {{text:"N.C.", color:"#8a8371"}};
  const p=String(l).split(":");
  const pc={{"C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,"E":4,"Fb":4,"F":5,"E#":5,"F#":6,"Gb":6,"G":7,"G#":8,"Ab":8,"A":9,"A#":10,"Bb":10,"B":11,"Cb":11}}[p[0]];
  const root = pc==null?0:pc;
  let q=(p[1]||"maj").split("(")[0];
  const text = NOTE[root] + (QTXT[q]!=null?QTXT[q]:q);
  const hue = (root*30)%360;
  return {{text, color:`hsl(${{hue}},42%,38%)`}};
}}
const gt = D.gt.map(g=>Object.assign({{}}, g, parseHarte(g.label)));

const scale = 90; // px/sec
const canvas = document.getElementById('canvas');
const gtStrip = document.getElementById('gtStrip');
const playhead = document.getElementById('playhead');
const audio = document.getElementById('audio');
const w = Math.max(320, D.duration * scale);
canvas.width = w; canvas.height = 80;
canvas.style.width = w+'px';
gtStrip.style.width = w+'px';
const ctx = canvas.getContext('2d');

function drawWave(peaks){{
  ctx.clearRect(0,0,w,80);
  ctx.fillStyle = '#efe8d6';
  ctx.fillRect(0,0,w,80);
  if(peaks && peaks.length){{
    ctx.fillStyle = '#b9b09a';
    const n = peaks.length;
    for(let x=0; x<w; x++){{
      const idx = Math.floor(x/w*n);
      const h = Math.max(1,(peaks[idx]||0)*64);
      ctx.fillRect(x, 40-h/2, 1, h);
    }}
  }}
}}
fetch('/api/waveform-peaks/'+encodeURIComponent(D.peaksSlug)).then(r=>r.ok?r.json():null)
  .then(d=>drawWave(d&&d.peaks)).catch(()=>drawWave(null));

gt.forEach((g,i)=>{{
  const b=document.createElement('div');
  b.className='gtBlock';
  b.style.left=(g.t0*scale)+'px';
  b.style.width=Math.max(2,(g.t1-g.t0)*scale-1)+'px';
  b.style.background=g.color;
  b.dataset.i=i;
  if((g.t1-g.t0)*scale > 26) b.textContent=g.text;
  b.title=g.text+'  '+g.t0.toFixed(2)+'s → '+g.t1.toFixed(2)+'s';
  b.onclick=()=>{{ audio.currentTime=g.t0; audio.play(); }};
  gtStrip.appendChild(b);
}});

function fmt(s){{ s=Math.max(0,Math.floor(s||0)); return Math.floor(s/60)+':'+String(s%60).padStart(2,'0'); }}
let lastActive=-1;
function tick(){{
  const t=audio.currentTime||0;
  playhead.style.left=Math.max(0,t*scale)+'px';
  document.getElementById('timeLbl').textContent=fmt(t)+' / '+fmt(D.duration);
  let idx=-1;
  for(let i=0;i<gt.length;i++){{ if(gt[i].t0<=t && t<gt[i].t1){{ idx=i; break; }} }}
  if(idx===-1){{ for(let i=gt.length-1;i>=0;i--){{ if(gt[i].t0<=t){{ idx=i; break; }} }} }}
  if(idx!==lastActive){{
    if(lastActive>=0){{ const prev=gtStrip.children[lastActive]; if(prev) prev.classList.remove('active'); }}
    if(idx>=0){{
      const cur=gtStrip.children[idx];
      if(cur){{ cur.classList.add('active');
        const wrap=document.getElementById('waveWrap');
        const bx=cur.offsetLeft;
        if(bx < wrap.scrollLeft+40 || bx > wrap.scrollLeft+wrap.clientWidth-80){{
          wrap.scrollTo({{left: Math.max(0,bx-120), behavior:'smooth'}});
        }}
      }}
      document.getElementById('curChordLabel').textContent = gt[idx].text;
      document.getElementById('curChordMeta').textContent =
        gt[idx].t0.toFixed(2)+'s → '+gt[idx].t1.toFixed(2)+'s  (span '+(gt[idx].t1-gt[idx].t0).toFixed(2)+'s)  ·  ground truth (RWC-Popular / Cho-Bello)';
    }}
    lastActive=idx;
  }}
}}
audio.addEventListener('timeupdate', tick);
audio.addEventListener('play', ()=>requestAnimationFrame(function loop(){{ tick(); if(!audio.paused) requestAnimationFrame(loop); }}));
document.getElementById('waveWrap').addEventListener('click', e=>{{
  if(e.target.classList.contains('gtBlock')) return;
  const rect=canvas.getBoundingClientRect();
  const x = e.clientX - rect.left + document.getElementById('waveWrap').scrollLeft;
  audio.currentTime = Math.max(0, x/scale);
}});
tick();
</script>
</body></html>"""

    return Response(page, mimetype="text/html")


@app.route("/billboard-gt-triage")
def billboard_gt_triage():
    """Triage list for the ~60-song Billboard GT-offset correction workflow
    (docs/known_issues.md "DATA bug, not display bug"). Flags each song by
    the duration-mismatch signal already computed during corpus search
    (|gt_dur - matched-video duration|): >2s is very likely a genuinely
    different edit (not just a phase shift — re-sourcing candidate, e.g.
    "The Commodores" had a long silent intro), the rest just need an
    offset check/nudge. No audio decoding here — this reads only the small
    cached search-result JSONs, so it's instant even though most of the
    corpus hasn't been downloaded/analysed yet."""
    from html import escape

    merged: dict[str, dict] = {}
    for p in _BILLBOARD_CORPUS_FILES:
        try:
            merged.update(json.loads(p.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            pass

    vid_to_file: dict[str, str] = {}
    for fname, vid in _yt_video_ids.items():
        vid_to_file.setdefault(vid, fname)

    offsets = _load_gt_offsets()
    track_ids_by_vid = {}
    for track_id, v in merged.items():
        best = v.get("best") or []
        if best:
            track_ids_by_vid[best[0]] = track_id

    rows = []
    for track_id, v in merged.items():
        best = v.get("best") or []
        if not best:
            continue
        vid = best[0]
        audio_dur = best[2] if len(best) > 2 else None
        gt_dur = v.get("gt_dur")
        mismatch = abs(audio_dur - gt_dur) if (audio_dur is not None and gt_dur is not None) else None
        severity = "unknown"
        if mismatch is not None:
            severity = "wrong-edit" if mismatch > 2.0 else "check"
        fname = vid_to_file.get(vid, "")
        corr = offsets.get(track_id)
        rows.append({
            "track_id": track_id, "artist": v.get("artist", ""), "title": v.get("title", ""),
            "video_id": vid, "gt_dur": gt_dur, "audio_dur": audio_dur, "mismatch": mismatch,
            "severity": severity, "file": fname,
            "has_offset": bool(corr), "offset_s": (corr or {}).get("offset_s"),
        })
    # worst mismatch first (None sorts last)
    rows.sort(key=lambda r: (-1 if r["mismatch"] is None else 0, -(r["mismatch"] or 0)))

    n_wrong = sum(1 for r in rows if r["severity"] == "wrong-edit")
    n_check = sum(1 for r in rows if r["severity"] == "check")
    n_corrected = sum(1 for r in rows if r["has_offset"])

    def row_html(r):
        cls = {"wrong-edit": "sev-wrong", "check": "sev-check", "unknown": "sev-unknown"}[r["severity"]]
        badge = {"wrong-edit": "likely wrong edit — re-source", "check": "needs offset check",
                  "unknown": "no duration data"}[r["severity"]]
        mism = f'{r["mismatch"]:+.1f}s' if r["mismatch"] is not None else "—"
        offset_badge = (f'<span class="pill pill-done">saved offset {r["offset_s"]:+.2f}s</span>'
                         if r["has_offset"] else '<span class="pill pill-todo">no correction yet</span>')
        if r["file"]:
            link = f'<a class="go" href="/gt-offset-fix?song={escape(r["file"])}">fix offset &rarr;</a>'
        else:
            link = '<a class="go go-dim" href="/library">not analysed yet &rarr;</a>'
        return f"""<tr class="{cls}">
          <td class="title">{escape(r["artist"])} &mdash; {escape(r["title"])}<div class="tid">track {escape(r["track_id"])} &middot; {escape(r["video_id"])}</div></td>
          <td class="num">{r["gt_dur"]:.1f}s</td>
          <td class="num">{"" if r["audio_dur"] is None else f'{r["audio_dur"]:.1f}s'}</td>
          <td class="num mism">{mism}</td>
          <td><span class="badge">{badge}</span></td>
          <td>{offset_badge}</td>
          <td>{link}</td>
        </tr>"""

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Billboard GT offset triage</title>
<style>
  :root {{ --paper:#f7f3e9; --card:#fffdf6; --ink:#1c1c1c; --rule:#b9b09a; --faint:#8a8371; --accent:#8a2b2b; --line:#e5dcc6; --green:#1f8a5b; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--paper); color:var(--ink); font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif; }}
  header {{ padding:16px 20px; background:var(--card); border-bottom:1px solid var(--line); }}
  h1 {{ margin:0 0 4px; font:italic 700 22px Georgia,'Times New Roman',serif; }}
  .sub {{ font:500 13px system-ui; color:var(--faint); }}
  .stats {{ display:flex; gap:18px; margin-top:10px; flex-wrap:wrap; }}
  .stat {{ font:600 12.5px system-ui; padding:4px 10px; border-radius:12px; border:1px solid var(--rule); }}
  .stat.wrong {{ color:#fff; background:var(--accent); border-color:var(--accent); }}
  .stat.check {{ color:#7a5c00; background:#fbe9b0; border-color:#e0c56a; }}
  .stat.done {{ color:#fff; background:var(--green); border-color:var(--green); }}
  table {{ width:100%; border-collapse:collapse; background:var(--card); }}
  th {{ text-align:left; font:600 11px system-ui; color:var(--faint); text-transform:uppercase; letter-spacing:.04em;
    padding:8px 12px; border-bottom:2px solid var(--line); position:sticky; top:0; background:var(--card); }}
  td {{ padding:9px 12px; border-bottom:1px solid var(--line); font:14px system-ui; vertical-align:middle; }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; color:var(--faint); }}
  td.mism {{ font-weight:700; }}
  .tid {{ font:11px system-ui; color:var(--faint); margin-top:2px; }}
  tr.sev-wrong td.mism {{ color:var(--accent); }}
  tr.sev-wrong {{ background:#fbeceb; }}
  tr.sev-check td.mism {{ color:#7a5c00; }}
  .badge {{ font:600 11px system-ui; padding:3px 8px; border-radius:10px; white-space:nowrap; }}
  tr.sev-wrong .badge {{ background:var(--accent); color:#fff; }}
  tr.sev-check .badge {{ background:#fbe9b0; color:#7a5c00; }}
  tr.sev-unknown .badge {{ background:#ddd; color:#555; }}
  .pill {{ font:600 11px system-ui; padding:3px 8px; border-radius:10px; white-space:nowrap; }}
  .pill-done {{ background:var(--green); color:#fff; }}
  .pill-todo {{ background:#eee; color:#888; }}
  a.go {{ font:600 12.5px system-ui; color:var(--accent); text-decoration:none; white-space:nowrap; }}
  a.go-dim {{ color:var(--faint); }}
  .wrap {{ overflow-x:auto; }}
</style>
</head><body>
<header>
  <h1>Billboard GT-offset triage</h1>
  <div class="sub">{len(rows)} corpus songs &middot; ranked by |GT duration &minus; matched-audio duration| &middot; McGill Billboard chords_full is relative to a different master recording than this corpus's YouTube audio &mdash; per-song offsets need hand correction.</div>
  <div class="stats">
    <span class="stat wrong">{n_wrong} likely wrong edit (&gt;2s mismatch)</span>
    <span class="stat check">{n_check} need offset check</span>
    <span class="stat done">{n_corrected} corrected so far</span>
  </div>
</header>
<div class="wrap">
<table>
  <thead><tr><th>song</th><th>GT dur</th><th>audio dur</th><th>mismatch</th><th>flag</th><th>correction</th><th></th></tr></thead>
  <tbody>
    {"".join(row_html(r) for r in rows)}
  </tbody>
</table>
</div>
</body></html>"""
    return Response(page, mimetype="text/html")


@app.route("/gt-offset-fix")
def gt_offset_fix():
    """Editable GT-offset correction view: extends /gt-playalong-training
    with a whole-timeline nudge control (fine 0.1s / coarse 1s steps),
    live re-sync of the GT block strip as the offset changes, a pre-seeded
    first-onset-alignment guess (see _estimate_gt_offset — same heuristic
    as scratchpad/offset_final.py, NOT reliable alone), and a save action
    that persists to data/cache/billboard_gt_offsets.json via
    /api/gt-offset/<track_id>. Once saved, _gt_chords_for_video() applies
    the offset everywhere (training-mode chart, gt-playalong*) automatically.

    ?song=<inferred_*.html chart filename> — same song resolution as
    /gt-playalong-training (needs the song already analysed/downloaded)."""
    from html import escape

    filename = request.args.get("song") or ""
    filename = re.sub(r"[^A-Za-z0-9_.\-]", "", filename)
    if not filename.startswith("inferred_") or not filename.endswith(".html"):
        return "<p>Pass ?song=inferred_&lt;slug&gt;.html (a training-mode chart).</p>", 400
    if not (PLOTS_DIR / filename).exists():
        return f"<p>No chart {escape(filename)}.</p>", 404

    video_id = _yt_video_ids.get(filename, "")
    if not video_id:
        return f"<p>{escape(filename)} has no YouTube video id.</p>", 404
    track_id, gt_raw = _gt_chords_for_video_raw(video_id)
    if not gt_raw:
        return (f"<p>{escape(filename)} is not a training-corpus song (no McGill "
                f"Billboard chords_full for video {escape(video_id)}).</p>"), 404

    audio_meta = _yt_audio_meta.get(filename) or {}
    audio_url = audio_meta.get("audio", "")
    audio_path = AUDIO_DIR / Path(audio_url).name if audio_url else None
    if not audio_url or not audio_path or not audio_path.exists():
        return f"<p>No downloaded audio for {escape(filename)}.</p>", 404
    slug = audio_path.stem

    saved = _load_gt_offsets().get(track_id or "", {})
    if "offset_s" in saved:
        initial_offset = saved["offset_s"]
        offset_source = saved.get("source", "manual")
    else:
        try:
            initial_offset = _estimate_gt_offset(audio_path, gt_raw)
        except Exception as e:
            log.warning("gt-offset-fix: onset guess failed for %s (%s)", slug, e)
            initial_offset = 0.0
        offset_source = "auto-onset (unsaved guess)"

    title = filename.removeprefix("inferred_").removesuffix(".html").replace("_", " ").title()
    duration = max((c["t1"] for c in gt_raw), default=0.0) + max(0.0, initial_offset) + 5.0

    chart_data = {
        "title": title, "gtRaw": gt_raw, "trackId": track_id,
        "audioUrl": f"/audio/{audio_path.name}", "peaksSlug": slug,
        "duration": duration, "backHref": f"/chart/{filename}",
        "initialOffset": initial_offset, "offsetSource": offset_source,
        "hasSaved": "offset_s" in saved,
    }

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>GT offset fix: {escape(title)}</title>
<style>
  :root {{ --paper:#f7f3e9; --card:#fffdf6; --ink:#1c1c1c; --rule:#b9b09a; --faint:#8a8371; --accent:#8a2b2b; --line:#e5dcc6; --green:#1f8a5b; }}
  * {{ box-sizing:border-box; -webkit-tap-highlight-color:transparent; }}
  html, body {{ margin:0; background:var(--paper); color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif; }}
  header {{ padding:14px 16px; background:var(--card); border-bottom:1px solid var(--line);
    display:flex; align-items:center; gap:10px; }}
  header a {{ font:600 12px system-ui; color:var(--faint); text-decoration:none; border:1px solid var(--rule);
    border-radius:20px; padding:5px 10px; flex:0 0 auto; }}
  h1 {{ margin:0; font:italic 600 17px Georgia,'Times New Roman',serif; flex:1; min-width:0;
    overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  #container {{ display:flex; flex-direction:column; }}
  #waveWrap {{ position:relative; overflow-x:auto; background:var(--card); border-bottom:1px solid var(--line); }}
  canvas {{ display:block; }}
  #gtStrip {{ position:relative; height:56px; }}
  .gtBlock {{ position:absolute; top:6px; bottom:6px; border-radius:6px; display:flex; align-items:center;
    justify-content:center; font:600 12.5px Georgia,serif; color:#fff; overflow:hidden; white-space:nowrap;
    cursor:pointer; transition:filter .08s; border:1px solid rgba(0,0,0,.15); }}
  .gtBlock.active {{ filter:brightness(1.12); box-shadow:0 0 0 2px var(--ink); z-index:5; }}
  .playhead {{ position:absolute; top:0; bottom:0; width:2px; background:var(--accent); z-index:10;
    pointer-events:none; box-shadow:0 0 4px var(--accent); }}
  #controls {{ padding:10px 16px; background:var(--card); display:flex; align-items:center; gap:12px;
    border-bottom:1px solid var(--line); flex-wrap:wrap; }}
  audio {{ flex:1; min-width:220px; height:34px; }}
  #offsetBar {{ padding:12px 16px; background:#fbf6e6; display:flex; align-items:center; gap:8px;
    border-bottom:1px solid var(--line); flex-wrap:wrap; }}
  #offsetBar label {{ font:600 12px system-ui; color:var(--faint); }}
  #offsetBar button {{ font:700 14px system-ui; border:1px solid var(--rule); background:var(--card);
    color:var(--ink); border-radius:8px; padding:6px 10px; cursor:pointer; }}
  #offsetBar button:active {{ background:var(--line); }}
  #offsetVal {{ font:700 16px 'SF Mono',Menlo,monospace; min-width:80px; text-align:center;
    padding:6px 8px; border:1px solid var(--rule); border-radius:8px; background:var(--card); }}
  #saveBtn {{ font:700 13px system-ui; background:var(--green); color:#fff; border:none; border-radius:8px;
    padding:8px 16px; cursor:pointer; }}
  #saveBtn:disabled {{ background:#bbb; cursor:default; }}
  #resetBtn {{ font:600 12px system-ui; color:var(--accent); background:none; border:1px solid var(--accent);
    border-radius:8px; padding:7px 12px; cursor:pointer; }}
  #status {{ font:600 12px system-ui; color:var(--faint); }}
  #status.dirty {{ color:#7a5c00; }}
  #status.saved {{ color:var(--green); }}
  #curChordCard {{ padding:14px; text-align:center; }}
  #curChordLabel {{ font:italic 700 40px Georgia,'Times New Roman',serif; color:var(--ink); }}
  #curChordMeta {{ font:500 13px system-ui; color:var(--faint); margin-top:4px; }}
  #hint {{ padding:0 16px 16px; font:italic 13px Georgia,serif; color:var(--faint); line-height:1.5; max-width:640px; }}
</style>
</head><body>
<div id="container">
  <header>
    <a href="{escape(chart_data['backHref'])}">&larr; chart</a>
    <a href="/billboard-gt-triage">&larr; triage list</a>
    <h1>GT offset fix — {escape(title)}</h1>
  </header>
  <div id="controls">
    <audio id="audio" controls preload="metadata" src="{escape(chart_data['audioUrl'])}"></audio>
    <span id="timeLbl" style="font:600 12px system-ui;color:var(--faint);white-space:nowrap;">0:00 / 0:00</span>
  </div>
  <div id="offsetBar">
    <label>whole-timeline offset</label>
    <button data-d="-1">&laquo; 1s</button>
    <button data-d="-0.1">&lsaquo; .1s</button>
    <span id="offsetVal">+0.00s</span>
    <button data-d="0.1">.1s &rsaquo;</button>
    <button data-d="1">1s &raquo;</button>
    <button id="resetBtn" title="reset to the auto onset-alignment guess">auto-guess</button>
    <button id="saveBtn">save correction</button>
    <span id="status"></span>
  </div>
  <div id="curChordCard">
    <div id="curChordLabel">&mdash;</div>
    <div id="curChordMeta">ground truth (McGill Billboard) &middot; tap a block below to seek</div>
  </div>
  <div id="waveWrap">
    <canvas id="canvas" height="80"></canvas>
    <div id="gtStrip"></div>
    <div id="playhead" class="playhead"></div>
  </div>
  <p id="hint">Positive offset shifts GT chord boundaries LATER (use when the audio's harmony change
    happens after the raw GT timestamp — the common case, since Billboard's masters usually have less
    lead-in than the YouTube upload). Nudge with the buttons or type an exact value, watch the blocks
    slide against the waveform, then press play and confirm the highlighted block matches what you hear
    before saving. The auto-guess (first strong onset vs GT's first chord) is a starting point only —
    it is fooled by intro flourishes/drum pickups on some songs, so always verify by ear.</p>
</div>
<script>
const D = {json.dumps(chart_data)};
const NOTE=["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"];
const QTXT={{"":"","maj":"","maj7":"maj7","min":"m","min7":"m7","7":"7","hdim7":"m7♭5","dim":"dim","dim7":"dim7","6":"6","min6":"m6","maj6":"6","9":"9","min9":"m9","sus4":"sus4","sus2":"sus2","aug":"aug"}};
function parseHarte(l){{
  if(!l || l==="N" || l==="X") return {{text:"N.C.", color:"#8a8371"}};
  const p=String(l).split(":");
  const pc={{"C":0,"C#":1,"Db":1,"D":2,"D#":3,"Eb":3,"E":4,"Fb":4,"F":5,"E#":5,"F#":6,"Gb":6,"G":7,"G#":8,"Ab":8,"A":9,"A#":10,"Bb":10,"B":11,"Cb":11}}[p[0]];
  const root = pc==null?0:pc;
  let q=(p[1]||"maj").split("(")[0];
  const text = NOTE[root] + (QTXT[q]!=null?QTXT[q]:q);
  const hue = (root*30)%360;
  return {{text, color:`hsl(${{hue}},42%,38%)`}};
}}
const gtParsed = D.gtRaw.map(g=>Object.assign({{}}, g, parseHarte(g.label)));
let offset = D.initialOffset;
let dirty = !D.hasSaved && Math.abs(offset) > 1e-9;

const scale = 90; // px/sec
const canvas = document.getElementById('canvas');
const gtStrip = document.getElementById('gtStrip');
const playhead = document.getElementById('playhead');
const audio = document.getElementById('audio');
const w = Math.max(320, D.duration * scale);
canvas.width = w; canvas.height = 80;
canvas.style.width = w+'px';
gtStrip.style.width = w+'px';
const ctx = canvas.getContext('2d');

function drawWave(peaks){{
  ctx.clearRect(0,0,w,80);
  ctx.fillStyle = '#efe8d6';
  ctx.fillRect(0,0,w,80);
  if(peaks && peaks.length){{
    ctx.fillStyle = '#b9b09a';
    const n = peaks.length;
    for(let x=0; x<w; x++){{
      const idx = Math.floor(x/w*n);
      const h = Math.max(1,(peaks[idx]||0)*64);
      ctx.fillRect(x, 40-h/2, 1, h);
    }}
  }}
}}
fetch('/api/waveform-peaks/'+encodeURIComponent(D.peaksSlug)).then(r=>r.ok?r.json():null)
  .then(d=>drawWave(d&&d.peaks)).catch(()=>drawWave(null));

function shifted(){{ return gtParsed.map(g=>({{...g, t0:g.t0+offset, t1:g.t1+offset}})); }}
let blocks = [];
function renderBlocks(){{
  gtStrip.innerHTML='';
  blocks = shifted();
  blocks.forEach((g,i)=>{{
    const b=document.createElement('div');
    b.className='gtBlock';
    b.style.left=(Math.max(0,g.t0)*scale)+'px';
    b.style.width=Math.max(2,(g.t1-g.t0)*scale-1)+'px';
    b.style.background=g.color;
    b.dataset.i=i;
    if((g.t1-g.t0)*scale > 26) b.textContent=g.text;
    b.title=g.text+'  '+g.t0.toFixed(2)+'s -> '+g.t1.toFixed(2)+'s';
    b.onclick=()=>{{ audio.currentTime=Math.max(0,g.t0); audio.play(); }};
    gtStrip.appendChild(b);
  }});
}}
renderBlocks();

function fmtOffset(v){{ return (v>=0?'+':'')+v.toFixed(2)+'s'; }}
function refreshOffsetUI(){{
  document.getElementById('offsetVal').textContent = fmtOffset(offset);
  const status = document.getElementById('status');
  if(dirty){{ status.textContent='unsaved changes'; status.className='dirty'; }}
  else {{ status.textContent = D.hasSaved ? 'saved' : ('auto guess (' + D.offsetSource + ')'); status.className = D.hasSaved ? 'saved' : ''; }}
  document.getElementById('saveBtn').disabled = false;
}}
refreshOffsetUI();

document.querySelectorAll('#offsetBar button[data-d]').forEach(btn=>{{
  btn.onclick=()=>{{
    offset = Math.round((offset + parseFloat(btn.dataset.d))*1000)/1000;
    dirty = true;
    renderBlocks();
    refreshOffsetUI();
    tick();
  }};
}});
document.getElementById('resetBtn').onclick=()=>{{
  offset = D.initialOffset;
  dirty = !D.hasSaved;
  renderBlocks(); refreshOffsetUI(); tick();
}};
document.getElementById('saveBtn').onclick=async()=>{{
  const status = document.getElementById('status');
  status.textContent='saving...'; status.className='dirty';
  try{{
    const r = await fetch('/api/gt-offset/'+encodeURIComponent(D.trackId), {{
      method:'POST', headers:{{'Content-Type':'application/json'}},
      body: JSON.stringify({{offset_s: offset, source: 'manual'}})
    }});
    if(!r.ok) throw new Error('save failed');
    D.hasSaved = true; dirty = false;
    status.textContent='saved'; status.className='saved';
  }}catch(e){{
    status.textContent='save failed — retry'; status.className='dirty';
  }}
}};

function fmt(s){{ s=Math.max(0,Math.floor(s||0)); return Math.floor(s/60)+':'+String(s%60).padStart(2,'0'); }}
let lastActive=-1;
function tick(){{
  const t=audio.currentTime||0;
  playhead.style.left=Math.max(0,t*scale)+'px';
  document.getElementById('timeLbl').textContent=fmt(t)+' / '+fmt(D.duration);
  let idx=-1;
  for(let i=0;i<blocks.length;i++){{ if(blocks[i].t0<=t && t<blocks[i].t1){{ idx=i; break; }} }}
  if(idx===-1){{ for(let i=blocks.length-1;i>=0;i--){{ if(blocks[i].t0<=t){{ idx=i; break; }} }} }}
  if(idx!==lastActive){{
    if(lastActive>=0){{ const prev=gtStrip.children[lastActive]; if(prev) prev.classList.remove('active'); }}
    if(idx>=0){{
      const cur=gtStrip.children[idx];
      if(cur){{ cur.classList.add('active');
        const wrap=document.getElementById('waveWrap');
        const bx=cur.offsetLeft;
        if(bx < wrap.scrollLeft+40 || bx > wrap.scrollLeft+wrap.clientWidth-80){{
          wrap.scrollTo({{left: Math.max(0,bx-120), behavior:'smooth'}});
        }}
      }}
      document.getElementById('curChordLabel').textContent = blocks[idx].text;
      document.getElementById('curChordMeta').textContent =
        blocks[idx].t0.toFixed(2)+'s -> '+blocks[idx].t1.toFixed(2)+'s  (span '+(blocks[idx].t1-blocks[idx].t0).toFixed(2)+'s)  ·  ground truth (McGill Billboard), offset '+fmtOffset(offset);
    }}
    lastActive=idx;
  }}
}}
audio.addEventListener('timeupdate', tick);
audio.addEventListener('play', ()=>requestAnimationFrame(function loop(){{ tick(); if(!audio.paused) requestAnimationFrame(loop); }}));
document.getElementById('waveWrap').addEventListener('click', e=>{{
  if(e.target.classList.contains('gtBlock')) return;
  const rect=canvas.getBoundingClientRect();
  const x = e.clientX - rect.left + document.getElementById('waveWrap').scrollLeft;
  audio.currentTime = Math.max(0, x/scale);
}});
tick();
</script>
</body></html>"""

    return Response(page, mimetype="text/html")


@app.route("/bar1-offset-fix")
def bar1_offset_fix():
    """Set-bar-1 tool: shift the PHASE of the chart's own bar grid (which
    detected beat is beat 1 of bar 1), distinct from the GT-offset tool above
    (which corrects ground-truth chord timestamps) and from the step-size fix
    already applied in chord_pipeline_v1/render_youtube_chart.py (real
    per-beat tempo via start_beat_idx — see docs/known_issues.md "Chart
    bar-layout bug"). That fix corrected how many chords land in each bar;
    it did not correct WHERE bar 1 starts if the beat tracker's beat 0 isn't
    the real downbeat (e.g. a pickup measure, or the tracker locking onto an
    off-beat accent).

    Mirrors /gt-offset-fix's pattern (waveform + audio element + overlay grid
    + nudge/slider + save) but applied to the chart's own bar grid instead of
    GT chords, and drawn as a flat linear timeline (not the iReal-style
    per-bar-box chart) so alignment is easy to see/hear precisely.

    ?song=<inferred_*.html chart filename>. Chord bar/beat/t0/t1 come from
    the chart's own baked payload (payload_from_chart_html); abs_beat is
    reconstructed as bar*bpb+beat, which is exact as long as the chart was
    last baked with offset_beats=0 (true for every chart today — this is a
    new feature). If a chart is later re-baked with a nonzero saved offset,
    re-opening this tool would reconstruct abs_beat already shifted; not
    fixed here (documented, not silently wrong: the slider would then be
    relative to the already-applied offset rather than absolute)."""
    from html import escape
    from harmonia.output.chart_model import payload_from_chart_html

    filename = request.args.get("song") or ""
    filename = re.sub(r"[^A-Za-z0-9_.\-]", "", filename)
    if not filename.startswith("inferred_") or not filename.endswith(".html"):
        return "<p>Pass ?song=inferred_&lt;slug&gt;.html (a rendered chart).</p>", 400
    if not (PLOTS_DIR / filename).exists():
        return f"<p>No chart {escape(filename)}.</p>", 404

    slug = filename.removeprefix("inferred_").removesuffix(".html")
    payload = payload_from_chart_html(PLOTS_DIR / filename)
    bpb = payload.get("bpb") or 4
    n_bars = payload.get("nBars") or 1
    offset_lo, offset_hi = _bar1_offset_bounds(bpb, n_bars)
    chords_raw = [
        {"t0": float(c.get("t0", 0.0)), "t1": float(c.get("t1", 0.0)),
         "abs_beat": int(c.get("bar", 0)) * bpb + int(c.get("beat", 0)),
         "label": ((c.get("lv") or {}).get("exact") or {}).get("ireal", "?")}
        for c in payload.get("chords", [])
    ]

    audio_meta = _yt_audio_meta.get(filename) or {}
    audio_url = audio_meta.get("audio", "")
    audio_path = AUDIO_DIR / Path(audio_url).name if audio_url else None
    if not audio_url or not audio_path or not audio_path.exists():
        return f"<p>No downloaded audio for {escape(filename)}.</p>", 404
    peaks_slug = audio_path.stem

    saved = _load_bar1_offsets().get(slug, {})
    initial_offset = int(saved.get("offset_beats", 0))

    title = slug.replace("_", " ").title()
    duration = max((c["t1"] for c in chords_raw), default=0.0) + 5.0

    chart_data = {
        "title": title, "chords": chords_raw, "slug": slug, "bpb": bpb,
        "audioUrl": f"/audio/{audio_path.name}", "peaksSlug": peaks_slug,
        "duration": duration, "backHref": f"/chart/{filename}",
        "initialOffset": initial_offset, "hasSaved": "offset_beats" in saved,
        "offsetLo": offset_lo, "offsetHi": offset_hi,
    }

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no">
<title>Set bar 1: {escape(title)}</title>
<style>
  :root {{ --paper:#f7f3e9; --card:#fffdf6; --ink:#1c1c1c; --rule:#b9b09a; --faint:#8a8371; --accent:#8a2b2b; --line:#e5dcc6; --green:#1f8a5b; }}
  * {{ box-sizing:border-box; -webkit-tap-highlight-color:transparent; }}
  html, body {{ margin:0; background:var(--paper); color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif; }}
  header {{ padding:14px 16px; background:var(--card); border-bottom:1px solid var(--line);
    display:flex; align-items:center; gap:10px; }}
  /* Matches the maroon "← Charts" pill every other page in the app shows
     (_BACK_BUTTON_HTML in harmonia_server.py) — this tool page previously
     used a plain bordered text link here, which was one of the "looks like
     a different, older app" signals reported 2026-07-17. */
  header a {{ font:700 13px system-ui,sans-serif; color:#fff; text-decoration:none;
    background:var(--accent); border:none; border-radius:20px; padding:7px 13px;
    box-shadow:0 2px 8px #0002; flex:0 0 auto; transition:transform .1s ease; }}
  header a:active {{ transform:scale(.93); }}
  h1 {{ margin:0; font:italic 600 17px Georgia,'Times New Roman',serif; flex:1; min-width:0;
    overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }}
  #container {{ display:flex; flex-direction:column; }}
  #waveWrap {{ position:relative; overflow-x:auto; background:var(--card); border-bottom:1px solid var(--line); }}
  canvas {{ display:block; }}
  #barStrip {{ position:relative; height:56px; }}
  .barLine {{ position:absolute; top:0; bottom:0; width:2px; background:var(--accent); }}
  .barLine.b1 {{ background:var(--green); width:3px; }}
  .barLabel {{ position:absolute; top:4px; font:700 11px Georgia,serif; color:var(--accent);
    white-space:nowrap; transform:translateX(2px); }}
  .barLabel.b1 {{ color:var(--green); font-size:13px; }}
  .chordLbl {{ position:absolute; top:30px; font:italic 11px Georgia,serif; color:var(--faint);
    white-space:nowrap; transform:translateX(2px); }}
  .playhead {{ position:absolute; top:0; bottom:0; width:2px; background:#1c1c1c88; z-index:10;
    pointer-events:none; }}
  #controls {{ padding:10px 16px; background:var(--card); display:flex; align-items:center; gap:12px;
    border-bottom:1px solid var(--line); flex-wrap:wrap; }}
  audio {{ flex:1; min-width:220px; height:34px; }}
  #offsetBar {{ padding:12px 16px; background:#fbf6e6; display:flex; flex-direction:column; gap:10px;
    border-bottom:1px solid var(--line); }}
  #offsetBar .row {{ display:flex; align-items:center; gap:10px; flex-wrap:wrap; }}
  #offsetBar label {{ font:600 12px system-ui; color:var(--faint); }}
  #offsetSlider {{ flex:1; min-width:160px; }}
  /* Pill buttons matching chart_interactive.py's .modal-panel button convention
     (#efe9d9 fill, #cfc7ae border, scale-down active state) instead of the
     plainer bordered rectangles this tool used before. */
  #offsetBar button {{ font:600 12px system-ui,sans-serif; border:1px solid #cfc7ae; background:#efe9d9;
    color:#4a4636; border-radius:8px; padding:7px 14px; cursor:pointer; transition:transform .1s ease, background .12s; }}
  #offsetBar button:active {{ background:#e0d5c0; transform:scale(.94); }}
  /* Whole-bar nudge buttons are visually distinct (accent-tinted) — they're a
     different, bigger-consequence operation (excluding N whole bars of
     intro, not a sub-bar phase nudge) and should not look identical to the
     "1 beat" buttons. See docs/known_issues.md 2026-07-17 entry. */
  #offsetBar button.barNudge {{ border-color:var(--accent); color:var(--accent); font-weight:700; }}
  #offsetBar button.barNudge:active {{ background:#8a2b2b22; }}
  #offsetVal {{ font:700 15px 'SF Mono',Menlo,monospace; min-width:110px; text-align:center;
    padding:6px 8px; border:1px solid var(--rule); border-radius:8px; background:var(--card); }}
  #saveBtn {{ font:700 13px system-ui,sans-serif; background:var(--green); color:#fff; border:none;
    border-radius:20px; padding:9px 18px; cursor:pointer; box-shadow:0 2px 8px #0002;
    transition:transform .1s ease; }}
  #saveBtn:active {{ transform:scale(.94); }}
  #status {{ font:600 12px system-ui; color:var(--faint); }}
  #status.dirty {{ color:#7a5c00; }}
  #status.saved {{ color:var(--green); }}
  #hint {{ padding:0 16px 16px; font:italic 13px Georgia,serif; color:var(--faint); line-height:1.5; max-width:640px; }}
</style>
</head><body>
<div id="container">
  <header>
    <a href="{escape(chart_data['backHref'])}">&larr; chart</a>
    <h1>Set bar 1 — {escape(title)}</h1>
  </header>
  <div id="controls">
    <audio id="audio" controls preload="metadata" src="{escape(chart_data['audioUrl'])}"></audio>
    <span id="timeLbl" style="font:600 12px system-ui;color:var(--faint);white-space:nowrap;">0:00 / 0:00</span>
  </div>
  <div id="offsetBar">
    <div class="row">
      <label>shift bar 1 by</label>
      <input type="range" id="offsetSlider" min="{offset_lo}" max="{offset_hi}" step="1" value="{initial_offset}">
      <span id="offsetVal">0 beats</span>
    </div>
    <div class="row">
      <button data-d="-1">&laquo; 1 beat</button>
      <button data-d="1">1 beat &raquo;</button>
      <button data-d="-{bpb}" class="barNudge">&laquo; 1 bar</button>
      <button data-d="{bpb}" class="barNudge">1 bar &raquo;</button>
      <button id="resetBtn" title="reset to 0 (tracker's own beat 0)">reset to 0</button>
      <button id="saveBtn">save bar-1 offset</button>
      <span id="status"></span>
    </div>
  </div>
  <div id="waveWrap">
    <canvas id="canvas" height="80"></canvas>
    <div id="barStrip"></div>
    <div id="playhead" class="playhead"></div>
  </div>
  <p id="hint">Two different corrections share this one control, both measured in beats
    (bpb={bpb} here): a <b>sub-bar nudge</b> (the "1 beat" buttons / slider, magnitude &lt; bpb)
    fine-tunes which detected beat counts as beat 1 — the first N beats become a pickup absorbed
    into bar 0. A <b>whole-bar shift</b> (the "1 bar" buttons, or any multiple of {bpb} beats)
    instead EXCLUDES that many bars of intro/pickup material from the front of the numbered
    chart entirely — use this when the beat tracker counted an instrumental intro as bars 1..N
    when the real bar 1 (e.g. the vocal entry) starts later. Excluded bars are hidden from the
    numbered chart, not deleted from the underlying audio/analysis — reachable range here is
    [{offset_lo}, {offset_hi}] beats, capped so at least one bar always remains. The green line
    marks where the new bar 1 starts; red lines mark every other bar. Drag the slider (or nudge)
    while listening until the green line lands exactly on the real downbeat, then save. Takes
    effect next time this song is analysed.</p>
</div>
<script>
const D = {json.dumps(chart_data)};
let offset = D.initialOffset;
let dirty = false;
const scale = 90; // px/sec
const bpb = D.bpb;
const offsetLo = D.offsetLo, offsetHi = D.offsetHi;
const canvas = document.getElementById('canvas');
const barStrip = document.getElementById('barStrip');
const playhead = document.getElementById('playhead');
const audio = document.getElementById('audio');
const w = Math.max(320, D.duration * scale);
canvas.width = w; canvas.height = 80;
canvas.style.width = w+'px';
barStrip.style.width = w+'px';
const ctx = canvas.getContext('2d');

function drawWave(peaks){{
  ctx.clearRect(0,0,w,80);
  ctx.fillStyle = '#efe8d6';
  ctx.fillRect(0,0,w,80);
  if(peaks && peaks.length){{
    ctx.fillStyle = '#b9b09a';
    const n = peaks.length;
    for(let x=0; x<w; x++){{
      const idx = Math.floor(x/w*n);
      const h = Math.max(1,(peaks[idx]||0)*64);
      ctx.fillRect(x, 40-h/2, 1, h);
    }}
  }}
}}
fetch('/api/waveform-peaks/'+encodeURIComponent(D.peaksSlug)).then(r=>r.ok?r.json():null)
  .then(d=>drawWave(d&&d.peaks)).catch(()=>drawWave(null));

// eff_beat is NOT clamped to 0 before dividing — mirrors the server-side fix
// in render_youtube_chart.py::chart_to_interactive_inputs. JS's own % is
// truncated (can return negative), not floor-mod, so beatOf needs the
// ((a%b)+b)%b idiom to match Python's // and % for negative eff_beat
// (pickup chords before the true bar-1 downbeat).
function effBeat(absBeat){{ return absBeat - offset; }}
function barOf(absBeat){{ return Math.max(0, Math.floor(effBeat(absBeat)/bpb)); }}
function beatOf(absBeat){{ const e=effBeat(absBeat); return ((e%bpb)+bpb)%bpb; }}

// Chord onsets are the only real anchor we have to real audio time (there is
// no continuous beat-times array at this API boundary — see the route's
// docstring). A bar boundary rarely coincides with an actual chord CHANGE
// (harmony often holds across a bar line), so drawing a line only where a
// chord happens to start there left most offsets showing NO visible grid at
// all. Instead, build (abs_beat, t0) control points from every known chord
// onset and linearly interpolate/extrapolate to get a time for ANY abs_beat
// — a genuine per-bar grid, decoupled from where the harmony changes.
const _pts = (()=>{{
  const seen = new Map();
  D.chords.forEach(c=>{{ if(!seen.has(c.abs_beat)) seen.set(c.abs_beat, c.t0); }});
  return Array.from(seen.entries()).map(([b,t])=>({{b,t}})).sort((a,z)=>a.b-z.b);
}})();
function timeForAbsBeat(b){{
  if(_pts.length===0) return 0;
  if(_pts.length===1) return _pts[0].t;
  if(b<=_pts[0].b){{
    const [p0,p1]=[_pts[0],_pts[1]];
    const slope=(p1.t-p0.t)/(p1.b-p0.b||1);
    return p0.t + slope*(b-p0.b);
  }}
  if(b>=_pts[_pts.length-1].b){{
    const [p0,p1]=[_pts[_pts.length-2],_pts[_pts.length-1]];
    const slope=(p1.t-p0.t)/(p1.b-p0.b||1);
    return p1.t + slope*(b-p1.b);
  }}
  for(let i=0;i<_pts.length-1;i++){{
    if(_pts[i].b<=b && b<=_pts[i+1].b){{
      const slope=(_pts[i+1].t-_pts[i].t)/(_pts[i+1].b-_pts[i].b||1);
      return _pts[i].t + slope*(b-_pts[i].b);
    }}
  }}
  return _pts[_pts.length-1].t;
}}

function renderStrip(){{
  barStrip.innerHTML='';
  const maxAbsBeat = Math.max(0, ..._pts.map(p=>p.b));
  const maxBar = barOf(maxAbsBeat);
  for(let b=0; b<=maxBar; b++){{
    // abs_beat at which bar b's eff_beat hits exactly 0 (its true downbeat) —
    // unclamped, so bar 0's own downbeat is at abs_beat=offset even though
    // bar 0 also absorbs any earlier pickup beats (eff_beat<0, still bar 0
    // after clamping in barOf/beatOf above).
    const boundaryAbsBeat = b*bpb + offset;
    const t = Math.max(0, timeForAbsBeat(boundaryAbsBeat));
    const line=document.createElement('div');
    line.className='barLine'+(b===0?' b1':'');
    line.style.left=(t*scale)+'px';
    barStrip.appendChild(line);
    const lbl=document.createElement('div');
    lbl.className='barLabel'+(b===0?' b1':'');
    lbl.style.left=(t*scale)+'px';
    lbl.textContent = b===0 ? 'bar 1' : ('bar '+(b+1));
    barStrip.appendChild(lbl);
  }}
  D.chords.forEach(c=>{{
    const cl=document.createElement('div');
    cl.className='chordLbl';
    cl.style.left=(c.t0*scale)+'px';
    cl.textContent=c.label;
    barStrip.appendChild(cl);
  }});
}}
renderStrip();

function fmtOffset(v){{ return (v>=0?'+':'')+v+' beat'+(Math.abs(v)===1?'':'s'); }}
function refreshUI(){{
  document.getElementById('offsetSlider').value = offset;
  document.getElementById('offsetVal').textContent = fmtOffset(offset);
  const status = document.getElementById('status');
  if(dirty){{ status.textContent='unsaved changes'; status.className='dirty'; }}
  else {{ status.textContent = D.hasSaved ? 'saved' : 'tracker default (0)'; status.className = D.hasSaved ? 'saved' : ''; }}
}}
refreshUI();

document.getElementById('offsetSlider').oninput=(e)=>{{
  offset = parseInt(e.target.value, 10) || 0;
  dirty = true;
  renderStrip(); refreshUI();
}};
document.querySelectorAll('#offsetBar button[data-d]').forEach(btn=>{{
  btn.onclick=()=>{{
    // Clamp to [offsetLo, offsetHi] — the server-computed safe range from
    // _bar1_offset_bounds() (caps at n_bars-1 bars so the chart can't be
    // fully emptied). The "1 beat" buttons nudge a sub-bar PHASE; the
    // "1 bar" buttons (delta = +/-bpb) nudge a whole-bar INTRO EXCLUSION —
    // both share this one offset value and this one clamp. Unlike the old
    // +/-bpb-only clamp, whole-bar multiples are now a legitimate, intended
    // destination, not an overshoot to guard against — see
    // docs/known_issues.md 2026-07-17 "Yesterday align-tool" entry (a real
    // case needed +8 beats = 2 bars to skip an instrumental intro).
    offset = Math.max(offsetLo, Math.min(offsetHi, offset + parseInt(btn.dataset.d, 10)));
    dirty = true;
    renderStrip(); refreshUI();
  }};
}});
document.getElementById('resetBtn').onclick=()=>{{
  offset = 0; dirty = (0 !== D.initialOffset);
  renderStrip(); refreshUI();
}};
document.getElementById('saveBtn').onclick=async()=>{{
  const status = document.getElementById('status');
  status.textContent='saving...'; status.className='dirty';
  try{{
    const r = await fetch('/api/bar1-offset/'+encodeURIComponent(D.slug), {{
      method:'POST', headers:{{'Content-Type':'application/json'}},
      body: JSON.stringify({{offset_beats: offset}})
    }});
    if(!r.ok) throw new Error('save failed');
    D.hasSaved = true; D.initialOffset = offset; dirty = false;
    status.textContent='saved'; status.className='saved';
  }}catch(e){{
    status.textContent='save failed — retry'; status.className='dirty';
  }}
}};

function fmt(s){{ s=Math.max(0,Math.floor(s||0)); return Math.floor(s/60)+':'+String(s%60).padStart(2,'0'); }}
function tick(){{
  const t=audio.currentTime||0;
  playhead.style.left=Math.max(0,t*scale)+'px';
  document.getElementById('timeLbl').textContent=fmt(t)+' / '+fmt(D.duration);
}}
audio.addEventListener('timeupdate', tick);
audio.addEventListener('play', ()=>requestAnimationFrame(function loop(){{ tick(); if(!audio.paused) requestAnimationFrame(loop); }}));
document.getElementById('waveWrap').addEventListener('click', e=>{{
  const rect=canvas.getBoundingClientRect();
  const x = e.clientX - rect.left + document.getElementById('waveWrap').scrollLeft;
  audio.currentTime = Math.max(0, x/scale);
  audio.play();
}});
tick();
</script>
</body></html>"""

    return Response(page, mimetype="text/html")


def _perfect_grid_for(slug: str, bpm_prior: float = 140.0, fit_max_bar: int = 7):
    """Load (or compute) the perfect constant-tempo grid for <slug>.

    Prefers a precomputed sidecar written by scripts/fit_beat_grid.py
    (docs/plots/annotations/irealb_<slug>_perfectgrid.json). Falls back to
    fitting on the fly from the gt-align corrections. Returns the fit dict or
    None if no corrected annotations exist.
    """
    sidecar = ANNOT_DIR / f"irealb_{slug}_perfectgrid.json"
    if sidecar.exists():
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except ValueError:
            pass
    annot = ANNOT_DIR / f"irealb_{slug}.html.json"
    if not annot.exists():
        return None
    try:
        from fit_beat_grid import fit_beat_grid  # scripts/ on sys.path
        chords = json.loads(annot.read_text(encoding="utf-8")).get("chords", [])
        return fit_beat_grid(chords, bpm_prior, fit_max_bar=fit_max_bar)
    except Exception as e:
        log.warning("perfect-grid fit failed for %s (%s)", slug, e)
        return None


@app.route("/gt-playalong-corrected")
def gt_playalong_corrected():
    """Perfect constant-tempo GT play-along: waveform + corrected chords snapped
    to a single fitted tempo. Overlays the rigid beat grid on the real audio so
    the user can hear whether the constant-tempo assumption holds, toggle between
    the perfect grid and the original DTW times, and spot where corrections are
    still needed. ?song=<slug>
    """
    from html import escape

    slug = lookup_slug(request.args.get("song") or "autumn_leaves")
    bpm_prior = float(request.args.get("bpm") or 140.0)

    grid = _perfect_grid_for(slug, bpm_prior=bpm_prior)
    if grid is None:
        return (f"<p>No corrected annotations for '{slug}'. Expected "
                f"docs/plots/annotations/irealb_{slug}.html.json (from gt-align).</p>"), 404

    audio_path = AUDIO_DIR / f"{slug}.m4a"
    if not audio_path.exists():
        return f"<p>No audio for '{slug}' at {audio_path}</p>", 404

    wf = _waveform_peaks(slug) or {}
    audio_dur = float(wf.get("duration") or 0.0)
    grid_end = max((c["t1_perfect"] for c in grid["chords"]), default=0.0)
    orig_end = max((c["t1_orig"] for c in grid["chords"]), default=0.0)
    duration = max(audio_dur, grid_end, orig_end)

    v = grid["validation"]
    chart_data = {
        "title": slug.replace("_", " ").title(),
        "slug": slug,
        "chords": grid["chords"],
        "beats": grid["beats"],
        "downbeats": grid["downbeats"],
        "audioUrl": f"/audio/{slug}.m4a",
        "duration": duration,
        "audioDur": audio_dur,
        "gridEnd": grid_end,
        "bpmFit": grid["bpm_fit"],
        "bpmPrior": grid.get("bpm_prior", bpm_prior),
        "bpmErrPct": grid.get("bpm_err_pct"),
        "slope": grid["slope_s_per_bar"],
        "nFit": v["n_fit_points"],
        "fitRms": v["fit_resid_rms_s"],
        "allRms": v["all_resid_rms_s"],
    }

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Perfect-Grid Play-Along: {escape(slug)}</title>
<style>
  * {{ box-sizing:border-box; }}
  html,body {{ margin:0; background:#0e1116; color:#e8edf4; font-family:system-ui,sans-serif; }}
  header {{ padding:14px 18px; background:#171c24; border-bottom:1px solid #2a3340; }}
  h1 {{ margin:0; font-size:17px; }}
  .sub {{ margin:6px 0 0; font-size:12px; color:#8b97a8; }}
  .kpis {{ display:flex; gap:10px; flex-wrap:wrap; margin-top:10px; }}
  .kpi {{ background:#1e2530; border:1px solid #2a3340; border-radius:6px; padding:6px 11px; font-size:12px; }}
  .kpi b {{ color:#00c9a7; font-size:15px; }}
  .kpi.warn b {{ color:#ffb454; }}
  .banner {{ margin:10px 18px 0; padding:9px 13px; background:#2a2015; border-left:3px solid #ffb454;
    border-radius:4px; font-size:12px; color:#e6d4b8; }}
  #controls {{ padding:10px 18px; display:flex; gap:10px; align-items:center; flex-wrap:wrap; }}
  button {{ background:#232c38; color:#e8edf4; border:1px solid #37445a; border-radius:6px;
    padding:7px 13px; font-size:13px; cursor:pointer; }}
  button.on {{ background:#00c9a7; color:#04140f; border-color:#00c9a7; font-weight:600; }}
  #curChord {{ font-size:15px; color:#00c9a7; font-weight:600; }}
  #waveWrap {{ margin:6px 0; overflow-x:auto; overflow-y:hidden; background:#12161d;
    border-top:1px solid #2a3340; border-bottom:1px solid #2a3340; position:relative; }}
  #stage {{ position:relative; height:230px; }}
  canvas {{ display:block; height:230px; }}
  #labels {{ position:absolute; top:4px; left:0; height:26px; pointer-events:none; }}
  .clab {{ position:absolute; transform:translateX(-2px); padding:2px 6px; font-size:11px;
    background:rgba(0,201,167,0.22); border:1px solid rgba(0,201,167,0.5); border-radius:3px;
    white-space:nowrap; color:#cffff2; pointer-events:auto; cursor:pointer; }}
  .clab.orig {{ background:rgba(255,140,66,0.18); border-color:rgba(255,140,66,0.45); color:#ffd9bf; }}
  #playhead {{ position:absolute; top:0; width:2px; height:230px; background:#6ea8ff; z-index:20; }}
  audio {{ width:calc(100% - 36px); margin:8px 18px; }}
  #foot {{ padding:8px 18px 20px; font-size:12px; color:#8b97a8; }}
</style></head><body>

<header>
  <h1>🎯 Perfect-Grid Play-Along — {escape(chart_data['title'])}</h1>
  <p class="sub">Corrected chords snapped to a single fitted tempo. Press play; teal
     markers are the perfect grid, orange (when shown) are the original DTW times.</p>
  <div class="kpis">
    <div class="kpi"><b id="k_bpm"></b> bpm fit</div>
    <div class="kpi">prior <b id="k_prior"></b></div>
    <div class="kpi warn"><b id="k_err"></b> vs prior</div>
    <div class="kpi"><b id="k_fitrms"></b>s fit RMS ({chart_data['nFit']} pts)</div>
    <div class="kpi warn"><b id="k_allrms"></b>s all-chord RMS</div>
  </div>
</header>

<div class="banner" id="banner"></div>

<div id="controls">
  <button id="btnPerfect" class="on">Perfect grid</button>
  <button id="btnOrig">Show original (DTW)</button>
  <button id="btnZoomOut">−</button><button id="btnZoomIn">+</button>
  <span>t: <span id="curTime">0:00</span> / <span id="durTime">0:00</span></span>
  <span>· now: <span id="curChord">—</span></span>
</div>

<audio id="audio" src="{escape(chart_data['audioUrl'])}" type="audio/mp4"
       playsinline controls preload="metadata"></audio>

<div id="waveWrap">
  <div id="stage">
    <canvas id="canvas"></canvas>
    <div id="labels"></div>
    <div id="playhead"></div>
  </div>
</div>

<div id="foot">
  Click any chord marker to seek there. The all-chord RMS residual measures how far
  the original DTW times drift from this constant tempo — large means the head tempo
  doesn't describe the whole song, so hand-correct more bars in gt-align and refit
  (<code>scripts/fit_beat_grid.py</code>).
</div>

<script>
const D = {json.dumps(chart_data)};
const cv = document.getElementById('canvas'), ctx = cv.getContext('2d');
const audio = document.getElementById('audio'), stage = document.getElementById('stage');
const wrap = document.getElementById('waveWrap'), labels = document.getElementById('labels');
const playhead = document.getElementById('playhead');
const H = 230;
let peaks = null, scale = 26, showPerfect = true, showOrig = false;

// KPI fill
document.getElementById('k_bpm').textContent = D.bpmFit;
document.getElementById('k_prior').textContent = D.bpmPrior;
document.getElementById('k_err').textContent = (D.bpmErrPct>=0?'+':'') + D.bpmErrPct + '%';
document.getElementById('k_fitrms').textContent = D.fitRms;
document.getElementById('k_allrms').textContent = D.allRms;
document.getElementById('banner').innerHTML =
  'Grid fit on <b>'+D.nFit+'</b> hand-corrected downbeats → <b>'+D.bpmFit+' bpm</b> '+
  '(fits them to '+D.fitRms+'s RMS). At this tempo the chart ends at <b>'+D.gridEnd.toFixed(1)+
  's</b> but the audio runs <b>'+D.audioDur.toFixed(1)+'s</b> and the original DTW times end near <b>'+
  (D.duration>200?'160s':D.gridEnd.toFixed(1))+'</b> — the constant head-tempo covers only the head. '+
  'Use this page to hear where it diverges.';

function fmt(s){{s=s||0;const m=Math.floor(s/60),ss=Math.floor(s%60);return m+':'+(ss<10?'0':'')+ss;}}

async function loadPeaks(){{
  try {{ const r = await fetch('/api/waveform-peaks/'+encodeURIComponent(D.slug));
    if (r.ok) {{ const d = await r.json(); peaks = d.peaks||[]; }} }} catch(e){{}}
  draw();
}}

function draw(){{
  const w = Math.max(600, D.duration*scale);
  cv.width = w; cv.height = H; stage.style.width = w+'px';
  labels.style.width = w+'px';

  ctx.fillStyle = '#12161d'; ctx.fillRect(0,0,w,H);
  const mid = H*0.55;
  // waveform
  if (peaks && peaks.length) {{
    ctx.fillStyle = '#3d4a5c';
    const n = peaks.length;
    // waveform maps across the AUDIO duration, not the (shorter) grid span
    const wAud = D.audioDur*scale || w;
    for (let x=0; x<wAud; x++){{ const idx=Math.floor(x/wAud*n);
      const h=Math.max(1,(peaks[idx]||0)*150); ctx.fillRect(x, mid-h/2, 1, h); }}
  }}
  // beat grid (perfect)
  if (showPerfect) {{
    ctx.strokeStyle='rgba(110,168,255,0.18)'; ctx.lineWidth=1;
    D.beats.forEach(t=>{{ const x=t*scale; ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke(); }});
    ctx.strokeStyle='rgba(110,168,255,0.55)'; ctx.lineWidth=2;
    D.downbeats.forEach(t=>{{ const x=t*scale; ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x,H); ctx.stroke(); }});
  }}
  // chord onset markers
  labels.innerHTML='';
  D.chords.forEach(c=>{{
    if (showPerfect) addMark(c, c.t0_perfect, false);
    if (showOrig)    addMark(c, c.t0_orig, true);
  }});
  syncPlayhead();
  document.getElementById('durTime').textContent = fmt(D.duration);
}}

function addMark(c, t, isOrig){{
  const x=t*scale;
  ctx.strokeStyle = isOrig ? 'rgba(255,140,66,0.8)' : 'rgba(0,201,167,0.9)';
  ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(x, isOrig?H*0.5:0); ctx.lineTo(x,H); ctx.stroke();
  const el=document.createElement('div');
  el.className='clab'+(isOrig?' orig':'');
  el.style.left=x+'px'; el.style.top=(isOrig?'H':'0');
  el.style.top = isOrig ? '2px' : '2px';
  if (isOrig) el.style.marginTop='0';
  el.textContent=c.label;
  el.title = 'bar '+c.bar+'.'+c.beat+'  '+(isOrig?'orig ':'perfect ')+t.toFixed(2)+'s  resid '+c.residual+'s';
  el.onclick=()=>{{ audio.currentTime=t; }};
  labels.appendChild(el);
}}

function syncPlayhead(){{
  const t=audio.currentTime||0; playhead.style.left=(t*scale)+'px';
  document.getElementById('curTime').textContent=fmt(t);
  let cur='—';
  const arr=D.chords;
  for (let i=arr.length-1;i>=0;i--){{ const on = showPerfect?arr[i].t0_perfect:arr[i].t0_orig;
    if (on<=t){{ cur=arr[i].label+' (bar '+arr[i].bar+')'; break; }} }}
  document.getElementById('curChord').textContent=cur;
  // auto-scroll
  const px=t*scale, vis=wrap.clientWidth;
  if (px < wrap.scrollLeft+40 || px > wrap.scrollLeft+vis-40)
    wrap.scrollLeft = px - vis*0.4;
}}

audio.addEventListener('timeupdate', syncPlayhead);
audio.addEventListener('seeked', syncPlayhead);

document.getElementById('btnPerfect').onclick=e=>{{ showPerfect=!showPerfect; e.target.classList.toggle('on',showPerfect); draw(); }};
document.getElementById('btnOrig').onclick=e=>{{ showOrig=!showOrig; e.target.classList.toggle('on',showOrig);
  e.target.textContent = showOrig?'Hide original (DTW)':'Show original (DTW)'; draw(); }};
document.getElementById('btnZoomIn').onclick=()=>{{ scale=Math.min(200,scale*1.3); draw(); }};
document.getElementById('btnZoomOut').onclick=()=>{{ scale=Math.max(8,scale/1.3); draw(); }};
cv.addEventListener('wheel', e=>{{ if(e.ctrlKey){{ e.preventDefault(); scale=Math.max(8,Math.min(200,scale*(e.deltaY<0?1.2:0.8))); draw(); }} }}, {{passive:false}});

loadPeaks();
</script>
</body></html>"""

    return Response(page, mimetype="text/html")


def _sectionwise_for(slug: str, bpm_prior: float = 181.0):
    """Load (or compute on the fly) the section-wise rigid-tempo alignment for
    <slug>. Prefers the sidecar written by scripts/align_by_sections.py
    (docs/plots/annotations/irealb_<slug>_sectionwise.json); otherwise computes
    it from the gt-align chart + inferred_<slug>.html. Returns the payload dict
    or None."""
    sidecar = ANNOT_DIR / f"irealb_{slug}_sectionwise.json"
    if sidecar.exists():
        try:
            return json.loads(sidecar.read_text(encoding="utf-8"))
        except ValueError:
            pass
    chart = ANNOT_DIR / f"irealb_{slug}.html.json"
    inferred = PLOTS_DIR / f"inferred_{slug}.html"
    if not (chart.exists() and inferred.exists()):
        return None
    try:
        from align_by_sections import align_sections_to_audio  # scripts/ on path
        payload, *_ = align_sections_to_audio(str(chart), str(inferred), bpm_prior)
        return payload
    except Exception as e:
        log.warning("sectionwise fit failed for %s (%s)", slug, e)
        return None


@app.route("/gt-playalong-sectionwise")
def gt_playalong_sectionwise():
    """Section-wise rigid-tempo play-along: each chart section (A/B/C) fit as its
    own constant-tempo block, located in the audio by inferred-chord proxy
    matching (scripts/align_by_sections.py). Chords are coloured by section;
    vamp / low-confidence regions are shaded and NOT treated as clean training
    data. ?song=<slug>&bpm=<prior>
    """
    from html import escape

    slug = lookup_slug(request.args.get("song") or "autumn_leaves")
    bpm_prior = float(request.args.get("bpm") or 181.0)

    pay = _sectionwise_for(slug, bpm_prior=bpm_prior)
    if pay is None:
        return (f"<p>No section-wise alignment for '{slug}'. Expected "
                f"docs/plots/annotations/irealb_{slug}_sectionwise.json, or a "
                f"gt-align chart + inferred_{slug}.html to compute it. Run "
                f"<code>scripts/align_by_sections.py</code>.</p>"), 404

    audio_path = AUDIO_DIR / f"{slug}.m4a"
    if not audio_path.exists():
        return f"<p>No audio for '{slug}' at {audio_path}</p>", 404

    wf = _waveform_peaks(slug) or {}
    audio_dur = float(wf.get("duration") or pay.get("audio_end_s") or 0.0)
    grid_end = max((c["t1_perfect"] for c in pay["chords"]), default=0.0)
    duration = max(audio_dur, grid_end, pay.get("audio_end_s", 0.0))

    n_clean = sum(1 for c in pay["chords"] if not c["is_vamp"])
    chart_data = {
        "title": slug.replace("_", " ").title(),
        "slug": slug,
        "chords": pay["chords"],
        "sections": pay["sections"],
        "vamps": pay["vamps"],
        "offset": pay["global_transpose_offset"],
        "audioUrl": f"/audio/{slug}.m4a",
        "duration": duration,
        "audioDur": audio_dur,
        "gridEnd": grid_end,
        "nClean": n_clean,
        "nTotal": len(pay["chords"]),
    }

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Section-wise Play-Along: {escape(slug)}</title>
<style>
  * {{ box-sizing:border-box; }}
  html,body {{ margin:0; background:#0e1116; color:#e8edf4; font-family:system-ui,sans-serif; }}
  header {{ padding:14px 18px; background:#171c24; border-bottom:1px solid #2a3340; }}
  h1 {{ margin:0; font-size:17px; }}
  .sub {{ margin:6px 0 0; font-size:12px; color:#8b97a8; max-width:900px; line-height:1.5; }}
  .kpis {{ display:flex; gap:10px; flex-wrap:wrap; margin-top:10px; }}
  .kpi {{ background:#1e2530; border:1px solid #2a3340; border-radius:6px; padding:6px 11px; font-size:12px; }}
  .kpi b {{ color:#00c9a7; font-size:15px; }}
  .kpi.warn b {{ color:#ffb454; }}
  #controls {{ padding:10px 18px; display:flex; gap:10px; align-items:center; flex-wrap:wrap; }}
  button {{ background:#232c38; color:#e8edf4; border:1px solid #37445a; border-radius:6px;
    padding:7px 13px; font-size:13px; cursor:pointer; }}
  button.on {{ background:#00c9a7; color:#04140f; border-color:#00c9a7; font-weight:600; }}
  .seclegend {{ display:flex; gap:12px; flex-wrap:wrap; font-size:11px; color:#9aa4b2; padding:0 18px; }}
  .seclegend i {{ display:inline-block; width:11px; height:11px; border-radius:2px; vertical-align:-1px; margin-right:4px; }}
  #waveWrap {{ margin:6px 0; overflow-x:auto; overflow-y:hidden; background:#12161d;
    border-top:1px solid #2a3340; border-bottom:1px solid #2a3340; position:relative; }}
  #stage {{ position:relative; height:250px; }}
  canvas {{ display:block; height:250px; }}
  #labels {{ position:absolute; top:4px; left:0; height:26px; pointer-events:none; }}
  .clab {{ position:absolute; transform:translateX(-2px); padding:2px 6px; font-size:11px;
    border-radius:3px; white-space:nowrap; pointer-events:auto; cursor:pointer; }}
  .clab.vamp {{ opacity:0.45; border-style:dashed !important; }}
  #playhead {{ position:absolute; top:0; width:2px; height:250px; background:#6ea8ff; z-index:20; }}
  #secbar {{ position:absolute; top:0; left:0; height:20px; pointer-events:none; }}
  .secblk {{ position:absolute; height:18px; border-radius:3px; font-size:10px; font-weight:700;
    padding:1px 4px; color:#04140f; pointer-events:auto; cursor:pointer; white-space:nowrap; }}
  audio {{ width:calc(100% - 36px); margin:8px 18px; }}
  #foot {{ padding:8px 18px 20px; font-size:12px; color:#8b97a8; max-width:900px; line-height:1.5; }}
</style></head><body>

<header>
  <h1>🧩 Section-wise Play-Along — {escape(chart_data['title'])}</h1>
  <p class="sub">Each chart section (A/B/C) is fit as its <b>own</b> constant-tempo block and
     located in the recording by matching the model's <b>inferred</b> chord sequence
     (chord-proxy matching), so section repeats and vamps don't have to share one global tempo.
     Global transposition offset detected: <b>+{chart_data['offset']} semitones</b>.
     Coloured blocks = matched sections; hatched red = vamp / low-confidence regions (not clean
     training data). Press play and listen for where the section fits hold.</p>
  <div class="kpis">
    <div class="kpi"><b id="k_clean"></b>/<span id="k_total"></span> clean chords</div>
    <div class="kpi"><b id="k_secs"></b> sections</div>
    <div class="kpi warn"><b id="k_vamps"></b> vamp regions</div>
    <div class="kpi">offset <b>+{chart_data['offset']}</b> st</div>
  </div>
</header>

<div class="seclegend" id="seclegend"></div>

<div id="controls">
  <button id="btnDTW">Show original (DTW)</button>
  <button id="btnZoomOut">−</button><button id="btnZoomIn">+</button>
  <span>t: <span id="curTime">0:00</span> / <span id="durTime">0:00</span></span>
  <span>· now: <span id="curChord">—</span></span>
</div>

<audio id="audio" src="{escape(chart_data['audioUrl'])}" type="audio/mp4"
       playsinline controls preload="metadata"></audio>

<div id="waveWrap">
  <div id="stage">
    <canvas id="canvas"></canvas>
    <div id="secbar"></div>
    <div id="labels"></div>
    <div id="playhead"></div>
  </div>
</div>

<div id="foot">
  Section blocks show fitted BPM and match score. A low score (hatched) means the inferred
  chords in that window don't spell the chart section — usually a solo/vamp where the head
  changes aren't played — so those bars are flagged <code>is_vamp</code> and excluded from
  clean training data. Click a chord or section block to seek. Adjust boundaries in gt-align
  and re-run <code>scripts/align_by_sections.py</code> to refine.
</div>

<script>
const D = {json.dumps(chart_data)};
const cv = document.getElementById('canvas'), ctx = cv.getContext('2d');
const audio = document.getElementById('audio'), stage = document.getElementById('stage');
const wrap = document.getElementById('waveWrap'), labels = document.getElementById('labels');
const secbar = document.getElementById('secbar'), playhead = document.getElementById('playhead');
const H = 250;
let peaks = null, scale = 26, showDTW = false;
const SECCOL = {{A:'#4c8dff', B:'#ff9f43', C:'#2dd4a8', D:'#c678dd'}};

document.getElementById('k_clean').textContent = D.nClean;
document.getElementById('k_total').textContent = D.nTotal;
document.getElementById('k_secs').textContent = D.sections.length;
document.getElementById('k_vamps').textContent = D.vamps.length;
// section legend
const legEl = document.getElementById('seclegend');
D.sections.forEach(s=>{{
  const sp=document.createElement('span');
  const col=SECCOL[s.label]||'#889';
  sp.innerHTML='<i style="background:'+col+'"></i>'+s.label+' '+s.bar_lo+'–'+s.bar_hi+
    ' · '+Math.round(s.bpm_fit)+'bpm · '+s.match_score.toFixed(2)+(s.is_vamp_flagged?' ⚠':'');
  legEl.appendChild(sp);
}});

function fmt(s){{s=s||0;const m=Math.floor(s/60),ss=Math.floor(s%60);return m+':'+(ss<10?'0':'')+ss;}}

async function loadPeaks(){{
  try {{ const r = await fetch('/api/waveform-peaks/'+encodeURIComponent(D.slug));
    if (r.ok) {{ const d = await r.json(); peaks = d.peaks||[]; }} }} catch(e){{}}
  draw();
}}

function draw(){{
  const w = Math.max(600, D.duration*scale);
  cv.width = w; cv.height = H; stage.style.width = w+'px';
  labels.style.width = w+'px'; secbar.style.width = w+'px';
  ctx.fillStyle = '#12161d'; ctx.fillRect(0,0,w,H);
  const mid = H*0.58;
  // vamp bands (behind waveform)
  D.vamps.forEach(v=>{{
    const x0=v.t_start*scale, x1=v.t_end*scale;
    ctx.fillStyle='rgba(255,92,92,0.07)'; ctx.fillRect(x0,0,x1-x0,H);
    // hatch
    ctx.strokeStyle='rgba(255,92,92,0.18)'; ctx.lineWidth=1;
    for(let x=x0;x<x1;x+=7){{ ctx.beginPath(); ctx.moveTo(x,0); ctx.lineTo(x-14,H); ctx.stroke(); }}
  }});
  // waveform
  if (peaks && peaks.length) {{
    ctx.fillStyle = '#3d4a5c';
    const n = peaks.length, wAud = D.audioDur*scale || w;
    for (let x=0; x<wAud; x++){{ const idx=Math.floor(x/wAud*n);
      const h=Math.max(1,(peaks[idx]||0)*150); ctx.fillRect(x, mid-h/2, 1, h); }}
  }}
  // section beat grids (per-section tempo)
  D.sections.forEach(s=>{{
    const col=SECCOL[s.label]||'#889';
    const a=hex2rgb(col);
    ctx.strokeStyle='rgba('+a+',0.4)'; ctx.lineWidth=1;
    for(let t=s.t_start; t<=s.t_end+1e-6; t+=s.slope_s_per_bar){{
      const x=t*scale; ctx.beginPath(); ctx.moveTo(x,20); ctx.lineTo(x,H); ctx.stroke();
    }}
  }});
  // section blocks bar
  secbar.innerHTML='';
  D.sections.forEach((s,i)=>{{
    const x0=s.t_start*scale, x1=s.t_end*scale, col=SECCOL[s.label]||'#889';
    const b=document.createElement('div'); b.className='secblk';
    b.style.left=x0+'px'; b.style.width=Math.max(x1-x0,14)+'px';
    b.style.background=col; b.style.opacity = s.is_vamp_flagged?0.4:0.95;
    b.textContent=s.label+' '+Math.round(s.bpm_fit);
    b.title=s.label+' bars '+s.bar_lo+'–'+s.bar_hi+'  '+s.t_start.toFixed(1)+'–'+s.t_end.toFixed(1)+
      's  '+Math.round(s.bpm_fit)+'bpm  score '+s.match_score.toFixed(2)+' cov '+s.coverage.toFixed(2);
    b.onclick=()=>{{ audio.currentTime=s.t_start; }};
    secbar.appendChild(b);
  }});
  // chord onset markers coloured by section
  labels.innerHTML='';
  D.chords.forEach(c=>{{
    const col=SECCOL[c.section]||'#889';
    const t = showDTW ? (c.t0_orig!=null?c.t0_orig:c.t0_perfect) : c.t0_perfect;
    const x=t*scale;
    ctx.strokeStyle=rgba(col, c.is_vamp?0.4:0.9); ctx.lineWidth=2;
    ctx.beginPath(); ctx.moveTo(x,20); ctx.lineTo(x,H); ctx.stroke();
    const el=document.createElement('div');
    el.className='clab'+(c.is_vamp?' vamp':'');
    el.style.left=x+'px'; el.style.top='2px';
    el.style.background=rgba(col,0.22); el.style.border='1px solid '+rgba(col,0.55);
    el.style.color='#e8f0ff';
    el.textContent=c.label;
    el.title='bar '+c.bar+'.'+c.beat+' ['+c.section+'#'+c.section_id+']  '+t.toFixed(2)+
      's  '+c.tempo_fit+'bpm  match '+c.match_score+(c.is_vamp?'  (vamp/uncertain)':'  (clean)');
    el.onclick=()=>{{ audio.currentTime=t; }};
    labels.appendChild(el);
  }});
  syncPlayhead();
  document.getElementById('durTime').textContent = fmt(D.duration);
}}

function hex2rgb(h){{ const n=parseInt(h.slice(1),16); return ((n>>16)&255)+','+((n>>8)&255)+','+(n&255); }}
function rgba(h,a){{ return 'rgba('+hex2rgb(h)+','+a+')'; }}

function syncPlayhead(){{
  const t=audio.currentTime||0; playhead.style.left=(t*scale)+'px';
  document.getElementById('curTime').textContent=fmt(t);
  let cur='—';
  for (let i=D.chords.length-1;i>=0;i--){{
    const on = showDTW ? (D.chords[i].t0_orig!=null?D.chords[i].t0_orig:D.chords[i].t0_perfect) : D.chords[i].t0_perfect;
    if (on<=t){{ const c=D.chords[i]; cur=c.label+' ('+c.section+' bar '+c.bar+(c.is_vamp?', vamp':'')+')'; break; }} }}
  document.getElementById('curChord').textContent=cur;
  const px=t*scale, vis=wrap.clientWidth;
  if (px < wrap.scrollLeft+40 || px > wrap.scrollLeft+vis-40)
    wrap.scrollLeft = px - vis*0.4;
}}

audio.addEventListener('timeupdate', syncPlayhead);
audio.addEventListener('seeked', syncPlayhead);
document.getElementById('btnDTW').onclick=e=>{{ showDTW=!showDTW; e.target.classList.toggle('on',showDTW);
  e.target.textContent = showDTW?'Hide original (DTW)':'Show original (DTW)'; draw(); }};
document.getElementById('btnZoomIn').onclick=()=>{{ scale=Math.min(200,scale*1.3); draw(); }};
document.getElementById('btnZoomOut').onclick=()=>{{ scale=Math.max(8,scale/1.3); draw(); }};
cv.addEventListener('wheel', e=>{{ if(e.ctrlKey){{ e.preventDefault(); scale=Math.max(8,Math.min(200,scale*(e.deltaY<0?1.2:0.8))); draw(); }} }}, {{passive:false}});

loadPeaks();
</script>
</body></html>"""

    return Response(page, mimetype="text/html")


@app.route("/annotator-v3")
def annotator_v3():
    """Music-aware waveform annotator (v3): server-decoded waveform envelope +
    <audio> playback (iOS-robust), draggable chord boundaries that rewrite the
    adjacent spans, add/remove boundaries, relabel. ?song=<slug>. Same save
    contract as /annotator (POST /api/annotations/<saveFile>)."""
    slug = lookup_slug(request.args.get("song") or "autumn_leaves")
    data, err = _build_annotator_data(slug)
    if err:
        return err
    page = ANNOTATOR_V3_TEMPLATE.replace("__ANNOT_DATA__", json.dumps(data))
    page = page.replace("</head>", _PWA_HEAD + "</head>", 1)

    # Inject beat-correction modal (Opus UX design)
    beat_modal_html = f"""
<div id="beatCorrectorModal" style="display:none;position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.7);z-index:9999;align-items:flex-end;justify-content:center;">
  <div style="background:var(--panel);width:100%;max-height:80vh;border-radius:12px 12px 0 0;padding:20px;padding-bottom:calc(20px+env(safe-area-inset-bottom));overflow-y:auto;">
    <h2 style="margin:0 0 12px;font-size:16px;font-weight:700;">Check Beat Grid</h2>
    <p style="margin:0 0 16px;font-size:12px;color:var(--faint);">Drag beat markers to correct beat phase. Tap "Correct & Infer" when ready.</p>
    <div id="beatCorrectorWave" style="position:relative;height:140px;background:var(--panel2);border:1px solid var(--line);border-radius:6px;margin-bottom:16px;overflow-x:auto;overflow-y:hidden;">
      <canvas id="beatCorrectorCanvas"></canvas>
    </div>
    <div id="beatConfidence" style="padding:8px 12px;background:var(--panel2);border-radius:4px;margin-bottom:12px;font-size:12px;color:var(--faint);">🔓 Detecting beats…</div>
    <div style="display:flex;gap:8px;">
      <button id="beatCorrectorInfer" style="flex:1;padding:12px;background:var(--accent);color:var(--bg);font-weight:600;border:none;border-radius:6px;cursor:pointer;">✓ Correct & Infer</button>
      <button id="beatCorrectorSkip" style="flex:1;padding:12px;background:var(--panel2);color:var(--ink);border:1px solid var(--line);font-weight:600;border-radius:6px;cursor:pointer;">Use As-Is</button>
    </div>
    <div id="beatCorrectorInfo" style="margin-top:12px;padding:8px;background:var(--panel2);border-radius:4px;font-size:11px;color:var(--faint);max-height:60px;overflow:hidden;">Ready</div>
  </div>
</div>
<script>
const beatCorrModal = {{
  modal: document.getElementById('beatCorrectorModal'),
  canvas: document.getElementById('beatCorrectorCanvas'),
  infer: document.getElementById('beatCorrectorInfer'),
  skip: document.getElementById('beatCorrectorSkip'),
  info: document.getElementById('beatCorrectorInfo'),
  beatTimes: [], beatTimesOrig: [], dragBeat: null,
  async init() {{
    const key = 'beatsCorrected:' + (D.slug || 'unknown');
    if (localStorage.getItem(key)) return;
    try {{
      const r = await fetch('/api/beat-grid-audio/' + encodeURIComponent(D.slug || 'autumn_leaves'));
      const grid = await r.json();
      this.beatTimes = grid.beat_times || [];
      this.beatTimesOrig = [...this.beatTimes];
      this.draw();
      this.modal.style.display = 'flex';
      this.wireEvents();
    }} catch(e) {{ this.info.textContent = 'Error: ' + e.message; }}
  }},
  draw() {{
    const ctx = this.canvas.getContext('2d');
    const w = Math.max(300, (D.duration || 0) * 80);
    this.canvas.width = w; this.canvas.height = 140;
    const grad = ctx.createLinearGradient(0, 0, w, 0);
    grad.addColorStop(0, '#2a3340'); grad.addColorStop(0.5, '#4a5a70'); grad.addColorStop(1, '#2a3340');
    ctx.fillStyle = grad; ctx.fillRect(0, 0, w, 140);
    this.beatTimes.forEach((t, i) => {{
      const x = t * 80; ctx.fillStyle = (i % 4) === 0 ? '#ffb454' : 'rgba(255,180,84,0.3)';
      ctx.fillRect(x - 1, 0, 3, 140);
    }});
    ctx.strokeStyle = '#8b97a8'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(0, 70); ctx.lineTo(w, 70); ctx.stroke();
  }},
  wireEvents() {{
    const self = this;
    let dragBeat = null;
    this.canvas.addEventListener('pointerdown', e => {{
      const rect = this.canvas.getBoundingClientRect();
      const x = e.clientX - rect.left, t = x / 80;
      let best = -1, bestDist = Infinity;
      self.beatTimes.forEach((bt, i) => {{
        const dx = Math.abs(bt * 80 - x);
        if (dx < 15 && dx < bestDist) {{ best = i; bestDist = dx; }}
      }});
      dragBeat = best >= 0 ? best : null;
    }});
    this.canvas.addEventListener('pointermove', e => {{
      if (dragBeat == null) return;
      const rect = this.canvas.getBoundingClientRect();
      const x = e.clientX - rect.left, t = x / 80;
      const orig = self.beatTimesOrig[dragBeat];
      self.beatTimes[dragBeat] = Math.max(orig - 0.2, Math.min(orig + 0.2, t));
      self.draw();
    }});
    this.canvas.addEventListener('pointerup', () => {{ dragBeat = null; }});
    this.infer.addEventListener('click', async () => {{
      this.infer.disabled = true;
      const delta_ms = (this.beatTimes[0] - this.beatTimesOrig[0]) * 1000;
      try {{
        const r = await fetch('/api/beat-0-shift/' + encodeURIComponent(D.slug || 'autumn_leaves'), {{
          method: 'POST', headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{ delta_ms }})
        }});
        localStorage.setItem(key, 'true');
        self.modal.style.display = 'none';
        setTimeout(() => {{ location.reload(); }}, 500);
      }} catch(e) {{ self.info.textContent = 'Error: ' + e.message; this.infer.disabled = false; }}
    }});
    this.skip.addEventListener('click', () => {{
      localStorage.setItem(key, 'true');
      this.modal.style.display = 'none';
    }});
  }}
}};
document.addEventListener('DOMContentLoaded', () => beatCorrModal.init());
</script>
"""
    page = page.replace("</body>", beat_modal_html + "</body>", 1)
    return Response(page, mimetype="text/html")


@app.route("/annotator-v4")
def annotator_v4():
    """Music-aware waveform annotator v4: beat-grid editor + chord events.

    Two-stage UI:
    1. Beat-grid editor: correct beat phase for first N bars, then lock & infer
    2. Chord event editor: add/delete chord boundaries, relabel

    Chord model: events (point markers) instead of intervals.
    ?song=<slug>
    """
    slug = lookup_slug(request.args.get("song") or "autumn_leaves")
    data, err = _build_annotator_data(slug)
    if err:
        return err
    page = ANNOTATOR_V4_TEMPLATE.replace("__ANNOT_DATA__", json.dumps(data))
    page = page.replace("</head>", _PWA_HEAD + "</head>", 1)
    return Response(page, mimetype="text/html")


def _beat_grid_for(slug: str, audio_path, tempo: float, duration: float) -> dict:
    """Beat/downbeat grid for the snap ruler. Prefers Mission-1's
    extract_beat_grid() on the real audio (cached to disk — librosa beat
    tracking is a few seconds); falls back to a uniform grid from the chart
    tempo if audio/librosa is unavailable."""
    BEATGRID_CACHE.mkdir(parents=True, exist_ok=True)
    cache = BEATGRID_CACHE / f"{slug}.json"
    if cache.exists():
        try:
            d = json.loads(cache.read_text(encoding="utf-8"))
            if d.get("beats"):
                return d
        except ValueError:
            pass
    result = None
    if audio_path is not None and audio_path.exists():
        try:
            from mission_1_build_benchmark import extract_beat_grid  # scripts/ on sys.path
            bg = extract_beat_grid(audio_path, bpm_hint=tempo)
            result = {
                "beats": [round(float(x), 4) for x in list(bg.beat_times)],
                "downbeats": [round(float(x), 4) for x in list(bg.downbeat_times)],
                "bpm": float(bg.bpm), "source": "extract_beat_grid",
            }
        except Exception as e:  # librosa/soundfile/pyRealParser missing, decode error…
            log.warning("extract_beat_grid failed for %s (%s) — uniform fallback", slug, e)
    if result is None:
        step = 60.0 / (tempo or 120.0)
        n = int((duration or 0.0) / step) + 4
        beats = [round(i * step, 4) for i in range(n)]
        result = {"beats": beats, "downbeats": beats[::4],
                  "bpm": float(tempo or 120.0), "source": "uniform"}
    try:
        cache.write_text(json.dumps(result), encoding="utf-8")
    except OSError:
        pass
    return result


@app.route("/api/beat-grid/<song>")
def api_beat_grid(song):
    """Beat/downbeat grid for <song> as JSON — the waveform annotator's beat
    layer can fetch this directly instead of relying on the embedded payload.
    Same cached extract_beat_grid() result the /annotator page ships inline."""
    slug = lookup_slug(song or "")
    chords, tempo = _load_ireal_alignment(slug)
    if not chords:
        return jsonify(error=f"no iReal chart for '{slug}'"), 404
    audio_path = AUDIO_DIR / f"{slug}.m4a"
    duration = max((c["t1"] for c in chords), default=0.0)
    grid = _beat_grid_for(slug, audio_path if audio_path.exists() else None,
                          float(tempo or 120), duration)
    return jsonify(grid)


def _build_annotator_data(slug: str):
    """Shared payload builder for the annotator routes. Returns (data, None) on
    success or (None, (msg, status)) if no chart could be loaded/generated."""
    chords, tempo = _load_ireal_alignment(slug)

    # If iReal chart is missing, try to generate it from the inferred chart's title
    if not chords:
        inferred_file = PLOTS_DIR / f"inferred_{slug}.html"
        if inferred_file.exists():
            try:
                html_text = inferred_file.read_text(encoding="utf-8")
                # Extract title from the inferred chart (look for <title> tag)
                title_match = re.search(r'<title>([^<]+)</title>', html_text)
                if title_match:
                    title = title_match.group(1).replace(" — ", " ").split(" • ")[0]
                    from harmonia.irealb_fetcher import search_community, render_irealb_chart
                    try:
                        results = search_community(title, max_results=1)
                        if results:
                            irealb_url = results[0]["irealb_url"]
                            ir_html = render_irealb_chart(irealb_url, chart_offset_s=0.0)
                            ir_path = PLOTS_DIR / f"irealb_{slug}.html"
                            ir_path.write_text(ir_html, encoding="utf-8")
                            _remember_ireal_url(f"inferred_{slug}.html", irealb_url)
                            chords, tempo = _load_ireal_alignment(slug)
                            log.info("Auto-generated iReal chart for %s", slug)
                    except Exception as e:
                        log.warning("Could not auto-generate iReal chart for %s: %s", slug, e)

                # Fallback: if iReal chart still missing, create a minimal placeholder
                # so /annotator doesn't fail. User can still edit the beat grid.
                if not chords:
                    ir_path = PLOTS_DIR / f"irealb_{slug}.html"
                    if not ir_path.exists():
                        # Create minimal iReal chart with placeholder chords (4 bars, 10s total)
                        placeholder_chords = [
                            {"label": "?", "t0": 0.0, "t1": 2.5, "bar": 0, "section": "A"},
                            {"label": "?", "t0": 2.5, "t1": 5.0, "bar": 1, "section": "A"},
                            {"label": "?", "t0": 5.0, "t1": 7.5, "bar": 2, "section": "A"},
                            {"label": "?", "t0": 7.5, "t1": 10.0, "bar": 3, "section": "A"},
                        ]
                        p_json = json.dumps({"chords": placeholder_chords, "tempo": int(tempo or 120)})
                        placeholder_html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Placeholder Chart</title></head>
<body><p>No iReal chart found. Annotation available via beat-grid editing.</p>
<script>window.P = {p_json};</script></body></html>"""
                        ir_path.write_text(placeholder_html, encoding="utf-8")
                        chords, tempo = _load_ireal_alignment(slug)
                        log.info("Created placeholder iReal chart for %s", slug)
            except Exception as e:
                log.warning("Could not attempt to auto-generate iReal chart: %s", e)

    if not chords:
        return (None, (f"No iReal chart for '{slug}'. Expected docs/plots/irealb_{slug}.html "
                       f"with a window.P payload.", 404))
    audio_path = AUDIO_DIR / f"{slug}.m4a"
    have_audio = audio_path.exists()
    duration = max((c["t1"] for c in chords), default=0.0)
    grid = _beat_grid_for(slug, audio_path if have_audio else None, tempo, duration)
    data = {
        "slug": slug,
        "title": slug.replace("_", " ").title(),
        "chords": chords,
        "audioUrl": f"/audio/{slug}.m4a" if have_audio else "",
        "beats": grid["beats"],
        "downbeats": grid["downbeats"],
        "gridSource": grid["source"],
        "bpm": grid["bpm"],
        "duration": duration,
        "tempo": tempo,
        "saveFile": f"inferred_{slug}.html",
        "snapTolMs": 250,
    }
    return (data, None)


@app.route("/annotator")
def annotator():
    """Manual chord-alignment tool. ?song=<slug> (default autumn_leaves)."""
    slug = lookup_slug(request.args.get("song") or "autumn_leaves")
    data, err = _build_annotator_data(slug)
    if err:
        return err
    page = ANNOTATOR_TEMPLATE.replace("__ANNOT_DATA__", json.dumps(data))
    page = page.replace("</head>", _PWA_HEAD + "</head>", 1)
    return Response(page, mimetype="text/html")


@app.route("/annotator-v2")
def annotator_v2():
    """Rebuilt, mobile-first waveform annotator (simple linear flow).
    ?song=<slug> (default autumn_leaves). Same save contract as /annotator
    (POST /api/annotations/<saveFile>)."""
    slug = lookup_slug(request.args.get("song") or "autumn_leaves")
    data, err = _build_annotator_data(slug)
    if err:
        return err
    page = ANNOTATOR_SIMPLE_TEMPLATE.replace("__ANNOT_DATA__", json.dumps(data))
    page = page.replace("</head>", _PWA_HEAD + "</head>", 1)
    return Response(page, mimetype="text/html")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    # Sets the CLI args on the runtime module (NOT a local/global name here):
    # request-time code reads them live as ``runtime.ARGS.<attr>``. Assigning the
    # attribute is what makes the reassignment visible to every reader — a
    # global-rebind of a plain module name here would only update THIS module and
    # leave ``runtime.ARGS`` at None forever (see runtime.py's module docstring #1).
    ap = argparse.ArgumentParser(description="Harmonia local server")
    ap.add_argument("--port", type=int, default=7771)
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--phase", type=int, default=1, choices=[1, 2, 3, 4])
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--no-madmom", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    runtime.ARGS = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if runtime.ARGS.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    url = f"http://localhost:{runtime.ARGS.port}"
    lan_ip = _lan_ip()
    print(f"Harmonia server →  {url}")
    if lan_ip:
        print(f"  on your iPhone (same Wi-Fi) →  http://{lan_ip}:{runtime.ARGS.port}")

    if not runtime.ARGS.no_open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    app.run(host="0.0.0.0", port=runtime.ARGS.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
