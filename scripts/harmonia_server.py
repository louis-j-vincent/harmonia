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
    # _bar1_offset_bounds moved to state.py (Phase 6c batch-2) with the
    # /api/bar1-offset POST route; re-imported so the /bar1-offset-fix page
    # route below still resolves it.  noqa: F401 re-export.
    _bar1_offset_bounds,
    # Writer helpers MOVED to state.py (mutating-routes serving refactor);
    # re-imported so server.X is state.X.  _section_labels_path/_save_section_labels
    # are now pure re-exports: both /api/section-labels routes (GET+POST) and their
    # _load_section_labels helper live in harmonia.serving.api as of Phase 6c, so
    # nothing in THIS module resolves them anymore.  noqa: F401 re-export.
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
# Audio-envelope / beat-grid compute-cache leaves now live in
# harmonia.serving.audio (Phase 6c architectural pass; _waveform_peaks/
# _beat_grid_for relocated there from state, _raw_beat_times_cached from this
# module). Re-imported so the staying page routes that call them (gt-align /
# gt-offset-fix / bar1-offset-fix / section-align / gt-playalong) resolve
# unchanged; _raw_beat_times_cached is NOT re-imported (only render used it, and
# render now imports it from audio directly).  noqa: F401 re-export.
from harmonia.serving.audio import _beat_grid_for, _waveform_peaks
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

# ── App factory (Phase 6c architectural pass) ─────────────────────────────────
# Construction is a FUNCTION (create_app), not an import-time side effect. The
# serving routes moved to harmonia.serving.api live on the ``api`` blueprint;
# this module's own ~17 page/debug routes are declared with the lightweight
# @route collector below — it just records ``(rule, view_func, options)`` at
# import time and create_app() REPLAYS them onto the app via ``add_url_rule``.
# Because add_url_rule defaults each endpoint to the view's ``__name__`` (bare),
# and the ``api`` blueprint is registered with name="" (Flask computes each
# endpoint as ``f"{name_prefix}.{name}.{endpoint}".lstrip(".")`` → an empty name
# yields the ORIGINAL bare endpoint), the resulting ``app.url_map`` is
# byte-identical to the old module-level ``@app.route`` + single blueprint
# wiring — same rules, same endpoint names, same methods; url_for() unchanged.
# (Two ``name=""`` blueprints would collide, which is why the page routes use
# the collector rather than a second blueprint.)  HARMONIA_FUSION_ALIGN and every
# other runtime flag are read live inside handlers, unaffected.
from harmonia.serving.api import api as api_bp

_PAGE_ROUTES: list = []


def route(rule, **options):
    """Deferred ``@app.route``: record the route now, register it in
    create_app(). Endpoint stays bare (add_url_rule defaults it to the view
    function's __name__), preserving the exact url_map."""
    def _decorator(fn):
        _PAGE_ROUTES.append((rule, fn, options))
        return fn
    return _decorator


def create_app() -> Flask:
    """Build the Flask app, register the serving blueprint + the collected page
    routes, and return it. No import-time side effects; tests/tooling can build a
    fresh, isolated app. The module-level ``app = create_app()`` at the bottom is
    the default instance ``main()`` runs and existing tooling (url_map dumps,
    test clients) reads as ``harmonia_server.app``."""
    app = Flask(__name__, static_folder=None)
    app.register_blueprint(api_bp, name="")
    for _rule, _view, _options in _PAGE_ROUTES:
        app.add_url_rule(_rule, view_func=_view, **_options)
    return app

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


# _extract_video_id MOVED to harmonia.serving.analysis (Phase 6c final cluster)
# alongside _run_analysis, its only caller. Pure regex leaf; not re-imported
# because nothing else in this module used it.

# ── Inject snippet ────────────────────────────────────────────────────────────
# _INJECT_MARKER / _BACK_BUTTON_HTML + _inject_overlay / _inject_back_button
# moved to harmonia.serving.render (serving refactor, render round) and imported
# back at module top. Pure string injectors (html in → html out), no beat/grid/
# model deps; call sites unchanged.


# ── Routes ────────────────────────────────────────────────────────────────────

@route("/sw.js")
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


@route("/debug/structure")
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


@route("/")
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


# /classic (classic_index) MOVED to the harmonia.serving.api blueprint (Phase 6c,
# read-only GET batch) — pure render_template_string(HOME_TEMPLATE) page, all deps
# extracted (config PLOTS_DIR, templates HOME_TEMPLATE, render _PWA_HEAD). Bare
# endpoint kept via name="".


# _raw_beat_times_cached MOVED to harmonia.serving.audio (Phase 6c architectural
# pass). It was consumed ONLY by harmonia.serving.render._chart_model_for (via the
# render↔server lazy back-import); render now imports it straight from audio, so
# the back-import is gone and nothing in this module referenced it.


# /api/billboard-corpus (api_billboard_corpus) + its sole helper
# _load_billboard_corpus MOVED to the harmonia.serving.api blueprint (Phase 6c,
# read-only GET batch). Both were pure reads over extracted leaves
# (_BILLBOARD_CORPUS_FILES in config, _yt_video_ids in state, _load_annotation in
# loaders); _billboard_video_to_track_id (in harmonia.serving.billboard_gt) is the
# other reader of _BILLBOARD_CORPUS_FILES. Bare endpoint kept via name="".


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
# harmonia.serving.state (Phase 6d), imported at module top. _bar1_offset_bounds
# (pure arithmetic leaf) ALSO moved to state.py (Phase 6c batch-2) and is
# re-imported at module top; it is still used by the /bar1-offset-fix page route
# below, while the extracted POST /api/bar1-offset route imports it from state
# directly.
# _apply_bar1_offset_to_payload MOVED to harmonia.serving.render (Phase 6c
# architectural pass) — it lived next to its only caller there anyway
# (_chart_model_for), and relocating it let the render↔server lazy back-import
# (``import scripts.harmonia_server as _srv``) be deleted. No server route called
# it, so nothing here needs it re-imported.


# GET+POST /api/bar1-offset/<slug> (api_bar1_offset_get / api_bar1_offset_save)
# MOVED to the harmonia.serving.api blueprint (Phase 6c batch-2). Deps are all
# extracted leaves: _load_bar1_offsets / _save_bar1_offset / _bar1_offset_bounds
# (all in state; _bar1_offset_bounds moved there this round), PLOTS_DIR (config),
# and a lazy in-body harmonia.output.chart_model import. Bare endpoints kept via
# name="".


# ── User-drawn song-structure section labels (2026-07-17) ────────────────────
# The auto SSM sections (P.sections / P.sectionChips) are the MODEL's guess; this
# is an independent, hand-drawn layer where the user marks "this is A, this is B"
# on the chart. Persisted as its own sidecar so it never collides with the
# annotation doc's last-write-wins /api/annotations POST (which posts the whole
# {annotator,chords,merges} on every chord edit and would otherwise clobber it).
# Same small-file GET/POST shape as /api/bar1-offset. Doc: {"labels": {"<bar>":
# "<label>", ...}, "updated": iso}. A label at bar b starts a named section that
# runs until the next labeled bar. Purely additive; render-only on the client.
#
# GET /api/section-labels/<filename> (api_section_labels_get) + its sole helper
# _load_section_labels MOVED to the harmonia.serving.api blueprint (Phase 6c,
# read-only GET batch) — pairing the GET with its already-moved POST sibling
# (api_section_labels_save). _section_labels_path (state) is the only dep. Bare
# endpoint kept via name="".


# ── Chord-audio snippet + /library MOVED to the harmonia.serving.api blueprint
# (Phase 6c, read-only GET batch):
#   * GET /api/chord-snippet/<filename> (api_chord_snippet) + its sole helper
#     _audio_path_for_chart — reads the retained docs/audio/<slug>.m4a (state
#     _yt_audio_meta + config AUDIO_DIR) and streams a sample-accurate WAV clip
#     via harmonia.models.audio_snippet (lazy in-body import); nothing cached to
#     disk, GET-only.
#   * /library (library) — pure render_template_string(LIBRARY_TEMPLATE) page
#     (config PLOTS_DIR, templates LIBRARY_TEMPLATE, render _PWA_HEAD).
# Bare endpoints kept via name="".


# 2026-07-18: no longer referenced — its only call site (serve_chart's
# baked-HTML swipe-nav injection) was removed when /chart/<file> became an
# unconditional redirect to the SPA (see serve_chart's docstring). Left in
# place rather than deleted since it's inert and reviving the baked-HTML
# path later (if ever) would want it back verbatim.


@route("/gt-align")
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


# POST /api/yt-search (api_yt_search) MOVED to the harmonia.serving.api blueprint
# (Phase 6c batch-2) — pure-network POST, deps are request/jsonify/log + a lazy
# in-body ``import yt_dlp``; no server-owned helper. Bare endpoint kept via name="".


# GET /api/annotations/<filename> (get_annotations) MOVED to the
# harmonia.serving.api blueprint (Phase 6c, read-only GET batch) — pairing the
# GET with its already-moved POST sibling (post_annotations). _load_annotation
# (loaders) is the only dep. Bare endpoint kept via name="".


# _chart_audio_path and _chord_at MOVED to harmonia.serving.analysis (Phase 6c
# final cluster) alongside their only caller, POST /api/reinfer. Pure leaves.


# /api/bar-merge-candidates/<filename> (api_bar_merge_candidates) and
# /api/section-merge-candidates/<filename> (api_section_merge_candidates) MOVED to
# the harmonia.serving.api blueprint (Phase 6c, read-only GET batch). Both are
# THIN passthroughs — read whichever precomputed scratchpad/*_candidates_*.json
# already exists (config REPO) and jsonify it, 200+empty on a missing file, no
# live recomputation, no state mutation. Bare endpoints kept via name="".


# _ireal_q_to_q5 MOVED to harmonia.serving.analysis (Phase 6c final cluster)
# alongside its only caller, POST /api/reinfer. Pure leaf.


# POST /api/reinfer/<filename> (api_reinfer) MOVED to the harmonia.serving.api
# blueprint (Phase 6c final cluster) — uses the analysis-module leaves
# _chart_audio_path / _chord_at / _ireal_q_to_q5 + lazy chord_pipeline_v1;
# transcodes via ffmpeg to a temp dir. Bare endpoint kept via name="".

# POST /api/analyze (api_analyze) MOVED to the harmonia.serving.api blueprint
# (Phase 6c final cluster) — spawns a background harmonia.serving.analysis.
# _run_analysis thread against the runtime.jobs registry (under _jobs_lock) and
# returns a job_id the client polls via GET /api/job/<job_id>. Bare endpoint kept
# via name="".


# /api/job/<job_id> (api_job) MOVED to the harmonia.serving.api blueprint
# (Phase 6c, read-only GET batch) — a pure read that copies the job dict under
# _jobs_lock (both from harmonia.serving.runtime, the SAME live objects the
# analyze/job writers mutate here). Bare endpoint kept via name="".


# POST /api/record-analyze (api_record_analyze) MOVED to the harmonia.serving.api
# blueprint (Phase 6c final cluster) — mic-recording analog of /api/analyze:
# transcodes the multipart upload, then spawns the SAME harmonia.serving.analysis.
# _run_analysis background thread (local_audio_path branch). Bare endpoint kept
# via name="".


# Jam Mode trio POST /api/jam/{start,chunk,stop} (api_jam_start / api_jam_chunk /
# api_jam_stop) MOVED to the harmonia.serving.api blueprint (Phase 6c batch-3) —
# they mutate the runtime.jam_sessions registry under its lock (SAME live objects,
# imported from runtime) + lazy harmonia.models.jam_mode / soundfile; no
# server-owned helper. Bare endpoints kept via name="".


# POST /api/tab-search (api_tab_search) and POST /api/tab-fetch (api_tab_fetch)
# MOVED to the harmonia.serving.api blueprint (Phase 6c batch-2) — pure-network
# POSTs over lazy harmonia.tab_fetcher imports; tab-fetch's sole helper
# _render_tab_page (a pure HTML-string leaf) moved with it. Deps otherwise are
# request/jsonify/log/re + PLOTS_DIR (config). /api/render-tab (below) STAYS —
# it renders via harmonia.tab_renderer.render_tab_chart, a different path.
# Bare endpoints kept via name="".


# POST /api/render-tab (api_render_tab) MOVED to the harmonia.serving.api
# blueprint (Phase 6c batch-3) — lazy harmonia.tab_fetcher / tab_renderer;
# PLOTS_DIR (config) + _remember_video_id (state), no server-owned helper. Bare
# endpoint kept via name="".


# _render_tab_page MOVED to the harmonia.serving.api blueprint (Phase 6c batch-2)
# alongside its sole caller, POST /api/tab-fetch (api_tab_fetch). Pure HTML-string
# leaf: in-body html/re only, reads the passed tab object, returns a string; no
# server module dep.


# POST /api/irealb-{align,render,import} (api_irealb_align / api_irealb_render /
# api_irealb_import) MOVED to the harmonia.serving.api blueprint (Phase 6c
# batch-3) — lazy pyRealParser + harmonia.irealb_{aligner,fetcher} / ireal_corpus
# / alignment_validator + (import) chart_slug (cache) / scripts.render_youtube_
# chart / chart_interactive; other-lane modules touched only via lazy in-body
# imports. PLOTS_DIR (config) + _remember_video_id (state). Bare endpoints kept
# via name="".


# (see the batch-3 breadcrumb above /api/irealb-render's former location —
# api_irealb_render and api_irealb_import moved with api_irealb_align.)


# _run_analysis MOVED to harmonia.serving.analysis (Phase 6c final cluster) —
# the background analyze/record-analyze worker. All its stateful deps go
# through extracted leaves (runtime jobs/ARGS/_ANALYZE_*, state registries +
# offset stores, config paths) and every heavy pipeline import stays lazy
# in-body, so the move dragged no server-owned mutable state. Its former
# callers (/api/analyze, /api/record-analyze) moved to serving/api.py with it.


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


# _waveform_peaks (pure audio-envelope compute-cache leaf) MOVED to
# harmonia.serving.state (Phase 6c batch-3) and re-imported at module top (still
# called by the staying gt-align / gt-offset-fix / bar1-offset-fix /
# section-align page routes). GET /api/waveform-peaks/<song> (api_waveform_peaks)
# MOVED to the harmonia.serving.api blueprint alongside it. Bare endpoint kept
# via name="".


# GET /api/grid-align-data/<song> (api_grid_align_data) MOVED to the
# harmonia.serving.api blueprint (Phase 6c batch-3) — self-contained diagnostic
# (lookup_slug + AUDIO_DIR/PLOTS_DIR + lazy librosa/numpy + chord_pipeline_v1/
# chart_model in-body), no server-owned helper. Bare endpoint kept via name="".


def _fusion_section_align_results(song_id: str, corpus: str, title: str, wav) -> "list[dict]":
    """ON-path adapter for the fusion kill-switch (``HARMONIA_FUSION_ALIGN``).

    Routes the section-align page through the productionised fusion DBN
    (``harmonia.align.chart_aligner.FusionChartAligner``) instead of the legacy
    ``align_tune_sections_to_audio``, then maps the returned ``ChartAlignment``
    onto the EXACT list-of-dicts ``results`` shape the route's downstream loop
    consumes — a drop-in, so the marker/section payload builder stays unchanged.

    Insertion-contract mapping (``ChartAlignment`` -> legacy ``results``, per
    section):
      * ``warp``   -> ``float`` (identity): fusion ``SectionMarker`` times are
        already absolute audio seconds, so ``warp(start_s) == start_s == m.t0``.
      * ``chords`` -> ``[{start_s: m.t0, end_s: m.t1, mma: m.mma}]`` per marker.
      * ``label`` / ``bar0`` / ``bar1`` -> section label + first/last marker
        global lattice bar index.
      * ``accepted`` -> ``True``: the fusion DBN has no accept/reject gate; every
        placement is part of its alignment (it flags low confidence via
        ``low_confidence_regions``, not per-section rejection).
      * ``shape_agreement`` / ``error`` / ``reason`` are legacy-only fields with
        no fusion analogue -> ``None`` (median gate n/a; the fusion path never
        runs the model shape-safeguard).
    """
    from harmonia.align.chart_aligner import FusionChartAligner

    chart_cfg = {"song_id": song_id, "ireal_file": corpus, "tune_title": title}
    alignment = FusionChartAligner().align(str(wav), chart_cfg)
    results: "list[dict]" = []
    for sec in alignment.sections:
        markers = sec.get("markers") or []
        results.append({
            "label": sec.get("label"),
            "bar0": markers[0].bar if markers else 0,
            "bar1": markers[-1].bar if markers else 0,
            "accepted": True,
            "warp": float,  # identity — fusion marker times are already absolute
            "chords": [{"start_s": m.t0, "end_s": m.t1, "mma": m.mma} for m in markers],
            "error": {"median_ms": None},
            "shape_agreement": None,
            "reason": None,
        })
    return results


@route("/debug/section-align")
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
            # KILL-SWITCH: HARMONIA_FUSION_ALIGN (default OFF). OFF keeps the exact
            # legacy call below, byte-identically (lossless no-op). ON routes through
            # the productionised fusion DBN via _fusion_section_align_results, which
            # maps ChartAlignment -> the same `results` shape this route consumes.
            if os.environ.get("HARMONIA_FUSION_ALIGN", "0").strip().lower() in ("1", "true", "yes", "on"):
                results = _fusion_section_align_results(slug, corpus, title, wav)
            else:
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


# GET /api/beat-grid-audio/<song> (api_beat_grid_audio) MOVED to the
# harmonia.serving.api blueprint (Phase 6c batch-3) — self-contained (lookup_slug
# + AUDIO_DIR + lazy in-body librosa/numpy, no server-owned helper). Bare
# endpoint kept via name="".


# POST /api/reinfer-from-beats/<song> (api_reinfer_from_beats) MOVED to the
# harmonia.serving.api blueprint (Phase 6c batch-2) — self-contained
# (lookup_slug + AUDIO_DIR + lazy infer_chords_v1/librosa/tempfile/pickle, no
# server-owned helper). NOTE (pre-existing, carried verbatim): its body uses a
# bare ``np`` with no numpy import and no module-level np in this module either,
# so its happy path already raised NameError -> caught -> 500 here too; the move
# preserved that exactly. Bare endpoint kept via name="".


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



# POST /api/beat-0-shift/<song> (api_beat_0_shift) MOVED to the
# harmonia.serving.api blueprint (Phase 6c batch-2) — self-contained
# (lookup_slug + AUDIO_DIR + lazy in-body librosa/numpy, no server-owned helper).
# Bare endpoint kept via name="".


@route("/gt-playalong")
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


@route("/gt-playalong-training")
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


@route("/rwc-playalong")
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


@route("/billboard-gt-triage")
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


@route("/gt-offset-fix")
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


@route("/bar1-offset-fix")
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


@route("/gt-playalong-corrected")
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


@route("/gt-playalong-sectionwise")
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


@route("/annotator-v3")
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


@route("/annotator-v4")
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


# _beat_grid_for (pure beat-grid compute-cache leaf) MOVED to
# harmonia.serving.state (Phase 6c batch-3) and re-imported at module top (still
# called by the staying gt-playalong page routes). GET /api/beat-grid/<song>
# (api_beat_grid) MOVED to the harmonia.serving.api blueprint alongside it —
# lookup_slug + _load_ireal_alignment (loaders) + AUDIO_DIR + _beat_grid_for.
# Bare endpoint kept via name="".


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


@route("/annotator")
def annotator():
    """Manual chord-alignment tool. ?song=<slug> (default autumn_leaves)."""
    slug = lookup_slug(request.args.get("song") or "autumn_leaves")
    data, err = _build_annotator_data(slug)
    if err:
        return err
    page = ANNOTATOR_TEMPLATE.replace("__ANNOT_DATA__", json.dumps(data))
    page = page.replace("</head>", _PWA_HEAD + "</head>", 1)
    return Response(page, mimetype="text/html")


@route("/annotator-v2")
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

# Default app instance, built now that every @route decorator above has been
# collected into _PAGE_ROUTES. This is the app main() runs and that tooling /
# tests read as ``harmonia_server.app``; build a fresh isolated one with
# create_app() when needed.
app = create_app()


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
