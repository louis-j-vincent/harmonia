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

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from flask import Flask, Response, jsonify, redirect, render_template_string, request, send_from_directory

log = logging.getLogger(__name__)

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


def _training_log_dir(song: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9_]", "", song or "") or "unknown"
    return TRAINING_LOGS_DIR / safe


def _annot_path(filename: str) -> Path:
    return ANNOT_DIR / f"{filename}.json"


def _load_annotation(filename: str) -> dict:
    try:
        return json.loads(_annot_path(filename).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"schema": 1, "chart": filename, "annotator": "", "modified": None,
                "chords": [], "merges": []}


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

_PWA_HEAD = """<link rel="manifest" href="/pwa/manifest.json">
<link rel="apple-touch-icon" href="/pwa/apple-touch-icon.png">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Harmonia">
<meta name="theme-color" content="#8a2b2b">
<!-- overrides the page's own viewport tag (last one wins) — locks pinch/
     double-tap zoom so it can't hijack the rotor-drag or swipe-nav gestures -->
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no">
<script>if("serviceWorker" in navigator){navigator.serviceWorker.register("/sw.js").then(function(reg){
  // This is an SPA: internal navigation (go()) never re-fetches app_shell.html,
  // so a tab/installed-PWA left open across a server-side UI change keeps
  // running the JS it loaded at open time indefinitely — no error, just
  // silently stale (root-caused 2026-07-15: "can't see the GT pill" after a
  // same-day feature landed with a verified-working server, because the
  // reporter's own browser tab predated the change). Surface a tap-to-reload
  // banner instead of relying on the user to know to hard-refresh.
  reg.addEventListener("updatefound", function(){
    var nw=reg.installing; if(!nw) return;
    nw.addEventListener("statechange", function(){
      if(nw.state==="installed" && navigator.serviceWorker.controller){
        var b=document.createElement("div");
        b.textContent="Update available — tap to refresh";
        b.style.cssText="position:fixed;left:50%;bottom:24px;transform:translateX(-50%);z-index:99999;background:#1a1a1a;color:#fff;font:600 13px system-ui,sans-serif;padding:10px 18px;border-radius:22px;box-shadow:0 10px 26px -10px rgba(0,0,0,.5);cursor:pointer;";
        b.onclick=function(){ location.reload(); };
        document.body.appendChild(b);
      }
    });
  });
});}</script>
<style>
/* which song, out of how many — quiet enough not to compete with the chart,
   but enough context to know where you are after a swipe */
.harm-pos { text-align:center; font:600 11px system-ui,sans-serif;
            color:#8a8371; letter-spacing:.04em; margin:-2px 0 12px; }
/* the docked YouTube player (z-index 9990) predates the rotor/options
   sheets — bump the sheets above it so opening one while the video is
   docked doesn't get its bottom edge hidden behind the video */
.modal { z-index:9995 !important; }
/* injected by harmonia_server.py so already-rendered charts get phone-width
   layout too, even if chart_interactive.py's own media query predates them */
@media (max-width: 640px) {
  /* clear the notch/status bar and the floating back button (standalone
     PWA mode has no Safari chrome to push content down for us) */
  .sheet { padding:calc(52px + env(safe-area-inset-top)) 8px 32px !important; }
  body { overscroll-behavior-y:none !important; -webkit-overflow-scrolling:touch !important; }
  .topbar h1 { font-size:17px !important; }
  .subhead { display:none !important; }
  .grid { grid-template-columns:repeat(4,1fr) !important; }
  .measure { min-height:66px !important; padding:4px 1px !important; }
  .chords { gap:2px !important; }
  .chord .root { font-size:24px !important; }
  .chord .qual { font-size:15px !important; }
}
@media (max-width: 360px) {
  .grid { grid-template-columns:repeat(2,1fr) !important; }
  .measure { min-height:80px !important; }
  .chord .root { font-size:30px !important; }
  .chord .qual { font-size:19px !important; }
}
/* a bar with 2+ chords shrinks so it doesn't wrap onto a second line and
   blow out that entire grid row's height (every measure in a CSS grid row
   grows to match the tallest cell in it) */
.measure:has(.chords > .chord:nth-child(2)) .chords { gap:8px !important; }
.measure:has(.chords > .chord:nth-child(2)) .chord .root { font-size:19px !important; }
.measure:has(.chords > .chord:nth-child(2)) .chord .qual { font-size:12px !important; }
@media (max-width: 640px) {
  .measure:has(.chords > .chord:nth-child(2)) .chord .root { font-size:15px !important; }
  .measure:has(.chords > .chord:nth-child(2)) .chord .qual { font-size:10px !important; }
}
@media (max-width: 360px) {
  .measure:has(.chords > .chord:nth-child(2)) .chord .root { font-size:19px !important; }
  .measure:has(.chords > .chord:nth-child(2)) .chord .qual { font-size:12px !important; }
}
/* ── Swipe transition: exit animation runs from the swipe handler inline;
   this is the entrance half, keyed by a flag the previous page set before
   navigating. Lives here (not in chart_interactive.py's own template) so it
   applies to every chart immediately, old or new, no migration needed. ── */
@keyframes harmEnterR { from{opacity:0; transform:translateX(26px);} to{opacity:1; transform:translateX(0);} }
@keyframes harmEnterL { from{opacity:0; transform:translateX(-26px);} to{opacity:1; transform:translateX(0);} }
html[data-enter="next"] .sheet { animation:harmEnterR .3s cubic-bezier(.22,.68,0,1); }
html[data-enter="prev"] .sheet { animation:harmEnterL .3s cubic-bezier(.22,.68,0,1); }
</style>
<script>
(function(){
  var d = sessionStorage.getItem("harmSwipeDir");
  if(d){ document.documentElement.setAttribute("data-enter", d); sessionStorage.removeItem("harmSwipeDir"); }
})();
</script>"""

app = Flask(__name__, static_folder=None)

# ── CLI args stored globally so routes can read them ─────────────────────────
_ARGS: argparse.Namespace | None = None

# ── Acoustic front-end for fresh /api/analyze requests ───────────────────────
# 2026-07-17: production deploy of the NNLS-24 feature front-end + music-x-lab
# routed-bass pipeline (docs/known_issues.md "music-x-lab BASS FRONT-END
# DEPLOYED"). This is the new default for freshly-analysed audio. Fully
# reversible WITHOUT a code change:
#   HARMONIA_ANALYZE_FRONTEND=bp48    → revert to the prior Billboard/BP48 chain
#   HARMONIA_ANALYZE_BASS=nnls24      → keep NNLS-24 features but drop music-x-lab bass
#   HARMONIA_ANALYZE_QUALITY=nnls24   → keep the in-house NNLS-24 root/quality heads
# The analyze route also try/excepts this path and falls back to the exact prior
# Billboard→infer_chords_v1 chain if the NNLS-24/musx pipeline raises, so a
# new-pipeline bug can never hard-break analysis for users.
#
# 2026-07-17 (DEPLOY-3): music-x-lab's OWN root/quality replace the NNLS-24 heads
# by default (FAIR bake-off: +7.3pp root / +13.5pp quality / +13.9pp joint on
# RWC). It reuses the same music-x-lab .lab already loaded for the routed bass, so
# there is no extra inference cost. NNLS-24 stays the bass root-veto only.
_ANALYZE_FEATURE_FRONTEND = os.environ.get("HARMONIA_ANALYZE_FRONTEND", "nnls24")
_ANALYZE_BASS_FRONTEND = os.environ.get("HARMONIA_ANALYZE_BASS", "musx")
_ANALYZE_QUALITY_FRONTEND = os.environ.get("HARMONIA_ANALYZE_QUALITY", "musx")
# Segmentation source (chord-CHANGE timing). "nnls" (default, unchanged): cut at
# every beat where the per-beat NNLS root argmax flips. "musx": use music-x-lab's
# OWN chord-change times snapped to the beat grid (boundary-F1 0.90 vs RWC GT,
# vs the NNLS argmax mechanism's documented over-segmentation). Opt-in via
# HARMONIA_ANALYZE_SEGSOURCE=musx; falls back to NNLS segs if musx is unavailable.
_ANALYZE_SEGMENT_SOURCE = os.environ.get("HARMONIA_ANALYZE_SEGSOURCE", "nnls")
# Beat-grid period (2026-07-19, "BAR-GRID vs REAL-MUSIC DRIFT"): "librosa"
# (default, bit-identical grid) vs "bestfit" (whole-song LSQ period; removes
# the systematic multi-bar drift, madmom-corroborated 11/14 songs — see
# scratchpad/beatgrid_madmom_validate.json). Staged rollout: opt-in only.
# Default flipped to "bestfit" 2026-07-19 with the pipeline default (commit
# cf0d1d1) — the env fallback had stayed "librosa" and silently overrode the
# shipped pipeline default on the analyze route (caught by the barlocked
# section-pass debugging). Rollback: HARMONIA_BEAT_PERIOD_MODE=librosa.
_ANALYZE_BEAT_PERIOD_MODE = os.environ.get("HARMONIA_BEAT_PERIOD_MODE", "bestfit")

# ── In-progress jobs: {job_id: {"status": ..., "url": ..., "out": ...}} ─────
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

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

_OVERLAY_HTML_TOOLS = r"""
<style>
/* ── Fallback FAB — only rendered (via JS) on pages that predate the
   Options sheet (tab/iReal comparison & diagnostic pages, not chart_
   interactive.py's own template). Normal chart pages hide this. ── */
#harm-fabs{
  position:fixed; bottom:24px; right:24px; z-index:9999;
  display:flex; flex-direction:column; gap:10px; align-items:flex-end;
}
.harm-fab{
  background:#8a2b2b; color:#fff; border:none; border-radius:50px;
  padding:11px 20px; font:700 14px system-ui,sans-serif;
  cursor:pointer; box-shadow:0 3px 12px #0004;
  display:flex; align-items:center; gap:8px; transition:background .15s;
  white-space:nowrap;
}
.harm-fab:hover { background:#a83333; }
.harm-fab svg { width:18px; height:18px; flex:0 0 auto; }
/* ── Shared modal backdrop ───────────────────────────── */
.harm-modal-bg{
  display:none; position:fixed; inset:0; background:#0007;
  z-index:10000; align-items:center; justify-content:center;
}
.harm-modal-bg.open { display:flex; }
.harm-modal{
  background:#f7f3e9; border-radius:14px; padding:28px 32px;
  max-width:520px; width:93%; box-shadow:0 8px 40px #0005;
  font-family:system-ui,sans-serif; max-height:90vh; overflow-y:auto;
}
.harm-modal h2 { margin:0 0 6px; font-size:18px; color:#1c1c1c; }
.harm-modal .sub { margin:0 0 16px; font-size:13px; color:#6b6050; }
.harm-input{
  width:100%; box-sizing:border-box; padding:9px 12px;
  border:1.5px solid #cfc7ae; border-radius:8px; font-size:14px;
  background:#fff; margin-bottom:10px;
}
.harm-input:focus { outline:none; border-color:#8a2b2b; }
.harm-row { display:flex; gap:10px; margin-bottom:4px; }
.harm-btn{
  flex:1; padding:10px; border:none; border-radius:8px;
  font:700 14px system-ui,sans-serif; cursor:pointer;
}
.harm-btn-primary { background:#8a2b2b; color:#fff; }
.harm-btn-primary:hover { background:#a83333; }
.harm-btn-primary:disabled { background:#bba0a0; cursor:default; }
.harm-btn-secondary { background:#e2dac4; color:#4a4636; }
.harm-btn-secondary:hover { background:#cfc7ae; }
.harm-status { margin-top:12px; font-size:13px; color:#4a4636; min-height:18px; }
.harm-status.err { color:#8a2b2b; }
.harm-spinner{
  display:inline-block; width:16px; height:16px; border-radius:50%;
  border:2.5px solid #cfc7ae; border-top-color:#8a2b2b;
  animation:harm-spin .7s linear infinite; vertical-align:middle; margin-right:6px;
}
@keyframes harm-spin { to { transform:rotate(360deg); } }
/* ── Tab results list ────────────────────────────────── */
#tab-results { margin-top:14px; }
.tab-result{
  border:1.5px solid #e2dac4; border-radius:8px; padding:10px 14px;
  margin-bottom:8px; cursor:pointer; transition:border-color .12s, background .12s;
  font-size:13px;
}
.tab-result:hover { border-color:#8a2b2b; background:#fdf8f0; }
.tab-result.selected { border-color:#8a2b2b; background:#fdf8f0; }
.tab-result-title { font-weight:700; font-size:14px; color:#1c1c1c; }
.tab-result-meta { color:#6b6050; margin-top:3px; font-size:12px; }
.tab-stars { color:#c07a20; font-size:13px; margin-right:4px; }
.tab-votes { color:#8a8371; }
.tab-type-badge{
  display:inline-block; background:#e2dac4; color:#4a4636;
  border-radius:4px; padding:1px 6px; font-size:11px; font-weight:700;
  margin-left:6px; vertical-align:middle;
}
</style>

<script>
// These used to be floating FABs sitting permanently over the chart. Same
// "chrome shouldn't compete with content" rule as the rotor/uncertainty/
// jazzify controls — on a chart page (which has the Options sheet) they now
// live as a section in there instead. Pages without an Options sheet (tab/
// iReal comparison views, not rendered by chart_interactive.py) keep the
// old floating FABs so the feature isn't silently lost on those pages.
(function(){
  var panel = document.querySelector("#optionsModal .modal-panel");
  function wire(btnId, modalBgId, focusId){
    document.getElementById(btnId).addEventListener("click", function(){
      if(typeof closeAllModals === "function") closeAllModals();
      document.getElementById(modalBgId).classList.add("open");
      document.getElementById(focusId).focus();
    });
  }
  if(panel){
    var hr = document.createElement("hr");
    var section = document.createElement("div");
    section.className = "opt-section";
    section.innerHTML =
      '<div class="opt-title">Import</div>' +
      '<div class="opt-row">' +
        '<button type="button" id="tab-fab">Guitar Tabs</button>' +
        '<button type="button" id="irealb-fab">iReal Pro</button>' +
        '<button type="button" id="yt-fab">Analyze YouTube</button>' +
      '</div>';
    panel.appendChild(hr);
    panel.appendChild(section);
  } else {
    var fabs = document.createElement("div");
    fabs.id = "harm-fabs";
    fabs.innerHTML =
      '<button class="harm-fab" id="tab-fab">Guitar Tabs</button>' +
      '<button class="harm-fab" id="irealb-fab">iReal Pro</button>' +
      '<button class="harm-fab" id="yt-fab">Analyze YouTube</button>';
    document.body.appendChild(fabs);
  }
  wire("tab-fab", "tab-modal-bg", "tab-title");
  wire("irealb-fab", "irealb-modal-bg", "irealb-title");
  wire("yt-fab", "yt-modal-bg", "yt-url");
})();
</script>

<!-- YouTube modal -->
<div id="yt-modal-bg" class="harm-modal-bg" onclick="if(event.target===this)closeYtModal()">
  <div class="harm-modal">
    <h2>Analyze a YouTube song</h2>
    <p class="sub">Paste a URL — Harmonia downloads the audio, infers chords, and opens the interactive chart.</p>
    <input id="yt-url" class="harm-input" type="url" placeholder="https://www.youtube.com/watch?v=..." autocomplete="off"
           onkeydown="if(event.key==='Enter')startAnalysis()">
    <div class="harm-row">
      <button class="harm-btn harm-btn-primary" id="yt-go" onclick="startAnalysis()">Analyze</button>
      <button class="harm-btn harm-btn-secondary" onclick="closeYtModal()">Cancel</button>
    </div>
    <div class="harm-status" id="yt-status"></div>
  </div>
</div>

<!-- iReal Pro modal -->
<div id="irealb-modal-bg" class="harm-modal-bg" onclick="if(event.target===this)closeIrealbModal()">
  <div class="harm-modal">
    <h2>iReal Pro chart</h2>
    <p class="sub">Search the iReal community or paste an <code>irealb://</code> URL directly from iReal Pro.</p>
    <input id="irealb-title"  class="harm-input" placeholder="Song title" autocomplete="off"
           onkeydown="if(event.key==='Enter')searchIrealb()">
    <input id="irealb-artist" class="harm-input" placeholder="Artist (optional)" autocomplete="off"
           onkeydown="if(event.key==='Enter')searchIrealb()">
    <div class="harm-row">
      <button class="harm-btn harm-btn-primary" id="irealb-search-btn" onclick="searchIrealb()">Search</button>
      <button class="harm-btn harm-btn-secondary" onclick="closeIrealbModal()">Cancel</button>
    </div>
    <div class="harm-status" id="irealb-status"></div>
    <div id="irealb-results"></div>
    <!-- Direct URL paste -->
    <details style="margin-top:14px;font-family:system-ui,sans-serif;font-size:12px;color:#6b6050;">
      <summary style="cursor:pointer;user-select:none">Or paste an irealb:// URL directly</summary>
      <div style="margin-top:8px;display:flex;gap:8px;">
        <input id="irealb-direct-url" class="harm-input" placeholder="irealb://..." style="margin-bottom:0;font-family:monospace;font-size:12px"
               onkeydown="if(event.key==='Enter')renderDirectIrealb()">
        <button class="harm-btn harm-btn-secondary" style="flex:0 0 auto;white-space:nowrap" onclick="renderDirectIrealb()">Load ↓</button>
      </div>
    </details>
    <!-- Chart offset + BPM override (shown after selecting a result) -->
    <div id="irealb-render-opts" style="display:none;margin-top:14px;border-top:1px solid #e2dac4;padding-top:12px;font-family:system-ui,sans-serif;font-size:12px;color:#4a4636;">
      <div style="display:flex;gap:14px;flex-wrap:wrap;margin-bottom:8px;">
        <label style="display:flex;align-items:center;gap:6px;">
          Chart starts at (s):
          <input type="number" id="irealb-offset" value="0" min="0" step="1" style="width:70px;padding:4px 6px;border:1px solid #cfc7ae;border-radius:6px;font-size:12px;">
        </label>
        <label style="display:flex;align-items:center;gap:6px;">
          BPM override:
          <input type="number" id="irealb-bpm" placeholder="auto" min="40" max="320" step="1" style="width:70px;padding:4px 6px;border:1px solid #cfc7ae;border-radius:6px;font-size:12px;">
        </label>
      </div>
      <button class="harm-btn harm-btn-primary" id="irealb-render-btn" style="width:100%" onclick="renderSelectedIrealb()">Render as Chart</button>
    </div>
  </div>
</div>

<!-- Guitar tabs modal -->
<div id="tab-modal-bg" class="harm-modal-bg" onclick="if(event.target===this)closeTabModal()">
  <div class="harm-modal">
    <h2>Guitar Tabs lookup</h2>
    <p class="sub">Search Ultimate Guitar by title and artist — results are ranked by rating × votes.</p>
    <input id="tab-title"  class="harm-input" placeholder="Song title" autocomplete="off"
           onkeydown="if(event.key==='Enter')searchTabs()">
    <input id="tab-artist" class="harm-input" placeholder="Artist (optional)" autocomplete="off"
           onkeydown="if(event.key==='Enter')searchTabs()">
    <div class="harm-row">
      <button class="harm-btn harm-btn-primary" id="tab-search-btn" onclick="searchTabs()">Search</button>
      <button class="harm-btn harm-btn-secondary" onclick="closeTabModal()">Cancel</button>
    </div>
    <div class="harm-status" id="tab-status"></div>
    <div id="tab-results"></div>
  </div>
</div>

<style>
/* ── Tab comparison panel ────────────────────────────── */
#tab-panel{
  display:none; position:fixed; top:0; right:0; width:280px; height:100vh;
  background:#f7f3e9; border-left:1px solid #cfc7ae;
  box-shadow:-4px 0 18px #0003; z-index:8888;
  font-family:system-ui,sans-serif; font-size:12px;
  overflow-y:auto; padding:16px 14px 32px;
}
#tab-panel.open { display:block; }
#tab-panel-close{
  position:absolute; top:10px; right:12px; background:none; border:none;
  font-size:18px; cursor:pointer; color:#6b6050; line-height:1;
}
#tab-panel h3 { margin:0 0 4px; font-size:14px; color:#1c1c1c; }
#tab-panel .tp-meta { color:#8a8371; margin-bottom:12px; font-size:11px; }
#tab-panel .tp-key  { background:#efe9d9; border-radius:6px; padding:6px 10px;
                       margin-bottom:10px; font-size:12px; }
.tp-legend { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:10px; }
.tp-dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:4px; }
.tp-legend span { display:flex; align-items:center; color:#4a4636; }
#tab-panel table { width:100%; border-collapse:collapse; font-size:11.5px; }
#tab-panel th { text-align:left; padding:3px 4px; color:#6b6050;
                border-bottom:1px solid #e2dac4; font-weight:600; }
#tab-panel td { padding:3px 4px; border-bottom:1px solid #f0ece0; }
#tab-panel tr.match-exact   td:nth-child(3) { color:#2a7a2a; font-weight:700; }
#tab-panel tr.match-family  td:nth-child(3) { color:#b07820; font-weight:600; }
#tab-panel tr.match-mismatch td:nth-child(3) { color:#8a2b2b; }
#tab-panel tr.match-gap     td:nth-child(3) { color:#aaa; font-style:italic; }
/* Chord dot markers on the grid */
.tab-dot{
  display:inline-block; width:8px; height:8px; border-radius:50%;
  position:absolute; top:3px; right:4px; pointer-events:none;
}
.tab-dot-exact    { background:#2a7a2a; }
.tab-dot-family   { background:#c07a20; }
.tab-dot-mismatch { background:#8a2b2b; }
</style>

<!-- Tab comparison side panel (injected into the live chart) -->
<div id="tab-panel">
  <button id="tab-panel-close" onclick="closeTabPanel()" title="Close">✕</button>
  <h3 id="tp-title">Tab comparison</h3>
  <div class="tp-meta" id="tp-meta"></div>
  <div class="tp-key" id="tp-key" style="display:none"></div>
  <div class="tp-legend">
    <span><span class="tp-dot" style="background:#2a7a2a"></span>exact</span>
    <span><span class="tp-dot" style="background:#c07a20"></span>family</span>
    <span><span class="tp-dot" style="background:#8a2b2b"></span>mismatch</span>
  </div>
  <div id="tp-stats" style="margin-bottom:10px;font-size:12px;color:#4a4636;"></div>
  <table>
    <thead><tr><th>Bar</th><th>Inferred</th><th>Tab</th></tr></thead>
    <tbody id="tp-rows"></tbody>
  </table>
</div>

<script>
(function(){
  function escHtml(s){ const d=document.createElement('div'); d.textContent=s||''; return d.innerHTML; }
  function starsHtml(r){ const f=Math.round(r); return '★'.repeat(f)+'☆'.repeat(5-f); }

  // ── YouTube modal ───────────────────────────────────────────────
  function closeYtModal(){
    document.getElementById('yt-modal-bg').classList.remove('open');
    if(typeof ytJargonStop==='function') ytJargonStop();
    setYtStatus('','');
    document.getElementById('yt-go').disabled=false;
  }
  window.closeYtModal = closeYtModal;
  window.closeModal   = closeYtModal;

  function setYtStatus(msg,cls){
    const s=document.getElementById('yt-status');
    s.textContent=msg; s.className='harm-status '+(cls||'');
  }

  // ── whimsical progress jargon, à la Claude Code's own spinner ──
  const YT_JARGON=["Reticulating the circle of fifths…","Debiasing the backbeat…",
    "Quantizing the swing feel…","Diagonalizing the ii–V–I…","Convolving with the changes…",
    "Untangling the voice leading…","Backpropagating through the bridge…",
    "Softmaxing the ambiguity…","Cross-validating the chorus…","Warming up the pitch detector…",
    "Aligning downbeats to reality…","Resolving enharmonic spellings…",
    "Bootstrapping the groove prior…","Denoising the chroma…","Tokenizing the tritone sub…",
    "Interrogating the bassline…","Regularizing the rubato…","Vectorizing the vamp…",
    "Annealing the modulation…","Gradient-descending the changes…"];
  let _ytJargonTimer=null, _ytJargonIdx=0;
  function ytJargonStart(){
    clearInterval(_ytJargonTimer);
    _ytJargonIdx=Math.floor(Math.random()*YT_JARGON.length);
    setYtStatus(YT_JARGON[_ytJargonIdx],'');
    _ytJargonTimer=setInterval(()=>{
      _ytJargonIdx=(_ytJargonIdx+1)%YT_JARGON.length;
      setYtStatus(YT_JARGON[_ytJargonIdx],'');
    },1600);
  }
  function ytJargonStop(){ clearInterval(_ytJargonTimer); _ytJargonTimer=null; }

  function pollJob(jobId){
    fetch('/api/job/'+jobId).then(r=>r.json()).then(d=>{
      if(d.status==='done'){ ytJargonStop(); window.location.href=d.url; }
      else if(d.status==='error'){
        ytJargonStop();
        document.getElementById('yt-go').disabled=false;
        setYtStatus(d.error||'Analysis failed.','err');
      } else {
        setTimeout(()=>pollJob(jobId),1500);
      }
    }).catch(()=>{ setTimeout(()=>pollJob(jobId),2500); });
  }

  window.startAnalysis = function(){
    const url=document.getElementById('yt-url').value.trim();
    if(!url){ setYtStatus('Please enter a YouTube URL.','err'); return; }
    document.getElementById('yt-go').disabled=true;
    ytJargonStart();
    // 2026-07-17: A/B boundary-segmentation override, carried on the page URL
    // (?seg=musx or ?seg=nnls) so it survives re-entering the same song from
    // an iPhone home-screen bookmark — no UI control added yet, URL-only.
    const _seg=new URLSearchParams(location.search).get('seg');
    const body={url}; if(_seg==='musx'||_seg==='nnls'){ body.seg_source=_seg; }
    fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
      .then(r=>r.json())
      .then(d=>{
        if(d.error){ ytJargonStop(); setYtStatus(d.error,'err'); document.getElementById('yt-go').disabled=false; return; }
        pollJob(d.job_id);
      })
      .catch(()=>{ ytJargonStop(); setYtStatus('Could not reach server.','err'); document.getElementById('yt-go').disabled=false; });
  };

  // ── Guitar tabs modal ───────────────────────────────────────────

  // Auto-fill from page title on open
  document.getElementById('tab-fab').addEventListener('click', ()=>{
    // Try to read title from the chart heading
    const h1 = document.querySelector('h1');
    if(h1 && !document.getElementById('tab-title').value){
      // Title may be "Song Name" or "Artist – Song Name" — split on em-dash
      const raw = h1.textContent.trim();
      const parts = raw.split(/\s*[–—]\s*/);
      if(parts.length >= 2){
        document.getElementById('tab-artist').value = parts[0];
        document.getElementById('tab-title').value  = parts.slice(1).join(' ');
      } else {
        document.getElementById('tab-title').value = raw;
      }
    }
  });

  function closeTabModal(){
    document.getElementById('tab-modal-bg').classList.remove('open');
    setTabStatus('','');
    document.getElementById('tab-results').innerHTML='';
    const fb=document.getElementById('tab-fetch-btn');
    if(fb) fb.style.display='none';
    const rb=document.getElementById('tab-render-btn');
    if(rb) rb.style.display='none';
    document.getElementById('tab-search-btn').disabled=false;
  }
  window.closeTabModal = closeTabModal;

  function setTabStatus(msg,cls){
    const s=document.getElementById('tab-status');
    if(msg && msg.startsWith('spinner:')){
      s.innerHTML='<span class="harm-spinner"></span>'+escHtml(msg.slice(8));
    } else {
      s.textContent=msg;
    }
    s.className='harm-status '+(cls||'');
  }

  window.searchTabs = function(){
    const title=document.getElementById('tab-title').value.trim();
    const artist=document.getElementById('tab-artist').value.trim();
    if(!title){ setTabStatus('Please enter a song title.','err'); return; }
    document.getElementById('tab-search-btn').disabled=true;
    document.getElementById('tab-results').innerHTML='';
    const fb=document.getElementById('tab-fetch-btn');
    if(fb) fb.style.display='none';
    _selectedTab=null;
    setTabStatus('spinner:Searching…','');
    fetch('/api/tab-search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({title,artist})})
      .then(r=>r.json())
      .then(d=>{
        document.getElementById('tab-search-btn').disabled=false;
        if(d.error){ setTabStatus(d.error,'err'); return; }
        setTabStatus('','');
        const results=d.results||[];
        if(!results.length){ setTabStatus('No results found.',''); return; }
        const container=document.getElementById('tab-results');
        results.forEach(r=>{
          const div=document.createElement('div');
          div.className='tab-result';
          div.innerHTML=
            '<div class="tab-result-title">'+escHtml(r.artist_name)+' — '+escHtml(r.song_name)+
            '<span class="tab-type-badge">'+escHtml(r.tab_type)+'</span></div>'+
            '<div class="tab-result-meta">'+
            '<span class="tab-stars">'+starsHtml(r.rating)+'</span>'+
            r.rating.toFixed(2)+' <span class="tab-votes">('+r.votes+' votes)</span>'+
            (r.tonality?' · Key: '+escHtml(r.tonality):'')+
            (r.difficulty?' · '+escHtml(r.difficulty):'')+
            '</div>';
          div.onclick=()=>selectTab(div,r);
          container.appendChild(div);
        });
      })
      .catch(()=>{ setTabStatus('Server error.','err'); document.getElementById('tab-search-btn').disabled=false; });
  };

  let _selectedTab=null;
  function selectTab(div, result){
    document.querySelectorAll('.tab-result').forEach(d=>d.classList.remove('selected'));
    div.classList.add('selected');
    _selectedTab=result;
    let btn=document.getElementById('tab-fetch-btn');
    if(!btn){
      btn=document.createElement('button');
      btn.id='tab-fetch-btn';
      btn.className='harm-btn harm-btn-primary';
      btn.style.cssText='width:100%;margin-top:12px';
      // Only show "Compare & merge" if we're on a live chart (P is defined)
      btn.textContent=typeof P!=='undefined' ? 'Compare & merge with inferred' : 'View chords';
      btn.onclick=typeof P!=='undefined' ? alignSelectedTab : fetchSelectedTab;
      document.getElementById('tab-results').after(btn);
    } else {
      btn.textContent=typeof P!=='undefined' ? 'Compare & merge with inferred' : 'View chords';
      btn.onclick=typeof P!=='undefined' ? alignSelectedTab : fetchSelectedTab;
    }
    btn.style.display='block';
    btn.disabled=false;
    // "Render as Chart" button — always visible
    let rbtn=document.getElementById('tab-render-btn');
    if(!rbtn){
      rbtn=document.createElement('button');
      rbtn.id='tab-render-btn';
      rbtn.className='harm-btn harm-btn-secondary';
      rbtn.style.cssText='width:100%;margin-top:6px';
      rbtn.textContent='Render as Chart';
      rbtn.onclick=renderSelectedTab;
      btn.after(rbtn);
    } else {
      rbtn.style.display='block';
      rbtn.disabled=false;
    }
  }

  // ── View raw tab page (non-chart context) ─────────────────────
  function fetchSelectedTab(){
    if(!_selectedTab) return;
    document.getElementById('tab-fetch-btn').disabled=true;
    setTabStatus('spinner:Fetching tab…','');
    fetch('/api/tab-fetch',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
      tab_url:_selectedTab.tab_url, song_name:_selectedTab.song_name,
      artist_name:_selectedTab.artist_name, rating:_selectedTab.rating,
      votes:_selectedTab.votes, tonality:_selectedTab.tonality,
    })})
      .then(r=>r.json())
      .then(d=>{
        document.getElementById('tab-fetch-btn').disabled=false;
        if(d.error){ setTabStatus(d.error,'err'); return; }
        setTabStatus('','');
        window.location.href=d.url;
      })
      .catch(()=>{ setTabStatus('Server error.','err'); document.getElementById('tab-fetch-btn').disabled=false; });
  }

  // ── Compare & merge (chart context) ───────────────────────────
  function alignSelectedTab(){
    if(!_selectedTab || typeof P==='undefined') return;
    document.getElementById('tab-fetch-btn').disabled=true;
    setTabStatus('spinner:Aligning…','');

    fetch('/api/tab-align',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        tab_url:     _selectedTab.tab_url,
        song_name:   _selectedTab.song_name,
        artist_name: _selectedTab.artist_name,
        rating:      _selectedTab.rating,
        votes:       _selectedTab.votes,
        tonality:    _selectedTab.tonality,
        chart_chords: P.chords,
      })
    })
    .then(r=>r.json())
    .then(d=>{
      document.getElementById('tab-fetch-btn').disabled=false;
      if(d.error){ setTabStatus(d.error,'err'); return; }
      setTabStatus('','');
      closeTabModal();
      applyAlignment(d, _selectedTab);
    })
    .catch(()=>{ setTabStatus('Server error.','err'); document.getElementById('tab-fetch-btn').disabled=false; });
  }

  // ── Apply alignment to the chart ──────────────────────────────
  function applyAlignment(data, tabResult){
    const anns = data.annotations || [];
    const xpose = data.transpose_semitones;
    const cost  = data.dtw_cost;

    // 1. Boost reinforcedConf for matching chords
    // (reinforcedConf is defined in the chart's own script)
    if(typeof reinforcedConf !== 'undefined'){
      anns.forEach(a=>{
        if(a.tab_conf_boost > 0){
          const cur = P.chords[a.chord_idx]?.lv?.seventh?.c ?? 0;
          reinforcedConf.set(a.chord_idx, Math.min(0.97, cur + a.tab_conf_boost));
        }
      });
      if(typeof render === 'function') render();
    }

    // 2. Add coloured dot to each measure cell in the grid
    document.querySelectorAll('.tab-dot').forEach(el=>el.remove());
    anns.forEach(a=>{
      if(a.match==='gap') return;
      const el=document.getElementById('chord-'+a.chord_idx);
      if(!el) return;
      const dot=document.createElement('span');
      dot.className='tab-dot tab-dot-'+a.match;
      dot.title='Tab: '+a.tab_chord+' ('+a.match+')';
      el.style.position='relative';
      el.appendChild(dot);
    });

    // 3. Open side panel
    const n=anns.length;
    const exact  = anns.filter(a=>a.match==='exact').length;
    const family = anns.filter(a=>a.match==='family').length;
    const miss   = anns.filter(a=>a.match==='mismatch').length;

    document.getElementById('tp-title').textContent=
      tabResult.artist_name+' — '+tabResult.song_name;
    document.getElementById('tp-meta').textContent=
      starsHtml(tabResult.rating)+' '+tabResult.rating.toFixed(2)+' ('+tabResult.votes+' votes)'+
      (tabResult.tonality?' · Key '+tabResult.tonality:'');

    const keyDiv=document.getElementById('tp-key');
    if(xpose===0){
      keyDiv.textContent='Same key as inferred chart ✓';
    } else {
      const SHARP=["C","C♯","D","D♯","E","F","F♯","G","G♯","A","A♯","B"];
      keyDiv.textContent='Tab transposed +'+xpose+' semitones to match chart key';
    }
    keyDiv.style.display='block';

    document.getElementById('tp-stats').innerHTML=
      '<b>'+exact+'</b> exact · <b>'+family+'</b> family · <b>'+miss+'</b> mismatch'+
      ' <span style="color:#8a8371">(DTW cost '+cost.toFixed(2)+')</span>';

    // Populate table
    const tbody=document.getElementById('tp-rows');
    tbody.innerHTML='';
    anns.forEach(a=>{
      const c=P.chords[a.chord_idx];
      if(!c) return;
      const SHARP=["C","C♯","D","D♯","E","F","F♯","G","G♯","A","A♯","B"];
      const inferredLabel=(c.root>=0?SHARP[c.root]:'')+
        (c.lv?.seventh?.q||'');
      const tr=document.createElement('tr');
      tr.className='match-'+a.match;
      tr.innerHTML='<td>'+(c.bar+1)+'</td>'+
        '<td>'+escHtml(inferredLabel)+'</td>'+
        '<td>'+escHtml(a.tab_chord||'—')+'</td>';
      tbody.appendChild(tr);
    });

    document.getElementById('tab-panel').classList.add('open');
  }

  window.closeTabPanel=function(){
    document.getElementById('tab-panel').classList.remove('open');
    // Remove dots from grid
    document.querySelectorAll('.tab-dot').forEach(el=>el.remove());
    // Clear confidence boosts
    if(typeof reinforcedConf!=='undefined'){ reinforcedConf.clear(); }
    if(typeof render==='function') render();
  };

  // ── Render tab as standalone chart ───────────────────────────────
  function renderSelectedTab(){
    if(!_selectedTab) return;
    const rbtn=document.getElementById('tab-render-btn');
    if(rbtn) rbtn.disabled=true;
    setTabStatus('spinner:Rendering chart from tab…','');
    // Get tempo from the current page's P if available, else 120
    const tempo=(typeof P!=='undefined'&&P.tempo)?P.tempo:120;
    // Get video_id if on a YouTube chart
    const vid=window.YT_VIDEO_ID||'';
    // Get song duration from YT player if available
    let duration_s=0;
    if(window._ytPlayer&&typeof window._ytPlayer.getDuration==='function'){
      try{ duration_s=window._ytPlayer.getDuration()||0; }catch(e){}
    }
    fetch('/api/render-tab',{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        tab_url:     _selectedTab.tab_url,
        song_name:   _selectedTab.song_name,
        artist_name: _selectedTab.artist_name,
        tempo:       tempo,
        duration_s:  duration_s,
        video_id:    vid,
      })
    })
    .then(r=>r.json())
    .then(d=>{
      if(rbtn) rbtn.disabled=false;
      if(d.error){ setTabStatus(d.error,'err'); return; }
      window.location.href=d.url;
    })
    .catch(()=>{ setTabStatus('Server error.','err'); if(rbtn) rbtn.disabled=false; });
  }
  window.renderSelectedTab=renderSelectedTab;
})();
</script>

<!-- ── First-use hint: swipe / rotor / motifs are all gesture-driven with
     zero visual affordance — shown once ever (localStorage flag), only on
     pages that have the topbar (chart pages, not comparison/diagnostic
     views). ── -->
<style>
#harm-hint { position:fixed; left:50%; bottom:0;
  transform:translateX(-50%) translateY(100%); opacity:0;
  width:min(92vw,380px); background:#1c1c1cee; color:#f0ead8;
  border-radius:16px 16px 0 0; padding:18px 20px calc(18px + env(safe-area-inset-bottom));
  box-shadow:0 -8px 30px #0006; font-family:system-ui,sans-serif;
  font-size:13px; line-height:1.5; z-index:10500;
  transition:opacity .35s ease, transform .35s ease; pointer-events:none; }
#harm-hint.show { opacity:1; transform:translateX(-50%) translateY(0); pointer-events:auto; }
#harm-hint .row { display:flex; align-items:center; gap:12px; margin-bottom:10px; }
#harm-hint .ic { font-size:19px; flex:0 0 auto; width:22px; text-align:center; }
#harm-hint button { width:100%; padding:10px; margin-top:2px; background:#8a2b2b;
  color:#fff; border:none; border-radius:9px; font:700 13px system-ui,sans-serif;
  cursor:pointer; transition:transform .1s ease; }
#harm-hint button:active { transform:scale(.97); }
</style>
<div id="harm-hint">
  <div class="row"><span class="ic">👉</span>Swipe left or right for the next/previous song</div>
  <div class="row"><span class="ic">🎛️</span>Tap the key pill, then drag the dial to transpose</div>
  <div class="row"><span class="ic">✦</span>Options → Motifs shows the song's repeating patterns</div>
  <button type="button" id="harm-hint-dismiss">Got it</button>
</div>
<script>
(function(){
  if(!document.getElementById('optionsModal')) return;   // only on chart pages
  if(localStorage.getItem('harmHintsSeen')) return;
  const el=document.getElementById('harm-hint');
  const dismiss=()=>{ el.classList.remove('show'); localStorage.setItem('harmHintsSeen','1'); };
  setTimeout(()=>el.classList.add('show'), 900);
  document.getElementById('harm-hint-dismiss').addEventListener('click', dismiss);
})();
</script>
"""

# Split out from _OVERLAY_HTML_TOOLS so the docked player's script can be
# gated separately (on window.HARM_AUDIO_URL) from the iReal Pro tools,
# which must run unconditionally. Some other page templates (render-tab,
# irealb-render, irealb-compare) still build their own separate embedded
# YouTube iframe with a #yt-player element — unrelated to this dock, which
# plays locally downloaded audio and never sets that id, so there's no
# collision risk between them.
_OVERLAY_HTML_YT = r"""
<!-- ── iReal Pro comparison tools (always active) + docked local-audio
     player (only active when HARM_AUDIO_URL is set) ── -->
<style>
#yt-player-dock{
  display:none; position:fixed; bottom:0; left:0; right:0; z-index:9990;
  background:#111; box-shadow:0 -4px 24px #0008;
  display:flex; align-items:stretch; gap:0;
}
#yt-player-dock.hidden { display:none !important; }
#yt-dock-thumb {
  flex:0 0 auto; width:120px; height:120px; background:#000 no-repeat center/cover;
  align-self:center; margin-left:10px; border-radius:8px;
}
#yt-dock-info {
  flex:1; padding:10px 16px; color:#ddd; font-family:system-ui,sans-serif;
  font-size:13px; display:flex; flex-direction:column; justify-content:center;
  overflow:hidden;
}
#yt-dock-title { font-weight:700; font-size:14px; color:#fff; margin-bottom:4px;
  white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
#yt-dock-chord { font-size:28px; font-weight:800; color:#7ef9aa;
  font-family:Georgia,serif; letter-spacing:0.02em; min-height:36px; }
#yt-dock-controls { display:flex; gap:8px; margin-top:8px; align-items:center; }
#yt-dock-controls button {
  background:#333; border:none; color:#ddd; border-radius:6px;
  padding:5px 12px; font-size:12px; cursor:pointer; transition:background .12s;
}
#yt-dock-controls button:hover { background:#555; }
#yt-dock-hide { position:absolute; top:6px; right:10px; background:none; border:none;
  color:#888; font-size:18px; cursor:pointer; z-index:1; line-height:1; }
#yt-dock-hide:hover { color:#ddd; }
/* now-playing highlight on chord cells */
.chord-now-playing {
  background: rgba(126,249,170,0.25) !important;
  border-radius: 4px;
  outline: 2px solid #7ef9aa;
  outline-offset: 1px;
  transition: background 0.08s;
}
</style>

<div id="yt-player-dock" class="hidden">
  <button id="yt-dock-hide" title="Hide player" onclick="document.getElementById('yt-player-dock').classList.add('hidden')">✕</button>
  <div id="yt-dock-thumb"></div>
  <div id="yt-dock-info">
    <div id="yt-dock-title">Loading…</div>
    <div id="yt-dock-chord"></div>
    <div id="yt-dock-controls">
      <button onclick="ytPlayer&&ytPlayer.seekTo(0)">⏮ Restart</button>
      <button id="yt-playpause" onclick="ytTogglePlay()">⏸ Pause</button>
      <span id="yt-dock-time" style="color:#888;font-size:11px;margin-left:4px"></span>
    </div>
  </div>
</div>
<audio id="harm-audio" preload="none" playsinline crossorigin="anonymous"></audio>

<script>
// ── iReal Pro modal — was previously nested inside the YouTube player's
// IIFE, gated behind "if(!window.YT_VIDEO_ID) return", which meant iReal
// search silently did nothing on any chart without a YouTube video. Its own
// IIFE now, unconditional. ──
(function(){
  function escHtml(s){ const d=document.createElement('div'); d.textContent=s||''; return d.innerHTML; }

  function closeIrealbModal(){
    document.getElementById('irealb-modal-bg').classList.remove('open');
    setIrealbStatus('','');
    document.getElementById('irealb-results').innerHTML='';
    document.getElementById('irealb-render-opts').style.display='none';
    document.getElementById('irealb-search-btn').disabled=false;
    document.getElementById('irealb-direct-url').value='';
    _selectedIrealb=null;
    _activeIrealbUrl=null;
  }
  window.closeIrealbModal=closeIrealbModal;

  function setIrealbStatus(msg,cls){
    const s=document.getElementById('irealb-status');
    if(msg && msg.startsWith('spinner:')){
      s.innerHTML='<span class="harm-spinner"></span>'+escHtml(msg.slice(8));
    } else { s.textContent=msg; }
    s.className='harm-status '+(cls||'');
  }

  // Auto-fill title from chart h1 on modal open
  document.getElementById('irealb-fab').addEventListener('click',()=>{
    const h1=document.querySelector('h1');
    if(h1 && !document.getElementById('irealb-title').value){
      const raw=h1.textContent.trim();
      const parts=raw.split(/\s*[–—]\s*/);
      if(parts.length>=2){
        document.getElementById('irealb-artist').value=parts[0];
        document.getElementById('irealb-title').value=parts.slice(1).join(' ');
      } else {
        document.getElementById('irealb-title').value=raw;
      }
    }
  });

  let _selectedIrealb=null;

  window.searchIrealb=function(){
    const title=document.getElementById('irealb-title').value.trim();
    const artist=document.getElementById('irealb-artist').value.trim();
    if(!title){ setIrealbStatus('Please enter a song title.','err'); return; }
    document.getElementById('irealb-search-btn').disabled=true;
    document.getElementById('irealb-results').innerHTML='';
    document.getElementById('irealb-render-opts').style.display='none';
    _selectedIrealb=null;
    setIrealbStatus('spinner:Searching iReal community…','');
    fetch('/api/irealb-search',{method:'POST',headers:{'Content-Type':'application/json'},
      body:JSON.stringify({title,artist})})
    .then(r=>r.json())
    .then(d=>{
      document.getElementById('irealb-search-btn').disabled=false;
      if(d.error){ setIrealbStatus(d.error,'err'); return; }
      setIrealbStatus('','');
      const results=d.results||[];
      if(!results.length){ setIrealbStatus('No results found.',''); return; }
      const container=document.getElementById('irealb-results');
      results.forEach(r=>{
        const div=document.createElement('div');
        div.className='tab-result';
        div.innerHTML=
          '<div class="tab-result-title">'+escHtml(r.title)+
          (r.composer?' <span style="font-weight:400;color:#6b6050">— '+escHtml(r.composer)+'</span>':'')+
          '</div>'+
          '<div class="tab-result-meta">'+
          'Key: '+escHtml(r.key||'?')+
          ' · '+escHtml(r.style||'?')+
          ' · '+escHtml(r.time_sig||'4/4')+
          '</div>';
        div.onclick=()=>selectIrealb(div,r);
        container.appendChild(div);
      });
    })
    .catch(()=>{ setIrealbStatus('Server error.','err'); document.getElementById('irealb-search-btn').disabled=false; });
  };

  function selectIrealb(div, result){
    document.querySelectorAll('#irealb-results .tab-result').forEach(d=>d.classList.remove('selected'));
    div.classList.add('selected');
    _selectedIrealb=result;
    _activeIrealbUrl=result.irealb_url;
    document.getElementById('irealb-render-opts').style.display='block';
    _updateIrealbRenderBtn();
  }

  // Track active URL regardless of source (search result or direct paste)
  let _activeIrealbUrl = null;

  function _updateIrealbRenderBtn(){
    const btn=document.getElementById('irealb-render-btn');
    if(!btn) return;
    const hasInferred = typeof P!=='undefined' && P.chords && P.chords.length
                        && P.chords[0] && 'root' in P.chords[0];
    btn.textContent = hasInferred ? 'Align to inferred chart ✦' : 'Render as Chart';
    btn.disabled = !_activeIrealbUrl;
  }
  document.getElementById('irealb-fab').addEventListener('click', _updateIrealbRenderBtn);

  window.renderSelectedIrealb=function(){
    if(!_selectedIrealb && !_activeIrealbUrl) return;
    _doRenderIrealb(_activeIrealbUrl || _selectedIrealb.irealb_url);
  };

  // "Load" button next to direct URL input: validate + show options, do NOT fire yet
  window.renderDirectIrealb=function(){
    const url=document.getElementById('irealb-direct-url').value.trim();
    if(!url.startsWith('irealb://')){ setIrealbStatus('Please paste an irealb:// URL.','err'); return; }
    _activeIrealbUrl = url;
    document.getElementById('irealb-render-opts').style.display='block';
    _updateIrealbRenderBtn();
    setIrealbStatus('URL loaded — set options then click the button below.','');
  };

  function _doRenderIrealb(irealb_url){
    const offset=parseFloat(document.getElementById('irealb-offset').value)||0;
    const bpm=parseInt(document.getElementById('irealb-bpm').value)||null;
    const vid=window.YT_VIDEO_ID||'';
    const btn=document.getElementById('irealb-render-btn');
    if(btn) btn.disabled=true;

    // If on an inferred chart page, use DTW alignment + comparison view
    const hasInferred = typeof P!=='undefined' && P.chords && P.chords.length
                        && P.chords[0] && 'root' in P.chords[0];
    if(hasInferred){
      setIrealbStatus('spinner:Aligning to inferred chart…','');
      fetch('/api/irealb-compare',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({irealb_url, p_chords:P.chords, bpm, video_id:vid})})
      .then(r=>r.json())
      .then(d=>{
        if(btn) btn.disabled=false;
        if(d.error){ setIrealbStatus(d.error,'err'); return; }
        setIrealbStatus(
          `Aligned: +${d.transpose_semitones} st · ${d.n_repeats}× form · `+
          `exact ${(d.exact_frac*100).toFixed(0)}% · family ${(d.family_frac*100).toFixed(0)}%`
        ,'');
        setTimeout(()=>{ window.location.href=d.url; }, 800);
      })
      .catch(()=>{ setIrealbStatus('Server error.','err'); if(btn) btn.disabled=false; });
    } else {
      setIrealbStatus('spinner:Rendering chart…','');
      fetch('/api/irealb-render',{method:'POST',headers:{'Content-Type':'application/json'},
        body:JSON.stringify({irealb_url,chart_offset_s:offset,tempo:bpm,video_id:vid})})
      .then(r=>r.json())
      .then(d=>{
        if(btn) btn.disabled=false;
        if(d.error){ setIrealbStatus(d.error,'err'); return; }
        window.location.href=d.url;
      })
      .catch(()=>{ setIrealbStatus('Server error.','err'); if(btn) btn.disabled=false; });
    }
  }
})();

// ── Docked local-audio player + chord sync. Plays back the audio we already
// downloaded to run inference, instead of re-embedding a YouTube iframe —
// sidesteps origin/CORS quirks, playsinline-forced-fullscreen, embedding-
// disabled videos, and duplicate-player collisions all at once. ──
(function(){
  if(!window.HARM_AUDIO_URL) return;

  function getChordTimes(){
    if(typeof P==='undefined') return [];
    return P.chords.map((c,i)=>({idx:i, t0:c.t0??null, t1:c.t1??null}))
                   .filter(c=>c.t0!==null);
  }

  function chordAt(times, t){
    if(!times.length) return -1;
    let best=-1;
    for(let i=times.length-1;i>=0;i--){
      if(times[i].t0<=t){ best=times[i].idx; break; }
    }
    return best;
  }

  function fmtTime(s){
    const m=Math.floor(s/60), ss=Math.floor(s%60);
    return m+':'+(ss<10?'0':'')+ss;
  }

  function chordLabel(idx){
    if(typeof P==='undefined'||idx<0||idx>=P.chords.length) return '';
    const c=P.chords[idx];
    if(c.label) return c.label;
    const SHARP=["C","C♯","D","D♯","E","F","F♯","G","G♯","A","A♯","B"];
    const root=c.root>=0?SHARP[c.root]:'';
    const lv=c.lv?.seventh||c.lv?.family||{};
    let q=lv.q||'';
    if(q===''||q==='maj') q='';
    else if(q==='-'||q==='min') q='m';
    else if(q==='-7') q='m7';
    else if(q==='^7') q='△7';
    else if(q==='h7') q='ø7';
    else if(q==='o') q='°';
    return root+q;
  }

  function scrollToChord(idx){
    const el=document.getElementById('chord-'+idx);
    if(!el) return;
    const rect=el.getBoundingClientRect();
    const dockH=document.getElementById('yt-player-dock').offsetHeight||180;
    const viewH=window.innerHeight-dockH;
    if(rect.top<60||rect.bottom>viewH-20){
      el.scrollIntoView({behavior:'smooth',block:'center'});
    }
  }

  const audio=document.getElementById('harm-audio');
  audio.preload='metadata';
  audio.src=window.HARM_AUDIO_URL;

  if(window.HARM_THUMB_URL){
    document.getElementById('yt-dock-thumb').style.backgroundImage="url('"+window.HARM_THUMB_URL+"')";
  }
  const title=document.querySelector('h1');
  if(title) document.getElementById('yt-dock-title').textContent=title.textContent.trim();
  document.getElementById('yt-player-dock').classList.remove('hidden');
  document.body.style.paddingBottom='196px';  // dock doesn't cover the last bars

  const _chordTimes=getChordTimes();
  let _currentChordIdx=-1;

  audio.addEventListener('timeupdate',()=>{
    const t=audio.currentTime;
    document.getElementById('yt-dock-time').textContent=fmtTime(t);
    const newIdx=chordAt(_chordTimes, t);
    if(newIdx!==_currentChordIdx){
      if(_currentChordIdx>=0){
        const old=document.getElementById('chord-'+_currentChordIdx);
        if(old) old.classList.remove('chord-now-playing');
      }
      if(newIdx>=0){
        const el=document.getElementById('chord-'+newIdx);
        if(el){ el.classList.add('chord-now-playing'); scrollToChord(newIdx); }
        document.getElementById('yt-dock-chord').textContent=chordLabel(newIdx);
      }
      _currentChordIdx=newIdx;
    }
  });
  audio.addEventListener('play', ()=>{
    const pp=document.getElementById('yt-playpause'); if(pp) pp.textContent='⏸ Pause';
  });
  audio.addEventListener('pause', ()=>{
    const pp=document.getElementById('yt-playpause'); if(pp) pp.textContent='▶ Play';
  });
  audio.addEventListener('error', ()=>{
    console.error('audio playback error', audio.error);
    const info=document.getElementById('yt-dock-info');
    if(info) info.innerHTML='<div id="yt-dock-title">Audio unavailable</div>'
      +'<div style="font-size:12px;color:#aaa">Could not play the downloaded audio.</div>';
  });

  window.ytTogglePlay=function(){ if(audio.paused) audio.play(); else audio.pause(); };
  window.ytPlayer={ seekTo:function(t){ audio.currentTime=t; } };
  window._ytPlayer={
    getDuration: ()=>audio.duration||0,
    getCurrentTime: ()=>audio.currentTime||0,
    getPlayerState: ()=>audio.paused?2:1,
  };
})();
</script>
</body></html>"""

_INJECT_MARKER = "</body></html>"

_BACK_BUTTON_HTML = """<a href="/" id="harm-back"
   style="position:fixed;top:max(12px,env(safe-area-inset-top));left:12px;z-index:9998;
   display:flex;align-items:center;gap:5px;background:#8a2b2bcc;color:#fff;
   text-decoration:none;font:700 13px system-ui,sans-serif;padding:7px 13px 7px 10px;
   border-radius:20px;box-shadow:0 2px 8px #0004;backdrop-filter:blur(4px);
   transition:transform .1s ease;">&larr; Charts</a>
<style>#harm-back:active{transform:scale(.93);}</style>
"""
# 2026-07-17: this used to try history.back() first ("if(history.length>1)")
# and only fall back to /library when there was no history to go back to.
# That's fragile: ANY page reached via a plain same-tab navigation lands in
# THIS tab's history stack, including tool pages like /bar1-offset-fix and
# /gt-align-fix. Reported bug: open the align tool (now a same-tab nav, not
# a real new tab — see app_shell.html's bar1-offset-fix button), tap its own
# "chart" link (a real navigation, bar1_offset_fix pushed a NEW entry), then
# tap this "Charts" button on the chart page — history.back() went to
# whatever was one step back in THIS tab's stack, which is the align tool,
# not the chart library. Dropped the history.back() shortcut (still true).
#
# 2026-07-17, second pass: the first pass pointed this at `/library` (its
# old href target) instead. That "fixed" the loop but broke the COMMON
# case for everyone: app_shell.html's SPA (served at "/") never navigates to
# a full /chart/<file> page itself (it renders charts in place via
# /api/chart-model) — so /chart/<file> is ALWAYS reached from outside the
# SPA (align tool, GT-align tool, direct link, PWA swipe-nav), and every one
# of those visitors then got dumped onto `/library`, a separate, much
# plainer static page (no search, no docked audio player, Jinja-rendered
# list) — reported as "when I click charts I get the old UI instead of the
# new one." `/` is the right target: app_shell.html's own `API.build()`
# calls `go("library")` on boot, so "/" already lands on exactly the "your
# charts" screen, in the current polished SPA — same destination intent,
# correct UI, and still a fresh navigation (no history.back(), so the
# original loop bug stays fixed too).


def _inject_overlay(html: str) -> str:
    """Inject the Guitar Tabs/iReal Pro tools, plus the docked local-audio
    player. The dock's own script no-ops unless window.HARM_AUDIO_URL is
    set, so this no longer needs to dodge pages with their own YouTube
    iframe (render-tab/irealb-* templates) the way it used to."""
    overlay = _OVERLAY_HTML_TOOLS + _OVERLAY_HTML_YT
    if _INJECT_MARKER in html:
        return html.replace(_INJECT_MARKER, overlay, 1)
    return html + overlay


def _inject_back_button(html: str) -> str:
    """Add a fixed 'back to chart list' link — standalone PWA mode has no
    Safari chrome, so there's otherwise no way back off a chart page."""
    return html.replace("<body>", "<body>" + _BACK_BUTTON_HTML, 1)


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


@app.route("/audio/<path:filename>")
def serve_audio(filename):
    """Serve downloaded song audio for the docked player. conditional=True
    (Flask's default) handles Range requests, which iOS Safari needs to
    seek in an <audio> element without redownloading the whole file."""
    p = AUDIO_DIR / filename
    if not p.exists() or p.parent != AUDIO_DIR:
        return "Not found", 404
    # Python's mimetypes module guesses "audio/mp4a-latm" for .m4a on some
    # systems — force the standard type iOS Safari expects for AAC/m4a.
    mimetype = "audio/mp4" if p.suffix == ".m4a" else None
    resp = send_from_directory(AUDIO_DIR, filename, conditional=True, mimetype=mimetype)
    # iOS Safari puts <audio crossorigin="anonymous"> media into CORS mode and
    # validates EVERY Range (206) response for an Access-Control-Allow-Origin
    # header — even same-origin. send_from_directory doesn't add it, so WebKit
    # silently taints the resource and playback (native Play button) does
    # nothing. Desktop Chromium is lenient about same-origin and plays anyway,
    # which masked this. Emit ACAO so any crossOrigin media element works.
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


@app.route("/pwa/<path:filename>")
def serve_pwa_asset(filename):
    """Serve the PWA manifest and home-screen icons."""
    p = PWA_DIR / filename
    if not p.exists() or p.parent != PWA_DIR:
        return "Not found", 404
    mimetype = "application/manifest+json" if p.suffix == ".json" else None
    return Response(p.read_bytes(), mimetype=mimetype)


_STRUCTURE_DEBUG_JSON = REPO / "scratchpad" / "real_structure_results.json"
_STRUCTURE_MULTILEVEL_JSON = REPO / "scratchpad" / "real_structure_multilevel.json"

# NEW debug routes (2026-07-18, chord-distance work — same authorization as
# /debug/structure above). These serve pre-built, self-contained static HTML
# files straight off disk — no server-side templating, so they can't drift
# from what was actually reviewed. Nothing else touched.
_METRIC_ARTIFACT_HTML = REPO / "scratchpad" / "structure_metric_artifact.html"
_SSM_VIZ_HTML = REPO / "scratchpad" / "bar_ssm_viz.html"


@app.route("/debug/metric-artifact")
def debug_metric_artifact():
    """V-measure block-level-vs-per-bar granularity artifact chart (4 iReal
    songs) — see docs/known_issues.md 'CORRECTION: the 0.732 clean-GT oracle'."""
    if not _METRIC_ARTIFACT_HTML.exists():
        return Response("not generated yet", status=404)
    return Response(_METRIC_ARTIFACT_HTML.read_bytes(), mimetype="text/html")


@app.route("/debug/ssm")
def debug_ssm():
    """Bar-to-bar chord-tone-distance self-similarity matrices (one clean
    iReal chart with GT, one real-audio song with raw NNLS chroma) — see
    docs/known_issues.md 'Hand-crafted CHORD-TONE-DISTANCE similarity'."""
    if not _SSM_VIZ_HTML.exists():
        return Response("not generated yet", status=404)
    return Response(_SSM_VIZ_HTML.read_bytes(), mimetype="text/html")


_SSM_MULTIGRAIN_HTML = REPO / "scratchpad" / "bar_ssm_multigrain_viz.html"


@app.route("/debug/ssm-multigrain")
def debug_ssm_multigrain():
    """Same two songs as /debug/ssm, but self-similarity at 5 granularities
    (1/2/4/8/16-bar blocks) side by side per song."""
    if not _SSM_MULTIGRAIN_HTML.exists():
        return Response("not generated yet", status=404)
    return Response(_SSM_MULTIGRAIN_HTML.read_bytes(), mimetype="text/html")


_DUAL_MATRIX_HTML = REPO / "scratchpad" / "dual_matrix_viz.html"


@app.route("/debug/dual-matrix")
def debug_dual_matrix():
    """Audio vs structural (decoded-chord) similarity matrices side by side
    at 8-bar grain, for the 3 real songs, plus the inferred section labels
    from clustering both together — see docs/known_issues.md ★ STRUCTURE /
    SEGMENTATION, 2026-07-18, the section-repeat-ranking diagnostic."""
    if not _DUAL_MATRIX_HTML.exists():
        return Response("not generated yet", status=404)
    return Response(_DUAL_MATRIX_HTML.read_bytes(), mimetype="text/html")


_CRITERIA_VIZ_HTML = REPO / "scratchpad" / "criteria_viz.html"


@app.route("/debug/criteria")
def debug_criteria():
    """Side-by-side comparison of 3 candidate section-matching criteria
    (all built on the Mantel-validated dual-matrix), at k=3/4/5, with the
    <=5-distinct-sections rule and the block0/block1 sanity check per
    criterion/song — see docs/known_issues.md ★ STRUCTURE / SEGMENTATION,
    2026-07-18, "no more than 4-5 sections" constraint work."""
    if not _CRITERIA_VIZ_HTML.exists():
        return Response("not generated yet", status=404)
    return Response(_CRITERIA_VIZ_HTML.read_bytes(), mimetype="text/html")


_K_PRIOR_VIZ_HTML = REPO / "scratchpad" / "k_prior_viz.html"


@app.route("/debug/k-prior")
def debug_k_prior():
    """Learned prior P(k|song_length_bars) from the full 1992-tune iReal
    corpus, combined with the silhouette clustering-quality signal into a
    principled k-selection rule — corpus-scale validation + the 3 real
    songs' chosen k, plotted against the corpus scatter. See
    docs/known_issues.md ★ STRUCTURE / SEGMENTATION, 2026-07-18."""
    if not _K_PRIOR_VIZ_HTML.exists():
        return Response("not generated yet", status=404)
    return Response(_K_PRIOR_VIZ_HTML.read_bytes(), mimetype="text/html")


_BARGRID_PLAYER_HTML = REPO / "scratchpad" / "bargrid_debug_player.html"


@app.route("/debug/bargrid-player")
def debug_bargrid_player():
    """Real waveform + the exact beat_grid()-derived bar timestamps the
    production chart uses, with a synced audio playhead and click-to-seek —
    built so the user can personally listen through a song end-to-end and
    verify by ear/eye whether the bar lines actually land on the downbeats,
    per the 2026-07-19 'la derivation des barres n'est pas du tout bonne'
    report even under fairly constant tempo. See docs/known_issues.md
    ★ CHART / BAR-GRID."""
    if not _BARGRID_PLAYER_HTML.exists():
        return Response("not generated yet", status=404)
    return Response(_BARGRID_PLAYER_HTML.read_bytes(), mimetype="text/html")


_REAL_TRANSFER_HTML = REPO / "scratchpad" / "real_transfer_viz.html"
_GRID_ALIGN_HTML = REPO / "scratchpad" / "grid_align_debug.html"


@app.route("/debug/merge-criterion")
def debug_merge_criterion():
    """2026-07-18 overnight continuation: Steps 2/3/4's bar-merge criterion,
    intro detector, and section detector (all trained/validated on clean
    iReal, see docs/known_issues.md "Step 2"/"Step 3"/"Step 4" entries)
    transferred to the 3 real-audio songs with NO ground truth — qualitative
    human-inspection page only, per the brief's explicit "human validation"
    requirement for the real-audio transfer step. Pre-built static HTML,
    same off-disk-serving pattern as /debug/ssm and /debug/metric-artifact
    above (can't drift from what was actually reviewed)."""
    if not _REAL_TRANSFER_HTML.exists():
        return Response("not generated yet", status=404)
    return Response(_REAL_TRANSFER_HTML.read_bytes(), mimetype="text/html")

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


_BAR_MERGE_GAME_HTML = REPO / "scratchpad" / "bar_merge_game.html"
_BAR_MERGE_GAME_DATA = REPO / "scratchpad" / "bar_merge_game_data.json"


@app.route("/debug/bar-merge-game")
def debug_bar_merge_game():
    """2026-07-18 chord-robustness reframe: interactive "pairs game" for
    confirming candidate bar-merges (from scratchpad/bar_merge_candidates.py,
    threshold+pairs on the untrained 1-bar raw-chroma SSM — see
    docs/known_issues.md "REFRAME: bar-merge SSM pooling") and POSTing
    confirmed spans to the EXISTING /api/reinfer/<filename> merge-pooling
    endpoint (harmonia.models.user_constraints.pool_beat_evidence).
    Separate new debug route per the user's explicit instruction NOT to
    edit chart_interactive.py's existing manual merge UI for this — same
    self-contained-HTML-off-disk pattern as every other /debug/* route
    tonight, candidate data precomputed (scratchpad/bar_merge_game_data.json,
    build via scratchpad/bar_merge_candidates.py) and templated in once at
    request time so the served page can't drift from what was reviewed."""
    if not _BAR_MERGE_GAME_HTML.exists() or not _BAR_MERGE_GAME_DATA.exists():
        return Response("not generated yet — run scratchpad/bar_merge_candidates.py "
                        "and rebuild bar_merge_game_data.json", status=404)
    html = _BAR_MERGE_GAME_HTML.read_text()
    data = _BAR_MERGE_GAME_DATA.read_text()  # already valid JSON text
    html = html.replace("__CANDIDATE_DATA__", data)
    return Response(html, mimetype="text/html")


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


def _chart_model_for(filename: str, include_gt: bool = True) -> dict:
    """ChartModel for a rendered chart — payload + sidecar + audio/video links.

    ``include_gt``: attach McGill Billboard ground-truth chords (training-mode
    songs only — see _gt_chords_for_video) as model["gt"]. Skipped for the
    /api/library summary loop (chart_summary never reads it, and a mirdata
    lookup per song on every library load is needless overhead)."""
    from harmonia.output.chart_model import payload_from_chart_html, to_chart_model

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
    return model


@app.route("/api/library")
def api_library():
    """Every chart we have, as library cards (title, key, bars, has-audio)."""
    from harmonia.output.chart_model import chart_summary

    charts = []
    for p in sorted(PLOTS_DIR.glob("inferred_*.html")):
        try:
            c = chart_summary(_chart_model_for(p.name, include_gt=False))
            c["mtime"] = p.stat().st_mtime  # lets the client offer a Recent sort too
            charts.append(c)
        except (OSError, ValueError, KeyError) as e:
            log.warning("Skipping %s in library: %s", p.name, e)
    # newest first — the chart you just analysed should be at the top
    charts.sort(key=lambda c: c["mtime"], reverse=True)
    return jsonify(charts=charts)


@app.route("/api/chart/<filename>", methods=["DELETE"])
def api_delete_chart(filename):
    """Remove a chart from the library (UX audit 2026-07-20: there was no way
    to delete or reorganize charts once analysed). Removes the chart HTML plus
    its registry entries (video-id link, retained-audio link, annotation
    sidecar) so nothing dangling is left behind; the underlying downloaded
    audio in docs/audio/ is NOT deleted (it may be cheap to keep and re-used
    if the same video is analysed again)."""
    p = PLOTS_DIR / filename
    if p.suffix != ".html" or p.parent != PLOTS_DIR or not p.name.startswith("inferred_"):
        return jsonify(error="Not found"), 404
    if not p.exists():
        return jsonify(error="Not found"), 404
    try:
        p.unlink()
    except OSError as e:
        return jsonify(error=str(e)), 500
    if filename in _yt_video_ids:
        del _yt_video_ids[filename]
        try:
            _YT_IDS_FILE.write_text(json.dumps(_yt_video_ids), encoding="utf-8")
        except OSError:
            log.warning("Could not persist YouTube video ids after deleting %s", filename)
    if filename in _yt_audio_meta:
        del _yt_audio_meta[filename]
        try:
            _YT_AUDIO_FILE.write_text(json.dumps(_yt_audio_meta), encoding="utf-8")
        except OSError:
            log.warning("Could not persist audio registry after deleting %s", filename)
    _annot_path(filename).unlink(missing_ok=True)
    return jsonify(ok=True)


_BILLBOARD_CORPUS_FILES = [
    REPO / "scratchpad" / "billboard_search_results_60.json",
    REPO / "scratchpad" / "billboard_search_results.json",
]


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


@app.route("/api/gt-offset/<track_id>", methods=["GET"])
def api_gt_offset_get(track_id):
    """Current saved GT-offset correction for a McGill Billboard track_id, if any."""
    return jsonify(_load_gt_offsets().get(track_id, {}))


@app.route("/api/gt-offset/<track_id>", methods=["POST"])
def api_gt_offset_save(track_id):
    """Persist a hand-corrected GT offset for a McGill Billboard track_id.
    Body: {"offset_s": float, "source": "manual"|"auto-onset" (optional)}.
    Clears the GT cache so /gt-playalong-training, the training-mode chart,
    and this route's own GET all reflect it immediately, no restart needed."""
    data = request.get_json(force=True, silent=True) or {}
    try:
        offset = float(data.get("offset_s"))
    except (TypeError, ValueError):
        return jsonify(error="offset_s must be a number"), 400
    _save_gt_offset(track_id, offset, source=data.get("source", "manual"))
    return jsonify(ok=True, track_id=track_id, offset_s=offset)


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
def _section_labels_path(filename: str) -> Path:
    return ANNOT_DIR / f"{filename}.sections.json"


def _load_section_labels(filename: str) -> dict:
    try:
        doc = json.loads(_section_labels_path(filename).read_text(encoding="utf-8"))
        if isinstance(doc, dict) and isinstance(doc.get("labels"), dict):
            return doc
    except (OSError, ValueError):
        pass
    return {"labels": {}}


def _save_section_labels(filename: str, labels: dict) -> dict:
    import datetime as _dt
    doc = {
        "labels": labels,
        "updated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
    ANNOT_DIR.mkdir(parents=True, exist_ok=True)
    _section_labels_path(filename).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return doc


@app.route("/api/section-labels/<filename>", methods=["GET"])
def api_section_labels_get(filename):
    """Current hand-drawn section labels for a chart (empty {labels:{}} if none)."""
    return jsonify(_load_section_labels(filename))


@app.route("/api/section-labels/<filename>", methods=["POST"])
def api_section_labels_save(filename):
    """Persist hand-drawn section labels. Body: {"labels": {"<bar>": "<label>"}}.
    Last-write-wins: the client posts the whole current map on every change,
    mirroring /api/annotations. Keys must be int-like bar indices; blank values
    drop that bar's label. Render-only — no re-inference in the request path."""
    data = request.get_json(force=True, silent=True) or {}
    labels = data.get("labels", {})
    if not isinstance(labels, dict):
        return jsonify(error="labels must be an object {bar: label}"), 400
    clean: dict[str, str] = {}
    for k, v in labels.items():
        try:
            bar = int(k)
        except (TypeError, ValueError):
            continue
        if bar < 0 or v is None:
            continue
        text = str(v).strip()[:24]
        if text:
            clean[str(bar)] = text
    saved = _save_section_labels(filename, clean)
    return jsonify(ok=True, filename=filename, **saved)


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


@app.route("/api/chart-model/<filename>")
def api_chart_model(filename):
    """The one clean shape the app UI consumes — see chart_model.to_chart_model."""
    p = PLOTS_DIR / filename
    if not p.exists() or p.suffix != ".html" or p.parent != PLOTS_DIR:
        return jsonify(error="Not found"), 404
    try:
        return jsonify(_chart_model_for(filename))
    except (OSError, ValueError, KeyError) as e:
        log.exception("chart-model failed for %s", filename)
        return jsonify(error=str(e)), 500


@app.route("/api/irealb-export/<filename>")
def api_irealb_export(filename):
    """Export a chart to an irealb:// URL (Share button, 2026-07-20) — the
    reverse of the existing iReal IMPORT path. iReal Pro opens this link
    directly (or it can be pasted into the app)."""
    p = PLOTS_DIR / filename
    if not p.exists() or p.suffix != ".html" or p.parent != PLOTS_DIR:
        return jsonify(error="Not found"), 404
    try:
        from harmonia.irealb_export import chart_model_to_irealb_url
        model = _chart_model_for(filename, include_gt=False)
        url = chart_model_to_irealb_url(model)
        return jsonify(url=url)
    except Exception as e:  # noqa: BLE001 — export is best-effort, never 500 the chart
        log.exception("irealb export failed for %s", filename)
        return jsonify(error=str(e)), 500


@app.route("/demo/progressive-analysis")
def demo_progressive_analysis():
    """Standalone, self-contained mockup of the proposed progressive-analysis
    screen (draft NNLS chords filling in, then corrected by music-x-lab) —
    for viewing on a real phone over the VPN. Not part of the app; served
    straight from scratchpad, no build step. Remove once the design is
    settled or ported into app_shell.html for real."""
    p = REPO / "scratchpad" / "progressive_analysis_demo.html"
    if not p.exists():
        return jsonify(error="demo file not found"), 404
    return Response(p.read_text(encoding="utf-8"), mimetype="text/html")


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
_SWIPE_NAV_JS = """<script>
(function(){
  const PREV="%%PREV%%", NEXT="%%NEXT%%";
  const sheet=document.querySelector(".sheet");
  if(!sheet) return;
  let sx=0, sy=0, dx=0, active=false, deciding=true, horizontal=false;

  function setX(px,animate){
    sheet.style.transition = animate ? "transform .3s cubic-bezier(.22,.68,0,1), opacity .3s" : "none";
    sheet.style.transform = "translateX("+px+"px)";
    sheet.style.opacity = String(Math.max(0.35, 1 - Math.abs(px)/window.innerWidth*0.9));
  }

  document.addEventListener("touchstart",e=>{
    if(e.target.closest(".wheel")||e.target.closest(".modal-panel")){active=false;return;}
    const t=e.touches[0]; sx=t.clientX; sy=t.clientY; dx=0;
    active=true; deciding=true; horizontal=false;
    sheet.style.transition="none";
  },{passive:true});

  document.addEventListener("touchmove",e=>{
    if(!active) return;
    const t=e.touches[0];
    dx=t.clientX-sx; const dy=t.clientY-sy;
    if(deciding){
      if(Math.abs(dx)<6 && Math.abs(dy)<6) return;
      horizontal=Math.abs(dx)>Math.abs(dy)*1.3;
      deciding=false;
      if(!horizontal){ active=false; return; }
    }
    // rubber-band toward a direction with nothing to swipe to (single-chart library)
    const hasTarget = dx<0 ? !!NEXT : !!PREV;
    setX(hasTarget ? dx : dx*0.28, false);
  },{passive:true});

  document.addEventListener("touchend",()=>{
    if(!active || !horizontal){ active=false; return; }
    active=false;
    const target = dx<0 ? NEXT : PREV;
    if(Math.abs(dx)>=70 && target){
      if(navigator.vibrate) navigator.vibrate(8);
      setX(dx<0 ? -window.innerWidth : window.innerWidth, true);
      sessionStorage.setItem("harmSwipeDir", dx<0 ? "next" : "prev");
      setTimeout(()=>{ location.href="/chart/"+target; }, 260);
    } else {
      setX(0,true);  // not a decisive swipe — spring back
    }
  },{passive:true});
})();
</script>"""


@app.route("/gt-align")
def gt_align():
    """GT alignment corrector: 4-bar focused waveform view with draggable chord
    markers, edge-gutter hit areas, continuous auto-pan on edge drag, timeline
    scrubbing, click-to-seek, and keyboard nudging.

    ?song=<slug>  →  drag iReal chords onto the audio to build GT alignment.
    """
    from html import escape

    slug = re.sub(r"[^A-Za-z0-9_]", "", (request.args.get("song") or "autumn_leaves"))

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


@app.route("/gt-chart")
def gt_chart():
    """Serve iReal ground-truth chart with YouTube video sync.

    ?song=<slug>  →  displays irealb_<slug>.html (ground truth) with YouTube/audio playback
    """
    slug = re.sub(r"[^A-Za-z0-9_]", "", (request.args.get("song") or "autumn_leaves"))
    filename = f"irealb_{slug}.html"
    p = PLOTS_DIR / filename

    if not p.exists():
        return f"<p>No iReal chart for {slug}</p>", 404

    content = p.read_text(encoding="utf-8")
    content = content.replace("</head>", _PWA_HEAD + "</head>", 1)

    # Inject YouTube video ID if available
    vid = _yt_video_ids.get(f"inferred_{slug}.html", "")
    if vid:
        content = content.replace(
            "</head>",
            f'<script>window.YT_VIDEO_ID="{vid}"; window.PAGE_TITLE="GT: {slug}";</script></head>',
            1,
        )

    # Inject audio metadata
    audio_meta = _yt_audio_meta.get(f"inferred_{slug}.html")
    if audio_meta and (AUDIO_DIR / Path(audio_meta["audio"]).name).exists():
        content = content.replace(
            "</head>",
            '<script>window.HARM_AUDIO_URL=' + json.dumps(audio_meta["audio"])
            + ';window.HARM_THUMB_URL=' + json.dumps(audio_meta.get("thumb", ""))
            + ';</script></head>',
            1,
        )

    # Add banner: "This is ground truth (iReal), not model inference"
    banner = '''<div style="position:fixed;top:0;right:0;background:#00c9a7;color:#0e1116;padding:8px 12px;font-size:11px;font-weight:700;z-index:100;border-radius:0 0 0 6px;">🎼 GROUND TRUTH (iReal)</div>'''
    content = content.replace("<body>", "<body>" + banner, 1)

    return Response(_inject_back_button(_inject_overlay(content)), mimetype="text/html")


@app.route("/chart/<filename>")
def serve_chart(filename):
    """Serve a chart HTML file with the YouTube overlay injected.

    2026-07-17: also re-derives the embedded ``const P = {...};`` payload
    under any saved bar-1 offset before serving. This route reads the baked
    HTML straight off disk (``p.read_text()``) — it is a SEPARATE code path
    from `_chart_model_for`/`/api/chart-model/<file>` (used by the SPA at
    "/"), which already applied `_apply_bar1_offset_to_payload`. Bug: a user
    who saves a bar-1 offset via /bar1-offset-fix and then lands here (its
    own "← chart" link points at this exact route) saw the ORIGINAL,
    un-shifted chart — the offset was real and persisted, but this route
    never read it. Mirrors the same re-derivation `_chart_model_for` does,
    but as a text substitution on the embedded payload rather than a
    structured API response, since this route serves the standalone
    self-contained chart page (chart_interactive.py's own client JS reads
    `P` directly, not `/api/chart-model`).

    2026-07-18: user complaint — following a `/chart/<file>` link (e.g. from
    the align tool, a swipe-nav, or a shared link) lands on this route's own
    baked-HTML rendering, which is a structurally separate, plainer UI than
    the SPA's own chart view at "/" (`API.build`/`openChart` via
    `/api/chart-model`) even though both show the same "Read/Analyse/
    Annotate" control — two divergent code paths for the same content (see
    docs/known_issues.md's two prior "old UI" entries, both patches WITHIN
    this route rather than a structural fix). Durable fix: this route now
    redirects to `/?open=<file>`, a new deep-link param `app_shell.html`'s
    `API.build` reads to call `openChart(file)` directly instead of landing
    on the library first — so any old-style `/chart/<file>` link now opens
    straight into the polished SPA showing that exact song.

    2026-07-18, later same day: the redirect briefly carried a content-check
    EXCEPTION for `inferred_autumn_leaves.html`, the one baked chart with a
    bar-merge-suggestions overlay string-patched directly into its static
    HTML (`#suggest-mode-btn`/`.sugg-badge`) that hadn't been ported to the
    SPA's own JS-driven chart renderer yet. User complaint #2 ("Toujours
    l'interface dégueu", still landing on the ugly chooser page for THIS
    song) made clear that exception was no longer acceptable — the user
    wants no ugly page anywhere, full stop. Fix: the bar-merge-suggestions
    overlay is now ALSO implemented natively in `app_shell.html` (see its
    own "BAR-MERGE SUGGESTIONS" comment block — `S.suggMode`/
    `suggestToggleBtn`/`paintSuggestions`/`openSuggestionSheet`, same
    `/api/bar-merge-candidates/<file>` + `/api/reinfer/<file>` endpoints,
    same preview-only semantics), so every song's suggestions are reachable
    from the SPA now, not just this one manually-patched static file. The
    exception is gone: this route redirects UNCONDITIONALLY (after the
    existence check below) — no content sniffing, no per-file exemption.
    The baked-HTML rendering pipeline that used to run below this point
    (bar-1 offset re-derivation, PWA head injection, YouTube-overlay
    injection, swipe-nav) is now dead for every file and has been removed;
    if a future feature needs the standalone baked-HTML path again, restore
    it from git history (this docstring's prior revision) rather than
    re-adding a content-sniffed exception here."""
    p = PLOTS_DIR / filename
    if not p.exists() or not p.suffix == ".html":
        return "Not found", 404
    return redirect(f"/?open={quote(filename)}", code=302)


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


@app.route("/api/annotations/<filename>", methods=["POST"])
def post_annotations(filename):
    """Persist the annotation sidecar. The client posts the whole current
    doc (annotator name + chords + merges) on every change — last-write-
    wins, no merge/conflict logic (single annotator per song, decided).
    Dumb on purpose: no re-inference in the request path."""
    data = request.get_json(silent=True) or {}
    doc = {
        "annotator": data.get("annotator", ""),
        "chords": data.get("chords", []),
        "merges": data.get("merges", []),
    }
    saved = _remember_annotation(filename, doc)
    return jsonify(saved)


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


@app.route("/api/correction-log/<song>", methods=["POST"])
def api_correction_log(song):
    """Persist ONE human-correction record as training data.

    The annotator POSTs one of these per corrected chord after a save: the
    model's original prediction, the human fix, the /api/reinfer diff it
    produced, and a small benefit analysis (see the task schema). We write one
    JSON file per correction to data/training_logs/<song>/<ts>_<user>_bar<N>.json
    so the corpus can be swept offline (~600 labelled model errors over 20 songs).

    This route is deliberately forgiving: the annotation itself already saved via
    /api/annotations, so logging is pure bonus. On any error we return a JSON
    error the client logs to console and ignores — the save flow never blocks."""
    data = request.get_json(silent=True) or {}
    if not data:
        return jsonify(error="empty correction payload"), 400

    d = _training_log_dir(song)
    try:
        d.mkdir(parents=True, exist_ok=True)   # recursive: fixes the perms/first-run case
    except OSError as e:
        log.warning("correction-log: cannot create %s (%s)", d, e)
        return jsonify(error=f"mkdir failed: {e}"), 500

    # Canonicalise the fields the schema promises even if the client omitted them.
    ts = data.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    data["timestamp"] = ts
    data["song"] = re.sub(r"[^A-Za-z0-9_]", "", song or "") or "unknown"

    # Filename: <timestamp>_<username>_bar<corrected_bar>.json. Colons are illegal
    # on some filesystems, so ISO 8601's are swapped for dashes.
    fs_ts = ts.replace(":", "-")
    user = re.sub(r"[^A-Za-z0-9_-]", "", (data.get("human_session") or "anon"))[:32] or "anon"
    bar = (data.get("original_prediction") or {}).get("bar", "x")
    stem = f"{fs_ts}_{user}_bar{bar}"
    path = d / f"{stem}.json"
    if path.exists():
        # Timestamp collision (two corrections in the same second) — microsecond suffix.
        stem = f"{stem}_{int(time.time() * 1e6) % 1_000_000:06d}"
        path = d / f"{stem}.json"

    try:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as e:
        log.warning("correction-log: write failed %s (%s)", path, e)
        return jsonify(error=f"write failed: {e}"), 500

    log.info("correction-log %s: wrote %s (self_corrected=%s, propagation=%s)",
             song, path.name,
             (data.get("benefit") or {}).get("self_corrected"),
             (data.get("benefit") or {}).get("propagation_count"))
    return jsonify(ok=True, file=path.name, path=str(path))


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


@app.route("/api/tab-align", methods=["POST"])
def api_tab_align():
    """Fetch a UG tab, align it to a chart payload, return per-chord annotations.

    Body: {
        tab_url, song_name, artist_name, rating, votes, tonality,
        chart_chords: [ {root, lv: {seventh: {q, c}}} ]   ← P.chords from the viewer
    }
    Returns: {
        transpose_semitones, dtw_cost,
        annotations: [ {chord_idx, tab_chord, match, tab_conf_boost} ]
    }
    """
    data = request.get_json(silent=True) or {}
    tab_url     = (data.get("tab_url") or "").strip()
    song_name   = (data.get("song_name") or "").strip()
    artist_name = (data.get("artist_name") or "").strip()
    rating      = float(data.get("rating") or 0)
    votes       = int(data.get("votes") or 0)
    tonality    = (data.get("tonality") or "").strip()
    chart_chords = data.get("chart_chords") or []

    if not tab_url:
        return jsonify(error="No tab_url provided"), 400
    if not chart_chords:
        return jsonify(error="No chart_chords provided"), 400

    try:
        from harmonia.tab_fetcher import TabResult, fetch_tab_chords
        from harmonia.tab_aligner import align_tab_to_chart

        stub = TabResult(id=0, song_name=song_name, artist_name=artist_name,
                         tab_type="Chords", rating=rating, votes=votes,
                         tonality=tonality, difficulty="", tab_url=tab_url, score=0)
        tab = fetch_tab_chords(stub)
        if tab is None:
            return jsonify(error="Could not fetch tab content"), 502

        result = align_tab_to_chart(
            chart_chords, tab.chords, tab_rating=rating, tab_votes=votes
        )
    except ImportError as e:
        return jsonify(error=str(e)), 500
    except Exception as e:
        log.exception("tab-align failed")
        return jsonify(error=str(e)), 500

    return jsonify(
        transpose_semitones=result.transpose_semitones,
        dtw_cost=result.dtw_cost,
        annotations=[
            {
                "chord_idx":      a.chord_idx,
                "tab_chord":      a.tab_chord,
                "match":          a.match,
                "tab_conf_boost": a.tab_conf_boost,
            }
            for a in result.annotations
        ],
    )


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


@app.route("/api/irealb-compare", methods=["POST"])
def api_irealb_compare():
    """iReal Pro grid with inferred chords overlaid in each cell.

    The iReal chart is rendered in its standard bar-grid form.
    Each cell shows the GT chord (large) and the inferred chord (small, below),
    colored by match quality.  Both are in the same key — iReal labels are
    transposed to match the inferred chart's key.

    Body: {irealb_url, p_chords, bpm (opt), video_id (opt)}
    Returns: {url: "/chart/compare_<slug>.html", ...stats}
    """
    import json as _json, re as _re, urllib.parse as _up
    data       = request.get_json(silent=True) or {}
    irealb_url = (data.get("irealb_url") or "").strip()
    p_chords   = data.get("p_chords") or []
    bpm        = data.get("bpm")
    bpm        = float(bpm) if bpm else None
    vid        = (data.get("video_id") or "").strip()

    if not irealb_url or not p_chords:
        return jsonify(error="irealb_url and p_chords required"), 400

    try:
        from pyRealParser import Tune
        from harmonia.data.ireal_corpus import tune_to_mma
        from harmonia.irealb_aligner import align_irealb_to_inferred
        from harmonia.irealb_fetcher import _esc

        decoded = _up.unquote(irealb_url)
        tunes   = Tune.parse_ireal_url(decoded)
        if not tunes:
            return jsonify(error="No tunes found"), 400
        tune = tunes[0]
        mma  = tune_to_mma(tune, tempo=int(bpm) if bpm else None)
        result = align_irealb_to_inferred(mma, p_chords, bpm_override=bpm)

        # ── label helpers ──────────────────────────────────────────────
        flat_names = ["C","D♭","D","E♭","E","F","G♭","G","A♭","A","B♭","B"]
        QUAL_LABEL = {
            "min7":"m7","min":"m","dom7":"7","maj7":"maj7","maj":"",
            "hdim7":"ø7","dim7":"°7","aug":"+","minmaj7":"mM7","sus4":"sus4",
        }
        def inf_label_from_pc(pc, q):
            if q.startswith(":"): q = q[1:]
            if pc < 0: return "N"
            return flat_names[pc % 12] + QUAL_LABEL.get(q, q)

        # Build a fast lookup: given audio time t → inferred label
        # (the inferred chord whose [t0,t1) contains t)
        inf_sorted = []
        for c in p_chords:
            t0 = float(c.get("t0") or 0)
            t1 = float(c.get("t1") or t0 + 0.5)
            pc = c.get("root", -1)
            q  = c.get("lv", {}).get("seventh", {}).get("q", "")
            inf_sorted.append((t0, t1, inf_label_from_pc(pc, q)))

        def inferred_at(t):
            for t0, t1, lbl in inf_sorted:
                if t0 <= t < t1:
                    return lbl
            return ""

        # ── group iReal chords by bar ──────────────────────────────────
        bars = {}   # bar_no → list of result.chords entries
        for c in result.chords:
            b = c["bar"]
            bars.setdefault(b, []).append(c)
        bar_nos = sorted(bars)

        # ── determine section breaks ───────────────────────────────────
        bar_section = {}
        prev_sec = ""
        for b in bar_nos:
            sec = bars[b][0]["section"]
            bar_section[b] = sec if sec != prev_sec else ""
            prev_sec = sec

        # ── build grid HTML ───────────────────────────────────────────
        MATCH_BG  = {"exact":  "rgba(34,197,94,.18)",
                     "family": "rgba(245,158,11,.18)",
                     "mismatch":"rgba(239,68,68,.18)",
                     "gap":    "rgba(148,163,184,.12)"}
        MATCH_BDR = {"exact":  "#16a34a", "family": "#d97706",
                     "mismatch":"#dc2626", "gap":    "#94a3b8"}

        grid_html = ""
        bars_per_row = 4
        for row_start in range(0, len(bar_nos), bars_per_row):
            row_bars = bar_nos[row_start : row_start + bars_per_row]

            # Section label for this row
            sec = bar_section.get(row_bars[0], "")
            sec_html = (f'<div class="sec-lbl"><span>{_esc(sec)}</span></div>'
                        if sec else '<div class="sec-lbl"></div>')

            row_html = f'<div class="ir-row">{sec_html}'
            for b in row_bars:
                chords_in_bar = bars[b]
                # bar number label
                row_html += f'<div class="ir-bar" data-bar="{b}">'
                row_html += f'<span class="bar-no">{b+1}</span>'
                for c in chords_in_bar:
                    t0 = c["t0"]
                    t1 = c["t1"]
                    m  = c["match"]
                    bg  = MATCH_BG[m]
                    bdr = MATCH_BDR[m]
                    # find what the model inferred at the midpoint of this chord
                    inf_lbl = inferred_at((t0 + t1) / 2) if t0 is not None else ""
                    cell_id = f'chord-{result.chords.index(c)}'
                    row_html += (
                        f'<div class="ir-cell m-{m}" id="{cell_id}" '
                        f'data-t0="{t0}" data-t1="{t1}" '
                        f'style="background:{bg};border-color:{bdr};" '
                        f'title="{_esc(c["label"])} vs {_esc(inf_lbl)} [{m}] {t0:.1f}s">'
                        f'<span class="gt">{_esc(c["label"])}</span>'
                        f'<span class="inf-lbl">{_esc(inf_lbl)}</span>'
                        f'</div>'
                    )
                row_html += '</div>'
            # pad empty bars
            for _ in range(bars_per_row - len(row_bars)):
                row_html += '<div class="ir-bar ir-empty"></div>'
            row_html += '</div>'
            grid_html += row_html

        # ── stats banner ───────────────────────────────────────────────
        ts = mma.time_signature or (4, 4)
        stats_html = (
            f'<span class="si">Key {_esc(mma.key or "?")}  ·  {mma.tempo} BPM  ·  {ts[0]}/{ts[1]}</span>'
            f'<span class="si">transpose +{result.transpose_semitones} st</span>'
            f'<span class="si">{result.n_repeats}× form</span>'
            f'<span class="sok">■ exact {result.exact_frac:.0%}</span>'
            f'<span class="sfam">■ family {result.family_frac:.0%}</span>'
            f'<span class="smiss">■ mismatch {result.mismatch_frac:.0%}</span>'
        )

        # ── legend ────────────────────────────────────────────────────
        legend_html = (
            '<div class="legend">'
            '<span class="leg-item"><span class="leg-swatch" style="background:rgba(34,197,94,.25);border-color:#16a34a"></span>exact root + quality family</span>'
            '<span class="leg-item"><span class="leg-swatch" style="background:rgba(245,158,11,.25);border-color:#d97706"></span>family match (root ok, quality differs)</span>'
            '<span class="leg-item"><span class="leg-swatch" style="background:rgba(239,68,68,.25);border-color:#dc2626"></span>root mismatch</span>'
            '<span class="leg-lbl" style="font-size:11px;color:#4a4636;font-family:system-ui,sans-serif">'
            'Each cell: <b>GT</b> above · <i>inferred</i> below</span>'
            '</div>'
        )

        # ── YouTube ────────────────────────────────────────────────────
        p_json = _json.dumps({"chords": result.chords, "tempo": mma.tempo})
        yt_dock = yt_script = ""
        if vid:
            yt_dock = (
                '<div id="yt-dock">'
                '<div id="yt-pw"><div id="yt-player"></div></div>'
                '<div id="yt-ctrl">'
                '<button onclick="ytToggle()">▶ / ⏸</button>'
                '<span id="yt-t">0:00</span>'
                '</div></div>'
            )
            yt_script = f"""
let _yp=null,_yr=false,_rf=null;
function ytToggle(){{if(!_yr)return;const s=_yp.getPlayerState();if(s===1)_yp.pauseVideo();else _yp.playVideo();}}
function onYouTubeIframeAPIReady(){{
  _yp=new YT.Player('yt-player',{{videoId:{_json.dumps(vid)},
    playerVars:{{origin:window.location.origin,controls:1}},
    events:{{onReady:()=>{{_yr=true;go();}},onStateChange:e=>{{if(e.data===1)go();else stop();}}}}
  }});
}}
function go(){{stop();_rf=requestAnimationFrame(loop);}}
function stop(){{if(_rf){{cancelAnimationFrame(_rf);_rf=null;}}}}
function loop(){{
  if(!_yr)return;
  const t=_yp.getCurrentTime();
  document.getElementById('yt-t').textContent=Math.floor(t/60)+':'+String(Math.floor(t%60)).padStart(2,'0');
  highlightAt(t);
  _rf=requestAnimationFrame(loop);
}}
const _s=document.createElement('script');
_s.src='https://www.youtube.com/iframe_api';document.head.appendChild(_s);
"""

        html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(tune.title)} — GT vs Inferred</title>
<style>
:root{{--paper:#f7f3e9;--ink:#1c1c1c;--rule:#b9b09a;--acc:#8a2b2b;--faint:#8a8371;}}
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:var(--paper);color:var(--ink);font-family:Georgia,serif;padding-bottom:160px;}}
.page{{max-width:960px;margin:0 auto;padding:22px 22px 0;}}
h1{{font-size:22px;margin-bottom:4px;}}
.meta{{font-size:12px;color:var(--faint);font-style:italic;margin-bottom:14px;}}
/* stats */
.stats{{display:flex;gap:10px;flex-wrap:wrap;font-family:system-ui,sans-serif;
        font-size:12px;margin-bottom:14px;padding:7px 12px;
        background:#efe9d9;border-radius:7px;align-items:center;}}
.sok{{color:#166534;font-weight:600;}}.sfam{{color:#92400e;font-weight:600;}}
.smiss{{color:#991b1b;font-weight:600;}}.si{{color:#4a4636;}}
/* legend */
.legend{{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-bottom:16px;
         font-family:system-ui,sans-serif;font-size:11px;color:#4a4636;}}
.leg-item{{display:flex;align-items:center;gap:5px;}}
.leg-swatch{{display:inline-block;width:14px;height:14px;border:1.5px solid;border-radius:3px;}}
/* grid */
.ir-grid{{display:flex;flex-direction:column;gap:3px;
          border-top:2.5px solid var(--acc);border-bottom:2.5px solid var(--acc);padding:8px 0;}}
.ir-row{{display:grid;grid-template-columns:26px repeat(4,1fr);gap:3px;}}
.sec-lbl{{display:flex;align-items:flex-start;justify-content:center;padding-top:6px;}}
.sec-lbl span{{font-family:system-ui,sans-serif;font-size:10px;font-weight:700;
   color:var(--acc);border:1.5px solid var(--acc);border-radius:3px;padding:1px 3px;}}
.ir-bar{{border:1px solid var(--rule);border-radius:3px;background:#fff;
          min-height:62px;position:relative;display:flex;flex-wrap:wrap;
          align-items:stretch;overflow:hidden;}}
.ir-empty{{border:1px dashed #e0d8c0;background:transparent;}}
.bar-no{{position:absolute;top:2px;left:3px;font-family:system-ui,sans-serif;
          font-size:8px;color:#b0a89a;line-height:1;}}
/* chord cells */
.ir-cell{{flex:1 1 40%;display:flex;flex-direction:column;align-items:center;
           justify-content:center;border:1.5px solid transparent;border-radius:2px;
           padding:14px 2px 4px;min-width:30px;cursor:default;transition:filter .08s;}}
.ir-cell+.ir-cell{{border-left:1px solid var(--rule);}}
.gt{{font-family:'Menlo','Courier New',monospace;font-size:13px;font-weight:600;
     color:var(--ink);line-height:1.2;}}
.inf-lbl{{font-family:'Menlo','Courier New',monospace;font-size:10px;font-weight:400;
           color:var(--faint);line-height:1.2;margin-top:2px;font-style:italic;}}
.ir-cell.now{{filter:brightness(1.06);outline:2px solid #555;outline-offset:1px;}}
/* youtube */
#yt-dock{{position:fixed;bottom:0;left:0;right:0;background:#111;
           display:flex;align-items:center;gap:12px;padding:8px 20px;z-index:200;}}
#yt-pw{{width:200px;height:113px;flex:0 0 200px;}}
#yt-player{{width:200px;height:113px;}}
#yt-ctrl{{color:#fff;font-family:system-ui,sans-serif;font-size:13px;
           display:flex;align-items:center;gap:10px;}}
#yt-ctrl button{{background:#333;color:#fff;border:none;border-radius:4px;
                  padding:4px 12px;cursor:pointer;}}
@media(prefers-color-scheme:dark){{
  :root{{--paper:#1c1a16;--ink:#f0ebe0;--rule:#3a3628;}}
  .ir-bar{{background:#252218;}}
  .stats{{background:#272420;}}
  .inf-lbl{{color:#8a8073;}}
}}
</style>
</head><body>
<div class="page">
  <h1>{_esc(tune.title)}</h1>
  <p class="meta">{_esc(tune.composer or "")} — GT chord chart with inferred chords overlaid</p>
  <div class="stats">{stats_html}</div>
  {legend_html}
  <div class="ir-grid">{grid_html}</div>
</div>
{yt_dock}
<script>
const P={p_json};
function highlightAt(t){{
  document.querySelectorAll('.ir-cell.now').forEach(e=>e.classList.remove('now'));
  document.querySelectorAll('.ir-cell').forEach(e=>{{
    const t0=parseFloat(e.dataset.t0),t1=parseFloat(e.dataset.t1);
    if(t0!=null&&!isNaN(t0)&&t>=t0&&t<t1) e.classList.add('now');
  }});
}}
{yt_script}
</script>
</body></html>"""

    except Exception as e:
        log.exception("irealb-compare failed")
        return jsonify(error=str(e)), 500

    try:
        slug_raw = _up.unquote(irealb_url).split("=")[0].replace("irealb://", "")
        slug = re.sub(r"[^a-z0-9]+", "_", slug_raw.lower()).strip("_") or "irealb"
    except Exception:
        slug = "irealb"

    out = PLOTS_DIR / f"compare_{slug[:60]}.html"
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
    )


@app.route("/api/irealb-search", methods=["POST"])
def api_irealb_search():
    """Search iReal Pro community for songs.

    Body: {title, artist}
    Returns: {results: [{title, composer, key, style, time_sig, irealb_url}]}
    """
    data   = request.get_json(silent=True) or {}
    title  = (data.get("title")  or "").strip()
    artist = (data.get("artist") or "").strip()
    if not title:
        return jsonify(error="No title provided"), 400
    query = f"{title} {artist}".strip()
    try:
        from harmonia.irealb_fetcher import search_community
        results = search_community(query)
    except Exception as e:
        log.exception("irealb-search failed")
        return jsonify(error=str(e)), 500
    return jsonify(results=results)


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


def _run_analysis(job_id: str, url: str, seg_source_override: str | None = None) -> None:
    def update(status, message="", **kw):
        with _jobs_lock:
            _jobs[job_id].update(status=status, message=message, **kw)

    # `stage` + `results` drive the app's Analysing screen. They are the real
    # steps, not a scripted animation: stage 1 is one call into infer_chords_v1
    # (Basic Pitch → beats → sections → key → chord HMM), which reports nothing
    # intermediate, so it stays lit for as long as the decode actually takes and
    # its result chip is filled from what the decode returned.
    results = ["downloading from YouTube", "notes, beats, sections, key, chords",
               "laying out the lead sheet"]

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
                    cache_dir=Path(_ARGS.cache_dir),
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
                    audio_path, cache_dir=Path(_ARGS.cache_dir),
                )
                backend_used = "billboard_bp48_60_rollaug_v1"
            except RuntimeError as e:
                log.warning("billboard backend unavailable (%s) — falling back to infer_chords_v1", e)
                pipeline_chart = infer_chords_v1(
                    audio_path,
                    cache_dir=Path(_ARGS.cache_dir),
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
                    activations = PitchExtractor(cache_dir=Path(_ARGS.cache_dir)).extract(audio_path)
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

LIBRARY_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Harmonia — your charts</title>
<style>
  :root { --paper:#f7f3e9; --ink:#1c1c1c; --rule:#b9b09a; --accent:#8a2b2b; --faint:#8a8371; }
  * { box-sizing:border-box; }
  body { background:var(--paper); color:var(--ink); margin:0;
         font-family:Georgia,'Times New Roman',serif; }
  .wrap { max-width:560px; margin:0 auto; padding:48px 32px; }

  .topline { display:flex; align-items:center; justify-content:space-between; margin-bottom:26px; }
  .topline h1 { font-size:26px; margin:0; }
  .back-pill { display:inline-flex; align-items:center; gap:6px; background:#efe9d9;
               border:1px solid #e2dac4; border-radius:20px; padding:9px 15px;
               min-height:40px; box-sizing:border-box; text-decoration:none;
               font:700 13px system-ui,sans-serif; color:#4a4636; flex:0 0 auto;
               transition:transform .1s ease, background .12s; }
  .back-pill:active { transform:scale(.95); background:#e2d9c2; }

  .section-label { font:700 11px system-ui,sans-serif; text-transform:uppercase;
                    letter-spacing:.06em; color:var(--faint); margin:0 0 8px 4px; }

  .chart-card { background:#efe9d9; border:1px solid #e2dac4; border-radius:14px;
                overflow:hidden; box-shadow:0 1px 3px #0001; }
  .chart-row { display:flex; align-items:center; gap:10px; padding:15px 18px;
               text-decoration:none; color:var(--ink); transition:background .12s; }
  .chart-row + .chart-row { border-top:1px solid #e2dac4; }
  .chart-row:hover, .chart-row:active { background:#e2d9c2; }
  .chart-row .name { flex:1; font-size:17px; min-width:0; overflow:hidden;
                      text-overflow:ellipsis; white-space:nowrap; }
  .chart-row .chev { color:#b9ac95; font-size:16px; font-family:system-ui,sans-serif; }
  .empty-state { padding:26px 18px; text-align:center; color:var(--faint);
                 font-style:italic; font-size:14px; }

  @media (max-width: 640px) {
    body { overscroll-behavior-y:none; -webkit-overflow-scrolling:touch; }
    .wrap { padding:calc(24px + env(safe-area-inset-top)) 18px
                    calc(28px + env(safe-area-inset-bottom)); }
    .topline h1 { font-size:22px; }
    .chart-row { padding:16px 18px; font-size:18px; }
  }
</style>
</head><body>
<div class="wrap">
  <div class="topline">
    <h1>Your charts</h1>
    <a class="back-pill" href="/">🔍 Search</a>
  </div>

  <p class="section-label">{% if charts %}{{ charts|length }} chart{{ 's' if charts|length != 1 else '' }}{% else %}Nothing yet{% endif %}</p>
  <div class="chart-card">
  {% for c in charts %}
    <a class="chart-row" href="/chart/{{ c.file }}"><span class="name">{{ c.name }}</span><span class="chev">›</span></a>
  {% else %}
    <div class="empty-state">No charts yet — search for a song to get started.</div>
  {% endfor %}
  </div>
</div>
</body></html>"""


HOME_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Harmonia</title>
<style>
  :root { --paper:#f7f3e9; --ink:#1c1c1c; --rule:#b9b09a; --accent:#8a2b2b; --faint:#8a8371; }
  * { box-sizing:border-box; }
  body { background:var(--paper); color:var(--ink); margin:0;
         font-family:Georgia,'Times New Roman',serif; }
  .wrap { max-width:560px; margin:0 auto; padding:48px 32px; }

  .brand { display:flex; align-items:center; justify-content:space-between; margin-bottom:30px; gap:10px; }
  .brand-id { display:flex; align-items:center; gap:12px; min-width:0; }
  .brand-mark { width:42px; height:42px; border-radius:12px; background:var(--paper);
                border:1.5px solid #e2dac4;
                color:var(--ink); font:italic 700 24px Georgia,'Times New Roman',serif;
                flex:0 0 auto; display:flex; align-items:center; justify-content:center;
                box-shadow:0 2px 6px #0001; }
  .brand h1 { font-size:23px; margin:0; }
  .lib-pill { display:inline-flex; align-items:center; gap:6px; background:#efe9d9;
              border:1px solid #e2dac4; border-radius:20px; padding:9px 15px;
              min-height:40px; box-sizing:border-box; text-decoration:none; white-space:nowrap;
              font:700 13px system-ui,sans-serif; color:#4a4636; flex:0 0 auto;
              transition:transform .1s ease, background .12s; }
  .lib-pill:active { transform:scale(.95); background:#e2d9c2; }

  h2.headline { font-size:25px; margin:0 0 6px; }
  .tagline { color:var(--faint); font-style:italic; font-size:14px; margin:0 0 20px; }

  .search-row { display:flex; gap:10px; margin-bottom:8px; }
  #q { flex:1; min-width:0; padding:13px 16px; border:1.5px solid #cfc7ae; border-radius:12px;
       font:16px system-ui,sans-serif; background:#fff; }
  #q:focus { outline:none; border-color:var(--accent); }
  #search-go { padding:0 20px; background:var(--accent); color:#fff; border:none;
               border-radius:12px; font:700 14px system-ui,sans-serif; cursor:pointer;
               transition:background .12s, transform .1s ease; }
  #search-go:active { transform:scale(.96); }
  #search-hint { font-size:12px; color:var(--faint); margin:0 0 22px; font-family:system-ui,sans-serif; }
  #search-hint a { color:var(--accent); }

  #results { display:flex; flex-direction:column; gap:2px; }
  .yt-card { display:flex; gap:12px; align-items:center; padding:10px;
             border-radius:12px; cursor:pointer; text-align:left; border:none;
             background:transparent; width:100%; font-family:system-ui,sans-serif;
             transition:background .12s; }
  .yt-card:active { background:#e2d9c2; }
  .yt-card img { width:96px; height:54px; border-radius:8px; object-fit:cover;
                 background:#e2dac4; flex:0 0 auto; }
  .yt-card .meta { min-width:0; flex:1; }
  .yt-card .title { font:600 14px/1.3 system-ui,sans-serif; color:var(--ink);
                     display:-webkit-box; -webkit-line-clamp:2; -webkit-box-orient:vertical;
                     overflow:hidden; }
  .yt-card .sub { font-size:12px; color:var(--faint); margin-top:3px; }

  #status { font-size:13px; color:#4a4636; margin:2px 0 14px; min-height:18px;
            font-family:system-ui,sans-serif; }
  #status.err { color:var(--accent); }
  .empty-hint { color:var(--faint); font-style:italic; font-size:14px; text-align:center;
                padding:20px 0; font-family:Georgia,serif; }

  @media (max-width: 640px) {
    body { overscroll-behavior-y:none; -webkit-overflow-scrolling:touch; }
    .wrap { padding:calc(24px + env(safe-area-inset-top)) 18px
                    calc(28px + env(safe-area-inset-bottom)); }
    h2.headline { font-size:22px; }
    #q { font-size:16px; padding:14px 16px; }
  }
</style>
</head><body>
<div class="wrap">
  <div class="brand">
    <div class="brand-id">
      <div class="brand-mark">h</div>
      <h1>Harmonia</h1>
    </div>
    <a class="lib-pill" href="/library">Your charts{% if n_charts %} · {{ n_charts }}{% endif %}</a>
  </div>

  <h2 class="headline">Find a song</h2>
  <p class="tagline">Search YouTube — Harmonia downloads it and infers the chords.</p>

  <div class="search-row">
    <input id="q" type="search" placeholder="Song title, artist…" autocomplete="off"
           onkeydown="if(event.key==='Enter'){event.preventDefault();doSearch();}">
    <button id="search-go" type="button" onclick="doSearch()">Search</button>
  </div>
  <p id="search-hint">Have a link already? <a href="#" id="paste-link">Paste a YouTube URL instead</a></p>

  <div id="status"></div>
  <div id="results"></div>
</div>
<script>
function setStatus(msg,cls){const s=document.getElementById('status');s.textContent=msg;s.className=cls||'';}

// ── whimsical progress jargon, à la Claude Code's own spinner — music
// theory and ML buzzwords mashed together, cycling while we wait ──
const JARGON=["Reticulating the circle of fifths…","Debiasing the backbeat…",
  "Quantizing the swing feel…","Diagonalizing the ii–V–I…","Convolving with the changes…",
  "Untangling the voice leading…","Backpropagating through the bridge…",
  "Softmaxing the ambiguity…","Cross-validating the chorus…","Warming up the pitch detector…",
  "Aligning downbeats to reality…","Resolving enharmonic spellings…",
  "Bootstrapping the groove prior…","Denoising the chroma…","Tokenizing the tritone sub…",
  "Interrogating the bassline…","Regularizing the rubato…","Vectorizing the vamp…",
  "Annealing the modulation…","Gradient-descending the changes…"];
let _jargonTimer=null, _jargonIdx=0;
function jargonStart(){
  clearInterval(_jargonTimer);
  _jargonIdx=Math.floor(Math.random()*JARGON.length);
  setStatus(JARGON[_jargonIdx],'');
  _jargonTimer=setInterval(()=>{
    _jargonIdx=(_jargonIdx+1)%JARGON.length;
    setStatus(JARGON[_jargonIdx],'');
  },1600);
}
function jargonStop(){ clearInterval(_jargonTimer); _jargonTimer=null; }

function poll(jobId){
  fetch('/api/job/'+jobId).then(r=>r.json()).then(d=>{
    if(d.status==='done'){ jargonStop(); window.location.href=d.url; }
    else if(d.status==='error'){ jargonStop(); setStatus(d.error||'Failed.','err'); }
    else { setTimeout(()=>poll(jobId),1500); }
  }).catch(()=>{ setTimeout(()=>poll(jobId),2500); });
}
function startAnalysis(url){
  document.getElementById('results').innerHTML='';
  jargonStart();
  fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url})})
    .then(r=>r.json())
    .then(d=>{
      if(d.error){jargonStop();setStatus(d.error,'err');return;}
      poll(d.job_id);
    });
}
document.getElementById('paste-link').addEventListener('click',e=>{
  e.preventDefault();
  const url=prompt('Paste a YouTube URL:');
  if(url && url.trim()) startAnalysis(url.trim());
});
function renderResults(results){
  const box=document.getElementById('results');
  box.innerHTML='';
  if(!results.length){
    box.innerHTML='<div class="empty-hint">No results — try a different search.</div>';
    return;
  }
  results.forEach(r=>{
    const btn=document.createElement('button');
    btn.type='button'; btn.className='yt-card';
    const img=document.createElement('img');
    img.src=r.thumb; img.loading='lazy'; img.alt='';
    const meta=document.createElement('div'); meta.className='meta';
    const title=document.createElement('div'); title.className='title'; title.textContent=r.title;
    const sub=document.createElement('div'); sub.className='sub';
    const mins=Math.floor((r.duration||0)/60), secs=String((r.duration||0)%60).padStart(2,'0');
    sub.textContent=[r.uploader, r.duration ? (mins+':'+secs) : null].filter(Boolean).join(' · ');
    meta.appendChild(title); meta.appendChild(sub);
    btn.appendChild(img); btn.appendChild(meta);
    btn.addEventListener('click',()=>startAnalysis('https://youtu.be/'+r.id));
    box.appendChild(btn);
  });
}
function doSearch(){
  const q=document.getElementById('q').value.trim();
  if(!q) return;
  document.getElementById('results').innerHTML='';
  jargonStart();
  fetch('/api/yt-search',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({q})})
    .then(r=>r.json())
    .then(d=>{
      jargonStop();
      if(d.error){ setStatus(d.error,'err'); return; }
      setStatus('','');
      renderResults(d.results||[]);
    })
    .catch(()=>{ jargonStop(); setStatus('Search failed — check your connection.','err'); });
}
</script>
</body></html>"""


ANNOTATOR_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no,viewport-fit=cover">
<title>Harmonia — Waveform Editor</title>
<style>
  :root{
    --bg:#0e1116; --panel:#171c24; --panel2:#1e2530; --ink:#e8edf4; --faint:#8b97a8;
    --line:#2a3340; --teal:#00c9a7; --teal-dim:#0b3d35; --amber:#ffb454; --accent:#6ea8ff;
    --danger:#ff5d6c; --ok:#37d67a; --wf:#4b5666; --downbeat:#cfd8e6;
  }
  *{box-sizing:border-box;-webkit-tap-highlight-color:transparent;}
  html,body{margin:0;background:var(--bg);color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,'SF Pro Text',system-ui,sans-serif;
    overscroll-behavior:none;overflow-x:hidden;max-width:100%;}
  body{padding-bottom:calc(84px + env(safe-area-inset-bottom));}
  a{color:var(--accent);}
  .top{position:sticky;top:0;z-index:20;background:linear-gradient(180deg,#0e1116 78%,#0e1116cc);
    padding:calc(8px + env(safe-area-inset-top)) 12px 8px;border-bottom:1px solid var(--line);}
  .toprow{display:flex;align-items:center;gap:10px;}
  .toprow h1{font-size:16px;margin:0;font-weight:700;flex:1;letter-spacing:.2px;
    overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
  .back{font:600 15px system-ui;color:var(--faint);text-decoration:none;padding:6px 8px;margin-left:-8px;}
  .who{background:var(--panel2);border:1px solid var(--line);color:var(--ink);
    border-radius:8px;padding:6px 8px;font:600 12px system-ui;width:92px;}
  .sub{display:flex;align-items:center;gap:6px;margin-top:6px;font:600 11px system-ui;color:var(--faint);flex-wrap:wrap;}
  .pill{background:var(--panel2);border:1px solid var(--line);border-radius:20px;padding:3px 9px;}
  .pill.src{color:var(--teal);border-color:var(--teal-dim);}
  /* transport */
  .transport{display:flex;align-items:center;gap:10px;padding:8px 12px 4px;}
  .playbtn{width:48px;height:48px;flex:none;border:none;border-radius:50%;background:var(--teal);
    color:#062;font-size:20px;display:flex;align-items:center;justify-content:center;}
  .playbtn:active{transform:scale(.94);}
  .clock{font:700 13px ui-monospace,Menlo,monospace;color:var(--ink);min-width:96px;}
  .clock .d{color:var(--faint);}
  .zoombtns{margin-left:auto;display:flex;gap:6px;}
  .zoombtns button{width:34px;height:34px;border:1px solid var(--line);background:var(--panel2);
    color:var(--ink);border-radius:9px;font:700 15px ui-monospace;}
  /* timeline */
  .tlwrap{position:relative;margin:2px 0 0;background:var(--panel);border-top:1px solid var(--line);
    border-bottom:1px solid var(--line);overflow:hidden;}
  .tlscroll{overflow-x:auto;overflow-y:hidden;-webkit-overflow-scrolling:touch;touch-action:pan-x;}
  .timeline{position:relative;height:300px;touch-action:none;}
  canvas#wf{position:absolute;left:0;top:0;z-index:0;display:block;}
  #beatlayer,#chordlayer{position:absolute;left:0;top:0;height:100%;z-index:1;pointer-events:none;}
  #chordlayer{z-index:2;}
  .beat{position:absolute;top:60px;bottom:0;width:22px;margin-left:-11px;pointer-events:auto;
    touch-action:none;cursor:ew-resize;z-index:1;}
  .beat i{position:absolute;left:10px;top:0;bottom:0;width:2px;background:var(--teal);opacity:.55;}
  .beat.db i{width:3px;left:9px;background:var(--downbeat);opacity:.85;}
  .beat.drag i{opacity:1;box-shadow:0 0 6px var(--teal);}
  .beat b{position:absolute;left:2px;top:2px;font:700 9px ui-monospace;color:var(--faint);}
  .beat.db b{color:var(--downbeat);}
  .chordbar{position:absolute;top:18px;height:40px;border-radius:8px;pointer-events:auto;
    display:flex;align-items:center;justify-content:center;overflow:hidden;
    border:1px solid #0006;box-shadow:0 1px 3px #0006;touch-action:none;}
  .chordbar span{font:700 12px 'SF Mono',ui-monospace,Menlo,monospace;color:#08110e;
    padding:0 12px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;pointer-events:none;
    text-shadow:0 1px 0 #fff5;}
  .chordbar.dirty{outline:2px solid var(--amber);outline-offset:-2px;}
  .chordbar .edge{position:absolute;top:0;bottom:0;width:16px;pointer-events:auto;touch-action:none;
    cursor:ew-resize;display:flex;align-items:center;justify-content:center;}
  .chordbar .edge.l{left:-2px;} .chordbar .edge.r{right:-2px;}
  .chordbar .edge::after{content:"";width:3px;height:22px;border-radius:2px;background:#0009;}
  .chordbar .edge:active::after{background:#000;}
  #playhead{position:absolute;top:0;bottom:0;width:2px;background:var(--danger);z-index:5;
    pointer-events:none;box-shadow:0 0 6px var(--danger);will-change:transform;}
  #playhead::before{content:"";position:absolute;top:0;left:-5px;width:12px;height:12px;
    background:var(--danger);border-radius:0 0 50% 50%;box-shadow:0 1px 4px #000a;}
  /* band labels down the left, over the scroller */
  .bandlabels{position:absolute;left:0;top:0;bottom:0;width:0;z-index:6;pointer-events:none;}
  .bandlabels span{position:absolute;left:4px;font:700 8px system-ui;color:#ffffff99;
    background:#0009;padding:1px 4px;border-radius:4px;letter-spacing:.4px;}
  /* hints */
  .hint{color:var(--faint);font:500 11.5px system-ui;padding:8px 12px 2px;line-height:1.5;}
  .hint b{color:var(--ink);}
  .legend{display:flex;gap:12px;padding:2px 12px 8px;font:600 10px system-ui;color:var(--faint);flex-wrap:wrap;}
  .legend i{display:inline-block;width:10px;height:10px;border-radius:3px;margin-right:4px;vertical-align:-1px;}
  /* save bar */
  .savebar{position:fixed;left:0;right:0;bottom:0;z-index:30;
    padding:10px 12px calc(10px + env(safe-area-inset-bottom));
    background:linear-gradient(0deg,#0e1116 65%,#0e1116cc);border-top:1px solid var(--line);
    display:flex;gap:10px;align-items:center;}
  .save{flex:1;min-height:52px;border:none;border-radius:14px;background:var(--teal);color:#062;
    font:800 16px system-ui;display:flex;align-items:center;justify-content:center;gap:8px;}
  .save:active{transform:scale(.98);} .save[disabled]{opacity:.5;}
  .stat{font:700 12px system-ui;color:var(--faint);min-width:64px;text-align:right;}
  /* toast */
  .toast{position:fixed;left:50%;bottom:96px;transform:translateX(-50%) translateY(20px);
    background:var(--ok);color:#062;font:800 13px system-ui;padding:10px 18px;border-radius:24px;
    opacity:0;transition:.25s;z-index:40;box-shadow:0 6px 20px #0008;pointer-events:none;max-width:92vw;text-align:center;}
  .toast.on{opacity:1;transform:translateX(-50%) translateY(0);}
  .toast.err{background:var(--danger);color:#fff;}
  /* chord editor modal */
  .modal{position:fixed;inset:0;z-index:50;background:#0009;display:none;align-items:flex-end;}
  .modal.on{display:flex;}
  .sheet{width:100%;background:var(--panel);border-top-left-radius:18px;border-top-right-radius:18px;
    border-top:1px solid var(--line);padding:14px 14px calc(16px + env(safe-area-inset-bottom));
    max-height:80vh;overflow:auto;}
  .sheet h3{margin:0 0 4px;font:800 15px system-ui;}
  .sheet .prev{font:800 22px 'SF Mono',ui-monospace,Menlo,monospace;color:var(--teal);margin:2px 0 12px;}
  .sheet .grp{font:700 10px system-ui;color:var(--faint);text-transform:uppercase;letter-spacing:.6px;margin:10px 0 6px;}
  .keys{display:grid;grid-template-columns:repeat(6,1fr);gap:6px;}
  .keys.q{grid-template-columns:repeat(4,1fr);}
  .keys button{min-height:44px;border:1px solid var(--line);background:var(--panel2);color:var(--ink);
    border-radius:10px;font:700 13px ui-monospace;}
  .keys button.on{background:var(--teal);color:#062;border-color:var(--teal);}
  .sheet .row{display:flex;gap:10px;margin-top:14px;}
  .sheet .row button{flex:1;min-height:48px;border-radius:12px;font:800 14px system-ui;border:1px solid var(--line);
    background:var(--panel2);color:var(--ink);}
  .sheet .row button.apply{background:var(--teal);color:#062;border-color:var(--teal);}
  .sheet .row button.del{color:var(--danger);}
  /* selection (keyboard focus + shift-click multi-select for merge) */
  .chordbar.sel{outline:2px solid var(--accent);outline-offset:-2px;z-index:3;}
  .chordbar.msel{outline:2px solid var(--amber);outline-offset:-2px;box-shadow:0 0 10px #ffb45480;}
  /* peak-snap cue: vertical line over the target waveform peak during a drag */
  #peakcue{position:absolute;top:60px;bottom:0;width:2px;margin-left:-1px;background:var(--accent);
    opacity:0;z-index:4;pointer-events:none;box-shadow:0 0 6px var(--accent);transition:opacity .08s;}
  #peakcue.on{opacity:.9;}
  /* beat-sync overlay dots (drift of each chord boundary vs nearest beat) */
  #synclayer{position:absolute;left:0;top:0;height:100%;z-index:3;pointer-events:none;}
  .syncdot{position:absolute;top:4px;width:9px;height:9px;border-radius:50%;margin-left:-4.5px;
    border:1px solid #0008;box-shadow:0 0 4px #000a;}
  /* tools row (peak-snap toggle · beat-sync overlay · merge) */
  .tools{display:flex;gap:8px;padding:2px 12px 6px;flex-wrap:wrap;align-items:center;}
  .tools button{border:1px solid var(--line);background:var(--panel2);color:var(--ink);
    border-radius:9px;padding:7px 11px;font:700 11.5px system-ui;display:flex;align-items:center;gap:5px;}
  .tools button.on{background:var(--teal-dim);border-color:var(--teal);color:var(--teal);}
  .tools button.merge{background:var(--panel2);}
  .tools button.merge.ready{background:var(--amber);color:#3a2600;border-color:var(--amber);}
  .tools .kbd{margin-left:auto;color:var(--faint);font:600 10px ui-monospace;}
</style>
</head><body>
<div class="top">
  <div class="toprow">
    <a class="back" href="/library">&larr;</a>
    <h1 id="ttl">Waveform Editor</h1>
    <input class="who" id="who" placeholder="your name" autocomplete="off">
  </div>
  <div class="sub">
    <span class="pill" id="nchords">0 chords</span>
    <span class="pill src" id="gridsrc">grid</span>
    <span class="pill" id="tempo">— bpm</span>
    <span class="pill" id="dur">0:00</span>
  </div>
</div>

<div class="transport">
  <button class="playbtn" id="play" aria-label="play">&#9654;</button>
  <div class="clock"><span id="cur">0:00.0</span><span class="d"> / <span id="tot">0:00.0</span></span></div>
  <div class="zoombtns">
    <button id="zout" aria-label="zoom out">&minus;</button>
    <button id="zin" aria-label="zoom in">+</button>
  </div>
</div>

<div class="tlwrap">
  <div class="tlscroll" id="scroll">
    <div class="timeline" id="timeline">
      <canvas id="wf"></canvas>
      <div id="beatlayer"></div>
      <div id="synclayer"></div>
      <div id="chordlayer"></div>
      <div id="peakcue"></div>
      <div id="playhead" style="transform:translateX(0)"></div>
    </div>
  </div>
  <div class="bandlabels">
    <span style="top:2px">+ ADD</span>
    <span style="top:22px">CHORDS</span>
    <span style="top:62px">BEATS</span>
    <span style="top:150px">WAVEFORM</span>
  </div>
</div>

<p class="hint"><b>Drag chord edges</b> to fine-tune spans &middot; <b>tap a chord</b> to relabel &middot;
<b>tap the + lane</b> (top) to add a boundary &middot; <b>drag teal beats</b> to fix alignment &middot;
<b>tap the waveform</b> to seek. &middot; <b>Shift-tap</b> chords to select for merge.</p>
<div class="tools">
  <button id="tmerge" class="merge">&#9776; Merge</button>
  <button id="tsync">&#9679; Beat-sync</button>
  <button id="tpeak" class="on">&#9650; Peak-snap</button>
  <span class="kbd">&larr;&rarr; nudge &middot; &#8679;+arrow coarse &middot; ⌘Z undo &middot; ⌘S save &middot; Space play &middot; Tab next</span>
</div>
<div class="legend">
  <span><i style="background:hsl(130 60% 45%)"></i>high conf</span>
  <span><i style="background:hsl(60 70% 50%)"></i>medium</span>
  <span><i style="background:hsl(0 70% 55%)"></i>low conf</span>
  <span><i style="background:var(--downbeat)"></i>downbeat</span>
  <span><i style="background:var(--teal)"></i>beat</span>
</div>

<div class="savebar">
  <button class="save" id="save">&#128190; Save alignment</button>
  <span class="stat" id="stat"></span>
</div>

<div class="modal" id="modal"><div class="sheet">
  <h3>Edit chord</h3>
  <div class="prev" id="mprev">C</div>
  <div class="grp">Root</div>
  <div class="keys" id="mroots"></div>
  <div class="grp">Quality</div>
  <div class="keys q" id="mquals"></div>
  <div class="row">
    <button class="del" id="mdel">Delete</button>
    <button id="mcancel">Cancel</button>
    <button class="apply" id="mapply">Apply</button>
  </div>
</div></div>

<div class="toast" id="toast"></div>

<script>
const D = __ANNOT_DATA__;
// ---------- constants / layout ----------
const H = 300, ADD_LANE = 18, CHORD_TOP = 18, CHORD_H = 40, WF_TOP = 72, WF_BOT = H - 6;
const WF_MID = (WF_TOP + WF_BOT) / 2, WF_HALF = (WF_BOT - WF_TOP) / 2;
const SNAP = (D.snapTolMs || 250) / 1000;   // beat-snap tolerance for chord edges
const BEAT_SNAP = 0.01;                      // beats snap to nearest 10 ms on release
const MINGAP = 0.05;
// pixels/second. Capped so the timeline canvas never exceeds the browser's
// ~32767 px hard limit (a 7-min track at 90 px/s would blow past it and the
// canvas silently paints nothing) — MAX_CANVAS_W keeps us safely under.
const MAX_CANVAS_W = 16000;
let PPS = 90;
let duration = D.duration || 1;
const maxPPS = () => Math.max(6, Math.floor(MAX_CANVAS_W / Math.max(1, duration)));
function clampPPS(){ PPS = Math.min(PPS, maxPPS()); }

// ---------- state ----------
let chords = D.chords.map((c,i)=>({ ...c, dirty:false,
  _origT0:+c.t0, _origLabel:c.label, key:c.bar+':'+c.beat }));
let beatsArr = (D.beats || []).slice();
const beatsOrig = (D.beats || []).slice();
const downSet = new Set((D.downbeats || []).map(x=>+x.toFixed(3)));
let beatDirty = new Array(beatsArr.length).fill(false);
let insCounter = 0, saved = true;
let existingChords = [], existingMerges = [];
// selection + undo + merge + tool toggles (speed-up features)
let selIdx = -1;                 // keyboard-focused chord
const selSet = new Set();        // shift-tapped chords staged for merge
const undoStack = [];            // JSON snapshots, most-recent last
const mergeLog = [];             // {label, parts:[{t0,t1,bar,beat,label}]} for unmerge
let peakSnap = true;             // snap drags to the nearest waveform peak
let syncOn = false;              // beat-sync drift overlay
const PEAK_WIN = 0.15;           // ±150 ms peak-snap window

// ---------- elements ----------
const scroll = document.getElementById('scroll');
const timeline = document.getElementById('timeline');
const canvas = document.getElementById('wf');
const ctx = canvas.getContext('2d');
const beatLayer = document.getElementById('beatlayer');
const chordLayer = document.getElementById('chordlayer');
const syncLayer = document.getElementById('synclayer');
const peakCue = document.getElementById('peakcue');
const playhead = document.getElementById('playhead');
const audio = new Audio();
audio.preload = 'auto'; audio.playsInline = true;

document.getElementById('ttl').textContent = D.title;
document.getElementById('nchords').textContent = chords.length + ' chords';
document.getElementById('tempo').textContent = Math.round(D.bpm||D.tempo||0) + ' bpm';
const gs = document.getElementById('gridsrc');
gs.textContent = D.gridSource==='extract_beat_grid' ? 'beat-grid' : 'grid: '+D.gridSource;

const who = document.getElementById('who');
who.value = localStorage.getItem('harmAnnotator')||'';
who.addEventListener('change',()=>localStorage.setItem('harmAnnotator',who.value.trim()));

// ---------- helpers ----------
const fmt = t => { t=Math.max(0,t); const m=Math.floor(t/60), s=t-60*m;
  return `${m}:${s<10?'0':''}${s.toFixed(1)}`; };
const fmtShort = t => { t=Math.max(0,t); const m=Math.floor(t/60), s=Math.round(t-60*m);
  return `${m}:${s<10?'0':''}${s}`; };
const totalW = () => Math.max(scroll.clientWidth||360, Math.round(duration*PPS));
const xToT = x => x / PPS;
const tToX = t => t * PPS;
function confColor(c){ const q=(c.conf!=null?c.conf:0.6);
  const hue = Math.round(q*130); return `hsl(${hue} 65% ${48-q*4}%)`; }
function nearestBeat(t){ let best=null,bd=1e9; for(const b of beatsArr){const d=Math.abs(b-t); if(d<bd){bd=d;best=b;}} return {b:best,d:bd}; }
// nearest waveform-energy peak (column of max amplitude) within ±PEAK_WIN of t
function nearestPeak(t){
  if(!peaks || !peaks.length) return null;
  const x0=Math.max(0,Math.floor(tToX(t-PEAK_WIN))), x1=Math.min(peaks.length-1,Math.ceil(tToX(t+PEAK_WIN)));
  let bx=-1,bv=-1; for(let x=x0;x<=x1;x++){ if((peaks[x]||0)>bv){bv=peaks[x]||0;bx=x;} }
  if(bx<0) return null; return { t:xToT(bx), v:bv, x:bx };
}
// best snap target for a drag release: nearest of {beat within SNAP, peak within PEAK_WIN}
function snapTarget(t){
  let best=t, bestD=Infinity;
  const nb=nearestBeat(t); if(nb.b!=null && nb.d<=SNAP && nb.d<bestD){ best=nb.b; bestD=nb.d; }
  if(peakSnap){ const np=nearestPeak(t); const d=np?Math.abs(np.t-t):Infinity;
    if(np && d<=PEAK_WIN && d<bestD){ best=np.t; bestD=d; } }
  return best;
}
// live peak-snap cue while dragging (accent line over the candidate peak)
function showPeakCue(t){
  if(!peakSnap){ peakCue.classList.remove('on'); return; }
  const np=nearestPeak(t);
  if(np && Math.abs(np.t-t)<=PEAK_WIN){ peakCue.style.left=tToX(np.t)+'px'; peakCue.classList.add('on'); }
  else peakCue.classList.remove('on');
}
function hidePeakCue(){ peakCue.classList.remove('on'); }
// undo: snapshot the mutable state before any edit, restore on ⌘Z
function snapshot(){
  undoStack.push(JSON.stringify({
    chords: chords.map(c=>({...c})), beatsArr: beatsArr.slice(),
    beatDirty: beatDirty.slice(), mergeLog: mergeLog.slice(), selIdx }));
  if(undoStack.length>60) undoStack.shift();
}
function undo(){
  if(!undoStack.length){ toast('Nothing to undo'); return; }
  const s=JSON.parse(undoStack.pop());
  chords=s.chords; beatsArr=s.beatsArr; beatDirty=s.beatDirty;
  mergeLog.length=0; s.mergeLog.forEach(m=>mergeLog.push(m));
  selIdx=(s.selIdx!=null?s.selIdx:-1); selSet.clear();
  reindexChords(); layoutBeats(); layoutChords(); markDirty(); toast('&#8630; Undo');
}
function markDirty(){ saved=false; updateStat(); }
function updateStat(){
  const nc = chords.filter(c=>c.dirty).length;
  const nb = beatDirty.filter(Boolean).length;
  const parts=[]; if(nb) parts.push(nb+' beat'); if(nc) parts.push(nc+' chord');
  document.getElementById('stat').textContent = parts.length? parts.join(' · ')+' edited' : (saved?'saved':'');
}

// ---------- beat numbering (1,2,3,4 within a bar; downbeat = 1) ----------
function beatNumber(i){
  let n=1;
  for(let j=i;j>=0;j--){ if(downSet.has(+beatsArr[j].toFixed(3))){ n = i-j+1; break; }
    if(j===0) n = i+1; }
  return n;
}

// ---------- waveform ----------
let peaks = null; // Float32Array length = totalW, 0..1 amplitude per pixel column
function drawWave(){
  clampPPS();
  const W = totalW();
  // dpr=1 on purpose: this is a backdrop, and doubling device pixels would
  // push a long-song canvas over the 32767 px limit (blank canvas bug).
  canvas.width = W; canvas.height = H;
  canvas.style.width = W+'px'; canvas.style.height = H+'px';
  timeline.style.width = W+'px';
  ctx.setTransform(1,0,0,1,0,0);
  ctx.clearRect(0,0,W,H);
  // add-lane (tap here to insert a chord boundary)
  ctx.fillStyle = '#141b16';
  ctx.fillRect(0,0,W,ADD_LANE);
  ctx.strokeStyle = '#2f6b4a'; ctx.setLineDash([5,5]);
  ctx.beginPath(); ctx.moveTo(0,ADD_LANE-0.5); ctx.lineTo(W,ADD_LANE-0.5); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#4f9c73'; ctx.font = '700 9px system-ui';
  for(let gx=6; gx<W; gx+=180) ctx.fillText('＋ tap to add a chord boundary', gx, 12);
  // waveform band background
  ctx.fillStyle = '#12161d'; ctx.fillRect(0,WF_TOP-2,W,WF_BOT-WF_TOP+4);
  // center line
  ctx.strokeStyle = '#232c38'; ctx.beginPath(); ctx.moveTo(0,WF_MID); ctx.lineTo(W,WF_MID); ctx.stroke();
  if(peaks){
    ctx.fillStyle = getComputedStyle(document.documentElement).getPropertyValue('--wf').trim()||'#4b5666';
    for(let x=0;x<W;x++){
      const a = peaks[x]||0; const h = Math.max(1, a*WF_HALF);
      ctx.fillRect(x, WF_MID-h, 1, h*2);
    }
  } else {
    ctx.fillStyle='#8b97a8'; ctx.font='12px system-ui';
    ctx.fillText('decoding audio…', 12, WF_MID);
  }
}
async function loadWaveform(){
  if(!D.audioUrl){ audio.remove?.(); drawWave(); return; }
  try{
    const resp = await fetch(D.audioUrl);
    const arr = await resp.arrayBuffer();
    // iOS-friendly: use direct URL for playback + Web Audio for waveform
    audio.src = D.audioUrl;
    audio.crossOrigin = 'anonymous';
    // Try to preload (iOS may block, that's OK)
    audio.load?.();
    // Add event listeners
    audio.addEventListener('loadstart', ()=>console.log('[audio] loadstart'));
    audio.addEventListener('canplay', ()=>console.log('[audio] canplay'));
    audio.addEventListener('error', (e)=>console.log('[audio] error:', e.target?.error));
    // Decode for waveform visualization (separate from playback)
    const AC = window.AudioContext || window.webkitAudioContext;
    const ac = new AC();
    const buf = await ac.decodeAudioData(arr.slice(0));
    duration = buf.duration || duration;
    decodedBuf = buf;
    clampPPS();
    document.getElementById('tot').textContent = fmt(duration);
    document.getElementById('dur').textContent = fmtShort(duration);
    computePeaks(buf);
    ac.close && ac.close();
    console.log('[audio] loaded ✓');
  }catch(e){ console.warn('[audio] decode error:',e); }
  drawWave(); layoutBeats(); layoutChords();
}
function computePeaks(buf){
  const W = totalW();
  const ch0 = buf.getChannelData(0);
  const ch1 = buf.numberOfChannels>1 ? buf.getChannelData(1) : null;
  const N = ch0.length, sr = buf.sampleRate;
  peaks = new Float32Array(W);
  let peakMax = 1e-6;
  for(let x=0;x<W;x++){
    const s0 = Math.floor(xToT(x)*sr), s1 = Math.min(N, Math.floor(xToT(x+1)*sr));
    let sum=0, cnt=0;
    for(let s=s0;s<s1;s+=4){ // stride 4 for speed
      let v = ch0[s]; if(ch1) v=(v+ch1[s])*0.5;
      sum += v*v; cnt++;
    }
    const rms = cnt? Math.sqrt(sum/cnt) : 0;
    peaks[x]=rms; if(rms>peakMax) peakMax=rms;
  }
  // normalise + gentle gamma so quiet passages stay visible
  for(let x=0;x<W;x++) peaks[x] = Math.pow(Math.min(1, peaks[x]/peakMax), 0.7);
}

// ---------- beat layer ----------
function layoutBeats(){
  const W = totalW();
  let html='';
  for(let i=0;i<beatsArr.length;i++){
    const t=beatsArr[i]; if(t> duration+1) continue;
    const x=tToX(t); const db=downSet.has(+t.toFixed(3));
    html+=`<div class="beat${db?' db':''}${beatDirty[i]?' drag':''}" data-b="${i}" style="left:${x}px">`
        +`<b>${beatNumber(i)}</b><i></i></div>`;
  }
  beatLayer.innerHTML=html; beatLayer.style.width=W+'px';
}
function moveBeatEl(i){
  const el=beatLayer.querySelector(`.beat[data-b="${i}"]`); if(!el) return;
  el.style.left=tToX(beatsArr[i])+'px';
}

// ---------- chord layer ----------
function layoutChords(){
  const W = totalW();
  let html='';
  chords.forEach((c,i)=>{
    const x=tToX(c.t0), w=Math.max(10, tToX(c.t1)-tToX(c.t0));
    const sel = (i===selIdx?' sel':'') + (selSet.has(i)?' msel':'');
    html+=`<div class="chordbar${c.dirty?' dirty':''}${sel}" data-i="${i}" `
        +`style="left:${x}px;width:${w}px;background:${confColor(c)}">`
        +`<div class="edge l" data-edge="l" data-i="${i}"></div>`
        +`<span>${(c.label||'?')}</span>`
        +`<div class="edge r" data-edge="r" data-i="${i}"></div></div>`;
  });
  chordLayer.innerHTML=html; chordLayer.style.width=W+'px';
  layoutSync();
}
// ---------- beat-sync overlay: per-boundary drift dot vs nearest beat ----------
function layoutSync(){
  if(!syncOn){ syncLayer.innerHTML=''; return; }
  let html='';
  chords.forEach(c=>{
    const nb=nearestBeat(c.t0); if(nb.b==null) return;
    const drift=nb.d;                       // seconds to nearest beat
    const col = drift<=0.1 ? 'var(--ok)' : 'var(--amber)';
    html+=`<div class="syncdot" style="left:${tToX(c.t0)}px;background:${col}" `
        +`title="drift ${(drift*1000).toFixed(0)} ms"></div>`;
  });
  syncLayer.innerHTML=html; syncLayer.style.width=totalW()+'px';
}
// ---------- selection ----------
function selectChord(i,{scrollTo=false,seek=false}={}){
  selIdx=(i>=0&&i<chords.length)?i:-1; layoutChords();
  if(selIdx>=0){
    if(seek) seekTo(chords[selIdx].t0);
    if(scrollTo){ const px=tToX(chords[selIdx].t0);
      if(px<scroll.scrollLeft+40 || px>scroll.scrollLeft+scroll.clientWidth-40)
        scroll.scrollLeft=px-scroll.clientWidth*0.4; }
  }
}
function toggleMulti(i){
  if(selSet.has(i)) selSet.delete(i); else selSet.add(i);
  updateMergeBtn(); layoutChords();
}
function updateMergeBtn(){
  const b=document.getElementById('tmerge');
  b.classList.toggle('ready', selSet.size>=2);
  b.innerHTML = selSet.size>=2 ? ('&#9776; Merge '+selSet.size) : '&#9776; Merge';
}
function moveChordEl(i){
  const el=chordLayer.querySelector(`.chordbar[data-i="${i}"]`); if(!el) return;
  const c=chords[i];
  el.style.left=tToX(c.t0)+'px';
  el.style.width=Math.max(10, tToX(c.t1)-tToX(c.t0))+'px';
  el.classList.toggle('dirty', !!c.dirty);
  el.style.background=confColor(c);
}
function reindexChords(){ chords.forEach((c,i)=>c.i=i); }

// ---------- edit ops ----------
function setChordEdge(i, side, t, {snap=true}={}){
  if(snap){ t=snapTarget(t); }
  if(side==='l'){
    const lo = i>0? chords[i-1].t0+MINGAP : 0;
    const hi = chords[i].t1 - MINGAP;
    t=Math.min(Math.max(t,lo),hi);
    chords[i].t0=t; if(i>0) chords[i-1].t1=t;
    chords[i].dirty=true; if(i>0) chords[i-1].dirty=true;
  }else{
    const lo = chords[i].t0 + MINGAP;
    const hi = i<chords.length-1? chords[i+1].t1-MINGAP : duration+5;
    t=Math.min(Math.max(t,lo),hi);
    chords[i].t1=t; if(i<chords.length-1) chords[i+1].t0=t;
    chords[i].dirty=true; if(i<chords.length-1) chords[i+1].dirty=true;
  }
  markDirty();
}
function addBoundaryAt(t){
  // beat-snap the new boundary
  const nb=nearestBeat(t); if(nb.b!=null && nb.d<=SNAP) t=nb.b;
  let idx=-1;
  for(let i=0;i<chords.length;i++){ if(chords[i].t0<=t && t<chords[i].t1){ idx=i; break; } }
  if(idx<0){ // past the last chord — extend a new one to the end
    idx=chords.length-1; if(idx<0) return;
  }
  const host=chords[idx];
  if(t<=host.t0+MINGAP || t>=host.t1-MINGAP) return; // too close to an existing edge
  snapshot();
  const nc={ i:idx+1, bar:host.bar, beat:900+(insCounter++), section:host.section,
    label:host.label, t0:t, t1:host.t1, match:'', conf:0.6, dirty:true, inserted:true,
    _origT0:t, _origLabel:host.label, key:'ins'+insCounter };
  host.t1=t; host.dirty=true;
  chords.splice(idx+1,0,nc); reindexChords();
  layoutChords(); markDirty();
  if(navigator.vibrate) navigator.vibrate(8);
  openEditor(idx+1);
}
function deleteChord(i){
  if(chords.length<=1) return;
  const c=chords[i];
  if(i>0) chords[i-1].t1 = c.t1, chords[i-1].dirty=true;
  else if(i<chords.length-1) chords[i+1].t0=c.t0, chords[i+1].dirty=true;
  chords.splice(i,1); reindexChords();
  if(selIdx>=chords.length) selIdx=chords.length-1;
  layoutChords(); markDirty();
}
// nudge the boundary between chord i and i-1 (arrow keys)
function nudgeChordStart(i, deltaSec){
  if(i<0||i>=chords.length) return;
  snapshot();
  setChordEdge(i,'l',chords[i].t0+deltaSec,{snap:false});
  moveChordEl(i); if(i>0) moveChordEl(i-1); layoutSync();
}
// merge the shift-selected adjacent same-label chords into one span
function mergeSelected(){
  const idxs=[...selSet].sort((a,b)=>a-b);
  if(idxs.length<2){ toast('Shift-tap 2+ adjacent chords first',true); return; }
  for(let k=1;k<idxs.length;k++) if(idxs[k]!==idxs[k-1]+1){ toast('Selection must be adjacent',true); return; }
  if(new Set(idxs.map(i=>chords[i].label)).size>1){ toast('Merge needs identical labels',true); return; }
  snapshot();
  const first=idxs[0], last=idxs[idxs.length-1];
  const rec={ label:chords[first].label,
    parts: idxs.map(i=>({t0:chords[i].t0,t1:chords[i].t1,bar:chords[i].bar,beat:chords[i].beat,label:chords[i].label})) };
  mergeLog.push(rec); existingMerges.push(rec);
  chords[first].t1=chords[last].t1; chords[first].dirty=true;
  chords.splice(first+1,last-first);
  reindexChords(); selSet.clear(); selIdx=first; updateMergeBtn();
  layoutChords(); markDirty(); toast('Merged '+idxs.length+' chords');
  if(navigator.vibrate) navigator.vibrate(8);
}
// revert the most recent merge (Ctrl/Cmd+M)
function unmerge(){
  const rec=mergeLog.pop();
  if(!rec){ toast('Nothing to unmerge',true); return; }
  const mi=existingMerges.lastIndexOf(rec); if(mi>=0) existingMerges.splice(mi,1);
  snapshot();
  let idx=chords.findIndex(c=>Math.abs(c.t0-rec.parts[0].t0)<1e-3 && c.label===rec.label);
  if(idx<0) idx=chords.findIndex(c=>c.label===rec.label);
  if(idx<0){ toast('Cannot locate merged chord',true); return; }
  const base=chords[idx];
  const rebuilt=rec.parts.map(p=>({ ...base, t0:p.t0, t1:p.t1, bar:p.bar, beat:p.beat,
    label:p.label, dirty:true, inserted:false, key:p.bar+':'+p.beat }));
  chords.splice(idx,1,...rebuilt);
  reindexChords(); selIdx=idx; layoutChords(); markDirty(); toast('Unmerged');
}

// ---------- pointer interactions ----------
let drag=null; // {type:'beat'|'edge', i, side, moved}
timeline.addEventListener('pointerdown', e=>{
  const beatEl=e.target.closest('.beat');
  const edgeEl=e.target.closest('.edge');
  const barEl=e.target.closest('.chordbar');
  if(barEl && e.shiftKey){ toggleMulti(+barEl.dataset.i); e.preventDefault(); return; }
  if(beatEl){
    const i=+beatEl.dataset.b; snapshot(); drag={type:'beat',i,moved:false};
    beatEl.classList.add('drag'); beatEl.setPointerCapture(e.pointerId); e.preventDefault(); return;
  }
  if(edgeEl){
    snapshot(); drag={type:'edge',i:+edgeEl.dataset.i,side:edgeEl.dataset.edge,moved:false};
    edgeEl.setPointerCapture(e.pointerId); e.preventDefault(); return;
  }
  if(barEl){ drag={type:'tap',i:+barEl.dataset.i,moved:false,x0:e.clientX}; return; }
  // empty area — decide by band
  const rect=timeline.getBoundingClientRect();
  const y=e.clientY-rect.top, x=(e.clientX-rect.left);
  if(y<CHORD_TOP+CHORD_H+6){ addBoundaryAt(xToT(x)); }
  else { seekTo(xToT(x)); }
},{passive:false});

timeline.addEventListener('pointermove', e=>{
  if(!drag) return;
  const rect=timeline.getBoundingClientRect();
  const t=xToT(Math.max(0, e.clientX-rect.left));
  if(drag.type==='beat'){
    drag.moved=true;
    let lo = drag.i>0? beatsArr[drag.i-1]+0.02 : 0;
    let hi = drag.i<beatsArr.length-1? beatsArr[drag.i+1]-0.02 : duration+2;
    beatsArr[drag.i]=Math.min(Math.max(t,lo),hi);
    moveBeatEl(drag.i); showPeakCue(beatsArr[drag.i]);
  }else if(drag.type==='edge'){
    drag.moved=true;
    setChordEdge(drag.i, drag.side, t, {snap:false});
    moveChordEl(drag.i); showPeakCue(t);
    if(drag.side==='l' && drag.i>0) moveChordEl(drag.i-1);
    if(drag.side==='r' && drag.i<chords.length-1) moveChordEl(drag.i+1);
  }else if(drag.type==='tap'){
    if(Math.abs(e.clientX-drag.x0)>6) drag.moved=true;
  }
},{passive:true});

timeline.addEventListener('pointerup', e=>{
  if(!drag) return;
  hidePeakCue();
  if(drag.type==='beat'){
    // snap to nearest waveform peak (if enabled/in range), then to the 10 ms grid
    beatsArr[drag.i]=snapTarget(beatsArr[drag.i]);
    beatsArr[drag.i]=Math.round(beatsArr[drag.i]/BEAT_SNAP)*BEAT_SNAP;
    beatDirty[drag.i]=Math.abs(beatsArr[drag.i]-beatsOrig[drag.i])>0.005;
    const el=beatLayer.querySelector(`.beat[data-b="${drag.i}"]`);
    if(el){ el.classList.toggle('drag',beatDirty[drag.i]); moveBeatEl(drag.i); }
    if(navigator.vibrate && beatDirty[drag.i]) navigator.vibrate(6);
    layoutSync(); markDirty();
  }else if(drag.type==='edge'){
    setChordEdge(drag.i, drag.side, drag.side==='l'?chords[drag.i].t0:chords[drag.i].t1, {snap:true});
    moveChordEl(drag.i);
    if(drag.i>0) moveChordEl(drag.i-1);
    if(drag.i<chords.length-1) moveChordEl(drag.i+1);
    layoutSync();
    if(navigator.vibrate) navigator.vibrate(5);
  }else if(drag.type==='tap' && !drag.moved){
    selIdx=drag.i; openEditor(drag.i);
  }
  drag=null;
});

// ---------- transport ----------
const playBtn=document.getElementById('play');
let audioSource=null, audioGain=null;
function seekTo(t){ t=Math.min(Math.max(0,t),duration); audio.currentTime=t; updatePlayhead(); }

function playAudio(){
  // Try HTML5 audio first
  const tryHTML5 = audio.play?.().catch(()=>{
    console.log('[play] HTML5 failed, trying Web Audio API');
    if(decodedBuf){
      try {
        const AC = window.AudioContext || window.webkitAudioContext;
        const ac = new AC();
        if(!audioSource) audioSource = ac.createBufferSource();
        if(!audioGain) { audioGain = ac.createGain(); audioGain.connect(ac.destination); }
        audioSource.buffer = decodedBuf;
        audioSource.connect(audioGain);
        audioSource.start(0, audio.currentTime || 0);
        console.log('[play] Web Audio started');
      } catch(e) { console.error('[play] Web Audio failed:', e); }
    }
  });
}

function pauseAudio(){
  if(audioSource) {
    try { audioSource.stop(); } catch(e) {}
    audioSource = null;
  }
  audio.pause?.();
}

playBtn.addEventListener('click',()=>{
  if(audio.paused && (!audioSource || audioSource.playbackRate === undefined)){
    playAudio();
    playBtn.innerHTML='&#10073;&#10073;'; rafLoop();
  } else {
    pauseAudio();
    playBtn.innerHTML='&#9654;';
  }
});
audio.addEventListener('play',()=>{ playBtn.innerHTML='&#10073;&#10073;'; rafLoop(); console.log('[play] started'); });
audio.addEventListener('pause',()=>{ playBtn.innerHTML='&#9654;'; console.log('[play] paused'); });
audio.addEventListener('ended',()=>{ playBtn.innerHTML='&#9654;'; console.log('[play] ended'); });
audio.addEventListener('loadedmetadata',()=>{ if(isFinite(audio.duration)&&audio.duration>0){
  duration=audio.duration; document.getElementById('tot').textContent=fmt(duration);
  document.getElementById('dur').textContent=fmtShort(duration); }});

let rafOn=false;
function rafLoop(){ if(rafOn) return; rafOn=true;
  const step=()=>{ updatePlayhead(); if(!audio.paused){ requestAnimationFrame(step); } else { rafOn=false; } };
  requestAnimationFrame(step);
}
function updatePlayhead(){
  const t=audio.currentTime||0;
  playhead.style.transform='translateX('+tToX(t)+'px)';
  document.getElementById('cur').textContent=fmt(t);
  // keep playhead in view while playing
  if(!audio.paused){
    const px=tToX(t), left=scroll.scrollLeft, w=scroll.clientWidth;
    if(px<left+w*0.15 || px>left+w*0.85) scroll.scrollLeft=px-w*0.4;
  }
}
audio.addEventListener('timeupdate',updatePlayhead);

// ---------- zoom ----------
function setZoom(f){
  const t=audio.currentTime||xToT(scroll.scrollLeft+scroll.clientWidth/2);
  PPS=Math.min(maxPPS(), Math.max(6, PPS*f));
  drawWave(); layoutBeats(); layoutChords(); updatePlayhead();
  scroll.scrollLeft=tToX(t)-scroll.clientWidth/2;
}
document.getElementById('zin').addEventListener('click',()=>{ if(peaks) computePeaks_lazy(); setZoom(1.4); });
document.getElementById('zout').addEventListener('click',()=>{ setZoom(1/1.4); });
let decodedBuf=null;
function computePeaks_lazy(){ if(decodedBuf) computePeaks(decodedBuf); }

// ---------- chord editor modal ----------
const NOTE_PC={C:0,D:2,E:4,F:5,G:7,A:9,B:11};
const ROOTS=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
// friendly label -> iReal tail
const QUALS=[['maj','^'],['maj7','^7'],['7','7'],['6','6'],['min','-'],['m7','-7'],
  ['m6','-6'],['m7b5','-7b5'],['dim7','o7'],['sus','sus'],['9','9'],['13','13']];
let editI=-1, editRootPc=0, editQual='';
function parseIreal(lbl){
  const m=/^([A-G])([#b]?)(.*)$/.exec(String(lbl||'').trim());
  if(!m) return {root:0,q:''};
  let pc=NOTE_PC[m[1]]; if(m[2]==='#')pc=(pc+1)%12; else if(m[2]==='b')pc=(pc+11)%12;
  return {root:pc,q:m[3]};
}
function buildLabel(pc,q){ return ROOTS[pc]+q; }
const modal=document.getElementById('modal');
function renderModalKeys(){
  document.getElementById('mroots').innerHTML=ROOTS.map((r,pc)=>
    `<button data-pc="${pc}" class="${pc===editRootPc?'on':''}">${r}</button>`).join('');
  document.getElementById('mquals').innerHTML=QUALS.map(([n,q])=>
    `<button data-q="${q}" class="${q===editQual?'on':''}">${n}</button>`).join('');
  document.getElementById('mprev').textContent=buildLabel(editRootPc,editQual);
}
function openEditor(i){
  editI=i; const p=parseIreal(chords[i].label);
  editRootPc=p.root; editQual=p.q; renderModalKeys(); modal.classList.add('on');
  // pause + park the playhead at this chord so the user hears the change point
  audio.pause(); seekTo(chords[i].t0);
}
modal.addEventListener('click',e=>{
  if(e.target===modal){ modal.classList.remove('on'); return; }
  const rb=e.target.closest('[data-pc]'); if(rb){ editRootPc=+rb.dataset.pc; renderModalKeys(); return; }
  const qb=e.target.closest('[data-q]'); if(qb){ editQual=qb.dataset.q; renderModalKeys(); return; }
});
document.getElementById('mcancel').addEventListener('click',()=>modal.classList.remove('on'));
document.getElementById('mapply').addEventListener('click',()=>{
  if(editI<0) return;
  const lbl=buildLabel(editRootPc,editQual);
  if(lbl!==chords[editI].label){ snapshot(); chords[editI].label=lbl; chords[editI].dirty=true; markDirty(); moveChordEl(editI); layoutChords(); }
  modal.classList.remove('on');
});
document.getElementById('mdel').addEventListener('click',()=>{
  if(editI>=0){ snapshot(); deleteChord(editI); } modal.classList.remove('on');
});

// ---------- tools row (peak-snap · beat-sync overlay · merge) ----------
document.getElementById('tpeak').addEventListener('click',function(){
  peakSnap=!peakSnap; this.classList.toggle('on',peakSnap);
  toast('Peak-snap '+(peakSnap?'on':'off'));
});
document.getElementById('tsync').addEventListener('click',function(){
  syncOn=!syncOn; this.classList.toggle('on',syncOn); layoutSync();
  toast('Beat-sync overlay '+(syncOn?'on':'off'));
});
document.getElementById('tmerge').addEventListener('click',mergeSelected);

// ---------- keyboard shortcuts ----------
document.addEventListener('keydown', e=>{
  const tag=(document.activeElement&&document.activeElement.tagName)||'';
  const typing = tag==='INPUT'||tag==='TEXTAREA';
  const mod = e.ctrlKey||e.metaKey;
  if(e.key===' ' && !typing){ e.preventDefault(); playBtn.click(); return; }
  if(mod && (e.key==='s'||e.key==='S')){ e.preventDefault(); saveBtn.click(); return; }
  if(mod && (e.key==='z'||e.key==='Z')){ e.preventDefault(); undo(); return; }
  if(mod && (e.key==='m'||e.key==='M')){ e.preventDefault(); unmerge(); return; }
  if(typing || mod) return;
  if(e.key==='Tab'){ e.preventDefault();
    selectChord(selIdx<0?0:(selIdx+1)%chords.length,{scrollTo:true,seek:true}); return; }
  if(e.key==='ArrowLeft'||e.key==='ArrowRight'){
    if(selIdx<0){ selectChord(0,{scrollTo:true}); return; }
    e.preventDefault();
    const dir=e.key==='ArrowRight'?1:-1;
    nudgeChordStart(selIdx, dir*(e.shiftKey?0.5:0.1));
  }
});

// ---------- load existing sidecar (resume prior alignment) ----------
fetch('/api/annotations/'+encodeURIComponent(D.saveFile)).then(r=>r.json()).then(doc=>{
  existingChords=doc.chords||[]; existingMerges=doc.merges||[];
  if(!who.value && doc.annotator) who.value=doc.annotator;
  const byKey={}; existingChords.forEach(c=>{ if('t0' in c) byKey[c.bar+':'+c.beat]=c; });
  let resumed=0;
  chords.forEach((c,i)=>{ const p=byKey[c.bar+':'+c.beat];
    if(p){ c.t0=+p.t0; if('t1' in p)c.t1=+p.t1; if(i>0)chords[i-1].t1=c.t0; c._origT0=+p.t0; c.dirty=false; resumed++; } });
  if(resumed){ saved=true; layoutChords(); updateStat(); }
}).catch(()=>{});

// ---------- save + logging ----------
const saveBtn=document.getElementById('save');
saveBtn.addEventListener('click',async()=>{
  const now=new Date().toISOString();
  const map={}; existingChords.forEach(c=>{ map[c.bar+':'+c.beat]={...c}; });
  chords.forEach(c=>{ const k=c.bar+':'+c.beat;
    map[k]={...(map[k]||{}), bar:c.bar, beat:c.beat, label:c.label, section:c.section,
            t0:+(+c.t0).toFixed(3), t1:+(+c.t1).toFixed(3), ts:now}; });
  const body={ annotator: who.value.trim()||'anon', chords:Object.values(map), merges:existingMerges };
  const dirtyChords = chords.filter(c=>c.dirty);
  const beatShifts = [];
  for(let i=0;i<beatsArr.length;i++){ if(beatDirty[i]) beatShifts.push({
    index:i, orig_s:+beatsOrig[i].toFixed(3), new_s:+beatsArr[i].toFixed(3),
    delta_ms:Math.round((beatsArr[i]-beatsOrig[i])*1000) }); }
  saveBtn.disabled=true;
  try{
    const doc=await fetch('/api/annotations/'+encodeURIComponent(D.saveFile),
      {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).then(r=>r.json());
    existingChords=doc.chords||[]; existingMerges=doc.merges||[];
    chords.forEach(c=>c.dirty=false); beatDirty.fill(false);
    saved=true; layoutChords(); layoutBeats(); updateStat();
    const nb=beatShifts.length, nc=dirtyChords.length;
    toast('&#10003; Saved: '+nb+' beat shift'+(nb===1?'':'s')+' + '+nc+' chord correction'+(nc===1?'':'s'));
    // fire-and-forget training logs — never block the save
    logBeatShifts(beatShifts);
    logChordCorrections(dirtyChords);
  }catch(e){ toast('Save failed — check connection',true); }
  finally{ saveBtn.disabled=false; }
});

async function logBeatShifts(shifts){
  if(!shifts.length) return;
  const rec={ song:D.slug, timestamp:new Date().toISOString(), type:'beat_alignment',
    human_session:who.value.trim()||'anon', n_beat_shifts:shifts.length, beat_adjustments:shifts };
  try{ await fetch('/api/correction-log/'+encodeURIComponent(D.slug),
    {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(rec)}); }
  catch(e){ console.warn('beat-shift log failed',e); }
}
async function logChordCorrections(dirty){
  for(const c of dirty){
    const parsed=parseIreal(c.label);
    let rr={}, diff=[], reinferErr=null, rejected=[];
    try{
      const body={confirms:[{t0:+c.t0,t1:+c.t1,root:parsed.root,q:parsed.q}],merges:[]};
      rr=await fetch('/api/reinfer/'+encodeURIComponent(D.saveFile),
        {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
        .then(r=>r.ok?r.json():r.json().then(j=>Promise.reject(j.error||('HTTP '+r.status))));
      diff=rr.diff||[]; rejected=rr.rejected||[];
    }catch(e){ reinferErr=String(e); console.warn('reinfer failed',e); }
    const hit=diff.find(d=>d.start_s<+c.t1 && d.end_s>+c.t0);
    const improvements=diff.filter(d=>((d.new_confidence||0)-(d.old_confidence||0))>0)
      .map(d=>({old_label:d.old_label,new_label:d.new_label,
        confidence_change:+(((d.new_confidence||0)-(d.old_confidence||0)).toFixed(3))}));
    const rec={ song:D.slug, timestamp:new Date().toISOString(), type:'chord_correction',
      human_session:who.value.trim()||'anon',
      original_prediction:{ bar:c.bar, beat:c.beat,
        chord: hit?hit.old_label:c._origLabel, confidence: hit?(hit.old_confidence!=null?hit.old_confidence:null):null,
        time_s:+(+c._origT0).toFixed(3) },
      human_correction:{ chord:c.label, time_s:+(+c.t0).toFixed(3), inserted:!!c.inserted },
      reinfer_result:{ n_chords_changed: rr.n_changed!=null?rr.n_changed:diff.length,
        diff: diff.map(d=>({old_label:d.old_label,new_label:d.new_label,
          confidence_change:+(((d.new_confidence||0)-(d.old_confidence||0)).toFixed(3)),
          start_s:d.start_s,end_s:d.end_s})),
        rejected_merges:rejected, error:reinferErr },
      benefit:{ self_corrected:!!hit, propagation_count:diff.length, improvements } };
    try{ await fetch('/api/correction-log/'+encodeURIComponent(D.slug),
      {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(rec)}); }
    catch(e){ console.warn('correction-log failed',e); }
  }
}

// ---------- toast ----------
const toastEl=document.getElementById('toast'); let toastT;
function toast(msg,err){ toastEl.innerHTML=msg; toastEl.className='toast on'+(err?' err':'');
  clearTimeout(toastT); toastT=setTimeout(()=>toastEl.className='toast'+(err?' err':''),2400); }

// ---------- init ----------
document.getElementById('tot').textContent=fmt(duration);
document.getElementById('dur').textContent=fmtShort(duration);
drawWave(); layoutBeats(); layoutChords();
loadWaveform().then(()=>{
  // stash decoded buffer for zoom re-peaking
});
window.addEventListener('resize',()=>{ drawWave(); layoutBeats(); layoutChords(); updatePlayhead(); });
// deep-link: /annotator?song=..&sel=N centers a chord
{ const s=parseInt(new URLSearchParams(location.search).get('sel'));
  if(!isNaN(s)&&s>=0&&s<chords.length) requestAnimationFrame(()=>{ scroll.scrollLeft=tToX(chords[s].t0)-scroll.clientWidth/2; }); }
</script>
</body></html>"""


# ── Rebuilt annotator (simple linear flow, iPhone Safari first) ─────────────────
ANNOTATOR_SIMPLE_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no,viewport-fit=cover">
<title>Harmonia — Chord Timing</title>
<style>
  :root{ --bg:#0e1116; --card:#171c24; --card2:#1e2530; --ink:#e8edf4; --faint:#94a1b3;
         --line:#2a3340; --teal:#00c9a7; --amber:#ffb454; --accent:#6ea8ff; --danger:#ff5d6c; --ok:#37d67a; }
  *{box-sizing:border-box;-webkit-tap-highlight-color:transparent;}
  html,body{margin:0;background:var(--bg);color:var(--ink);overscroll-behavior:none;overflow-x:hidden;
    font-family:-apple-system,BlinkMacSystemFont,system-ui,sans-serif;}
  body{padding-bottom:calc(88px + env(safe-area-inset-bottom));}
  .wrap{max-width:640px;margin:0 auto;}
  /* header */
  header{position:sticky;top:0;z-index:10;background:#0e1116;border-bottom:1px solid var(--line);
    padding:calc(8px + env(safe-area-inset-top)) 14px 8px;display:flex;align-items:center;gap:10px;}
  header a.back{color:var(--faint);text-decoration:none;font:600 15px system-ui;padding:6px;margin-left:-6px;}
  header h1{font-size:15px;margin:0;font-weight:700;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
  input.who{width:88px;background:var(--card2);border:1px solid var(--line);color:var(--ink);
    border-radius:8px;padding:6px 8px;font:600 12px system-ui;}
  /* instructions */
  .howto{background:var(--card);border:1px solid var(--line);border-radius:12px;margin:12px 14px;padding:12px 14px;
    font:500 13px system-ui;color:var(--faint);line-height:1.5;}
  .howto b{color:var(--ink);}
  /* loading overlay */
  #loader{position:fixed;inset:0;z-index:100;background:var(--bg);display:flex;flex-direction:column;
    align-items:center;justify-content:center;gap:14px;padding:24px;}
  #loader h2{font:800 18px system-ui;margin:0;}
  #loader .steps{font:600 13px ui-monospace,Menlo,monospace;color:var(--faint);line-height:1.9;text-align:left;}
  #loader .steps .done{color:var(--ok);} #loader .steps .fail{color:var(--danger);}
  #loader .spin{width:34px;height:34px;border:3px solid var(--line);border-top-color:var(--teal);
    border-radius:50%;animation:sp 0.8s linear infinite;}
  @keyframes sp{to{transform:rotate(360deg);}}
  /* section card */
  .card{background:var(--card);border:1px solid var(--line);border-radius:14px;margin:12px 14px;padding:14px;}
  .card h3{margin:0 0 10px;font:700 11px system-ui;letter-spacing:.6px;text-transform:uppercase;color:var(--faint);}
  /* player */
  .player{display:flex;align-items:center;gap:14px;}
  .playbtn{width:64px;height:64px;flex:none;border:none;border-radius:50%;background:var(--teal);color:#052;
    font-size:26px;display:flex;align-items:center;justify-content:center;box-shadow:0 4px 14px #00c9a733;}
  .playbtn:active{transform:scale(.94);} .playbtn[disabled]{background:var(--card2);color:var(--faint);box-shadow:none;}
  .pcol{flex:1;min-width:0;}
  .clock{font:700 15px ui-monospace,Menlo,monospace;} .clock .d{color:var(--faint);}
  input[type=range]{width:100%;height:32px;background:transparent;margin:4px 0 0;-webkit-appearance:none;}
  input[type=range]::-webkit-slider-runnable-track{height:6px;border-radius:3px;background:var(--card2);}
  input[type=range]::-webkit-slider-thumb{-webkit-appearance:none;width:22px;height:22px;margin-top:-8px;
    border-radius:50%;background:var(--teal);border:2px solid #0e1116;}
  .volrow{display:flex;align-items:center;gap:10px;margin-top:8px;color:var(--faint);font:600 12px system-ui;}
  .volrow input{flex:1;}
  /* current chord */
  .nowbig{font:800 34px 'SF Mono',ui-monospace,Menlo,monospace;color:var(--teal);text-align:center;
    padding:6px 0;letter-spacing:1px;min-height:44px;}
  .nowsub{text-align:center;color:var(--faint);font:600 12px system-ui;margin-top:-4px;}
  /* waveform */
  .wfscroll{overflow-x:auto;overflow-y:hidden;-webkit-overflow-scrolling:touch;border-radius:10px;
    background:var(--card2);position:relative;touch-action:pan-x;}
  .wftrack{position:relative;height:120px;}
  .wftrack canvas{position:absolute;left:0;top:0;display:block;}
  .region{position:absolute;top:8px;height:60px;border-radius:7px;background:#00c9a722;border:1px solid #00c9a755;
    display:flex;align-items:center;justify-content:center;overflow:hidden;}
  .region.dirty{background:#ffb45422;border-color:var(--amber);}
  .region.sel{background:#6ea8ff33;border-color:var(--accent);}
  .region span{font:700 12px 'SF Mono',ui-monospace,Menlo,monospace;color:var(--ink);pointer-events:none;
    white-space:nowrap;padding:0 6px;text-overflow:ellipsis;overflow:hidden;}
  .handle{position:absolute;top:0;bottom:0;width:44px;touch-action:none;cursor:ew-resize;z-index:3;
    display:flex;align-items:center;justify-content:center;}
  .handle.l{left:-22px;} .handle.r{right:-22px;}
  .handle::after{content:"";width:4px;height:44px;border-radius:2px;background:var(--teal);opacity:.75;}
  .region.dirty .handle::after{background:var(--amber);opacity:1;}
  .handle.drag::after{opacity:1;box-shadow:0 0 8px var(--teal);}
  #playhead{position:absolute;top:0;bottom:0;width:2px;background:var(--danger);z-index:4;pointer-events:none;
    box-shadow:0 0 6px var(--danger);}
  .wfnote{padding:10px;color:var(--faint);font:600 12px system-ui;text-align:center;}
  /* chord list */
  .clist{display:flex;flex-direction:column;gap:6px;}
  .crow{display:flex;align-items:center;gap:8px;background:var(--card2);border:1px solid var(--line);
    border-radius:10px;padding:8px 10px;}
  .crow.sel{border-color:var(--accent);} .crow.dirty{border-color:var(--amber);}
  .crow.playing{background:#00c9a71a;border-color:var(--teal);}
  .crow .lab{flex:1;min-width:0;font:700 15px 'SF Mono',ui-monospace,Menlo,monospace;
    overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
  .crow .t{font:600 12px ui-monospace,Menlo,monospace;color:var(--faint);min-width:54px;text-align:right;}
  .crow .jump{width:44px;height:44px;flex:none;border:1px solid var(--line);background:var(--card);color:var(--accent);
    border-radius:9px;font-size:16px;}
  .crow .nudge{width:44px;height:44px;flex:none;border:1px solid var(--line);background:var(--card);color:var(--ink);
    border-radius:9px;font:800 20px ui-monospace;}
  .crow .nudge:active,.crow .jump:active{background:var(--card2);}
  /* save bar */
  .savebar{position:fixed;left:0;right:0;bottom:0;z-index:20;padding:10px 14px calc(10px + env(safe-area-inset-bottom));
    background:linear-gradient(0deg,#0e1116 70%,#0e1116cc);border-top:1px solid var(--line);
    display:flex;gap:10px;align-items:center;}
  .savebar .save{flex:1;min-height:54px;border:none;border-radius:14px;background:var(--teal);color:#052;
    font:800 16px system-ui;} .savebar .save:active{transform:scale(.98);} .savebar .save[disabled]{opacity:.5;}
  .savebar .stat{font:700 12px system-ui;color:var(--faint);min-width:70px;text-align:right;}
  /* toast */
  .toast{position:fixed;left:50%;bottom:100px;transform:translateX(-50%) translateY(16px);opacity:0;
    background:var(--ok);color:#052;font:800 13px system-ui;padding:11px 18px;border-radius:22px;z-index:60;
    transition:.25s;pointer-events:none;max-width:90vw;text-align:center;box-shadow:0 6px 20px #0008;}
  .toast.on{opacity:1;transform:translateX(-50%) translateY(0);} .toast.err{background:var(--danger);color:#fff;}
</style></head>
<body>
<div id="loader">
  <div class="spin"></div>
  <h2 id="ltitle">Loading…</h2>
  <div class="steps">
    <div id="s-chords">• Loading chords…</div>
    <div id="s-audio">• Loading audio…</div>
    <div id="s-wave">• Decoding waveform…</div>
  </div>
</div>

<div class="wrap">
  <header>
    <a class="back" href="/">‹ Back</a>
    <h1 id="title">Song</h1>
    <input class="who" id="who" placeholder="your name" autocomplete="off">
  </header>

  <div class="howto">
    <b>Adjust chord timings.</b> Play the song, then drag a chord's <b>edge handles</b> left/right on the
    waveform to line it up with what you hear. Fine-tune with the <b>−/+</b> buttons in the list
    (±0.1s). Tap <b>▸</b> to jump the audio there. Press <b>Save</b> when done.
  </div>

  <div class="card">
    <h3>Player</h3>
    <div class="player">
      <button class="playbtn" id="play" disabled>▶</button>
      <div class="pcol">
        <div class="clock"><span id="cur">0:00</span><span class="d"> / <span id="tot">0:00</span></span></div>
        <input type="range" id="seek" min="0" max="1000" value="0">
        <div class="volrow"><span>Vol</span><input type="range" id="vol" min="0" max="100" value="100"></div>
      </div>
    </div>
    <div class="nowbig" id="nowchord">—</div>
    <div class="nowsub" id="nowsub">current chord</div>
  </div>

  <div class="card">
    <h3>Waveform</h3>
    <div class="wfscroll" id="wfscroll"><div class="wftrack" id="wftrack">
      <canvas id="wf"></canvas><div id="playhead"></div>
    </div></div>
    <div class="wfnote" id="wfnote" style="display:none"></div>
  </div>

  <div class="card">
    <h3>Chords</h3>
    <div class="clist" id="clist"></div>
  </div>
</div>

<div class="savebar">
  <button class="save" id="save" disabled>Save changes</button>
  <div class="stat" id="stat">0 edited</div>
</div>
<div class="toast" id="toast"></div>

<audio id="audio" preload="auto" playsinline crossorigin="anonymous"></audio>

<script>
const D = __ANNOT_DATA__;
const PPS = 60;                 // px per second on the waveform
const chords = D.chords.map(c=>({ ...c, t0:+c.t0, t1:+c.t1, _o0:+c.t0, _o1:+c.t1, dirty:false }));
let duration = Math.max(D.duration||0, chords.length?chords[chords.length-1].t1:0) || 1;
let selIdx = -1, wfReady = false;

const $ = id=>document.getElementById(id);
const audio = $('audio');
const fmt = s=>{ s=Math.max(0,s||0); const m=Math.floor(s/60), ss=Math.floor(s%60); return m+':'+String(ss).padStart(2,'0'); };
function step(id, ok, txt){ const e=$(id); if(!e) return; e.className = ok?'done':'fail'; e.textContent=(ok?'✓ ':'✗ ')+txt; }

// ---------- LOAD SEQUENCE ----------
$('title').textContent = D.title;
$('ltitle').textContent = D.title;
step('s-chords', true, chords.length+' chords loaded');

function finishLoad(){ setTimeout(()=>{ $('loader').style.display='none'; }, 250); }

if(!D.audioUrl){
  step('s-audio', false, 'no audio for this song');
  step('s-wave', false, 'skipped (no audio)');
  $('play').disabled = true;
  $('wfnote').style.display='block'; $('wfnote').textContent='Audio not available — timings still editable.';
  buildAll(); finishLoad();
}else{
  audio.src = D.audioUrl;
  audio.addEventListener('loadedmetadata', ()=>{
    if(isFinite(audio.duration) && audio.duration>0) duration = Math.max(duration, audio.duration);
    step('s-audio', true, 'audio ready ('+fmt(duration)+')');
    $('play').disabled = false; $('tot').textContent = fmt(duration);
    buildAll();
    decodeWaveform();     // independent of playback
  }, {once:true});
  audio.addEventListener('error', ()=>{
    step('s-audio', false, 'audio failed to load');
    $('play').disabled = true;
    $('wfnote').style.display='block'; $('wfnote').textContent='Audio playback not available.';
    buildAll(); finishLoad();
  }, {once:true});
  // safety: never hang the loader forever
  setTimeout(()=>{ if($('loader').style.display!=='none' && !$('play').disabled) finishLoad(); }, 6000);
}

// ---------- WAVEFORM DECODE (visual only; playback uses <audio>) ----------
async function decodeWaveform(){
  const track = $('wftrack'), cv = $('wf');
  const W = Math.max(Math.ceil(duration*PPS), 300), H = 120;
  track.style.width = W+'px'; cv.width = W; cv.height = H;
  try{
    const buf = await fetch(D.audioUrl).then(r=>r.arrayBuffer());
    const AC = window.AudioContext || window.webkitAudioContext;
    const ac = new AC();
    const audioBuf = await new Promise((res,rej)=>{
      const p = ac.decodeAudioData(buf, res, rej); if(p&&p.then) p.then(res,rej);
    });
    ac.close && ac.close();
    drawPeaks(cv, audioBuf);
    wfReady = true; step('s-wave', true, 'waveform ready');
  }catch(e){
    console.warn('waveform decode failed', e);
    step('s-wave', false, 'waveform unavailable');
    drawFlat(cv);
    $('wfnote').style.display='block'; $('wfnote').textContent='Waveform preview unavailable (drag still works).';
  }
  finishLoad(); layoutRegions();
}
function drawPeaks(cv, ab){
  const ctx = cv.getContext('2d'); const W=cv.width, H=cv.height, mid=H/2;
  const ch = ab.getChannelData(0); const per = Math.max(1, Math.floor(ch.length/W));
  ctx.clearRect(0,0,W,H); ctx.fillStyle='#4b5666';
  for(let x=0;x<W;x++){ let mn=1,mx=-1; const s=x*per;
    for(let i=0;i<per;i++){ const v=ch[s+i]||0; if(v<mn)mn=v; if(v>mx)mx=v; }
    const y1=mid+mn*mid*0.92, y2=mid+mx*mid*0.92;
    ctx.fillRect(x, y1, 1, Math.max(1,y2-y1)); }
}
function drawFlat(cv){ const ctx=cv.getContext('2d'); ctx.strokeStyle='#3a4452'; ctx.beginPath();
  ctx.moveTo(0,cv.height/2); ctx.lineTo(cv.width,cv.height/2); ctx.stroke(); }

// ---------- BUILD UI ----------
function buildAll(){ buildRegions(); buildList(); layoutRegions(); updateStat(); $('tot').textContent=fmt(duration);
  const t=$('wftrack'); if(t.style.width==='') t.style.width=Math.max(Math.ceil(duration*PPS),300)+'px';
}

function buildRegions(){
  const track = $('wftrack');
  [...track.querySelectorAll('.region')].forEach(e=>e.remove());
  chords.forEach((c,i)=>{
    const r = document.createElement('div'); r.className='region'; r.dataset.i=i;
    r.innerHTML = '<span>'+esc(c.label||'?')+'</span>'+
      '<div class="handle l" data-edge="l"></div><div class="handle r" data-edge="r"></div>';
    r.addEventListener('click', ev=>{ if(ev.target.classList.contains('handle'))return; select(i,true); });
    r.querySelector('.handle.l').addEventListener('pointerdown', ev=>startDrag(ev,i,'l'));
    r.querySelector('.handle.r').addEventListener('pointerdown', ev=>startDrag(ev,i,'r'));
    track.appendChild(r);
  });
}
function layoutRegions(){
  const track=$('wftrack');
  chords.forEach((c,i)=>{
    const r=track.querySelector('.region[data-i="'+i+'"]'); if(!r) return;
    r.style.left=(c.t0*PPS)+'px'; r.style.width=Math.max(6,(c.t1-c.t0)*PPS)+'px';
    r.classList.toggle('dirty', c.dirty); r.classList.toggle('sel', i===selIdx);
  });
}

function buildList(){
  const list=$('clist'); list.innerHTML='';
  chords.forEach((c,i)=>{
    const row=document.createElement('div'); row.className='crow'; row.dataset.i=i;
    row.innerHTML =
      '<button class="jump" title="jump here">▸</button>'+
      '<div class="lab">'+esc(c.label||'?')+'</div>'+
      '<div class="t">'+c.t0.toFixed(2)+'s</div>'+
      '<button class="nudge" data-d="-1">−</button>'+
      '<button class="nudge" data-d="1">+</button>';
    row.querySelector('.jump').addEventListener('click', ()=>{ seekTo(c.t0); select(i,true); });
    row.querySelector('.lab').addEventListener('click', ()=>select(i,true));
    row.querySelectorAll('.nudge').forEach(b=>b.addEventListener('click',()=>{
      nudge(i, (+b.dataset.d)*0.1); }));
    list.appendChild(row);
  });
  refreshList();
}
function refreshList(){
  const t = audio.currentTime||0;
  [...$('clist').children].forEach(row=>{
    const i=+row.dataset.i, c=chords[i];
    row.querySelector('.t').textContent=c.t0.toFixed(2)+'s';
    row.classList.toggle('dirty', c.dirty);
    row.classList.toggle('sel', i===selIdx);
    row.classList.toggle('playing', t>=c.t0 && t<c.t1);
  });
}

// ---------- EDITING ----------
function markDirty(i){ if(i>=0&&i<chords.length){ const c=chords[i];
  c.dirty = Math.abs(c.t0-c._o0)>1e-3 || Math.abs(c.t1-c._o1)>1e-3; } }
function setBoundary(i, edge, t){
  // clamp within neighbours; keep contiguous with the adjacent chord
  const lo = edge==='l' ? (i>0?chords[i-1].t0+0.05:0) : (chords[i].t0+0.05);
  const hi = edge==='l' ? (chords[i].t1-0.05) : (i<chords.length-1?chords[i+1].t1-0.05:duration);
  t = Math.min(Math.max(t, lo), hi);
  if(edge==='l'){ chords[i].t0=t; if(i>0){ chords[i-1].t1=t; markDirty(i-1);} }
  else{ chords[i].t1=t; if(i<chords.length-1){ chords[i+1].t0=t; markDirty(i+1);} }
  markDirty(i);
}
function nudge(i, d){ setBoundary(i,'l', chords[i].t0 + d); layoutRegions(); refreshList(); updateStat(); select(i,false); }

let drag=null;
function startDrag(ev,i,edge){
  ev.preventDefault(); ev.stopPropagation();
  const h=ev.currentTarget; h.classList.add('drag'); h.setPointerCapture(ev.pointerId);
  drag={i,edge,h}; select(i,false);
  h.addEventListener('pointermove', onDrag); h.addEventListener('pointerup', endDrag);
  h.addEventListener('pointercancel', endDrag);
}
function onDrag(ev){ if(!drag) return; ev.preventDefault();
  const rect=$('wftrack').getBoundingClientRect();
  const x = ev.clientX - rect.left;   // track is untransformed; scroll handled by offset already in clientX
  setBoundary(drag.i, drag.edge, x/PPS);
  layoutRegions(); refreshList(); updateStat();
}
function endDrag(ev){ if(!drag) return; drag.h.classList.remove('drag');
  drag.h.removeEventListener('pointermove', onDrag); drag.h.removeEventListener('pointerup', endDrag);
  drag.h.removeEventListener('pointercancel', endDrag); drag=null; }

function select(i, scroll){ selIdx=i; layoutRegions(); refreshList();
  if(scroll){ const sc=$('wfscroll'); const x=chords[i].t0*PPS;
    sc.scrollTo({left:Math.max(0,x-sc.clientWidth/3), behavior:'smooth'}); } }

function updateStat(){ const n=chords.filter(c=>c.dirty).length;
  $('stat').textContent=n+' edited'; $('save').disabled = n===0; }

// ---------- PLAYBACK ----------
$('play').addEventListener('click', ()=>{ if(audio.paused){ audio.play().catch(()=>{}); } else audio.pause(); });
audio.addEventListener('play', ()=>{ $('play').textContent='❚❚'; tick(); });
audio.addEventListener('pause',()=>{ $('play').textContent='▶'; });
audio.addEventListener('ended',()=>{ $('play').textContent='▶'; });
$('seek').addEventListener('input', e=>{ if(isFinite(audio.duration)) seekTo(audio.duration*e.target.value/1000); });
$('vol').addEventListener('input', e=>{ audio.volume=e.target.value/100; });
function seekTo(t){ if(isFinite(audio.duration)) audio.currentTime=Math.min(Math.max(0,t),audio.duration); updateNow(); }

let raf=0;
function tick(){ cancelAnimationFrame(raf); const loop=()=>{ updateNow(); if(!audio.paused) raf=requestAnimationFrame(loop); }; loop(); }
function updateNow(){
  const t=audio.currentTime||0;
  $('cur').textContent=fmt(t);
  if(isFinite(audio.duration)&&audio.duration>0) $('seek').value=Math.round(t/audio.duration*1000);
  $('playhead').style.left=(t*PPS)+'px';
  // auto-scroll waveform to follow playhead
  const sc=$('wfscroll'), px=t*PPS;
  if(px < sc.scrollLeft+20 || px > sc.scrollLeft+sc.clientWidth-60) sc.scrollLeft = px - sc.clientWidth*0.4;
  const cur = chords.find(c=>t>=c.t0 && t<c.t1);
  $('nowchord').textContent = cur?cur.label:'—';
  refreshList();
}

// ---------- SAVE ----------
fetch('/api/annotations/'+encodeURIComponent(D.saveFile)).then(r=>r.json()).then(doc=>{
  if(!$('who').value && doc.annotator) $('who').value=doc.annotator;
  const by={}; (doc.chords||[]).forEach(c=>{ if('t0' in c) by[c.bar+':'+c.beat]=c; });
  let resumed=0;
  chords.forEach((c,i)=>{ const p=by[c.bar+':'+c.beat]; if(p){ c.t0=+p.t0; if('t1' in p)c.t1=+p.t1;
    c._o0=+p.t0; c._o1=('t1' in p)?+p.t1:c._o1; if(i>0)chords[i-1].t1=c.t0; resumed++; } });
  if(resumed){ layoutRegions(); buildList(); updateStat(); }
}).catch(()=>{});

$('save').addEventListener('click', async ()=>{
  const now=new Date().toISOString();
  const body={ annotator:($('who').value.trim()||'anon'),
    chords: chords.map(c=>({ bar:c.bar, beat:c.beat, label:c.label, section:c.section,
      t0:+c.t0.toFixed(3), t1:+c.t1.toFixed(3), ts:now })), merges:[] };
  $('save').disabled=true;
  try{
    const doc=await fetch('/api/annotations/'+encodeURIComponent(D.saveFile),
      {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).then(r=>r.json());
    chords.forEach(c=>{ c._o0=c.t0; c._o1=c.t1; c.dirty=false; });
    layoutRegions(); refreshList(); updateStat();
    toast('✓ Saved '+(doc.chords?doc.chords.length:chords.length)+' chords');
  }catch(e){ toast('Save failed — check connection', true); }
  finally{ updateStat(); }
});

// ---------- misc ----------
let toastT; function toast(m, err){ const e=$('toast'); e.textContent=m; e.className='toast on'+(err?' err':'');
  clearTimeout(toastT); toastT=setTimeout(()=>e.className='toast'+(err?' err':''),2400); }
function esc(s){ return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c])); }
window.addEventListener('resize', layoutRegions);
</script>
</body></html>"""


# ── Manual chord-alignment annotator (iPhone-first) ─────────────────────────────
#
# GET /annotator?song=<slug>
#   Loads the iReal GT chords + their Mission-1 initial alignment (DTW t0/t1
#   from docs/plots/irealb_<slug>.html) and the song audio, and serves a
#   touch-optimised page where the user drags/snaps each chord boundary to the
#   audio and saves the corrected times. Save reuses POST /api/annotations/
#   (non-destructive merge — existing quality corrections are preserved).

BEATGRID_CACHE = REPO / "data" / "cache" / "beat_grid"
WAVEFORM_CACHE = REPO / "data" / "cache" / "waveform_peaks"


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
    slug = re.sub(r"[^A-Za-z0-9_]", "", song or "")
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

    slug = re.sub(r"[^A-Za-z0-9_]", "", song or "")
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

        return jsonify({
            "song": slug, "duration_s": duration_s,
            "tempo_bpm_stock": tempo_bpm, "tempo_bpm_bestfit": 60.0 / period_best,
            "raw_beat_times": [round(float(t), 4) for t in raw_beats],
            "uniform_grid_times": [round(float(t), 4) for t in uniform_grid],
            "bestfit_grid_times": [round(float(t), 4) for t in bestfit_grid],
            "displayed_chords": displayed,
            "downbeat_times": downbeats,
            "audio_url": f"/audio/{slug}.m4a",
        })
    except Exception as e:
        log.exception(f"grid-align-data error for {slug}")
        return jsonify(error=str(e)), 500


@app.route("/debug/grid-align")
def debug_grid_align():
    """Interactive audio+waveform grid-alignment diagnostic (2026-07-20).

    Lets the user personally listen through any cached song with FOUR
    overlaid hypotheses (raw detected beats / stock uniform grid / bestfit
    grid / the chart actually shown today) and a synced, click-to-seek
    playhead — so "the grid still looks wrong" can be confirmed or refuted
    by eye+ear on the real audio, not by a self-reported offline metric."""
    if not _GRID_ALIGN_HTML.exists():
        return Response("grid-align page not found", status=404)
    return Response(_GRID_ALIGN_HTML.read_bytes(), mimetype="text/html")


@app.route("/api/beat-grid-audio/<song>")
def api_beat_grid_audio(song):
    """Detected beat times + tempo from audio for waveform V4 beat-grid editor.

    Returns {beat_times: [...], tempo_bpm: X, duration_s: Y, n_bars: Z}
    """
    import librosa
    import librosa.beat
    import numpy as np

    slug = re.sub(r"[^A-Za-z0-9_]", "", song or "")
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

    slug = re.sub(r"[^A-Za-z0-9_]", "", song or "")
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

ANNOTATOR_V4_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no,viewport-fit=cover">
<title>Harmonia — Waveform Annotator v4</title>
<style>
  :root { --bg:#0e1116; --panel:#171c24; --panel2:#1e2530; --ink:#e8edf4; --faint:#8b97a8;
    --line:#2a3340; --teal:#00c9a7; --amber:#ffb454; --accent:#6ea8ff; --danger:#ff5d6c; --ok:#37d67a; }
  * { box-sizing:border-box; -webkit-tap-highlight-color:transparent; user-select:none; }
  html,body { margin:0; background:var(--bg); color:var(--ink); overflow-x:hidden;
    font-family:-apple-system,BlinkMacSystemFont,system-ui,sans-serif; }
  body { padding-bottom:calc(120px + env(safe-area-inset-bottom)); }

  header { position:sticky; top:0; z-index:20; background:var(--bg); border-bottom:1px solid var(--line);
    padding:calc(8px + env(safe-area-inset-top)) 12px 8px; }
  .hrow { display:flex; align-items:center; gap:10px; }
  .hrow a { color:var(--faint); font:600 17px system-ui; padding:6px 8px; text-decoration:none; }
  .hrow h1 { font-size:15px; margin:0; font-weight:700; flex:1; }
  .status { font-size:12px; color:var(--faint); }
  .status.locked { color:var(--ok); }
  .status.editing { color:var(--amber); }

  #waveContainer { position:relative; height:140px; background:var(--panel); border-bottom:1px solid var(--line);
    overflow-x:auto; overflow-y:hidden; }
  #canvas { display:block; }

  .beatMarker { position:absolute; width:3px; height:100%; background:rgba(255,180,84,0.3); opacity:0.5; z-index:5; }
  .beatMarker.downbeat { background:var(--amber); opacity:0.7; }

  .chordMarker { position:absolute; width:10px; height:28px; top:50%; transform:translate(-50%,-50%);
    background:var(--teal); border-radius:2px; cursor:pointer; z-index:10; border:2px solid transparent;
    transition:all 0.1s; }
  .chordMarker.locked { background:var(--danger); opacity:0.5; cursor:not-allowed; }
  .chordMarker.selected { border-color:var(--accent); }

  #controls { display:flex; gap:8px; padding:12px; background:var(--panel); border-bottom:1px solid var(--line);
    overflow-x:auto; flex-wrap:wrap; }
  button { padding:8px 16px; background:var(--panel2); border:1px solid var(--line); color:var(--ink);
    border-radius:4px; font:600 13px system-ui; cursor:pointer; white-space:nowrap; }
  button:active { background:var(--accent); }
  button:disabled { opacity:0.5; cursor:not-allowed; }

  audio { width:100%; }

  #info { padding:12px; background:var(--panel2); font-size:12px; color:var(--faint); line-height:1.5; }
  #info.stage2 { background:var(--panel); }

  #modal { display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.7);
    z-index:100; align-items:flex-end; }
  #modal.on { display:flex; }
  #sheet { background:var(--panel); width:100%; border-radius:12px 12px 0 0; padding:16px;
    padding-bottom:calc(16px + env(safe-area-inset-bottom)); }
  #sheet h2 { margin:0 0 16px; font-size:16px; }
  #roots, #quals { display:grid; grid-template-columns:repeat(6,1fr); gap:8px; margin-bottom:16px; }
  #roots button, #quals button { padding:12px; }
  #roots button.on, #quals button.on { background:var(--accent); color:var(--bg); }
  #sheet-footer { display:flex; gap:8px; margin-top:16px; }
  #sheet-footer button { flex:1; }
</style>
</head><body>

<header>
  <div class="hrow">
    <a href="/">←</a>
    <h1 id="title">Annotator v4</h1>
    <span id="status" class="status">…</span>
  </div>
</header>

<div id="waveContainer">
  <canvas id="canvas"></canvas>
</div>

<div id="controls">
  <button id="lockBtn" disabled>🔓 Correct & Infer</button>
  <button id="addBtn" style="display:none;">➕ Add Chord</button>
  <button id="resetBtn">↻ Reset Beats</button>
  <button id="saveBtn" style="display:none;">💾 Save</button>
  <button id="zoomIn">🔍+ Zoom</button>
  <button id="zoomOut">🔍- Zoom</button>
</div>

<div id="info">
  <div id="stage">🎵 <span id="song">—</span> | Drag beat markers to correct, then tap "Correct & Infer"</div>
</div>

<audio id="audio" crossOrigin="anonymous" playsinline controls></audio>

<div id="modal">
  <div id="sheet">
    <h2>Relabel Chord</h2>
    <div>Now: <strong id="mprev">—</strong></div>
    <div style="margin:16px 0;">Root:</div>
    <div id="roots"></div>
    <div style="margin:16px 0;">Quality:</div>
    <div id="quals"></div>
    <div id="sheet-footer">
      <button id="mapply">Apply</button>
      <button id="mdel" style="color:var(--danger);">Delete</button>
      <button id="mcancel">Cancel</button>
    </div>
  </div>
</div>

<script>
const D = __ANNOT_DATA__;
const $=(id)=>document.getElementById(id);
const log=console.log;

// ──── STATE ────
let mode='beatEditor'; // 'beatEditor' | 'chordEditor'
let beatTimes=[], beatTimesOrig=[];
let chords=[];
let tempo=120, duration=0;
let scale=60; // px/sec
let dragBeat=null;
let editChordIdx=-1;
let selRoot='C', selQual='';

const ROOTS=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
const QUALS=['','m','7','maj7','m7','m7b5','dim','aug'];

const canvas=$('canvas'), ctx=canvas.getContext('2d');
const audio=$('audio');

// ──── INIT ────
async function init(){
  $('song').textContent=D.title||'Song';
  audio.src=D.audioUrl;
  duration=D.duration||0;

  try{
    const r=await fetch('/api/beat-grid-audio/'+encodeURIComponent(D.slug));
    const grid=await r.json();
    beatTimes=grid.beat_times||[];
    beatTimesOrig=[...beatTimes];
    tempo=grid.tempo_bpm||120;
    updateUI();
    draw();
  }catch(e){ log('Error loading beats:',e); }
}

function updateUI(){
  const st=$('status');
  if(mode==='beatEditor'){
    st.textContent=`${beatTimes.length} beats`;
    st.classList.add('editing');
    st.classList.remove('locked');
    $('lockBtn').style.display='block';
    $('addBtn').style.display='none';
    $('saveBtn').style.display='none';
    $('info').classList.remove('stage2');
    $('stage').textContent='🎵 '+D.title+' | Tap "Correct & Infer" when ready';
  }else{
    st.textContent='✓ Chord editing';
    st.classList.add('locked');
    st.classList.remove('editing');
    $('lockBtn').style.display='none';
    $('addBtn').style.display='block';
    $('saveBtn').style.display='block';
    $('info').classList.add('stage2');
    $('stage').textContent='🎚️ Tap chord to relabel, tap area to add. Press & hold to delete.';
  }
}

function draw(){
  const w=Math.max(300, duration*scale);
  canvas.width=w;
  canvas.height=140;

  // Waveform bg
  const grad=ctx.createLinearGradient(0,0,w,0);
  grad.addColorStop(0,'#2a3340'); grad.addColorStop(0.5,'#4a5a70'); grad.addColorStop(1,'#2a3340');
  ctx.fillStyle=grad; ctx.fillRect(0,0,w,140);

  if(mode==='beatEditor'){
    // Draw beat grid
    beatTimes.forEach((t,i)=>{
      const x=t*scale;
      ctx.fillStyle=(i%4===0)?'#ffb454':'rgba(255,180,84,0.3)';
      ctx.fillRect(x-1,0,3,140);
    });

    // Labels for every 4 beats (bar)
    ctx.fillStyle='#8b97a8'; ctx.font='11px system-ui';
    for(let i=0;i<beatTimes.length;i+=4){
      const x=beatTimes[i]*scale;
      ctx.fillText('B'+(i/4|0), x+4, 20);
    }
  }else{
    // Draw beat grid lightly + chords
    beatTimes.forEach((t,i)=>{
      const x=t*scale;
      ctx.fillStyle=(i%4===0)?'rgba(255,180,84,0.2)':'rgba(255,180,84,0.1)';
      ctx.fillRect(x-1,0,2,140);
    });

    // Draw chord spans
    ctx.fillStyle='rgba(0,201,167,0.15)';
    for(let i=0;i<chords.length;i++){
      const c=chords[i], cn=chords[i+1];
      const x0=c.t*scale, x1=(cn?cn.t:duration)*scale;
      ctx.fillRect(x0,55,x1-x0,30);
    }
  }

  ctx.strokeStyle='#8b97a8'; ctx.lineWidth=1;
  ctx.beginPath(); ctx.moveTo(0,70); ctx.lineTo(w,70); ctx.stroke();
}

canvas.addEventListener('pointerdown',e=>{
  if(mode!=='beatEditor') return;
  const rect=canvas.getBoundingClientRect();
  const x=e.clientX-rect.left;
  const t=x/scale;

  let best=-1, bestDist=Infinity;
  beatTimes.forEach((bt,i)=>{
    const dx=Math.abs(bt*scale-x);
    if(dx<15 && dx<bestDist){ best=i; bestDist=dx; }
  });

  if(best>=0){ dragBeat=best; }
});

canvas.addEventListener('pointermove',e=>{
  if(dragBeat==null) return;
  const rect=canvas.getBoundingClientRect();
  const x=e.clientX-rect.left;
  const t=x/scale;
  const orig=beatTimesOrig[dragBeat];
  beatTimes[dragBeat]=Math.max(orig-0.2, Math.min(orig+0.2, t));
  draw();
});

canvas.addEventListener('pointerup',()=>{ dragBeat=null; });

// ──── BEAT CORRECTION ────
$('lockBtn').addEventListener('click',async()=>{
  $('lockBtn').disabled=true;
  try{
    const r=await fetch('/api/reinfer-from-beats/'+encodeURIComponent(D.slug),{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify({
        corrected_beat_times: beatTimes.slice(0,Math.min(8,beatTimes.length)),
        n_locked_beats: Math.min(8,beatTimes.length),
        tempo_bpm: tempo
      })
    });
    const result=await r.json();
    beatTimes=result.beat_times||beatTimes;
    chords=(D.chords||[]).map((c,i)=>({
      t: c.t0,
      label: c.label,
      dirty: false,
      _orig: c.label
    }));
    mode='chordEditor';
    updateUI();
    draw();
  }catch(e){ alert('Infer failed: '+e.message); }finally{
    $('lockBtn').disabled=false;
  }
});

// ──── CHORD EDITING ────
canvas.addEventListener('click',e=>{
  if(mode!=='chordEditor') return;
  const rect=canvas.getBoundingClientRect();
  const x=e.clientX-rect.left;
  const t=x/scale;

  let best=-1, bestDist=Infinity;
  chords.forEach((c,i)=>{
    const dx=Math.abs(c.t*scale-x);
    if(dx<20 && dx<bestDist){ best=i; bestDist=dx; }
  });

  if(best>=0){ openEditor(best); }
  else { addChordAt(t); }
});

function addChordAt(t){
  if(t<0.1){alert('Cannot add at start'); return;}
  const prev=chords.find(c=>c.t<t), next=chords.find(c=>c.t>t);
  const label=prev?prev.label:'C';
  chords.push({t, label, dirty:true});
  chords.sort((a,b)=>a.t-b.t);
  draw();
}

function openEditor(i){
  editChordIdx=i;
  const p=(chords[i].label||'').match(/^([A-G][#b]?)(.*)$/)||['','C',''];
  selRoot=p[1].replace('b','');
  selQual=p[2];
  drawSheet();
  $('modal').classList.add('on');
}

function drawSheet(){
  $('mprev').textContent=selRoot+selQual;
  $('roots').innerHTML=ROOTS.map(r=>\`<button data-r="\${r}" class="\${r===selRoot?'on':''}">\${r}</button>\`).join('');
  $('quals').innerHTML=QUALS.map(q=>\`<button data-q="\${q}" class="\${q===selQual?'on':''}">\${q||'maj'}</button>\`).join('');
}

$('roots').addEventListener('click',e=>{
  const b=e.target.closest('button');
  if(b){selRoot=b.dataset.r; drawSheet();}
});

$('quals').addEventListener('click',e=>{
  const b=e.target.closest('button');
  if(b){selQual=b.dataset.q; drawSheet();}
});

$('mapply').addEventListener('click',()=>{
  chords[editChordIdx].label=selRoot+selQual;
  chords[editChordIdx].dirty=true;
  draw();
  $('modal').classList.remove('on');
});

$('mdel').addEventListener('click',()=>{
  if(editChordIdx<=0){ alert('Cannot delete first chord'); $('modal').classList.remove('on'); return; }
  chords.splice(editChordIdx,1);
  draw();
  $('modal').classList.remove('on');
});

$('mcancel').addEventListener('click',()=>{ $('modal').classList.remove('on'); });

$('saveBtn').addEventListener('click',async()=>{
  const now=new Date().toISOString();
  const body={
    annotator:'v4-user',
    chords:chords.map(c=>({t:c.t, label:c.label, ts:now})),
    merges:[]
  };
  try{
    await fetch('/api/annotations/'+encodeURIComponent(D.saveFile),{
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body:JSON.stringify(body)
    });
    alert('✓ Saved '+chords.length+' chords');
  }catch(e){alert('Save failed: '+e.message);}
});

$('resetBtn').addEventListener('click',()=>{ beatTimes=[...beatTimesOrig]; draw(); });
$('zoomIn').addEventListener('click',()=>{ scale*=1.4; draw(); });
$('zoomOut').addEventListener('click',()=>{ scale/=1.4; draw(); });

init();
</script>
</body></html>
"""

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

ANNOTATOR_V3_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,maximum-scale=1,user-scalable=no,viewport-fit=cover">
<title>Harmonia — Waveform Annotator v3</title>
<style>
  :root{ --bg:#0e1116; --panel:#171c24; --panel2:#1e2530; --ink:#e8edf4; --faint:#8b97a8;
    --line:#2a3340; --teal:#00c9a7; --amber:#ffb454; --accent:#6ea8ff; --danger:#ff5d6c; --ok:#37d67a;
    --bar:#3a4658; --db:#cfd8e6; }
  *{box-sizing:border-box;-webkit-tap-highlight-color:transparent;-webkit-user-select:none;user-select:none;}
  html,body{margin:0;background:var(--bg);color:var(--ink);overscroll-behavior:none;overflow-x:hidden;max-width:100%;
    font-family:-apple-system,BlinkMacSystemFont,system-ui,sans-serif;}
  body{padding-bottom:calc(84px + env(safe-area-inset-bottom));}
  a{color:var(--accent);text-decoration:none;}
  header{position:sticky;top:0;z-index:20;background:#0e1116;border-bottom:1px solid var(--line);
    padding:calc(8px + env(safe-area-inset-top)) 12px 8px;}
  .hrow{display:flex;align-items:center;gap:10px;}
  .hrow a.back{color:var(--faint);font:600 17px system-ui;padding:6px 8px;margin-left:-8px;}
  .hrow h1{font-size:15px;margin:0;font-weight:700;flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
  input.who{width:88px;background:var(--panel2);border:1px solid var(--line);color:var(--ink);
    border-radius:8px;padding:6px 8px;font:600 12px system-ui;}
  .sub{display:flex;gap:6px;margin-top:6px;font:600 11px system-ui;color:var(--faint);flex-wrap:wrap;}
  .pill{background:var(--panel2);border:1px solid var(--line);border-radius:20px;padding:3px 9px;}
  .pill.src{color:var(--teal);border-color:#0b3d35;}
  /* transport */
  .transport{display:flex;align-items:center;gap:12px;padding:10px 12px 6px;}
  .playbtn{width:52px;height:52px;flex:none;border:none;border-radius:50%;background:var(--teal);color:#052;
    font-size:22px;display:flex;align-items:center;justify-content:center;box-shadow:0 4px 14px #00c9a733;}
  .playbtn:active{transform:scale(.94);} .playbtn[disabled]{background:var(--panel2);color:var(--faint);box-shadow:none;}
  .clock{font:700 14px ui-monospace,Menlo,monospace;min-width:104px;} .clock .d{color:var(--faint);}
  .nowchord{margin-left:auto;font:800 20px 'SF Mono',ui-monospace,Menlo,monospace;color:var(--teal);
    min-width:56px;text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:40vw;}
  /* timeline */
  .tlwrap{background:var(--panel);border-top:1px solid var(--line);border-bottom:1px solid var(--line);overflow:hidden;}
  .tlscroll{overflow-x:auto;overflow-y:hidden;-webkit-overflow-scrolling:touch;touch-action:pan-x;}
  .timeline{position:relative;height:240px;touch-action:none;}
  canvas#wf{position:absolute;left:0;top:0;z-index:0;display:block;}
  #grid,#chords,#playhead,#addcue{position:absolute;top:0;left:0;height:100%;}
  #grid{z-index:1;pointer-events:none;} #chords{z-index:2;}
  .cspan{position:absolute;top:24px;height:40px;border-radius:8px;display:flex;align-items:center;
    justify-content:center;overflow:hidden;border:1px solid #0006;box-shadow:0 1px 3px #0006;touch-action:none;}
  .cspan span{font:700 12px 'SF Mono',ui-monospace,Menlo,monospace;color:#08110e;padding:0 10px;white-space:nowrap;
    overflow:hidden;text-overflow:ellipsis;pointer-events:none;text-shadow:0 1px 0 #fff5;}
  .cspan.dirty{outline:2px solid var(--amber);outline-offset:-2px;}
  /* draggable boundary between two chords */
  .bnd{position:absolute;top:16px;bottom:0;width:30px;margin-left:-15px;z-index:3;touch-action:none;
    display:flex;justify-content:center;cursor:ew-resize;}
  .bnd i{width:3px;height:100%;background:var(--teal);opacity:.85;border-radius:2px;box-shadow:0 0 4px #00c9a780;}
  .bnd.drag i{opacity:1;width:4px;box-shadow:0 0 10px var(--teal);}
  .bnd.snap i{background:var(--amber);box-shadow:0 0 10px var(--amber);}
  #playhead{width:2px;background:var(--danger);z-index:5;pointer-events:none;box-shadow:0 0 6px var(--danger);
    will-change:transform;}
  #addcue{width:2px;margin-left:-1px;background:var(--accent);z-index:4;opacity:0;pointer-events:none;
    box-shadow:0 0 6px var(--accent);}
  #addcue.on{opacity:.9;}
  .bandlbl{position:absolute;left:5px;font:700 8px system-ui;color:#ffffff88;background:#0009;padding:1px 5px;
    border-radius:4px;letter-spacing:.4px;z-index:6;pointer-events:none;}
  /* controls */
  .tools{display:flex;gap:8px;padding:8px 12px 4px;flex-wrap:wrap;align-items:center;}
  .tools button{border:1px solid var(--line);background:var(--panel2);color:var(--ink);border-radius:9px;
    padding:8px 11px;font:700 12px system-ui;display:flex;align-items:center;gap:5px;min-height:40px;}
  .tools button.on{background:#0b3d35;border-color:var(--teal);color:var(--teal);}
  .tools .zoom{margin-left:auto;display:flex;gap:6px;}
  .tools .zoom button{width:40px;justify-content:center;font:800 16px ui-monospace;}
  .hint{color:var(--faint);font:500 11.5px system-ui;padding:4px 12px 8px;line-height:1.5;}
  .hint b{color:var(--ink);}
  /* save bar */
  .savebar{position:fixed;left:0;right:0;bottom:0;z-index:30;
    padding:10px 12px calc(10px + env(safe-area-inset-bottom));
    background:linear-gradient(0deg,#0e1116 65%,#0e1116cc);border-top:1px solid var(--line);
    display:flex;gap:10px;align-items:center;}
  .save{flex:1;min-height:52px;border:none;border-radius:14px;background:var(--teal);color:#052;
    font:800 16px system-ui;} .save:active{transform:scale(.98);} .save[disabled]{opacity:.5;}
  .stat{font:700 12px system-ui;color:var(--faint);min-width:70px;text-align:right;}
  /* relabel sheet */
  .modal{position:fixed;inset:0;z-index:50;background:#0009;display:none;align-items:flex-end;}
  .modal.on{display:flex;}
  .sheet{width:100%;background:var(--panel);border-radius:18px 18px 0 0;border-top:1px solid var(--line);
    padding:14px 14px calc(16px + env(safe-area-inset-bottom));max-height:82vh;overflow:auto;}
  .sheet h3{margin:0 0 2px;font:800 15px system-ui;} .sheet .prev{font:800 22px 'SF Mono',ui-monospace;
    color:var(--teal);margin:2px 0 12px;}
  .grp{font:700 10px system-ui;color:var(--faint);text-transform:uppercase;letter-spacing:.6px;margin:10px 0 6px;}
  .keys{display:grid;grid-template-columns:repeat(6,1fr);gap:6px;} .keys.q{grid-template-columns:repeat(4,1fr);}
  .keys button{min-height:44px;border:1px solid var(--line);background:var(--panel2);color:var(--ink);
    border-radius:10px;font:700 13px ui-monospace;}
  .keys button.on{background:var(--teal);color:#052;border-color:var(--teal);}
  .srow{display:flex;gap:10px;margin-top:14px;} .srow button{flex:1;min-height:48px;border-radius:12px;
    font:800 14px system-ui;border:1px solid var(--line);background:var(--panel2);color:var(--ink);}
  .srow button.apply{background:var(--teal);color:#052;border-color:var(--teal);} .srow button.del{color:var(--danger);}
  /* toast + loader */
  .toast{position:fixed;left:50%;bottom:98px;transform:translateX(-50%) translateY(16px);opacity:0;
    background:var(--ok);color:#052;font:800 13px system-ui;padding:11px 18px;border-radius:22px;z-index:60;
    transition:.25s;pointer-events:none;max-width:90vw;text-align:center;box-shadow:0 6px 20px #0008;}
  .toast.on{opacity:1;transform:translateX(-50%) translateY(0);} .toast.err{background:var(--danger);color:#fff;}
  #loader{position:fixed;inset:0;z-index:100;background:var(--bg);display:flex;flex-direction:column;
    align-items:center;justify-content:center;gap:14px;color:var(--faint);font:600 13px ui-monospace;}
  #loader .spin{width:34px;height:34px;border:3px solid var(--line);border-top-color:var(--teal);border-radius:50%;
    animation:sp .8s linear infinite;} @keyframes sp{to{transform:rotate(360deg);}}
</style></head><body>
<div id="loader"><div class="spin"></div><div id="lmsg">Loading waveform…</div></div>

<header>
  <div class="hrow">
    <a class="back" href="/">&lsaquo;</a>
    <h1 id="ttl">Annotator</h1>
    <input class="who" id="who" placeholder="your name" autocomplete="off">
  </div>
  <div class="sub">
    <span class="pill" id="nchords">0 chords</span>
    <span class="pill" id="sig">4/4</span>
    <span class="pill" id="bpm">— bpm</span>
    <span class="pill src" id="gridsrc">grid</span>
    <span class="pill" id="dur">0:00</span>
  </div>
</header>

<div class="transport">
  <button class="playbtn" id="play" aria-label="play" disabled>&#9654;</button>
  <div class="clock"><span id="cur">0:00.0</span><span class="d"> / <span id="tot">0:00.0</span></span></div>
  <div class="nowchord" id="now">—</div>
</div>

<div class="tlwrap"><div class="tlscroll" id="scroll"><div class="timeline" id="timeline">
  <canvas id="wf"></canvas>
  <div id="grid"></div>
  <div id="chords"></div>
  <div id="addcue"></div>
  <div id="playhead" style="transform:translateX(0)"></div>
  <span class="bandlbl" style="top:4px">CHORDS &middot; drag teal lines to move &middot; tap waveform to seek</span>
</div></div></div>

<div class="tools">
  <button id="tadd">&#65291; Add boundary</button>
  <button id="tsnap" class="on">&#9673; Snap to beat</button>
  <div class="zoom">
    <button id="zout" aria-label="zoom out">&minus;</button>
    <button id="zin" aria-label="zoom in">+</button>
  </div>
</div>
<p class="hint"><b>Drag a teal line</b> to move a chord change (both neighbours follow) &middot;
<b>Add boundary</b> then tap the timeline to split a chord &middot;
<b>Tap a chord</b> to relabel or delete its boundary.</p>

<div class="savebar">
  <button class="save" id="save">&#128190; Save alignment</button>
  <span class="stat" id="stat"></span>
</div>

<div class="modal" id="modal"><div class="sheet">
  <h3>Edit chord</h3><div class="prev" id="mprev">C</div>
  <div class="grp">Root</div><div class="keys" id="mroots"></div>
  <div class="grp">Quality</div><div class="keys q" id="mquals"></div>
  <div class="srow">
    <button class="del" id="mdel">Delete boundary</button>
    <button id="mcancel">Cancel</button>
    <button class="apply" id="mapply">Apply</button>
  </div>
</div></div>
<div class="toast" id="toast"></div>

<script>
const D = __ANNOT_DATA__;
// ---------- music model ----------
let duration = D.duration || 1;
const beats = (D.beats||[]).slice();
const downSet = new Set((D.downbeats||[]).map(x=>+x.toFixed(3)));
// chords: harmonic spans sharing boundaries. key = bar:beat (save address).
let chords = D.chords.map(c=>({ bar:c.bar, beat:c.beat, section:c.section||'', label:c.label||'?',
  t0:+c.t0, t1:+c.t1, conf:c.conf, dirty:false, _o0:+c.t0, _o1:+c.t1,
  key:c.bar+':'+c.beat, added:false }));
chords.sort((a,b)=>a.t0-b.t0);
// derive a beats-per-bar guess from downbeat spacing (for the time-sig pill)
function beatsPerBar(){ const dbs=[...downSet]; if(dbs.length<2||beats.length<2) return 4;
  const gaps=[]; for(let i=1;i<dbs.length;i++){ let n=0; for(const b of beats) if(b>=dbs[i-1]-1e-3&&b<dbs[i]-1e-3)n++; if(n>0)gaps.push(n);}
  if(!gaps.length) return 4; gaps.sort((a,b)=>a-b); return gaps[Math.floor(gaps.length/2)]; }
const BPB = beatsPerBar();

// ---------- layout constants ----------
const H=240, CH_TOP=24, CH_H=40;
const WF_TOP=70, WF_BOT=H-6, WF_MID=(WF_TOP+WF_BOT)/2, WF_HALF=(WF_BOT-WF_TOP)/2;
const MINGAP=0.06;                          // min chord length (s)
const SNAP=(D.snapTolMs||250)/1000;         // snap-to-beat tolerance
const MAX_CANVAS_W=16000;
let PPS=Math.max(24, Math.min(120, 900/Math.max(1,duration)*10)); // fit-ish default
const maxPPS=()=>Math.max(6, Math.floor(MAX_CANVAS_W/Math.max(1,duration)));
function clampPPS(){ PPS=Math.max(10, Math.min(PPS, maxPPS())); }

// ---------- elements ----------
const $=id=>document.getElementById(id);
const scroll=$('scroll'), timeline=$('timeline'), canvas=$('wf'), ctx=canvas.getContext('2d');
const gridL=$('grid'), chordL=$('chords'), playhead=$('playhead'), addcue=$('addcue');
const audio=new Audio(); audio.preload='auto'; audio.playsInline=true; audio.crossOrigin='anonymous';
let peaks=null, snapOn=true, addMode=false, saved=true, playing=false;

// ---------- header ----------
$('ttl').textContent=D.title;
$('sig').textContent=BPB+'/4';
$('bpm').textContent=Math.round(D.bpm||D.tempo||0)+' bpm';
$('gridsrc').textContent=D.gridSource==='extract_beat_grid'?'beat-grid':'grid: '+(D.gridSource||'?');
const who=$('who'); who.value=localStorage.getItem('harmAnnotator')||'';
who.addEventListener('change',()=>localStorage.setItem('harmAnnotator',who.value.trim()));

// ---------- helpers ----------
const fmt=t=>{t=Math.max(0,t);const m=Math.floor(t/60),s=t-60*m;return `${m}:${s<10?'0':''}${s.toFixed(1)}`;};
const fmtS=t=>{t=Math.max(0,t);const m=Math.floor(t/60),s=Math.round(t-60*m);return `${m}:${s<10?'0':''}${s}`;};
const totalW=()=>Math.max(scroll.clientWidth||360, Math.round(duration*PPS));
const tToX=t=>t*PPS, xToT=x=>x/PPS;
function nearestBeat(t){let best=null,bd=1e9;for(const b of beats){const d=Math.abs(b-t);if(d<bd){bd=d;best=b;}}return{b:best,d:bd};}
function confColor(c){const q=(c.conf!=null?c.conf:0.6);return `hsl(${Math.round(q*130)} 62% ${48-q*4}%)`;}
function markDirty(){saved=false;updateStat();}
function updateStat(){const n=chords.filter(c=>c.dirty||c.added).length;
  $('stat').textContent=n?`${n} edited`:(saved?'saved':'');}
let toastT; function toast(m,err){const e=$('toast');e.textContent=m;e.className='toast on'+(err?' err':'');
  clearTimeout(toastT);toastT=setTimeout(()=>e.className='toast'+(err?' err':''),2400);}

// ---------- waveform ----------
function drawWave(){
  clampPPS(); const W=totalW();
  canvas.width=W; canvas.height=H; canvas.style.width=W+'px'; canvas.style.height=H+'px';
  timeline.style.width=W+'px';
  ctx.setTransform(1,0,0,1,0,0); ctx.clearRect(0,0,W,H);
  ctx.fillStyle='#12161d'; ctx.fillRect(0,WF_TOP-2,W,WF_BOT-WF_TOP+4);
  ctx.strokeStyle='#232c38'; ctx.beginPath(); ctx.moveTo(0,WF_MID); ctx.lineTo(W,WF_MID); ctx.stroke();
  if(peaks && peaks.length){
    ctx.fillStyle='#4b5666';
    const n=peaks.length;
    for(let x=0;x<W;x++){
      const idx=Math.min(n-1, Math.floor(xToT(x)/duration*n));
      const a=peaks[idx]||0, h=Math.max(1,a*WF_HALF);
      ctx.fillRect(x, WF_MID-h, 1, h*2);
    }
  } else { ctx.fillStyle='#8b97a8'; ctx.font='12px system-ui'; ctx.fillText('decoding audio…',12,WF_MID); }
}
async function loadPeaks(){
  if(!D.slug){ drawWave(); return; }
  try{
    const r=await fetch('/api/waveform-peaks/'+encodeURIComponent(D.slug));
    if(r.ok){ const d=await r.json(); peaks=d.peaks; if(d.duration) duration=Math.max(duration,d.duration); }
  }catch(e){ console.warn('peaks fetch failed',e); }
  drawWave();
}

// ---------- bar / beat grid ----------
function layoutGrid(){
  const W=totalW(); let html='';
  for(const t of beats){ if(t>duration+1) continue; const x=tToX(t); const db=downSet.has(+t.toFixed(3));
    html+=`<div style="position:absolute;left:${x}px;top:${WF_TOP}px;bottom:0;width:${db?2:1}px;
      background:${db?'var(--db)':'#2f3a49'};opacity:${db?.7:.4}"></div>`; }
  // bar numbers along the top of the waveform band
  let bar=1;
  for(const t of beats){ if(!downSet.has(+t.toFixed(3))) continue; if(t>duration+1) continue;
    html+=`<div style="position:absolute;left:${tToX(t)+2}px;top:${WF_TOP+1}px;font:700 8px ui-monospace;color:#cfd8e6aa">${bar++}</div>`; }
  gridL.innerHTML=html; gridL.style.width=W+'px';
}

// ---------- chord spans + boundaries ----------
function layoutChords(){
  const W=totalW(); let html='';
  chords.forEach((c,i)=>{
    const x=tToX(c.t0), w=Math.max(8,tToX(c.t1)-tToX(c.t0));
    html+=`<div class="cspan${c.dirty?' dirty':''}" data-i="${i}" style="left:${x}px;width:${w}px;background:${confColor(c)}">`
        +`<span>${esc(c.label||'?')}</span></div>`;
  });
  // one draggable boundary per internal edge (shared by chords[i-1] & chords[i])
  for(let i=1;i<chords.length;i++){
    const x=tToX(chords[i].t0);
    html+=`<div class="bnd" data-b="${i}" style="left:${x}px"><i></i></div>`;
  }
  chordL.innerHTML=html; chordL.style.width=W+'px';
}
function esc(s){return String(s).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}

// ---------- reflow everything ----------
function relayout(){ drawWave(); layoutGrid(); layoutChords(); placePlayhead(); }

// ---------- playback + playhead ----------
function placePlayhead(){ playhead.style.transform='translateX('+tToX(audio.currentTime||0)+'px)'; }
function nowChordAt(t){ for(const c of chords) if(t>=c.t0-1e-3 && t<c.t1) return c; return null; }
let raf=0;
function tick(){ const t=audio.currentTime||0; placePlayhead();
  $('cur').textContent=fmt(t); const c=nowChordAt(t); $('now').textContent=c?c.label:'—';
  // keep playhead in view
  const x=tToX(t); if(x<scroll.scrollLeft+30||x>scroll.scrollLeft+scroll.clientWidth-30)
    scroll.scrollLeft=x-scroll.clientWidth/2;
  if(playing) raf=requestAnimationFrame(tick); }
$('play').addEventListener('click',async()=>{
  if(audio.paused){ try{ await audio.play(); playing=true; $('play').innerHTML='&#10073;&#10073;'; tick(); }
    catch(e){ toast('Tap again to allow audio',true); } }
  else { audio.pause(); playing=false; $('play').innerHTML='&#9654;'; cancelAnimationFrame(raf); }
});
audio.addEventListener('ended',()=>{ playing=false; $('play').innerHTML='&#9654;'; cancelAnimationFrame(raf); });

// ---------- pointer: seek / add / drag ----------
let drag=null; // {b, startX}
function evX(e){ const r=timeline.getBoundingClientRect(); const cx=(e.touches?e.touches[0]:e).clientX;
  return cx-r.left; }
// boundary drag (pointerdown on .bnd)
chordL.addEventListener('pointerdown',e=>{
  const bnd=e.target.closest('.bnd');
  if(bnd){ e.preventDefault(); const b=+bnd.dataset.b; drag={b,el:bnd}; bnd.classList.add('drag');
    bnd.setPointerCapture&&bnd.setPointerCapture(e.pointerId); return; }
});
chordL.addEventListener('pointermove',e=>{
  if(!drag) return; e.preventDefault();
  let t=xToT(evX(e)); const i=drag.b;
  const lo=chords[i-1].t0+MINGAP, hi=chords[i].t1-MINGAP;
  t=Math.max(lo,Math.min(hi,t));
  let snapped=false;
  if(snapOn){ const nb=nearestBeat(t); if(nb.b!=null&&nb.d<=SNAP){ t=Math.max(lo,Math.min(hi,nb.b)); snapped=true; } }
  chords[i-1].t1=t; chords[i].t0=t;
  drag.el.style.left=tToX(t)+'px'; drag.el.classList.toggle('snap',snapped);
  // live-resize the two neighbouring spans without full relayout
  const L=chordL.querySelector(`.cspan[data-i="${i-1}"]`), R=chordL.querySelector(`.cspan[data-i="${i}"]`);
  if(L) L.style.width=Math.max(8,tToX(chords[i-1].t1)-tToX(chords[i-1].t0))+'px';
  if(R){ R.style.left=tToX(chords[i].t0)+'px'; R.style.width=Math.max(8,tToX(chords[i].t1)-tToX(chords[i].t0))+'px'; }
});
function endDrag(e){
  if(!drag) return; const i=drag.b; drag.el.classList.remove('drag','snap');
  chords[i-1].dirty=chords[i-1].t0!==chords[i-1]._o0||chords[i-1].t1!==chords[i-1]._o1;
  chords[i].dirty=chords[i].t0!==chords[i]._o0||chords[i].t1!==chords[i]._o1;
  drag=null; layoutChords(); markDirty();
}
chordL.addEventListener('pointerup',endDrag); chordL.addEventListener('pointercancel',endDrag);

// tap on empty timeline: seek, or (add mode) split the chord there
timeline.addEventListener('click',e=>{
  if(drag) return;
  if(e.target.closest('.bnd')) return;
  const t=xToT(evX(e));
  const cspan=e.target.closest('.cspan');
  if(addMode){ addBoundaryAt(t); return; }
  if(cspan){ openEditor(+cspan.dataset.i); return; }
  seek(t);
});
function seek(t){ t=Math.max(0,Math.min(duration,t)); audio.currentTime=t; placePlayhead();
  $('cur').textContent=fmt(t); const c=nowChordAt(t); $('now').textContent=c?c.label:'—'; }
// live add-cue while in add mode
timeline.addEventListener('pointermove',e=>{ if(!addMode||drag){addcue.classList.remove('on');return;}
  let t=xToT(evX(e)); if(snapOn){const nb=nearestBeat(t); if(nb.b!=null&&nb.d<=SNAP)t=nb.b;}
  addcue.style.left=tToX(t)+'px'; addcue.classList.add('on'); });
timeline.addEventListener('pointerleave',()=>addcue.classList.remove('on'));

function addBoundaryAt(t){
  if(snapOn){ const nb=nearestBeat(t); if(nb.b!=null&&nb.d<=SNAP) t=nb.b; }
  // find the chord span containing t
  let idx=-1; for(let i=0;i<chords.length;i++){ if(t>chords[i].t0+MINGAP && t<chords[i].t1-MINGAP){ idx=i; break; } }
  if(idx<0){ toast('Tap inside a chord to split it',true); return; }
  const c=chords[idx];
  // synthetic unique address so the save sidecar (keyed by bar:beat) never collides
  const nb={ bar:c.bar, beat:900+(insCounter++), section:c.section, label:c.label,
    t0:t, t1:c.t1, conf:0.6, dirty:true, added:true, _o0:t, _o1:c.t1 };
  nb.key=nb.bar+':'+nb.beat;
  c.t1=t; c.dirty=true;
  chords.splice(idx+1,0,nb);
  layoutChords(); markDirty(); toast('Boundary added');
}
let insCounter=0;

// ---------- zoom ----------
function zoom(f){ const t=xToT(scroll.scrollLeft+scroll.clientWidth/2); PPS*=f; clampPPS();
  relayout(); scroll.scrollLeft=tToX(t)-scroll.clientWidth/2; }
$('zin').addEventListener('click',()=>zoom(1.4)); $('zout').addEventListener('click',()=>zoom(1/1.4));
$('tsnap').addEventListener('click',()=>{ snapOn=!snapOn; $('tsnap').classList.toggle('on',snapOn); });
$('tadd').addEventListener('click',()=>{ addMode=!addMode; $('tadd').classList.toggle('on',addMode);
  addcue.classList.remove('on'); toast(addMode?'Tap a chord to split it':'Add mode off'); });

// ---------- relabel / delete sheet ----------
const ROOTS=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B'];
const QUALS=['','m','7','maj7','m7','m7b5','dim','aug','6','m6','sus4','9','13','7b9','7#9','+'];
let editIdx=-1, selRoot='C', selQual='';
function parseLabel(l){ const m=(l||'').match(/^([A-G][#b]?)(.*)$/); return m?{root:m[1],q:m[2]}:{root:'C',q:''}; }
function openEditor(i){ editIdx=i; const p=parseLabel(chords[i].label);
  selRoot=p.root.replace('b',''); selQual=p.q; drawSheet(); $('modal').classList.add('on'); }
function drawSheet(){
  $('mprev').textContent=selRoot+selQual;
  $('mroots').innerHTML=ROOTS.map(r=>`<button data-r="${r}" class="${r===selRoot?'on':''}">${r}</button>`).join('');
  $('mquals').innerHTML=QUALS.map(q=>`<button data-q="${q}" class="${q===selQual?'on':''}">${q||'maj'}</button>`).join('');
}
$('mroots').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;selRoot=b.dataset.r;drawSheet();});
$('mquals').addEventListener('click',e=>{const b=e.target.closest('button');if(!b)return;selQual=b.dataset.q;drawSheet();});
$('mcancel').addEventListener('click',()=>$('modal').classList.remove('on'));
$('mapply').addEventListener('click',()=>{ const c=chords[editIdx]; const nl=selRoot+selQual;
  if(nl!==c.label){ c.label=nl; c.dirty=true; markDirty(); } layoutChords(); $('modal').classList.remove('on'); });
$('mdel').addEventListener('click',()=>{ // delete this chord's LEFT boundary (merge into previous)
  const i=editIdx; if(i<=0){ toast('Cannot remove the first boundary',true); $('modal').classList.remove('on'); return; }
  chords[i-1].t1=chords[i].t1; chords[i-1].dirty=true; chords.splice(i,1);
  layoutChords(); markDirty(); $('modal').classList.remove('on'); toast('Boundary removed'); });

// ---------- save / resume ----------
$('save').addEventListener('click',async()=>{
  const now=new Date().toISOString();
  const body={ annotator:(who.value.trim()||'anon'),
    chords: chords.map(c=>({ bar:c.bar, beat:c.beat, section:c.section, label:c.label,
      t0:+c.t0.toFixed(3), t1:+c.t1.toFixed(3), ts:now })), merges:[] };
  $('save').disabled=true;
  try{
    const doc=await fetch('/api/annotations/'+encodeURIComponent(D.saveFile),
      {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).then(r=>r.json());
    chords.forEach(c=>{ c._o0=c.t0; c._o1=c.t1; c.dirty=false; c.added=false; });
    saved=true; layoutChords(); updateStat();
    toast('✓ Saved '+(doc.chords?doc.chords.length:chords.length)+' chords');
  }catch(e){ toast('Save failed — check connection',true); }
  finally{ $('save').disabled=false; }
});
// resume prior edits (match by saved t0/t1 onto the same bar:beat address)
fetch('/api/annotations/'+encodeURIComponent(D.saveFile)).then(r=>r.json()).then(doc=>{
  if(!who.value && doc.annotator) who.value=doc.annotator;
  const by={}; (doc.chords||[]).forEach(c=>{ if('t0' in c) by[c.bar+':'+c.beat]=c; });
  let n=0; chords.forEach(c=>{ const p=by[c.bar+':'+c.beat]; if(p){ c.t0=+p.t0; if('t1' in p)c.t1=+p.t1;
    if(p.label)c.label=p.label; c._o0=c.t0; c._o1=c.t1; n++; } });
  if(n){ chords.sort((a,b)=>a.t0-b.t0); relayout(); }
}).catch(()=>{});

// ---------- boot ----------
$('nchords').textContent=chords.length+' chords';
$('dur').textContent=fmtS(duration); $('tot').textContent=fmt(duration);
if(D.audioUrl){ audio.src=D.audioUrl; audio.addEventListener('canplay',()=>$('play').disabled=false,{once:true});
  audio.addEventListener('loadedmetadata',()=>{ if(audio.duration&&isFinite(audio.duration)){
    duration=Math.max(duration,audio.duration); $('tot').textContent=fmt(duration); $('dur').textContent=fmtS(duration); relayout(); } });
  audio.load(); setTimeout(()=>{ if($('play').disabled) $('play').disabled=false; },1500);
} else { $('play').disabled=true; }
(async()=>{ await loadPeaks(); relayout(); $('loader').style.display='none'; })();
window.addEventListener('resize',relayout);
</script>
</body></html>"""


@app.route("/api/beat-0-shift/<song>", methods=["POST"])
def api_beat_0_shift(song):
    """Shift beat 0 by delta_ms, extrapolate entire grid, re-infer chords.

    Request: {delta_ms: int}  (positive = beat too early, shift forward)
    Response: {beat_times: [...], chords: [...], note: "..."}
    """
    import librosa
    import numpy as np

    slug = re.sub(r"[^A-Za-z0-9_]", "", song or "")
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

    slug = re.sub(r"[^A-Za-z0-9_]", "", (request.args.get("song") or "autumn_leaves"))

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

    rwcid = re.sub(r"[^A-Za-z0-9_]", "", request.args.get("song") or "")
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

    slug = re.sub(r"[^A-Za-z0-9_]", "", (request.args.get("song") or "autumn_leaves"))
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

    slug = re.sub(r"[^A-Za-z0-9_]", "", (request.args.get("song") or "autumn_leaves"))
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
    slug = re.sub(r"[^A-Za-z0-9_]", "", (request.args.get("song") or "autumn_leaves"))
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
    slug = re.sub(r"[^A-Za-z0-9_]", "", (request.args.get("song") or "autumn_leaves"))
    data, err = _build_annotator_data(slug)
    if err:
        return err
    page = ANNOTATOR_V4_TEMPLATE.replace("__ANNOT_DATA__", json.dumps(data))
    page = page.replace("</head>", _PWA_HEAD + "</head>", 1)
    return Response(page, mimetype="text/html")


def _load_ireal_alignment(slug: str):
    """Parse docs/plots/irealb_<slug>.html → (chords, tempo).

    Each chord: {i, bar, beat, section, label, t0, t1}. `beat` is the 0-based
    ordinal within its bar (the irealb payload has no beat-in-bar offset), so
    (bar, beat) is a unique per-song key — the sidecar's chord address (§3).
    t0/t1 are the DTW-aligned starting suggestion the user corrects from.
    """
    p = PLOTS_DIR / f"irealb_{slug}.html"
    if not p.exists():
        return None, None
    m = re.search(r"window\.P\s*=\s*(\{.*?\})\s*;", p.read_text(encoding="utf-8"), re.S)
    if not m:
        return None, None
    try:
        payload = json.loads(m.group(1))
    except ValueError:
        return None, None
    tempo = float(payload.get("tempo") or 120)
    chords, bar_counts = [], {}
    for idx, c in enumerate(payload.get("chords", [])):
        bar = int(c.get("bar", 0))
        beat = bar_counts.get(bar, 0)
        bar_counts[bar] = beat + 1
        # irealb payloads carry no per-chord posterior; the DTW `match` field
        # (exact|mismatch vs the acoustic reading) is the only confidence-like
        # signal available, so the waveform UI colours bars from it:
        # exact -> high (green), mismatch -> low (red), unknown -> mid (amber).
        match = c.get("match", "")
        conf = 0.9 if match == "exact" else (0.25 if match == "mismatch" else 0.6)
        chords.append({
            "i": idx, "bar": bar, "beat": beat,
            "section": c.get("section", ""), "label": c.get("label", ""),
            "t0": float(c.get("t0", 0.0)), "t1": float(c.get("t1", 0.0)),
            "match": match, "conf": conf,
        })
    return chords, tempo


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
    slug = re.sub(r"[^A-Za-z0-9_]", "", song or "")
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
    slug = re.sub(r"[^A-Za-z0-9_]", "", (request.args.get("song") or "autumn_leaves"))
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
    slug = re.sub(r"[^A-Za-z0-9_]", "", (request.args.get("song") or "autumn_leaves"))
    data, err = _build_annotator_data(slug)
    if err:
        return err
    page = ANNOTATOR_SIMPLE_TEMPLATE.replace("__ANNOT_DATA__", json.dumps(data))
    page = page.replace("</head>", _PWA_HEAD + "</head>", 1)
    return Response(page, mimetype="text/html")


@app.route("/diagnostics/<filename>")
def serve_diagnostic(filename):
    """Serve diagnostic HTML files from docs/plots/."""
    # Sanitize filename to prevent directory traversal
    filename = re.sub(r"[^A-Za-z0-9_\-.]", "", filename)
    p = PLOTS_DIR / filename
    if not p.exists() or not p.suffix == ".html":
        return f"<p>Diagnostic {filename} not found</p>", 404
    return send_from_directory(PLOTS_DIR, filename, mimetype="text/html")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    global _ARGS
    ap = argparse.ArgumentParser(description="Harmonia local server")
    ap.add_argument("--port", type=int, default=7771)
    ap.add_argument("--no-open", action="store_true")
    ap.add_argument("--phase", type=int, default=1, choices=[1, 2, 3, 4])
    ap.add_argument("--cache-dir", default="data/cache")
    ap.add_argument("--no-madmom", action="store_true")
    ap.add_argument("--verbose", "-v", action="store_true")
    _ARGS = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if _ARGS.verbose else logging.INFO,
        format="%(levelname)s  %(message)s",
    )

    url = f"http://localhost:{_ARGS.port}"
    lan_ip = _lan_ip()
    print(f"Harmonia server →  {url}")
    if lan_ip:
        print(f"  on your iPhone (same Wi-Fi) →  http://{lan_ip}:{_ARGS.port}")

    if not _ARGS.no_open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    app.run(host="0.0.0.0", port=_ARGS.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
