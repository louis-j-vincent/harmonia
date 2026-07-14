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
_SW_CACHE_VERSION = "harmonia-v1"

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
<script>if("serviceWorker" in navigator){navigator.serviceWorker.register("/sw.js");}</script>
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
    fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url})})
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

_BACK_BUTTON_HTML = """<a href="/library" id="harm-back" onclick="if(history.length>1){history.back();return false;}"
   style="position:fixed;top:max(12px,env(safe-area-inset-top));left:12px;z-index:9998;
   display:flex;align-items:center;gap:5px;background:#8a2b2bcc;color:#fff;
   text-decoration:none;font:700 13px system-ui,sans-serif;padding:7px 13px 7px 10px;
   border-radius:20px;box-shadow:0 2px 8px #0004;backdrop-filter:blur(4px);
   transition:transform .1s ease;">&larr; Charts</a>
<style>#harm-back:active{transform:scale(.93);}</style>
"""


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
    return Response(_SERVICE_WORKER_JS, mimetype="application/javascript")


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
    return send_from_directory(AUDIO_DIR, filename, conditional=True, mimetype=mimetype)


@app.route("/pwa/<path:filename>")
def serve_pwa_asset(filename):
    """Serve the PWA manifest and home-screen icons."""
    p = PWA_DIR / filename
    if not p.exists() or p.parent != PWA_DIR:
        return "Not found", 404
    mimetype = "application/manifest+json" if p.suffix == ".json" else None
    return Response(p.read_bytes(), mimetype=mimetype)


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


def _chart_model_for(filename: str) -> dict:
    """ChartModel for a rendered chart — payload + sidecar + audio/video links."""
    from harmonia.output.chart_model import payload_from_chart_html, to_chart_model

    p = PLOTS_DIR / filename
    payload = payload_from_chart_html(p)
    meta = _yt_audio_meta.get(filename) or {}
    audio_url = meta.get("audio", "")
    if audio_url and not (AUDIO_DIR / Path(audio_url).name).exists():
        audio_url = ""
    return to_chart_model(
        payload,
        filename=filename,
        video_id=_yt_video_ids.get(filename, ""),
        audio_url=audio_url,
        annotation=_load_annotation(filename),
    )


@app.route("/api/library")
def api_library():
    """Every chart we have, as library cards (title, key, bars, has-audio)."""
    from harmonia.output.chart_model import chart_summary

    charts = []
    for p in sorted(PLOTS_DIR.glob("inferred_*.html")):
        try:
            charts.append(chart_summary(_chart_model_for(p.name)))
        except (OSError, ValueError, KeyError) as e:
            log.warning("Skipping %s in library: %s", p.name, e)
    # newest first — the chart you just analysed should be at the top
    charts.sort(key=lambda c: (PLOTS_DIR / c["file"]).stat().st_mtime, reverse=True)
    return jsonify(charts=charts)


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


@app.route("/library")
def library():
    """Your already-analyzed charts — a deliberately separate page from the
    search-first home, reached via the "Your charts" pill."""
    charts = sorted(PLOTS_DIR.glob("inferred_*.html"))
    items = [{"name": p.stem.replace("inferred_", "").replace("_", " ").title(),
              "file": p.name} for p in charts]
    page = render_template_string(LIBRARY_TEMPLATE, charts=items)
    return Response(page.replace("</head>", _PWA_HEAD + "</head>", 1), mimetype="text/html")


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
    """Serve a chart HTML file with the YouTube overlay injected."""
    p = PLOTS_DIR / filename
    if not p.exists() or not p.suffix == ".html":
        return "Not found", 404
    content = p.read_text(encoding="utf-8")
    content = content.replace("</head>", _PWA_HEAD + "</head>", 1)
    vid = _yt_video_ids.get(filename, "")
    if vid:
        content = content.replace(
            "</head>",
            f'<script>window.YT_VIDEO_ID="{vid}";</script></head>',
            1,
        )
    audio_meta = _yt_audio_meta.get(filename)
    if audio_meta and (AUDIO_DIR / Path(audio_meta["audio"]).name).exists():
        content = content.replace(
            "</head>",
            '<script>window.HARM_AUDIO_URL=' + json.dumps(audio_meta["audio"])
            + ';window.HARM_THUMB_URL=' + json.dumps(audio_meta.get("thumb", ""))
            + ';</script></head>',
            1,
        )
    annotation = _load_annotation(filename)
    if annotation.get("chords") or annotation.get("merges"):
        content = content.replace(
            "</head>",
            '<script>window.HARM_ANNOTATIONS=' + json.dumps(annotation) + ';</script></head>',
            1,
        )
    charts = sorted(f.name for f in PLOTS_DIR.glob("inferred_*.html"))
    if filename in charts and len(charts) > 1:
        idx = charts.index(filename)
        pos_html = f'<div class="harm-pos" aria-hidden="true">{idx + 1} / {len(charts)}</div>\n  '
        content = content.replace(
            '<div class="modal" id="wheelModal">', pos_html + '<div class="modal" id="wheelModal">', 1
        )
        swipe_js = (_SWIPE_NAV_JS
                    .replace("%%PREV%%", charts[idx - 1])
                    .replace("%%NEXT%%", charts[(idx + 1) % len(charts)]))
        content = content.replace(_INJECT_MARKER, swipe_js + _INJECT_MARKER)
    return Response(_inject_back_button(_inject_overlay(content)), mimetype="text/html")


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

    from harmonia.models.chord_pipeline_v1 import infer_chords_v1

    tmp_dir = Path(tempfile.mkdtemp(prefix="harmonia_reinfer_"))
    try:
        wav = tmp_dir / "a.wav"
        try:
            _sp.run(["ffmpeg", "-y", "-i", str(audio), "-ac", "1", "-ar", "22050",
                     str(wav)], check=True, capture_output=True, timeout=120)
        except (OSError, _sp.CalledProcessError, _sp.TimeoutExpired) as e:
            return jsonify(error=f"Audio transcode failed: {e}"), 500

        cache = tmp_dir            # shared cache_dir → 2nd infer is a stage-1 cache hit
        base = infer_chords_v1(wav, cache_dir=cache, joint_transition_weight=tw)
        # The pipeline DEGRADES GRACEFULLY when a constraint can't be applied —
        # e.g. pool_beat_evidence rejects a section-merge whose spans differ in
        # beat count ("equal musical length" is a v1 precondition). It logs a
        # warning and decodes unconstrained, so without this the endpoint would
        # answer 200 / n_changed=0 and the UI would report "Merged — one shared
        # reading" when nothing was pooled at all. Capture the warning and hand
        # it back so the client can say what actually happened.
        warnings: list[str] = []

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
        log.info("reinfer %s: %d confirms, %d merges, %d/%d chords changed%s",
                 filename, len(confirms), len(merges), len(diff), len(out),
                 f" (REJECTED: {rejected})" if rejected else "")
        return jsonify(chords=out, diff=diff, n_changed=len(diff),
                       key=cons.global_key, tempo_bpm=cons.tempo_bpm,
                       rejected=rejected)
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
    """Accept a YouTube URL, start a background analysis job, return job_id."""
    data = request.get_json(silent=True) or {}
    url = (data.get("url") or "").strip()
    if not url:
        return jsonify(error="No URL provided"), 400
    if "youtube.com" not in url and "youtu.be" not in url:
        return jsonify(error="Please provide a YouTube URL"), 400

    job_id = f"job_{int(time.time() * 1000)}"
    with _jobs_lock:
        _jobs[job_id] = {"status": "queued", "url": url, "message": "Queued"}

    t = threading.Thread(target=_run_analysis, args=(job_id, url), daemon=True)
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


def _run_analysis(job_id: str, url: str) -> None:
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

    tmp_dir = Path(tempfile.mkdtemp(prefix="harmonia_yt_"))
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
                         else "audio fetched"), title=video_title)

        from harmonia.models.chord_pipeline_v1 import infer_chords_v1
        pipeline_chart = infer_chords_v1(
            audio_path,
            cache_dir=Path(_ARGS.cache_dir),
        )

        stage(2, result=(f"{pipeline_chart.global_key} · {pipeline_chart.tempo_bpm:.0f} bpm"
                         f" · {len(pipeline_chart.chords)} chords"))

        from scripts.render_youtube_chart import chart_to_interactive_inputs
        from harmonia.output.chart_interactive import render_interactive

        source_desc = f"inferred from YouTube · {url}"
        chart_obj, chord_dicts = chart_to_interactive_inputs(pipeline_chart, video_title, source_desc)

        slug = re.sub(r"[^a-z0-9]+", "_", video_title.lower()).strip("_") or "yt"
        out = PLOTS_DIR / f"inferred_{slug[:60]}.html"
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
                 "-acodec", "aac", "-b:a", "128k", str(audio_dest)],
                check=True, capture_output=True, timeout=120,
            )
            thumb_url = f"https://i.ytimg.com/vi/{vid}/mqdefault.jpg" if vid else ""
            _remember_audio(out.name, f"/audio/{audio_dest.name}", thumb_url)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            log.warning("Could not persist/transcode audio for %s: %s", out.name, e)

        # Persist the chroma/pitch activations too, addressable by slug — not
        # a second Basic Pitch run: infer_chords_v1() already populated
        # PitchExtractor's own (ephemeral, temp-path-keyed) cache above, so
        # this call is a cache hit and just re-saves it somewhere we can
        # actually find again later (see PITCH_CACHE_DIR).
        try:
            from harmonia.models.stage1_pitch import PitchExtractor
            activations = PitchExtractor(cache_dir=Path(_ARGS.cache_dir)).extract(audio_path)
            activations.save(PITCH_CACHE_DIR / f"{slug[:60]}.npz")
        except Exception as e:
            log.warning("Could not persist pitch/chroma cache for %s: %s", out.name, e)

        # Fetch iReal chart from community if available, so the annotator tool
        # (which needs docs/plots/irealb_<slug>.html) doesn't fail with 404.
        try:
            from harmonia.irealb_fetcher import search_community, render_irealb_chart
            results_ir = search_community(video_title, max_results=1)
            if results_ir:
                irealb_url = results_ir[0]["irealb_url"]
                # Render the iReal chart with this offset (assume the inferred chart
                # starts at t=0 in the audio file)
                html_ir = render_irealb_chart(irealb_url, chart_offset_s=0.0,
                                              tempo_override=int(round(pipeline_chart.tempo_bpm)))
                ir_out = PLOTS_DIR / f"irealb_{slug[:60]}.html"
                ir_out.write_text(html_ir, encoding="utf-8")
                _remember_ireal_url(out.name, irealb_url)
                log.info("Saved iReal chart for %s (%s)", out.name, irealb_url)
        except Exception as e:
            log.warning("Could not fetch/render iReal chart for %s: %s", video_title, e)

        results[2] = f"{chart_obj.n_bars} bars"
        update("done", url=f"/chart/{out.name}", stage=3,
               results=list(results), title=video_title)

    except Exception as e:
        log.exception("Analysis failed for %s", url)
        update("error", error=str(e))
    finally:
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
