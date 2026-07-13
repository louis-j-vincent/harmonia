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
<audio id="harm-audio" preload="none"></audio>

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
        cons = infer_chords_v1(wav, cache_dir=cache, joint_transition_weight=tw,
                               user_constraints=constraints)
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
        log.info("reinfer %s: %d confirms, %d merges, %d/%d chords changed",
                 filename, len(confirms), len(merges), len(diff), len(out))
        return jsonify(chords=out, diff=diff, n_changed=len(diff),
                       key=cons.global_key, tempo_bpm=cons.tempo_bpm)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


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
<title>Harmonia — Align Chords</title>
<style>
  :root{
    --bg:#0e1116; --panel:#171c24; --panel2:#1e2530; --ink:#e8edf4; --faint:#8b97a8;
    --line:#2a3340; --teal:#00c9a7; --teal-dim:#0b3d35; --amber:#ffb454; --accent:#6ea8ff;
    --danger:#ff5d6c; --ok:#37d67a;
  }
  *{box-sizing:border-box;-webkit-tap-highlight-color:transparent;}
  html,body{margin:0;background:var(--bg);color:var(--ink);
    font-family:-apple-system,BlinkMacSystemFont,'SF Pro Text',system-ui,sans-serif;
    overscroll-behavior-y:none;overflow-x:hidden;max-width:100%;}
  body{padding-bottom:calc(96px + env(safe-area-inset-bottom));}
  a{color:var(--accent);}
  .top{position:sticky;top:0;z-index:20;background:linear-gradient(180deg,#0e1116 70%,#0e1116cc);
    padding:calc(8px + env(safe-area-inset-top)) 12px 8px;border-bottom:1px solid var(--line);
    overflow-x:hidden;}
  .toprow{display:flex;align-items:center;gap:10px;}
  .toprow h1{font-size:16px;margin:0;font-weight:700;flex:1;letter-spacing:.2px;}
  .back{font:600 13px system-ui;color:var(--faint);text-decoration:none;padding:6px 8px;margin-left:-8px;}
  .who{background:var(--panel2);border:1px solid var(--line);color:var(--ink);
    border-radius:8px;padding:6px 8px;font:600 12px system-ui;width:96px;}
  .sub{display:flex;align-items:center;gap:8px;margin-top:6px;font:600 11px system-ui;color:var(--faint);flex-wrap:wrap;}
  .pill{background:var(--panel2);border:1px solid var(--line);border-radius:20px;padding:3px 9px;}
  .pill.src{color:var(--teal);border-color:var(--teal-dim);}
  audio{width:100%;min-width:0;max-width:100%;margin-top:8px;height:38px;}
  .list{padding:8px 10px 10px;display:flex;flex-direction:column;gap:7px;max-width:100%;}
  .chord{background:var(--panel);border:1px solid var(--line);border-radius:12px;
    padding:10px 12px;display:grid;grid-template-columns:auto 1fr auto;align-items:center;
    column-gap:10px;row-gap:8px;min-height:56px;
    transition:border-color .12s,background .12s;position:relative;}
  .chord.playing{border-color:var(--teal);box-shadow:0 0 0 1px var(--teal) inset;}
  .chord.sel{background:var(--panel2);border-color:var(--accent);}
  .chord.dirty::after{content:"";position:absolute;top:8px;right:8px;width:7px;height:7px;
    border-radius:50%;background:var(--amber);}
  .sec{min-width:22px;height:22px;border-radius:6px;display:flex;align-items:center;justify-content:center;
    font:800 11px system-ui;color:#0e1116;background:var(--faint);}
  .sec.A{background:#6ea8ff;} .sec.B{background:#ffb454;} .sec.C{background:#c88bff;}
  .sec.D{background:#7bd88f;}
  .nm{min-width:0;font:700 17px 'SF Mono',ui-monospace,Menlo,monospace;letter-spacing:.3px;
    overflow:hidden;text-overflow:ellipsis;white-space:nowrap;}
  .tm{font:700 15px ui-monospace,Menlo,monospace;color:var(--teal);min-width:66px;
    text-align:right;justify-self:end;white-space:nowrap;}
  .tm .d{color:var(--faint);font-size:12px;}
  .snapchip{font:700 9px system-ui;color:var(--teal);border:1px solid var(--teal-dim);
    border-radius:10px;padding:1px 6px;margin-left:6px;opacity:0;transition:opacity .15s;}
  .snapchip.on{opacity:1;}
  /* editor */
  .editor{overflow:hidden;max-height:0;transition:max-height .2s ease;grid-column:1/-1;width:100%;}
  .chord.sel .editor{max-height:220px;}
  .edwrap{width:100%;margin-top:10px;}
  .ruler{position:relative;height:64px;background:var(--panel);border:1px solid var(--line);
    border-radius:10px;overflow:hidden;touch-action:none;}
  .tick{position:absolute;top:10px;bottom:10px;width:1px;background:var(--line);}
  .tick.db{top:4px;bottom:4px;width:2px;background:#3a4655;}
  .ticklab{position:absolute;bottom:2px;font:600 8px ui-monospace;color:var(--faint);transform:translateX(-50%);}
  .handle{position:absolute;top:0;bottom:0;width:3px;background:var(--teal);
    box-shadow:0 0 8px var(--teal);}
  .handle::before{content:"";position:absolute;top:-2px;left:-13px;width:28px;height:28px;
    background:var(--teal);border-radius:50%;box-shadow:0 2px 8px #000a;}
  .handle::after{content:"◀▶";position:absolute;top:4px;left:-11px;width:24px;text-align:center;
    font-size:9px;color:#062;letter-spacing:-1px;}
  .edbtns{display:flex;gap:6px;margin-top:8px;}
  .b{flex:1;min-height:44px;border:1px solid var(--line);background:var(--panel2);color:var(--ink);
    border-radius:10px;font:700 13px system-ui;display:flex;align-items:center;justify-content:center;gap:5px;}
  .b:active{transform:scale(.96);}
  .b.play{color:var(--teal);} .b.now{background:var(--teal);color:#062;border-color:var(--teal);}
  .b.nudge{max-width:56px;font-family:ui-monospace;}
  .savebar{position:fixed;left:0;right:0;bottom:0;z-index:30;
    padding:10px 12px calc(10px + env(safe-area-inset-bottom));
    background:linear-gradient(0deg,#0e1116 65%,#0e1116cc);border-top:1px solid var(--line);
    display:flex;gap:10px;align-items:center;}
  .save{flex:1;min-height:52px;border:none;border-radius:14px;background:var(--teal);color:#062;
    font:800 16px system-ui;display:flex;align-items:center;justify-content:center;gap:8px;}
  .save:active{transform:scale(.98);} .save[disabled]{opacity:.5;}
  .stat{font:700 12px system-ui;color:var(--faint);min-width:60px;text-align:right;}
  .toast{position:fixed;left:50%;bottom:96px;transform:translateX(-50%) translateY(20px);
    background:var(--ok);color:#062;font:800 13px system-ui;padding:10px 18px;border-radius:24px;
    opacity:0;transition:.25s;z-index:40;box-shadow:0 6px 20px #0008;pointer-events:none;}
  .toast.on{opacity:1;transform:translateX(-50%) translateY(0);}
  .toast.err{background:var(--danger);color:#fff;}
  .hint{color:var(--faint);font:500 12px system-ui;padding:2px 4px 8px;line-height:1.5;}
</style>
</head><body>
<div class="top">
  <div class="toprow">
    <a class="back" href="/library">&larr;</a>
    <h1 id="ttl">Align</h1>
    <input class="who" id="who" placeholder="your name" autocomplete="off">
  </div>
  <div class="sub">
    <span class="pill" id="nchords">0 chords</span>
    <span class="pill src" id="gridsrc">grid</span>
    <span class="pill" id="tempo">— bpm</span>
  </div>
  <audio id="audio" controls preload="metadata" playsinline></audio>
</div>
<p class="hint">Play the song. When you hear a chord change, tap it and press
<b>&#9201; Set&nbsp;here</b>, or drag the teal handle on its ruler. Boundaries snap to the beat grid. Then <b>Save</b>.</p>
<div class="list" id="list"></div>
<div class="savebar">
  <button class="save" id="save">&#128190; Save alignment</button>
  <span class="stat" id="stat"></span>
</div>
<div class="toast" id="toast"></div>
<script>
const D = __ANNOT_DATA__;
const beats = D.beats||[], downSet = new Set((D.downbeats||[]).map(x=>x.toFixed(2)));
const SNAP = (D.snapTolMs||250)/1000, MINGAP = 0.05, HALF = 2.5;
let chords = D.chords.map(c=>({...c, dirty:false}));
let sel = -1, saved = true;
const audio = document.getElementById('audio');
const listEl = document.getElementById('list');

document.getElementById('ttl').textContent = D.title;
document.getElementById('nchords').textContent = chords.length + ' chords';
document.getElementById('tempo').textContent = Math.round(D.bpm) + ' bpm';
const gs = document.getElementById('gridsrc');
gs.textContent = D.gridSource==='extract_beat_grid' ? 'beat-grid' : 'grid: '+D.gridSource;
if(D.audioUrl){ audio.src = D.audioUrl; } else { audio.replaceWith(Object.assign(document.createElement('div'),{className:'hint',textContent:'(no audio file found for this song)'})); }
const who = document.getElementById('who');
who.value = localStorage.getItem('harmAnnotator')||'';
who.addEventListener('change',()=>localStorage.setItem('harmAnnotator',who.value.trim()));

const fmt = t => { t=Math.max(0,t); const m=Math.floor(t/60), s=t-60*m;
  return `${m}:${s<10?'0':''}${s.toFixed(1)}`; };
function nearestBeat(t){ let best=null,bd=1e9; for(const b of beats){const d=Math.abs(b-t); if(d<bd){bd=d;best=b;}} return {b:best,d:bd}; }
function clampT0(i,t){
  const lo = i>0 ? chords[i-1].t0+MINGAP : 0;
  const hi = i<chords.length-1 ? chords[i+1].t0-MINGAP : (D.duration+5);
  return Math.min(Math.max(t,lo),hi);
}
function setT0(i,t,{snap=false}={}){
  let snapped=false;
  if(snap){ const nb=nearestBeat(t); if(nb.b!=null && nb.d<=SNAP){ t=nb.b; snapped=true; } }
  t = clampT0(i, t);
  chords[i].t0 = t;
  if(i>0) chords[i-1].t1 = t;         // boundaries stay contiguous
  chords[i].dirty = true; chords[i].snapped = snapped;
  markDirty();
  return snapped;
}
function markDirty(){ saved=false; updateStat(); }
function updateStat(){
  const n = chords.filter(c=>c.dirty).length;
  document.getElementById('stat').textContent = n? n+' edited' : (saved?'saved':'');
}

function rowHTML(c,i){
  const secCls = (c.section||'').replace(/[^A-D]/g,'');
  return `<div class="sec ${secCls}">${c.section||'·'}</div>
    <div class="nm">${c.label||'?'}</div>
    <div class="tm">${fmt(c.t0)}<span class="snapchip${c.snapped?' on':''}">snap</span></div>
    <div class="editor"><div class="edwrap">
      <div class="ruler" data-i="${i}"></div>
      <div class="edbtns">
        <button class="b play" data-act="play">&#9654; Play</button>
        <button class="b nudge" data-act="m">-50</button>
        <button class="b nudge" data-act="p">+50</button>
        <button class="b now" data-act="now">&#9201; Set here</button>
      </div>
    </div></div>`;
}
function render(){
  listEl.innerHTML='';
  chords.forEach((c,i)=>{
    const el=document.createElement('div');
    el.className='chord'+(i===sel?' sel':'')+(c.dirty?' dirty':'');
    el.id='ch'+i; el.innerHTML=rowHTML(c,i);
    el.addEventListener('click',ev=>{ if(ev.target.closest('.editor')) return; select(i); });
    listEl.appendChild(el);
  });
  if(sel>=0) drawRuler(sel);
  updateStat();
}
function refreshRow(i){
  const el=document.getElementById('ch'+i); if(!el) return;
  el.className='chord'+(i===sel?' sel':'')+(chords[i].dirty?' dirty':'')+(el.classList.contains('playing')?' playing':'');
  el.querySelector('.tm').innerHTML = fmt(chords[i].t0)+`<span class="snapchip${chords[i].snapped?' on':''}">snap</span>`;
}
function select(i){
  const prev=sel; sel=i;
  if(prev>=0) refreshRow(prev);
  const pe=document.getElementById('ch'+prev); if(pe) pe.classList.remove('sel');
  const el=document.getElementById('ch'+i); el.classList.add('sel');
  el.scrollIntoView({block:'nearest',behavior:'smooth'});
  drawRuler(i);
}

// ---- ruler (drag + snap grid visual) ----
function drawRuler(i){
  const el=document.getElementById('ch'+i); if(!el) return;
  const r=el.querySelector('.ruler'); if(!r) return;
  const c=chords[i];
  const winStart=c.t0-HALF, winEnd=c.t0+HALF, span=winEnd-winStart;
  r.dataset.ws=winStart; r.dataset.we=winEnd;
  const W=r.clientWidth||358;
  let html='';
  for(const b of beats){ if(b<winStart||b>winEnd) continue;
    const x=(b-winStart)/span*W;
    const db=downSet.has(b.toFixed(2));
    html+=`<div class="tick${db?' db':''}" style="left:${x}px"></div>`;
    if(db) html+=`<div class="ticklab" style="left:${x}px">${b.toFixed(1)}</div>`;
  }
  const hx=(c.t0-winStart)/span*W;
  html+=`<div class="handle" style="left:${hx}px"></div>`;
  r.innerHTML=html;
}
function rulerToTime(r,clientX){
  const rect=r.getBoundingClientRect();
  const ws=+r.dataset.ws, we=+r.dataset.we;
  let x=(clientX-rect.left)/rect.width; x=Math.min(1,Math.max(0,x));
  return ws + x*(we-ws);
}
let dragI=-1;
listEl.addEventListener('pointerdown',e=>{
  const r=e.target.closest('.ruler'); if(!r) return;
  dragI=+r.dataset.i; r.setPointerCapture(e.pointerId);
  const t=rulerToTime(r,e.clientX); setT0(dragI,t); drawRuler(dragI); refreshRow(dragI);
  e.preventDefault();
},{passive:false});
listEl.addEventListener('pointermove',e=>{
  if(dragI<0) return;
  const r=document.getElementById('ch'+dragI).querySelector('.ruler');
  const t=rulerToTime(r,e.clientX);
  // live-preview snap
  const nb=nearestBeat(t); const willSnap=nb.b!=null&&nb.d<=SNAP;
  chords[dragI].t0 = clampT0(dragI, willSnap?nb.b:t);
  if(dragI>0) chords[dragI-1].t1=chords[dragI].t0;
  chords[dragI].snapped=willSnap; chords[dragI].dirty=true;
  const el=document.getElementById('ch'+dragI);
  el.querySelector('.snapchip')?.classList.toggle('on',willSnap);
  const W=r.clientWidth, ws=+r.dataset.ws, we=+r.dataset.we;
  const hx=(chords[dragI].t0-ws)/(we-ws)*W;
  const h=r.querySelector('.handle'); if(h) h.style.left=hx+'px';
  el.querySelector('.tm').firstChild.textContent=fmt(chords[dragI].t0);
},{passive:true});
listEl.addEventListener('pointerup',e=>{
  if(dragI<0) return;
  setT0(dragI, chords[dragI].t0, {snap:true});
  if(navigator.vibrate&&chords[dragI].snapped) navigator.vibrate(6);
  drawRuler(dragI); refreshRow(dragI); dragI=-1;
});

// ---- edit buttons ----
listEl.addEventListener('click',e=>{
  const btn=e.target.closest('[data-act]'); if(!btn) return;
  const i=sel; if(i<0) return;
  const act=btn.dataset.act;
  if(act==='play'){ audio.currentTime=Math.max(0,chords[i].t0); audio.play(); }
  else if(act==='m'){ setT0(i,chords[i].t0-0.05); drawRuler(i); refreshRow(i); }
  else if(act==='p'){ setT0(i,chords[i].t0+0.05); drawRuler(i); refreshRow(i); }
  else if(act==='now'){ setT0(i,audio.currentTime,{snap:true}); drawRuler(i); refreshRow(i);
    if(navigator.vibrate) navigator.vibrate(8); }
});

// ---- playhead sync ----
let lastPlaying=-1;
audio.addEventListener('timeupdate',()=>{
  const t=audio.currentTime; let cur=-1;
  for(let i=0;i<chords.length;i++){ if(chords[i].t0<=t){cur=i;} else break; }
  if(cur!==lastPlaying){
    if(lastPlaying>=0) document.getElementById('ch'+lastPlaying)?.classList.remove('playing');
    if(cur>=0){ const el=document.getElementById('ch'+cur); if(el){ el.classList.add('playing');
      if(!audio.paused){ const rc=el.getBoundingClientRect();
        if(rc.top<120||rc.bottom>window.innerHeight-120) el.scrollIntoView({block:'center',behavior:'smooth'});}}}
    lastPlaying=cur;
  }
});

// ---- load existing sidecar (resume prior alignment, keep quality corrections) ----
let existingChords=[], existingMerges=[];
fetch('/api/annotations/'+encodeURIComponent(D.saveFile)).then(r=>r.json()).then(doc=>{
  existingChords = doc.chords||[]; existingMerges = doc.merges||[];
  if(!who.value && doc.annotator){ who.value=doc.annotator; }
  // resume: if a prior save carried t0 for our (bar,beat) keys, adopt it
  const byKey={}; existingChords.forEach(c=>{ if('t0' in c) byKey[c.bar+':'+c.beat]=c; });
  let resumed=0;
  chords.forEach((c,i)=>{ const p=byKey[c.bar+':'+c.beat];
    if(p){ c.t0=+p.t0; if(i>0)chords[i-1].t1=c.t0; c.dirty=false; resumed++; } });
  if(resumed){ saved=true; render(); }
}).catch(()=>{});

// ---- save (non-destructive merge into the existing sidecar) ----
const saveBtn=document.getElementById('save');
saveBtn.addEventListener('click',()=>{
  const now=new Date().toISOString();
  const map={}; existingChords.forEach(c=>{ map[c.bar+':'+c.beat]={...c}; });
  chords.forEach(c=>{ const k=c.bar+':'+c.beat;
    map[k]={...(map[k]||{}), bar:c.bar, beat:c.beat, label:c.label, section:c.section,
            t0:+c.t0.toFixed(3), t1:+c.t1.toFixed(3), ts:now}; });
  const body={ annotator: who.value.trim()||'anon',
    chords: Object.values(map), merges: existingMerges };
  saveBtn.disabled=true;
  fetch('/api/annotations/'+encodeURIComponent(D.saveFile),
    {method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)})
   .then(r=>r.json()).then(doc=>{
     existingChords=doc.chords||[]; existingMerges=doc.merges||[];
     chords.forEach(c=>c.dirty=false); saved=true; render();
     toast('Saved '+chords.length+' chord times &#10003;');
   }).catch(()=>toast('Save failed — check connection',true))
   .finally(()=>saveBtn.disabled=false);
});
const toastEl=document.getElementById('toast');
let toastT;
function toast(msg,err){ toastEl.innerHTML=msg; toastEl.className='toast on'+(err?' err':'');
  clearTimeout(toastT); toastT=setTimeout(()=>toastEl.className='toast'+(err?' err':''),1800); }

render();
window.addEventListener('resize',()=>{ if(sel>=0) drawRuler(sel); });
// deep-link: /annotator?song=..&sel=N preselects a chord (opens its ruler)
{ const s=parseInt(new URLSearchParams(location.search).get('sel'));
  if(!isNaN(s)&&s>=0&&s<chords.length) requestAnimationFrame(()=>select(s)); }
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
        chords.append({
            "i": idx, "bar": bar, "beat": beat,
            "section": c.get("section", ""), "label": c.get("label", ""),
            "t0": float(c.get("t0", 0.0)), "t1": float(c.get("t1", 0.0)),
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


@app.route("/annotator")
def annotator():
    """Manual chord-alignment tool. ?song=<slug> (default autumn_leaves)."""
    slug = re.sub(r"[^A-Za-z0-9_]", "", (request.args.get("song") or "autumn_leaves"))
    chords, tempo = _load_ireal_alignment(slug)
    if not chords:
        return (f"No iReal chart for '{slug}'. Expected docs/plots/irealb_{slug}.html "
                f"with a window.P payload.", 404)
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
    page = ANNOTATOR_TEMPLATE.replace("__ANNOT_DATA__", json.dumps(data))
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
