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
import logging
import os
import re
import shutil
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

from flask import Flask, Response, jsonify, redirect, render_template_string, request

log = logging.getLogger(__name__)

PLOTS_DIR = REPO / "docs" / "plots"
PWA_DIR = REPO / "docs" / "pwa"

_PWA_HEAD = """<link rel="manifest" href="/pwa/manifest.json">
<link rel="apple-touch-icon" href="/pwa/apple-touch-icon.png">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="black-translucent">
<meta name="apple-mobile-web-app-title" content="Harmonia">
<meta name="theme-color" content="#8a2b2b">
<style>
/* injected by harmonia_server.py so already-rendered charts get phone-width
   layout too, even if chart_interactive.py's own media query predates them */
@media (max-width: 640px) {
  .sheet { padding:16px 8px 32px !important; }
  h1 { font-size:22px !important; }
  .controls { padding:12px 10px !important; gap:10px !important; font-size:12px !important;
              justify-content:center !important; }
  .transposeCtl { width:100% !important; justify-content:center !important; }
  .motif-toggle { border-left:none !important; margin-left:0 !important; padding-left:0 !important; }
  .legend { margin-left:0 !important; width:100% !important; justify-content:center !important; }
  .grid { grid-template-columns:repeat(4,1fr) !important; }
  .measure { min-height:66px !important; padding:4px 1px !important; }
  .chords { gap:2px !important; }
  .chord .root { font-size:24px !important; }
  .chord .qual { font-size:15px !important; }
  /* drawer popovers become a centred bottom sheet so they can never run
     off a phone-width screen, regardless of where their trigger sits */
  .drawer-panel {
    position:fixed !important; left:50% !important; right:auto !important; top:auto !important;
    bottom:max(16px,env(safe-area-inset-bottom)) !important;
    transform:translateX(-50%) !important;
    width:calc(100vw - 32px) !important; max-width:360px !important;
    max-height:65vh !important; overflow-y:auto !important;
    white-space:normal !important; box-sizing:border-box !important; z-index:500 !important;
  }
}
@media (max-width: 360px) {
  .grid { grid-template-columns:repeat(2,1fr) !important; }
  .measure { min-height:80px !important; }
  .chord .root { font-size:30px !important; }
  .chord .qual { font-size:19px !important; }
}
</style>"""

app = Flask(__name__, static_folder=None)

# ── CLI args stored globally so routes can read them ─────────────────────────
_ARGS: argparse.Namespace | None = None

# ── In-progress jobs: {job_id: {"status": ..., "url": ..., "out": ...}} ─────
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()

# ── YouTube video ID registry: {html_filename → video_id} ────────────────────
_yt_video_ids: dict[str, str] = {}


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

_OVERLAY_HTML = r"""
<style>
/* ── FAB buttons ─────────────────────────────────────── */
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

<div id="harm-fabs">
  <button class="harm-fab" id="tab-fab"
          onclick="document.getElementById('tab-modal-bg').classList.add('open');document.getElementById('tab-title').focus()">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M9 19c-5 1.5-5-2.5-7-3m14 6v-3.87a3.37 3.37 0 0 0-.94-2.61c3.14-.35 6.44-1.54 6.44-7A5.44 5.44 0 0 0 20 4.77 5.07 5.07 0 0 0 19.91 1S18.73.65 16 2.48a13.38 13.38 0 0 0-7 0C6.27.65 5.09 1 5.09 1A5.07 5.07 0 0 0 5 4.77a5.44 5.44 0 0 0-1.5 3.78c0 5.42 3.3 6.61 6.44 7A3.37 3.37 0 0 0 9 18.13V22"/>
    </svg>
    Guitar Tabs
  </button>
  <button class="harm-fab" id="irealb-fab"
          onclick="document.getElementById('irealb-modal-bg').classList.add('open');document.getElementById('irealb-title').focus()">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
      <path d="M9 18V5l12-2v13"/><circle cx="6" cy="18" r="3"/><circle cx="18" cy="16" r="3"/>
    </svg>
    iReal Pro
  </button>
  <button class="harm-fab" id="yt-fab"
          onclick="document.getElementById('yt-modal-bg').classList.add('open');document.getElementById('yt-url').focus()">
    <svg viewBox="0 0 24 24" fill="currentColor">
      <path d="M23.5 6.2a3 3 0 0 0-2.1-2.1C19.5 3.5 12 3.5 12 3.5s-7.5 0-9.4.6A3 3 0 0 0 .5 6.2 31 31 0 0 0 0 12a31 31 0 0 0 .5 5.8 3 3 0 0 0 2.1 2.1c1.9.6 9.4.6 9.4.6s7.5 0 9.4-.6a3 3 0 0 0 2.1-2.1A31 31 0 0 0 24 12a31 31 0 0 0-.5-5.8z"/>
      <polygon points="9.7 15.5 15.8 12 9.7 8.5 9.7 15.5" fill="#f7f3e9"/>
    </svg>
    Analyze YouTube
  </button>
</div>

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
    setYtStatus('','');
    document.getElementById('yt-go').disabled=false;
  }
  window.closeYtModal = closeYtModal;
  window.closeModal   = closeYtModal;

  function setYtStatus(msg,cls){
    const s=document.getElementById('yt-status');
    s.textContent=msg; s.className='harm-status '+(cls||'');
  }

  function pollJob(jobId){
    fetch('/api/job/'+jobId).then(r=>r.json()).then(d=>{
      if(d.status==='done'){ window.location.href=d.url; }
      else if(d.status==='error'){
        document.getElementById('yt-go').disabled=false;
        setYtStatus(d.error||'Analysis failed.','err');
      } else {
        setYtStatus(d.message||'Processing…','');
        setTimeout(()=>pollJob(jobId),1500);
      }
    }).catch(()=>setYtStatus('Server error.','err'));
  }

  window.startAnalysis = function(){
    const url=document.getElementById('yt-url').value.trim();
    if(!url){ setYtStatus('Please enter a YouTube URL.','err'); return; }
    document.getElementById('yt-go').disabled=true;
    setYtStatus('Submitting…','');
    fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url})})
      .then(r=>r.json())
      .then(d=>{
        if(d.error){ setYtStatus(d.error,'err'); document.getElementById('yt-go').disabled=false; return; }
        setYtStatus('Downloading…','');
        pollJob(d.job_id);
      })
      .catch(()=>{ setYtStatus('Could not reach server.','err'); document.getElementById('yt-go').disabled=false; });
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

<!-- ── YouTube sync player (only active when YT_VIDEO_ID is set) ── -->
<style>
#yt-player-dock{
  display:none; position:fixed; bottom:0; left:0; right:0; z-index:9990;
  background:#111; box-shadow:0 -4px 24px #0008;
  display:flex; align-items:stretch; gap:0;
}
#yt-player-dock.hidden { display:none !important; }
#yt-iframe-wrap {
  flex:0 0 auto; width:320px; height:180px; background:#000;
}
#yt-iframe-wrap iframe { width:100%; height:100%; display:block; border:none; }
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
  <div id="yt-iframe-wrap"><div id="yt-player"></div></div>
  <div id="yt-dock-info">
    <div id="yt-dock-title">Loading…</div>
    <div id="yt-dock-chord"></div>
    <div id="yt-dock-controls">
      <button onclick="ytPlayer&&ytPlayer.seekTo(0,true)">⏮ Restart</button>
      <button id="yt-playpause" onclick="ytTogglePlay()">⏸ Pause</button>
      <span id="yt-dock-time" style="color:#888;font-size:11px;margin-left:4px"></span>
    </div>
  </div>
</div>

<script>
(function(){
  if(!window.YT_VIDEO_ID) return;   // only runs on YouTube-derived charts

  // Build timing array from P.chords (populated after chart script runs)
  function getChordTimes(){
    if(typeof P==='undefined') return [];
    return P.chords.map((c,i)=>({idx:i, t0:c.t0??null, t1:c.t1??null}))
                   .filter(c=>c.t0!==null);
  }

  // Find which chord is playing at time t
  function chordAt(times, t){
    if(!times.length) return -1;
    // last chord whose t0 <= t (and t < t1 if available)
    let best=-1;
    for(let i=0;i<times.length;i++){
      if(times[i].t0<=t){
        if(times[i].t1===null || t<times[i].t1) best=times[i].idx;
        else if(times[i].t1!==null && t>=times[i].t1) best=times[i].idx; // past end, keep updating
      }
    }
    // refine: find tightest bracket
    best=-1;
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
    // iReal Pro charts store the label directly
    if(c.label) return c.label;
    const SHARP=["C","C♯","D","D♯","E","F","F♯","G","G♯","A","A♯","B"];
    const root=c.root>=0?SHARP[c.root]:'';
    // Use the same level/quality the chart is currently showing
    const lv=c.lv?.seventh||c.lv?.family||{};
    let q=lv.q||'';
    // Typeset quality a bit
    if(q===''||q==='maj') q='';
    else if(q==='-'||q==='min') q='m';
    else if(q==='-7') q='m7';
    else if(q==='^7') q='△7';
    else if(q==='h7') q='ø7';
    else if(q==='o') q='°';
    return root+q;
  }

  // ── iReal Pro modal ─────────────────────────────────────────────────
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

  let ytPlayer=null, _currentChordIdx=-1, _chordTimes=[], _rafId=null;
  window.ytPlayer=null;

  // Scroll the active chord into view
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

  function syncLoop(){
    if(!ytPlayer||typeof ytPlayer.getCurrentTime!=='function'){
      _rafId=requestAnimationFrame(syncLoop); return;
    }
    const state=ytPlayer.getPlayerState?.()??-1;
    const t=ytPlayer.getCurrentTime();
    document.getElementById('yt-dock-time').textContent=fmtTime(t);

    // Update play/pause button
    const pp=document.getElementById('yt-playpause');
    if(pp) pp.textContent = (state===1)?'⏸ Pause':'▶ Play';

    const newIdx=chordAt(_chordTimes, t);
    if(newIdx!==_currentChordIdx){
      // Remove old highlight
      if(_currentChordIdx>=0){
        const old=document.getElementById('chord-'+_currentChordIdx);
        if(old) old.classList.remove('chord-now-playing');
      }
      // Apply new highlight
      if(newIdx>=0){
        const el=document.getElementById('chord-'+newIdx);
        if(el){
          el.classList.add('chord-now-playing');
          scrollToChord(newIdx);
        }
        document.getElementById('yt-dock-chord').textContent=chordLabel(newIdx);
      }
      _currentChordIdx=newIdx;
    }
    _rafId=requestAnimationFrame(syncLoop);
  }

  window.ytTogglePlay=function(){
    if(!ytPlayer) return;
    const state=ytPlayer.getPlayerState?.()??-1;
    if(state===1) ytPlayer.pauseVideo(); else ytPlayer.playVideo();
  };

  // YouTube IFrame API callback
  window.onYouTubeIframeAPIReady=function(){
    ytPlayer=new YT.Player('yt-player',{
      videoId: window.YT_VIDEO_ID,
      playerVars:{autoplay:0,modestbranding:1,rel:0,controls:1,origin:window.location.origin},
      events:{
        onReady: function(e){
          window.ytPlayer=ytPlayer;
          window._ytPlayer=ytPlayer;
          _chordTimes=getChordTimes();
          // Set title
          const title=document.querySelector('h1');
          if(title) document.getElementById('yt-dock-title').textContent=title.textContent.trim();
          document.getElementById('yt-player-dock').classList.remove('hidden');
          // Add bottom padding to page so dock doesn't cover last bars
          document.body.style.paddingBottom='196px';
          syncLoop();
        },
        onStateChange: function(e){ /* syncLoop handles everything */ }
      }
    });
  };

  // Load the IFrame API
  const tag=document.createElement('script');
  tag.src='https://www.youtube.com/iframe_api';
  document.head.appendChild(tag);
})();
</script>
</body></html>"""

_INJECT_MARKER = "</body></html>"

_BACK_BUTTON_HTML = """<a href="/" id="harm-back" onclick="if(history.length>1){history.back();return false;}"
   style="position:fixed;top:max(12px,env(safe-area-inset-top));left:12px;z-index:9998;
   display:flex;align-items:center;gap:5px;background:#8a2b2bcc;color:#fff;
   text-decoration:none;font:700 13px system-ui,sans-serif;padding:7px 13px 7px 10px;
   border-radius:20px;box-shadow:0 2px 8px #0004;backdrop-filter:blur(4px);">&larr; Charts</a>
"""


def _inject_overlay(html: str) -> str:
    """Inject the YouTube overlay into a chart HTML page."""
    if _INJECT_MARKER in html:
        return html.replace(_INJECT_MARKER, _OVERLAY_HTML, 1)
    return html + _OVERLAY_HTML


def _inject_back_button(html: str) -> str:
    """Add a fixed 'back to chart list' link — standalone PWA mode has no
    Safari chrome, so there's otherwise no way back off a chart page."""
    return html.replace("<body>", "<body>" + _BACK_BUTTON_HTML, 1)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/pwa/<path:filename>")
def serve_pwa_asset(filename):
    """Serve the PWA manifest and home-screen icons."""
    p = PWA_DIR / filename
    if not p.exists() or p.parent != PWA_DIR:
        return "Not found", 404
    mimetype = "application/manifest+json" if p.suffix == ".json" else None
    return Response(p.read_bytes(), mimetype=mimetype)


@app.route("/")
def index():
    """List all available chart HTML files."""
    charts = sorted(PLOTS_DIR.glob("inferred_*.html"))
    items = [{"name": p.stem.replace("inferred_", "").replace("_", " ").title(),
              "file": p.name} for p in charts]
    page = render_template_string(INDEX_TEMPLATE, charts=items)
    return Response(page.replace("</head>", _PWA_HEAD + "</head>", 1), mimetype="text/html")


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
    return Response(_inject_back_button(_inject_overlay(content)), mimetype="text/html")


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
        _yt_video_ids[out.name] = vid
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
        _yt_video_ids[out.name] = vid

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
        _yt_video_ids[out.name] = vid

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
        _yt_video_ids[out.name] = vid
    return jsonify(url=f"/chart/{out.name}")


def _run_analysis(job_id: str, url: str) -> None:
    def update(status, message="", **kw):
        with _jobs_lock:
            _jobs[job_id].update(status=status, message=message, **kw)

    tmp_dir = Path(tempfile.mkdtemp(prefix="harmonia_yt_"))
    try:
        update("running", "Downloading audio…")

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

        update("running", f'Running chord inference on "{video_title}"…')

        from harmonia.models.chord_pipeline_v1 import infer_chords_v1
        pipeline_chart = infer_chords_v1(
            audio_path,
            cache_dir=Path(_ARGS.cache_dir),
        )

        update("running", "Rendering chart…")

        from scripts.render_youtube_chart import chart_to_interactive_inputs
        from harmonia.output.chart_interactive import render_interactive

        source_desc = f"inferred from YouTube · {url}"
        chart_obj, chord_dicts = chart_to_interactive_inputs(pipeline_chart, video_title, source_desc)

        slug = re.sub(r"[^a-z0-9]+", "_", video_title.lower()).strip("_") or "yt"
        out = PLOTS_DIR / f"inferred_{slug[:60]}.html"
        render_interactive(chart_obj, chord_dicts, out, bars_per_row=4)

        vid = _extract_video_id(url)
        if vid:
            _yt_video_ids[out.name] = vid

        update("done", url=f"/chart/{out.name}")

    except Exception as e:
        log.exception("Analysis failed for %s", url)
        update("error", error=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


# ── Index template ─────────────────────────────────────────────────────────────

INDEX_TEMPLATE = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Harmonia</title>
<style>
  :root { --paper:#f7f3e9; --ink:#1c1c1c; --rule:#b9b09a; --accent:#8a2b2b; }
  body { background:var(--paper); color:var(--ink); margin:0;
         font-family:Georgia,'Times New Roman',serif; }
  .wrap { max-width:640px; margin:0 auto; padding:48px 32px; }
  h1 { font-size:34px; margin:0 0 6px; }
  .sub { color:#8a8371; font-style:italic; margin-bottom:36px; font-size:15px; }
  ul { list-style:none; padding:0; margin:0 0 36px; }
  li { border-bottom:1px solid var(--rule); }
  li a { display:block; padding:13px 4px; text-decoration:none; color:var(--ink);
         font-size:17px; transition:color .12s; }
  li a:hover { color:var(--accent); }
  .yt-section { background:#efe9d9; border:1px solid #e2dac4; border-radius:10px;
                padding:20px 24px; font-family:system-ui,sans-serif; }
  .yt-section h2 { margin:0 0 8px; font-size:16px; }
  .yt-section p { margin:0 0 12px; font-size:13px; color:#6b6050; }
  .yt-row { display:flex; gap:10px; }
  #yt-url { flex:1; padding:9px 12px; border:1.5px solid #cfc7ae; border-radius:8px;
            font-size:14px; background:#fff; }
  #yt-url:focus { outline:none; border-color:var(--accent); }
  #yt-go { padding:9px 20px; background:var(--accent); color:#fff; border:none;
           border-radius:8px; font:700 14px system-ui,sans-serif; cursor:pointer; }
  #yt-go:hover { background:#a83333; }
  #yt-status { margin-top:10px; font-size:13px; color:#4a4636; min-height:18px; }
  #yt-status.err { color:var(--accent); }
  #yt-spinner { display:none; width:18px; height:18px; border-radius:50%;
                border:2.5px solid #cfc7ae; border-top-color:var(--accent);
                animation:spin .7s linear infinite; display:inline-block; vertical-align:middle; }
  @keyframes spin { to { transform:rotate(360deg); } }
</style>
</head><body>
<div class="wrap">
  <h1>Harmonia</h1>
  <p class="sub">Interactive chord charts</p>

  <ul>
  {% for c in charts %}
    <li><a href="/chart/{{ c.file }}">{{ c.name }}</a></li>
  {% endfor %}
  </ul>

  <div class="yt-section">
    <h2>Analyze a YouTube song</h2>
    <p>Paste a URL — Harmonia downloads the audio, infers chords, and opens an interactive chart.</p>
    <div class="yt-row">
      <input id="yt-url" type="url" placeholder="https://www.youtube.com/watch?v=…"
             onkeydown="if(event.key==='Enter')startAnalysis()">
      <button id="yt-go" onclick="startAnalysis()">Analyze</button>
    </div>
    <div id="yt-status"></div>
  </div>
</div>
<script>
function setStatus(msg,cls){const s=document.getElementById('yt-status');s.textContent=msg;s.className=cls||'';}
function poll(jobId){
  fetch('/api/job/'+jobId).then(r=>r.json()).then(d=>{
    if(d.status==='done'){ window.location.href=d.url; }
    else if(d.status==='error'){ setStatus(d.error||'Failed.','err'); document.getElementById('yt-go').disabled=false; }
    else { setStatus(d.message||'Processing…',''); setTimeout(()=>poll(jobId),1500); }
  });
}
function startAnalysis(){
  const url=document.getElementById('yt-url').value.trim();
  if(!url){setStatus('Please enter a YouTube URL.','err');return;}
  document.getElementById('yt-go').disabled=true;
  setStatus('Submitting…','');
  fetch('/api/analyze',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({url})})
    .then(r=>r.json())
    .then(d=>{
      if(d.error){setStatus(d.error,'err');document.getElementById('yt-go').disabled=false;return;}
      setStatus('Downloading…','');
      poll(d.job_id);
    });
}
</script>
</body></html>"""


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
