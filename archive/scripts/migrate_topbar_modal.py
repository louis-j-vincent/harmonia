"""One-off migration #4: retrofit the minimal-chrome topbar + bottom-sheet
modal redesign onto already-rendered chart HTML, without re-running
inference. Mirrors, verbatim, the edits applied to
harmonia/output/chart_interactive.py in this session:

  - .controls (always-visible toolbar) -> .topbar (key pill + options icon)
  - 3 separate drawers (Uncertainty/Scales/Jazzify) + motif buttons + legend
    -> one consolidated #optionsModal bottom sheet
  - the transpose wheel -> its own #wheelModal bottom sheet, opened from the
    key pill instead of sitting inline in the toolbar
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLOTS_DIR = REPO / "docs" / "plots"

REPLACEMENTS: list[tuple[str, str]] = []

# The body-markup block contains %%TITLE%%/%%SUB%%/%%COMPOSER%% placeholders
# in the *template*, but already-rendered files have those substituted with
# each song's real title/key/composer — so it can't be a static string pair.
# Matched separately, per file, with the captured values spliced into NEW.
_BODY_OLD_RE = re.compile(
    r'<body><div class="sheet">\n'
    r'  <h1>(?P<title>.*?)</h1>\n'
    r'  <div class="subhead"><span>(?P<sub>.*?)</span><span>(?P<composer>.*?)</span></div>\n'
    r'  <div class="controls">.*?<div id="motif-overlay">',
    re.DOTALL,
)
_BODY_NEW_TMPL = """<body><div class="sheet">
  <div class="topbar">
    <button type="button" class="pill" id="keyPillBtn" aria-label="Change key"><span id="keyPillLabel">Key</span></button>
    <h1>%%TITLE%%</h1>
    <button type="button" class="icon-btn" id="optionsBtn" aria-label="Options">
      <svg width="18" height="18" viewBox="0 0 24 24" fill="currentColor">
        <circle cx="5" cy="12" r="2.2"/><circle cx="12" cy="12" r="2.2"/><circle cx="19" cy="12" r="2.2"/>
      </svg>
    </button>
  </div>
  <div class="subhead"><span>%%SUB%%</span><span>%%COMPOSER%%</span></div>

  <!-- Transpose rotor — opened from the key pill -->
  <div class="modal" id="wheelModal">
    <div class="modal-backdrop" data-close></div>
    <div class="modal-panel">
      <div class="modal-handle"></div>
      <div class="transposeCtl">
        <div class="wheel" id="transposeWheel"></div>
        <span class="transposeLabel" id="transposeLabel"></span>
        <input type="hidden" id="transpose" value="0">
      </div>
    </div>
  </div>

  <!-- Everything else — opened from the options button -->
  <div class="modal" id="optionsModal">
    <div class="modal-backdrop" data-close></div>
    <div class="modal-panel">
      <div class="modal-handle"></div>

      <div class="opt-section">
        <div class="opt-info"><span>%%SUB%%</span><span>%%COMPOSER%%</span></div>
      </div>
      <hr>
      <div class="opt-section">
        <div class="opt-title">Uncertainty</div>
        <label>Level
          <select id="level">
            <option value="auto">Auto (certainty-gated)</option>
            <option value="family">Family (triad)</option>
            <option value="seventh">7th</option>
            <option value="exact">Exact</option>
          </select>
        </label>
        <label>Colour scale
          <select id="scale">
            <option value="warm">Warm</option>
            <option value="rg">Red → Green</option>
            <option value="gray">Grayscale</option>
          </select>
        </label>
        <label id="gate">Gate ≥ <span id="thv">0.60</span>
          <input type="range" id="thresh" min="0.4" max="0.95" step="0.05" value="0.6">
        </label>
        <span class="legend">unsure<span class="bar" id="legbar"></span>sure</span>
      </div>
      <hr>
      <div class="opt-section">
        <div class="opt-title">Scales</div>
        <label><input type="checkbox" id="hl"> Show scale bands</label>
        <label id="sv">View
          <select id="scaleview">
            <option value="one">Natural (one)</option>
            <option value="all">All fitting (jazz)</option>
          </select>
        </label>
      </div>
      <hr>
      <div class="opt-section">
        <div class="opt-title">Jazzify <span id="jv">0</span></div>
        <label>Intensity
          <input type="range" id="jazz" min="0" max="5" step="1" value="0">
        </label>
        <div class="opt-row">
          <button type="button" id="reroll" title="Resample with same intensity">Re-roll</button>
          <button type="button" id="resetbars" title="Reset per-bar overrides">Reset bars</button>
        </div>
        <label><input type="checkbox" id="fuse"> Synthesize view</label>
      </div>
      <hr>
      <div class="opt-section">
        <div class="opt-title">Motif analyser</div>
        <div class="opt-row">
          <button type="button" id="motifmode-btn"><span class="icon">◈</span> Motifs</button>
          <button type="button" id="motif-style-btn" title="Switch motif style">🌃 Neon Lights</button>
        </div>
      </div>
    </div>
  </div>

  <div id="motif-overlay">"""


def _migrate_body(text: str) -> tuple[str, bool]:
    m = _BODY_OLD_RE.search(text)
    if not m:
        return text, False
    new = (_BODY_NEW_TMPL
           .replace("%%TITLE%%", m.group("title"))
           .replace("%%SUB%%", m.group("sub"))
           .replace("%%COMPOSER%%", m.group("composer")))
    return text[:m.start()] + new + text[m.end():], True


def _pair(old: str, new: str) -> None:
    REPLACEMENTS.append((old, new))


# 1) header CSS: sheet/h1/subhead/controls/transposeCtl -> topbar/pill/icon-btn
_pair(
    """  .sheet { max-width:980px; margin:0 auto; padding:28px 32px 48px; }
  h1 { text-align:center; font-size:30px; margin:0 0 4px; }
  .subhead { display:flex; justify-content:space-between; color:var(--faint);
             font-style:italic; font-size:14px; margin-bottom:16px; }
  .controls { display:flex; gap:18px; flex-wrap:wrap; align-items:center;
              background:#efe9d9; border:1px solid #e2dac4; border-radius:10px;
              padding:11px 16px; margin-bottom:14px; font-family:system-ui,sans-serif;
              font-size:13px; color:#4a4636; }
  .controls label { display:flex; align-items:center; gap:7px; }
  .transposeCtl { display:flex; align-items:center; gap:9px; }""",
    """  .sheet { max-width:980px; margin:0 auto; padding:28px 32px 48px; }
  h1 { text-align:center; font-size:30px; margin:0 0 4px; }
  .subhead { display:flex; justify-content:space-between; color:var(--faint);
             font-style:italic; font-size:14px; margin-bottom:16px; }

  /* ── Minimal top bar: a key pill and an options button are the only chrome
     visible by default. Everything else — transpose, uncertainty, scales,
     jazzify, motifs — lives in on-demand bottom sheets, so the chord grid
     is the first and only thing you see when you open a chart. ── */
  .topbar { display:flex; align-items:center; gap:10px; margin-bottom:10px; }
  .topbar h1 { flex:1; min-width:0; margin:0; font-size:22px; text-align:center;
               overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
  .pill { display:inline-flex; align-items:center; gap:5px; background:#efe9d9;
          border:1px solid #e2dac4; border-radius:20px; padding:8px 15px;
          font:700 13px system-ui,sans-serif; color:#4a4636; cursor:pointer;
          flex:0 0 auto; transition:transform .1s ease, background .12s; }
  .pill::after { content:"⌄"; opacity:.5; font-size:10px; }
  .pill:active { transform:scale(.94); background:#e2d9c2; }
  .icon-btn { display:inline-flex; align-items:center; justify-content:center;
              width:36px; height:36px; border-radius:50%; background:#efe9d9;
              border:1px solid #e2dac4; color:#4a4636; cursor:pointer; flex:0 0 auto;
              transition:transform .1s ease, background .12s; }
  .icon-btn:active { transform:scale(.9); background:#e2d9c2; }
  .icon-btn svg { pointer-events:none; }
  .transposeCtl { display:flex; flex-direction:column; align-items:center; gap:12px; }""",
)

# 2) legend: no longer a toolbar sibling pushed right by margin-left:auto
_pair(
    """  .legend { display:flex; align-items:center; gap:8px; margin-left:auto; }
  .legend .bar { width:110px; height:12px; border-radius:6px; }""",
    """  .legend { display:flex; align-items:center; gap:8px; justify-content:center; }
  .legend .bar { width:110px; height:12px; border-radius:6px; }""",
)

# 3) drawer CSS -> modal CSS
_pair(
    """  /* ── Collapsible drawer controls ── */
  .drawer { position:relative; }
  .drawer-btn { padding:5px 12px; border:1px solid #cfc7ae; border-radius:7px;
                background:#efe9d9; cursor:pointer; font:600 12px system-ui,sans-serif;
                color:#4a4636; transition:background .12s; display:inline-flex;
                align-items:center; gap:5px; }
  .drawer-btn::after { content:"▾"; font-size:9px; opacity:.55; transition:transform .15s; }
  .drawer.open .drawer-btn { background:#ddd3be; }
  .drawer.open .drawer-btn::after { transform:rotate(-180deg); }
  .drawer-panel { display:none; position:absolute; top:calc(100% + 7px); left:0; z-index:200;
                  background:#f7f3e9; border:1px solid #cfc7ae; border-radius:10px;
                  padding:13px 16px; box-shadow:0 6px 22px #0002,0 1px 4px #0001;
                  flex-direction:column; gap:11px; white-space:nowrap;
                  font-family:system-ui,sans-serif; font-size:13px; color:#4a4636;
                  min-width:180px; }
  .drawer.open .drawer-panel { display:flex; }
  .drawer-panel label { display:flex; align-items:center; gap:8px; }
  .drawer-panel hr { border:none; border-top:1px solid #e2dac4; margin:0; }
  .drawer-panel button { padding:5px 12px; border:1px solid #cfc7ae; border-radius:6px;
                         background:#efe9d9; cursor:pointer; font:12px system-ui,sans-serif;
                         color:#4a4636; }
  .drawer-panel button:hover { background:#e0d5c0; }""",
    """  /* ── Bottom-sheet modal: rotor + the consolidated options panel ── */
  .modal { position:fixed; inset:0; z-index:600; display:none; }
  .modal.open { display:block; }
  .modal-backdrop { position:absolute; inset:0; background:#1c1c1c5c;
                     opacity:0; transition:opacity .22s ease; }
  .modal.open .modal-backdrop { opacity:1; }
  .modal-panel { position:absolute; left:50%; bottom:0; transform:translate(-50%,105%);
                 width:min(94vw,440px); max-height:78vh; overflow-y:auto;
                 background:#f7f3e9; border:1px solid #e2dac4; border-bottom:none;
                 border-radius:18px 18px 0 0; padding:10px 20px calc(20px + env(safe-area-inset-bottom));
                 box-shadow:0 -8px 30px #0003; font-family:system-ui,sans-serif;
                 font-size:13px; color:#4a4636; transition:transform .32s cubic-bezier(.32,.9,.35,1); }
  .modal.open .modal-panel { transform:translate(-50%,0); }
  .modal-handle { width:36px; height:4px; border-radius:2px; background:#cfc7ae;
                  margin:2px auto 12px; }
  .opt-section { display:flex; flex-direction:column; gap:10px; padding:12px 0; }
  .opt-section label { display:flex; align-items:center; gap:8px; }
  .opt-title { font:700 11px system-ui,sans-serif; text-transform:uppercase;
               letter-spacing:.05em; color:#8a8371; }
  .opt-row { display:flex; gap:8px; flex-wrap:wrap; }
  .opt-info { display:flex; justify-content:space-between; gap:10px; font-style:italic;
              color:#6b6050; font-size:12.5px; }
  .modal-panel hr { border:none; border-top:1px solid #e2dac4; margin:0; }
  .modal-panel button { padding:7px 14px; border:1px solid #cfc7ae; border-radius:8px;
                         background:#efe9d9; cursor:pointer; font:600 12px system-ui,sans-serif;
                         color:#4a4636; transition:transform .1s ease, background .12s; }
  .modal-panel button:active { transform:scale(.94); background:#e0d5c0; }""",
)

# 4) drop the now-unused .motif-toggle divider rule
_pair(
    """  /* ══ MOTIF ANALYSER ══ */
  .motif-toggle { border-left:1px solid #cfc7ae; padding-left:14px; margin-left:6px; }
  #motifmode-btn {""",
    """  /* ══ MOTIF ANALYSER ══ */
  #motifmode-btn {""",
)

# 5) neon-mode toolbar tint: .controls/.drawer-btn -> .pill/.icon-btn
_pair(
    """  body[data-motif-style="full"].motif-active .controls { background:#0c1522; border-color:#1e2d45; color:#8ab8d8; }
  body[data-motif-style="full"].motif-active .drawer-btn { background:#0c1522; border-color:#1e2d45; color:#8ab8d8; }""",
    """  body[data-motif-style="full"].motif-active .pill,
  body[data-motif-style="full"].motif-active .icon-btn { background:#0c1522; border-color:#1e2d45; color:#8ab8d8; }""",
)

# 6) mobile media query: drop drawer/controls-specific rules, now dead
_pair(
    """  @media (max-width: 640px) {
    /* clear the notch/status bar and the floating back button (standalone
       PWA mode has no Safari chrome to push content down for us) */
    .sheet { padding:calc(52px + env(safe-area-inset-top)) 8px 32px; }
    body { overscroll-behavior-y:none; -webkit-overflow-scrolling:touch; }
    h1 { font-size:22px; }
    .subhead { font-size:12px; flex-direction:column; gap:2px; }
    .controls { padding:12px 10px; gap:10px; font-size:12px;
                justify-content:center; }
    .transposeCtl { width:100%; justify-content:center; }
    .wheel { width:84px; height:84px; }
    .wheel button { width:24px; height:24px; margin:-12px 0 0 -12px; font-size:9px; }
    .wheel .hub { width:30px; height:30px; font-size:9px; }
    .transposeLabel { min-width:0; font-size:11px; }
    .motif-toggle { border-left:none; margin-left:0; padding-left:0; }
    .legend { margin-left:0; width:100%; justify-content:center; }
    .legend .bar { width:70px; }
    /* drawer popovers become a centred bottom sheet — an absolute popover
       anchored to its trigger button can run off a phone-width screen */
    .drawer-panel {
      position:fixed; left:50%; right:auto; top:auto;
      bottom:max(16px,env(safe-area-inset-bottom));
      transform:translateX(-50%);
      width:calc(100vw - 32px); max-width:360px;
      max-height:65vh; overflow-y:auto;
      white-space:normal; box-sizing:border-box; z-index:500;
    }
    .grid { grid-template-columns:repeat(4,1fr) !important; }
    .measure { min-height:66px; padding:4px 1px; }
    .chords { gap:2px; }
    .chord .root { font-size:24px; }
    .chord .qual { font-size:15px; }
    .seclabel { width:15px; height:15px; font-size:9px; }
    /* touch press feedback — no :hover on a phone, so give taps their own cue */
    .drawer-btn, #motifmode-btn, #motif-style-btn, .drawer-panel button {
      transition:transform .1s ease, background .1s ease;
    }
    .drawer-btn:active, #motifmode-btn:active, #motif-style-btn:active,
    .drawer-panel button:active { transform:scale(.93); }
    #motifpanel { font-size:12px; gap:10px; padding:10px 12px; }
    #motifstats { margin-left:0; }
  }""",
    """  @media (max-width: 640px) {
    /* clear the notch/status bar and the floating back button (standalone
       PWA mode has no Safari chrome to push content down for us) */
    .sheet { padding:calc(52px + env(safe-area-inset-top)) 8px 32px; }
    body { overscroll-behavior-y:none; -webkit-overflow-scrolling:touch; }
    .topbar h1 { font-size:17px; }
    /* the song info is duplicated compactly at the top of the Options sheet */
    .subhead { display:none; }
    .wheel { width:84px; height:84px; }
    .wheel button { width:24px; height:24px; margin:-12px 0 0 -12px; font-size:9px; }
    .wheel .hub { width:30px; height:30px; font-size:9px; }
    .transposeLabel { min-width:0; font-size:11px; }
    .legend .bar { width:70px; }
    .grid { grid-template-columns:repeat(4,1fr) !important; }
    .measure { min-height:66px; padding:4px 1px; }
    .chords { gap:2px; }
    .chord .root { font-size:24px; }
    .chord .qual { font-size:15px; }
    .seclabel { width:15px; height:15px; font-size:9px; }
    #motifpanel { font-size:12px; gap:10px; padding:10px 12px; }
    #motifstats { margin-left:0; }
  }""",
)

# 8) JS: key pill label follows the rotor
_pair(
    """  const label=keyLabel(P.home.tonic,P.home.mode,offset);
  document.getElementById("transposeLabel").textContent=label;
  const hub=document.getElementById("wheelHub");
  if(hub) hub.textContent=label.replace(/ major| minor/,"");
}""",
    """  const label=keyLabel(P.home.tonic,P.home.mode,offset);
  document.getElementById("transposeLabel").textContent=label;
  const hub=document.getElementById("wheelHub");
  if(hub) hub.textContent=label.replace(/ major| minor/,"");
  const pill=document.getElementById("keyPillLabel");
  if(pill) pill.textContent=label;
}""",
)

# 9) JS: drawer open/close -> modal open/close
_pair(
    """// ── Drawer open/close ──
document.querySelectorAll('.drawer-btn').forEach(btn=>{
  btn.addEventListener("click",e=>{
    e.stopPropagation();
    const drawer=btn.closest('.drawer');
    const wasOpen=drawer.classList.contains('open');
    // close all drawers first
    document.querySelectorAll('.drawer.open').forEach(d=>d.classList.remove('open'));
    if(!wasOpen) drawer.classList.add('open');
  });
});
document.addEventListener("click",()=>{
  document.querySelectorAll('.drawer.open').forEach(d=>d.classList.remove('open'));
});
document.querySelectorAll('.drawer-panel').forEach(p=>p.addEventListener("click",e=>e.stopPropagation()));""",
    """// ── Modal open/close (rotor + options bottom sheets) — only the key pill
// and the options button are visible by default; everything else lives here ──
function openModal(id){ closeAllModals(); document.getElementById(id).classList.add("open"); }
function closeAllModals(){ document.querySelectorAll(".modal.open").forEach(m=>m.classList.remove("open")); }
document.getElementById("keyPillBtn").addEventListener("click",()=>openModal("wheelModal"));
document.getElementById("optionsBtn").addEventListener("click",()=>openModal("optionsModal"));
document.querySelectorAll("[data-close]").forEach(el=>el.addEventListener("click",closeAllModals));
document.addEventListener("keydown",e=>{ if(e.key==="Escape") closeAllModals(); });""",
)


def main() -> None:
    files = sorted(PLOTS_DIR.glob("inferred_*.html"))
    patched, untouched = 0, 0
    for f in files:
        text = f.read_text(encoding="utf-8")
        text, body_hit = _migrate_body(text)
        hits = 1 if body_hit else 0
        for old, new in REPLACEMENTS:
            if old in text:
                text = text.replace(old, new, 1)
                hits += 1
        if hits == 0:
            untouched += 1
            print(f"skip {f.name}: no known blocks found (pre-dates this CSS lineage)")
            continue
        f.write_text(text, encoding="utf-8")
        patched += 1
        total = len(REPLACEMENTS) + 1
        print(f"patched {f.name} ({hits}/{total} blocks" + ("" if hits == total else ", partial") + ")")
    print(f"\n{patched} patched, {untouched} untouched")


if __name__ == "__main__":
    sys.exit(main())
