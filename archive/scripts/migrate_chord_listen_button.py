#!/usr/bin/env python3
"""Migrate existing baked charts to add the Annotate-tab "Listen to real audio"
button (🎧) beside the chord editor's synth-preview play button.

2026-07-17: the chord editor already had a ▶ button that plays a *synthesized*
arpeggio of the current label. This adds a second button that plays the EXACT
[t0,t1) span of the song's downloaded recording for the chord being corrected —
so you can hear what's actually in the audio while choosing the right label.
Served by /api/chord-snippet (harmonia.models.audio_snippet, bleed-fixed
frame-clip convention).

The three inserted blocks (button HTML, CSS, JS handler) are byte-identical to
what harmonia/output/chart_interactive.py now emits, so:
  • newly-rendered charts already contain them (this script skips those), and
  • re-running this migration is a no-op on an up-to-date chart.

Usage:
    python scripts/migrate_chord_listen_button.py
"""
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLOTS_DIR = REPO / "docs" / "plots"

# ── Anchors (old, exactly as prior renders emitted them) ─────────────────────
_OLD_BTN = ('<button type="button" id="ce-play" title="Play this chord" '
            'aria-label="Play chord">&#9654;</button>')
_NEW_BTN = ('<button type="button" id="ce-play" title="Play this chord (synth)" '
            'aria-label="Play chord">&#9654;</button>\n'
            '          <button type="button" id="ce-listen" '
            'title="Listen to the real recording of this exact chord" '
            'aria-label="Listen to real audio">&#127911;</button>')

_OLD_CSS = ("  #ce-play { padding:4px 9px; font-size:11px; border-radius:50%; width:26px; height:26px;\n"
            "             display:flex; align-items:center; justify-content:center; }")
_NEW_CSS = _OLD_CSS + (
    "\n  #ce-listen { padding:4px 9px; font-size:12px; border-radius:50%; width:26px; height:26px;\n"
    "             display:flex; align-items:center; justify-content:center; margin-left:4px; }\n"
    "  #ce-listen.loading { opacity:0.55; }\n"
    "  #ce-listen.playing { background:#8a2b2b; color:#f7f3e9; border-color:#8a2b2b; }")

_OLD_JS = "  document.getElementById('ce-play').addEventListener('click', ()=>playChordArpeggio(ceRoot, ceQual));"
_NEW_JS = _OLD_JS + """

  // ── "Listen to the real audio" — plays the EXACT [t0,t1) span of the song's
  // downloaded recording for the chord being edited, so you can hear what's
  // actually there while deciding the correct label. Fetches a sample-accurate
  // WAV from /api/chord-snippet (bleed-fixed frame-clip convention); no synth.
  // Clips are cached per-span in-memory (blob URLs) so re-taps are instant. ──
  const ceListenBtn = document.getElementById('ce-listen');
  let ceSnippetAudio = null;         // single reusable <audio> element
  const ceSnippetCache = new Map();  // "t0:t1" -> object URL
  function ceStopSnippet(){
    if(ceSnippetAudio){ ceSnippetAudio.pause(); }
    ceListenBtn.classList.remove('playing','loading');
  }
  async function ceListen(){
    if(_editingIdx<0) return;
    const c = P.chords[_editingIdx];
    if(c==null || c.t0==null || c.t1==null){
      ceListenBtn.title = 'No audio span for this chord'; return;
    }
    // Toggle: a second tap while playing stops it.
    if(ceSnippetAudio && !ceSnippetAudio.paused){ ceStopSnippet(); return; }
    const filename = location.pathname.split('/').pop();
    const key = c.t0.toFixed(3)+':'+c.t1.toFixed(3);
    try{
      let url = ceSnippetCache.get(key);
      if(!url){
        ceListenBtn.classList.add('loading');
        const qs = '?t0='+encodeURIComponent(c.t0)+'&t1='+encodeURIComponent(c.t1);
        const r = await fetch('/api/chord-snippet/'+encodeURIComponent(filename)+qs);
        if(!r.ok) throw new Error('snippet '+r.status);
        url = URL.createObjectURL(await r.blob());
        ceSnippetCache.set(key, url);
        ceListenBtn.classList.remove('loading');
      }
      if(!ceSnippetAudio){
        ceSnippetAudio = new Audio();
        ceSnippetAudio.addEventListener('ended', ceStopSnippet);
        ceSnippetAudio.addEventListener('pause', ()=>ceListenBtn.classList.remove('playing'));
      }
      ceSnippetAudio.src = url;
      ceListenBtn.classList.add('playing');
      await ceSnippetAudio.play();
    }catch(err){
      ceListenBtn.classList.remove('loading','playing');
      ceListenBtn.title = 'Audio unavailable for this chord';
    }
  }
  ceListenBtn.addEventListener('click', ceListen);"""


def migrate_chart(p: Path) -> str:
    html = p.read_text(encoding="utf-8")
    if 'id="ce-listen"' in html:
        return "skip (already has ce-listen)"
    if _OLD_BTN not in html:
        return "skip (no ce-play editor button — not an annotate chart)"
    missing = [n for n, s in (("css", _OLD_CSS), ("js", _OLD_JS)) if s not in html]
    if missing:
        return f"WARN: button present but missing anchors {missing} — left untouched"
    html = html.replace(_OLD_BTN, _NEW_BTN, 1)
    html = html.replace(_OLD_CSS, _NEW_CSS, 1)
    html = html.replace(_OLD_JS, _NEW_JS, 1)
    p.write_text(html, encoding="utf-8")
    return "migrated"


def main():
    charts = sorted(PLOTS_DIR.glob("inferred_*.html"))
    counts: dict[str, int] = {}
    for p in charts:
        res = migrate_chart(p)
        tag = res.split(" ")[0]
        counts[tag] = counts.get(tag, 0) + 1
        if tag not in ("skip",):
            print(f"{p.name}: {res}")
    print(f"\n{len(charts)} charts — " + ", ".join(f"{k}:{v}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
