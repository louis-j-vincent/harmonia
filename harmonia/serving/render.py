"""harmonia/serving/render.py — chart-model / chart-render helpers (Phase 6b PORT).

Moved verbatim out of ``scripts/harmonia_server.py`` (Phase 6b of the serving
refactor). This is a behavior-preserving MOVE: the function body is byte-for-byte
the original ``_chart_model_for``; only a lazy dependency-binding preamble was
added at the top so the helper can live here without an import cycle.

``_chart_model_for`` is the single non-route helper in the server that wraps
``harmonia.output.chart_model`` (``payload_from_chart_html`` / ``to_chart_model``).
The persistent registries/offsets it needs (``_yt_audio_meta``, ``_yt_video_ids``,
``_load_bar1_offsets``) now live in ``harmonia.serving.state`` (Phase 6d) and are
imported at module top — state is a leaf module (no server import), so this is
safe and no longer needs the lazy back-import. The McGill-GT lookup
(``_gt_chords_for_video``) likewise now comes from ``harmonia.serving.billboard_gt``
(another leaf) at module top. It still depends on server-owned state out of scope
for this round (sidecar stores, the bar-1 offset transform,
``_raw_beat_times_cached``); those remain bound lazily from
``scripts.harmonia_server`` at call time. ``scripts/harmonia_server.py`` imports
``_chart_model_for`` back from here, so all existing call sites are unchanged.
"""

from __future__ import annotations

import os
from pathlib import Path

from harmonia.serving.audio import _raw_beat_times_cached
from harmonia.serving.config import AUDIO_DIR, PLOTS_DIR
from harmonia.serving.billboard_gt import _gt_chords_for_video
from harmonia.serving.loaders import _load_annotation
from harmonia.serving.state import _load_bar1_offsets, _yt_audio_meta, _yt_video_ids
from harmonia.serving.templates import _OVERLAY_HTML_TOOLS, _OVERLAY_HTML_YT


def _chart_model_for(filename: str, include_gt: bool = True) -> dict:
    """ChartModel for a rendered chart — payload + sidecar + audio/video links.

    ``include_gt``: attach McGill Billboard ground-truth chords (training-mode
    songs only — see _gt_chords_for_video) as model["gt"]. Skipped for the
    /api/library summary loop (chart_summary never reads it, and a mirdata
    lookup per song on every library load is needless overhead)."""
    from harmonia.output.chart_model import payload_from_chart_html, to_chart_model

    # Phase 6c architectural pass: the last server-owned deps this helper leaned
    # on are now extracted leaves, so the lazy ``import scripts.harmonia_server``
    # back-import is GONE — every name below is bound at module top.
    #   PLOTS_DIR / AUDIO_DIR              — harmonia.serving.config
    #   _yt_audio_meta / _yt_video_ids /
    #     _load_bar1_offsets              — harmonia.serving.state
    #   _gt_chords_for_video             — harmonia.serving.billboard_gt
    #   _load_annotation                 — harmonia.serving.loaders
    #   _raw_beat_times_cached           — harmonia.serving.audio
    #   _apply_bar1_offset_to_payload    — defined in THIS module (below)
    # The body is byte-for-byte the original _chart_model_for.
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
    # Default ON 2026-07-20 (validated->prod): corpus mean 84->27ms (-68%),
    # improves all 7 matched songs, zero regression (only chord/section
    # DISPLAY TIMING moves — labels, sections, folds are untouched, verified
    # byte-identical). Rollback: HARMONIA_BOUNDARY_SNAP=0.
    if os.environ.get("HARMONIA_BOUNDARY_SNAP", "1") == "1" and audio_url:
        bt = _raw_beat_times_cached(Path(audio_url).stem)
        if bt:
            model["beatTimes"] = bt
            # Snap the playhead's bar EDGES to the real beats (2026-07-30). The
            # client used to do this itself, per reconstructed chord span; it
            # now happens once here, on the authoritative per-bar map, with an
            # order guard so a compressed pass can't have a bar snapped away.
            try:
                from harmonia.output.chart_display import snap_bar_spans_to_beats
                snap_bar_spans_to_beats(model.get("sections") or [], bt)
            except Exception:
                pass
    return model


# ── Bar-1 offset payload transform (serving refactor, Phase 6c architectural
# pass) ────────────────────────────────────────────────────────────────────
# MOVED VERBATIM out of scripts/harmonia_server.py. This was the LAST consumer
# behind the render↔server lazy back-import: _chart_model_for (above) is its
# only caller, so relocating it here — next to that caller — lets the
# ``import scripts.harmonia_server as _srv`` block be deleted entirely. It reads
# nothing server-owned: a pure payload→payload transform whose one external dep,
# ``rebalance_near_boundary_onsets``, is pulled lazily in-body from
# scripts.render_youtube_chart (resolved at call time, scripts/ on sys.path).
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


# ── Presentation / injection helpers (serving refactor, render round) ──────
# Moved verbatim out of ``scripts/harmonia_server.py``: the PWA <head> snippet
# and the two pure string-injection helpers (overlay tools + back button).
# All are pure string manipulation (html in → html out); no beat/grid/model
# deps. The overlay HTML fragments they splice in (_OVERLAY_HTML_TOOLS /
# _OVERLAY_HTML_YT) already live in serving/templates.py and are imported
# above. ``scripts/harmonia_server.py`` imports these back so call sites are
# unchanged.

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


# Split out from _OVERLAY_HTML_TOOLS so the docked player's script can be
# gated separately (on window.HARM_AUDIO_URL) from the iReal Pro tools,
# which must run unconditionally. Some other page templates (render-tab,
# irealb-render, irealb-compare) still build their own separate embedded
# YouTube iframe with a #yt-player element — unrelated to this dock, which
# plays locally downloaded audio and never sets that id, so there's no
# collision risk between them.

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
