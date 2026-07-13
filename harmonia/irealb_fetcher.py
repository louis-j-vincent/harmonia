"""Fetch and render iReal Pro chord charts.

Two entry points:
  search_community(query)       — scrape ireal.pro community for matching songs
  render_irealb_chart(url, …)   — convert an irealb:// URL → interactive HTML

The rendered HTML exposes window.P.chords with {label, t0, t1} so the existing
YouTube sync overlay (harmonia_server._OVERLAY_HTML) can highlight chords in
real time. Timestamps are derived from BPM + chart_offset_s.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request

from pyRealParser import Tune

from harmonia.data.ireal_corpus import sectionized_measures, split_chords, tune_to_mma

_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Harmonia/1.0"


# ── Community search ──────────────────────────────────────────────────────────

def search_community(query: str, max_results: int = 8) -> list[dict]:
    """Search iReal Pro community (irealpro.com/music) for songs.

    Returns list of dicts: {title, composer, key, style, time_sig, irealb_url}
    The irealb_url is URL-encoded (as it appears in the HTML) — pass it as-is
    to render_irealb_chart which will decode it before parsing.
    """
    search_url = "https://www.irealpro.com/music/?s=" + urllib.parse.quote(query)
    req = urllib.request.Request(search_url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        raise RuntimeError(f"iReal community search failed: {exc}") from exc

    # Decode HTML entities, then extract irealb:// URLs
    html = re.sub(r"&#0*38;|&amp;", "&", html)
    raw_urls = re.findall(r"irealb://[^\s\"'<>]+", html)

    results: list[dict] = []
    seen: set[str] = set()
    for raw_url in raw_urls:
        if len(results) >= max_results:
            break
        try:
            decoded = urllib.parse.unquote(raw_url)
            tunes = Tune.parse_ireal_url(decoded)
        except Exception:
            continue
        for tune in tunes:
            if len(results) >= max_results:
                break
            if tune.title in seen:
                continue
            seen.add(tune.title)
            ts = tune.time_signature or (4, 4)
            results.append({
                "title":      tune.title,
                "composer":   tune.composer or "",
                "key":        tune.key or "",
                "style":      tune.style or "",
                "time_sig":   f"{ts[0]}/{ts[1]}",
                "irealb_url": raw_url,   # keep raw (URL-encoded)
            })

    return results


# ── Chart renderer ────────────────────────────────────────────────────────────

def render_irealb_chart(
    irealb_url: str,
    *,
    chart_offset_s: float = 0.0,
    tempo_override: int | None = None,
) -> str:
    """Convert an irealb:// URL to a standalone interactive HTML chord chart.

    window.P.chords  — [{label, t0, t1, bar, section}] for YouTube sync
    Chord elements   — <div id="chord-{i}"> for highlighting by the overlay
    Timestamps       — chart_offset_s + beat_index * (60 / bpm)
    """
    decoded = urllib.parse.unquote(irealb_url)
    tunes = Tune.parse_ireal_url(decoded)
    if not tunes:
        raise ValueError("No tunes found in irealb URL")
    tune = tunes[0]

    mma = tune_to_mma(tune, tempo=tempo_override)
    bpm = mma.tempo
    spb = 60.0 / bpm  # seconds per beat

    # ── Flatten to per-chord timeline ──────────────────────────────────
    p_chords: list[dict] = []
    total_beats = 0
    for bar_no, section, slots in mma.timeline:
        bpb = mma.beats_per_bar
        for k, (beat_offset, ireal_token, _mma_chord) in enumerate(slots):
            next_beat = slots[k + 1][0] if k + 1 < len(slots) else bpb
            dur = max(next_beat - beat_offset, 1)
            abs_beat = total_beats + beat_offset
            t0 = chart_offset_s + abs_beat * spb
            t1 = chart_offset_s + (abs_beat + dur) * spb
            # Clean up raw token (strip iReal marker junk)
            label = re.sub(r"[npWNQUSr]+$", "", ireal_token).strip() or "N.C."
            p_chords.append({
                "label":   label,
                "t0":      round(t0, 3),
                "t1":      round(t1, 3),
                "bar":     bar_no - 1,   # 0-indexed
                "section": section,
            })
        total_beats += bpb

    # ── Group chords by bar ────────────────────────────────────────────
    bars: list[list[int]] = []   # bars[i] = list of chord indices
    current_bar = -1
    for i, ch in enumerate(p_chords):
        if ch["bar"] != current_bar:
            bars.append([])
            current_bar = ch["bar"]
        bars[-1].append(i)

    # ── Build HTML grid: 4 bars per row ───────────────────────────────
    grid_html = ""
    bar_ptr = 0
    while bar_ptr < len(bars):
        row_bars = bars[bar_ptr: bar_ptr + 4]

        # Section label for this row (show if it changed)
        first_sec = p_chords[row_bars[0][0]]["section"] if row_bars else ""
        prev_sec  = p_chords[bars[bar_ptr - 1][0]]["section"] if bar_ptr > 0 else ""
        sec_html  = (f'<span>{_esc(first_sec)}</span>'
                     if first_sec and first_sec != prev_sec else "")

        row_html = f'<div class="ir-row"><div class="ir-sec">{sec_html}</div>'
        for chord_indices in row_bars:
            row_html += '<div class="ir-bar">'
            for ci in chord_indices:
                row_html += (f'<div class="ir-cell" id="chord-{ci}">'
                             f'{_esc(p_chords[ci]["label"])}</div>')
            row_html += '</div>'
        for _ in range(4 - len(row_bars)):
            row_html += '<div class="ir-bar ir-empty"></div>'
        row_html += '</div>'
        grid_html += row_html
        bar_ptr += 4

    ts = mma.time_signature or (4, 4)
    total_bars = len(bars)
    p_json = json.dumps({"chords": p_chords, "tempo": bpm})

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(tune.title)} — iReal Pro Chart</title>
<style>
  :root{{--paper:#f7f3e9;--ink:#1c1c1c;--rule:#b9b09a;--accent:#8a2b2b;--faint:#8a8371;
         --teal:#00c9a7;}}
  *{{box-sizing:border-box;}}
  body{{background:var(--paper);color:var(--ink);margin:0;
       font-family:Georgia,'Times New Roman',serif;}}
  .sheet{{max-width:900px;margin:0 auto;padding:28px 28px 80px;}}
  h1{{text-align:center;font-size:24px;margin:0 0 4px;}}
  .meta{{text-align:center;color:var(--faint);font-style:italic;font-size:13px;margin-bottom:20px;}}
  .ir-grid{{display:flex;flex-direction:column;gap:3px;
            border-top:2px solid var(--accent);border-bottom:2px solid var(--accent);
            padding:8px 0;}}
  .ir-row{{display:grid;grid-template-columns:28px repeat(4,1fr);gap:3px;align-items:stretch;}}
  .ir-sec{{display:flex;align-items:flex-start;justify-content:center;padding-top:6px;}}
  .ir-sec span{{font-family:system-ui,sans-serif;font-size:10px;font-weight:700;
    color:var(--accent);border:1.5px solid var(--accent);border-radius:3px;
    padding:1px 4px;line-height:1;}}
  .ir-bar{{display:flex;gap:0;border:1px solid var(--rule);border-radius:3px;
           min-height:48px;background:#fff;overflow:hidden;}}
  .ir-empty{{border:1px dashed #e0d8c0;background:transparent;}}
  .ir-cell{{flex:1;display:flex;align-items:center;justify-content:center;
            font-family:'Menlo','Courier New',monospace;font-size:13px;color:var(--ink);
            padding:4px 3px;text-align:center;line-height:1.25;}}
  .ir-cell+.ir-cell{{border-left:1px solid var(--rule);}}
  .chord-now-playing{{
    background:rgba(0,201,167,0.25)!important;
    outline:2px solid var(--teal);outline-offset:1px;border-radius:2px;
  }}
  .ir-source{{font-family:system-ui,sans-serif;font-size:11px;color:var(--faint);
    margin-top:12px;text-align:right;}}
</style>
</head><body>
<div class="sheet">
  <h1>{_esc(tune.title)}</h1>
  <p class="meta">{_esc(tune.composer or "")}{"&ensp;·&ensp;" if tune.composer else ""}Key: {_esc(mma.key or "")} &ensp;·&ensp; {_esc(mma.style or "")} &ensp;·&ensp; {bpm} BPM &ensp;·&ensp; {ts[0]}/{ts[1]} &ensp;·&ensp; {total_bars} bars</p>
  <div class="ir-grid">
{grid_html}  </div>
  <p class="ir-source">iReal Pro chart · chart starts at {chart_offset_s:.1f}s into video</p>
</div>
<script>window.P = {p_json};</script>
</body></html>"""


def _esc(s: str) -> str:
    return (s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
