"""
tab_renderer.py — parse a UG raw_content tab string into an interactive chart.

Converts [ch]..[/ch] / [tab]..[/tab] markup into a bar list, then calls
render_interactive to produce an HTML lead sheet identical in look to the
audio-inferred charts.

Two line types handled:
  - [tab]...[/tab] blocks: first line = chord positions, second line = lyrics → 1 bar.
  - Loose chord lines: N [ch]...[/ch] on one line → 1 bar (N ≤ bpb) or multiple bars.

Section headers [Intro], [Verse 1], [Chorus], etc. drive section_per_bar labels.
Duplicate section names are suffixed: Chorus, Chorus 2, Chorus 3, ...
"""

from __future__ import annotations

import re
from pathlib import Path

# ── Regexps ────────────────────────────────────────────────────────────────────

_CH_RE   = re.compile(r'\[ch\](.*?)\[/ch\]')
_TAB_RE  = re.compile(r'\[tab\](.*?)\[/tab\]', re.DOTALL)
# Full-line section marker: [Something] with nothing else on the line
_SEC_RE  = re.compile(r'^\s*\[([^\]]+)\]\s*$')
# Tags that are NOT section markers
_TAG_SKIP = re.compile(r'^(/?)(?:ch|tab)$', re.IGNORECASE)


def _has_ch(line: str) -> bool:
    return bool(_CH_RE.search(line))


def _section_label(name: str, seen: dict[str, int]) -> str:
    """First occurrence → bare name; subsequent → 'Name N'."""
    seen[name] = seen.get(name, 0) + 1
    count = seen[name]
    return name if count == 1 else f"{name} {count}"


def _chord_visual_positions(chord_line: str) -> tuple[list[tuple[int, str]], str]:
    """Return ((visual_pos, chord_name), ...) and the fully-stripped chord line.

    visual_pos is the character offset of the chord name in the stripped string
    (i.e. after removing all [ch] and [/ch] markup).
    """
    starts: list[int] = []
    names: list[str] = []
    stripped = ""
    i = 0
    text = chord_line
    while i < len(text):
        if text[i : i + 4] == "[ch]":
            end = text.find("[/ch]", i + 4)
            if end < 0:
                # Malformed — skip the open tag, treat rest as literal
                stripped += text[i]
                i += 1
                continue
            chord_name = text[i + 4 : end]
            starts.append(len(stripped))
            names.append(chord_name)
            stripped += chord_name
            i = end + 5  # past [/ch]
        elif text[i : i + 5] == "[/ch]":
            i += 5  # stray close tag
        else:
            stripped += text[i]
            i += 1
    return list(zip(starts, names)), stripped


def parse_tab(raw_content: str, bpb: int = 4) -> list[dict]:
    """Parse UG raw_content into a list of bar dicts.

    Returns:
        [{"section": "Verse 1",
          "chords": [{"beat": 0, "ireal": "F"}, {"beat": 2, "ireal": "A-"}]},
         ...]
    One dict per bar; beat values are in [0, bpb-1].
    """
    from harmonia.tab_aligner import _ug_to_ireal  # noqa: PLC0415

    bars: list[dict] = []
    current_section = ""
    seen_sections: dict[str, int] = {}

    def add_bar(chords: list[dict], section: str) -> None:
        bars.append({"section": section, "chords": chords})

    def process_loose_line(line: str, section: str) -> None:
        """Loose chord line (not inside [tab]) → one or more bars."""
        raw_chords = _CH_RE.findall(line)
        if not raw_chords:
            return
        ireal_chords = [_ug_to_ireal(c) for c in raw_chords]
        n = len(ireal_chords)
        if n <= bpb:
            # One bar; space the N chords evenly across bpb beats
            entries = [{"beat": (i * bpb) // n, "ireal": ir}
                       for i, ir in enumerate(ireal_chords)]
            add_bar(entries, section)
        else:
            # More than one bar's worth; fill bpb chords per bar
            import math as _math
            n_bars = _math.ceil(n / bpb)
            for bi in range(n_bars):
                chunk = ireal_chords[bi * bpb : (bi + 1) * bpb]
                entries = [{"beat": j, "ireal": ir} for j, ir in enumerate(chunk)]
                add_bar(entries, section)

    def process_tab_block(inner: str, section: str) -> None:
        """Inner content of [tab]...[/tab] → one bar per chord-line found."""
        lines = inner.split("\n")
        i = 0
        while i < len(lines):
            chord_line = lines[i]
            if not _has_ch(chord_line):
                i += 1
                continue

            # This line carries chord markers.  The next line (if it has no chord
            # markers) is the lyric line; we use its length as the reference span.
            lyric_line = ""
            if i + 1 < len(lines) and not _has_ch(lines[i + 1]):
                lyric_line = lines[i + 1]
                i += 2
            else:
                i += 1

            positions, stripped = _chord_visual_positions(chord_line)
            if not positions:
                continue

            ref_len = max(len(lyric_line), len(stripped), 1)
            entries: list[dict] = []
            for vis_pos, name in positions:
                beat = int(vis_pos / ref_len * bpb)
                beat = max(0, min(beat, bpb - 1))
                entries.append({"beat": beat, "ireal": _ug_to_ireal(name)})

            if not entries:
                continue

            # If all chords collapsed to beat 0 (very short reference line),
            # space them evenly instead.
            if len(entries) > 1 and all(e["beat"] == 0 for e in entries):
                n = len(entries)
                for j, e in enumerate(entries):
                    e["beat"] = (j * bpb) // n

            # Deduplicate same-beat entries — keep last (most rightward chord wins).
            seen_beats: dict[int, int] = {}  # beat → index in entries
            for idx, e in enumerate(entries):
                seen_beats[e["beat"]] = idx
            entries = [entries[i] for i in sorted(seen_beats.values())]

            add_bar(entries, section)

    # ── Split raw_content into non-tab text segments and [tab] blocks ─────────
    segments: list[tuple[str, str]] = []  # (type, content)
    pos = 0
    for m in _TAB_RE.finditer(raw_content):
        before = raw_content[pos : m.start()]
        if before:
            segments.append(("text", before))
        segments.append(("tab", m.group(1)))
        pos = m.end()
    tail = raw_content[pos:]
    if tail:
        segments.append(("text", tail))

    # ── Process segments ───────────────────────────────────────────────────────
    for seg_idx, (seg_type, seg_content) in enumerate(segments):
        # Look ahead: is the very next segment a [tab] block?
        next_is_tab = (seg_idx + 1 < len(segments) and segments[seg_idx + 1][0] == "tab")

        if seg_type == "tab":
            process_tab_block(seg_content, current_section)
        else:
            for raw_line in seg_content.split("\n"):
                line = raw_line.strip()
                if not line:
                    continue
                # Check for a full-line section marker like [Intro] or [Verse 1]
                sec_m = _SEC_RE.match(line)
                if sec_m:
                    name = sec_m.group(1).strip()
                    if not _TAG_SKIP.match(name):
                        current_section = _section_label(name, seen_sections)
                    continue
                # Loose chord line: skip if it's a section-header summary line
                # (UG tabs always put "F  G  Am  G" before the [tab] blocks to
                # show the section's chord loop — it duplicates the [tab] content).
                if _has_ch(line):
                    if next_is_tab:
                        continue  # header line — the [tab] blocks that follow are the real bars
                    process_loose_line(line, current_section)
                # Plain text lines (lyrics, annotations) → skip

    return bars


def _expand_repeats(bars: list[dict], duration_s: float, bpb: int, tempo: int) -> list[dict]:
    """Expand bars to fill duration_s by repeating sections in song-form order.

    UG tabs write each section once; the real song loops them.  Strategy:
    1. Compute target bar count from duration and BPM.
    2. Cycle through the bar list (whole passes + partial last pass) to reach target.
    3. Section labels get " ×N" suffix on repeat passes so the UI shows structure.
    """
    if not bars or duration_s <= 0:
        return bars

    bar_dur = bpb * 60.0 / max(tempo, 1)
    target_bars = max(len(bars), round(duration_s / bar_dur))

    if target_bars <= len(bars):
        return bars  # already covers the duration

    # Build a pass-labelled infinite cycle, stop at target_bars
    # Pass 1 = original labels; pass 2+ gets " ×N" suffix
    expanded: list[dict] = []
    pass_num = 1
    pos = 0
    while len(expanded) < target_bars:
        bar = bars[pos % len(bars)]
        if pos > 0 and pos % len(bars) == 0:
            pass_num += 1
        label = bar["section"]
        if pass_num > 1 and label:
            label = f"{label} ×{pass_num}"
        expanded.append({"section": label, "chords": bar["chords"]})
        pos += 1

    return expanded


def render_tab_chart(
    raw_content: str,
    title: str,
    artist: str,
    bpb: int = 4,
    bars_per_row: int = 4,
    tempo: int = 120,
    duration_s: float = 0.0,
    out_path: Path | None = None,
) -> Path:
    """Parse raw_content and render to an interactive HTML chart.

    Args:
        duration_s: Actual song duration in seconds (from YT player or audio).
                    When > 0, sections are expanded/repeated to fill the duration.
    Returns the output path.  If out_path is None, writes to
    docs/plots/tab_{slug}.html relative to the repo root.
    """
    from harmonia.output.chart_interactive import render_interactive  # noqa: PLC0415
    from harmonia.output.chart_render import Chart  # noqa: PLC0415

    bars = parse_tab(raw_content, bpb=bpb)
    if not bars:
        bars = [{"section": "", "chords": [{"beat": 0, "ireal": "N.C."}]}]

    if duration_s > 0:
        bars = _expand_repeats(bars, duration_s, bpb, tempo)

    n_bars = len(bars)
    bar_duration_s = bpb * 60.0 / tempo
    beat_duration_s = 60.0 / tempo

    # Build chord_dicts for render_interactive
    chord_dicts: list[dict] = []
    for bar_idx, bar in enumerate(bars):
        chords_in_bar = bar["chords"]
        for ci, chord in enumerate(chords_in_bar):
            beat = chord["beat"]
            start_s = bar_idx * bar_duration_s + beat * beat_duration_s
            if ci + 1 < len(chords_in_bar):
                next_beat = chords_in_bar[ci + 1]["beat"]
                end_s = bar_idx * bar_duration_s + next_beat * beat_duration_s
            else:
                end_s = (bar_idx + 1) * bar_duration_s
            ireal = chord["ireal"]
            chord_dicts.append({
                "bar":     bar_idx,
                "beat":    beat,
                "start_s": round(start_s, 3),
                "end_s":   round(end_s, 3),
                "levels": {
                    "family":  {"ireal": ireal, "conf": 0.85},
                    "seventh": {"ireal": ireal, "conf": 0.85},
                    "exact":   {"ireal": ireal, "conf": 0.85},
                },
            })

    chart = Chart(
        title=title,
        composer=artist,
        key="",
        style=f"from UltimateGuitar · {artist}",
        tempo=tempo,
        time_signature=(bpb, 4),
        n_bars=n_bars,
        section_per_bar=[bar["section"] for bar in bars],
    )

    if out_path is None:
        slug = re.sub(r"[^a-z0-9]+", "_", f"{artist}_{title}".lower()).strip("_") or "tab"
        repo = Path(__file__).resolve().parent.parent
        out_path = repo / "docs" / "plots" / f"tab_{slug[:60]}.html"

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    render_interactive(chart, chord_dicts, out_path, bars_per_row=bars_per_row)
    return out_path
