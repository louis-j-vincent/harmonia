"""Fetch and render iReal Pro chord charts.

Two entry points:
  search_community(query)       — scrape ireal.pro community for matching songs
  render_irealb_chart(url, …)   — convert an irealb:// URL → interactive HTML

The rendered HTML exposes window.P.chords with {label, t0, t1} so the existing
YouTube sync overlay (harmonia_server._OVERLAY_HTML) can highlight chords in
real time. Timestamps are derived from BPM + chart_offset_s.
"""

from __future__ import annotations

import http.cookiejar
import json
import re
import time
import urllib.parse
import urllib.request

from pyRealParser import Tune

from harmonia.data.ireal_corpus import sectionized_measures, split_chords, tune_to_mma

_UA = "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Harmonia/1.0"


# ── Community search ──────────────────────────────────────────────────────────

_MAIN_PLAYLISTS_URL = "https://www.irealpro.com/main-playlists/"
_FORUM_BASE = "https://forums.irealpro.com"


def _iter_tunes_in_blob(raw_blob: str):
    """Yield ``(segment, Tune)`` for each song in one ``irealb://`` blob.

    A blob is a whole playlist: MANY songs joined by ``===``. ``segment`` is
    the raw slice for a SINGLE song, re-encodable into a single-tune URL (so
    importing song #7 of a playlist renders song #7, not tunes[0]).
    """
    decoded = urllib.parse.unquote(raw_blob)
    m = re.match(r"irealb://(.+)", decoded)
    if not m:
        return
    for seg in re.split("===", m.group(1)):
        if not seg:
            continue
        try:
            yield seg, Tune(seg)
        except Exception:
            continue


def _result_dict(seg: str, tune: Tune, source: str) -> dict:
    ts = tune.time_signature or (4, 4)
    return {
        "title":      tune.title,
        "composer":   tune.composer or "",
        "key":        tune.key or "",
        "style":      tune.style or "",
        "time_sig":   f"{ts[0]}/{ts[1]}",
        # SINGLE-tune URL (re-encoded from just this song's segment) — NOT the
        # multi-song playlist blob; render/import always take tunes[0].
        "irealb_url": "irealb://" + urllib.parse.quote(seg, safe="="),
        "source":     source,
    }


def _search_main_playlists(query: str, max_results: int) -> list[dict]:
    """The ~2200 jazz-standard corpus on ``/main-playlists/`` (fast, one page).

    irealpro.com restructured after this module was first written — the old
    ``/music/?s=<query>`` search endpoint 404s now (confirmed 2026-07-20).
    The standards content moved to ``/main-playlists/``: 6 big playlist blobs
    that decode to ~2236 tunes total (jazz standards, indexed by COMPOSER in
    "Last First" order — NOT performer). No server-side filter param remains,
    so fetch once and filter locally by word-overlap on title+composer.
    """
    req = urllib.request.Request(_MAIN_PLAYLISTS_URL, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode("utf-8", errors="replace")
    except Exception as exc:
        raise RuntimeError(f"iReal main-playlists search failed: {exc}") from exc

    html = re.sub(r"&#0*38;|&amp;", "&", html)
    raw_blobs = re.findall(r"irealb://[^\s\"'<>]+", html)

    # Word-overlap match (2026-07-20): ALL query words appearing somewhere in
    # title+composer — forgiving of an artist name pasted after the title,
    # still precise at ~2200 candidates.
    q_words = [w for w in re.split(r"\s+", query.strip().lower()) if w]
    results: list[dict] = []
    seen: set[str] = set()
    for raw_blob in raw_blobs:
        for seg, tune in _iter_tunes_in_blob(raw_blob):
            if len(results) >= max_results:
                return results
            haystack = f"{tune.title} {tune.composer or ''}".lower()
            if tune.title in seen or (q_words and not all(w in haystack for w in q_words)):
                continue
            seen.add(tune.title)
            results.append(_result_dict(seg, tune, "standards"))
    return results


def _forum_opener() -> urllib.request.OpenerDirector:
    cj = http.cookiejar.CookieJar()
    op = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))
    op.addheaders = [("User-Agent", _UA)]
    return op


def _search_forum(query: str, max_results: int, *,
                  max_threads: int = 6, pace_s: float = 0.35) -> list[dict]:
    """Search the iReal Pro XenForo community forum (forums.irealpro.com).

    This is where the REAL breadth lives. ``/main-playlists/`` is only the
    ~2200 jazz standards; the forum's genre subforums (jazz, pop-rock-blues,
    brazilian-latin, country-folk, holiday/film/worship, …) hold thousands of
    user-submitted charts for specific songs/artists that are NOT standards —
    e.g. Nina Simone, "Feeling Good" (Newley/Bricusse), Bob Dylan. Those
    return zero from main-playlists no matter how good the local filter is,
    because the tune simply isn't in that corpus (confirmed 2026-07-21).

    Guests can search without logging in via XenForo's stored-search flow:
      1. GET the search form for the ``_xfToken`` CSRF token,
      2. POST ``keywords`` to ``/search/search`` (302 → a results page),
      3. fetch the top matching THREADS and pull their ``irealb://`` blobs.
    The forum's own relevance ranking does the hard part — a thread for a
    Nina Simone cover often has no "Nina Simone" in the tune's composer field
    (that's the songwriter), so we trust the search over a re-applied title
    filter for focused (single-song) threads and only word-filter the big
    multi-song playlist threads.

    Paced: at most ``max_threads`` thread fetches, ``pace_s`` apart — a
    polite read-only scraper, same posture as the rest of this module. This
    is a SUPPLEMENT; callers should let its failure fall back to standards
    rather than blank the whole search.
    """
    op = _forum_opener()
    try:
        form = op.open(_FORUM_BASE + "/search/?type=post", timeout=20).read().decode("utf-8", "replace")
    except Exception as exc:
        raise RuntimeError(f"iReal forum search failed (form fetch): {exc}") from exc
    m = re.search(r'name="_xfToken"\s+value="([^"]*)"', form)
    token = m.group(1) if m else ""

    data = urllib.parse.urlencode({
        "keywords": query, "order": "relevance",
        "_xfToken": token, "search_type": "post",
    }).encode()
    req = urllib.request.Request(_FORUM_BASE + "/search/search", data=data,
                                 headers={"User-Agent": _UA,
                                          "Referer": _FORUM_BASE + "/search/"})
    try:
        page = op.open(req, timeout=20).read().decode("utf-8", "replace")
    except Exception as exc:
        raise RuntimeError(f"iReal forum search failed (post): {exc}") from exc

    # thread (slug, id) in relevance order, de-duped by id
    threads: list[tuple[str, str]] = []
    seen_ids: set[str] = set()
    for slug, tid in re.findall(r'/threads/([a-z0-9\-]+)\.(\d+)', page):
        if tid in seen_ids:
            continue
        seen_ids.add(tid)
        threads.append((slug, tid))

    q_words = [w for w in re.split(r"\s+", query.strip().lower()) if w]
    sig_words = [w for w in q_words if len(w) >= 4]  # drop "in"/"the" noise
    out: list[dict] = []
    seen_titles: set[str] = set()
    for i, (slug, tid) in enumerate(threads[:max_threads]):
        if len(out) >= max_results:
            break
        if i:
            time.sleep(pace_s)
        try:
            th = op.open(f"{_FORUM_BASE}/threads/{slug}.{tid}/", timeout=20).read().decode("utf-8", "replace")
        except Exception:
            continue
        th = re.sub(r"&#0*38;|&amp;", "&", th)
        tunes = [pair for b in re.findall(r"irealb://[^\s\"'<>]+", th)
                 for pair in _iter_tunes_in_blob(b)]
        big = len(tunes) > 8  # a shared multi-song playlist, not a single song
        for seg, tune in tunes:
            if len(out) >= max_results:
                break
            title = (tune.title or "").strip()
            if not title or title.lower() in seen_titles:
                continue
            if big and sig_words:
                hay = f"{title} {tune.composer or ''}".lower()
                if not any(w in hay for w in sig_words):
                    continue
            seen_titles.add(title.lower())
            out.append(_result_dict(seg, tune, "forum"))
    return out


def search_community(query: str, max_results: int = 8) -> list[dict]:
    """Search the iReal Pro community for songs matching ``query``.

    Two sources, standards first then forum as a supplement:
      * ``_search_main_playlists`` — the ~2200 jazz standards on
        ``/main-playlists/`` (one fast page fetch).
      * ``_search_forum`` — forums.irealpro.com, thousands of user-submitted
        charts for non-standard songs/artists across every genre subforum.

    Merged and de-duped by title. Returns dicts with the shape the UI expects
    plus a ``source`` key ("standards" | "forum") for transparency:
    {title, composer, key, style, time_sig, irealb_url, source}.

    Why two sources (2026-07-21): main-playlists is COMPLETE only for jazz
    standards — a search for "Nina Simone" or "feeling good" returns 0 there
    no matter the filter, because the songs aren't in that corpus. Almost all
    of iReal Pro's actual community breadth lives on the forum, reachable by
    guest search. This does NOT cover 100% of iReal charts in existence
    (there is no single public index of all of them, and forum coverage is
    whatever users have posted), but it takes common non-standard queries
    from "0 results" to real, importable charts. See docs/known_issues.md.
    """
    try:
        results = _search_main_playlists(query, max_results)
    except Exception:
        results = []  # forum below may still save the search

    if len(results) < max_results:
        seen = {r["title"].lower() for r in results}
        try:
            for r in _search_forum(query, max_results - len(results)):
                if r["title"].lower() in seen:
                    continue
                seen.add(r["title"].lower())
                results.append(r)
                if len(results) >= max_results:
                    break
        except Exception:
            # Forum is a supplement — never let its failure blank out the
            # standards results we already have.
            pass

    return results


# ── iReal token -> Harte sev_h (root:quality), the format chord_pipeline_v1
# emits and chart_interactive.py/chart_model.py already know how to render.
# Inverse of scripts/render_youtube_chart.py's _QUALITY_TO_IREAL (kept as its
# own small table here rather than importing scripts/ from harmonia/ — the
# wrong dependency direction — since it's the same handful of entries).
_NOTE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_NOTE_SHARP = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_IREAL_TOKEN_TO_SEV = {
    "": "maj", "-": "min", "o": "dim", "+": "aug",
    "sus2": "sus2", "sus": "sus4",
    "^7": "maj7", "-7": "min7", "7": "7", "-^7": "minmaj7",
    "h7": "hdim7", "o7": "dim7", "+^7": "augmaj7", "+7": "aug7", "7sus": "7sus4",
    "9": "9", "-9": "min9", "^9": "maj9", "7b9": "7b9", "7#9": "7#9",
    "9sus": "9sus4", "^9#11": "maj9#11",
    "-11": "min11", "11": "dom11", "^11": "maj11",
    "13": "dom13", "-13": "min13", "^13": "maj13",
    # Sixth chords (user report 2026-07-21: "Bb6 devient B" — a bare "6" fell
    # all the way through _approx_ireal_quality's fallthrough to a plain
    # major triad, silently dropping the 6th entirely). "69"/"6/9" (six-nine)
    # has no dedicated slot in this vocabulary either; approximated as a
    # plain 6 rather than dropped to a triad — see _approx_ireal_quality.
    "6": "6", "-6": "min6", "69": "6", "6/9": "6",
}


def _approx_ireal_quality(rest: str) -> str:
    """Best-effort family approximation for an iReal quality suffix that isn't
    an exact key in _IREAL_TOKEN_TO_SEV.

    Real iReal charts (user-submitted, arbitrary) use a MUCH richer vocabulary
    than our own model ever needs to express (6, 69, alt, b13, #11, sus4b9,
    add9, …) — chord_pipeline_v1's sev_h space was never built to hold all of
    it, so an EXACT round-trip isn't possible for those. Confirmed bug
    2026-07-20: without this, "D7b13" and "G-6" both silently fell through to
    the dict's blanket default of "maj" — root shown right, quality WRONG
    (a plain "D" / "G" where the source said 7b13 / minor-6). Keeping the
    right FAMILY (dominant vs minor vs diminished vs augmented) is a real,
    visible improvement over a wrong triad, even when the exact extension
    can't be spelled in our vocabulary.
    """
    if "o" in rest or "dim" in rest:
        return "dim7" if "7" in rest else "dim"
    if rest.startswith("-") or rest.startswith("h"):
        if "7" in rest:
            return "hdim7" if rest.startswith("h") else "min7"
        return "min"
    if "+" in rest or "aug" in rest:
        return "aug7" if "7" in rest else "aug"
    if "sus" in rest:
        return "7sus4" if "7" in rest else "sus4"
    if "13" in rest:
        return "dom13"
    if "11" in rest:
        return "dom11"
    if "9" in rest:
        return "9"
    if "^" in rest or "maj" in rest:
        return "maj7" if "7" in rest else "maj"
    if "7" in rest:
        return "7"
    if "6" in rest:
        return "min6" if rest.startswith("-") else "6"
    return "maj"  # no matching family at all — triad is the honest floor


_BASS_RE = re.compile(r"^([A-G])([#b]*)$")


def _parse_ireal_chord_token(token: str) -> tuple[int, str, int | None] | None:
    """'C-7' -> (0, 'min7', None); 'Bb/D' -> (10, 'maj', 2) (slash-bass);
    'F^7' -> (5, 'maj7', None); None for N.C./blank tokens.

    A trailing "/<note>" is a slash-bass (root pc, kept separately) — but
    "6/9" (a six-nine CHORD, not a bass) also contains a "/", and "9" isn't a
    valid note letter, so splitting only when the part after "/" matches
    A-G[#b]* distinguishes the two without a special case for "6/9" itself
    (confirmed bug 2026-07-21: slash-bass chords silently lost their bass
    note downstream — this is where it must be captured, since the label
    format "root:quality/bass" this feeds is understood everywhere below).
    """
    token = re.sub(r"[npWNQUSr]+$", "", token or "").strip()
    m = re.match(r"^([A-G])([#b]*)(.*)$", token)
    if not m:
        return None
    letter, acc, rest = m.groups()
    pc = (_NOTE_PC[letter] + acc.count("#") - acc.count("b")) % 12
    bass_pc = None
    if "/" in rest:
        head, _, tail = rest.rpartition("/")
        bm = _BASS_RE.match(tail)
        if bm:
            b_letter, b_acc = bm.groups()
            bass_pc = (_NOTE_PC[b_letter] + b_acc.count("#") - b_acc.count("b")) % 12
            rest = head
    sev = _IREAL_TOKEN_TO_SEV.get(rest)
    if sev is None:
        sev = _approx_ireal_quality(rest)
    return pc, sev, bass_pc


def irealb_tune_to_chord_chart(irealb_url: str, *, tempo_override: int | None = None):
    """Parse an irealb:// URL (one tune) into a ``ChordChart`` — the SAME
    shape ``infer_chords_v1`` produces — so an imported chart goes through
    the identical ``chart_to_interactive_inputs`` / ``render_interactive``
    pipeline as a real analysis and is fully SPA-compatible (opens, plays
    [no audio], edits, sorts, deletes, exports — everything the library
    already does for an analysed chart).

    2026-07-20: the previous ``render_irealb_chart`` (below) emits a
    different, older payload shape (``window.P.chords`` with no root pc / iReal
    -token levels) built for a now-dead standalone YouTube-sync overlay page.
    Since ``/chart/<file>`` unconditionally redirects into the SPA
    (``/?open=``), which expects the ChordModel shape, every iReal import
    silently failed to open (``_chart_model_for`` choking on the wrong
    shape) — this is the fix, not a patch on the old renderer.

    Sections come straight from iReal's own ``*A``/``*B`` bar markers
    (``MMAChart.timeline``'s per-bar section field) — that's exactly the
    structure information the format already carries, grouped into
    contiguous runs; no re-detection needed, unlike a freshly analysed
    audio chart where that structure has to be inferred.
    """
    from harmonia.pipeline import ChordChart

    decoded = urllib.parse.unquote(irealb_url)
    tunes = Tune.parse_ireal_url(decoded)
    if not tunes:
        raise ValueError("No tunes found in irealb URL")
    tune = tunes[0]

    mma = tune_to_mma(tune, tempo=tempo_override)
    bpm = mma.tempo
    spb = 60.0 / bpm
    bpb = mma.beats_per_bar

    # NOTE on precision: chart_to_interactive_inputs (the shared renderer)
    # recovers (bar, beat) from start_s via TRUNCATING division
    # (`int(t0 / beat_dur)`), not round(). Every boundary here sits at an
    # EXACT multiple of the beat duration (a clean synthetic grid, unlike
    # real audio's organic timing) — rounding start_s/end_s to 3 decimals,
    # as chord_pipeline_v1's own output does, would occasionally round a
    # boundary DOWN by a fraction of a millisecond, and truncation then
    # drops it a whole beat (confirmed bug 2026-07-20: "Ah7" landed at bar 3
    # beat 3 instead of bar 4 beat 0, cascading every chord after it one beat
    # early). Fix: keep full float precision here (no measurement noise to
    # round away) plus a tiny forward epsilon as a safety margin against
    # float representation error in either direction.
    _EPS = 1e-6
    chords: list[dict] = []
    sections: list[dict] = []
    total_beats = 0
    cur_label, cur_start_beat = None, 0
    for bar_no, section, slots in mma.timeline:
        if section != cur_label:
            if cur_label is not None:
                sections.append({
                    "start_s": cur_start_beat * spb + _EPS,
                    "end_s": total_beats * spb + _EPS,
                    "n_bars": round((total_beats - cur_start_beat) / bpb),
                    "label": cur_label,
                })
            cur_label, cur_start_beat = section, total_beats
        for k, (beat_offset, ireal_token, _mma_chord) in enumerate(slots):
            next_beat = slots[k + 1][0] if k + 1 < len(slots) else bpb
            dur = max(next_beat - beat_offset, 1)
            abs_beat = total_beats + beat_offset
            t0, t1 = abs_beat * spb + _EPS, (abs_beat + dur) * spb + _EPS
            parsed = _parse_ireal_chord_token(ireal_token)
            if parsed is None:
                label = "N"
            else:
                pc, sev, bass_pc = parsed
                label = f"{_NOTE_SHARP[pc]}:{sev}"
                # "root:quality/bass" — the raw label format app_shell.html's
                # prettyChordLabel already documents/parses (chart_interactive
                # .py's parse_token round-trips it via the iReal token too);
                # encoding it here is what was missing (2026-07-21 report:
                # "les slash chords se perdent"). label_to_ireal (render_
                # youtube_chart.py) must split it back off before doing its
                # quality lookup — fixed alongside this.
                if bass_pc is not None:
                    label += f"/{_NOTE_SHARP[bass_pc]}"
            chords.append({
                "label": label, "start_s": t0, "end_s": t1,
                "duration_beats": dur, "confidence": 1.0,
            })
        total_beats += bpb
    if cur_label is not None:
        sections.append({
            "start_s": cur_start_beat * spb + _EPS,
            "end_s": total_beats * spb + _EPS,
            "n_bars": round((total_beats - cur_start_beat) / bpb),
            "label": cur_label,
        })

    # coalesce adjacent identical labels (iReal repeats a chord across bars
    # explicitly — e.g. a "%"/x — so without this every held bar is its own
    # chord entry, unlike a real analysis's already-coalesced output)
    coalesced: list[dict] = []
    for c in chords:
        if coalesced and coalesced[-1]["label"] == c["label"]:
            coalesced[-1]["end_s"] = c["end_s"]
            coalesced[-1]["duration_beats"] += c["duration_beats"]
        else:
            coalesced.append(dict(c))

    duration_s = total_beats * spb
    key_raw = tune.key or "C"
    is_minor = key_raw.endswith("-")
    key_tonic = key_raw[:-1] if is_minor else key_raw
    return ChordChart(
        source_path=f"irealb:{tune.title}", duration_s=duration_s,
        tempo_bpm=round(bpm, 1), time_signature=f"{bpb}/4",
        global_key=f"{key_tonic or 'C'} {'minor' if is_minor else 'major'}",
        global_key_confidence=1.0, style="ireal-import", modulations=[],
        chords=coalesced, segments=[], sections=sections,
    )


# ── Chart renderer (LEGACY — dead code path, see irealb_tune_to_chord_chart
# above for the fix; kept only because scripts/harmonia_server.py's old
# YouTube-sync overlay demo route still calls it) ─────────────────────────────

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
