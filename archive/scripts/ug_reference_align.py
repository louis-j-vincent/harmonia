"""A REFERENCE chart written from an Ultimate Guitar tab, on OUR bar grid.

« teste sur d'autres musiques qui ont des bonnes notes guitartabs alors stp,
  et aussi pour verifier sors moi le chart inféré par guitar tabs » (Louis,
  2026-08-05)

What this produces, per song:

* ``harmonia_min/state/charts/min_<stem>__ug.json`` — a ChartModel whose BAR
  GRID, beat times and key come from a real ``pipeline.analyze()`` run, but
  whose CHORDS and SECTIONS come from the tab. Opens at ``/?open=min_<stem>__ug``
  beside our own chart. Title carries "— TAB" so it cannot be mistaken.
* ``scratchpad/ugref/<stem>.json`` — everything the report page needs
  (both section strips, both per-bar chord strings, the alignment diagnostics).

Method, and where it can be wrong
---------------------------------
The tab is untimed: a chord sits over a lyric, not over a second. Timing comes
from ``scratchpad/ug_align.py`` (written earlier the same day, reused as-is):

1. the tab is expanded into duration-weighted slots and warped against our own
   chart with a chord-TONE cosine cost (DTW, free endpoints in a leeway band so
   an un-transcribed intro is skipped rather than stretched);
2. **lyric anchors** — demucs isolates the vocals, whisper transcribes them with
   WORD timestamps, the transcript is sequence-aligned to the tab's lyric lines,
   and confident monotonic matches become hard (slot, frame) anchors the DTW
   must pass through. This is the part that does not depend on our own harmony.

Two honesty gates are carried through to the page, never hidden:

* **contrast test** — the aligned DTW cost against the same DTW on circularly
  shifted copies of the tab. A one-chord vamp scores the same however you shift
  it; that reads ``harmony-underdetermined`` and means phase 1 alone proves
  nothing (the lyric anchors may still carry it).
* **root agreement** — the fraction of aligned tab chords whose root matches our
  chart at the same instant. It is NOT a score of the tab: it is a joint measure
  of the alignment and of our chords, and it is reported as such.

A song is DROPPED, not fudged, when the tab has too few lyric lines to anchor
(instrumental-heavy) or when both gates fail.

    .venv/bin/python scripts/ug_reference_align.py [--songs a,b] [--no-phase2]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scratchpad"))

import ug_align as UA                                            # noqa: E402
from harmonia.theory.local_key import parse_token                # noqa: E402

CHARTS = HERE / "harmonia_min/state/charts"
OUTDIR = HERE / "scratchpad/ugref"
PC_FLAT = UA.PC_FLAT

# ── the songs, with the tab chosen for each ─────────────────────────────────
# Selection rule (project trust order: iReal Pro > UG >= 4.7* > other tabs >
# model output): among the CHORD tabs UG returns for the title+artist, the
# highest-rated one with >= 100 votes. Rating/votes below are what UG served on
# 2026-08-05 and are re-read from the live page at run time, never trusted from
# here.
SONGS: dict[str, dict] = {
    "maroon_5_this_love": dict(
        title="This Love",
        url="https://tabs.ultimate-guitar.com/tab/maroon-5/this-love-chords-786697"),
    "norah_jones_don_t_know_why": dict(
        title="Don't Know Why",
        url="https://tabs.ultimate-guitar.com/tab/norah-jones/dont-know-why-chords-839620"),
    "mayer_hawthorne_the_walk": dict(
        title="The Walk",
        url="https://tabs.ultimate-guitar.com/tab/mayer-hawthorne/the-walk-chords-1132291"),
    "ben_e_king_stand_by_me_audio": dict(
        title="Stand By Me",
        url="https://tabs.ultimate-guitar.com/tab/ben-e-king/stand-by-me-chords-1041639"),
    "the_police_every_breath_you_take_official_music_video": dict(
        title="Every Breath You Take",
        url="https://tabs.ultimate-guitar.com/tab/the-police/every-breath-you-take-chords-1864436"),
    "let_it_be_remastered_2009": dict(
        title="Let It Be",
        url="https://tabs.ultimate-guitar.com/tab/the-beatles/let-it-be-chords-1089764"),
    "the_ronettes_be_my_baby_music_video": dict(
        title="Be My Baby",
        url="https://tabs.ultimate-guitar.com/tab/the-ronettes/be-my-baby-chords-1475374"),
    "maroon_5_she_will_be_loved_official_music_video": dict(
        title="She Will Be Loved",
        url="https://tabs.ultimate-guitar.com/tab/maroon-5/she-will-be-loved-chords-995857"),
    "elton_john_goodbye_yellow_brick_road_lyrics": dict(
        title="Goodbye Yellow Brick Road",
        url="https://tabs.ultimate-guitar.com/tab/elton-john/goodbye-yellow-brick-road-chords-1498172"),
    "bruno_mars_the_lazy_song_official_music_video": dict(
        title="The Lazy Song",
        url="https://tabs.ultimate-guitar.com/tab/bruno-mars/the-lazy-song-chords-1761018"),
    "katy_perry_hot_n_cold_official_music_video": dict(
        title="Hot N Cold",
        url="https://tabs.ultimate-guitar.com/tab/katy-perry/hot-n-cold-chords-733932"),
    "bruno_mars_grenade_official_music_video": dict(
        title="Grenade",
        url="https://tabs.ultimate-guitar.com/tab/bruno-mars/grenade-chords-996462"),
    "aretha_franklin_chain_of_fools_official_lyric_video": dict(
        title="Chain Of Fools",
        url="https://tabs.ultimate-guitar.com/tab/aretha-franklin/chain-of-fools-chords-1212253"),
}

MIN_RATING, MIN_VOTES = 4.7, 100

# ── when an alignment is not good enough to publish ─────────────────────────
# All three tests use ONLY the tab and the vocal transcript — never our chords,
# never our sections — so dropping on them cannot bias the comparison in our
# favour.
#   * fewer than MIN_ANCHORS lyric anchors, or anchors spanning less than
#     MIN_SPAN of the song: most of the tab is then placed by harmonic cost
#     alone, and the contrast test says that cost is frequently
#     underdetermined;
#   * leave-one-out anchor error above MAX_LOO_MEDIAN seconds: at ~120 BPM a
#     bar is 2 s, so 4 s is two bars — past that the per-bar section strip
#     compares the wrong bars and a "reference" chart would be fiction.
MIN_ANCHORS, MIN_SPAN, MAX_LOO_MEDIAN = 6, 0.45, 4.0


# ── our side: one real pipeline.analyze() run, spied on for the grid ────────

def capture_ours(stem: str, title: str) -> dict:
    """Run the real pipeline once; return {grid, bars, segs, model}.

    ``segs`` is the UNFOLDED section list the shipped detector returned — the
    thing the tab is compared against. ``bars`` is captured by reference so it
    holds the FINAL (re-folded) chords, i.e. what the app renders.
    """
    from harmonia_min import sections as hs
    from harmonia_min import pipeline as _pl
    audio = HERE / f"docs/audio/{stem}.m4a"
    real = hs.detect_sections
    cap: dict = {}

    def spy(grid, arr, times, bars=None, **kw):
        out = real(grid, arr, times, bars, **kw)
        cap.update(grid=list(grid), bars=bars, segs=copy.deepcopy(out))
        return out

    hs.detect_sections = spy
    try:
        model = _pl.analyze(audio, title=title, file_key=f"min_{stem}",
                            audio_url=f"/audio/{stem}.m4a")
    finally:
        hs.detect_sections = real
    cap["model"] = model
    return cap


def ours_chords(bars: list[list[dict]]) -> list[dict]:
    """Our chord timeline in ug_align's `our_tokens` shape."""
    out = []
    for bar in bars:
        for c in bar:
            out.append({"tok": "NC" if c["nc"] else PC_FLAT[c["root"] % 12] + c["q"],
                        "t0": float(c["t0"]), "t1": float(c["t1"]),
                        "nc": bool(c["nc"]), "conf": float(c.get("c", 0.0))})
    out.sort(key=lambda c: c["t0"])
    return out


# ── tab side ───────────────────────────────────────────────────────────────

_NUM = re.compile(r"[\s\-_]*\b(\d+|i{1,3}v?|iv|vi{0,3})\b\s*$", re.I)


def norm_section(name: str) -> str:
    """"Verse 2" and "Verse" are the same SECTION TYPE; "Pre-chorus" and
    "Pre-Chorus" too. Tab writers number occurrences, which would otherwise
    make Every Breath You Take look like it has nine different sections."""
    s = (name or "?").strip()
    s = _NUM.sub("", s)
    s = re.sub(r"\s+", " ", s).strip().lower()
    s = s.replace("prechorus", "pre-chorus").replace("pre chorus", "pre-chorus")
    s = s.replace("postchorus", "post-chorus").replace("post chorus", "post-chorus")
    return s.title() if s else "?"


def tab_blocks(events) -> list[dict]:
    """Consecutive aligned chords sharing a section header → one block with a
    real [t0, t1) in our seconds."""
    blocks: list[dict] = []
    for e in events:
        if e.t0 is None:
            continue
        name = norm_section(e.section)
        raw = (e.section or "?").strip()
        if blocks and blocks[-1]["raw"] == raw:
            blocks[-1]["t1"] = max(blocks[-1]["t1"], float(e.t1))
            blocks[-1]["ev"].append(e)
        else:
            blocks.append({"name": name, "raw": raw, "t0": float(e.t0),
                           "t1": float(e.t1), "ev": [e]})
    return blocks


def bar_labels(blocks, grid) -> list[str | None]:
    """One tab-section name per bar — whichever block covers the bar's CENTRE.
    ``None`` = the tab does not cover this bar (un-transcribed intro/outro).
    Never guessed."""
    n = len(grid) - 1
    lab: list[str | None] = [None] * n
    for b in range(n):
        c = 0.5 * (grid[b] + grid[b + 1])
        for blk in blocks:
            if blk["t0"] <= c < blk["t1"]:
                lab[b] = blk["name"]
                break
    return lab


def occurrences(labels) -> list[tuple[str, int, int]]:
    """Run-length-encode per-bar labels into (name, b0, b1_exclusive)."""
    occ, i, n = [], 0, len(labels)
    while i < n:
        if labels[i] is None:
            i += 1
            continue
        j = i
        while j < n and labels[j] == labels[i]:
            j += 1
        occ.append((labels[i], i, j))
        i = j
    return occ


# ── the alignment itself (ug_align machinery, our chart as the time side) ───

def loo_anchor_error(C, anchors, times) -> dict:
    """LEAVE-ONE-OUT anchor error, in seconds — the one alignment check that
    uses NEITHER our chords NOR our sections.

    Definition: hide one lyric anchor, re-run the anchored DTW on the others,
    read where the path puts that anchor's tab position, and compare with the
    time whisper actually heard those words. It answers "if the alignment had
    not been told about this line, how many seconds off would it have placed
    it" — the honest size of the timing error between two anchors. Reported as
    the median and the 90th percentile over all anchors.
    """
    errs: list[float] = []
    for k in range(len(anchors)):
        rest = [(a[0], a[1]) for i, a in enumerate(anchors) if i != k]
        if len(rest) < 2:
            continue
        path, _ = UA.anchored_dtw(C, rest)
        slot, frame = anchors[k][0], anchors[k][1]
        hits = [f for s, f in path if s == slot]
        if not hits:
            continue
        pred = float(times[int(np.median(hits))])
        errs.append(abs(pred - float(times[frame])))
    if not errs:
        return {"n": 0, "median": None, "p90": None, "errors": []}
    return {"n": len(errs), "median": float(np.median(errs)),
            "p90": float(np.quantile(errs, 0.90)),
            "errors": [round(e, 2) for e in errs]}


def align_tab(stem: str, url: str, ours: list[dict], phase2: bool = True) -> dict:
    tab = UA.fetch_ug(url)
    events, lyrics, notes = UA.parse_tab(tab.content)
    sh = UA.detect_shift(events, ours, tab)
    for e in events:
        e.tok_snd = UA.transpose_ireal(e.tok, sh["shift"])
    runs = UA.build_runs(events, key="tok_snd")

    times, F, _ = UA.our_frames(ours)
    V, owner = UA.expand_slots(runs, times[-1] - times[0])
    C = UA.cost_matrix(V, F, owner, runs)
    path, cost = UA.dtw(C)
    ct = UA.contrast_test(C, cost)
    mode, anchors_used = "phase1", []

    loo = {"n": 0, "median": None, "p90": None, "errors": []}
    if phase2:
        segs = UA.transcribe(stem, "small")
        if segs:
            anchors = UA.lyric_anchors(events, lyrics, segs, runs, owner, times)
            if len(anchors) >= 2:
                path, cost = UA.anchored_dtw(C, [(a[0], a[1]) for a in anchors])
                mode = "phase2"
                anchors_used = [{"slot": int(a[0]), "t": float(times[a[1]]),
                                 "sim": float(a[2]), "line": a[3]} for a in anchors]
                loo = loo_anchor_error(C, anchors, times)
    UA.path_to_times(path, owner, times, runs, events)

    # root agreement — a JOINT measure of the alignment and of our chords
    agree = tot = 0
    for e in events:
        e.mismatch = False
        if e.t0 is None or e.nomusic:
            continue
        mid = 0.5 * (e.t0 + e.t1)
        c = next((c for c in ours if c["t0"] <= mid < c["t1"]), None)
        if c is None or c["nc"]:
            continue
        tot += 1
        if parse_token(c["tok"])[0] == parse_token(e.tok_snd)[0]:
            agree += 1
        else:
            e.mismatch = True

    span = 0.0
    if anchors_used:
        span = (anchors_used[-1]["t"] - anchors_used[0]["t"]) / \
               max(times[-1] - times[0], 1e-6)
    return {"tab": tab, "events": events, "lyrics": lyrics, "notes": notes,
            "shift": sh, "mode": mode, "anchors": anchors_used,
            "cost": cost, "contrast": ct, "n_anchors": len(anchors_used),
            "anchor_span": span, "loo": loo,
            "agree": agree, "n_compared": tot, "agree_rate": agree / max(tot, 1)}


# ── tab → ChartModel ───────────────────────────────────────────────────────

UNCOVERED = "hors tab"


def build_ug_chart(stem: str, title: str, base: dict, grid: list[float],
                   events, blocks, info: dict) -> dict:
    """Our grid + our beats + THE TAB'S chords and sections."""
    bpb = int(base["bpb"])
    n = len(grid) - 1
    bars: list[list[dict]] = [[] for _ in range(n)]

    # Snap every tab chord onset to the nearest HALF-BAR line of our grid.
    # Justified by measurement, not by taste: over the four songs aligned first,
    # the onsets' position inside a bar is sharply bimodal at 0 and 0.5 (This
    # Love 112 chords: 67 in the first quarter-bar, 29 around the half; Every
    # Breath You Take 50: 41 in the first quarter), with a jitter of ~0.15 bar
    # — that is a tab writing on bar lines and half-bars, plus DTW noise. The
    # snap moves a chord by at most a quarter-bar (~0.6 s), well under the
    # measured leave-one-out alignment error, and it is what turns the chart
    # from "every bar carries the previous chord" into a readable one.
    half = []
    for b in range(n):
        half.append(grid[b])
        half.append(0.5 * (grid[b] + grid[b + 1]))
    half.append(grid[n])
    half_arr = np.asarray(half)

    placed = []
    for e in events:
        if e.t0 is None or e.nomusic or e.tok_snd in ("", "NC"):
            continue
        idx = int(np.abs(half_arr - e.t0).argmin())
        b = min(idx // 2, n - 1)
        k = 0 if idx % 2 == 0 else bpb // 2
        root, q, bass = parse_token(e.tok_snd)
        t0 = float(half_arr[idx])
        placed.append({"bar": b, "beat": int(k), "root": int(root), "q": q,
                       "bass": -1 if bass is None else int(bass),
                       "t0": round(t0, 3),
                       "t1": round(max(float(e.t1), t0 + 0.05), 3)})
    for a, z in zip(placed, placed[1:]):        # a chord runs to the next one
        a["t1"] = max(z["t0"], a["t0"] + 0.05)

    # one chord per (bar, beat); consecutive duplicates of the same chord merge
    for p in placed:
        cur = bars[p["bar"]]
        if cur and cur[-1]["beat"] == p["beat"]:
            cur[-1]["t1"] = max(cur[-1]["t1"], p["t1"])
            continue
        prev = None
        for b in range(p["bar"], -1, -1):
            if bars[b]:
                prev = bars[b][-1]
                break
        if prev is not None and prev["root"] == p["root"] and prev["q"] == p["q"]:
            prev["t1"] = max(prev["t1"], p["t1"])
            continue
        cur.append({**p, "nc": False, "c": 1.0, "n": 0})
    for b in range(n):                     # a bar cannot hold more than bpb
        bars[b] = bars[b][:bpb]

    # Louis's bar rule: a bar with no onset writes its sounding chord, carried
    prev = None
    for b in range(n):
        first = bars[b][0] if bars[b] else None
        if first is None and prev is not None:
            bars[b].append({**prev, "carry": True, "bar": b, "beat": 0,
                            "t0": round(grid[b], 3), "t1": round(grid[b + 1], 3)})
        elif first is not None and first["beat"] > 0 and prev is not None:
            bars[b].insert(0, {**prev, "carry": True, "bar": b, "beat": 0,
                               "t0": round(grid[b], 3), "t1": first["t0"]})
        if bars[b]:
            prev = bars[b][-1]
    for b in range(n):                     # leading bars the tab never reaches
        if not bars[b]:
            bars[b].append({"root": 0, "q": "", "bass": -1, "nc": True, "c": 1.0,
                            "n": 0, "bar": b, "beat": 0,
                            "t0": round(grid[b], 3), "t1": round(grid[b + 1], 3)})

    from collections import Counter
    fam = Counter()
    for bar in bars:
        for c in bar:
            if not c["nc"] and not c.get("carry"):
                fam[(c["root"], c["q"][:1])] += 1
    for bar in bars:
        for c in bar:
            c["n"] = 0 if c["nc"] else fam[(c["root"], c["q"][:1])]

    # sections: the tab's blocks, snapped to bars; anything the tab never
    # covers becomes its own "hors tab" section rather than being absorbed.
    labels = bar_labels(blocks, grid)
    occ = occurrences(labels)
    spans: list[tuple[str, int, int]] = []
    b = 0
    for name, s, e_ in occ:
        if s > b:
            spans.append((UNCOVERED, b, s - 1))
        spans.append((name, s, e_ - 1))
        b = e_
    if b < n:
        spans.append((UNCOVERED, b, n - 1))
    if not spans:
        spans = [(UNCOVERED, 0, n - 1)]

    sections = []
    for si, (name, b0, b1) in enumerate(spans):
        sections.append({
            "id": f"S{si}", "label": name, "tag": "", "reps": 1,
            "spans": [[grid[b0], grid[b1 + 1]]],
            "barRanges": [[b0, b1]],
            "bars": bars[b0:b1 + 1],
            "barSpans": [[[grid[bb], grid[bb + 1]]] for bb in range(b0, b1 + 1)],
        })

    # colours / key from the TAB's chords, through the shipped analyser
    flat = [c for bar in bars for c in bar]
    key, key_name, key_segments = base["key"], base["keyName"], base.get("keySegments")
    try:
        from harmonia_min.harmonic_key import analyze_harmony
        from harmonia_min.nnls_features import extract_bothchroma
        arr, times = extract_bothchroma(HERE / f"docs/audio/{stem}.m4a")
        H = analyze_harmony(arr, times, flat)
        for i, c in enumerate(flat):
            c["colour"] = H["colours"][i]
        key_segments = H["segments"]
        main = max(H["segments"], key=lambda s: s["t1"] - s["t0"])
        key = {"tonic": main["tonic"], "mode": main["mode"]}
        maj = main["tonic"] if main["mode"] == "major" else (main["tonic"] + 3) % 12
        names = ("C C# D Eb E F F# G G# A Bb B" if maj in (7, 2, 9, 4, 11)
                 else "C Db D Eb E F Gb G Ab A Bb B").split()
        key_name = f"{names[main['tonic']]} {main['mode']}"
    except Exception as exc:                        # reported, never silent
        print(f"    [warn] harmonic_key on tab chords failed: "
              f"{type(exc).__name__}: {exc} — keeping our own key/colours")

    return {
        "file": f"min_{stem}__ug", "title": f"{title} — TAB", "video_id": "",
        "audio_url": f"/audio/{stem}.m4a",
        "key": key, "keyName": key_name, "keySegments": key_segments,
        "bpb": bpb, "nBars": n,
        "barGrid": grid, "beatTimes": base["beatTimes"],
        "form": None, "fold": None, "sections": sections,
        "meta": {**base.get("meta", {}), "engine": "ultimate-guitar-tab",
                 "ug_url": info["tab"].url, "ug_rating": info["tab"].rating,
                 "ug_votes": info["tab"].votes, "ug_capo": info["tab"].capo,
                 "align_mode": info["mode"], "align_anchors": info["n_anchors"],
                 "align_verdict": info["contrast"]["verdict"],
                 "root_agreement": round(info["agree_rate"], 4)},
    }


# ── per-bar chord strings, for the side-by-side comparison ─────────────────

def bar_strings(bars, n) -> list[str]:
    out = []
    for b in range(n):
        toks = []
        for c in (bars[b] if b < len(bars) else []):
            if c["nc"]:
                toks.append("N.C.")
            elif c.get("carry") and toks:
                continue
            else:
                toks.append(PC_FLAT[c["root"] % 12] + c["q"])
        # a carried bar writes its chord once
        if not toks and b < len(bars) and bars[b]:
            c = bars[b][0]
            toks = ["N.C." if c["nc"] else PC_FLAT[c["root"] % 12] + c["q"]]
        out.append(" ".join(dict.fromkeys(toks)) or "·")
    return out


# ── driver ─────────────────────────────────────────────────────────────────

def run_song(stem: str, title: str, url: str, phase2: bool) -> dict | None:
    print(f"\n=== {title}  [{stem}]")
    cap = capture_ours(stem, title)
    grid, bars, segs, model = cap["grid"], cap["bars"], cap["segs"], cap["model"]
    n = len(grid) - 1
    ours = ours_chords(bars)
    print(f"  ours: {n} bars, {len(segs)} sections "
          f"({' '.join(s['label'] for s in segs)})")

    info = align_tab(stem, url, ours, phase2=phase2)
    tab = info["tab"]
    print(f"  UG: {tab.artist} — {tab.song}  {tab.rating:.2f}*/{tab.votes}v  "
          f"capo={tab.capo}  shift +{info['shift']['shift']} "
          f"({info['shift']['source']})")
    if tab.rating < MIN_RATING or tab.votes < MIN_VOTES:
        print(f"  DROP: tab below the trust bar ({MIN_RATING}*/{MIN_VOTES} votes)")
        return None
    lm = info["loo"]
    print(f"  align: {info['mode']} ({info['n_anchors']} lyric anchors covering "
          f"{info['anchor_span']*100:.0f}% of the song), cost {info['cost']:.3f}, "
          f"{info['contrast']['verdict']} "
          f"(gain {info['contrast']['rel_gain']*100:+.0f}%), "
          f"leave-one-out anchor error median "
          f"{('%.1fs' % lm['median']) if lm['median'] is not None else '—'} "
          f"/ p90 {('%.1fs' % lm['p90']) if lm['p90'] is not None else '—'}, "
          f"root agreement {info['agree_rate']*100:.1f}% "
          f"on {info['n_compared']} chords")
    if info["mode"] == "phase1" and info["contrast"]["verdict"] != "determinate":
        print("  DROP: no lyric anchors AND the harmonic contrast test says the "
              "alignment is not determinate — nothing here is trustworthy.")
        return None
    why = []
    if info["n_anchors"] < MIN_ANCHORS:
        why.append(f"only {info['n_anchors']} lyric anchors (< {MIN_ANCHORS})")
    if info["anchor_span"] < MIN_SPAN:
        why.append(f"anchors span only {info['anchor_span']*100:.0f}% of the song "
                   f"(< {MIN_SPAN*100:.0f}%)")
    if lm["median"] is None or lm["median"] > MAX_LOO_MEDIAN:
        why.append(f"leave-one-out anchor error median "
                   f"{('%.1f s' % lm['median']) if lm['median'] else '—'} "
                   f"(> {MAX_LOO_MEDIAN:.0f} s)")
    if why:
        print("  DROP: the alignment is not good enough to publish — "
              + "; ".join(why) + ".")
        return {"__dropped__": stem, "why": why,
                "rating": tab.rating, "votes": tab.votes, "url": url,
                "title": title, "n_anchors": info["n_anchors"],
                "anchor_span": info["anchor_span"], "loo_median": lm["median"]}

    blocks = tab_blocks(info["events"])
    labels = bar_labels(blocks, grid)
    occ = occurrences(labels)
    names = list(dict.fromkeys(o[0] for o in occ))
    print(f"  tab: {len(occ)} section blocks, {len(names)} distinct "
          f"({' '.join(names)}); {sum(l is None for l in labels)} bar(s) uncovered")

    chart = build_ug_chart(stem, title, model, grid, info["events"], blocks, info)
    CHARTS.mkdir(parents=True, exist_ok=True)
    (CHARTS / f"min_{stem}__ug.json").write_text(json.dumps(chart))
    # our own chart beside it, so the two open side by side in the library —
    # but NEVER overwrite one that already exists (another session may own it,
    # and a chart can carry annotations).
    ours_path = CHARTS / f"min_{stem}.json"
    if not ours_path.exists():
        ours_path.write_text(json.dumps(model))
        print(f"  -> /?open=min_{stem}  (our chart, newly written)")
    print(f"  -> /?open=min_{stem}__ug")

    ug_flat: list[list[dict]] = [[] for _ in range(n)]
    for s in chart["sections"]:
        b0, b1 = s["barRanges"][0]
        for k, bb in enumerate(range(b0, b1 + 1)):
            ug_flat[bb] = s["bars"][k]

    rec = {
        "stem": stem, "title": title,
        "url": url, "rating": tab.rating, "votes": tab.votes,
        "artist": tab.artist, "song": tab.song, "capo": tab.capo,
        "tonality": tab.tonality, "shift": info["shift"]["shift"],
        "shift_source": info["shift"]["source"],
        "align_mode": info["mode"], "n_anchors": info["n_anchors"],
        "anchor_span": info["anchor_span"], "loo": info["loo"],
        "cost": info["cost"], "verdict": info["contrast"]["verdict"],
        "rel_gain": info["contrast"]["rel_gain"],
        "agree_rate": info["agree_rate"], "n_compared": info["n_compared"],
        "parse_notes": info["notes"],
        "grid": grid, "n_bars": n,
        "tab_labels": labels, "tab_occ": occ, "tab_names": names,
        "our_segs": [{"b0": s["b0"], "b1": s["b1"], "label": s["label"]} for s in segs],
        "our_bar_chords": bar_strings(bars, n),
        "tab_bar_chords": bar_strings(ug_flat, n),
        "anchors": info["anchors"][:60],
    }
    OUTDIR.mkdir(parents=True, exist_ok=True)
    (OUTDIR / f"{stem}.json").write_text(json.dumps(rec, indent=1))
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--songs", default="", help="comma-separated stems")
    ap.add_argument("--no-phase2", action="store_true")
    a = ap.parse_args()
    want = [s for s in a.songs.split(",") if s] or list(SONGS)
    kept, dropped = [], []
    for stem in want:
        cfg = SONGS[stem]
        try:
            rec = run_song(stem, cfg["title"], cfg["url"], not a.no_phase2)
        except Exception as exc:
            import traceback
            traceback.print_exc()
            rec = {"__dropped__": stem, "title": cfg["title"], "url": cfg["url"],
                   "why": [f"{type(exc).__name__}: {exc}"]}
        if rec and "__dropped__" not in rec:
            kept.append(stem)
            continue
        dropped.append(rec or {"__dropped__": stem, "title": cfg["title"],
                               "url": cfg["url"], "why": ["see log"]})
        for p in (OUTDIR / f"{stem}.json", CHARTS / f"min_{stem}__ug.json"):
            if p.exists():
                p.unlink()                 # never leave a chart we do not stand by
    OUTDIR.mkdir(parents=True, exist_ok=True)
    old = json.loads((OUTDIR / "_dropped.json").read_text()) \
        if (OUTDIR / "_dropped.json").exists() else []
    old = [d for d in old if d["__dropped__"] not in want] + dropped
    (OUTDIR / "_dropped.json").write_text(json.dumps(old, indent=1))
    print(f"\nkept {len(kept)} / dropped {len(dropped)}")
    for d in dropped:
        print(f"  dropped: {d['__dropped__']} — {'; '.join(d['why'])}")


if __name__ == "__main__":
    main()
