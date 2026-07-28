#!/usr/bin/env python3
"""section_merge_declined.py — 2026-07-29.

Generate SECTION-level merge SUGGESTIONS for the human-confirm "merge game"
(scratchpad/section_merge_game.html), driven by the *arbiter's declined pairs*.

The connection this closes
--------------------------
The production section arbiter (``harmonia.models.section_arbiter`` /
``chart_model._sections_by_largest_unit``) deliberately UNDER-splits: when two
spans have matching harmony but a *distinctive chord* is present in one and
absent from the other (the ``veto``), or their energy differs, it keeps them as
SEPARATE letters rather than risk hiding a real section (user error-preference,
2026-07-21: "je préfère l'erreur 2 à l'erreur 1"). That is exactly the right
default for a suggest-then-confirm loop: the pairs the arbiter *declined to
merge under ambiguity* are the high-value questions to put to a human.

A "declined pair" here = two DIFFERENTLY-LABELLED sections whose bar-root
sequences MATCH on the arbiter's own similarity (``section_arbiter.sim`` >=
MATCH) — i.e. harmony says "these could be the same" — that the arbiter
nonetheless split, either because the distinctive-chord ``veto`` fired
(``tier="veto"``, the prime case: it carries a concrete musical reason) or
because they fell just under the linkage threshold / a phase offset
(``tier="near"``). Every candidate ships with the SPECIFIC reason (which bar,
which chord) so the confirm card can explain itself in one plain sentence.

Reuses ``section_arbiter.sim`` / ``.veto`` verbatim (one source of truth for
the thresholds and the phase-tolerant matching), so a suggestion is consistent
with the split the live pipeline actually made.

Data source: the ``const P = {...}`` payload baked into every
``docs/plots/inferred_*.html`` (sectionChips + per-bar chords) — decoupled from
the pipeline, so this never needs the server or the (concurrent-WIP) chart_model.

Output: scratchpad/section_merge_game_data.json — the game's candidate deck.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.models.section_arbiter import sim as arb_sim, veto as arb_veto, MATCH

PLOTS = REPO / "docs" / "plots"
OUT = Path(__file__).resolve().parent / "section_merge_game_data.json"

FLOOR = 0.70             # a suggestion needs >=70% of bars identical (equal length)
MAX_PER_SONG = 4         # keep the deck varied across songs
MIN_SEC_BARS = 4         # ignore fragment "sections" (turnarounds, tags)
MAX_LEN_DIFF = 1         # a repeat has the SAME bar-length (±1 for a pickup/tag);
#                          the arbiter's sim is only meaningful on equal-length
#                          blocks — comparing a 8-bar to a 16-bar section matches
#                          spuriously on a shared chord in the overlap.
NEAR_IDENTICAL = 0.90    # >= this -> "looks identical"; else "one chord apart"

# noise words stripped from a slug to make a readable card title
_NOISE = {"official", "music", "video", "lyric", "audio", "remastered", "hd",
          "4k", "readable", "npattern", "nfix", "bestfit", "original", "show",
          "a", "colors", "the", "feat", "ft", "hq", "mv", "live", "acoustic",
          "ireal", "vevo", "1977", "2009"}


def _title(slug: str) -> str:
    words = [w for w in slug.split("_") if w and w.lower() not in _NOISE]
    return " ".join(words[:5]).title() if words else slug

# pitch-class -> name (flats, the display convention in the handoff)
_PC = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
# iReal quality token -> compact superscript-ish glyph for the card
_Q = {"": "", "^7": "maj7", "6": "6", "7": "7", "-": "m", "-7": "m7",
      "-^7": "m(maj7)", "o": "dim", "-7b5": "m7b5", "h7": "m7b5", "o7": "dim7",
      "9": "9", "7b9": "7b9", "13": "13", "-6": "m6", "+": "aug"}


def _load_payload(html_path: Path) -> dict | None:
    """Extract the ``const P = {...};`` JSON object from a baked chart."""
    txt = html_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"const\s+P\s*=\s*", txt)
    if not m:
        return None
    i = txt.index("{", m.end())
    depth, j, instr, esc = 0, i, False, False
    while j < len(txt):
        c = txt[j]
        if instr:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                instr = False
        else:
            if c == '"':
                instr = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
        j += 1
    try:
        return json.loads(txt[i:j + 1])
    except Exception:
        return None


def _bar_times(chords: list[dict], n_bars: int) -> list[float]:
    """Downbeat time (seconds) of every bar 0..n_bars, on the BEAT grid.

    ``sectionChips`` starts are NOT guaranteed to land on a bar line, and the
    chord list is sparse (held chords appear once), so snapping a section to the
    nearest chord *onset* lands bar 1 up to a bar off and starts playback
    mid-bar. Here we rebuild a real per-bar grid: each beat-0 chord onset anchors
    its bar's downbeat, and bars between two anchors are spaced uniformly (a held
    chord spanning k bars is split into k equal bars). Returns ``bar_time`` with
    ``bar_time[b]`` = downbeat of bar b and ``bar_time[n_bars]`` = song end, so a
    section can start and play exactly on the downbeat of its first bar.
    """
    anchors: list[tuple[int, float]] = []
    seen: set[int] = set()
    for c in sorted(chords, key=lambda c: (int(c["bar"]), int(c.get("beat", 0)))):
        b = int(c["bar"])
        if int(c.get("beat", 0)) == 0 and b not in seen and 0 <= b <= n_bars:
            anchors.append((b, float(c["t0"])))
            seen.add(b)
    end_t = float(max(chords, key=lambda c: float(c["t1"]))["t1"])
    if not anchors:
        return [end_t * b / max(n_bars, 1) for b in range(n_bars + 1)]
    anchors.append((n_bars, end_t))  # cap so the last held region is spaced too
    bar_time = [0.0] * (n_bars + 1)
    for (b0, t0), (b1, t1) in zip(anchors, anchors[1:]):
        span = max(b1 - b0, 1)
        for b in range(b0, b1):
            if 0 <= b <= n_bars:
                bar_time[b] = t0 + (t1 - t0) * (b - b0) / span
    bar_time[n_bars] = end_t
    # bars before the first anchor (rare pickup) share the first downbeat
    for b in range(anchors[0][0]):
        bar_time[b] = anchors[0][1]
    return bar_time


def _section_bars(P: dict) -> list[dict]:
    """One entry per SECTION: {label, bar0, t0, t1, bars:[{root,q,c,name}]}.

    Per-bar representative = the chord SOUNDING at that bar's downbeat, with
    held chords FORWARD-FILLED across the bars they span (the chord list is
    sparse — one entry per change — so a chord held for 3 bars appears once and
    must be carried, or the section fingerprint fills with holes). Split-bar
    second chords (beat>0) don't override the downbeat. Same grain as the live
    clustering (per-bar downbeat root sequence).

    ``bar0`` and the play ``t0``/``t1`` are snapped to the BEAT GRID
    (:func:`_bar_times`): a section starts on the downbeat of its first bar, so
    the chord shown as "bar 1" is the chord you hear when playback begins — not a
    chord onset a bar early because ``start_s`` fell mid-bar.
    """
    chips = P.get("sectionChips") or []
    chords = P.get("chords") or []
    if not chips or not chords:
        return []
    n_bars = int(P.get("nBars") or (max(c["bar"] for c in chords) + 1))
    bar_time = _bar_times(chords, n_bars)
    # forward-fill: rep[b] = last chord whose (bar,beat) <= (b, downbeat)
    events = sorted(chords, key=lambda c: (int(c["bar"]), int(c.get("beat", 0))))
    bar_chord: dict[int, dict] = {}
    cur = None
    ci = 0
    for b in range(n_bars):
        while ci < len(events) and (int(events[ci]["bar"]), int(events[ci].get("beat", 0))) <= (b, 0):
            cur = events[ci]
            ci += 1
        if cur is not None:
            bar_chord[b] = cur
    # section start bars from start_s -> nearest BAR DOWNBEAT (on the beat grid)
    starts = []
    for chip in chips:
        ts = float(chip["start_s"])
        bar0 = min(range(n_bars), key=lambda b: abs(bar_time[b] - ts))
        starts.append((chip["label"], bar0, ts))
    secs = []
    for k, (label, bar0, _ts) in enumerate(starts):
        bar1 = starts[k + 1][1] if k + 1 < len(starts) else n_bars
        t0 = bar_time[bar0]                 # play/display start = bar 1 downbeat
        bars = []
        for b in range(bar0, bar1):
            c = bar_chord.get(b)
            if c is None:
                bars.append(None)
                continue
            q = c["lv"]["exact"]["q"]
            root = int(c["root"])
            bars.append({
                "root": root,
                "q": q,
                "c": float(c["lv"]["exact"].get("c", 1.0)),
                "name": _PC[root % 12] + (_Q.get(q, q)),
            })
        t1 = bar_time[bar1]                 # end = next section's downbeat (grid)
        secs.append({"label": label, "bar0": bar0, "t0": round(t0, 2),
                     "t1": round(float(t1), 2), "bars": bars})
    return secs


def _roots(sec: dict) -> list[int | None]:
    return [b["root"] if b else None for b in sec["bars"]]


def _veto_reason(a: dict, b: dict) -> dict | None:
    """If a distinctive chord blocks the merge, name it (bar index + chord).

    Mirrors ``section_arbiter.veto``: a chord (here, root) recurring >=2 bars in
    one section but wholly ABSENT from the other. Returns the strongest such
    offender for the card, plus which bars differ, or None if nothing vetoes.
    """
    ra = [r for r in _roots(a) if r is not None]
    rb = [r for r in _roots(b) if r is not None]
    if not ra or not rb:
        return None
    from collections import Counter
    ca, cb, sa, sb = Counter(ra), Counter(rb), set(ra), set(rb)
    offenders = []
    for src, cnt, other_set, side in ((a, ca, sb, "left"), (b, cb, sa, "right")):
        for root, n in cnt.items():
            if n >= 2 and n >= 0.2 * len([x for x in _roots(src) if x is not None]) and root not in other_set:
                # find a representative bar + chord name in that section
                name = next((bb["name"] for bb in src["bars"] if bb and bb["root"] == root), _PC[root % 12])
                offenders.append((n, side, root, name))
    if not offenders:
        return None
    offenders.sort(reverse=True)
    n, side, root, name = offenders[0]
    return {"side": side, "chord": name, "n_bars": n, "root": root}


def _aligned(a: dict, b: dict) -> list[dict]:
    """Bar-by-bar alignment for the side-by-side card (min overlap length)."""
    out = []
    k = min(len(a["bars"]), len(b["bars"]))
    for t in range(k):
        ba, bb = a["bars"][t], b["bars"][t]
        same = bool(ba and bb and ba["root"] == bb["root"])
        out.append({
            "i": t,
            "left": ba["name"] if ba else "·",
            "right": bb["name"] if bb else "·",
            "same": same,
        })
    return out


def _reason_text(a: dict, b: dict, sim: float, vr: dict | None, aligned: list) -> str:
    n = len(aligned)
    diffs = [al for al in aligned if not al["same"]]
    if not diffs:
        return f"All {n} bars identical — almost certainly the same section played twice."
    if vr:
        where = a["label"] if vr["side"] == "left" else b["label"]
        return (f"{n - len(diffs)} of {n} bars match. The one thing keeping them "
                f"apart: {where} plays {vr['chord']} where the other doesn't.")
    exd = diffs[0]
    bars = "bar" if len(diffs) == 1 else "bars"
    where = ", ".join(str(d["i"] + 1) for d in diffs[:3])
    return (f"{n - len(diffs)} of {n} bars match. They differ at {bars} {where}"
            f" (e.g. {exd['left']} vs {exd['right']}).")


def generate_for(html_path: Path, limit: int = MAX_PER_SONG) -> list[dict]:
    P = _load_payload(html_path)
    if not P:
        return []
    secs = _section_bars(P)
    if len(secs) < 2:
        return []
    slug = html_path.stem.replace("inferred_", "")
    title = _title(slug)
    tonic = (P.get("home") or {}).get("tonic")
    keyName = P.get("keyName", "")
    out = []
    for i in range(len(secs)):
        for j in range(i + 1, len(secs)):
            a, b = secs[i], secs[j]
            if a["label"] == b["label"]:
                continue  # already one letter -> already merged
            ra = [x for x in _roots(a) if x is not None]
            rb = [x for x in _roots(b) if x is not None]
            if len(ra) < MIN_SEC_BARS or len(rb) < MIN_SEC_BARS:
                continue  # skip fragment sections (turnarounds, tags)
            if abs(len(ra) - len(rb)) > MAX_LEN_DIFF:
                continue  # a repeat is the SAME length — never merge unequal sections
            s = arb_sim(ra, rb)
            if s < FLOOR:
                continue  # need >=70% of bars identical to even suggest it
            vr = _veto_reason(a, b)
            aligned = _aligned(a, b)
            # a genuine "one chord apart" needs a distinctive chord AND not too
            # many other diffs; otherwise it's just a strong near-identical.
            n_diff = sum(1 for al in aligned if not al["same"])
            if s >= NEAR_IDENTICAL or not vr:
                tier = "near"          # looks identical (or no single distinctive chord)
            elif n_diff <= max(2, len(aligned) // 4):
                tier = "veto"          # same but for one distinctive chord
            else:
                tier = "near"
            out.append({
                "song": slug, "title": title, "keyName": keyName, "tonic": tonic,
                "left": {"label": a["label"], "t0": a["t0"], "t1": a["t1"],
                         "bar0": a["bar0"], "nbars": len(a["bars"]),
                         "bars": [bb["name"] if bb else "·" for bb in a["bars"]]},
                "right": {"label": b["label"], "t0": b["t0"], "t1": b["t1"],
                          "bar0": b["bar0"], "nbars": len(b["bars"]),
                          "bars": [bb["name"] if bb else "·" for bb in b["bars"]]},
                "sim": round(float(s), 3),
                "tier": tier,
                "veto": vr,
                "aligned": aligned,
                "reason": _reason_text(a, b, s, vr, aligned),
            })
    # dedup: a song with two same-letter sections (two B's) yields the identical
    # "i vs B" comparison twice — show it once (keyed on labels + bar content).
    seen: set = set()
    deduped = []
    for c in out:
        key = (c["left"]["label"], c["right"]["label"],
               tuple(c["left"]["bars"]), tuple(c["right"]["bars"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(c)
    # rank: veto (concrete reason) first, then near, then by harmony sim
    tier_rank = {"veto": 0, "near": 1, "weak": 2}
    deduped.sort(key=lambda c: (tier_rank[c["tier"]], -c["sim"]))
    return deduped[:limit]


def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    files = sorted(PLOTS.glob("inferred_*.html"))
    if only:
        files = [f for f in files if only in f.name]
    deck, per_song = [], {}
    for f in files:
        cands = generate_for(f)
        if cands:
            per_song[f.stem] = len(cands)
            deck.extend(cands)
    # global order: all veto cards first (best teaching examples), then near
    tier_rank = {"veto": 0, "near": 1, "weak": 2}
    deck.sort(key=lambda c: (tier_rank[c["tier"]], -c["sim"]))
    meta = {
        "n_candidates": len(deck),
        "n_songs": len(per_song),
        "by_tier": {t: sum(1 for c in deck if c["tier"] == t) for t in ("veto", "near", "weak")},
        "note": ("Arbiter's declined pairs: harmony matched (sim>=%.2f) but the "
                 "split stood. tier=veto carries a concrete distinctive-chord "
                 "reason; tier=near was split by linkage/phase. Every card needs "
                 "human confirmation (real-audio same-section precision ~0.5)." % MATCH),
    }
    OUT.write_text(json.dumps({"deck": deck, "meta": meta}, indent=2))
    print(f"{len(deck)} candidates across {len(per_song)} songs -> {OUT.name}", file=sys.stderr)
    print(f"  by tier: {meta['by_tier']}", file=sys.stderr)
    for song, n in sorted(per_song.items(), key=lambda kv: -kv[1])[:12]:
        print(f"    {n:2d}  {song}", file=sys.stderr)


if __name__ == "__main__":
    main()
