"""ChartModel — the single normalised shape the app UI consumes.

The inference payload baked into every ``inferred_*.html`` (``const P = {…}``,
built by :mod:`harmonia.output.chart_interactive`) is per-chord and
under-structured: per-bar section *letters* rather than section spans, a
three-level confidence ladder rather than one number, no repeat folding, and
no cap on chords per bar. The app UI (docs/app_shell.html) wants exactly one
shape, documented in the design handoff:

    {title, video_id, audio_url, key:{tonic,mode}, bpb,
     sections:[{id, label, tag, reps, spans:[[t0,t1],…], bars:[Bar,…]}, …]}
    Bar   = [Chord] | [Chord, Chord]      (2 = split bar; never more)
    Chord = {root:0..11, q:<iReal tail>, c:0..1, t0, t1, sug?:[{root,q,c}], confirmed?}

This module is that adapter and the *only* place the messy→clean translation
happens; the UI never sees a raw payload. Rules implemented here mirror the
handoff's normalisation list (roots as pitch classes, one key per tune,
sections as real spans, repeats folded, ≤2 chords/bar, honest ``c``, seconds
for ``t0``/``t1``).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from .chart_interactive import _parse_home_key

# The confidence ladder level whose (q, c) the app displays. "exact" is what the
# chart shows at full depth; the UI does its own family/seventh collapse under
# low certainty (that IS the level-of-detail idea), so it needs the honest
# exact-level number, not a pre-collapsed one.
_DISPLAY_LEVEL = "exact"

_PAYLOAD_RE = re.compile(r"^const P = (\{.*\});\s*$", re.MULTILINE)
_KEYNAME_RE = re.compile(r"Key ([A-G][b#]?(?:\s*(?:major|minor|maj|min|m))?)")


def payload_from_chart_html(path: str | Path) -> dict:
    """Recover the inference payload from a rendered chart.

    Charts are the durable artifact of a run (the pipeline's own output is not
    persisted), so re-deriving a ChartModel means reading it back out of the
    HTML. Raises ValueError if the file isn't a chart we rendered.

    Charts baked before the ``_parse_home_key`` "major"-reads-as-minor fix have
    a wrong ``home.mode`` and no ``keyName``; the subhead still carries the raw
    key string, so recover it from there rather than re-rendering the library.
    """
    text = Path(path).read_text(encoding="utf-8")
    m = _PAYLOAD_RE.search(text)
    if not m:
        raise ValueError(f"No inference payload (const P = …) in {path}")
    payload = json.loads(m.group(1))
    if not payload.get("keyName"):
        km = _KEYNAME_RE.search(text)
        if km:
            payload["keyName"] = km.group(1).strip()
    return payload


def _title_from_filename(filename: str) -> str:
    stem = Path(filename).stem
    return stem.replace("inferred_", "").replace("_", " ").strip().title() or "Untitled"


def _bar_key(bar: list[dict]) -> tuple:
    """Identity of a bar's *music* (roots + qualities), for repeat folding."""
    return tuple((c["root"], c["q"]) for c in bar)


def to_chart_model(
    payload: dict,
    *,
    filename: str = "",
    title: str = "",
    video_id: str = "",
    audio_url: str = "",
    annotation: dict | None = None,
    fold_repeats: bool = True,
) -> dict:
    """Normalise a chart payload (+ its sidecar) into a ChartModel.

    ``annotation`` is the sidecar doc (docs/annotation_sidecar_schema.md); its
    ``chords`` corrections are applied and marked ``confirmed`` (c = 1) so the
    UI shows them locked and never re-decodes them.
    """
    ann = annotation or {}
    # sidecar corrections are keyed on (bar, beat) — see the schema's §5.1 note
    # on why the raw array index is not a stable key across re-renders.
    fixes = {(c["bar"], c.get("beat", 0)): c for c in ann.get("chords", [])}

    bpb = payload.get("bpb") or 4
    n_bars = payload.get("nBars") or 0
    per_bar_label: list[str] = payload.get("sections") or []

    # ── chords → bars ────────────────────────────────────────────────────────
    bars: list[list[dict]] = [[] for _ in range(n_bars)]
    for c in payload.get("chords", []):
        bar = c.get("bar", 0)
        if not 0 <= bar < n_bars:
            continue
        lv = (c.get("lv") or {}).get(_DISPLAY_LEVEL) or {}
        root, q, conf = c.get("root", 0) % 12, lv.get("q", ""), float(lv.get("c", 0.0))
        beat = c.get("beat", 0)
        is_nc = bool(c.get("nc"))
        confirmed = False
        fix = fixes.get((bar, beat))
        if fix:
            root, q, conf, confirmed = fix["root"] % 12, fix.get("q", ""), 1.0, True
            is_nc = False          # a user correction turns an N cell into a chord
        if is_nc:
            # No-chord: sentinel q="N", conf 0.  Distinct (root,q) so _bar_key
            # folds N bars together and never with a real C major bar.
            q, conf = "N", 0.0
        entry = {
            "root": root, "q": q, "c": round(min(max(conf, 0.0), 1.0), 4),
            # (bar, beat) is the annotation sidecar's key — carry it through so
            # a correction made in the app can be written back to the sidecar.
            "bar": bar, "beat": beat,
            "t0": float(c.get("t0", 0.0)), "t1": float(c.get("t1", 0.0)),
        }
        if is_nc:
            entry["nc"] = True
        if confirmed:
            entry["confirmed"] = True
        if c.get("sug"):
            entry["sug"] = [{"root": s["root"] % 12, "q": s.get("q", ""),
                             "c": round(float(s.get("c", 0.0)), 4)}
                            for s in c["sug"][:3]]
        bars[bar].append(entry)

    for i, bar in enumerate(bars):
        bar.sort(key=lambda e: e["beat"])
        if len(bar) > 2:
            # ≤2 chords per bar: keep the two the model is surest about, in
            # time order. 3+ in a bar is nearly always segmentation noise, and
            # the iReal grid has no way to draw it.
            keep = sorted(sorted(bar, key=lambda e: -e["c"])[:2], key=lambda e: e["beat"])
            bars[i] = keep

    runs = _section_runs(payload, bars, n_bars, per_bar_label)

    # Raw sections (one per changepoint run, reps=1) — folding + rank-relabel
    # happen below.
    raw: list[dict] = []
    for r in runs:
        sec_bars = bars[r["bar0"]: r["bar1"] + 1]
        raw.append({
            "id": r["label"], "label": r["label"], "tag": "", "reps": 1,
            "bars": sec_bars, "spans": [_span_of(sec_bars)],
            "barRanges": [[r["bar0"], r["bar1"]]],
        })

    # iReal-style repeat folding (user directive 2026-07-19): a section — or a
    # multi-section LOOP UNIT — whose content repeats k times renders ONCE badged
    # ×k, when the passes are (near-)identical.  Handles the alternating vamp
    # (barlocked emits the loop's two phases as A,B,A,B… so consecutive-identical
    # folding never fires — the loop unit period is 2) as well as a plain
    # consecutive repeat (period 1).  Each pass keeps its own span so the playhead
    # tracks all k passes (SPA loadModel spans-per-pass).
    sections = _fold_section_loops(raw) if fold_repeats else raw

    # Letters by DISTINCT CONTENT TYPE first (so barlocked's over-segmented loop
    # phases collapse to one letter), THEN merge adjacent same-letter sections —
    # the chart converges toward the canonical form (user directive 2026-07-19:
    # "il ne devrait y avoir qu'un A et un B par chanson").
    _relabel_by_reps(sections)
    sections = _coalesce_adjacent_same_letter(sections)
    sections = _coalesce_if_unreadable(sections)
    _relabel_by_reps(sections)          # re-rank after merges settle the reps

    # Non-adjacent occurrences of the SAME letter (a real return of the material,
    # e.g. verse … chorus … verse) get A¹ / A² occurrence tags.
    by_label: dict[str, list[dict]] = {}
    for s in sections:
        by_label.setdefault(s["label"], []).append(s)
    sup = "¹²³⁴⁵⁶⁷⁸⁹"
    for label, group in by_label.items():
        if len(group) > 1:
            for i, s in enumerate(group):
                s["id"] = f"{label}{i + 1}"
                s["tag"] = sup[i] if i < len(sup) else str(i + 1)

    home = payload.get("home") or {}
    tonic, mode = int(home.get("tonic", 0)) % 12, home.get("mode", "major")
    if payload.get("keyName"):
        # Prefer the raw key string: charts baked before the _parse_home_key
        # fix carry a wrong home.mode ("G# major" → minor).
        tonic, mode = _parse_home_key(payload["keyName"])
    return {
        "file": filename,
        "title": title or _title_from_filename(filename),
        "video_id": video_id,
        "audio_url": audio_url,
        "key": {"tonic": tonic, "mode": mode},
        "bpb": bpb,
        "nBars": n_bars,
        "sections": sections,
        "merges": ann.get("merges", []),
    }


_MAX_SECTIONS = 12
_RANK_ALPHA = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def _is_form_label(lbl: str) -> bool:
    return bool(lbl) and lbl.lower() != "intro" and len(lbl) <= 2 and " " not in lbl


def _bars_near_eq(a: list[list[dict]], b: list[list[dict]], frac: float = 0.7) -> bool:
    """Two bar sequences are (near-)identical: same length and ≥ ``frac`` of bars
    share the exact (root, quality) content.  ``frac`` < 1 tolerates the per-pass
    decode noise the user's directive calls out ("(near-)identical post-Occam")."""
    if len(a) != len(b):
        return False
    if not a:
        return True
    same = sum(1 for x, y in zip(a, b) if _bar_key(x) == _bar_key(y))
    return same / len(a) >= frac


def _section_span_bars(sec_bars: list[list[dict]], bar0: int) -> tuple[float, float]:
    times = [c["t0"] for b in sec_bars for c in b] + [c["t1"] for b in sec_bars for c in b]
    return (min(times), max(times)) if times else (0.0, 0.0)


def _fold_section_loops(raw: list[dict], frac: float = 0.7) -> list[dict]:
    """Fold a repeating BAR loop into one ×k block (iReal ‖: … :‖ ×k).

    The repetition in real charts lives at the BAR level (henny's 2-bar A|Bm7
    vamp, a 4- or 8-bar verse loop), NOT at barlocked's section level — barlocked
    emits irregular variable-length runs that never match each other, so a
    section-level fold is a no-op (measured on abba/henny).  So flatten each
    contiguous run of form-letter sections to its bars, find the bar period ``P``
    ∈ {2,4,8} whose consecutive near-identical P-bar blocks fold away the most
    bars (Occam: biggest compression, ties → smallest P), and emit each folded
    run as ONE super-section: bars = the first P-bar block, reps = k, and every
    pass keeps its own time span so the playhead tracks all k passes (SPA
    loadModel spans-per-pass).  ``Intro`` (non-form-letter) sections are hard
    boundaries, passed through untouched.  A chart with no bar-level repeat passes
    through unchanged.
    """
    out: list[dict] = []
    i = 0
    n = len(raw)
    while i < n:
        if not _is_form_label(str(raw[i]["label"])):
            out.append(raw[i]); i += 1
            continue
        # gather a maximal run of contiguous form-letter sections
        j = i
        run_bars: list[list[dict]] = []
        run_secs: list[dict] = []
        run_bar0 = raw[i]["barRanges"][0][0]
        while j < n and _is_form_label(str(raw[j]["label"])):
            run_bars.extend(raw[j]["bars"]); run_secs.append(raw[j]); j += 1
        folded = _fold_bar_run(run_bars, run_bar0, raw[i]["label"], frac)
        # ABSTAIN → preserve the ORIGINAL section breakdown (never merge a
        # noisy run into one fake loop; anti-crush).  Only replace when a real
        # dominant loop was found.
        out.extend(folded if folded is not None else run_secs)
        i = j
    return out


def _fold_bar_run(bars: list[list[dict]], bar0: int, label: str,
                  frac: float = 0.7, dominance: float = 0.6) -> "list[dict] | None":
    """Fold a flat bar run to its MODAL P-bar loop block ×k, iReal-style.

    Strict consecutive-block folding fragments under decode noise (a single
    off-loop bar splits the run).  Instead, for each period P ∈ {2,4,8}, take the
    modal P-bar block and count how many of the run's ``m//P`` blocks are
    near-identical to it.  If a period's modal block explains ≥ ``dominance`` of
    the blocks, the run IS that loop → emit ONE super-section (bars = the modal
    block, reps = number of blocks, each pass keeping its own span for playback).
    This gives a clean "A|Bm7 ×N" for a genuine single-loop body and, crucially,
    ABSTAINS (returns the run unfolded) for a verse/chorus body where no single
    block dominates — so real structure is never crushed into a fake loop.
    Prefer the period with the highest dominance; ties → smaller P.
    """
    m = len(bars)

    def _dominant_block(P: int):
        nb = m // P
        if nb < 2:
            return None, 0.0
        blocks = [bars[x * P:(x + 1) * P] for x in range(nb)]
        best_mode, best_hits = None, 0
        for cand in blocks:
            hits = sum(1 for b in blocks if _bars_near_eq(cand, b, frac))
            if hits > best_hits:
                best_hits, best_mode = hits, cand
        return best_mode, best_hits / nb

    pick = None
    for P in (2, 4, 8):
        if P * 2 > m:
            continue
        mode, dom = _dominant_block(P)
        if dom >= dominance and (pick is None or dom > pick[2]):
            pick = (P, mode, dom)
    if pick is None:
        return None                        # abstain — caller preserves structure
    P, mode, _dom = pick
    nb = m // P
    merged = {"id": label, "label": label, "tag": "", "reps": nb,
              "bars": mode, "spans": [], "barRanges": []}
    for r in range(nb):
        blk = bars[r * P:(r + 1) * P]
        merged["spans"].append(list(_section_span_bars(blk, bar0 + r * P)))
        merged["barRanges"].append([bar0 + r * P, bar0 + (r + 1) * P - 1])
    out = [merged]
    rem = m - nb * P                       # trailing partial loop, kept unfolded
    if rem:
        tail = bars[nb * P:]
        out.append({"id": label, "label": label, "tag": "", "reps": 1,
                    "bars": tail,
                    "spans": [list(_section_span_bars(tail, bar0 + nb * P))],
                    "barRanges": [[bar0 + nb * P, bar0 + m - 1]]})
    return out


def _section_root_set(sec: dict) -> set:
    """A section's CONTENT TYPE fingerprint = the SET of its distinct chord ROOTS.
    Length-invariant (a verse decoded 8 vs 12 bars is the same type) and robust to
    per-pass quality wobble; two sections are the same content when their root sets
    overlap strongly (Jaccard), a genuinely different progression gets a different
    set → a different letter.  N (no-chord) cells are ignored."""
    return {c["root"] for bar in sec.get("bars", []) for c in bar
            if c.get("q") != "N"}


def _root_sets_match(a: set, b: set, jaccard: float = 0.5) -> bool:
    if not a and not b:
        return True
    inter = len(a & b)
    union = len(a | b) or 1
    return inter / union >= jaccard


def _relabel_by_reps(sections: list[dict]) -> None:
    """Assign section letters by DISTINCT CONTENT TYPE (user directive 2026-07-19:
    "il ne devrait y avoir qu'un A et un B par chanson; s'il y en a d'autres ce
    sont des C et des D").  A letter names a CONTENT TYPE, not an occurrence:

      * cluster form-letter sections by content signature (distinct chord set);
      * two clusters with different content NEVER share a letter;
      * all occurrences of the same content carry the same letter;
      * rank clusters by (total folded reps desc, total bars desc, first
        appearance asc) → **A = most-repeated distinct material, B = second, …**,
        "on commence toujours par A" from the first-appearance tie-break.

    Mutates ``sections`` in place (label + id).  Intro / non-form-letter labels
    untouched.  Deterministic → stable across fresh runs."""
    clusters: list[dict] = []          # {roots, reps, bars, first, members:[idx]}
    for i, s in enumerate(sections):
        if not _is_form_label(str(s.get("label", ""))):
            continue
        rs = _section_root_set(s)
        reps = int(s.get("reps", 1) or 1)
        c = next((c for c in clusters if _root_sets_match(c["roots"], rs)), None)
        if c is None:
            c = {"roots": set(rs), "reps": 0, "bars": 0, "first": i, "members": []}
            clusters.append(c)
        c["roots"] |= rs               # grow the cluster fingerprint
        c["reps"] += reps
        c["bars"] += reps * len(s.get("bars", []))
        c["members"].append(i)
    if not clusters:
        return
    clusters.sort(key=lambda c: (-c["reps"], -c["bars"], c["first"]))
    for rank, c in enumerate(clusters):
        letter = _RANK_ALPHA[rank] if rank < len(_RANK_ALPHA) else "?"
        for i in c["members"]:
            sections[i]["label"] = sections[i]["id"] = letter


def _coalesce_adjacent_same_letter(sections: list[dict]) -> list[dict]:
    """Merge consecutive sections that carry the same letter (same content type)
    into one — barlocked over-segments a loop into alternating phase-sections that
    the content relabel just gave a single letter, so this converges them to the
    canonical one-block-per-type form.  Bars/spans/barRanges concatenate; reps of
    a single merged block is 1 (its repeats are internal), so ranking still works."""
    out: list[dict] = []
    for s in sections:
        if out and _is_form_label(str(out[-1]["label"])) and out[-1]["label"] == s["label"]:
            prev = out[-1]
            prev["bars"] = prev["bars"] + s["bars"]
            prev["spans"] = [[prev["spans"][0][0], s["spans"][-1][1]]]
            prev["barRanges"] = [[prev["barRanges"][0][0], s["barRanges"][-1][1]]]
            prev["reps"] = 1
        else:
            out.append(s)
    return out


def _coalesce_if_unreadable(sections: list[dict]) -> list[dict]:
    """Merge adjacent same-letter sections when there are too many to read.

    A 6-minute track with a solo over the form segments into ~40 four-bar
    chips, none of which fold (every pass is voiced a little differently), and
    a 40-chip form ribbon is not a form. Falling back to the coarse letter
    blocks keeps the ribbon usable; the bars themselves are untouched, so
    nothing about the chart is lost — only the merge granularity.
    """
    if len(sections) <= _MAX_SECTIONS:
        return sections
    out: list[dict] = []
    for s in sections:
        if out and out[-1]["label"] == s["label"]:
            prev = out[-1]
            prev["bars"].extend(s["bars"])
            prev["spans"] = [[prev["spans"][0][0], s["spans"][-1][1]]]
            prev["barRanges"] = [[prev["barRanges"][0][0], s["barRanges"][-1][1]]]
            prev["reps"] = 1
        else:
            out.append(s)
    return out


def _looks_like_a_form_letter(label: str) -> bool:
    """Per-bar section labels are supposed to be form letters (A/B/C). On
    real-audio charts they are the *key name* instead ("G# major") — the
    pipeline fills `section_per_bar` with the local key. Anything with a space
    or longer than two characters is not a form letter."""
    return bool(label) and len(label) <= 2 and " " not in label


def _section_runs(payload: dict, bars: list[list[dict]], n_bars: int,
                  per_bar_label: list[str]) -> list[dict]:
    """Sections as real spans of bars, in play order.

    Preference order, per the handoff's rule 3:
      1. ``sectionChips`` — the pipeline's own changepoint segmentation, as
         (label, start_s). Consecutive chips with the same letter are one
         section (chips are ~4-bar blocks; a run of A chips is one A).
      2. per-bar letters — symbolic (iReal) charts, where they really are A/B/C.
      3. one section "A" over the whole tune — always renders, never lies.
    """
    chips = payload.get("sectionChips") or []
    if chips:
        # Each chip is one segment of the changepoint segmentation, so two
        # adjacent "A" chips are two PASSES of A — not one long A. Keep them
        # separate: identical passes fold to ×N below, and passes the model
        # read differently stay apart as A¹/A², which is what merge is for.
        bar_t0 = [b[0]["t0"] if b else None for b in bars]
        starts: list[tuple[int, str]] = []
        for chip in chips:
            t = float(chip.get("start_s", 0.0))
            bar = next((i for i, t0 in enumerate(bar_t0) if t0 is not None and t0 >= t - 1e-6), None)
            if bar is None:
                continue
            lbl = str(chip.get("label") or "A")
            if starts and starts[-1][0] == bar:
                continue                       # two chips inside one bar
            starts.append((bar, lbl))
        if starts:
            if starts[0][0] != 0:
                starts.insert(0, (0, starts[0][1]))
            runs = []
            for i, (bar0, lbl) in enumerate(starts):
                bar1 = (starts[i + 1][0] - 1) if i + 1 < len(starts) else n_bars - 1
                if bar1 >= bar0:
                    runs.append({"label": lbl, "bar0": bar0, "bar1": bar1})
            if runs:
                return runs

    if any(_looks_like_a_form_letter(x) for x in per_bar_label):
        runs = []
        for b in range(n_bars):
            lbl = per_bar_label[b] if b < len(per_bar_label) else ""
            lbl = lbl if _looks_like_a_form_letter(lbl) else "A"
            if runs and runs[-1]["label"] == lbl:
                runs[-1]["bar1"] = b
            else:
                runs.append({"label": lbl, "bar0": b, "bar1": b})
        if runs:
            return runs

    return [{"label": "A", "bar0": 0, "bar1": max(n_bars - 1, 0)}]


def _span_of(sec_bars: list[list[dict]]) -> list[float]:
    """[t0, t1] of a run of bars — the audio span a merge would pool."""
    times = [c["t0"] for b in sec_bars for c in b] + [c["t1"] for b in sec_bars for c in b]
    return [min(times), max(times)] if times else [0.0, 0.0]


def chart_summary(model: dict) -> dict:
    """One library-card's worth of a ChartModel (no bars) — for /api/library."""
    n_bars = model.get("nBars", 0)
    key = model["key"]
    return {
        "file": model["file"], "title": model["title"],
        "key": key, "bars": n_bars,
        "sections": "".join(s["label"] for s in model["sections"]),
        "hasAudio": bool(model.get("audio_url")),
    }
