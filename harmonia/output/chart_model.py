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

    sections: list[dict] = []
    for r in runs:
        sec_bars = bars[r["bar0"]: r["bar1"] + 1]
        span = _span_of(sec_bars)
        prev = sections[-1] if sections else None
        same_music = (fold_repeats and prev is not None
                      and prev["label"] == r["label"]
                      and [_bar_key(b) for b in prev["bars"]] == [_bar_key(b) for b in sec_bars])
        if same_music:
            # Fold: identical consecutive sections render once, badged ×N. Both
            # passes keep their own time span so the playhead (and a pooled
            # re-infer) still address the right audio.
            prev["reps"] += 1
            prev["spans"].append(span)
            prev["barRanges"].append([r["bar0"], r["bar1"]])
            continue
        sections.append({
            "id": r["label"], "label": r["label"], "tag": "", "reps": 1,
            "bars": sec_bars, "spans": [span], "barRanges": [[r["bar0"], r["bar1"]]],
        })

    sections = _coalesce_if_unreadable(sections)

    # Same-letter sections that were NOT folded (the model read them
    # differently) get A¹ / A² tags — those are exactly the pairs worth merging.
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
