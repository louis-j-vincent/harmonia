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
import os as _os_cm
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
    consumed_fixes: set[tuple[int, int]] = set()
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
            consumed_fixes.add((bar, beat))
        if is_nc:
            # No-chord: sentinel q="N", conf 0.  Distinct (root,q) so _bar_key
            # folds N bars together and never with a real C major bar.
            q, conf = "N", 0.0
        # A split-bar's KEPT half (the raw chord that already existed) needs
        # its t0/t1 SHRUNK to make room for the new second half — the sidecar
        # fix carries the client's already-computed midpoint split, so once a
        # fix exists it's the source of truth for timing too, not just root/q
        # (2026-07-20; without this the kept half kept displaying/playing its
        # original full-bar span even though a second chord now shares the
        # bar with it).
        t0 = float(fix.get("t0", c.get("t0", 0.0))) if fix else float(c.get("t0", 0.0))
        t1 = float(fix.get("t1", c.get("t1", 0.0))) if fix else float(c.get("t1", 0.0))
        entry = {
            "root": root, "q": q, "c": round(min(max(conf, 0.0), 1.0), 4),
            # (bar, beat) is the annotation sidecar's key — carry it through so
            # a correction made in the app can be written back to the sidecar.
            "bar": bar, "beat": beat,
            "t0": t0, "t1": t1,
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

    # Split-bar additions (2026-07-20): a sidecar fix whose (bar, beat) never
    # existed in the raw inference output — the app's "Split into two" action
    # adds a genuinely NEW second-half chord, not a correction to an existing
    # one, so there's nothing in payload["chords"] for the loop above to
    # attach it to. Synthesize an entry directly from the fix (it already
    # carries t0/t1 from the client's own split — see saveAnnotations).
    for (bar, beat), fix in fixes.items():
        if (bar, beat) in consumed_fixes or not 0 <= bar < n_bars:
            continue
        bars[bar].append({
            "root": fix["root"] % 12, "q": fix.get("q", ""), "c": 1.0,
            "bar": bar, "beat": beat, "confirmed": True,
            "t0": float(fix.get("t0", 0.0)), "t1": float(fix.get("t1", 0.0)),
        })

    for i, bar in enumerate(bars):
        bar.sort(key=lambda e: e["beat"])
        if len(bar) > 2:
            # ≤2 chords per bar: keep the two the model is surest about, in
            # time order. 3+ in a bar is nearly always segmentation noise, and
            # the iReal grid has no way to draw it.
            keep = sorted(sorted(bar, key=lambda e: -e["c"])[:2], key=lambda e: e["beat"])
            bars[i] = keep

    runs = _section_runs(payload, bars, n_bars, per_bar_label)

    # Trusted-boundary charts (2026-07-20 — iReal imports): the whole point of
    # _sections_by_largest_unit / _fold_section_loops below is to RECOVER
    # section structure that barlocked's noisy audio-decode boundaries can't
    # be trusted to give directly. An iReal import's sections come straight
    # from the source's own *A/*B/*C markers — already ground-truth, nothing
    # to recover — so the loop-fold heuristic has nothing to fix and only
    # risk mis-firing on unusually clean, exactly-cyclic content (confirmed:
    # it pooled a legitimate A(×2)/B/C into one fake "3 reps" block, eating
    # the real B/C boundary). Payload opts out via "sections_trusted": true.
    if payload.get("sections_trusted"):
        sections = [{"id": r["label"], "label": r["label"], "tag": "", "reps": 1,
                    "bars": bars[r["bar0"]: r["bar1"] + 1],
                    "spans": [_span_of(bars[r["bar0"]: r["bar1"] + 1])],
                    "barRanges": [[r["bar0"], r["bar1"]]]} for r in runs]
        home = payload.get("home") or {}
        tonic, mode = int(home.get("tonic", 0)) % 12, home.get("mode", "major")
        if payload.get("keyName"):
            tonic, mode = _parse_home_key(payload["keyName"])
        return {
            "file": filename, "title": title or _title_from_filename(filename),
            "video_id": video_id, "audio_url": audio_url,
            "key": {"tonic": tonic, "mode": mode}, "bpb": bpb, "nBars": n_bars,
            "sections": sections, "merges": ann.get("merges", []),
        }

    # ── LARGEST-REPEATING-UNIT sections (user design principle 2026-07-20): the
    # section entity is the LARGEST span that repeats (≥2×) — a phrase (8/16 bars),
    # NOT the small P2/P4 loop that lives inside it.  Find the largest bar-multiple
    # lag L with strong self-recurrence, cut at L-boundaries, cluster L-blocks by
    # ORDERED content into letters.  Only OVERRIDES the changepoint runs when it
    # finds a clear phrase-repeating structure — else falls back (no regression).
    lu = _sections_by_largest_unit(bars, n_bars) if fold_repeats else None
    if lu is not None:
        # Largest-unit path already produced final folded + rank-lettered phrase
        # sections; use them directly (its ordered-content clustering is stricter
        # than the changepoint fold/relabel below and must not be re-merged).
        sections = _coalesce_if_unreadable(lu)
    else:
        # Raw sections (one per changepoint run, reps=1).
        raw = []
        for r in runs:
            sec_bars = bars[r["bar0"]: r["bar1"] + 1]
            raw.append({
                "id": r["label"], "label": r["label"], "tag": "", "reps": 1,
                "bars": sec_bars, "spans": [_span_of(sec_bars)],
                "barRanges": [[r["bar0"], r["bar1"]]],
            })

        # iReal-style repeat folding (user directive 2026-07-19): a section — or a
        # multi-section LOOP UNIT — whose content repeats k times renders ONCE
        # badged ×k, when the passes are (near-)identical.  Each pass keeps its
        # own span so the playhead tracks all k passes (SPA loadModel).
        sections = _fold_section_loops(raw)

        # Letters by DISTINCT CONTENT TYPE first, THEN merge adjacent same-letter
        # sections — converge toward the canonical form ("un A et un B par chanson").
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


def _bar_root_seq(bars: list[list[dict]], n_bars: int) -> list[int]:
    """One dominant chord-root per bar; a held/empty/N bar inherits the previous
    bar's root (a held chord still sounds) so recurrence sees the real harmony."""
    seq: list[int] = []
    prev = -1
    for b in range(n_bars):
        bar = bars[b] if b < len(bars) else []
        r = next((c["root"] for c in bar if c.get("q") != "N"), None)
        if r is None:
            r = prev
        seq.append(r)
        prev = r
    return seq


def _sections_by_largest_unit(bars: list[list[dict]], n_bars: int, *,
                              cands=(16, 8), rec_min: float = 0.55,
                              match: float = 0.6):
    """Sections = the LARGEST repeating phrase (user principle 2026-07-20).

    Find the largest bar-multiple lag ``L`` ∈ ``cands`` whose L-shifted root
    recurrence clears ``rec_min`` (a genuine phrase repeat, not a bar-loop), cut
    the song into L-blocks, and cluster blocks by ORDERED content (sequence
    near-equality, ``match``) into section letters — the ordered signature fixes
    the verse/chorus chord-SET over-merge.  Returns raw section dicts, or ``None``
    when no phrase-scale repeat is found (caller keeps the changepoint sections —
    no regression on through-composed / short songs).
    """
    if n_bars < 16:
        return None
    R = _bar_root_seq(bars, n_bars)
    # 8-BAR BASE SCALE (validated 2026-07-21, docs/research_sessions/section_
    # discrimination_grammar_2026-07-20.md ckpt 8): the shipped 16-first grain
    # collapses repetitive pop to ONE letter (16-bar blocks so long every block
    # matches every other → over-merge 90% vs GT). Base the grid on 8 bars — the
    # modal phrase in BOTH genres (corpus §H) — and let a 16-bar unit emerge via
    # the agglomerative MERGE of two adjacent same-cluster 8-blocks (fold below).
    # This dropped symbolic over-merge 90.5%→29% on pop, matching the user's
    # confirmed "more sections > fewer" error-preference.  Kill-switch:
    # HARMONIA_SECTION_CANDS="16,8" restores the old behaviour.
    cands = tuple(int(x) for x in
                  _os_cm.environ.get("HARMONIA_SECTION_CANDS", "8,16").split(","))
    L = None
    for cand in cands:
        if 2 * cand > n_bars:
            continue
        rec = sum(1 for b in range(cand, n_bars) if R[b] == R[b - cand]) / (n_bars - cand)
        if rec >= rec_min:
            L = cand
            break
    if L is None:
        return None
    blocks = [(i, min(i + L, n_bars)) for i in range(0, n_bars, L)]
    if len(blocks) < 2:
        return None

    seqs = [R[b0:b1] for (b0, b1) in blocks]
    nb = len(blocks)

    # PHASE-TOLERANT block matching (user diagnosis 2026-07-20, D8-bis): on real
    # audio a phrase repeated N times phase-DRIFTS by a bar or so by its later
    # passes, so a strict position-by-position compare of two same-content blocks
    # collapses to ~0 and the drifted pass gets split into a FALSE new section
    # letter ("the same chords but shifted → separate cluster → labelled B").
    # Measured on Let It Be: identical-but-1-bar-drifted 8-bar blocks score strict
    # 0.00 but phase-tolerant 1.00.  Fix: allow a small bar LAG (±_PHASE_MAXLAG)
    # when comparing, take the best alignment.  The lag is capped tiny so it only
    # absorbs drift, never aligns two genuinely different progressions by sliding
    # them arbitrarily.  A length-mismatched trailing partial block is compared on
    # its overlap (so a short song-end fragment still merges into its phrase rather
    # than minting a letter).  Kill-switch HARMONIA_SECTION_PHASE_TOL=0.
    _phase_tol = _os_cm.environ.get("HARMONIA_SECTION_PHASE_TOL", "1") == "1"
    _PHASE_MAXLAG = 1
    # A shifted (phase≠0) alignment is only TRUSTED if it is a STRONG match — real
    # drift makes two same-content blocks align almost perfectly under one small
    # lag (Let It Be: 1.00), whereas two GENUINELY DIFFERENT sections (Bein' Green's
    # AABA bridge, a verse vs chorus) only find a WEAK coincidental partial overlap
    # when slid (~0.6).  Requiring a lagged match to clear _PHASE_STRICT before it
    # can raise the similarity stops the slide from dissolving real B sections
    # (over-merge regression measured with a plain max-over-lags).  Lag-0 keeps the
    # normal ``match`` threshold.
    _PHASE_STRICT = 0.80

    def _overlap_match(a, b):
        k = min(len(a), len(b))
        if k == 0:
            return 0.0
        return sum(1 for x, y in zip(a[:k], b[:k]) if x == y) / k

    def _sim(i, j):
        a, b = seqs[i], seqs[j]
        if not a or not b:
            return 0.0
        base = (sum(1 for x, y in zip(a, b) if x == y) / len(a)
                if len(a) == len(b) else _overlap_match(a, b))
        if not _phase_tol:
            return base if len(a) == len(b) else 0.0
        # phase-tolerant: a small lag may recover a drifted repetition, but only a
        # STRONG shifted match (>= _PHASE_STRICT) is trusted (else weak coincidental
        # overlap between different sections would over-merge).  min-length overlap
        # also lets a trailing partial block merge into its phrase.
        best = base
        for lag in range(1, _PHASE_MAXLAG + 1):
            for m in (_overlap_match(a[lag:], b), _overlap_match(a, b[lag:])):
                if m >= _PHASE_STRICT:
                    best = max(best, m)
        return best

    # SINGLE-LINKAGE clustering over the L-blocks (union-find): two phrases are the
    # same section if they match ≥ ``match`` — so a repeating phrase whose passes
    # each differ from the FIRST by decode noise but chain through intermediates
    # still merges into ONE letter (Let It Be's 8-bar blocks all match some other
    # at 0.75–0.88 → one A ×N, the user's "one clear A").  Uses the bar-ROOT
    # SEQUENCE (ordered), so verse/chorus that share a chord set but differ in
    # order stay distinct (fixes the Jaccard over-merge).
    parent = list(range(nb))
    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(nb):
        for j in range(i + 1, nb):
            if _sim(i, j) >= match:
                parent[_find(i)] = _find(j)
    groups: dict[int, list[int]] = {}
    for i in range(nb):
        groups.setdefault(_find(i), []).append(i)
    clusters = list(groups.values())
    block_label = [_find(i) for i in range(nb)]
    # require a genuine repeat: at least one cluster with ≥2 blocks
    if not any(len(c) >= 2 for c in clusters):
        return None
    # anti-fragmentation guard: a real pop/rock form has FEW section types (verse/
    # chorus/bridge ≈ 2–4).  If single-linkage still shatters into >4 clusters
    # (a through-composed jazz head like Autumn Leaves, or a decode too noisy for
    # any two phrases to match), this is not a clean phrase structure → fall back
    # (no regression).  Also require the dominant phrase to cover a real share of
    # the song, so a one-off repeat in an otherwise-unique sequence doesn't win.
    if len(clusters) > 4:
        return None
    if max(len(c) for c in clusters) < max(2, 0.3 * nb):
        return None
    # INTRO = the LEADING run of blocks before the first occurrence of the
    # DOMINANT phrase, WHEN those leading clusters never recur afterwards (user
    # convention 2026-07-20: letters are for REPEATED / non-leading-distinct
    # material; a leading-only phrase — even one that repeats a couple times at the
    # very start — is an Intro, rendered label-only; a distinct one-off in the
    # MIDDLE stays a lettered bridge).  Let It Be: B×2·A×15·C → Intro·A×15·B;
    # Stand By Me: B·A×2·C → Intro·A×2·B.
    dominant_root = max(groups, key=lambda r: (len(groups[r]),
                        sum(blocks[i][1] - blocks[i][0] for i in groups[r])))
    first_dom = next((bi for bi in range(nb) if _find(bi) == dominant_root), 0)
    lead_roots = {_find(bi) for bi in range(first_dom)}
    # leading clusters must NOT appear at/after first_dom (else they're a real
    # recurring section that merely opens the song → keep their letters)
    recurs = any(_find(bi) in lead_roots for bi in range(first_dom, nb))
    intro_upto = first_dom if (first_dom > 0 and not recurs) else 0
    intro_roots = {_find(bi) for bi in range(intro_upto)}
    # letters by CHRONOLOGICAL first-appearance over the NON-intro clusters (user
    # 2026-07-20, correcting D9: "la première partie c'est A, la deuxième c'est B
    # … c'est l'ordre chronologique, alphabétique").  The FIRST distinct content to
    # appear is A, the next new distinct content is B, etc. — repetition COUNT no
    # longer decides the LETTER (it only decides clustering/merging, unchanged).
    rest = [g for r, g in groups.items() if r not in intro_roots]
    order = sorted(rest, key=lambda g: min(g))       # earliest block first
    letter_of = {}
    for rank, g in enumerate(order):
        for i in g:
            letter_of[i] = chr(ord("A") + rank) if rank < 26 else "?"
    out: list[dict] = []
    if intro_upto > 0:                    # collapse the leading one-off run → Intro
        ib0, ib1 = blocks[0][0], blocks[intro_upto - 1][1]
        ibars = bars[ib0:ib1]
        out.append({"id": "Intro", "label": "Intro", "tag": "", "reps": 1,
                    "bars": ibars, "spans": [_span_of(ibars)],
                    "barRanges": [[ib0, ib1 - 1]]})
    for bi in range(intro_upto, nb):
        b0, b1 = blocks[bi]
        letter = letter_of[_find(bi)]
        sec_bars = bars[b0:b1]
        span = _span_of(sec_bars)
        # fold ADJACENT same-cluster blocks into one ×N (repeating phrase shown once)
        if out and out[-1]["label"] == letter and _find(bi) == _find(bi - 1):
            prev = out[-1]
            prev["reps"] += 1
            prev["spans"].append(span)
            prev["barRanges"].append([b0, b1 - 1])
            prev["_blockIdxs"].append(bi)
        else:
            out.append({"id": letter, "label": letter, "tag": "", "reps": 1,
                        "bars": sec_bars, "spans": [span], "barRanges": [[b0, b1 - 1]],
                        "_blockIdxs": [bi]})

    # RECURRING-VOCAB representative (user report 2026-07-20, She Will Be Loved: an
    # Eb chorus chord "never shown").  A folded ×N section shows ONE representative
    # phrase; the FIRST block drops any chord the first pass missed to local decode
    # noise but that RECURS across the other passes (SWBL's first A block is the
    # Cm–Bb vamp with no Eb, while ~2/3 of the A blocks DO carry the Eb).  Neither
    # "first" (misses Eb) nor "medoid" (the Cm–Bb vamp is the tightest sub-cluster
    # → still no Eb) nor "richest" (over-writes one-off noise, against the
    # under-write principle) is right.  Pick the block that best COVERS the
    # cluster's RECURRING vocabulary — roots present in ≥ ``_REP_VOCAB_FRAC`` of the
    # section's blocks — so a chord that recurs (Eb, 8/13) surfaces while a one-off
    # decode artifact does not.  Structure (labels/reps/spans/barRanges — playback)
    # unchanged; only the DISPLAYED bars change.  Kill-switch HARMONIA_FOLD_REP=0.
    _REP_VOCAB_FRAC = 0.4
    if _os_cm.environ.get("HARMONIA_FOLD_REP", "1") != "0":
        for sec in out:
            idxs = sec.get("_blockIdxs") or []
            if sec["reps"] < 2 or sec["label"] == "Intro" or len(idxs) < 3:
                continue
            block_roots = [{e["root"] for bar in bars[blocks[i][0]:blocks[i][1]]
                            for e in bar if e.get("q") != "N"} for i in idxs]
            from collections import Counter as _Ctr
            freq = _Ctr(r for rs in block_roots for r in rs)
            need = max(2, int(round(_REP_VOCAB_FRAC * len(idxs))))
            vocab = {r for r, c in freq.items() if c >= need}
            if not vocab:
                continue

            def _cover(k):
                covered = len(block_roots[k] & vocab)
                extra = len(block_roots[k] - vocab)      # penalise one-off noise
                return (covered, -extra, -idxs[k])
            best_k = max(range(len(idxs)), key=_cover)
            mb0, mb1 = blocks[idxs[best_k]]
            sec["bars"] = bars[mb0:mb1]
    for sec in out:
        sec.pop("_blockIdxs", None)
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
    """Assign section letters by DISTINCT CONTENT TYPE, ordered CHRONOLOGICALLY
    (user 2026-07-20, correcting the earlier repetition-rank rule: "la première
    partie c'est A, la deuxième c'est B, pas l'inverse — c'est l'ordre
    chronologique, alphabétique").  A letter names a CONTENT TYPE, not an
    occurrence:

      * cluster form-letter sections by content signature (distinct chord set);
      * two clusters with different content NEVER share a letter;
      * all occurrences of the same content carry the same letter;
      * order clusters by FIRST APPEARANCE in time → **A = the first distinct
        content to appear, B = the next new distinct content, …** — repetition
        COUNT no longer decides the letter (only clustering stays content-based).
        (Function name kept for call-site stability; behaviour is now by-appearance.)

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
    clusters.sort(key=lambda c: c["first"])          # chronological first-appearance
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
