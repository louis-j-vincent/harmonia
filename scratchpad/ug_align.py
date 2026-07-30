#!/usr/bin/env python3
"""ug_align.py — give every chord of an Ultimate Guitar tab a timestamp in our audio.

    .venv/bin/python scratchpad/ug_align.py <slug> <ug_url_or_cached_html> [options]

WHAT THIS IS
------------
A *forced aligner* for chord charts.  The tab tells us WHICH chords happen and
in WHAT ORDER; it says nothing about WHEN.  The audio says when things change
but our chord model is what we are trying to check.  So: hold the UG chord
SEQUENCE fixed and search only over its timing — a monotone segmentation of the
audio into N spans, one per UG chord, scored by chord-tone/chroma agreement.

This is deliberately NOT "run our recogniser and match strings".  Nothing here
reads our inferred chords except for the diagnostic plot; the alignment is
tab-vs-audio only, so the result is usable as reference ground truth for the
very model it would otherwise be circular with.

METHOD (phase 1, harmony only)
------------------------------
1. Parse the UG tab -> ordered chord events with section labels and a *relative
   duration weight* (character span of the chord over the lyric line beneath it,
   which is a surprisingly good proxy for sung time; bare chord lines get one
   bar each).
2. Turn each chord into a 24-d template: 12 treble bins = ``local_key.chord_pcs``
   weights (chord-TONE weighted, so Bb sits closer to Gm than to F — never
   root-only, per the standing rule), 12 bass bins = bass/root emphasis.
3. NNLS-chroma the audio (cached), rotate A-first -> C-first, split bass/treble.
4. Segmental Viterbi over (chord index x frame): every chord gets exactly one
   contiguous span, spans are in tab order, with a log-ratio duration prior
   against the tab's own relative weights and free (unpenalised-ish) skip at the
   head and tail so an intro or a fade the tab never wrote does not drag the
   first/last chord over it.
5. Identifiability audit — because a plausible alignment on a one-chord vamp is
   not an alignment at all (see Chain of Fools).  Reported, never hidden.

Phase 2 (``--asr``) adds lyric anchors: Whisper word timestamps fuzzy-matched to
the tab's own lyric lines become hard constraints that the DP must pass through.
That is what cracks a song whose harmony is constant.

Outputs (all under scratchpad/):
  ug_align_<slug>.json   landmark table + per-chord timestamps + audit numbers
  ug_align_<slug>.png    our chart chords vs aligned UG chords, shared time axis
"""
from __future__ import annotations

import argparse
import html as _html
import json
import math
import re
import subprocess
import sys
import unicodedata
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.theory.local_key import (chord_pcs, parse_token,  # noqa: E402
                                       prefer_flats, transpose_token)


def _prefer_flats_tok(tok: str, semis: int) -> bool:
    """Spell the transposed root the way the destination key normally is."""
    root, q, _ = parse_token(tok)
    mode = "minor" if q[:1] in ("-", "h") or q.startswith("o") else "major"
    return prefer_flats((root + semis) % 12, mode)

SCRATCH = REPO / "scratchpad"
UG_CACHE = SCRATCH / "ug_cache"
PLOTS = REPO / "docs" / "plots"
AUDIO = REPO / "docs" / "audio"

_PC_FLAT = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


# --------------------------------------------------------------------------- #
# 1. Ultimate Guitar: fetch + parse
# --------------------------------------------------------------------------- #
_UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
       "(KHTML, like Gecko) Version/17.4 Safari/605.1.15")


def fetch_ug(url: str, cache_name: str | None = None) -> str:
    """Fetch a UG page (one request, cached on disk — be polite)."""
    UG_CACHE.mkdir(parents=True, exist_ok=True)
    if cache_name is None:
        cache_name = re.sub(r"[^a-z0-9]+", "_", url.lower().split("/")[-1])[:80] + ".html"
    path = UG_CACHE / cache_name
    if path.exists():
        return path.read_text(encoding="utf-8", errors="ignore")
    out = subprocess.run(["curl", "-sL", "-A", _UA, url], capture_output=True, text=True)
    if out.returncode != 0 or len(out.stdout) < 2000:
        raise RuntimeError(f"UG fetch failed for {url} (rc={out.returncode}, "
                           f"{len(out.stdout)} bytes)")
    path.write_text(out.stdout, encoding="utf-8")
    return out.stdout


def ug_store(page_html: str) -> dict:
    """`<div class="js-store" data-content="...">` -> store.page.data."""
    m = re.search(r'<div class="js-store" data-content="(.*?)"></div>', page_html, re.S)
    if not m:
        raise RuntimeError("no js-store div — UG layout changed or page is a captcha")
    return json.loads(_html.unescape(m.group(1)))["store"]["page"]["data"]


# --- UG chord spelling -> iReal token understood by local_key.chord_pcs ----- #
# chord_pcs() reads its quality tail with `startswith("m")` == minor, so a raw
# "Cmaj7" would silently become C MINOR.  Every UG suffix is mapped explicitly.
_QUAL_MAP = [
    ("maj7", "^7"), ("maj9", "^7"), ("maj13", "^7"), ("M7", "^7"), ("Δ", "^7"),
    ("m7b5", "h7"), ("m7-5", "h7"), ("ø", "h7"), ("dim7", "o7"), ("dim", "o"),
    ("°7", "o7"), ("°", "o"),
    ("mmaj7", "-^7"), ("mM7", "-^7"),
    ("m11", "-7"), ("m9", "-7"), ("m13", "-7"), ("m7", "-7"), ("m6", "-6"),
    ("min7", "-7"), ("min", "-"), ("m", "-"),
    ("sus4", "sus"), ("sus2", "sus2"), ("sus", "sus"),
    ("add9", ""), ("add2", ""), ("add11", ""), ("add4", ""),
    ("aug", "+"), ("+", "+"),
    ("13", "13"), ("11", "9"), ("9", "9"), ("7", "7"), ("6", "6"), ("5", ""),
]
_ROOT_RE = re.compile(r"^([A-G])([b#]?)(.*)$")


def ug_to_ireal(tok: str) -> str | None:
    """"Cadd9" -> "C", "Cmaj7" -> "C^7", "Bbm7" -> "Bb-7", "C/E" -> "C/E"."""
    tok = tok.strip().replace("♭", "b").replace("♯", "#")
    head, _, bass = tok.partition("/")
    m = _ROOT_RE.match(head)
    if not m:
        return None
    letter, acc, tail = m.groups()
    tail = tail.strip()
    q = ""
    while tail:
        for pat, rep in _QUAL_MAP:
            if tail.startswith(pat):
                # first suffix wins; "add9"/"5" etc. contribute nothing
                if not q:
                    q = rep
                tail = tail[len(pat):]
                break
        else:
            tail = tail[1:]          # unknown char (parens, '*') — skip
    out = letter + acc + q
    if bass:
        bm = _ROOT_RE.match(bass.strip())
        if bm:
            out += "/" + bm.group(1) + bm.group(2)
    return out


@dataclass
class UGChord:
    idx: int
    tok: str            # iReal token
    raw: str            # as written in the tab
    section: str
    block: int          # blank-line-separated block index
    line: int
    weight: float       # relative duration weight
    lyric: str = ""     # lyric text starting under this chord (ASR anchoring)
    nomusic: bool = False   # the tab marked this stretch "(No music)"
    t0: float = float("nan")
    t1: float = float("nan")


_CH_RE = re.compile(r"\[ch\](.*?)\[/ch\]")
_HDR_RE = re.compile(r"^\s*\[([^\]/]+)\]\s*$")
_REPEAT_RE = re.compile(r"[\(\[]?\s*(?:x\s*(\d+)|(\d+)\s*x)\s*[\)\]]?\s*$", re.I)
_NOMUSIC_RE = re.compile(r"\(\s*no\s+music\s*\)", re.I)


def _strip_markup(line: str) -> tuple[str, list[tuple[int, str]]]:
    """Remove [tab]/[ch] markup, returning plain text + (column, chord) list."""
    line = line.replace("[tab]", "").replace("[/tab]", "")
    out, chords, i = [], [], 0
    col = 0
    while i < len(line):
        m = _CH_RE.match(line, i)
        if m:
            chords.append((col, m.group(1)))
            out.append(m.group(1))
            col += len(m.group(1))
            i = m.end()
        else:
            out.append(line[i])
            col += 1
            i += 1
    return "".join(out), chords


def parse_ug_tab(page_html: str) -> tuple[list[UGChord], dict]:
    d = ug_store(page_html)
    tab, tv = d["tab"], d["tab_view"]
    meta = {
        "song": tab.get("song_name"), "artist": tab.get("artist_name"),
        "rating": round(float(tab.get("rating") or 0), 3),
        "votes": int(tab.get("votes") or 0),
        "tonality": (tv.get("meta") or {}).get("tonality"),
        "capo": int((tv.get("meta") or {}).get("capo") or 0),
        "url": tab.get("tab_url"), "tab_id": tab.get("id"),
    }
    content = tv["wiki_tab"]["content"]
    # A capo means the tab is written in SHAPES, not in sounding pitch: This
    # Love's 4.86* tab is written in Am with capo 3 and sounds in Cm.  Every
    # downstream number here (chroma templates, landmarks, our chart) is in
    # sounding pitch, so the capo is applied at parse time and nowhere else.
    capo = meta["capo"]

    chords: list[UGChord] = []
    section, block = "", 0
    lines = content.replace("\r\n", "\n").split("\n")
    prev_blank = True
    pending: list[UGChord] = []     # chords of the line we just read
    # word stream for phase-2 lyric anchoring: every lyric word in tab order,
    # tagged with the chord sounding above it and whether the tab marked its
    # line "(No music)".  Kept separate from UGChord.lyric because a chord can
    # own several lyric lines (a-cappella blocks have no chord line at all).
    wstream: list[dict] = []
    line_cols: list[tuple[int, int]] = []   # (column, chord idx) of last chord line
    nomusic = False

    def flush(lyric_line: str | None):
        """Assign duration weights to `pending` from the lyric line beneath."""
        if not pending:
            return
        cols = [c._col for c in pending]           # type: ignore[attr-defined]
        end = max(len(lyric_line or ""), cols[-1] + 4)
        for k, c in enumerate(pending):
            nxt = cols[k + 1] if k + 1 < len(cols) else end
            span = max(nxt - cols[k], 1)
            # a lyric line makes the span meaningful; a bare chord line is
            # normally one bar per chord regardless of how it was spaced out.
            c.weight = float(span) if lyric_line else 4.0
            if lyric_line:
                c.lyric = lyric_line[cols[k]:nxt].strip()
        pending.clear()

    for raw_line in lines:
        line = raw_line.rstrip()
        if not line.strip():
            flush(None)
            if not prev_blank:
                block += 1
            prev_blank = True
            continue
        prev_blank = False
        if _NOMUSIC_RE.fullmatch(line.strip()):
            nomusic = True                       # marks the lines that FOLLOW
            continue
        h = _HDR_RE.match(line)
        # guard against the markup tokens only — "[Chorus]" IS a section header
        # (an earlier `"ch" not in name[:2]` test silently swallowed every
        # chorus label and put chorus chords in the previous section).
        if h and h.group(1).strip().lower() not in {"ch", "tab"}:
            flush(None)
            section = h.group(1).strip()
            continue
        plain, found = _strip_markup(line)
        if found:
            flush(None)
            line_cols, nomusic = [], False
            rep = _REPEAT_RE.search(plain)
            n_rep = int(rep.group(1) or rep.group(2)) if rep else 1
            n_rep = min(max(n_rep, 1), 8)
            for r in range(n_rep):
                for col, raw in found:
                    tok = ug_to_ireal(raw)
                    if tok is None:
                        continue
                    if capo:
                        tok = transpose_token(tok, capo,
                                              flats=_prefer_flats_tok(tok, capo))
                    c = UGChord(idx=len(chords), tok=tok, raw=raw, section=section,
                                block=block, line=len(chords), weight=4.0)
                    c._col = col              # type: ignore[attr-defined]
                    chords.append(c)
                    if r == n_rep - 1:
                        pending.append(c)
                if r < n_rep - 1:             # repeats get bar-length weights
                    for c in chords[-len(found):]:
                        c.weight = 4.0
            for c in pending:
                line_cols.append((c._col, c.idx))  # type: ignore[attr-defined]
        else:
            # a lyric line dates the chord line above it (if any); a lyric line
            # with no chord line above still belongs to the last chord — that is
            # exactly the a-cappella case, and dropping it would throw away the
            # only timing evidence such a passage has.
            flush(plain)
            for m_ in re.finditer(r"\S+", plain):
                toks = _norm_words(m_.group(0))
                if not toks:
                    continue
                col = m_.start()
                owner = chords[-1].idx if chords else 0
                for cc, ci in line_cols:
                    if cc <= col:
                        owner = ci
                for t in toks:
                    wstream.append({"w": t, "chord": owner, "nomusic": nomusic})
    flush(None)
    for i, c in enumerate(chords):
        c.idx = i
    for w in wstream:
        if w["nomusic"] and 0 <= w["chord"] < len(chords):
            chords[w["chord"]].nomusic = True
    meta["_words"] = wstream
    return chords, meta


# --------------------------------------------------------------------------- #
# 2. Templates + audio features
# --------------------------------------------------------------------------- #
def chord_template(tok: str) -> np.ndarray:
    """24-d C-first template: [0:12] treble chord tones, [12:24] bass emphasis."""
    v = np.zeros(24)
    for pc, w in chord_pcs(tok, include_bass=False).items():
        v[pc % 12] += w
    root, _, bass = parse_token(tok)
    b = bass if bass is not None else root
    v[12 + b % 12] += 2.0
    v[12 + (root + 7) % 12] += 0.4        # the fifth is audible in the bass too
    tr, ba = v[:12], v[12:]
    if tr.sum():
        tr /= np.linalg.norm(tr)
    if ba.sum():
        ba /= np.linalg.norm(ba)
    return v


def load_chroma(slug: str, hop: float = 0.1) -> tuple[np.ndarray, np.ndarray]:
    """NNLS bothchroma -> (F,24) C-first [bass|treble], resampled to `hop` s."""
    from harmonia.models.nnls_features import extract_bothchroma
    arr, times = extract_bothchroma(AUDIO / f"{slug}.m4a")
    arr = np.asarray(arr, dtype=float)
    # index 0 = A: pc p sits at column (p - 9) % 12
    roll = np.array([(p - 9) % 12 for p in range(12)])
    bass, treb = arr[:, :12][:, roll], arr[:, 12:24][:, roll]
    dur = float(times[-1])
    n = int(dur / hop) + 1
    grid = np.arange(n) * hop
    idx = np.clip(np.searchsorted(times, grid), 0, len(times) - 1)
    B, T = bass[idx], treb[idx]
    # log-compress then L2-normalise each frame half (chroma dynamic range is
    # huge; raw magnitudes make loud frames dominate the DTW cost)
    def norm(X):
        X = np.log1p(50.0 * np.maximum(X, 0))
        n_ = np.linalg.norm(X, axis=1, keepdims=True)
        return X / np.maximum(n_, 1e-9)
    return np.hstack([norm(B), norm(T)]), grid


W_TREBLE, W_BASS = 0.72, 0.28


def cost_matrix(chords: list[UGChord], feats: np.ndarray) -> np.ndarray:
    """C[j,t] = 1 - weighted cosine(template_j, frame_t).  Chord-tone based."""
    tpl = np.stack([chord_template(c.tok) for c in chords])           # (N,24)
    simb = feats[:, :12] @ tpl[:, 12:].T                              # (F,N)
    simt = feats[:, 12:] @ tpl[:, :12].T
    return (1.0 - (W_BASS * simb + W_TREBLE * simt)).T                # (N,F)


# --------------------------------------------------------------------------- #
# 3. Segmental Viterbi with duration prior + free head/tail
# --------------------------------------------------------------------------- #
ANCHOR_GAMMA = 3.0     # cost of being 1 s outside an anchor window


@dataclass
class AlignResult:
    bounds: np.ndarray                 # (N+1,) frame indices
    cost: float
    hop: float
    n_frames: int
    anchors_used: int = 0
    notes: list[str] = field(default_factory=list)


def segmental_align(C: np.ndarray, weights: np.ndarray, hop: float,
                    *, beta: float = 0.9, skip_mult: float = 1.0,
                    dmin_s: float = 0.35, dmax_s: float = 40.0,
                    anchors: dict[int, tuple[int, int]] | None = None,
                    scale: float | None = None) -> AlignResult:
    """Cut the audio into N contiguous spans, one per chord, in order.

    ``beta`` weights a log-ratio duration prior against the tab's own relative
    weights.  ``anchors[j] = (lo, hi)`` forces chord j to START inside [lo, hi)
    — the phase-2 lyric constraint.

    The cost is measured RELATIVE to a neutral per-frame charge
    ``c_skip = skip_mult * mean(C)``, which is also what an *uncovered* head or
    tail frame costs.  This is not cosmetic: with a raw (non-centred) cost, the
    total shrinks with the number of frames covered, so the optimum is to cram
    all N chords into the shortest legal span.  That bug put all 119 This Love
    chords into the last 40 s of the song (chord #0 at 162.3 s) before it was
    caught by the landmarks.  Centring makes "cover this frame" break even
    against "skip it" for an average-fitting chord, so only chords that explain
    the audio BETTER than average pay for their airtime.
    """
    N, F = C.shape
    c_skip = skip_mult * float(C.mean())
    C = C - c_skip
    CS = np.zeros((N, F + 1))
    np.cumsum(C, axis=1, out=CS[:, 1:])

    w = np.asarray(weights, dtype=float)
    w = np.maximum(w, 0.25) / w.sum()
    if scale is None:
        scale = F                       # frames per unit weight (tab covers all)
    d_exp = np.maximum(w * scale, 1.0)

    dmin, dmax = max(1, int(dmin_s / hop)), int(dmax_s / hop)
    INF = 1e18
    best = np.full((N + 1, F + 1), INF)
    back = np.zeros((N + 1, F + 1), dtype=np.int32)
    # free begin: the head is charged the same neutral rate as any frame, which
    # after centring is exactly 0 — so an unwritten intro costs nothing, and a
    # written one is still preferred because its chords beat neutral.
    best[0, : F + 1] = 0.0

    for j in range(N):
        row = np.full(F + 1, INF)
        bck = np.zeros(F + 1, dtype=np.int32)
        prev = best[j]
        win = (anchors or {}).get(j)
        if win is not None:
            # SOFT anchors.  Masking the window hard made the constraint set
            # INFEASIBLE on This Love (40 anchors): every path hit INF, argmin
            # returned frame 0 and the aligner emitted an all-zeros alignment
            # while still printing a cost and a verdict.  A quadratic pull is
            # effectively as strong when the anchors agree and degrades
            # gracefully when they do not.
            lo, hi = win
            f = np.arange(F + 1)
            over = np.maximum(0, lo - f) + np.maximum(0, f - hi)
            prev = prev + ANCHOR_GAMMA * (over * hop) ** 2
        dmax_j = min(dmax, F)
        for d in range(dmin, dmax_j + 1):
            pen = beta * (math.log(d / d_exp[j]) ** 2)
            # segment [b-d, b) for chord j
            cand = prev[: F + 1 - d] + (CS[j, d:] - CS[j, : F + 1 - d]) + pen
            tgt = row[d:]
            upd = cand < tgt
            if upd.any():
                idxs = np.nonzero(upd)[0]
                row[idxs + d] = cand[idxs]
                bck[idxs + d] = idxs
        best[j + 1] = row
        back[j + 1] = bck

    end = int(np.argmin(best[N]))      # tail frames are neutral, i.e. free
    if not np.isfinite(best[N, end]) or best[N, end] >= INF / 2:
        raise RuntimeError(
            f"alignment infeasible: {N} chords need >= {N * dmin * hop:.0f}s "
            f"but the audio is {F * hop:.0f}s (or the anchors conflict)")
    bounds = np.zeros(N + 1, dtype=int)
    bounds[N] = end
    for j in range(N, 0, -1):
        bounds[j - 1] = back[j, bounds[j]]
    return AlignResult(bounds=bounds, cost=float(best[N, end]), hop=hop, n_frames=F)


def align(chords: list[UGChord], feats: np.ndarray, hop: float,
          anchors: dict[int, tuple[int, int]] | None = None,
          **kw) -> tuple[AlignResult, np.ndarray]:
    """Two passes: the second re-estimates the tab's time scale from the first."""
    C = cost_matrix(chords, feats)
    w = np.array([c.weight for c in chords])
    r1 = segmental_align(C, w, hop, anchors=anchors, **kw)
    span = max(r1.bounds[-1] - r1.bounds[0], 1)
    r2 = segmental_align(C, w, hop, anchors=anchors, scale=float(span), **kw)
    r = r2 if r2.cost <= r1.cost else r1
    if anchors:
        r.anchors_used = len(anchors)
    for j, c in enumerate(chords):
        c.t0, c.t1 = float(r.bounds[j] * hop), float(r.bounds[j + 1] * hop)
    return r, C


# --------------------------------------------------------------------------- #
# 4. Identifiability audit — is the timing even recoverable from harmony?
# --------------------------------------------------------------------------- #
def support(C: np.ndarray, res: AlignResult) -> np.ndarray:
    """Per-chord audio support: mean cost over its span, centred on neutral.

    Negative = the audio backs this chord better than an average chord would.
    Positive = the aligner placed it somewhere the audio contradicts.  Needed
    because a tab can contain material the RECORDING does not: the 4.83* Close
    to You tab appends an alternate all-C ending, and the DP happily stretched
    it over the real Db outro (197-220 s) with every global number still fine.
    """
    N, F = C.shape
    neutral = float(C.mean())
    b = res.bounds
    return np.array([C[j, b[j]:max(b[j] + 1, b[j + 1])].mean() - neutral
                     for j in range(N)])


def unsupported_spans(chords: list[UGChord], sup: np.ndarray,
                      min_len: float = 6.0) -> list[dict]:
    """Contiguous runs of chords the audio contradicts (support > 0)."""
    out, run = [], []
    for c, s in zip(chords, sup):
        if s > 0:
            run.append(c)
        else:
            if run and run[-1].t1 - run[0].t0 >= min_len:
                out.append({"t0": round(run[0].t0, 1), "t1": round(run[-1].t1, 1),
                            "n": len(run), "from": run[0].idx, "to": run[-1].idx})
            run = []
    if run and run[-1].t1 - run[0].t0 >= min_len:
        out.append({"t0": round(run[0].t0, 1), "t1": round(run[-1].t1, 1),
                    "n": len(run), "from": run[0].idx, "to": run[-1].idx})
    return out


def identifiability(chords: list[UGChord], C: np.ndarray, res: AlignResult) -> dict:
    """Two numbers that say whether to BELIEVE the alignment.

    ``boundary_contrast``: mean cosine distance between consecutive chord
    templates.  If the tab is a one-chord vamp this is ~0 and no chroma method
    on earth can time it.
    ``cost_contrast``: how much worse a *random* monotone segmentation is than
    the fitted one, in units of the random spread.  ~0 means the cost surface is
    flat: the alignment we printed is one of a million equally good ones.
    """
    tpl = np.stack([chord_template(c.tok) for c in chords])
    tn = tpl / np.maximum(np.linalg.norm(tpl, axis=1, keepdims=True), 1e-9)
    if len(chords) > 1:
        bc = float(np.mean(1.0 - np.sum(tn[:-1] * tn[1:], axis=1)))
    else:
        bc = 0.0

    N, F = C.shape
    b = res.bounds
    fit = float(np.mean([C[j, b[j]:max(b[j] + 1, b[j + 1])].mean() for j in range(N)]))
    rng = np.random.default_rng(0)
    lo, hi = int(b[0]), int(b[-1])
    rnd = []
    for _ in range(200):
        cut = np.sort(rng.integers(lo, max(hi, lo + N + 1), size=N - 1))
        bb = np.concatenate([[lo], cut, [hi]])
        rnd.append(np.mean([C[j, bb[j]:max(bb[j] + 1, bb[j + 1])].mean()
                            for j in range(N)]))
    rnd = np.array(rnd)
    cc = float((rnd.mean() - fit) / max(rnd.std(), 1e-9))
    verdict = ("harmony-underdetermined" if (bc < 0.12 or cc < 1.0) else
               "weak" if cc < 2.5 else "ok")
    if verdict == "harmony-underdetermined" and res.anchors_used:
        # the harmony still says nothing — but the timing is no longer free,
        # it is pinned by lyric anchors.  Two different claims; keep them apart.
        verdict = "harmony-underdetermined/ASR-anchored"
    return {"boundary_contrast": round(bc, 4), "cost_contrast": round(cc, 3),
            "fit_cost": round(fit, 4), "random_cost": round(float(rnd.mean()), 4),
            "anchors": int(res.anchors_used), "verdict": verdict}


# --------------------------------------------------------------------------- #
# 5. Our side: the baked payload (for the plot + the disagreement list)
# --------------------------------------------------------------------------- #
def load_payload(slug: str) -> dict | None:
    p = PLOTS / f"inferred_{slug}.html"
    if not p.exists():
        return None
    txt = p.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"const\s+P\s*=\s*", txt)
    if not m:
        return None
    i = txt.index("{", m.end())
    depth, j, instr, esc = 0, i, False, False
    while j < len(txt):
        ch = txt[j]
        if instr:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                instr = False
        else:
            if ch == '"':
                instr = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
        j += 1
    try:
        return json.loads(txt[i:j + 1])
    except Exception:
        return None


def payload_chords(P: dict) -> list[dict]:
    out = []
    for c in P.get("chords", []):
        q = ((c.get("lv") or {}).get("exact") or {}).get("q", "")
        out.append({"t0": float(c["t0"]), "t1": float(c["t1"]),
                    "root": int(c["root"]), "q": q,
                    "nc": bool(c.get("nc")),
                    "name": _PC_FLAT[int(c["root"]) % 12] + (q or "")})
    return out


# --------------------------------------------------------------------------- #
# 6. Phase 2: lyric anchors from Whisper
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[a-z']+")


def _norm_words(s: str) -> list[str]:
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode()
    return _WORD_RE.findall(s.lower())


def transcribe(slug: str, model_name: str = "small") -> list[dict]:
    """Whisper word timestamps, cached to scratchpad/asr_<slug>.json."""
    cache = SCRATCH / f"asr_{slug}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    import whisper
    model = whisper.load_model(model_name)
    # Music is not the domain Whisper's defaults were tuned for: with them,
    # `small` declared the first 32 s of Chain of Fools to be non-speech and
    # emitted 135 words for a 169 s song. Disabling the no-speech/logprob gates
    # and the previous-text conditioning (which lets one bad segment cascade
    # over a repetitive lyric) is what makes it usable here.
    r = model.transcribe(str(AUDIO / f"{slug}.m4a"), language="en",
                         word_timestamps=True, verbose=False,
                         condition_on_previous_text=False,
                         no_speech_threshold=None, logprob_threshold=None,
                         compression_ratio_threshold=None,
                         temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))
    words = []
    for seg in r["segments"]:
        for w in seg.get("words", []):
            tok = _norm_words(w["word"])
            if tok:
                words.append({"w": tok[0], "t": float(w["start"]),
                              "e": float(w["end"])})
    cache.write_text(json.dumps(words))
    return words


def lyric_anchors(chords: list[UGChord], tab_words: list[dict],
                  asr: list[dict], duration: float, *,
                  min_run: int = 3) -> tuple[dict, list[dict], list]:
    """Monotone word alignment tab-lyrics <-> ASR, then one anchor per chord.

    An earlier version demanded an *exact* run of N words unique in the ASR
    stream; on Chain of Fools that produced **0 anchors from 590 ASR words** —
    ASR never reproduces a lyric sheet verbatim (mishearings, ad-libs, repeated
    hooks), so exact+unique is the wrong test.

    Instead: ``difflib.SequenceMatcher`` over the two normalised word streams.
    It is monotone by construction (so anchors can never cross), tolerant of
    insertions/deletions/errors, and its matching *blocks* of >= ``min_run``
    consecutive identical words are precisely the runs worth trusting.  Each
    chord's anchor is the ASR time of the earliest matched word that belongs to
    it; anchors are then forced strictly increasing.
    """
    import difflib
    A = [w["w"] for w in tab_words]
    B = [w["w"] for w in asr]
    sm = difflib.SequenceMatcher(None, A, B, autojunk=False)
    t_of: dict[int, float] = {}
    for i, j, n in sm.get_matching_blocks():
        if n >= min_run:
            for k in range(n):
                t_of[i + k] = asr[j + k]["t"]

    per_chord: dict[int, float] = {}
    for i, w in enumerate(tab_words):
        if i in t_of and w["chord"] not in per_chord:
            per_chord[w["chord"]] = t_of[i]

    cand, last = [], -1e9
    for ci in sorted(per_chord):
        t = per_chord[ci]
        if t <= last:                      # keep the anchor set strictly monotone
            continue
        cand.append((ci, t))
        last = t

    keep = _consistent_chain(cand, duration, len(chords))
    anchors = dict(keep)
    dropped = [(ci, round(t, 1)) for ci, t in cand if ci not in anchors]
    report = [{"chord": ci, "tok": chords[ci].tok, "t": round(t, 2)}
              for ci, t in keep]
    return anchors, report, dropped


RATE_LO, RATE_HI = 0.05, 6.0     # x the song's mean seconds-per-chord


def _consistent_chain(cand: list[tuple[int, float]], duration: float,
                      n_chords: int) -> list[tuple[int, float]]:
    """Keep the longest subset of anchors whose implied pace stays plausible.

    Repeated lyrics are the failure mode: on This Love, `difflib` matched the
    FIRST chorus's words to the LAST chorus's audio, so 11 of 40 anchors were
    +101 s out — monotone, so monotonicity alone did not catch them, but they
    imply 52 s per chord across one gap and 0.02 s per chord across the next.
    So the test is on the *rate*: any pair of consecutive kept anchors must
    imply between 0.05x and 6x the song's mean seconds-per-chord.  Longest
    valid chain by DP (O(n^2), n ~ 40).
    """
    if len(cand) < 2:
        return cand
    mean = max(duration / max(n_chords, 1), 1e-6)
    lo, hi = RATE_LO * mean, RATE_HI * mean
    n = len(cand)
    dp = [1] * n
    back = [-1] * n
    for j in range(n):
        for i in range(j):
            di = cand[j][0] - cand[i][0]
            dt = cand[j][1] - cand[i][1]
            if di <= 0 or dt <= 0:
                continue
            if lo <= dt / di <= hi and dp[i] + 1 > dp[j]:
                dp[j], back[j] = dp[i] + 1, i
    j = int(max(range(n), key=lambda k: dp[k]))
    out = []
    while j >= 0:
        out.append(cand[j])
        j = back[j]
    return out[::-1]


def nomusic_spans(tab_words: list[dict], asr: list[dict]) -> list[dict]:
    """Time the tab's "(No music)" stretches from the ASR word stream.

    These are the a-cappella passages: the tab asserts no instrument plays, so
    they are a free, labelled negative set for chord-vs-no-chord — but only if
    they can be timed, and harmony obviously cannot time them.
    """
    import difflib
    A = [w["w"] for w in tab_words]
    B = [w["w"] for w in asr]
    sm = difflib.SequenceMatcher(None, A, B, autojunk=False)
    t_of: dict[int, float] = {}
    for i, j, n in sm.get_matching_blocks():
        if n >= 2:
            for k in range(n):
                t_of[i + k] = asr[j + k]["t"]
    out, cur = [], None

    def close(c):
        if c and c["times"]:
            out.append({"t0": round(min(c["times"]), 2),
                        "t1": round(max(c["times"]), 2),
                        "n_matched": len(c["times"]),
                        "chord": c["chord"]})

    for i, w in enumerate(tab_words):
        if w["nomusic"]:
            if cur is None:
                cur = {"times": [], "chord": w["chord"]}
            if i in t_of:
                cur["times"].append(t_of[i])
        else:
            close(cur)
            cur = None
    close(cur)
    return out


def anchors_to_windows(anchors: dict[int, float], hop: float, F: int,
                       window: float = 4.0,
                       nm_spans: list[dict] | None = None,
                       n_chords: int = 0) -> dict[int, tuple[int, int]]:
    """Turn anchor times into hard START windows for the DP.

    A word starts *inside* its chord's span, not at its onset, so the chord must
    start no later than the word and not absurdly earlier.

    A "(No music)" span adds a second, stronger constraint: the tab says one
    chord is in force across the whole a-cappella passage, so the NEXT chord
    cannot begin before that passage ends.  Without it the duration prior cuts
    the owning chord short (Chain of Fools: chord 20 ended at 79.2 s while the
    passage it owns runs to 96.0 s).
    """
    out = {}
    for j, t in anchors.items():
        f = int(t / hop)
        out[j] = (max(0, f - int(window / hop)), min(F, f + int(1.0 / hop)))
    for s in (nm_spans or []):
        ci = int(s["chord"])
        f0, f1 = int(s["t0"] / hop), int(s["t1"] / hop)
        lo, hi = out.get(ci, (0, F))
        out[ci] = (lo, min(hi, f0))                  # starts before the passage
        if ci + 1 < n_chords:
            lo2, hi2 = out.get(ci + 1, (0, F))
            out[ci + 1] = (max(lo2, f1), max(hi2, f1 + 1))
    return out


# --------------------------------------------------------------------------- #
# 7. Plot
# --------------------------------------------------------------------------- #
def plot(slug: str, chords: list[UGChord], ours: list[dict], res: AlignResult,
         audit: dict, landmarks: list[dict], out: Path, meta: dict):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    dur = res.n_frames * res.hop
    cmap = plt.get_cmap("hsv")

    def col(tok):
        r, _, _ = parse_token(tok)
        return cmap((r % 12) / 12.0)

    rows = 3
    fig, ax = plt.subplots(figsize=(19, 5.2), dpi=130)
    ax.set_xlim(0, dur)
    ax.set_ylim(0, rows)

    # row 2 (top): UG aligned
    for c in chords:
        if not np.isfinite(c.t0):
            continue
        wid = max(c.t1 - c.t0, 0.05)
        ax.add_patch(Rectangle((c.t0, 2.05), wid, 0.8, facecolor=col(c.tok),
                               edgecolor="white", lw=0.5, alpha=0.85))
        if wid > dur / 90:
            # SOUNDING name (post-capo) — the written shape would not line up
            # with our row and reads as a bug when it is only a capo.
            ax.text(c.t0 + wid / 2, 2.45, c.tok, ha="center", va="center",
                    fontsize=6, rotation=0)
    # row 1: ours
    for c in ours:
        wid = max(c["t1"] - c["t0"], 0.05)
        fc = "0.85" if c["nc"] else col(_PC_FLAT[c["root"]])
        ax.add_patch(Rectangle((c["t0"], 1.05), wid, 0.8, facecolor=fc,
                               edgecolor="white", lw=0.5, alpha=0.85))
        if wid > dur / 90:
            ax.text(c["t0"] + wid / 2, 1.45, c["name"], ha="center", va="center",
                    fontsize=6)
    # row 0: disagreement (root mismatch at 0.5 s resolution)
    step = 0.5
    tt = np.arange(0, dur, step)
    ug_root = np.full(len(tt), -1)
    for c in chords:
        if np.isfinite(c.t0):
            r, _, _ = parse_token(c.tok)
            ug_root[(tt >= c.t0) & (tt < c.t1)] = r
    our_root = np.full(len(tt), -1)
    for c in ours:
        our_root[(tt >= c["t0"]) & (tt < c["t1"])] = -2 if c["nc"] else c["root"]
    both = (ug_root >= 0) & (our_root >= 0)
    agree = both & (ug_root == our_root)
    disag = both & (ug_root != our_root)
    for t, a in zip(tt, agree):
        if a:
            ax.add_patch(Rectangle((t, 0.15), step, 0.55, facecolor="#2e8b57",
                                   lw=0, alpha=0.75))
    for t, dd in zip(tt, disag):
        if dd:
            ax.add_patch(Rectangle((t, 0.15), step, 0.55, facecolor="#c0392b",
                                   lw=0, alpha=0.9, hatch="///"))
    pct = 100.0 * agree.sum() / max(both.sum(), 1)

    for lm in landmarks:
        ax.axvline(lm["target"], color="k", ls="--", lw=1.2, alpha=0.8)
        ax.text(lm["target"], 2.95, lm["name"], fontsize=7, ha="center",
                va="bottom", rotation=0)
        if lm.get("got") is not None:
            ax.axvline(lm["got"], color="#0057b8", ls="-", lw=1.6, alpha=0.9)

    ax.set_yticks([0.42, 1.45, 2.45])
    ax.set_yticklabels(["root agree", "OURS (chart)", f"UG aligned"], fontsize=9)
    ax.set_xlabel("time (s)")
    ax.set_title(
        f"{meta.get('song')} — UG tab {meta.get('tab_id')} "
        f"({meta.get('rating')}★/{meta.get('votes')}, tonality "
        f"{meta.get('tonality')}) vs our chart   |   root agreement {pct:.0f}%   |   "
        f"identifiability: {audit['verdict']} "
        f"(contrast {audit['cost_contrast']}σ, Δtpl {audit['boundary_contrast']})"
        + (f"   |   {res.anchors_used} lyric anchors" if res.anchors_used else ""),
        fontsize=10)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out)
    plt.close(fig)
    return pct


# --------------------------------------------------------------------------- #
# 8. Landmarks (the validation contract — hits AND misses get printed)
# --------------------------------------------------------------------------- #
LANDMARKS = {
    # This Love: the tab is written in Am shapes with capo 3, so its SOUNDING
    # chorus tail is Cm-F-Ab-G.  The landmark is the sounding F (written "D").
    "maroon_5_this_love": [
        {"name": "chorus-tail F #1", "target": 57.9, "tol": 4.0,
         "match": {"nth_tok": ("F", 0)}},
        {"name": "chorus-tail F #2", "target": 108.4, "tol": 4.0,
         "match": {"nth_tok": ("F", 1)}},
        {"name": "chorus-tail F #3", "target": 158.9, "tol": 4.0,
         "match": {"nth_tok": ("F", 2)}},
        # "verse G7s inside 0-40 s" is a RANGE test, not a point test: the
        # sounding-G chords of the intro + verse 1 (written E7/G#) must all
        # land in the first 40 s.
        {"name": "intro+V1 G7s in 0-40s", "target": 20.0, "tol": 20.0,
         "match": {"root_in_sections": (7, ("Intro", "Verse 1"))}},
    ],
    "carpenters_close_to_you": [
        {"name": "Db-section start", "target": 98.0, "tol": 3.0,
         "match": {"first_root": 1}},
    ],
    # target = centre of the brief's ~75-95 s window
    "aretha_franklin_chain_of_fools_official_lyric_video": [
        {"name": "(No music) verse-2 [harmony]", "target": 85.0, "tol": 10.0,
         "match": {"nomusic_chord": True}},
        {"name": "(No music) verse-2 [ASR-timed]", "target": 85.0, "tol": 10.0,
         "match": {"nomusic_span": True}},
    ],
}


def score_landmarks(slug: str, chords: list[UGChord],
                    nm_spans: list[dict] | None = None) -> list[dict]:
    out = []
    lms = LANDMARKS.get(slug, [])
    for k, lm in enumerate(lms):
        m = lm["match"]
        got, which = None, None
        if "nomusic_span" in m:
            # phase 2 only: timed directly off the ASR word stream, so this row
            # does NOT inherit the harmony alignment's (in)validity.
            if nm_spans:
                s = max(nm_spans, key=lambda s: s["n_matched"])
                got = (s["t0"] + s["t1"]) / 2
                which = (f"ASR-timed {s['t0']}-{s['t1']}s, "
                         f"{s['n_matched']} words matched")
            else:
                which = "no ASR (--asr not given, or no match)"
            out.append({"name": lm["name"], "target": lm["target"],
                        "tol": lm["tol"],
                        "got": None if got is None else round(float(got), 2),
                        "which": which,
                        "err": None if got is None else round(float(got - lm["target"]), 2),
                        "hit": got is not None and abs(got - lm["target"]) <= lm["tol"]})
            continue
        if "nomusic_chord" in m:
            hits = [c for c in chords if c.nomusic]
            if hits:
                lo_, hi_ = min(c.t0 for c in hits), max(c.t1 for c in hits)
                got = (lo_ + hi_) / 2
                which = f"{len(hits)} chord(s) span {lo_:.1f}-{hi_:.1f}s"
            else:
                which = "tab has no (No music) marker"
            out.append({"name": lm["name"], "target": lm["target"],
                        "tol": lm["tol"],
                        "got": None if got is None else round(float(got), 2),
                        "which": which,
                        "err": None if got is None else round(float(got - lm["target"]), 2),
                        "hit": got is not None and abs(got - lm["target"]) <= lm["tol"]})
            continue
        if "root_in_sections" in m:
            r_, secs = m["root_in_sections"]
            hits = [c for c in chords
                    if parse_token(c.tok)[0] == r_ and c.section in secs]
            if hits:
                lo_, hi_ = min(c.t0 for c in hits), max(c.t0 for c in hits)
                got = (lo_ + hi_) / 2
                inside = (lo_ >= lm["target"] - lm["tol"]
                          and hi_ <= lm["target"] + lm["tol"])
                which = (f"{len(hits)}x G-rooted in {lo_:.1f}-{hi_:.1f}s"
                         + ("" if inside else "  OUT OF RANGE"))
                out.append({"name": lm["name"], "target": lm["target"],
                            "tol": lm["tol"], "got": round(float(got), 2),
                            "which": which, "err": round(float(got - lm["target"]), 2),
                            "hit": bool(inside)})
                continue
        if "nth_tok" in m:
            tok, n = m["nth_tok"]
            # nth *occasion* the chord appears (consecutive repeats collapse)
            occ, prev = [], None
            for c in chords:
                if c.tok == tok and prev != tok:
                    occ.append(c)
                prev = c.tok
            if n < len(occ):
                got, which = occ[n].t0, f"chord #{occ[n].idx} {occ[n].raw}"
        elif "nth_root" in m:
            r_, n = m["nth_root"]
            occ, prev = [], None
            for c in chords:
                rr = parse_token(c.tok)[0]
                if rr == r_ and prev != rr:
                    occ.append(c)
                prev = rr
            if n < len(occ):
                got, which = occ[n].t0, f"chord #{occ[n].idx} {occ[n].raw}"
        elif "first_root" in m:
            for c in chords:
                r, _, _ = parse_token(c.tok)
                if r == m["first_root"]:
                    got, which = c.t0, f"chord #{c.idx} {c.raw}"
                    break
        elif "lyric" in m:
            for c in chords:
                if m["lyric"] in c.lyric:
                    got, which = (c.t0 + c.t1) / 2, f"chord #{c.idx} {c.raw}"
                    break
        elif m.get("tok"):
            cands = [c for c in chords if c.tok == m["tok"]]
            if cands:
                c = min(cands, key=lambda c: abs(c.t0 - lm["target"]))
                got, which = c.t0, f"chord #{c.idx} {c.raw}"
        rec = {"name": lm["name"], "target": lm["target"], "tol": lm["tol"],
               "got": None if got is None else round(float(got), 2),
               "which": which}
        rec["err"] = None if got is None else round(float(got) - lm["target"], 2)
        rec["hit"] = got is not None and abs(got - lm["target"]) <= lm["tol"]
        out.append(rec)
    return out


# --------------------------------------------------------------------------- #
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("slug")
    ap.add_argument("source", help="UG url or a cached .html path")
    ap.add_argument("--hop", type=float, default=0.1)
    ap.add_argument("--beta", type=float, default=0.9)
    ap.add_argument("--asr", action="store_true", help="phase 2: lyric anchors")
    ap.add_argument("--asr-model", default="small")
    ap.add_argument("--tag", default="")
    a = ap.parse_args(argv)

    src = Path(a.source)
    page = src.read_text(encoding="utf-8", errors="ignore") if src.exists() \
        else fetch_ug(a.source)
    chords, meta = parse_ug_tab(page)
    print(f"[ug] {meta['artist']} — {meta['song']}  {meta['rating']}★/"
          f"{meta['votes']} votes  tonality={meta['tonality']} capo={meta['capo']}")
    print(f"[ug] {len(chords)} chord events, "
          f"{len(set(c.tok for c in chords))} distinct, "
          f"{len(set(c.section for c in chords))} sections")
    if meta["rating"] and meta["rating"] < 4.7:
        print(f"[warn] rating {meta['rating']} < 4.7 — below the trust bar")

    feats, grid = load_chroma(a.slug, hop=a.hop)
    F = len(grid)
    print(f"[audio] {F} frames @ {a.hop}s = {F * a.hop:.1f}s")

    # tonality cross-check: our chroma's tonic vs the tab's declared tonality
    from harmonia.theory.key_profiles import infer_key
    hist = feats[:, 12:].sum(axis=0)
    ki = infer_key(hist)
    ug_ton = (meta.get("tonality") or "")
    ug_root = _ROOT_RE.match(ug_ton)
    ug_pc = (parse_token(ug_ton)[0] if ug_root else None)
    key_ok = (ug_pc is None or ug_pc == ki.tonic % 12)
    print(f"[key] tab says {ug_ton!r}; our chroma says {ki.key_name} "
          f"(conf {ki.confidence:.2f}) -> tonic {'MATCH' if key_ok else 'MISMATCH'}")
    if not key_ok:
        print("[warn] tab tonic != our measured tonic: the tab may be transposed, "
              "or the video pitch-shifted. Alignment below is suspect.")

    anchors_w, anchor_rep, nm_spans = None, [], []
    if a.asr:
        asr = transcribe(a.slug, a.asr_model)
        tab_words = meta.pop("_words", [])
        print(f"[asr] {len(asr)} ASR words vs {len(tab_words)} tab words")
        anc, anchor_rep, dropped = lyric_anchors(chords, tab_words, asr,
                                                 F * a.hop)
        nm_spans = nomusic_spans(tab_words, asr)
        anchors_w = anchors_to_windows(anc, a.hop, F, nm_spans=nm_spans,
                                       n_chords=len(chords))
        print(f"[asr] {len(anc)} lyric anchors kept, {len(dropped)} dropped as "
              f"rate-inconsistent (repeated-lyric mismatches) -> "
              f"{len(anchors_w)} constrained chords of {len(chords)}")
        if dropped:
            print(f"[asr] dropped: {dropped}")
        if nm_spans:
            print(f"[asr] (No music) spans: "
                  + ", ".join(f"{s['t0']}-{s['t1']}s ({s['n_matched']}w)"
                              for s in nm_spans))
    else:
        meta.pop("_words", None)

    res, C = align(chords, feats, a.hop, anchors=anchors_w, beta=a.beta)
    audit = identifiability(chords, C, res)
    sup = support(C, res)
    unsup = unsupported_spans(chords, sup)
    audit["unsupported_frac"] = round(float((sup > 0).mean()), 3)
    print(f"[audit] {audit}")
    if audit["verdict"].startswith("harmony-underdetermined"):
        print("[audit] !! HARMONY CANNOT TIME THIS TAB — the printed alignment is "
              "one of many equally good ones. Use --asr.")

    if unsup:
        print("[support] spans the AUDIO CONTRADICTS (tab material not in this "
              "recording, or a misplacement):")
        for u in unsup:
            print(f"          {u['t0']:>6.1f}-{u['t1']:<6.1f}s  chords "
                  f"{u['from']}-{u['to']} ({u['n']})")

    lms = score_landmarks(a.slug, chords, nm_spans)
    P = load_payload(a.slug)
    ours = payload_chords(P) if P else []
    tag = f"_{a.tag}" if a.tag else ""
    png = SCRATCH / f"ug_align_{a.slug}{tag}.png"
    pct = plot(a.slug, chords, ours, res, audit, lms, png, meta)

    print("\nLANDMARKS")
    print(f"{'name':<32}{'target':>8}{'got':>9}{'err':>8}  hit  which")
    for lm in lms:
        got_s = "-" if lm["got"] is None else f"{lm['got']:.1f}"
        err_s = "-" if lm["err"] is None else f"{lm['err']:+.1f}"
        print(f"{lm['name']:<32}{lm['target']:>8.1f}{got_s:>9}{err_s:>8}"
              f"  {'HIT ' if lm['hit'] else 'MISS'}  {lm['which'] or ''}")
    print(f"\nroot agreement with our chart: {pct:.1f}%   plot: {png}")

    out = {"slug": a.slug, "meta": meta, "audit": audit, "landmarks": lms,
           "root_agreement_pct": round(float(pct), 2),
           "anchors": anchor_rep, "unsupported_spans": unsup,
           "chords": [{"idx": c.idx, "raw": c.raw, "tok": c.tok,
                       "section": c.section, "block": c.block,
                       "t0": round(c.t0, 3), "t1": round(c.t1, 3),
                       "support": round(float(-sup[c.idx]), 4)}
                      for c in chords]}
    jp = SCRATCH / f"ug_align_{a.slug}{tag}.json"
    jp.write_text(json.dumps(out, indent=1))
    print(f"json: {jp}")
    return out


if __name__ == "__main__":
    main()
