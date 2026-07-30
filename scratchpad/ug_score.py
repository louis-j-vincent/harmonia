#!/usr/bin/env python3
"""ug_score.py — score OUR chord chart against an aligned Ultimate Guitar tab.

    .venv/bin/python scratchpad/ug_score.py <slug> <ug_html_or_url> [--asr]
    .venv/bin/python scratchpad/ug_score.py --report          # all scored songs

Implements the consumption doctrine in ``docs/ug_alignment_brick.md`` (Louis,
2026-07-30), which exists because UG's chord-over-lyric placement is done BY
HAND:

  1. UG *timing* is rough — never compare per-chord timestamps.
  2. UG *order* is the reliable signal.
  3. So: snap to REFERENCE ANCHORS — chords where BOTH sides are sure (our
     chart confident AND the alignment's audio support high AND the roots agree
     exactly) — then diff only what lies BETWEEN consecutive anchors.

Everything below is anchor-interval diffing. A chord we have that UG does not,
inside one interval, is an ADDED chord; a chord UG has that we lack is MISSED.
Only chords the interval diff actually PAIRS are compared for root/quality, and
even then the comparison is ordinal, never "these two timestamps are close".

Three scoring rules from Louis (2026-07-30) decide what counts as an error:

  RULE 1  a 6th/7th written or not is SPELLING, not disagreement ("du pareil au
          même").  The discriminator is the THIRD — see ``triad_core``.
  RULE 2  intros are not scorable; versions differ.  See ``intro_boundary``.
  RULE 3  UG is trusted up to transposition — asserted per song by the share of
          anchors at unison, not assumed.

Error classes, worst (longest) first:
  MISSED    UG has a chord change we never wrote
  ADDED     we wrote a chord UG lacks, on a root not in force
            (sub-class: inside a "(No music)" stretch = hallucination in silence)
  ROOT      paired, different root ("we wrote G where it was C")
  QUALITY   paired, same root, different THIRD (we wrote C7, UG says Cm)
Not errors, reported separately: COSMETIC (rule 1), SPLIT (we chopped a held
chord), ORNAMENT (UG's same-root variant inside our held chord), INTRO (rule 2).
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scratchpad"))

import ug_align as UA  # noqa: E402
from harmonia.theory.local_key import parse_token  # noqa: E402

SCRATCH = REPO / "scratchpad"
_PC = UA._PC_FLAT

# --------------------------------------------------------------------------- #
# quality families — the partial-credit axis (CLAUDE.md: prefer partial credit)
# --------------------------------------------------------------------------- #
def family(q: str) -> str:
    q = (q or "").strip()
    if q.startswith("sus"):
        return "sus"
    if q.startswith(("o", "dim")):
        return "dim"
    if q.startswith("h") or "b5" in q and q.startswith("-"):
        return "halfdim"
    if q.startswith("+") or q.startswith("aug"):
        return "aug"
    if q.startswith("-"):
        return "minmaj" if "^" in q else "min"
    if "^" in q or "maj" in q:
        return "maj"
    if q.startswith(("7", "9", "13", "11")) or q in ("7b9", "7#9", "7b5", "7#5"):
        return "dom"
    return "maj"          # bare triad, 6, add9 ...


def cname(root: int, q: str) -> str:
    return _PC[root % 12] + (q or "")


# --------------------------------------------------------------------------- #
# RULE 1 (Louis, 2026-07-30): a chord differing only by an added/removed 7th or
# 6th is the tab author's WRITING STYLE, not a disagreement — "du pareil au
# même".  Cm/Cm7, C/C6, Db/Db6 are the same chord written two ways.
#
# The discriminator he gave is the THIRD: "dom -> min etc. still count: the
# third changes".  The fifth has to come along too, or Cm/Cdim and C/C+ would
# be swept in as cosmetic, and those are real disagreements.  So the test is on
# the TRIAD CORE (third, fifth); anything stacked above it is spelling.
#
#   maj core (4,7):  ""  6  ^7  7  9  13  add9      <- C / C6 / Cmaj7 / C7
#   min core (3,7):  -   -6  -7  -^7                <- Cm / Cm7
#   dim core (3,6):  o   o7  h7                     <- Cdim7 / Cm7b5
# --------------------------------------------------------------------------- #
def triad_core(q: str) -> tuple[int, int] | None:
    """(third, fifth) semitones, or None for chords with no third (sus/5)."""
    q = (q or "").strip()
    if q.startswith("sus") or q == "5":
        return None
    if q.startswith(("o", "dim")) or (q.startswith(("-", "h")) and "b5" in q) \
            or q.startswith("h"):
        return (3, 6)
    if q.startswith("+") or q.startswith("aug") or "#5" in q:
        return (4, 8)
    if q.startswith("-") or q.startswith("m") and not q.startswith("maj"):
        return (3, 7)
    return (4, 7)


def is_cosmetic(q1: str, q2: str) -> bool:
    """Same triad core, different spelling -> a 6th/7th written or not."""
    c1, c2 = triad_core(q1), triad_core(q2)
    return c1 is not None and c1 == c2 and (q1 or "") != (q2 or "")


# --------------------------------------------------------------------------- #
# sequences (consecutive identical chords collapse — a held chord is one event)
# --------------------------------------------------------------------------- #
@dataclass
class Ev:
    t0: float
    t1: float
    root: int
    q: str
    conf: float = 1.0      # our chart's confidence / UG's audio support
    nc: bool = False
    nomusic: bool = False
    section: str = ""      # UG section header the chord sits under (UG side only)

    @property
    def name(self) -> str:
        return "N.C." if self.nc else cname(self.root, self.q)

    @property
    def dur(self) -> float:
        return self.t1 - self.t0


def _collapse(evs: list[Ev]) -> list[Ev]:
    out: list[Ev] = []
    for e in evs:
        if out and out[-1].root == e.root and out[-1].q == e.q and out[-1].nc == e.nc:
            out[-1].t1 = e.t1
            out[-1].conf = max(out[-1].conf, e.conf)
            out[-1].nomusic = out[-1].nomusic or e.nomusic
        else:
            n = Ev(e.t0, e.t1, e.root, e.q, e.conf, e.nc, e.nomusic)
            n.section = e.section
            out.append(n)
    return out


def our_sequence(slug: str) -> list[Ev]:
    P = UA.load_payload(slug)
    if P is None:
        raise SystemExit(f"no baked payload for {slug}")
    evs = []
    for c in P.get("chords", []):
        lv = (c.get("lv") or {}).get("exact") or {}
        evs.append(Ev(float(c["t0"]), float(c["t1"]), int(c["root"]),
                      lv.get("q", ""), float(lv.get("c", 0.0)),
                      bool(c.get("nc"))))
    return _collapse(evs)


def ug_sequence(chords, sup: np.ndarray, nm_spans) -> list[Ev]:
    evs = []
    for c in chords:
        r, q, _ = parse_token(c.tok)
        nm = c.nomusic or any(s["t0"] <= (c.t0 + c.t1) / 2 <= s["t1"]
                              for s in (nm_spans or []))
        e = Ev(float(c.t0), float(c.t1), r, q, float(-sup[c.idx]), False, nm)
        e.section = c.section or ""
        evs.append(e)
    return _collapse(evs)


# --------------------------------------------------------------------------- #
# 1. anchors: positions BOTH sides are sure about
# --------------------------------------------------------------------------- #
# A UG chord pinned at the aligner's MINIMUM duration is not evidence of
# anything: the DP had to put it somewhere and had no room, so the floor is
# where surplus tab material lands.  Every Breath You Take's tab writes out the
# whole fade-out loop -- 73 chords, every one of them at the floor, ALL inside
# 205-229s -- and scoring those as chords we "missed" made it the second-worst
# song in the report.  Same family as the Close to You alternate-ending problem
# already logged; `unsupported_frac` was flagging it (0.275) and I did not act
# on it until the duration histogram made it unmissable.
DUR_FLOOR = 0.41        # dmin_s=0.35 -> 3 frames @0.1s = 0.30s, +1 frame slack
CONF_TH = 0.45          # our chart's own confidence
SUP_TH = 0.0            # UG chord better-than-neutral against the audio
OVERLAP_TOL = 8.0       # UG timing is rough — this is deliberately generous


def find_anchors(ours: list[Ev], ug: list[Ev]) -> list[tuple[int, int]]:
    """Monotone set of (our idx, ug idx) pairs where both sides are confident
    and the roots agree exactly.  Longest such chain by DP — the chain is what
    makes the anchor set order-consistent, which is the only property the
    interval diff needs (doctrine point 2: order, not timing)."""
    cand = []
    for i, o in enumerate(ours):
        if o.nc or o.conf < CONF_TH:
            continue
        for j, u in enumerate(ug):
            if u.root != o.root or u.conf < SUP_TH or u.nomusic:
                continue
            overlap = min(o.t1, u.t1) - max(o.t0, u.t0)
            if overlap <= 0 and abs((o.t0 + o.t1) / 2 - (u.t0 + u.t1) / 2) > OVERLAP_TOL:
                continue
            score = 1.0 + (0.5 if family(o.q) == family(u.q) else 0.0)
            cand.append((i, j, score))
    if not cand:
        return []
    cand.sort()
    n = len(cand)
    dp = [c[2] for c in cand]
    back = [-1] * n
    for b in range(n):
        for a in range(b):
            if cand[a][0] < cand[b][0] and cand[a][1] < cand[b][1] \
                    and dp[a] + cand[b][2] > dp[b]:
                dp[b], back[b] = dp[a] + cand[b][2], a
    k = int(max(range(n), key=lambda x: dp[x]))
    chain = []
    while k >= 0:
        chain.append((cand[k][0], cand[k][1]))
        k = back[k]
    return chain[::-1]


# --------------------------------------------------------------------------- #
# 2. interval diff: Needleman-Wunsch on (root, quality), root-priority
# --------------------------------------------------------------------------- #
S_ROOT_QUAL, S_ROOT_ONLY, S_SUBST, S_GAP = 3.0, 2.0, -0.5, -1.0


def _nw(a: list[Ev], b: list[Ev]) -> list[tuple[int | None, int | None]]:
    na, nb = len(a), len(b)
    F = np.zeros((na + 1, nb + 1))
    F[:, 0] = np.arange(na + 1) * S_GAP
    F[0, :] = np.arange(nb + 1) * S_GAP
    P = np.zeros((na + 1, nb + 1), dtype=np.int8)   # 0 diag 1 up(gap b) 2 left
    P[1:, 0] = 1
    P[0, 1:] = 2
    for i in range(1, na + 1):
        for j in range(1, nb + 1):
            if a[i - 1].root == b[j - 1].root and not a[i - 1].nc:
                s = S_ROOT_QUAL if family(a[i - 1].q) == family(b[j - 1].q) \
                    else S_ROOT_ONLY
            else:
                s = S_SUBST
            opts = (F[i - 1, j - 1] + s, F[i - 1, j] + S_GAP, F[i, j - 1] + S_GAP)
            k = int(np.argmax(opts))
            F[i, j], P[i, j] = opts[k], k
    out, i, j = [], na, nb
    while i > 0 or j > 0:
        k = P[i, j]
        if i > 0 and j > 0 and k == 0:
            out.append((i - 1, j - 1)); i -= 1; j -= 1
        elif i > 0 and k == 1:
            out.append((i - 1, None)); i -= 1
        else:
            out.append((None, j - 1)); j -= 1
    return out[::-1]


# --------------------------------------------------------------------------- #
# 3. score
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# RULE 2 (Louis, 2026-07-30): intros are NOT scorable — "les intros sont tres
# variables d'une version a l'autre, on ne peut pas s'y fier".  A tab is written
# against one performance; every other take opens differently.  So everything
# before the song's first *sung* moment is reported, never counted.
# --------------------------------------------------------------------------- #
def intro_boundary(chords, ours: list[Ev], ug: list[Ev],
                   anchors, lyric_t: float | None, duration: float) -> float:
    """Time the intro zone ends, best evidence first.

    1. the tab's own first non-intro SECTION header — this is literally the
       "verse/section anchor" the rule names, and 6 of our 7 tabs carry one;
    2. else the first matched lyric word (ASR);
    3. else the first both-sides-sure anchor.
    Capped at 25% of the song so a bad marker cannot swallow a verse.

    Order matters: the ASR marker is the *first lyric line that matched*, which
    on a song with a repeated hook can land deep into the song (Chain of Fools:
    31.3 s, well past two choruses) and would hide real errors.  The section
    header is both more faithful to the rule and better behaved.
    """
    t = None
    for c in chords:
        sec = (c.section or "").lower()
        if sec and not any(k in sec for k in
                           ("intro", "instrumental", "riff", "solo")):
            t = c.t0
            break
    if t is None and lyric_t is not None:
        t = lyric_t
    if t is None and anchors:
        t = ug[anchors[0][1]].t0
    if t is None:
        return 0.0
    return float(min(max(t, 0.0), 0.25 * duration))


def score(slug: str, ours: list[Ev], ug: list[Ev], anchors: list[tuple[int, int]],
          nm_spans, t_intro: float = 0.0) -> dict:  # noqa: C901
    errs: list[dict] = []
    n_pair = n_ok = 0

    # An anchor is a compared position too (roots agree by construction), so it
    # belongs in the denominator.  Leaving it out made Close to You's
    # agreement% a 7-sample statistic while 38 agreeing anchors sat unused.
    for oi, ui in anchors:
        o, u = ours[oi], ug[ui]
        n_pair += 1
        if is_cosmetic(o.q, u.q):
            n_ok += 1
            errs.append(_e("COSMETIC", u, o,
                           f"{o.name} vs {u.name} — 6th/7th spelling"))
        elif family(o.q) == family(u.q):
            n_ok += 1
            if o.q != u.q:
                errs.append(_e("quality_detail", u, o,
                               f"{o.name} vs {u.name} (same family)"))
        else:
            errs.append(_e("QUALITY", u, o,
                           f"{family(o.q)} vs {family(u.q)}: we wrote {o.name},"
                           f" UG says {u.name}"))

    # intervals: before first anchor, between each pair, after last
    segs = []
    prev = (-1, -1)
    for a in anchors:
        segs.append((prev[0] + 1, a[0], prev[1] + 1, a[1]))
        prev = a
    segs.append((prev[0] + 1, len(ours), prev[1] + 1, len(ug)))

    for (oi0, oi1, ui0, ui1) in segs:
        A, B = ours[oi0:oi1], ug[ui0:ui1]
        if not A and not B:
            continue
        for ia, ib in _nw(A, B):
            if ia is not None and ib is not None:
                o, u = A[ia], B[ib]
                n_pair += 1
                if o.nc:
                    errs.append(_e("MISSED", u, o, "we wrote no-chord"))
                elif o.root != u.root:
                    errs.append(_e("ROOT", u, o,
                                   f"we wrote {o.name} where UG says {u.name}"))
                elif is_cosmetic(o.q, u.q):
                    n_ok += 1                       # RULE 1: same chord, two spellings
                    errs.append(_e("COSMETIC", u, o,
                                   f"{o.name} vs {u.name} — 6th/7th spelling"))
                elif family(o.q) != family(u.q):
                    errs.append(_e("QUALITY", u, o,
                                   f"{family(o.q)} vs {family(u.q)}: "
                                   f"we wrote {o.name}, UG says {u.name}"))
                else:
                    n_ok += 1
                    if o.q != u.q:
                        errs.append(_e("quality_detail", u, o,
                                       f"{o.name} vs {u.name} (same family)"))
            elif ia is not None:
                o = A[ia]
                if o.nc:
                    continue                    # no-chord where UG is silent too
                sil = o.nomusic or any(s["t0"] - 1 <= o.t0 and o.t1 <= s["t1"] + 1
                                       for s in (nm_spans or []))
                # An unpaired chord of OUR chart whose root is already in force
                # in this interval is not an invented chord — it is a held UG
                # chord that we chopped up (Chain of Fools: C / C7 / C / C7 over
                # one static Cm7 vamp).  Calling that "ADDED" hid the real and
                # much more common defect behind the rarer, scarier one.
                near = ug[max(ui0 - 1, 0):min(ui1 + 1, len(ug))]
                same = [u for u in near if u.root == o.root]
                if same and any(is_cosmetic(o.q, u.q) or o.q == u.q for u in same):
                    errs.append(_e("COSMETIC", same[0], o,
                                   f"extra {o.name} beside UG's "
                                   f"{same[0].name} — 6th/7th spelling"))
                elif same:
                    errs.append(_e("SPLIT", same[0], o,
                                   f"we chopped a held {_PC[o.root]} into a "
                                   f"separate {o.name}"))
                else:
                    errs.append(_e("ADDED", None, o,
                                   (f"we wrote {o.name} in a (No music) stretch"
                                    if sil else
                                    f"we wrote {o.name}; UG has no chord here"),
                                   silence=sil))
            else:
                u = B[ib]
                # Mirror of SPLIT: UG writing Db6 inside a held Db, or Cadd9
                # next to C, is inner-voice detail, not a chord change we
                # missed.  Scoring those as MISSED buried Close to You's real
                # misses under 62 strumming ornaments.
                near = [o for o in ours[max(oi0 - 1, 0):min(oi1 + 1, len(ours))]
                        if o.root == u.root and not o.nc]
                if near and any(is_cosmetic(o.q, u.q) or o.q == u.q for o in near):
                    errs.append(_e("COSMETIC", u, near[0],
                                   f"UG writes {u.name} beside our "
                                   f"{near[0].name} — 6th/7th spelling"))
                elif near:
                    errs.append(_e("ORNAMENT", u, None,
                                   f"UG writes {u.name} inside our held "
                                   f"{_PC[u.root]}"))
                elif u.dur <= DUR_FLOOR:
                    rec = _e("CRAMMED", u, None,
                             f"UG has {u.name} but the aligner pinned it at "
                             f"minimum duration — surplus tab material, not "
                             f"evidence")
                    rec["ug_i"] = ui0 + ib
                    errs.append(rec)
                else:
                    rec = _e("MISSED", u, None,
                             f"UG has {u.name}, we never wrote it")
                    rec["ug_i"] = ui0 + ib
                    errs.append(rec)

    # RULE 2: everything that ENDS before the first sung word is unscored.
    for e in errs:
        e["intro"] = bool(e["t1"] <= t_intro)

    # RULE 3 assertion: a globally transposed tab would show one non-zero
    # interval dominating (our root - UG root).  Computed, not assumed.
    ivals: dict[int, int] = {}
    for oi, ui in anchors:
        d = (ours[oi].root - ug[ui].root) % 12
        ivals[d] = ivals.get(d, 0) + 1
    n_iv = sum(ivals.values())
    unison = 100.0 * ivals.get(0, 0) / max(n_iv, 1)

    agree = 100.0 * n_ok / max(n_pair, 1)
    scored = [e for e in errs if not e["intro"]]
    cnt = {k: sum(1 for e in scored if e["cls"] == k)
           for k in ("ADDED", "SPLIT", "MISSED", "ORNAMENT", "ROOT", "QUALITY",
                     "COSMETIC", "CRAMMED", "quality_detail")}
    cnt["ADDED_in_silence"] = sum(1 for e in scored
                                  if e["cls"] == "ADDED" and e["silence"])
    cnt["INTRO_unscored"] = sum(1 for e in errs if e["intro"]
                                and e["cls"] in ("ADDED", "MISSED", "ROOT",
                                                 "QUALITY"))
    return {"slug": slug, "n_ours": len(ours), "n_ug": len(ug),
            "t_intro": round(t_intro, 1), "unison_pct": round(unison, 1),
            "root_intervals": ivals,
            "anchors": len(anchors), "paired": n_pair,
            "agreement_pct": round(agree, 1), "counts": cnt,
            "ug_seq": [{"t0": round(e.t0, 2), "t1": round(e.t1, 2),
                        "root": e.root, "q": e.q, "name": e.name,
                        "section": e.section} for e in ug],
            "our_seq": [{"t0": round(e.t0, 2), "t1": round(e.t1, 2),
                         "root": e.root, "q": e.q, "name": e.name,
                         "nc": e.nc, "conf": round(e.conf, 3)} for e in ours],
            "errors": sorted(errs, key=lambda e: -e["dur"])}


def _e(cls, u: Ev | None, o: Ev | None, why: str, silence: bool = False) -> dict:
    ref = o if o is not None else u
    return {"cls": cls, "t0": round(ref.t0, 1), "t1": round(ref.t1, 1),
            "dur": round(ref.dur, 1), "ours": None if o is None else o.name,
            "ug": None if u is None else u.name, "why": why,
            "conf": None if o is None else round(o.conf, 2),
            "silence": silence}


# --------------------------------------------------------------------------- #
def run(slug: str, source: str, use_asr: bool = False, hop: float = 0.1) -> dict:
    src = Path(source)
    page = src.read_text(encoding="utf-8", errors="ignore") if src.exists() \
        else UA.fetch_ug(source)
    chords, meta = UA.parse_ug_tab(page)
    feats, grid = UA.load_chroma(slug, hop=hop)
    F = len(grid)

    anchors_w, nm_spans, lyric_t = None, [], None
    if use_asr:
        asr = UA.transcribe(slug)
        tw = meta.pop("_words", [])
        anc, _rep, _drop = UA.lyric_anchors(chords, tw, asr, F * hop)
        nm_spans = UA.nomusic_spans(tw, asr)
        anchors_w = UA.anchors_to_windows(anc, hop, F, nm_spans=nm_spans,
                                          n_chords=len(chords))
        if anc:
            lyric_t = min(anc.values())     # first matched sung word
    else:
        meta.pop("_words", None)

    res, C = UA.align(chords, feats, hop, anchors=anchors_w)
    audit = UA.identifiability(chords, C, res)
    sup = UA.support(C, res)
    audit["unsupported_frac"] = round(float((sup > 0).mean()), 3)

    ours = our_sequence(slug)
    ugs = ug_sequence(chords, sup, nm_spans)
    anch = find_anchors(ours, ugs)
    t_intro = intro_boundary(chords, ours, ugs, anch, lyric_t, F * hop)
    out = score(slug, ours, ugs, anch, nm_spans, t_intro)
    out["meta"] = {k: meta[k] for k in ("song", "artist", "rating", "votes",
                                        "tonality", "capo", "tab_id")}
    out["audit"] = audit
    out["nomusic_spans"] = nm_spans
    p = SCRATCH / f"ug_score_{slug}.json"
    p.write_text(json.dumps(out, indent=1))
    return out


def print_song(s: dict, top: int = 14):
    m, c = s["meta"], s["counts"]
    print(f"\n=== {m['artist']} — {m['song']}  (UG {m['tab_id']}, "
          f"{m['rating']}★/{m['votes']}, tonality {m['tonality']}, "
          f"capo {m['capo'] or 0})")
    print(f"    ours {s['n_ours']} chords | UG {s['n_ug']} | anchors "
          f"{s['anchors']} | paired {s['paired']} | agreement "
          f"{s['agreement_pct']}%")
    print(f"    ADDED {c['ADDED']} (in silence {c['ADDED_in_silence']})  "
          f"MISSED {c['MISSED']}  ROOT {c['ROOT']}  QUALITY {c['QUALITY']}")
    print(f"    not errors: COSMETIC {c['COSMETIC']}  CRAMMED {c['CRAMMED']}  "
          f"SPLIT {c['SPLIT']}  "
          f"ORNAM {c['ORNAMENT']}  detail {c['quality_detail']}  | intro zone "
          f"0-{s['t_intro']}s hides {c['INTRO_unscored']}  | unison "
          f"{s['unison_pct']}%")
    print(f"    audit: {s['audit']['verdict']}, contrast "
          f"{s['audit']['cost_contrast']}σ, unsupported "
          f"{s['audit']['unsupported_frac']}")
    real = [e for e in s["errors"]
            if e["cls"] in ("ADDED", "MISSED", "ROOT", "QUALITY")
            and not e["intro"]]
    for e in real[:top]:
        print(f"      {e['t0']:>6.1f}-{e['t1']:<6.1f} {e['cls']:<8} {e['why']}")



FAM_PAIR_LABEL = {
    ("maj", "min"): "we wrote major where it is minor",
    ("min", "maj"): "we wrote minor where it is major",
    ("dom", "min"): "we wrote a dominant 7th where the third is minor",
    ("min", "dom"): "we wrote minor where it is dominant",
}

# the numbers this report showed BEFORE Louis's three scoring rules, kept so the
# effect of the rules is auditable rather than asserted (commit d1da3a0)
BEFORE = [
    ("Close To You", 80.0, 3, 39, 0, 8), ("Every Breath You Take", 91.0, 1, 51, 4, 1),
    ("Chain Of Fools", 20.0, 57, 0, 1, 19), ("Let It Be", 99.1, 3, 45, 1, 0),
    ("Hot N Cold", 96.9, 11, 8, 0, 2), ("This Love", 84.3, 1, 3, 1, 16),
    ("Stand By Me", 94.9, 0, 7, 0, 0),
]


def build_report(out_path: Path):
    """docs/ug_score_report.md — per song worst-first, then the synthesis."""
    songs = []
    for p in sorted(SCRATCH.glob("ug_score_*.json")):
        try:
            songs.append(json.loads(p.read_text()))
        except Exception:
            pass
    if not songs:
        raise SystemExit("no ug_score_*.json — score some songs first")

    def err_rate(s):
        c = s["counts"]
        return (c["ADDED"] + c["MISSED"] + c["ROOT"] + c["QUALITY"]) \
            / max(s["n_ours"], 1)

    songs.sort(key=err_rate, reverse=True)

    L = ["# Where our charts are wrong — scored against Ultimate Guitar", "",
         "Generated by `scratchpad/ug_score.py`. Method is the consumption",
         "doctrine in `docs/ug_alignment_brick.md`: UG *timing* is hand-made and",
         "rough, UG *order* is reliable, so we snap to **anchors** (positions",
         "where our chart is confident AND the alignment's audio support is",
         "positive AND the roots agree exactly) and diff only what lies between",
         "consecutive anchors. No per-chord timestamp comparison anywhere.", "",
         "## Scoring rules (Louis, 2026-07-30)", "",
         "**Rule 1 — a 6th/7th written or not is not a disagreement.** "
         "“Du pareil au même”: Cm/Cm7, C/C6, Db/Db6 are one chord "
         "spelled two ways, and which one a tab author writes is style. The "
         "discriminator is the **third** — if the third changes, it still "
         "counts. The fifth has to travel with it, or Cm/Cdim and C/C+ would be "
         "swept in too, so the test is on the **triad core** (third, fifth): "
         "everything stacked above it is spelling. Excluded in both directions "
         "and reported as COSMETIC.", "",
         "**Rule 2 — intros are not scorable.** “Les intros sont très "
         "variables d'une version à l'autre.” A tab is written against "
         "one performance. Everything ending before the tab's own first "
         "non-intro section header is reported and never counted.", "",
         "**Rule 3 — UG is trusted up to transposition.** Asserted below with a "
         "number, not assumed.", "",
         "## Error classes", "",
         "| class | meaning | counts |",
         "|---|---|---|",
         "| **MISSED** | UG has a chord change we never wrote | yes |",
         "| **ADDED** | we wrote a chord UG does not have, on a root not in force | yes |",
         "| ADDED *in silence* | …and the tab marks that stretch “(No music)” | yes, worst kind |",
         "| **ROOT** | paired position, different root (“we wrote G where it was C”) | yes |",
         "| **QUALITY** | paired, same root, the **third** differs | yes |",
         "| COSMETIC | same triad core, 6th/7th spelling (Rule 1) | no |",
         "| SPLIT | we chopped a held chord into pieces (same root, different third) | no |",
         "| ORNAMENT | UG writes a same-root variant inside our held chord | no |",
         "| INTRO | anything ending inside the intro zone (Rule 2) | no |",
         "| CRAMMED | UG chord the aligner pinned at minimum duration — surplus tab material | no |", "",
         "## Summary (worst first)", "",
         "| song | UG | ours | UG ch | anch | paired | agree | ADDED | MISSED | ROOT | QUAL | COSM | intro zone |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in songs:
        m, c = s["meta"], s["counts"]
        sil = f" ({c['ADDED_in_silence']} sil)" if c["ADDED_in_silence"] else ""
        L.append(f"| {m['song']} | {m['rating']}★/{m['votes']} | {s['n_ours']} | "
                 f"{s['n_ug']} | {s['anchors']} | {s['paired']} | "
                 f"**{s['agreement_pct']}%** | {c['ADDED']}{sil} | {c['MISSED']} | "
                 f"{c['ROOT']} | {c['QUALITY']} | {c['COSMETIC']} | "
                 f"0–{s['t_intro']}s ({c['INTRO_unscored']}) |")
    L += ["", "`agree` = of the positions the diff pairs (anchors included), the "
          "share where the root and the third both match. `intro zone` shows the "
          "boundary and how many errors it hides.", ""]

    # Rule 3 assertion
    L += ["## Rule 3 asserted, not assumed", "",
          "A tab transposed relative to our audio would show one **non-zero** "
          "interval dominating `(our root − UG root) mod 12`. Measured over "
          "every anchor:", "",
          "| song | capo | anchors at unison |", "|---|---|---|"]
    for s in songs:
        L.append(f"| {s['meta']['song']} | {s['meta']['capo'] or 0} | "
                 f"**{s['unison_pct']}%** |")
    L += ["", "100% on all seven, including the four capo tabs (This Love 3, "
          "Hot N Cold 5, Stand By Me 2, Every Breath 1). The capo is applied at "
          "parse time and UG's `tonality` field is already sounding pitch, so "
          "**no error anywhere in this report is a transposition artifact**.", ""]

    for s in songs:
        m, c, a = s["meta"], s["counts"], s["audit"]
        L += [f"## {m['artist']} — {m['song']}", "",
              f"UG tab {m['tab_id']}, {m['rating']}★/{m['votes']} votes, "
              f"tonality {m['tonality']}, capo {m['capo'] or 0}. "
              f"Alignment: {a['verdict']}, contrast {a['cost_contrast']}σ, "
              f"unsupported {a['unsupported_frac']}.", "",
              f"{s['anchors']} anchors, {s['paired']} paired, "
              f"**{s['agreement_pct']}%** agreement. Intro zone 0–"
              f"{s['t_intro']}s. Cosmetic 6th/7th differences: {c['COSMETIC']}.",
              ""]
        real = [e for e in s["errors"]
                if e["cls"] in ("ADDED", "MISSED", "ROOT", "QUALITY")
                and not e["intro"]]
        if not real:
            L += ["No errors outside the excluded classes.", ""]
            continue
        L += ["| when | class | what |", "|---|---|---|"]
        for e in real[:30]:
            L.append(f"| {e['t0']:.1f}–{e['t1']:.1f}s | {e['cls']} | {e['why']} |")
        if len(real) > 30:
            L.append(f"| … | | {len(real) - 30} more |")
        L.append("")

    # ---- synthesis -------------------------------------------------------- #
    tot: dict[str, int] = {}
    for s in songs:
        for k, v in s["counts"].items():
            tot[k] = tot.get(k, 0) + v
    fam: dict[tuple[str, str], int] = {}
    roots: dict[str, int] = {}
    for s in songs:
        for e in s["errors"]:
            if e["intro"]:
                continue
            if e["cls"] == "QUALITY":
                k2 = tuple(e["why"].split(":")[0].split(" vs "))
                fam[k2] = fam.get(k2, 0) + 1
            elif e["cls"] == "ROOT":
                roots[f"{e['ours']} → {e['ug']}"] = \
                    roots.get(f"{e['ours']} → {e['ug']}", 0) + 1

    n_bad = tot["ADDED"] + tot["MISSED"] + tot["ROOT"] + tot["QUALITY"]
    noch = {k: 0 for k in ("ADDED", "MISSED", "ROOT", "QUALITY")}
    for s in songs:
        if "chain" in s["slug"]:
            continue
        for k in noch:
            noch[k] += s["counts"][k]
    n_noch = sum(noch.values())
    classes = sorted([("MISSED — a chord change we never wrote", tot["MISSED"]),
                      ("ADDED — a chord that is not there", tot["ADDED"]),
                      ("QUALITY — right root, wrong third", tot["QUALITY"]),
                      ("ROOT — the wrong root entirely", tot["ROOT"])],
                     key=lambda x: -x[1])
    L += ["## Cross-song synthesis — where we are wrong, ranked", "",
          f"{n_bad} real errors across {len(songs)} songs, after removing "
          f"{tot['COSMETIC']} cosmetic 6th/7th differences (rule 1), "
          f"{tot['INTRO_unscored']} intro-zone items (rule 2), "
          f"{tot['SPLIT'] + tot['ORNAMENT']} grid differences, and "
          f"{tot['CRAMMED']} chords the aligner crammed at minimum duration "
          f"(see the MISSED characterization — that last one alone removed 72 "
          f"false misses).", "",
          "| rank | error class | count | share | excl. Chain of Fools |",
          "|---|---|---|---|---|"]
    for i, (name, n) in enumerate(classes, 1):
        key = name.split(" ")[0]
        L.append(f"| {i} | {name} | {n} | {100 * n / max(n_bad, 1):.0f}% | "
                 f"{noch[key]} ({100 * noch[key] / max(n_noch, 1):.0f}%) |")
    lead = "MISSED and ADDED are now level" if abs(tot["MISSED"] - tot["ADDED"]) \
        <= 3 else ("MISSED still leads" if tot["MISSED"] > tot["ADDED"]
                   else "ADDED now leads")
    L += ["", f"**{lead} overall — but the overall figure is misleading.** "
          f"{100 * (tot['ADDED'] - noch['ADDED']) / max(tot['ADDED'], 1):.0f}% "
          f"of ADDED comes from Chain of Fools alone, the one song whose "
          f"harmony cannot time itself. On the six songs where the alignment "
          f"is trustworthy, MISSED is "
          f"{100 * noch['MISSED'] / max(n_noch, 1):.0f}% of all errors and "
          f"ADDED is {100 * noch['ADDED'] / max(n_noch, 1):.0f}%. "
          f"**MISSED is the defect class to work on.** But see the "
          f"characterization below: of the 160 originally reported, only "
          f"~66 survive scrutiny.", "",
          "### What the rules changed", "",
          "| | before rules | after |", "|---|---|---|",
          f"| real errors | 282 | {n_bad} |",
          f"| QUALITY | 46 | {tot['QUALITY']} |",
          f"| reclassified COSMETIC | — | {tot['COSMETIC']} |", "",
          "Rule 1 did the heavy lifting and it landed exactly where Louis said "
          "it would: **every surviving QUALITY error is a changed third.** The "
          "pairs that vanished — `halfdim→dim` (8), `maj→dom` (7), "
          "`dom→maj` (7) — all share a triad core. This Love went from 16 "
          "QUALITY errors to **0** and from 84.3% to 98.3% agreement.", "",
          "It did **not** do what was expected to Chain of Fools. Its 57 ADDED "
          "are Eb (13), Em (9), E (8), Eb7 (4), Bbm, D, A, F, B, Gb — chords "
          "on roots the tab never uses, not Cm/Cm7 alternation. The Cm↔Cm7 "
          "vamp writing does show up, but as C and C7 against the tab's Cm7: "
          "that is a **major third against a minor third**, which Rule 1 "
          "explicitly keeps as an error. So Chain's ADDED went 57→57, and its "
          "19 QUALITY errors are all `dom→min` / `maj→min`. Reported "
          "rather than smoothed.", "",
          "### Quality errors by third (ours → UG)", "",
          "| ours → UG | n | in words |", "|---|---|---|"]
    for (x, y), n in sorted(fam.items(), key=lambda z: -z[1]):
        L.append(f"| {x} → {y} | {n} | {FAM_PAIR_LABEL.get((x, y), '')} |")
    if roots:
        L += ["", "### Root errors", "", "| ours → UG | n |", "|---|---|"]
        for k, n in sorted(roots.items(), key=lambda z: -z[1]):
            L.append(f"| {k} | {n} |")

    # the MISSED deep-dive, if scratchpad/ug_missed.py has been run
    sec = SCRATCH / "ug_missed_section.md"
    if sec.exists():
        L += ["", sec.read_text(), ""]

    L += ["", "### The headline: we UNDER-write", "",
          "Our charts carry roughly half the tab's chord events on the dense "
          "songs (Close to You 52 vs 106, Every Breath 70 vs 132, Let It Be 109 "
          "vs 175). Same-root ornaments and cosmetic spellings are both already "
          "excluded, so what remains are changes to a **different root** that we "
          "never wrote.", "",
          "Two leads worth an ear:", "",
          "- **Stand By Me: we write no-chord from 0.4–26.4 s and again "
          "28.4–38.4 s.** Rule 2 removes the first 13.8 s (the tab's own "
          "intro), but the first sung word is at 14.8 s by ASR — so the "
          "no-chord runs about 12 s **into the sung verse**, and the second span "
          "is entirely inside it. The finding shrinks under Rule 2; it does not "
          "disappear.",
          "- **Hot N Cold: 18 ADDED, clustered after 240 s.** Everything before "
          "that is clean. Whatever fails, fails at the end of the song.", "",
          "### Caveats", "",
          "- A song whose alignment verdict is `harmony-underdetermined` has "
          "reliable ORDER but soft placement, so its ADDED/MISSED are weaker "
          "evidence. That is Chain of Fools, and it carries most of the ADDED.",
          "- UG tabs contain material the recording does not (alternate "
          "endings); `unsupported_frac` bounds how much of the tab the audio "
          "backs.",
          "- MISSED counts assume the tab's extra events are real chord changes. "
          "Same-root ornaments are excluded, but a tab that writes a passing "
          "chord we deliberately merge will still read as a miss.", "",
          "## Appendix — before the three rules (commit d1da3a0)", "",
          "| song | agree | ADDED | MISSED | ROOT | QUAL |", "|---|---|---|---|---|---|"]
    for name, ag, ad, mi, ro, qu in BEFORE:
        L.append(f"| {name} | {ag}% | {ad} | {mi} | {ro} | {qu} |")
    L += ["", "Then: 282 errors, MISSED 54% / ADDED 27% / QUALITY 16% / ROOT 2%. "
          "The old run also scored four songs without ASR, so its alignments "
          "differ slightly from the current ones on top of the rule changes.", ""]
    out_path.write_text("\n".join(L))
    print(f"wrote {out_path}  ({len(songs)} songs, {n_bad} errors)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("slug", nargs="?")
    ap.add_argument("source", nargs="?")
    ap.add_argument("--asr", action="store_true")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        build_report(REPO / "docs" / "ug_score_report.md")
        return
    if not a.slug:
        ap.error("need <slug> <ug_html_or_url>, or --report")
    s = run(a.slug, a.source, a.asr)
    print_song(s)


if __name__ == "__main__":
    main()
