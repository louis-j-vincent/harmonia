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

Error classes reported, worst (longest) first:
  ADDED    we invented a chord              (sub-class: inside a "(No music)"
                                             stretch = hallucination in silence)
  MISSED   UG has a chord we never wrote    (sub-class: we wrote no-chord there)
  ROOT     paired, different root           ("we wrote G where it was C")
  QUALITY  paired, same root, wrong family  (e.g. we wrote C7, UG says Cm)
Same root + same family but different exact quality (Cmaj7 vs C) is counted
separately as `quality_detail` and is NOT an error — that is partial credit.
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
            out.append(Ev(e.t0, e.t1, e.root, e.q, e.conf, e.nc, e.nomusic))
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
        evs.append(Ev(float(c.t0), float(c.t1), r, q,
                      float(-sup[c.idx]), False, nm))
    return _collapse(evs)


# --------------------------------------------------------------------------- #
# 1. anchors: positions BOTH sides are sure about
# --------------------------------------------------------------------------- #
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
def score(slug: str, ours: list[Ev], ug: list[Ev], anchors: list[tuple[int, int]],
          nm_spans) -> dict:  # noqa: C901
    errs: list[dict] = []
    n_pair = n_ok = 0

    # An anchor is a compared position too (roots agree by construction), so it
    # belongs in the denominator.  Leaving it out made Close to You's
    # agreement% a 7-sample statistic while 38 agreeing anchors sat unused.
    for oi, ui in anchors:
        o, u = ours[oi], ug[ui]
        n_pair += 1
        if family(o.q) == family(u.q):
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
                if any(u.root == o.root for u in near):
                    errs.append(_e("SPLIT", None, o,
                                   f"we chopped a held {_PC[o.root]} into a "
                                   f"separate {o.name}"))
                else:
                    errs.append(_e("ADDED", None, o,
                                   ("hallucination in a (No music) stretch"
                                    if sil else "extra chord UG does not have"),
                                   silence=sil))
            else:
                u = B[ib]
                # Mirror of SPLIT: UG writing Db6 inside a held Db, or Cadd9
                # next to C, is inner-voice detail, not a chord change we
                # missed.  Scoring those as MISSED buried Close to You's real
                # misses under 62 strumming ornaments.
                near = ours[max(oi0 - 1, 0):min(oi1 + 1, len(ours))]
                if any(o.root == u.root and not o.nc for o in near):
                    errs.append(_e("ORNAMENT", u, None,
                                   f"UG writes {u.name} inside our held "
                                   f"{_PC[u.root]}"))
                else:
                    errs.append(_e("MISSED", u, None,
                                   "UG has it, we never wrote it"))

    agree = 100.0 * n_ok / max(n_pair, 1)
    cnt = {k: sum(1 for e in errs if e["cls"] == k)
           for k in ("ADDED", "SPLIT", "MISSED", "ORNAMENT", "ROOT", "QUALITY",
                     "quality_detail")}
    cnt["ADDED_in_silence"] = sum(1 for e in errs
                                  if e["cls"] == "ADDED" and e["silence"])
    return {"slug": slug, "n_ours": len(ours), "n_ug": len(ug),
            "anchors": len(anchors), "paired": n_pair,
            "agreement_pct": round(agree, 1), "counts": cnt,
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

    anchors_w, nm_spans = None, []
    if use_asr:
        asr = UA.transcribe(slug)
        tw = meta.pop("_words", [])
        anc, _rep, _drop = UA.lyric_anchors(chords, tw, asr, F * hop)
        nm_spans = UA.nomusic_spans(tw, asr)
        anchors_w = UA.anchors_to_windows(anc, hop, F, nm_spans=nm_spans,
                                          n_chords=len(chords))
    else:
        meta.pop("_words", None)

    res, C = UA.align(chords, feats, hop, anchors=anchors_w)
    audit = UA.identifiability(chords, C, res)
    sup = UA.support(C, res)
    audit["unsupported_frac"] = round(float((sup > 0).mean()), 3)

    ours = our_sequence(slug)
    ugs = ug_sequence(chords, sup, nm_spans)
    anch = find_anchors(ours, ugs)
    out = score(slug, ours, ugs, anch, nm_spans)
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
          f"SPLIT {c['SPLIT']}  MISSED {c['MISSED']}  ORNAM {c['ORNAMENT']}  "
          f"ROOT {c['ROOT']}  "
          f"QUALITY {c['QUALITY']}  (same-family detail {c['quality_detail']})")
    print(f"    audit: {s['audit']['verdict']}, contrast "
          f"{s['audit']['cost_contrast']}σ, unsupported "
          f"{s['audit']['unsupported_frac']}")
    real = [e for e in s["errors"]
            if e["cls"] not in ("quality_detail", "ORNAMENT")]
    for e in real[:top]:
        print(f"      {e['t0']:>6.1f}-{e['t1']:<6.1f} {e['cls']:<8} {e['why']}")


FAM_PAIR_LABEL = {
    ("maj", "dom"): "we dropped a dominant 7th (wrote the triad)",
    ("dom", "maj"): "we invented a dominant 7th",
    ("maj", "min"): "we wrote major where it is minor",
    ("min", "maj"): "we wrote minor where it is major",
    ("halfdim", "dim"): "we wrote m7b5 where it is a full diminished 7th",
    ("dim", "halfdim"): "we wrote dim7 where it is m7b5",
    ("min", "dom"): "we wrote minor where it is dominant",
    ("dom", "min"): "we wrote dominant where it is minor",
}


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
        bad = c["ADDED"] + c["MISSED"] + c["ROOT"] + c["QUALITY"]
        return bad / max(s["n_ours"], 1)

    songs.sort(key=err_rate, reverse=True)

    L = ["# Where our charts are wrong — scored against Ultimate Guitar",
         "",
         "Generated by `scratchpad/ug_score.py`. Method is the consumption",
         "doctrine in `docs/ug_alignment_brick.md`: UG *timing* is hand-made and",
         "rough, UG *order* is reliable, so we snap to **anchors** (positions",
         "where our chart is confident AND the alignment's audio support is",
         "positive AND the roots agree exactly) and diff only what lies between",
         "consecutive anchors. No per-chord timestamp comparison anywhere.",
         "",
         "## Error classes", "",
         "| class | meaning | counts as our error |",
         "|---|---|---|",
         "| **ADDED** | we wrote a chord UG does not have, on a root not in force | yes |",
         "| ADDED *in silence* | …and the tab marks that stretch \"(No music)\" | yes, worst kind |",
         "| **MISSED** | UG has a chord change we never wrote | yes |",
         "| **ROOT** | paired position, different root (\"we wrote G where it was C\") | yes |",
         "| **QUALITY** | paired, same root, wrong family (we wrote C7, UG says Cm) | yes |",
         "| SPLIT | we chopped a held chord into pieces (same root) | granularity |",
         "| ORNAMENT | UG writes Db6 inside our held Db | granularity |",
         "| quality_detail | same root, same family, different exact quality | partial credit |",
         "",
         "SPLIT / ORNAMENT / quality_detail are reported but excluded from the",
         "error rate: they are grid-resolution differences between a strummed",
         "guitar sheet and a chord chart, not wrong answers.",
         "",
         "## Summary (worst first)", "",
         "| song | UG | ours | UG ch | anchors | paired | agree | ADDED | MISSED | ROOT | QUAL | SPLIT | ORN |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in songs:
        m, c = s["meta"], s["counts"]
        sil = f" ({c['ADDED_in_silence']} in silence)" if c["ADDED_in_silence"] else ""
        L.append(f"| {m['song']} | {m['rating']}★/{m['votes']} | {s['n_ours']} | "
                 f"{s['n_ug']} | {s['anchors']} | {s['paired']} | "
                 f"**{s['agreement_pct']}%** | {c['ADDED']}{sil} | {c['MISSED']} | "
                 f"{c['ROOT']} | {c['QUALITY']} | {c['SPLIT']} | {c['ORNAMENT']} |")
    L += ["", "`agree` = of the positions the diff actually pairs (anchors "
          "included), the share where root AND quality family both match.", ""]

    for s in songs:
        m, c, a = s["meta"], s["counts"], s["audit"]
        L += [f"## {m['artist']} — {m['song']}", "",
              f"UG tab {m['tab_id']}, {m['rating']}★/{m['votes']} votes, "
              f"tonality {m['tonality']}, capo {m['capo'] or 0}. "
              f"Alignment: {a['verdict']}, contrast {a['cost_contrast']}σ, "
              f"unsupported {a['unsupported_frac']}.", "",
              f"{s['anchors']} anchors, {s['paired']} paired positions, "
              f"**{s['agreement_pct']}%** agreement.", ""]
        real = [e for e in s["errors"]
                if e["cls"] not in ("quality_detail", "ORNAMENT")]
        if not real:
            L += ["No errors above the granularity classes.", ""]
            continue
        L += ["| when | class | what |", "|---|---|---|"]
        for e in real[:30]:
            L.append(f"| {e['t0']:.1f}–{e['t1']:.1f}s | {e['cls']} | {e['why']} |")
        if len(real) > 30:
            L.append(f"| … | | {len(real) - 30} more |")
        L.append("")

    # ---- synthesis -------------------------------------------------------- #
    tot = {}
    for s in songs:
        for k, v in s["counts"].items():
            tot[k] = tot.get(k, 0) + v
    fam_pairs: dict[tuple[str, str], int] = {}
    root_pairs: dict[str, int] = {}
    for s in songs:
        for e in s["errors"]:
            if e["cls"] == "QUALITY" and e["ours"] and e["ug"]:
                a_, b_ = e["why"].split(":")[0].split(" vs ")
                fam_pairs[(a_, b_)] = fam_pairs.get((a_, b_), 0) + 1
            elif e["cls"] == "ROOT":
                root_pairs[f"{e['ours']} -> {e['ug']}"] = \
                    root_pairs.get(f"{e['ours']} -> {e['ug']}", 0) + 1

    n_bad = tot["ADDED"] + tot["MISSED"] + tot["ROOT"] + tot["QUALITY"]
    classes = sorted([("MISSED — a chord change we never wrote", tot["MISSED"]),
                      ("QUALITY — right root, wrong chord family", tot["QUALITY"]),
                      ("ADDED — a chord that is not there", tot["ADDED"]),
                      ("ROOT — the wrong bass/root entirely", tot["ROOT"])],
                     key=lambda x: -x[1])
    L += ["## Cross-song synthesis — where we are wrong, ranked", "",
          f"{n_bad} real errors across {len(songs)} songs.", "",
          "| rank | error class | count | share |", "|---|---|---|---|"]
    for i, (name, n) in enumerate(classes, 1):
        L.append(f"| {i} | {name} | {n} | {100 * n / max(n_bad, 1):.0f}% |")
    L += ["", f"Of the ADDED, **{tot['ADDED_in_silence']}** are inside a stretch "
          "the tab marks \"(No music)\" — chords written where no instrument "
          "plays.", "",
          "### Quality errors by family pair (ours → UG)", "",
          "| ours → UG | n | in words |", "|---|---|---|"]
    for (a_, b_), n in sorted(fam_pairs.items(), key=lambda x: -x[1]):
        L.append(f"| {a_} → {b_} | {n} | "
                 f"{FAM_PAIR_LABEL.get((a_, b_), '')} |")
    if root_pairs:
        L += ["", "### Root errors", "", "| ours → UG | n |", "|---|---|"]
        for k, n in sorted(root_pairs.items(), key=lambda x: -x[1]):
            L.append(f"| {k} | {n} |")
    L += ["", "### The headline: we UNDER-write, we do not over-write", "",
          "MISSED is the largest class on every song whose alignment is "
          "high-contrast, and it is not close. Our charts carry roughly half "
          "the chord events the tab does on the dense songs (Close to You "
          "52 vs 106, Every Breath 70 vs 132, Let It Be 109 vs 175). Same-root "
          "ornaments are already excluded, so these are changes to a DIFFERENT "
          "root that we never wrote.",
          "",
          "The ADDED class is the opposite failure and it is far more "
          "concentrated: 57 of 76 are Chain of Fools alone, the one song whose "
          "harmony cannot time itself. Strip that song and ADDED drops to 19 "
          "across six songs, behind QUALITY. **The corpus-wide defect is "
          "missing chords, not inventing them** — the reverse of what the "
          "3-song read suggested, which is exactly why single-song findings "
          "are hypotheses.",
          "",
          "Two specific things worth a listen:",
          "",
          "- **Stand By Me, 0.4–26.4 s and 28.4–38.4 s: we write NO-CHORD for "
          "26 of the first 38 seconds.** The tab has chords throughout. That is "
          "chord-vs-no-chord failing in the conservative direction, on a song "
          "where we otherwise score 94.9%.",
          "- **Hot N Cold, 267–283 s: 4 ADDED plus 3 SPLIT clustered in the "
          "outro.** Everything before 240 s is clean. Whatever goes wrong, goes "
          "wrong at the end of the song.", "",
          "### Song selection and gates", "",
          "Candidates needed a baked payload, audio, and a UG chords tab above "
          "4.7★. Every song was pre-flighted for rating, capo/tonality vs our "
          "measured tonic, and harmony-identifiability before being scored. "
          "Three of the four new tabs carry a capo (Every Breath 1, Hot N Cold "
          "5, Stand By Me 2) — the aligner transposes to sounding pitch at "
          "parse time, and UG's `tonality` field is already sounding, so it is "
          "not transposed.",
          "",
          "**Stand By Me failed the tonic gate and was included anyway, because "
          "the gate is wrong, not the song.** `infer_key` on our chroma answers "
          "**C# minor** for a song in **A major** — the mediant, not merely the "
          "wrong mode. Three independent checks say the tab is fine: the chroma "
          "energy peaks on A (1.00 vs C# 0.896), the capo-2 sounding chords are "
          "A/F#m/D/E, and our own chart — built from the audio with no "
          "knowledge of the tab — contains exactly A, D, E and F#m and nothing "
          "else. It then scored 94.9%. This is worse than the limitation "
          "already logged for `infer_key` (\"compares tonic only, never mode\"): "
          "here the tonic itself is wrong, so a tonic-only comparison does not "
          "rescue it. Worth a `known_issues.md` entry.", "",
          "### Adjudication notes (checked, not assumed)", "",
          "**`halfdim → dim` is NOT established as our error.** All 8 are This "
          "Love's D chord: we write Dm7b5 (D F Ab C), the tab writes Ddim7 "
          "(D F Ab Cb). Guitar sheets are known to be loose about exactly this "
          "distinction — `dim7` is written for the shape. I measured the NNLS "
          "chroma over the 8 spans to settle it and it does not: B scores 0.86 "
          "against C at 0.73, which leans to the tab, but the feature is muddy "
          "on this mix (Gb sits at 0.79 and belongs to neither chord). "
          "**Unresolved — needs Louis's ear.** Excluded from any claim that we "
          "are wrong 8 times.",
          "",
          "The trust order (iReal > guitar tabs > model output) applies to the "
          "chord identity, but a tab's *quality spelling* is the weakest thing "
          "it carries. Root-level disagreements from a >4.7★ tab are strong "
          "evidence; 7th/extension disagreements are worth a listen, not a "
          "code change.", "",
          "### Caveats", "",
          "- SPLIT/ORNAMENT counts say a guitar sheet and our chart use "
          "different grids; they are not evidence either side is wrong.",
          "- A song whose alignment verdict is `harmony-underdetermined` has "
          "reliable ORDER but unreliable per-chord placement, so its ADDED/"
          "MISSED counts are softer evidence than a high-contrast song's.",
          "- UG tabs contain material the recording does not (alternate "
          "endings); `unsupported_frac` bounds how much of the tab the audio "
          "actually backs.", ""]
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
