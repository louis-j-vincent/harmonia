#!/usr/bin/env python3
"""Failure-mode analysis for Harmonia inferred charts.

Two evidence tiers (trust order per CLAUDE.md: iReal GT > model self-signal):

  1. GT-anchored (pilots with an irealb_<slug>.html reference chart): align the
     inferred per-beat chords to each GT chord span by time overlap, majority-vote
     the inferred root/quality-family over the span, and compare. This produces the
     real root- and quality-confusion matrices.

  2. Model self-signal (all inferred charts, no GT needed): each inferred chord
     carries `sug` (the model's own ranked alternatives). A chosen root that has a
     perfect-fifth competitor (+-7 semitones) near the top of `sug` flags the known
     5th-apart acoustic-confusion risk without needing labels.

Emits a single JSON blob (stdout or --out) consumed by the dashboard builder.
"""
from __future__ import annotations
import json, re, sys, argparse
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PLOTS = REPO / "docs" / "plots"

ROOT_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NOTE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
# families we bucket qualities into for partial-credit scoring
FAMILIES = ["maj", "min", "dom", "hdim", "dim", "sus", "other"]


def brace_match(text: str, start: int) -> str | None:
    """Return the `{...}` object literal beginning at/after `start`, respecting
    JS strings so braces inside quotes don't unbalance the scan."""
    j = text.find("{", start)
    if j < 0:
        return None
    depth = 0
    instr = False
    esc = False
    q = ""
    k = j
    while k < len(text):
        ch = text[k]
        if instr:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == q:
                instr = False
        else:
            if ch in "\"'":
                instr = True
                q = ch
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    return text[j:k + 1]
        k += 1
    return None


def load_inferred(slug: str) -> dict | None:
    p = PLOTS / f"inferred_{slug}.html"
    if not p.exists():
        return None
    t = p.read_text(encoding="utf-8")
    m = re.search(r"\bconst P\s*=", t)
    if not m:
        return None
    blob = brace_match(t, m.end())
    if not blob:
        return None
    try:
        return json.loads(blob)
    except ValueError:
        return None


def load_gt(slug: str) -> list[dict] | None:
    """irealb_<slug>.html carries `window.P = {chords:[{label,t0,t1,...}], tempo}`."""
    p = PLOTS / f"irealb_{slug}.html"
    if not p.exists():
        return None
    t = p.read_text(encoding="utf-8")
    m = re.search(r"window\.P\s*=", t)
    if not m:
        return None
    blob = brace_match(t, m.end())
    if not blob:
        return None
    try:
        return json.loads(blob).get("chords", [])
    except ValueError:
        return None


def parse_label(lbl: str) -> tuple[int | None, str]:
    """iReal label -> (root pitch-class, quality-tail). Strips /bass inversions
    (functional root only, matching how POP909/iReal GT encodes root)."""
    s = str(lbl or "").strip().split("/")[0].strip()
    m = re.match(r"^([A-G])([#b]?)(.*)$", s)
    if not m:
        return None, ""
    pc = NOTE_PC[m.group(1)]
    if m.group(2) == "#":
        pc = (pc + 1) % 12
    elif m.group(2) == "b":
        pc = (pc + 11) % 12
    return pc, m.group(3).strip()


def quality_family(q: str) -> str:
    """Bucket an iReal quality tail into a coarse family (order matters: the more
    specific half-diminished/diminished tests run before the plain-minor test)."""
    q = (q or "").strip()
    ql = q.lower()
    if not q:
        return "maj"
    if "7b5" in ql or ql.startswith("h") or "ø" in q or "m7b5" in ql:
        return "hdim"
    if ql.startswith("o") or ql.startswith("dim") or ql.startswith("-o"):
        return "dim"
    if "sus" in ql:
        return "sus"
    if ql.startswith("-") or ql.startswith("m") and not ql.startswith("maj"):
        return "min"
    if ql[0] in "^6" or ql.startswith("maj") or ql.startswith("add") or ql.startswith("9^"):
        return "maj"
    if ql[0] in "79" or ql.startswith("11") or ql.startswith("13"):
        return "dom"
    return "other"


def inferred_quality(ch: dict) -> str:
    """The quality the chart displays: prefer the finest level whose confidence
    still leads, mirroring the chart's own exact>=seventh>=family fallback."""
    lv = ch.get("lv") or {}
    ex, sev, fam = lv.get("exact"), lv.get("seventh"), lv.get("family")
    for level in (ex, sev, fam):
        if level and level.get("q") is not None:
            return level["q"]
    return ""


def sample_inferred_over_span(inf_chords: list[dict], t0: float, t1: float):
    """Overlap-weighted majority vote of inferred (root, family) across [t0,t1]."""
    root_w: dict[int, float] = {}
    fam_w: dict[str, float] = {}
    conf_num = conf_den = 0.0
    for c in inf_chords:
        a, b = c.get("t0", 0.0), c.get("t1", 0.0)
        ov = min(t1, b) - max(t0, a)
        if ov <= 0:
            continue
        r = c.get("root", -1)
        if r is None or r < 0:
            continue
        fam = quality_family(inferred_quality(c))
        root_w[r] = root_w.get(r, 0.0) + ov
        fam_w[fam] = fam_w.get(fam, 0.0) + ov
        cval = ((c.get("lv") or {}).get("exact") or {}).get("c", 0.0) or 0.0
        conf_num += cval * ov
        conf_den += ov
    if not root_w:
        return None
    root = max(root_w, key=root_w.get)
    fam = max(fam_w, key=fam_w.get)
    conf = conf_num / conf_den if conf_den else 0.0
    return root, fam, conf


def fifth_competition(ch: dict) -> bool:
    """True when the model's top alternative root sits a perfect fifth (+-7 st)
    from the chosen root with >=60% of the chosen root's confidence — the
    signature of 5th-apart acoustic ambiguity."""
    root = ch.get("root", -1)
    sug = ch.get("sug") or []
    if root is None or root < 0 or not sug:
        return False
    chosen_c = None
    for s in sug:
        if s.get("root") == root:
            chosen_c = s.get("c", 0.0)
            break
    if chosen_c is None:
        chosen_c = sug[0].get("c", 0.0)
    for s in sug:
        r = s.get("root", -1)
        if r is None or r < 0 or r == root:
            continue
        if (r - root) % 12 in (5, 7) and s.get("c", 0.0) >= 0.6 * (chosen_c or 1e-9):
            return True
    return False


def collapse_segments(inf_chords: list[dict]):
    """Merge consecutive per-beat inferred chords with the same (root, family)
    into segments — the inferred chart's own chord list."""
    segs = []
    for c in inf_chords:
        r = c.get("root", -1)
        if r is None or r < 0:
            continue
        fam = quality_family(inferred_quality(c))
        if segs and segs[-1]["root"] == r and segs[-1]["fam"] == fam:
            segs[-1]["t1"] = c.get("t1", segs[-1]["t1"])
        else:
            segs.append({"root": r, "fam": fam, "t0": c.get("t0", 0.0), "t1": c.get("t1", 0.0)})
    return segs


def nw_align(a: list[int], b: list[int], match=2, mism=-1, gap=-1):
    """Needleman-Wunsch global alignment of two root sequences; returns matched
    (i, j) index pairs. Used only to bound the GT comparison, not to score it."""
    n, m = len(a), len(b)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        D[i][0] = i * gap
    for j in range(1, m + 1):
        D[0][j] = j * gap
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            s = match if a[i - 1] == b[j - 1] else mism
            D[i][j] = max(D[i - 1][j - 1] + s, D[i - 1][j] + gap, D[i][j - 1] + gap)
    i, j, pairs = n, m, []
    while i > 0 and j > 0:
        s = match if a[i - 1] == b[j - 1] else mism
        if D[i][j] == D[i - 1][j - 1] + s:
            pairs.append((i - 1, j - 1)); i -= 1; j -= 1
        elif D[i][j] == D[i - 1][j] + gap:
            i -= 1
        else:
            j -= 1
    return pairs[::-1]


def gt_integrity(slug: str) -> dict | None:
    """Diagnose whether an irealb_<slug>.html GT chart can anchor an evaluation.
    Returns duration spans plus the two competing (and mutually contradictory)
    root-accuracy estimates that prove the timeline is unusable:
      * time_overlap_acc  — pessimistic (misaligned timelines -> ~chance)
      * seq_align_acc     — optimistic  (free alignment cherry-picks 1-of-N segs)
    """
    gt = load_gt(slug)
    inf = load_inferred(slug)
    if not gt or not inf:
        return None
    inf_chords = inf.get("chords", [])
    gt_dur = max((c.get("t1", 0.0) for c in gt), default=0.0)
    inf_dur = max((c.get("t1", 0.0) for c in inf_chords), default=0.0)
    gtp = [parse_label(g.get("label", "")) for g in gt]
    gt_roots = [r for r, _ in gtp if r is not None]

    # (a) absolute-time overlap sampling
    to_n = to_ok = 0
    for g in gt:
        gr, _ = parse_label(g.get("label", ""))
        if gr is None:
            continue
        s = sample_inferred_over_span(inf_chords, g.get("t0", 0.0), g.get("t1", 0.0))
        if not s:
            continue
        to_n += 1
        to_ok += (s[0] == gr)

    # (b) timeline-free NW sequence alignment against collapsed segments
    segs = collapse_segments(inf_chords)
    seg_roots = [s["root"] for s in segs]
    pairs = nw_align(gt_roots, seg_roots)
    sa_ok = sum(1 for gi, si in pairs if gt_roots[gi] == seg_roots[si])

    return {
        "slug": slug,
        "gt_dur": round(gt_dur, 1), "inf_dur": round(inf_dur, 1),
        "dur_ratio": round(inf_dur / gt_dur, 2) if gt_dur else None,
        "n_gt": len(gt_roots), "n_inf_segments": len(segs),
        "time_overlap_acc": round(to_ok / to_n, 3) if to_n else None,
        "seq_align_acc": round(sa_ok / len(pairs), 3) if pairs else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    inferred_files = sorted(PLOTS.glob("inferred_*.html"))
    slugs = [f.stem[len("inferred_"):] for f in inferred_files]

    # --- tier 1 (BLOCKED): GT-anchored integrity check ---
    # The irealb GT charts and the inferred charts do NOT share a timeline
    # (GT spans are far shorter than the inferred chart), so no valid GT eval is
    # possible from these artifacts. We surface the evidence rather than a number.
    gt_checks = [c for slug in slugs if (c := gt_integrity(slug))]

    # --- tier 2 (VALID): model self-signal across all inferred charts ---
    # Every inferred chord carries `sug` (the model's own ranked root/quality
    # alternatives) and `lv.exact.c` (chosen-chord confidence). This is fully
    # grounded, needs no labels, and directly exposes where the model wavers.
    root_confus = [[0.0] * 12 for _ in range(12)]          # [chosen][competitor] conf mass
    qual_waver = [[0.0] * len(FAMILIES) for _ in range(len(FAMILIES))]  # [chosen fam][competitor fam]
    fam_dist: dict[str, int] = {f: 0 for f in FAMILIES}
    conf_bins = [0, 0, 0, 0, 0]        # <.2 .2-.4 .4-.6 .6-.8 >=.8
    fifth_dir = {"fifth_up_+7": 0, "fourth_up_+5": 0}
    total_chords = total_fifth = 0
    songs = []
    for slug in slugs:
        inf = load_inferred(slug)
        if not inf:
            continue
        chs = inf.get("chords", [])
        nfifth = low = 0
        conf_sum = 0.0
        for c in chs:
            total_chords += 1
            root = c.get("root", -1)
            chosen_fam = quality_family(inferred_quality(c))
            fam_dist[chosen_fam] += 1
            cval = ((c.get("lv") or {}).get("exact") or {}).get("c", 0.0) or 0.0
            conf_sum += cval
            conf_bins[min(4, int(cval / 0.2))] += 1
            if cval < 0.4:
                low += 1
            # accumulate competitor mass from the model's own suggestion list
            for s in (c.get("sug") or []):
                sr = s.get("root", -1)
                sc = s.get("c", 0.0) or 0.0
                if sr is None or sr < 0:
                    continue
                if root is not None and root >= 0 and sr != root:
                    root_confus[root][sr] += sc
                    qual_waver[FAMILIES.index(chosen_fam)][FAMILIES.index(quality_family(s.get("q", "")))] += sc
            if fifth_competition(c):
                nfifth += 1
                total_fifth += 1
                # directionality of the top competing fifth
                sug = c.get("sug") or []
                for s in sug:
                    sr = s.get("root", -1)
                    if sr is None or sr < 0 or sr == root:
                        continue
                    d = (sr - root) % 12
                    if d == 7:
                        fifth_dir["fifth_up_+7"] += 1; break
                    if d == 5:
                        fifth_dir["fourth_up_+5"] += 1; break
        songs.append({
            "slug": slug, "n_chords": len(chs),
            "fifth_competition": nfifth,
            "fifth_pct": round(100 * nfifth / len(chs), 1) if chs else 0.0,
            "low_conf_pct": round(100 * low / len(chs), 1) if chs else 0.0,
            "mean_conf": round(conf_sum / len(chs), 3) if chs else 0.0,
            "tonic": (inf.get("home") or {}).get("tonic"),
            "mode": (inf.get("home") or {}).get("mode"),
            "key": inf.get("keyName"),
        })

    out = {
        "roots": ROOT_NAMES, "families": FAMILIES,
        "gt_integrity": {
            "note": ("irealb GT charts are on a different (shorter) timeline than the "
                     "inferred charts, so no valid GT-anchored root/quality eval is "
                     "possible from these artifacts. Both alignment strategies contradict "
                     "each other, proving the mismatch."),
            "pilots": gt_checks,
        },
        "self_signal": {
            "root_confusability": [[round(v, 2) for v in row] for row in root_confus],
            "quality_waver": [[round(v, 2) for v in row] for row in qual_waver],
            "family_distribution": fam_dist,
            "conf_bins": conf_bins,
            "conf_bin_edges": ["<0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", ">=0.8"],
            "total_chords": total_chords,
            "total_fifth_competition": total_fifth,
            "fifth_competition_pct": round(100 * total_fifth / total_chords, 1) if total_chords else 0.0,
            "fifth_direction": fifth_dir,
            "low_conf_total": conf_bins[0] + conf_bins[1],
            "low_conf_pct": round(100 * (conf_bins[0] + conf_bins[1]) / total_chords, 1) if total_chords else 0.0,
            "n_songs": len(songs), "songs": songs,
        },
    }
    js = json.dumps(out, indent=1)
    if args.out:
        Path(args.out).write_text(js, encoding="utf-8")
        print(f"wrote {args.out}  ({len(songs)} charts, {total_chords} chords, "
              f"{len(gt_checks)} GT-integrity checks)", file=sys.stderr)
    else:
        print(js)


if __name__ == "__main__":
    main()
