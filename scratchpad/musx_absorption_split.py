"""musx_absorption_split.py -- cross-song replication of the Let It Be
MISSED PRESENT/ABSENT split.

CONTEXT (docs/ug_score_report.md, "Falsification test result", main session
2026-07-30). On Let It Be, every MISSED chord (a chord Ultimate Guitar's tab
has that our chart doesn't) was checked against music-x-lab's own frame-level
decode (``data/cache/musx_infer/<slug>_submission.lab`` -- this is upstream of
our bar-grid/min-duration/fold post-processing, so it screens hypothesis 1
"our chart is just too coarse-grained to write short chords"). The 38 MISSED
chords split into:

  - 24 ABSENT from musx entirely -- "decoder-level absorption": e.g. all 13
    D-7 passing chords decode as their shared-tone neighbour F:maj, because
    Dm7 (D-F-A-C) contains F-A-C and only the D in the bass tells them apart.
    No amount of chart-layer grain fix can recover these; the fix (if any)
    is bass-informed root discrimination AT DECODE TIME.
  - 14 PRESENT in musx (real decoded segments, some 3.4s long) but then lost
    somewhere between musx and the baked chart -- a chart-layer bug, not a
    decoder problem.

Single-song result = a HYPOTHESIS (CLAUDE.md rule #5, "single-song findings
are hypotheses"; Song 001's beat-phase correlation collapsed corpus-wide the
same way). This script repeats the exact split on five more songs to see if
the ~24/14 (63/37) ratio holds, or was a Let It Be idiosyncrasy (that song has
an unusually repetitive C-Dm7-C vamp).

DEFINITION USED (mirrors the mission brief exactly; deviations noted below)
----------------------------------------------------------------------------
For each MISSED error (``cls == "MISSED"``, ``ug`` not null) in
``scratchpad/ug_score_<slug>.json``:

  1. Parse the UG root pc with ``harmonia.models.musx_bass._parse_root``
     (handles the UG shorthand root+accidental spelling directly, e.g.
     "D-7" -> D, "Bb-" -> Bb, "G^7" -> G).
  2. PRESENT(strict) if music-x-lab's .lab has ANY segment with that root pc
     overlapping ``[t0, t1)`` (zero tolerance).
  3. PRESENT(+-0.5s) if the same check passes after expanding the MISSED span
     by 0.5s on each side (UG's hand-made timing per docs/ug_alignment_brick.md).
  4. ABSENT = NOT present at +-0.5s. This is the "genuinely absent even after
     forgiving UG's timing" bucket, and it's what the ABSENT-specific columns
     (shared-tone absorption, decoded-at-midpoint) are computed over. Using
     the loose tolerance for this call (rather than strict) avoids counting a
     chord as a "decoder failure" when it's really just a UG timestamp that's
     off by a beat -- the same slop class the main report already documents
     (docs/ug_score_report.md "Diff slop", -16 of 82 misses were this).
  5. For every ABSENT row: read whatever music-x-lab DID decode at the span
     midpoint (root + quality), and test shared-tone absorption via pc-set
     subset in EITHER direction (missed-chord-pcs subset-of decoded-pcs, or
     vice versa -- a triad passing chord absorbed into a 7th chord is the
     vice-versa case).

QUALITY -> PC-SET MAPPING (mission brief gave 8 example shapes for the UG
shorthand; extended here, noted explicitly)
----------------------------------------------------------------------------
UG shorthand (brief's own rule, applied by longest/most-specific prefix
match against the suffix after root+accidentals): "-7" -> min7, "^7" -> maj7,
"h7" -> halfdim7, "sus" -> sus4, "o" -> dim, "-" -> min, "7" -> dom7,
"6" -> maj (add6 doesn't change the triad), "" -> maj. None of the six target
songs' MISSED chords actually need the "h7"/"o"/"sus" branches (checked: the
observed UG quality tokens are only "", "-", "-7", "6", "^7") but they're
implemented for completeness/robustness.

music-x-lab's own Harte vocabulary (from its fixed submission dict, see
``harmonia.models.musx_bass._MUSX_Q_TO_SEV``) needs its own map since it is
NOT the UG shorthand -- e.g. musx writes "min7" not "-7". DEVIATION from the
brief: the brief only defined 8 shapes; music-x-lab's dict has 17 qualities.
Extended tensions (9, min9, maj9, 11, 13) are collapsed onto their base
triad+7th core (9->dom7, min9->min7, maj9->maj7, 11->dom7, 13->dom7) since the
9th/11th/13th never changes whether the triad+7th core is shared -- that core
is what "absorption" is actually about. aug and dim7 get their own pc sets
(not in the brief's 8) since collapsing them into maj/dim would misstate the
subset test.
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harmonia.models.musx_bass import _parse_root  # noqa: E402

SLUGS = [
    "let_it_be_remastered_2009",   # baseline replication check (already published)
    "carpenters_close_to_you",
    "the_police_every_breath_you_take_official_music_video",
    "katy_perry_hot_n_cold_official_music_video",
    "maroon_5_this_love",
    "ben_e_king_stand_by_me_audio",
]

WINDOW = 0.5    # UG hand-made timing slop, docs/ug_alignment_brick.md

# pc-set offsets from root, for the UG-shorthand quality buckets named in the brief
UG_SHAPES = {
    "min7": {0, 3, 7, 10},
    "maj7": {0, 4, 7, 11},
    "hdim7": {0, 3, 6, 10},
    "sus4": {0, 5, 7},
    "dim": {0, 3, 6},
    "min": {0, 3, 7},
    "dom7": {0, 4, 7, 10},
    "maj": {0, 4, 7},
}

# music-x-lab's fixed submission-dict qualities -> pc-set offsets from root.
# (Direct map, not routed through UG_SHAPES, since musx's own vocabulary is
# Harte spelling, not UG shorthand -- see module docstring.)
MUSX_SHAPES = {
    "maj": {0, 4, 7},
    "min": {0, 3, 7},
    "7": {0, 4, 7, 10},
    "maj7": {0, 4, 7, 11},
    "min7": {0, 3, 7, 10},
    "dim": {0, 3, 6},
    "dim7": {0, 3, 6, 9},
    "hdim7": {0, 3, 6, 10},
    "aug": {0, 4, 8},
    "sus2": {0, 2, 7},
    "sus4": {0, 5, 7},
    "sus4(b7)": {0, 5, 7, 10},
    "9": {0, 4, 7, 10},     # dom7 core, 9th tension dropped for the subset test
    "min9": {0, 3, 7, 10},  # min7 core
    "maj9": {0, 4, 7, 11},  # maj7 core
    "11": {0, 4, 7, 10},    # dom7 core
    "13": {0, 4, 7, 10},    # dom7 core
}


# ── loaders (same shape as scratchpad/bass_premise_check.py; not imported
#    from it since that file is read-only and its loaders are npz/probs-
#    specific, but the .lab loader here is byte-identical in behaviour) ──────

def load_score(slug: str) -> dict:
    return json.loads((REPO / "scratchpad" / f"ug_score_{slug}.json").read_text())


def load_musx_lab(slug: str) -> list[tuple[float, float, str]]:
    p = REPO / "data" / "cache" / "musx_infer" / f"{slug}_submission.lab"
    out = []
    for line in p.read_text().splitlines():
        f = line.split()
        if len(f) >= 3:
            out.append((float(f[0]), float(f[1]), f[2]))
    return out


# ── quality parsing ──────────────────────────────────────────────────────────

def ug_root_shape(name: str) -> tuple[int | None, str, set[int] | None]:
    """(root pc, shape name, absolute pc set) for a UG chord name.

    Root via ``_parse_root`` (letter + contiguous #/b accidentals). Shape via
    the brief's own longest-prefix rule on whatever suffix remains.
    """
    root = _parse_root(name)
    if root is None:
        return None, "?", None
    n = 1
    while n < len(name) and name[n] in "#b":
        n += 1
    suffix = name[n:]
    if suffix.startswith("-7"):
        shape = "min7"
    elif suffix.startswith("^7"):
        shape = "maj7"
    elif suffix.startswith("h7"):
        shape = "hdim7"
    elif suffix.startswith("sus"):
        shape = "sus4"
    elif suffix.startswith("o"):
        shape = "dim"
    elif suffix.startswith("-"):
        shape = "min"
    elif suffix.startswith("7"):
        shape = "dom7"
    elif suffix.startswith("6"):
        shape = "maj"
    elif suffix == "":
        shape = "maj"
    else:
        shape = "maj"  # unrecognized suffix; none observed in the 6-song set (see docstring)
    pcs = {(root + o) % 12 for o in UG_SHAPES[shape]}
    return root, shape, pcs


def musx_root_shape(label: str | None) -> tuple[int | None, str | None, set[int] | None]:
    """(root pc, Harte quality, absolute pc set) for a music-x-lab label."""
    if not label or label in ("N", "X"):
        return None, None, None
    body = label.split("/", 1)[0]
    if ":" in body:
        root_sym, q = body.split(":", 1)
    else:
        root_sym, q = body, "maj"
    root = _parse_root(root_sym)
    if root is None:
        return None, None, None
    offsets = MUSX_SHAPES.get(q)
    if offsets is None:
        return root, q, None  # unknown quality token -- no pc set, no absorption test
    pcs = {(root + o) % 12 for o in offsets}
    return root, q, pcs


# ── span logic ───────────────────────────────────────────────────────────────

def _overlaps(a0: float, a1: float, b0: float, b1: float) -> bool:
    return a0 < b1 and b0 < a1


def present(labels, want_root: int, t0: float, t1: float, tol: float) -> bool:
    a, b = t0 - tol, t1 + tol
    for s0, s1, lab in labels:
        if lab in ("N", "X"):
            continue
        root_sym = lab.split("/", 1)[0].split(":", 1)[0]
        r = _parse_root(root_sym)
        if r == want_root and _overlaps(a, b, s0, s1):
            return True
    return False


def label_at(labels, t: float) -> str | None:
    for s0, s1, lab in labels:
        if s0 <= t < s1:
            return lab
    return None


# ── per-song run ─────────────────────────────────────────────────────────────

def run(slug: str) -> dict:
    sc = load_score(slug)
    labels = load_musx_lab(slug)

    rows = []
    for e in sc["errors"]:
        if e["cls"] != "MISSED" or e.get("ug") is None:
            continue
        t0, t1 = float(e["t0"]), float(e["t1"])
        root, shape, pcs = ug_root_shape(e["ug"])
        if root is None:
            rows.append({"t0": t0, "t1": t1, "ug": e["ug"], "unparseable": True})
            continue

        pres_strict = present(labels, root, t0, t1, 0.0)
        pres_loose = present(labels, root, t0, t1, WINDOW)
        absent = not pres_loose

        row = {
            "t0": t0, "t1": t1, "ug": e["ug"], "unparseable": False,
            "want_root": root, "want_shape": shape, "want_pcs": sorted(pcs),
            "present_strict": pres_strict, "present_loose": pres_loose,
            "absent": absent,
        }

        if absent:
            mid = 0.5 * (t0 + t1)
            dec_label = label_at(labels, mid)
            dec_root, dec_q, dec_pcs = musx_root_shape(dec_label)
            absorption = None
            if pcs is not None and dec_pcs is not None:
                absorption = pcs.issubset(dec_pcs) or dec_pcs.issubset(pcs)
            row.update({
                "decoded_at_mid": dec_label, "decoded_root": dec_root,
                "decoded_quality": dec_q,
                "decoded_pcs": sorted(dec_pcs) if dec_pcs is not None else None,
                "shared_tone_absorption": absorption,
            })
        rows.append(row)

    parseable = [r for r in rows if not r["unparseable"]]
    n = len(parseable)
    n_strict = sum(r["present_strict"] for r in parseable)
    n_loose = sum(r["present_loose"] for r in parseable)
    absent_rows = [r for r in parseable if r["absent"]]
    n_absent = len(absent_rows)
    absorbed_rows = [r for r in absent_rows if r["shared_tone_absorption"]]
    n_absorbed = len(absorbed_rows)

    shape_counts = Counter(
        f"{r['ug']} swallowed by {r['decoded_at_mid']}" for r in absorbed_rows
    )
    top_shape, top_n = (shape_counts.most_common(1)[0] if shape_counts else (None, 0))

    summary = {
        "slug": slug,
        "n_missed": n,
        "n_unparseable": len(rows) - n,
        "present_strict": n_strict,
        "present_loose": n_loose,
        "absent": n_absent,
        "absent_shared_tone_absorption": n_absorbed,
        "absent_absorption_share": round(n_absorbed / n_absent, 3) if n_absent else None,
        "top_absorbed_shape": top_shape,
        "top_absorbed_shape_n": top_n,
        "absorbed_shape_counts": shape_counts.most_common(),
    }
    return {"summary": summary, "rows": rows}


def main() -> None:
    out_dir = REPO / "scratchpad"
    all_summaries = []
    for slug in SLUGS:
        score_path = out_dir / f"ug_score_{slug}.json"
        lab_path = REPO / "data" / "cache" / "musx_infer" / f"{slug}_submission.lab"
        if not score_path.exists():
            print(f"SKIP {slug}: no ug_score json")
            continue
        if not lab_path.exists():
            print(f"SKIP {slug}: no musx .lab at {lab_path}")
            continue
        res = run(slug)
        (out_dir / f"musx_absorption_{slug}.json").write_text(json.dumps(res, indent=1))
        s = res["summary"]
        print(f"{slug}: n={s['n_missed']} present_strict={s['present_strict']} "
              f"present_loose={s['present_loose']} absent={s['absent']} "
              f"absorbed={s['absent_shared_tone_absorption']} "
              f"top={s['top_absorbed_shape']} x{s['top_absorbed_shape_n']}")
        all_summaries.append(s)

    print("\n| song | n MISSED | PRESENT strict | PRESENT +-0.5s | ABSENT | "
          "ABSENT absorbed | top absorbed shape |")
    print("|---|---|---|---|---|---|---|")
    for s in all_summaries:
        top = (f"{s['top_absorbed_shape']} x{s['top_absorbed_shape_n']}"
               if s['top_absorbed_shape'] else "--")
        print(f"| {s['slug']} | {s['n_missed']} | {s['present_strict']} | "
              f"{s['present_loose']} | {s['absent']} | "
              f"{s['absent_shared_tone_absorption']} | {top} |")


if __name__ == "__main__":
    main()
