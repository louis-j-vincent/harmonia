"""alignment_validator.py — structural QA gate for iReal↔inferred alignment (Mission 6).

Given a candidate alignment (``irealb_aligner.IRealbAlignmentResult``) plus the
inferred chord content (per-segment ``(root, quality, t0, t1)``), decide whether
the chart's own form (A-A-B-A, repeats, section lengths) *survives* the mapping
onto the inferred chords — WITHOUT any absolute-time ground truth.

Three relative signals (see ``docs/mission_6_elastic_matching_design.md``):

  Signal 1  repeat consistency  — same-label sections should have similar inferred
            content.  ``within − cross`` cosine is the aggregate gate; a
            *per-instance sibling-mean z-score* localises a single slipped repeat.
  Signal 2  boundary agreement  — inferred section boundaries vs chart boundaries
            (elastic IoU / F1).  Abstains (nan) when no inferred boundaries.
  Signal 3  per-section family-fraction — worst section's inferred↔GT agreement.

Premise-check refinements baked in (``docs/mission_6_premise_check_results.md``):

  * The global ``within − cross`` Δ is √N-diluted and nearly blind to a *single*
    localised slip → localisation uses the **per-instance sibling-mean z-score**,
    not the global Δ.
  * The z-score is contrast-limited (fires only when the swapped sections are
    harmonically distinct) → when ``within − cross`` is near zero because the
    sections are genuinely similar, return ``UNVERIFIABLE`` (bridge-contrast gate).

Pure/numpy-only; unit-testable without audio.  All audio work already happened
upstream in ``align_irealb_to_inferred``; this consumes its result.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from harmonia.models.section_structure import (
    build_chord_ssm,
    detect_section_boundaries,
)
from harmonia.tab_aligner import _ROOTS, _FLAT_MAP, _family


# ── outputs ────────────────────────────────────────────────────────────────────
@dataclass
class SectionReport:
    label: str                 # iReal section label, e.g. "A"
    instance: int              # 1-based ordinal of this label's repeat (A#1, A#2, …)
    bar0: int                  # chart bar span (0-indexed, inclusive)
    bar1: int
    t0: float                  # audio time span
    t1: float
    family_frac: float         # (#exact+#family)/#chords in this section instance
    n_chords: int
    own_sim: float = float("nan")   # mean cosine to same-label siblings (Signal 1)
    xmatch: float = float("nan")    # best cosine to a *different*-label section
    z: float = float("nan")         # standardised own_sim (per-instance outlier score)
    fingerprint: np.ndarray = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return f"{self.label}#{self.instance}"


@dataclass
class AlignmentValidation:
    align_score: float                 # [0,1] overall coherence (aggregate); nan if UNVERIFIABLE
    verdict: str                       # OK | SUSPECT | MISALIGNED | UNVERIFIABLE
    repeat_consistency: float          # within − cross (Signal 1, aggregate)
    z_outlier: float                   # per-instance z-score of the worst section (localisation)
    boundary_f1: float                 # Signal 2 (nan if abstained)
    min_section_family_frac: float     # Signal 3 (worst section)
    suspect_sections: list[str]        # e.g. ["A#2"] localised via z-outlier / family dip
    bridge_contrast: float             # within − cross; < gate ⇒ slip undetectable
    sections: list[SectionReport]
    notes: list[str] = field(default_factory=list)


# ── fingerprint of inferred content ─────────────────────────────────────────────
# Fixed quality vocabulary so fingerprints are stable across calls/tests.
_QUAL_VOCAB = ["maj", "min", "dom", "hdim", "dim", "aug", "sus"]
_QIDX = {q: i for i, q in enumerate(_QUAL_VOCAB)}
_N_QUAL = len(_QUAL_VOCAB)


def _qual_family_idx(quality: str) -> int:
    return _QIDX.get(_family(quality or ""), 0)


def _fingerprint(segments, n_pitches: int = 12) -> np.ndarray:
    """L2-normalised mean ``[root one-hot(12) | quality-family one-hot]`` over the
    inferred ``(root_pc, quality)`` segments mapped under one section instance."""
    dim = n_pitches + _N_QUAL
    if not segments:
        return np.zeros(dim)
    feats = []
    for root, q in segments:
        v = np.zeros(dim)
        if root is not None and root >= 0:
            v[int(root) % n_pitches] = 1.0
        v[n_pitches + _qual_family_idx(q)] = 1.0
        n = np.linalg.norm(v)
        feats.append(v / n if n > 1e-9 else v)
    fp = np.mean(feats, axis=0)
    n = np.linalg.norm(fp)
    return fp / n if n > 1e-9 else fp


# ── normalise the inferred content into (root_pc, quality, t0, t1) tuples ───────
def _as_segments(inferred) -> list[tuple[int, str, float, float]]:
    """Accept a list of InferredChord objects / dicts / tuples → uniform tuples."""
    out = []
    for c in inferred:
        if isinstance(c, (tuple, list)):
            root, q, t0, t1 = (list(c) + [None, None, None, None])[:4]
        elif isinstance(c, dict):
            root = c.get("pc", c.get("root", -1))
            q = c.get("quality")
            if q is None:
                q = c.get("lv", {}).get("seventh", {}).get("q", "")
            if isinstance(q, str) and q.startswith(":"):
                q = q[1:]
            t0 = c.get("t0")
            t1 = c.get("t1")
        else:  # object with attributes (InferredChord)
            root = getattr(c, "pc", getattr(c, "root", -1))
            q = getattr(c, "quality", "")
            t0 = getattr(c, "t0", None)
            t1 = getattr(c, "t1", None)
        t0 = float(t0) if t0 is not None else 0.0
        t1 = float(t1) if t1 is not None else t0
        out.append((int(root) if root is not None else -1, q or "", t0, t1))
    return out


# ── GT chord parsing (from a transposed display label like "C#min7") ────────────
def _parse_display_label(label: str) -> tuple[int, str]:
    """Return (root_pc, family) of a display label emitted by the aligner.

    Labels look like ``"C#min7"``, ``"Bbmaj7"``, ``"F7"``, ``"N.C."``.  We parse
    the leading root (sharp or flat spelling) and pass the remainder to ``_family``.
    """
    if not label or label in ("N", "N.C."):
        return -1, "maj"
    s = label.strip()
    root = s[0].upper()
    rest = s[1:]
    if rest[:1] in ("#", "b", "♯", "♭"):
        acc = rest[0]
        root2 = root + ("#" if acc in "#♯" else "b")
        rest = rest[1:]
        root2 = _FLAT_MAP.get(root2, root2)
    else:
        root2 = root
    try:
        pc = _ROOTS.index(root2)
    except ValueError:
        return -1, "maj"
    return pc, _family(rest)


# ── section instances (contiguous runs of result.chords, per-label ordinal) ─────
def _section_instances(result_chords):
    """Group ``result.chords`` into contiguous section runs.

    Returns a list of dicts: ``{label, instance, bar0, bar1, t0, t1, chords}``
    where ``instance`` is the 1-based ordinal of that label (A#1, A#2, …) and
    ``chords`` is the list of GT chord dicts in the run.
    """
    runs = []
    label_counts: dict[str, int] = {}
    cur = None
    for ch in result_chords:
        lbl = ch.get("section", "?")
        if cur is None or lbl != cur["label"]:
            if cur is not None:
                runs.append(cur)
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
            cur = dict(label=lbl, instance=label_counts[lbl],
                       bar0=ch.get("bar", 0), bar1=ch.get("bar", 0),
                       t0=None, t1=None, chords=[])
        cur["bar1"] = ch.get("bar", cur["bar1"])
        t0 = ch.get("t0")
        t1 = ch.get("t1")
        if t0 is not None:
            cur["t0"] = t0 if cur["t0"] is None else min(cur["t0"], t0)
        if t1 is not None:
            cur["t1"] = t1 if cur["t1"] is None else max(cur["t1"], t1)
        cur["chords"].append(ch)
    if cur is not None:
        runs.append(cur)
    # backfill Nones
    for r in runs:
        r["t0"] = 0.0 if r["t0"] is None else float(r["t0"])
        r["t1"] = r["t0"] if r["t1"] is None else float(r["t1"])
    return runs


def _segments_in_span(segments, t0, t1):
    """Inferred (root, quality) tuples whose midpoint falls in [t0, t1]."""
    out = []
    for root, q, s0, s1 in segments:
        mid = 0.5 * (s0 + s1)
        if t0 <= mid <= t1:
            out.append((root, q))
    return out


def _segment_at(segments, t):
    """Inferred (root, quality) whose [t0,t1] contains t (nearest-midpoint fallback)."""
    for root, q, s0, s1 in segments:
        if s0 <= t < s1:
            return root, q
    if not segments:
        return -1, ""
    best = min(segments, key=lambda s: abs(0.5 * (s[2] + s[3]) - t))
    return best[0], best[1]


# ── Signal 1: repeat consistency + per-instance z-score ─────────────────────────
def _repeat_consistency_with_z(labels, fps):
    """Return (within, cross, z_outlier, bridge_contrast, per_instance).

    within/cross: aggregate mean cosine over same/diff-label instance pairs.
    bridge_contrast = within − cross.
    per_instance: list of (own_sim, xmatch, z) aligned with ``labels``.  ``z`` is
    the standardised own_sim over all *repeated*-label instances (a slipped repeat
    is a strong negative outlier).  z_outlier = min z (nan if <2 repeated
    instances or zero spread).
    """
    n = len(labels)
    S = np.array([[float(a @ b) for b in fps] for a in fps]) if n else np.zeros((0, 0))
    within, cross = [], []
    for i in range(n):
        for j in range(i + 1, n):
            (within if labels[i] == labels[j] else cross).append(S[i, j])
    w = float(np.mean(within)) if within else float("nan")
    c = float(np.mean(cross)) if cross else float("nan")
    bridge = (w - c) if (within and cross) else float("nan")

    counts = {lbl: labels.count(lbl) for lbl in set(labels)}
    own = [float("nan")] * n
    xmatch = [float("nan")] * n
    for i in range(n):
        sibs = [S[i, j] for j in range(n) if j != i and labels[j] == labels[i]]
        others = [S[i, j] for j in range(n) if labels[j] != labels[i]]
        if sibs:
            own[i] = float(np.mean(sibs))
        if others:
            xmatch[i] = float(np.max(others))

    # standardise own_sim over instances whose label repeats (>=2)
    rep_idx = [i for i in range(n) if counts[labels[i]] >= 2 and not np.isnan(own[i])]
    z = [float("nan")] * n
    z_outlier = float("nan")
    if len(rep_idx) >= 2:
        vals = np.array([own[i] for i in rep_idx])
        mu, sd = float(vals.mean()), float(vals.std())
        if sd > 1e-6:
            for i in rep_idx:
                z[i] = (own[i] - mu) / sd
            z_outlier = float(min(z[i] for i in rep_idx))
    per_instance = list(zip(own, xmatch, z))
    return w, c, z_outlier, bridge, per_instance


# ── Signal 2: boundary agreement (elastic F1) ───────────────────────────────────
def _chart_boundary_bars(result_chords):
    """Interior chart boundary bars = bars where the section label changes."""
    bnds, last = [], None
    for ch in result_chords:
        lbl = ch.get("section", "?")
        if last is not None and lbl != last:
            bnds.append(int(ch.get("bar", 0)))
        last = lbl
    return sorted(set(bnds))


def _boundary_f1(chart_bnds, inferred_bnds, tol=2):
    if not chart_bnds or not inferred_bnds:
        return float("nan")
    matched_c = sum(any(abs(cb - ib) <= tol for ib in inferred_bnds) for cb in chart_bnds)
    matched_i = sum(any(abs(ib - cb) <= tol for cb in chart_bnds) for ib in inferred_bnds)
    rec = matched_c / len(chart_bnds)
    prec = matched_i / len(inferred_bnds)
    return 0.0 if (rec + prec) == 0 else 2 * rec * prec / (rec + prec)


# ── Signal 3: per-section family fraction (recomputed from inferred content) ─────
def _section_family_frac(section, segments):
    """(#root-or-family match)/#chords between a section's GT chords and the
    inferred content mapped under each chord's time span."""
    hits, tot = 0, 0
    for ch in section["chords"]:
        gt_pc, gt_fam = _parse_display_label(ch.get("label", ""))
        if gt_pc < 0:
            continue
        tot += 1
        t = 0.5 * (float(ch.get("t0") or section["t0"]) + float(ch.get("t1") or section["t1"]))
        inf_pc, inf_q = _segment_at(segments, t)
        if inf_pc == gt_pc:            # exact root → exact or family
            hits += 1
        elif inf_pc >= 0 and _family(inf_q) == gt_fam:
            hits += 0.5                # right family, wrong root (partial credit)
    return (hits / tot) if tot else float("nan")


# ── top-level ────────────────────────────────────────────────────────────────────
def validate_alignment(
    result,
    inferred,
    chart=None,
    *,
    beats_per_bar: int = 4,
    inferred_beats=None,
    weights=(0.70, 0.30),
    thresholds=(0.75, 0.55),
    contrast_gate: float = 0.04,
    z_gate: float = -2.5,
    slip_gate: float = 0.15,
    family_dip: float = 0.30,
    family_floor: float = 0.55,
) -> AlignmentValidation:
    """Judge whether an iReal↔inferred alignment is structurally coherent.

    Args:
        result: ``IRealbAlignmentResult`` (needs ``.chords`` with section/label/t0/t1/bar).
        inferred: inferred chord content — list of ``InferredChord`` /
            ``{root|pc, quality|lv, t0, t1}`` dicts / ``(root, q, t0, t1)`` tuples.
        chart: optional ``MMAChart`` (unused for the core signals; reserved).
        beats_per_bar: bars↔beats conversion for Signal 2.
        inferred_beats: optional whole-song per-beat ``(root_rel, q_idx)`` sequence
            for Signal 2's inferred-boundary detection.  If ``None``, Signal 2 abstains.
        weights: ``(w_family, w_boundary)`` — the aggregate score is the inferred↔GT
            family agreement (mean over sections) plus, when available, boundary_f1.
            NB: this deviates from the design skeleton's ``sig(repeat_consistency)``
            term.  Calibration (docs/mission_6_phase2_results.md) showed a bridge-
            contrast term in the *score* falsely penalises clean low-contrast tunes
            (30%→5% FP when removed); repeat_consistency stays as the abstain gate
            and the localiser's basis, not a positive score term.  Likewise the
            aggregate uses MEAN family, not min — a single slipped section is √N-
            diluted in the aggregate *by design* and is caught by the per-instance
            outlier downgrade below, exactly the premise-check's headline finding.
        thresholds: ``(hi, lo)`` — ``score>=hi`` OK, ``>=lo`` SUSPECT, else MISALIGNED.
        contrast_gate: below this bridge contrast, slip is undetectable → UNVERIFIABLE.
        z_gate: per-instance z below this confirms a localised slip (reported; the
            firing decision leans on ``slip_gate`` which is robust to inference noise).
        slip_gate: ``xmatch − own_sim`` above this flags a localised slip — a section
            that resembles *another* section more than its own siblings.  This is the
            load-bearing localiser: a genuine swap sends it ~0.5+, inference noise
            keeps it negative, giving clean separation.
        family_dip: a section whose family_frac is this far below the median is suspect
            (only if also below ``family_floor``, to survive per-section noise).
        family_floor: absolute family_frac below which a section may be flagged.
    """
    segments = _as_segments(inferred)
    sections = _section_instances(result.chords)

    labels = [s["label"] for s in sections]
    fps = [_fingerprint(_segments_in_span(segments, s["t0"], s["t1"])) for s in sections]
    within, cross, z_outlier, bridge, per_inst = _repeat_consistency_with_z(labels, fps)
    rep_cons = bridge

    # Signal 3
    fam_fracs = [_section_family_frac(s, segments) for s in sections]
    valid_fam = [f for f in fam_fracs if not np.isnan(f)]
    min_fam = min(valid_fam) if valid_fam else float("nan")
    median_fam = float(np.median(valid_fam)) if valid_fam else float("nan")

    # Signal 2
    chart_bnds = _chart_boundary_bars(result.chords)
    if inferred_beats is not None and len(inferred_beats):
        inf_beat_bnds = detect_section_boundaries(build_chord_ssm(list(inferred_beats)),
                                                  beats_per_bar=beats_per_bar)
        inf_bar_bnds = sorted(set(b // beats_per_bar for b in inf_beat_bnds))
    else:
        inf_bar_bnds = []
    bf1 = _boundary_f1(chart_bnds, inf_bar_bnds)

    # assemble per-section reports
    reports = []
    for s, fp, ff, (own, xm, zz) in zip(sections, fps, fam_fracs, per_inst):
        reports.append(SectionReport(
            label=s["label"], instance=s["instance"], bar0=s["bar0"], bar1=s["bar1"],
            t0=s["t0"], t1=s["t1"], family_frac=ff,
            n_chords=len(s["chords"]), own_sim=own, xmatch=xm, z=zz, fingerprint=fp))

    notes: list[str] = []

    # ── Gate 1: bridge-contrast abstain (Signal 1 blind when sections are similar) ─
    no_signal1 = np.isnan(rep_cons)
    if no_signal1:
        notes.append("no repeated section / <2 labels; Signal 1 unavailable")
    if np.isnan(bf1):
        notes.append("no inferred boundaries; Signal 2 abstained")

    if (no_signal1 or rep_cons < contrast_gate) and np.isnan(bf1):
        why = ("low bridge contrast; slip undetectable"
               if not no_signal1 else "no repeats + no inferred boundaries")
        notes.append(why)
        return AlignmentValidation(
            align_score=float("nan"), verdict="UNVERIFIABLE",
            repeat_consistency=rep_cons, z_outlier=z_outlier, boundary_f1=bf1,
            min_section_family_frac=min_fam if valid_fam else float("nan"),
            suspect_sections=[], bridge_contrast=rep_cons, sections=reports, notes=notes)

    # ── localisation, gated on bridge contrast ────────────────────────────────────
    # Primary localiser = the cross-match slip signature (xmatch − own_sim): a
    # slipped instance resembles some OTHER section more than its own siblings.
    # This is robust to inference noise (noise lowers own_sim but not xmatch, so
    # the difference stays negative), unlike a raw z-outlier which noise can trip.
    suspect: list[str] = []
    contrast_ok = (not np.isnan(rep_cons)) and rep_cons >= contrast_gate
    if contrast_ok:
        for r in reports:
            slip_fires = (not np.isnan(r.xmatch) and not np.isnan(r.own_sim)
                          and (r.xmatch - r.own_sim) > slip_gate)
            # z confirms in concert with a (weaker) cross-match tilt …
            z_confirms = ((not np.isnan(r.z)) and r.z < z_gate
                          and not np.isnan(r.xmatch) and not np.isnan(r.own_sim)
                          and (r.xmatch - r.own_sim) > slip_gate * 0.5)
            # … or a *very* strong pure own-fit outlier (a swap whose donor shares
            # a root family with the victim's label leaves no cross-match tilt, but
            # the within-label agreement still collapses far past any noise level).
            z_strong = (not np.isnan(r.z)) and r.z < 3.0 * z_gate / 2.5  # z < -3.0
            if slip_fires or z_confirms or z_strong:
                suspect.append(r.name)
        # degenerate same-label group fully flagged → keep the best cross-matcher
        by_label: dict[str, list[SectionReport]] = {}
        for r in reports:
            by_label.setdefault(r.label, []).append(r)
        for lbl, grp in by_label.items():
            flagged = [r for r in grp if r.name in suspect]
            if len(grp) >= 2 and len(flagged) == len(grp):
                keep = max(flagged, key=lambda r: (r.xmatch if not np.isnan(r.xmatch) else -1))
                for r in flagged:
                    if r.name != keep.name:
                        suspect.remove(r.name)
    # Signal 3 localised family dip: only a section that is BOTH well below the
    # song median AND below an absolute floor (survives per-section noise).
    if not np.isnan(median_fam):
        for r in reports:
            if (not np.isnan(r.family_frac) and r.family_frac < family_floor
                    and r.family_frac < median_fam - family_dip
                    and r.name not in suspect):
                suspect.append(r.name)

    # ── aggregate score + verdict ─────────────────────────────────────────────────
    # Aggregate coherence = mean inferred↔GT family agreement (+ boundary_f1 when
    # available).  A GLOBAL slip drives every section's family down → low score →
    # MISALIGNED.  A LOCALISED slip barely moves the mean (√N dilution) → the
    # per-instance outlier downgrade below is what catches it.
    w_fam, w_bnd = weights
    mean_fam = float(np.mean(valid_fam)) if valid_fam else float("nan")
    parts, wsum = 0.0, 0.0
    if valid_fam:
        parts += w_fam * mean_fam; wsum += w_fam
    if not np.isnan(bf1):
        parts += w_bnd * bf1; wsum += w_bnd
    score = float(parts / wsum) if wsum > 0 else float("nan")

    hi, lo = thresholds
    if np.isnan(score):
        verdict = "UNVERIFIABLE"
    elif score >= hi:
        verdict = "OK"
    elif score >= lo:
        verdict = "SUSPECT"
    else:
        verdict = "MISALIGNED"

    # Uniform family FLOOR = global fault (wrong transpose / whole-chorus slip):
    # every section's inferred content contradicts its GT even though the sections
    # stay internally self-consistent (so Signal 1 alone is fooled).  The design's
    # disambiguation branch: uniform-low family + (high or n/a) repeat_consistency
    # ⇒ MISALIGNED.  min_family alone can be masked by a strong Signal 1, so gate
    # on the MEDIAN section family here, not the worst.
    if not np.isnan(median_fam) and median_fam < 0.45:
        if verdict != "MISALIGNED":
            notes.append("uniform low family fraction → global fault (transpose/chorus slip)")
        verdict = "MISALIGNED"

    # a localised slip can leave the aggregate high (√N dilution) — if a strong
    # per-instance outlier fired, flag at least SUSPECT so slip-recall tracks it.
    if suspect and verdict == "OK":
        verdict = "SUSPECT"
        notes.append("localised outlier fired though aggregate score high")

    return AlignmentValidation(
        align_score=score, verdict=verdict, repeat_consistency=rep_cons,
        z_outlier=z_outlier, boundary_f1=bf1,
        min_section_family_frac=min_fam if valid_fam else float("nan"),
        suspect_sections=suspect, bridge_contrast=rep_cons,
        sections=reports, notes=notes)
