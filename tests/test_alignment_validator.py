"""Red-first synthetic tests for the Mission-6 alignment validator.

Five controlled cases exercise the three signals and the two premise-check
refinements (per-instance z-score localiser + bridge-contrast abstain gate):

  1. clean aligned form            → OK,          z_outlier is nan (no outlier)
  2. global slip (label remap)     → MISALIGNED,  z_outlier nan (global fault)
  3. localised slip (1 instance)   → OK/SUSPECT,  z_outlier < -2, suspect=[victim]
  4. low bridge-contrast form      → UNVERIFIABLE, bridge_contrast < gate
  5. phase offset (content shift)  → SUSPECT/MISALIGNED, family drops

All synthetic: 1 bar = 1 chord = 1 second, no audio.
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from harmonia.models.alignment_validator import validate_alignment
from harmonia.tab_aligner import _ROOTS


# ── synthetic case builder ──────────────────────────────────────────────────────
def build_case(form, gt_content, inf_content=None, dt=1.0):
    """Build (result, inferred_segments) from a symbolic form.

    form: list of section labels, e.g. ["A", "A", "B", "A"].
    gt_content: {label: [(root_pc, quality_str), ...]} — the GT chords (one per bar).
    inf_content: optional {label: [...]} for the *inferred* content (defaults to
        gt_content — a perfectly aligned song).  Pass a different mapping to slip.
    """
    inf_content = inf_content or gt_content
    chords, segs = [], []
    bar, t = 0, 0.0
    for lbl in form:
        gt = gt_content[lbl]
        inf = inf_content[lbl]
        for k in range(len(gt)):
            g_root, g_q = gt[k]
            chords.append({"label": _ROOTS[g_root] + g_q, "section": lbl,
                           "bar": bar, "t0": t, "t1": t + dt, "match": "exact"})
            i_root, i_q = inf[k]
            segs.append((i_root, i_q, t, t + dt))
            bar += 1
            t += dt
    return SimpleNamespace(chords=chords), segs


# distinct A / B / C content (disjoint roots → low cross similarity)
A_CONTENT = [(0, "maj7"), (5, "7"), (7, "min7"), (2, "7")]      # C  F7  Gm7 D7
B_CONTENT = [(1, "min7"), (3, "7"), (8, "maj7"), (10, "min7")]  # Dbm7 Eb7 Abmaj7 Bbm7
C_CONTENT = [(4, "maj7"), (9, "7"), (11, "min7"), (6, "7")]     # E  A7  Bm7 F#7


def test_clean_aligned_is_ok():
    form = ["A", "A", "B", "A"]
    content = {"A": A_CONTENT, "B": B_CONTENT}
    result, segs = build_case(form, content)
    v = validate_alignment(result, segs)

    assert v.verdict == "OK", (v.verdict, v.align_score, v.notes)
    assert v.suspect_sections == []
    assert math.isnan(v.z_outlier)              # all A instances identical → no outlier
    assert v.repeat_consistency > 0.04          # real bridge contrast
    assert v.min_section_family_frac > 0.99


def test_global_slip_is_misaligned():
    # Global label remap: every A instance carries B content and vice-versa.
    # within-label agreement survives (all A's identical) so Signal 1 sees nothing,
    # but every section's inferred content contradicts its GT → family collapses.
    form = ["A", "A", "B", "A"]
    gt = {"A": A_CONTENT, "B": B_CONTENT}
    inf = {"A": B_CONTENT, "B": A_CONTENT}
    result, segs = build_case(form, gt, inf)
    v = validate_alignment(result, segs)

    assert v.verdict == "MISALIGNED", (v.verdict, v.align_score, v.min_section_family_frac)
    assert math.isnan(v.z_outlier)              # global, not a single outlier
    assert v.min_section_family_frac < 0.2


def test_localized_slip_localises_via_zscore():
    # Multi-chorus interleaved form (A,B,C × 6 = the realistic looped-video case,
    # instances separated so each A is its own run); corrupt the 4th A (A#4) with
    # C content — an outlier that must localise via the per-instance z-score.
    form = ["A", "B", "C"] * 6
    content = {"A": A_CONTENT, "B": B_CONTENT, "C": C_CONTENT}
    result, segs = build_case(form, content)

    # A#4 is the 4th "A" run: chorus index 3 → sections [9,10,11] → A is section 9,
    # spanning bars 9*4 .. 9*4+3.
    victim_bar0 = (3 * 3) * 4
    for k, (r, q) in enumerate(C_CONTENT):
        b = victim_bar0 + k
        t = float(b)
        segs[b] = (r, q, t, t + 1.0)

    v = validate_alignment(result, segs)

    assert v.verdict in ("OK", "SUSPECT"), v.verdict
    assert "A#4" in v.suspect_sections, v.suspect_sections
    assert v.z_outlier < -2.0, v.z_outlier      # sharp per-instance outlier


def test_low_contrast_is_unverifiable():
    # A and B are harmonically identical → within == cross → slip undetectable.
    form = ["A", "A", "B", "A"]
    content = {"A": A_CONTENT, "B": A_CONTENT}   # B == A
    result, segs = build_case(form, content)
    v = validate_alignment(result, segs)

    assert v.verdict == "UNVERIFIABLE", (v.verdict, v.bridge_contrast)
    assert v.bridge_contrast < 0.04
    assert math.isnan(v.align_score)


def test_phase_offset_drops_family_and_flags():
    # Shift ALL inferred content earlier by 2 bars: every GT chord now sees the
    # wrong inferred chord.  Sections stay internally consistent (Signal 1 fine,
    # z nan) but family_frac collapses → SUSPECT/MISALIGNED.
    form = ["A", "A", "B", "A"]
    content = {"A": A_CONTENT, "B": B_CONTENT}
    result, segs = build_case(form, content)

    clean = validate_alignment(result, segs)

    shifted = segs[2:] + segs[:2]               # rotate content by 2 bars
    shifted = [(r, q, float(i), float(i) + 1.0) for i, (r, q, _, _) in enumerate(shifted)]
    v = validate_alignment(result, shifted)

    assert v.verdict in ("SUSPECT", "MISALIGNED"), v.verdict
    assert v.min_section_family_frac < clean.min_section_family_frac
