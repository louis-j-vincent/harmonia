"""
Symbolic SECTION alignment for Harmonia (Phase 2) — phase-search for constant-BPM rigid sections.

Goal
----
Align rigid chart sections (A1, A2, B, C, ...) onto a per-beat inferred chord stream,
assuming constant BPM (~181) and zero gaps (back-to-back sections). Use unweighted
root-match rate (binary per-bar, no confidence weighting) as the metric.

**CORRECTED DESIGN (2026-07-14)** after Phase 1 validation:
  - Song structure: AABC (not vamp-heavy)
  - Constant BPM throughout (no per-section fitting)
  - Metric: unweighted root-match = (# bars with root match) / (total bars)
  - Phase 1 error: B offset +12 bars (from confidence-weighted metrics masking true ~43% accuracy)
  - Phase 2 insight: B's low match% is inference quality, not misalignment

**Why this is NOT an OT/Hungarian problem:**
With constant BPM and rigid back-to-back sections, the problem reduces to 1D PHASE SEARCH:
  - Unknown: t_bar0 (time of bar 0)
  - Constraint: all bar times are locked to each other via constant BPM
  - Result: section positions are trivial (A1 at t_bar0, A2 at t_bar0+10.6s, etc.)
  - Optimization: exhaustive search over candidate t_bar0 values

No assignment problem (sections are ordered); no OT needed (not a mass transport problem).
The algorithm is phase search + soft temporal assignment (Gaussian weighting to neighboring bars).

Literature anchors:
  - Soft-DTW: Cuturi & Blondel (2017), soft temporal alignment via entropic-OT
  - Phase recovery in beat tracking: Ellis (2007), Ellis & Masataka (2012)
  - Tempo tracking with constant BPM: Goto (2001), McVicar et al. (2011)
"""

from __future__ import annotations

import json
import math
import os
import sys
from dataclasses import dataclass, asdict
from typing import Optional, Sequence

# Allow `python scripts/optimal_section_alignment.py` (sys.path[0] = scripts/).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from harmonia.tab_aligner import _parse_ireal


# ─────────────────────────────────────────────────────────────────────────────
# Data contracts
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class ChartSection:
    """One rigid chart section, e.g. A1 = bars 0..7."""
    idx: int
    label: str                       # "A", "B", "C", ...
    bar_lo: int
    bar_hi: int                      # inclusive
    chords: list[tuple[int, str]]    # [(root_pc, quality), ...] one per bar

    @property
    def n_bars(self) -> int:
        return self.bar_hi - self.bar_lo + 1

    @property
    def roots(self) -> list[int]:
        return [pc for pc, _ in self.chords]


@dataclass
class InferredChord:
    """One inferred audio chord (per-beat or per-segment)."""
    t_start: float
    t_end: float
    root_pc: int
    quality: str
    confidence: float = 1.0

    @property
    def midpoint(self) -> float:
        return 0.5 * (self.t_start + self.t_end)


@dataclass
class SectionAlignment:
    """Alignment result for one section at a given phase."""
    idx: int
    label: str
    bar_lo: int
    bar_hi: int
    t_start: float              # expected start time at this phase
    t_end: float                # expected end time
    root_match_count: int       # unweighted: how many bars had matching root
    root_match_rate: float      # root_match_count / n_bars, in [0,1]
    per_bar_match: list[bool]   # True if bar matched, False otherwise
    fitted_bpm: float           # fitted BPM from actual duration / expected duration
    residual_phase: float       # RMS residual beat-phase deviation (seconds)


@dataclass
class PhaseSearchResult:
    """Result of phase search."""
    best_phase_s: float         # best t_bar0 (seconds)
    total_score: float          # total root-match rate across all sections
    mean_score: float           # average per-section root-match rate
    sections: list[SectionAlignment]
    fitted_bpm_global: float    # global BPM fitted from all sections
    metrics: dict               # diagnostics


# ─────────────────────────────────────────────────────────────────────────────
# Loaders
# ─────────────────────────────────────────────────────────────────────────────

def load_chart_sections(chart_json: dict) -> list[ChartSection]:
    """Load chart sections from a sectionwise.json or explicit schema."""
    secs = chart_json.get("sections", [])
    flat = chart_json.get("chords")

    # group flat chords by section id
    by_sec: dict[int, list[tuple[int, str]]] = {}
    if flat:
        for c in flat:
            sid = c.get("section_id", c.get("section"))
            root = c.get("root_pc")
            if "label" in c:
                _, q = _parse_ireal(c["label"])
            else:
                q = c.get("quality", "")
            by_sec.setdefault(sid, []).append((root if root is not None else -1, q))

    out: list[ChartSection] = []
    for s in secs:
        idx = s.get("idx", len(out))
        chords = _coerce_chords(s.get("chords")) if s.get("chords") else by_sec.get(idx, [])
        out.append(ChartSection(
            idx=idx,
            label=str(s.get("label", "?")),
            bar_lo=int(s["bar_lo"]),
            bar_hi=int(s["bar_hi"]),
            chords=chords,
        ))
    return out


def _coerce_chords(raw) -> list[tuple[int, str]]:
    out = []
    for c in raw:
        if isinstance(c, str):
            out.append(_parse_ireal(c))
        elif isinstance(c, (list, tuple)):
            out.append((int(c[0]), str(c[1]) if len(c) > 1 else ""))
        elif isinstance(c, dict):
            if "label" in c:
                out.append(_parse_ireal(c["label"]))
            else:
                out.append((int(c.get("root_pc", -1)), str(c.get("quality", ""))))
    return out


def normalize_inferred(raw: Sequence) -> list[InferredChord]:
    """Normalise a heterogeneous inferred-chord stream to InferredChord list."""
    out: list[InferredChord] = []
    for c in raw:
        if isinstance(c, InferredChord):
            out.append(c); continue
        t0 = _first(c, ["t_start", "t0", "start", "t0_perfect", "t0_orig"])
        t1 = _first(c, ["t_end", "t1", "end", "t1_perfect"])
        if "label" in c and "root_pc" not in c:
            pc, q = _parse_ireal(c["label"])
        else:
            pc = int(_first(c, ["root_pc", "root"], default=-1))
            q = str(_first(c, ["quality", "qual"], default=""))
        conf = float(_first(c, ["conf", "confidence", "prob"], default=1.0))
        if t0 is None:
            continue
        if t1 is None:
            t1 = t0 + 0.33  # ~one beat at 181 BPM if missing
        out.append(InferredChord(float(t0), float(t1), pc, q, conf))
    out.sort(key=lambda x: x.t_start)
    return out


def _first(d: dict, keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


# ─────────────────────────────────────────────────────────────────────────────
# Core phase-search algorithm
# ─────────────────────────────────────────────────────────────────────────────

def unweighted_root_match(
    section: ChartSection,
    inferred: list[InferredChord],
    t_bar0: float,
    bar_dur_s: float,
    τ: float = 0.5,
) -> tuple[int, list[bool]]:
    """Unweighted root-match for a section at phase t_bar0.

    For each bar in the section, check if ANY inferred chord in that bar's
    time window has a matching root (unweighted — no confidence involved).

    τ is the Gaussian softness parameter (seconds) for soft temporal assignment:
      - τ=0: hard grid snapping (only the closest bar counts)
      - τ>0: soft assignment (nearby bars contribute with Gaussian decay)

    Returns (total_matches, per_bar_match_list).
    """
    matches = []
    for bar_idx, chart_root in enumerate(section.roots):
        if chart_root < 0:  # N.C.
            continue

        # Expected bar time
        bar_i = section.bar_lo + bar_idx
        t_bar_start = t_bar0 + bar_i * bar_dur_s
        t_bar_end = t_bar0 + (bar_i + 1) * bar_dur_s

        # Hard assignment: inferred chords overlapping the bar
        if τ == 0.0:
            # Hard: is there an overlapping inferred chord with matching root?
            found = False
            for ic in inferred:
                if ic.t_end >= t_bar_start and ic.t_start <= t_bar_end:
                    if ic.root_pc == chart_root:
                        found = True
                        break
            matches.append(found)
        else:
            # Soft: accumulate weighted contribution from nearby inferred chords
            match_mass = 0.0
            for ic in inferred:
                if ic.root_pc == chart_root:
                    # Gaussian weight based on distance to bar midpoint
                    bar_mid = 0.5 * (t_bar_start + t_bar_end)
                    dist = abs(ic.midpoint - bar_mid)
                    weight = math.exp(-(dist ** 2) / (2 * τ ** 2))
                    match_mass += weight
            # Threshold at 0.5 (more than half contribution = match)
            matches.append(match_mass >= 0.5)

    return sum(matches), matches


def compute_section_alignment(
    section: ChartSection,
    inferred: list[InferredChord],
    t_bar0: float,
    bpm_prior: float,
    beats_per_bar: int = 4,
    τ: float = 0.5,
) -> SectionAlignment:
    """Compute alignment for one section at phase t_bar0."""
    bar_dur_s = 60.0 / (bpm_prior / beats_per_bar)

    match_count, per_bar = unweighted_root_match(section, inferred, t_bar0, bar_dur_s, τ)
    rate = match_count / len(per_bar) if per_bar else 0.0

    # Expected duration and fitted BPM
    t_start = t_bar0 + section.bar_lo * bar_dur_s
    t_end = t_bar0 + (section.bar_hi + 1) * bar_dur_s
    actual_dur = t_end - t_start
    fitted_bpm = 60.0 * section.n_bars / (actual_dur / beats_per_bar) if actual_dur > 0 else bpm_prior

    # Residual phase: RMS deviation of inferred-chord midpoints from expected bar grid
    residuals = []
    for bar_idx, chart_root in enumerate(section.roots):
        if chart_root < 0:
            continue
        bar_i = section.bar_lo + bar_idx
        bar_mid_expected = t_bar0 + (bar_i + 0.5) * bar_dur_s
        for ic in inferred:
            if ic.t_start <= bar_mid_expected <= ic.t_end:
                residuals.append(abs(ic.midpoint - bar_mid_expected))
                break
    residual_phase = math.sqrt(sum(r**2 for r in residuals) / len(residuals)) if residuals else 0.0

    return SectionAlignment(
        idx=section.idx,
        label=section.label,
        bar_lo=section.bar_lo,
        bar_hi=section.bar_hi,
        t_start=t_start,
        t_end=t_end,
        root_match_count=match_count,
        root_match_rate=round(rate, 3),
        per_bar_match=per_bar,
        fitted_bpm=round(fitted_bpm, 1),
        residual_phase=round(residual_phase, 4),
    )


def search_best_phase(
    chart_sections: list[ChartSection],
    inferred: list[InferredChord],
    bpm_prior: float,
    beats_per_bar: int = 4,
    search_range: tuple[float, float] = (0.0, 10.0),
    search_step: float = 0.05,
    τ: float = 0.5,
) -> PhaseSearchResult:
    """Exhaustive phase search for the best t_bar0.

    Search over [search_range[0], search_range[1]] in steps of search_step.
    """
    candidates = [i * search_step for i in range(int((search_range[1] - search_range[0]) / search_step) + 1)]
    candidates = [search_range[0] + c for c in candidates]

    best_phase, best_score = None, -1.0
    best_sections = None

    for t_bar0 in candidates:
        # Compute alignment for each section at this phase
        alignments = []
        total_score = 0.0
        for section in chart_sections:
            align = compute_section_alignment(section, inferred, t_bar0, bpm_prior, beats_per_bar, τ)
            alignments.append(align)
            total_score += align.root_match_rate

        if total_score > best_score:
            best_score, best_phase = total_score, t_bar0
            best_sections = alignments

    if best_sections is None:
        best_sections = [compute_section_alignment(s, inferred, search_range[0], bpm_prior, beats_per_bar, τ)
                         for s in chart_sections]

    # Fitted global BPM from all sections combined
    bar_dur_s = 60.0 / (bpm_prior / beats_per_bar)
    total_bars = sum(s.n_bars for s in chart_sections)
    total_dur = (best_sections[-1].t_end - best_sections[0].t_start) if best_sections else 0.0
    fitted_bpm_global = 60.0 * total_bars / (total_dur / beats_per_bar) if total_dur > 0 else bpm_prior

    return PhaseSearchResult(
        best_phase_s=round(best_phase, 3) if best_phase is not None else 0.0,
        total_score=round(best_score, 3),
        mean_score=round(best_score / len(chart_sections), 3) if chart_sections else 0.0,
        sections=best_sections,
        fitted_bpm_global=round(fitted_bpm_global, 1),
        metrics={
            "search_range": search_range,
            "search_step": search_step,
            "soft_tau": τ,
            "n_candidate_phases": len(candidates),
        },
    )


# ─────────────────────────────────────────────────────────────────────────────
# Public entry point
# ─────────────────────────────────────────────────────────────────────────────

def align_sections_optimal(
    chart_json: dict,
    inferred_chords,
    bpm_prior: float,
    phase1_results: Optional[dict] = None,
    *,
    beats_per_bar: Optional[int] = None,
    search_range: tuple[float, float] = (0.0, 10.0),
    search_step: float = 0.05,
    soft_tau: float = 0.5,
) -> dict:
    """Phase-search alignment for constant-BPM rigid sections.

    Parameters
    ----------
    chart_json : dict       Chart with sections (bar ranges) and chords.
    inferred_chords : list  Per-beat inferred chords (heterogeneous ok).
    bpm_prior : float       Nominal BPM (e.g., 181 for Autumn Leaves).
    phase1_results : dict   Unused (for compatibility); phase search is independent.
    beats_per_bar : int     Default 4.
    search_range : tuple    Phase search window in seconds (default 0–10s).
    search_step : float     Phase search granularity (default 0.05s → 200 candidates).
    soft_tau : float        Gaussian softness for temporal assignment (0=hard grid, >0=soft).

    Returns
    -------
    dict with sections, metrics, and diagnostics in a sectionwise.json-compatible format.
    """
    bpb = beats_per_bar or int(chart_json.get("beats_per_bar", 4))
    sections = load_chart_sections(chart_json)
    inferred = normalize_inferred(inferred_chords)
    audio_end = float(chart_json.get("audio_end_s", inferred[-1].t_end if inferred else 0.0))

    # Phase search
    result = search_best_phase(
        sections, inferred, bpm_prior, beats_per_bar=bpb,
        search_range=search_range, search_step=search_step, τ=soft_tau,
    )

    return {
        "algorithm": "phase_search_constant_bpm",
        "solver": "exhaustive_1d_phase_search",
        "metric": "unweighted_root_match_rate",
        "bpm_prior": bpm_prior,
        "bpm_fitted_global": result.fitted_bpm_global,
        "beats_per_bar": bpb,
        "audio_end_s": audio_end,
        "best_phase_s": result.best_phase_s,
        "total_root_match_score": result.total_score,
        "mean_root_match_rate": result.mean_score,
        "sections": [asdict(s) for s in result.sections],
        "metrics": result.metrics,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Smoke test
# ─────────────────────────────────────────────────────────────────────────────

def _smoke():
    import os
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = os.path.join(here, "docs/plots/annotations/irealb_autumn_leaves_sectionwise.json")
    chart = json.load(open(path))

    # Synthesise an inferred stream from the chart's perfect grid
    # (circular by construction; real Phase 1 inferred chords replace this)
    inferred = []
    for c in chart["chords"]:
        t0 = c.get("t0_perfect", c.get("t0_orig"))
        t1 = c.get("t1_perfect", (t0 + 0.33) if t0 else None)
        if t0 is None:
            continue
        inferred.append({
            "t_start": t0, "t_end": t1,
            "root_pc": c["root_pc"], "quality": _parse_ireal(c["label"])[1],
            "conf": 1.0,
        })

    out = align_sections_optimal(chart, inferred, bpm_prior=chart["bpm_prior"], soft_tau=0.5)
    print(f"Algorithm: {out['algorithm']}")
    print(f"Solver: {out['solver']}")
    print(f"Metric: {out['metric']}")
    print(f"Best phase: {out['best_phase_s']}s")
    print(f"Total root-match score: {out['total_root_match_score']}")
    print(f"Mean per-section rate: {out['mean_root_match_rate']}")
    print(f"Fitted global BPM: {out['bpm_fitted_global']}")
    print()
    for s in out["sections"]:
        print(f"  {s['label']:<2} bars {s['bar_lo']:<2}-{s['bar_hi']:<2}  "
              f"[{s['t_start']:<6.1f}–{s['t_end']:<6.1f}s]  "
              f"root-match {s['root_match_rate']:.2f} ({s['root_match_count']}/{len(s['per_bar_match'])}) "
              f"fitted_bpm {s['fitted_bpm']}")
    return out


if __name__ == "__main__":
    _smoke()
