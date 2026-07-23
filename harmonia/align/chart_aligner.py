"""harmonia.align.chart_aligner — the productionized chart-to-audio ALIGNER seam.

This is the refactor-lane-compatible *brick* for the chart→audio aligner. The
concurrent refactor uses an ABC + factory + dataclass-at-every-seam; the
chart-to-audio aligner is the one stage with no home yet, and the fusion DBN
(``harmonia.align.fusion.align_fusion``) is its drop-in implementation. This
module inserts that DBN into the refactor's architecture WITHOUT touching any of
its files: our brick, their conventions, a pure adapter in between.

The two public dataclasses (``SectionMarker`` / ``ChartAlignment``) mirror the
section-marker payload the serving layer already consumes:

  * ``harmonia.serving.loaders._load_ireal_alignment`` reads a per-chord
    ``{i, bar, beat, section, label, t0, t1, match, conf}`` sidecar;
  * ``scripts/harmonia_server.py``'s section-align route writes
    ``markers=[{t0, t1, mma, sectionIdx}]`` + ``sections=[{idx, label, bar0,
    bar1, t0, t1, acceptedWithModel, ...}]`` from
    ``align_tune_sections_to_audio``.

``SectionMarker`` carries the UNION of the per-chord fields both consumers need
(``t0, t1, label, mma, section_idx, bar, conf``) so a serving route can build
EITHER payload from a ``ChartAlignment`` with no information loss — that is the
"lossless adapter" contract. The one representational note (see
``_to_chart_alignment``): the fusion output carries the shipped-schema chord
*label* (e.g. ``"C:maj7"``), not the raw iReal ``mma`` token, so ``mma`` is set
to the label. Serving uses ``mma`` for display only and ``_load_ireal_alignment``
reads ``label`` directly, so this is lossless for both call sites.

The aligner itself is an ABC (``ChartAligner``) with one concrete implementation
today (``FusionChartAligner``). ``FusionChartAligner.align`` is a THIN wrapper:
it calls ``align_fusion`` and hands the resulting ``FusionAlignment`` to the pure,
audio-free adapter ``_to_chart_alignment``. No numerics live in the wrapper — the
adapter is a total function of the ``FusionAlignment``, so it is unit-testable
with a hand-built alignment and is guaranteed lossless vs ``align_fusion`` (the
validation harness diffs markers + downbeat + confidence byte-for-byte).

BOUNDARY. This module lives entirely in OUR lane. It imports the stable
``harmonia.align.fusion`` read-only and creates no import cycle. The refactor's
serving/stages/core files are untouched; the ONE serving-side change that would
route the live section-align page through this aligner is a coordinate-with-Louis
item, kill-switched (see ``docs/fusion_aligner_design.md`` 2026-07-24b).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from harmonia.align.fusion import (
    FusionAlignment,
    Placement,
    align_fusion,
)

_METER = 4


# ═════════════════════════════════════════════════════════════════════════════
# Seam dataclasses — the section-marker payload shape (refactor convention)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class SectionMarker:
    """One placed chart chord on the audio timeline — the per-chord unit of the
    serving payload.

    Carries the UNION of the fields the two serving consumers need so either
    payload is derivable with no loss:
      * server section-align markers: ``{t0, t1, mma, sectionIdx}``  (``mma`` <-
        ``label``, ``sectionIdx`` <- ``section_idx``);
      * ``_load_ireal_alignment`` per-chord: ``{i, bar, beat, section, label,
        t0, t1, match, conf}`` (``i``/``beat``/``match`` are derived at
        payload-build time from the ordered marker list; ``section``/``bar``/
        ``label``/``t0``/``t1``/``conf`` come straight from here).
    """
    t0: float
    t1: float
    label: str                # shipped-schema chord label, e.g. "C:maj7"
    mma: str                  # iReal-token slot; == label (see module docstring)
    section_idx: int          # order index of the containing section (serving sectionIdx)
    bar: int                  # global lattice bar index (monotone; (bar, beat) is a per-song key)
    conf: float               # per-chord confidence (inherits the section posterior confidence)


@dataclass
class ChartAlignment:
    """A chart aligned to an audio recording — the ``ChartAligner`` output seam.

    ``sections`` is the serving-facing list: one dict per placed chart section,
    each ``{label, t0, t1, markers: [SectionMarker], confidence}``. The remaining
    fields carry the fusion DBN's timing grid + the self-detection signals the
    dataset gate and the waveform UI consume.

    ``downbeat_confidence`` / ``downbeat_flagged`` expose the fusion downbeat's
    posterior (the ``FusionAlignment.downbeat`` reconciliation) so a consumer
    (e.g. the dataset harvest's beat-lock fold) can read the productionized DBN's
    confidence WITHOUT reaching back into the ``FusionAlignment`` — the adapter
    stays the single source of truth for the seam.
    """
    sections: list[dict]                     # [{label, t0, t1, markers:[SectionMarker], confidence}]
    beats: np.ndarray                        # constant-tempo lattice (s)
    downbeat_phase: int                      # chosen global downbeat phase (0..meter-1)
    tempo_bpm: float                         # 60 / beat_period
    whole_song_confidence: float             # pooled self-detection confidence [0,1]
    low_confidence_regions: list[dict]       # [{label, t0, t1, confidence, reliability, reason}]
    downbeat_confidence: float = 0.0         # fusion downbeat posterior confidence [0,1]
    downbeat_flagged: bool = False           # fusion downbeat precision-first flag
    meter: int = _METER
    diag: dict = field(default_factory=dict)


# ═════════════════════════════════════════════════════════════════════════════
# The aligner ABC + the fusion implementation
# ═════════════════════════════════════════════════════════════════════════════

class ChartAligner(ABC):
    """Abstract chart-to-audio aligner (refactor ABC + factory convention).

    ``align`` takes the source ``audio`` (a path) and a ``chart`` (a brick0
    BATCH1-style song-cfg dict: ``song_id`` + the iReal chart reference
    ``ireal_file`` / ``tune_title`` + optional timing hints ``human_anchor`` /
    ``onset_nudge`` / ``scored_end``) and returns a :class:`ChartAlignment`.
    ``features`` may pre-supply the expensive per-song extracts (see
    ``harmonia.align.fusion.extract_features``) to skip re-extraction.
    """

    @abstractmethod
    def align(self, audio, chart: dict, *,
              features: Optional[dict] = None) -> ChartAlignment:
        ...


class FusionChartAligner(ChartAligner):
    """The fusion-DBN aligner: ``align_fusion`` + the pure lossless adapter.

    Thin by design — ``align`` runs the DBN and delegates ALL shaping to
    ``_to_chart_alignment``, which is a total function of the ``FusionAlignment``
    (no audio, no re-computation). Optional fusion knobs (``w_bass`` / ``w_hr`` /
    ``salience_weighting``) pass through so a caller can reproduce a specific
    fusion configuration; the defaults are the frozen-reproduction operating
    point (harmony-dominant), unchanged from ``align_fusion``.
    """

    def __init__(self, *, w_bass: Optional[float] = None,
                 w_hr: Optional[float] = None,
                 salience_weighting: bool = True):
        self.w_bass = w_bass
        self.w_hr = w_hr
        self.salience_weighting = salience_weighting

    def align(self, audio, chart: dict, *,
              features: Optional[dict] = None) -> ChartAlignment:
        kw: dict = dict(features=features, salience_weighting=self.salience_weighting)
        if self.w_bass is not None:
            kw["w_bass"] = self.w_bass
        if self.w_hr is not None:
            kw["w_hr"] = self.w_hr
        fa = align_fusion(audio, chart, **kw)
        return _to_chart_alignment(fa)


# ═════════════════════════════════════════════════════════════════════════════
# The pure adapter — FusionAlignment -> ChartAlignment (audio-free, lossless)
# ═════════════════════════════════════════════════════════════════════════════

def _marker_bar(t0: float, phase: float, period: float) -> int:
    """Global lattice bar index of a marker starting at ``t0``. Chords are laid
    on integer beats of the constant-tempo lattice (origin ``phase``, spacing
    ``period``), so ``round((t0 - phase) / period)`` recovers the beat index
    exactly (sub-beat offset is a global phase slide already folded into the
    lattice); ``// meter`` gives the bar. Monotone in ``t0`` => ``(bar, beat)``
    is a per-song-unique chord address, matching ``_load_ireal_alignment``."""
    if period <= 0:
        return 0
    beat_idx = int(round((t0 - phase) / period))
    return max(beat_idx, 0) // _METER


def _placement_window(pl: Placement, beats: np.ndarray, period: float
                      ) -> tuple[float, float]:
    """Time window [t0, t1) of a placement's beat span on the lattice, with the
    same end-of-lattice extrapolation ``build_gt_chords`` uses (beats past the
    grid extend at the constant period)."""
    n = len(beats)

    def _bt(idx: int) -> float:
        if idx < n:
            return float(beats[idx])
        return float(beats[-1]) + (idx - n + 1) * period if n else idx * period

    return _bt(pl.start_beat), _bt(pl.start_beat + pl.n_beats)


def _assign_section(c0: float, c1: float, windows: list[tuple[float, float]]) -> int:
    """Index of the placement a chord [c0, c1] belongs to: the window it OVERLAPS
    most (a chord at a section boundary goes to the section it mostly sits in). A
    chord that overlaps nothing (a merge that landed a span in a gap) goes to the
    nearest window by centre. Guarantees a total assignment => a chord<->marker
    bijection (no double-count on abutting windows, no drop in a gap)."""
    best_i, best_ov = -1, 0.0
    for i, (w0, w1) in enumerate(windows):
        ov = min(c1, w1) - max(c0, w0)
        if ov > best_ov:
            best_ov, best_i = ov, i
    if best_i >= 0:
        return best_i
    mid = 0.5 * (c0 + c1)
    return int(np.argmin([abs(mid - 0.5 * (w0 + w1)) for w0, w1 in windows]))


def _to_chart_alignment(fa: FusionAlignment) -> ChartAlignment:
    """Pure adapter: map a ``FusionAlignment`` to a ``ChartAlignment`` losslessly.

    Every placement becomes one section; each ``gt_chord`` is assigned to the
    placement whose beat window it OVERLAPS most (``_assign_section``), so the
    chords partition exactly over the sections — a chord<->marker bijection with
    no double-count where two sections abut and no drop for a boundary/merged
    span. Section ``t0/t1`` are the first/last marker times — exactly how the
    server derives a section's span. The per-chord ``conf`` inherits the
    placement's posterior ``confidence`` (the fusion confidence is per-section,
    the honest granularity). The downbeat phase/confidence/flag, the whole-song
    confidence, and the low-confidence regions pass straight through. No audio,
    no re-computation — a total function of ``fa``.
    """
    beats = np.asarray(fa.beats, float)
    period = float(fa.period)
    phase = float(fa.phase)
    chords = sorted(fa.gt_chords, key=lambda c: float(c["t0"]))
    windows = [_placement_window(pl, beats, period) for pl in fa.placements]

    markers_by_sec: list[list[SectionMarker]] = [[] for _ in fa.placements]
    for c in chords:
        t0, t1 = float(c["t0"]), float(c["t1"])
        if not windows:
            break
        si = _assign_section(t0, t1, windows)
        lbl = c.get("label", "N")
        markers_by_sec[si].append(SectionMarker(
            t0=t0, t1=t1, label=lbl, mma=lbl, section_idx=si,
            bar=_marker_bar(t0, phase, period),
            conf=round(float(fa.placements[si].confidence), 4)))

    sections: list[dict] = []
    for si, pl in enumerate(fa.placements):
        markers = markers_by_sec[si]
        sec_t0 = markers[0].t0 if markers else None
        sec_t1 = markers[-1].t1 if markers else None
        sections.append(dict(
            label=pl.label, t0=sec_t0, t1=sec_t1, markers=markers,
            confidence=round(float(pl.confidence), 4)))

    db = fa.downbeat
    tempo_bpm = 60.0 / period if period > 0 else 0.0
    return ChartAlignment(
        sections=sections,
        beats=beats,
        downbeat_phase=int(db.phase),
        tempo_bpm=round(float(tempo_bpm), 3),
        whole_song_confidence=round(float(fa.whole_song_confidence), 4),
        low_confidence_regions=list(fa.low_confidence_regions),
        downbeat_confidence=round(float(db.confidence), 4),
        downbeat_flagged=bool(db.flagged),
        meter=int(getattr(db, "meter", _METER)),
        diag=dict(song_id=fa.song_id, transpose=fa.transpose,
                  n_sections=len(sections),
                  n_markers=sum(len(s["markers"]) for s in sections),
                  downbeat_form_phase=int(db.form_phase),
                  downbeat_acoustic_phase=int(db.acoustic_phase),
                  downbeat_anticipation=bool(db.anticipation)))


__all__ = ["SectionMarker", "ChartAlignment", "ChartAligner", "FusionChartAligner"]
