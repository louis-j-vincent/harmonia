"""harmonia/stages/chord_head.py — Phase 3 ChordHead PORT (nnls24 live path).

A clean, config-driven decomposition of the ~30-flag ``infer_chords_v1`` chord
stage. This module owns the *chord stage* only: it consumes the already-built
beat grid (from ``stages/beat_grid``) plus a config object, and produces the
per-segment chord labels, the coalesced ChordChart spans, and the symbolic
section pass. Everything upstream (audio load, beat tracking, best-fit period)
is the beat_grid stage's job and is NOT repeated here.

═══════════════════════════════════════════════════════════════════════════════
THE SEAM (read-only refs into harmonia/models/chord_pipeline_v1.py)
═══════════════════════════════════════════════════════════════════════════════
The nnls24 chord stage is the self-contained function ``_infer_nnls24``
(chord_pipeline_v1.py L3509-3985). ``infer_chords_v1`` builds the beat grid
(L4241-4322: ``y,sr = sf.read`` → beatthis → ``_bestfit_beat_period`` → circular
-mean phase → uniform ``bt``) and then, for ``feature_frontend=="nnls24"``,
dispatches to ``_infer_nnls24`` and returns early (L4337-4343). So the chord
stage BEGINS at that dispatch: its inputs are exactly
``(audio_path, bt, tempo_bpm, duration_s, period, seventh_gate, *frontends)``,
and it ENDS at the returned ``ChordChart``.

This ChordHead ports the DETERMINISTIC, net-covered core of that stage — the
three intermediates the extended parity net freezes (see
``harmonia/eval/parity.py::_nnls24_stages`` and known_issues "PARITY-NET nnls24
EXTENSION"):

  1. ``features``     — VAMP bothchroma (cached) → per-beat feat24 → root head
                        posterior → global key.
  2. ``precoalesce``  — root-change segmentation → per-segment (root, quality,
                        bass) via ``_label_segments`` → ``_coalesce_labeled``.
                        THE load-bearing chord-port golden (pre-coalesce AND
                        coalesced labels).
  3. ``sections``     — the SYMBOLIC bar-locked section pass at anchor 0
                        (``_pool_root_proba_to_bars`` + barlocked SSM).

The pure chord-stage logic (segmentation, per-segment labeling, coalesce,
fifth-correction, the bar-pooling and barlocked gating) is RE-IMPLEMENTED here
so the port is a genuine, reviewable move rather than a thin wrapper; it is then
proven byte-identical to the live code and the frozen goldens
(tests/test_chord_head_parity.py). The genuine model primitives — the VAMP
plugin, the trained MLP heads, the music-x-lab subprocess, the barlocked SSM,
and key inference — are imported as leaves (they are "the model", not the
god-function's inlined orchestration).

═══════════════════════════════════════════════════════════════════════════════
STEP B (2026-07-26) — THIS MODULE IS NOW THE SOLE IMPLEMENTATION
═══════════════════════════════════════════════════════════════════════════════
``chord_pipeline_v1._infer_nnls24`` no longer contains any chord-stage logic: it
is a ~30-line adapter that maps its positional signature onto a
``ChordHeadConfig`` and calls :meth:`NNLS24ChordHead.run_full`. The ~440-line
inline body (and the ``HARMONIA_CHORDHEAD`` kill-switch that used to select
between the two) are deleted. Everything the inline path did now lives here:

  * the audio tail — flux / Beat This! ``sota_downbeat_phase`` grid anchor,
    native bar grid, ``_flux_anchored_bar_root``, ``_section_fallback``, the
    Occam post-pass, the music-x-lab 2-chord-per-bar split, onset hints,
    ``_finalize_chords`` and the ``ChordChart`` assembly;
  * every front-end combo, not just the frozen oracle — ``segment_source="musx"``
    (music-x-lab boundary segmentation), the music-x-lab-unavailable degradation
    to the pure NNLS-24 heads, the raw-energy ``_nnls_no_chord_segs`` gate that
    fires when no music-x-lab N mask is available, and the heads-missing
    single-chord fallback;
  * the five ``progress_cb`` call sites (``key`` → ``draft`` → per-fold
    ``chords`` → ``sections`` → final ``chords``) with identical payloads and
    ordering (``beats`` is fired by ``infer_chords_v1`` itself, upstream).

Still imported as LEAVES (deliberately — these are "the model", and re-deriving
their float math would risk drift): the VAMP plugin, the trained MLP heads, the
music-x-lab subprocess, ``barlocked_sections``, key inference, Beat This!, the
flux comb, ``_apply_occam_to_coalesced``, the calibration map, the 2-chord split
and the onset hints.

RESIDUAL DUPLICATION (deliberate, one rung left — Louis's, coordinated with the
parity lane): the small pure helpers ``_root_change_segs`` / ``_label_segments``
/ ``_coalesce_labeled`` / ``_drop_leading_outlier`` / ``_finalize_chords`` /
``_fifth_corrected_quality`` / ``_pool_root_proba_to_bars`` /
``_barlocked_sections_or_none`` still exist BOTH here and in
``chord_pipeline_v1``, because ``harmonia/eval/parity.py::_nnls24_stages``
(the 3rd copy of the sequence), ``harmonia/models/jam_mode.py`` and two tests
import them from there. They are gated identical by the parity net every run.
Collapsing them belongs to the same move that collapses ``parity._nnls24_stages``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# The nnls24 stage's operational logging used to be emitted under the
# ``harmonia.models.chord_pipeline_v1`` logger (it lived inside that module).
# It now emits under ``harmonia.stages.chord_head``.  The only name-scoped
# consumer in the tree is serving/api.py's ``_CatchRejections`` handler, which
# filters for "rejected" / "partially applied" — both produced by
# ``pool_beat_evidence`` on the BP48 path, which is still in chord_pipeline_v1.
# So no consumer loses a message it was reading.
logger = logging.getLogger(__name__)

# ── constants (re-implemented from chord_pipeline_v1 so this module is
#    independent of the contended core file) ─────────────────────────────────
NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
NO_CHORD_LABEL = "N"
_NNLS_Q_TO_HARTE = {
    "maj": "maj", "min": "min", "dom": "7", "hdim": "hdim7",
    "dim": "dim", "aug": "aug", "sus": "sus4",
}


# ═══════════════════════════════════════════════════════════════════════════
# Config object — the ~30 infer_chords_v1 flags, named
# ═══════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class ChordHeadConfig:
    """Named config for the nnls24 chord stage.

    Only the flags ``_infer_nnls24`` actually consumes are live here. The other
    ~24 ``infer_chords_v1`` kwargs (``theta``, ``cell``, ``use_beat_seq``,
    ``use_ctx_model``, ``use_joint_decode``, ``use_semi_markov``, the
    ``joint_*`` / ``semi_markov_*`` / ``llm_*`` families, ``user_constraints`` …)
    are BP48-path only: the nnls24 branch returns BEFORE any of them is read
    (chord_pipeline_v1.py L4337-4343), so they are intentionally absent from
    this config rather than carried as dead fields.

    ``ChordHeadConfig.live_defaults()`` tracks the SHIPPED live path.  Since
    2026-07-27 it NO LONGER equals ``benchmark_set.LIVE_ORACLE_KWARGS``: the
    live default moved ``segment_source`` "nnls" -> "musx_redecode", while
    LIVE_ORACLE_KWARGS stays pinned to the 2026-07-22 config its committed
    goldens were captured under.  Anything gating those goldens must construct
    its config from LIVE_ORACLE_KWARGS explicitly, not from ``live_defaults()``.
    """

    # ── chord-stage front-end selectors (the live-path knobs) ──
    feature_frontend: str = "nnls24"     # this head IS the nnls24 head
    bass_frontend: str = "musx"          # sounding-bass source: "nnls24" | "musx"
    quality_frontend: str = "musx"       # root+quality source: "nnls24" | "musx"
    # Segmentation source (chord-CHANGE timing).  DEFAULT FLIPPED 2026-07-27 to
    # "musx_redecode" — see the ``label_stage`` docstring and
    # ``harmonia/models/musx_redecode.py`` for the evidence (+2.20 pp partial on
    # the 7-song frozen benchmark; Louis's A/B verdict "'our cuts' is really
    # worse than the other three, with 'model, ON + timing fix' clearly ahead").
    #   "musx_redecode" — beat-aware, latency-compensated re-decode of
    #                     music-x-lab's FRAME posteriors (== the A/B page's
    #                     "persistence ON + timing fix" lane).  Degrades to
    #                     "nnls" on any failure, NEVER to "musx".
    #   "nnls"          — per-beat NNLS root-argmax flip (the A/B page's
    #                     "our cuts" lane; the pre-2026-07-27 default).
    #   "musx"          — raw music-x-lab .lab boundaries.  MEASURED WORSE
    #                     (−1.97 pp, 2026-07-26): its boundaries are +113 ms
    #                     late.  Kept only for reproducing that refutation.
    # Rollback without a code change: HARMONIA_ANALYZE_SEGSOURCE=nnls (server) or
    # HARMONIA_MUSX_REDECODE=0 (hard kill switch, any caller).
    segment_source: str = "musx_redecode"

    # ── consumed-but-inert on the nnls24 path (kept for interface fidelity) ──
    seventh_gate: float = 0.0            # DEAD on nnls24 (quality via head/musx, not gated)
    audio_domain: str = "real"          # not read in the _infer_nnls24 body

    # ── section pass (only ``section_mode`` affects the net-covered symbolic pass) ──
    section_mode: str = "barlocked"     # env HARMONIA_SECTION_MODE
    section_min_duration_s: float = 20.0

    # ── audio-path tail knobs (follow-up; NOT reproduced in this slice) ──
    grid_anchor: str = "flux"           # env HARMONIA_GRID_ANCHOR
    grid_anchor_sota: str = "on"        # env HARMONIA_GRID_ANCHOR_SOTA
    native_bargrid: bool = False        # env HARMONIA_NATIVE_BARGRID
    occam_postpass: bool = True         # env HARMONIA_OCCAM_POSTPASS
    musx_2chord_bar: bool = True        # env HARMONIA_MUSX_2CHORD_BAR
    musx_onset_hint: bool = True        # env HARMONIA_MUSX_ONSET_HINT

    # BP48-only flags observed at call time but ignored here (audit trail only).
    ignored_bp48_flags: tuple[str, ...] = field(default_factory=tuple)

    @property
    def want_musx(self) -> bool:
        return self.bass_frontend == "musx" or self.quality_frontend == "musx"

    @classmethod
    def live_defaults(cls) -> "ChordHeadConfig":
        """The SHIPPED live production defaults (segment_source="musx_redecode"
        since 2026-07-27).  NOT the frozen parity oracle any more — see the
        class docstring; golden-gated code must use LIVE_ORACLE_KWARGS."""
        return cls()

    @classmethod
    def from_infer_kwargs(cls, **kwargs) -> "ChordHeadConfig":
        """Build from a raw ``infer_chords_v1(**kwargs)`` call.

        Picks the chord-stage-relevant subset; records the BP48-only flags that
        were passed (non-default) in ``ignored_bp48_flags`` for auditability
        rather than silently dropping them.
        """
        relevant = {
            "feature_frontend", "bass_frontend", "quality_frontend",
            "segment_source", "seventh_gate", "audio_domain",
        }
        kw = {k: kwargs[k] for k in relevant if k in kwargs}
        ignored = tuple(sorted(k for k in kwargs if k not in relevant
                               and k not in ("progress_cb", "cache_dir",
                                             "beat_backend", "beat_period_mode")))
        return cls(ignored_bp48_flags=ignored, **kw)


# ═══════════════════════════════════════════════════════════════════════════
# Result container
# ═══════════════════════════════════════════════════════════════════════════
@dataclass
class ChordStageResult:
    """Outputs of the net-covered chord stage (one nnls24 decode)."""

    # features
    arr: np.ndarray                 # (T, 24) cached VAMP bothchroma
    times: np.ndarray               # (T,) frame times
    feat: np.ndarray                # (n_beats, 24) C-frame, L2-per-half
    beat_proba: np.ndarray          # (n_beats, 12) per-beat root posterior
    key_name: str
    key_confidence: float

    # pre-coalesce / coalesced labels
    segs: list[tuple[int, int]]
    seg_bounds: list[tuple[float, float]]
    labeled: list[tuple[float, float, str, float]]   # PRE-coalesce (t0,t1,label,conf)
    coalesced: list[list]                            # [t0,t1,label,conf*dur,dur]

    # symbolic sections
    bar_root: np.ndarray            # (n_bars, 12)
    sections: list[dict] | None     # barlocked or None (defer to acoustic fallback)


# ═══════════════════════════════════════════════════════════════════════════
# ChordHead — the ported nnls24 chord stage
# ═══════════════════════════════════════════════════════════════════════════
class NNLS24ChordHead:
    """The nnls24 chord stage as a clean component.

    Usage (chord stage only — beat grid comes from the beat_grid stage):

        head = NNLS24ChordHead(ChordHeadConfig.live_defaults())
        res = head.run(audio_path, bt, period, duration_s)
        res.labeled      # pre-coalesce chord labels
        res.coalesced    # coalesced spans
        res.sections     # symbolic bar-locked sections
    """

    def __init__(self, config: ChordHeadConfig | None = None):
        self.config = config or ChordHeadConfig.live_defaults()

    # ── stage 1: features ────────────────────────────────────────────────────
    def extract_features(self, audio_path: Path, bt: np.ndarray):
        """Cached VAMP bothchroma → per-beat feat24 → root posterior → key.

        Mirrors ``_infer_nnls24`` L3566-3576 exactly. Leaf primitives
        (``extract_bothchroma``, ``pool_beats``, the trained ``root_proba``
        head, ``infer_key``) are imported, not re-implemented.
        """
        from harmonia.models import nnls_features as nf
        from harmonia.theory.key_profiles import infer_key

        heads = nf.get_heads()
        if heads is None:
            raise RuntimeError("nnls24_heads.npz unavailable — cannot run ChordHead")

        arr, times = nf.extract_bothchroma(audio_path)   # cache HIT (stem-keyed)
        feat = nf.pool_beats(arr, times, bt)             # (n_beats, 24) C-frame
        beat_proba = heads.root_proba(feat)              # (n_beats, 12)
        key_result = infer_key(feat[:, 12:].sum(0))
        return heads, arr, times, feat, beat_proba, key_result

    # ── stage 2: segmentation + per-segment labels + coalesce ────────────────
    @staticmethod
    def _root_change_segs(beat_proba: np.ndarray) -> list[tuple[int, int]]:
        """Cut wherever the per-beat root argmax changes (gmerge).

        Faithful re-implementation of chord_pipeline_v1._root_change_segs
        (L1832-1848).
        """
        pred = beat_proba.argmax(1)
        n = len(pred)
        if n == 0:
            return []
        cuts = [0] + [b for b in range(1, n) if pred[b] != pred[b - 1]] + [n]
        return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]

    @staticmethod
    def _fifth_corrected_quality(sev_h: str, root: int, treble: np.ndarray,
                                 margin: float = 1.3, min_energy: float = 0.12) -> str:
        """min7<->hdim7 / min<->dim tie-break on the direct 5th-bin evidence.

        Faithful re-implementation of chord_pipeline_v1._fifth_corrected_quality
        (L3357-3398).
        """
        pairs = {"min7": "hdim7", "hdim7": "min7", "min": "dim", "dim": "min"}
        if sev_h not in pairs:
            return sev_h
        p5 = float(treble[(root + 7) % 12])
        b5 = float(treble[(root + 6) % 12])
        lo, hi = min(p5, b5), max(p5, b5)
        if hi < min_energy or lo <= 0 or hi / lo < margin:
            return sev_h
        wants_dim_side = b5 > p5
        is_dim_side = sev_h in ("hdim7", "dim")
        if wants_dim_side == is_dim_side:
            return sev_h
        return pairs[sev_h]

    def _label_segments(
        self, segs, seg_bounds, beat_proba, feat, bass_half, heads,
        musx_seg_rq=None, musx_seg_bass=None, seg_no_chord=None,
    ) -> list[tuple[float, float, str, float]]:
        """Per-segment (root, quality, bass) → label, one pass.

        Faithful re-implementation of chord_pipeline_v1._label_segments
        (L3401-3447). ``heads.quality_idx`` / ``mxb.routed_bass_pc`` are leaves.
        """
        from harmonia.models import musx_bass as mxb

        if seg_no_chord is None:
            seg_no_chord = np.zeros(len(seg_bounds), dtype=bool)
        labeled: list[tuple[float, float, str, float]] = []
        for i, (s, e) in enumerate(segs):
            if seg_no_chord[i]:
                t0, t1 = seg_bounds[i]
                labeled.append((t0, t1, NO_CHORD_LABEL, 0.0))
                continue
            p_seg = beat_proba[s:e].sum(0)
            nnls_root = int(p_seg.argmax())
            seg_feat = feat[s:e].mean(0, keepdims=True)
            root, sev_h, conf = None, None, None
            if musx_seg_rq is not None:
                mx_root, mx_sev = musx_seg_rq[i]
                if mx_root >= 0 and mx_sev is not None:
                    root, sev_h = mx_root, mx_sev
                    conf = float(p_seg[root] / max(p_seg.sum(), 1e-9))
            if root is None:
                root = nnls_root
                q_idx = int(heads.quality_idx(seg_feat, np.array([root]))[0])
                sev_h = _NNLS_Q_TO_HARTE.get(heads.qualities[q_idx], "maj")
                conf = float(p_seg[root] / max(p_seg.sum(), 1e-9))
            sev_h = self._fifth_corrected_quality(sev_h, root, seg_feat[0, 12:])
            nnls_bass = int(bass_half[s:e].sum(0).argmax())
            if musx_seg_bass is not None:
                bass_pc = mxb.routed_bass_pc(int(musx_seg_bass[i]), nnls_bass, root)
            else:
                bass_pc = nnls_bass
            label = f"{NOTE[root]}:{sev_h}"
            if bass_pc != root:
                label += f"/{NOTE[bass_pc]}"
            t0, t1 = seg_bounds[i]
            labeled.append((t0, t1, label, conf))
        return labeled

    @staticmethod
    def _coalesce_labeled(
        labeled: list[tuple[float, float, str, float]],
    ) -> list[list]:
        """Merge adjacent equal labels; duration-weighted confidence sum.

        Faithful re-implementation of chord_pipeline_v1._coalesce_labeled
        (L3450-3465).
        """
        coalesced: list[list] = []
        for t0, t1, label, conf in labeled:
            dur = max(t1 - t0, 1e-9)
            if coalesced and coalesced[-1][2] == label:
                p = coalesced[-1]
                p[1] = t1
                p[3] += conf * dur
                p[4] += dur
            else:
                coalesced.append([t0, t1, label, conf * dur, dur])
        return coalesced

    def label_stage(self, audio_path, bt, feat, beat_proba, heads, *,
                    arr=None, times=None, period=None, key_name=None,
                    progress_cb=None):
        """Segmentation → musx per-segment lookups → per-segment labels → coalesce.

        The FINAL pass of the nnls24 chord stage, for EVERY front-end combo
        (not just the frozen oracle):

          * ``segment_source`` — ``"musx_redecode"`` (DEFAULT since 2026-07-27:
            beat-aware + latency-compensated re-decode of music-x-lab's frame
            posteriors, used as BOTH the boundary source and the musx label
            source), ``"nnls"`` (per-beat root-change) or ``"musx"`` (raw
            music-x-lab change times snapped to beats — measured −1.97 pp,
            kept only to reproduce that refutation).  Every musx variant
            degrades silently to the NNLS segs on any failure;
          * ``bass_frontend`` / ``quality_frontend`` — ``"nnls24"`` or ``"musx"``;
            a music-x-lab failure degrades silently to the pure NNLS-24 heads
            (this must never crash the server path — CLAUDE.md #6);
          * the raw-energy ``_nnls_no_chord_segs`` gate, which replaces the
            per-segment N mask whenever music-x-lab did NOT supply the
            root/quality front-end (i.e. ``musx_seg_rq is None``).

        ``progress_cb`` (server path only) additionally fires:
          * ``"draft"`` — a full pure-NNLS chart computed BEFORE the slow
            music-x-lab call, so the UI can show a rough chart in a few seconds;
          * ``"chords"`` once per music-x-lab ensemble fold as the folds land
            (skipped on a musx cache hit — there is nothing to poll).
        Both need ``arr``/``times``/``period``/``key_name``; both are
        best-effort (any failure is swallowed and just skips the callback).
        """
        from harmonia.models import musx_bass as mxb
        from harmonia.models.chord_pipeline_v1 import (
            _fit_harmonic_grid, _get_nnls24_conf_map, _musx_boundary_segs,
            _nnls_no_chord_segs,
        )

        cfg = self.config
        n_beats = len(feat)

        # segmentation: per-beat root-change on the harmonic grid (the default)
        grid = _fit_harmonic_grid(beat_proba)
        segs = self._root_change_segs(beat_proba)
        logger.debug("nnls24: %d-beat grid, %d root-change segs", grid, len(segs))

        # music-x-lab's frame posteriors, re-decoded on OUR beat grid with a
        # per-song latency correction (DEFAULT since 2026-07-27 — see
        # ChordHeadConfig.segment_source and harmonia/models/musx_redecode.py).
        # Fills ``mx_labels`` here so the ``cfg.want_musx`` block below reuses
        # the SAME timeline for root/quality/bass/no-chord instead of paying for
        # a second (subprocess) music-x-lab run: this is the validated config,
        # where the re-decode is both the boundary source and the label source.
        # Any failure degrades to the NNLS root-change segs + the raw .lab.
        mx_labels = None
        # Did the re-decode actually drive this decode?  Read by the Occam gate
        # in ``run_full``, which is only validated under this segmentation.
        self._used_redecode = False
        if cfg.segment_source == "musx_redecode":
            try:
                from harmonia.models import musx_redecode as mxr
                if not mxr.enabled():
                    raise RuntimeError("HARMONIA_MUSX_REDECODE=0 (kill switch)")
                _lab, _lat = mxr.redecode_audio(
                    audio_path, bt, latency_grid=mxr.latency_grid_from_env())
                _rsegs = _musx_boundary_segs(_lab, bt, n_beats)
                if _rsegs:
                    logger.info("nnls24: segmentation from music-x-lab RE-DECODE "
                                "(%d segs; was %d NNLS root-change; latency "
                                "%.0f ms)", len(_rsegs), len(segs), _lat * 1000)
                    segs = _rsegs
                    mx_labels = _lab
                    self._used_redecode = True
            except Exception as exc:  # pragma: no cover - env-dependent
                logger.warning("nnls24: musx re-decode unavailable (%s); keeping "
                               "NNLS root-change segs + raw .lab labels", exc)

        # Legacy opt-in: replace the per-beat-argmax root-change segmentation
        # with music-x-lab's OWN raw .lab chord-change times (snapped to the
        # nearest beat).  MEASURED WORSE end-to-end (−1.97 pp, 2026-07-26) —
        # its boundaries are systematically +113 ms late, which is exactly what
        # the re-decode path above corrects.  Any failure degrades silently to
        # the NNLS root-change segs.
        if cfg.segment_source == "musx":
            try:
                _mx = mxb.musx_labels(audio_path)
                _msegs = _musx_boundary_segs(_mx, bt, n_beats)
                if _msegs:
                    logger.info("nnls24: segmentation from music-x-lab boundaries "
                                "(%d segs; was %d NNLS root-change)",
                                len(_msegs), len(segs))
                    segs = _msegs
            except Exception as exc:  # pragma: no cover - env-dependent
                logger.warning("nnls24: musx-boundary segmentation failed (%s); "
                               "keeping NNLS root-change segs", exc)

        bass_half = feat[:, :12]
        seg_bounds = [(float(bt[s]), float(bt[min(e, len(bt) - 1)]))
                      for (s, e) in segs]

        # ── Draft pass (progress_cb only): pure-NNLS labels, no music-x-lab ───
        # Runs BEFORE the (slow, ~10-30s) music-x-lab call below.  Reuses the
        # exact same per-segment logic as the final pass (``_label_segments``
        # with no musx_* overrides), so draft and final can never silently
        # diverge in mechanism — only in which inputs were available when each
        # ran.  See docs/inference_pipeline_timing_and_animation_scope.md.
        if progress_cb is not None:
            try:
                _draft_no_chord = _nnls_no_chord_segs(arr, times, bt, segs)
                _draft_labeled = self._label_segments(
                    segs, seg_bounds, beat_proba, feat, bass_half, heads,
                    seg_no_chord=_draft_no_chord)
                _draft_coalesced = self._drop_leading_outlier(
                    self._coalesce_labeled(_draft_labeled), period)
                _draft_chords, _ = self._finalize_chords(
                    _draft_coalesced, period, key_name, _get_nnls24_conf_map())
                progress_cb("draft", {"chords": _draft_chords})
            except Exception:  # noqa: BLE001 — draft preview is best-effort
                logger.warning("nnls24: progress_cb('draft') failed", exc_info=True)

        # music-x-lab is loaded ONCE and shared by the bass front-end (rule F)
        # and the root/quality front-end — both are midpoint lookups over the
        # same .lab.
        # (``mx_labels`` may ALREADY hold the re-decoded timeline — set above by
        # the "musx_redecode" segmentation branch.  Do not clobber it.)
        musx_seg_bass = musx_seg_rq = None
        # No-chord (N) mask, one bool per segment.  Primary source = music-x-lab's
        # explicit "N"/"X" token (trustworthy); fallback = the raw-NNLS energy
        # gate below.  A True entry becomes a first-class N.C. cell (empty
        # render, confidence 0) instead of an invented NNLS-argmax chord
        # (known_issues.md 2026-07-19 ★ CHORDS / NO-CHORD).
        seg_no_chord = np.zeros(len(seg_bounds), dtype=bool)

        def _chords_from_musx_labels(mx_labels_snapshot):
            """One snapshot of music-x-lab labels (a single ensemble fold, or the
            final 5-fold average) -> a finalized chords_out list, via the exact
            same per-segment mechanism used everywhere else."""
            m_bass = (mxb.bass_pc_per_segment(mx_labels_snapshot, seg_bounds)
                      if cfg.bass_frontend == "musx" else None)
            m_rq = (mxb.root_quality_per_segment(mx_labels_snapshot, seg_bounds)
                    if cfg.quality_frontend == "musx" else None)
            m_no_chord = mxb.no_chord_per_segment(mx_labels_snapshot, seg_bounds)
            _labeled = self._label_segments(
                segs, seg_bounds, beat_proba, feat, bass_half, heads,
                musx_seg_rq=m_rq, musx_seg_bass=m_bass, seg_no_chord=m_no_chord)
            _co = self._drop_leading_outlier(
                self._coalesce_labeled(_labeled), period)
            _chords, _ = self._finalize_chords(
                _co, period, key_name, _get_nnls24_conf_map())
            return _chords

        def _musx_fold_progress(fold_i, n_folds, fold_labels):
            if progress_cb is None:
                return
            try:
                progress_cb("chords", {"chords": _chords_from_musx_labels(fold_labels),
                                       "fold": fold_i, "n_folds": n_folds})
            except Exception:  # noqa: BLE001 — fold preview is best-effort
                logger.warning("nnls24: fold-progress callback failed", exc_info=True)

        if cfg.want_musx:
            try:
                if mx_labels is None:                 # not already re-decoded
                    mx_labels = mxb.musx_labels(      # cache HIT (stem-keyed)
                        audio_path,
                        progress_cb=(_musx_fold_progress if progress_cb is not None
                                     else None))
                if cfg.bass_frontend == "musx":
                    musx_seg_bass = mxb.bass_pc_per_segment(mx_labels, seg_bounds)
                if cfg.quality_frontend == "musx":
                    musx_seg_rq = mxb.root_quality_per_segment(mx_labels, seg_bounds)
                seg_no_chord = mxb.no_chord_per_segment(mx_labels, seg_bounds)
                if seg_no_chord.any():
                    logger.warning("nnls24: music-x-lab marks %d/%d segments as "
                                   "no-chord (N) — rendering as N.C.",
                                   int(seg_no_chord.sum()), len(seg_bounds))
                logger.info("nnls24: music-x-lab active (%d segs; bass=%s quality=%s)",
                            len(seg_bounds), cfg.bass_frontend == "musx",
                            cfg.quality_frontend == "musx")
            except Exception as exc:  # pragma: no cover - env-dependent
                logger.warning("nnls24: music-x-lab unavailable (%s); "
                               "falling back to NNLS-24 heads", exc)
                musx_seg_bass = None
                musx_seg_rq = None

        # NNLS-only no-chord gate: when music-x-lab supplied no N mask (clone
        # absent or the quality front-end is the in-house heads), fall back to
        # the raw-energy detector so a chordless intro still renders empty
        # instead of an invented argmax chord.  Skipped entirely when
        # music-x-lab's own N is available (the trustworthy source).
        if musx_seg_rq is None:
            seg_no_chord = _nnls_no_chord_segs(arr, times, bt, segs)
            if seg_no_chord.any():
                logger.warning("nnls24: raw-energy N gate flags %d/%d segments as "
                               "no-chord (musx-N unavailable) — rendering as N.C.",
                               int(seg_no_chord.sum()), len(segs))

        labeled = self._label_segments(
            segs, seg_bounds, beat_proba, feat, bass_half, heads,
            musx_seg_rq=musx_seg_rq, musx_seg_bass=musx_seg_bass,
            seg_no_chord=seg_no_chord)
        # coalesce adjacent same-label segments; confidence aggregates as the
        # duration-weighted mean of the segment scores.
        coalesced = self._coalesce_labeled(labeled)
        return segs, seg_bounds, labeled, coalesced, mx_labels

    # ── stage 3: symbolic bar-locked sections (anchor 0) ─────────────────────
    @staticmethod
    def _pool_root_proba_to_bars(
        beat_proba: np.ndarray, bt: np.ndarray, period: float,
        beats_per_bar: int = 4, anchor_beats: int = 0,
    ):
        """Per-beat root posteriors → per-bar 12-d posterior + uniform bar times.

        Faithful re-implementation of chord_pipeline_v1._pool_root_proba_to_bars
        (L2617-2675). The uniform-grid bar binding is load-bearing for chart
        alignment — reproduced verbatim.
        """
        n_beats = min(len(beat_proba), len(bt) - 1)
        if n_beats < 2 or period <= 0:
            return np.zeros((0, 12)), []
        bar_len = beats_per_bar * period
        anchor_t = anchor_beats * period
        mids = (bt[:n_beats] + bt[1:n_beats + 1]) / 2.0
        bar_of = np.floor((mids - anchor_t) / bar_len).astype(int)
        keep = bar_of >= 0
        bar_of = bar_of[keep]
        kept_proba = beat_proba[:n_beats][keep]
        n_bars = int(bar_of.max()) + 1 if len(bar_of) else 0
        if n_bars < 2:
            return np.zeros((0, 12)), []
        bar_root = np.zeros((n_bars, 12), dtype=np.float64)
        bar_times: list = []
        eps = 0.25 * period
        for b in range(n_bars):
            sel = kept_proba[bar_of == b]
            if len(sel):
                v = sel.mean(0).astype(np.float64)
                s = v.sum()
                bar_root[b] = v / s if s > 1e-9 else v
            bar_times.append((anchor_t + b * bar_len + eps,
                              anchor_t + (b + 1) * bar_len + eps))
        return bar_root, bar_times

    def symbolic_sections(self, beat_proba, bt, period, duration_s,
                          tonic_pc, anchor_beats=0):
        """Symbolic bar-locked section pass; ``None`` if disabled/short/degenerate.

        Faithful re-implementation of chord_pipeline_v1._barlocked_sections_or_none
        (L2678-2739) — the audio-free symbolic fallback the golden freezes. The
        ``HARMONIA_SECTION_MODE`` env read and the ``barlocked_sections`` SSM
        primitive are preserved (env read honoured via config default; the SSM
        is a genuine leaf import).
        """
        cfg = self.config
        mode = os.environ.get("HARMONIA_SECTION_MODE", cfg.section_mode)
        if mode != "barlocked":
            logger.info("nnls24 sections: barlocked DISABLED "
                        "(HARMONIA_SECTION_MODE=%r) — using acoustic fallback", mode)
            return None
        if duration_s < cfg.section_min_duration_s:
            logger.warning("nnls24 sections: barlocked SKIPPED (duration %.1fs "
                           "< %.0fs)", duration_s, cfg.section_min_duration_s)
            return None
        try:
            from harmonia.models.section_structure import barlocked_sections
            bar_root, bar_times = self._pool_root_proba_to_bars(
                beat_proba, bt, period, anchor_beats=anchor_beats)
            if len(bar_root) < 2:
                logger.warning("nnls24 sections: barlocked DEFERRED — too few "
                               "bars (%d) at this grid", len(bar_root))
                return None
            # barlocked itself returns [] when its labelling collapses to a
            # single label (a near-single-chord loop) — that IS the defer signal.
            secs = barlocked_sections(bar_root, bar_times, tonic_pc=tonic_pc)
            if not secs:
                logger.warning("nnls24 sections: barlocked DEFERRED to acoustic — "
                               "single-label collapse over %d bars", len(bar_root))
                return None
            logger.warning("nnls24 sections: barlocked FIRED — %d sections over "
                           "%d bars, labels=%s", len(secs), len(bar_root),
                           "".join(s["label"][0] for s in secs))
            return secs
        except Exception as exc:  # noqa: BLE001 — never break analyze over sections
            logger.warning("nnls24 sections: barlocked FAILED (%s) — acoustic "
                           "fallback", exc)
            return None

    @staticmethod
    def _note_name_to_pc(name: str) -> int:
        """Pitch class of a key/chord root name. Re-impl of
        chord_pipeline_v1._note_name_to_pc (L110-117)."""
        base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
        tok = name.strip().split()[0] if name.strip() else "C"
        pc = base.get(tok[0].upper(), 0)
        if len(tok) > 1 and tok[1] in "#b":
            pc += 1 if tok[1] == "#" else -1
        return pc % 12

    # ── orchestration: the three net-covered stages ──────────────────────────
    def run(self, audio_path: Path, bt: np.ndarray, period: float,
            duration_s: float) -> ChordStageResult:
        """Run the deterministic, parity-net-covered chord stage end to end."""
        bt = np.asarray(bt, dtype=float)
        heads, arr, times, feat, beat_proba, key_result = self.extract_features(
            audio_path, bt)
        segs, seg_bounds, labeled, coalesced, _mx = self.label_stage(
            audio_path, bt, feat, beat_proba, heads, arr=arr, times=times,
            period=period, key_name=key_result.key_name)

        try:
            tonic_pc = self._note_name_to_pc(key_result.key_name.split()[0])
        except Exception:  # noqa: BLE001
            tonic_pc = None
        bar_root, _bar_times = self._pool_root_proba_to_bars(
            beat_proba, bt, period, anchor_beats=0)
        sections = self.symbolic_sections(
            beat_proba, bt, period, duration_s, tonic_pc=tonic_pc, anchor_beats=0)

        return ChordStageResult(
            arr=arr, times=times, feat=feat, beat_proba=beat_proba,
            key_name=key_result.key_name,
            key_confidence=float(key_result.confidence),
            segs=segs, seg_bounds=seg_bounds, labeled=labeled, coalesced=coalesced,
            bar_root=bar_root, sections=sections,
        )

    # ═══════════════════════════════════════════════════════════════════════
    # FULL PORT — the audio tail (flux/sota anchor, Occam, finalize, 2-chord,
    # onset hints, ChordChart assembly).  Reproduces the WHOLE ``_infer_nnls24``
    # for the live oracle (progress_cb=None) — see the tail at
    # chord_pipeline_v1.py L3741-3985.
    #
    # DETERMINISM (screened 2026-07-23, CLAUDE.md #2): the tail is bit-stable
    # run-to-run on all 8 frozen songs (labels/starts/confs/sections/anchor
    # identical) — so it is ported and gated on EXACT labels + float-eps confs,
    # not brittle audio-float internals (the bp48 lesson).
    #
    # Faithfulness boundary: the tail's genuine primitives (Beat This! sota
    # downbeat, flux comb, per-bar root, Occam arbitration, calibration map,
    # 2-chord split, onset hints, section fallback) are IMPORTED as leaves — the
    # branching/env-read ORCHESTRATION that used to be inline in the god-function
    # is what moves here.  The two pure finalizers are re-implemented below.
    # ═══════════════════════════════════════════════════════════════════════

    @staticmethod
    def _drop_leading_outlier(coalesced: list[list], period: float) -> list[list]:
        """Drop a leading spurious sub-beat low-confidence chord (pre-song noise).
        Re-impl of chord_pipeline_v1._drop_leading_outlier (L3468-3481)."""
        drop_cap = 4.0 * period
        while len(coalesced) > 1:
            t0, t1, label, cds, ds = coalesced[0]
            conf_raw0 = cds / max(ds, 1e-9)
            if t0 <= 1e-3 and (t1 - t0) < 1.2 * period and conf_raw0 < 0.5 \
                    and coalesced[1][0] <= drop_cap:
                coalesced[1][0] = t0
                coalesced.pop(0)
            else:
                break
        return coalesced

    @staticmethod
    def _finalize_chords(coalesced, period, key_name, conf_map):
        """coalesced spans -> (chords_out, segments_out), calibrated confidence.
        Re-impl of chord_pipeline_v1._finalize_chords (L3484-3506)."""
        chords_out, segments_out = [], []
        for t0, t1, label, conf_sum, dur_sum in coalesced:
            conf_raw = conf_sum / dur_sum
            if label == NO_CHORD_LABEL:
                conf_raw = 0.0
                conf = 0.0
            else:
                conf = (float(np.interp(conf_raw, conf_map[0], conf_map[1]))
                        if conf_map is not None else conf_raw)
            n_b = max(1, round((t1 - t0) / period))
            chords_out.append({
                "label": label, "start_s": round(t0, 3), "end_s": round(t1, 3),
                "duration_beats": n_b, "confidence": round(conf, 4),
                "confidence_raw": round(conf_raw, 4), "suggestions": [],
            })
            segments_out.append({"start_s": round(t0, 3), "end_s": round(t1, 3),
                                 "key": key_name, "n_beats": n_b})
        return chords_out, segments_out

    def _section_anchor_pass(self, audio_path, arr, times, heads, beat_proba,
                             bt, period, duration_s, tonic_pc,
                             beat_times_real=None):
        """The flux / Beat This! sota downbeat-anchored section decision.

        Returns ``(sections_out, occam_bars, anchor)``.  The primitives
        (sota downbeat / flux comb / native bar grid / barlocked SSM / acoustic
        fallback) are leaves; the branching + env reads are what lives here.

        Chain: Beat This! ``sota_downbeat_phase`` (default ON — confidently
        right on 6/8 screened songs, correctly abstains on rubato) → chroma-flux
        comb → structure-crispness tie-break when the comb is weak (ratio <
        1.05) → optional native per-downbeat bar grid (default OFF) → per-bar
        root posteriors → barlocked sections; then the symbolic barlocked pass
        at the chosen anchor, then the librosa-Laplacian acoustic fallback.
        """
        from harmonia.models.chord_pipeline_v1 import (
            _flux_anchored_bar_root, _flux_downbeat_phase, _section_fallback,
            _structure_anchor_phase, native_bargrid_enabled,
        )
        from harmonia.models.section_structure import barlocked_sections

        cfg = self.config
        anchor = 0
        sections_out = None
        occam_bars = None
        grid_mode = os.environ.get("HARMONIA_GRID_ANCHOR", cfg.grid_anchor)
        section_mode = os.environ.get("HARMONIA_SECTION_MODE", cfg.section_mode)
        if (grid_mode in ("flux", "structure") and section_mode == "barlocked"
                and duration_s >= cfg.section_min_duration_s):
            try:
                bar_period = 4.0 * period
                phi = ratio = None
                if os.environ.get("HARMONIA_GRID_ANCHOR_SOTA",
                                  cfg.grid_anchor_sota) == "on":
                    try:
                        from harmonia.models.downbeat_anchor import sota_downbeat_phase
                        sota = sota_downbeat_phase(audio_path, bar_period)
                    except Exception as exc:  # noqa: BLE001
                        logger.warning("nnls24 sota-anchor failed (%s) — flux "
                                       "fallback", exc)
                        sota = None
                    if sota is not None:
                        phi, ratio = sota
                        logger.warning("nnls24 sota-anchor (beat_this): downbeat "
                                       "phase %d beats — skipping flux/structure "
                                       "anchor", phi)
                if phi is None:
                    phi, ratio = _flux_downbeat_phase(
                        arr, times, bar_period, audio_path=audio_path)
                if ratio < 1.05:
                    sphi, _ss = _structure_anchor_phase(beat_proba, tonic_pc=tonic_pc)
                    logger.warning("nnls24 flux-anchor: weak comb (ratio %.3f) — "
                                   "structure-crispness tie-break phase %d",
                                   ratio, sphi)
                    phi = sphi
                native_bnds = None
                if native_bargrid_enabled():
                    try:
                        from harmonia.models.beat_grid import native_bar_grid
                        from harmonia.models.downbeat_anchor import beat_this_downbeats
                        dbs, dconf = beat_this_downbeats(audio_path)
                        bts_real = np.asarray(
                            beat_times_real if beat_times_real is not None else bt,
                            dtype=float)
                        native_bnds, nanchor, nmode = native_bar_grid(
                            dbs, dconf, bts_real, period, flux_phi=phi)
                        logger.warning(
                            "nnls24 native-bargrid ON: mode=%s (conf=%.2f, %d "
                            "native downbeats) anchor=%s", nmode, dconf, len(dbs),
                            nanchor)
                        if native_bnds is not None and nanchor is not None:
                            phi = int(nanchor)   # grid_anchor_beats <- bar_times[0]
                    except Exception as exc:  # noqa: BLE001 — never break analyse
                        logger.warning("nnls24 native-bargrid failed (%s) — flux "
                                       "grid", exc)
                        native_bnds = None
                bar_root, bar_times = _flux_anchored_bar_root(
                    arr, times, heads, phi, bar_period, bnds=native_bnds)
                logger.warning("nnls24 flux-anchor: downbeat phase %d beats "
                               "(comb ratio %.3f, %d bars)", phi, ratio,
                               len(bar_root))
                if len(bar_root) >= 2:
                    secs = barlocked_sections(bar_root, bar_times, tonic_pc=tonic_pc)
                    # The Occam post-pass keys off the flux per-bar posteriors +
                    # non-N runs, NOT the barlocked sections — so make the bars
                    # available even when barlocked collapses/returns [].
                    occam_bars = (bar_root, bar_times, secs or [])
                    if secs:
                        sections_out = secs
                        anchor = phi
                        logger.warning("nnls24 flux-anchor sections: %d, labels=%s",
                                       len(secs),
                                       "".join(s["label"][0] for s in secs))
            except Exception as exc:  # noqa: BLE001 — never break analyze
                logger.warning("nnls24 flux-anchor failed (%s) — falling back", exc)
                sections_out = None
        if sections_out is None:
            sections_out = self.symbolic_sections(
                beat_proba, bt, period, duration_s, tonic_pc=tonic_pc,
                anchor_beats=anchor)
        if sections_out is None:
            sections_out = _section_fallback([], audio_path, duration_s)
        return sections_out, occam_bars, anchor

    def _heads_missing_chart(self, audio_path, duration_s, tempo_bpm):
        """Single-chord chart when the trained NNLS-24 heads are absent.

        Never crashes the server path — same degenerate fallback the inline
        ``_infer_nnls24`` emitted (no sections, no anchor, no beat_times).
        """
        from harmonia.models.chord_pipeline_v1 import ChordChart

        logger.warning("infer_chords_v1(nnls24): heads missing — single-chord "
                       "fallback")
        return ChordChart(
            source_path=str(audio_path), duration_s=duration_s,
            tempo_bpm=round(tempo_bpm, 1), time_signature="4/4",
            global_key="C major", global_key_confidence=0.0, style="v1-nnls24",
            modulations=[],
            chords=[{"label": "C:maj", "start_s": 0.0, "end_s": duration_s,
                     "duration_beats": 1, "confidence": 0.0}],
            segments=[{"start_s": 0.0, "end_s": duration_s, "key": "C major",
                       "n_beats": 1}],
        )

    def run_full(self, audio_path: Path, bt: np.ndarray, period: float,
                 duration_s: float, tempo_bpm: float,
                 beat_times_real: np.ndarray | None = None,
                 progress_cb: "object | None" = None):
        """Full nnls24 chord stage → final ChordChart (audio tail included).

        THE single implementation of the shipped nnls24 chord stage:
        ``chord_pipeline_v1._infer_nnls24`` is a thin adapter over this method
        (STEP B, 2026-07-26).  Byte-identical to the pre-B inline path — proven
        on the 8 frozen_parity songs (full ChordChart: labels exact, floats eps)
        for the live oracle, and on a 4-combo × 2-song front-end trace for the
        non-oracle combos + the heads-missing fallback.

        ``progress_cb(kind, payload)`` (server path): fires ``"key"`` →
        ``"draft"`` → per-fold ``"chords"`` → ``"sections"`` → final
        ``"chords"``.  ``infer_chords_v1`` fires ``"beats"`` before calling in.

        ``beat_times_real`` populates ``ChordChart.beat_times`` and feeds the
        OFF-by-default native-bargrid path; ``None`` leaves both inert.
        """
        from harmonia.models import nnls_features as nf
        from harmonia.models.chord_pipeline_v1 import (
            ChordChart, _apply_occam_to_coalesced, _attach_musx_onset_hints,
            _get_nnls24_conf_map, _split_collapsed_bars_via_musx,
        )

        cfg = self.config
        bt = np.asarray(bt, dtype=float)

        # ── REAL-BEAT GRID brick (default-OFF; env HARMONIA_REAL_BEAT_GRID) ──
        # ``grid`` mode swaps the synthetic constant-tempo lattice built in
        # chord_pipeline_v1.py:3920 for the beats Beat This! actually detected,
        # so chroma pooling, the musx re-decode's allowed-transition set, the
        # segmentation indices AND the emitted chord times all live on real
        # onsets.  OFF (the default) returns ``bt`` unchanged — identity, so the
        # shipped path is byte-for-byte what it was.  See
        # harmonia/models/beat_grid.py for the evidence + the guards.
        from harmonia.models import beat_grid as _bg
        bt, _grid_info = _bg.apply_real_beat_grid(
            bt, beat_times_real, duration_s, period)

        if nf.get_heads() is None:
            return self._heads_missing_chart(audio_path, duration_s, tempo_bpm)

        heads, arr, times, feat, beat_proba, key_result = self.extract_features(
            audio_path, bt)

        # Global key is computed EARLY (inside extract_features) so progress_cb
        # can surface it during the fast ~4-6s NNLS stage, well before
        # music-x-lab even starts — docs/inference_pipeline_timing_and_
        # animation_scope.md.
        if progress_cb is not None:
            try:
                progress_cb("key", {"key": key_result.key_name,
                                    "confidence": round(key_result.confidence, 4)})
            except Exception:  # noqa: BLE001 — progress must never break analyze
                logger.warning("nnls24: progress_cb('key') failed", exc_info=True)

        segs, seg_bounds, labeled, coalesced, mx_labels = self.label_stage(
            audio_path, bt, feat, beat_proba, heads, arr=arr, times=times,
            period=period, key_name=key_result.key_name, progress_cb=progress_cb)

        # Drop a leading spurious outlier chord (pre-song video noise): only the
        # very first coalesced span(s), only if sub-beat AND low raw confidence,
        # capped at one bar total, absorbed into the following chord.
        coalesced = self._drop_leading_outlier(coalesced, period)

        # NOTE: chords_out is built AFTER the section pass — the Occam post-pass
        # needs the barlocked loop families + the flux-anchored per-bar root
        # posteriors, so section structure is computed first.
        try:
            tonic_pc = self._note_name_to_pc(key_result.key_name.split()[0])
        except Exception:  # noqa: BLE001
            tonic_pc = None
        sections_out, occam_bars, anchor = self._section_anchor_pass(
            audio_path, arr, times, heads, beat_proba, bt, period, duration_s,
            tonic_pc, beat_times_real=beat_times_real)
        if progress_cb is not None:
            try:
                progress_cb("sections", {"n_sections": len(sections_out or [])})
            except Exception:  # noqa: BLE001
                logger.warning("nnls24: progress_cb('sections') failed",
                               exc_info=True)

        # ── Occam post-pass (default ON; rollback HARMONIA_OCCAM_POSTPASS=0) ──
        # Uses ONLY the song's own structure (barlocked loop families + flux
        # per-bar posteriors); NO corpus grammar/LM prior.  Compresses decode
        # noise on a clean vamp into its repeating pattern, keeping only
        # margin-surviving deviations (100.00% anti-crush on 25,120 pop400 GT
        # bars).  Needs the flux bar grid (else no-op).
        if (os.environ.get("HARMONIA_OCCAM_POSTPASS",
                           "1" if cfg.occam_postpass else "0") == "1"
                and occam_bars is not None):
            try:
                _bp, _btimes, _secs = occam_bars
                new_coalesced, _decisions = _apply_occam_to_coalesced(
                    coalesced, _bp, _btimes, _secs, period)
                _applied = [d for d in _decisions if d.get("applied")]
                # ── GT-free loop-family gate (2026-07-27) ───────────────────
                # SCOPED to the re-decode segmentation, which is the only
                # configuration it was measured under. It is NOT a no-op on the
                # legacy nnls segmentation: the frozen parity net caught it
                # changing let_it_be_remastered_2009 (golden 120 chords / 129
                # segments with Occam applied -> 112 / 45 with the gate
                # rejecting it), so defaulting it ON everywhere would have
                # silently altered charts on a path where nothing justified it.
                # Default therefore follows the segmentation; HARMONIA_OCCAM_GATE
                # (0/1) still forces it either way for A/B.
                # Occam is tuned to the OLD (nnls root-argmax) segmentation and
                # is the single blocker on the re-decode flip: under the new
                # boundaries it fires on 2/7 frozen songs and swings close_to_you
                # by 13.2 pp (−13.21 ON vs +5.17 OFF) while helping stand_by_me
                # (+3.16).  The two are cleanly separated by statistics Occam
                # ALREADY computes, so the gate needs no ground truth: accept a
                # loop family only if coverage >= 0.95 AND it kept 0 deviations.
                # (close_to_you: cov=0.78 dev=4 and cov=0.67 dev=16 → reject;
                #  stand_by_me: cov=0.97 dev=0 → accept.)  Verified below to be
                # a 0.0000 no-op on the pre-flip shipped chart, whose only family
                # anywhere is stand_by_me's cov=0.98 dev=0.  Caveat: N=2 songs
                # actually exercise it.  All-or-nothing per song: if ANY applied
                # family fails the gate the whole post-pass is discarded (a
                # per-family veto would need occam_compress_bars itself to take
                # a family filter — deliberately not touched here).
                _gate_default = "1" if getattr(self, "_used_redecode", False) else "0"
                if (_applied and os.environ.get("HARMONIA_OCCAM_GATE",
                                                _gate_default) == "1"):
                    _bad = [d for d in _applied
                            if float(d.get("coverage", 0.0)) < 0.95
                            or int(d.get("kept_deviations", 0)) != 0]
                    if _bad:
                        logger.warning(
                            "nnls24 OCCAM: GATE REJECTED %d/%d loop famil%s (%s) "
                            "— post-pass discarded, chart left unchanged",
                            len(_bad), len(_applied),
                            "y" if len(_bad) == 1 else "ies",
                            ", ".join("cov=%.2f/dev=%d" % (
                                d.get("coverage", 0.0), d.get("kept_deviations", 0))
                                for d in _bad))
                        _applied = []
                if _applied and new_coalesced is not coalesced:
                    logger.warning("nnls24 OCCAM: compressed %d loop-famil%s (%s); "
                                   "%d spans -> %d", len(_applied),
                                   "y" if len(_applied) == 1 else "ies",
                                   ", ".join("vocab=%s/cov=%.2f/dev=%d" % (
                                       d["vocab"], d["coverage"],
                                       d["kept_deviations"]) for d in _applied),
                                   len(coalesced), len(new_coalesced))
                    for d in _decisions:
                        if "kept_deviation" in d:
                            logger.warning(
                                "nnls24 OCCAM bar %d: %s (root=%s snap=%s "
                                "conf=%.2f lr=%.2f log_odds=%.2f)", d["bar"],
                                "KEPT turnaround" if d["kept_deviation"]
                                else "snapped->vocab",
                                NOTE[d.get("root", d.get("was_root", 0))],
                                NOTE[d["snap_root"]], d.get("conf", 0),
                                d.get("lr", 0), d.get("log_odds", 0))
                    coalesced = new_coalesced
                else:
                    logger.warning("nnls24 OCCAM: no loop family compressed — "
                                   "chart left unchanged")
            except Exception as exc:  # noqa: BLE001 — never break analyze
                logger.warning("nnls24 OCCAM post-pass failed (%s) — unchanged", exc)

        # ── build chords_out from the (possibly Occam-compressed) spans ──
        # No-chord spans: the calibrator is fitted on chord-bearing RWC blocks
        # (no reject option), so any confidence it emits on N is meaningless —
        # _finalize_chords clamps those to 0.
        conf_map = _get_nnls24_conf_map()
        chords_out, segments_out = self._finalize_chords(
            coalesced, period, key_result.key_name, conf_map)

        # Split a collapsed full-bar chord into its 2 real chords where
        # music-x-lab has a sustained 2-chords-per-bar rhythm.  Layout-
        # preserving: the bar's downbeat chord is unchanged; the 2nd chord is
        # added WITHIN the bar.  Kill-switch HARMONIA_MUSX_2CHORD_BAR=0.
        if (mx_labels is not None
                and os.environ.get("HARMONIA_MUSX_2CHORD_BAR",
                                   "1" if cfg.musx_2chord_bar else "0") != "0"):
            _ns = _split_collapsed_bars_via_musx(chords_out, mx_labels, period)
            if _ns:
                logger.warning("nnls24: split %d collapsed full-bar chord(s) into "
                               "2-chords/bar from music-x-lab (fast harmonic "
                               "rhythm)", _ns)
        # ── REAL-BEAT GRID brick, ``snap`` mode (default-OFF) ────────────────
        # Timing-only sibling of the ``grid`` mode above: decode exactly as
        # today on the lattice, then re-lay the FINAL boundary times onto their
        # nearest detected beat (capped at half a beat, monotone, endpoints
        # pinned).  A no-op unless HARMONIA_REAL_BEAT_GRID=snap.
        _bg.snap_chord_times_to_beats(
            chords_out, segments_out, beat_times_real, duration_s, period)

        # Attach trusted DISPLAY onsets from music-x-lab's change-times.
        # Display-only: (bar, beat) layout untouched, playhead snaps to the
        # accurate onset.  Kill-switch HARMONIA_MUSX_ONSET_HINT=0.
        if (mx_labels is not None
                and os.environ.get("HARMONIA_MUSX_ONSET_HINT",
                                   "1" if cfg.musx_onset_hint else "0") != "0"):
            _nh = _attach_musx_onset_hints(chords_out, mx_labels, period)
            if _nh:
                logger.info("nnls24: attached music-x-lab display onsets to "
                            "%d/%d chords", _nh, len(chords_out))
        logger.info("infer_chords_v1(nnls24): %d chords, key=%s, tempo=%.1f BPM",
                    len(chords_out), key_result.key_name, tempo_bpm)
        if progress_cb is not None:
            try:
                progress_cb("chords", {"chords": chords_out})
            except Exception:  # noqa: BLE001
                logger.warning("nnls24: progress_cb('chords') failed", exc_info=True)

        return ChordChart(
            source_path=str(audio_path), duration_s=duration_s,
            tempo_bpm=round(tempo_bpm, 1), time_signature="4/4",
            global_key=key_result.key_name,
            global_key_confidence=round(key_result.confidence, 4),
            style="v1-nnls24", modulations=[],
            chords=chords_out, segments=segments_out, sections=sections_out,
            grid_anchor_beats=int(anchor),
            beat_times=([float(t) for t in beat_times_real]
                        if beat_times_real is not None else []),
        )
