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
WHAT THIS DELIBERATELY DOES NOT PORT (documented follow-up — CLAUDE.md #4)
═══════════════════════════════════════════════════════════════════════════════
The live ``_infer_nnls24`` also runs, AFTER the coalesce, an AUDIO-touching and
env-gated tail that the parity net does NOT cover (it needs fresh audio
inference; the golden freezes only the symbolic pass + a marker):
  * the flux / Beat This! ``sota_downbeat_phase`` grid anchor + native bar grid,
  * ``_flux_anchored_bar_root`` + ``_section_fallback`` (librosa-Laplacian),
  * the Occam post-pass, the music-x-lab 2-chord-per-bar split, onset hints,
  * the ``progress_cb`` draft/fold callbacks.
Those are gated behind ``ChordHeadConfig`` flags (``grid_anchor``,
``occam_postpass``, ``musx_2chord_bar``, ``musx_onset_hint`` …) so the interface
is complete, but they are NOT reproduced or gated in this Phase-3 slice. Porting
them is a follow-up that needs the audio-path net extension first.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

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

    ``ChordHeadConfig.live_defaults()`` == ``benchmark_set.LIVE_ORACLE_KWARGS``
    for the chord-stage-relevant subset (the frozen parity oracle).
    """

    # ── chord-stage front-end selectors (the live-path knobs) ──
    feature_frontend: str = "nnls24"     # this head IS the nnls24 head
    bass_frontend: str = "musx"          # sounding-bass source: "nnls24" | "musx"
    quality_frontend: str = "musx"       # root+quality source: "nnls24" | "musx"
    segment_source: str = "nnls"         # "nnls" (root-change) | "musx" (boundaries)

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
        """The frozen live production oracle (== LIVE_ORACLE_KWARGS subset)."""
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

    def label_stage(self, audio_path, bt, feat, beat_proba, heads):
        """Segmentation → musx per-segment lookups → per-segment labels → coalesce.

        Reproduces the FINAL pass of ``_infer_nnls24`` for the live default
        (``segment_source='nnls'``, ``bass_frontend='musx'``,
        ``quality_frontend='musx'`` — see L3586, L3612, L3685-3728), which is
        exactly what the parity golden ``nnls24_precoalesce`` freezes
        (parity.py L482-493).
        """
        from harmonia.models import musx_bass as mxb

        cfg = self.config
        # segmentation (nnls root-change is the live default; musx-boundary is a
        # documented extension the net does not cover — see module docstring)
        if cfg.segment_source == "musx":
            raise NotImplementedError(
                "segment_source='musx' (_musx_boundary_segs) is a documented "
                "follow-up; the parity-gated live path uses 'nnls'.")
        segs = self._root_change_segs(beat_proba)
        seg_bounds = [(float(bt[s]), float(bt[min(e, len(bt) - 1)]))
                      for (s, e) in segs]
        bass_half = feat[:, :12]

        mx_labels = None
        musx_seg_bass = musx_seg_rq = None
        seg_no_chord = np.zeros(len(seg_bounds), dtype=bool)
        if cfg.want_musx:
            mx_labels = mxb.musx_labels(audio_path)      # cache HIT (stem-keyed)
            if cfg.bass_frontend == "musx":
                musx_seg_bass = mxb.bass_pc_per_segment(mx_labels, seg_bounds)
            if cfg.quality_frontend == "musx":
                musx_seg_rq = mxb.root_quality_per_segment(mx_labels, seg_bounds)
            seg_no_chord = mxb.no_chord_per_segment(mx_labels, seg_bounds)

        labeled = self._label_segments(
            segs, seg_bounds, beat_proba, feat, bass_half, heads,
            musx_seg_rq=musx_seg_rq, musx_seg_bass=musx_seg_bass,
            seg_no_chord=seg_no_chord)
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
            return None
        if duration_s < cfg.section_min_duration_s:
            return None
        try:
            from harmonia.models.section_structure import barlocked_sections
            bar_root, bar_times = self._pool_root_proba_to_bars(
                beat_proba, bt, period, anchor_beats=anchor_beats)
            if len(bar_root) < 2:
                return None
            secs = barlocked_sections(bar_root, bar_times, tonic_pc=tonic_pc)
            if not secs:
                return None
            return secs
        except Exception:  # noqa: BLE001 — never break analyze over sections
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
            audio_path, bt, feat, beat_proba, heads)

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
                             bt, period, duration_s, tonic_pc):
        """The flux / Beat This! sota downbeat-anchored section decision.

        Faithful re-impl of the ``_infer_nnls24`` section block (L3762-3879,
        progress_cb=None branch): returns ``(sections_out, occam_bars, anchor)``.
        Primitives (sota/flux/native-bargrid/barlocked/fallback) are leaves.
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
                if os.environ.get("HARMONIA_GRID_ANCHOR_SOTA", cfg.grid_anchor_sota) == "on":
                    try:
                        from harmonia.models.downbeat_anchor import sota_downbeat_phase
                        sota = sota_downbeat_phase(audio_path, bar_period)
                    except Exception:  # noqa: BLE001
                        sota = None
                    if sota is not None:
                        phi, ratio = sota
                if phi is None:
                    phi, ratio = _flux_downbeat_phase(
                        arr, times, bar_period, audio_path=audio_path)
                if ratio < 1.05:
                    sphi, _ss = _structure_anchor_phase(beat_proba, tonic_pc=tonic_pc)
                    phi = sphi
                native_bnds = None
                if native_bargrid_enabled():
                    try:
                        from harmonia.models.beat_grid import native_bar_grid
                        from harmonia.models.downbeat_anchor import beat_this_downbeats
                        dbs, dconf = beat_this_downbeats(audio_path)
                        bts_real = np.asarray(bt, dtype=float)
                        native_bnds, nanchor, _nmode = native_bar_grid(
                            dbs, dconf, bts_real, period, flux_phi=phi)
                        if native_bnds is not None and nanchor is not None:
                            phi = int(nanchor)
                    except Exception:  # noqa: BLE001
                        native_bnds = None
                bar_root, bar_times = _flux_anchored_bar_root(
                    arr, times, heads, phi, bar_period, bnds=native_bnds)
                if len(bar_root) >= 2:
                    secs = barlocked_sections(bar_root, bar_times, tonic_pc=tonic_pc)
                    occam_bars = (bar_root, bar_times, secs or [])
                    if secs:
                        sections_out = secs
                        anchor = phi
            except Exception:  # noqa: BLE001 — never break analyze over sections
                sections_out = None
        if sections_out is None:
            sections_out = self.symbolic_sections(
                beat_proba, bt, period, duration_s, tonic_pc=tonic_pc,
                anchor_beats=anchor)
        if sections_out is None:
            sections_out = _section_fallback([], audio_path, duration_s)
        return sections_out, occam_bars, anchor

    def run_full(self, audio_path: Path, bt: np.ndarray, period: float,
                 duration_s: float, tempo_bpm: float,
                 beat_times_real: np.ndarray | None = None):
        """Full nnls24 chord stage → final ChordChart (audio tail included).

        Byte-identical to ``_infer_nnls24(...)`` for the live oracle
        (progress_cb=None).  ``beat_times_real`` only populates the (uncaptured)
        ``ChordChart.beat_times`` field and the OFF-by-default native-bargrid
        path; passing ``None`` does not change any captured output (empirically
        confirmed by the byte-identity gate).
        """
        from harmonia.models.chord_pipeline_v1 import (
            ChordChart, _apply_occam_to_coalesced, _attach_musx_onset_hints,
            _get_nnls24_conf_map, _split_collapsed_bars_via_musx,
        )

        cfg = self.config
        bt = np.asarray(bt, dtype=float)
        heads, arr, times, feat, beat_proba, key_result = self.extract_features(
            audio_path, bt)
        segs, seg_bounds, labeled, coalesced, mx_labels = self.label_stage(
            audio_path, bt, feat, beat_proba, heads)
        coalesced = self._drop_leading_outlier(coalesced, period)

        try:
            tonic_pc = self._note_name_to_pc(key_result.key_name.split()[0])
        except Exception:  # noqa: BLE001
            tonic_pc = None
        sections_out, occam_bars, anchor = self._section_anchor_pass(
            audio_path, arr, times, heads, beat_proba, bt, period, duration_s,
            tonic_pc)

        # ── Occam post-pass (default ON) ──
        if (os.environ.get("HARMONIA_OCCAM_POSTPASS",
                           "1" if cfg.occam_postpass else "0") == "1"
                and occam_bars is not None):
            try:
                _bp, _btimes, _secs = occam_bars
                new_coalesced, _dec = _apply_occam_to_coalesced(
                    coalesced, _bp, _btimes, _secs, period)
                if [d for d in _dec if d.get("applied")] and new_coalesced is not coalesced:
                    coalesced = new_coalesced
            except Exception:  # noqa: BLE001 — never break analyze over Occam
                pass

        conf_map = _get_nnls24_conf_map()
        chords_out, segments_out = self._finalize_chords(
            coalesced, period, key_result.key_name, conf_map)

        # ── music-x-lab 2-chord-per-bar split + display onset hints (default ON) ──
        if (mx_labels is not None
                and os.environ.get("HARMONIA_MUSX_2CHORD_BAR",
                                   "1" if cfg.musx_2chord_bar else "0") != "0"):
            _split_collapsed_bars_via_musx(chords_out, mx_labels, period)
        if (mx_labels is not None
                and os.environ.get("HARMONIA_MUSX_ONSET_HINT",
                                   "1" if cfg.musx_onset_hint else "0") != "0"):
            _attach_musx_onset_hints(chords_out, mx_labels, period)

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
