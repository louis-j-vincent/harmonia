"""harmonia/eval/parity.py — parity gate harness for the rewrite.

Captures and diffs ChordChart outputs (old vs new) with field-specific tolerances.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DiffResult:
    """Single field diff between old and new ChordChart."""
    field: str
    old_value: Any
    new_value: Any
    match: bool
    tolerance: float | None = None
    error: float | None = None
    details: str = ""


@dataclass
class ChartDiff:
    """Full diff between two ChordChart JSONs."""
    name: str
    song_id: str
    matches: bool
    differences: list[DiffResult]

    def report(self, verbose=False) -> str:
        """Render a human-readable diff report."""
        status = "✓ MATCH" if self.matches else "✗ DIFFER"
        lines = [f"{status}: {self.name} (song={self.song_id})"]

        if self.differences:
            for diff in self.differences:
                if not diff.match or verbose:
                    lines.append(
                        f"  {diff.field}: {diff.old_value} → {diff.new_value} "
                        f"(tol={diff.tolerance}, error={diff.error})"
                    )

        return "\n".join(lines)


def load_chart_json(path: Path) -> dict:
    """Load a ChordChart JSON file."""
    with open(path) as f:
        return json.load(f)


def save_chart_json(chart: dict, path: Path) -> None:
    """Save a ChordChart JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(chart, f, indent=2)


def diff_charts(
    old_chart: dict,
    new_chart: dict,
    name: str = "chart",
    float_tol: float = 1e-4,
    label_exact: bool = True,
) -> ChartDiff:
    """Compare two ChordChart dicts field-by-field.

    Args:
        old_chart: old JSON dict
        new_chart: new JSON dict
        name: song name for reporting
        float_tol: tolerance for floating-point fields (tempo, confidence, etc.)
        label_exact: if True, chord labels must match exactly; if False, allow label order variance

    Returns:
        ChartDiff with all field-level diffs
    """
    diffs = []

    # Top-level scalar fields (exact match for most, tolerance for floats)
    scalar_fields = {
        "source_path": (str, True),
        "duration_s": (float, False),
        "tempo_bpm": (float, False),
        "time_signature": (str, True),
        "global_key": (str, True),
        "global_key_confidence": (float, False),
        "style": (str, True),
    }

    for field, (expected_type, exact) in scalar_fields.items():
        old_val = old_chart.get(field)
        new_val = new_chart.get(field)

        if exact:
            match = old_val == new_val
            error = None
        else:  # float, with tolerance
            match = (old_val is not None and new_val is not None and
                    abs(float(old_val) - float(new_val)) < float_tol)
            error = (abs(float(old_val) - float(new_val))
                    if old_val is not None and new_val is not None else None)

        if not match:
            diffs.append(DiffResult(
                field=field,
                old_value=old_val,
                new_value=new_val,
                match=match,
                tolerance=float_tol if not exact else None,
                error=error,
            ))

    # Chord sequence (critical): compare label-for-label
    old_chords = old_chart.get("chords", [])
    new_chords = new_chart.get("chords", [])

    if len(old_chords) != len(new_chords):
        diffs.append(DiffResult(
            field="chords.length",
            old_value=len(old_chords),
            new_value=len(new_chords),
            match=False,
        ))

    for i, (old_ch, new_ch) in enumerate(zip(old_chords, new_chords)):
        # Exact label match
        if old_ch.get("label") != new_ch.get("label"):
            diffs.append(DiffResult(
                field=f"chords[{i}].label",
                old_value=old_ch.get("label"),
                new_value=new_ch.get("label"),
                match=False,
            ))

        # Time boundaries: tight tolerance (50ms)
        for bound in ["start_s", "end_s"]:
            old_b = old_ch.get(bound)
            new_b = new_ch.get(bound)
            error = abs(float(old_b) - float(new_b)) if old_b is not None and new_b is not None else None
            match = error is not None and error < 0.050
            if not match:
                diffs.append(DiffResult(
                    field=f"chords[{i}].{bound}",
                    old_value=old_b,
                    new_value=new_b,
                    match=match,
                    tolerance=0.050,
                    error=error,
                ))

        # Confidence scores: loose tolerance (0.01)
        for conf_field in ["confidence", "confidence_raw", "root_conf"]:
            old_c = old_ch.get(conf_field)
            new_c = new_ch.get(conf_field)
            if old_c is not None and new_c is not None:
                error = abs(float(old_c) - float(new_c))
                match = error < 0.01
                if not match:
                    diffs.append(DiffResult(
                        field=f"chords[{i}].{conf_field}",
                        old_value=old_c,
                        new_value=new_c,
                        match=match,
                        tolerance=0.01,
                        error=error,
                    ))

    # Segments (structure boundaries)
    old_segs = old_chart.get("segments", [])
    new_segs = new_chart.get("segments", [])

    if len(old_segs) != len(new_segs):
        diffs.append(DiffResult(
            field="segments.length",
            old_value=len(old_segs),
            new_value=len(new_segs),
            match=False,
        ))

    for i, (old_s, new_s) in enumerate(zip(old_segs, new_segs)):
        # Segment boundaries with 50ms tolerance
        for bound in ["start_s", "end_s"]:
            old_b = old_s.get(bound)
            new_b = new_s.get(bound)
            error = abs(float(old_b) - float(new_b)) if old_b is not None and new_b is not None else None
            match = error is not None and error < 0.050
            if not match:
                diffs.append(DiffResult(
                    field=f"segments[{i}].{bound}",
                    old_value=old_b,
                    new_value=new_b,
                    match=match,
                    tolerance=0.050,
                    error=error,
                ))

    song_id = old_chart.get("source_path", "unknown").split("/")[-1]
    return ChartDiff(
        name=name,
        song_id=song_id,
        matches=len(diffs) == 0,
        differences=diffs,
    )


# ═══════════════════════════════════════════════════════════════════════════
# Per-stage golden capture + diff (Phase 0: the frozen parity safety net)
# ═══════════════════════════════════════════════════════════════════════════
# capture(song) dumps every pipeline stage it can cleanly reach for one song;
# save_golden / load_golden persist it; diff() reports per-stage divergence with
# tolerances (EXACT for labels/ints/strings/shapes, epsilon for float vectors).
#
# The frozen ORACLE is the live production path (== PipelineConfig.live_defaults,
# nnls24 + musx), verified deterministic across independent processes. Stages:
#   beats            — beatthis + bestfit grid   [ON the live path; re-derived]
#   live_chart.*     — infer_chords_v1(live)     [AUTHORITATIVE end-to-end]
#   bp48_features    — Basic-Pitch stem features [Phase-1 bp48 port target]
#   bp48_pooled/     — pool→beat_seq→segment→key [Phase-2 port targets]
#     beat_proba/segments/key                     (bp48 head; NOT on live path)
#
# Everything an array is stored as {shape,dtype,__sha256__,stats}: the sha256 of
# the raw bytes makes the self-consistency gate a bitwise-exact check, while the
# stats let a later cross-VERSION diff apply an epsilon tolerance.

import hashlib as _hashlib

import numpy as _np

CAPTURE_SCHEMA_VERSION = 4  # v4 = +live_grid_anchor (audio-tail downbeat-phase decision);
#                              v3 = +nnls24 chord-stage intermediates (v2 per-stage; v1 raw JSON)

# Stages capture() reaches vs not (surfaced in the golden meta, honesty bar).
STAGES_CAPTURED = [
    "beats", "bp48_features", "bp48_pooled", "bp48_beat_proba",
    "bp48_segments", "bp48_key",
    # nnls24 chord-stage intermediates (v3 — the live-path Phase-3 chord PORT
    # targets, captured deterministically from CACHED features; see _nnls24_stages).
    "nnls24_features", "nnls24_precoalesce", "nnls24_sections",
    "live_key", "live_tempo", "live_grid_anchor",
    "live_chords", "live_segments", "live_sections",
]
STAGES_NOT_YET_CAPTURED = [
    # honest record of what is NOT frozen yet (plan Phase 0 "documented, not faked")
    # nnls24_features / nnls24_precoalesce / nnls24_sections (symbolic barlocked
    # pass) are now CAPTURED (v3).  What is STILL not frozen on the section path:
    "nnls24 LIVE section output — the AUDIO-touching flux-anchor (sota Beat This! "
    "downbeat + _flux_anchored_bar_root) and _section_fallback (librosa-Laplacian); "
    "needs fresh audio inference, so nnls24_sections freezes the symbolic barlocked "
    "pass + a marker instead (see _nnls24_stages.__audio_dependent_not_captured__)",
    "BP48-§10b section phase-correction internals (period_bars/shift/apply_phase_shift "
    "in infer_chords_v1) — a BP48-path concept, not part of any nnls24 code path",
    "aligned_corpus chord-label capture (chord-label net; deferred)",
    "POP909 beat/alignment capture (no rendered audio; render forbidden)",
]


def _arr_summary(a) -> dict:
    """Summarize an ndarray: exact byte sha256 + shape/dtype + float stats."""
    a = _np.ascontiguousarray(a)
    flat = a.astype(_np.float64).ravel() if a.size else _np.zeros(0)
    return {
        "shape": list(a.shape),
        "dtype": str(a.dtype),
        "__sha256__": _hashlib.sha256(a.tobytes()).hexdigest(),
        "stats": {
            "sum": round(float(flat.sum()), 6) if a.size else 0.0,
            "mean": round(float(flat.mean()), 6) if a.size else 0.0,
            "min": round(float(flat.min()), 6) if a.size else 0.0,
            "max": round(float(flat.max()), 6) if a.size else 0.0,
            "l2": round(float(_np.linalg.norm(flat)), 6) if a.size else 0.0,
        },
    }


def decode_to_wav(m4a_path: Path, wav_path: Path) -> Path:
    """Decode an .m4a to mono wav (soundfile can't read m4a; infer_chords_v1
    needs a soundfile-readable file — matches the server's transcode step)."""
    import librosa
    import soundfile as sf
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    y, sr = librosa.load(str(m4a_path), sr=None, mono=True)
    sf.write(str(wav_path), y, sr)
    return wav_path


def _beats_stage(wav_path: Path, beat_period_mode: str = "bestfit") -> dict:
    """Beat + downbeat + bestfit grid — faithfully replicates infer_chords_v1
    lines ~4265-4322 (beat_backend='beatthis'). ON the live chart path."""
    import soundfile as sf
    from harmonia.models.chord_pipeline_v1 import _bestfit_beat_period, _get_beatthis

    y, sr = sf.read(str(wav_path))
    y = (y.mean(1) if getattr(y, "ndim", 1) > 1 else y)
    duration_s = len(y) / sr

    f2b = _get_beatthis()
    if f2b is None:
        return {"__missing__": "beatthis unavailable", "duration_s": round(duration_s, 6)}
    bts, dbs = f2b(str(wav_path))
    bts = _np.asarray(bts, dtype=float)
    dbs = _np.asarray(dbs, dtype=float)
    tempo_bpm = 60.0 / float(_np.median(_np.diff(bts)))

    period = 60.0 / max(tempo_bpm, 1.0)
    if beat_period_mode == "bestfit":
        period = _bestfit_beat_period(bts, period)
        tempo_bpm = 60.0 / period
    ang = 2 * _np.pi * (bts % period) / period
    phase = (_np.angle(_np.mean(_np.exp(1j * ang))) % (2 * _np.pi)) * period / (2 * _np.pi)
    bt = _np.arange(phase, duration_s + period, period)
    bt = _np.unique(_np.concatenate([[0.0], bt, [duration_s]]))

    return {
        "provenance": "beatthis + bestfit grid (infer_chords_v1 L4265-4322); ON live path",
        "duration_s": round(float(duration_s), 6),
        "sr": int(sr),
        "n_beats_raw": int(len(bts)),
        "n_downbeats": int(len(dbs)),
        "tempo_bpm": round(float(tempo_bpm), 6),
        "period_s": round(float(period), 6),
        "phase_s": round(float(phase), 6),
        "n_grid": int(len(bt)),
        "beat_times_raw": _arr_summary(bts),
        "downbeats": _arr_summary(dbs),
        "grid": _arr_summary(bt),
        "_bt": bt,  # popped before serialization; fed to bp48 + nnls24 stages
        "_period": float(period),        # popped; nnls24 sections need the exact period
        "_duration_s": float(duration_s),  # popped; nnls24 sections >=20s gate
    }


def _bp48_stages(wav_path: Path, bt, cache_dir: Path) -> dict:
    """bp48 feature → pool → beat_seq → segment → key.  These are the Phase-1/2
    PORT targets for the bp48 head; they are NOT on the live nnls24 chart path,
    so they are frozen as independent intermediates (labeled as such)."""
    from harmonia.core.features import ActivationResult, FeatureExtractor
    from harmonia.models.chord_pipeline_v1 import (
        _coarse_segments, _get_beat_seq, _pool_beats, _reg_raw, _root_change_segs,
    )
    from harmonia.models.stage1_pitch import BASIC_PITCH_FRAME_RATE
    from harmonia.theory.key_profiles import infer_key

    out: dict = {}
    acts = FeatureExtractor.create("bp48", cache_dir=cache_dir).extract(wav_path)
    assert isinstance(acts, ActivationResult)
    out["bp48_features"] = {
        "provenance": "FeatureExtractor.create('bp48') — exact call infer_chords_v1 makes; Phase-1 port target",
        "n_frames": int(acts.frame_times.shape[0]),
        "frame_rate_hz": round(float(BASIC_PITCH_FRAME_RATE), 7),
        "onsets": _arr_summary(acts.onsets),
        "activations": _arr_summary(acts.activations),
        "frame_times": _arr_summary(acts.frame_times),
    }

    onset_b = _pool_beats(acts.frame_times, acts.onsets, bt)
    note_b = _pool_beats(acts.frame_times, acts.activations, bt)
    out["bp48_pooled"] = {
        "provenance": "_pool_beats(SUM) on bp48 acts + grid; Phase-2 target",
        "n_beats": int(len(onset_b)),
        "onset_b": _arr_summary(onset_b),
        "note_b": _arr_summary(note_b),
    }

    beat_seq = _get_beat_seq()
    if beat_seq is None:
        out["bp48_beat_proba"] = {"__missing__": "beat_seq model unavailable"}
        out["bp48_segments"] = {"__missing__": "needs beat_proba"}
    else:
        beat_proba = beat_seq.predict_proba(onset_b, note_b)
        mean_conf = float(beat_proba.max(1).mean())
        out["bp48_beat_proba"] = {
            "provenance": "_get_beat_seq().predict_proba; Phase-2 target",
            "shape": list(beat_proba.shape),
            "mean_conf": round(mean_conf, 6),
            "root_argmax": [int(x) for x in beat_proba.argmax(1)],
            "beat_proba": _arr_summary(beat_proba),
        }
        # default path: use_harmonic_grid=True, use_bass_tracking=False (L4430-4462)
        if mean_conf < 0.30:
            segs = _coarse_segments(onset_b, theta=0.08, cell=2)
            seg_fn = "_coarse_segments(theta=0.08,cell=2) [low-conf acoustic fallback]"
        else:
            segs = _root_change_segs(beat_proba)
            seg_fn = "_root_change_segs [gmerge, default]"
        out["bp48_segments"] = {
            "provenance": f"segmentation branch (mean_conf {'<' if mean_conf < 0.30 else '>='} 0.30): {seg_fn}",
            "n_segments": int(len(segs)),
            "boundaries": [[int(s), int(e)] for s, e in segs],
        }

    key_res = infer_key(_reg_raw(onset_b.sum(0)))
    out["bp48_key"] = {
        "provenance": "infer_key(_reg_raw(onset_b.sum(0))); Phase-4 key target",
        "key_name": key_res.key_name,
        "confidence": round(float(key_res.confidence), 6),
    }
    return out


def _nnls24_stages(wav_path: Path, bt, period: float, duration_s: float,
                   cache_dir: Path) -> dict:
    """nnls24 chord-stage intermediates — features → pre-coalesce labels →
    symbolic barlocked sections.  Sibling of ``_bp48_stages``: mirrors the
    INTERNAL call sequence of ``_infer_nnls24`` (feature_frontend='nnls24',
    bass_frontend='musx', quality_frontend='musx', segment_source='nnls' — the
    frozen ``LIVE_ORACLE_KWARGS``) on CACHED features; modifies NO pipeline source.

    These three blocks are the Phase-3 chord PORT targets (a port touching the
    nnls24 chord stage was NOT covered by the parity net — the ``[MED —
    safety-net GAP]`` known-issue).  They ARE on the live chart path, so they are
    frozen as the label-for-label / vector-for-vector oracle for that port.

    Cache discipline: ``extract_bothchroma`` and ``musx_labels`` are STEM-keyed
    caches — a HIT loads the .npz/.lab and never touches audio (a MISS would pay a
    cold VAMP / music-x-lab decode, which this deliberately refuses: it records a
    ``__needs_inference__`` marker instead — the honesty bar).  Reuses the beats
    stage's grid ``bt`` + ``period`` + ``duration_s``; adds NO new beat inference.

    What is NOT reproduced here (honest remainder, CLAUDE.md #4): the LIVE nnls24
    SECTION output additionally runs the audio-touching flux-anchor (Beat This!
    downbeat via ``sota_downbeat_phase`` + ``_flux_anchored_bar_root``) and the
    ``_section_fallback`` librosa-Laplacian pass.  Those need fresh audio
    inference, so ``nnls24_sections`` freezes only the deterministic SYMBOLIC
    barlocked pass (``_barlocked_sections_or_none`` at anchor 0 — the live path's
    own symbolic fallback) plus a marker naming the un-captured audio path.
    """
    from harmonia.models import musx_bass as mxb
    from harmonia.models import nnls_features as nf
    from harmonia.models.chord_pipeline_v1 import (
        _barlocked_sections_or_none, _coalesce_labeled, _label_segments,
        _note_name_to_pc, _pool_root_proba_to_bars, _root_change_segs,
    )
    from harmonia.theory.key_profiles import infer_key

    stem = Path(wav_path).stem
    nnls_cache = Path(cache_dir) / "nnls_infer" / f"{stem}.npz"
    musx_cache = Path(cache_dir) / "musx_infer" / f"{stem}_submission.lab"
    if not (nnls_cache.exists() and musx_cache.exists()):
        marker = {"__needs_inference__":
                  f"cache MISS (nnls={nnls_cache.exists()}, musx={musx_cache.exists()}) "
                  "— capturing would pay a cold VAMP / music-x-lab decode; refused"}
        return {"nnls24_features": dict(marker), "nnls24_precoalesce": dict(marker),
                "nnls24_sections": dict(marker)}

    heads = nf.get_heads()
    if heads is None:
        marker = {"__needs_inference__": "nnls24 heads (nnls24_heads.npz) unavailable"}
        return {"nnls24_features": dict(marker), "nnls24_precoalesce": dict(marker),
                "nnls24_sections": dict(marker)}

    bt = _np.asarray(bt, dtype=float)
    out: dict = {}

    # ── nnls24_features: cached VAMP bothchroma → beat-pooled feat24 → root head ─
    arr, times = nf.extract_bothchroma(wav_path)     # cache HIT (stem-keyed)
    feat = nf.pool_beats(arr, times, bt)             # (n_beats, 24) C-frame
    n_beats = int(len(feat))
    beat_proba = heads.root_proba(feat)              # (n_beats, 12)
    key_result = infer_key(feat[:, 12:].sum(0))
    out["nnls24_features"] = {
        "provenance": "nf.extract_bothchroma → pool_beats → heads.root_proba — exact "
                      "_infer_nnls24 sequence (LIVE_ORACLE_KWARGS), from CACHE; Phase-3 target",
        "n_frames": int(arr.shape[0]),
        "n_beats": n_beats,
        "bothchroma_arr": _arr_summary(arr),
        "bothchroma_times": _arr_summary(times),
        "feat": _arr_summary(feat),
        "beat_proba": _arr_summary(beat_proba),
        "key_name": key_result.key_name,
        "key_confidence": round(float(key_result.confidence), 6),
    }

    # ── nnls24_precoalesce: _label_segments BEFORE _coalesce_labeled ────────────
    # segment_source='nnls' → _root_change_segs; bass+quality='musx' → per-segment
    # music-x-lab root/quality/bass + N mask (all cached).  This IS the live final
    # pass (want_musx True ⇒ the NNLS raw-energy N gate branch is skipped).
    segs = _root_change_segs(beat_proba)
    seg_bounds = [(float(bt[s]), float(bt[min(e, len(bt) - 1)])) for (s, e) in segs]
    bass_half = feat[:, :12]
    mx_labels = mxb.musx_labels(wav_path)            # cache HIT (stem-keyed)
    musx_seg_bass = mxb.bass_pc_per_segment(mx_labels, seg_bounds)
    musx_seg_rq = mxb.root_quality_per_segment(mx_labels, seg_bounds)
    seg_no_chord = mxb.no_chord_per_segment(mx_labels, seg_bounds)
    labeled = _label_segments(
        segs, seg_bounds, beat_proba, feat, bass_half, heads,
        musx_seg_rq=musx_seg_rq, musx_seg_bass=musx_seg_bass,
        seg_no_chord=seg_no_chord)
    coalesced = _coalesce_labeled(labeled)
    out["nnls24_precoalesce"] = {
        "provenance": "_label_segments BEFORE _coalesce_labeled (seg=nnls, bass/quality=musx) "
                      "— the load-bearing Phase-3 chord-port golden",
        "n_segs": int(len(segs)),
        "n_precoalesce": int(len(labeled)),
        "n_coalesced": int(len(coalesced)),
        "seg_boundaries": [[int(s), int(e)] for (s, e) in segs],
        "labels": [{"start_s": round(float(t0), 3), "end_s": round(float(t1), 3),
                    "label": lab, "conf": round(float(conf), 6)}
                   for (t0, t1, lab, conf) in labeled],
    }

    # ── nnls24_sections: symbolic barlocked pass (deterministic; no audio) ──────
    try:
        _tonic_pc = _note_name_to_pc(key_result.key_name.split()[0])
    except Exception:  # noqa: BLE001
        _tonic_pc = None
    bar_root, _bar_times = _pool_root_proba_to_bars(
        beat_proba, bt, period, anchor_beats=0)
    secs = _barlocked_sections_or_none(
        beat_proba, bt, period, float(duration_s), tonic_pc=_tonic_pc, anchor_beats=0)
    out["nnls24_sections"] = {
        "provenance": "symbolic barlocked nnls24 section pass "
                      "(_barlocked_sections_or_none, anchor_beats=0) + _pool_root_proba_to_bars; "
                      "deterministic from cached features",
        "barlocked_fired": bool(secs is not None),
        "n_bars": int(len(bar_root)),
        "bar_root": _arr_summary(bar_root),
        "sections": [{"start_s": s.get("start_s"), "end_s": s.get("end_s"),
                      "n_bars": s.get("n_bars"), "label": s.get("label")}
                     for s in (secs or [])],
        "__audio_dependent_not_captured__": [
            "LIVE flux-anchor downbeat: sota_downbeat_phase (Beat This! on AUDIO) / "
            "_flux_downbeat_phase / _flux_anchored_bar_root → barlocked_sections",
            "_section_fallback librosa-Laplacian acoustic sections (touches audio)",
            "BP48-§10b phase-correction internals period_bars/shift/apply_phase_shift "
            "(an infer_chords_v1 BP48-path concept; not on any nnls24 code path)",
        ],
    }
    return out


def _live_chart_stages(wav_path: Path, cache_dir: Path, oracle_kwargs: dict) -> dict:
    """Run the AUTHORITATIVE live production path and decompose the ChordChart
    into per-stage goldens (key / tempo / chords / segments / sections)."""
    from harmonia.models.chord_pipeline_v1 import infer_chords_v1

    chart = infer_chords_v1(wav_path, cache_dir=cache_dir, **oracle_kwargs)
    return {
        "live_key": {
            "global_key": chart.global_key,
            "global_key_confidence": round(float(chart.global_key_confidence), 6),
        },
        "live_tempo": {
            "tempo_bpm": round(float(chart.tempo_bpm), 6),
            "time_signature": chart.time_signature,
        },
        # The AUDIO-TAIL downbeat-phase decision (flux / Beat This! sota anchor):
        # ``grid_anchor_beats`` is the chosen bar-1 phase that drives the bar grid
        # + section boundaries.  Deterministic run-to-run (verified 2026-07-23) so
        # captured EXACT.  This is the one tail DECISION not already implied by
        # live_chords/live_sections — the Phase-3 FULL ChordHead port gates on it.
        "live_grid_anchor": {
            "grid_anchor_beats": int(getattr(chart, "grid_anchor_beats", 0) or 0),
        },
        "live_chords": [
            {"label": c["label"], "start_s": c["start_s"], "end_s": c["end_s"],
             "duration_beats": c["duration_beats"],
             "confidence": c.get("confidence"), "confidence_raw": c.get("confidence_raw"),
             "root_conf": c.get("root_conf")}
            for c in chart.chords
        ],
        "live_segments": [
            {"start_s": s["start_s"], "end_s": s["end_s"], "n_beats": s.get("n_beats"),
             "key": s.get("key")}
            for s in chart.segments
        ],
        "live_sections": [
            {"start_s": s.get("start_s"), "end_s": s.get("end_s"),
             "n_bars": s.get("n_bars"), "label": s.get("label")}
            for s in (chart.sections or [])
        ],
    }


def capture(entry: dict, *, wav_path: Path, cache_dir: Path,
            oracle_kwargs: dict | None = None, decode: bool = True) -> dict:
    """Capture every reachable pipeline stage for one benchmark song.

    Args:
        entry:  a PARITY-net entry from benchmark_set.parity_songs() (needs
                'song_id' and 'audio_path').
        wav_path: where the decoded wav lives / should be written. The caller
                owns its lifecycle (decode → capture → cleanup) so disk stays
                bounded; if `decode` and it is absent, we decode the m4a here.
        cache_dir: pipeline cache dir (data/cache) — bp48/nnls/musx caches.
        oracle_kwargs: infer_chords_v1 kwargs (defaults to the frozen live path).

    Returns a JSON-serializable per-stage golden dict. Each stage is wrapped:
    on failure the stage records {'__error__': ...} (honesty bar — a stage that
    can't be captured is recorded, never faked).
    """
    import time as _time

    from harmonia.eval.benchmark_set import LIVE_ORACLE_KWARGS

    oracle_kwargs = dict(oracle_kwargs if oracle_kwargs is not None else LIVE_ORACLE_KWARGS)
    song_id = entry["song_id"]
    m4a_path = Path(entry["audio_path"])
    wav_path = Path(wav_path)
    if decode and not wav_path.exists():
        decode_to_wav(m4a_path, wav_path)

    stages: dict = {}

    def _run(name_group, fn):
        try:
            res = fn()
            if isinstance(res, dict) and name_group is None:
                stages.update(res)
            else:
                stages[name_group] = res
        except Exception as exc:  # noqa: BLE001 — record, never fake
            import traceback
            stages[name_group or "unknown"] = {
                "__error__": f"{type(exc).__name__}: {exc}",
                "__traceback__": traceback.format_exc()[-800:],
            }

    beats = None
    try:
        beats = _beats_stage(wav_path, oracle_kwargs.get("beat_period_mode", "bestfit"))
    except Exception as exc:  # noqa: BLE001
        import traceback
        beats = {"__error__": f"{type(exc).__name__}: {exc}",
                 "__traceback__": traceback.format_exc()[-800:]}
    bt = beats.pop("_bt", None) if isinstance(beats, dict) else None
    _period = beats.pop("_period", None) if isinstance(beats, dict) else None
    _duration_s = beats.pop("_duration_s", None) if isinstance(beats, dict) else None
    stages["beats"] = beats

    if bt is not None:
        _run(None, lambda: _bp48_stages(wav_path, bt, cache_dir))
    else:
        stages["bp48_features"] = {"__missing__": "no beat grid (beats stage failed)"}

    if bt is not None and _period is not None and _duration_s is not None:
        _run(None, lambda: _nnls24_stages(wav_path, bt, _period, _duration_s, cache_dir))
    else:
        _miss = {"__missing__": "no beat grid/period (beats stage failed)"}
        stages["nnls24_features"] = dict(_miss)
        stages["nnls24_precoalesce"] = dict(_miss)
        stages["nnls24_sections"] = dict(_miss)

    _run(None, lambda: _live_chart_stages(wav_path, cache_dir, oracle_kwargs))

    return {
        "schema_version": CAPTURE_SCHEMA_VERSION,
        "song_id": song_id,
        "audio_path": str(m4a_path),
        "wav_basename": wav_path.name,
        "oracle_kwargs": oracle_kwargs,
        "captured_at": _time.strftime("%Y-%m-%dT%H:%M:%S"),
        "stages_captured": STAGES_CAPTURED,
        "stages_not_yet_captured": STAGES_NOT_YET_CAPTURED,
        "stages": stages,
    }


GOLDEN_DIR = Path(__file__).resolve().parent / "golden" / "frozen_parity"


def save_golden(cap: dict, golden_dir: Path | None = None) -> Path:
    golden_dir = Path(golden_dir) if golden_dir else GOLDEN_DIR
    golden_dir.mkdir(parents=True, exist_ok=True)
    path = golden_dir / f"{cap['song_id']}.golden.json"
    path.write_text(json.dumps(cap, indent=2, sort_keys=True))
    return path


def load_golden(song_id: str, golden_dir: Path | None = None) -> dict:
    golden_dir = Path(golden_dir) if golden_dir else GOLDEN_DIR
    return json.loads((golden_dir / f"{song_id}.golden.json").read_text())


# ── per-stage diff ───────────────────────────────────────────────────────────

def _diff_value(path: str, old, new, float_tol: float, out: list) -> None:
    """Recursive tolerance-aware comparator. Appends divergences to `out`."""
    # array-summary dict
    if isinstance(old, dict) and "__sha256__" in old:
        if not (isinstance(new, dict) and "__sha256__" in new):
            out.append((path, "type-mismatch", old.get("__sha256__"), new)); return
        if old["shape"] != new["shape"]:
            out.append((path + ".shape", "shape", old["shape"], new["shape"])); return
        if old["__sha256__"] == new["__sha256__"]:
            return  # bitwise identical
        # hashes differ → fall back to epsilon on stats
        for k in ("sum", "mean", "min", "max", "l2"):
            ov, nv = old["stats"].get(k), new["stats"].get(k)
            if ov is None or nv is None or abs(ov - nv) > float_tol:
                out.append((path + f".stats.{k}", "float", ov, nv)); return
        out.append((path, "sha256-diff-within-eps", old["__sha256__"][:8], new["__sha256__"][:8]))
        return
    if isinstance(old, dict):
        if not isinstance(new, dict):
            out.append((path, "type-mismatch", type(old).__name__, type(new).__name__)); return
        for k in sorted(set(old) | set(new)):
            if k.startswith("__traceback__") or k in ("captured_at",):
                continue
            if k not in old:
                out.append((path + "." + k, "added", None, "present")); continue
            if k not in new:
                out.append((path + "." + k, "removed", "present", None)); continue
            _diff_value(path + "." + k, old[k], new[k], float_tol, out)
        return
    if isinstance(old, list):
        if not isinstance(new, list):
            out.append((path, "type-mismatch", "list", type(new).__name__)); return
        if len(old) != len(new):
            out.append((path + ".len", "length", len(old), len(new))); return
        for i, (o, n) in enumerate(zip(old, new)):
            _diff_value(f"{path}[{i}]", o, n, float_tol, out)
        return
    if isinstance(old, bool) or isinstance(new, bool):
        if old != new:
            out.append((path, "bool", old, new)); return
    if isinstance(old, float) or isinstance(new, float):
        try:
            if abs(float(old) - float(new)) > float_tol:
                out.append((path, "float", old, new))
        except (TypeError, ValueError):
            if old != new:
                out.append((path, "value", old, new))
        return
    # int / str / None: exact
    if old != new:
        out.append((path, "exact", old, new))


def diff(golden: dict, fresh: dict, *, float_tol: float = 1e-6) -> dict:
    """Diff two captures stage-by-stage. Labels/ints/strings/shapes are EXACT;
    float vectors compared bitwise (sha256) then epsilon on stats. Returns a
    report dict; report['divergences'] == [] means zero divergence (gate pass).
    """
    out: list = []
    g_stages = golden.get("stages", {})
    f_stages = fresh.get("stages", {})
    per_stage: dict = {}
    for stage in sorted(set(g_stages) | set(f_stages)):
        local: list = []
        if stage not in g_stages:
            local.append((stage, "added-stage", None, "present"))
        elif stage not in f_stages:
            local.append((stage, "removed-stage", "present", None))
        else:
            _diff_value(stage, g_stages[stage], f_stages[stage], float_tol, local)
        per_stage[stage] = {"diverged": len(local), "detail": local[:20]}
        out.extend(local)
    return {
        "song_id": golden.get("song_id"),
        "total_divergences": len(out),
        "zero_divergence": len(out) == 0,
        "per_stage": per_stage,
        "divergences": out,
    }


def _diff_report_str(rep: dict) -> str:
    tag = "✓ ZERO-DIVERGENCE" if rep["zero_divergence"] else f"✗ {rep['total_divergences']} DIVERGENCE(S)"
    lines = [f"{tag}: {rep['song_id']}"]
    for stage, s in rep["per_stage"].items():
        mark = "·" if s["diverged"] == 0 else "✗"
        lines.append(f"  {mark} {stage:<18} diverged={s['diverged']}")
        for (p, kind, o, n) in s["detail"]:
            if s["diverged"]:
                lines.append(f"      {p} [{kind}] {str(o)[:40]} → {str(n)[:40]}")
    return "\n".join(lines)


# ── CLI: capture the benchmark set, and run the self-consistency gate ─────────

def _cli() -> int:  # pragma: no cover
    import argparse
    import shutil

    from harmonia.eval.benchmark_set import (
        LIVE_ORACLE_KWARGS, REPO, parity_songs, write_manifest_json,
    )

    ap = argparse.ArgumentParser(description="Frozen parity benchmark harness")
    ap.add_argument("cmd", choices=["capture", "gate", "verify"],
                    help="capture goldens | in-process 2-run gate | cross-process verify vs saved golden")
    ap.add_argument("--songs", default="", help="comma-sep song_ids (default: all capturable)")
    ap.add_argument("--cache-dir", default=str(REPO / "data" / "cache"))
    ap.add_argument("--min-free-gib", type=float, default=1.5)
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir)
    wav_dir = REPO / "data" / "cache" / "_parity_wav_tmp"
    wav_dir.mkdir(parents=True, exist_ok=True)

    songs = [s for s in parity_songs() if s["capturable"]]
    if args.songs:
        want = set(args.songs.split(","))
        songs = [s for s in songs if s["song_id"] in want]

    def free_gib() -> float:
        return round(shutil.disk_usage(REPO).free / 2**30, 3)

    write_manifest_json()
    print(f"[manifest written]  disk_free={free_gib()} GiB  songs={len(songs)}")

    if args.cmd == "capture":
        for s in songs:
            if free_gib() < args.min_free_gib:
                print(f"STOP: disk {free_gib()} GiB < {args.min_free_gib} GiB floor"); return 2
            wav = wav_dir / f"{s['song_id']}.wav"
            print(f"\n=== {s['song_id']} (dur~{s['duration_s']}s)  free={free_gib()} GiB ===")
            cap = capture(s, wav_path=wav, cache_dir=cache_dir, oracle_kwargs=LIVE_ORACLE_KWARGS)
            p = save_golden(cap)
            wav.unlink(missing_ok=True)
            errs = [k for k, v in cap["stages"].items() if isinstance(v, dict) and ("__error__" in v or "__missing__" in v)]
            print(f"  saved {p.name}  stages_ok={len(cap['stages'])-len(errs)}/{len(cap['stages'])}"
                  f"{'  ERR/MISSING:'+','.join(errs) if errs else ''}  free={free_gib()} GiB")
        return 0

    if args.cmd == "verify":
        # cross-process gate: this is a SEPARATE python process from `capture`.
        # Re-decode a fresh wav (fresh mtime → bp48 re-extracts) + re-capture,
        # then diff against the golden saved by the capture process. Zero
        # divergence ⇒ the pipeline is reproducible across independent processes.
        all_zero = True
        for s in songs:
            if free_gib() < args.min_free_gib:
                print(f"STOP: disk {free_gib()} GiB < floor"); return 2
            try:
                golden = load_golden(s["song_id"])
            except FileNotFoundError:
                print(f"  {s['song_id']}: NO GOLDEN — run `capture` first"); all_zero = False; continue
            wav = wav_dir / f"{s['song_id']}.wav"
            decode_to_wav(Path(s["audio_path"]), wav)
            fresh = capture(s, wav_path=wav, cache_dir=cache_dir, decode=False)
            wav.unlink(missing_ok=True)
            rep = diff(golden, fresh)
            all_zero &= rep["zero_divergence"]
            print(_diff_report_str(rep) + f"   free={free_gib()} GiB")
        print(f"\nGATE (cross-process): {'GREEN (all zero-divergence)' if all_zero else 'RED (divergence found)'}")
        return 0 if all_zero else 1

    # gate: capture twice per song in-process independent, diff to zero
    all_zero = True
    for s in songs:
        if free_gib() < args.min_free_gib:
            print(f"STOP: disk {free_gib()} GiB < floor"); return 2
        wav = wav_dir / f"{s['song_id']}.wav"
        decode_to_wav(Path(s["audio_path"]), wav)
        c1 = capture(s, wav_path=wav, cache_dir=cache_dir, decode=False)
        c2 = capture(s, wav_path=wav, cache_dir=cache_dir, decode=False)
        wav.unlink(missing_ok=True)
        rep = diff(c1, c2)
        all_zero &= rep["zero_divergence"]
        print(_diff_report_str(rep) + f"   free={free_gib()} GiB")
    print(f"\nGATE: {'GREEN (all zero-divergence)' if all_zero else 'RED (nondeterminism found)'}")
    return 0 if all_zero else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_cli())
