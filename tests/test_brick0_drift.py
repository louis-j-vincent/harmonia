"""Unit tests for the Brick 0 aligner-v5 TEMPO-DRIFT detector + piecewise-constant-
per-section tempo model (scripts/brick0_propose.py).

Audio-free: they exercise the classifier/grid math on synthetic offset ramps that
mirror the real batch-1 measurements, so the GUARDRAIL (fire only on a monotone,
materially-large, well-fit ramp; leave constant/rubato songs untouched) is locked in
without a Beat This! / librosa run.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

import brick0_propose as bp  # noqa: E402


def _rows(ts, deltas, agr=0.35, n_beats=64):
    """(t_center, delta, agr_best, n_beats) tuples in the offset_ramp format."""
    return [(float(t), float(d), float(agr), n_beats) for t, d in zip(ts, deltas)]


# ── the DRIFT case: a clean monotone ramp (Blue Bossa: +0.45 -> -1.25 over 22) ──

def test_monotone_ramp_classified_as_drift():
    bp_grid = 60.0 / 170.75                      # ~0.3515 s (Blue Bossa)
    ts = np.linspace(22.0, 494.0, 22)
    deltas = np.linspace(0.45, -1.25, 22)
    deltas += 0.05 * np.sin(np.arange(22))       # mild jitter, still monotone-ish
    d = bp.detect_drift(_rows(ts, deltas), bp_grid)
    assert d["classification"] == "drift"
    assert d["apply"] is True
    assert abs(d["pearson"]) >= bp._DRIFT_R_MIN
    assert d["span_beats"] >= bp._DRIFT_SPAN_BEATS
    # coeffs must encode a real negative slope (accelerating band)
    assert d["slope_s_per_s"] < 0


# ── the GUARDRAIL cases: constant + rubato songs must NOT get drift ─────────────

def test_flat_tiny_ramp_not_applied():
    """Blue Bossa backing: monotone (Pearson ~1) but span 0.2s = half a beat.
    The MAGNITUDE gate must veto it (else the guardrail is too weak)."""
    bp_grid = 60.0 / 150.0                        # 0.4 s
    ts = np.linspace(17.5, 299.0, 12)
    deltas = np.linspace(-0.10, 0.10, 12)
    d = bp.detect_drift(_rows(ts, deltas, agr=0.53), bp_grid)
    assert d["apply"] is False
    assert d["classification"] == "flat"
    assert d["span_beats"] < bp._DRIFT_SPAN_BEATS


def test_erratic_ramp_not_applied():
    """Close-To-You-style section jitter: swings both ways, low monotonicity, even
    with a large span -> 'erratic', never chased (that would be the banned free warp)."""
    bp_grid = 0.5
    ts = np.linspace(10.0, 60.0, 6)
    deltas = [0.0, 1.5, -0.2, 1.6, -0.1, 1.4]
    d = bp.detect_drift(_rows(ts, deltas), bp_grid)
    assert d["apply"] is False
    assert d["classification"] in ("erratic", "flat")
    assert abs(d["pearson"]) < bp._DRIFT_R_MIN


def test_insufficient_units_not_applied():
    d = bp.detect_drift(_rows([10.0, 20.0], [0.0, -1.0]), 0.35)
    assert d["apply"] is False
    assert d["classification"] == "insufficient"


# ── the grid warp: smooth, monotone, and it FLATTENS a synthetic drift ─────────

def test_drift_grid_is_monotone_and_matches_offset_model():
    beats = np.arange(0.0, 200.0, 0.3515)
    coeffs = [0.0, -0.004, 0.5]                   # linear: delta(t) = -0.004 t + 0.5
    w = bp.drift_grid(beats, coeffs)
    assert np.all(np.diff(w) > 0)                 # strictly increasing (usable grid)
    expect = beats + (coeffs[1] * beats + coeffs[2])
    assert np.allclose(w, expect, atol=1e-9)


def test_drift_grid_flattens_a_known_ramp():
    """If chords truly sit at grid + delta_true(t), warping the grid by the fitted
    delta model must drive the residual offset (delta_true - warp) toward ~0."""
    beats = np.arange(0.0, 500.0, 0.3515)
    coeffs = [0.0, -0.004, 0.5]                   # the drift the band actually has
    true_pos = beats + bp._drift_offset(coeffs, beats)     # where the audio really is
    w = bp.drift_grid(beats, coeffs)
    resid = true_pos - w                          # residual after correction
    assert np.max(np.abs(resid)) < 1e-6           # fully flattened


# ── per-section BPM series: smooth, monotone, no jumps (Louis's guardrail) ─────

def test_section_bpms_smooth_and_monotone():
    # A QUADRATIC offset (constant acceleration) => the per-section BPM genuinely
    # RAMPS (a pure-linear offset would give a single re-tempo = equal BPMs). These
    # coeffs mirror Blue Bossa's flat-then-drop curve (~170 -> ~173 BPM).
    beat_period = 0.3515
    beats = np.arange(0.0, 520.0, beat_period)
    coeffs = [-1.5e-5, 0.003, 0.4]                # c2<0 => band accelerates
    w = bp.drift_grid(beats, coeffs)
    assert np.all(np.diff(w) > 0)                 # grid still usable
    pls = [bp.Placement(chorus=c, sec_idx=0, label="A", start_beat=c * 64,
                        n_beats=64, agreement=0.35, occ=c) for c in range(22)]
    series = np.array([s["bpm"] for s in bp.section_bpms(pls, w, beat_period)])
    assert not np.any(np.isnan(series))
    steps = np.diff(series)
    # monotone (band accelerates) and — the GUARDRAIL — NO jumps: each section can
    # only TRACK the single global ramp, so successive BPM steps are tiny + smooth.
    assert np.all(steps > 0)                      # strictly increasing (real ramp)
    assert series.max() - series.min() > 1.0      # a genuine drift, not degenerate
    assert np.max(np.abs(steps)) < 0.5            # cannot jump — only track


# ── WINDOWED (localized) drift — aligner v6 (Louis round 10) ───────────────────
# The whole-song detector (above) only fires on a monotone WHOLE-SONG ramp. v6
# generalizes it to piecewise: change-point-segment the ramp into DRIFT spans
# (monotone, materially large, well-fit) vs CONSTANT spans (flat). These lock in
# the two localized cases (Every Breath's flat-head-then-tail-drift, Close To You's
# converging-head-then-flat) + the guardrails (constant/rubato/sub-threshold decline).


def test_windowed_flat_head_then_drift_tail():
    """Every Breath: FLAT 0->~42s then an accelerating DRIFT tail. The whole-song
    Pearson washes out, but the piecewise detector must find a drift span."""
    bp_grid = 60.0 / 114.0                          # ~0.526 s (Every Breath)
    ts = np.linspace(5.0, 120.0, 16)
    deltas = np.concatenate([np.full(6, 0.2), np.linspace(0.2, -1.6, 10)])
    d = bp.detect_windowed_drift(_rows(ts, deltas), bp_grid)
    assert d["classification"] == "drift"
    assert d["apply"] is True
    assert d["windowed"] is True                    # >1 segment => genuinely windowed
    assert d["n_drift"] >= 1
    kinds = [s["kind"] for s in d["segments"]]
    assert "drift" in kinds
    # a drift span must clear the SAME material-magnitude gate as the whole-song case
    drift_seg = next(s for s in d["segments"] if s["kind"] == "drift")
    assert drift_seg["span_beats"] >= bp._DRIFT_SPAN_BEATS


def test_windowed_converging_head_then_flat():
    """Close To You: a CONVERGING head drift (large enough), then FLAT. The detector
    must localize the drift to the HEAD and leave the tail constant."""
    bp_grid = 60.0 / 88.9                            # ~0.675 s (Close To You)
    ts = np.linspace(5.0, 180.0, 16)
    deltas = np.concatenate([np.linspace(1.5, 0.0, 8), np.full(8, 0.0)])
    d = bp.detect_windowed_drift(_rows(ts, deltas), bp_grid)
    assert d["classification"] == "drift"
    assert d["windowed"] is True
    # head is the drift span, tail is flat
    assert d["segments"][0]["kind"] == "drift"
    assert d["segments"][-1]["kind"] == "flat"


def test_windowed_whole_song_is_single_segment():
    """Blue Bossa's clean monotone whole-song ramp must stay a SINGLE drift segment
    (windowed=False) so it keeps its v5 whole-song (<=quadratic) treatment."""
    bp_grid = 60.0 / 170.75
    ts = np.linspace(22.0, 494.0, 22)
    deltas = np.linspace(0.45, -1.25, 22)
    d = bp.detect_windowed_drift(_rows(ts, deltas), bp_grid)
    assert d["classification"] == "drift"
    assert d["apply"] is True
    assert d["windowed"] is False                   # one segment -> not windowed
    assert d["n_segments"] == 1


def test_windowed_constant_song_not_applied():
    """A flat constant ramp (Blue Bossa backing) must NOT be carved into a spurious
    drift span by the change-point search."""
    bp_grid = 60.0 / 150.0
    ts = np.linspace(17.5, 299.0, 12)
    deltas = np.linspace(-0.10, 0.10, 12)
    d = bp.detect_windowed_drift(_rows(ts, deltas, agr=0.53), bp_grid)
    assert d["apply"] is False
    assert d["classification"] == "flat"


def test_windowed_erratic_span_vetoes():
    """A materially-large but NON-monotone span (Georgia-style rubato) must veto the
    whole song as 'erratic' — never chased."""
    bp_grid = 0.5
    ts = np.linspace(10.0, 70.0, 8)
    deltas = [0.0, 1.6, -0.2, 1.7, -0.1, 1.5, -0.3, 1.6]
    d = bp.detect_windowed_drift(_rows(ts, deltas), bp_grid)
    assert d["apply"] is False
    assert d["classification"] == "erratic"


def test_windowed_subthreshold_head_declined():
    """Close To You's REAL head convergence is only ~0.6s (< 1.5 beats): the detector
    localizes the change-point but must DECLINE to warp a sub-threshold drift."""
    bp_grid = 60.0 / 88.9                            # ~0.675 s
    ts = np.linspace(6.0, 165.0, 6)
    deltas = [0.15, -0.5, -0.45, 0.05, 0.4, -0.15]  # the real CTY section ramp
    d = bp.detect_windowed_drift(_rows(ts, deltas), bp_grid)
    assert d["apply"] is False
    assert d["classification"] == "flat"            # every span below the beat-gate


def test_windowed_insufficient_units():
    d = bp.detect_windowed_drift(_rows([10.0, 20.0], [0.0, -1.0]), 0.35)
    assert d["apply"] is False
    assert d["classification"] == "insufficient"


def test_windowed_grid_flattens_a_piecewise_ramp():
    """If chords sit at grid + a piecewise offset (flat, then a linear drift, then
    held), warping by the knot model must drive the residual toward ~0 over the
    labeled span and keep the grid strictly increasing (usable)."""
    beats = np.arange(0.0, 200.0, 0.3515)
    knots = [(0.0, 0.0), (40.0, 0.0), (120.0, -1.5), (200.0, -1.5)]
    kt = np.array([k[0] for k in knots]); kd = np.array([k[1] for k in knots])
    true_pos = beats + np.interp(beats, kt, kd)     # where the audio really is
    w = bp.windowed_drift_grid(beats, knots)
    assert np.all(np.diff(w) > 0)                   # strictly increasing
    resid = true_pos - w
    assert np.max(np.abs(resid)) < 1e-6             # fully flattened


def test_segment_ramp_finds_the_changepoint():
    """The change-point DP must place the break at the flat->drift transition, not
    mid-segment (a penalised, not greedy, fit)."""
    ts = np.linspace(0.0, 150.0, 16)
    ds = np.concatenate([np.full(8, 0.1), np.linspace(0.1, -2.0, 8)])
    w = np.full(16, 0.4)
    bps = bp._segment_ramp(ts, ds, w)
    assert len(bps) == 1
    assert 6 <= bps[0] <= 9                          # break near index 8 (t~=80)


# ── FREEZE GUARD — aligner v6 (Louis round 10): never overwrite a verified song ──

def test_is_frozen_reads_verified(tmp_path, monkeypatch):
    monkeypatch.setattr(bp, "GOLDEN", tmp_path)
    (tmp_path / "frozen.gt.json").write_text('{"verified": true}')
    (tmp_path / "open.gt.json").write_text('{"verified": false}')
    assert bp.is_frozen("frozen") is True
    assert bp.is_frozen("open") is False
    assert bp.is_frozen("missing") is False          # no file -> not frozen


def test_process_skips_frozen_song(tmp_path, monkeypatch):
    """A verified=true golden must make process() return None and touch NOTHING —
    the guard fires before any audio work (so this stays audio-free)."""
    monkeypatch.setattr(bp, "GOLDEN", tmp_path)
    (tmp_path / "sbm.gt.json").write_text('{"verified": true}')
    before = (tmp_path / "sbm.gt.json").read_bytes()
    song = dict(song_id="sbm", title="Stand By Me", audio="does/not/exist.m4a",
                ireal_file="x", tune_title="y")
    out = bp.process(song, tmp_path, write=True)      # would crash on the fake audio
    assert out is None                                # ...but the guard returns first
    assert (tmp_path / "sbm.gt.json").read_bytes() == before  # byte-identical


# ── FORM-PERIODIC VAMP PROPAGATION — aligner v6b (Louis's Autumn Leaves) ────────
# The min-gap DP opens only the FIRST turnaround vamp and tiles the rest of the
# choruses contiguously, so every later chorus drifts early by the accumulated
# missing vamps. propagate_form_vamps seeds the vamp from that one confirmed gap and
# propagates it after each chorus, snapping to the actual low-agreement turnaround,
# accepted only where it RE-ALIGNS the following chorus. These lock the mechanism +
# the guardrail (no seed / no gain => revert), audio-free via a synthetic per-beat
# chroma where each chorus's beats carry their section chord-tones and the vamp beats
# are flat (low agreement).

def _mk_section(label, tmpl_rows, ck):
    """A minimal Section: template = one-hot chord-tone rows, beat_chords sized to it."""
    tmpl = np.zeros((len(tmpl_rows), 12))
    for i, pcs in enumerate(tmpl_rows):
        for pc in pcs:
            tmpl[i, pc % 12] = 1.0
    return bp.Section(label=label, n_bars=len(tmpl_rows) // 4 or 1,
                      beat_chords=[None] * len(tmpl_rows), template=tmpl, content_key=ck)


def _synth_frames(sections, chorus_starts, N):
    """Per-beat chroma (ftimes==beats so beat_sync_chroma is identity): chorus beats
    carry their section chord-tones (high agreement), everything else is flat (~0)."""
    frames = np.full((N, 12), 0.1)
    for cs in chorus_starts:
        off = 0
        for s in sections:
            for pos in range(s.n_beats):
                b = cs + off + pos
                if b < N:
                    frames[b] = s.template[pos] + 0.02
            off += s.n_beats
    return frames


def _sec_pair():
    A = _mk_section("A", [[0, 4, 7]] * 4 + [[5, 9, 0]] * 4, ck=("A",))
    B = _mk_section("B", [[2, 6, 9]] * 4 + [[7, 11, 2]] * 4, ck=("B",))
    return [A, B]


def test_form_vamp_propagates_and_realigns():
    """3 choruses separated by 8-beat vamps at [0,24,48]; the input placement has only
    the FIRST vamp (chorus 2 tiled contiguous @40, WRONG). Propagation must open the
    second vamp and re-align chorus 2 to its true start (48)."""
    sections = _sec_pair()
    beat_period = 0.5
    N = 68
    beats = np.arange(N) * beat_period
    ftimes = beats.copy()
    frames = _synth_frames(sections, [0, 24, 48], N)
    # input placements = min-gap DP result: one seed vamp, then contiguous (mis-tiled)
    P = bp.Placement
    placements = [
        P(0, 0, "A", 0, 8, 0.9), P(0, 1, "B", 8, 8, 0.9),      # chorus 0
        P(1, 0, "A", 24, 8, 0.9), P(1, 1, "B", 32, 8, 0.9),    # chorus 1 (after seed vamp)
        P(2, 0, "A", 40, 8, 0.3), P(2, 1, "B", 48, 8, 0.3),    # chorus 2 CONTIGUOUS (wrong)
    ]
    new_pl, new_diag, rep = bp.propagate_form_vamps(
        sections, frames, ftimes, beats, beat_period, placements, dict(song_score=0.0))
    assert rep["accepted"] is True
    assert rep["seed_vamp_beats"] == 8                         # the confirmed seed vamp
    assert rep["n_vamps"] == 2                                 # after chorus 0 AND 1
    assert rep["delta"] > 0                                    # self-check improved
    # chorus 2 re-aligned from beat 40 -> 48 (its true, vamp-separated start)
    ch2 = sorted((p for p in new_pl if p.chorus == 2), key=lambda p: p.start_beat)
    assert ch2 and ch2[0].start_beat == 48
    # every propagated chorus now sits on real content -> high agreement
    assert all(p.agreement > 0.5 for p in new_pl)


def test_form_vamp_no_seed_not_applied():
    """A song whose min-gap DP opened NO large gap (contiguous choruses) has no seed
    vamp to propagate -> return unchanged, accepted=False (guardrail)."""
    sections = _sec_pair()
    beat_period = 0.5
    N = 52
    beats = np.arange(N) * beat_period
    frames = _synth_frames(sections, [0, 16, 32], N)           # truly contiguous choruses
    P = bp.Placement
    placements = [
        P(0, 0, "A", 0, 8, 0.9), P(0, 1, "B", 8, 8, 0.9),
        P(1, 0, "A", 16, 8, 0.9), P(1, 1, "B", 24, 8, 0.9),
        P(2, 0, "A", 32, 8, 0.9), P(2, 1, "B", 40, 8, 0.9),
    ]
    new_pl, new_diag, rep = bp.propagate_form_vamps(
        sections, frames, beats.copy(), beats, beat_period, placements, dict(song_score=0.9))
    assert rep["accepted"] is False
    assert "no seed vamp" in rep["reason"]
    assert new_pl is placements                               # returned unchanged


def test_form_vamp_reverts_when_no_gain():
    """A seed vamp exists but the audio is FLAT everywhere (no real turnarounds to snap
    to): propagation must find no confirming vamp and revert (accepted=False)."""
    sections = _sec_pair()
    beat_period = 0.5
    N = 68
    beats = np.arange(N) * beat_period
    frames = np.full((N, 12), 0.1)                            # flat: nothing to align to
    P = bp.Placement
    placements = [
        P(0, 0, "A", 0, 8, 0.1), P(0, 1, "B", 8, 8, 0.1),
        P(1, 0, "A", 24, 8, 0.1), P(1, 1, "B", 32, 8, 0.1),   # seed vamp = 8 beats
        P(2, 0, "A", 40, 8, 0.1), P(2, 1, "B", 48, 8, 0.1),
    ]
    new_pl, new_diag, rep = bp.propagate_form_vamps(
        sections, frames, beats.copy(), beats, beat_period, placements, dict(song_score=0.0))
    assert rep["accepted"] is False
    assert new_pl is placements


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
