"""Jam Mode (2026-07-20) — near-live loop detection from a growing mic buffer.

The premise (validated first, cheaply, per CLAUDE.md rule #2 — see
scratchpad/jam_premise_check.py and the session's known_issues.md entry): a
fast, no-music-x-lab NNLS-24 decode of a SHORT real-audio buffer, plus a
simple autocorrelation over the resulting per-beat chord-label sequence,
recovers a real repeating loop with rising per-slot vote confidence as more
repeats accumulate — using entirely existing, already-tested decode machinery
(``harmonia.models.chord_pipeline_v1``'s draft-pass helpers), nothing new on
the DSP side.

This module is the SESSION/STATE layer on top of that: accumulate a mic
buffer, periodically redecode it and re-detect the loop's period, refine a
per-slot majority vote across completed repeats within the CURRENT part, and
flag a probable part change when a sustained recent run stops agreeing with
that vote. Deliberately "near-live, not live-live" per the user's own framing
("ce n'est pas obligé d'être en live live, on a le droit d'avoir un delta de
quelques secondes") — accuracy over latency.

Design note on why every ``update()`` redecodes the WHOLE buffer from
scratch rather than accumulating state across polls: an incremental /
persisted vote histogram would need the beat grid's phase to stay pinned
across polls, but each poll's beat-tracking pass is independent and can
shift its phase slightly as more audio arrives — persisting votes across
that would silently misalign old beats into the wrong slot over time
(exactly the calibration-drift trap CLAUDE.md rule #1 warns about). Fully
self-consistent, at the cost of getting slower as a jam runs long — flagged
below, not solved.

Does NOT solve: true low-latency (<1s) response, incremental (non-full-
buffer) redecoding, or more than a two-way (current part vs. new part) split
— a jam with 3+ distinct sections will keep detecting "a new part" against
whichever part is current rather than recognizing a RETURN to an earlier
one. Flagged, not built, given the scope of this pass.
"""
from __future__ import annotations

import logging
import time
from collections import Counter
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# A part-change needs a sustained mismatch run, not one passing chord — the
# user explicitly allows "a few seconds' delta" of latency/error here in
# exchange for not falsely splitting on a single embellished passing chord.
PART_CHANGE_MIN_MISMATCH_S = 6.0
PART_CHANGE_MISMATCH_RATE = 0.6   # ≥60% of that window must disagree with the vote

# Loop detection looks at only the RECENT tail of the current part, not the
# whole thing from its start — confirmed necessary on real audio (a jam's own
# warm-up/noisy lead-in, or just the song's intro before a real vamp settles
# in, dilutes a whole-buffer autocorrelation score below threshold even once
# the last 20+ seconds are a clean, confidently-repeating loop). This also
# caps redecode cost from growing unboundedly with a long current part.
RECENT_WINDOW_BEATS = 64


def _beat_grid(y: np.ndarray, sr: int) -> tuple[np.ndarray, float]:
    """Same tempo-grid de-jitter as chord_pipeline_v1's stage 2 (reused, not
    reimplemented, so this can never silently calibrate differently)."""
    import librosa
    from harmonia.models.chord_pipeline_v1 import _bestfit_beat_period

    duration_s = len(y) / sr
    tempo_arr, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    tempo_bpm = float(np.atleast_1d(tempo_arr)[0])
    beat_times_raw = librosa.frames_to_time(beat_frames, sr=sr)
    if len(beat_times_raw) < 4:
        return np.array([0.0, duration_s]), 0.0
    period = 60.0 / max(tempo_bpm, 1.0)
    period = _bestfit_beat_period(beat_times_raw, period)
    ang = 2 * np.pi * (beat_times_raw % period) / period
    phase = (np.angle(np.mean(np.exp(1j * ang))) % (2 * np.pi)) * period / (2 * np.pi)
    bt = np.arange(phase, duration_s, period)
    bt = np.unique(np.concatenate([[0.0], bt, [duration_s]]))
    return bt, 60.0 / period


def fast_draft_decode(audio_path: Path) -> "dict | None":
    """Beat-track + NNLS-24 heads decode, NO music-x-lab (too slow for
    near-live use) — the exact same per-segment logic as chord_pipeline_v1's
    "draft" progress_cb branch, factored out so jam mode can call it directly
    on a short buffer instead of going through the full analyze pipeline.

    Returns None on anything too short/silent to beat-track (caller treats
    that as "nothing to show yet", not an error).
    """
    import soundfile as sf
    from harmonia.models import nnls_features as nf
    from harmonia.models.chord_pipeline_v1 import (
        _root_change_segs, _label_segments, _coalesce_labeled,
        _drop_leading_outlier, _nnls_no_chord_segs,
    )

    heads = nf.get_heads()
    if heads is None:
        return None

    y, sr = sf.read(audio_path)
    y = (y.mean(1) if y.ndim > 1 else y).astype("float32")
    bt, tempo_bpm = _beat_grid(y, sr)
    if tempo_bpm <= 0 or len(bt) < 5:
        return None
    period = 60.0 / tempo_bpm

    # extract_bothchroma caches by filename STEM — jam mode redecodes a
    # GROWING buffer under the SAME path each poll, so a stale cache hit
    # would silently serve last poll's (shorter) features forever. Disable
    # caching here; this is exactly the calibration trap CLAUDE.md rule #1
    # warns about (a cache key that doesn't cover the thing that changed).
    arr, times = nf.extract_bothchroma(audio_path, use_cache=False)
    feat = nf.pool_beats(arr, times, bt)
    if len(feat) < 5:
        return None
    beat_proba = heads.root_proba(feat)
    segs = _root_change_segs(beat_proba)
    seg_bounds = [(float(bt[s]), float(bt[min(e, len(bt) - 1)])) for (s, e) in segs]
    bass_half = feat[:, :12]
    no_chord = _nnls_no_chord_segs(arr, times, bt, segs)
    labeled = _label_segments(segs, seg_bounds, beat_proba, feat, bass_half, heads,
                              seg_no_chord=no_chord)
    coalesced = _coalesce_labeled(labeled)
    coalesced = _drop_leading_outlier(coalesced, period)
    if not coalesced:
        return None

    # Per-beat label lookup — the unit loop-detection operates on, since a
    # live jam has no known bar length a priori (unlike the batch pipeline's
    # bar-locked sections).
    beat_labels: list[str] = []
    ci = 0
    for bi in range(len(bt) - 1):
        t = bt[bi]
        while ci < len(coalesced) - 1 and t >= coalesced[ci][1]:
            ci += 1
        beat_labels.append(coalesced[ci][2])

    return {"tempo_bpm": tempo_bpm, "period_s": period, "bt": bt,
            "beat_labels": beat_labels}


def detect_loop_period(beat_labels: list[str], min_p: int = 4, max_p: int = 32,
                        min_score: float = 0.65) -> "tuple[int, float] | None":
    """Smallest period P (Occam — same tie-break convention as
    ``_fold_bar_run``) whose lag-P autocorrelation over LABEL identity (root+
    quality, so a re-harmonized repeat doesn't falsely match) clears
    ``min_score``. Returns None (abstain) below that.

    ``min_score`` is deliberately lower than a first pass at 0.75 (see
    scratchpad/jam_premise_check.py's follow-up debugging, 2026-07-20): real
    audio's true period scored only 0.696 on a genuine, audibly clean 2-chord
    vamp — per-repeat classifier noise on one ambiguous chord (dim vs. min)
    was enough to push it under 0.75, which then silently locked onto 2×the
    true period instead (a correct but needlessly-doubled loop — the same
    "octave" ambiguity as this project's known tempo-doubling issues). Lower
    threshold trades some false-lock risk on truly noisy/silent input for
    correctly finding the SMALLEST real period on real, moderately-noisy
    audio — matches the user's own explicit preference for responsiveness
    over precision here ("un delta de quelques secondes où on se trompe").
    """
    n = len(beat_labels)
    labels = np.asarray(beat_labels, dtype=object)
    for p in range(min_p, min(max_p, n // 2) + 1):
        score = float(np.mean(labels[:-p] == labels[p:]))
        if score >= min_score:
            return p, score
    return None


class LoopVotes:
    """Per-slot majority vote over completed loop repeats — the "gets cleaner
    the more times it's played" mechanism the user asked for. Built fresh
    from one label sequence each time (see module docstring for why this is
    never persisted/accumulated across polls)."""

    def __init__(self, period: int):
        self.period = period
        self.slots: list[Counter] = [Counter() for _ in range(period)]
        self.n_beats = 0

    def add(self, beat_labels: list[str]) -> None:
        for bi, lbl in enumerate(beat_labels):
            self.slots[bi % self.period][lbl] += 1
        self.n_beats = len(beat_labels)

    def best(self) -> list[dict]:
        out = []
        for slot in self.slots:
            if not slot:
                out.append({"label": "N", "confidence": 0.0})
                continue
            lbl, cnt = slot.most_common(1)[0]
            out.append({"label": lbl, "confidence": round(cnt / sum(slot.values()), 3)})
        return out

    def n_reps(self) -> int:
        return self.n_beats // self.period if self.period else 0


class JamSession:
    """One live jam: a growing mono float32 buffer + the current best-fit
    loop, split into finalized "parts" whenever a sustained new pattern is
    detected. NOT thread-safe beyond Python's GIL — fine for the dev server's
    single-process in-memory session dict; would need real locking under a
    multi-worker deployment."""

    def __init__(self, sr: int = 44100):
        self.sr = sr
        self.buf = np.zeros(0, dtype=np.float32)
        self.created = time.time()
        self.parts: list[dict] = []
        self.part_boundary_s = 0.0     # current part starts here in buffer-time
        self.cur: "dict | None" = None  # {"loop","period_beats","n_reps"} or None

    def append(self, chunk: np.ndarray) -> None:
        self.buf = np.concatenate([self.buf, chunk.astype(np.float32)])

    def elapsed_s(self) -> float:
        return len(self.buf) / self.sr

    def update(self, tmp_wav: Path) -> dict:
        import soundfile as sf
        sf.write(tmp_wav, self.buf, self.sr)
        decoded = fast_draft_decode(tmp_wav)
        if decoded is None:
            self.cur = None
            return self.state()

        bt, beat_labels = decoded["bt"], decoded["beat_labels"]
        part_start_idx = int(np.searchsorted(bt[:-1], self.part_boundary_s))
        start_idx = max(part_start_idx, len(beat_labels) - RECENT_WINDOW_BEATS)
        cur_labels = beat_labels[start_idx:]

        period_info = detect_loop_period(cur_labels)
        if period_info is None:
            self.cur = None
            return self.state()

        period, _score = period_info
        votes = LoopVotes(period)
        votes.add(cur_labels)

        new_part_at: "float | None" = None
        if votes.n_reps() >= 2:
            window_beats = max(period, int(round(PART_CHANGE_MIN_MISMATCH_S / decoded["period_s"])))
            tail_len = min(window_beats, len(cur_labels))
            tail_off = len(cur_labels) - tail_len
            best = votes.best()
            mismatches = sum(
                1 for i in range(tail_len)
                if cur_labels[tail_off + i] != best[(tail_off + i) % period]["label"]
            )
            if tail_len and mismatches / tail_len >= PART_CHANGE_MISMATCH_RATE:
                new_part_at = float(bt[start_idx + tail_off])

        if new_part_at is not None:
            self.parts.append({
                "loop": votes.best(), "period_beats": period,
                "tempo_bpm": round(decoded["tempo_bpm"], 1), "n_reps": votes.n_reps(),
            })
            self.part_boundary_s = new_part_at
            self.cur = None
        else:
            self.cur = {"loop": votes.best(), "period_beats": period, "n_reps": votes.n_reps(),
                       "tempo_bpm": round(decoded["tempo_bpm"], 1)}
        return self.state()

    def state(self) -> dict:
        return {"elapsed_s": round(self.elapsed_s(), 1), "parts": self.parts, "current": self.cur}
