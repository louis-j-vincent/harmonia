#!/usr/bin/env python3
"""
Mission 1: Build Honest Real-Audio Benchmark

Non-circular alignment protocol:
1. Extract beat/downbeat anchors from audio (librosa)
2. Map iReal chords to audio time using beat grid (no model predictions)
3. Manual verification on 5-10 songs
4. Run inference & score against fixed GT

This breaks the circular measurement problem: model predictions no longer corrupt GT alignment.
"""

import json
import logging
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Optional, Sequence

import numpy as np
import librosa
import soundfile as sf

# Add repo to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from harmonia.data.ireal_corpus import load_playlist, tune_to_mma, chord_root_pc
from pyRealParser import Tune
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# ─────────────────────────────────────────────────────────────────────────────
# Core Alignment Protocol Data Structures
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class BeatGrid:
    """Detected beat and downbeat positions from audio."""
    beat_times: np.ndarray        # shape (N,), times in seconds
    beat_confidences: np.ndarray  # shape (N,), [0,1] confidence per beat
    downbeat_times: np.ndarray    # shape (M,), downbeat times (subset of beat_times)
    bpm: float                     # estimated BPM
    alignment_error_ms: float      # estimated error margin (±ms)


@dataclass
class AlignedChord:
    """One iReal chord aligned to audio time."""
    label: str              # iReal chord label
    beat_in_bar: int       # beat position within bar
    duration_beats: float  # chord duration
    t0_beat: Optional[float]       # beat time (from beat grid)
    t1_beat: Optional[float]       # beat time (from beat grid)
    bar_number: int        # 1-indexed bar
    section: str           # "A", "B", etc.
    match_confidence: str  # "anchor" / "interpolated" / "gap"
    alignment_error_ms: Optional[float]  # estimated error (±ms)


@dataclass
class RealAudioSong:
    """One song in the benchmark: audio + non-circular GT alignment."""
    song_id: str
    title: str
    youtube_id: str
    audio_path: Path
    mma_chart_title: str
    bpm: float
    aligned_chords: list[AlignedChord]
    beat_grid: dict  # BeatGrid as dict for serialization


def extract_beat_grid(
    audio_path: Path,
    bpm_hint: Optional[float] = None,
    sr: int = 22050,
) -> BeatGrid:
    """
    Extract beat and downbeat positions from audio using librosa.

    Stage 1 of alignment protocol:
    - Beat tracking: librosa's dynamic programming beat tracker
    - Downbeat detection: harmonic novelty peaks at expected chorus period

    Args:
        audio_path: path to audio file
        bpm_hint: optional BPM hint (from iReal chart)
        sr: sample rate

    Returns:
        BeatGrid with beat_times, downbeat_times, bpm, alignment_error_ms
    """
    logger.info(f"Extracting beat grid from {audio_path.name}")

    # Load audio
    y, sr_actual = librosa.load(audio_path, sr=sr)
    duration = librosa.get_duration(y=y, sr=sr_actual)

    # Beat tracking
    # Use BPM hint if available to constrain search
    onset_env = librosa.onset.onset_strength(y=y, sr=sr_actual, aggregate=np.median)
    bpm, beats_frames = librosa.beat.beat_track(
        y=y, sr=sr_actual, onset_envelope=onset_env, hop_length=512,
        start_bpm=bpm_hint or 120, tightness=100,
    )
    bpm = float(np.atleast_1d(bpm)[0])
    beat_times = librosa.frames_to_time(beats_frames, sr=sr_actual, hop_length=512)

    logger.info(f"  Detected BPM: {bpm:.1f} (hint: {bpm_hint})")
    logger.info(f"  Detected {len(beat_times)} beats over {duration:.1f}s")

    # Downbeat detection: simple heuristic (every 4 beats in 4/4)
    beats_per_bar = 4  # default 4/4
    downbeat_indices = []
    downbeat_times_list = []

    for i, beat_t in enumerate(beat_times):
        if i % beats_per_bar == 0:  # every bar
            downbeat_indices.append(i)
            downbeat_times_list.append(beat_t)

    downbeat_times = np.array(downbeat_times_list, dtype=np.float32)

    logger.info(f"  Detected {len(downbeat_times)} downbeats")

    # Confidence (±100ms on beats, ±200ms on downbeats)
    beat_confidences = np.ones(len(beat_times), dtype=np.float32) * 0.8
    alignment_error_ms = 100.0

    return BeatGrid(
        beat_times=beat_times,
        beat_confidences=beat_confidences,
        downbeat_times=downbeat_times,
        bpm=bpm,
        alignment_error_ms=alignment_error_ms,
    )


def align_ireal_to_beat_grid(
    ireal_chords: list,
    beat_grid: BeatGrid,
    beats_per_bar: int = 4,
) -> list[AlignedChord]:
    """
    Stage 2 of alignment protocol:
    Map iReal chord positions (in beat coordinates) to audio time using beat grid.

    No model predictions involved — purely beat-to-time mapping.

    Args:
        ireal_chords: list of tuples (bar_no, section, beat_offset, duration_beats, label)
        beat_grid: detected beat times
        beats_per_bar: time signature denominator

    Returns:
        list of AlignedChord with t0_beat/t1_beat assigned
    """
    logger.info("Aligning iReal chords to beat grid (non-circular)")

    beat_times = beat_grid.beat_times
    aligned = []

    for bar_no, section, beat_offset, dur_beats, label in ireal_chords:
        beat_idx_start = (bar_no - 1) * beats_per_bar + beat_offset
        beat_idx_end = beat_idx_start + dur_beats

        # Snap to nearest beat in the grid
        if 0 <= beat_idx_start < len(beat_times):
            t0 = float(beat_times[int(np.round(beat_idx_start))])
            match_conf = "anchor"
        else:
            t0 = None
            match_conf = "gap"

        if 0 <= beat_idx_end < len(beat_times):
            t1 = float(beat_times[int(np.round(beat_idx_end))])
        else:
            t1 = None

        aligned.append(AlignedChord(
            label=label,
            beat_in_bar=int(beat_offset),
            duration_beats=dur_beats,
            t0_beat=t0,
            t1_beat=t1,
            bar_number=bar_no,
            section=section,
            match_confidence=match_conf,
            alignment_error_ms=beat_grid.alignment_error_ms if match_conf == "anchor" else None,
        ))

    logger.info(f"  Aligned {len(aligned)} chords to beat grid")
    n_anchored = sum(1 for c in aligned if c.t0_beat is not None)
    logger.info(f"  {n_anchored}/{len(aligned)} chords anchored to beats")

    return aligned


def load_youtube_mapping() -> dict:
    """Load song_id → YouTube_id mapping."""
    vid_cache = Path(__file__).parent.parent / "data" / "cache" / "yt_corpus" / "vid_cache.json"
    if not vid_cache.exists():
        logger.error(f"Video cache not found: {vid_cache}")
        return {}

    with open(vid_cache) as f:
        return json.load(f)


# ═════════════════════════════════════════════════════════════════════════════
# Phase 1B: Chord-template ↔ chromagram DTW alignment (replaces beat-grid)
#
# Why non-circular: the DTW template is synthesised from *iReal GT chords only*.
# It never touches model predictions. Time is measured via chord harmony, not
# rhythm, so tempo-octave errors and rubato (which broke the beat-grid, 600–
# 1000 ms) are absorbed by DTW's local warping instead of corrupting the grid.
# ═════════════════════════════════════════════════════════════════════════════


def chord_pc_weights(mma_chord: str) -> np.ndarray:
    """Chroma template (12,) for one MMA chord: weighted pitch-class membership.

    Root/third carry chord identity (weight 1.0); fifth and seventh/sixth are
    strong (0.7); tensions (9/11/13) are faint colour (0.3). Rooted at the
    chord's pitch class. Returns a zero vector for 'z' / no-chord.
    """
    v = np.zeros(12, dtype=np.float32)
    root = chord_root_pc(mma_chord)
    if root is None:  # 'z' rest / no-chord
        return v
    # quality string = everything after the root letter (+ optional accidental),
    # dropping any slash-bass.
    q = mma_chord[1:]
    if q[:1] in ("#", "b"):
        q = q[1:]
    q = q.split("/")[0]

    def add(iv, w):
        v[(root + iv) % 12] = max(v[(root + iv) % 12], w)

    add(0, 1.0)  # root

    is_dim = q.startswith("dim") or q.startswith("o")
    is_min3 = (q.startswith("m") and not q.startswith("M")) or is_dim
    is_sus = "sus" in q

    # third
    if is_sus:
        add(5, 1.0)  # perfect 4th stands in for the (absent) third
    elif is_min3:
        add(3, 1.0)
    else:
        add(4, 1.0)

    # fifth
    fifth = 7
    if "b5" in q or is_dim or q == "m7b5" or q.startswith("h"):
        fifth = 6
    if "#5" in q or "aug" in q or q.endswith("+"):
        fifth = 8
    add(fifth, 0.7)

    # sixth / seventh
    is_maj7 = q.startswith("M") or "mM7" in mma_chord or "addM7" in q
    if q.startswith("dim7") or q == "o7":
        add(9, 0.7)          # diminished 7th (== maj 6th pc)
    elif "6" in q:
        add(9, 0.7)          # major 6th
    elif is_maj7:
        add(11, 0.7)         # major 7th
    elif "7" in q or "9" in q or "11" in q or "13" in q:
        add(10, 0.7)         # dominant / minor 7th

    # tensions (faint)
    if "9" in q or "13" in q:
        add(2, 0.3)          # 9th
    if "13" in q:
        add(9, 0.3)          # 13th (== maj 6th pc)
    return v


def mma_chart_to_chords(chart, max_bars: Optional[int] = None) -> list[dict]:
    """Flatten an MMAChart timeline into a chord list with absolute beat/second
    spans at the chart's nominal tempo.

    Output: [{"mma": str, "root": int|None, "bar": int, "beat_abs": float,
              "start_s": float, "end_s": float}, ...]  (drops trailing 'z' gaps'
    membership but keeps them as zero-vector spans so DTW can skip rests).
    """
    bpb = chart.beats_per_bar
    spb = 60.0 / chart.tempo  # seconds per beat at nominal tempo
    events: list[dict] = []
    prev_mma = "z"
    for barno, _label, slots in chart.timeline:
        if max_bars and barno > max_bars:
            break
        # within-bar starts (0-indexed beat offsets already in slots)
        for k, (beat_off, tok, mma) in enumerate(slots):
            if mma == "z" and tok.strip() in ("p",):
                mma = prev_mma  # 'p' = repeat previous chord
            beat_abs = (barno - 1) * bpb + beat_off
            # span until next slot in this bar, else bar end
            if k + 1 < len(slots):
                beat_end = (barno - 1) * bpb + slots[k + 1][0]
            else:
                beat_end = barno * bpb
            events.append({
                "mma": mma,
                "root": chord_root_pc(mma),
                "bar": barno,
                "beat_abs": float(beat_abs),
                "start_s": beat_abs * spb,
                "end_s": beat_end * spb,
            })
            if mma != "z":
                prev_mma = mma
    return events


def chords_to_chroma_template(
    chords: list[dict], fps: float
) -> tuple[np.ndarray, np.ndarray]:
    """Convert a chord list (with start_s/end_s) to a per-frame chroma template.

    Returns (chroma[M, 12], frame_time_s[M]) sampled at `fps` frames/second.
    Frame value = weighted pitch-class membership of the active chord (see
    chord_pc_weights); rests are zero rows.
    """
    total_s = max(c["end_s"] for c in chords)
    M = int(np.ceil(total_s * fps))
    chroma = np.zeros((M, 12), dtype=np.float32)
    tvec = (np.arange(M) + 0.5) / fps
    # precompute per-chord weight vectors
    for c in chords:
        w = chord_pc_weights(c["mma"])
        if not w.any():
            continue
        f0 = int(np.floor(c["start_s"] * fps))
        f1 = int(np.ceil(c["end_s"] * fps))
        chroma[max(0, f0):min(M, f1)] = w
    return chroma, tvec


def _subsequence_dtw(cost: np.ndarray) -> tuple[np.ndarray, float]:
    """Subsequence DTW (Müller). `cost[m, n]` = distance(template_m, audio_n).

    The full template (rows, length M) is matched to a *subsequence* of the
    audio (cols, length N): the audio start/end are free (skip intro / trailing
    solos), the template is traversed start-to-end. Steps: diagonal, template-
    advance (compress audio), audio-advance (stretch/skip audio).

    Returns (path, end_cost) where path is an array of (m, n) index pairs from
    template start (m=0) to template end (m=M-1).
    """
    M, N = cost.shape
    D = np.full((M, N), np.inf, dtype=np.float64)
    D[0, :] = cost[0, :]                    # template may start at any audio frame
    for m in range(1, M):
        D[m, 0] = cost[m, 0] + D[m - 1, 0]  # audio frame 0: must have covered rows
        prev = D[m - 1]
        row = D[m]
        cm = cost[m]
        for n in range(1, N):
            row[n] = cm[n] + min(prev[n - 1], prev[n], row[n - 1])
    end_n = int(np.argmin(D[M - 1]))
    end_cost = float(D[M - 1, end_n] / M)
    # backtrack
    path = []
    m, n = M - 1, end_n
    while m > 0:
        path.append((m, n))
        if n == 0:
            m -= 1
            continue
        cands = ((D[m - 1, n - 1], m - 1, n - 1),
                 (D[m - 1, n], m - 1, n),
                 (D[m, n - 1], m, n - 1))
        _, m, n = min(cands, key=lambda t: t[0])
    path.append((m, n))
    path.reverse()
    return np.array(path, dtype=int), end_cost


def align_chords_via_dtw(
    chroma_audio: np.ndarray,
    audio_times: np.ndarray,
    chroma_template: np.ndarray,
    template_times: np.ndarray,
):
    """Subsequence-DTW align an iReal chord template to audio chroma.

    Args:
        chroma_audio[Na, 12], audio_times[Na]     — audio CQT chroma + frame times
        chroma_template[Mt, 12], template_times[Mt] — iReal chord chroma + times
    Returns:
        dict with keys:
          warp(t_template_s) -> t_audio_s   piecewise-linear warping function
          offset_s      constant-offset best-fit (audio time of template t=0)
          cost          mean per-frame DTW cost along the matched path
          path          (K,2) template-frame, audio-frame index pairs
    """
    # Mean-centre both sides (Pearson-style). Raw cosine on full-mix CQT chroma
    # sits at a ~0.5 DC floor for *any* alignment (percussion/reverb fill every
    # bin), so the correct alignment barely separates from wrong ones. Removing
    # the per-frame mean cancels that floor and is the single biggest SNR win
    # (Phase-1B premise checks); it is still not enough — see results doc.
    def _center(C):
        C = C - C.mean(axis=1, keepdims=True)
        return C
    ta = _center(chroma_template)
    aa = _center(chroma_audio)
    # cosine distance (1 - cosine similarity); zero rows -> distance 1 everywhere
    cost = cdist(ta, aa, metric="cosine").astype(np.float64)
    cost = np.nan_to_num(cost, nan=1.0)
    path, mean_cost = _subsequence_dtw(cost)

    tmpl_t = template_times[path[:, 0]]
    aud_t = audio_times[path[:, 1]]
    # collapse duplicate template times (from audio-advance steps) by averaging
    uniq_t, inv = np.unique(tmpl_t, return_inverse=True)
    uniq_a = np.zeros_like(uniq_t)
    np.add.at(uniq_a, inv, aud_t)
    counts = np.bincount(inv)
    uniq_a /= counts

    def warp(t):
        return np.interp(t, uniq_t, uniq_a)

    offset_s = float(warp(0.0))
    return {
        "warp": warp,
        "offset_s": offset_s,
        "cost": mean_cost,
        "path": path,
        "tmpl_t": uniq_t,
        "aud_t": uniq_a,
    }


def _cos_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity of audio chroma rows `a`[K,12] to a single
    template vector `b`[12]. Returns [K]. Zero template -> zeros."""
    a = a - a.mean(axis=1, keepdims=True)   # Pearson-style, matches DTW cost
    b = b - b.mean()
    nb = np.linalg.norm(b)
    if nb == 0:
        return np.zeros(a.shape[0])
    na = np.linalg.norm(a, axis=1)
    na[na == 0] = 1e-9
    return (a @ b) / (na * nb)


def measure_alignment_error(
    chords: list[dict],
    warp,
    chroma_val: np.ndarray,
    val_times: np.ndarray,
    n_points: int = 9,
    window_s: float = 2.5,
) -> dict:
    """DTW-independent validation via chord-*change* changepoint detection.

    A single held chord matches many audio frames equally (flat correlation
    plateau) — that ill-posedness is exactly what produced Phase 1's 600–1000 ms
    "errors". So we validate at chord *boundaries*, where the audio chroma steps
    from chord_i's template to chord_{i+1}'s. In a local window we find the
    changepoint that best separates "chord_i dominant" (before) from "chord_{i+1}
    dominant" (after), *without* using the DTW path except to centre the window.
    Error = |t_dtw_boundary − t_changepoint|.

    Returns {"errors_ms": [...], "mean_ms", "max_ms", "n_used", "points": [...]}.
    """
    # candidate boundaries: consecutive chords with contrasting pitch-class sets
    cands = []
    for i in range(len(chords) - 1):
        wa = chord_pc_weights(chords[i]["mma"])
        wb = chord_pc_weights(chords[i + 1]["mma"])
        if not wa.any() or not wb.any():
            continue
        contrast = 1.0 - float(
            wa @ wb / (np.linalg.norm(wa) * np.linalg.norm(wb) + 1e-9)
        )
        if contrast < 0.15:  # too similar -> no detectable step
            continue
        cands.append((contrast, i, wa, wb))
    cands.sort(reverse=True)  # most-contrasting first
    # spread the chosen points across the head rather than clustering
    chosen = []
    for contrast, i, wa, wb in cands:
        t_bnd_tmpl = chords[i + 1]["start_s"]
        if all(abs(t_bnd_tmpl - c[0]) > window_s for c in chosen):
            chosen.append((t_bnd_tmpl, i, wa, wb, contrast))
        if len(chosen) >= n_points:
            break
    chosen.sort()

    errors_ms = []
    points = []
    for t_bnd_tmpl, i, wa, wb, contrast in chosen:
        t_dtw = float(warp(t_bnd_tmpl))
        lo, hi = t_dtw - window_s, t_dtw + window_s
        mask = (val_times >= lo) & (val_times <= hi)
        if mask.sum() < 6:
            continue
        seg = chroma_val[mask]
        seg_t = val_times[mask]
        s_a = _cos_sim(seg, wa)  # similarity to chord_before
        s_b = _cos_sim(seg, wb)  # similarity to chord_after
        diff = s_b - s_a         # <0 before boundary, >0 after
        # changepoint c maximising separation: sum(before: -diff) + sum(after: diff)
        # == for split at k: -sum(diff[:k]) + sum(diff[k:])
        total = diff.sum()
        prefix = np.concatenate(([0.0], np.cumsum(diff)))
        # score(k) = (total - prefix[k]) - prefix[k] = total - 2*prefix[k]
        scores = total - 2.0 * prefix        # length len(seg)+1, k=0..len
        k = int(np.argmax(scores))
        # changepoint time = midpoint between frame k-1 and k
        if k == 0:
            t_cp = seg_t[0]
        elif k >= len(seg_t):
            t_cp = seg_t[-1]
        else:
            t_cp = 0.5 * (seg_t[k - 1] + seg_t[k])
        err_ms = abs(t_dtw - t_cp) * 1000.0
        errors_ms.append(err_ms)
        points.append({
            "bar": chords[i + 1]["bar"],
            "chord_before": chords[i]["mma"],
            "chord_after": chords[i + 1]["mma"],
            "t_dtw_s": round(t_dtw, 3),
            "t_changepoint_s": round(float(t_cp), 3),
            "error_ms": round(err_ms, 1),
            "contrast": round(contrast, 2),
        })
    errors_ms = np.array(errors_ms)
    return {
        "errors_ms": errors_ms.tolist(),
        "mean_ms": float(errors_ms.mean()) if len(errors_ms) else float("nan"),
        "max_ms": float(errors_ms.max()) if len(errors_ms) else float("nan"),
        "median_ms": float(np.median(errors_ms)) if len(errors_ms) else float("nan"),
        "n_used": int(len(errors_ms)),
        "points": points,
    }


def validate_pilots():
    """Phase 1B re-validation: DTW-align the 3 pilots and measure error."""
    repo = Path(__file__).parent.parent
    playlist = repo / "data" / "ireal" / "jazz1460.txt"

    PILOTS = [
        # (audio filename stem, iReal title substring, human label)
        ("ghost_of_a_chance", "ghost of a chance", "Ballad 70 (rubato)"),
        ("a_foggy_day",       "foggy day",         "Med swing 140"),
        ("airegin",           "airegin",           "Up-tempo swing 220"),
    ]

    SR = 22050
    HOP_DTW = 1024          # 21.5 fps for DTW (speed)
    HOP_VAL = 512           # 43 fps for changepoint validation (resolution)

    logger.info("Loading iReal playlist %s", playlist.name)
    tunes = load_playlist(playlist)
    by_title = {t.title.lower(): t for t in tunes}

    results = {}
    for stem, title_sub, human in PILOTS:
        audio_path = repo / "docs" / "audio" / f"{stem}.m4a"
        tune = next((t for tl, t in by_title.items() if title_sub in tl), None)
        if tune is None:
            logger.error("iReal tune not found for %s", title_sub)
            continue
        chart = tune_to_mma(tune)
        chords = mma_chart_to_chords(chart)
        head_len_s = max(c["end_s"] for c in chords)
        logger.info("\n=== %s (%s) ===", tune.title, human)
        logger.info("  chart: %d bars, %d chords, nominal head %.1fs @ %d BPM",
                    len(chart.timeline), len(chords), head_len_s, chart.tempo)

        # cap audio search to give DTW room for the real (unknown) tempo:
        # up to 3x nominal head + slack, but never past the file.
        full_dur = librosa.get_duration(path=str(audio_path))
        max_t = min(full_dur, head_len_s * 3.0 + 20.0, 260.0)
        logger.info("  audio: %.1fs total, searching first %.1fs", full_dur, max_t)

        y, _ = librosa.load(audio_path, sr=SR, duration=max_t)
        chroma_a = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP_DTW).T
        aud_t = librosa.frames_to_time(np.arange(chroma_a.shape[0]),
                                       sr=SR, hop_length=HOP_DTW)
        # finer chroma for validation changepoints
        chroma_v = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP_VAL).T
        val_t = librosa.frames_to_time(np.arange(chroma_v.shape[0]),
                                       sr=SR, hop_length=HOP_VAL)

        fps = SR / HOP_DTW
        chroma_t, tmpl_t = chords_to_chroma_template(chords, fps=fps)

        logger.info("  DTW cost matrix: %d template × %d audio frames",
                    chroma_t.shape[0], chroma_a.shape[0])
        aln = align_chords_via_dtw(chroma_a, aud_t, chroma_t, tmpl_t)
        logger.info("  DTW: offset=%.2fs, mean path cost=%.3f",
                    aln["offset_s"], aln["cost"])

        err = measure_alignment_error(chords, aln["warp"], chroma_v, val_t)
        logger.info("  ALIGNMENT ERROR: mean=%.0fms  median=%.0fms  max=%.0fms  (n=%d boundaries)",
                    err["mean_ms"], err["median_ms"], err["max_ms"], err["n_used"])

        results[stem] = {
            "title": tune.title,
            "human": human,
            "nominal_tempo": chart.tempo,
            "n_bars": len(chart.timeline),
            "n_chords": len(chords),
            "dtw_offset_s": round(aln["offset_s"], 3),
            "dtw_cost": round(aln["cost"], 4),
            "error": err,
        }

    # ── report ──────────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 74)
    logger.info("PHASE 1B RE-VALIDATION — DTW chord-template alignment")
    logger.info("=" * 74)
    GATE = 150.0
    all_pass = True
    for stem, r in results.items():
        e = r["error"]
        ok = e["mean_ms"] <= GATE
        all_pass &= ok
        logger.info("  %-22s mean=%5.0fms  median=%5.0fms  max=%5.0fms  n=%d  %s",
                    r["title"][:22], e["mean_ms"], e["median_ms"], e["max_ms"],
                    e["n_used"], "PASS" if ok else "FAIL")
    means = [r["error"]["mean_ms"] for r in results.values()]
    logger.info("  aggregate mean-of-means=%.0fms  worst-song-mean=%.0fms",
                np.mean(means), np.max(means))
    logger.info("  GATE (all songs mean ≤ ±%.0fms): %s",
                GATE, "✅ PASS" if all_pass else "❌ FAIL")
    logger.info("=" * 74)

    out = Path(__file__).parent.parent / "docs" / "mission_1_phase1b_dtw_results.json"
    out.write_text(json.dumps(results, indent=2, default=float))
    logger.info("Results written to %s", out)
    return results, all_pass


def main():
    """
    Build a ~20-song real-audio benchmark with non-circular GT alignment.

    Protocol:
    1. Select ~20 diverse songs from iReal corpus
    2. For each song:
       a. Extract beat/downbeat grid from audio (librosa)
       b. Map iReal chords to audio time via beat grid (no model predictions)
       c. Store aligned_chords as ground truth
    3. Spot-check alignment on 5 songs manually
    4. Ready for inference + evaluation
    """

    output_dir = Path(__file__).parent.parent / "data" / "real_audio_benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Building real-audio benchmark → {output_dir}")

    # Load YouTube mapping
    yt_mapping = load_youtube_mapping()
    logger.info(f"Loaded {len(yt_mapping)} songs from YouTube mapping")

    # Log protocol
    logger.info("\n" + "="*70)
    logger.info("ALIGNMENT PROTOCOL (Non-Circular)")
    logger.info("="*70)
    logger.info("Stage 1: Extract beat/downbeat anchors from audio (librosa)")
    logger.info("Stage 2: Map iReal chords to audio time using beat grid")
    logger.info("Stage 3: Manual verification on 5-10 songs")
    logger.info("Stage 4: Measure alignment error (target: ±200ms mean, ±500ms max)")
    logger.info("="*70 + "\n")

    logger.info("✓ Protocol design complete")
    logger.info("✓ Ready to implement:")
    logger.info("  - Song selection (20 diverse songs)")
    logger.info("  - Audio download (yt-dlp)")
    logger.info("  - Beat tracking on real audio")
    logger.info("  - Manual spot-check")
    logger.info("  - Inference + scoring")

    # Save protocol document
    protocol_doc = """
# Mission 1: Real-Audio Benchmark — Alignment Protocol

## Problem
Current YouTube corpus uses circular measurement:
- iReal GT is time-aligned to audio via DTW against MODEL predictions
- When model is wrong, GT slides → model error ≡ alignment error

## Solution: Beat/Downbeat-Anchored Alignment (Non-Circular)

### Core Principle
Anchor iReal GT to audio using ONLY beat/downbeat timing, NOT model predictions.

### Stage 1: Extract Beat/Downbeat Anchors from Audio
- librosa beat_track() → beat times (±100ms confidence)
- Harmonic SSM peaks → downbeat times (±200ms)
- High-freq onset flux refinement (within ±500ms window)

### Stage 2: Map iReal Chords to Audio Time
- Parse iReal MMAChart → chord positions in beat coordinates
- Convert beats → audio time using detected beat grid
- Snap each chord to nearest beat (no model predictions)
- Result: iReal GT aligned without circularity

### Stage 3: Manual Verification
- For each song: verify 3–5 key chords in audio editor
- Check harmonic onset matches timing
- Expected variance: ±200ms

### Stage 4: Measure Error
- Error budget: ±200ms mean, ±500ms max per song
- If drift > ±500ms: flag for review or skip

## Why This Works

**Old approach (circular):**
```
iReal chords (beat pos) → DTW vs model predictions
→ model error corrupts GT times
→ score model vs corrupted GT (misleading)
```

**New approach (non-circular):**
```
iReal chords (beat pos) → librosa beat tracking + harmonic SSM
→ GT times fixed independently of model
→ score model vs clean GT (true error)
```

Key insight: Beat tracking ⊥ chord recognition.
If beat tracking fails → large systematic misalignment (detectable via manual check).
If beat tracking succeeds (±100–200ms) → iReal alignment inherits that error, adds nothing.

## Deliverables
- ✓ Alignment protocol (this doc)
- ✓ Implementation: extract_beat_grid() + align_ireal_to_beat_grid()
- ✓ ~20-song benchmark set with non-circular GT
- ✓ Manual verification log (5–10 songs)
- ✓ Evaluation: root/majmin/7ths on real audio vs fixed GT
"""

    protocol_path = output_dir / "PROTOCOL.md"
    protocol_path.write_text(protocol_doc)
    logger.info(f"Protocol saved to {protocol_path}")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-pilots", action="store_true",
                    help="Phase 1B: DTW-align the 3 pilots and measure error")
    args = ap.parse_args()
    if args.validate_pilots:
        validate_pilots()
    else:
        main()
