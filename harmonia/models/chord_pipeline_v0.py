"""harmonia/models/chord_pipeline_v0.py — v0 inference pipeline as a reusable module.

Extracted from scripts/pipeline_v0.py.  Uses bass-register root detection +
a trained LogisticRegression family classifier instead of the frozen chord_hmm
template-matching approach.

Chain:
    1. Beat tracking         librosa.beat.beat_track
    2. Basic Pitch features  PitchExtractor → onset_probs / note_probs
    3. Pool to beats         mean frame activity per beat interval
    4. Chord segmentation    running-segment scan with chroma + bass novelty
    5. Root per segment      bass-register chroma argmax (MIDI 21–72)
    6. Family per segment    LR classifier trained on audio_chord_features.npz
    7. Key inference         harmonia.theory.key_profiles.infer_key on global chroma

The LR classifier (StandardScaler + LogisticRegression) is fit once at module
import time from data/cache/audio_chord_features.npz.
"""

from __future__ import annotations

import logging
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

from harmonia.models.stage1_pitch import PitchExtractor
from harmonia.pipeline import ChordChart
from harmonia.theory.key_profiles import infer_key

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FAMILIES = ["major", "minor", "diminished", "augmented", "suspended"]
NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

# Short-form quality strings that match _QUALITY_TO_IREAL in render_youtube_chart.py
FAM_SHORT = {
    "major":      "maj",
    "minor":      "min",
    "diminished": "dim",
    "augmented":  "aug",
    "suspended":  "sus4",
}

REPO = Path(__file__).resolve().parent.parent.parent
CLEAN_FEAT = REPO / "data" / "cache" / "audio_chord_features.npz"

# ---------------------------------------------------------------------------
# Classifier — fit once at import time
# ---------------------------------------------------------------------------

_sc = None
_clf = None


def _load_classifier():
    """Fit (or return cached) StandardScaler + LogisticRegression."""
    global _sc, _clf
    if _sc is not None:
        return _sc, _clf

    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    if not CLEAN_FEAT.exists():
        raise FileNotFoundError(
            f"Chord feature cache not found: {CLEAN_FEAT}\n"
            "Run scripts/build_audio_chord_features.py first."
        )

    d = np.load(CLEAN_FEAT, allow_pickle=True)
    # Normalize each 12-dim block independently before concatenating.
    # The raw training features are summed over whole chord segments (large
    # magnitude), while inference features are per-beat (small magnitude).
    # Whole-vector L2-norm is insufficient because note_probs are nearly
    # uniform across pitch classes (saturated Basic Pitch sustain channel),
    # so after a single L2 the uniform note block dominates and dilutes the
    # discriminative onset/bass signal. Per-block norm keeps each view at
    # unit length regardless of its absolute scale, preserving maj/min
    # separation in the onset third (m3 0.51 vs 0.17, M3 0.08 vs 0.24).
    def _norm_block(v):
        n = np.linalg.norm(v, axis=1, keepdims=True)
        return v / np.where(n > 0, n, 1.0)

    Xc = np.hstack([_norm_block(d["onset"]), _norm_block(d["note"]),
                    _norm_block(d["bass"]),  _norm_block(d["treble"])])
    _sc = StandardScaler().fit(Xc)
    _clf = LogisticRegression(max_iter=2000).fit(_sc.transform(Xc), d["family"].astype(int))
    logger.debug("chord_pipeline_v0: classifier fit (%d training examples)", len(Xc))
    return _sc, _clf


# Eagerly fit at import so the first infer_chords_v0 call isn't slower.
try:
    _load_classifier()
except FileNotFoundError as _e:
    logger.warning("chord_pipeline_v0: %s — classifier will be fit on first call", _e)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _pool_to_beats(
    frame_times: np.ndarray,
    probs: np.ndarray,
    beat_times: np.ndarray,
) -> np.ndarray:
    """Return (n_beats-1, 88) array: mean frame activity within each beat interval."""
    out = np.zeros((len(beat_times) - 1, probs.shape[1]), dtype=np.float32)
    for b in range(len(beat_times) - 1):
        mask = (frame_times >= beat_times[b]) & (frame_times < beat_times[b + 1])
        if mask.any():
            out[b] = probs[mask].mean(0)
    return out


def _chroma_of(v88: np.ndarray) -> np.ndarray:
    """Fold 88-dim pitch vector into 12-dim chroma (A4 = MIDI 69 = PC 9)."""
    c = np.zeros(12)
    for k in range(88):
        c[(k + 21) % 12] += v88[k]
    return c


def _reg(v88: np.ndarray, lo: int, hi: int) -> np.ndarray:
    """Register-filtered chroma: sum only MIDI pitches in [21+lo, 21+hi)."""
    c = np.zeros(12)
    for k in range(88):
        if lo <= k < hi:
            c[(21 + k) % 12] += v88[k]
    return c


def _unit_chroma(v88: np.ndarray, lo: int = 0, hi: int = 200) -> np.ndarray:
    c = _reg(v88, lo, hi)
    n = np.linalg.norm(c)
    return c / n if n > 1e-9 else c


def _classify(seg_on: np.ndarray, seg_nt: np.ndarray, sc, clf) -> tuple[int, str, float]:
    """Return (root_pc, family_string, confidence) for a segment."""
    bass = _reg(seg_on, 0, 52)
    root = int(bass.argmax()) if bass.sum() > 1e-6 else int(_chroma_of(seg_on).argmax())

    def rr(c):
        return np.roll(c, -root)

    def _nb(v):
        n = np.linalg.norm(v); return v / n if n > 0 else v

    feat = np.hstack([
        _nb(rr(_chroma_of(seg_on))),
        _nb(rr(_chroma_of(seg_nt))),
        _nb(rr(_reg(seg_on, 0, 52))),
        _nb(rr(_reg(seg_on, 60, 200))),
    ])
    proba = clf.predict_proba(sc.transform(feat[None]))[0]
    fam_idx = int(proba.argmax())
    return root, FAMILIES[fam_idx], float(proba[fam_idx])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def infer_chords_v0(
    audio_path: Path,
    *,
    cell: int = 2,
    nov: float = 0.35,
    cache_dir: Path | None = None,
) -> ChordChart:
    """Run the v0 inference chain on an audio file and return a ChordChart.

    Args:
        audio_path:  Path to the audio file (any format soundfile can read,
                     or an opus/m4a if ffmpeg is available via librosa).
        cell:        Minimum chord duration in beats before a change is allowed.
        nov:         Chroma / bass novelty threshold to declare a new chord.
        cache_dir:   Cache directory for Basic Pitch activations (None = no cache).

    Returns:
        ChordChart compatible with harmonia.pipeline.ChordChart and accepted by
        scripts/render_youtube_chart.chart_to_interactive_inputs.

    Note:
        duration_beats is derived purely from beat-time arithmetic; time_signature
        is hardcoded "4/4".  Modulations are not detected (empty list).
    """
    audio_path = Path(audio_path)
    sc, clf = _load_classifier()

    # ── 1. Load audio ─────────────────────────────────────────────────────────
    logger.info("chord_pipeline_v0: loading %s", audio_path.name)
    y, sr = sf.read(audio_path)
    y = y.mean(1) if y.ndim > 1 else y
    y = y.astype("float32")
    duration_s = len(y) / sr

    # ── 2. Beat tracking ──────────────────────────────────────────────────────
    tempo_arr, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    tempo_bpm = float(np.atleast_1d(tempo_arr)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)

    if len(beat_times) < 4:
        # Degenerate file: return a single "C:maj" placeholder
        logger.warning("chord_pipeline_v0: too few beats detected (%d)", len(beat_times))
        return ChordChart(
            source_path=str(audio_path),
            duration_s=duration_s,
            tempo_bpm=tempo_bpm,
            time_signature="4/4",
            global_key="C major",
            global_key_confidence=0.0,
            style="v0",
            modulations=[],
            chords=[{"label": "Cmaj", "start_s": 0.0, "end_s": duration_s,
                     "duration_beats": 1, "confidence": 0.0}],
            segments=[{"start_s": 0.0, "end_s": duration_s, "key": "C major", "n_beats": 1}],
        )

    # ── 3. Basic Pitch features ───────────────────────────────────────────────
    logger.info("chord_pipeline_v0: extracting pitch features…")
    ex = PitchExtractor(cache_dir=cache_dir)
    acts = ex.extract(audio_path)

    onset_b = _pool_to_beats(acts.frame_times, acts.onset_probs, beat_times)
    note_b  = _pool_to_beats(acts.frame_times, acts.note_probs,  beat_times)
    nbz = len(onset_b)

    # ── 4. Global key inference ───────────────────────────────────────────────
    global_onset_chroma = _chroma_of(onset_b.sum(0))
    global_key_result = infer_key(global_onset_chroma)

    # ── 5. Running-segment chord change detector ──────────────────────────────
    segs: list[tuple[int, int, np.ndarray, np.ndarray]] = []
    run_on = run_nt = None
    run_start = 0

    for b in range(nbz):
        if onset_b[b].sum() < 1e-6:
            continue
        if run_on is None:
            run_on, run_nt, run_start = onset_b[b].copy(), note_b[b].copy(), b
            continue

        ref_ch   = _unit_chroma(run_on)
        ref_bass = _unit_chroma(run_on, 0, 52)
        beat_ch   = _unit_chroma(onset_b[b])
        beat_bass = _unit_chroma(onset_b[b], 0, 52)

        novelty      = 1.0 - float(ref_ch   @ beat_ch)
        bass_novelty = 1.0 - float(ref_bass @ beat_bass)

        changed = (b - run_start) >= cell and (novelty > nov or bass_novelty > nov)
        if changed:
            segs.append((run_start, b, run_on, run_nt))
            run_on, run_nt, run_start = onset_b[b].copy(), note_b[b].copy(), b
        else:
            run_on += onset_b[b]
            run_nt += note_b[b]

    if run_on is not None:
        segs.append((run_start, nbz, run_on, run_nt))

    # ── 6. Classify each segment → label ──────────────────────────────────────
    beat_dur_s = 60.0 / max(tempo_bpm, 1.0)
    chords_out: list[dict] = []
    segments_out: list[dict] = []

    for s, e, son, snt in segs:
        root, fam, confidence = _classify(son, snt, sc, clf)

        t_start = float(beat_times[s])
        t_end   = float(beat_times[min(e, len(beat_times) - 1)])
        n_beats = max(1, round((t_end - t_start) / beat_dur_s))

        label = f"{NOTE[root]}{FAM_SHORT[fam]}"

        chords_out.append({
            "label":          label,
            "start_s":        round(t_start, 3),
            "end_s":          round(t_end, 3),
            "duration_beats": n_beats,
            "confidence":     round(confidence, 4),
        })
        segments_out.append({
            "start_s": round(t_start, 3),
            "end_s":   round(t_end, 3),
            "key":     global_key_result.key_name,
            "n_beats": n_beats,
        })

    logger.info(
        "chord_pipeline_v0: %d chords, key=%s, tempo=%.1f BPM",
        len(chords_out), global_key_result.key_name, tempo_bpm,
    )

    return ChordChart(
        source_path=str(audio_path),
        duration_s=duration_s,
        tempo_bpm=round(tempo_bpm, 1),
        time_signature="4/4",
        global_key=global_key_result.key_name,
        global_key_confidence=round(global_key_result.confidence, 4),
        style="v0",
        modulations=[],
        chords=chords_out,
        segments=segments_out,
    )
