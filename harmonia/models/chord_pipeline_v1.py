"""harmonia/models/chord_pipeline_v1.py — production Gen-2 inference pipeline.

Chain (all Gen-2 improvements wired in):
  1. Load audio
  2. Beat tracking — librosa tempo-grid de-jitter (metronomic assumption: accurate
     tempo, jittery phase → impose uniform grid at detected tempo + circular-mean
     phase). Costs ~20 majmin pts less than raw librosa beats.
  3. Basic Pitch (onset_probs + note_probs, 88-dim piano roll)
  4. Pool to beats — sum pooling per beat interval
  5. Coarse segmentation — fixed 2-beat grid + chroma+bass cosine novelty (θ=0.08),
     coalesce adjacent same-label segments (recall-tuned, undoes over-segmentation
     cheaply). Forced cut at 0.
  6. Root per segment — beat-sequence model (BeatSeqModel, 88.3% CV per-beat root)
     via sum of per-beat probabilities over each segment, then argmax.
     Fallback: trained segment-level root model (RootModel, 93% CV) if beat-seq
     model is unavailable.
  7. Family per segment — trained LR on root-shifted, L2-normalised audio features
     (82.8% majmin end-to-end on 30 disjoint hold-out songs). If the ctx model is
     saved (ctx_family_model.npz), use entropy-gated blend (87.5% oracle family).
  8. Seventh quality — trained LR on same features (88% seventh on oracle segs).
  9. Global key — Krumhansl-Schmuckler on raw onset chroma.
 10. Coalesce adjacent same-label chords.

Critical correctness invariants (each one a past calibration bug):
  - norm_blocks: L2-normalise each 12-dim chroma block before the family
    classifier — raw summed chroma scales with segment length and lands off the
    training distribution (Issue #10, +24pp majmin).
  - Root shift: family features are root-shifted (roll by -root) so the model
    learns quality-independent-of-key.
  - chroma88: MIDI pitch 21+k, register [lo, hi) in MIDI numbers (not key index).
  - pool_beats: SUM over frames per beat interval (not mean) — beat_seq_model was
    trained with sum pooling on onset_probs/note_probs.
  - Beat grid: tempo-grid de-jitter required; raw librosa beat times jitter ~±10%
    of one beat and destroy pooling quality on metronomic audio.

Public API:
    from harmonia.models.chord_pipeline_v1 import infer_chords_v1
    chart = infer_chords_v1(Path("song.wav"))   # → ChordChart
"""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import TYPE_CHECKING

import librosa
import numpy as np
import soundfile as sf

from harmonia.models.stage1_pitch import PitchExtractor
from harmonia.pipeline import ChordChart
from harmonia.theory.key_profiles import infer_key

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent.parent
MODELS = REPO / "harmonia" / "models"
CLEAN_FEAT = REPO / "data" / "cache" / "audio_chord_features.npz"

FAMILIES = ["major", "minor", "diminished", "augmented", "suspended"]
NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def _note_name_to_pc(name: str) -> int:
    """Pitch class of a key/chord root name like 'C', 'Bb major', 'F# minor'."""
    base = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
    tok = name.strip().split()[0] if name.strip() else "C"
    pc = base.get(tok[0].upper(), 0)
    if len(tok) > 1 and tok[1] in "#b":
        pc += 1 if tok[1] == "#" else -1
    return pc % 12
FAM_HARTE = {
    "major": "maj", "minor": "min", "diminished": "dim",
    "augmented": "aug", "suspended": "sus4",
}
B7_HARTE = {
    "majT": "maj", "minT": "min", "dimT": "dim", "augT": "aug", "susT": "sus4",
    "maj7": "maj7", "min7": "min7", "dom7": "7", "m7b5": "hdim7", "dim7": "dim7",
    "minmaj7": "minmaj7", "7sus4": "sus4", "aug7": "aug", "augmaj7": "aug",
}


# ── diatonic quality prior (issue #20) ────────────────────────────────────────
# Pop/standards corpora are strongly diatonic (POP909 = 93.3% of GT chords are
# diatonic in the local key; jazz1460 = only 49.4%).  A confidence-gated,
# section-local diatonic prior therefore *helps* pop/standards ("Georgia On My
# Mind"-type maj/min/dom flips) while jazz users can disable it.
#
# Tables mirror scripts/check_diatonic_premise_pop909.py exactly (the premise
# that PASSed on POP909) — degree = (root_pc - tonic) % 12.
#   *_OK    : set of q5 names that count as diatonic at that degree (no override)
#   *_CANON : the single canonical q5 the prior snaps a non-diatonic quality TO
_DIA_MAJOR_OK = {
    0: {"major", "maj7"}, 2: {"minor"}, 4: {"minor"}, 5: {"major", "maj7"},
    7: {"dom7", "major"}, 9: {"minor"}, 11: {"dim"},
}
_DIA_MAJOR_CANON = {0: "major", 2: "minor", 4: "minor", 5: "major",
                    7: "dom7", 9: "minor", 11: "dim"}
_DIA_MINOR_OK = {
    0: {"minor"}, 2: {"dim"}, 3: {"major", "maj7"}, 5: {"minor"},
    7: {"minor", "dom7", "major"}, 8: {"major", "maj7"}, 10: {"dom7", "major"},
    11: {"dim"},
}
_DIA_MINOR_CANON = {0: "minor", 2: "dim", 3: "major", 5: "minor",
                    7: "dom7", 8: "major", 10: "dom7", 11: "dim"}

# sev_h (Harte quality) → coarse q5 name for the diatonic test.  None = a quality
# with no place in the diatonic tables (sus/aug) → never overridden.
_SEV_TO_Q5 = {
    "maj": "major", "maj7": "maj7", "min": "minor", "min7": "minor",
    "7": "dom7", "hdim7": "dim", "dim7": "dim", "dim": "dim", "minmaj7": "minor",
    "aug": None, "sus4": None, "sus2": None, "7sus4": None,
}
# whether a sev_h already carries a seventh — the prior preserves the acoustic
# model's triad-vs-seventh extension decision and only corrects maj/min/dom/dim.
_SEV_IS_SEVENTH = {
    "maj": False, "min": False, "dim": False, "aug": False, "sus4": False,
    "maj7": True, "min7": True, "7": True, "hdim7": True, "dim7": True,
    "minmaj7": True,
}


def _q5_to_sev(q5: str, is_seventh: bool) -> str | None:
    """Canonical q5 → Harte sev_h, at the requested extension level."""
    if q5 == "major":
        return "maj7" if is_seventh else "maj"
    if q5 == "minor":
        return "min7" if is_seventh else "min"
    if q5 == "dom7":
        return "7" if is_seventh else "maj"      # dominant triad == major triad
    if q5 == "dim":
        return "dim7" if is_seventh else "dim"
    return None


def apply_diatonic_prior(
    root: int, sev_h: str, conf: float,
    tonic: int, mode: str, key_conf: float,
    *,
    diatonic_boost: float = 4.0,
    threshold_chromatic: float = 0.80,
    key_conf_min: float = 0.30,
) -> str:
    """Return a (possibly diatonically-corrected) Harte quality for one segment.

    Implements the confidence-gated combination
        log_posterior(q) = log P_acoustic(q) + w · log(diatonic_boost)·1[q diatonic]
    reduced to the two-candidate (acoustic-arg-max vs canonical-diatonic) case,
    where `w = 1` iff the acoustic quality is uncertain (`conf < threshold_chromatic`)
    AND the local key is reliable (`key_conf >= key_conf_min`), else `w = 0`.

    Args:
        root:      predicted root pitch class (0–11).
        sev_h:     acoustic Harte quality (e.g. "maj7", "min7", "7", "dim").
        conf:      acoustic confidence for that quality (family/ctx max-prob).
        tonic/mode/key_conf: local key from infer_key() over the segment window.

    Returns:
        sev_h unchanged, or the canonical diatonic quality when the prior fires.

    Does NOT: touch roots; touch sus/aug qualities; fire on a chromatic root
    (degree outside the diatonic table) or when the acoustic quality is already
    diatonic — those are pass-through so real secondary dominants / borrowed
    chords the model sees clearly survive.
    """
    if key_conf < key_conf_min or conf >= threshold_chromatic:
        return sev_h
    q5 = _SEV_TO_Q5.get(sev_h)
    if q5 is None:
        return sev_h
    deg = (root - tonic) % 12
    ok = (_DIA_MAJOR_OK if mode == "major" else _DIA_MINOR_OK).get(deg)
    canon = (_DIA_MAJOR_CANON if mode == "major" else _DIA_MINOR_CANON).get(deg)
    if canon is None or ok is None:
        return sev_h                       # chromatic root — prior not applicable
    if q5 in ok:
        return sev_h                       # already diatonic — keep acoustic call
    # Non-diatonic call under a reliable key + uncertain acoustics.  Boost the
    # diatonic quality; flip only if it wins the (crude) 2-way log comparison.
    log_ac = math.log(max(conf, 1e-6))
    log_dia = math.log(max(1.0 - conf, 1e-6)) + math.log(max(diatonic_boost, 1e-6))
    if log_dia <= log_ac:
        return sev_h
    return _q5_to_sev(canon, _SEV_IS_SEVENTH.get(sev_h, False)) or sev_h


# ── low-level helpers ─────────────────────────────────────────────────────────

def _chroma88(v88: np.ndarray, lo: int = 0, hi: int = 200) -> np.ndarray:
    """MIDI-register-filtered L2-normalised chroma from an 88-dim piano-roll vector.

    lo/hi are MIDI numbers (21 = A0 = lowest key). Returns a unit 12-vector.
    This matches the feature definition in root_model_experiment.chroma88 exactly.
    """
    c = np.zeros(12)
    for k in range(88):
        m = 21 + k
        if lo <= m < hi:
            c[m % 12] += v88[k]
    n = np.linalg.norm(c)
    return c / n if n > 1e-9 else c


def _reg_raw(v88: np.ndarray, lo: int = 0, hi: int = 200) -> np.ndarray:
    """Raw (un-normalised) register-filtered chroma."""
    c = np.zeros(12)
    for k in range(88):
        m = 21 + k
        if lo <= m < hi:
            c[m % 12] += v88[k]
    return c


def _norm_blocks(x: np.ndarray) -> np.ndarray:
    """L2-normalise each consecutive 12-dim chroma block in a flat feature row.

    Critical: raw summed chroma scales with segment length; normalising each
    12-dim block makes the family classifier duration-invariant (Issue #10).
    """
    x = np.asarray(x, float)
    y = x.reshape(*x.shape[:-1], x.shape[-1] // 12, 12)
    n = np.linalg.norm(y, axis=-1, keepdims=True)
    return (y / (n + 1e-9)).reshape(x.shape)


def _pool_beats(frame_times: np.ndarray, probs: np.ndarray,
                beat_times: np.ndarray) -> np.ndarray:
    """Sum (not mean) frame activity within each beat interval → (n_beats, 88).

    Uses SUM pooling: beat_seq_model was trained with sum pooling.  The family
    classifier's training features also use sum pooling (build_audio_chord_features
    sums across the segment).
    """
    out = np.zeros((len(beat_times) - 1, probs.shape[1]), dtype=np.float32)
    for b in range(len(beat_times) - 1):
        mask = (frame_times >= beat_times[b]) & (frame_times < beat_times[b + 1])
        if mask.any():
            out[b] = probs[mask].sum(0)
    return out


def _cos_dist(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    if na < 1e-9 or nb < 1e-9:
        return 0.0
    return float(1.0 - (a @ b) / (na * nb))


def _feat24(v88: np.ndarray) -> np.ndarray:
    """24d feature for change detection: normalised full + bass chroma."""
    ch = _reg_raw(v88, 0, 200); ba = _reg_raw(v88, 0, 52)
    ch /= (np.linalg.norm(ch) + 1e-9); ba /= (np.linalg.norm(ba) + 1e-9)
    return np.concatenate([ch, ba])


def _deinvert_bass_proba(beat_proba_seg: np.ndarray) -> np.ndarray:
    """Correct the beat_seq root probability for likely 2nd-inversion (5th-in-bass) chords.

    The beat_seq model was trained on MMA where bass = root always.  In piano
    voicings the 5th is often the lowest note, causing the model to predict the
    5th instead of the root.  This is the #1 error pattern (53 % of bass argmax
    errors, 15 % of beat_seq errors are a P5 above GT).

    Heuristic: for each root candidate k, check if the note a P5 above (k+7)
    also has high probability (the inverted-5th scenario: predicted k is really
    the 5th, and (k-7)%12 is the true root).  Blend by returning a corrected
    distribution that shifts weight from k toward (k-7)%12 proportionally.
    """
    p = beat_proba_seg.copy()
    # For every pitch class: if p[k] is large AND p[(k+7)%12] is also large,
    # it's ambiguous whether k or (k-7)%12 is the root.  Apply a soft re-weight:
    # push half the p[k] mass toward the candidate 5th-below.
    p_shift = p.copy()
    for k in range(12):
        fifth_above = (k + 7) % 12
        # fifth_above being prominent means k might be the 5th (root is k-7)
        if p[fifth_above] > 0.4 * p[k]:
            # candidate: actual root might be (fifth_above - 7) % 12 = (k) ... wait
            # If k is being predicted and fifth_above is also prominent:
            # Scenario: k is the 5th of chord rooted at (k - 7) % 12
            true_root_cand = (k - 7) % 12  # = (k + 5) % 12
            weight = 0.4 * p[fifth_above] / (p[k] + 1e-9) * p[k]
            p_shift[true_root_cand] += weight
            p_shift[k] -= weight
    p_shift = np.clip(p_shift, 0, None)
    s = p_shift.sum()
    return p_shift / s if s > 1e-9 else p


# ── model wrappers ────────────────────────────────────────────────────────────

class _RootModel:
    """Trained 12-way root LR loaded from root_model.npz."""

    def __init__(self, path: Path) -> None:
        d = np.load(path)
        self.mean = d["mean"]; self.scale = d["scale"]
        self.coef = d["coef"]; self.intercept = d["intercept"]
        self.classes = d["classes"]
        self._has_templates = (len(self.mean) == 60)

    def _features(self, seg88: np.ndarray) -> np.ndarray:
        f = np.concatenate([
            _chroma88(seg88),
            _chroma88(seg88),           # note channel — use onset as proxy
            _chroma88(seg88, 0, 52),
            _chroma88(seg88, 60, 200),
        ])
        if self._has_templates:
            # template features expected: max cosine over 5 family templates per root
            FAM_TONES = {"major": [0, 4, 7], "minor": [0, 3, 7], "diminished": [0, 3, 6],
                         "augmented": [0, 4, 8], "suspended": [0, 5, 7]}
            oc = _chroma88(seg88)
            templates = []
            for r in range(12):
                best = max(
                    np.dot(oc, np.roll(t / np.linalg.norm(t + 1e-9), r))
                    for tones in FAM_TONES.values()
                    for t in [np.array([1.0 if i in tones else 0.0 for i in range(12)])]
                )
                templates.append(best)
            f = np.concatenate([f, templates])
        return f

    def predict(self, seg_on: np.ndarray, seg_nt: np.ndarray) -> int:
        f = np.concatenate([
            _chroma88(seg_on),
            _chroma88(seg_nt),
            _chroma88(seg_on, 0, 52),
            _chroma88(seg_on, 60, 200),
        ])
        if self._has_templates:
            FAM_TONES = {"major": [0, 4, 7], "minor": [0, 3, 7], "diminished": [0, 3, 6],
                         "augmented": [0, 4, 8], "suspended": [0, 5, 7]}
            oc = _chroma88(seg_on)
            templates = []
            for r in range(12):
                best = -1.0
                for tones in FAM_TONES.values():
                    t = np.zeros(12)
                    for off in tones:
                        t[(r + off) % 12] = 1.0
                    t /= (np.linalg.norm(t) + 1e-9)
                    best = max(best, float(oc @ t))
                templates.append(best)
            f = np.concatenate([f, templates])
        z = (f - self.mean) / self.scale
        logits = z @ self.coef.T + self.intercept
        return int(self.classes[np.argmax(logits)])


class _BeatSeqModel:
    """Per-beat root model from beat_seq_model.npz (88.3% CV, ±2 context)."""

    def __init__(self, path: Path) -> None:
        d = np.load(path)
        self.mean = d["mean"]; self.scale = d["scale"]
        self.coef = d["coef"]; self.intercept = d["intercept"]
        self.classes = d["classes"]
        self.window = int(d["window"][0])

    def predict_proba(self, onset_b: np.ndarray, note_b: np.ndarray) -> np.ndarray:
        """Return (n_beats, 12) root probabilities indexed by pitch class 0–11."""
        n = len(onset_b); w = self.window
        # 48d per beat: chroma88(onset) + chroma88(note) + bass + treble
        F = np.zeros((n, 48), dtype=np.float32)
        for b in range(n):
            F[b] = np.concatenate([
                _chroma88(onset_b[b]),
                _chroma88(note_b[b]),
                _chroma88(onset_b[b], 0, 52),
                _chroma88(onset_b[b], 60, 200),
            ])
        d_feat = F.shape[1]
        X = np.zeros((n, d_feat * (2 * w + 1)), dtype=np.float32)
        for b in range(n):
            row = []
            for delta in range(-w, w + 1):
                nb = b + delta
                row.append(F[nb] if 0 <= nb < n else np.zeros(d_feat))
            X[b] = np.concatenate(row)
        z = (X - self.mean) / self.scale
        logits = z @ self.coef.T + self.intercept       # (n, n_classes)
        logits -= logits.max(1, keepdims=True)
        exp = np.exp(logits)
        proba_cls = exp / exp.sum(1, keepdims=True)
        proba_pc = np.zeros((n, 12), dtype=np.float32)
        for i, c in enumerate(self.classes):
            proba_pc[:, int(c)] = proba_cls[:, i]
        return proba_pc


class _BeatSeqModelV3:
    """Dual-output beat model from beat_seq_model_v3.npz.

    Root head: canonical MLP (key-invariant; 78.2% per-beat root on CV).
    Quality head: LR on DFT magnitudes (62.2% majmin on CV).

    In the production pipeline we use v2 for root (88.3%) and only use
    this model's quality head as an additional segmentation boundary signal:
    a quality change (major→minor) at a grid boundary keeps the split even
    when the root argmax is the same on both sides.
    """

    def __init__(self, path: Path) -> None:
        d = np.load(path, allow_pickle=True)
        self.window = int(d["window"][0])
        # root MLP weights
        self._rW1 = d["root_W1"].astype(np.float32)
        self._rb1 = d["root_b1"].astype(np.float32)
        self._rW2 = d["root_W2"].astype(np.float32)
        self._rb2 = d["root_b2"].astype(np.float32)
        self._rmean = d["root_mean"].astype(np.float32)
        self._rscale = d["root_scale"].astype(np.float32)
        self._use_template = bool(d["root_use_template"][0])
        self._mu = d["template_mu"].astype(np.float32)
        self._sigma = d["template_sigma"].astype(np.float32)
        # quality head (LR over DFT magnitudes)
        self._qmean  = d["qual_mean"].astype(np.float32)
        self._qscale = d["qual_scale"].astype(np.float32)
        self._qcoef  = d["qual_coef"].astype(np.float32)
        self._qint   = d["qual_intercept"].astype(np.float32)
        self._qcls   = d["qual_classes"]

    def _beat_feat(self, onset_b: np.ndarray, note_b: np.ndarray) -> np.ndarray:
        n = len(onset_b)
        F = np.zeros((n, 48), dtype=np.float32)
        for b in range(n):
            F[b] = np.concatenate([
                _chroma88(onset_b[b]),
                _chroma88(note_b[b]),
                _chroma88(onset_b[b], 0, 52),
                _chroma88(onset_b[b], 60, 200),
            ])
        return F

    def _windowed(self, onset_b: np.ndarray, note_b: np.ndarray) -> np.ndarray:
        F = self._beat_feat(onset_b, note_b)
        n, d = F.shape; w = self.window
        out = np.zeros((n, d * (2 * w + 1)), dtype=np.float32)
        for b in range(n):
            row = []
            for delta in range(-w, w + 1):
                nb = b + delta
                row.append(F[nb] if 0 <= nb < n else np.zeros(d, dtype=np.float32))
            out[b] = np.concatenate(row)
        return out

    @staticmethod
    def _roll_idx(d: int, r: int) -> np.ndarray:
        idx = np.arange(d)
        for start in range(0, d, 12):
            idx[start:start + 12] = start + (np.arange(12) + r) % 12
        return idx

    @staticmethod
    def _dft_feat(X: np.ndarray) -> np.ndarray:
        n, d = X.shape; nb = d // 12
        mags = np.abs(np.fft.rfft(X.reshape(n, nb, 12), n=12, axis=2))[:, :, :7]
        return mags.reshape(n, nb * 7).astype(np.float32)

    def root_proba(self, onset_b: np.ndarray, note_b: np.ndarray) -> np.ndarray:
        """(n,12) per-beat root probabilities (canonical MLP, 78.2% CV)."""
        X = self._windowed(onset_b, note_b)
        n = X.shape[0]; scores = np.zeros((n, 12), np.float32)
        cb = self.window * 4  # onset-full block of centre beat
        for r in range(12):
            Xr = X[:, self._roll_idx(X.shape[1], r)]
            if self._use_template:
                x = Xr[:, cb * 12:(cb + 1) * 12]
                ll = (-0.5 * (((x - self._mu) / self._sigma) ** 2).sum(1)).astype(np.float32)
                Xr = np.concatenate([Xr, ll[:, None]], axis=1)
            z = (Xr - self._rmean) / self._rscale
            h = np.maximum(z @ self._rW1 + self._rb1, 0.0)
            scores[:, r] = (h @ self._rW2 + self._rb2)[:, 0]
        scores -= scores.max(1, keepdims=True)
        e = np.exp(scores)
        return e / e.sum(1, keepdims=True)

    def qual_proba(self, onset_b: np.ndarray, note_b: np.ndarray) -> np.ndarray:
        """(n,5) per-beat quality probabilities (LR-DFT, 62.2% majmin CV)."""
        X = self._windowed(onset_b, note_b)
        D = self._dft_feat(X)
        z = (D - self._qmean) / self._qscale
        logits = z @ self._qcoef.T + self._qint
        logits -= logits.max(1, keepdims=True)
        e = np.exp(logits)
        p = e / e.sum(1, keepdims=True)
        out = np.zeros((len(X), 5), np.float32)
        for i, c in enumerate(self._qcls):
            out[:, int(c)] = p[:, i]
        return out

    def predict_proba(self, onset_b: np.ndarray, note_b: np.ndarray):
        """Returns (root_proba (n,12), qual_proba (n,5))."""
        return self.root_proba(onset_b, note_b), self.qual_proba(onset_b, note_b)


class _BeatSeqModelV4:
    """Per-beat ROOT model from beat_seq_model_v4.npz — the in-place root scorer.

    Winner of the 2026-07-09 per-beat bake-off (docs/known_issues.md #18): a
    key-agnostic canonical MLP ensembled with a bass-anchored LR.  On a clean
    disjoint jazz split, per-beat root: v2-style LR 86.7% → this 93.3%; and on
    held-out POP909 001-005: v2 79.4% → this 80.4% (boundary beats 72.8→75.1).

    Interface mirrors _BeatSeqModel: predict_proba(onset_b, note_b) -> (n,12).

      canon head : for each candidate root r, roll the ±window window by -r, shared
                   MLP -> scalar; softmax over 12 = key-agnostic root posterior.
      bass head  : anchor rotation on the observed bass PC (argmax of centre beat's
                   bass chroma), roll by -anchor, LR predicts offset=(root-bass)%12,
                   mapped back to absolute root.
    final posterior = normalize(softmax(canon) + P_abs(bass)).
    """

    def __init__(self, path: Path) -> None:
        d = np.load(path, allow_pickle=True)
        self.window = int(d["window"][0])
        self._rW1, self._rb1 = d["root_W1"], d["root_b1"]
        self._rW2, self._rb2 = d["root_W2"], d["root_b2"]
        self._rmean, self._rscale = d["root_mean"], d["root_scale"]
        self._use_template = bool(d["root_use_template"][0])
        self._mu, self._sigma = d["template_mu"], d["template_sigma"]
        self._bmean, self._bscale = d["ba_mean"], d["ba_scale"]
        self._bcoef, self._bint = d["ba_coef"], d["ba_intercept"]
        self._bclasses = d["ba_classes"]

    @staticmethod
    def _roll_idx(d: int, r: int) -> np.ndarray:
        idx = np.arange(d)
        for s in range(0, d, 12):
            idx[s:s + 12] = s + (np.arange(12) + r) % 12
        return idx

    def _windowed(self, onset_b: np.ndarray, note_b: np.ndarray) -> np.ndarray:
        n = len(onset_b); w = self.window
        F = np.zeros((n, 48), np.float32)
        for b in range(n):
            F[b] = np.concatenate([
                _chroma88(onset_b[b]), _chroma88(note_b[b]),
                _chroma88(onset_b[b], 0, 52), _chroma88(onset_b[b], 60, 200),
            ])
        out = np.zeros((n, 48 * (2 * w + 1)), np.float32)
        for b in range(n):
            row = [F[b + o] if 0 <= b + o < n else np.zeros(48, np.float32)
                   for o in range(-w, w + 1)]
            out[b] = np.concatenate(row)
        return out

    def _canon_proba(self, X: np.ndarray) -> np.ndarray:
        n, d = X.shape
        sc = np.zeros((n, 12), np.float32)
        cb = self.window * 4  # centre-beat onset-full block
        for r in range(12):
            Xr = X[:, self._roll_idx(d, r)]
            feat = Xr
            if self._use_template:
                x = Xr[:, cb * 12:(cb + 1) * 12]
                ll = (-0.5 * (((x - self._mu) / self._sigma) ** 2).sum(1)).astype(np.float32)
                feat = np.concatenate([Xr, ll[:, None]], axis=1)
            z = (feat - self._rmean) / self._rscale
            h = np.maximum(z @ self._rW1 + self._rb1, 0.0)
            sc[:, r] = (h @ self._rW2 + self._rb2)[:, 0]
        sc -= sc.max(1, keepdims=True)
        e = np.exp(sc)
        return e / e.sum(1, keepdims=True)

    def _ba_proba(self, X: np.ndarray) -> np.ndarray:
        d = X.shape[1]
        c = self.window * 48  # centre beat's 48d block; bass sub-block at +24:+36
        a = X[:, c + 24:c + 36].argmax(1).astype(int)
        Xw = np.empty_like(X)
        for i in range(len(X)):
            Xw[i] = X[i, self._roll_idx(d, int(a[i]))]
        z = (Xw - self._bmean) / self._bscale
        lg = z @ self._bcoef.T + self._bint
        lg -= lg.max(1, keepdims=True)
        e = np.exp(lg); p = e / e.sum(1, keepdims=True)
        P_off = np.zeros((len(X), 12), np.float32)
        for i, cl in enumerate(self._bclasses):
            P_off[:, int(cl)] = p[:, i]
        out = np.zeros_like(P_off)
        for i in range(len(P_off)):
            out[i] = np.roll(P_off[i], int(a[i]))  # P_abs[(a+off)%12] = P_off[off]
        return out

    def predict_proba(self, onset_b: np.ndarray, note_b: np.ndarray) -> np.ndarray:
        """(n,12) per-beat root probabilities indexed by pitch class 0-11."""
        X = self._windowed(onset_b, note_b)
        p = self._canon_proba(X) + self._ba_proba(X)
        return p / p.sum(1, keepdims=True)


class _FamilyClassifier:
    """Baseline LR family classifier from audio_chord_features.npz.

    Features: [norm(onset), norm(note), norm(bass), norm(treble)] each 12d,
    root-shifted so root lands at index 0.  StandardScaler applied.
    """

    def __init__(self) -> None:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler

        if not CLEAN_FEAT.exists():
            raise FileNotFoundError(
                f"Feature cache missing: {CLEAN_FEAT}\n"
                "Run scripts/build_audio_chord_features.py first."
            )
        d = np.load(CLEAN_FEAT, allow_pickle=True)
        Xc = _norm_blocks(np.hstack([d["onset"], d["note"], d["bass"], d["treble"]]))
        self.sc = StandardScaler().fit(Xc)
        self.clf = LogisticRegression(max_iter=2000).fit(
            self.sc.transform(Xc), d["family"].astype(int)
        )
        # seventh model
        b7y = d["base7"].astype(int)
        self.b7_clf = LogisticRegression(max_iter=2000).fit(
            self.sc.transform(Xc), b7y
        )
        self.base7_labels = [str(x) for x in d["base7_labels"]]
        logger.debug("_FamilyClassifier: fit on %d examples", len(Xc))

    def predict(self, root: int, seg_on: np.ndarray, seg_nt: np.ndarray,
                seg_bs: np.ndarray, seg_tr: np.ndarray,
                seventh_gate: float = 0.0) -> tuple[str, str, float]:
        """Return (family_harte, seventh_harte, confidence).

        seg_on/seg_nt: 88-dim summed piano-roll vectors.
        seg_bs/seg_tr: 12-dim register chroma (from _reg_raw).
        seventh_harte falls back to family_harte when confidence < seventh_gate.
        """
        # Convert 88-dim to 12-dim chroma before rolling — mirrors label_segment in engine
        ch_on = _reg_raw(seg_on); ch_nt = _reg_raw(seg_nt)
        rr = lambda c: np.roll(c, -root)
        f = _norm_blocks(np.hstack([rr(ch_on), rr(ch_nt), rr(seg_bs), rr(seg_tr)]))
        Xf = self.sc.transform(f[None])
        p = self.clf.predict_proba(Xf)[0]
        fam_idx = int(p.argmax())
        fam = FAMILIES[fam_idx]
        conf = float(p[fam_idx])
        fam_h = FAM_HARTE[fam]

        # seventh level
        p7 = self.b7_clf.predict_proba(Xf)[0]
        if p7.max() >= seventh_gate:
            b7_key = self.base7_labels[int(self.b7_clf.classes_[p7.argmax()])]
            sev_h = B7_HARTE.get(b7_key, fam_h)
        else:
            sev_h = fam_h
        return fam_h, sev_h, conf


class _CtxFamilyClassifier:
    """Entropy-gated context-MLP family classifier from ctx_family_model.npz.

    Saved by scripts/train_ctx_family_model.py.  Falls back gracefully to
    _FamilyClassifier if the file is absent.
    """

    def __init__(self, path: Path, base_clf: _FamilyClassifier) -> None:
        import torch, torch.nn as nn

        d = np.load(path, allow_pickle=True)
        self._base = base_clf
        self._w = float(d["gate_w"])
        self._b = float(d["gate_b"])
        flat_dim = int(d["flat_dim"])
        sc_mean = d["sc_mean"]; sc_std = d["sc_std"]
        self._sc_mean = sc_mean; self._sc_std = sc_std

        # reconstruct MLP architecture
        hidden1 = int(d["hidden1"]); hidden2 = int(d["hidden2"])
        self._mlp = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, hidden1), nn.LayerNorm(hidden1), nn.GELU(), nn.Dropout(0.0),
            nn.Linear(hidden1, hidden2), nn.LayerNorm(hidden2), nn.GELU(), nn.Dropout(0.0),
            nn.Linear(hidden2, 5),
        )
        mlp_state = {k: torch.tensor(v) for k, v in d["mlp_state"].item().items()}
        self._mlp.load_state_dict(mlp_state)
        self._mlp.eval()
        self._dist = {k: d[f"dist_{k}"] for k in
                      ["major_mu", "major_std", "minor_mu", "minor_std",
                       "diminished_mu", "diminished_std", "augmented_mu", "augmented_std",
                       "suspended_mu", "suspended_std"]}

    def predict(self, root: int, seg_on: np.ndarray, seg_nt: np.ndarray,
                seg_bs: np.ndarray, seg_tr: np.ndarray,
                chroma_abs: np.ndarray,
                ctx_ll_mats: list[np.ndarray],
                ctx_roots: list[int],
                seventh_gate: float = 0.0,
                bsm_probs_abs: np.ndarray | None = None) -> tuple[str, str, float]:
        """Return (family_harte, seventh_harte, confidence) using entropy gate."""
        import torch
        # base prediction (always used for seventh + gate)
        fam_h_base, sev_h, conf_base = self._base.predict(
            root, seg_on, seg_nt, seg_bs, seg_tr, seventh_gate
        )
        # ctx logits
        # chroma_mean root-shifted
        cn = np.linalg.norm(chroma_abs)
        cm = (chroma_abs / cn if cn > 1e-9 else chroma_abs).astype(np.float32)
        ll_mat = _compute_key_family_ll(cm, self._dist)  # (5,12)

        # build context tensor
        k = (len(ctx_ll_mats) - 1) // 2  # should be 4
        ctx_tensor = np.zeros((len(ctx_ll_mats), 5, 12), dtype=np.float32)
        for j, (ll_j, root_j) in enumerate(zip(ctx_ll_mats, ctx_roots)):
            if ll_j is not None:
                delta = (root_j - root) % 12
                ctx_tensor[j] = np.roll(ll_j, -delta, axis=1)

        ctx_flat = ctx_tensor.reshape(-1)
        X_ctx = np.concatenate([cm, ctx_flat])
        X_ctx = ((X_ctx - self._sc_mean) / (self._sc_std + 1e-9)).astype(np.float32)

        with torch.no_grad():
            logits_ctx = self._mlp(torch.tensor(X_ctx[None])).numpy()[0]

        # base logits from family classifier (same 12-dim conversion as _FamilyClassifier.predict)
        ch_on = _reg_raw(seg_on); ch_nt = _reg_raw(seg_nt)
        rr = lambda c: np.roll(c, -root)
        f = _norm_blocks(np.hstack([rr(ch_on), rr(ch_nt), rr(seg_bs), rr(seg_tr)]))
        logits_base = self._base.clf.decision_function(self._base.sc.transform(f[None]))[0]

        # entropy gate
        def _softmax(lg):
            lg = lg - lg.max()
            e = np.exp(lg); return e / e.sum()
        pb = _softmax(logits_base); pc = _softmax(logits_ctx)
        H = -float((pb * np.log(pb + 1e-12)).sum())
        alpha = 1.0 / (1.0 + np.exp(-(self._w * H + self._b)))
        p_mix = alpha * pb + (1.0 - alpha) * pc
        fam_idx = int(p_mix.argmax())
        fam = FAMILIES[fam_idx]
        conf = float(p_mix[fam_idx])
        return FAM_HARTE[fam], sev_h, conf


def _compute_key_family_ll(chroma_mean: np.ndarray, dist: dict) -> np.ndarray:
    """(5, 12) log-likelihood matrix for family × key, matching experiment_ctx_model."""
    ll = np.zeros((5, 12), dtype=np.float32)
    for fi, fam in enumerate(FAMILIES):
        mu = dist[f"{fam}_mu"]; std = dist[f"{fam}_std"]
        for r in range(12):
            x = np.roll(chroma_mean, -r)
            ll[fi, r] = float(-0.5 * np.sum(((x - mu) / std) ** 2) - np.sum(np.log(std)))
    return ll


class _CtxFamilyClassifierV2:
    """Dual-head ctx MLP from ctx_v2.npz.

    Features (684d): chroma(12) + ctx_ll(540) + root_intervals(108) + bsm_rel(12) + bsm_abs(12).
    Outputs: family_head(5) + root_head(12) off a shared 256→128 trunk.
    Entropy-gated blend with base LR classifier for family; root head used separately.

    Trained with oracle MIDI beat rolls for bsm features.  At inference, bsm_probs_abs
    comes from beat_seq_model running on Basic Pitch piano rolls — slightly noisier.
    """

    def __init__(self, path: Path, base_clf: _FamilyClassifier) -> None:
        import torch
        import torch.nn as nn

        d = np.load(path, allow_pickle=True)
        self._base = base_clf
        self._w = float(d["gate_w"])
        self._b = float(d["gate_b"])
        flat_dim = int(d["flat_dim"])
        self._sc_mean = d["sc_mean"].astype(np.float32)
        self._sc_std  = d["sc_std"].astype(np.float32)
        self._dist = {k: d[f"dist_{k}"] for k in
                      ["major_mu", "major_std", "minor_mu", "minor_std",
                       "diminished_mu", "diminished_std", "augmented_mu", "augmented_std",
                       "suspended_mu", "suspended_std"]}

        class _MLPv2(nn.Module):
            def __init__(self):
                super().__init__()
                self.shared = nn.Sequential(
                    nn.Flatten(),
                    nn.Linear(flat_dim, 256), nn.LayerNorm(256), nn.GELU(), nn.Dropout(0.0),
                    nn.Linear(256, 128),      nn.LayerNorm(128), nn.GELU(), nn.Dropout(0.0),
                )
                self.family_head = nn.Linear(128, 5)
                self.root_head   = nn.Linear(128, 12)
            def forward(self, x):
                h = self.shared(x)
                return self.family_head(h), self.root_head(h)

        self._mlp = _MLPv2()
        mlp_state = {k: torch.tensor(v) for k, v in d["mlp_state"].item().items()}
        self._mlp.load_state_dict(mlp_state)
        self._mlp.eval()

    def predict(self, root: int, seg_on: np.ndarray, seg_nt: np.ndarray,
                seg_bs: np.ndarray, seg_tr: np.ndarray,
                chroma_abs: np.ndarray,
                ctx_ll_mats: list[np.ndarray],
                ctx_roots: list[int],
                seventh_gate: float = 0.0,
                bsm_probs_abs: np.ndarray | None = None) -> tuple[str, str, float]:
        """Return (family_harte, seventh_harte, confidence) using dual-head ctx v2."""
        import torch

        # base prediction (used for seventh + entropy gate)
        fam_h_base, sev_h, conf_base = self._base.predict(
            root, seg_on, seg_nt, seg_bs, seg_tr, seventh_gate
        )

        # chroma_mean: root-shifted, L2-normed (same as v1)
        cn = np.linalg.norm(chroma_abs)
        cm = (chroma_abs / cn if cn > 1e-9 else chroma_abs).astype(np.float32)
        ll_mat = _compute_key_family_ll(cm, self._dist)  # unused for features, kept for consistency

        # ctx_flat (540d): 9 × 5 × 12 neighbour ll_mats, key-unified to current root
        k = (len(ctx_ll_mats) - 1) // 2  # should be 4
        ctx_tensor = np.zeros((len(ctx_ll_mats), 5, 12), dtype=np.float32)
        for j, (ll_j, root_j) in enumerate(zip(ctx_ll_mats, ctx_roots)):
            if ll_j is not None:
                delta = (root_j - root) % 12
                ctx_tensor[j] = np.roll(ll_j, -delta, axis=1)
        ctx_flat = ctx_tensor.reshape(-1)  # (540,)

        # root_interval one-hots (108d): (root_j - root_i) % 12 for 9 window positions
        W = len(ctx_roots)  # 9
        root_inv = np.zeros(W * 12, dtype=np.float32)
        for j, root_j in enumerate(ctx_roots):
            delta = int((root_j - root) % 12)
            root_inv[j * 12 + delta] = 1.0

        # bsm features (24d)
        if bsm_probs_abs is None:
            bsm_probs_abs = np.full(12, 1.0 / 12, dtype=np.float32)
        bsm_abs = bsm_probs_abs.astype(np.float32)
        bsm_rel = np.roll(bsm_abs, -root).astype(np.float32)

        # assemble 684d feature vector
        X_ctx = np.concatenate([cm, ctx_flat, root_inv, bsm_rel, bsm_abs])
        X_ctx = ((X_ctx - self._sc_mean) / (self._sc_std + 1e-9)).astype(np.float32)

        with torch.no_grad():
            fam_logits, _ = self._mlp(torch.tensor(X_ctx[None]))
            logits_ctx = fam_logits.numpy()[0]

        # base logits from LR family classifier
        ch_on = _reg_raw(seg_on); ch_nt = _reg_raw(seg_nt)
        rr = lambda c: np.roll(c, -root)
        f = _norm_blocks(np.hstack([rr(ch_on), rr(ch_nt), rr(seg_bs), rr(seg_tr)]))
        logits_base = self._base.clf.decision_function(self._base.sc.transform(f[None]))[0]

        def _softmax(lg):
            lg = lg - lg.max()
            e = np.exp(lg); return e / e.sum()

        pb = _softmax(logits_base); pc = _softmax(logits_ctx)
        H = -float((pb * np.log(pb + 1e-12)).sum())
        alpha = 1.0 / (1.0 + np.exp(-(self._w * H + self._b)))
        p_mix = alpha * pb + (1.0 - alpha) * pc
        fam_idx = int(p_mix.argmax())
        fam = FAMILIES[fam_idx]
        conf = float(p_mix[fam_idx])
        return FAM_HARTE[fam], sev_h, conf


# ── module-level lazy-loaded models ──────────────────────────────────────────

_family_clf: _FamilyClassifier | None = None
_ctx_clf: _CtxFamilyClassifier | _CtxFamilyClassifierV2 | None = None
_beat_seq: _BeatSeqModel | _BeatSeqModelV4 | None = None
_beat_seq_v3: _BeatSeqModelV3 | None = None
_root_mdl: _RootModel | None = None


def _get_family_clf() -> _FamilyClassifier:
    global _family_clf
    if _family_clf is None:
        _family_clf = _FamilyClassifier()
    return _family_clf


def _get_ctx_clf() -> _CtxFamilyClassifier | _CtxFamilyClassifierV2 | None:
    global _ctx_clf
    if _ctx_clf is not None:
        return _ctx_clf
    # prefer ctx_v2 (dual-head, 684d), then v1 large, then v1 small
    v2_path = MODELS / "ctx_v2.npz"
    if v2_path.exists():
        try:
            import torch
            _ctx_clf = _CtxFamilyClassifierV2(v2_path, _get_family_clf())
            logger.info("chord_pipeline_v1: loaded ctx v2 model (684d dual-head)")
            return _ctx_clf
        except Exception as e:
            logger.warning("chord_pipeline_v1: ctx_v2.npz load failed (%s) — falling back", e)
    for candidate in ("ctx_family_model_large.npz", "ctx_family_model.npz"):
        ctx_path = MODELS / candidate
        if not ctx_path.exists():
            continue
        try:
            import torch
            _ctx_clf = _CtxFamilyClassifier(ctx_path, _get_family_clf())
            logger.info("chord_pipeline_v1: loaded ctx family model %s", candidate)
            return _ctx_clf
        except Exception as e:
            logger.warning("chord_pipeline_v1: ctx model %s load failed (%s)", candidate, e)
    return None


def _get_beat_seq() -> _BeatSeqModel | _BeatSeqModelV4 | None:
    global _beat_seq
    if _beat_seq is not None:
        return _beat_seq
    # prefer v4 (canonical ⊕ bass-anchored, per-beat bake-off winner) over v2 over v1
    v4 = MODELS / "beat_seq_model_v4.npz"
    if v4.exists():
        try:
            _beat_seq = _BeatSeqModelV4(v4)
            logger.info("chord_pipeline_v1: loaded beat_seq_model_v4 (canon⊕bass-anchored root)")
            return _beat_seq
        except Exception as exc:
            logger.warning("chord_pipeline_v1: v4 load failed (%s) — falling back to v2", exc)
    for candidate in ("beat_seq_model_v2.npz", "beat_seq_model.npz"):
        p = MODELS / candidate
        if p.exists():
            _beat_seq = _BeatSeqModel(p)
            logger.info("chord_pipeline_v1: loaded %s", candidate)
            return _beat_seq
    return _beat_seq


def _get_beat_seq_v3() -> _BeatSeqModelV3 | None:
    global _beat_seq_v3
    if _beat_seq_v3 is not None:
        return _beat_seq_v3
    p = MODELS / "beat_seq_model_v3.npz"
    if p.exists():
        try:
            _beat_seq_v3 = _BeatSeqModelV3(p)
            logger.info("chord_pipeline_v1: loaded beat_seq_model_v3 (quality-boundary head)")
        except Exception as exc:
            logger.warning("chord_pipeline_v1: v3 load failed (%s)", exc)
    return _beat_seq_v3


def _get_root_mdl() -> _RootModel | None:
    global _root_mdl
    if _root_mdl is not None:
        return _root_mdl
    p = MODELS / "root_model.npz"
    if p.exists():
        _root_mdl = _RootModel(p)
    return _root_mdl


# ── progression-encoder quality reranker (issue #21) ──────────────────────────
# Second pass over the classified chord SEQUENCE (not the audio): a small
# non-causal transformer (harmonia/models/progression_encoder.py) refines each
# segment's coarse 5-class quality (maj/min/dom/hdim/dim) from its ±6-chord
# harmonic neighbourhood.  Motivation: the per-segment acoustic classifier is
# IID given audio and under-recalls the *grammatical* dom family (54% in prod);
# the encoder scores 86.8% dom recall standalone by reading ii-V-I context.
# The dom↔maj distinction is a 7ths-level (tetrad) call, invisible to majmin.

# Fine Harte quality (sev_h emitted by the family/ctx classifier) → 5-class q5.
_HARTE_TO_Q5NAME = {
    "maj": "maj", "min": "min", "dim": "dim", "aug": "maj",
    "sus4": "maj", "sus2": "maj",
    "maj7": "maj", "min7": "min", "7": "dom", "hdim7": "hdim",
    "dim7": "dim", "minmaj7": "min", "aug7": "dom", "augmaj7": "maj",
}
# q5 index → (triad Harte, seventh Harte) canonical form, used when the encoder
# overrides the family; preserve the acoustic triad-vs-seventh choice.
_Q5IDX_TO_HARTE = {
    0: ("maj", "maj7"),    # maj
    1: ("min", "min7"),    # min
    2: ("7", "7"),         # dom (dominant is inherently a seventh)
    3: ("hdim7", "hdim7"), # hdim
    4: ("dim", "dim7"),    # dim
}
_SEVENTH_HARTE = {"maj7", "min7", "7", "hdim7", "dim7", "minmaj7", "aug7", "augmaj7"}

_prog_encoder = None
_prog_encoder_loaded = False


def _get_progression_encoder():
    """Lazy-load the ProgressionEncoder checkpoint (None if unavailable)."""
    global _prog_encoder, _prog_encoder_loaded
    if _prog_encoder_loaded:
        return _prog_encoder
    _prog_encoder_loaded = True
    path = MODELS / "progression_encoder.pt"
    if not path.exists():
        logger.warning("chord_pipeline_v1: progression_encoder.pt missing — reranker off")
        return None
    try:
        from harmonia.models.progression_encoder import load_encoder
        _prog_encoder = load_encoder(path, device="cpu")
        logger.info("chord_pipeline_v1: loaded progression_encoder (quality reranker)")
    except Exception as exc:
        logger.warning("chord_pipeline_v1: progression_encoder load failed (%s)", exc)
        _prog_encoder = None
    return _prog_encoder


def _harte_to_q5idx(sev_h: str):
    from harmonia.models.progression_encoder import QUAL5_IDX
    fam = _HARTE_TO_Q5NAME.get(sev_h)
    return QUAL5_IDX[fam] if fam is not None else None


def rerank_progression_qualities(
    roots: list[int], sev_hs: list[str], confs: list[float],
    *, weight: float = 0.5, encoder=None,
) -> list[str]:
    """Second-pass ProgressionEncoder quality rerank over a chord sequence.

    Args:
        roots:   per-segment root pitch-class (0-11).
        sev_hs:  per-segment fine Harte quality (e.g. "maj7", "7", "min").
        confs:   per-segment acoustic confidence (family/ctx max-prob), in [0,1].
        weight:  progression_weight in log_post = log_acoustic + w·log_encoder.
        encoder: preloaded ProgressionEncoder (defaults to the module singleton).

    Returns:
        A new list of Harte qualities; unchanged where the encoder agrees with
        the acoustic call or the segment quality is outside the 5-class vocab.
        Preserves the acoustic triad-vs-seventh choice when the family flips.
    """
    out = list(sev_hs)
    encoder = encoder if encoder is not None else _get_progression_encoder()
    N = len(roots)
    if encoder is None or N == 0:
        return out

    import torch

    from harmonia.models.progression_encoder import CTX, MASK_ID, WINDOW

    q5 = [_harte_to_q5idx(s) for s in sev_hs]

    R = np.zeros((N, WINDOW), dtype=np.int64)
    Q = np.full((N, WINDOW), MASK_ID, dtype=np.int64)
    C = np.zeros((N, WINDOW), dtype=np.float32)
    P_mask = np.ones((N, WINDOW), dtype=bool)  # True = padding
    for i in range(N):
        for jj in range(WINDOW):
            k = i + jj - CTX
            if 0 <= k < N and q5[k] is not None:
                R[i, jj] = (roots[k] - roots[i]) % 12
                Q[i, jj] = q5[k]
                C[i, jj] = confs[k]
                P_mask[i, jj] = False

    with torch.no_grad():
        logits = encoder(
            torch.from_numpy(R), torch.from_numpy(Q),
            torch.from_numpy(C), torch.from_numpy(P_mask),
        )
        enc_logp = torch.log_softmax(logits, dim=-1).numpy()  # (N,5) log-probs

    for i in range(N):
        if q5[i] is None:
            continue
        # acoustic 5-class log-probs: confidence-gated one-hot on the greedy q5
        c = float(min(max(confs[i], 1e-3), 1.0 - 1e-3))
        aco = np.full(5, np.log((1.0 - c) / 4.0), dtype=np.float32)
        aco[q5[i]] = np.log(c)
        combined = aco + weight * enc_logp[i]
        new_q5 = int(combined.argmax())
        if new_q5 != q5[i]:
            triad, seventh = _Q5IDX_TO_HARTE[new_q5]
            out[i] = seventh if sev_hs[i] in _SEVENTH_HARTE else triad
    return out


# ── segmentation ──────────────────────────────────────────────────────────────

def _fit_harmonic_grid(beat_proba: np.ndarray) -> int:
    """Estimate per-song harmonic grid resolution: 2 or 4 beats.

    Checks how often consecutive 2-beat windows share the same argmax root.
    Stability > 0.65 → chords hold for a full bar → 4-beat grid.
    Based on POP909 corpus: 43% of bars are 4-beat (once/bar), 39% are 2-beat.
    """
    if len(beat_proba) < 4:
        return 2
    roots = beat_proba.argmax(1)
    n_pairs = len(roots) // 2
    same = sum(int(roots[2 * i] == roots[2 * i + 1]) for i in range(n_pairs))
    return 4 if (same / n_pairs) > 0.65 else 2


def _make_grid_segs(n_beats: int, grid: int) -> list[tuple[int, int]]:
    """Fixed grid segmentation: non-overlapping windows of `grid` beats."""
    return [(i, min(i + grid, n_beats)) for i in range(0, n_beats, grid)]


def _root_change_segs(beat_proba: np.ndarray) -> list[tuple[int, int]]:
    """Segment wherever the per-beat root argmax changes (gmerge).

    Unlike _merge_grid_by_root (fixed 2/4-beat cells, then merged — which
    UNDER-segments across ii-V changes that fall mid-cell), this cuts at any
    beat, so a chord that changes every 2 beats is captured exactly.  On
    held-out irealb end-to-end (detected beats) this lifts MIREX root from
    70.8% (grid-merge) to 88.7% — within 3pp of the oracle-segmentation
    ceiling (91.7%); on POP909 it matches the best grid baseline.  See
    docs/known_issues.md #18 (per-beat bake-off + segmentation follow-up).
    """
    pred = beat_proba.argmax(1)
    n = len(pred)
    if n == 0:
        return []
    cuts = [0] + [b for b in range(1, n) if pred[b] != pred[b - 1]] + [n]
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]


def _merge_grid_by_root(segs: list[tuple[int, int]],
                        beat_proba: np.ndarray) -> list[tuple[int, int]]:
    """Merge adjacent grid cells whose beat_proba argmax agrees on root.

    Runs before segment classification so merged cells pool more beats for
    a more stable root vote. Also makes the grid robust when the tempo
    estimate is off (adjacent 'beats' then share the same root prediction
    and collapse into a single segment automatically).
    """
    merged = [list(segs[0])]
    for s, e in segs[1:]:
        cur_r = int(beat_proba[merged[-1][0]:merged[-1][1]].sum(0).argmax())
        new_r = int(beat_proba[s:e].sum(0).argmax())
        if cur_r == new_r:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(m[0], m[1]) for m in merged]


def _merge_grid_by_root_and_bass(
    segs: list[tuple[int, int]],
    beat_proba: np.ndarray,
    onset_b: np.ndarray,
    qual_proba: np.ndarray | None = None,
) -> list[tuple[int, int]]:
    """Merge adjacent grid cells. Keep boundary when root OR bass OR quality changes.

    Extends _merge_grid_by_root with two additional boundary signals:
    - Bass pitch class: dominant PC in MIDI 21–51 (piano-roll indices 0–30).
      Catches inversions and slash chords (e.g. C/E → keeps the split because
      the bass note E ≠ C even though the root stays C in the harmonic grid).
    - Quality change (if qual_proba from _BeatSeqModelV3 is supplied): a chord
      family change (major→minor, major→dom7, …) is kept as a segment boundary
      even when the argmax root is the same on both sides.

    Bass aggregation: sum across segment beats before argmax so walking-bass
    fluctuations within a chord are smoothed away.  Falls back to root when
    the bass register is silent (e.g. solo piano, no low notes).
    """
    merged = [list(segs[0])]
    for s, e in segs[1:]:
        cs, ce = merged[-1][0], merged[-1][1]

        # root
        cur_r = int(beat_proba[cs:ce].sum(0).argmax())
        new_r = int(beat_proba[s:e].sum(0).argmax())
        root_same = (cur_r == new_r)

        # bass
        cur_bv = _reg_raw(onset_b[cs:ce].sum(0), 0, 52)
        new_bv = _reg_raw(onset_b[s:e].sum(0), 0, 52)
        cur_bass = int(cur_bv.argmax()) if cur_bv.sum() > 1e-6 else cur_r
        new_bass = int(new_bv.argmax()) if new_bv.sum() > 1e-6 else new_r
        bass_same = (cur_bass == new_bass)

        # quality (optional)
        qual_same = True
        if qual_proba is not None:
            cur_q = int(qual_proba[cs:ce].sum(0).argmax())
            new_q = int(qual_proba[s:e].sum(0).argmax())
            qual_same = (cur_q == new_q)

        if root_same and bass_same and qual_same:
            merged[-1][1] = e
        else:
            merged.append([s, e])
    return [(m[0], m[1]) for m in merged]


# TCS projection matrix (Harte & Sandler 2006) — same geometry as chord_hmm.hcdf
# but applied here to (n, 12) root-probability vectors rather than 88-dim piano rolls.
_r12 = np.arange(12, dtype=np.float32)
_TCS12 = np.stack([
    np.sin(_r12 * 7 * np.pi / 6), np.cos(_r12 * 7 * np.pi / 6),
    np.sin(_r12 * 3 * np.pi / 2), np.cos(_r12 * 3 * np.pi / 2),
    np.sin(_r12 * 2 * np.pi / 3), np.cos(_r12 * 2 * np.pi / 3),
], axis=0).astype(np.float32)  # (6, 12)
del _r12


def _tcs12(prob: np.ndarray) -> np.ndarray:
    """(N, 12) root probabilities → (N, 6) Tonal Centroid Space vectors."""
    s = prob.sum(axis=-1, keepdims=True)
    p = prob / np.where(s > 1e-9, s, 1.0)
    if p.ndim == 1:
        return (p @ _TCS12.T)
    return p @ _TCS12.T


def _merge_grid_by_divergence(segs: list[tuple[int, int]],
                               beat_proba: np.ndarray,
                               threshold: float = 0.15) -> list[tuple[int, int]]:
    """Merge adjacent grid cells with low harmonic divergence.

    Soft version of _merge_grid_by_root: uses TCS distance on segment-averaged
    beat_proba (Option C from the handoff) rather than argmax equality.
    Cells whose root distributions are harmonically close in Tonal Centroid Space
    (e.g., C and Am) are merged; harmonically distant cells (e.g., C and F#) are
    kept separate. More selective than argmax equality because:
      - Handles ambiguous C vs C# without always splitting (TCS dist is small)
      - Handles same-argmax C-major vs C-minor correctly (small TCS dist → merge)
      - threshold ~0.15: fires on genuine root changes, ignores beat-to-beat noise

    threshold: TCS distance above which grid cells are treated as separate chords.
    """
    merged = [list(segs[0])]
    for s, e in segs[1:]:
        prev_avg = beat_proba[merged[-1][0]:merged[-1][1]].mean(0)  # (12,)
        next_avg = beat_proba[s:e].mean(0)
        dist = float(np.linalg.norm(_tcs12(prev_avg) - _tcs12(next_avg)))
        if dist > threshold:
            merged.append([s, e])
        else:
            merged[-1][1] = e
    return [(m[0], m[1]) for m in merged]


def _coarse_segments(onset_b: np.ndarray, theta: float = 0.08,
                     cell: int = 2) -> list[tuple[int, int]]:
    """Fixed cell-beat grid + cosine novelty merge.

    theta=0.08 is the recall-tuned value (lower θ → more segments, coalesce
    same-label removes false positives cheaply).  cell=2 gives AUC 0.962 for
    change vs hold on the iReal corpus.
    """
    nb = len(onset_b)
    blocks: list[tuple[int, int]] = []
    s = 0
    while s < nb:
        blocks.append((s, min(s + cell, nb)))
        s += cell

    bfeat = [_feat24(onset_b[a:b].sum(0)) for a, b in blocks]
    novs = [_cos_dist(bfeat[i], bfeat[i - 1]) for i in range(1, len(blocks))]

    segs = [list(blocks[0])]
    for i, (a, b) in enumerate(blocks[1:]):
        if novs[i] > theta:
            segs.append([a, b])
        else:
            segs[-1][1] = b
    return [(s[0], s[1]) for s in segs]


# ── beat-feature extraction (used by YouTube corpus builder) ─────────────────

from dataclasses import dataclass as _dc


@_dc
class BeatFeatures:
    """Intermediate beat-level features from the v1 pipeline, before classifiers run."""
    onset_b: "np.ndarray"    # (n_beats, 88) sum-pooled onset probs
    note_b: "np.ndarray"     # (n_beats, 88) sum-pooled note probs
    beat_times: "np.ndarray" # (n_beats+1,) beat boundary times in seconds
    tempo_bpm: float


def extract_beat_features(
    audio_path: Path,
    *,
    cache_dir: Path | None = None,
) -> BeatFeatures:
    """Run steps 1–4 of chord_pipeline_v1 and return raw beat-level features.

    Used by the YouTube corpus builder to extract training features from real
    audio without running the full classifier chain.
    """
    audio_path = Path(audio_path)
    y, sr = sf.read(audio_path)
    y = (y.mean(1) if y.ndim > 1 else y).astype("float32")
    duration_s = len(y) / sr

    tempo_arr, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    tempo_bpm = float(np.atleast_1d(tempo_arr)[0])
    beat_times_raw = librosa.frames_to_time(beat_frames, sr=sr)

    period = 60.0 / max(tempo_bpm, 1.0)
    ang = 2 * np.pi * (beat_times_raw % period) / period
    phase = (np.angle(np.mean(np.exp(1j * ang))) % (2 * np.pi)) * period / (2 * np.pi)
    bt = np.arange(phase, duration_s + period, period)
    bt = np.unique(np.concatenate([[0.0], bt, [duration_s]]))

    ex = PitchExtractor(cache_dir=cache_dir)
    acts = ex.extract(audio_path)

    onset_b = _pool_beats(acts.frame_times, acts.onset_probs, bt)
    note_b  = _pool_beats(acts.frame_times, acts.note_probs,  bt)
    return BeatFeatures(onset_b=onset_b, note_b=note_b, beat_times=bt, tempo_bpm=tempo_bpm)


# ── main inference ────────────────────────────────────────────────────────────

def infer_chords_v1(
    audio_path: Path,
    *,
    theta: float = 0.08,
    cell: int = 2,
    seventh_gate: float = 0.0,
    cache_dir: Path | None = None,
    use_beat_seq: bool = True,
    use_ctx_model: bool = True,
    use_harmonic_grid: bool = True,
    use_bass_tracking: bool = False,
    use_diatonic_prior: bool = False,
    diatonic_boost: float = 4.0,
    threshold_chromatic: float = 0.80,
    use_progression_prior: bool = True,
    progression_weight: float = 0.5,
) -> ChordChart:
    """Infer a ChordChart from an audio file using the Gen-2 pipeline.

    Args:
        audio_path:    Path to audio (any format soundfile can read).
        theta:         Cosine novelty threshold for chord-change detection (0.08
                       is the recall-tuned value; increase to reduce fragmentation).
        cell:          Coarse grid cell size in beats (2 is optimal per AUC analysis).
        seventh_gate:  Descend to the seventh level only when max-prob >= this
                       (0.0 = always report seventh; 0.6 = confident-only).
        cache_dir:     Cache directory for Basic Pitch activations.
        use_beat_seq:    Use the beat-sequence root model (88.3% CV) when available.
        use_ctx_model:   Use the entropy-gated ctx MLP family model when saved.
        use_bass_tracking: Also split on bass-PC or quality changes (v3 head) at
                         grid boundaries.  Disabled by default: on piano-only renders
                         the bass alternates root–fifth within a chord, so extra splits
                         get coalesced back by step 9 and add only overhead.  Useful
                         for real multi-track audio with a stem-isolated bass guitar.
        use_diatonic_prior: Apply a section-local, confidence-gated diatonic
                         quality prior (issue #20).  **Default OFF.**  The GT
                         premise holds (POP909 is ~93% diatonic in the local key)
                         but the *inferred* local key is not accurate enough to
                         exploit it: end-to-end it is a coin-flip (best config
                         POP909 majmin +0.1pp, default config −0.6pp; jazz1460
                         −0.8pp).  Kept as opt-in infrastructure — see
                         apply_diatonic_prior and docs/known_issues.md #20.  The
                         real lever is better local-key inference, not the prior.
        diatonic_boost:  Strength of the diatonic prior (log-weight base, 4.0).
        threshold_chromatic: Acoustic-confidence gate; at/above it the acoustic
                         call is trusted and the prior is skipped (0.80, the
                         least-bad opt-in value from the POP909 sweep).
        use_progression_prior: Second-pass ProgressionEncoder quality rerank
                         (issue #21).  Refines each segment's coarse quality
                         (maj/min/dom/hdim/dim) from its ±6-chord context via a
                         learned transformer over the classified sequence.  Its
                         main lever is dom recall (a 7ths-level call), so it
                         moves the sevenths metric more than majmin.
        progression_weight: Encoder weight in the log-posterior combination
                         (log_acoustic + w·log_encoder); 0 = acoustic only.

    Returns:
        ChordChart with fields populated for the interactive renderer.
    """
    audio_path = Path(audio_path)
    logger.info("chord_pipeline_v1: %s", audio_path.name)

    # ── 1. Load audio ─────────────────────────────────────────────────────────
    y, sr = sf.read(audio_path)
    y = (y.mean(1) if y.ndim > 1 else y).astype("float32")
    duration_s = len(y) / sr

    # ── 2. Beat tracking — tempo-grid de-jitter ───────────────────────────────
    # librosa tempo is accurate to ~1%; per-beat times jitter.  On metronomic
    # audio, impose a uniform grid at detected tempo + circular-mean phase.
    # This recovers ~20 majmin pts vs raw librosa beat times.
    tempo_arr, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    tempo_bpm = float(np.atleast_1d(tempo_arr)[0])
    beat_times_raw = librosa.frames_to_time(beat_frames, sr=sr)

    if len(beat_times_raw) < 4:
        logger.warning("chord_pipeline_v1: too few beats (%d)", len(beat_times_raw))
        return ChordChart(
            source_path=str(audio_path), duration_s=duration_s,
            tempo_bpm=tempo_bpm, time_signature="4/4",
            global_key="C major", global_key_confidence=0.0, style="v1",
            modulations=[],
            chords=[{"label": "Cmaj", "start_s": 0.0, "end_s": duration_s,
                     "duration_beats": 1, "confidence": 0.0}],
            segments=[{"start_s": 0.0, "end_s": duration_s, "key": "C major", "n_beats": 1}],
        )

    period = 60.0 / max(tempo_bpm, 1.0)
    ang = 2 * np.pi * (beat_times_raw % period) / period
    phase = (np.angle(np.mean(np.exp(1j * ang))) % (2 * np.pi)) * period / (2 * np.pi)
    bt = np.arange(phase, duration_s + period, period)
    bt = np.unique(np.concatenate([[0.0], bt, [duration_s]]))

    # ── 3. Basic Pitch features ───────────────────────────────────────────────
    ex = PitchExtractor(cache_dir=cache_dir)
    acts = ex.extract(audio_path)

    # ── 4. Pool to beats (SUM) ────────────────────────────────────────────────
    onset_b = _pool_beats(acts.frame_times, acts.onset_probs, bt)  # (n_beats, 88)
    note_b  = _pool_beats(acts.frame_times, acts.note_probs,  bt)  # (n_beats, 88)
    n_beats = len(onset_b)

    # ── 5. Beat-sequence root probabilities ──────────────────────────────────
    beat_seq = _get_beat_seq() if use_beat_seq else None
    beat_proba: np.ndarray | None = None
    if beat_seq is not None:
        beat_proba = beat_seq.predict_proba(onset_b, note_b)  # (n_beats, 12)

    # v3 quality head for segmentation boundary detection
    qual_proba: np.ndarray | None = None
    if use_bass_tracking and beat_proba is not None:
        bsv3 = _get_beat_seq_v3()
        if bsv3 is not None:
            qual_proba = bsv3.qual_proba(onset_b, note_b)  # (n_beats, 5)

    # ── 6. Segmentation ───────────────────────────────────────────────────────
    # Harmonic grid: estimate 2- or 4-beat resolution from beat_proba consistency.
    # Bass and quality signals complement the root argmax as boundary detectors:
    # a split is kept if root OR bass OR quality changes across a grid boundary.
    if use_harmonic_grid and beat_proba is not None:
        # Confidence gate: beat_seq max-prob is a proxy for pooling quality.
        # Low confidence indicates broken beat alignment (e.g. 2x tempo bug) or
        # a key with very few iReal training examples (e.g. F#). In those cases
        # acoustic novelty (cosine of raw onset chroma) is more robust.
        mean_conf = float(beat_proba.max(1).mean())
        if mean_conf < 0.30:
            logger.debug(
                "harmonic grid: low beat_proba confidence (%.2f) — acoustic fallback",
                mean_conf,
            )
            segs = _coarse_segments(onset_b, theta=theta, cell=cell)
        else:
            grid = _fit_harmonic_grid(beat_proba)
            segs = _make_grid_segs(n_beats, grid)
            if use_bass_tracking:
                segs = _merge_grid_by_root_and_bass(segs, beat_proba, onset_b, qual_proba)
                logger.debug(
                    "harmonic grid: %d-beat  conf=%.2f  %d segs (root+bass+qual merge)",
                    grid, mean_conf, len(segs),
                )
            else:
                # gmerge: cut wherever the per-beat root changes (not fixed grid
                # cells). +17.9pp MIREX root end-to-end on held-out irealb vs the
                # old grid-cell merge, which under-segmented across mid-cell ii-V
                # changes.  See _root_change_segs / known_issues #18.
                segs = _root_change_segs(beat_proba)
                logger.debug(
                    "harmonic grid: %d-beat est, gmerge (root-change) conf=%.2f  %d segs",
                    grid, mean_conf, len(segs),
                )
    else:
        segs = _coarse_segments(onset_b, theta=theta, cell=cell)

    root_mdl = _get_root_mdl()

    # ── 7–8. Classify each segment ────────────────────────────────────────────
    fam_clf = _get_family_clf()
    ctx_clf = _get_ctx_clf() if use_ctx_model else None

    # Pre-compute per-segment chroma means + ll_mats for ctx model (if needed)
    seg_ll_mats: list[np.ndarray | None] = [None] * len(segs)
    seg_roots: list[int] = [0] * len(segs)
    if ctx_clf is not None:
        dist_path = REPO / "data" / "cache" / "ltas_family_dist.npz"
        if dist_path.exists():
            _d = np.load(dist_path)
            _dist = {k: _d[k] for k in _d.files}
            for idx, (s, e) in enumerate(segs):
                seg_on_sum = onset_b[s:e].sum(0)
                ch_abs = _reg_raw(seg_on_sum)
                cn = np.linalg.norm(ch_abs)
                cm = (ch_abs / cn if cn > 1e-9 else ch_abs).astype(np.float32)
                seg_ll_mats[idx] = _compute_key_family_ll(cm, _dist)
        else:
            logger.warning("chord_pipeline_v1: ltas_family_dist.npz missing — ctx model disabled")
            ctx_clf = None

    labeled: list[tuple[float, float, str, str, float]] = []  # (t_start, t_end, fam_h, sev_h, conf)

    for idx, (s, e) in enumerate(segs):
        seg_on = onset_b[s:e].sum(0)   # (88,)
        seg_nt = note_b[s:e].sum(0)
        seg_bs = _reg_raw(seg_on, 0, 52)
        seg_tr = _reg_raw(seg_on, 60, 200)

        # root
        if beat_proba is not None:
            p_seg = beat_proba[s:e].sum(0)
            root = int(np.argmax(p_seg))
        elif root_mdl is not None:
            root = root_mdl.predict(seg_on, seg_nt)
        else:
            bass = _reg_raw(seg_on, 0, 52)
            root = int(bass.argmax()) if bass.sum() > 1e-6 else int(_reg_raw(seg_on).argmax())

        seg_roots[idx] = root

        # family + seventh
        if ctx_clf is not None and seg_ll_mats[idx] is not None:
            k_ctx = 4
            ctx_ll = [seg_ll_mats[max(0, idx - k_ctx + j)] if 0 <= idx - k_ctx + j < len(segs) else None
                      for j in range(2 * k_ctx + 1)]
            ctx_rt = [seg_roots[max(0, idx - k_ctx + j)] if 0 <= idx - k_ctx + j < len(segs) else 0
                      for j in range(2 * k_ctx + 1)]
            seg_on_sum = onset_b[s:e].sum(0)
            ch_abs = _reg_raw(seg_on_sum)
            bsm_abs = beat_proba[s:e].mean(0) if beat_proba is not None else None
            fam_h, sev_h, conf = ctx_clf.predict(
                root, seg_on, seg_nt, seg_bs, seg_tr,
                ch_abs, ctx_ll, ctx_rt, seventh_gate,
                bsm_probs_abs=bsm_abs,
            )
        else:
            fam_h, sev_h, conf = fam_clf.predict(
                root, seg_on, seg_nt, seg_bs, seg_tr, seventh_gate
            )

        # ── diatonic quality prior (issue #20) ────────────────────────────────
        # Correct maj/min/dom family flips in diatonic contexts when the acoustic
        # call is uncertain and a reliable local key pins the expected quality.
        if use_diatonic_prior:
            seg_len = e - s
            if seg_len < 8:                       # < ~2 bars: widen to ±4 bars
                c = (s + e) // 2
                lo, hi = max(0, c - 16), min(n_beats, c + 16)
            else:
                lo, hi = s, e
            loc_chroma = _reg_raw(onset_b[lo:hi].sum(0))
            kp = infer_key(loc_chroma)
            sev_h = apply_diatonic_prior(
                root, sev_h, conf, kp.tonic, kp.mode, kp.confidence,
                diatonic_boost=diatonic_boost, threshold_chromatic=threshold_chromatic,
            )

        t_start = float(bt[s])
        t_end   = float(bt[min(e, len(bt) - 1)])
        label = f"{NOTE[root]}:{sev_h}"
        labeled.append((t_start, t_end, fam_h, sev_h, conf, label))

    # ── 8b. Progression-encoder quality rerank (second pass, issue #21) ────────
    if use_progression_prior and labeled:
        try:
            sev_seq = [lab[3] for lab in labeled]
            conf_seq = [lab[4] for lab in labeled]
            new_sev = rerank_progression_qualities(
                list(seg_roots), sev_seq, conf_seq, weight=progression_weight,
            )
            for i, ns in enumerate(new_sev):
                if ns != labeled[i][3]:
                    t0, t1, fam_h, _old, conf, _lab = labeled[i]
                    labeled[i] = (t0, t1, fam_h, ns, conf,
                                  f"{NOTE[seg_roots[i]]}:{ns}")
        except Exception as exc:
            logger.warning("chord_pipeline_v1: progression rerank failed (%s)", exc)

    # ── 9. Coalesce adjacent same-label segments ──────────────────────────────
    coalesced: list[tuple[float, float, str, float]] = []
    for t0, t1, fam_h, sev_h, conf, label in labeled:
        if coalesced and coalesced[-1][2] == label:
            coalesced[-1] = (coalesced[-1][0], t1, label, max(coalesced[-1][3], conf))
        else:
            coalesced.append((t0, t1, label, conf))

    # ── 10. Global key ────────────────────────────────────────────────────────
    global_chroma = _reg_raw(onset_b.sum(0))
    key_result = infer_key(global_chroma)

    # ── 10b. Section structure (issue #22) ────────────────────────────────────
    # Symbolic chord-SSM + jazz form-length prior recovers 8/16-bar section
    # boundaries that the chord-level gmerge segmentation cannot (it cuts at
    # every chord change).  Runs on the classified per-beat chord sequence
    # (root relative to global tonic + quality index) so it is key-invariant.
    sections_out: list[dict] = []
    if beat_proba is not None and n_beats >= 32:
        from harmonia.models.section_structure import (
            build_chord_ssm,
            detect_section_boundaries,
            label_sections,
        )
        tonic_pc = _note_name_to_pc(key_result.key_name)
        qi: dict[str, int] = {}
        seq: list[tuple[int, int]] = [(-1, -1)] * n_beats
        for (s, e), root, lab in zip(segs, seg_roots, labeled):
            sev_h = lab[3]
            q = qi.setdefault(sev_h, len(qi))
            for b in range(s, min(e, n_beats)):
                seq[b] = ((root - tonic_pc) % 12, q)
        ssm = build_chord_ssm(seq)
        bnds = detect_section_boundaries(ssm, beats_per_bar=4)
        cut_beats = [0] + bnds + [n_beats]
        sec_labels = label_sections(ssm, cut_beats)
        for i in range(len(cut_beats) - 1):
            s_b, e_b = cut_beats[i], cut_beats[i + 1]
            sections_out.append({
                "start_s": round(float(bt[s_b]), 3),
                "end_s":   round(float(bt[min(e_b, len(bt) - 1)]), 3),
                "n_bars":  max(1, round((e_b - s_b) / 4)),
                "label":   sec_labels[i] if i < len(sec_labels) else "A",
            })
        logger.info("chord_pipeline_v1: %d sections", len(sections_out))

    # ── 11. Build ChordChart ──────────────────────────────────────────────────
    beat_dur_s = period
    chords_out = []
    segments_out = []
    for t0, t1, label, conf in coalesced:
        n_b = max(1, round((t1 - t0) / beat_dur_s))
        chords_out.append({
            "label":          label,
            "start_s":        round(t0, 3),
            "end_s":          round(t1, 3),
            "duration_beats": n_b,
            "confidence":     round(conf, 4),
        })
        segments_out.append({
            "start_s": round(t0, 3),
            "end_s":   round(t1, 3),
            "key":     key_result.key_name,
            "n_beats": n_b,
        })

    logger.info(
        "chord_pipeline_v1: %d chords, key=%s, tempo=%.1f BPM",
        len(chords_out), key_result.key_name, tempo_bpm,
    )

    return ChordChart(
        source_path=str(audio_path),
        duration_s=duration_s,
        tempo_bpm=round(tempo_bpm, 1),
        time_signature="4/4",
        global_key=key_result.key_name,
        global_key_confidence=round(key_result.confidence, 4),
        style="v1",
        modulations=[],
        chords=chords_out,
        segments=segments_out,
        sections=sections_out,
    )
