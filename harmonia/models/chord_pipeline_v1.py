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
from typing import TYPE_CHECKING, Literal

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
    return_post: bool = False,
) -> str | tuple[str, float | None]:
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
        return_post: also return the normalized 2-way posterior of the winning
            quality when the prior actually fired a flip, else ``None`` (the
            acoustic conf still describes an unchanged label).  Fix for the
            stale-confidence bug (audit 2026-07-13): a flipped label must not
            carry the pre-flip acoustic confidence.

    Returns:
        sev_h unchanged, or the canonical diatonic quality when the prior fires.
        With ``return_post=True``: ``(quality, posterior_or_None)``.

    Does NOT: touch roots; touch sus/aug qualities; fire on a chromatic root
    (degree outside the diatonic table) or when the acoustic quality is already
    diatonic — those are pass-through so real secondary dominants / borrowed
    chords the model sees clearly survive.
    """
    def _ret(sev: str, post: float | None):
        return (sev, post) if return_post else sev

    if key_conf < key_conf_min or conf >= threshold_chromatic:
        return _ret(sev_h, None)
    q5 = _SEV_TO_Q5.get(sev_h)
    if q5 is None:
        return _ret(sev_h, None)
    deg = (root - tonic) % 12
    ok = (_DIA_MAJOR_OK if mode == "major" else _DIA_MINOR_OK).get(deg)
    canon = (_DIA_MAJOR_CANON if mode == "major" else _DIA_MINOR_CANON).get(deg)
    if canon is None or ok is None:
        return _ret(sev_h, None)           # chromatic root — prior not applicable
    if q5 in ok:
        return _ret(sev_h, None)           # already diatonic — keep acoustic call
    # Non-diatonic call under a reliable key + uncertain acoustics.  Boost the
    # diatonic quality; flip only if it wins the (crude) 2-way log comparison.
    log_ac = math.log(max(conf, 1e-6))
    log_dia = math.log(max(1.0 - conf, 1e-6)) + math.log(max(diatonic_boost, 1e-6))
    if log_dia <= log_ac:
        return _ret(sev_h, None)
    flipped = _q5_to_sev(canon, _SEV_IS_SEVENTH.get(sev_h, False))
    if flipped is None:
        return _ret(sev_h, None)
    # Posterior of the decision actually made: the normalized 2-way winner.
    p_dia = math.exp(log_dia)
    post = p_dia / (p_dia + math.exp(log_ac))
    return _ret(flipped, post)


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

    def _proba_family_and_b7(self, root: int, seg_on: np.ndarray, seg_nt: np.ndarray,
                              seg_bs: np.ndarray, seg_tr: np.ndarray,
                              ) -> tuple[np.ndarray, np.ndarray, list[str]]:
        """Raw (p_fam, p7, b7_labels_aligned) posteriors for this segment.

        p_fam is (5,) in FAMILIES order. p7 is (k,) aligned column-for-column
        with b7_labels_aligned (b7_clf.classes_ decoded through base7_labels —
        NOT the same order as self.base7_labels itself). Shared by `predict`
        and by the ctx classifiers (which reuse this base's b7 head for the
        seventh-level split even when they override the family posterior).
        """
        ch_on = _reg_raw(seg_on); ch_nt = _reg_raw(seg_nt)
        rr = lambda c: np.roll(c, -root)
        f = _norm_blocks(np.hstack([rr(ch_on), rr(ch_nt), rr(seg_bs), rr(seg_tr)]))
        Xf = self.sc.transform(f[None])
        p_fam = self.clf.predict_proba(Xf)[0]
        p7 = self.b7_clf.predict_proba(Xf)[0]
        b7_labels_aligned = [self.base7_labels[int(c)] for c in self.b7_clf.classes_]
        return p_fam, p7, b7_labels_aligned

    def predict(self, root: int, seg_on: np.ndarray, seg_nt: np.ndarray,
                seg_bs: np.ndarray, seg_tr: np.ndarray,
                seventh_gate: float = 0.0,
                return_q5proba: bool = False,
                ) -> tuple[str, str, float] | tuple[str, str, float, np.ndarray]:
        """Return (family_harte, seventh_harte, confidence[, q5_logprobs]).

        seg_on/seg_nt: 88-dim summed piano-roll vectors.
        seg_bs/seg_tr: 12-dim register chroma (from _reg_raw).
        seventh_harte falls back to family_harte when confidence < seventh_gate.
        If return_q5proba, a 4th element is appended: a (5,) log-probability
        vector over q5 (maj/min/dom/hdim/dim), combining the family and
        seventh posteriors (see _family_q5_logprobs) — the real-probability
        replacement for the confidence-gated one-hot acoustic prior used by
        rerank_progression_qualities.
        """
        p, p7, b7_labels_aligned = self._proba_family_and_b7(
            root, seg_on, seg_nt, seg_bs, seg_tr
        )
        fam_idx = int(p.argmax())
        fam = FAMILIES[fam_idx]
        conf = float(p[fam_idx])
        fam_h = FAM_HARTE[fam]

        # seventh level
        if p7.max() >= seventh_gate:
            b7_key = b7_labels_aligned[int(p7.argmax())]
            sev_h = B7_HARTE.get(b7_key, fam_h)
        else:
            sev_h = fam_h

        if return_q5proba:
            q5_logp = _family_q5_logprobs(p, p7, b7_labels_aligned)
            return fam_h, sev_h, conf, q5_logp
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
                bsm_probs_abs: np.ndarray | None = None,
                return_q5proba: bool = False,
                ) -> tuple[str, str, float] | tuple[str, str, float, np.ndarray]:
        """Return (family_harte, seventh_harte, confidence[, q5_logprobs])
        using entropy gate. See _FamilyClassifier.predict for return_q5proba.
        """
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

        if return_q5proba:
            _, p7, b7_labels_aligned = self._base._proba_family_and_b7(
                root, seg_on, seg_nt, seg_bs, seg_tr
            )
            q5_logp = _family_q5_logprobs(p_mix, p7, b7_labels_aligned)
            return FAM_HARTE[fam], sev_h, conf, q5_logp
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
        # Volet 2 (#20/#23): a model trained with --local-key {v2,v3} carries a
        # 117d key-relative local-key block (9 window positions × 13 = degree
        # one-hot(12) + mode bit) APPENDED to the base 684d features → 801d.
        # ``_lk_dim`` = flat_dim − 684 tells predict() how large that trailing
        # block is (0 for a plain 684d model).  ``local_key_mode`` records which
        # teacher produced it (off/v2/v3) — informational.
        self._lk_dim = max(0, flat_dim - 684)
        try:
            self.local_key_mode = str(d["local_key_mode"])
        except (KeyError, ValueError):
            self.local_key_mode = "off"
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
                bsm_probs_abs: np.ndarray | None = None,
                return_q5proba: bool = False,
                lk_block: np.ndarray | None = None,
                ) -> tuple[str, str, float] | tuple[str, str, float, np.ndarray]:
        """Return (family_harte, seventh_harte, confidence[, q5_logprobs])
        using dual-head ctx v2. See _FamilyClassifier.predict for return_q5proba.

        ``lk_block`` (117d, volet 2): the key-relative local-key context block for
        this segment (9 window positions × 13d), appended when the model was
        trained with ``--local-key`` (``self._lk_dim > 0``).  It is the second
        pass of the two-pass inference scheme (#20/#23): pass 1 classifies
        quality with no local-key feature, the rule-based teacher reads a local
        key per chord off that predicted sequence, and pass 2 re-runs this
        predict() with the resulting ``lk_block``.  ``None`` → an all-zero block
        (neutral placeholder, e.g. pass 1 of a two-pass run or a mis-sized call).
        Ignored entirely for a 684d model (``self._lk_dim == 0``).
        """
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
        # append the 117d key-relative local-key block for an 801d model (volet 2)
        if self._lk_dim > 0:
            if lk_block is None:
                lk = np.zeros(self._lk_dim, dtype=np.float32)
            else:
                lk = np.asarray(lk_block, dtype=np.float32).ravel()
                if lk.shape[0] != self._lk_dim:   # defensive: pad/trim to model dim
                    fixed = np.zeros(self._lk_dim, dtype=np.float32)
                    fixed[:min(self._lk_dim, lk.shape[0])] = lk[:self._lk_dim]
                    lk = fixed
            X_ctx = np.concatenate([X_ctx, lk])
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

        if return_q5proba:
            _, p7, b7_labels_aligned = self._base._proba_family_and_b7(
                root, seg_on, seg_nt, seg_bs, seg_tr
            )
            q5_logp = _family_q5_logprobs(p_mix, p7, b7_labels_aligned)
            return FAM_HARTE[fam], sev_h, conf, q5_logp
        return FAM_HARTE[fam], sev_h, conf


# ── module-level lazy-loaded models ──────────────────────────────────────────

_family_clf: _FamilyClassifier | None = None
_ctx_clf: _CtxFamilyClassifier | _CtxFamilyClassifierV2 | None = None
_ctx_clf_v3: _CtxFamilyClassifierV2 | None = None
_ctx_clf_v3_loaded: bool = False
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


def _get_ctx_clf_v3() -> _CtxFamilyClassifierV2 | None:
    """Lazy-load the 801d key-relative ctx classifier (ctx_v3.npz) for two-pass
    inference (#20/#23, volet 2).  Distinct singleton from :func:`_get_ctx_clf`
    (which loads the 684d ctx_v2 used for pass 1) so the caller controls which
    variant runs — see ``ctx_classifier_variant`` in :func:`infer_chords_v1`.
    Returns None if ctx_v3.npz is absent or fails to load (caller falls back to
    the 684d pass-1 labels)."""
    global _ctx_clf_v3, _ctx_clf_v3_loaded
    if _ctx_clf_v3_loaded:
        return _ctx_clf_v3
    _ctx_clf_v3_loaded = True
    p = MODELS / "ctx_v3.npz"
    if not p.exists():
        logger.warning("chord_pipeline_v1: ctx_v3.npz missing — 801d two-pass unavailable")
        return None
    try:
        import torch  # noqa: F401
        _ctx_clf_v3 = _CtxFamilyClassifierV2(p, _get_family_clf())
        logger.info("chord_pipeline_v1: loaded ctx v3 model (%dd, local-key=%s)",
                    684 + _ctx_clf_v3._lk_dim, _ctx_clf_v3.local_key_mode)
    except Exception as e:
        logger.warning("chord_pipeline_v1: ctx_v3.npz load failed (%s)", e)
        _ctx_clf_v3 = None
    return _ctx_clf_v3


def _sev_to_localkey_token(root: int, sev_h: str) -> str:
    """Build an iReal-style chord token (e.g. ``"C7"``, ``"A-"``) from a predicted
    (root pc, Harte quality) so the rule-based local-key teacher
    (:func:`continuity_scale_track_v2`) can parse it in two-pass inference.

    The teacher only reads the token's functional class + chord tones
    (``quality_class`` / ``core_tones``), so the tail just has to route through
    those correctly; exact iReal spelling is not required."""
    tail = {
        "maj": "", "maj7": "^7", "6": "6",
        "min": "-", "min7": "-7", "minmaj7": "-^7", "min6": "-6",
        "7": "7", "9": "7", "13": "7",
        "hdim7": "h7", "dim": "o", "dim7": "o7",
        "aug": "+", "sus4": "sus", "sus2": "sus2", "7sus4": "7sus",
    }.get(sev_h, "")
    return f"{NOTE[int(root) % 12]}{tail}"


def _localkey_track_from_qualities_v2(
    roots: list[int], sev_hs: list[str], home_tonic: int, home_mode: str,
) -> list[tuple[int, int]]:
    """Per-chord ``(scale_degree, mode_bit)`` from the raw v2 continuity teacher,
    run over a predicted (root, quality) sequence — the pass-1 → local-key step of
    two-pass inference (#20/#23, volet 2).

    Mirrors ``scripts/train_ctx_model_v2._song_local_key_labels`` (``mode="v2"``)
    but sourced from *predicted* tokens rather than GT iReal tokens (the price of
    non-circularity: the teacher sees the noisy pass-1 quality, not clean GT).
    ``degree = (root - local_tonic) % 12``; ``mode_bit`` 0 major / 1 minor.
    Falls back to ``(root, 0)`` per chord on any teacher failure."""
    from harmonia.theory.local_key import continuity_scale_track_v2
    n = len(roots)
    if n == 0:
        return []
    tokens = [_sev_to_localkey_token(roots[i], sev_hs[i]) for i in range(n)]
    try:
        track = continuity_scale_track_v2(tokens, home_tonic=home_tonic,
                                          home_mode=home_mode)
    except Exception:
        return [(int(r) % 12, 0) for r in roots]
    out: list[tuple[int, int]] = []
    for i in range(n):
        sc = track[i]
        deg = (int(roots[i]) - int(sc["tonic"])) % 12
        out.append((int(deg), 0 if sc["mode"] == "major" else 1))
    return out


def _localkey_window_block(lk_pos: list[tuple[int, int]], i: int,
                           k: int = 4) -> np.ndarray:
    """Assemble the 117d key-relative local-key block for segment ``i`` from the
    per-chord ``(degree, mode_bit)`` list — a windowed one-hot exactly matching
    ``train_ctx_model_v2._localkey_ctx_onehots``: 9 window positions (−k..+k),
    each 13d = degree one-hot(12) ⊕ mode bit, carrying chord ``i+offset``'s own
    degree/mode (0 outside the sequence)."""
    LK_POS_DIM, LK_DEG_DIM = 13, 12
    W = 2 * k + 1
    out = np.zeros(W * LK_POS_DIM, dtype=np.float32)
    n = len(lk_pos)
    for j_idx, offset in enumerate(range(-k, k + 1)):
        ni = i + offset
        if 0 <= ni < n:
            base = j_idx * LK_POS_DIM
            deg, mbit = lk_pos[ni]
            out[base + int(deg) % 12] = 1.0
            out[base + LK_DEG_DIM] = float(mbit)
    return out


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


_jazz_dur_prior: dict | None = None


def _get_jazz_duration_prior() -> dict:
    """Load the jazz1460 symbolic chord-duration prior for the semi-Markov decode
    ({"pooled": (D,), "per_q5": (5, D)}); build via scripts/build_duration_prior_jazz.py."""
    global _jazz_dur_prior
    if _jazz_dur_prior is not None:
        return _jazz_dur_prior
    p = REPO / "data" / "cache" / "duration_prior_jazz1460.npz"
    if not p.exists():
        raise FileNotFoundError(
            f"{p} missing — run scripts/build_duration_prior_jazz.py first")
    d = np.load(p)
    _jazz_dur_prior = {"pooled": d["pooled"], "per_q5": d["per_q5"]}
    return _jazz_dur_prior


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

# base7 labels that resolve to each side of the two q5-relevant splits.
# major family (FAMILIES idx 0) splits into q5 "maj" vs "dom" (dominant 7th);
# diminished family (FAMILIES idx 2) splits into q5 "dim" vs "hdim". minor,
# augmented and suspended families map to a single q5 class each (min / maj /
# maj respectively — see _HARTE_TO_Q5NAME), so no split is needed for them.
_MAJOR_TO_MAJ_B7 = {"majT", "maj7"}
_MAJOR_TO_DOM_B7 = {"dom7"}
_DIM_TO_DIM_B7 = {"dimT", "dim7"}
_DIM_TO_HDIM_B7 = {"m7b5"}


def _family_q5_logprobs(p_fam: np.ndarray, p7: np.ndarray,
                         b7_labels_aligned: list[str]) -> np.ndarray:
    """Combine a 5-class family posterior with the base7 (seventh) posterior
    into a real 5-class q5 (maj/min/dom/hdim/dim) log-probability vector.

    `p_fam` is in FAMILIES order (major, minor, diminished, augmented,
    suspended) — the convention used throughout this module (fam_idx =
    p_fam.argmax(); FAMILIES[fam_idx]).  `p7`/`b7_labels_aligned` are the
    b7_clf.predict_proba(...) output and its column->label alignment (see
    _FamilyClassifier._proba_family_and_b7).

    Only the major and diminished branches need splitting (maj-vs-dom,
    dim-vs-hdim); the split fraction is the b7 posterior's *relative* mass
    between the two branches' labels, renormalized within that branch (the
    family decision itself already comes from p_fam, not from p7).  This
    replaces the previous confidence-gated one-hot placed on the greedy q5
    class, which pinned the acoustic prior near-degenerate whenever
    conf > ~0.65 and starved the ProgressionEncoder of any real evidence to
    argue against.
    """
    eps = 1e-9

    def _mass(labels: set[str]) -> float:
        idx = [i for i, lab in enumerate(b7_labels_aligned) if lab in labels]
        return float(p7[idx].sum()) if idx else 0.0

    maj_mass = _mass(_MAJOR_TO_MAJ_B7)
    dom_mass = _mass(_MAJOR_TO_DOM_B7)
    tot_major_b7 = maj_mass + dom_mass
    dom_frac = dom_mass / tot_major_b7 if tot_major_b7 > eps else 0.0

    dim_mass = _mass(_DIM_TO_DIM_B7)
    hdim_mass = _mass(_DIM_TO_HDIM_B7)
    tot_dim_b7 = dim_mass + hdim_mass
    hdim_frac = hdim_mass / tot_dim_b7 if tot_dim_b7 > eps else 0.0

    p_major, p_minor, p_dim, p_aug, p_sus = (float(x) for x in p_fam)
    q = np.array([
        p_major * (1.0 - dom_frac) + p_aug + p_sus,  # maj
        p_minor,                                      # min
        p_major * dom_frac,                           # dom
        p_dim * hdim_frac,                             # hdim
        p_dim * (1.0 - hdim_frac),                      # dim
    ], dtype=np.float64)
    q = np.clip(q, eps, None)
    q = q / q.sum()
    return np.log(q).astype(np.float32)


_Q5_NAMES = ("maj", "min", "dom", "hdim", "dim")


def _top_chord_suggestions(
    p_root: np.ndarray, q5_logp: np.ndarray, k: int = 5,
) -> list[dict]:
    """Joint root x q5-quality candidate list, sorted by probability.

    `p_root` (12,) and `q5_logp` (5,) are the two posteriors already computed
    per segment (beat-sequence root model, family/seventh classifier) and
    otherwise discarded after argmax — this just keeps the top-k instead of
    only the winner, under an independence assumption (joint = root_prob *
    q5_prob) that is a real approximation but a reasonable one: the two
    models are trained on different features (root model: pitch-class
    activations; q5 classifier: family/seventh heads) and not jointly
    calibrated, so no true joint distribution exists to draw from.
    """
    root_sum = float(p_root.sum())
    p_root_n = p_root / root_sum if root_sum > 1e-9 else np.full(12, 1 / 12)
    q5_p = np.exp(q5_logp)
    q5_p = q5_p / q5_p.sum()

    top_roots = np.argsort(p_root_n)[::-1][:3]
    top_q5 = np.argsort(q5_p)[::-1][:3]
    cands = []
    for r in top_roots:
        for qi in top_q5:
            cands.append({
                "root": int(r), "q5": _Q5_NAMES[qi],
                "prob": float(p_root_n[r] * q5_p[qi]),
            })
    cands.sort(key=lambda c: -c["prob"])
    total = sum(c["prob"] for c in cands[:k]) or 1.0
    out = cands[:k]
    for c in out:
        c["prob"] = round(c["prob"] / total, 4)
    return out


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


# ── local-key sequence model (LocalKeySeqGRU v3) ──────────────────────────────
# Transpose-equivariant per-chord local-key tagger, distilled from the rule-based
# heuristic the user chose (theory.local_key.continuity_scale_track_v2) with the
# v3 secondary-dominant-chain consolidation. Wired here as a *diatonic-prior
# reranker* (issue #20/#23): after the first acoustic quality pass, it labels a
# local key per chord over the whole song, and apply_diatonic_prior uses that key
# to correct maj/min/dom family flips in diatonic contexts. Historically the
# diatonic prior was opt-in and net-neutral because the *inferred* local key
# (infer_key over a chroma window) was not reliable enough; this model is the
# reliability upgrade that motivated re-enabling the prior.
CONF_CALIBRATION_PATH = REPO / "data" / "cache" / "confidence_calibration.npz"
CONF_CALIBRATION_REAL_PATH = REPO / "data" / "cache" / "confidence_calibration_real.npz"

# Cache: {"synth": callable_or_None, "real": callable_or_None}
_CONF_CAL_CACHE: dict[str, object] = {}


def _get_conf_calibrator(audio_domain: str = "synth"):
    """Lazy-load the isotonic confidence-calibration map for ``audio_domain``.

    Two domain-specific maps exist (Mission 4, issue #19/#26 — real-audio
    calibration):

    - ``synth`` → ``confidence_calibration.npz``: breakpoints ``x`` (raw FUSED
      score = family/rerank conf × span root posterior) → ``y`` (empirical
      P(root+q5-family correct)) fitted by scripts/fit_confidence_calibration.py
      on held-out jazz1460 MMA renders (test ECE 0.039). Fed the fused raw.

    - ``real`` → ``confidence_calibration_real.npz``: fitted on the yt_corpus_50
      real-audio segments (iReal GT, DTW-aligned; scripts/fit_confidence_calibration_real.py)
      mapping the QUALITY head's raw confidence (confidence_raw, NOT the fused
      score) → real-audio P(q5 correct). Fed ``conf`` (confidence_raw), not the
      fused raw, because it was fit on that score. On real recordings the synth
      map is badly miscalibrated (measured ECE 0.465, and it *amplifies*
      overconfidence to 0.533) because real-audio confidence is near
      non-discriminative — even conf≈0.98 → ~48% correct. The real map collapses
      displayed confidence toward the measured base rate (~0.44); 5-fold
      song-held-out CV ECE 0.007. **Caveat:** fit on a proxy score (baseline LR
      _FamilyClassifier on cached feat48), not the production ctx/joint
      confidence_raw, and root_conf is not folded in; it is robust to that
      mismatch only because it is nearly flat. See docs/known_issues.md #19/#26.

    Returns a callable raw→calibrated, or None when the file is absent (the
    pipeline then reports the raw conf). Display-layer only: labels and every
    internal gate are computed before this map is ever applied.

    The returned callable carries a ``score_kind`` attribute (``"fused"`` or
    ``"conf"``) declaring which raw score it was fitted on, so the pipeline feeds
    it the matching input (issue #29). The kind is read from the saved map's
    ``score_kind`` field; absent that (legacy maps) it defaults per domain —
    ``synth``→``fused`` (already fit on conf×root_conf), ``real``→``conf`` (the
    old root-blind fit). Mission 3's refit saves ``score_kind="fused"`` so the
    real path also folds in root uncertainty.
    """
    if audio_domain in _CONF_CAL_CACHE:
        return _CONF_CAL_CACHE[audio_domain]
    path = CONF_CALIBRATION_REAL_PATH if audio_domain == "real" else CONF_CALIBRATION_PATH
    cal = None
    if not path.exists():
        logger.info("chord_pipeline_v1: %s missing — raw confidence", path.name)
    else:
        try:
            d = np.load(path, allow_pickle=True)
            x, y = d["x"].astype(float), d["y"].astype(float)
            cal = lambda s: float(np.interp(s, x, y))  # noqa: E731
            default_kind = "conf" if audio_domain == "real" else "fused"
            cal.score_kind = (str(d["score_kind"]) if "score_kind" in d.files
                              else default_kind)
            logger.info("chord_pipeline_v1: loaded %s calibration (%d breakpoints, "
                        "score=%s)", audio_domain, len(x), cal.score_kind)
        except Exception as exc:
            logger.warning("chord_pipeline_v1: %s calibration load failed (%s)",
                           audio_domain, exc)
    _CONF_CAL_CACHE[audio_domain] = cal
    return cal


_NOTE_TO_PC: dict[str, int] | None = None


def _span_root_conf(beat_proba: np.ndarray | None, bt: np.ndarray,
                    t0: float, t1: float, label: str) -> float | None:
    """Mean per-beat root posterior over [t0, t1) at the label's root pc.

    The root-side half of the fused display confidence (audit step 1b): the
    family/ctx conf never sees the root at all, so a confidently-wrong root
    used to surface as a confident chord. None when the root model is off,
    the label has no root, or the span covers no beats.
    """
    global _NOTE_TO_PC
    if beat_proba is None or ":" not in label:
        return None
    if _NOTE_TO_PC is None:
        _NOTE_TO_PC = {n: i for i, n in enumerate(NOTE)}
    pc = _NOTE_TO_PC.get(label.split(":", 1)[0])
    if pc is None:
        return None
    s = int(np.searchsorted(bt, t0, side="left"))
    e = int(np.searchsorted(bt, t1, side="left"))
    e = min(max(e, s + 1), len(beat_proba))
    if s >= len(beat_proba):
        return None
    return float(beat_proba[s:e, pc].mean())


_LOCAL_KEY_SEQ_MODEL = None
_LOCAL_KEY_SEQ_LOADED = False
LOCAL_KEY_SEQ_PATH = REPO / "data" / "cache" / "local_key_seq_gru.pt"


def _get_local_key_seq_model():
    """Lazy-load the LocalKeySeqGRU checkpoint (None if unavailable)."""
    global _LOCAL_KEY_SEQ_MODEL, _LOCAL_KEY_SEQ_LOADED
    if _LOCAL_KEY_SEQ_LOADED:
        return _LOCAL_KEY_SEQ_MODEL
    _LOCAL_KEY_SEQ_LOADED = True
    if not LOCAL_KEY_SEQ_PATH.exists():
        logger.warning("chord_pipeline_v1: local_key_seq_gru.pt missing — local-key prior off")
        return None
    try:
        from harmonia.models.local_key_seq_model import load_seq_model
        _LOCAL_KEY_SEQ_MODEL = load_seq_model(LOCAL_KEY_SEQ_PATH, device="cpu")
        logger.info("chord_pipeline_v1: loaded local_key_seq_gru (diatonic-prior reranker)")
    except Exception as exc:
        logger.warning("chord_pipeline_v1: local_key_seq_gru load failed (%s)", exc)
        _LOCAL_KEY_SEQ_MODEL = None
    return _LOCAL_KEY_SEQ_MODEL


def local_key_track_from_qualities(
    roots: list[int], sev_hs: list[str], global_tonic: int, *, model=None,
) -> list[tuple[int, str, float]] | None:
    """Per-chord local key ``(tonic, mode, confidence)`` for a classified sequence.

    Runs LocalKeySeqGRU over the ``(root, predicted-quality)`` sequence of a whole
    song and returns one absolute local key per position. Roots are encoded
    *relative to the song's global tonic* (the model is transpose-equivariant by
    construction — see local_key_seq_data), so ``global_tonic`` must be the real
    inferred global tonic, not an arbitrary origin. Positions whose Harte quality
    has no q5 mapping (sus/aug) are fed as ``maj`` for the tagger's benefit only;
    the diatonic prior never fires on them anyway.

    ``confidence`` is the per-position softmax max — a proxy for how sure the
    tagger is about the local key at that chord, used as ``key_conf`` by
    apply_diatonic_prior.
    """
    model = model if model is not None else _get_local_key_seq_model()
    if model is None or not roots:
        return None
    import torch

    from harmonia.models.local_key_seq_data import rel_features, rel_to_abs_key
    from harmonia.models.local_key_seq_model import collate

    seq_rel: list[tuple[int, int]] = []
    for r, sev in zip(roots, sev_hs):
        q5 = _harte_to_q5idx(sev)
        seq_rel.append(((int(r) - global_tonic) % 12, q5 if q5 is not None else 0))

    intervals, dom_prep = rel_features(seq_rel)
    item = {"seq": seq_rel, "intervals": intervals, "dom_prep": dom_prep,
            "y": [0] * len(seq_rel)}
    root_t, qual_t, interval_t, dp_t, lengths_t, _ = collate([item], "cpu")
    with torch.no_grad():
        logits = model(root_t, qual_t, lengths_t, interval_t, dp_t)[0]  # (T,24)
        probs = torch.softmax(logits, dim=-1)
        conf_t, idx_t = probs.max(-1)

    out: list[tuple[int, str, float]] = []
    for i in range(len(seq_rel)):
        abs_idx = rel_to_abs_key(int(idx_t[i]), global_tonic)
        out.append((abs_idx % 12, "major" if abs_idx < 12 else "minor",
                    float(conf_t[i])))
    return out


def rerank_local_key_qualities(
    roots: list[int], sev_hs: list[str], confs: list[float],
    global_tonic: int, *,
    boost: float = 4.0, threshold_chromatic: float = 0.80,
    key_conf_min: float = 0.30, model=None,
    return_post: bool = False,
) -> list[str] | tuple[list[str], list[float | None]]:
    """Second-pass diatonic-prior quality rerank driven by LocalKeySeqGRU (#20/#23).

    Mirrors :func:`rerank_progression_qualities`: a *whole-sequence* second pass
    that refines each segment's coarse quality using context the per-segment
    first pass cannot see. Here the context is a learned local-key label per
    chord; :func:`apply_diatonic_prior` snaps a non-diatonic, acoustically
    *uncertain* maj/min/dom/dim call to the diatonic quality of that key.

    Args mirror the diatonic-prior gate: ``boost`` is ``diatonic_boost``,
    ``threshold_chromatic`` the acoustic-confidence gate (skip the prior when the
    acoustic call is confident), ``key_conf_min`` the minimum tagger confidence.
    Returns a new Harte-quality list; unchanged where the prior does not fire.
    With ``return_post=True`` also returns a per-position posterior list:
    the 2-way winner probability where the prior flipped the label, ``None``
    elsewhere (stale-confidence fix, audit 2026-07-13).
    """
    out = list(sev_hs)
    posts: list[float | None] = [None] * len(sev_hs)
    keys = local_key_track_from_qualities(roots, sev_hs, global_tonic, model=model)
    if keys is None:
        return (out, posts) if return_post else out
    for i, (root, sev) in enumerate(zip(roots, sev_hs)):
        tonic, mode, kconf = keys[i]
        out[i], posts[i] = apply_diatonic_prior(
            int(root), sev, float(confs[i]), tonic, mode, kconf,
            diatonic_boost=boost, threshold_chromatic=threshold_chromatic,
            key_conf_min=key_conf_min, return_post=True,
        )
    return (out, posts) if return_post else out


def _progression_fusion_bonus_fn(
    roots: list[int], q5: list[int | None], confs: list[float],
    *, weight: float, encoder=None, subtract_prior: bool = False,
):
    """Closure ``(seg_idx, cand_root) -> (5,) λ·log P_enc`` for H2 shallow fusion.

    Builds the ProgressionEncoder's per-quality log-prob for segment ``i`` given
    its harmonic neighbourhood (the pass-1 decoded ``(root, q5)`` sequence), with
    the CENTRE MASKED so the encoder scores a pure grammar conditional
    ``P(q_i | context)`` rather than echoing its own greedy call. Root-dependent:
    the context intervals ``(root_k − cand_root) % 12`` shift with the candidate
    centre root, so a top-2/3 root gets its own grammar score — the reason this
    enters the joint decode per candidate rather than as a post-hoc rerank.
    Returns ``weight * log_softmax(logits)`` (a (5,) vector), memoised per
    ``(i, root)``. ``None`` when the encoder is unavailable → decode unchanged.
    """
    encoder = encoder if encoder is not None else _get_progression_encoder()
    if encoder is None or not roots:
        return None
    import torch

    from harmonia.models.progression_encoder import CTX, MASK_ID, WINDOW
    N = len(roots)
    cache: dict[tuple[int, int], np.ndarray] = {}

    # H3: the encoder's OWN marginal log P(q) — its prediction under an all-masked
    # (empty) context. Subtracting it turns the fusion term from log P_enc(q|ctx)
    # into the log-likelihood-RATIO log[P_enc(q|ctx)/P_enc(q)], the standard
    # density-ratio / internal-LM-subtraction remedy for the label-bias the raw
    # LM carries (Korzeniowski: keep the label prior uniform). Removes the
    # majority-major base rate the acoustic emission already encodes.
    prior_lp = np.zeros(5, dtype=np.float64)
    if subtract_prior:
        R0 = np.zeros(WINDOW, dtype=np.int64)
        Q0 = np.full(WINDOW, MASK_ID, dtype=np.int64)
        C0 = np.zeros(WINDOW, dtype=np.float32)
        Pm0 = np.ones(WINDOW, dtype=bool)
        Pm0[CTX] = False  # centre present but masked; all neighbours padded
        with torch.no_grad():
            lg = encoder(torch.from_numpy(R0[None]), torch.from_numpy(Q0[None]),
                         torch.from_numpy(C0[None]), torch.from_numpy(Pm0[None]))
            prior_lp = torch.log_softmax(lg, dim=-1).numpy()[0].astype(np.float64)

    def bonus(i: int, root: int) -> np.ndarray:
        key = (i, int(root))
        hit = cache.get(key)
        if hit is not None:
            return hit
        R = np.zeros(WINDOW, dtype=np.int64)
        Q = np.full(WINDOW, MASK_ID, dtype=np.int64)
        C = np.zeros(WINDOW, dtype=np.float32)
        Pm = np.ones(WINDOW, dtype=bool)  # True = padding
        for jj in range(WINDOW):
            k = i + jj - CTX
            if k == i:
                Pm[jj] = False           # centre present but MASKED (Q=MASK,C=0)
                continue
            if 0 <= k < N and q5[k] is not None:
                R[jj] = (int(roots[k]) - int(root)) % 12
                Q[jj] = int(q5[k])
                C[jj] = float(confs[k])
                Pm[jj] = False
        with torch.no_grad():
            logits = encoder(
                torch.from_numpy(R[None]), torch.from_numpy(Q[None]),
                torch.from_numpy(C[None]), torch.from_numpy(Pm[None]),
            )
            lp = torch.log_softmax(logits, dim=-1).numpy()[0]  # (5,)
        out = (weight * (lp.astype(np.float64) - prior_lp))
        cache[key] = out
        return out

    return bonus


def rerank_progression_qualities(
    roots: list[int], sev_hs: list[str], confs: list[float],
    *, weight: float = 0.5, encoder=None,
    aco_logprobs: list[np.ndarray] | None = None,
    return_post: bool = False,
) -> list[str] | tuple[list[str], list[float | None]]:
    """Second-pass ProgressionEncoder quality rerank over a chord sequence.

    Args:
        roots:   per-segment root pitch-class (0-11).
        sev_hs:  per-segment fine Harte quality (e.g. "maj7", "7", "min").
        confs:   per-segment acoustic confidence (family/ctx max-prob), in [0,1].
        weight:  progression_weight in log_post = log_acoustic + w·log_encoder.
        encoder: preloaded ProgressionEncoder (defaults to the module singleton).
        aco_logprobs: optional per-segment (5,) real q5 log-probability vector
            (from _FamilyClassifier/_CtxFamilyClassifier*.predict(...,
            return_q5proba=True), combining the family and seventh posteriors
            — see _family_q5_logprobs). When given, used directly as the
            acoustic prior instead of the confidence-gated one-hot fallback
            (issue #21: the one-hot pins the prior near-degenerate whenever
            conf > ~0.65, starving the encoder of any real evidence to argue
            against). Falls back to the one-hot when None, or per-segment when
            an entry is None (e.g. no ctx model loaded for that call site).

    Returns:
        A new list of Harte qualities; unchanged where the encoder agrees with
        the acoustic call or the segment quality is outside the 5-class vocab.
        Preserves the acoustic triad-vs-seventh choice when the family flips.
        With ``return_post=True`` also returns a per-position posterior list:
        ``softmax(log_acoustic + w·log_encoder)[chosen_q5]`` — the normalized
        value of the actual decision variable — where the rerank flipped the
        label, ``None`` elsewhere (stale-confidence fix, audit 2026-07-13).
    """
    out = list(sev_hs)
    posts: list[float | None] = [None] * len(sev_hs)
    encoder = encoder if encoder is not None else _get_progression_encoder()
    N = len(roots)
    if encoder is None or N == 0:
        return (out, posts) if return_post else out

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
        aco_lp = aco_logprobs[i] if aco_logprobs is not None else None
        if aco_lp is not None:
            # real per-q5 log-probs from the family/seventh classifier heads
            aco = np.asarray(aco_lp, dtype=np.float32)
        else:
            # fallback: confidence-gated one-hot on the greedy q5
            c = float(min(max(confs[i], 1e-3), 1.0 - 1e-3))
            aco = np.full(5, np.log((1.0 - c) / 4.0), dtype=np.float32)
            aco[q5[i]] = np.log(c)
        combined = aco + weight * enc_logp[i]
        new_q5 = int(combined.argmax())
        if new_q5 != q5[i]:
            triad, seventh = _Q5IDX_TO_HARTE[new_q5]
            out[i] = seventh if sev_hs[i] in _SEVENTH_HARTE else triad
            # Posterior of the decision actually made: softmax over the same
            # combined score that produced the argmax.
            z = combined - combined.max()
            p = np.exp(z)
            posts[i] = float(p[new_q5] / p.sum())
    return (out, posts) if return_post else out


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


# ── Mission 5: LLM-prior glue (Part A) ────────────────────────────────────────
# The LLM/offline analyst (scripts/llm_chord_priors.py) emits an analysis JSON;
# ``to_bayesian_factors`` turns it into the decoder factor object. This glue maps
# that object onto the four ``joint_decode`` seams (tonic, q5_bonus, pool_groups;
# transition bias intentionally OFF — the bigram slot is saturated, #27). Default
# OFF behind ``use_llm_priors`` — see docs/mission_5_bayesian_integration.md.

# Gate on the analyst's self-reported key confidence: below this we keep the
# audio-inferred tonic (a low-confidence analyst must not override a key the
# acoustic front-end may have gotten right).
LLM_KEY_TRUST = 0.60


def bars_to_segment_groups(
    pool_group_bars: "list[list[tuple[int, int]]]",
    segs: "list[tuple[int, int]]",
    beat_times: np.ndarray,
    *,
    beats_per_bar: int = 4,
) -> list[list[int]]:
    """Map analyst repeat/parallel bar-spans → tied ``joint_decode`` segment groups.

    ``pool_group_bars`` (from ``BayesianFactors.pool_group_bars``) is a list of
    GROUPS; each group is a list of equal-length spans ``(bar_start, bar_end)``
    (1-indexed, inclusive) the analyst asserts are the SAME material. Pooling in
    ``joint_decode`` ties a set of segment indices to ONE decoded label and sums
    their emission (√N denoising, #28). Corresponding *slots* across parallel
    spans are what should tie — bar 1↔9, 2↔10, … for a span pair (1-8, 9-16) —
    NOT the whole 8-bar strain collapsed to one chord. So we emit one tie-group
    per within-span slot.

    Segment boundaries are audio-beat indices into ``beat_times``; bars are
    symbolic. Without downbeat ground truth we assume a fixed ``beats_per_bar``
    (4/4 default) and locate each bar's covering segment by its centre beat. A
    bar with no covering segment (audio shorter than the chart, off-grid) is
    skipped; a slot that ends up tying fewer than two distinct segments is
    dropped.

    Returns a list of segment-index tie groups suitable for
    ``joint_decode(pool_groups=...)``. NOTE what this does NOT solve (CLAUDE.md
    #4): it does not recover bar↔beat *phase* — it assumes bar 1 starts at beat
    0. A pickup bar or a mis-phased beat grid will misalign the slots; the tie is
    only as good as the fixed-grid assumption.
    """
    n_seg = len(segs)

    def _seg_of_bar(bar: int) -> int | None:
        # 1-indexed bar → covering segment by centre-beat containment.
        centre = (bar - 1) * beats_per_bar + beats_per_bar / 2.0
        for i, (s, e) in enumerate(segs):
            if s <= centre < e:
                return i
        # fallback: max beat-overlap
        lo, hi = (bar - 1) * beats_per_bar, bar * beats_per_bar
        best, best_ov = None, 0.0
        for i, (s, e) in enumerate(segs):
            ov = max(0.0, min(e, hi) - max(s, lo))
            if ov > best_ov:
                best, best_ov = i, ov
        return best

    groups: list[list[int]] = []
    for span_group in pool_group_bars:
        spans = [(int(s), int(e)) for (s, e) in span_group if int(e) >= int(s)]
        if len(spans) < 2:
            continue
        slot_len = min(e - s + 1 for (s, e) in spans)
        for k in range(slot_len):
            tied: list[int] = []
            for (s, _e) in spans:
                si = _seg_of_bar(s + k)
                if si is not None and 0 <= si < n_seg:
                    tied.append(si)
            tied = sorted(set(tied))
            if len(tied) > 1:
                groups.append(tied)
    return groups


def apply_llm_priors(
    analysis: dict,
    segs: "list[tuple[int, int]]",
    beat_times: np.ndarray,
    *,
    inferred_tonic: int,
    max_nats: float = 8.0,
    beats_per_bar: int = 4,
) -> dict:
    """Convert an analyst analysis JSON → Bayesian factors via the 4 seams.

    Args:
        analysis: dict from ``llm_chord_priors.offline_analyze`` (or the LLM path).
        segs: segment (beat-index) ranges from the pipeline's segmentation.
        beat_times: pipeline beat grid, for the bar↔segment mapping.
        inferred_tonic: audio-inferred tonic pc, used when the analyst's key
            confidence is below :data:`LLM_KEY_TRUST`.
        max_nats: ceiling on prior strength (default 8, ~5× weaker than a user
            confirm's ~40; scaled further by analyst confidence in
            ``to_bayesian_factors``).

    Returns:
        dict with keys: ``tonic`` (int), ``q5_bonus`` (callback for
        ``joint_decode``), ``pool_groups`` (segment-index tie groups),
        ``factors`` (the underlying :class:`BayesianFactors`, for logging).
    """
    from scripts.llm_chord_priors import to_bayesian_factors

    f = to_bayesian_factors(analysis, max_nats=max_nats)

    # Seam 1: tonic (KEY_TRUST gate).
    tonic = f.tonic if f.confidence >= LLM_KEY_TRUST else inferred_tonic

    # Seam 2: q5_bonus callback. seg_idx is unused in the v1 (position-agnostic)
    # marginal prior — it is the hook Part C's section-conditional prior fills.
    def q5_bonus(seg_idx: int, root: int) -> np.ndarray:
        row = np.zeros(5)
        for q5, nats in f.quality_bonus.get(root, {}).items():
            if 0 <= q5 < 5:
                row[q5] = float(nats)
        return row

    # Seam 3: pool_groups (repeat spans → tied segment indices).
    pool_groups = bars_to_segment_groups(
        f.pool_group_bars, segs, beat_times, beats_per_bar=beats_per_bar
    )

    # Seam 4: transition bias OFF (saturated, #27).

    return dict(tonic=tonic, q5_bonus=q5_bonus, pool_groups=pool_groups, factors=f)


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
    ctx_classifier_variant: Literal["684d", "801d_two_pass"] = "684d",
    use_harmonic_grid: bool = True,
    use_bass_tracking: bool = False,
    use_diatonic_prior: bool = False,
    diatonic_boost: float = 4.0,
    threshold_chromatic: float = 0.80,
    use_progression_prior: bool = False,
    progression_weight: float = 2.0,
    use_local_key_prior: bool = False,
    local_key_weight: float = 4.0,
    local_key_threshold_chromatic: float = 0.80,
    use_phase_correction: bool = True,
    use_joint_decode: bool = True,
    joint_K: int = 3,
    joint_transition_weight: float = 0.0,
    joint_local_key_transition: bool = False,
    joint_progression_fusion: bool = False,
    joint_progression_weight: float = 0.5,
    joint_fusion_iters: int = 1,
    joint_fusion_subtract_prior: bool = False,
    use_semi_markov: bool = True,
    semi_markov_dur_weight: float = 0.25,
    semi_markov_qual_weight: float = 0.0,
    semi_markov_per_quality_dur: bool = False,
    user_constraints: dict | None = None,
    audio_domain: Literal["synth", "real"] = "real",
    use_llm_priors: bool = False,
    llm_analysis: dict | None = None,
    llm_song: str | None = None,
    llm_playlist: "Path | None" = None,
    llm_max_nats: float = 8.0,
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
        ctx_classifier_variant: Which ctx family classifier to run (#20/#23, volet 2).
                         "684d" (default) = the current ctx_v2 model, single pass.
                         "801d_two_pass" = the key-relative ctx_v3 model with the
                         117d local-key block, run as a *two-pass* scheme: pass 1
                         is the 684d ctx_v2 classifier over the whole song →
                         predicted (root, quality) sequence; the raw-v2 continuity
                         teacher reads a local key per chord off THAT (noisy,
                         non-circular) sequence; pass 2 re-runs the 801d model per
                         segment with the resulting local-key block, refining
                         quality. Opt-in; falls back to the pass-1 labels if
                         ctx_v3.npz is unavailable.
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
                         learned transformer over the classified sequence.
                         DEFAULT OFF since 2026-07-13 (issue #25): the +1.0pp
                         that justified default-ON came from the bypass harness
                         (eval_irealb_e2e.py, no ctx model); on the real path
                         it measures −3.6pp majmin (jazz1460 held-out, n=25)
                         and never fires on POP909. The encoder's information
                         belongs in the joint decode (audit step 2), not a
                         greedy post-hoc override.
        progression_weight: Encoder weight in the log-posterior combination
                         (log_acoustic + w·log_encoder); 0 = acoustic only.
                         Default 2.0, calibrated for the real per-q5 acoustic
                         prior (see rerank_progression_qualities); irealb e2e
                         sweep at {0.2,0.5,1.0,2.0} peaked at 2.0 (85.0% majmin,
                         59.0% 7ths) vs the old one-hot-gated prior's 84.7%/58.9%
                         at w=0.5 and the no-encoder baseline's 84.0%/58.6%.
        use_local_key_prior: Second-pass LocalKeySeqGRU diatonic-prior rerank
                         (issue #20/#23).  **Default OFF.**  Runs the
                         transpose-equivariant per-chord local-key tagger over
                         the first-pass (root, quality) sequence, then snaps
                         non-diatonic, acoustically-uncertain maj/min/dom/dim
                         calls to the local key's diatonic quality — the
                         reliability upgrade over the old chroma-window
                         infer_key() diatonic prior (use_diatonic_prior), which
                         was net-neutral because the inferred key was too noisy.
        local_key_weight: Diatonic boost for the local-key prior (== diatonic_boost
                         forwarded to apply_diatonic_prior); higher = the prior
                         wins more family flips (4.0 default).
        local_key_threshold_chromatic: Acoustic-confidence gate for the local-key
                         prior (0.80); at/above it the acoustic call is trusted.
        use_joint_decode: Segment-level JOINT (root x quality) Viterbi decode
                         (audit step 2, harmonia/models/joint_decode.py): top-K
                         candidate roots x 5 qualities per segment, coupled by
                         the scale-relative progression bigram as a transition
                         factor; subsumes (and disables) the two-pass/local-key/
                         progression rerankers. Segmentation is unchanged -- the
                         decode only relabels. Default OFF.
        joint_K:         Candidate roots per segment for the joint decode (3;
                         GT-root top-3 coverage on real segments is 99.3%).
        joint_transition_weight: Weight on the progression-bigram transition
                         factor. Default 0.0 (emission-only) -- the fit-split
                         sweep (idx 20..30, w in {0,.1,.25,.5,1,2}) found ANY
                         positive weight of the corpus bigram net-negative on
                         jazz (it snaps min/hdim/dim toward the majority-major
                         prior; jazz is ~49% diatonic). The joint gain comes
                         from the root x quality emission coupling itself.
        joint_local_key_transition: (H1, #27) re-reference the bigram transition
                         to a per-chord LOCAL key. **Default OFF, dead end** — the
                         local tonic churns on 46% of adjacent pairs, so a
                         globally-fit bigram under a shifting reference is strictly
                         worse than global. Kept for the record / re-fit attempts.
        joint_progression_fusion / joint_progression_weight / joint_fusion_iters
                         / joint_fusion_subtract_prior: (H2/H3, #27) ASR-style
                         shallow fusion of the ProgressionEncoder as a per-cand-root
                         EMISSION factor (centre masked); subtract_prior turns it
                         into a density-ratio log[P(q|ctx)/P(q)]. **Default OFF,
                         dead end** — both carry/over-correct the label-bias and are
                         net-negative on jazz majmin (optimum λ→0); residual errors
                         are acoustic, not grammatical (see #27 Mission 1).
        use_semi_markov: Per-beat semi-Markov (explicit-duration) decode (#27
                         Mission 2, harmonia/models/semi_markov_decode.py). When
                         ON (**default**), the segmentation is DISCARDED and an
                         explicit-duration Viterbi over (root×q5) with a
                         jazz1460-fit duration prior (~0 mass on 1/3-beat chords)
                         decides the boundaries; the joint decode above then
                         labels root×quality on those segments. GATE PASSED
                         2026-07-13: jazz held-out root 88.7→89.4, majmin
                         86.2→86.6; POP909 5-song root 76.9→78.6, majmin
                         50.1→51.1, 7ths 45.9→47.0 (all up). Gracefully falls
                         back to root-change segmentation if the (gitignored)
                         duration-prior npz is absent.
        semi_markov_dur_weight: Prior temperature on the duration term (0 = the
                         decode reduces bit-exactly to the root-change
                         segmentation ⇒ production joint decode). 0.25 (default)
                         won the fit sweep {0,.25,.5,1}; higher weights over-merge
                         and eat short dim/hdim chords.
        semi_markov_qual_weight: Weight on the v3 per-beat quality head as a
                         boundary EMISSION signal (0 = root-only decode, default).
                         The v3 head is only 51.7% q5-exact per-beat so it is not
                         trusted for the label — the joint decode re-labels
                         quality on the decoded segments.
        semi_markov_per_quality_dur: Use the per-q5 density-ratio duration shape
                         (log[P(d|q)/P(d)]) instead of the pooled prior. Default
                         OFF — the pooled prior injects zero quality label-bias
                         (the Korzeniowski discipline; see semi_markov_decode.py).
        user_constraints: optional collaborative-editing factors (Mission 3,
                         handoff §8). A JSON-friendly dict
                         ``{"confirms": [{"t0","t1","root","q5"?}, ...],
                            "merges":   [{"spans": [[t0,t1], ...]}, ...]}``.
                         chord-confirm → dominant emission clamp on the confirmed
                         (root, q5) cells + a duration-boundary hint (propagates
                         to neighbours through the joint decode's transition
                         factor). section-merge → tie corresponding segments and
                         pool their emission log-scores (P3). ``None`` (default)
                         is bit-identical to production. Only active with
                         use_joint_decode=True.
        use_llm_priors:  Inject LLM/offline-analyst priors (Mission 5, Part A)
                         into the joint decode via three seams: tonic override
                         (gated by LLM_KEY_TRUST), per-root q5 quality bonus, and
                         repeat-span pooling. Transition bias is intentionally
                         OFF (#27 saturated slot). **Default OFF** — bit-identical
                         to production for non-opted callers. Only active with
                         use_joint_decode=True.
        llm_analysis:    Pre-computed analyst JSON (from
                         llm_chord_priors.analyze/offline_analyze). When
                         use_llm_priors and this is None, the analyst is run on
                         (llm_song, llm_playlist) via the OFFLINE path.
        llm_song / llm_playlist: tune title + iReal playlist to derive priors
                         from, when llm_analysis is not supplied.
        llm_max_nats:    Ceiling on LLM prior strength in nats (default 8.0,
                         scaled by the analyst's confidence).

    Returns:
        ChordChart with fields populated for the interactive renderer.
    """
    audio_path = Path(audio_path)
    logger.info("chord_pipeline_v1: %s", audio_path.name)

    # ── User-constraint factors (Mission 3) ───────────────────────────────────
    _confirms: list = []
    _merges: list = []
    if user_constraints:
        from harmonia.models.user_constraints import ChordConfirm, SectionMerge
        for c in user_constraints.get("confirms", []) or []:
            _confirms.append(c if isinstance(c, ChordConfirm) else ChordConfirm(
                t0=float(c["t0"]), t1=float(c["t1"]), root=int(c["root"]),
                q5=(None if c.get("q5") is None else int(c["q5"])),
                bonus=float(c.get("bonus", 20.0))))
        for m in user_constraints.get("merges", []) or []:
            _merges.append(m if isinstance(m, SectionMerge) else SectionMerge(
                spans=[(float(a), float(b)) for a, b in m["spans"]]))

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

    # ── 5·U. Section-merge (P3): pool per-beat evidence across tied spans ──────
    # A user merge asserts two spans are the same material; pooling their per-
    # beat root posterior AND raw onset/note features (before segmentation, so
    # BOTH spans then segment + classify on the summed, denoised evidence) is the
    # "superimposed observations" √N win — gated by the user's assertion, never a
    # blind average (Candidate C's failure). Beat-level (not segment-level)
    # because equal musical length ⇒ equal beat count, robust to the two spans
    # segmenting into different numbers of chords.
    if _merges and beat_proba is not None:
        from harmonia.models.user_constraints import pool_beat_evidence
        try:
            beat_proba, onset_b, note_b = pool_beat_evidence(
                _merges, bt, beat_proba, onset_b, note_b)
            logger.info("chord_pipeline_v1: pooled %d merge group(s)", len(_merges))
        except ValueError as exc:
            logger.warning("chord_pipeline_v1: section-merge rejected (%s)", exc)

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

    # ── 6·SM. Per-beat semi-Markov (explicit-duration) re-segmentation (#27 M2) ─
    # When enabled, the segmentation above is DISCARDED: an explicit-duration
    # Viterbi over (root × q5) with a jazz1460-fit duration prior decides both the
    # boundaries and the per-segment root itself (headline lever = root; the
    # duration prior — ~0 mass on 1/3-beat chords — resists carving a spurious
    # 1-beat span around a single 5th-apart wrong beat).  Quality is NOT trusted
    # to the decode (v3 head is weak per-beat); the ctx classifier re-labels
    # quality on the decoded segments below.  See semi_markov_decode.py.
    if use_semi_markov and beat_proba is not None:
        try:
            from harmonia.models.semi_markov_decode import semi_markov_decode
            _dp = _get_jazz_duration_prior()
            _qp = None
            if semi_markov_qual_weight > 0.0:
                _bsv3 = _get_beat_seq_v3()
                if _bsv3 is not None:
                    _qp = _bsv3.qual_proba(onset_b, note_b)
            _dec = semi_markov_decode(
                beat_proba, dur_pmf=_dp, qual_proba=_qp,
                qual_weight=semi_markov_qual_weight,
                dur_weight=semi_markov_dur_weight,
                per_quality_duration=semi_markov_per_quality_dur,
            )
            segs = [(s, e) for (s, e, _r, _q) in _dec["segments"]]
            logger.debug("semi-Markov: %d segs (dur_w=%.2f qual_w=%.2f)",
                         len(segs), semi_markov_dur_weight, semi_markov_qual_weight)
        except FileNotFoundError as exc:
            # Duration prior npz is gitignored; on a fresh checkout it may be
            # absent. Fall back to the root-change segmentation above rather than
            # crash — equivalent to use_semi_markov=False (dur_weight=0).
            logger.warning("chord_pipeline_v1: semi-Markov disabled (%s)", exc)

    # ── 6·U. User chord-confirm duration-boundary hints (Mission 3) ────────────
    # Carve segment boundaries at each confirmed span's endpoints so the clamp
    # lands on a slot the user actually delimited (and freeing a neighbour of the
    # confirmed beats' evidence is itself a propagation channel at w=0).
    if _confirms and beat_proba is not None:
        from harmonia.models.user_constraints import confirm_cut_beats, force_boundaries
        segs = force_boundaries(segs, confirm_cut_beats(_confirms, bt))

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
    seg_q5_logp: list[np.ndarray | None] = []  # real per-q5 log-probs, for the encoder rerank

    # Semi-Markov (#27 M2) supplies the SEGMENTATION (boundaries) via its
    # duration-prior decode; the joint decode below then labels root×quality on
    # those segments exactly as in production (its top-K root×quality coupling is
    # what earns majmin 88.4 — a forced-root path throws that away).  So the
    # semi-Markov here is a drop-in replacement for `_root_change_segs`, tested by
    # whether duration-aware boundaries beat root-change-argmax boundaries.

    # ── 7·J. JOINT (root × quality) decode (audit build-order step 2) ──────────
    # Replaces the greedy argmax-root + quality-at-that-root labeling below with a
    # single MAP inference: candidate roots = top-K of the segment-summed beat
    # posterior, ×5 qualities, coupled across segments by the scale-relative
    # progression bigram as a TRANSITION factor (see harmonia/models/joint_decode.py).
    # Root and quality are decided together (the true root is in beat_seq's top-2
    # for ~86% of root errors but the greedy path commits to top-1 before quality
    # is computed).  Runs INSTEAD of the per-segment loop; the two-pass/local-key/
    # progression rerankers below (greedy post-hoc overrides this decode subsumes)
    # are skipped.  conf = the state's forward–backward max-marginal posterior.
    if use_joint_decode and beat_proba is not None:
        from harmonia.models.joint_decode import joint_decode

        gk_j = infer_key(_reg_raw(onset_b.sum(0)))
        # Greedy top-1 roots supply the ctx classifier's neighbour context for a
        # non-argmax candidate root (v1: only the current segment's root varies).
        greedy_roots = [int(beat_proba[s:e].sum(0).argmax()) for (s, e) in segs]

        def _joint_classify(idx: int, root: int):
            s, e = segs[idx]
            seg_on = onset_b[s:e].sum(0)
            seg_nt = note_b[s:e].sum(0)
            seg_bs = _reg_raw(seg_on, 0, 52)
            seg_tr = _reg_raw(seg_on, 60, 200)
            if ctx_clf is not None and seg_ll_mats[idx] is not None:
                k_ctx = 4
                ctx_ll = [seg_ll_mats[max(0, idx - k_ctx + j)] if 0 <= idx - k_ctx + j < len(segs) else None
                          for j in range(2 * k_ctx + 1)]
                ctx_rt = [greedy_roots[max(0, idx - k_ctx + j)] if 0 <= idx - k_ctx + j < len(segs) else 0
                          for j in range(2 * k_ctx + 1)]
                ch_abs = _reg_raw(seg_on)
                bsm_abs = beat_proba[s:e].mean(0)
                return ctx_clf.predict(
                    root, seg_on, seg_nt, seg_bs, seg_tr, ch_abs, ctx_ll, ctx_rt,
                    seventh_gate, bsm_probs_abs=bsm_abs, return_q5proba=True,
                )
            return fam_clf.predict(
                root, seg_on, seg_nt, seg_bs, seg_tr, seventh_gate, return_q5proba=True,
            )

        # User chord-confirm factors (Mission 3): per-segment emission clamps.
        # (section-merge is handled earlier as beat-level evidence pooling, step
        # 5·U; joint_decode's segment-level pool_groups remains available as a
        # tested API but the pipeline prefers the robust beat-level path.)
        _seg_cons = None
        _pool_groups = None
        if _confirms:
            from harmonia.models.user_constraints import build_segment_constraints
            _seg_cons = build_segment_constraints(_confirms, segs, bt)
            # Propagation channel #2 (Mission 3): feed each confirmed ROOT into
            # the ctx family classifier's NEIGHBOUR-root context (ctx_rt, ±4
            # segments). This sharpens the neighbours' QUALITY predictions with
            # the corrected root context WITHOUT routing through the progression
            # quality-bigram (which over-smooths jazz toward major, issue #25) —
            # the classifier already consumes neighbour roots as a feature, so a
            # confirmed root is exactly the evidence it wants. Complements the
            # emission clamp (which propagates through the transition factor).
            for i, con in enumerate(_seg_cons):
                if con is not None and con.get("root") is not None:
                    greedy_roots[i] = int(con["root"]) % 12

        # LLM/offline-analyst priors (Mission 5, Part A). Fills three seams:
        # tonic override (KEY_TRUST-gated), q5 quality bonus, repeat-span pooling.
        # Default OFF ⇒ _tonic_j == gk_j.tonic, _llm_q5 None, pool unchanged
        # (bit-identical to production).
        _tonic_j = gk_j.tonic
        _llm_q5 = None
        if use_llm_priors:
            _analysis = llm_analysis
            if _analysis is None and llm_song is not None:
                from scripts.llm_chord_priors import load_chart, offline_analyze
                _pl = Path(llm_playlist) if llm_playlist else (
                    REPO / "data" / "ireal" / "jazz1460.txt")
                _analysis = offline_analyze(load_chart(llm_song, _pl))
            if _analysis is not None:
                _llm = apply_llm_priors(
                    _analysis, segs, bt, inferred_tonic=gk_j.tonic,
                    max_nats=llm_max_nats)
                _tonic_j = _llm["tonic"]
                _llm_q5 = _llm["q5_bonus"]
                if _llm["pool_groups"]:
                    _pool_groups = (_pool_groups or []) + _llm["pool_groups"]
                logger.info(
                    "chord_pipeline_v1: LLM priors ON — tonic=%d (conf %.2f, "
                    "strength %.1f nats), %d q-roots, %d pool group(s)",
                    _tonic_j, _llm["factors"].confidence, _llm["factors"].strength,
                    len(_llm["factors"].quality_bonus), len(_llm["pool_groups"]))

        dec = joint_decode(segs, beat_proba, _joint_classify, _tonic_j,
                           K=joint_K, transition_weight=joint_transition_weight,
                           q5_bonus=_llm_q5,
                           constraints=_seg_cons, pool_groups=_pool_groups)
        # H1 (#27): re-reference the transition to a per-chord LOCAL key read off
        # the pass-1 (root, quality) labels, then re-decode. A ii-V-I inside a
        # tonicization then scores on the bigram's diatonic diagonal instead of
        # looking chromatic w.r.t. the global tonic. Two-pass by necessity: the
        # local key needs the whole pass-1 sequence. Only runs at positive weight
        # (at w=0 the transition is inert, so the second pass is a no-op).
        if joint_local_key_transition and joint_transition_weight > 0.0:
            lk_pos = _localkey_track_from_qualities_v2(
                list(dec["roots"]), list(dec["sev_h"]), gk_j.tonic, gk_j.mode,
            )
            local_tonic = [(int(dec["roots"][i]) - int(lk_pos[i][0])) % 12
                           for i in range(len(segs))]
            dec = joint_decode(segs, beat_proba, _joint_classify, _tonic_j,
                               K=joint_K, transition_weight=joint_transition_weight,
                               local_tonic=local_tonic, q5_bonus=_llm_q5,
                               constraints=_seg_cons, pool_groups=_pool_groups)
        # H2 (#27): ASR-style SHALLOW FUSION of the ProgressionEncoder as an
        # EMISSION factor. Decode once → score each candidate (root, q5) with the
        # encoder's grammar conditional P(q_i | neighbourhood) (centre masked) →
        # re-decode with λ·log P_enc folded into the emission BEFORE the joint
        # argmax (jointly with the root choice — the principled successor to the
        # reversed #21 post-hoc rerank). 1+ iterations.
        if joint_progression_fusion and joint_progression_weight > 0.0:
            for _ in range(max(1, joint_fusion_iters)):
                bonus_fn = _progression_fusion_bonus_fn(
                    list(dec["roots"]), list(dec["q5"]), list(dec["conf"]),
                    weight=joint_progression_weight,
                    subtract_prior=joint_fusion_subtract_prior,
                )
                if bonus_fn is None:
                    break
                # NOTE: progression fusion and LLM q5_bonus share the same
                # emission slot; when both are enabled the fusion bonus wins here
                # (both are OFF by default, so this is not composed in production).
                dec = joint_decode(
                    segs, beat_proba, _joint_classify, _tonic_j,
                    K=joint_K, transition_weight=joint_transition_weight,
                    q5_bonus=bonus_fn,
                    constraints=_seg_cons, pool_groups=_pool_groups)
        for idx, (s, e) in enumerate(segs):
            root = dec["roots"][idx]
            seg_roots[idx] = root
            fam_h, sev_h, conf = dec["fam_h"][idx], dec["sev_h"][idx], dec["conf"][idx]
            q5_logp = dec["q5_logp"][idx]
            seg_q5_logp.append(q5_logp)
            p_seg = beat_proba[s:e].sum(0)
            suggestions = _top_chord_suggestions(p_seg, q5_logp)
            t_start = float(bt[s])
            t_end = float(bt[min(e, len(bt) - 1)])
            label = f"{NOTE[root]}:{sev_h}"
            labeled.append((t_start, t_end, fam_h, sev_h, conf, label, suggestions))

    for idx, (s, e) in enumerate(segs) if not (use_joint_decode and beat_proba is not None) else []:
        seg_on = onset_b[s:e].sum(0)   # (88,)
        seg_nt = note_b[s:e].sum(0)
        seg_bs = _reg_raw(seg_on, 0, 52)
        seg_tr = _reg_raw(seg_on, 60, 200)

        # root
        p_seg: np.ndarray | None = None
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
            # return_q5proba is cheap (a few numpy ops on already-computed
            # posteriors) — always request it, only used downstream if
            # use_progression_prior triggers the encoder rerank.
            fam_h, sev_h, conf, q5_logp = ctx_clf.predict(
                root, seg_on, seg_nt, seg_bs, seg_tr,
                ch_abs, ctx_ll, ctx_rt, seventh_gate,
                bsm_probs_abs=bsm_abs,
                return_q5proba=True,
            )
        else:
            fam_h, sev_h, conf, q5_logp = fam_clf.predict(
                root, seg_on, seg_nt, seg_bs, seg_tr, seventh_gate,
                return_q5proba=True,
            )
        seg_q5_logp.append(q5_logp)
        suggestions = _top_chord_suggestions(p_seg, q5_logp) if p_seg is not None else []

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
        labeled.append((t_start, t_end, fam_h, sev_h, conf, label, suggestions))

    # ── 8·0. Two-pass 801d key-relative reclassification (issue #20/#23) ───────
    # The 801d ctx classifier learned to USE a per-chord local-key feature in its
    # weights (not a post-hoc rerank).  That feature can't be computed in the
    # per-segment first pass (a local key needs the whole predicted sequence), so
    # we do it here: (1) read a local key per chord off the pass-1 (root, quality)
    # sequence with the RAW v2 continuity teacher — the bootstrap-winning source
    # (v2 > v3); (2) re-run the 801d model per segment with the resulting 117d
    # block, replacing the pass-1 quality.  This is the realizable (non-circular)
    # version of the bootstrap upper bound, which used GT-quality context.  Runs
    # BEFORE the local-key / progression rerankers so they see refined qualities.
    if (ctx_classifier_variant == "801d_two_pass" and use_ctx_model
            and labeled and ctx_clf is not None and not use_joint_decode):
        ctx_clf_v3 = _get_ctx_clf_v3()
        if ctx_clf_v3 is not None and ctx_clf_v3._lk_dim > 0:
            try:
                gk2 = infer_key(_reg_raw(onset_b.sum(0)))
                sev_seq = [lab[3] for lab in labeled]
                lk_pos = _localkey_track_from_qualities_v2(
                    list(seg_roots), sev_seq, gk2.tonic, gk2.mode,
                )
                k_ctx = 4
                for idx, (s, e) in enumerate(segs):
                    seg_on = onset_b[s:e].sum(0)
                    seg_nt = note_b[s:e].sum(0)
                    seg_bs = _reg_raw(seg_on, 0, 52)
                    seg_tr = _reg_raw(seg_on, 60, 200)
                    root = seg_roots[idx]
                    ctx_ll = [seg_ll_mats[idx - k_ctx + j]
                              if 0 <= idx - k_ctx + j < len(segs) else None
                              for j in range(2 * k_ctx + 1)]
                    ctx_rt = [seg_roots[idx - k_ctx + j]
                              if 0 <= idx - k_ctx + j < len(segs) else 0
                              for j in range(2 * k_ctx + 1)]
                    ch_abs = _reg_raw(seg_on)
                    bsm_abs = beat_proba[s:e].mean(0) if beat_proba is not None else None
                    lk_block = _localkey_window_block(lk_pos, idx, k=k_ctx)
                    fam_h2, sev_h2, conf2, q5_logp2 = ctx_clf_v3.predict(
                        root, seg_on, seg_nt, seg_bs, seg_tr,
                        ch_abs, ctx_ll, ctx_rt, seventh_gate,
                        bsm_probs_abs=bsm_abs, return_q5proba=True,
                        lk_block=lk_block,
                    )
                    seg_q5_logp[idx] = q5_logp2
                    t0, t1, _fam, _old, _conf, _lab, sugg = labeled[idx]
                    labeled[idx] = (t0, t1, fam_h2, sev_h2, conf2,
                                    f"{NOTE[root]}:{sev_h2}", sugg)
            except Exception as exc:
                logger.warning("chord_pipeline_v1: 801d two-pass reclassify failed (%s)", exc)

    # ── 8a. Local-key diatonic-prior rerank (second pass, issue #20/#23) ───────
    # Two-pass by necessity: the LocalKeySeqGRU tagger needs the WHOLE (root,
    # quality) sequence to read a local key per chord (a descending-fifths
    # dominant chain only resolves at its end), but the per-segment loop above
    # only knows each segment's own acoustic quality when it runs — future
    # segments are unlabeled. So we run it here, once, over the completed
    # first-pass sequence, then let apply_diatonic_prior snap non-diatonic,
    # acoustically-uncertain family calls (the "A major where La minor is
    # expected" flip) to the local key's diatonic quality. Placed BEFORE the
    # progression rerank so the encoder sees diatonic-cleaned context.
    if use_local_key_prior and labeled and not use_joint_decode:
        try:
            global_chroma_lk = _reg_raw(onset_b.sum(0))
            gk = infer_key(global_chroma_lk)
            sev_seq = [lab[3] for lab in labeled]
            conf_seq = [lab[4] for lab in labeled]
            new_sev, new_post = rerank_local_key_qualities(
                list(seg_roots), sev_seq, conf_seq, gk.tonic,
                boost=local_key_weight,
                threshold_chromatic=local_key_threshold_chromatic,
                return_post=True,
            )
            for i, ns in enumerate(new_sev):
                if ns != labeled[i][3]:
                    t0, t1, fam_h, _old, conf, _lab, sugg = labeled[i]
                    # Stale-confidence fix (audit 2026-07-13): a flipped label
                    # carries the posterior of the decision that flipped it,
                    # not the pre-rerank acoustic confidence.
                    if new_post[i] is not None:
                        conf = float(new_post[i])
                    labeled[i] = (t0, t1, fam_h, ns, conf,
                                  f"{NOTE[seg_roots[i]]}:{ns}", sugg)
        except Exception as exc:
            logger.warning("chord_pipeline_v1: local-key prior rerank failed (%s)", exc)

    # ── 8b. Progression-encoder quality rerank (second pass, issue #21) ────────
    if use_progression_prior and labeled and not use_joint_decode:
        try:
            sev_seq = [lab[3] for lab in labeled]
            conf_seq = [lab[4] for lab in labeled]
            new_sev, new_post = rerank_progression_qualities(
                list(seg_roots), sev_seq, conf_seq, weight=progression_weight,
                aco_logprobs=seg_q5_logp, return_post=True,
            )
            for i, ns in enumerate(new_sev):
                if ns != labeled[i][3]:
                    t0, t1, fam_h, _old, conf, _lab, sugg = labeled[i]
                    # Stale-confidence fix (audit 2026-07-13): see 8a above.
                    if new_post[i] is not None:
                        conf = float(new_post[i])
                    labeled[i] = (t0, t1, fam_h, ns, conf,
                                  f"{NOTE[seg_roots[i]]}:{ns}", sugg)
        except Exception as exc:
            logger.warning("chord_pipeline_v1: progression rerank failed (%s)", exc)

    # ── 9. Coalesce adjacent same-label segments ──────────────────────────────
    # Suggestions travel with whichever merged sub-segment had the higher
    # confidence — same rule already used for `conf` itself.
    coalesced: list[tuple[float, float, str, float, list[dict]]] = []
    for t0, t1, fam_h, sev_h, conf, label, suggestions in labeled:
        if coalesced and coalesced[-1][2] == label:
            prev = coalesced[-1]
            new_conf = max(prev[3], conf)
            new_sugg = suggestions if conf >= prev[3] else prev[4]
            coalesced[-1] = (prev[0], t1, label, new_conf, new_sugg)
        else:
            coalesced.append((t0, t1, label, conf, suggestions))

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
            apply_phase_shift,
            build_chord_ssm,
            correct_section_phase,
            detect_section_boundaries,
            estimate_base_period_bars,
            label_sections,
            load_progression_model,
        )
        tonic_pc = _note_name_to_pc(key_result.key_name)
        qi: dict[str, int] = {}
        seq: list[tuple[int, int]] = [(-1, -1)] * n_beats
        # Parallel per-beat (root_rel, q5-family) sequence for phase correction,
        # which needs real functional qualities (the qi index above is arbitrary).
        seq_q5: list[tuple[int, int] | None] = [None] * n_beats
        for (s, e), root, lab in zip(segs, seg_roots, labeled):
            sev_h = lab[3]
            q = qi.setdefault(sev_h, len(qi))
            q5 = _harte_to_q5idx(sev_h)
            for b in range(s, min(e, n_beats)):
                seq[b] = ((root - tonic_pc) % 12, q)
                if q5 is not None:
                    seq_q5[b] = ((root - tonic_pc) % 12, q5)
        ssm = build_chord_ssm(seq)
        bnds = detect_section_boundaries(ssm, beats_per_bar=4)

        # ── Phase correction (issue #22 cycle-shift, e.g. Let It Be) ───────────
        # detect_section_boundaries assumes phase 0; recover the true loop phase
        # from harmonic-progression likelihood (tonic-opening bias), not from
        # downbeat GT (unavailable on real audio).
        if use_phase_correction:
            try:
                period_bars = estimate_base_period_bars(ssm, beats_per_bar=4)
                prog_model = load_progression_model()
                if period_bars and prog_model is not None:
                    shift = correct_section_phase(seq_q5, period_bars, 4, prog_model)
                    if shift:
                        bnds = apply_phase_shift(bnds, shift, 4, n_beats)
                        logger.info(
                            "chord_pipeline_v1: section phase shift +%d bars", shift)
            except Exception as exc:
                logger.warning("chord_pipeline_v1: phase correction failed (%s)", exc)

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
    # Display-layer confidence (audit step 1b): fuse the quality conf with the
    # span's root posterior (the quality heads never see the root, so a
    # confidently-wrong root used to surface as a confident chord), then map
    # through the fitted isotonic calibration when available.  Labels and every
    # internal gate were computed above — nothing here can change a decision.
    # audio_domain selects the calibration map (Mission 4). "synth" maps the
    # FUSED raw (conf × root_conf); "real" maps the quality confidence_raw
    # (conf) it was fitted on — see _get_conf_calibrator. Default "real" for the
    # server path (users analyse real recordings); eval harnesses on MMA renders
    # pass audio_domain="synth".
    conf_cal = _get_conf_calibrator(audio_domain)
    beat_dur_s = period
    chords_out = []
    segments_out = []
    for t0, t1, label, conf, suggestions in coalesced:
        n_b = max(1, round((t1 - t0) / beat_dur_s))
        root_conf = _span_root_conf(beat_proba, bt, t0, t1, label)
        raw = conf if root_conf is None else conf * root_conf
        # The map declares which raw score it was fitted on (issue #29): a
        # "fused" map (conf × root_conf) folds in root uncertainty; a legacy
        # "conf" map (old root-blind real fit) takes the quality conf alone.
        score_kind = getattr(conf_cal, "score_kind", None)
        cal_input = conf if score_kind == "conf" else raw
        conf_out = conf_cal(cal_input) if conf_cal is not None else conf
        chords_out.append({
            "label":          label,
            "start_s":        round(t0, 3),
            "end_s":          round(t1, 3),
            "duration_beats": n_b,
            "confidence":     round(conf_out, 4),
            "confidence_raw": round(conf, 4),
            "root_conf":      round(root_conf, 4) if root_conf is not None else None,
            "suggestions":    suggestions,
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
