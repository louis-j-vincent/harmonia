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
FAM_HARTE = {
    "major": "maj", "minor": "min", "diminished": "dim",
    "augmented": "aug", "suspended": "sus4",
}
B7_HARTE = {
    "majT": "maj", "minT": "min", "dimT": "dim", "augT": "aug", "susT": "sus4",
    "maj7": "maj7", "min7": "min7", "dom7": "7", "m7b5": "hdim7", "dim7": "dim7",
    "minmaj7": "minmaj7", "7sus4": "sus4", "aug7": "aug", "augmaj7": "aug",
}


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
                seventh_gate: float = 0.0) -> tuple[str, str, float]:
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


# ── module-level lazy-loaded models ──────────────────────────────────────────

_family_clf: _FamilyClassifier | None = None
_ctx_clf: _CtxFamilyClassifier | None = None
_beat_seq: _BeatSeqModel | None = None
_root_mdl: _RootModel | None = None


def _get_family_clf() -> _FamilyClassifier:
    global _family_clf
    if _family_clf is None:
        _family_clf = _FamilyClassifier()
    return _family_clf


def _get_ctx_clf() -> _CtxFamilyClassifier | None:
    global _ctx_clf
    if _ctx_clf is not None:
        return _ctx_clf
    # prefer the large model (300-song) over the small one (60-song) when both exist
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


def _get_beat_seq() -> _BeatSeqModel | None:
    global _beat_seq
    if _beat_seq is not None:
        return _beat_seq
    p = MODELS / "beat_seq_model.npz"
    if p.exists():
        _beat_seq = _BeatSeqModel(p)
    return _beat_seq


def _get_root_mdl() -> _RootModel | None:
    global _root_mdl
    if _root_mdl is not None:
        return _root_mdl
    p = MODELS / "root_model.npz"
    if p.exists():
        _root_mdl = _RootModel(p)
    return _root_mdl


# ── segmentation ──────────────────────────────────────────────────────────────

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
        use_beat_seq:  Use the beat-sequence root model (88.3% CV) when available.
        use_ctx_model: Use the entropy-gated ctx MLP family model when saved.

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

    # ── 5. Coarse segmentation ────────────────────────────────────────────────
    segs = _coarse_segments(onset_b, theta=theta, cell=cell)

    # ── 6. Beat-sequence root probabilities ───────────────────────────────────
    beat_seq = _get_beat_seq() if use_beat_seq else None
    beat_proba: np.ndarray | None = None
    if beat_seq is not None:
        beat_proba = beat_seq.predict_proba(onset_b, note_b)  # (n_beats, 12)

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
            fam_h, sev_h, conf = ctx_clf.predict(
                root, seg_on, seg_nt, seg_bs, seg_tr,
                ch_abs, ctx_ll, ctx_rt, seventh_gate,
            )
        else:
            fam_h, sev_h, conf = fam_clf.predict(
                root, seg_on, seg_nt, seg_bs, seg_tr, seventh_gate
            )

        t_start = float(bt[s])
        t_end   = float(bt[min(e, len(bt) - 1)])
        label = f"{NOTE[root]}:{sev_h}"
        labeled.append((t_start, t_end, fam_h, sev_h, conf, label))

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
    )
