"""nnls_features.py — NNLS-24 chroma front-end for the production inference path.

The opt-in counterpart to Basic Pitch (BP48) in chord_pipeline_v1: extracts the
real Mauch & Dixon NNLS-Chroma VAMP plugin's 24-dim `bothchroma` (bass|treble),
pooled per beat, and loads the trained root + quality heads
(harmonia/models/nnls24_heads.npz, from scripts/train_nnls24_heads.py).

Feature convention (matched byte-for-byte to the RWC training features in
data/cache/rwc/rwc_nnls24.npz — see scratchpad/nnls_real_extract.py):
  * VAMP `nnls-chroma:nnls-chroma` output `bothchroma`, 24-dim, index 0 = A.
  * per beat interval [t0,t1): MEAN bothchroma over frames in the interval.
  * roll each 12-half by 9 -> C-first pitch-class frame.
  * L2-normalise each 12-half independently.  Stacked -> (…, 24).

Bass is an UNTRAINED argmax on the (C-frame) bass half — no weights, free on all
audio (SESSION_PRESENTATION_2026_07_17: NNLS bass-argmax is the winning sounding-
bass estimator, 0.776 all / 0.743 inversions).

WHY this is opt-in and not a silent default (CLAUDE.md rule #6): swapping the
feature front-end changes every downstream intermediate (beat root posteriors,
segmentation, quality).  infer_chords_v1(feature_frontend="nnls24") selects it;
"bp48" (default) is bit-identical to before.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parent.parent.parent
MODELS = REPO / "harmonia" / "models"
HEADS_NPZ = MODELS / "nnls24_heads.npz"
_CACHE_DIR = REPO / "data" / "cache" / "nnls_infer"

SR = 44100
_ROLL_TO_C = 9  # index 0 = A -> roll by 9 puts C at index 0


def _l2(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


# ── raw VAMP extraction (cached per audio file) ──────────────────────────────

def extract_bothchroma(audio_path: Path, *, use_cache: bool = True):
    """Run the NNLS-Chroma VAMP plugin -> (arr (T,24) index0=A, times (T,)).

    Cached to data/cache/nnls_infer/<name>_<mtime>.npz.  Raises RuntimeError with
    an actionable message if the `vamp` module or the nnls-chroma plugin is
    unavailable (the plugin is a native VAMP library, not a pip package).
    """
    audio_path = Path(audio_path)
    key = f"{audio_path.stem}_{int(audio_path.stat().st_mtime)}.npz"
    cache = _CACHE_DIR / key
    if use_cache and cache.exists():
        z = np.load(cache)
        return z["arr"], z["times"]

    try:
        import librosa
        import vamp
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "NNLS-24 front-end needs the `vamp` python module (pip install vamp) "
            "and the native NNLS-Chroma VAMP plugin on VAMP_PATH."
        ) from e

    plugins = vamp.list_plugins()
    if "nnls-chroma:nnls-chroma" not in plugins:
        raise RuntimeError(
            "NNLS-Chroma VAMP plugin not found (have: "
            f"{[p for p in plugins if 'nnls' in p.lower()]}). Install libnnls-chroma "
            "and point VAMP_PATH at it."
        )

    y, _ = librosa.load(str(audio_path), sr=SR, mono=True)
    out = vamp.collect(y.astype(np.float32), SR, "nnls-chroma:nnls-chroma",
                       output="bothchroma")
    step, arr = out["matrix"]                 # arr (T,24), index 0 = A
    arr = np.asarray(arr, np.float32)
    times = np.arange(arr.shape[0]) * float(step)
    if use_cache:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(cache, arr=arr, times=times)
    return arr, times


def pool_beats(arr: np.ndarray, times: np.ndarray,
               beat_times: np.ndarray) -> np.ndarray:
    """Pool raw bothchroma to (n_beats, 24) C-frame, L2-per-half feature rows.

    n_beats = len(beat_times) - 1.  Empty intervals fall back to the nearest
    frame (same rule as the training extractor's block_both).
    """
    n = len(beat_times) - 1
    out = np.zeros((n, 24), np.float32)
    for b in range(n):
        a, c = beat_times[b], beat_times[b + 1]
        m = (times >= a) & (times < c)
        if not m.any():
            j = int(np.argmin(np.abs(times - 0.5 * (a + c))))
            seg = arr[j]
        else:
            seg = arr[m].mean(0)
        bass = np.roll(seg[:12], _ROLL_TO_C)
        treb = np.roll(seg[12:], _ROLL_TO_C)
        out[b] = np.concatenate([_l2(bass), _l2(treb)])
    return out


# ── trained heads (root + quality cascade) ───────────────────────────────────

class NNLS24Heads:
    """Loads nnls24_heads.npz and serves per-row root/quality posteriors.

    Root head consumes the absolute 24-d vector; quality head consumes the
    bass|treble rotated so the predicted root sits at index 0 (deployable
    cascade).  Reconstructs the training MLP (multihead_training.MLP) and loads
    weights from the npz arrays — torch is required at inference (already a
    pipeline dependency for the ctx models).
    """

    def __init__(self, path: Path = HEADS_NPZ) -> None:
        import torch  # noqa: F401
        import sys
        sys.path.insert(0, str(REPO / "scratchpad"))
        from multihead_training import MLP

        d = np.load(path, allow_pickle=True)
        self.qualities = [str(q) for q in d["qualities"]]
        hid = tuple(int(x) for x in d["hid"])

        def _build(prefix: str, din: int, dout: int):
            import torch as _t
            m = MLP(din, dout, hid)
            state = {k[len(prefix) + 2:]: _t.tensor(d[k])
                     for k in d.files if k.startswith(prefix + "__")}
            m.load_state_dict(state)
            m.eval()
            return m

        self._root = _build("root", int(d["root_din"][0]), int(d["root_dout"][0]))
        self._qual = _build("qual", int(d["qual_din"][0]), int(d["qual_dout"][0]))
        self.sanity = d["sanity"].tolist() if "sanity" in d.files else None

    def _proba(self, model, X: np.ndarray) -> np.ndarray:
        import torch
        with torch.no_grad():
            return torch.softmax(
                model(torch.tensor(np.asarray(X, np.float32))), 1).numpy()

    def root_proba(self, feat24: np.ndarray) -> np.ndarray:
        """(n,12) root pitch-class posteriors from the absolute nnls24 rows."""
        feat24 = np.atleast_2d(feat24).astype(np.float32)
        return self._proba(self._root, feat24)

    def quality_idx(self, feat24: np.ndarray, roots: np.ndarray) -> np.ndarray:
        """(n,) argmax quality index (self.qualities order); cascade-rotated."""
        feat24 = np.atleast_2d(feat24).astype(np.float32)
        roots = np.atleast_1d(roots).astype(int)
        bass, treb = feat24[:, :12], feat24[:, 12:]
        out = np.empty((len(feat24), 24), np.float32)
        for i in range(len(feat24)):
            r = roots[i] % 12
            out[i, :12] = np.roll(bass[i], -r)
            out[i, 12:] = np.roll(treb[i], -r)
        return self._proba(self._qual, out).argmax(1)


_heads: NNLS24Heads | None = None


def get_heads() -> NNLS24Heads | None:
    """Lazy singleton; returns None (with a warning) if the checkpoint is absent."""
    global _heads
    if _heads is not None:
        return _heads
    if not HEADS_NPZ.exists():
        logger.warning("nnls_features: %s missing — run scripts/train_nnls24_heads.py",
                       HEADS_NPZ)
        return None
    _heads = NNLS24Heads()
    return _heads
