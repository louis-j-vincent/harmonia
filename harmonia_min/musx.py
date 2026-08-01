"""harmonia_min/musx.py — music-x-lab frame posteriors + beat-grid re-decode.

COPIED 2026-07-30 from harmonia/models/musx_redecode.py (feat/minimal-pipeline
rebuild) with three changes only:
  * ``musx_dir()`` inlined (was harmonia.models.musx_bass.musx_dir) — resolves
    the vendored ISMIR2019 clone via HARMONIA_MUSX_DIR or harmonia/third_party/.
  * ``REPO`` depth fixed (this package sits at repo root, not two levels down).
  * the old pipeline's env-flag helpers (``enabled``/``latency_grid_from_env``)
    dropped — harmonia_min's orchestration decides, not env vars.

The science is unchanged; see the original module's header for the full
measurement record (+2.20 pp partial-credit, latency selected per song by the
decoder's own path log-likelihood, boundaries land exactly on OUR beats).
Posterior cache: data/cache/musx_probs/<audio stem>.npz (shared with the old
pipeline — same extractor, same layout: [triad(T,73), bass(T,13), s7(T,4),
s9(T,4), s11(T,3), s13(T,3)] on the 23.22 ms frame grid).
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[1]

# music-x-lab frame grid (settings.py: DEFAULT_SR=22050, DEFAULT_HOP_LENGTH=512).
MUSX_SR = 22050
MUSX_HOP = 512
FRAME_DT = MUSX_HOP / MUSX_SR          # 0.023219954648526078 s  (43.07 fps)

# The 5-fold ensemble shipped with the clone (== MODEL_NAMES in chord_recognition.py).
MODEL_NAMES = ['joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s%d.best' % i
               for i in range(5)]

#: Candidate latencies (seconds) searched per song.  Spans the measured range
#: (+46…+289 ms across the 7 frozen songs) with a 40 ms step.
DEFAULT_LATENCY_GRID: tuple[float, ...] = (0.0, 0.04, 0.08, 0.12, 0.16,
                                           0.20, 0.24, 0.28)

#: Viterbi change penalty on the beat grid.  Pooled optimum 40; LOSO-stable
#: (per-fold picks were 40 on 6/7 songs, 55 on the seventh).
DEFAULT_PENALTY = 40.0

_PROB_CACHE = REPO / "data" / "cache" / "musx_probs"


# ── beat_arr: the transition structure the vendored decoder already supports ──

def make_beat_arr(n_frame: int, beat_times, latency: float = 0.0,
                  downbeat_times=None, beats_per_bar: int = 4) -> np.ndarray:
    """Mirror of ``XHMMDecoder._XHMMDecoder__get_beat_arr``, fed OUR beat grid.

    Semantics consumed by the clone's ``decode``:
      0 -> no chord change may occur at this frame;
      1 -> change costs ``diff_trans_penalty``;
      2 / 3 / 4 -> change costs ``beat_trans_penalty[0/1/2]``
                   (downbeat / mid-bar / other beat).

    ``latency`` displaces the whole grid later, so that a decoder whose evidence
    arrives late still gets a legal transition at the right musical instant; the
    caller shifts the decoded times back by the same amount.

    NOTE: ``downbeat_times`` is REQUIRED in harmonia_min (2026-07-31). The old
    accuracy study measured graded ≈ flat on label overlap (0.6627 vs 0.6644),
    but placement is what the chart lives on: with a flat penalty, last-beat
    and next-downbeat cost the same, and at phrase turns (ambiguous frames)
    noise put chords one beat early (Louis's This Love report, 1:56 / 2:52).
    Downbeat-graded costs break exactly that tie toward the downbeat.
    """
    arr = np.ones(int(n_frame), dtype=np.int8)
    bt = np.asarray(beat_times, dtype=float) + float(latency)
    fr = np.round(bt / FRAME_DT).astype(int)
    keep = (fr >= 0) & (fr < n_frame)
    fr_k = fr[keep]
    if len(fr_k) < 2:
        return arr
    for i in range(len(fr_k) - 1):
        arr[fr_k[i] + 1:fr_k[i + 1]] = 0
    if downbeat_times is None:
        return arr
    db = np.asarray(downbeat_times, dtype=float)
    if len(db) < 3:
        return arr
    arr[fr_k] = 4
    idx = np.unique([int(np.argmin(np.abs(bt - t))) for t in db])
    idx = idx[(idx >= 0) & (idx < len(fr))]
    f_db = fr[idx]
    f_db = f_db[(f_db >= 0) & (f_db < n_frame)]
    mid = idx + beats_per_bar // 2
    mid = mid[mid < len(fr)]
    f_mid = fr[mid]
    f_mid = f_mid[(f_mid >= 0) & (f_mid < n_frame)]
    arr[f_mid] = 3
    arr[f_db] = 2
    return arr


# ── the vendored clone (imported, never edited) ──────────────────────────────

def _musx_dir() -> Path:
    """The vendored music-x-lab clone (entry script + pretrained weights)."""
    candidates = []
    env = os.environ.get("HARMONIA_MUSX_DIR")
    if env:
        candidates.append(Path(env))
    candidates.append(REPO / "harmonia" / "third_party"
                      / "ISMIR2019-Large-Vocabulary-Chord-Recognition")
    for d in candidates:
        if (d / "chord_recognition.py").exists() and \
           list((d / "cache_data").glob("*.sdict")):
            return d
    raise RuntimeError("harmonia_min.musx: music-x-lab clone not found "
                       f"(tried {[str(c) for c in candidates]})")


class _InMusxDir:
    """cwd + sys.path into the clone; the clone resolves 'cache_data' relatively."""

    def __enter__(self):
        self._cwd = Path.cwd()
        d = _musx_dir()
        os.chdir(d)
        if str(d) not in sys.path:
            sys.path.insert(0, str(d))
        return d

    def __exit__(self, *exc):
        os.chdir(self._cwd)
        return False


def frame_posteriors(audio_path: Path | str, *, use_cache: bool = True,
                     cache_dir: Path | None = None) -> list[np.ndarray]:
    """The 5-fold-averaged frame posteriors ``chord_recognition.py`` throws away.

    Returns ``[triad(73), bass(13), s7(4), s9(4), s11(3), s13(3)]``, each
    ``(n_frame, k)`` on the 23.22 ms grid.  Triad column 0 is ``N``; column
    ``i>=1`` is root ``(i-1)%12`` with triad type ``(i-1)//12+1`` in
    {maj,min,sus4,sus2,dim,aug}.

    Cached (stem-keyed, like ``musx_bass``) to ``data/cache/musx_probs``.  A song
    costs ~3–9 MB compressed.
    """
    audio_path = Path(audio_path).resolve()
    cdir = Path(cache_dir) if cache_dir is not None else _PROB_CACHE
    cache = cdir / f"{audio_path.stem}.npz"
    names = ("triad", "bass", "s7", "s9", "s11", "s13")
    if use_cache and cache.exists():
        z = np.load(cache)
        return [z[n] for n in names]

    with _InMusxDir():
        from mir import io, DataEntry
        from mir.nn.train import NetworkInterface
        from extractors.cqt import CQTV2
        from chordnet_ismir_naive import ChordNet
        entry = DataEntry()
        entry.prop.set('sr', MUSX_SR)
        entry.prop.set('hop_length', MUSX_HOP)
        entry.append_file(str(audio_path), io.MusicIO, 'music')
        entry.append_extractor(CQTV2, 'cqt')
        cqt = entry.cqt
        acc = None
        for name in MODEL_NAMES:
            net = NetworkInterface(ChordNet(None), name, load_checkpoint=False)
            out = net.inference(cqt)
            acc = list(out) if acc is None else [a + b for a, b in zip(acc, out)]
            del net
    probs = [(a / len(MODEL_NAMES)).astype(np.float32) for a in acc]
    if use_cache:
        cdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **dict(zip(names, probs)))
    return probs


def _decoder(penalty: float, beat_trans_penalty=(15.0, 45.0, 100.0),
             chord_dict: str = "submission"):
    from extractors.xhmm_ismir import XHMMDecoder
    return XHMMDecoder(template_file=f'data/{chord_dict}_chord_list.txt',
                       diff_trans_penalty=float(penalty),
                       beat_trans_penalty=tuple(beat_trans_penalty))


def _tags_to_lab(tags: list[str], latency: float) -> list[tuple[float, float, str]]:
    out, last, n = [], 0, len(tags)
    for i in range(n):
        if i + 1 == n or tags[i + 1] != tags[i]:
            t0 = last * FRAME_DT - latency
            t1 = (i + 1) * FRAME_DT - latency
            if t1 > 0:
                out.append((max(0.0, t0), t1, tags[i]))
            last = i + 1
    return out


def path_loglik(logprob: np.ndarray, names: list[str],
                lab: list[tuple[float, float, str]], penalty: float,
                n_frame: int, latency: float = 0.0) -> float:
    """The decoder's own Viterbi objective for a decoded labelling.

    Used as the GT-FREE selector over candidate latencies: same observations,
    same emission model, only the legal-transition set moves.
    """
    idx = {n: i for i, n in enumerate(names)}
    tag = np.zeros(n_frame, dtype=int)
    for t0, t1, s in lab:
        if s not in idx:
            return float("-inf")
        a = max(0, int(round((t0 + latency) / FRAME_DT)))
        b = min(n_frame, int(round((t1 + latency) / FRAME_DT)))
        tag[a:b] = idx[s]
    em = float(logprob[np.arange(n_frame), tag].sum())
    return em - float(penalty) * max(0, len(lab) - 1)


def redecode(beat_times, probs: list[np.ndarray], *, downbeat_times,
             penalty: float = DEFAULT_PENALTY,
             latency_grid=DEFAULT_LATENCY_GRID,
             beats_per_bar: int = 4,
             beat_trans_penalty=(15.0, 45.0, 100.0),
             chord_dict: str = "submission",
             ) -> tuple[list[tuple[float, float, str]], float]:
    """Beat-aware, latency-compensated re-decode -> (labels, chosen latency).

    Boundaries land exactly on ``beat_times``.  The latency is selected per song
    by maximising the decoder's own path log-likelihood — no ground truth.
    """
    n_frame = int(probs[0].shape[0])
    plist = [np.asarray(p, dtype=np.float64) for p in probs]
    best = (float("-inf"), 0.0, [])
    with _InMusxDir():
        hmm = _decoder(penalty, beat_trans_penalty, chord_dict)
        names, logprob = hmm.get_chord_tag_obs(plist)
        for L in latency_grid:
            arr = make_beat_arr(n_frame, beat_times, L, downbeat_times,
                                beats_per_bar=beats_per_bar)
            tags = hmm.decode(plist, arr)
            lab = _tags_to_lab(tags, L)
            ll = path_loglik(logprob, names, lab, penalty, n_frame, L)
            if ll > best[0]:
                best = (ll, float(L), lab)
    logger.info("musx_redecode: latency %.0f ms selected (loglik %.1f), %d segments",
                best[1] * 1000, best[0], len(best[2]))
    return best[2], best[1]
