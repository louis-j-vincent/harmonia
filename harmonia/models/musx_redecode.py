"""musx_redecode.py — DEFAULT-OFF brick: re-decode music-x-lab's FRAME posteriors
on our beat grid, with a per-song latency correction chosen without ground truth.

WHY (measured 2026-07-27, `docs/research_sessions/musx_frame_posteriors_2026-07-27.md`)
--------------------------------------------------------------------------------------
The shipped chart reads music-x-lab through its ``.lab`` file only.  That ``.lab``
is produced by the vendored clone's own Viterbi with ``use_beats=False``, i.e. a
**flat ~30-nat change penalty applied at every 23.2 ms frame** — mathematically a
memoryless (geometric) duration prior, the regime Korzeniowski & Widmer (ISMIR
2018) identify as "the model degenerates into a smoother".  Two consequences were
measured here:

1. **music-x-lab's boundaries are systematically LATE.**  Against the 7 frozen
   hand-verified songs, with a fixed GT match (N=457): our chart's boundaries are
   unbiased (median **−4.0 ms**), the raw ``.lab`` boundaries are **+112.8 ms**
   late, and a maximum-likelihood change point computed from the frame posteriors
   is **+134.4 ms** late.  Simply shifting the raw ``.lab`` 160 ms earlier is worth
   +1.59 pp partial-credit — which also supplies the missing mechanism for the
   2026-07-26 refutation of ``segment_source="musx"`` (−1.97 pp): its boundaries
   arrive after the chord actually changed.
2. **The posteriors are nonetheless a MORE PRECISE boundary estimator than what we
   ship** — pooled with each song's own median removed (so this is precision, not
   bias): MAD **84.5 ms → 71.7 ms**, within-100 ms **0.536 → 0.611**.  They are
   better; they are just biased.

So the fix is not a novelty curve (chroma novelty, onset strength and HPSS are all
already measured and refuted) but a **re-decode**: allow chord changes only on our
Beat This! beats, and shift the allowed-transition grid by the latency.

WHAT THIS DOES
--------------
``redecode()`` builds the transition-structure array the vendored
``XHMMDecoder`` already supports but that ``chord_recognition.py`` never switches
on (``use_beats``), feeds it OUR beat grid displaced by a latency ``L``, calls the
clone's own Viterbi, and shifts the decoded times back by ``L`` so every boundary
lands exactly on a beat.

``L`` is chosen **per song and without ground truth**, by maximising the decoder's
own Viterbi path log-likelihood over a small grid of candidate latencies: the same
observations and the same emission model, only the allowed-transition set moves,
so the latency whose beat grid best explains the posteriors wins.  That matters —
a single GLOBAL constant L does not survive leave-one-song-out (+0.54 pp only,
per-song optima span 60–340 ms), whereas the likelihood-selected per-song L does.

MEASURED (7 frozen songs, LOSO on the one fitted knob, `accuracy_score`)
------------------------------------------------------------------------
Fed back into the real chord head as BOTH the musx label source and the
segmentation source, with the Occam post-pass gated (see below):

    partial_credit  0.6419 -> 0.6639   (+2.20 pp)      <- headline
    mirex_root      0.7367 -> 0.7503   (+1.36 pp)
    mirex_majmin    0.7102 -> 0.7300   (+1.98 pp)
    mirex_sevenths  0.5173 -> 0.5332   (+1.58 pp)
    bass_root       0.7440 -> 0.7584   (+1.44 pp)
    strict          0.4774 -> 0.4890   (+1.16 pp)

Per song (partial): bein_green +3.45, blue_bossa +2.25, backing +4.67,
close_to_you +5.17, stand_by_me 0.00, every_breath **−1.20**, georgia **−1.03**.
With Occam simply OFF instead of gated it is +1.90 pp partial / +1.04 pp root.

Of the gain, **80 % is boundary placement**: re-decoding the boundaries but
re-sampling the OLD flat-`.lab` labels onto them already gives +1.24 pp of the
+1.55 pp standalone gain (labels alone are worth +0.28 pp).

WHAT THIS DOES **NOT** SOLVE (CLAUDE.md rule #4)
------------------------------------------------
* **It needs the Occam post-pass gated.**  With Occam ON as it ships, the same
  configuration is only +1.11 pp partial, because close_to_you swings by
  **13.2 pp** between Occam ON and OFF under the new segmentation (Occam picks a
  different loop family off the changed per-bar posterior).  Under this
  segmentation Occam fires on only 2/7 songs: close_to_you −13.21 pp
  (its families are ``cov=0.78 dev=4`` and ``cov=0.67 dev=16``) and stand_by_me
  +3.16 pp (``cov=0.97 dev=0``).  A GT-free gate — accept a loop family only if
  ``cov >= 0.95 and dev == 0`` — separates them, and is a *verified no-op on
  today's shipped chart*.  That gate is NOT implemented here (the post-pass lives
  in guarded files); it was measured by env-toggling and rests on N=2 songs.
* **It does not fix the beat grid.**  Only 18 % of GT chord changes lie within
  30 ms of one of our beats (35 % within 60 ms, 57 % within 100 ms).  Every
  beat-quantised chart, ours included, is capped by that.
* **It does not reach the label-recut oracle.**  Handing this decoder the GT change
  times as its only legal transition frames gives partial 0.6903 / root 0.7728 —
  still ~3 pp of root below the 2026-07-26 oracle (0.8023) that re-cuts *our own*
  labels on GT spans.  The residual is label identity, not boundary placement.
* **Downbeat-graded transition costs do not help** (best graded 0.6627 vs flat
  0.6644), and an explicit iReal-derived duration prior (semi-Markov) strictly
  hurts (0.6652 at lambda=0 -> 0.6365 at lambda=60) — see the session log.
* Two songs regress; this is not a universal win.

DEFAULT OFF.  Nothing in the shipped pipeline imports this module.
"""
from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

REPO = Path(__file__).resolve().parents[2]

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


def enabled() -> bool:
    """Default OFF.  Set HARMONIA_MUSX_REDECODE=1 to opt in (nothing reads this
    yet — the brick is unwired; the flag exists so a future wiring is a one-liner
    with a kill switch, matching `no_chord_policy` / `seventh_upgrade`)."""
    return os.environ.get("HARMONIA_MUSX_REDECODE", "0") == "1"


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

    NOTE: ``downbeat_times`` is supported because the vendored decoder supports
    it, but it was MEASURED not to help on the frozen benchmark (best graded
    0.6627 vs flat 0.6644) — the uniform beat grid's bar phase is only 0.30–0.50
    pure against Beat This!'s downbeats on 3 of 7 songs.  Left in for future work,
    not recommended.
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
    from harmonia.models.musx_bass import musx_dir
    d = musx_dir()
    if d is None:
        raise RuntimeError(
            "musx_redecode: music-x-lab clone not found (see musx_bass.musx_dir)")
    return d


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


def redecode(beat_times, probs: list[np.ndarray], *,
             penalty: float = DEFAULT_PENALTY,
             latency_grid=DEFAULT_LATENCY_GRID,
             downbeat_times=None,
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
            arr = make_beat_arr(n_frame, beat_times, L, downbeat_times)
            tags = hmm.decode(plist, arr)
            lab = _tags_to_lab(tags, L)
            ll = path_loglik(logprob, names, lab, penalty, n_frame, L)
            if ll > best[0]:
                best = (ll, float(L), lab)
    logger.info("musx_redecode: latency %.0f ms selected (loglik %.1f), %d segments",
                best[1] * 1000, best[0], len(best[2]))
    return best[2], best[1]


def redecode_audio(audio_path: Path | str, beat_times, **kw):
    """Convenience: extract (or load cached) posteriors, then re-decode."""
    return redecode(beat_times, frame_posteriors(audio_path), **kw)
