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
import threading
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
                  downbeat_times=None, beats_per_bar: int = 4,
                  quarter_beats=None) -> np.ndarray:
    """Mirror of ``XHMMDecoder._XHMMDecoder__get_beat_arr``, fed OUR beat grid.

    Semantics consumed by the clone's ``decode``:
      0 -> no chord change may occur at this frame;
      1 -> change costs ``diff_trans_penalty``;
      2 / 3 / 4 -> change costs ``beat_trans_penalty[0/1/2]``
                   (downbeat / mid-bar / other beat).

    ``quarter_beats`` opens the quarter-bar level (feat/quarter-bar, 2026-08-07):
      None  -> half-bar only, the shipped behaviour (every grade-4 beat zeroed);
      "all" -> every beat may carry a change, at ``beat_trans_penalty[2]``;
      iterable of beat INDICES (into ``beat_times``) -> only those beats keep
      grade 4 — the targeted mode, fed by a more-chords-here detector.

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
    # HALF-BAR ONLY (Louis, 2026-08-01): « les demi-barres permettent d'avoir
    # un premier niveau de fiabilité » — at this reliability level, chord
    # changes are STRUCTURALLY restricted to bar and half-bar positions;
    # quarter-bar refinement is a LATER pass, run only once the half-bar
    # level is validated. Implemented below by zeroing every non-half-bar
    # beat after the downbeat grading (0 = no transition allowed).
    # RULING LIFTED 2026-08-07 (Louis: « on ne met plus de restrictions sur
    # la granularité ») — the LIVE pipeline and the folding template decode
    # now pass quarter_beats="all", so every beat is legal at the graded
    # cost; this function's None default keeps the restricted behaviour for
    # callers that still want it (HARMONIA_QUARTER_BAR=off, experiments).
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
    if quarter_beats is None:
        arr[arr == 4] = 0    # quarter-bar beats: forbidden at this level
    elif isinstance(quarter_beats, str):
        if quarter_beats != "all":
            raise ValueError(f"quarter_beats: unknown mode {quarter_beats!r}")
    else:
        qi = np.asarray(sorted({int(b) for b in quarter_beats}), dtype=int)
        qi = qi[(qi >= 0) & (qi < len(fr))]
        f_q = fr[qi]
        f_q = f_q[(f_q >= 0) & (f_q < n_frame)]
        allowed = np.zeros(n_frame, dtype=bool)
        allowed[f_q] = True
        arr[(arr == 4) & ~allowed] = 0
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


# ── le GPU d'Apple, mesuré bit-à-bit identique ──────────────────────────────
# 2026-08-18. L'inférence des 5 réseaux est l'étape la plus chère du chemin
# froid (14,4 s sur les 9 min de h_D3VFfhvs4, contre 3,9 s de CQT et 11,8 s de
# battues). Elle tourne sur MPS sans rien changer à la sortie :
#
#   morceau            durée   CPU      MPS            accords redécodés
#   0DdCoNbbRvQ         252 s   6,20 s   1,75 s (x3,5)  100/100 IDENTIQUES
#   autumn_leaves       422 s  10,96 s   2,52 s (x4,3)  160/160 IDENTIQUES
#   4JkIs37a2JE         235 s   5,96 s   1,68 s (x3,6)  163/163 IDENTIQUES
#   DksSPZTZES0         290 s   8,06 s   2,00 s (x4,0)  118/118 IDENTIQUES
#
# « identiques » au sens fort : argmax des postérieures 100 % égal, |Δ| max
# 0,0000, et les segments du re-décodage égaux label ET frontière.
#
# NE PAS étendre à Beat This! : mesuré le même jour sur le même morceau,
# 7,3 s en CPU contre 67,5 s en MPS (x9 PLUS LENT, battues identiques à 0 ms).
# Le tracker reste sur CPU — ce qui tombe bien, les deux étapes peuvent alors
# tourner en parallèle sur deux unités différentes.
#
# Un seul obstacle technique : `ChordNet.init_hidden` fabrique h0/c0 sur CPU en
# dur (`torch.zeros(...)`, cuda ou rien), donc le LSTM reçoit un état caché
# CPU pour une entrée MPS — « Placeholder storage has not been allocated on MPS
# device ». On le corrige ici, dans NOTRE module, sans éditer le clone.
_DEVICE: str | None = None


def _device() -> str:
    """'mps' si disponible, sinon 'cpu'. Coupe-circuit HARMONIA_MUSX_DEVICE."""
    global _DEVICE
    if _DEVICE is None:
        want = os.environ.get("HARMONIA_MUSX_DEVICE", "auto").strip().lower()
        if want in ("cpu", "mps"):
            _DEVICE = want
        else:
            import torch
            _DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
        logger.info("musx: inférence sur %s", _DEVICE)
    return _DEVICE


def _patch_init_hidden(chordnet_module) -> None:
    """h0/c0 sur le device des poids, au lieu de CPU en dur."""
    import torch

    def init_hidden(self, batch_size, hidden_dim):
        dev = next(self.parameters()).device
        z = torch.zeros(2, batch_size, hidden_dim // 2, device=dev)
        return (z, z.clone())

    chordnet_module.ChordNet.init_hidden = init_hidden


_NETS: list | None = None
_NETS_LOCK = threading.Lock()


def _nets() -> list:
    """Les 5 réseaux, chargés UNE FOIS par processus, déjà sur le bon device.

    Ils étaient reconstruits à chaque appel (`NetworkInterface(...)` ×5, 0,7 s
    de disque) — supportable sur CPU, ruineux sur MPS : chaque copie laisse ses
    tampons dans le cache d'allocation du GPU, que rien ne rend. Mesuré le
    2026-08-18 en rechargeant à chaque fois : la 3e analyse d'un même processus
    mettait 26 s là où la 1re en mettait 6,6. Les poids ne changent jamais ;
    on les garde.
    """
    global _NETS
    if _NETS is None:
        with _InMusxDir():
            from mir.nn.train import NetworkInterface
            from chordnet_ismir_naive import ChordNet
            import chordnet_ismir_naive as _cn
            _patch_init_hidden(_cn)
            dev = _device()
            _NETS = [NetworkInterface(ChordNet(None), n, load_checkpoint=False)
                     .net.eval().to(dev) for n in MODEL_NAMES]
        logger.info("musx: %d réseaux chargés sur %s", len(_NETS), _device())
    return _NETS


def _run_nets(cqt: np.ndarray) -> list[np.ndarray]:
    """Les 6 flux de postérieures, moyennés sur les 5 folds. UN SEUL chemin.

    `frame_posteriors` et `posteriors_from_cqt` passaient tous deux par
    `NetworkInterface.inference`, qui appelle `init_settings(False)` — lequel
    fait `self.cpu()` et ramènerait donc les poids sur CPU à chaque appel. On
    place les modules nous-mêmes et on appelle `ChordNet.inference` directement.

    Le verrou sérialise deux analyses simultanées (deux onglets, deux jobs) :
    les modules sont partagés, et rien ne dit qu'un GPU Metal aime deux
    inférences concurrentes sur les mêmes poids.
    """
    import torch
    dev = _device()
    with _NETS_LOCK:
        nets = _nets()
        x = torch.tensor(np.asarray(cqt), dtype=torch.float32).to(dev)
        acc = None
        with torch.no_grad():
            for m in nets:
                out = m.inference(x)
                acc = list(out) if acc is None else [a + b for a, b in zip(acc, out)]
        del x
        if dev == "mps":
            torch.mps.empty_cache()
    return [(a / len(MODEL_NAMES)).astype(np.float32) for a in acc]


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
        from extractors.cqt import CQTV2
        entry = DataEntry()
        entry.prop.set('sr', MUSX_SR)
        entry.prop.set('hop_length', MUSX_HOP)
        entry.append_file(str(audio_path), io.MusicIO, 'music')
        entry.append_extractor(CQTV2, 'cqt')
        cqt = np.asarray(entry.cqt)
    probs = _run_nets(cqt)
    if use_cache:
        cdir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, **dict(zip(names, probs)))
    return probs


# ── le CQT, et l'inférence à partir d'un CQT fabriqué ───────────────────────
# Louis, 2026-08-08, après avoir écouté les quatre agrégations possibles :
# « les CQT moyennés ça marche très bien (additions en amont) ». Empiler les
# répétitions AVANT le modèle, dans le domaine spectre, demande deux choses
# que le module ne savait pas faire : rendre le CQT d'une chanson, et faire
# tourner l'ensemble de réseaux sur un CQT qu'on a soi-même construit.

_CQT_CACHE = REPO / "data" / "cache" / "musx_cqt"


def song_cqt(audio_path: Path | str, *, use_cache: bool = True) -> np.ndarray:
    """Le CQT que musx mange, sur la MÊME grille que frame_posteriors.

    Cache disque (~2 Mo par morceau) clé par stem, comme les postérieures —
    et avec le même défaut assumé : un fichier remplacé sous le même nom lit
    un cache périmé, videz `data/cache/musx_cqt/` après.
    """
    audio_path = Path(audio_path).resolve()
    cache = _CQT_CACHE / f"{audio_path.stem}.npz"
    if use_cache and cache.exists():
        return np.load(cache)["cqt"]
    with _InMusxDir():
        from mir import io, DataEntry
        from extractors.cqt import CQTV2
        entry = DataEntry()
        entry.prop.set('sr', MUSX_SR)
        entry.prop.set('hop_length', MUSX_HOP)
        entry.append_file(str(audio_path), io.MusicIO, 'music')
        entry.append_extractor(CQTV2, 'cqt')
        cqt = np.asarray(entry.cqt)
    if use_cache:
        _CQT_CACHE.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache, cqt=cqt.astype(np.float32))
    return cqt


def posteriors_from_cqt(cqt: np.ndarray) -> list[np.ndarray]:
    """Les 6 flux de postérieures pour un CQT quelconque — y compris un CQT
    MOYENNÉ sur plusieurs répétitions, ce qui est tout l'intérêt."""
    return _run_nets(np.asarray(cqt))


# ── label confidence ────────────────────────────────────────────────────────
# musx triad-plane family index (1-based; see frame_posteriors above) for each
# label quality the decoder can emit.
TRIAD_FAMILY = {
    "maj": 1, "maj7": 1, "7": 1, "9": 1, "maj9": 1, "13": 1,
    "maj/3": 1, "maj/5": 1, "maj/b7": 1, "maj/2": 1,
    "min": 2, "min7": 2, "min9": 2, "min/b3": 2, "min/5": 2,
    "min/b7": 2, "min/2": 2,
    "sus4": 3, "sus4(b7)": 3, "11": 3,
    "sus2": 4,
    "dim": 5, "dim7": 5, "hdim7": 5,
    "aug": 6,
}


def label_confidence(triad: np.ndarray, t0: float, t1: float,
                     label: str) -> float:
    """Mean triad-plane posterior of `label` over the frames [t0, t1).

    THE single definition of "how well does the evidence support this chord",
    deliberately shared by both places that need it:

      * `pipeline._segment_confidence`, on the raw per-song posteriors;
      * `folding._template_chords`, on the AVERAGED posteriors a fold decoded
        from.

    They must be the same quantity on the same scale. Measured 2026-08-01 on
    GuitarSet, when they were not: folded chords were the most accurate spans in
    the set (95.2% root vs 87.5%) while carrying the LOWEST confidences, because
    folding wrote `0.5 + 0.08*n_obs` — a repetition count — into the same field.
    Pooling the two scales cost 2 points of AUC (0.819 -> 0.800), which is the
    margin that decides whether `c` is usable in the LM-intervention rule at all.

    Returns 0.5 (neutral) for an unusable window or an unknown quality rather
    than guessing — a wrong confidence is worse than an uninformative one.
    """
    a = max(0, int(round(t0 / FRAME_DT)))
    b = min(triad.shape[0], int(round(t1 / FRAME_DT)))
    if b <= a:
        return 0.5
    if label == "N":
        col = 0
    else:
        root_s, _, qual = label.partition(":")
        fam = TRIAD_FAMILY.get(qual)
        if fam is None:
            return 0.5
        from harmonia_min.labels import parse_root
        col = 1 + (fam - 1) * 12 + parse_root(root_s)
    return float(triad[a:b, col].mean())


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
                n_frame: int, latency: float = 0.0,
                guard_frames: int = 0) -> float:
    """The decoder's own Viterbi objective for a decoded labelling.

    Used as the GT-FREE selector over candidate latencies: same observations,
    same emission model, only the legal-transition set moves.

    ``guard_frames`` (audit fix, 2026-08-02): when ``_tags_to_lab`` shifts a
    candidate's spans back by ``latency``, a span whose true start was
    negative gets clamped to ``t0=0`` — so re-adding ``latency`` here to
    recover the original frame index lands PAST the clamped span's real
    start.  The frames in that gap (``~latency`` worth, at the head of the
    file) are never written into ``tag`` and silently keep the array's
    default value, index 0.  Index 0 is ``names[0]``, which
    ``XHMMDecoder.__init_known_chord_names`` always prepends as the literal
    ``"N"`` (no-chord) tag — NOT the chord vocabulary's first data row
    (``C:min/b7`` in ``submission_chord_list.txt``); an earlier read of this
    code called it a "C:min/b7" penalty, which is the wrong tag identity.
    Either way, every candidate with ``latency > 0`` pays an emission cost on
    those head frames that candidate ``latency == 0`` never pays, which is an
    apples-to-oranges comparison. Fix: score every candidate over the SAME
    frame window by dropping ``guard_frames`` from both ends of the sum
    (``redecode`` sizes it once from ``max(latency_grid)`` so it is identical
    across the whole grid, including L=0). Measured impact (This Love,
    2026-08-02): 2-8 nats out of an 8000-22000 nat range across the grid —
    real but three orders of magnitude too small to be why L=0 wins; see
    ``docs/musx_latency_ab.md``.
    """
    idx = {n: i for i, n in enumerate(names)}
    tag = np.zeros(n_frame, dtype=int)
    for t0, t1, s in lab:
        if s not in idx:
            return float("-inf")
        a = max(0, int(round((t0 + latency) / FRAME_DT)))
        b = min(n_frame, int(round((t1 + latency) / FRAME_DT)))
        tag[a:b] = idx[s]
    lo = max(0, int(guard_frames))
    hi = max(lo, n_frame - int(guard_frames))
    em = float(logprob[np.arange(lo, hi), tag[lo:hi]].sum())
    return em - float(penalty) * max(0, len(lab) - 1)


def redecode(beat_times, probs: list[np.ndarray], *, downbeat_times,
             penalty: float = DEFAULT_PENALTY,
             latency_grid=DEFAULT_LATENCY_GRID,
             beats_per_bar: int = 4,
             beat_trans_penalty=(15.0, 45.0, 100.0),
             chord_dict: str = "submission",
             quarter_beats=None,
             ) -> tuple[list[tuple[float, float, str]], float]:
    """Beat-aware, latency-compensated re-decode -> (labels, chosen latency).

    Boundaries land exactly on ``beat_times``.  The latency is selected per song
    by maximising the decoder's own path log-likelihood — no ground truth.
    ``quarter_beats`` — see ``make_beat_arr``: None (half-bar only, shipped),
    "all", or beat indices where a quarter-bar change is allowed.
    """
    n_frame = int(probs[0].shape[0])
    plist = [np.asarray(p, dtype=np.float64) for p in probs]
    best = (float("-inf"), 0.0, [])
    # Same guard for every candidate in this grid (see path_loglik docstring):
    # sized off the LARGEST latency so no candidate — including L=0 — ever
    # scores frames another candidate is forced to skip.
    guard = int(np.ceil(max((abs(float(L)) for L in latency_grid), default=0.0)
                        / FRAME_DT))
    with _InMusxDir():
        hmm = _decoder(penalty, beat_trans_penalty, chord_dict)
        names, logprob = hmm.get_chord_tag_obs(plist)
        for L in latency_grid:
            arr = make_beat_arr(n_frame, beat_times, L, downbeat_times,
                                beats_per_bar=beats_per_bar,
                                quarter_beats=quarter_beats)
            tags = hmm.decode(plist, arr)
            lab = _tags_to_lab(tags, L)
            ll = path_loglik(logprob, names, lab, penalty, n_frame, L,
                             guard_frames=guard)
            if ll > best[0]:
                best = (ll, float(L), lab)
    logger.info("musx_redecode: latency %.0f ms selected (loglik %.1f), %d segments",
                best[1] * 1000, best[0], len(best[2]))
    return best[2], best[1]
