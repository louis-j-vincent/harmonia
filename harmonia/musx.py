"""music-x-lab frame posteriors + beat-grid re-decode.

COPIED 2026-07-30 from harmonia/models/musx_redecode.py (feat/minimal-pipeline
rebuild) with three changes only:
  * ``musx_dir()`` inlined (was harmonia.models.musx_bass.musx_dir) — resolves
    the vendored ISMIR2019 clone via ``harmonia.settings.SETTINGS.musx_dir``
    (itself ``HARMONIA_MUSX_DIR`` or ``third_party/musx_ismir2019``).
  * ``REPO``-relative paths dropped (refactor, 2026-09-14): every path here
    now resolves from ``harmonia.settings``/``harmonia.cache``, never from
    the module's own file location (checklist item 9).
  * the old pipeline's env-flag helpers (``enabled``/``latency_grid_from_env``)
    dropped — the pipeline's orchestration decides, not env vars.

The science: beat-grid re-decode, boundaries land exactly on OUR beats (see
the original module's header for the measurement record, +2.20 pp
partial-credit). The per-song LATENCY SEARCH that came with it was removed
on 2026-09-15 (audit, accepted by Louis): measured on 46 songs, musx's
posterior changes sit 23–46 ms BEFORE the Beat This! beat, never after —
the +46…+289 ms range it searched came from another beat tracker — and,
run again by the fold's template decode, the search aliased by one beat
period on 18 songs (chords written one beat early). One law: no latency.
History: ``docs/audit_2026-09-15_plan.md``, ``docs/musx_latency_ab.md``.
Posterior cache: via ``harmonia.cache`` (kind "musx_probs", stem-keyed, shared
with the old pipeline — same extractor, same layout: [triad(T,73), bass(T,13),
s7(T,4), s9(T,4), s11(T,3), s13(T,3)] on the 23.22 ms frame grid).
"""
from __future__ import annotations

import logging
import os
import sys
import threading
from pathlib import Path

import numpy as np

from harmonia import cache
from harmonia.settings import SETTINGS

logger = logging.getLogger(__name__)

# music-x-lab frame grid (settings.py: DEFAULT_SR=22050, DEFAULT_HOP_LENGTH=512).
MUSX_SR = 22050
MUSX_HOP = 512
FRAME_DT = MUSX_HOP / MUSX_SR          # 0.023219954648526078 s  (43.07 fps)

# The 5-fold ensemble shipped with the clone (== MODEL_NAMES in chord_recognition.py).
MODEL_NAMES = ['joint_chord_net_ismir_naive_v1.0_reweight(0.0,10.0)_s%d.best' % i
               for i in range(5)]

#: Viterbi change penalty — **INERTE sur le corps d'un morceau** (mesuré le
#: 2026-09-16, en cherchant à illustrer ce qu'elle fait). De 5 a 200, Ready rend
#: exactement les memes 86 accords.
#:
#: Pourquoi : le decodeur vendu n'applique `diff_trans_penalty` que la ou
#: `beat_arr[t] == 1` (`xhmm_ismir.py:120`), et `make_beat_arr` ci-dessous ne
#: laisse la valeur 1 que sur les images HORS de la grille de temps — avant le
#: premier temps et apres le dernier, 49 images sur 6671 pour Ready. Partout
#: ailleurs c'est 0 (aucun changement permis) ou 2/3/4, donc
#: `beat_trans_penalty`. Le vrai curseur de densite d'accords, c'est ce
#: triplet-la : 5/15/30 donne 92 accords, 40/80/200 en donne 78.
#:
#: Le commentaire d'origine annoncait « pooled optimum 40 ; LOSO-stable (40 sur
#: 6/7 morceaux, 55 sur le septieme) ». Aucune trace de cette etude dans docs/
#: ni archive/, corpus non nomme — et elle porterait de toute facon sur un
#: curseur qui ne bouge rien. Valeur gardee telle quelle : la changer ne peut
#: rien casser, mais rien ne justifie de la bouger non plus.
DEFAULT_PENALTY = 40.0

#: Bridge for callers that still want the raw folder (e.g. `span_rescore`
#: building its own filename) — the convention lives in `harmonia.cache`.
_PROB_CACHE = cache.folder("musx_probs")


# ── beat_arr: the transition structure the vendored decoder already supports ──

def make_beat_arr(n_frame: int, beat_times, downbeat_times=None,
                  beats_per_bar: int = 4, quarter_beats=None) -> np.ndarray:
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

    NOTE: ``downbeat_times`` is REQUIRED (2026-07-31). The old
    accuracy study measured graded ≈ flat on label overlap (0.6627 vs 0.6644),
    but placement is what the chart lives on: with a flat penalty, last-beat
    and next-downbeat cost the same, and at phrase turns (ambiguous frames)
    noise put chords one beat early (Louis's This Love report, 1:56 / 2:52).
    Downbeat-graded costs break exactly that tie toward the downbeat.
    """
    arr = np.ones(int(n_frame), dtype=np.int8)
    bt = np.asarray(beat_times, dtype=float)
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
    """The vendored music-x-lab clone (entry script + pretrained weights).

    Resolved once, by `harmonia.settings.SETTINGS.musx_dir` (env
    `HARMONIA_MUSX_DIR` or, since the 2026-09-14 refactor sprint 1,
    `third_party/musx_ismir2019` — the clone is a third-party dependency
    this module executes, not a piece of the old package).
    """
    d = SETTINGS.musx_dir
    if (d / "chord_recognition.py").exists() and \
       list((d / "cache_data").glob("*.sdict")):
        return d
    raise RuntimeError("harmonia.musx: music-x-lab clone not found "
                       f"(tried {d})")


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
    """'mps' si disponible, sinon 'cpu'. Coupe-circuit SETTINGS.musx_device."""
    global _DEVICE
    if _DEVICE is None:
        want = SETTINGS.musx_device.strip().lower()
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

    Cached (stem-keyed, like ``musx_bass``) via ``harmonia.cache`` (kind
    "musx_probs") when ``cache_dir`` is None, else at the given directory
    exactly as before.  A song costs ~3–9 MB compressed.
    """
    audio_path = Path(audio_path).resolve()
    names = ("triad", "bass", "s7", "s9", "s11", "s13")
    explicit_dir = Path(cache_dir) if cache_dir is not None else None
    if use_cache:
        if explicit_dir is not None:
            cfile = explicit_dir / f"{audio_path.stem}.npz"
            if cfile.exists():
                z = np.load(cfile)
                return [z[n] for n in names]
        else:
            d = cache.load_npz("musx_probs", audio_path)
            if d is not None:
                return [d[n] for n in names]

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
        if explicit_dir is not None:
            explicit_dir.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(explicit_dir / f"{audio_path.stem}.npz",
                               **dict(zip(names, probs)))
        else:
            cache.save_npz("musx_probs", audio_path, **dict(zip(names, probs)))
    return probs


# ── le CQT, et l'inférence à partir d'un CQT fabriqué ───────────────────────
# Louis, 2026-08-08, après avoir écouté les quatre agrégations possibles :
# « les CQT moyennés ça marche très bien (additions en amont) ». Empiler les
# répétitions AVANT le modèle, dans le domaine spectre, demande deux choses
# que le module ne savait pas faire : rendre le CQT d'une chanson, et faire
# tourner l'ensemble de réseaux sur un CQT qu'on a soi-même construit.

#: Bridge for callers that want the raw folder — see `_PROB_CACHE` above.
_CQT_CACHE = cache.folder("musx_cqt")


def song_cqt(audio_path: Path | str, *, use_cache: bool = True) -> np.ndarray:
    """Le CQT que musx mange, sur la MÊME grille que frame_posteriors.

    Cache disque (~2 Mo par morceau) via `harmonia.cache` (kind "musx_cqt"),
    clé par stem comme les postérieures — et avec le même défaut assumé : un
    fichier remplacé sous le même nom lit un cache périmé, videz le dossier
    `musx_cqt` après.
    """
    audio_path = Path(audio_path).resolve()
    if use_cache:
        d = cache.load_npz("musx_cqt", audio_path)
        if d is not None:
            return d["cqt"]
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
        cache.save_npz("musx_cqt", audio_path, cqt=cqt.astype(np.float32))
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

      * `bars._segment_confidence`, on the raw per-song posteriors;
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
        from harmonia.labels import parse_root
        col = 1 + (fam - 1) * 12 + parse_root(root_s)
    return float(triad[a:b, col].mean())


def _decoder(penalty: float, beat_trans_penalty=(15.0, 45.0, 100.0),
             chord_dict: str = "submission"):
    from extractors.xhmm_ismir import XHMMDecoder
    return XHMMDecoder(template_file=f'data/{chord_dict}_chord_list.txt',
                       diff_trans_penalty=float(penalty),
                       beat_trans_penalty=tuple(beat_trans_penalty))


def _tags_to_lab(tags: list[str]) -> list[tuple[float, float, str]]:
    """Frame tags -> (t0, t1, label) segments on the 23.22 ms grid."""
    out, last, n = [], 0, len(tags)
    for i in range(n):
        if i + 1 == n or tags[i + 1] != tags[i]:
            out.append((last * FRAME_DT, (i + 1) * FRAME_DT, tags[i]))
            last = i + 1
    return out


def redecode(beat_times, probs: list[np.ndarray], *, downbeat_times,
             penalty: float = DEFAULT_PENALTY,
             beats_per_bar: int = 4,
             beat_trans_penalty=(15.0, 45.0, 100.0),
             chord_dict: str = "submission",
             quarter_beats=None,
             ) -> list[tuple[float, float, str]]:
    """Beat-aware re-decode -> [(t0, t1, label)]; boundaries land exactly on
    ``beat_times``.

    No latency compensation (2026-09-15, module docstring): the legal
    transitions sit on the beats themselves. What this does NOT solve: a
    chord change played ahead of the beat (an anticipation, measured at
    about −220 ms on ``gbO7qQliXT8``) still lands on the nearest legal
    beat, before or after — the grid has no half-beat slot.
    ``quarter_beats`` — see ``make_beat_arr``: None (half-bar only), "all",
    or beat indices where a quarter-bar change is allowed.
    """
    n_frame = int(probs[0].shape[0])
    plist = [np.asarray(p, dtype=np.float64) for p in probs]
    with _InMusxDir():
        hmm = _decoder(penalty, beat_trans_penalty, chord_dict)
        arr = make_beat_arr(n_frame, beat_times, downbeat_times,
                            beats_per_bar=beats_per_bar,
                            quarter_beats=quarter_beats)
        tags = hmm.decode(plist, arr)
    lab = _tags_to_lab(tags)
    logger.info("musx_redecode: %d segments", len(lab))
    return lab
