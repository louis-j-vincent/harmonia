"""Les huit endroits où le calcul peut changer de moteur — et rien d'autre.

POURQUOI CE MODULE EXISTE (portage mobile, tâche P2c, 2026-09-23).
`harmonia/` est le code du Mac ET du téléphone. Sur Android il n'y a ni
`ffmpeg`, ni `librosa`, ni `vamp`, ni `scipy`, ni `torch` — mais le pipeline,
lui, ne change pas d'un iota : mêmes mesures, mêmes accords, même chart.
Plutôt que semer des `if android:` dans les modules feuilles (ce qui aurait
fait deux comportements à lire dans chaque fonction), on nomme ICI les huit
appels qui touchent une bibliothèque native, et on les rend REMPLAÇABLES.

CE QUE CE MODULE NE FAIT PAS, et c'est le point : il ne contient AUCUNE
implémentation par défaut. Chaque module feuille garde la sienne, inchangée,
dans son propre fichier ; il l'exécute par `BACKENDS.call("<fente>", sa_
fonction_actuelle, ...)`. Quand personne n'a rien installé, `call` appelle
exactement cette fonction-là — le comportement Mac ne peut donc pas dériver
par omission, parce qu'il n'y a rien à ré-écrire pour l'obtenir.

LE CHOIX DU MOTEUR SE FAIT PAR DU CODE, PAS PAR L'ENVIRONNEMENT.
`harmonia/settings.py` reste le seul lecteur de `os.environ` (décision du
refactor 2026-09-14). Le point d'entrée Android
(`mobile/android/app/src/main/python/mobile_entry.py`) appelle
`mobile.harmonia_mobile.backends_mobile.install()` AVANT d'importer le
pipeline ; rien d'autre ne parle de plateforme.

LES HUIT FENTES, et le contrat de chacune :

    load_audio(path, sr, duration=None) -> np.ndarray float32, mono, (n,)
        Le décodage. Référence Mac = `librosa.load(..., mono=True)` (qui sur
        cette machine passe par CoreAudio via `audioread.macca`, PAS ffmpeg —
        mesuré en P2a). Android = `mobile_audio.load_audio` (MediaCodec).
    duration_seconds(path) -> float
        La durée. Référence = `ffprobe`. Android = `MediaExtractor`.
    slice_to_wav(path, seconds, out_wav, sr) -> None
        Écrit les `seconds` premières secondes en wav mono `sr`. Sert au SEUL
        chart de tête (`pipeline._head_chart`), un aperçu. Référence = ffmpeg.
    beats(path) -> (beats, downbeats), en secondes
        Beat This!. Référence = le paquet `beat_this` (torch) avec son repli
        ffmpeg. Android = mel numpy + `.pte` ExecuTorch + post-traitement
        numpy (`mobile/harmonia_mobile/beats_infer.py`).
    musx_features(path) -> np.ndarray (n_frames, 288) float32
        Le CQT hybride que mange ChordNet. Référence = le clone vendu
        (`extractors/cqt.py:CQTV2`, donc `librosa.core.hybrid_cqt`).
        Android = `mobile/harmonia_mobile/cqt_musx.py`.
    musx_posteriors(cqt) -> list de 6 np.ndarray float32
        [triad(T,73), bass(T,13), s7(T,4), s9(T,4), s11(T,3), s13(T,3)],
        moyennés sur les 5 folds. Référence = les 5 réseaux torch.
        Android = `mobile/harmonia_mobile/musx_net.py` (numpy).
    nnls_bothchroma(path) -> (arr (T,24) index 0 = A, times (T,))
        Le chroma NNLS. Référence = le plugin VAMP natif.
        Android = `mobile/harmonia_mobile/nnls_chroma.py`.
    find_peaks(x, height, distance) -> indices (np.ndarray)
        Les pics locaux de `sections/similarity._peaks`. Référence =
        `scipy.signal.find_peaks`. Android = équivalent numpy.

CE QUE CE MODULE NE RÉSOUT PAS (à dire, pas à découvrir plus tard) :
* il ne garantit RIEN sur la parité numérique des remplaçants — c'est le
  travail des rapports de brique (`docs/portage_mobile/p2*.md`) et du
  `mobile/tools/parity_pipeline.py` qui rejoue les 3 morceaux de seed ;
* il ne couvre pas les appels natifs HORS chemin critique : l'import YouTube
  (`server/youtube.py`), le ré-encodage micro/upload (`server/routes/
  analyze.py:_ffmpeg`), les quatre courbes de diagnostic de `debut.pistes`
  (`_spectre`, ffmpeg) et SongFormer (éteint, `SETTINGS.sections="aucune"`).
  Chacun a son propre plan dans `docs/portage_mobile/`.
"""
from __future__ import annotations

import time


class Backends:
    """Huit fentes, `None` par défaut = « le module feuille fait comme avant »."""

    SLOTS = ("load_audio", "duration_seconds", "slice_to_wav", "beats",
             "musx_features", "musx_posteriors", "nnls_bothchroma",
             "find_peaks")

    def __init__(self) -> None:
        for s in self.SLOTS:
            setattr(self, s, None)
        #: secondes cumulées par fente depuis le dernier `reset_timings()`.
        #: Mesuré ici et pas chez l'appelant pour que le chemin Mac et le
        #: chemin mobile soient chronométrés au MÊME endroit.
        self.timings: dict[str, float] = {}

    def install(self, **fentes) -> None:
        """Pose des remplaçants. Un nom inconnu lève — une faute de frappe qui
        laisserait silencieusement le moteur du Mac en place sur le téléphone
        serait exactement le bug de calibration muet de CLAUDE.md règle 1."""
        for k, v in fentes.items():
            if k not in self.SLOTS:
                raise KeyError(f"fente inconnue: {k!r} (connues: {self.SLOTS})")
            setattr(self, k, v)

    def reset(self) -> None:
        """Remet tout sur le moteur du Mac (utilisé par les tests)."""
        for s in self.SLOTS:
            setattr(self, s, None)
        self.reset_timings()

    def reset_timings(self) -> None:
        self.timings = {}

    def call(self, slot: str, default, *args, **kw):
        """Exécute la fente `slot`, ou `default` si personne ne l'a remplacée."""
        fn = getattr(self, slot) or default
        t0 = time.perf_counter()
        try:
            return fn(*args, **kw)
        finally:
            self.timings[slot] = round(
                self.timings.get(slot, 0.0) + time.perf_counter() - t0, 3)


#: L'unique registre du processus.
BACKENDS = Backends()
