"""Installe les moteurs « téléphone » dans `harmonia.backends` (tâche P2c).

UN SEUL APPEL, `install()`, AVANT d'importer le pipeline. C'est le seul
endroit du portage qui sait qu'on n'est pas sur le Mac — `harmonia/` n'a
aucun `if android:`, et `harmonia/settings.py` reste le seul lecteur de
`os.environ` (aucune variable ne choisit de moteur ici, c'est du code).

    from mobile.harmonia_mobile import backends_mobile
    backends_mobile.install()                 # les 5 briques numpy/.pte
    from harmonia import pipeline              # ... puis seulement

LES CINQ BRIQUES, et d'où elles viennent :

| fente             | moteur mobile                              | rapport |
|-------------------|--------------------------------------------|---------|
| `beats`           | mel numpy + `.pte` ExecuTorch + post numpy | `t1c_beat_this_complet.md` |
| `musx_features`   | CQT hybride numpy                          | `p2b_cqt_musx.md` |
| `musx_posteriors` | ChordNet numpy (5 folds)                   | `p0_t1_executorch.md` + T1b |
| `nnls_bothchroma` | NNLS-chroma numpy (FISTA batché)           | `p0_t2b_nnls_vitesse.md` |
| `find_peaks`      | maxima locaux numpy                        | (ici, `_find_peaks_numpy`) |

LE DÉCODAGE AUDIO EST UN CHOIX EXPLICITE, `audio=` :

* `"pcm_dump"` (DÉFAUT sur le Mac) — relit le PCM que le pipeline du Mac
  produit AUJOURD'HUI, site d'appel par site d'appel, figé une fois par
  `mobile/tools/dump_reference_pcm.py` sous `.venv`. Pourquoi pas librosa
  directement : les deux venvs n'ont PAS la même librosa (0.11.0 côté
  `.venv`, avec le repli `audioread`/CoreAudio ; 1.0.0 côté
  `~/harmonia/.venv-mobile`, qui a supprimé ce repli et ne sait plus ouvrir
  un `.m4a`) — « décoder avec librosa des deux côtés » n'aurait donc pas été
  le même décodage. Le décodage est ainsi tenu CONSTANT et la parité mesure
  les cinq briques de calcul, rien d'autre. Le décodeur Android a son propre
  rapport (`p2a_decodeur_audio.md` : décalage 0 échantillon, diff max 0,0 en
  tête, +348 à +873 échantillons en QUEUE de fichier).
* `"reference"` — `librosa.load` en direct (utilisable seulement dans un venv
  où librosa sait ouvrir le fichier, donc PAS `.venv-mobile`).
* `"ffmpeg"` — décodage par sous-processus `ffmpeg -f f32le`, pour MESURER
  ce que coûte un décodeur différent de la référence, sans device Android.
* `"mediacodec"` — le vrai chemin Android (`mobile_audio.load_audio`, pont
  Chaquopy vers `AudioDecoder.kt`). N'existe que sur le téléphone.

CE QUE CE MODULE NE RÉSOUT PAS :
* `beats` a besoin du runtime ExecuTorch. Sur le Mac, le paquet pip
  `executorch` importe `torch` (vérifié : `import executorch.runtime` ->
  `'torch' in sys.modules`) ; sur Android le runtime est une bibliothèque
  native, sans torch. La preuve « aucun paquet interdit » (`mobile/tools/
  purity_check.py`) porte donc sur les QUATRE briques numpy, pas sur beats.
* `musx_posteriors` peut retomber sur les réseaux torch, mais SEULEMENT si
  on le demande explicitement (`musx="torch_reference"`) — jamais en
  silence, parce qu'un repli muet vers torch donnerait une parité parfaite
  en ne mesurant rien (CLAUDE.md, « silent fallbacks are URGENT »).
"""
from __future__ import annotations

import subprocess
import wave
from pathlib import Path

import numpy as np

MODELS = Path(__file__).resolve().parents[1] / "models"
_FOLDS = [MODELS / f"musx_fold{i}_weights.npz" for i in range(5)]

_weights_cache: list | None = None
_audio_impl = None


# ── décodage ────────────────────────────────────────────────────────────────

def audio_librosa_reference(path, sr: int, duration: float | None = None,
                            variant: str = ""):
    """Le décodeur de RÉFÉRENCE du Mac. Nommé, jamais implicite : il importe
    `librosa`, interdit sur le téléphone — sa seule raison d'être est de tenir
    le décodage constant pendant qu'on mesure les cinq briques de calcul."""
    import librosa
    y, _ = librosa.load(str(path), sr=int(sr), mono=True, duration=duration)
    return np.asarray(y, dtype=np.float32)


PCM_DIR = Path(__file__).resolve().parents[1] / "tools" / "parity_out" / "pcm"


def audio_pcm_dump(path, sr: int, duration: float | None = None,
                   variant: str = ""):
    """Lit le PCM de référence figé par `mobile/tools/dump_reference_pcm.py`.

    C'est le mode de parité P2c sur le Mac : le décodage est tenu CONSTANT
    (octet pour octet celui du pipeline d'aujourd'hui, site d'appel par site
    d'appel), donc un écart de chart ne peut venir que des cinq briques de
    calcul. Aucun import interdit : `np.load` et rien d'autre."""
    stem = Path(path).stem
    tag = f"{int(sr)}" if duration is None else f"{int(sr)}w{int(duration)}"
    tag += variant
    f = PCM_DIR / f"{stem}__{tag}.npy"
    if not f.exists():
        raise RuntimeError(
            f"PCM de référence absent: {f} — lancer "
            f".venv/bin/python mobile/tools/dump_reference_pcm.py")
    return np.load(f).astype(np.float32, copy=False)


def audio_ffmpeg(path, sr: int, duration: float | None = None,
                 variant: str = ""):
    """Décodage par sous-processus (stdlib seulement côté Python). Sert à
    mesurer l'effet d'un décodeur DIFFÉRENT de la référence sur le Mac."""
    cmd = ["ffmpeg", "-v", "error", "-i", str(path)]
    if duration is not None:
        cmd += ["-t", str(duration)]
    cmd += ["-ac", "1", "-ar", str(int(sr)), "-f", "f32le", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    return np.frombuffer(raw, dtype=np.float32).copy()


def audio_mediacodec(path, sr: int, duration: float | None = None,
                     variant: str = ""):
    """Le vrai chemin Android — `mobile_audio.load_audio` (P2a)."""
    import mobile_audio
    y, _ = mobile_audio.load_audio(str(path), int(sr))
    y = np.asarray(y, dtype=np.float32)
    if duration is not None:
        y = y[:int(round(duration * sr))]
    return y


def _load(path, sr, duration=None, variant: str = ""):
    """`variant` ne sert QU'au mode `pcm_dump` : il dit quel décodage de
    référence rejouer quand le pipeline du Mac en emploie plusieurs sur le
    MÊME fichier au MÊME taux. C'est le cas à 22050 Hz — `beat_this` reçoit un
    transcodage ffmpeg, musx reçoit `librosa.load`. Sur le téléphone il n'y a
    qu'un décodeur (MediaCodec) et ce paramètre n'existe plus."""
    return _audio_impl(path, sr, duration, variant)


def duration_seconds(path) -> float:
    """La durée, sans ffprobe : la longueur du signal décodé.

    Ce que ça ne résout PAS : sur Android `MediaExtractor.KEY_DURATION` est
    moins cher et devrait passer devant (P2a a mesuré qu'il rend une valeur
    LÉGÈREMENT PLUS GRANDE que le PCM réellement décodé) ; ici on prend la
    mesure qui ne peut pas mentir sur ce qu'on va analyser."""
    return float(len(_load(path, 22050)) / 22050.0)


def slice_to_wav(path, seconds: float, out_wav, sr: int) -> None:
    """Les `seconds` premières secondes en wav mono 16 bits — comme `ffmpeg`,
    dont le wav par défaut est déjà du `pcm_s16le`. Sert au SEUL chart de
    tête (un aperçu, `pipeline._head_chart`)."""
    y = _load(path, sr, duration=seconds)
    pcm = np.clip(np.asarray(y, dtype=np.float32), -1.0, 1.0)
    pcm = (pcm * 32767.0).astype("<i2")
    with wave.open(str(out_wav), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(sr))
        w.writeframes(pcm.tobytes())


# ── les cinq briques ────────────────────────────────────────────────────────

def beats_pte(audio_path):
    from mobile.harmonia_mobile.beats_infer import beats_from_audio
    # variant="beats" : sur le Mac, Beat This! est nourri par un transcodage
    # ffmpeg, PAS par `librosa.load` (aucun lecteur de beat_this n'ouvre un
    # .m4a sur cette machine). Les deux décodages diffèrent assez pour changer
    # le nombre de battues (348 vs 345 sur Stand By Me) — donc la parité de la
    # PLOMBERIE doit rejouer le décodage de CE site d'appel-là.
    y = _load(audio_path, 22050, variant="beats")
    return beats_from_audio(y)


def musx_features_numpy(audio_path) -> np.ndarray:
    from mobile.harmonia_mobile.cqt_musx import song_cqt_288
    y = _load(audio_path, 22050)
    return np.asarray(song_cqt_288(y, sr=22050), dtype=np.float32)


def _weights() -> list:
    global _weights_cache
    if _weights_cache is None:
        from mobile.harmonia_mobile.musx_net import load_weights_npz
        missing = [p.name for p in _FOLDS if not p.exists()]
        if missing:
            raise RuntimeError(
                f"poids musx numpy absents ({missing}) — lancer "
                f"mobile/tools/dump_musx_weights.py")
        _weights_cache = [load_weights_npz(p) for p in _FOLDS]
    return _weights_cache


def musx_posteriors_numpy(cqt) -> list:
    from mobile.harmonia_mobile.musx_net import predict_ensemble
    return predict_ensemble(np.asarray(cqt, dtype=np.float32), _weights())


def musx_torch_reference(cqt) -> list:
    """Les 5 réseaux TORCH, derrière la même couture. Explicitement nommé :
    c'est le moteur du Mac, autorisé uniquement pour isoler le reste de la
    plomberie quand le port numpy n'est pas encore là."""
    from harmonia.musx import _run_nets
    return _run_nets(np.asarray(cqt))


def nnls_bothchroma_numpy(audio_path):
    from mobile.harmonia_mobile.nnls_chroma import extract_bothchroma_numpy
    y = _load(audio_path, 44100)
    return extract_bothchroma_numpy(np.asarray(y, dtype=np.float32), 44100)


# ── find_peaks en numpy ─────────────────────────────────────────────────────

def _local_maxima_1d(x: np.ndarray) -> np.ndarray:
    """Les maxima locaux, plateaux compris — port direct de
    `scipy.signal._peak_finding_utils._local_maxima_1d` : un plateau de
    valeurs égales encadré de valeurs plus basses rend son point MILIEU
    (`(bord_gauche + bord_droit) // 2`), pas son premier échantillon."""
    out = []
    i, i_max = 1, len(x) - 1
    while i < i_max:
        if x[i - 1] < x[i]:
            i_ahead = i + 1
            while i_ahead < i_max and x[i_ahead] == x[i]:
                i_ahead += 1
            if x[i_ahead] < x[i]:
                out.append((i + i_ahead - 1) // 2)
                i = i_ahead
        i += 1
    return np.asarray(out, dtype=np.intp)


def _find_peaks_numpy(x, *, height, distance):
    """`scipy.signal.find_peaks(x, height=…, distance=…)`, indices seulement.

    Même ORDRE que scipy — hauteur d'abord, distance ensuite — et même règle
    de distance (le plus haut se sert le premier, puis on efface tout ce qui
    est à moins de `distance`). Vérifié par `mobile/tools/purity_check.py`.
    """
    x = np.asarray(x, dtype=np.float64)
    peaks = _local_maxima_1d(x)
    if height is not None and len(peaks):
        peaks = peaks[x[peaks] >= height]
    if distance is not None and len(peaks) > 1:
        # `np.argsort` SANS `kind=` : c'est l'appel exact de
        # `scipy.signal._peak_finding_utils._select_by_peak_distance`. Un tri
        # STABLE cassait les ex-æquo dans l'autre sens et donnait 84 vecteurs
        # différents sur 300 (mesuré, `mobile/tools/check_find_peaks.py`) —
        # sur des signaux à plateaux, les ex-æquo sont la règle, pas le cas
        # tordu.
        prio = np.argsort(x[peaks])
        keep = np.ones(len(peaks), dtype=bool)
        for i in range(len(peaks) - 1, -1, -1):
            j = int(prio[i])
            if not keep[j]:
                continue
            k = j - 1
            while k >= 0 and peaks[j] - peaks[k] < distance:
                keep[k] = False
                k -= 1
            k = j + 1
            while k < len(peaks) and peaks[k] - peaks[j] < distance:
                keep[k] = False
                k += 1
        peaks = peaks[keep]
    return peaks


# ── installation ────────────────────────────────────────────────────────────

_AUDIO = {"pcm_dump": audio_pcm_dump,
          "reference": audio_librosa_reference,
          "ffmpeg": audio_ffmpeg,
          "mediacodec": audio_mediacodec}


def install(*, audio: str = "pcm_dump", musx: str = "numpy") -> None:
    """Pose les huit fentes. `audio` et `musx` sont des CHOIX, pas des replis."""
    global _audio_impl, _weights_cache
    if audio not in _AUDIO:
        raise KeyError(f"audio={audio!r} inconnu (connus: {sorted(_AUDIO)})")
    _audio_impl = _AUDIO[audio]
    from harmonia.backends import BACKENDS
    BACKENDS.install(
        load_audio=_load,
        duration_seconds=duration_seconds,
        slice_to_wav=slice_to_wav,
        beats=beats_pte,
        musx_features=musx_features_numpy,
        musx_posteriors=(musx_posteriors_numpy if musx == "numpy"
                         else musx_torch_reference),
        nnls_bothchroma=nnls_bothchroma_numpy,
        find_peaks=_find_peaks_numpy,
    )
