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
import threading
import wave
from pathlib import Path

import numpy as np

#: Où sont les poids. Sur le Mac c'est `mobile/models/` à côté du code ; sur
#: Android le code vit dans le zip d'assets de Chaquopy (pas un chemin
#: fichier) et les poids ont été recopiés dans `filesDir` par Kotlin —
#: `install(models_dir=...)` le dit alors explicitement.
MODELS = Path(__file__).resolve().parents[1] / "models"
_models_dir: Path = MODELS
_pte_path: Path | None = None

_weights_cache: list | None = None
_audio_impl = None


def _folds() -> list[Path]:
    return [_models_dir / f"musx_fold{i}_weights.npz" for i in range(5)]


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


#: Cache d'UN SEUL décodage (P2d, 2026-09-23). Mesuré sur l'émulateur : sans
#: lui, une analyse décode le même fichier SIX fois (durée, aperçu de tête,
#: battues, CQT musx, chroma NNLS, enveloppe) — 18 à 21 s par décodage sur
#: l'AVD, soit ~100 s jetés sur un morceau de 3 min 25. Une seule entrée, pas
#: un LRU : le pipeline demande 22050 Hz quatre fois de suite (dont deux dans
#: des threads parallèles), PUIS 44100, PUIS 8000 — une entrée suffit à
#: attraper tous les doublons, et n'immobilise jamais deux signaux à la fois,
#: ce qui compte sur un appareil à 2 Go.
_cache_key = None
_cache_val = None

#: Un VERROU, pas seulement un cache. `analyze_steps` lance les battues et
#: musx dans deux threads qui réclament le même fichier au même taux à la
#: même seconde. Sans verrou, deux décodages MediaCodec partent en parallèle
#: sur le même fichier et le décodeur natif AVORTE le process (SIGABRT,
#: message « ptr », mesuré sur l'émulateur le 2026-09-23 — l'analyse mourait
#: une seconde après le CQT du chart de tête, sans trace Python). Avec le
#: verrou, le second thread attend et repart du cache : c'est aussi la seule
#: façon que le cache serve à quelque chose ici.
_decode_lock = threading.Lock()


def _load(path, sr, duration=None, variant: str = ""):
    """`variant` ne sert QU'au mode `pcm_dump` : il dit quel décodage de
    référence rejouer quand le pipeline du Mac en emploie plusieurs sur le
    MÊME fichier au MÊME taux. C'est le cas à 22050 Hz — `beat_this` reçoit un
    transcodage ffmpeg, musx reçoit `librosa.load`. Sur le téléphone il n'y a
    qu'un décodeur (MediaCodec) et ce paramètre n'existe plus.

    Le signal rendu par le cache est marqué NON INSCRIPTIBLE : deux étages
    partagent alors le même tableau, et une modification en place de l'un
    corromprait l'autre en silence. Mieux vaut une exception numpy nette au
    premier écrivain (aucun des cinq étages n'écrit dans son entrée
    aujourd'hui — vérifié en lançant la parité complète sur le Mac)."""
    global _cache_key, _cache_val
    key = (str(path), int(sr), None if duration is None else float(duration),
           variant)
    with _decode_lock:
        if key == _cache_key and _cache_val is not None:
            return _cache_val
        y = _audio_impl(path, sr, duration, variant)
        y = np.asarray(y, dtype=np.float32)
        try:
            y.flags.writeable = False
        except ValueError:
            pass  # vue sur un buffer non possédé : tant pis, on ne fige pas
        _cache_key, _cache_val = key, y
        return y


def forget_decoded() -> None:
    """Relâche le signal en cache. Appelé entre deux analyses ; sur un
    appareil serré, garder 36 Mo de PCM pendant que le réseau musx travaille
    n'a aucun intérêt."""
    global _cache_key, _cache_val
    _cache_key, _cache_val = None, None


def duration_seconds(path) -> float:
    """La durée, sans ffprobe.

    DEUX MESURES POSSIBLES, et le choix n'est pas neutre. La longueur du PCM
    décodé ne peut pas mentir sur ce qu'on va analyser, mais elle coûte un
    décodage complet du fichier — 17,9 s mesurées sur l'émulateur pour un
    morceau de 3 min 25, pour un nombre qui ne sert qu'à la fiche du morceau
    et à décider s'il y a un aperçu de tête. `MediaExtractor.KEY_DURATION`
    coûte quelques millisecondes. On prend donc la clé quand elle est là, et
    le décodage sinon.

    Ce que ça ne résout PAS : P2a a mesuré que la clé rend une valeur un peu
    PLUS GRANDE que le PCM réellement décodé (~300 échantillons sur les 3
    morceaux de seed, soit ~14 ms). Aucun accord, aucune mesure et aucune
    battue ne dépend de cette valeur — le chart est bâti sur le PCM — mais
    la durée affichée dans la bibliothèque peut différer du Mac au dernier
    chiffre."""
    if _audio_impl is audio_mediacodec:
        import mobile_audio
        return float(mobile_audio.duration_seconds(str(path)))
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
    from mobile.harmonia_mobile.beats_infer import beats_from_audio, DEFAULT_PTE
    # variant="beats" : sur le Mac, Beat This! est nourri par un transcodage
    # ffmpeg, PAS par `librosa.load` (aucun lecteur de beat_this n'ouvre un
    # .m4a sur cette machine). Les deux décodages diffèrent assez pour changer
    # le nombre de battues (348 vs 345 sur Stand By Me) — donc la parité de la
    # PLOMBERIE doit rejouer le décodage de CE site d'appel-là.
    y = _load(audio_path, 22050, variant="beats")
    return beats_from_audio(y, _pte_path or DEFAULT_PTE)


def musx_features_numpy(audio_path) -> np.ndarray:
    from mobile.harmonia_mobile.cqt_musx import song_cqt_288
    y = _load(audio_path, 22050)
    return np.asarray(song_cqt_288(y, sr=22050), dtype=np.float32)


def _weights() -> list:
    global _weights_cache
    if _weights_cache is None:
        from mobile.harmonia_mobile.musx_net import load_weights_npz
        folds = _folds()
        missing = [p.name for p in folds if not p.exists()]
        if missing:
            raise RuntimeError(
                f"poids musx numpy absents dans {_models_dir} ({missing}) — "
                f"lancer mobile/tools/dump_musx_weights.py")
        _weights_cache = [load_weights_npz(p) for p in folds]
    return _weights_cache


#: Taille de tuile temporelle du CNN musx (P2d). `None` = le chemin d'origine,
#: sans tuilage — c'est ce que le Mac garde, pour que la parité mesurée par
#: P2c reste comparable octet pour octet. Un entier borne la mémoire : le
#: calcul est le MÊME (chaque convolution est tuilée séparément, les
#: statistiques d'InstanceNorm restent prises sur le morceau entier), écart
#: mesuré 0.0 exactement sur les 3 morceaux (`mobile/tools/parity_out/
#: mem_chunk_parity.json`).
_musx_chunk_frames: int | None = None


def musx_posteriors_numpy(cqt) -> list:
    from mobile.harmonia_mobile.musx_net import predict_ensemble
    return predict_ensemble(np.asarray(cqt, dtype=np.float32), _weights(),
                            _musx_chunk_frames)


def musx_torch_reference(cqt) -> list:
    """Les 5 réseaux TORCH, derrière la même couture. Explicitement nommé :
    c'est le moteur du Mac, autorisé uniquement pour isoler le reste de la
    plomberie quand le port numpy n'est pas encore là."""
    from harmonia.musx import _run_nets
    return _run_nets(np.asarray(cqt))


#: Même idée que `_musx_chunk_frames`, pour l'étage FFT du chroma NNLS.
#: `None` = chemin d'origine. Le coupable mesuré était l'étage FFT
#: (`windowed` (T,16384) float64 et son `rfft` (T,8193) complex128, 758 Mo
#: CHACUN sur un morceau de 4 min 28) ; tuilé, la sortie est BIT-IDENTIQUE
#: (0.0, pas « sous 1e-6 ») et l'estimation d'accordage reste prise sur le
#: morceau entier — la couper aurait décalé tout l'aval en silence.
_nnls_chunk_frames: int | None = None


def nnls_bothchroma_numpy(audio_path):
    from mobile.harmonia_mobile.nnls_chroma import extract_bothchroma_numpy
    y = _load(audio_path, 44100)
    return extract_bothchroma_numpy(np.asarray(y, dtype=np.float32), 44100,
                                    chunk_frames=_nnls_chunk_frames)


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


def install(*, audio: str = "pcm_dump", musx: str = "numpy",
            beats_runtime: str = "python", models_dir=None,
            pte_path=None, beats_threads: int = 0,
            musx_chunk_frames: int | None = None,
            nnls_chunk_frames: int | None = None) -> None:
    """Pose les huit fentes. Tout est un CHOIX explicite, jamais un repli.

    `beats_runtime` dit QUI exécute le `.pte` de Beat This! :

    * `"python"` (défaut, le Mac) — `executorch.runtime`, le paquet pip. C'est
      ce runtime qui a produit toute la parité de T1c et de P2c ; le laisser
      par défaut garantit qu'aucune mesure Mac ne bouge à cause de P2d.
    * `"java"` (Android) — `mobile_beats_runtime.install()`, c'est-à-dire
      `libexecutorch.so` via JNI. `mobile_beats_runtime` n'est importable que
      sur le téléphone (il importe `java`), d'où l'import tardif ici.

    `musx_chunk_frames` : la tuile temporelle du CNN musx. `None` (défaut,
    le Mac) garde le chemin d'origine. Sur un appareil à 2 Go il FAUT une
    valeur (512 mesuré) — sans elle le réseau alloue ~2 Go de tableaux
    intermédiaires sur un morceau de 3 min 25 et le système tue l'app. Le
    résultat est identique, pas approché.

    `models_dir` / `pte_path` : sur Android le code Python vit dans le zip
    d'assets de Chaquopy, qui n'est pas un chemin fichier — les poids sont
    recopiés dans `filesDir` au premier lancement et leur emplacement est
    passé ici. Sur le Mac, ne rien passer garde `mobile/models/`.
    """
    global _audio_impl, _weights_cache, _models_dir, _pte_path
    global _musx_chunk_frames, _nnls_chunk_frames
    if audio not in _AUDIO:
        raise KeyError(f"audio={audio!r} inconnu (connus: {sorted(_AUDIO)})")
    _audio_impl = _AUDIO[audio]
    _musx_chunk_frames = musx_chunk_frames
    _nnls_chunk_frames = nnls_chunk_frames

    if models_dir is not None:
        _models_dir = Path(models_dir)
        _weights_cache = None
    if pte_path is not None:
        _pte_path = Path(pte_path)
    elif models_dir is not None:
        _pte_path = _models_dir / "beat_this_final0_fixed1500.pte"

    if beats_runtime == "java":
        import mobile_beats_runtime
        if _pte_path is None:
            raise ValueError("beats_runtime='java' exige pte_path ou models_dir")
        mobile_beats_runtime.install(str(_pte_path), threads=beats_threads)
    elif beats_runtime == "python":
        from mobile.harmonia_mobile import beats_infer
        beats_infer.set_chunk_runner(None)
    else:
        raise KeyError(f"beats_runtime={beats_runtime!r} inconnu "
                       f"(connus: 'python', 'java')")
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
