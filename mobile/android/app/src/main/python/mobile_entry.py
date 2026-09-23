"""Point d'entrée Python de l'app Android — PAS un fichier de `harmonia/`.

Appelé une fois par `MainActivity.kt` via Chaquopy, APRÈS que Kotlin a copié
les 2-3 charts + audio + `chart_meta.json` de seed (assets/harmonia_seed/,
packagés dans l'APK, illisibles directement comme chemins fichier) vers
`context.getFilesDir()` (stockage privé de l'app, inscriptible). Son seul
travail : poser les variables d'environnement AVANT d'importer `harmonia`
(règle de `harmonia/settings.py` : il ne lit `os.environ` qu'à l'import, une
fois), créer les dossiers d'état que `settings.py` documente ne PAS créer
lui-même, puis démarrer Flask dans un thread.

Ce que ce fichier ne fait PAS : copier les assets (Kotlin, `MainActivity.kt`
— l'API `AssetManager` n'est pas un chemin fichier direct) ; choisir le
port (passé par Kotlin, doit rester libre sur le téléphone — jamais 7772 sur
le Mac, voir docs/portage_mobile/PLAN.md) ; lancer une analyse (hors
périmètre P1).
"""
from __future__ import annotations

import os
import threading


def _install_mobile_backends(models_dir: str) -> None:
    """Pose les moteurs « téléphone » AVANT que quoi que ce soit n'importe
    `harmonia.pipeline` (P2d, 2026-09-23).

    L'ordre n'est pas décoratif. `harmonia/backends.py` est un registre de
    fentes : tant que personne n'a rien installé, chaque étage appelle son
    implémentation Mac (ffmpeg, librosa, torch, le plugin VAMP) — aucune
    n'existe ici, et l'analyse échouerait au premier étage. `install()` doit
    donc précéder le premier `import harmonia.pipeline`, qui a lieu dans
    `harmonia.server.jobs`, lui-même importé par `create_app()`.

    Les six choix, tous explicites (jamais un repli silencieux) :

    * `audio="mediacodec"` — le décodeur Kotlin de P2a. C'est le SEUL
      décodeur du téléphone, là où le Mac en emploie deux sur le même
      fichier (ffmpeg pour Beat This!, CoreAudio pour musx et le chroma) :
      P2c a mesuré que cette divergence-là, à elle seule, change le compte de
      battues. Elle est structurelle, pas un bug.
    * `beats_runtime="java"` — `libexecutorch.so` par JNI, puisque le paquet
      pip `executorch` (et le `torch` qu'il tire) n'existe pas ici.
    * `musx="numpy"` — le port numpy des 5 réseaux (T1b) ; ExecuTorch a été
      écarté pour musx, mesuré 8 à 10x trop lent.
    * `models_dir` — `filesDir/harmonia_models`, où Kotlin a recopié les
      assets ; le code Python, lui, vit dans un zip et n'a pas de voisin
      `models/`.
    * `beats_threads=0` — autant de threads XNNPACK que de coeurs.
    * `musx_chunk_frames=256` — le CNN musx est tuilé en temps. Sans ça
      l'analyse d'un morceau de 3 min 25 alloue ~2 Go et l'AVD (2 Go de RAM)
      tue l'app avant le premier accord ; mesuré. Le calcul est le même, à
      0.0 près sur les 3 morceaux.
    * `nnls_chunk_frames=256` — même chose pour l'étage FFT du chroma NNLS,
      qui était le pic SUIVANT (1,7 Go rien que pour ses deux tableaux de
      fenêtres et de spectres). Sortie bit-identique ; l'accordage reste
      estimé sur le morceau entier.
    """
    import logging
    import os
    import sys
    from mobile.harmonia_mobile import backends_mobile

    log = logging.getLogger("harmonia.mobile.entry")

    # QUEL `.pte` POUR BEAT THIS!, ET POURQUOI IL Y EN A DEUX (P2d).
    #
    # Le `.pte` de production est délégué à XNNPACK : c'est celui que T1c a
    # validé et celui qui donnera la vitesse sur le téléphone de Louis. Il ne
    # tourne PAS sur l'émulateur : le CPU invité y annonce `sme2` (les bits de
    # la puce Apple de l'hôte traversent l'hyperviseur) mais l'instruction SME
    # `rdsvl`, que XNNPACK exécute une fois pour lire la longueur de vecteur
    # en streaming, lève un SIGILL. Constaté, désassemblé, pas supposé.
    #
    # Le `.pte` PORTABLE (sans délégué) n'est donc PAS embarqué dans l'APK :
    # il se pousse à la main (`adb push ... files/harmonia_models/`) quand on
    # veut faire tourner l'émulateur. S'il est là, on le prend, et on le DIT
    # dans le log — jamais un repli muet, et jamais dans l'APK, pour qu'un
    # chiffre de vitesse pris sur l'émulateur ne puisse pas être confondu avec
    # celui du téléphone.
    xnn = os.path.join(models_dir, "beat_this_final0_fixed1500.pte")
    portable = os.path.join(models_dir, "beat_this_final0_fixed1500_portable.pte")
    if os.path.exists(portable):
        pte = portable
        log.warning("BEAT THIS! en mode PORTABLE (%s) — sans XNNPACK : "
                    "correct, mais la VITESSE mesurée ici ne vaut RIEN pour "
                    "le téléphone", portable)
    else:
        pte = xnn
        log.info("Beat This! délégué à XNNPACK (%s)", xnn)

    backends_mobile.install(
        audio="mediacodec",
        beats_runtime="java",
        musx="numpy",
        models_dir=models_dir,
        pte_path=pte,
        beats_threads=0,
        musx_chunk_frames=256,
        nnls_chunk_frames=256,
    )
    # PAS D'APERÇU DE TÊTE SUR TÉLÉPHONE, POUR L'INSTANT (P2d, 2026-09-23).
    #
    # `_head_chart` analyse les ~45 premières secondes pour afficher un chart
    # tout de suite. Sur 45 s, le mel fait ~1138 trames, donc MOINS que le
    # bloc de 1500 : `split_piece` rend alors un bloc court, et le `.pte` de
    # Beat This! est exporté en FORME FIXE (1, 1500, 128). Le runtime natif
    # refuse net — « Attempted to resize a static tensor. Expected shape
    # (1, 1500, 128), but received (1, 1138, 128) » — et abandonne le
    # process. Mesuré sur l'émulateur, deux fois.
    #
    # Ce n'est PAS un problème d'Android : le même chemin sur le Mac ferait la
    # même chose. Ni T1c ni P2c ne l'avaient vu, parce que T1c ne mesurait que
    # des morceaux entiers (« tout bloc réel fait 1500 trames ») et que P2c
    # tournait avec `head_s=0`. C'est P2d qui l'a trouvé.
    #
    # La VRAIE correction est de router les blocs courts vers le `.pte`
    # DYNAMIQUE, déjà exporté par T1c et jamais exercé. Elle n'est pas faite
    # ici (il faut un second module chargé, et sa variante portable pour
    # l'émulateur). En attendant, l'aperçu est désactivé plutôt que rendu faux
    # par un remplissage de zéros — bourrer le bloc jusqu'à 1500 changerait
    # l'attention du transformer, donc les battues, sans rien signaler.
    #
    # Effet de bord assumé : sur téléphone le chart n'apparaît qu'à la fin de
    # l'analyse, pas après 45 s. À rouvrir quand le `.pte` dynamique sera
    # branché.
    from harmonia import pipeline as _pipeline
    _pipeline.HEAD_MIN_SONG = float("inf")

    # (plus d'assert « pipeline pas encore importé » : on vient de l'importer
    # nous-mêmes pour éteindre l'aperçu. L'ordre qui compte est que
    # `backends_mobile.install()` ait eu lieu AVANT, ce qui est le cas — et
    # de toute façon `harmonia.backends` est un registre consulté à l'APPEL,
    # pas capturé à l'import : l'ordre n'est un piège que pour les constantes
    # de module, comme celle qu'on change juste au-dessus.)


def start_server(state_dir: str, audio_dir: str, port: int,
                 models_dir: str | None = None,
                 musx_dir: str | None = None) -> None:
    # DOIT précéder tout `import harmonia*` — settings.py lit ces deux noms
    # une seule fois, à l'import du module (voir sa docstring, note
    # "portage mobile 2026-09-23").
    os.environ["HARMONIA_STATE_DIR"] = state_dir
    os.environ["HARMONIA_AUDIO_DIR"] = audio_dir
    os.environ["HARMONIA_PORT"] = str(port)
    if musx_dir:
        # La 3e (et dernière) variable que lit `harmonia/settings.py`. Le
        # clone musx vendu a été recopié des assets vers `filesDir` par
        # Kotlin ; sans ce pointeur, `settings` chercherait
        # `REPO/third_party/musx_ismir2019` — c'est-à-dire dans l'archive
        # Python de Chaquopy, où il n'y a pas de vrai dossier.
        os.environ["HARMONIA_MUSX_DIR"] = musx_dir

    for sub in (
        "human/sections", "human/sections_draft", "human/annotations",
        "human/marks", "cache/charts", "cache/beats", "cache/songformer",
        "cache/reports",
    ):
        os.makedirs(os.path.join(state_dir, sub), exist_ok=True)

    # AVANT create_app() : create_app importe les blueprints, dont
    # `routes/analyze.py` -> `server/jobs.py` -> `harmonia.pipeline`.
    if models_dir:
        _install_mobile_backends(models_dir)

    from harmonia.server.app import create_app

    app = create_app()

    # Route de debug P2a (mobile_debug.py, PAS dans harmonia/) : compare le
    # décodeur MediaCodec/Kotlin au Mac via /api/debug/decode. Montée ici,
    # jamais dans harmonia.server.app — hors périmètre du serveur de prod.
    import mobile_debug
    app.register_blueprint(mobile_debug.bp)

    # P2d : chronomètres par brique (décodage / battues / CQT / réseau musx /
    # chroma). Enveloppe les fentes déjà posées, ne change aucun calcul.
    if models_dir:
        import mobile_timing
        mobile_timing.install()
        app.register_blueprint(mobile_timing.bp)

    # Import YouTube P3b (mobile_youtube.py, PAS dans harmonia/) : remplace
    # harmonia.server.youtube._ytdlp_bin/_video_meta/_download_audio (qui
    # shellent vers un binaire yt-dlp + Firefox + PO-Token server, aucun des
    # trois présent sur Android) par yt_dlp.YoutubeDL en process, cookies
    # statiques + QuickJS embarqué. Monkeypatch explicite, documenté dans
    # docs/portage_mobile/p3b_youtube_cookies_qjs.md — harmonia/server/
    # youtube.py n'est pas modifié.
    import mobile_youtube
    mobile_youtube.install()
    # P3c (2026-09-23) : POST /api/mobile/youtube_login -> lance
    # YouTubeLoginActivity (session device, remplace le fichier de cookies
    # statique du Mac). N'existe que sur mobile, comme mobile_debug.
    app.register_blueprint(mobile_youtube.bp)

    def _run() -> None:
        # use_reloader=False : un reloader forkerait un second process, que
        # Chaquopy ne sait pas faire sur Android. threaded=True : la WebView
        # peut tirer plusieurs requêtes en parallèle (audio Range + JSON).
        app.run(host="127.0.0.1", port=port, threaded=True, use_reloader=False)

    threading.Thread(target=_run, daemon=True, name="harmonia-flask").start()
