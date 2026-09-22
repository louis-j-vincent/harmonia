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


def start_server(state_dir: str, audio_dir: str, port: int) -> None:
    # DOIT précéder tout `import harmonia*` — settings.py lit ces deux noms
    # une seule fois, à l'import du module (voir sa docstring, note
    # "portage mobile 2026-09-23").
    os.environ["HARMONIA_STATE_DIR"] = state_dir
    os.environ["HARMONIA_AUDIO_DIR"] = audio_dir
    os.environ["HARMONIA_PORT"] = str(port)

    for sub in (
        "human/sections", "human/sections_draft", "human/annotations",
        "human/marks", "cache/charts", "cache/beats", "cache/songformer",
        "cache/reports",
    ):
        os.makedirs(os.path.join(state_dir, sub), exist_ok=True)

    from harmonia.server.app import create_app

    app = create_app()

    def _run() -> None:
        # use_reloader=False : un reloader forkerait un second process, que
        # Chaquopy ne sait pas faire sur Android. threaded=True : la WebView
        # peut tirer plusieurs requêtes en parallèle (audio Range + JSON).
        app.run(host="127.0.0.1", port=port, threaded=True, use_reloader=False)

    threading.Thread(target=_run, daemon=True, name="harmonia-flask").start()
