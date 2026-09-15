"""Point d'entrée `python -m harmonia.server` — démarre l'app sur SETTINGS.port.

Ce que ce module ne fait PAS : construire l'app (voir `app.py`) ni choisir le
port (voir `settings.py` — `HARMONIA_PORT`/`HARMONIA_MIN_PORT`).
"""
from __future__ import annotations

import logging

from harmonia.server.app import app
from harmonia.server.jobs import CHARTS_DIR
from harmonia.settings import SETTINGS

log = logging.getLogger("harmonia.server")


def main() -> None:
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    log.info("harmonia.server on http://0.0.0.0:%d (charts: %s)",
              SETTINGS.port, CHARTS_DIR)
    app.run(host="0.0.0.0", port=SETTINGS.port, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
