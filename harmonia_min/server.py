"""Pont vers `harmonia.server` (refactor, sprints 11-14). Disparaît avec
harmonia_min (sprint 22).

L'app elle-même vit dans `harmonia.server.app` ; ce fichier ne fait que
réexporter ce que les tests et les scripts importent encore par
`harmonia_min.server`, et garder `python -m harmonia_min.server` vivant
jusqu'à ce que sprint 22 le retire.
"""
from __future__ import annotations

from harmonia.server.app import app
from harmonia.server.jobs import CHARTS_DIR, META_PATH, _jobs, _load_chart_meta, _run_job
from harmonia.server.routes.library import _pretty_title
from harmonia.server.routes.sections import SECTIONS_DIR, SECTIONS_DRAFT_DIR
from harmonia.settings import SETTINGS

PORT = SETTINGS.port


def main() -> None:
    from harmonia.server.__main__ import main as _main
    _main()


if __name__ == "__main__":
    main()
