"""Le seul endroit où l'application lit son environnement.

Tout ce qui se règle est ici : chemins, port, appareil musx. Les choix
d'algorithme (détecteur de sections, loi de merge, granularité, repli) ne se
règlent PAS : ce sont des constantes, décidées par Louis (plan du refactor,
2026-09-14, décision 4), qui changent par un commit et une page avant/après,
jamais par une variable d'environnement. Dix-huit drapeaux `HARMONIA_*` lus
à leur point d'usage, c'est ce que ce module remplace.

Ce que ce module ne fait PAS : créer des dossiers ou vérifier que les
fichiers existent — chaque étage le fait pour ce qu'il lit, et échoue fort.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _env(name: str, default: str) -> str:
    return (os.environ.get(name) or "").strip() or default


@dataclass(frozen=True)
class Settings:
    repo: Path = REPO
    #: Jusqu'au sprint 15 (état humain / caches séparés), l'état reste celui
    #: de harmonia_min : c'est la bibliothèque que le rapport d'or compare.
    state_dir: Path = REPO / "harmonia_min" / "state"
    data_cache: Path = REPO / "data" / "cache"
    audio_dir: Path = REPO / "docs" / "audio"
    assets: Path = Path(__file__).resolve().parent / "assets"
    #: Le clone music-x-lab (ISMIR 2019) est EXÉCUTÉ, pas seulement lu : le
    #: décodeur et les extracteurs s'importent depuis son dossier.
    musx_dir: Path = Path(_env("HARMONIA_MUSX_DIR",
                               str(REPO / "third_party" / "musx_ismir2019")))
    musx_device: str = _env("HARMONIA_MUSX_DEVICE", "auto")
    #: `HARMONIA_MIN_PORT` reste accepté jusqu'à la disparition de
    #: harmonia_min (sprint 22) : c'est le nom que le worktree utilise.
    port: int = int(_env("HARMONIA_PORT", _env("HARMONIA_MIN_PORT", "7772")))

    # ── Choix d'algorithme : des constantes, pas des réglages ──────────────
    sections: str = "songformer"      # seul détecteur (2026-08-18, à l'oreille)
    merge: str = "mean"               # postérieures moyennées (2026-08-19)
    quarter_bar: bool = True          # tout temps peut porter un accord (2026-08-07)
    fold_loop: str = "occurrence"     # l'occurrence entière se replie (2026-08-08)
    fold_gate: str = "letter"         # veto de cohérence par lettre entière
    fold_transpose: bool = False      # pas de transposition avant l'empilement


SETTINGS = Settings()
