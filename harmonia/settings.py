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
    #: État split (décision 10, sprint 15) : `state/human/` est suivi par
    #: git (petit JSON écrit à la main ou par un geste de Louis — sections,
    #: annotations, marques, titres) ; `state/cache/` ne l'est pas
    #: (régénérable — charts, battues, sections songformer, rapports). Un
    #: chart perdu par un fichier gitignored (2026-08-12) est l'incident qui
    #: a motivé la séparation ; voir `.gitignore`.
    state_dir: Path = REPO / "state"
    human_dir: Path = REPO / "state" / "human"
    cache_dir: Path = REPO / "state" / "cache"
    charts_dir: Path = REPO / "state" / "cache" / "charts"
    beats_dir: Path = REPO / "state" / "cache" / "beats"
    songformer_dir: Path = REPO / "state" / "cache" / "songformer"
    reports_dir: Path = REPO / "state" / "cache" / "reports"
    sections_dir: Path = REPO / "state" / "human" / "sections"
    #: Les gestes en cours de l'outil sections (brouillons) — séparés de
    #: `sections_dir`, la vérité terrain validée à la main.
    sections_draft_dir: Path = REPO / "state" / "human" / "sections_draft"
    annotations_dir: Path = REPO / "state" / "human" / "annotations"
    #: La marque « Set bar 1 » de Louis, une par audio (`<stem>.json`,
    #: `{"bar1": <secondes>}`) — la SEULE source à l'exécution (voir
    #: `server.jobs.bar1_for`). Avant le sprint 15 elle ne vivait que dans le
    #: champ `bar1` du chart régénérable, ce qui l'a perdue le 2026-08-13.
    marks_dir: Path = REPO / "state" / "human" / "marks"
    chart_meta_path: Path = REPO / "state" / "human" / "chart_meta.json"
    folders_path: Path = REPO / "state" / "human" / "folders.json"
    data_cache: Path = REPO / "data" / "cache"
    audio_dir: Path = REPO / "docs" / "audio"
    assets: Path = Path(__file__).resolve().parent / "assets"
    #: Le clone music-x-lab (ISMIR 2019) est EXÉCUTÉ, pas seulement lu : le
    #: décodeur et les extracteurs s'importent depuis son dossier.
    musx_dir: Path = Path(_env("HARMONIA_MUSX_DIR",
                               str(REPO / "third_party" / "musx_ismir2019")))
    musx_device: str = _env("HARMONIA_MUSX_DEVICE", "auto")
    #: `HARMONIA_MIN_PORT` survit VOLONTAIREMENT au sprint 22 (2026-09-16),
    #: alors que `harmonia_min` disparaît : c'est le nom qu'emploient les
    #: serveurs de worktree pour ne PAS se poser sur :7772
    #: (`docs/refactor_2026-09/sprints.md`, « serveur du worktree sur :7773 »).
    #: Le retirer ferait retomber ces serveurs sur 7772 par défaut, c'est-à-dire
    #: sur l'app vivante — exactement l'incident qu'on évite. À retirer le jour
    #: où les worktrees passeront à `HARMONIA_PORT`, pas avant.
    port: int = int(_env("HARMONIA_PORT", _env("HARMONIA_MIN_PORT", "7772")))

    # ── Choix d'algorithme : des constantes, pas des réglages ──────────────
    sections: str = "songformer"      # seul détecteur (2026-08-18, à l'oreille)
    merge: str = "mean"               # postérieures moyennées (2026-08-19)
    quarter_bar: bool = True          # tout temps peut porter un accord (2026-08-07)
    fold_loop: str = "occurrence"     # l'occurrence entière se replie (2026-08-08)
    fold_gate: str = "letter"         # veto de cohérence par lettre entière
    fold_transpose: bool = False      # pas de transposition avant l'empilement
    #: FUSIONNER LES LETTRES QUI SONT LE MÊME PASSAGE — loi depuis le
    #: 2026-09-18. songformer nomme par le RÔLE (couplet/refrain/instrumental),
    #: pas par la musique : sur Fallin', qui est une seule boucle E-…B- du
    #: début à la fin, il rend huit lettres. La comparaison se fait sur les
    #: ACCORDS DÉCODÉS (voir `folding.MERGE_LETTERS_ACCORDS` pour ce qui a été
    #: mesuré, essayé et écarté — la version chroma cassait Let It Be).
    #: Arbitré sur six morceaux que Louis a vérifiés à l'oreille, et contrôlé
    #: contre les structures iReal, seules structures écrites par un humain
    #: qu'on ait : `docs/plots/sections_6morceaux.html`.
    #:
    #: ALLUMÉ le 2026-09-19 à la demande de Louis (« mets moi merge on et
    #: montres moi ce que donnes les sections avec »), après une journée où je
    #: l'avais laissé éteint faute d'avoir arbitré dans le bon régime. Ce qui
    #: reste vrai et non résolu : sur une ré-analyse fraîche, Let It Be
    #: fusionne `A` et `C` à 1.00 alors qu'ils doivent rester séparés — le
    #: dossier est dans known_issues, et la page avant/après le montre morceau
    #: par morceau.
    merge_letters: bool = True


SETTINGS = Settings()
