"""harmonia/sections — où viennent les frontières et les lettres d'un chart.

Un seul détecteur : `songformer` (`harmonia/sections/songformer.py`). Louis,
2026-08-18, après avoir écouté `/plots/songformer.html` où sa propre
annotation, SongFormer et l'ancien détecteur maison jouaient côte à côte sur
dix-neuf morceaux : « je suis d'accord avec lui partout, on le prend en
prod ». Le refactor (plan 2026-09-14, décision 4) en fait le SEUL détecteur —
l'ancien dispatcher à quatre voies (`harmonia_min/sections.py` : songformer /
voice / harmonic / chroma) et les modules qu'il aiguillait sont supprimés ;
git history les garde.

[{b0, b1, label}] sur les indices de mesure, contigu, couvrant : le contrat
que l'ancien dispatcher documentait déjà et que `songformer.detect_sections`
tient seul désormais.

CE QUE CE MODULE NE FAIT PAS :
  * **pas de repli.** Si le processus enfant de songformer meurt (mémoire
    épuisée, code de sortie 137/-9) ou dépasse son délai, l'exception REMONTE
    ici sans être rattrapée. Avant, un enfant mort faisait retomber le chart
    sur le détecteur `voice` avec un simple ERROR dans le journal — le chart
    qui sortait n'était plus celui que Louis avait validé à l'oreille, et rien
    ne le disait à l'écran. Décision de Louis (2026-09-14) : une analyse dont
    songformer échoue doit ÉCHOUER, pas dégrader en silence — c'est
    `analyze_steps`, puis le job du serveur, qui l'enregistrent comme
    `refine_error` au lieu de rendre un chart non validé.
  * **pas de repliement (folding).** Ce module rend des frontières et des
    rôles (intro / couplet / refrain / pont / outro) — l'empilement des
    occurrences d'une même lettre, la longueur à laquelle chacune s'écrit,
    les queues, restent le travail de `harmonia.folding` en aval.
"""
from __future__ import annotations

from . import songformer


def detect_sections(grid: list[float], audio, form_start=None) -> list[dict]:
    """[{b0, b1, label}] sur les indices de mesure — contigu, couvrant.

    `form_start` : la marque « Set bar 1 » de l'utilisateur — quand elle est
    passée, il n'y a pas d'intro à trouver, l'appelant a déjà découpé ce qui
    précède (voir `songformer.detect_sections`).
    """
    return songformer.detect_sections(grid, audio, form_start=form_start)
