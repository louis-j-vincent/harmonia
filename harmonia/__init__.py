"""Harmonia — l'application, réécrite étage par étage (refactor 2026-09).

Le refactor est terminé : `harmonia_min`, l'app de référence qui a servi de
témoin pendant vingt et un sprints, est supprimée (sprint 22, 2026-09-16) ;
il ne reste d'elle que `harmonia_min/chord_lm/`, une recherche en cours qui
ne fait pas partie de l'app. Ce paquet est le seul moteur. Le rapport d'or
(`tools/golden.py`) compare toujours la bibliothèque à une baseline gelée.
Plan et journal : `docs/refactor_2026-09/`.
"""
