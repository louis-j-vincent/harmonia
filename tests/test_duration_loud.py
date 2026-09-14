"""tests/test_duration_loud.py — la durée échoue fort, elle ne rend jamais 0.

Errors not to carry #1 (docs/refactor_2026-09/plan.md) : `_duree()` rendait
0.0 sur toute erreur ffprobe, sans logger — un silence qui aurait pu faire
lire « ce morceau dure 0 s » sans que personne ne le sache.
`harmonia.pipeline.duration_seconds()` la remplace : sur un fichier qui
n'existe pas, ffprobe échoue et la fonction doit lever, pas rendre 0.0.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from harmonia.pipeline import duration_seconds


def test_duration_seconds_leve_sur_fichier_absent():
    with pytest.raises(RuntimeError):
        duration_seconds(Path("/nonexistent.m4a"))
