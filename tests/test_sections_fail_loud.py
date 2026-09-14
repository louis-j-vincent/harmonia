"""tests/test_sections_fail_loud.py — songformer sans repli : un enfant mort
fait ÉCHOUER l'analyse, red-first sur la décision de Louis (2026-09-14).

Avant refactor sprint 9, un enfant songformer mort (mémoire épuisée, code de
sortie 137/-9) faisait silencieusement retomber le chart sur le détecteur
`voice` — le chart qui sortait n'était plus celui que Louis avait validé à
l'oreille, et rien à l'écran ne le disait. Il n'y a plus de repli : l'échec du
détecteur doit remonter tel quel jusqu'à l'appelant (`analyze_steps`, puis le
job du serveur, qui l'enregistrent comme `refine_error`).
"""
from __future__ import annotations

import uuid

import pytest

import harmonia.sections as S
import harmonia.sections.songformer as SF


def test_enfant_mort_fait_echouer_lanalyse(tmp_path, monkeypatch):
    # nom unique : garantit un cache songformer FROID (jamais analysé), donc
    # `segments()` doit forcément passer par `_dans_un_enfant`.
    audio = tmp_path / f"jamais_vu_{uuid.uuid4().hex}.wav"
    audio.write_bytes(b"\x00" * 16)

    def _enfant_mort(_audio):
        raise RuntimeError("mémoire épuisée : l'OS a tué le processus SongFormer")

    monkeypatch.setattr(SF, "_dans_un_enfant", _enfant_mort)

    grid = [float(i) for i in range(9)]      # 8 mesures — au-delà du raccourci n<2
    with pytest.raises(RuntimeError, match="mémoire épuisée"):
        S.detect_sections(grid, audio)
