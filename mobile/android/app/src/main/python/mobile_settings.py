"""Réglage persistant : quel `.pte` de Beat This! charger PAR DÉFAUT au
démarrage de l'app (Réglages généraux, 2026-10-28, demande de Louis après le
switch à chaud de la veille — « je veux un écran de réglages, pas lié à une
chanson, où je choisis le modèle par défaut »).

PAS dans `harmonia/` : état propre à la plateforme mobile. Stocké comme un
petit JSON écrit à la main sous `state/human/` (même convention que
`chart_meta.json`/`folders.json` : un réglage que Louis fait, pas un cache
régénérable — voir docs/STATE.md, la distinction `human/` vs `cache/`).

POURQUOI UN MODULE À PART, PAS DANS `mobile_beatmodel.py`. `mobile_entry.py`
doit pouvoir lire ce choix AVANT que Flask (donc `mobile_beatmodel`, un
blueprint monté après `create_app()`) n'existe, pour décider quel `.pte`
charger au tout premier `BeatsRuntime.load()`. Sans ce module séparé, le
choix de Louis ne survivrait qu'en mémoire — perdu à chaque relance de l'app,
ce qu'il a explicitement demandé de corriger.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger("harmonia.mobile.settings")

#: mêmes trois noms et fichiers que `mobile_beatmodel.py` — dupliqué plutôt
#: que couplé (import Flask-free exprès, voir docstring du module).
FILES = {
    "final0": "beat_this_final0_fixed1500.pte",
    "small0": "beat_this_small0_fixed1500.pte",
    "int8": "beat_this_final0_int8_fixed1500.pte",
}


def _path(human_dir: str | Path) -> Path:
    return Path(human_dir) / "mobile_settings.json"


def read_default_beat_model(human_dir: str | Path) -> str | None:
    """Le nom ('final0'/'small0'/'int8') choisi dans Réglages, ou `None` si
    jamais réglé — l'appelant décide du repli (toujours `final0` partout
    ailleurs dans ce portage, pour que le silence reste le choix le plus
    prudent)."""
    p = _path(human_dir)
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError):
        return None
    name = data.get("beat_model")
    return name if name in FILES else None


def write_default_beat_model(human_dir: str | Path, name: str) -> None:
    if name not in FILES:
        raise ValueError(f"nom de modèle inconnu : {name!r}, attendu un de {sorted(FILES)}")
    p = _path(human_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    data: dict = {}
    try:
        data = json.loads(p.read_text())
    except (OSError, ValueError):
        pass
    data["beat_model"] = name
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False))
    log.info("réglage persisté : modèle de battues par défaut = %s (%s)", name, p)
