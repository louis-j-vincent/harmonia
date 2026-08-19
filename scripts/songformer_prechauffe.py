#!/usr/bin/env python3
"""scripts/songformer_prechauffe.py — remplir le cache SongFormer d'avance.

Le modèle coûte ~1 min de chargement par processus puis quelques dizaines de
secondes par morceau, sur CPU. Une fois le résultat en cache
(`harmonia_min/state/songformer/`), l'analyse d'un chart ne le repaie jamais.
On le lance donc une fois sur toute la bibliothèque, en tâche de fond.

    .venv/bin/python scripts/songformer_prechauffe.py
"""
from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

from harmonia_min import songformer as SF   # noqa: E402

ETAT = REPO / "harmonia_min" / "state"
AUD = REPO / "docs" / "audio"


def main() -> None:
    audios, vus = [], set()
    for p in sorted((ETAT / "charts").glob("min_*.json")):
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        a = AUD / Path(m.get("audio_url") or "x").name
        if a.exists() and a.name not in vus:
            vus.add(a.name)
            audios.append(a)
    reste = [a for a in audios if not SF._cle(a).exists()]
    print(f"{len(audios)} morceaux, {len(audios) - len(reste)} déjà en cache, "
          f"{len(reste)} à faire")
    for i, a in enumerate(reste, 1):
        t = time.time()
        try:
            segs = SF.segments(a)
        except Exception:
            logging.exception("RATÉ %s", a.name)
            continue
        print(f"[{i}/{len(reste)}] {a.name} — {len(segs)} segments en "
              f"{time.time() - t:.0f} s")


if __name__ == "__main__":
    main()
