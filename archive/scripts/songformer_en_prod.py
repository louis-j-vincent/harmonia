#!/usr/bin/env python3
"""scripts/songformer_en_prod.py — re-découper les charts DÉJÀ analysés.

Depuis le 2026-08-18, toute NOUVELLE analyse passe par SongFormer
(`harmonia_min/sections.py`). Les charts déjà dans la bibliothèque, eux,
gardent le découpage du détecteur d'avant tant que personne ne les rouvre.
Ce script leur applique le nouveau, SANS toucher à leur grille de mesures.

CE QU'IL NE FAIT PAS, exprès : il ne relance ni les battues, ni musx, ni le
décodage. Une re-analyse complète referait la grille, et une grille qui bouge
décale les annotations posées dessus — c'est déjà arrivé (bein_green, Grenade,
Another Day, 2026-08-18). Ici on ne change que la couche du dessus : les
sections, et les accords empilés qui en sont la conséquence — exactement le
chemin de la Soudure (`server.soudure_valider`), avec `refold` au milieu.

Les charts pour lesquels Louis a une annotation à la main dans
`state/sections/` sont SAUTÉS par défaut : son découpage à lui n'est pas une
sortie de détecteur, et l'écraser serait perdre son travail. `--tout` les
inclut quand même.

    .venv/bin/python scripts/songformer_en_prod.py [--tout] [--sec] [--un STEM]

`--sec` n'écrit rien et montre ce qui changerait.
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harmonia_min import songformer as SF          # noqa: E402
from harmonia_min.refold import refold             # noqa: E402
from harmonia_min.soudure import sections_pour_chart  # noqa: E402

ETAT = REPO / "harmonia_min" / "state"
CHARTS = ETAT / "charts"
AUD = REPO / "docs" / "audio"
BAK = ETAT / "charts.bak_songformer"


def resume(secs) -> str:
    return " ".join(f"{s['label']}[{s['b0'] + 1}-{s['b1'] + 1}]" for s in secs)


def ancien(chart) -> str:
    out = []
    for s in chart.get("sections") or []:
        for a, b in s.get("barRanges") or []:
            out.append((a, f"{s['label']}[{a + 1}-{b + 1}]"))
    return " ".join(t for _, t in sorted(out))


def main() -> None:
    tout = "--tout" in sys.argv
    sec = "--sec" in sys.argv
    un = sys.argv[sys.argv.index("--un") + 1] if "--un" in sys.argv else None
    a_la_main = {f.stem for f in (ETAT / "sections").glob("*.json")}

    faits, sautes, rates = 0, [], []
    for p in sorted(CHARTS.glob("min_*.json")):
        stem = p.stem[len("min_"):]
        if un and stem != un:
            continue
        chart = json.loads(p.read_text(encoding="utf-8"))
        audio = AUD / Path(chart.get("audio_url") or "x").name
        grid = chart.get("barGrid") or []
        if not audio.exists() or len(grid) < 3:
            sautes.append((stem, "pas d'audio ou pas de grille"))
            continue
        if stem in a_la_main and not tout:
            sautes.append((stem, "annotation à la main — gardée"))
            continue
        if not SF._cle(audio).exists():
            sautes.append((stem, "pas encore dans le cache SongFormer"))
            continue
        try:
            segs = SF.detect_sections(grid, audio)
        except Exception as exc:
            rates.append((stem, f"{type(exc).__name__}: {exc}"))
            continue
        secs = [{"label": s["label"], "mesure_debut": s["b0"] + 1,
                 "mesure_fin": s["b1"] + 1} for s in segs]
        bars, rap = refold(chart, secs, AUD)
        neuves = sections_pour_chart(chart, secs, bars=bars)
        if not neuves:
            rates.append((stem, "aucune section utilisable"))
            continue
        print(f"\n{stem}")
        print(f"   avant : {ancien(chart)}")
        print(f"   après : {resume(segs)}")
        print(f"   empilement : "
              + (f"{rap.get('n_reecrites')} mesures réécrites" if rap.get("ok")
                 else f"REFUSÉ ({rap.get('raison')})"))
        if sec:
            continue
        BAK.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, BAK / p.name)
        chart["sections"] = neuves
        chart["fold"] = rap.get("rapport") or {}
        chart["form"] = None
        p.write_text(json.dumps(chart, ensure_ascii=False), encoding="utf-8")
        faits += 1

    print(f"\n{'(à sec) ' if sec else ''}{faits} chart(s) réécrit(s)"
          + (f", copies dans {BAK}" if faits else ""))
    for stem, why in sautes:
        print(f"   sauté  {stem} — {why}")
    for stem, why in rates:
        print(f"   RATÉ   {stem} — {why}")


if __name__ == "__main__":
    main()
