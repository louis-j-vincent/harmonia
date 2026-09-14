"""Capture l'entrée EXACTE du repli d'affichage, morceau par morceau.

Le chart servi par l'app (`harmonia_min/state/charts/min_*.json`) ne contient
plus la suite d'accords DÉPLIÉE : `minimal_fold` n'y écrit qu'un bloc par
lettre. Impossible, donc, de vérifier depuis le chart qu'une forme minimale est
bien reconstructible — la seule chose qu'on pourrait comparer, c'est le bloc
avec lui-même.

On rejoue donc l'analyse et on espionne l'appel à `minimal_fold`, dont les
arguments sont précisément ce qui manque : les sections détectées (une
occurrence chacune), la liste `bars` mesure par mesure APRÈS le repli
d'observation (phase 1, celui qui réécrit les accords), la grille et le rapport
de repli. C'est l'état que voit le repli d'affichage, donc l'état contre lequel
une forme minimale doit être jugée.

    python scripts/minimal_form_capture.py            # tous les charts
    python scripts/minimal_form_capture.py bein_green # un seul

Le cache vit dans `data/cache/minimal_form/<stem>.pkl` (~10 s par morceau au
premier passage, instantané ensuite).
"""
from __future__ import annotations

import copy
import os
import pickle
import sys
import time
from pathlib import Path

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

CACHE = HERE / "data/cache/minimal_form"
AUDIO = HERE / "docs/audio"
CHARTS = HERE / "harmonia_min/state/charts"


def stems() -> list[str]:
    """Les morceaux de la bibliothèque qui ont un audio local."""
    have = {p.stem for p in AUDIO.glob("*.m4a")}
    return sorted(p.stem[4:] for p in CHARTS.glob("min_*.json")
                  if p.stem[4:] in have)


def capture(stem: str, rebuild: bool = False) -> dict:
    """Rejoue l'analyse et rend {grid, bars, sections, fold, bpb, title}."""
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{stem}.pkl"
    if p.exists() and not rebuild:
        with open(p, "rb") as f:
            return pickle.load(f)

    import harmonia_min.folding as FD
    from harmonia_min import pipeline as _pl

    got: dict = {}
    real = FD.minimal_fold

    def spy(sections, bars, grid, fold_report):
        # deepcopy AVANT l'appel : `minimal_fold` ne mute pas, mais la suite du
        # pipeline (clé harmonique, suggestions musx) écrit dans les accords.
        got["sections"] = copy.deepcopy(sections)
        got["bars"] = copy.deepcopy(bars)
        got["grid"] = [float(x) for x in grid]
        got["fold"] = copy.deepcopy(fold_report)
        return real(sections, bars, grid, fold_report)

    FD.minimal_fold = spy
    try:
        model = _pl.analyze(AUDIO / f"{stem}.m4a", title=stem,
                            file_key=stem, audio_url="")
    finally:
        FD.minimal_fold = real
    if "bars" not in got:
        raise RuntimeError(f"{stem}: minimal_fold jamais appelé")
    got["stem"] = stem
    got["bpb"] = model["bpb"]
    got["title"] = model["title"]
    got["folded_sections"] = copy.deepcopy(model["sections"])
    with open(p, "wb") as f:
        pickle.dump(got, f)
    return got


def main() -> None:
    todo = sys.argv[1:] or stems()
    for i, st in enumerate(todo, 1):
        t = time.time()
        try:
            d = capture(st)
        except Exception as e:                       # noqa: BLE001
            print(f"[{i}/{len(todo)}] {st[:50]:52} ÉCHEC {type(e).__name__}: {e}",
                  flush=True)
            continue
        print(f"[{i}/{len(todo)}] {st[:50]:52} {len(d['bars']):4d} mesures, "
              f"{len(d['sections']):2d} sections  ({time.time() - t:.0f} s)",
              flush=True)


if __name__ == "__main__":
    main()
