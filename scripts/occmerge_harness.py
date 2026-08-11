"""Harnais du chantier merge-d'occurrences (branche feat/occurrence-merge).

Pour chaque chart de la bibliothèque live, il :
  1. rejoue le pipeline jusqu'au chart BRUT (bars première passe, avant tout
     fold) — caches chauds, ~2 s par morceau ;
  2. vérifie que sa grille colle à celle du chart live (sinon le morceau est
     sauté : les barRanges des sections ne seraient plus comparables) ;
  3. reconstruit les sections par occurrence depuis le chart live (une
     pseudo-section par barRange d'une lettre) ;
  4. rejoue fold_letter_groups sur une COPIE des bars brutes pour chaque
     variante (prod, gate bi-mesure, médiane, produit, entropie, basse
     sortie…) ;
  5. sauve un instantané JSON par morceau — la matière première de toutes
     les pages d'écoute du handoff 2026-08-08.

    HARMONIA_MUSX_DIR=<clone ISMIR de l'arbre principal> \
    python scripts/occmerge_harness.py [--out DIR] [--only STEM]
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

LIVE = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
LIVE_CHARTS = LIVE / "harmonia_min" / "state" / "charts"
AUDIO_DIR = LIVE / "docs" / "audio"

VARIANTS: dict[str, dict] = {
    "prod": {},
    # l'agrégation retenue par Louis à l'oreille (2026-08-08) : on empile les
    # répétitions AVANT le modèle, dans le domaine du spectre, puis on
    # ré-infère. `cqt_check` y ajoute son contrôle d'adhésion.
    "cqt": {"combine": "cqt"},
    "cqt_check": {"combine": "cqt", "check_thr": 0.60},
    "cqt_bibar": {"combine": "cqt", "gate": "bibar"},
    "bibar": {"gate": "bibar"},
    "median": {"combine": "median"},
    "trim20": {"combine": "trim20"},
    "logpool": {"combine": "logpool"},
    "entropy": {"weight": "entropy"},
    "bassout": {"bass_mode": "skip"},
}


def _jsonable(x):
    if isinstance(x, dict):
        return {k: _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in sorted(x)] if isinstance(x, set) \
            else [_jsonable(v) for v in x]
    if isinstance(x, (np.floating, np.integer)):
        return x.item()
    return x


def raw_chart(audio: Path):
    """Le modèle BRUT du pipeline (bars première passe), générateur refermé
    avant sections/fold — le même code que la prod, arrêté plus tôt."""
    from harmonia_min.pipeline import analyze_steps
    gen = analyze_steps(audio, title=audio.stem, file_key=f"min_{audio.stem}",
                        audio_url=f"/audio/{audio.name}")
    kind, model = next(gen)
    assert kind == "raw", kind
    snap = copy.deepcopy(model)
    gen.close()
    return snap


def run_song(stem: str, chart: dict, out_dir: Path, variants=VARIANTS):
    audio = AUDIO_DIR / f"{stem}.m4a"
    if not audio.exists():
        return f"{stem}: audio absent"
    raw = raw_chart(audio)
    g0, g1 = raw["barGrid"], chart["barGrid"]
    drift = max((abs(a - b) for a, b in zip(g0, g1)), default=0.0)
    if len(g0) != len(g1) or drift > 0.02:
        # Le message ne disait que le nombre de mesures : quand seul le TEMPS
        # avait bougé, il affichait « 84 mesures vs 84 », ce qui ne veut rien
        # dire. Un chart live plus vieux que le traqueur de battues actuel
        # tombe exactement là.
        why = (f"{len(g0)-1} mesures contre {len(g1)-1}"
               if len(g0) != len(g1)
               else f"même nombre de mesures mais {drift*1000:.0f} ms de "
                    f"décalage — le chart live est plus ancien que les "
                    f"battues d'aujourd'hui")
        return f"{stem}: grille divergente ({why}) — sauté"
    grid, bpb = raw["barGrid"], raw["bpb"]
    raw_bars = raw["sections"][0]["bars"]          # _one_section: tout le morceau

    # sections par occurrence, reconstruites du chart live (ordre temporel)
    occs = []
    for sec in chart["sections"]:
        for c0, c1 in sec["barRanges"]:
            occs.append({"label": sec["label"], "barRanges": [[c0, c1]]})
    occs.sort(key=lambda s: s["barRanges"][0][0])

    from harmonia_min import musx as _musx
    from harmonia_min.nnls_features import extract_bothchroma
    probs = _musx.frame_posteriors(audio)
    arr, times = extract_bothchroma(audio)
    need_cqt = any(kw.get("combine") == "cqt" for kw in variants.values())
    cqt = _musx.song_cqt(audio) if need_cqt else None

    from harmonia_min.folding import fold_letter_groups
    snap = {"stem": stem, "title": chart.get("title", stem), "bpb": bpb,
            "grid": grid, "audio": f"/audio/{audio.name}",
            "occurrences": [{"label": s["label"], "range": s["barRanges"][0]}
                            for s in occs],
            "raw_bars": raw_bars, "variants": {}}
    for name, kw in variants.items():
        bars_v = copy.deepcopy(raw_bars)
        report = fold_letter_groups(occs, bars_v, grid, probs, bpb,
                                    arr=arr, times=times, cqt=cqt, **kw)
        snap["variants"][name] = {"bars": bars_v, "report": _jsonable(report)}
    out = out_dir / f"{stem}.json"
    out.write_text(json.dumps(_jsonable(snap)))
    n_diff = {}
    ref = snap["variants"]["prod"]["bars"]
    for name, v in snap["variants"].items():
        if name == "prod":
            continue
        d = sum(1 for a, b in zip(ref, v["bars"])
                if [(c["root"], c["q"], c["nc"]) for c in a]
                != [(c["root"], c["q"], c["nc"]) for c in b])
        n_diff[name] = d
    return f"{stem}: ok ({len(raw_bars)} mesures) — diffs vs prod {n_diff}"


def main(argv):
    out_dir = Path(argv[argv.index("--out") + 1]) if "--out" in argv else \
        Path(__file__).resolve().parents[1] / "harmonia_min" / "state" / "occmerge"
    only = argv[argv.index("--only") + 1] if "--only" in argv else None
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(LIVE_CHARTS.glob("min_*.json")):
        chart = json.loads(p.read_text())
        stem = chart["file"].removeprefix("min_")
        if only and only != stem:
            continue
        try:
            msg = run_song(stem, chart, out_dir)
        except Exception as e:
            msg = f"{stem}: ERREUR {type(e).__name__}: {e}"
        print(msg, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
