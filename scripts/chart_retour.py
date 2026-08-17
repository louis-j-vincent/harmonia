#!/usr/bin/env python3
"""scripts/chart_retour.py — écrire les sections de `retour` dans de VRAIS charts.

Louis, 2026-08-16 : « alors tu me fais le chart harmonia stp, et tout doit y
être ; les accords qui ne sont pas dans des sections, tu leur crées de nouvelles
sections pour voir. »

Ce script ne touche AUCUN chart existant. Il écrit à côté, sous le préfixe
`retour_`, des charts que le serveur sert tels quels sur `/min/retour_<stem>` et
qui apparaissent dans la bibliothèque. Les originaux `min_*` restent intacts :
une loi de découpage ne passe en prod que sur décision de Louis, et deux
sessions travaillent sur le même arbre.

CE QU'ON ÉCRIT. Une section de l'algo est une boucle de `L` mesures jouée
plusieurs fois. Le format de chart, lui, veut un motif écrit une fois plus la
liste de ses reprises — c'est exactement la même chose : **un tour de boucle =
une reprise**. Une suite de 3 tours donne donc 3 entrées dans `barRanges`, et
le motif écrit une fois fait `L` mesures.

LE RESTE. Chaque passage laissé sans section devient une section à lui seul,
étiqueté avec la lettre suivante, pour que RIEN ne manque au chart — c'est la
demande de Louis, et c'est aussi la façon la plus rapide de voir ce que l'algo
n'a pas su rattacher.

CE QUI DOIT VOYAGER AVEC L'ACCORD (2026-08-17). On reconstruit chaque case à
partir de `prompter.chords` — le chart BRUT, le décodage à plat capturé avant
que les sections et le repli ne touchent à quoi que ce soit
(`pipeline.prompter_chords`). C'est la bonne source pour les accords, et c'est
la règle de Louis : la vérité est le chart brut, les sections sont un affichage
par-dessus. Mais elle ne porte QUE les accords : ni les candidats du modèle
(`sug`) ni le compte de répétitions (`n`). Sans eux, le mode Annotate du chart
n'avait plus rien à montrer — et il s'inventait des accords.
Les deux sont donc recalculés ici : `sug` par musx sur l'empan réel de chaque
case (`span_rescore.musx_suggestions`, la même source que la pipeline), `n` par
`chord_confidence.repetition_counts` sur tout le morceau.

    .venv/bin/python scripts/chart_retour.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from harmonia.output.chord_confidence import (  # noqa: E402
    chord_key as _cle_accord, repetition_counts as _comptes)
from harmonia_min import retour as R          # noqa: E402
from demo_retour import MORCEAUX              # noqa: E402

CHARTS = REPO / "harmonia_min" / "state" / "charts"
AUDIO = REPO / "docs" / "audio"


def _accords_de_mesure(chart: dict, b: int, comptes) -> list[dict]:
    """Les accords qui sonnent dans la mesure `b`, au format que la vue attend."""
    grid = chart["barGrid"]
    bpb = int(chart.get("bpb") or 4)
    t0, t1 = float(grid[b]), float(grid[b + 1])
    duree = max(t1 - t0, 1e-6)
    out = []
    for c in (chart.get("prompter") or {}).get("chords") or []:
        a, z = max(t0, float(c["t0"])), min(t1, float(c["t1"]))
        # Les éclats de frontière ne sont pas des accords. Un accord qui déborde
        # d'un centième de seconde sur la mesure voisine y apparaissait comme un
        # deuxième accord : deux symboles dans une case qui n'en joue qu'un.
        # Un vrai demi-temps occupe la moitié de la mesure ; on garde à partir
        # d'un dixième.
        if (z - a) / duree < 0.10:
            continue
        beat = int(round((a - t0) / duree * bpb))
        out.append({"root": int(c["root"]), "q": c.get("q") or "",
                    "bass": int(c.get("bass", -1)), "nc": bool(c.get("nc")),
                    # « carry » = l'accord a commencé AVANT cette mesure. La
                    # vue s'en sert pour ne pas réécrire un accord qui dure :
                    # mal calculé, chaque mesure affichait aussi l'accord de la
                    # précédente.
                    "carry": float(c["t0"]) < t0 - 1e-3,
                    "beat": max(0, min(bpb - 1, beat)), "bar": b,
                    "c": float(c.get("c") or 0.0), "t0": a, "t1": z,
                    # `n` = combien de fois le MORCEAU joue cet accord. C'est
                    # l'observable dont `c` est tiré, et la seule chose que
                    # l'éditeur d'annotation dit à voix haute sur un accord
                    # (« played 7 times in this song »). 0 pour un N.C.
                    "n": 0 if c.get("nc") else int(comptes.get(
                        _cle_accord(int(c["root"]), c.get("q") or ""), 0)),
                    "colour": "harmonic"})
    if not out:      # une mesure sans rien : la vue veut au moins un jeton
        out = [{"root": 0, "q": "", "bass": -1, "nc": True, "carry": False,
                "beat": 0, "bar": b, "c": 0.0, "t0": t0, "t1": t1, "n": 0,
                "colour": "harmonic"}]
    return out


def _section(chart: dict, label: str, L: int, departs: list[int],
             comptes) -> dict:
    """Un motif de `L` mesures et la liste des mesures où il repart."""
    grid = chart["barGrid"]
    n = len(grid) - 1
    departs = [d for d in departs if d + L <= n]
    modele = departs[0]
    return {
        "id": f"L{label}",
        "label": label,
        "tag": "",
        "reps": len(departs),
        "spans": [[float(grid[d]), float(grid[d + L])] for d in departs],
        "barRanges": [[d, d + L - 1] for d in departs],
        "bars": [_accords_de_mesure(chart, modele + k, comptes)
                 for k in range(L)],
        "barSpans": [[[float(grid[d + k]), float(grid[d + k + 1])]
                      for d in departs] for k in range(L)],
    }


def _reste_en_sections(res: dict) -> list[tuple]:
    """Les passages sans section, découpés en morceaux d'un seul tenant."""
    out, courant = [], []
    for b in res["reste"]:
        if courant and b == courant[-1] + 1:
            courant.append(b)
        else:
            if courant:
                out.append((courant[0], courant[-1]))
            courant = [b]
    if courant:
        out.append((courant[0], courant[-1]))
    return out


def _candidats(chart: dict, secs: list[dict]) -> tuple[int, int]:
    """Les candidats du modèle (`sug`) sur l'empan réel de chaque case.

    Rend `(cases servies, cases sonnantes)`. Pas de cache musx = AUCUN
    candidat, et on le dit : l'éditeur préfère n'avoir rien à montrer plutôt
    qu'une liste fabriquée (c'était le bug du 2026-08-17).
    """
    accords = [c for s in secs for bar in s["bars"] for c in bar
               if not c.get("nc")]
    nom = Path(chart.get("audio_url") or "").name
    if not nom:
        return 0, len(accords)
    from harmonia_min.musx import frame_posteriors
    from harmonia_min.span_rescore import musx_cache_path, musx_suggestions
    audio = AUDIO / nom
    if not musx_cache_path(audio).exists():
        return 0, len(accords)
    probs = frame_posteriors(audio)          # cache seul, l'audio n'est pas relu
    return musx_suggestions(probs, accords), len(accords)


def construire(fichier: str) -> dict | None:
    chart = json.loads((CHARTS / f"{fichier}.json").read_text(encoding="utf-8"))
    S = R.ssm_mesures(chart)
    if S is None:
        return None
    res = R.sections(S)

    # comptés sur TOUT le morceau (la liste à plat), pas sur les motifs écrits
    # une fois : « joué 7 fois » parle de la chanson, pas du chart.
    comptes = _comptes((c.get("root"), c.get("q") or "", bool(c.get("nc")))
                       for c in (chart.get("prompter") or {}).get("chords") or [])

    secs, fold = [], {}
    for s in res["sections"]:
        # un tour de boucle = une reprise du motif
        departs = [su["b0"] + t * s["L"]
                   for su in s["suites"] for t in range(su["tours"])]
        if not departs:
            continue
        secs.append(_section(chart, s["label"], s["L"], departs, comptes))
        fold[s["label"]] = {"period": s["L"], "n_obs": [], "variants": [],
                            "changed": [], "cv_skip": [], "pos_skip": [],
                            "coh": []}

    # LE RESTE, une section par passage — « tout doit y être »
    lettres = [x for x in R.LETTRES if x not in {s["label"] for s in secs}]
    for i, (a, b) in enumerate(_reste_en_sections(res)):
        lab = lettres[i] if i < len(lettres) else f"r{i}"
        secs.append(_section(chart, lab, b - a + 1, [a], comptes))
        fold[lab] = {"period": b - a + 1, "n_obs": [], "variants": [],
                     "changed": [], "cv_skip": [], "pos_skip": [], "coh": []}

    secs.sort(key=lambda s: s["barRanges"][0][0])
    servis, sonnants = _candidats(chart, secs)
    neuf = dict(chart)
    neuf["file"] = f"retour_{Path(chart['audio_url']).stem}"
    neuf["title"] = (chart.get("title") or "") + " — retour"
    neuf["sections"] = secs
    neuf["fold"] = fold
    neuf["meta"] = {**(chart.get("meta") or {}),
                    "source": "harmonia_min.retour",
                    "substrat": R.SUBSTRAT,
                    "candidats": [servis, sonnants]}
    return neuf


def main() -> None:
    for fichier, titre in MORCEAUX:
        neuf = construire(fichier)
        if neuf is None:
            print(f"  — {fichier} : pas de matrice, sauté")
            continue
        p = CHARTS / f"{neuf['file']}.json"
        p.write_text(json.dumps(neuf), encoding="utf-8")
        lettres = " ".join(f"{s['label']}×{s['reps']}" for s in neuf["sections"])
        servis, sonnants = neuf["meta"]["candidats"]
        print(f"  {p.name}  ({len(neuf['sections'])} sections : {lettres})")
        print(f"     /min/{neuf['file']}")
        if servis < sonnants:
            print(f"     ⚠ candidats du modèle : {servis}/{sonnants} cases — "
                  f"pas de cache musx, le mode Annotate n'aura rien à montrer")
        else:
            print(f"     candidats du modèle : {servis}/{sonnants} cases")


if __name__ == "__main__":
    main()
