#!/usr/bin/env python3
"""scripts/chart_pli.py — écrire la règle du repliement dans de VRAIS charts.

Louis, 2026-08-18 : « c'est top, tu peux me montrer les charts générés comme
ça ? »

La règle est celle qu'il a dictée sur trois messages (voir demo_repliement.py) :
on découpe en tours ancrés sur ses sections, on groupe par ressemblance
harmonique SANS regarder les lettres, on ne replie qu'à partir de trois tours,
une case qui alterne avec le rang du tour donne une 1re et une 2e fin plutôt
qu'une variante, et une variante doit revenir √n fois pour être écrite.

Ce script ne touche AUCUN chart existant : il écrit à côté, sous le préfixe
`pli_`, que le serveur sert tel quel sur `/min/pli_<stem>` et qui apparaît dans
la bibliothèque. Même précaution que `chart_retour.py` : une loi de découpage
ne passe en prod que sur décision de Louis, et deux sessions travaillent sur le
même arbre.

CE QUI EST ÉCRIT DANS L'ACCORD. La variante voyage dans un champ `var` (le
texte du moins commun). L'app la rend en petit au-dessus de l'accord, comme
iRealb — c'est la seule chose que ce chart demande à l'écran de savoir faire en
plus.

    .venv/bin/python scripts/chart_pli.py
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harmonia_min.folding import _bar_vecs                  # noqa: E402
from harmonia_min.nnls_features import extract_bothchroma   # noqa: E402
from harmonia_min.sections import halfbar_features          # noqa: E402
from harmonia_min.soudure import accords_par_mesure         # noqa: E402

NOTE = "C Db D Eb E F Gb G Ab A Bb B".split()
CHARTS = REPO / "harmonia_min" / "state" / "charts"
MARQUES = REPO / "harmonia_min" / "state" / "sections"
LETTRES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"

P = 4
SEUIL = 0.80
MIN_TOURS = 3

MORCEAUX = ["min_maroon_5_this_love", "min_X-yIEMduRXk",
            "min_norah_jones_don_t_know_why"]


def nom(c: dict) -> str:
    return "N.C." if c.get("nc") else NOTE[c["root"] % 12] + (c.get("q") or "")


def texte(bar) -> str:
    return " ".join(nom(c) for c in bar) or "%"


def min_var(n: int) -> int:
    return max(2, math.ceil(math.sqrt(n)))


def construire(stem: str) -> dict | None:
    chart = json.loads((CHARTS / f"{stem}.json").read_text(encoding="utf-8"))
    grid = chart["barGrid"]
    n_bars = chart.get("nBars") or (len(grid) - 1)
    audio = REPO / "docs" / "audio" / Path(chart["audio_url"]).name
    if not audio.exists():
        return None
    arr, times = extract_bothchroma(audio)
    Vb = _bar_vecs(halfbar_features(grid, arr, times), n_bars)
    bars = accords_par_mesure(chart)

    fm = MARQUES / f"{Path(chart['audio_url']).stem}.json"
    if fm.exists():
        ancres = [(x["b0"], x["b1"])
                  for x in json.loads(fm.read_text(encoding="utf-8"))["sections"]]
    else:
        ancres = [tuple(r) for s in chart.get("sections") or []
                  for r in (s.get("barRanges") or [])]
    departs, rang = [], {}
    for b0, b1 in ancres:
        for t in range((b1 - b0 + 1) // P):
            departs.append(b0 + t * P)
            rang[b0 + t * P] = t
    departs = sorted(set(departs))
    if len(departs) < 2:
        return None

    def sim(a, b):
        return float(np.mean([Vb[a + k] @ Vb[b + k] for k in range(P)]))

    restants, groupes = list(departs), []
    while restants:
        centre = max(restants,
                     key=lambda a: sum(sim(a, b) >= SEUIL for b in restants))
        g = [b for b in restants if sim(centre, b) >= SEUIL]
        groupes.append(sorted(g))
        restants = [b for b in restants if b not in g]
    groupes.sort(key=lambda x: x[0])

    def tete(t):
        return t.split(" ")[0]

    # chaque groupe -> une lettre (les groupes non repliés en prennent une aussi,
    # sinon le morceau aurait des trous)
    secs, i_lettre = [], 0
    for g in groupes:
        txts = {a: [texte(bars[a + k]) for k in range(P)] for a in g}
        replie = len(g) >= MIN_TOURS
        divergentes = [k for k in range(P) if len({txts[a][k] for a in g}) > 1]
        alterne = None
        for k in divergentes:
            par = {}
            for a in g:
                par.setdefault(rang.get(a, 0) % 2, set()).add(tete(txts[a][k]))
            if (len(par) == 2 and all(len(v) == 1 for v in par.values())
                    and par[0] != par[1] and alterne is None):
                alterne = k

        lab = LETTRES[i_lettre % len(LETTRES)]
        i_lettre += 1

        if not replie:
            # chaque tour garde son texte : une section par tour
            for a in g:
                secs.append(_section(chart, bars, lab, a, P, [a]))
                lab = LETTRES[i_lettre % len(LETTRES)]
                i_lettre += 1
            continue

        if alterne is not None:
            # la boucle fait 2P mesures : on écrit la paire, pas le tour
            paires = sorted(a for a in g if rang.get(a, 0) % 2 == 0
                            and (a + P) in rang)
            if paires:
                secs.append(_section(chart, bars, lab, paires[0], 2 * P, paires,
                                     modele_de=g, rang=rang, alterne=True))
                continue

        secs.append(_section(chart, bars, lab, g[0], P, g, modele_de=g))

    secs.sort(key=lambda s: s["barRanges"][0][0])
    neuf = dict(chart)
    neuf["file"] = f"pli_{Path(chart['audio_url']).stem}"
    neuf["title"] = (chart.get("title") or "") + " — pli"
    neuf["sections"] = secs
    neuf["fold"] = {}
    neuf["form"] = None
    neuf["meta"] = {**(chart.get("meta") or {}), "source": "chart_pli",
                    "seuil": SEUIL, "min_tours": MIN_TOURS, "tour": P}
    return neuf


def _section(chart, bars, lab, modele, L, departs, *, modele_de=None,
             rang=None, alterne=False) -> dict:
    """Une section écrite une fois, jouée `len(departs)` fois.

    `modele_de` : les tours dont on tire le texte écrit — la majorité case par
    case, et le moins commun dans `var` s'il revient √n fois.
    """
    grid = chart["barGrid"]
    n = len(grid) - 1
    departs = [d for d in departs if d + L <= n]
    ecrit = []
    for k in range(L):
        b = modele + k
        case = [dict(c) for c in bars[b]]
        if modele_de and len(modele_de) >= MIN_TOURS:
            # la case homologue de chaque tour du groupe
            if alterne and rang is not None:
                pairs = [a for a in modele_de
                         if rang.get(a, 0) % 2 == (0 if k < L // 2 else 1)]
                homo = [a + (k % (L // 2)) for a in pairs]
            else:
                homo = [a + k for a in modele_de]
            vals = [texte(bars[x]) for x in homo if x < n]
            uniq = sorted(set(vals), key=lambda x: (-vals.count(x), x))
            if uniq:
                # la case écrite EST la majorité du groupe, pas celle du modèle
                gagnant = next((x for x in homo
                                if x < n and texte(bars[x]) == uniq[0]), None)
                if gagnant is not None:
                    case = [dict(c) for c in bars[gagnant]]
                var = [u for u in uniq[1:] if vals.count(u) >= min_var(len(vals))]
                # UNE VARIANTE EST UN ACCORD, pas une mesure. Sur le refrain
                # d'Easy On Me la case majoritaire est `F/A G-7` et la variante
                # `G-7 F` : deux accords contre deux accords, écrits en exposant
                # ça déborde sur la case voisine et ça ne veut plus rien dire.
                # Une mesure qui change de DÉCOUPAGE n'est pas une variation de
                # lecture, c'est autre chose — on ne l'écrit pas.
                if (var and case and len(case) == 1
                        and " " not in var[0] and " " not in uniq[0]):
                    case[0]["var"] = var[0]
        for c in case:
            c["bar"] = b
        ecrit.append(case)
    return {
        "id": f"L{lab}{modele}", "label": lab, "tag": "", "reps": len(departs),
        "spans": [[float(grid[d]), float(grid[min(d + L, n)])] for d in departs],
        "barRanges": [[d, min(d + L - 1, n - 1)] for d in departs],
        "bars": ecrit,
        "barSpans": [[[float(grid[d + k]), float(grid[min(d + k + 1, n)])]
                      for d in departs] for k in range(L)],
    }


def main() -> None:
    for stem in MORCEAUX:
        neuf = construire(stem)
        if neuf is None:
            print(f"  — {stem} : sauté")
            continue
        p = CHARTS / f"{neuf['file']}.json"
        p.write_text(json.dumps(neuf), encoding="utf-8")
        forme = " ".join(f"{s['label']}×{s['reps']}" if s["reps"] > 1
                         else s["label"] for s in neuf["sections"])
        n_var = sum(1 for s in neuf["sections"] for bar in s["bars"]
                    for c in bar if c.get("var"))
        print(f"  {p.name}  ({len(neuf['sections'])} sections : {forme})")
        print(f"     {n_var} variante(s) écrite(s)")
        print(f"     http://100.89.209.63:7772/?open={neuf['file']}")


if __name__ == "__main__":
    main()
