"""Le tableau de bord du découpage : frontières justes, sections exactes.

    .venv/bin/python scripts/bench_sections.py [--voix]

Deux nombres par morceau, et rien d'autre :

  * **frontières** — combien des siennes on retrouve à la mesure EXACTE, et
    combien des nôtres sont justes (rappel / précision) ;
  * **étiquettes** — parmi les sections qu'on a posées au bon endroit, combien
    portent la même lettre que la sienne, à renommage près (on cherche la
    meilleure correspondance lettre-à-lettre, sinon « A » contre « B » compterait
    faux alors que le découpage est le même).

Il existe pour une raison : `mots4` a sept morceaux exacts et cinq faux, et
toute modification du groupage doit être jugée sur les DOUZE avant d'être
gardée (règle d'erreur #6 du CLAUDE.md — un échange de composant déplace plus
que la métrique visée).
"""
from __future__ import annotations

import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "scratchpad"))

from ssm_zoo import SONGS, AUDIO, gt_sections     # noqa: E402
from vote_fill import fill                        # noqa: E402
import mots4                                      # noqa: E402
import order_bundle                               # noqa: E402


def etiquettes(nous, toi, n):
    """La part des mesures dont l'étiquette est juste, au meilleur renommage.

    On apparie nos lettres aux siennes par vote majoritaire (la lettre à nous
    qui recouvre le plus de mesures d'une lettre à lui la prend), puis on compte
    les mesures d'accord. C'est la mesure de recouvrement pondérée habituelle,
    insensible au nom des lettres.
    """
    a = np.empty(n, object); b = np.empty(n, object)
    for s in nous:
        a[s["b0"]:s["b1"] + 1] = str(s.get("nu", s["label"])).rstrip("′")
    for s in toi:
        b[s["b0"]:s["b1"] + 1] = str(s["label"])
    paires = Counter((x, y) for x, y in zip(a, b) if x is not None and y is not None)
    carte, pris = {}, set()
    for (x, y), _c in paires.most_common():
        if x in carte or y in pris:
            continue
        carte[x], _ = y, pris.add(y)
    return float(np.mean([carte.get(x) == y for x, y in zip(a, b)]))


def une(stem, voix=False):
    b = order_bundle.get(stem)
    n = b["n"]
    Sv = np.nan_to_num(np.asarray(b["M"], float))
    essais = {}
    for lien in ("complet", "moyen"):
        R = fill(b, stem, lien=lien)
        se, _d = mots4.sections(R["mot"], R["x0"], n, R["sim"], b["start"],
                                cuts=set(R["cuts"]), Sv=Sv,
                                stem=stem if voix else None)
        essais[lien] = (mots4.cout(se), se)
    lien = min(essais, key=lambda k: (essais[k][0], k != "complet"))
    nous = essais[lien][1]
    toi = gt_sections(stem)["sections"]
    nb = {s["b0"] for s in nous if s["b0"] > 0}
    tb = {s["b0"] for s in toi if s["b0"] > 0}
    return {"n": n, "nous": nous, "toi": toi, "bons": len(nb & tb),
            "nous_n": len(nb), "toi_n": len(tb),
            "eti": etiquettes(nous, toi, n), "lien": lien}


def main():
    voix = "--voix" in sys.argv
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    todo = [(s, t) for s, t in SONGS if s in args] or list(SONGS)
    tot = [0, 0, 0]
    eti = []
    print(f"{'morceau':24s} {'frontières':>18s}  {'étiquettes':>10s}")
    for stem, title in todo:
        if not (AUDIO / f"{stem}.m4a").exists():
            continue
        r = une(stem, voix)
        tot[0] += r["bons"]; tot[1] += r["nous_n"]; tot[2] += r["toi_n"]
        eti.append(r["eti"])
        exact = "  EXACT" if r["bons"] == r["nous_n"] == r["toi_n"] else ""
        print(f"{title:24s} {r['bons']:2d}/{r['nous_n']:2d} justes · "
              f"{r['bons']:2d}/{r['toi_n']:2d} siennes  {r['eti']:9.0%}{exact}")
    print(f"\n{'TOTAL':24s} précision {tot[0]}/{tot[1]} = {tot[0]/max(1,tot[1]):.0%}"
          f" · rappel {tot[0]}/{tot[2]} = {tot[0]/max(1,tot[2]):.0%}"
          f" · étiquettes {np.mean(eti):.0%}")


if __name__ == "__main__":
    main()
