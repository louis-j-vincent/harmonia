#!/usr/bin/env python3
"""scripts/chart_pile_demo.py — le chart de Bora Bora avec la nouvelle pile.

Louis, 2026-08-19 : « fais moi une démo sur chart bora bora ».

On écrit un chart JUMEAU, à côté de l'original, pour qu'il puisse passer de
l'un à l'autre dans la bibliothèque de l'app et lire la différence sur le
chart lui-même :

    Bora Bora                            la pile d'aujourd'hui (par lettre)
    Bora Bora — pile par mesure + ton    gate="bar", transpose=True

MÊME AUDIO, MÊME GRILLE, MÊMES SECTIONS. Une seule chose change : la loi
d'empilement. C'est ce qui rend la comparaison lisible — si les sections
bougeaient aussi, on ne saurait plus à quoi attribuer l'écart.

Les mesures que la nouvelle loi réécrit sont listées à la fin, pour pouvoir
aller les écouter directement.

    .venv/bin/python scripts/chart_pile_demo.py [--chart min_T64BgKEL-Sw]
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

ETAT = REPO / "harmonia_min" / "state"
AUD = REPO / "docs" / "audio"
SUFFIXE = "_pile"

NOTES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def txt(c) -> str:
    if c.get("nc"):
        return "N.C."
    t = NOTES[c["root"] % 12] + (c.get("q") or "")
    if c.get("bass", -1) >= 0 and c["bass"] != c["root"]:
        t += "/" + NOTES[c["bass"] % 12]
    return t


def mesure(bar) -> str:
    return " ".join(txt(c) for c in bar) if bar else "·"


def main() -> None:
    stem = "min_T64BgKEL-Sw"
    if "--chart" in sys.argv:
        stem = sys.argv[sys.argv.index("--chart") + 1]
    src = ETAT / "charts" / f"{stem}.json"
    chart = json.loads(src.read_text(encoding="utf-8"))
    grid = chart["barGrid"]
    audio = AUD / Path(chart["audio_url"]).name
    bpb = int(chart.get("bpb") or 4)

    from harmonia_min import musx as _musx
    from harmonia_min.folding import fold_letter_groups
    from harmonia_min.nnls_features import extract_bothchroma
    from harmonia_min.soudure import accords_par_mesure, sections_pour_chart

    brut = accords_par_mesure(chart)          # le brut : la vérité terrain
    probs = _musx.frame_posteriors(audio)
    arr, times = extract_bothchroma(audio)
    cqt = _musx.song_cqt(audio)

    # LES SECTIONS DU CHART D'ORIGINE, inchangées : seule la loi de pile bouge.
    secs, fold_in = [], []
    for sec in chart.get("sections") or []:
        for b0, b1 in sec.get("barRanges") or []:
            lab = str(sec.get("label") or "?")
            secs.append({"label": lab, "mesure_debut": int(b0) + 1,
                         "mesure_fin": int(b1) + 1})
            fold_in.append({"label": lab, "barRanges": [[int(b0), int(b1)]]})
    secs.sort(key=lambda s: s["mesure_debut"])
    fold_in.sort(key=lambda s: s["barRanges"][0][0])

    avant = copy.deepcopy(brut)
    fold_letter_groups(fold_in, avant, grid, probs, bpb, arr=arr, times=times,
                       combine="cqt", cqt=cqt, loop="occurrence")
    apres = copy.deepcopy(brut)
    rap = fold_letter_groups(fold_in, apres, grid, probs, bpb, arr=arr,
                             times=times, combine="cqt", cqt=cqt,
                             loop="occurrence", gate="bar", transpose=True)

    jumeau = dict(chart)
    # `file` porte le préfixe `min_`, comme dans les charts d'origine —
    # c'est la clé que l'app et le lien `?open=` utilisent.
    jumeau["file"] = f"{stem}{SUFFIXE}"
    jumeau["title"] = (chart.get("title") or stem) + " — pile par mesure + ton"
    jumeau["sections"] = sections_pour_chart(chart, secs, bars=apres)
    jumeau["fold"] = rap
    jumeau["form"] = None
    out = ETAT / "charts" / f"{stem}{SUFFIXE}.json"
    out.write_text(json.dumps(jumeau, ensure_ascii=False), encoding="utf-8")

    diff = [b for b in range(len(brut)) if mesure(avant[b]) != mesure(apres[b])]
    print(f"écrit {out.name}")
    print(f"\n{len(diff)} mesure(s) où les deux lois écrivent autre chose :\n")
    for b in diff:
        print(f"  mes. {b + 1:>3}  ({grid[b]:6.1f} s)   "
              f"aujourd'hui : {mesure(avant[b]):<22}  "
              f"nouvelle : {mesure(apres[b])}")
    for L, r in rap.items():
        if r.get("demiton"):
            qui = sorted({v for v in r["demiton"].values()})
            print(f"\n  lettre {L} : un passage ramené de {qui} demi-ton(s) "
                  "avant la pile")
    print(f"\n  http://100.89.209.63:7772/?open={jumeau['file']}")
    print(f"  http://100.89.209.63:7772/?open={stem}   (l'original)")


if __name__ == "__main__":
    main()
