"""La forme minimale RENDUE PAR LE CHART DE L'APP, pas par une page à part.

    python scripts/minimal_form_charts.py [<stem> ...]
    python scripts/minimal_form_charts.py --clean      # retire les entrées

Louis, 2026-08-08 : « top pour le repliement minimal, montre-moi ce que ça
donnerait en visualisation chart harmonia ».

`/reports/forme_minimale.html` montre la forme dans une page de démonstration.
Ce n'est pas ce qu'il demande : il veut la voir DANS le chart, avec la vraie
grille, le vrai curseur de lecture et le vrai rendu. Donc aucun code de l'app
n'est touché ici — on écrit des ChartModel supplémentaires, exactement comme le
faisaient les variantes `__ug`/`__dict`, et l'app les dessine avec son propre
moteur, inchangé.

    /?open=min_<stem>__mini

CE QUE L'APP SAIT DESSINER, ET CE QU'IL FAUT LUI DONNER. Une section de
ChartModel est « un bloc de mesures écrites, joué N fois » :

    {label, reps, bars, barRanges[N], spans[N], barSpans[écrite -> fenêtres]}

`barSpans[r]` est la clé : la liste des fenêtres de temps où la mesure ÉCRITE
numéro r se joue réellement. C'est ce qui permet à une seule mesure dessinée de
s'allumer à ses six passages. La forme minimale se traduit donc sans rien
inventer : une cellule devient un bloc dont chaque mesure s'allume à toutes ses
reprises, une retouche devient un petit bloc qui ne s'allume qu'aux reprises
qui la portent, un littéral un bloc qui ne s'allume qu'à son seul endroit.

UNE MESURE ÉCRITE NE S'ALLUME QU'À UN SEUL ENDROIT À LA FOIS. Les positions
qu'une retouche remplace sont retirées des fenêtres de la cellule : sans ça, à
la reprise où la fin change, la cellule ET la retouche s'allumeraient ensemble
et le curseur mentirait sur ce qu'on entend.

CE QUE ÇA N'EST PAS : ces entrées sont des DÉMONSTRATIONS, pas la bibliothèque.
Elles portent le suffixe `__mini` et le titre « · MINI », et `--clean` les
retire toutes. Rien dans l'app ne change tant que Louis n'a pas tranché.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

from minimal_form_capture import capture, stems                       # noqa: E402
from harmonia_min import minimal_form as MF                           # noqa: E402

CHARTS = HERE / "harmonia_min/state/charts"
SUFFIX = "__mini"


def _win(grid, b):
    """La fenêtre de temps de la mesure absolue b."""
    return [grid[b], grid[min(len(grid) - 1, b + 1)]]


def sections_of(form, ranges, grid, all_bars, endings_only=False):
    """SongForm -> les sections que le chart de l'app sait dessiner.

    LES MESURES ÉCRITES SONT REPRISES DU MORCEAU, PAS DE LA FORME. `minimal_form`
    travaille sur des SIGNATURES `(root, q, nc)` — juste assez pour décider ce
    qui boucle — alors que le chart dessine des accords complets (tenues,
    enrichissements, alternatives, confiance). Recopier la signature telle quelle
    donnait un modèle que `loadModel` refusait, et l'app retombait sur la
    bibliothèque sans rien dire. On va donc chercher, pour chaque mesure écrite,
    la vraie mesure à l'endroit qu'elle représente.
    """
    out = []
    n_bars = len(all_bars)
    for lf in form.letters:
        for i, blk in enumerate(lf.blocks):
            name = lf.name(i)
            occ0 = ranges.get(lf.label) or [(0, 0)]

            def absolute(occ, off):
                base = occ0[occ][0] if occ < len(occ0) else 0
                return base + off

            # CE QUE LE CHART SAIT DESSINER, ET CE QU'IL NE SAIT PAS. Une
            # section de ChartModel est « un bloc de mesures, joué N fois » :
            # il n'existe aucune façon d'y écrire « à la 2e reprise, la mesure 1
            # devient N.C. ». Une retouche AU MILIEU devient donc une ligne à
            # elle seule, avec une mesure remplie et trois vides — plus verbeux
            # que la grille dépliée, et illisible. En mode `endings_only` on ne
            # garde que les retouches de FIN (la « 2e fois » du musicien, que
            # l'app dessine bien) et toute reprise à retouche interne est
            # écrite au long.
            inline = set()
            if endings_only:
                for ri, r in enumerate(blk.renditions):
                    if any(not blk.patches[pi].is_ending(blk.period)
                           for pi in r.patches):
                        inline.add(ri)
            covered = []
            for ri, r in enumerate(blk.renditions):
                hid = set()
                for pi in r.patches:
                    p = blk.patches[pi]
                    if endings_only and not p.is_ending(blk.period):
                        continue
                    hid |= set(range(p.pos, p.pos + len(p.bars)))
                covered.append(hid)
            keep = [r for ri, r in enumerate(blk.renditions) if ri not in inline]

            if blk.period and blk.alias_of is None:
                rows = [[] for _ in range(blk.period)]
                cell_bars = []
                for k in range(blk.period):
                    src = next((r for ri, r in enumerate(blk.renditions)
                                if k not in covered[ri]), blk.renditions[0])
                    cell_bars.append(all_bars[absolute(src.occ, src.b0 + k)])
                for ri, r in enumerate(blk.renditions):
                    if ri in inline:
                        continue
                    for k in range(blk.period):
                        if k in covered[ri]:
                            continue
                        b = absolute(r.occ, r.b0 + k)
                        if 0 <= b < n_bars:
                            rows[k].append(_win(grid, b))
                out.append({
                    "id": f"M{name}", "label": name, "tag": "",
                    "reps": len(keep),
                    "spans": [[grid[absolute(r.occ, r.b0)],
                               grid[min(len(grid) - 1,
                                        absolute(r.occ, r.b0) + blk.period)]]
                              for r in keep],
                    "barRanges": [[absolute(r.occ, r.b0),
                                   absolute(r.occ, r.b0) + blk.period - 1]
                                  for r in keep],
                    "bars": cell_bars,
                    "barSpans": rows,
                })

            for pi, p in enumerate(blk.patches):
                if endings_only and not p.is_ending(blk.period):
                    continue
                who = [r for ri, r in enumerate(blk.renditions)
                       if pi in r.patches and ri not in inline]
                if not who:
                    continue
                tag = "fin" if p.is_ending(blk.period) else f"mes. {p.pos + 1}"
                rows = [[] for _ in range(len(p.bars))]
                for r in who:
                    for k in range(len(p.bars)):
                        b = absolute(r.occ, r.b0 + p.pos + k)
                        if 0 <= b < n_bars:
                            rows[k].append(_win(grid, b))
                out.append({
                    "id": f"M{name}p{pi}", "label": f"{name} · {tag}", "tag": "",
                    "reps": len(who),
                    "spans": [[grid[absolute(r.occ, r.b0 + p.pos)],
                               grid[min(len(grid) - 1,
                                        absolute(r.occ, r.b0 + p.pos)
                                        + len(p.bars))]] for r in who],
                    "barRanges": [[absolute(r.occ, r.b0 + p.pos),
                                   absolute(r.occ, r.b0 + p.pos)
                                   + len(p.bars) - 1] for r in who],
                    "bars": [all_bars[absolute(who[0].occ, who[0].b0 + p.pos + k)]
                             for k in range(len(p.bars))],
                    "barSpans": rows,
                })

            for ri in sorted(inline):
                r = blk.renditions[ri]
                a = absolute(r.occ, r.b0)
                out.append({
                    "id": f"M{name}i{ri}", "label": f"{name} · au long",
                    "tag": "", "reps": 1,
                    "spans": [[grid[a], grid[min(len(grid) - 1,
                                                 a + blk.period)]]],
                    "barRanges": [[a, a + blk.period - 1]],
                    "bars": [all_bars[a + k] for k in range(blk.period)
                             if a + k < n_bars],
                    "barSpans": [[_win(grid, a + k)] for k in range(blk.period)
                                 if a + k < n_bars],
                })

            for li, lit in enumerate(blk.literals):
                a = absolute(lit.occ, lit.b0)
                out.append({
                    "id": f"M{name}l{li}", "label": f"{name} · au long",
                    "tag": "", "reps": 1,
                    "spans": [[grid[a], grid[min(len(grid) - 1,
                                                 a + len(lit.bars))]]],
                    "barRanges": [[a, a + len(lit.bars) - 1]],
                    "bars": [all_bars[a + k] for k in range(len(lit.bars))],
                    "barSpans": [[_win(grid, a + k)]
                                 for k in range(len(lit.bars))],
                })
    return out


def build(stem, endings_only=False, suffix=SUFFIX):
    src = CHARTS / f"min_{stem}.json"
    if not src.exists():
        raise FileNotFoundError(src)
    model = json.loads(src.read_text(encoding="utf-8"))
    d = capture(stem)
    form = MF.compress_song(d["sections"], d["bars"])
    ranges = MF.occurrence_ranges(d["sections"])
    secs = sections_of(form, ranges, d["grid"], d["bars"], endings_only)
    key = f"min_{stem}{suffix}"
    model["sections"] = secs
    model["file"] = key
    model["title"] = f"{model.get('title') or stem} · MINI"
    model["fold"] = None
    (CHARTS / f"{key}.json").write_text(json.dumps(model), encoding="utf-8")
    return key, form, sum(len(s["bars"]) for s in secs)


def main():
    if "--clean" in sys.argv:
        gone = 0
        for p in sorted(list(CHARTS.glob(f"*{SUFFIX}.json"))
                        + list(CHARTS.glob("*__mini2.json"))):
            p.unlink()
            gone += 1
        print(f"{gone} entrées « · MINI » retirées")
        return
    eo = "--endings-only" in sys.argv
    sfx = "__mini2" if eo else SUFFIX
    todo = [a for a in sys.argv[1:] if not a.startswith("-")] or stems()
    for st in todo:
        try:
            key, form, written = build(st, endings_only=eo, suffix=sfx)
        except Exception as exc:                       # noqa: BLE001
            print(f"  {st[:44]:46} {type(exc).__name__}: {exc}")
            continue
        print(f"  {st[:44]:46} {written:>3} mesures écrites / "
              f"{form.played:>3} jouées = {written / max(1, form.played):.2f}"
              f"   /?open={key}")


if __name__ == "__main__":
    main()
