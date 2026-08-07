"""La cohérence globale décide — et c'est elle qui choisit l'intro.

    python scripts/coherence.py [<stem> ...]

Louis, 2026-08-07 :

  « Let It Be, il y a un A qu'on ne détecte pas, pourquoi ? Il faudrait un score
    de cohérence globale. Ensuite il faut absolument que tu apprennes à bien
    détecter les intros, je te laisse gérer ça : une chose à prendre en compte,
    **l'intro influe sur la cohérence des sections totales**. »

Sa dernière phrase donne la réponse aux deux problèmes d'un coup. Si une intro
fausse décale tout le reste, alors la bonne intro est **celle qui rend la chanson
la plus cohérente**. On arrête donc de deviner l'intro avec une règle locale (la
voix entre ici, donc ça commence là) et on l'ESSAIE : chaque longueur d'intro
candidate produit un découpage complet, on note ce découpage, on garde le
meilleur.

POURQUOI UN SEUIL PAR PIC NE PEUT PAS SUFFIRE. Le A manquant de Let It Be, mesuré :

    A mesure 17   harmonie 0.999   chant 0.860   score 0.905   retenu
    A mesure 37   harmonie 0.994   chant 0.137   score 0.426   REJETÉ
    A mesure 49   harmonie 0.995   chant 0.877   score 0.905   retenu

L'harmonie le voit aussi bien que les deux autres ; c'est le chant qui l'écrase,
et pour cause — cinq de ses huit mesures sont muettes, c'est le solo de guitare.
Faire abstenir le chant quand il n'entend rien remonte le score à 0.577, ce qui
est juste mais ne suffit pas : sur Let It Be l'harmonie est plate, donc le pic
n'est pas *pointu* et le score le punit. Ce A ne se justifie pas par sa propre
valeur — il se justifie parce qu'il **complète la forme**. Il faut donc une clé
qui note le découpage ENTIER, pas chaque pic isolément.

LA CLÉ. Celle qui a déjà fonctionné dans `blocks_flex` : le coût d'écriture,
Occam pris au pied de la lettre. Combien de mesures faut-il écrire pour rendre la
chanson ?

    coût  =  mesures non expliquées
           + une fois la longueur de chaque bloc DISTINCT
           + `LINE` par passage (une ligne sur la grille coûte de la place)

Une reprise n'est comptée comme expliquée qu'à hauteur de son score : à 0.5 elle
explique la moitié de ses mesures. Une reprise faible ne paie donc pas sa ligne
et se fait refuser toute seule — le seuil cesse d'être une décision et devient
une conséquence.

`LINE` est le seul réglage, et il est mesuré contre la vérité de Louis, pas
choisi.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from pattern_lanes import load                                            # noqa: E402
import vocal_anchor as VA                                                 # noqa: E402
import blocks8 as B8                                                      # noqa: E402
import melody_ssm as MS                                                   # noqa: E402
import melody_check as MC                                                 # noqa: E402
import vocal_melody as VM                                                 # noqa: E402
import voice_first as VF                                                  # noqa: E402
import score_eval as SE                                                   # noqa: E402

FLOOR = 0.40       # sous ça un pic n'est même pas un candidat
UNIT = 2


def anchor(S, M, mute, n, start, block, floor=FLOOR, line=2.0, max_blocks=20):
    """Ancre en partant de `start`, et laisse le COÛT décider de chaque reprise."""
    from scipy.signal import find_peaks
    claimed = np.zeros(n, bool)
    runs, cursor = [], start
    while cursor + block <= n and len(runs) < max_blocks:
        if claimed[cursor:cursor + block].any():
            cursor += UNIT
            continue
        cur_m = MC.slide_on(M, cursor, block, n)
        cur_h = MC.slide_on(S, cursor, block, n)
        sc = VF.block_score(cur_m, cur_h, cursor, n, mute=mute, block=block)
        idx, _ = find_peaks(sc, height=floor, distance=UNIT)
        cand = [int(p) for p in idx
                if (p - cursor) % UNIT == 0 and abs(p - cursor) >= block
                and p + block <= n and not claimed[p:p + block].any()]
        occ = []
        for p in sorted(cand, key=lambda p: -sc[p]):
            if any(abs(p - q) < block for q in occ):
                continue
            # UNE REPRISE DOIT PAYER SA LIGNE. Elle n'explique ses mesures qu'à
            # hauteur de son score ; si ça ne couvre pas le coût d'un passage
            # sur la grille, elle ne vaut pas la peine d'être écrite.
            if sc[p] * block <= line:
                continue
            occ.append(p)
        occ.sort()
        if occ:
            runs.append({"b0": cursor, "occ": occ, "block": block,
                         "scores": [float(sc[p]) for p in occ]})
            for c in [cursor] + occ:
                claimed[c:min(n, c + block)] = True
        cursor += block
    return runs, claimed


def cost(runs, n, start, block, line=2.0):
    """Le coût d'écriture du découpage entier. Plus il est bas, plus c'est cohérent."""
    explained = 0.0
    written = float(start)                       # l'intro s'écrit en entier
    passes = 0
    covered = np.zeros(n, bool)
    for r in runs:
        written += r["block"]                    # le bloc distinct, une fois
        for p, s in zip([r["b0"]] + r["occ"], [1.0] + r["scores"]):
            explained += s * min(block, n - p)
            covered[p:min(n, p + block)] = True
            passes += 1
    rest = float(np.sum(~covered[start:]))
    return (rest + max(0.0, (n - start) - explained) * .5
            + written + line * passes) / max(1, n)


def best_intro(S, M, mute, n, block, line=2.0, cands=None):
    """L'INTRO EST CE QU'ON ESSAIE, plus ce qu'on devine.

    Louis : « l'intro influe sur la cohérence des sections totales ». Donc on
    parcourt les débuts possibles, on construit le découpage complet pour
    chacun, et on garde celui qui coûte le moins cher à écrire. La règle de la
    voix reste dans la liste des candidats — elle a raison souvent — mais elle
    n'est plus seule à décider.
    """
    out = []
    for st in (cands if cands is not None else range(0, min(n // 2, 17), UNIT)):
        runs, _ = anchor(S, M, mute, n, st, block, line=line)
        out.append({"start": st, "runs": runs,
                    "cost": cost(runs, n, st, block, line)})
    return min(out, key=lambda o: o["cost"]), out


def prep(stem):
    S, n, g = load(stem)
    voc = VA.separate_vocals(HERE / f"docs/audio/{stem}.m4a")
    onset, *_ = B8.sing_onset(voc)
    vstart = MS.voice_start(VA.bar_of(g, onset), n) or 0
    tt, ff, vv, rr = VM.track_f0(voc)
    notes, _ = VM.melody_notes(tt, ff, vv, rr)
    notes, _ = VM.clean(notes)
    M, mute = MS.melody_bars(notes, g, n)
    return S, M, mute, n, g, vstart


def main():
    T = SE.truth()
    stems = sys.argv[1:] or [k for k, v in T.items() if v.get("sections")]
    print(f"{'morceau':<34} {'voix':>5} {'coût':>5} {'toi':>5}   frontières / correspondance")
    tot = {}
    for st in stems:
        if not (HERE / f"docs/audio/{st}.m4a").exists():
            continue
        S, M, mute, n, g, vstart = prep(st)
        ref = SE.truth_labels(T[st]["sections"], n)
        his = next((s["b1"] + 1 for s in T[st]["sections"]
                    if s["label"] == "intro"), 0)
        for block in (8, 4):
            for line in (1.0, 2.0, 3.0):
                best, alls = best_intro(S, M, mute, n, block, line)
                pred = np.full(n, -1)
                for i, r in enumerate(best["runs"]):
                    for c in [r["b0"]] + r["occ"]:
                        pred[c:min(n, c + block)] = i
                pred[:best["start"]] = -1
                e, c2 = SE.correspondence(pred, ref)
                pr, rc = SE.bounds_pr(SE.bounds(pred), SE.bounds(ref))
                key = (block, line)
                tot.setdefault(key, []).append((best["start"] == his, e, c2, pr, rc))
                if line == 2.0 and block == 8:
                    print(f"{st[:33]:<34} {vstart:>5} {best['start']:>5} {his:>5}"
                          f"   {e:.2f} / {c2:.2f}   (justes {pr:.2f} trouv. {rc:.2f})")
    print(f"\n{'bloc':>5} {'ligne':>6} {'intro juste':>12} {'frontières':>11} "
          f"{'corresp.':>9} {'justes':>7} {'trouv.':>7}")
    for (block, line), rows in sorted(tot.items()):
        a = np.array([[float(x) for x in r] for r in rows])
        print(f"{block:>5} {line:>6.1f} {a[:,0].mean():>11.0%} {a[:,1].mean():>11.2f} "
              f"{a[:,2].mean():>9.2f} {a[:,3].mean():>7.2f} {a[:,4].mean():>7.2f}")


if __name__ == "__main__":
    main()
