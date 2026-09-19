"""tools/sections_nouveautes.py — CE QUE L'APP DONNERAIT AVEC LES NOUVEAUTÉS

    .venv/bin/python -m tools.sections_nouveautes SORTIE.json

Rejouable : c'est la mesure qui a produit `docs/plots/sections_nouveautes_
2026-09-20.html`. À relancer après toute touche à `sections/nouveaute.py`, à
`sections/simulation.py` ou à la grille — le chiffre qu'elle rend n'a de sens
que comparé au précédent., dans le vrai régime d'usage.

Le régime, c'est celui du bouton « Valider les sections » : Louis pose UNE
brique, et la machine doit en déduire le reste. On simule donc exactement ça —
sa PREMIÈRE section qui n'est pas une intro — et on compare trois choses :

    ce qu'il a écrit à la main        (la vérité)
    ce que l'app rend AUJOURD'HUI     (briques + agglomération des 4 mots)
    ce qu'elle rendrait AVEC          (briques + transposition + nouveauté
                                       calée sur la grille de sa brique)

LA GRILLE VIENT DE SA BRIQUE, et c'est la seule origine honnête : dans l'app,
c'est la seule chose qu'on ait. Mesuré sur ses 22 annotations, 76 % de ses
frontières tombent sur une grille de 4 mesures.
"""
import json, sys, os
from pathlib import Path
sys.path.insert(0, os.getcwd())
import numpy as np
from harmonia import musx as M
from harmonia.sections import similarity as VS
from harmonia.sections.nouveaute import courbe_de_nouveaute, frontieres
from harmonia.sections.simulation import cherche_brique, occurrences_de_tous, remplir_par_algo

AUDIO = Path("docs/audio")
CHARTS = Path("state/cache/charts")
INTROS = ("intro", "silence", "outro")
#: la période que ses annotations imposent (180/238 frontières, 22 morceaux)
PERIODE = 4
#: au-delà, une proposition n'est pas « un peu à côté » de la grille : elle est
#: ailleurs, et on la jette plutôt que de la tirer de force
TOL_GRILLE = 2

_charts = None
def chart_de(stem):
    global _charts
    if _charts is None:
        _charts = {}
        for p in CHARTS.glob("*.json"):
            try: d = json.loads(p.read_text())
            except Exception: continue
            s = Path(d.get("audio_url") or "").stem
            if s: _charts.setdefault(s, (p.stem, d))
    return _charts.get(stem, (None, None))

def cale_sur_grille(prop, b0, n, g=PERIODE, tol=TOL_GRILLE):
    out = []
    for p in prop:
        q = b0 + round((p - b0) / g) * g
        if 0 < q < n and abs(q - p) <= tol: out.append(q)
    return sorted(set(out))

def blocs_depuis(frontieres_, occ, n):
    """Des frontières + les briques → des plages contiguës qui pavent."""
    bornes = sorted({0, n} | set(frontieres_)
                    | {o["m0"] for o in occ} | {o["m1"] + 1 for o in occ})
    bornes = [b for b in bornes if 0 <= b <= n]
    par_brique = {(o["m0"], o["m1"]): o["label"] for o in occ}
    out, libres, vus = [], [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                            if c not in {o["label"].upper() for o in occ}], {}
    for a, b in zip(bornes, bornes[1:]):
        if b - a < 1: continue
        lab = par_brique.get((a, b - 1))
        out.append({"m0": a, "m1": b - 1,
                    "label": lab if lab else "?", "brique": lab is not None})
    # les trous sans nom prennent une lettre neuve, par ordre d'apparition
    k = 0
    for bl in out:
        if bl["brique"]: continue
        bl["label"] = libres[k % len(libres)] if libres else "?"; k += 1
    return out

def frontieres_de(blocs):
    return sorted({b["m0"] for b in blocs if b["m0"] > 0})

resultats = []
for p in sorted(Path("state/human/sections").glob("*.json")):
    stem = p.stem
    h = json.loads(p.read_text())
    secs = h.get("sections") or []
    if len(secs) < 4: continue
    fkey, ch = chart_de(stem)
    if ch is None or not (AUDIO / f"{stem}.m4a").exists(): continue
    grid = ch.get("barGrid") or []
    n = len(grid) - 1
    if n < 8: continue
    siennes = sorted({s["b0"] for s in secs if 0 < s["b0"] < n})

    # SA BRIQUE : la première section qui n'est pas une intro
    brique = next((s for s in secs
                   if (s.get("label") or "").strip().lower() not in INTROS
                   and s["b1"] < n), None)
    if brique is None: continue
    humain = [{"label": brique["label"], "mesure_debut": brique["b0"] + 1,
               "mesure_fin": min(brique["b1"], n - 1) + 1}]

    try:
        occ, _ecartes, diags = occurrences_de_tous(ch, humain, AUDIO)
        avant = remplir_par_algo(ch, occ, AUDIO)
        triad = M.frame_posteriors(AUDIO / f"{stem}.m4a")[0]
        S = VS.ssm(triad, grid)
        cou = courbe_de_nouveaute(S)
        occupe = np.zeros(n, bool)
        for o in occ: occupe[max(0, o["m0"]):min(n, o["m1"] + 1)] = True
        trous, i = [], 0
        while i < n:
            if occupe[i]: i += 1; continue
            j = i
            while j + 1 < n and not occupe[j + 1]: j += 1
            if j - i + 1 >= 4: trous.append((i, j))
            i = j + 1
        brut = frontieres(S, dans=trous, tuilage=False)
        cales = cale_sur_grille(brut, brique["b0"], n)
        apres = blocs_depuis(cales, occ, n)
    except Exception as exc:
        print(f"  !! {stem}: {exc}"); continue

    def sc(fr):
        j = sum(1 for b in siennes if any(abs(b - x) <= 1 for x in fr))
        inv = sum(1 for x in fr if not any(abs(b - x) <= 1 for b in siennes))
        return j, inv
    f_av, f_ap = frontieres_de([{"m0": b["m0"]} for b in avant]), frontieres_de(apres)
    resultats.append({
        "stem": stem, "file": fkey, "n": n, "siennes": siennes,
        "brique": {"label": brique["label"], "m0": brique["b0"], "m1": brique["b1"]},
        "occ": occ, "diag": diags,
        "avant": [{"m0": b["m0"], "m1": b["m1"], "label": b["label"],
                   "source": b.get("source", "")} for b in avant],
        "apres": apres,
        "frontieres_avant": f_av, "frontieres_apres": f_ap,
        "score_avant": sc(f_av), "score_apres": sc(f_ap),
        "nouveaute_brute": brut, "nouveaute_calee": cales,
        "courbe": [round(float(x), 4) for x in cou],
    })
    ja, ia = sc(f_av); jp, ip = sc(f_ap)
    tr = sum(1 for o in occ if o.get("rot"))
    print(f"{stem[:34]:<34} {len(siennes):>3} front. | avant {ja:>2}✓ {ia:>2}✗ "
          f"| après {jp:>2}✓ {ip:>2}✗ | reprises {len(occ)-1} (dont {tr} transposées)")

Path(sys.argv[1]).write_text(json.dumps(resultats, ensure_ascii=False))
A = sum(r["score_avant"][0] for r in resultats); Ai = sum(r["score_avant"][1] for r in resultats)
B = sum(r["score_apres"][0] for r in resultats); Bi = sum(r["score_apres"][1] for r in resultats)
T = sum(len(r["siennes"]) for r in resultats)
print("-" * 100)
print(f"{len(resultats)} morceaux, {T} frontières de Louis")
print(f"   AUJOURD'HUI : {A} justes ({100*A/T:.0f} %), {Ai} inventées")
print(f"   AVEC        : {B} justes ({100*B/T:.0f} %), {Bi} inventées")
