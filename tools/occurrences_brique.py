"""Occurrences d'une brique : ce que fait la prod, et un candidat — page de diagnostic.

Louis, 2026-09-30 : « une fois que l'annotateur a donné une première section,
il faudrait pouvoir BIEN détecter les occurrences de cette section ».

On simule ce geste sur ses découpages validés (`state/human/sections/*.json`,
`validated: true`) : pour chaque lettre, sa PREMIÈRE occurrence devient la
brique, et on compare à ses vraies occurrences

  * ce que rend la prod : `harmonia.sections.simulation.cherche_brique`
    (brique glissée sur la SSM chord-tone, longueur FIXE, seuil
    médiane + 1,5 σ borné [0,85 ; 0,999], pics sans recouvrement,
    transpositions) ;
  * deux candidats, `cherche_fixe(plafond=0,97)` et `cherche_candidat`
    (ci-dessous) — rien n'est branché.

LECTURE SEULE. Ne modifie rien sous `state/`. Les grilles de mesures viennent
de la sauvegarde de chart dont le nombre de mesures égale le `n` de la vérité
terrain (les `min_*.json` ont disparu de `state/cache/charts/` le 2026-09-30).

    .venv/bin/python -m tools.occurrences_brique            # texte, corpus
    .venv/bin/python -m tools.occurrences_brique --page     # + la page HTML
"""
from __future__ import annotations

import argparse
import glob
import html
import json
import os
from pathlib import Path

import numpy as np

from harmonia import musx
from harmonia.sections import similarity as VS
from harmonia.sections import simulation as SIM
from harmonia.settings import SETTINGS

REPO = Path(__file__).resolve().parents[1]
SANS_REPRISE = {"intro", "outro", "silence", "queue"}
PAGE = REPO / "docs" / "plots" / "occurrences_brique_2026-09-30.html"

# ── les candidats ───────────────────────────────────────────────────────────
#: une mesure « colle » à la mesure de même rang dans la brique au-dessus de ça
#: (cosinus chord-tone ; vraies reprises mesurées 0,85–1,0, accords différents
#: 0,1–0,7). = `simulation.PLANCHER_REPRISE`.
TAU = 0.85
#: une section, c'est au moins 4 mesures (Louis, 2026-08-16)
MIN_RUN = 4
#: CANDIDAT 1 — le plafond du seuil. En prod il vaut 0,999 : quand
#: médiane + 1,5 σ dépasse 1 (morceau où la brique revient souvent), le seuil
#: devient 0,999 et recale des reprises à 0,989–0,998.
PLAFOND_CANDIDAT = 0.97
#: ... mais seulement pour une brique d'au moins 8 mesures. Mesuré : sur une
#: brique de 1 à 4 mesures, 0,97 ramasse tout ce qui contient les mêmes
#: accords (This Lazy Song C : +5 fausses ; 0DdCoNbbRvQ B, 1 mesure : +6) —
#: même raison que `simulation.MIN_MESURES_TRANSPOSE`.
PLAFOND_MIN_MESURES = 8


def plafond_candidat(L: int) -> float:
    return PLAFOND_CANDIDAT if L >= PLAFOND_MIN_MESURES else SIM.PLAFOND_REPRISE


def _chart_for(stem: str, n: int):
    """La sauvegarde de chart la plus récente dont la grille a n mesures."""
    for p in sorted(glob.glob(str(REPO / f"state/cache/*/min_{stem}.json")),
                    key=os.path.getmtime, reverse=True):
        try:
            c = json.loads(Path(p).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if len(c.get("barGrid") or []) - 1 == n:
            return c
    return None


def seuil_morceau(dom, plafond: float) -> float:
    """La loi de `simulation.cherche_brique`, plafond en paramètre."""
    return min(plafond, max(SIM.PLANCHER_REPRISE,
                            float(np.median(dom) + SIM.Z_REPRISE * dom.std())))


def cherche_fixe(S, V, b0: int, b1: int, plafond: float = SIM.PLAFOND_REPRISE):
    """Réécriture de `simulation.cherche_brique` sur des matrices déjà
    calculées, plafond en paramètre. Au plafond de la prod elle rend
    EXACTEMENT la même chose (vérifié par `_verifie_prod`)."""
    n = S.shape[0]
    L = b1 - b0 + 1
    dom = np.asarray(VS._slide(S, b0, L, n))[:max(1, n - L + 1)]
    thr = seuil_morceau(dom, plafond)
    pris = [(b0, b1)]
    out = [{"m0": b0, "m1": b1, "rot": 0, "alt": [], "src": "brique"}]

    def moissonne(scores, th, rot):
        for b in sorted(range(len(scores)), key=lambda x: -scores[x]):
            if scores[b] < th:
                break
            if any(not (b + L <= c or d < b) for c, d in pris):
                continue
            pris.append((b, b + L - 1))
            out.append({"m0": b, "m1": b + L - 1, "rot": rot, "alt": [],
                        "src": "fixe", "score": float(scores[b])})
    moissonne(dom, thr, 0)
    if L >= SIM.MIN_MESURES_TRANSPOSE:
        th = min(plafond, thr + SIM.BONUS_TRANSPOSE)
        for k in range(1, 12):
            Sk = V @ np.roll(V, k, axis=1).T
            moissonne(np.asarray(VS._slide(Sk, b0, L, n))[:len(dom)], th, k)
    return sorted(out, key=lambda o: o["m0"]), thr


def _chaine(S, b0: int, L: int, c: int, libre) -> list[int]:
    """Les mesures de la chaîne qui part de `c`, brique en phase 0 : la mesure
    c+t contre la mesure t de la brique, tant que ça colle (≥ TAU). Une seule
    mesure qui ne colle pas, suivie d'une qui colle, est tolérée (bruit de
    décodage). La chaîne doit COMMENCER par une mesure qui colle."""
    n = S.shape[0]
    if not libre[c] or S[b0, c] < TAU:
        return []
    t, last, trous = 0, -1, 0
    while t < L and c + t < n and libre[c + t]:
        if S[b0 + t, c + t] >= TAU:
            last, trous = t, 0
        else:
            trous += 1
            if trous > 1:
                break
        t += 1
    return list(range(c, c + last + 1))


def cherche_candidat(S, V, b0: int, b1: int):
    """CANDIDAT 1 + 2 — rien n'est branché.

    1. la loi de la prod, plafond abaissé à `PLAFOND_CANDIDAT` ;
    2. dans ce qui reste libre, les passages PLUS COURTS que la brique : une
       chaîne mesure par mesure (voir `_chaine`) d'au moins
       max(4, L/2) mesures, dont la moyenne passe le MÊME seuil du morceau
       (le seuil relatif garde son rôle : il protège des jumeaux) ;
    3. une chaîne qui s'arrête dans le dernier quart de la brique est une FIN
       ALTERNATIVE : l'occurrence garde la longueur de la brique, et les
       mesures qui ne collent pas sont marquées `alt`.

    CE QUE ÇA NE RÉSOUT PAS : les jumeaux harmoniques (deux sections aux
    mêmes accords) ; une occurrence qui commence au milieu de la brique ; une
    occurrence PLUS LONGUE que la brique (ses mesures en trop restent libres) ;
    le choix de phase quand la brique contient une boucle interne (décalage
    de 4 mesures).
    """
    n = S.shape[0]
    L = b1 - b0 + 1
    out, thr = cherche_fixe(S, V, b0, b1, plafond_candidat(L))
    libre = np.ones(n, bool)
    for o in out:
        libre[o["m0"]:o["m1"] + 1] = False
    mini = max(MIN_RUN, -(-L // 2))
    if L <= mini:
        return out, thr
    while True:
        best = None
        for c in range(n):
            r = _chaine(S, b0, L, c, libre)
            if len(r) < mini:
                continue
            fin = L - len(r) <= max(1, L // 4)
            m1 = r[0] + L - 1 if fin else r[-1]
            if m1 >= n or not libre[r[0]:m1 + 1].all():
                m1 = r[-1]
                fin = False
            moy = float(np.mean([S[b0 + t, r[0] + t] for t in range(len(r))]))
            if moy < thr:
                continue
            cle = (len(r), moy)
            if best is None or cle > best[0]:
                best = (cle, r, m1, fin)
        if best is None:
            break
        _, r, m1, fin = best
        alt = [x for x in range(r[0], m1 + 1) if S[b0 + x - r[0], x] < TAU] if fin else []
        out.append({"m0": r[0], "m1": m1, "rot": 0, "alt": alt,
                    "src": "fin_alt" if fin else "court"})
        libre[r[0]:m1 + 1] = False
    return sorted(out, key=lambda o: o["m0"]), thr


# ── la simulation, lettre par lettre ─────────────────────────────────────────
def _fam(label: str) -> str:
    return (label or "").rstrip("′'″").strip()


def analyse_morceau(stem: str) -> dict | None:
    d = json.loads((SETTINGS.sections_dir / f"{stem}.json").read_text(encoding="utf-8"))
    if not d.get("validated"):
        return None
    c = _chart_for(stem, d["n"])
    if c is None:
        return None
    secs = sorted(d["sections"], key=lambda s: s["b0"])
    audio = SETTINGS.audio_dir / (Path(c["audio_url"]).stem + ".m4a")
    triad = musx.frame_posteriors(audio)[0]
    grid = c["barGrid"]
    V = VS.harmonic_vectors(triad, grid)
    S = V @ V.T
    n = S.shape[0]
    lettres = []
    vues = set()
    for s in secs:
        lab = _fam(s["label"])
        if lab.lower() in SANS_REPRISE or lab in vues:
            continue
        vues.add(lab)
        vraies = [(x["b0"], x["b1"], x["label"]) for x in secs if _fam(x["label"]) == lab]
        if len(vraies) < 2:
            continue
        b0, b1 = s["b0"], s["b1"]
        L = b1 - b0 + 1
        prod, info = SIM.cherche_brique(c, b0, b1, SETTINGS.audio_dir, label=lab)
        dom = np.asarray(VS._slide(S, b0, L, n))[:max(1, n - L + 1)]
        rotmax = np.zeros_like(dom)
        rotarg = np.zeros(len(dom), int)
        for k in range(1, 12):
            dk = np.asarray(VS._slide(V @ np.roll(V, k, axis=1).T, b0, L, n))[:len(dom)]
            better = dk > rotmax
            rotmax[better], rotarg[better] = dk[better], k
        fixe, _ = cherche_fixe(S, V, b0, b1)
        assert [(o["m0"], o["m1"], o["rot"]) for o in fixe] == \
            [(o[0], o[1], o[3]) for o in prod], f"réécriture ≠ prod sur {stem} {lab}"
        plafond, thr_p = cherche_fixe(S, V, b0, b1, plafond_candidat(L))
        cand, thr_c = cherche_candidat(S, V, b0, b1)
        lettres.append({
            "label": lab, "b0": b0, "b1": b1, "L": L,
            "vraies": vraies, "info": info,
            "prod": [{"m0": o[0], "m1": o[1], "score": float(o[2]), "rot": int(o[3])} for o in prod],
            "plafond": plafond, "thr_plafond": thr_p, "cand": cand, "thr_cand": thr_c,
            "courbe": [round(float(x), 3) for x in dom],
            "courbe_brute": dom, "thr_prod": seuil_morceau(dom, SIM.PLAFOND_REPRISE),
            "courbe_rot": [round(float(x), 3) for x in rotmax],
            "rot_arg": [int(x) for x in rotarg],
            "S": S, "V": V,
        })
    return {"stem": stem, "n": n, "secs": secs, "grid": grid,
            "audio": "/audio/" + audio.name, "lettres": lettres}


def _chevauche(a0, a1, b0, b1) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def _par_mesure(S, b0, L, g0, g1, n):
    """Ressemblance mesure par mesure de l'occurrence vraie à la brique, en
    phase 0 (la mesure g0+t contre la mesure (t mod L) de la brique)."""
    return [round(float(S[b0 + (t % L), g0 + t]), 2) for t in range(g1 - g0 + 1) if g0 + t < n]


def classe(let: dict, n: int, cands: list[dict], secs) -> tuple[list[dict], list[dict]]:
    """Pour chaque vraie occurrence : trouvée / décalée / longueur / ratée, et
    pourquoi. Pour chaque trouvaille hors vérité : sur quoi elle tombe."""
    S, b0, L = let["S"], let["b0"], let["L"]
    thr = let["thr_prod"]
    rows = []
    for g0, g1, glab in let["vraies"]:
        if (g0, g1) == (let["b0"], let["b1"]):
            continue
        Lg = g1 - g0 + 1
        pm = _par_mesure(S, b0, L, g0, g1, n)
        sc = float(let["courbe_brute"][g0]) if g0 < len(let["courbe_brute"]) else None
        rs = let["courbe_rot"][g0] if g0 < len(let["courbe_rot"]) else None
        best = None
        for o in cands:
            if o.get("src") == "brique":
                continue
            ov = _chevauche(g0, g1, o["m0"], o["m1"])
            if ov and (best is None or ov > best[0]):
                best = (ov, o)
        if best is None:
            etat = "ratée"
        else:
            o = best[1]
            if (o["m0"], o["m1"]) == (g0, g1):
                etat = "trouvée"
            elif o["m0"] == g0:
                etat = f"trouvée, {o['m1'] - o['m0'] + 1} mes. au lieu de {Lg}"
            else:
                etat = f"décalée de {o['m0'] - g0:+d} mes."
            if o.get("rot"):
                etat += f" (transposée {o['rot']})"
        # la cause, lue sur les chiffres de CETTE occurrence
        n_ok = sum(1 for x in pm if x >= TAU)
        # la même occurrence calée sur la MEILLEURE phase de la brique (elle
        # peut commencer sur la 2e boucle d'une brique qui en contient trois)
        ok_best = max(sum(1 for t in range(Lg) if g0 + t < n
                          and S[b0 + ((k + t) % L), g0 + t] >= TAU) for k in range(L))
        fin_ko = len(pm) >= 4 and all(x >= TAU for x in pm[:max(1, len(pm) - max(1, L // 4))]) \
            and any(x < TAU for x in pm[-max(1, L // 4):])
        if Lg != L and max(n_ok, ok_best) >= len(pm) - 1:
            cause = f"longueur différente ({Lg} mes. contre {L}), le contenu colle"
        elif fin_ko:
            cause = "fin alternative : le début colle, la fin change"
        elif sc is not None and sc >= TAU and thr is not None and sc < thr:
            cause = (f"seuil trop haut : {sc:.4f} < {thr:.4f}"
                     + (" (plafond 0,999)" if thr >= 0.998 else " (seuil du morceau)"))
        elif rs is not None and rs >= 0.9 and (sc or 0) < 0.85:
            cause = "transposée"
        elif n_ok < len(pm) * 0.75:
            cause = f"variante : {len(pm) - n_ok} mesure(s) sur {len(pm)} ne collent pas"
        else:
            cause = ""
        rows.append({"g0": g0, "g1": g1, "label": glab, "Lg": Lg, "par_mesure": pm,
                     "score": sc, "score_rot": rs, "etat": etat, "cause": cause})
    fausses = []
    for o in cands:
        if o.get("src") == "brique" or (o["m0"], o["m1"]) == (b0, let["b1"]):
            continue
        if any(_chevauche(g0, g1, o["m0"], o["m1"]) * 2 >= (o["m1"] - o["m0"] + 1)
               for g0, g1, _ in let["vraies"]):
            continue
        sous = sorted({x["label"] for x in secs if _chevauche(x["b0"], x["b1"], o["m0"], o["m1"])})
        if o.get("rot"):
            pourquoi = f"même musique transposée de {o['rot']} demi-tons — tu l'as nommée {'/'.join(sous)}"
        elif any(s.lower() == "intro" for s in sous):
            pourquoi = "l'intro joue les mêmes accords"
        else:
            pourquoi = f"jumeau harmonique : {'/'.join(sous)} a les mêmes accords (au moins sur ce morceau-là)"
        fausses.append({"m0": o["m0"], "m1": o["m1"], "sous": sous, "rot": o.get("rot", 0),
                        "pourquoi": pourquoi})
    return rows, fausses


def texte(res: dict) -> str:
    out = []
    n = res["n"]
    for let in res["lettres"]:
        info = let["info"]
        out.append(f"{res['stem'][:30]:30s} {let['label']:4s} brique {let['b0'] + 1}-{let['b1'] + 1} "
                   f"(L={let['L']}) seuil={info['seuil']} raison={info['raison']}")
        for nom, cands in (("prod", let["prod"]), ("plafond .97", let["plafond"]), ("candidat", let["cand"])):
            rows, fausses = classe(let, n, cands, res["secs"])
            out.append(f"   [{nom}]")
            for r in rows:
                out.append(f"     {r['g0'] + 1}-{r['g1'] + 1} ({r['Lg']}) {r['etat']:34s} {r['cause']}")
            for f in fausses:
                out.append(f"     EN TROP {f['m0'] + 1}-{f['m1'] + 1} {f['pourquoi']}")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", action="store_true")
    ap.add_argument("stems", nargs="*")
    a = ap.parse_args()
    stems = a.stems or [Path(p).stem for p in sorted(glob.glob(str(SETTINGS.sections_dir / "*.json")))]
    tous = []
    for s in stems:
        r = analyse_morceau(s)
        if r is None:
            continue
        tous.append(r)
        print(texte(r))
    if a.page:
        from tools.occurrences_brique_page import ecrire_page
        ecrire_page(tous, PAGE)
        print("page :", PAGE)


if __name__ == "__main__":
    main()
