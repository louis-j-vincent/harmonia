"""Occurrences d'une brique, v4 : la règle 5 de Louis, terme par terme.

Louis, 2026-09-30 nuit, après `docs/plots/the_walk_decalage.html` (The Walk :
les boucles de A posées 2 mesures trop tôt, parce que 7 boucles décalées
battaient 6 boucles en phase) :

    « si on hésite entre un départ et celui de ses deux voisins, on prend
    celui qui est plus grand que ses voisins d'avant et d'après (généralise
    au cas où on aurait des boucles plus grandes que 2) ».

Implémentée dans `harmonia.sections.simulation.maxima_locaux`. Cette page
compare v3 (règles 1–4 : le même code avec `REGLE5 = False`) et v4 (le code
live) sur les découpages validés de Louis, et montre pour chaque lettre qui
change les départs rivaux : chacun, son meilleur voisin à moins d'une boucle,
et qui gagne.

LECTURE SEULE. Ne modifie rien sous `state/`.

    .venv/bin/python -m tools.occurrences_brique_v4            # texte, corpus
    .venv/bin/python -m tools.occurrences_brique_v4 --page     # + la page
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

from harmonia.sections import simulation as SIM
from harmonia.settings import SETTINGS
from tools import occurrences_brique_v2 as V2
from tools.occurrences_brique_page import CSS, JS

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "docs" / "plots" / "occurrences_brique_v4_2026-09-30.html"
_esc = V2._esc

def _analyse_5b(stem: str):
    """La variante 5b, NON branchée, pour arbitrage : deux départs sont à
    égalité quand leur écart est sous la TOLÉRANCE DU MORCEAU de la règle 4
    (2 × max(1 − R, écart du groupe du haut, 0,01)) au lieu de 0,01. Même
    code live, seul le seuil d'égalité de `maxima_locaux` change."""
    ref0, max0 = SIM.reference_du_morceau, SIM.maxima_locaux
    box = {}

    def ref(pics):
        box["r"] = ref0(pics)
        return box["r"]

    def mx(c, Lb, cands):
        r, b = box.get("r"), SIM.BRUIT_PIC
        SIM.BRUIT_PIC = r["tolerance"] if r else b
        try:
            return max0(c, Lb, cands)
        finally:
            SIM.BRUIT_PIC = b
    SIM.reference_du_morceau, SIM.maxima_locaux = ref, mx
    try:
        return V2.analyse_morceau(stem)
    finally:
        SIM.reference_du_morceau, SIM.maxima_locaux = ref0, max0


def analyse(stem: str):
    """Le même morceau : règle 5 coupée (v3), live (v4), variante 5b."""
    try:
        SIM.REGLE5 = False
        r3 = V2.analyse_morceau(stem)
    finally:
        SIM.REGLE5 = True
    r4 = V2.analyse_morceau(stem)
    if r4 is None:
        return None
    r5 = _analyse_5b(stem)
    for l3, l4, l5 in zip(r3["lettres"], r4["lettres"], r5["lettres"]):
        l4["avant"], l4["n_avant"], l4["det3"] = l3["apres"], l3["n_apres"], l3["det"]
        l4["v5b"], l4["n_v5b"], l4["det5b"] = l5["apres"], l5["n_apres"], l5["det"]
    return r4


def departs(occ, b0, b1) -> str:
    return " ".join(f"{x + 1}" + (f"↑{r}" if r else "") for x, y, _s, r in occ if y < b0 or x > b1)


#: « ce que ça montre », écrit en lisant `python -m tools.occurrences_brique_v4`
#: (mesures en numérotation 1, comme dans l'app)
DIRES = {
    ("mayer_hawthorne_the_walk", "A"):
        "<b>Le cas pour lequel la règle a été faite — réglé.</b> v3 posait les boucles à 27, 31, …, 51 : "
        "7 boucles décalées de 2, dont les deux du bord (27 et 51, à 0,91) commencent sur le C♯m du B. "
        "v4 : 27 a un voisin nettement plus haut (29, 0,996), 51 aussi (49, 0,997) — ce ne sont plus des "
        "départs, et les boucles repassent en phase : <b>29, 33, 37, 41, 45, 49</b>. Même chose en fin de "
        "morceau : 83, 87, 91, 95 → 85, 89, 93. Justes 40 → 40, en trop 6 → 4. Plus de décalage.",
    ("bruno_mars_the_lazy_song_official_video", "A"):
        "<b>Perte : 8 mesures justes, à toi de trancher.</b> Tes A en 20–27 et 58–65 démarrent en 20/24 et "
        "58/62. Mais la boucle (deux accords qui alternent) colle presque aussi bien décalée de 2 mesures, "
        "et ici c'est le décalé qui est un peu plus haut : 22 = 0,977 contre 24 = 0,963 ; 60 = 0,991 contre "
        "58 = 0,963 et 62 = 0,967. Écarts de 0,014 à 0,028, au-dessus du 0,01 de bruit : le décalé gagne, "
        "24–27 disparaît et 58–65 devient 60–63. La variante 5b (égalité jusqu'à la tolérance du morceau, "
        "0,073) garde tout, comme v3.",
    ("maroon_5_she_will_be_loved_official_music_video", "A"):
        "<b>Perte : 4 mesures justes, à toi de trancher.</b> Ton A en 34–41 : le départ juste 34 vaut 0,977, "
        "son voisin décalé 36 vaut 0,994 (écart 0,017) — 34 est effacé, 34–37 n'est plus trouvé (38 reste). "
        "Même mécanisme que Lazy Song. 5b (tolérance 0,043) garde 34.",
    ("maroon_5_she_will_be_loved_official_music_video", "D"):
        "Gain sans effet sur tes mesures justes : dans l'intro (jumeau de D, en trop de toute façon), 13 "
        "(0,974) est effacé par 15 (0,993) ; 17 (0,991) reste à égalité avec 15, mais la couverture prend "
        "alors 15 puis 21 — 8 mesures en trop en moins. 5b garde v3 ici (écart sous sa tolérance, 0,042).",
}

TITRES = {
    "mayer_hawthorne_the_walk": "The Walk — Mayer Hawthorne",
    "bruno_mars_the_lazy_song_official_video": "The Lazy Song — Bruno Mars",
    "maroon_5_she_will_be_loved_official_music_video": "She Will Be Loved — Maroon 5",
}


def _rivaux(let) -> str:
    """Zoom sur les départs où v3 et v4 diffèrent : chaque départ qui passe la
    règle 3, son meilleur rival à moins d'une boucle, qui gagne."""
    det, d3, d5 = let["det"], let["det3"], let["det5b"]
    Lb = det["boucle"]
    pk = {q["p"]: q for q in det["pics"]}
    passe = sorted(p for p, q in pk.items() if q["ok"] or q.get("pas_maximum") or q.get("sous_reference"))
    pris3 = {q["p"] for q in d3["pics"] if q["pris"]}
    pris4 = {q["p"] for q in det["pics"] if q["pris"]}
    pris5 = {q["p"] for q in d5["pics"] if q["pris"]}
    diff = pris3 ^ pris4
    zone = sorted({p for p in passe if any(abs(p - d) < 2 * Lb for d in diff)})
    if not zone:
        return ""
    W, padL, H, y0, y1 = 1000, 60, 200, 22, 150
    lo = min(pk[p]["c"] for p in zone) - 0.01
    yv = lambda v: y0 + (y1 - y0) * (1 - v) / max(1 - lo, 1e-6)
    a, z = zone[0], zone[-1]
    xv = lambda p: padL + (W - padL - 20) * (p - a + 0.5) / max(z - a + 1, 1)
    o = [f'<svg viewBox="0 0 {W} {H}" class="plot" role="img">']
    for v in sorted({round(lo + (1 - lo) * f, 3) for f in (0, .5, 1)}):
        o.append(f'<line class="grid" x1="{padL}" x2="{W - 20}" y1="{yv(v):.1f}" y2="{yv(v):.1f}"/>'
                 f'<text x="{padL - 4}" y="{yv(v) + 3:.1f}" class="tick" text-anchor="end">{v:.3f}</text>')
    grid, stem = let["_grid"], let["_stem"]
    rows = []
    for p in zone:
        q = pk[p]
        vs = [r for r in passe if r != p and abs(r - p) < Lb]
        best = max(vs, key=lambda r: pk[r]["c"]) if vs else None
        m = set(range(p, p + Lb))
        juste = len(m & let["verite"]) * 2 >= Lb
        tolere = not juste and len(m & (let["tol_verdict"] | let["tol"])) * 2 >= Lb
        ou = "juste" if juste else ("accepté (ton verdict)" if tolere else "hors vérité")
        cls = "d-vrai" if juste else ("d-tol" if tolere else "d-faux")
        eff = not q.get("maximum_local", True)
        if eff and best is not None:
            o.append(f'<line class="arrow" x1="{xv(p):.1f}" y1="{yv(q["c"]):.1f}" '
                     f'x2="{xv(best):.1f}" y2="{yv(pk[best]["c"]):.1f}"/>')
        t0, t1 = grid[p], grid[min(len(grid) - 1, p + Lb)]
        o.append(f'<g class="play" data-stem="{_esc(stem)}" data-t0="{t0:.2f}" data-t1="{t1:.2f}">'
                 f'<circle class="{cls}{" d-cut" if eff else ""}" cx="{xv(p):.1f}" cy="{yv(q["c"]):.1f}" r="6"/>'
                 f'<title>départ mes. {p + 1} · {q["c"]:.3f} · {ou} · '
                 f'{"effacé par " + str(best + 1) if eff else "maximum local"} · clic = écouter</title></g>'
                 f'<text x="{xv(p):.1f}" y="{y1 + 16}" class="tick" text-anchor="middle">{p + 1}</text>')
        for yy, s_ in ((y1 + 30, pris3), (y1 + 42, pris4)):
            if p in s_:
                o.append(f'<rect class="{"r-ok" if juste else ("r-tol" if tolere else "r-ko")}" x="{xv(p) - 4:.1f}" y="{yy - 8}" '
                         f'width="8" height="8"/>')
        ecart = f'{pk[best]["c"] - q["c"]:+.3f}' if best is not None else "—"
        rows.append(f'<tr><td class="num">{p + 1}</td><td class="num">{q["c"]:.3f}</td>'
                    f'<td class="num">{best + 1 if best is not None else "—"}</td>'
                    f'<td class="num">{pk[best]["c"] if best is not None else 0:.3f}</td>'
                    f'<td class="num">{ecart}</td><td>{ou}</td>'
                    f'<td class="{"ko" if eff else "ok"}">{"effacé" if eff else "maximum local"}</td>'
                    f'<td>{"pris" if p in pris3 else ""}</td><td>{"pris" if p in pris4 else ""}</td>'
                    f'<td>{"pris" if p in pris5 else ""}</td></tr>')
    o.append(f'<text x="4" y="{y1 + 30}" class="lab">v3 pris</text>'
             f'<text x="4" y="{y1 + 42}" class="lab">v4 pris</text></svg>')
    table = ('<table><tr><th class="num">départ</th><th class="num">score</th><th class="num">meilleur rival '
             f'(±{Lb - 1})</th><th class="num">son score</th><th class="num">écart</th><th>dans ta vérité ?</th>'
             '<th>règle 5 (0,01)</th><th>v3</th><th>v4</th><th>5b</th></tr>' + "".join(rows) + '</table>')
    return ('<div class="sub">Zoom — les départs qui passent la règle 3 autour des changements. Point plein '
            '= maximum local, point pâle = effacé (la flèche va vers le rival qui l\'efface). Vert = départ '
            'juste dans ta vérité, bleu = accepté par ton verdict, rouge = non. Carrés = départs pris par v3 / v4.</div>'
            f'<div class="tablewrap">{"".join(o)}</div><div class="tablewrap">{table}</div>')


def _chiffres(let) -> str:
    f = lambda nom, n: (f'<tr><td>{nom}</td><td class="num">{n["justes"]}</td><td class="num">{n["ratees"]}</td>'
                        f'<td class="num">{n["trop"]}</td></tr>')
    return ('<table class="mini"><tr><th></th><th class="num">justes</th><th class="num">ratées</th>'
            '<th class="num">en trop</th></tr>' + f("v3 (règles 1–4)", let["n_avant"])
            + f("v4 (+ règle 5, live)", let["n_apres"]) + f("5b (variante, non branchée)", let["n_v5b"])
            + f'</table><div class="sub">en mesures, sur {len(let["verite"])} de ta vérité pour cette lettre</div>')


CSS4 = """
.d-vrai { fill:var(--ok); } .d-faux { fill:var(--ko); } .d-tol { fill:#4d8fe0; } .d-tol.d-cut { stroke:#4d8fe0; }
.d-cut { fill-opacity:.22; stroke-width:1.6; } .d-vrai.d-cut { stroke:var(--ok); } .d-faux.d-cut { stroke:var(--ko); }
.arrow { stroke:var(--mut); stroke-width:1.2; stroke-dasharray:3 2; }
tr.chg td { font-weight:600; } td.loss { color:var(--ko); font-weight:700; }
"""


def ecrire_page(tous: list[dict]) -> None:
    corps = []
    par = {r["stem"]: r for r in tous}
    for stem in TITRES:
        r = par.get(stem)
        if not r:
            continue
        corps.append(f'<h2>{_esc(TITRES[stem])}</h2>')
        for let in r["lettres"]:
            if (stem, let["label"]) not in DIRES:
                continue
            let["_stem"], let["_grid"] = stem, r["grid"]
            det = let["det"]
            corps.append(f'<h3>« {_esc(let["label"])} » = mesures {let["b0"] + 1}–{let["b1"] + 1} '
                         f'({let["L"]} mes.), boucle cherchée {det["boucle"]} mes., accords qui tournent '
                         f'toutes les {det["sous_periode"]} mes.</h3>')
            corps.append(f'<div class="say"><b>Ce que ça montre.</b> {DIRES[(stem, let["label"])]}</div>')
            corps.append(f'<div class="sub">départs trouvés hors brique — v3 : '
                         f'{departs(let["avant"], let["b0"], let["b1"])}<br>v4 : '
                         f'{departs(let["apres"], let["b0"], let["b1"])}<br>5b : '
                         f'{departs(let["v5b"], let["b0"], let["b1"])}</div>')
            corps.append(_rivaux(let))
            svg = V2._svg(r, let).replace('class="lab">avant</text>', 'class="lab">v3</text>') \
                                 .replace('class="lab">après</text>', 'class="lab">v4</text>')
            corps.append('<div class="tablewrap">' + svg + '</div>')
            corps.append('<div class="tablewrap">' + _chiffres(let) + '</div>')
    lignes, tot = [], {k: [0, 0, 0] for k in ("n_avant", "n_apres", "n_v5b")}
    for r in tous:
        for let in r["lettres"]:
            for k in tot:
                n = let[k]
                tot[k][0] += n["justes"]
                tot[k][1] += n["ratees"]
                tot[k][2] += n["trop"]
            a, p, b = let["n_avant"], let["n_apres"], let["n_v5b"]
            if a == p == b and let["avant"] == let["apres"]:
                continue
            perte = p["justes"] < a["justes"]
            lignes.append(
                f'<tr class="chg"><td>{_esc(r["stem"][:34])}</td><td>{_esc(let["label"])}</td>'
                f'<td class="num{" loss" if perte else ""}">{a["justes"]} → {p["justes"]}</td>'
                f'<td class="num">{a["trop"]} → {p["trop"]}</td>'
                f'<td class="num">{b["justes"]} / {b["trop"]}</td></tr>')
    t3, t4, t5 = tot["n_avant"], tot["n_apres"], tot["n_v5b"]
    n_let = sum(len(r["lettres"]) for r in tous)
    n_chg = len(lignes)
    page = f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Occurrences d'une brique v4</title>
<style>{CSS}{V2.CSS2}{CSS4}</style></head><body>
<h1>Ta règle 5 — un départ doit dépasser ses voisins</h1>
<div class="sub">Ta règle : « si on hésite entre un départ et celui de ses deux voisins, on prend celui qui est
plus grand que ses voisins d'avant et d'après (généralise au cas où on aurait des boucles plus grandes que 2) ».
Branchée dans « Valider les sections » (active au prochain redémarrage du serveur, pas fait). v3 = règles 1–4 ;
v4 = v3 + la règle 5.</div>
<div class="say"><b>La règle, telle qu'implémentée.</b> Parmi les départs qui passent la règle 3, un départ n'est
gardé que s'il est le plus haut de tous ceux qui sont à moins d'une boucle de lui (± 3 mesures pour une boucle
de 4, ± 7 pour 8) ; à 0,01 près, deux départs sont à égalité et restent tous les deux (sinon un plateau où tout
vaut 0,995–1,0 serait réduit à un départ au hasard). La zone que tu as tracée ne compte pas comme voisine.
Ensuite, comme avant, on prend le plus de boucles sans recouvrement — mais seulement parmi ces départs-là.</div>
<div class="say"><b>En bref.</b> {n_let} lettres sur tes découpages validés ; {n_chg} changent.
<b>The Walk A</b> : réglé — boucles en 29, 33, …, 49 au lieu de 27, 31, …, 51, plus de décalage.
<b>Pertes</b> : Lazy Song A (−8 mesures justes) et She Will Be Loved A (−4) — là, le départ juste est un peu
<i>plus bas</i> que son voisin décalé de 2 (0,963 contre 0,977–0,991) : deux accords qui alternent, la courbe ne
sait pas la phase et la règle tranche sur du bruit. Total justes {t3[0]} → {t4[0]}, en trop {t3[2]} → {t4[2]}.
<br><b>À toi de trancher</b> : la variante <b>5b</b> (non branchée) compte deux départs à égalité tant que leur
écart reste sous la tolérance du morceau que tu as validée pour la règle 4 (0,04–0,07 ici) au lieu de 0,01. Elle
règle The Walk pareil et ne perd rien : justes {t5[0]}, en trop {t5[2]}.</div>
<div class="legend">Courbe (sous chaque zoom) : comme en v3 ; lignes « v3 » et « v4 » = les occurrences trouvées.
<span style="background:var(--ok)"></span>juste <span style="background:#4d8fe0"></span>accepté par ton verdict
<span style="background:var(--ko)"></span>en trop. <b>Clic sur un point ou un bloc = écouter.</b></div>
<div id="player">clic sur un bloc pour écouter</div>
{"".join(corps)}
<h2>Tout le corpus : les lettres qui changent</h2>
<div class="tablewrap"><table><tr><th>morceau</th><th>lettre</th><th class="num">justes v3 → v4</th>
<th class="num">en trop v3 → v4</th><th class="num">5b : justes / en trop</th></tr>{"".join(lignes)}
<tr><td><b>total</b></td><td></td><td class="num">{t3[0]} → {t4[0]}</td><td class="num">{t3[2]} → {t4[2]}</td>
<td class="num">{t5[0]} / {t5[2]}</td></tr></table></div>
<p class="sub">En rouge : mesures justes perdues. Toutes les autres lettres : identiques en v3, v4 et 5b.
Généré par <code>python -m tools.occurrences_brique_v4 --page</code> ; v4 = le code live
(<code>harmonia.sections.simulation.maxima_locaux</code>), v3 = le même avec <code>REGLE5 = False</code>,
5b = le même avec l'égalité à la tolérance de la règle 4.</p>
<script>{JS}</script>
</body></html>"""
    PAGE.write_text(page, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", action="store_true")
    ap.add_argument("stems", nargs="*")
    a = ap.parse_args()
    stems = a.stems or [Path(p).stem for p in sorted(glob.glob(str(SETTINGS.sections_dir / "*.json")))]
    tous = []
    for s in stems:
        r = analyse(s)
        if r is None:
            continue
        tous.append(r)
        for let in r["lettres"]:
            x, y = let["n_avant"], let["n_apres"]
            chg = x != y or let["avant"] != let["apres"]
            print(f"{s[:30]:30s} {let['label']:6s} v3 j/r/t={x['justes']}/{x['ratees']}/{x['trop']}  "
                  f"v4={y['justes']}/{y['ratees']}/{y['trop']}{'  <<' if chg else ''}")
            if chg:
                print("     v3 :", departs(let["avant"], let["b0"], let["b1"]))
                print("     v4 :", departs(let["apres"], let["b0"], let["b1"]))
    if a.page:
        ecrire_page(tous)
        print("page :", PAGE)


if __name__ == "__main__":
    main()
