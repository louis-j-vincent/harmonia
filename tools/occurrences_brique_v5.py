"""Occurrences d'une brique, v5 : règle 5b + candidats en suspens, terme par terme.

Louis, 2026-09-30, sur `docs/plots/occurrences_brique_v4_2026-09-30.html` :

    « Oui pour la 5b très bien, mais alors il faut trancher par rapport à la
    cohérence globale du morceau, en cherchant d'autres sections et en gardant
    ces 2 candidats potentiels sans en acter un. Puis on voit ce qui reste
    quand les autres sections ont mangé ce qu'elles ont à manger. »

Implémenté dans `harmonia.sections.simulation` (`maxima_locaux(tol=)`,
`resoudre_suspens`, `occurrences_de_tous`). Contrairement aux pages v2–v4
(une lettre à la fois), cette page rejoue LE VRAI FLUX de « Valider les
sections » : pour chaque découpage validé, Louis est supposé avoir tracé la
PREMIÈRE occurrence de chaque lettre (intro, outro compris), et
`occurrences_de_tous` tourne sur le chart réel. Trois versions :
  * v4  : `EGALITE_5B = False`, `SUSPENS = False` (égalité à 0,01, choix au score) ;
  * 5b  : `EGALITE_5B = True`,  `SUSPENS = False` (la variante de la page v4) ;
  * v5  : le code live (5b + suspens tranchés par la cohérence globale).

LECTURE SEULE. Ne modifie rien sous `state/`.

    .venv/bin/python -m tools.occurrences_brique_v5            # texte, corpus
    .venv/bin/python -m tools.occurrences_brique_v5 --page     # + la page
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

from harmonia.sections import simulation as SIM
from harmonia.settings import SETTINGS
from tools import occurrences_brique as OB
from tools import occurrences_brique_v2 as V2
from tools.occurrences_brique_page import CSS, JS

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "docs" / "plots" / "occurrences_brique_v5_2026-09-30.html"
_esc = V2._esc
VERSIONS = {"v4": (False, False), "5b": (True, False), "v5": (True, True)}


def _flux(chart, traits, egal, susp):
    e0, s0 = SIM.EGALITE_5B, SIM.SUSPENS
    SIM.EGALITE_5B, SIM.SUSPENS = egal, susp
    try:
        occ, _e, diags = SIM.occurrences_de_tous(chart, traits, SETTINGS.audio_dir)
    finally:
        SIM.EGALITE_5B, SIM.SUSPENS = e0, s0
    return occ, diags


def analyse(stem: str) -> dict | None:
    d = json.loads((SETTINGS.sections_dir / f"{stem}.json").read_text(encoding="utf-8"))
    if not d.get("validated"):
        return None
    chart = OB._chart_for(stem, d["n"])
    if chart is None:
        return None
    secs = sorted(d["sections"], key=lambda s: s["b0"])
    n = len(chart["barGrid"]) - 1
    traits, vues = [], set()
    for s in secs:                                  # la 1re occurrence de chaque lettre
        lab = OB._fam(s["label"])
        if lab in vues:
            continue
        vues.add(lab)
        traits.append({"label": lab, "mesure_debut": s["b0"] + 1, "mesure_fin": s["b1"] + 1})
    res = {"stem": stem, "n": n, "secs": secs, "grid": chart["barGrid"], "traits": traits}
    for nom, (e, s) in VERSIONS.items():
        occ, diags = _flux(chart, traits, e, s)
        res[nom] = occ
        if nom == "v5":
            res["groupes"] = [g for dg in diags for g in dg.get("suspens_global") or []]
    fin_annot = max(s["b1"] for s in secs)
    lettres = []
    for t in traits:
        lab = t["label"]
        if lab.lower() in OB.SANS_REPRISE:
            continue
        fam = V2.MEME_FAMILLE.get(stem, {}).get(lab, {lab})
        verite = V2._mesures([(x["b0"], x["b1"]) for x in secs if OB._fam(x["label"]) == lab])
        tol = set(range(fin_annot + 1, n))
        for x in secs:
            f = OB._fam(x["label"])
            if f != lab and (f in fam or f in V2.TOLERE.get(stem, {}).get(lab, set())):
                tol |= set(range(x["b0"], x["b1"] + 1))
        row = {"label": lab, "verite": verite, "tol": tol}
        for nom in VERSIONS:
            m = V2._mesures([(o["m0"], o["m1"]) for o in res[nom] if o["label"] == lab])
            row[nom] = {"justes": len(m & verite), "ratees": len(verite - m),
                        "trop": len(m - verite - tol)}
        lettres.append(row)
    res["lettres"] = lettres
    return res


def _mangeur(res, alt):
    """Qui, dans le résultat v5, occupe les mesures d'une alternative perdante."""
    qui = set()
    for a, z, *_r in alt:
        for o in res["v5"]:
            if not (z < o["m0"] or o["m1"] < a):
                qui.add(f'{o["label"]} {o["m0"] + 1}–{o["m1"] + 1}')
    return ", ".join(sorted(qui)) or "—"


def texte(res) -> str:
    out = []
    for g in res["groupes"]:
        alts = ["/".join(str(o[0] + 1) for o in alt) for alt in g["alternatives"]]
        t = g["termes"]
        out.append(f'   suspens {g["label"]}: ' + " | ".join(
            f'{a} (mang {x["mangees"]} cont {x["contacts"]} frag {x["fragments"]} sc {x["score"]})'
            for a, x in zip(alts, t))
            + f' → {alts[g["choix"]]} [{g["raison"]}] (seul : {alts[g["choix_seul"]]} [{g["raison_seul"]}])')
    for r in res["lettres"]:
        a, b, c = r["v4"], r["5b"], r["v5"]
        if a != b or b != c:
            out.append(f'   {r["label"]:6s} v4 {a["justes"]}/{a["ratees"]}/{a["trop"]}  '
                       f'5b {b["justes"]}/{b["ratees"]}/{b["trop"]}  v5 {c["justes"]}/{c["ratees"]}/{c["trop"]}')
    return "\n".join(out)


# ── la page ──────────────────────────────────────────────────────────────────

#: « ce que ça montre », écrit en lisant `python -m tools.occurrences_brique_v5`
DIRES: dict[str, str] = {
    "mayer_hawthorne_the_walk":
        "<b>Le décalage reste réglé.</b> En 5b, 27 et 51 (0,91) restent effacés par 29 et 49 (0,996) : l'écart "
        "(0,086) dépasse la tolérance du morceau (0,049). Les boucles de A restent en phase, 29, 33, …, 49, "
        "comme en v4. Un seul suspens, à la fin : six calages rivaux entre 85 et 95 (dans ton B et ton outro). "
        "Ton trait outro mange 93 et 95 ; il reste 85 et 89 dans tous les cas. Rien à choisir : « mangée ».",
    "bruno_mars_the_lazy_song_official_music_video":
        "<b>La perte de v4 est rattrapée, sans suspens.</b> En 5b, 58 (0,963) et 62 (0,967) ne sont plus "
        "effacés par 60 (0,991) : l'écart est sous la tolérance (0,073). Ensemble, ils casent 2 boucles, 60 seul "
        "une : la couverture choisit 58 + 62, c'est structurel, pas du bruit. A : 40 → 48 mesures justes.",
    "maroon_5_she_will_be_loved_official_music_video":
        "<b>Mieux, mais les jumeaux se déplacent.</b> 34 revient (5b : 0,977 contre 0,994, sous la tolérance "
        "0,043). Dix suspens (A et D, calés à 2 mesures près dans tes B et C : 21/23, 50/52, 80/82, 88/90, "
        "96/98), tous mangés par tes autres sections. A ne prend donc plus les zones de C d'abord : ses 16 "
        "mesures en trop disparaissent. Mais ces zones libérées sont prises par des jumeaux de B (24 en trop "
        "dans tes C, 88–103), et C perd 8 mesures justes. Sur A + B + C : justes 56 → 64, en trop 32 → 24.",
    "bobby_hebb_sunny_official_audio":
        "<b>Le coût de la règle, à toi de trancher.</b> Ton A en 18–25 était en suspens : 18 (0,950) contre 22 "
        "(0,874), à égalité parce que la tolérance du morceau est large (0,10). Pendant qu'il attend, la boucle "
        "de C (4 mesures, mêmes accords, 0,984) prend 18–21 et 22–25 : A perd 8 mesures justes, C en a 8 en "
        "trop. A et C jouent les mêmes accords (v4 posait déjà des A sur tes C 50–57 et 66–73) ; avant, A "
        "gardait 18–25 seulement parce que tu avais tracé A avant C.",
    "X-yIEMduRXk":
        "Deux suspens de C (1/5 et 23/27), tous deux dans tes A. Ton trait A et ses reprises les mangent : "
        "rien ne change.",
}
TITRES = {
    "mayer_hawthorne_the_walk": "The Walk — Mayer Hawthorne",
    "bruno_mars_the_lazy_song_official_music_video": "The Lazy Song — Bruno Mars",
    "maroon_5_she_will_be_loved_official_music_video": "She Will Be Loved — Maroon 5",
    "bobby_hebb_sunny_official_audio": "Sunny — Bobby Hebb",
}
PALETTE = ["#e07b39", "#3b82c4", "#4caf6e", "#b05bc9", "#d4b106", "#c94f5b", "#2aa5a0",
           "#7a6ff0", "#8d6e63", "#5f7d8c"]
RAISONS = {"tout_mange": "tous les calages sont mangés par d'autres sections : rien n'est posé",
           "mangee": "l'autre calage est mangé par une autre section",
           "phase": "collé aux autres occurrences de la brique (même phase)",
           "fragments": "ne laisse pas de bout orphelin (&lt; 4 mesures)",
           "score": "rien d'autre ne tranche : le score"}


def _couleurs(res):
    labs = []
    for s in res["secs"]:
        f = OB._fam(s["label"])
        if f not in labs:
            labs.append(f)
    return {f: PALETTE[i % len(PALETTE)] for i, f in enumerate(labs)}


def _bandes(res) -> str:
    n, grid, stem = res["n"], res["grid"], res["stem"]
    col = _couleurs(res)
    W, padL, padR = 1000, 70, 8
    bw = (W - padL - padR) / n
    x = lambda b: padL + b * bw                                         # noqa: E731
    rows = [("toi", [(s["b0"], s["b1"], OB._fam(s["label"]), "") for s in res["secs"]])]
    for nom in ("v4", "v5"):
        rows.append((nom, [(o["m0"], o["m1"], o["label"],
                            "suspens" if o.get("suspens") else ("trace" if o.get("trace") else ""))
                           for o in res[nom]]))
    H = 30 + 26 * len(rows) + 40
    o = [f'<svg viewBox="0 0 {W} {H}" class="plot" role="img">']
    for k, (nom, blocs) in enumerate(rows):
        yy = 10 + 26 * k
        o.append(f'<text x="4" y="{yy + 13}" class="lab">{nom}</text>')
        for m0, m1, lab, sorte in blocs:
            t0, t1 = grid[m0], grid[min(len(grid) - 1, m1 + 1)]
            c = col.get(lab, "#999")
            dash = ' stroke-dasharray="3 2"' if sorte == "suspens" else ""
            fill = c if nom == "toi" or sorte == "trace" else c + "88"
            o.append(f'<g class="play" data-stem="{_esc(stem)}" data-t0="{t0:.2f}" data-t1="{t1:.2f}">'
                     f'<rect x="{x(m0):.1f}" y="{yy}" width="{bw * (m1 - m0 + 1) - 1:.1f}" height="18" rx="2" '
                     f'fill="{fill}" stroke="{c}" stroke-width="{2 if sorte == "suspens" else 1}"{dash}/>'
                     f'<text x="{x(m0) + 2:.1f}" y="{yy + 13}" class="seclab">{_esc(lab)}</text>'
                     f'<title>{nom} · {_esc(lab)} · mes. {m0 + 1}–{m1 + 1}'
                     f'{" · en suspens, tranché" if sorte == "suspens" else ""} · clic = écouter</title></g>')
    # les groupes en suspens : chaque alternative, en cadre sous la ligne v5
    yy = 10 + 26 * len(rows)
    o.append(f'<text x="4" y="{yy + 11}" class="lab">suspens</text>')
    for g in res["groupes"]:
        for k, alt in enumerate(g["alternatives"]):
            gagne = k == g["choix"] and g["raison"] != "tout_mange"
            for a, z, *_r in alt:
                o.append(f'<rect x="{x(a):.1f}" y="{yy + 2 + 8 * (k % 2)}" width="{bw * (z - a + 1) - 1:.1f}" '
                         f'height="6" fill="{col.get(g["label"], "#999") if gagne else "none"}" '
                         f'stroke="{col.get(g["label"], "#999")}"><title>{_esc(g["label"])} en suspens · '
                         f'mes. {a + 1}–{z + 1} · {"retenu" if gagne else "écarté"}</title></rect>')
    for b in range(0, n, 8):
        o.append(f'<text x="{x(b):.1f}" y="{H - 4}" class="tick">{b + 1}</text>')
    o.append("</svg>")
    return "".join(o)


def _table_groupes(res) -> str:
    if not res["groupes"]:
        return '<div class="sub">Aucun calage en suspens dans ce morceau.</div>'
    rows = []
    for g in res["groupes"]:
        for k, (alt, t) in enumerate(zip(g["alternatives"], g["termes"])):
            dep = ", ".join(str(o[0] + 1) for o in alt)
            sc = " ".join(f"{o[2]:.3f}" for o in alt)
            gagne = k == g["choix"] and g["raison"] != "tout_mange"
            rows.append(
                f'<tr class="{"chg" if gagne else ""}"><td>{_esc(g["label"])}</td><td class="num">{dep}</td>'
                f'<td class="num">{sc}</td><td class="num">{t["mangees"]}</td>'
                f'<td>{_esc(_mangeur(res, alt)) if t["mangees"] else ""}</td>'
                f'<td class="num">{t["contacts"]}</td><td class="num">{t["fragments"]}</td>'
                f'<td class="{"ok" if gagne else "ko"}">{"retenu" if gagne else ("mangé" if not t["vivantes"] else "écarté")}</td>'
                f'<td>{RAISONS[g["raison"]] if gagne or (k == 0 and g["raison"] == "tout_mange") else ""}</td>'
                f'<td>{"oui" if k == g["choix_seul"] else ""}</td></tr>')
    return ('<table><tr><th>lettre</th><th class="num">départ(s)</th><th class="num">score(s)</th>'
            '<th class="num">boucles mangées</th><th>par</th><th class="num">contacts (phase)</th>'
            '<th class="num">bouts orphelins</th><th>v5</th><th>pourquoi</th><th>choix brique seule</th></tr>'
            + "".join(rows) + '</table>')


def _chiffres(res) -> str:
    rows = []
    for r in res["lettres"]:
        a, b, c = r["v4"], r["5b"], r["v5"]
        f = lambda n: f'{n["justes"]} / {n["ratees"]} / {n["trop"]}'   # noqa: E731
        chg = "chg" if (a != b or b != c) else ""
        rows.append(f'<tr class="{chg}"><td>{_esc(r["label"])}</td><td class="num">{len(r["verite"])}</td>'
                    f'<td class="num">{f(a)}</td><td class="num">{f(b)}</td><td class="num">{f(c)}</td></tr>')
    return ('<table class="mini"><tr><th>lettre</th><th class="num">ta vérité</th>'
            '<th class="num">v4 j / r / t</th><th class="num">5b seule</th><th class="num">v5</th></tr>'
            + "".join(rows) + '</table><div class="sub">j = justes, r = ratées, t = en trop, en mesures.</div>')


def ecrire_page(tous: list[dict]) -> None:
    corps = []
    par = {r["stem"]: r for r in tous}
    detail = [s for s in TITRES if s in par] + sorted(
        s for s, r in par.items() if s not in TITRES and (r["groupes"] or any(
            x["v4"] != x["v5"] for x in r["lettres"])))
    for stem in detail:
        r = par[stem]
        corps.append(f'<h2>{_esc(TITRES.get(stem, stem))}</h2>')
        if stem in DIRES:
            corps.append(f'<div class="say"><b>Ce que ça montre.</b> {DIRES[stem]}</div>')
        tr = " · ".join(f'{t["label"]} {t["mesure_debut"]}–{t["mesure_fin"]}' for t in r["traits"])
        corps.append(f'<div class="sub">Tes traits supposés (1re occurrence de chaque lettre) : {_esc(tr)}</div>')
        corps.append('<div class="tablewrap">' + _bandes(r) + '</div>')
        corps.append('<div class="tablewrap">' + _table_groupes(r) + '</div>')
        corps.append('<div class="tablewrap">' + _chiffres(r) + '</div>')
    tot = {k: [0, 0, 0] for k in VERSIONS}
    lignes = []
    for r in tous:
        for x in r["lettres"]:
            for k in VERSIONS:
                tot[k][0] += x[k]["justes"]
                tot[k][1] += x[k]["ratees"]
                tot[k][2] += x[k]["trop"]
            if x["v4"] != x["v5"] or x["5b"] != x["v5"]:
                f = lambda n: f'{n["justes"]} / {n["ratees"]} / {n["trop"]}'   # noqa: E731
                lignes.append(f'<tr><td>{_esc(r["stem"][:34])}</td><td>{_esc(x["label"])}</td>'
                              f'<td class="num">{f(x["v4"])}</td><td class="num">{f(x["5b"])}</td>'
                              f'<td class="num">{f(x["v5"])}</td></tr>')
    n_g = sum(len(r["groupes"]) for r in tous)
    par_raison = {}
    for r in tous:
        for g in r["groupes"]:
            par_raison[g["raison"]] = par_raison.get(g["raison"], 0) + 1
    noms = {"tout_mange": "tout mangé", "mangee": "mangé", "phase": "phase", "fragments": "bouts orphelins",
            "score": "score"}
    rais = ", ".join(f"{v} par « {noms[k]} »" for k, v in sorted(par_raison.items(), key=lambda kv: -kv[1]))
    f = lambda t: f"{t[0]} / {t[1]} / {t[2]}"   # noqa: E731
    page = f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Occurrences d'une brique v5</title>
<style>{CSS}{V2.CSS2}
tr.chg td {{ font-weight:600; }}
</style></head><body>
<h1>5b + candidats en suspens, tranchés par tout le morceau</h1>
<div class="sub">Ta décision : « Oui pour la 5b très bien, mais alors il faut trancher par rapport à la
cohérence globale du morceau, en cherchant d'autres sections et en gardant ces 2 candidats potentiels sans en
acter un. Puis on voit ce qui reste quand les autres sections ont mangé ce qu'elles ont à manger. »
Branché dans « Valider les sections » (actif au prochain redémarrage du serveur, pas fait).</div>
<div class="say"><b>Ce qui est fait.</b>
(1) <b>5b</b> : deux départs voisins sont à égalité tant que leur écart reste sous la tolérance du morceau de
ta règle 4. (2) Si deux calages à égalité casent <b>autant de boucles</b>, aucun n'est posé : ils restent
<b>en suspens</b>. (3) Quand tu cliques « Valider », toutes tes briques et leurs reprises sûres sont posées
d'abord ; ensuite, pour chaque suspens : un calage dont les mesures sont déjà prises par une autre section perd ;
puis celui qui colle aux autres occurrences de la brique (même phase) ; puis celui qui ne laisse pas de bout
orphelin de moins de 4 mesures ; le score seulement en dernier. Le calage retenu sort en pointillés
(« par ressemblance ») : c'est une proposition, retranchée à chaque « Valider », donc quand tu ajoutes une
brique.</div>
<div class="say"><b>Comment on simule.</b> Contrairement aux pages v2–v4 (une lettre à la fois), on rejoue le
vrai « Valider » : sur chacun de tes découpages validés, tu es supposé avoir tracé la première occurrence de
chaque lettre (intro et outro compris), toutes d'un coup. v4 = l'égalité à 0,01 et le choix au score ;
« 5b seule » = la variante de la page v4 ; v5 = le code live.
<br><b>En bref.</b> {n_g} calages en suspens sur le corpus ({rais}). Total justes / ratées / en trop :
v4 {f(tot["v4"])} · 5b seule {f(tot["5b"])} · <b>v5 {f(tot["v5"])}</b>.</div>
<div class="legend">Bandes : « toi » = tes sections ; « v4 », « v5 » = ce que rend Valider (plein = ton trait,
pâle = reprise trouvée, <b>pointillé = calage en suspens tranché</b>) ; « suspens » = les calages rivaux (plein
= retenu, vide = écarté). Même couleur = même lettre. <b>Clic sur un bloc = écouter.</b></div>
<div id="player">clic sur un bloc pour écouter</div>
{"".join(corps)}
<h2>Tout le corpus : les lettres qui changent</h2>
<div class="tablewrap"><table><tr><th>morceau</th><th>lettre</th><th class="num">v4 j / r / t</th>
<th class="num">5b seule</th><th class="num">v5</th></tr>{"".join(lignes)}
<tr><td><b>total</b></td><td></td><td class="num">{f(tot["v4"])}</td><td class="num">{f(tot["5b"])}</td>
<td class="num">{f(tot["v5"])}</td></tr></table></div>
<p class="sub">Généré par <code>python -m tools.occurrences_brique_v5 --page</code> ; v5 =
<code>harmonia.sections.simulation.occurrences_de_tous</code> (live), v4 / 5b = le même avec
<code>EGALITE_5B</code> / <code>SUSPENS</code> coupés.</p>
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
        t = texte(r)
        if t:
            print(s[:40])
            print(t)
    tot = {k: [sum(x[k][m] for r in tous for x in r["lettres"]) for m in ("justes", "ratees", "trop")]
           for k in VERSIONS}
    print("total j/r/t", tot)
    if a.page:
        ecrire_page(tous)
        print("page :", PAGE)


if __name__ == "__main__":
    main()
