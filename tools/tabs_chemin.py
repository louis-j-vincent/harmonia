"""Tout le chemin d'un accord de tab jusqu'à sa mesure, terme par terme.

Louis, 2026-09-18, après avoir vu le POC : « tu pourrais me montrer exactement
comment tu fais matcher le prior et la logproba, j'aimerais pouvoir comprendre
tout le chemin ».

La page déroule les six étapes sur UN morceau, avec les vrais nombres et des
mesures qu'on écoute au doigt. Elle ne recalcule rien à côté du code : elle lit
la table de programmation dynamique que `tab_align.aligner` remplit, par
`tab_align.table` et `tab_align.detail_mesure`. C'est volontaire — une page
d'explication qui refait le calcul dans son coin finit par expliquer autre
chose que ce qui tourne, et ce projet l'a déjà payé une fois.

Elle montre aussi les FRONTIÈRES de section, parce que c'est ce que Louis a
trouvé « quasi bon mais pas bon », et que le désaccord est arbitrable à
l'oreille : le tab ouvre la section sur la tonique, notre chart l'ouvre sur la
dominante qui précède.

    python -m tools.tabs_chemin
    python -m tools.tabs_chemin --chart min_xxx --requete "Artiste Titre"
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

from harmonia.integrations.tab_align import (NOMS, aligner, compresser,
                                             detail_mesure,
                                             meilleure_transposition, nom_q5,
                                             part_des_temoins,
                                             prior_de_grille, priors_musx,
                                             sequence_du_tab, table)
from harmonia.settings import SETTINGS

#: Grenade : 95 % d'accord avec musx, et les cinq frontières « en retard » y
#: tombent toutes sur le même accord — le morceau le plus lisible du POC.
DEFAUT = ("min_bruno_mars_grenade_official_music_video", "Bruno Mars Grenade")


def rassembler(cle: str, requete: str) -> dict | None:
    """Refait l'alignement du POC et garde de quoi l'expliquer."""
    from harmonia.integrations.tab_fetcher import fetch_tab_chords, search_tabs

    p = SETTINGS.charts_dir / f"{cle}.json"
    if not p.exists():
        print(f"   {cle} : pas de chart")
        return None
    chart = json.loads(p.read_text(encoding="utf-8"))
    stem = Path(chart.get("audio_url") or "").stem
    audio = SETTINGS.audio_dir / f"{stem}.m4a"
    grille = [float(t) for t in (chart.get("barGrid") or [])]
    if not audio.exists() or len(grille) < 3:
        print(f"   {cle} : pas d'audio ou grille trop courte")
        return None

    res = [r for r in search_tabs(requete, tab_types=("Chords",), max_results=6)
           if r.rating >= 4.5]
    if not res:
        print(f"   {cle} : aucun tab au-dessus de 4,5★")
        return None
    tab = fetch_tab_chords(res[0])
    seq = compresser(sequence_du_tab(tab.raw_content))
    P = priors_musx(audio, grille)
    dec, scores = meilleure_transposition(seq, P)
    chemin = aligner(seq, P, dec)
    t = table(seq, P, dec)

    mesures = []
    for b, i in enumerate(chemin):
        c = seq[i]
        top = int(np.argmax(P[b]))
        mesures.append({
            "b": b, "case": i,
            "accord": nom_q5((c["root"] + dec) % 12, c["q5"]),
            "musx": nom_q5(top // 5, top % 5), "p_musx": float(P[b][top]),
            "section": c.get("section") or "",
            "t0": grille[b], "t1": grille[b + 1]})

    # les frontières que le tab pose, et celles que notre chart pose
    f_tab = [b for b in range(1, len(mesures))
             if mesures[b]["section"] != mesures[b - 1]["section"]]
    f_chart = sorted({a for s in (chart.get("sections") or [])
                      for a, _ in (s.get("barRanges") or [])})
    f_chart = [b for b in f_chart if 0 < b < len(mesures)]

    return {"cle": cle, "titre": chart.get("title") or cle,
            "audio": chart.get("audio_url") or f"/audio/{stem}.m4a",
            "tab": res[0], "seq": seq, "P": P, "dec": dec, "chemin": chemin,
            "table": t, "grille": grille, "mesures": mesures,
            "prior": prior_de_grille(seq),
            "f_tab": f_tab, "f_chart": f_chart,
            "n_mesures": len(mesures)}


# ── rendu ───────────────────────────────────────────────────────────────────

def n(x: float, d: int = 3) -> str:
    """Un nombre lisible, virgule décimale, signe explicite pour les logs."""
    s = f"{x:+.{d}f}" if d and abs(x) < 100 else f"{x:.{d}f}"
    return s.replace(".", ",")


def bouton(d: dict, m: dict, texte: str) -> str:
    return (f"<button class=ec onclick=\"jouer('{html.escape(d['audio'])}',"
            f"{m['t0']:.2f},{m['t1']:.2f})\">{texte}</button>")


def etape(numero: int, titre: str, corps: str, quoi: str = "") -> str:
    q = f"<p class=quoi>{quoi}</p>" if quoi else ""
    return (f"<section><h2><span class=et>{numero}</span>{titre}</h2>"
            f"{q}{corps}</section>")


def bloc_cases(d: dict) -> str:
    """Étape 1 — le tab devient une suite de cases."""
    L = ["<table class=t><tr><th>case<th>accord<th>lignes du tab<th>section"]
    for i, c in enumerate(d["seq"][:10]):
        L.append(f"<tr><td class=g>{i}<td><b>{nom_q5(c['root'], c['q5'])}</b>"
                 f"<td class=g>{c.get('repetitions', 1)}"
                 f"<td class=g>{html.escape(c.get('section') or '—')}")
    L.append("</table>")
    tot = sum(max(1, c.get("repetitions", 1)) for c in d["seq"])
    return ("".join(L) +
            f"<p class=note>{len(d['seq'])} cases en tout, pour {tot} lignes "
            f"d'accord écrites dans le tab et {d['n_mesures']} mesures de "
            f"musique. Un accord écrit au-dessus de trois lignes de paroles "
            f"est un accord tenu, pas trois changements : on le replie en une "
            f"case qui pèse 3.</p>")


def bloc_cadence(d: dict) -> str:
    """Étape 2 — la grille dit à quelle vitesse ça bouge."""
    t, r = d["table"], d["table"]["r"]
    L = ["<table class=t><tr><th>la mesure consomme<th>prix a priori"]
    for k, c in enumerate(t["cout"]):
        mark = " class=on" if k == round(r) else ""
        L.append(f"<tr{mark}><td>{k} case{'s' if k > 1 else ''}"
                 f"<td class=num>{n(float(c))}")
    L.append("</table>")
    return ("".join(L) +
            f"<p class=note>Le tab a {len(d['seq'])} cases pour "
            f"{d['n_mesures']} mesures : une mesure en consomme "
            f"<b>{n(r, 2)}</b> en moyenne. C'est <b>r</b>. Le prix ci-dessus "
            f"est <b>log Poisson(k ; r)</b> — la loi de « combien de cases "
            f"cette mesure avale ». Elle pique en k ≈ r et punit autant le "
            f"surplace que la course.</p>")


def bloc_musx(d: dict, b: int) -> str:
    """Étape 3 — ce que musx entend sur une mesure."""
    P, m = d["P"], d["mesures"][b]
    ordre = np.argsort(-P[b])[:5]
    L = ["<table class=t><tr><th>accord<th>ce que musx lui donne"]
    for c in ordre:
        c = int(c)
        L.append(f"<tr><td><b>{nom_q5(c // 5, c % 5)}</b>"
                 f"<td class=num>{n(float(P[b][c]), 3)[1:]}"
                 f"<td class=barre><i style=\"width:{P[b][c]*100:.0f}%\"></i>")
    L.append("</table>")
    return ("".join(L) +
            f"<p class=note>Mesure {b+1}. musx ne rend pas un accord, il rend "
            f"une probabilité sur les 60 accords possibles (12 fondamentales × "
            f"5 familles). {bouton(d, m, 'écouter cette mesure')}</p>")


def bloc_distance(d: dict, b: int, case: int) -> str:
    """Étape 4 — la distance musicale transforme ce qu'il entend en vote."""
    c = d["seq"][case]
    cible = ((c["root"] + d["dec"]) % 12, c["q5"])
    parts = part_des_temoins(d["P"][b], cible)
    L = ["<table class=t><tr><th>musx envisage<th>il y croit<th>× proximité"
         "<th>= il verse"]
    for idx, p, pr, v in parts:
        L.append(f"<tr><td><b>{nom_q5(idx // 5, idx % 5)}</b>"
                 f"<td class=num>{n(p, 3)[1:]}<td class=num>{n(pr, 2)[1:]}"
                 f"<td class=num>{n(v, 3)[1:]}")
    somme = sum(v for *_, v in parts)
    L.append(f"<tr class=on><td colspan=3>somme<td class=num>{n(somme, 3)[1:]}")
    L.append("</table>")
    lv = d["table"]["E"][b, case]
    return ("".join(L) +
            f"<p class=note>La case dit <b>{nom_q5(*cible)}</b>. On ne demande "
            f"pas à musx s'il a dit exactement ça — on lui demande combien il "
            f"verse <i>pour</i> ça. Chaque accord qu'il envisage verse sa "
            f"probabilité, escomptée par sa <b>proximité musicale</b> à la "
            f"case : le recouvrement des notes. Un Fm qui partage deux notes "
            f"sur trois avec un Dm verse les deux tiers ; un C# ne verse "
            f"rien.<br>La somme fait <b>{n(somme, 3)[1:]}</b>, et la "
            f"log-vraisemblance de la case est son logarithme : "
            f"<b>{n(float(lv))}</b>.</p>")


def bloc_decision(d: dict, b: int, pourquoi: str) -> str:
    """Étape 5 — les deux termes s'additionnent, le plus grand gagne."""
    det = detail_mesure(d["seq"], d["P"], b, d["dec"])
    m = d["mesures"][b]
    L = ["<table class=t><tr><th>si on avance de<th>on tombe sur"
         "<th>l'audio en dit<th>la grille en dit<th>total"]
    for c in det["candidats"]:
        mark = " class=on" if c["retenue"] else ""
        L.append(f"<tr{mark}><td class=g>{c['saut']} case"
                 f"{'s' if c['saut'] > 1 else ''}<td><b>{c['accord']}</b>"
                 f"<td class=num>{n(c['lv'])}<td class=num>{n(c['prior'])}"
                 f"<td class=num><b>{n(c['total'])}</b>")
    L.append("</table>")
    # la ligne retenue n'est pas toujours celle qui a le plus gros total : la
    # PD choisit le meilleur CHEMIN, pas le meilleur coup. Le dire, sinon ça
    # se lit comme une erreur de calcul.
    loc = max(det["candidats"], key=lambda c: c["total"])
    ret = next((c for c in det["candidats"] if c["retenue"]), loc)
    ecart = ""
    if loc["case"] != ret["case"]:
        ecart = (f"<br><b>Ici le plus gros total n'est pas celui qu'on garde</b> :"
                 f" {loc['accord']} marque {n(loc['total'])}, {ret['accord']} "
                 f"marque {n(ret['total'])}. On garde quand même "
                 f"{ret['accord']}, parce que ce qui vient après s'enchaîne "
                 f"mieux : la table cherche le meilleur <i>chemin</i> d'un bout "
                 f"à l'autre, pas le meilleur coup mesure par mesure. Sur ce "
                 f"morceau ça arrive sur un quart des mesures, et j'ai mesuré "
                 f"l'alternative — écrire le premier accord avalé plutôt que le "
                 f"dernier fait tomber l'accord avec musx de 95 % à 94 % ici, "
                 f"et de 79 % à 74 % sur This Love.")
    return ("".join(L) +
            f"<p class=note>Mesure {b+1}, en venant de la case "
            f"{det['case_precedente']}. {pourquoi} "
            f"{bouton(d, m, 'écouter')}{ecart}</p>")


def bloc_frontieres(d: dict) -> str:
    """Étape 6 — là où le tab et notre chart ne sont pas d'accord."""
    L = []
    paires = []
    for b in d["f_tab"]:
        if not d["f_chart"]:
            continue
        proche = min(d["f_chart"], key=lambda x: abs(x - b))
        if 0 < abs(b - proche) <= 3:
            paires.append((b, proche))
    if not paires:
        return "<p class=note>Le tab et le chart posent les mêmes frontières.</p>"
    for b, proche in paires[:6]:
        nom = html.escape(d["mesures"][b]["section"])
        L.append(f"<div class=front><div class=fh>« {nom} » — "
                 f"le tab l'ouvre mesure <b>{b+1}</b>, notre chart mesure "
                 f"<b>{proche+1}</b></div><div class=fr>")
        for x in range(min(b, proche) - 1, max(b, proche) + 2):
            if not 0 <= x < d["n_mesures"]:
                continue
            m = d["mesures"][x]
            cl = "mm"
            if x == b:
                cl += " tab"
            if x == proche:
                cl += " chart"
            L.append(f"<div class=\"{cl}\" onclick=\"jouer("
                     f"'{html.escape(d['audio'])}',{m['t0']:.2f},"
                     f"{m['t1']:.2f})\"><span class=g>{x+1}</span>"
                     f"<b>{m['accord']}</b></div>")
        L.append("</div></div>")
    return "".join(L)


CSS = """
*{box-sizing:border-box}
body{margin:0 auto;padding:16px 12px 48px;background:#faf6ec;color:#2c2820;
 font:15px/1.5 system-ui,-apple-system,sans-serif;max-width:760px}
h1{font-size:20px;margin:0 0 8px;letter-spacing:-.01em}
h2{font-size:16px;margin:0 0 6px;display:flex;align-items:center;gap:8px}
.et{display:inline-flex;width:22px;height:22px;flex:0 0 22px;border-radius:50%;
 background:#2c2820;color:#faf6ec;font-size:12px;align-items:center;
 justify-content:center;font-weight:600}
section{margin:22px 0;padding:14px;background:#fff;border:1px solid #e6dfcc;
 border-radius:12px}
.lede{background:#fdf6df;border:1px solid #ecdfae;border-radius:12px;
 padding:12px 14px;margin:0 0 6px;font-size:14.5px}
.quoi{margin:0 0 10px;color:#6b6453;font-size:13.5px}
.note{margin:10px 0 0;color:#6b6453;font-size:13px;line-height:1.55}
table.t{border-collapse:collapse;width:100%;font-size:13.5px}
table.t th{text-align:left;font-weight:600;color:#8a8371;font-size:11.5px;
 text-transform:uppercase;letter-spacing:.04em;padding:0 8px 5px 0;
 border-bottom:1px solid #e6dfcc}
table.t td{padding:5px 8px 5px 0;border-bottom:1px solid #f2ecdc}
table.t tr.on td{background:#e9f3e6;font-weight:600}
td.g{color:#8a8371}
td.num{font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}
td.barre{width:34%}
td.barre i{display:block;height:7px;background:#b9cfae;border-radius:4px}
button.ec{font:inherit;font-size:12.5px;border:1px solid #ccc2a6;
 background:#fff;border-radius:999px;padding:3px 11px;cursor:pointer;
 color:#2c2820;white-space:nowrap}
button.ec:active{background:#efe7d0}
.front{margin:0 0 14px}
.fh{font-size:13.5px;margin:0 0 6px}
.fr{display:flex;gap:6px;flex-wrap:wrap}
.mm{flex:1 1 66px;min-width:66px;border:1px solid #e6dfcc;border-radius:9px;
 padding:7px 4px;text-align:center;background:#fdfbf4;cursor:pointer}
.mm span{display:block;font-size:10.5px;color:#a49b82}
.mm.tab{border-color:#7fa86b;background:#eef5ea;border-width:2px}
.mm.chart{border-color:#c98b5e;background:#fbf0e6;border-width:2px}
.lg{display:flex;gap:14px;font-size:12.5px;color:#6b6453;margin:8px 0 0;
 flex-wrap:wrap}
.lg i{display:inline-block;width:11px;height:11px;border-radius:3px;
 vertical-align:-1px;margin-right:4px}
"""

JS = r"""
const au=document.getElementById('au');let stop=null;
au.addEventListener('timeupdate',()=>{if(stop!=null&&au.currentTime>=stop){au.pause();stop=null;}});
function jouer(src,t0,t1){
  const go=()=>{try{au.currentTime=t0;}catch(e){}stop=t1+0.15;au.play().catch(()=>{});};
  if(au.getAttribute('src')!==src){au.setAttribute('src',src);
    au.addEventListener('loadedmetadata',go,{once:true});au.load();}
  else go();
}
"""


def page(d: dict, facile: int, dur: int, bord: int) -> str:
    B = [f"<h1>Comment un accord de tab atterrit sur une mesure</h1>",
         "<div class=lede>Le tab dit <b>quoi</b> et <b>dans quel ordre</b> ; "
         "l'audio dit <b>quand</b>. Chaque mesure choisit une case du tab, et "
         "ce choix est la somme de <b>deux logarithmes</b> : ce que l'audio "
         "pense de cette case, et ce que la grille du tab pense de cette "
         "vitesse. Deux logs dans la même unité, donc pas de poids à régler "
         "entre eux — juste une probabilité jointe.</div>",
         f"<p class=note>{html.escape(d['titre'])} · tab "
         f"{d['tab'].rating:.2f}★ · transposé de +{d['dec']} demi-tons</p>"]

    B.append(etape(1, "Le tab devient une suite de cases", bloc_cases(d),
                   "On lit les accords dans l'ordre du document. Une case = "
                   "un accord qui change."))
    B.append(etape(2, "La grille dit à quelle vitesse ça bouge",
                   bloc_cadence(d),
                   "C'est le « prior sur la grille » : avant d'écouter quoi "
                   "que ce soit, on sait déjà à quel rythme les accords "
                   "tournent dans ce tab."))
    B.append(etape(3, "musx écoute et rend une probabilité",
                   bloc_musx(d, dur),
                   "Pas un accord : une distribution. C'est ce qui permet de "
                   "la faire parler pour un accord qu'il n'a pas nommé."))
    B.append(etape(4, "La proximité musicale transforme ça en vote pour une case",
                   bloc_distance(d, dur, d["chemin"][dur]),
                   "L'étape que Louis demandait : comment on passe de « musx "
                   "pense ça » à « donc cette case vaut tant »."))
    B.append(etape(5, "Les deux termes s'additionnent, le plus grand gagne",
                   bloc_decision(d, facile,
                                 "Ici l'audio est net et il tranche tout seul.")
                   + bloc_decision(
                       d, dur,
                       "Ici l'audio hésite, et c'est la grille qui départage.")
                   + bloc_decision(
                       d, bord,
                       "Et ici, une frontière de section."),
                   "Une seule règle, répétée mesure après mesure : "
                   "<b>audio + grille</b>, et on garde le plus grand. Le "
                   "chemin complet est le meilleur enchaînement de ces "
                   "choix, pas la suite des meilleurs choix isolés."))
    B.append(etape(6, "Là où le tab et notre chart ne sont pas d'accord",
                   bloc_frontieres(d) +
                   "<div class=lg><span><i style=background:#eef5ea;"
                   "border:2px solid #7fa86b></i>le tab ouvre ici</span>"
                   "<span><i style=background:#fbf0e6;border:2px solid "
                   "#c98b5e></i>notre chart ouvre ici</span></div>"
                   "<p class=note>Touche une mesure pour l'écouter.<br>"
                   "Sur ce morceau le désaccord a toujours la même forme : le "
                   "tab termine chaque section sur le <b>A</b>, la dominante "
                   "qui ramène à la maison, et ouvre la suivante sur le "
                   "<b>Dm</b>. Notre chart, lui, ouvre la section <b>sur ce "
                   "A</b>, une mesure plus tôt. Les deux lectures se "
                   "défendent : le A est soit la dernière mesure de ce qui "
                   "finit, soit la première de ce qui commence. C'est une "
                   "question d'oreille, pas de calcul — et c'est toi "
                   "l'oreille.</p>",
                   "C'est ce que tu as trouvé « quasi bon mais pas bon »."))

    return ("<!doctype html><html lang=fr><meta charset=utf-8>"
            "<meta name=viewport content='width=device-width,initial-scale=1'>"
            "<title>Le chemin d'un accord de tab</title>"
            f"<style>{CSS}</style>" + "".join(B) +
            "<audio id=au preload=none></audio>"
            f"<script>{JS}</script>")


def choisir_mesures(d: dict) -> tuple:
    """Une mesure facile, une mesure dure, une frontière — sur ce morceau-ci.

    On ne code pas des numéros en dur : un tab qui change, et la page
    illustrerait des mesures qui n'illustrent plus rien.
    """
    E, chemin = d["table"]["E"], d["chemin"]
    lv = [float(E[b, chemin[b]]) for b in range(d["n_mesures"])]
    facile = int(np.argmax(lv))
    bords = set(d["f_tab"]) | {x + 1 for x in d["f_tab"]}
    cand = [b for b in range(1, d["n_mesures"]) if b not in bords]
    dur = min(cand, key=lambda b: lv[b]) if cand else 1
    bord = d["f_tab"][1] if len(d["f_tab"]) > 1 else (d["f_tab"] or [1])[0]
    return facile, dur, bord


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chart", default=None)
    ap.add_argument("--requete", default=None)
    ap.add_argument("--out", type=Path,
                    default=SETTINGS.reports_dir / "tabs_chemin.html")
    a = ap.parse_args(argv)
    cle, req = (a.chart, a.requete) if a.chart and a.requete else DEFAUT
    print(f"→ {req}")
    d = rassembler(cle, req)
    if not d:
        return 1
    facile, dur, bord = choisir_mesures(d)
    print(f"   mesure nette {facile+1} · mesure douteuse {dur+1} · "
          f"frontière {bord+1}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(page(d, facile, dur, bord), encoding="utf-8")
    print(f"→ {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
