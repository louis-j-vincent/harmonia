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

from harmonia.integrations.tab_align import (NOMS, compresser,
                                             meilleure_transposition, nom_q5,
                                             part_des_temoins, poser_tout,
                                             prior_de_grille, priors_musx,
                                             sequence_du_tab, termes,
                                             termes_du_segment)
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
    temps = [float(t) for t in (chart.get("beatTimes") or [])]
    if not audio.exists() or len(grille) < 3 or len(temps) < 8:
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

    Pt = priors_musx(audio, temps)
    bpb = int(chart.get("bpb") or 4)
    origine = int(np.argmin([abs(t - grille[0]) for t in temps]))
    chemin = poser_tout(seq, Pt, dec, bpb=bpb, origine=origine)
    if chemin is None:
        print(f"   {cle} : plus d'accords que de temps")
        return None
    T = termes(seq, Pt, dec, bpb=bpb, origine=origine)

    # chaque accord devient un segment [premier temps, dernier temps]
    segments = []
    for t, i in enumerate(chemin):
        if segments and segments[-1]["case"] == i:
            segments[-1]["fin"] = t + 1
            continue
        c = seq[i]
        segments.append({"case": i, "debut": t, "fin": t + 1,
                         "accord": nom_q5((c["root"] + dec) % 12, c["q5"]),
                         "section": c.get("section") or ""})
    for g in segments:
        g["t0"] = temps[g["debut"]]
        g["t1"] = temps[min(g["fin"], len(temps) - 1)]
        g["mesure"] = int(np.searchsorted(grille, g["t0"] + 1e-6)) or 1

    f_tab = [k for k in range(1, len(segments))
             if segments[k]["section"] != segments[k - 1]["section"]]
    f_chart = sorted({a for sc in (chart.get("sections") or [])
                      for a, _ in (sc.get("barRanges") or [])})

    return {"cle": cle, "titre": chart.get("title") or cle,
            "audio": chart.get("audio_url") or f"/audio/{stem}.m4a",
            "tab": res[0], "seq": seq, "P": Pt, "dec": dec, "chemin": chemin,
            "T": T, "grille": grille, "temps": temps, "bpb": bpb,
            "origine": origine, "segments": segments,
            "prior": prior_de_grille(seq),
            "f_tab": f_tab, "f_chart": f_chart,
            "n_temps": len(chemin), "n_mesures": len(grille) - 1}


# ── rendu ───────────────────────────────────────────────────────────────────

def n(x: float, d: int = 3) -> str:
    """Un nombre lisible, virgule décimale, signe explicite pour les logs."""
    s = f"{x:+.{d}f}" if d and abs(x) < 100 else f"{x:.{d}f}"
    return s.replace(".", ",")


def bouton(d: dict, t0: float, t1: float, texte: str) -> str:
    return (f"<button class=ec onclick=\"jouer('{html.escape(d['audio'])}',"
            f"{t0:.2f},{t1:.2f})\">{texte}</button>")


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
            f"<p class=note>{len(d['seq'])} accords en tout, pour {tot} lignes "
            f"écrites dans le tab, {d['n_mesures']} mesures et "
            f"{d['n_temps']} temps de musique. Un accord écrit au-dessus de "
            f"trois lignes de paroles est un accord tenu, pas trois "
            f"changements : on le replie en une case qui pèse 3.<br>"
            f"<b>Les {len(d['seq'])} sont posés</b>, dans l'ordre, du premier "
            f"temps au dernier. Aucun n'est jeté : il y a plus d'accords que "
            f"de mesures, mais bien moins que de temps.</p>")


def bloc_cadence(d: dict) -> str:
    """Étape 2 — ce que le tab annonce AVANT d'écouter quoi que ce soit."""
    T = d["T"]
    L = ["<table class=t><tr><th>accord<th>lignes du tab<th>durée annoncée"]
    for i, c in enumerate(d["seq"][:6]):
        L.append(f"<tr><td><b>{nom_q5((c['root'] + d['dec']) % 12, c['q5'])}</b>"
                 f"<td class=g>{c.get('repetitions', 1)}"
                 f"<td class=num>{n(float(T['attendue'][i]), 1)[1:]} temps")
    L.append("</table>")
    dep = T["depart"]
    o, b = d["origine"], d["bpb"]
    M = ["<table class=t><tr><th>un accord qui commence<th>coûte"]
    noms = {0: "sur le 1er temps de la mesure", b // 2: "au milieu de la mesure"}
    for k in range(b):
        t = o + k
        while t >= len(dep):
            t -= b
        M.append(f"<tr{' class=on' if k == 0 else ''}>"
                 f"<td>{noms.get(k, f'sur le temps {k + 1}')}"
                 f"<td class=num>{n(float(dep[t]), 2)}")
    M.append("</table>")
    return ("".join(L) + "".join(M) +
            "<p class=note>Deux choses, et elles viennent du tab seul.<br>"
            "<b>La durée</b> : un accord écrit au-dessus de trois lignes de "
            "paroles attend trois fois plus de temps qu'un accord écrit "
            "au-dessus d'une seule. On paie l'écart à cette durée.<br>"
            "<b>La place</b> : un accord change de préférence sur un temps "
            "fort. Mesuré sur ces deux morceaux sans aucun prior, 62 % et "
            "63 % des changements tombaient déjà sur le premier temps de la "
            "mesure, 93 % et 92 % sur un temps fort — le prior ne fait que "
            "finir le travail (97 % et 98 %), pour 0,3 point d'accord avec "
            "musx en moins.</p>")


def bloc_musx(d: dict, b: int) -> str:
    """Étape 3 — ce que musx entend sur une mesure."""
    P = d["P"]
    ordre = np.argsort(-P[b])[:5]
    L = ["<table class=t><tr><th>accord<th>ce que musx lui donne"]
    for c in ordre:
        c = int(c)
        L.append(f"<tr><td><b>{nom_q5(c // 5, c % 5)}</b>"
                 f"<td class=num>{n(float(P[b][c]), 3)[1:]}"
                 f"<td class=barre><i style=\"width:{P[b][c]*100:.0f}%\"></i>")
    L.append("</table>")
    return ("".join(L) +
            f"<p class=note>Temps {b+1}, dans la mesure {1 + int(np.searchsorted(d['grille'], d['temps'][b] + 1e-6)) - 1}. "
            f"musx ne rend pas un accord, il rend une probabilité sur les 60 "
            f"accords possibles (12 fondamentales × 5 familles). "
            f"{bouton(d, d['temps'][b], d['temps'][b+1], 'écouter ce temps')}</p>")


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
    lv = d["T"]["E"][b, case]
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


def bloc_decision(d: dict, k: int, pourquoi: str) -> str:
    """Étape 5 — les trois termes s'additionnent, le plus grand gagne.

    On montre LA vraie décision : où tombe la frontière entre l'accord `k` et
    le suivant. Les deux accords couvrent ensemble les mêmes temps quelle que
    soit la ligne, donc les totaux se comparent.

    Une première version faisait varier la FIN de l'accord `k` sans rien
    mettre derrière. Les temps laissés libres n'étaient payés par personne,
    donc un accord plus court semblait toujours moins cher, et la ligne
    retenue n'était jamais celle du plus gros total. Un tableau qui ne compare
    pas la même chose sur chaque ligne ne compare rien.
    """
    g, h = d["segments"][k], d["segments"][k + 1]
    T = d["T"]
    debut, fin_h = g["debut"], h["fin"]
    coupes = sorted({c for c in range(g["fin"] - 2, g["fin"] + 3)
                     if debut < c < fin_h})
    L = [f"<table class=t><tr><th>frontière<th>{html.escape(g['accord'])}"
         f"<th>{html.escape(h['accord'])}<th>l'audio<th>durée<th>place<th>total"]
    for c in coupes:
        a = termes_du_segment(T, g["case"], debut, c)
        b = termes_du_segment(T, h["case"], c, fin_h)
        tot = a["total"] + b["total"]
        mark = " class=on" if c == g["fin"] else ""
        L.append(f"<tr{mark}><td class=g>temps {c + 1}"
                 f"<td class=g>{a['temps']}<td class=g>{b['temps']}"
                 f"<td class=num>{n(a['audio'] + b['audio'], 2)}"
                 f"<td class=num>{n(a['duree'] + b['duree'], 2)}"
                 f"<td class=num>{n(a['depart'] + b['depart'], 2)}"
                 f"<td class=num><b>{n(tot, 2)}</b>")
    L.append("</table>")
    return ("".join(L) +
            f"<p class=note>Où s'arrête le <b>{html.escape(g['accord'])}</b> "
            f"(le {k+1}<sup>e</sup> accord du tab, mesure {g['mesure']}) et où "
            f"commence le <b>{html.escape(h['accord'])}</b> ? {pourquoi} "
            f"{bouton(d, g['t0'], h['t1'], 'écouter les deux')}<br>"
            f"Sur chaque ligne les deux accords couvrent les mêmes temps : "
            f"seule la frontière bouge, donc les totaux se comparent.</p>")


def bloc_frontieres(d: dict) -> str:
    """Étape 6 — là où le tab et notre chart ne posent pas la frontière."""
    L, paires = [], []
    for k in d["f_tab"]:
        g = d["segments"][k]
        b = g["mesure"] - 1                      # 0-indexé
        cand = [x for x in d["f_chart"] if 0 < x < d["n_mesures"]]
        if not cand:
            continue
        proche = min(cand, key=lambda x: abs(x - b))
        if 0 < abs(b - proche) <= 3:
            paires.append((k, b, proche))
    if not paires:
        return ("<p class=note>Le tab et le chart posent les mêmes "
                "frontières.</p>")
    for k, b, proche in paires[:6]:
        g = d["segments"][k]
        L.append(f"<div class=front><div class=fh>« "
                 f"{html.escape(g['section'])} » — le tab l'ouvre sur le "
                 f"<b>{html.escape(g['accord'])}</b> de la mesure "
                 f"<b>{b+1}</b>, notre chart ouvre mesure "
                 f"<b>{proche+1}</b></div><div class=fr>")
        for x in range(min(b, proche) - 1, max(b, proche) + 2):
            if not 0 <= x < d["n_mesures"]:
                continue
            dedans = [y["accord"] for y in d["segments"] if y["mesure"] == x + 1]
            cl = "mm" + (" tab" if x == b else "") + (" chart" if x == proche else "")
            L.append(f"<div class=\"{cl}\" onclick=\"jouer("
                     f"'{html.escape(d['audio'])}',{d['grille'][x]:.2f},"
                     f"{d['grille'][x+1]:.2f})\"><span class=g>{x+1}</span>"
                     f"<b>{html.escape(' '.join(dedans) or '—')}</b></div>")
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


def page(d: dict, facile: int, dur: int, seg: int) -> str:
    B = [f"<h1>Où chaque accord du tab commence</h1>",
         "<div class=lede><b>Tous</b> les accords du tab sont posés sur le "
         "morceau, dans l'ordre, du premier temps au dernier. La seule "
         "question est <i>où chacun commence</i>, et elle se tranche par une "
         "somme de <b>trois logarithmes</b> : ce que l'audio pense de cet "
         "accord, la durée que le tab lui annonce, et la place du temps où il "
         "commence. Trois logs dans la même unité, donc pas de poids à régler "
         "entre eux — une seule probabilité jointe qu'on lit en "
         "logarithme.</div>",
         f"<p class=note>{html.escape(d['titre'])} · tab "
         f"{d['tab'].rating:.2f}★ · transposé de +{d['dec']} demi-tons · "
         f"{len(d['seq'])} accords sur {d['n_temps']} temps</p>"]

    B.append(etape(1, "Le tab devient une suite d'accords", bloc_cases(d),
                   "On lit les accords dans l'ordre du document. Une case = "
                   "un accord qui change."))
    B.append(etape(2, "Le tab annonce une durée et une place",
                   bloc_cadence(d),
                   "C'est le « prior sur la grille » : avant d'écouter quoi "
                   "que ce soit, on sait déjà combien de temps chaque accord "
                   "devrait durer, et qu'un accord change sur un temps "
                   "fort."))
    B.append(etape(3, "musx écoute et rend une probabilité",
                   bloc_musx(d, dur),
                   "Pas un accord : une distribution. C'est ce qui permet de "
                   "la faire parler pour un accord qu'il n'a pas nommé."))
    B.append(etape(4, "La proximité musicale transforme ça en vote pour une case",
                   bloc_distance(d, dur, d["chemin"][dur]),
                   "L'étape que Louis demandait : comment on passe de « musx "
                   "pense ça » à « donc cette case vaut tant »."))
    B.append(etape(5, "Les trois termes s'additionnent, le plus grand gagne",
                   bloc_decision(
                       d, seg,
                       "Une durée de plus ou de moins, et voilà ce que ça "
                       "coûte.")
                   + bloc_decision(
                       d, max(1, seg - 1),
                       "L'accord juste avant, pour voir la même règle sur un "
                       "autre cas."),
                   "Une seule règle, répétée accord après accord : "
                   "<b>audio + durée + place</b>, et on garde le plus grand. "
                   "Le découpage complet est le meilleur enchaînement de ces "
                   "choix, pas la suite des meilleurs choix isolés — et les "
                   "deux bouts sont fixés : le premier accord ouvre le "
                   "morceau, le dernier le ferme."))
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
                   "l'oreille.<br>Rappel : le tab n'a aucune mesure. Les "
                   "numéros ci-dessus sont ceux que <i>ce placement</i> lui "
                   "donne. Ce que le tab dit vraiment est ordinal : quel "
                   "accord ouvre la section.</p>",
                   "C'est ce que tu as trouvé « quasi bon mais pas bon »."))

    return ("<!doctype html><html lang=fr><meta charset=utf-8>"
            "<meta name=viewport content='width=device-width,initial-scale=1'>"
            "<title>Le chemin d&#39;un accord de tab</title>"
            f"<style>{CSS}</style>" + "".join(B) +
            "<audio id=au preload=none></audio>"
            f"<script>{JS}</script>")


def choisir_mesures(d: dict) -> tuple:
    """Un temps net, un temps douteux, une frontière — sur ce morceau-ci.

    On ne code pas des numéros en dur : un tab qui change, et la page
    illustrerait des endroits qui n'illustrent plus rien.
    """
    E, chemin = d["T"]["E"], d["chemin"]
    lv = [float(E[t, chemin[t]]) for t in range(len(chemin))]
    facile = int(np.argmax(lv))
    dur = int(np.argmin(lv))
    # un segment bien au milieu, assez long pour que les durées voisines
    # existent toutes
    longs = [k for k, g in enumerate(d["segments"])
             if g["fin"] - g["debut"] >= 4 and 0 < k < len(d["segments"]) - 2
             and d["segments"][k + 1]["fin"] - d["segments"][k + 1]["debut"] >= 3]
    seg = longs[len(longs) // 2] if longs else 1
    return facile, dur, seg


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
    facile, dur, seg = choisir_mesures(d)
    print(f"   temps net {facile+1} · temps douteux {dur+1} · "
          f"accord illustré n°{seg+1}")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(page(d, facile, dur, seg), encoding="utf-8")
    print(f"→ {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
