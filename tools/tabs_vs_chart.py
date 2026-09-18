"""Le tab posé, et le chart qu'on écrit vraiment — mesure par mesure.

Louis, 2026-09-18 : « alors pas bon, compare maintenant au vrai chart qu'on
ferait pour cette chanson, et compare les alignements ».

Jusqu'ici on comparait le placement du tab au top-1 de musx, temps par temps.
C'est une mesure interne : musx est la matière première des deux côtés, donc
elle ne peut pas dire si le résultat ressemble à un chart. La vraie question
est celle-ci — **est-ce que le tab, posé, donne le même chart que le nôtre ?**

Deux pièges, tous deux traités ici.

  * LE CHART SERVI EST REPLIÉ. 54 mesures écrites pour 100 mesures de musique
    sur Grenade : une section revient quatre fois et n'est écrite qu'une fois.
    `deplier` rend chaque occurrence à sa place sur la ligne du temps, via
    `barSpans`. Comparer sans déplier n'aurait comparé que le début.
  * LES VOCABULAIRES DIFFÈRENT. Le chart écrit `-7`, `h7`, `^7` ; le tab écrit
    `m7`, `m7b5`, `maj7`. Les deux sont ramenés aux cinq familles de musx,
    donc on compare des accords, pas des orthographes. On perd la couleur —
    c'est assumé, c'est tout ce que les postérieures savent dire.

    python -m tools.tabs_vs_chart
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

from harmonia.integrations import tab_structure as TS
from harmonia.integrations.tab_align import (NOMS, compresser, lire_accord,
                                             meilleure_transposition, nom_q5,
                                             poser_tout, priors_musx,
                                             sequence_du_tab)
from harmonia.settings import SETTINGS

PAIRES = [("min_maroon_5_this_love", "Maroon 5 This Love"),
          ("min_bruno_mars_grenade_official_music_video", "Bruno Mars Grenade")]

#: du vocabulaire du chart vers celui d'un tab, pour passer par un seul
#: lecteur d'accord — et donc par une seule définition des cinq familles
VERS_TAB = {"": "", "-": "m", "-7": "m7", "o": "dim", "h7": "m7b5",
            "^7": "maj7", "7": "7", "^": "maj7", "+": "", "sus": "sus4"}


def texte_du_chart(ch: dict) -> str:
    """L'accord tel que le chart l'écrit, en toutes lettres."""
    if ch.get("nc"):
        return "N.C."
    s = NOMS[int(ch.get("root", 0)) % 12] + str(ch.get("q") or "")
    b = ch.get("bass", -1)
    if b is not None and int(b) >= 0 and int(b) != int(ch.get("root", -1)):
        s += "/" + NOMS[int(b) % 12]
    return s


def deplier(chart: dict) -> list[dict]:
    """Toutes les occurrences de tous les accords du chart, sur la ligne du temps.

    `barSpans[i]` donne, pour la i-ème mesure ÉCRITE d'une section, une plage
    de temps par occurrence. C'est ce qui permet de rendre à un refrain écrit
    une fois ses quatre passages réels.
    """
    out = []
    for sec in chart.get("sections") or []:
        spans = sec.get("barSpans") or []
        for i, bar in enumerate(sec.get("bars") or []):
            for j, (a0, a1) in enumerate(spans[i] if i < len(spans) else []):
                duree = max(1e-6, a1 - a0)
                for k, ch in enumerate(bar):
                    # `beat` est la position dans la mesure ; la mesure vaut
                    # bpb temps, donc on la traduit en fraction de la plage
                    t0 = a0 + duree * (float(ch.get("beat") or 0)
                                       / max(1, int(chart.get("bpb") or 4)))
                    t1 = (a0 + duree * (float(bar[k + 1].get("beat") or 0)
                                        / max(1, int(chart.get("bpb") or 4)))
                          if k + 1 < len(bar) else a1)
                    texte = texte_du_chart(ch)
                    lu = lire_accord(NOMS[int(ch.get("root", 0)) % 12]
                                     + VERS_TAB.get(str(ch.get("q") or ""),
                                                    str(ch.get("q") or "")))
                    out.append({"t0": t0, "t1": t1, "texte": texte,
                                "nc": bool(ch.get("nc")),
                                "root": None if (lu is None or ch.get("nc"))
                                        else lu["root"],
                                "q5": None if (lu is None or ch.get("nc"))
                                      else lu["q5"],
                                "bass": (None if int(ch.get("bass", -1)) < 0
                                         else int(ch["bass"]) % 12),
                                "section": sec.get("label"), "occ": j})
    out.sort(key=lambda x: x["t0"])
    return out


def etudier(cle: str, requete: str) -> dict | None:
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
    seq = compresser(sequence_du_tab(fetch_tab_chords(res[0]).raw_content))
    P = priors_musx(audio, grille)
    dec, _ = meilleure_transposition(seq, P)
    Pt = priors_musx(audio, temps)
    bpb = int(chart.get("bpb") or 4)
    origine = int(np.argmin([abs(t - grille[0]) for t in temps]))
    chemin = poser_tout(seq, Pt, dec, bpb=bpb, origine=origine)
    if chemin is None:
        print(f"   {cle} : plus d'accords que de temps")
        return None

    tabseg = []
    for t, i in enumerate(chemin):
        if tabseg and tabseg[-1]["i"] == i:
            tabseg[-1]["t1"] = temps[min(t + 1, len(temps) - 1)]
            continue
        c = seq[i]
        # la basse est DANS le tab (« E7/G# » huit fois sur This Love) et se
        # transpose comme la fondamentale. La jeter, c'est jeter la moitié de
        # ce que le projet cherche : sa cible est la basse qui SONNE.
        b = None if c.get("bass") is None else (c["bass"] + dec) % 12
        tabseg.append({"i": i, "t0": temps[t],
                       "t1": temps[min(t + 1, len(temps) - 1)],
                       "root": (c["root"] + dec) % 12, "q5": c["q5"],
                       "bass": b,
                       "texte": nom_q5((c["root"] + dec) % 12, c["q5"], b),
                       "section": c.get("section") or ""})
    notre = deplier(chart)

    def a_l_instant(segs, t):
        for g in segs:
            if g["t0"] - 1e-6 <= t < g["t1"] - 1e-6:
                return g
        return None

    # l'accord de chaque côté, temps par temps
    pareil = racine = avec_basse = compares = 0
    ecarts = []
    for t in range(len(chemin)):
        x, y = a_l_instant(notre, temps[t]), a_l_instant(tabseg, temps[t])
        if x is None or y is None or x["root"] is None:
            continue
        compares += 1
        if x["root"] == y["root"]:
            racine += 1
            bx = x.get("bass") if x.get("bass") is not None else x["root"]
            by = y.get("bass") if y.get("bass") is not None else y["root"]
            if bx % 12 == by % 12:
                avec_basse += 1
        if (x["root"], x["q5"]) == (y["root"], y["q5"]):
            pareil += 1
        else:
            ecarts.append((t, x["texte"], y["texte"]))

    # et les CHANGEMENTS : où chacun décide qu'un accord commence
    def departs(segs):
        d = set()
        for g in segs:
            k = int(np.argmin([abs(u - g["t0"]) for u in temps]))
            if abs(temps[k] - g["t0"]) < 0.12:
                d.add(k)
        return d

    dn, dt = departs(notre), departs(tabseg)

    # la forme déduite de la grille posée, au rasoir d'Occam
    mots = TS.mots_par_mesure(tabseg, grille, temps)
    f = TS.forme(mots)
    # et celle de notre chart, pour comparer : une lettre par mesure, repliée
    par_mesure = [None] * (len(grille) - 1)
    for sec in chart.get("sections") or []:
        for a, b in (sec.get("barRanges") or []):
            for k in range(max(0, a), min(len(par_mesure), b + 1)):
                par_mesure[k] = sec.get("label")
    notre_forme, prec = [], object()
    for lab in par_mesure:
        if lab != prec:
            notre_forme.append([lab, 1])
            prec = lab
        else:
            notre_forme[-1][1] += 1

    return {
        "forme": f, "mots": mots,
        "notre_forme": " ".join(f"{l or '?'}" for l, _ in notre_forme),
        "variantes": TS.variantes(f["motifs"]),
        "cle": cle, "titre": chart.get("title") or cle,
        "audio": chart.get("audio_url") or f"/audio/{stem}.m4a",
        "grille": grille, "temps": temps, "bpb": bpb,
        "notre": notre, "tab": tabseg, "decalage": dec,
        "n_mesures": len(grille) - 1,
        "n_notre": len(notre), "n_tab": len(tabseg),
        "ecrit_notre": sum(len(b) for s in (chart.get("sections") or [])
                           for b in (s.get("bars") or [])),
        "accord": pareil / max(1, compares), "compares": compares,
        "racine": racine / max(1, compares),
        "basse": avec_basse / max(1, compares),
        "changements_communs": len(dn & dt),
        "changements_notre": len(dn), "changements_tab": len(dt),
        "ecarts": ecarts,
    }


def ranger(segs: list[dict], grille: list[float], temps: list[float],
           n: int) -> list[list[dict]]:
    """Les segments d'une voix, rangés dans les mesures et à leur largeur.

    Un accord qui déborde d'une mesure sur l'autre est dessiné dans les deux,
    mais marqué `tenu` dans la seconde : il n'y COMMENCE pas. Sans ça une
    mesure qui porte deux accords en paraît quatre, et chaque accord paraît
    durer autant que ses voisins.
    """
    out = []
    for b in range(n):
        a0, a1 = grille[b], grille[b + 1]
        dedans = []
        for g in segs:
            if not (g["t0"] < a1 - 1e-6 and g["t1"] > a0 + 1e-6):
                continue
            part = sum(1 for t in temps
                       if max(a0, g["t0"]) - 1e-6 <= t < min(a1, g["t1"]) - 1e-6)
            dedans.append({**g, "part": max(1, part),
                           "tenu": b > 0 and g["t0"] < a0 - 1e-6})
        out.append(dedans)
    return out


def verdict(x: dict | None, y: dict | None) -> str:
    """Vert : le même accord, basse comprise. Ambre : la même fondamentale et
    la même basse, une couleur différente. Rouge : pas la même fondamentale,
    ou pas la même basse — les deux sont de vraies erreurs, parce que la cible
    de ce projet est la basse qui SONNE (2026-07-16)."""
    if x is None or y is None or x.get("root") is None:
        return ""
    bx = x.get("bass") if x.get("bass") is not None else x["root"]
    by = y.get("bass") if y.get("bass") is not None else y["root"]
    if x["root"] != y["root"] or bx % 12 != by % 12:
        return "faux"
    return "" if x["q5"] == y["q5"] else "ortho"


CSS = """
*{box-sizing:border-box}
body{margin:0 auto;padding:14px 12px 44px;background:#faf6ec;color:#2c2820;
 font:15px/1.5 system-ui,-apple-system,sans-serif;max-width:900px}
h1{font-size:19px;margin:0 0 6px}
h2{font-size:16px;margin:0 0 4px}
.lede{background:#fdf6df;border:1px solid #ecdfae;border-radius:12px;
 padding:11px 13px;margin:0 0 10px;font-size:14px}
.note{color:#6b6453;font-size:12.5px;margin:8px 0 0}
.carte{background:#fff;border:1px solid #e6dfcc;border-radius:12px;
 padding:13px;margin:16px 0}
.forme{background:#fbf7ea;border:1px solid #ece2c6;border-radius:9px;
 padding:10px;margin:0 0 10px}
.fl{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap;margin:0 0 4px}
.fl span{font-size:11.5px;color:#8a8371;min-width:150px}
.fl b{font:700 13.5px ui-monospace,monospace;letter-spacing:.09em;
 word-break:break-word}
.mot{display:flex;gap:3px;align-items:center;flex-wrap:wrap;margin:5px 0 0;
 font-size:11.5px}
.mot>b{width:16px;font-weight:700}
.mot>span{min-width:46px;color:#a49b82}
.mot i{font:600 11px ui-monospace,monospace;font-style:normal;background:#fff;
 border-radius:4px;padding:2px 4px;white-space:nowrap}
.chiffres{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12.5px;
 color:#6b6453;margin:4px 0 10px}
.grille{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:3px}
.mes{position:relative;min-width:0;border-radius:6px;background:#f5f0e0;
 padding:13px 3px 3px;cursor:pointer;-webkit-tap-highlight-color:transparent}
.mes .no{position:absolute;top:1px;left:4px;font:600 9px system-ui;color:#a49b82}
.voix{display:flex;gap:2px;min-height:21px;align-items:stretch}
.voix+.voix{margin-top:2px}
.ac{flex:1 1 0;min-width:0;display:flex;align-items:center;justify-content:center;
 border-radius:4px;background:#fff;font:700 12px ui-monospace,monospace;
 padding:2px 1px;overflow:hidden}
.ac.tenu{background:transparent;font-weight:400;color:#bdb49a}
.ac.faux{background:#f7dcd3;color:#8f3a1e}
.ac.ortho{background:#fbf0cf;color:#7a5f14}
.eti{position:absolute;right:4px;top:1px;font:600 8.5px system-ui;color:#bdb49a}
.sec{grid-column:1/-1;font:600 10px system-ui;letter-spacing:.06em;
 text-transform:uppercase;color:#8a8371;margin:9px 0 0}
.lg{display:flex;gap:12px;flex-wrap:wrap;font-size:12px;color:#6b6453;
 margin:10px 0 0}
.lg i{display:inline-block;width:10px;height:10px;border-radius:3px;
 vertical-align:-1px;margin-right:4px;border:1px solid #0001}
.row{display:flex;gap:6px;flex-wrap:wrap;margin-top:10px}
button{border:1px solid #d8cfb4;border-radius:9px;background:#fff;
 padding:8px 11px;font:600 13px system-ui;cursor:pointer;min-height:42px;
 color:#2c2820}
button.on{background:#2c2820;border-color:#2c2820;color:#fff}
"""

JS = r"""
const au=document.getElementById('au');let stop=null;
au.addEventListener('timeupdate',()=>{if(stop!=null&&au.currentTime>=stop){au.pause();stop=null;}});
function jouer(src,t0,t1){
  const go=()=>{try{au.currentTime=t0;}catch(e){}stop=t1+0.1;au.play().catch(()=>{});};
  if(au.getAttribute('src')!==src){au.setAttribute('src',src);
    au.addEventListener('loadedmetadata',go,{once:true});au.load();}
  else go();
}
let V={};
fetch("/api/verdicts/tabs_vs_chart").then(r=>r.json()).then(d=>{V=d.reponses||{};peindre();})
                                    .catch(()=>peindre());
function peindre(){
  for(const c of document.querySelectorAll('.carte')){
    const v=V[c.dataset.cle];
    for(const b of c.querySelectorAll('.vd button')) b.classList.toggle('on', b.dataset.v===v);
  }
}
function juger(cle,quoi){
  V[cle]=quoi; peindre();
  fetch("/api/verdicts/tabs_vs_chart",{method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({reponses:V})}).catch(()=>{});
}
"""


def carte(d: dict) -> str:
    n = d["n_mesures"]
    A = ranger(d["notre"], d["grille"], d["temps"], n)
    B = ranger(d["tab"], d["grille"], d["temps"], n)
    f = d["forme"]
    L = [f"<div class=carte data-cle=\"{html.escape(d['cle'])}\">",
         f"<h2>{html.escape(d['titre'])}</h2>",
         "<div class=forme>",
         f"<div class=fl><span>la forme déduite du tab</span>"
         f"<b>{html.escape(TS.mot(f['sections']))}</b></div>",
         f"<div class=fl><span>la forme de notre chart</span>"
         f"<b>{html.escape(d['notre_forme'])}</b></div>",
         f"<p class=note>{f['cout'][0]} mesures à écrire, "
         f"{f['cout'][1]} sections différentes, {f['cout'][2]} posées — "
         f"c'est l'écriture la plus courte qui explique les "
         f"{d['n_mesures']} mesures du morceau.</p>"]
    for lettre, bloc in sorted(f["motifs"].items()):
        L.append(f"<div class=mot><b>{lettre}</b>"
                 f"<span class=g>{len(bloc)} mes.</span>"
                 + "".join(f"<i>{html.escape(' '.join(x) or '·')}</i>"
                           for x in bloc) + "</div>")
    if d["variantes"]:
        L.append("<p class=note>Ne diffèrent que par la FIN, donc peut-être "
                 "une seule section : "
                 + ", ".join(f"<b>{a}</b> et <b>{b}</b> ({k} mesure"
                             f"{'s' if k > 1 else ''})"
                             for a, b, k in d["variantes"])
                 + ". On ne les fond pas tout seul — c'est à toi.</p>")
    L.append("</div>")
    L += [
         "<div class=chiffres>"
         f"<span>notre chart <b>{d['ecrit_notre']}</b> accords écrits, "
         f"<b>{d['n_notre']}</b> joués</span>"
         f"<span>le tab <b>{d['n_tab']}</b> posés</span>"
         f"<span>même accord <b>{d['accord']:.0%}</b> des temps</span>"
         f"<span>même fondamentale <b>{d['racine']:.0%}</b></span>"
         f"<span>même fondamentale ET basse <b>{d['basse']:.0%}</b></span>"
         f"<span>changements au même temps <b>{d['changements_communs']}</b> "
         f"sur {d['changements_notre']} / {d['changements_tab']}</span>"
         "</div>", "<div class=grille>"]
    section = object()
    for b in range(n):
        lab = (B[b][0]["section"] if B[b] else "")
        if lab != section:
            section = lab
            if lab:
                L.append(f"<div class=sec>{html.escape(str(lab))}</div>")
        lignes = []
        for voix, dedans in (("nous", A[b]), ("tab", B[b])):
            cells = []
            for g in dedans:
                if voix == "nous":
                    v = ""
                else:
                    # celui qui chevauche LE PLUS, pas le premier venu :
                    # sinon un accord du tab qui déborde d'un souffle sur la
                    # mesure d'avant se fait comparer au voisin et sort rouge
                    # alors qu'il est juste
                    x = max(A[b], key=lambda y: max(
                        0.0, min(y["t1"], g["t1"]) - max(y["t0"], g["t0"])),
                        default=None)
                    if x is not None and min(x["t1"], g["t1"]) <= max(
                            x["t0"], g["t0"]):
                        x = None
                    v = verdict(x, g)
                cl = "ac" + (" tenu" if g["tenu"] else "") + (f" {v}" if v else "")
                cells.append(
                    f"<div class='{cl}' style=flex-grow:{g['part']}>"
                    + ("·" if g["tenu"] else html.escape(g["texte"])) + "</div>")
            lignes.append(f"<div class=voix>{''.join(cells) or '&nbsp;'}</div>")
        L.append(
            f"<div class=mes onclick=\"jouer('{html.escape(d['audio'])}',"
            f"{d['grille'][b]:.2f},{d['grille'][b+1]:.2f})\">"
            f"<div class=no>{b+1}</div>{''.join(lignes)}</div>")
    L.append("</div>")
    L.append("<div class=lg><span>ligne du haut : <b>notre chart</b></span>"
             "<span>ligne du bas : <b>le tab posé</b></span>"
             "<span><i style=background:#fff></i>même accord</span>"
             "<span><i style=background:#fbf0cf></i>même fondamentale, "
             "écrite autrement</span>"
             "<span><i style=background:#f7dcd3></i>fondamentale différente"
             "</span><span><i style=background:#faf6ec></i>· accord tenu"
             "</span></div>")
    cle = html.escape(d["cle"])
    L.append("<div class='row vd'>"
             f"<button data-v=tab onclick=\"juger('{cle}','tab')\">"
             "le tab a raison</button>"
             f"<button data-v=nous onclick=\"juger('{cle}','nous')\">"
             "notre chart a raison</button>"
             f"<button data-v=melange onclick=\"juger('{cle}','melange')\">"
             "ça dépend des endroits</button>"
             f"<button data-v='sais pas' onclick=\"juger('{cle}','sais pas')\">"
             "je ne sais pas</button></div>")
    return "".join(L) + "</div>"


def page(songs: list[dict]) -> str:
    B = ["<h1>Le tab posé, et le chart qu'on écrit vraiment</h1>",
         "<div class=lede>Deux lignes par mesure : <b>en haut notre chart</b> "
         "tel que l'app le sert, <b>en bas le tab posé</b>. Le chart servi est "
         "replié — une section écrite une fois est jouée quatre — donc il est "
         "déplié ici pour que chaque occurrence soit à sa place.<br>"
         "Les deux vocabulaires sont ramenés aux cinq familles de musx, donc "
         "on compare des accords et pas des orthographes. Touche une mesure "
         "pour l'écouter.<br>En tête de chaque morceau, <b>la forme déduite "
         "de la grille</b> : l'écriture la plus courte qui l'explique, une "
         "lettre par section, ce qui se répète n'étant écrit qu'une fois.</div>"]
    for s in songs:
        B.append(carte(s))
    return ("<!-- tools/tabs_vs_chart.py -->"
            "<meta charset=utf-8>"
            "<meta name=viewport content='width=device-width,initial-scale=1'>"
            "<title>Le tab contre notre chart</title><style>" + CSS +
            "</style><body>" + "".join(B) +
            "<audio id=au preload=none playsinline></audio>"
            "<script>" + JS + "</script>")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chart", default=None)
    ap.add_argument("--requete", default=None)
    ap.add_argument("--out", type=Path,
                    default=SETTINGS.reports_dir / "tabs_vs_chart.html")
    a = ap.parse_args(argv)
    paires = [(a.chart, a.requete)] if a.chart and a.requete else PAIRES
    songs = []
    for cle, req in paires:
        print(f"→ {req}")
        d = etudier(cle, req)
        if not d:
            continue
        songs.append(d)
        print(f"   même accord {d['accord']:.1%} · même fondamentale "
              f"{d['racine']:.1%} · {d['changements_communs']}/"
              f"{d['changements_notre']} changements au même temps")
    if not songs:
        print("rien à montrer")
        return 1
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(page(songs), encoding="utf-8")
    print(f"\n→ {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
