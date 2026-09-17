"""Le chart brut, et les étapes de l'algorithme posées DESSUS.

Louis, 2026-09-17 : « je veux le chart brut avec itérativement au cours du
temps les sections qui y sont détectées, que je puisse les voir et les relier
au morceau — là je vois aa bbb cc, ça ne me parle pas ».

Il a raison : un mot de lettres ne dit rien tant qu'on ne voit pas à quelles
mesures il correspond. Cette page montre donc le chart comme l'app le montre,
quatre mesures par ligne avec les accords, et par-dessus l'état de
l'algorithme À CHAQUE ÉTAPE :

    étape 0   les jetons nus — une lettre par bi-mesure
    étape 1…n après chaque soudure, les unités telles qu'elles sont
    final     les blocs nommés, avec la provenance de chaque nom

Chaque unité s'écoute au doigt. C'est tout l'objet de la page : entendre ce
que « a » veut dire.

CE QU'ELLE NE FAIT PAS : corriger quoi que ce soit (Louis, le même jour :
« n'essaye pas de fix, c'est moi qui dirai comment fix »).

    python -m tools.chart_etapes --chart min_xxx --trait A:7-14
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

from harmonia.settings import SETTINGS
from tools.annotation_degats import couleur, nom_accord


def cherche_brique(chart, b0: int, b1: int, seuil: float = 0.90) -> list[tuple]:
    """Les reprises de la brique [b0, b1] dans le morceau, sans recouvrement.

    Louis, 2026-09-17 : « une fois que j'ai annoté une section, ça devient une
    brique, et la première chose à faire c'est de trouver d'autres occurrences
    de cette section ». C'est le contraire de ce que fait l'algorithme
    aujourd'hui, qui jette la brique dans une agglomération générique.

    On fait glisser la brique le long de la matrice de ressemblance chord-tone
    — celle-là même que l'app utilise déjà pour répondre « où ce bloc se
    rejoue-t-il ? » — et on retient les pics par score décroissant, en
    refusant tout recouvrement.

    PRÉMISSE VÉRIFIÉE avant d'écrire une ligne d'algorithme (règle 2 du
    CLAUDE.md), sur Don't Want My Love avec la brique A de Louis (mesures
    7-14) : la recherche rend 7-14, 15-22 et 34-41 — exactement les trois
    occurrences que SongFormer avait trouvées, et exactement celles que
    l'algorithme actuel rate (il propose 15-22, 29-36, 49-56). Le résultat ne
    bouge pas entre 0,85 et 0,95 de seuil. Sur la brique B (23-33) il rend
    23-33 et 42-52 ; sur l'outro (49-56), lui seul — un outro ne se rejoue pas.

    CE QUE ÇA NE RÉSOUT PAS : rien n'est branché. C'est une mesure, pas un
    correctif — Louis tranchera.
    """
    import numpy as np

    from harmonia import musx as _musx
    from harmonia.sections.similarity import _slide, ssm
    grid = chart["barGrid"]
    n = len(grid) - 1
    L = b1 - b0 + 1
    stem = Path(chart.get("audio_url") or "").stem
    S = ssm(_musx.frame_posteriors(SETTINGS.audio_dir / f"{stem}.m4a")[0], grid)
    sc = _slide(S, b0, L, n)
    pris, out = [], []
    for b in sorted(range(n - L + 1), key=lambda x: -sc[x]):
        if sc[b] < seuil:
            break
        if any(not (b + L <= c or c + L <= b) for c in pris):
            continue
        pris.append(b)
        out.append((b, b + L - 1, float(sc[b])))
    return sorted(out)


def remplir_par_algo(chart, occ, label):
    """Les trous entre les occurrences, remplis par l'algorithme des 4 mots.

    Les occurrences de la brique de Louis entrent TOUTES comme unités gelées —
    pas seulement celle qu'il a tracée. L'agglomération ne travaille donc que
    dans les trous, et ne peut plus donner son nom à un bloc qu'elle a
    fabriqué elle-même.
    """
    from harmonia.phrases4 import grouper_restes, merges4, nommer, phrases
    from harmonia.soudure import mot_sur_traits

    traits = [(b0, b1) for b0, b1, _ in occ]
    bornes, mot, _src = mot_sur_traits(chart, traits,
                                       audio_dir=SETTINGS.audio_dir)
    idx = {b: j for j, b in enumerate(bornes)}
    fixes = [(idx[b0], idx[b1 + 1] - 1) for b0, b1 in traits if b0 in idx]
    depart, j = [], 0
    for j0, j1 in fixes:
        while j < j0:
            depart.append((j, j + 1, mot[j])); j += 1
        depart.append((j0, j1 + 1, mot[j0:j1 + 1])); j = j1 + 1
    while j < len(mot):
        depart.append((j, j + 1, mot[j])); j += 1
    geles = {(j0, j1 + 1) for j0, j1 in fixes}
    _secs, info = phrases(mot, depart=depart, geles=geles)
    steps = merges4(mot, cible=info["cible"], depart=depart, geles=geles)
    blocs = grouper_restes(nommer(steps[-1]["jetons"], info["cible"],
                                  geles=geles), mot, info["cible"])
    siens = {(j0, j1 + 1) for j0, j1 in fixes}
    # Les lettres neuves ne doivent pas rentrer en collision avec la sienne.
    libres = [c for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ" if c != label.upper()]
    renom, k, out = {}, 0, []
    for b in blocs:
        sien = (b["j0"], b["j1"]) in siens
        if sien:
            nom = label
        else:
            if b["label"] not in renom:
                renom[b["label"]] = libres[k % len(libres)]; k += 1
            nom = renom[b["label"]] + ("′" if b["prime"] else "")
        out.append({"m0": bornes[b["j0"]], "m1": bornes[b["j1"]] - 1,
                    "nom": nom, "txt": b["type"], "humain": sien})
    return out


def remplir_par_songformer(chart, occ, label, auto_sections):
    """Les trous, remplis par ce que SongFormer avait trouvé.

    Les mesures couvertes par la brique de Louis gardent SON nom ; partout
    ailleurs on reprend l'étiquette de SongFormer, et les mesures voisines de
    même étiquette redeviennent un bloc. C'est le contraire de ce que fait la
    route aujourd'hui, qui jette purement et simplement ce découpage.
    """
    n = int(chart.get("nBars") or 0)
    par_mes = ["?"] * n
    for sec in auto_sections or []:
        for a, b in (sec.get("barRanges") or []):
            for i in range(max(0, a), min(n, b + 1)):
                par_mes[i] = sec.get("label") or "?"
    # Ses occurrences sont des blocs À PART, jamais fondues entre elles : deux
    # occurrences voisines de la même brique restent deux occurrences. Les
    # fondre affichait « A 7-22 » là où le morceau joue deux fois A.
    pris = [False] * n
    blocs = []
    for b0, b1, _ in occ:
        b0, b1 = max(0, b0), min(n - 1, b1)
        blocs.append({"m0": b0, "m1": b1, "nom": label, "txt": "toi",
                      "humain": True})
        for i in range(b0, b1 + 1):
            pris[i] = True
    i = 0
    while i < n:
        if pris[i]:
            i += 1
            continue
        j = i
        while j + 1 < n and not pris[j + 1] and par_mes[j + 1] == par_mes[i]:
            j += 1
        blocs.append({"m0": i, "m1": j, "nom": par_mes[i],
                      "txt": "songformer", "humain": False})
        i = j + 1
    return sorted(blocs, key=lambda b: b["m0"])


def _sous_etape0(label, b0, b1, avant, apres, mot_av, mot_ap, n_unites) -> str:
    """Ce que le trait a VRAIMENT changé avant même que l'algorithme démarre.

    Pas seulement « il est gelé ». Sur Don't Want My Love, la machine avait
    trouvé une COUTURE — un jeton d'une seule mesure à la 30ᵉ, posé là où le
    morceau a une mesure en trop. `jetons_sur_traits` repose une grille
    régulière : la couture disparaît, et tout ce qui suit la mesure 30 se
    décale d'une mesure. Trois jetons changent alors de lettre, à l'autre bout
    du morceau, très loin du trait. C'est mesuré ici et écrit sur la page
    plutôt que deviné.
    """
    bouts = [f"ton trait {label} (mesures {b0}-{b1}) entre comme UNE unité "
             f"gelée ; {len(apres)-1} jetons, {n_unites} unités"]
    d = [i for i, (x, y) in enumerate(zip(avant, apres)) if x != y]
    if d:
        bouts.append(f"⚠ les bords bougent à partir de la mesure "
                     f"{min(avant[d[0]], apres[d[0]]) + 1} : la couture que la "
                     "machine avait posée là disparaît, et tout ce qui suit se "
                     "décale")
    chg = [i for i, (x, y) in enumerate(zip(mot_av, mot_ap)) if x != y]
    if chg:
        bouts.append(f"⚠ {len(chg)} jeton(s) changent de lettre, dont le n°"
                     f"{chg[0]} — loin de ton trait")
    return " · ".join(bouts)


def etapes(cle: str, label: str, b0: int, b1: int) -> dict:
    """Le chart, et l'état des unités après chaque soudure."""
    from harmonia.phrases4 import (cout, grouper_restes, merges4, nommer,
                                   phrases)
    from harmonia.soudure import (accords_par_mesure, mot_sur_traits,
                                  song_du_chart, traits_propres)

    chart = json.loads((SETTINGS.charts_dir / f"{cle}.json")
                       .read_text(encoding="utf-8"))
    song = song_du_chart(chart, audio_dir=SETTINGS.audio_dir)
    n = int(chart.get("nBars") or 0)
    grid = [float(t) for t in (chart.get("barGrid") or [])]
    stem = Path(chart.get("audio_url") or "").stem

    gardes, _ = traits_propres(
        [{"label": label, "mesure_debut": b0, "mesure_fin": b1}],
        song["n_mesures"])
    bornes, mot, src_mot = mot_sur_traits(
        chart, [(x, y) for x, y, _ in gardes], audio_dir=SETTINGS.audio_dir)
    idx = {b: j for j, b in enumerate(bornes)}
    fixes = [(idx[x], idx[y + 1] - 1, lab) for x, y, lab in gardes]

    depart, j = [], 0
    for j0, j1, _ in fixes:
        while j < j0:
            depart.append((j, j + 1, mot[j])); j += 1
        depart.append((j0, j1 + 1, mot[j0:j1 + 1])); j = j1 + 1
    while j < len(mot):
        depart.append((j, j + 1, mot[j])); j += 1
    geles = {(j0, j1 + 1) for j0, j1, _ in fixes}

    # l'hypothèse retenue, pour ne dérouler QUE celle-là
    _secs, info = phrases(mot, depart=depart, geles=geles)
    cible = info["cible"]
    steps = merges4(mot, cible=cible, depart=depart, geles=geles)

    siens = {(j0, j1 + 1) for j0, j1, _ in fixes}

    def en_mesures(toks):
        # `humain` distingue LA section annotée des bi-mesures brutes. Louis,
        # 2026-09-17 : « pour moi l'étape 0 ça devrait être juste mon
        # annotation de la section A et rien d'autre, là pourquoi il y a déjà
        # d'autres sections annotées ». Il a raison : il n'y en a qu'une. Les
        # autres unités ne sont pas des sections, ce sont les jetons de deux
        # mesures, la matière première que l'algorithme va souder. Les peindre
        # pareil laissait croire à un découpage déjà fait.
        return [{"m0": bornes[t[0]], "m1": bornes[t[1]] - 1, "txt": t[2],
                 "humain": (t[0], t[1]) in siens} for t in toks]

    # AVANT TON TRAIT — la grille que la machine se donne toute seule. Louis,
    # 2026-09-17 : « et ça c'est à quel moment que l'humain a noté quelque
    # chose ? moi j'ai annoté une section, c'est l'étape 0 ou 1 ? ». Ni l'une
    # ni l'autre : son trait entre AVANT l'étape 0, et il ne fait pas que se
    # figer — les bords des jetons épousent ses frontières, donc le mot lui-
    # même change. Cette vue est là pour qu'on voie ce que son trait déplace.
    vues = [{"titre": "avant ton trait",
             "sous": f"la machine seule découpe en {len(song['jetons'])-1} "
                     "bi-mesures posées de deux en deux, et lit un mot dessus",
             "unites": [{"m0": song["jetons"][k], "m1": song["jetons"][k+1]-1,
                         "txt": song["mot"][k]}
                        for k in range(len(song["jetons"]) - 1)]},
            {"titre": "étape 0 — ton trait est posé",
             "sous": "UNE seule section est annotée, la tienne (cadre "
                     "noir) ; tout le reste n'est pas découpé, ce sont les "
                     "bi-mesures brutes — " + _sous_etape0(
                         label, b0, b1, song["jetons"], bornes,
                         song["mot"], mot, len(depart)),
             "unites": en_mesures(depart)}]
    for k, s in enumerate(steps[1:], 1):
        vues.append({
            "titre": f"étape {k} — on soude « {s['paire'][0]} » + "
                     f"« {s['paire'][1]} »",
            "sous": f"cette paire revient {s['compte']} fois ; "
                    f"il reste {len(s['jetons'])} unités",
            "unites": en_mesures(s["jetons"])})

    blocs = grouper_restes(nommer(steps[-1]["jetons"], cible, geles=geles),
                           mot, cible)
    vues.append({
        "titre": "final — les blocs, et leur nom",
        "sous": f"cible {cible} bi-mesures, coût {cout(blocs, cible)} ; "
                "les trop courts ont été recollés à leur voisin",
        "unites": [{"m0": bornes[b["j0"]], "m1": bornes[b["j1"]] - 1,
                    "txt": b["type"],
                    "nom": b["label"] + ("′" if b["prime"] else ""),
                    "queue": bool(b["queue"])} for b in blocs]})

    # et ce que la route rend vraiment, avec la provenance de chaque nom
    from harmonia.server.app import create_app
    r = create_app().test_client().post(
        f"/api/sections/inferer/{cle}",
        json={"humain": [{"label": label, "mesure_debut": b0,
                          "mesure_fin": b1}]}).get_json()
    vues.append({
        "titre": "ce que l'app écrit",
        "sous": "le nom de chaque bloc, et par quelle règle il l'a reçu",
        "unites": [{"m0": s["mesure_debut"] - 1, "m1": s["mesure_fin"] - 1,
                    "txt": s.get("source") or "", "nom": s["label"]}
                   for s in (r.get("sections") or [])]})

    # LA PISTE DE LOUIS, mesurée : et si on cherchait simplement sa brique ?
    occ = cherche_brique(chart, b0 - 1, b1 - 1)
    vues.append({
        "titre": "si on cherchait ta brique",
        "sous": f"ta section {label} glissée le long du morceau : "
                f"{len(occ)} occurrence(s) au-dessus de 0,90, sans "
                "recouvrement — écoute-les pour juger",
        "unites": [{"m0": x, "m1": y, "txt": f"{sc:.2f}", "nom": label,
                    "humain": (x == b0 - 1)} for x, y, sc in occ]})

    # LES DEUX FAÇONS DE REMPLIR LES TROUS. Louis, 2026-09-17 : « il faut
    # impérativement caler ma brique, puis montre ce que ça fait dans les deux
    # cas, remplir avec l'algo actuel vs remplir avec SongFormer, toujours en
    # utilisant ma brique et ses occurrences détectées en priorité ».
    from harmonia.pipeline import analyze
    auto = analyze(SETTINGS.audio_dir / f"{stem}.m4a",
                   title=chart.get("title") or stem, file_key=cle,
                   audio_url=f"/audio/{stem}.m4a")
    vues.append({
        "titre": "trous remplis par l'algo",
        "sous": "tes occurrences sont gelées ; l'algorithme des quatre mots ne "
                "travaille plus que DANS les trous",
        "unites": remplir_par_algo(chart, occ, label)})
    vues.append({
        "titre": "trous remplis par SongFormer",
        "sous": "tes occurrences d'abord ; partout ailleurs, ce que SongFormer "
                "avait trouvé, au lieu de le jeter",
        "unites": remplir_par_songformer(chart, occ, label,
                                         auto.get("sections"))})

    return {
        "cle": cle, "titre": chart.get("title") or stem, "n": n,
        "audio": chart.get("audio_url") or f"/audio/{stem}.m4a",
        "grid": [round(t, 3) for t in grid], "bpb": int(chart.get("bpb") or 4),
        "accords": [" ".join(nom_accord(c) for c in (b or [])) or "·"
                    for b in accords_par_mesure(chart)][:n],
        "mot": mot, "source_mot": src_mot, "trait": (label, b0, b1),
        "vues": vues,
    }


CSS = """
*{box-sizing:border-box}
body{margin:0 auto;padding:14px 12px 40px;background:#faf6ec;color:#2c2820;
 font:15px/1.45 system-ui,-apple-system,sans-serif;max-width:860px}
h1{font-size:19px;margin:0 0 6px}
.note{color:#8a8371;font-size:12.5px}
.lede{background:#fff7df;border:1px solid #e8d9a8;border-radius:10px;
 padding:10px 12px;font-size:14px;margin:10px 0}
.pas{display:flex;gap:6px;flex-wrap:wrap;margin:12px 0 4px;
 position:sticky;top:0;background:#faf6ec;padding:8px 0;z-index:5}
.pas button{border:1px solid #d8cfb4;border-radius:9px;background:#fff;
 padding:7px 10px;font:600 12.5px system-ui;cursor:pointer;min-height:40px;
 color:#2c2820}
.pas button.on{background:#2c2820;border-color:#2c2820;color:#fff}
.tete{font:600 15px system-ui;margin:2px 0}
.stete{color:#8a8371;font-size:12.5px;margin-bottom:8px}
.grille{display:grid;grid-template-columns:repeat(4,1fr);gap:3px}
.mes{position:relative;min-height:56px;border-radius:5px;padding:14px 5px 5px;
 background:#f3edda;cursor:pointer;-webkit-tap-highlight-color:transparent}
.mes .no{position:absolute;top:2px;left:5px;font:600 10px system-ui;
 color:#8a8371;font-variant-numeric:tabular-nums}
.mes .ac{font:600 13.5px ui-monospace,monospace;color:#2c2820;line-height:1.3;
 word-break:break-word}
.mes.deb{border-top-left-radius:5px;border-bottom-left-radius:5px}
.badge{position:absolute;top:-1px;right:3px;font:700 10px system-ui;color:#fff;
 padding:1px 5px;border-radius:0 0 4px 4px;opacity:.72}
.badge.moi{opacity:1;box-shadow:0 0 0 2px #2c2820}
.lg{font:600 12.5px system-ui;color:#8a8371;margin:10px 0 2px}
"""

JS = r"""
const au=document.getElementById('au');let stop=null;
au.addEventListener('timeupdate',()=>{if(stop!=null&&au.currentTime>=stop){au.pause();stop=null;}});
function jouer(t0,t1){
  const src=D.audio;
  const go=()=>{try{au.currentTime=t0;}catch(e){}stop=t1;au.play().catch(()=>{});};
  if(au.getAttribute('src')!==src){au.setAttribute('src',src);
    au.addEventListener('loadedmetadata',go,{once:true});au.load();}
  else go();
}
function teinte(txt){
  // même teinte pour un même contenu, d'une étape à l'autre : c'est ce qui
  // permet de SUIVRE une unité pendant qu'elle grossit.
  const T=["#8a2b2b","#2f5fa8","#1f7a6b","#b06a1f","#7b4ea3","#4a7c3f",
           "#a8336a","#556b2f","#8a6d3b","#3b6f8a"];
  const s=String(txt||"?").toLowerCase();
  if(s==="intro"||s==="outro"||s==="silence"||s==="humain") return "#8a8371";
  let h=0; for(const c of s) h+=c.charCodeAt(0);
  return T[h%T.length];
}
let vue=0;
function peindre(){
  const v=D.vues[vue];
  document.getElementById('tete').textContent=v.titre;
  document.getElementById('stete').textContent=v.sous;
  for(const b of document.querySelectorAll('.pas button'))
    b.classList.toggle('on', +b.dataset.i===vue);
  const cells=document.querySelectorAll('.mes');
  for(const c of cells){ c.style.background="#f3edda"; c.style.boxShadow="";
    const b=c.querySelector('.badge'); if(b) b.remove(); }
  // Une unité de Louis se voit ; une bi-mesure brute se devine. Sans cette
  // différence la page laissait croire que tout était déjà annoté.
  const brut = v.unites.filter(u=>!u.humain).length>8;
  for(const u of v.unites){
    const coul=teinte(u.nom||u.txt);
    const fort = u.humain || !brut;
    for(let m=u.m0;m<=u.m1 && m<cells.length;m++){
      const c=cells[m];
      c.style.background=coul+(fort?"33":"12");
      c.style.boxShadow="inset "+(fort?"4":"2")+"px 0 0 "
                        +(m===u.m0?(fort?coul:coul+"66"):"transparent")
                        +(u.humain?", 0 0 0 2px #2c2820":"");
      c.onclick=()=>jouer(D.grid[u.m0], D.grid[Math.min(u.m1+1,D.grid.length-1)]);
    }
    const tete=cells[u.m0];
    if(tete && (fort || u.nom)){
      const b=document.createElement('div');
      b.className='badge'+(u.humain?' moi':''); b.style.background=coul;
      b.textContent=(u.humain?"toi · ":"")+(u.nom? u.nom+" · ":"")+(u.txt||"");
      tete.appendChild(b);
    }
  }
}
window.addEventListener('DOMContentLoaded',()=>{
  const bar=document.getElementById('pas');
  D.vues.forEach((v,i)=>{
    const b=document.createElement('button');
    b.dataset.i=i; b.textContent=v.titre.split(" — ")[0];
    if(i<2) b.style.borderColor="#4a7c3f";   // les deux vues qui encadrent son geste
    b.onclick=()=>{vue=i;peindre();};
    bar.appendChild(b);
  });
  peindre();
});
"""


def page(d: dict) -> str:
    lab, b0, b1 = d["trait"]
    cells = []
    for i in range(d["n"]):
        cells.append(f"<div class=mes><div class=no>{i+1}</div>"
                     f"<div class=ac>{html.escape(d['accords'][i] if i < len(d['accords']) else '·')}</div></div>")
    corps = (
        f"<h1>{html.escape(d['titre'])}</h1>"
        "<div class=lede>Le chart brut, et l'algorithme posé dessus, étape par "
        "étape. <b>Touche une mesure pour écouter l'unité entière</b> à "
        "laquelle elle appartient — c'est là qu'on entend ce qu'une lettre veut "
        f"dire.<br><br>Ton trait ici : <b>{html.escape(lab)} sur les mesures "
        f"{b0}-{b1}</b>. <b>Il entre AVANT l'étape 0</b> — compare les deux "
        "premières vues : il ne fait pas que se figer, il déplace aussi les "
        "bords des jetons, donc le mot que la machine lit. Les étapes 1 et "
        "suivantes sont les soudures de la machine, qui ne touchent jamais à "
        "ton bloc.<br>"
        "Pendant les étapes, la teinte suit le CONTENU d'une unité : deux "
        "unités de même teinte ont le même contenu, et c'est exactement ce qui "
        "leur fera porter le même nom. Sur les deux dernières vues elle suit le "
        "NOM.</div>"
        "<div class=pas id=pas></div>"
        "<div class=tete id=tete></div><div class=stete id=stete></div>"
        "<div class=grille>" + "".join(cells) + "</div>"
        f"<p class=note>{d['n']} mesures · {d['bpb']} temps par mesure · "
        f"mot calculé sur : {html.escape(d['source_mot'])}</p>")
    data = json.dumps({"audio": d["audio"], "grid": d["grid"],
                       "vues": d["vues"]}, ensure_ascii=False)
    return ("<!-- tools/chart_etapes.py -->"
            "<meta charset=utf-8>"
            "<meta name=viewport content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(d['titre'])} — les étapes sur le chart</title>"
            "<style>" + CSS + "</style><body>" + corps
            + "<audio id=au preload=auto playsinline></audio>"
            "<script>const D=" + data + ";</script><script>" + JS + "</script>")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chart", default="min_B6AHb9W_LkM")
    ap.add_argument("--trait", default="A:7-14", help="LABEL:debut-fin")
    ap.add_argument("--titre", default=None)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args(argv)
    label, plage = a.trait.split(":")
    deb, fin = plage.split("-")
    d = etapes(a.chart, label, int(deb), int(fin))
    if a.titre:
        d["titre"] = a.titre
    out = a.out or (SETTINGS.reports_dir / f"chart_etapes_{a.chart}.html")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page(d), encoding="utf-8")
    print(f"→ {out}\n   {len(d['vues'])} étapes · {d['n']} mesures")
    for v in d["vues"]:
        print(f"   {v['titre']}  ({len(v['unites'])} unités)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
