"""Les trois étages, à voir et à écouter — pas un tableau de chiffres.

    .venv/bin/python scripts/uvw_demo.py
    ->  /plots/uvw_demo.html

Louis, 2026-08-17 : « show me demo always instead of showing numbers ».

Ce que la page montre, dans l'ordre où ça se décide :

  1. L'ALPHABET des bi-mesures, dur contre mou, en deux bandes de couleurs
     alignées sur les mesures. Une couleur = une lettre. On VOIT la version
     molle donner deux couleurs à la même musique.
  2. LA LIAISON : les soudures que l'étage V propose, et la vraie couture du
     morceau posée par-dessus. On VOIT le bloc qui l'enjambe.
  3. Chaque plaque se tape et se joue.

Les chiffres correspondants sont dans `docs/uvw_trois_etages.md`.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))

BASE = "http://100.89.209.63:7772"
SORTIE = HERE / "docs" / "plots" / "uvw_demo.html"
FICHIER = "min_X-yIEMduRXk"
SEUIL_DUR = 0.93
M = 8
K = 6

#: Ce que le morceau fait vraiment, lu à la main : une liaison de 2 mesures
#: avant le couplet 2, et un accord final tout seul. C'est la cible de l'étage
#: de liaison — la page les pose en repères, elle ne les donne pas au calcul.
VRAIES_QUEUES = [(25, 26), (63, 63)]


def alphabet_dur(S, bi):
    lab, k = [], 0
    for i, (a, _b) in enumerate(bi):
        t = None
        for j in range(i):
            c = bi[j][0]
            if float(np.mean([S[c, a], S[min(c + 1, len(S) - 1),
                                         min(a + 1, len(S) - 1)]])) >= SEUIL_DUR:
                t = lab[j]
                break
        if t is None:
            t = k
            k += 1
        lab.append(t)
    return lab


def alphabet_mou(X, m=M):
    from sklearn.decomposition import NMF
    mm = min(m, len(X) - 1)
    f = NMF(n_components=mm, init="nndsvda", max_iter=800, random_state=0)
    return [int(x) for x in np.argmax(f.fit_transform(X), axis=1)]


def liaison(H, k=K):
    """L'étage V : toutes les paires adjacentes, encodées par fente."""
    from sklearn.decomposition import NMF
    Hn = H / np.clip(H.sum(1, keepdims=True), 1e-9, None)
    P = np.array([np.concatenate([Hn[i], Hn[i + 1]]) for i in range(len(Hn) - 1)])
    f = NMF(n_components=k, init="nndsvda", max_iter=800, random_state=0)
    U = f.fit_transform(P)
    V = f.components_
    Un = U / np.clip(U.sum(1, keepdims=True), 1e-9, None)
    res = np.array([np.linalg.norm(P[i] - U[i] @ V) /
                    max(np.linalg.norm(P[i]), 1e-9) for i in range(len(P))])
    force = Un.max(1) * (1 - res)
    pris, blocs = set(), []
    for i in np.argsort(-force):
        if i in pris or (i + 1) in pris:
            continue
        pris.add(int(i)); pris.add(int(i + 1))
        blocs.append({"i": int(i), "force": round(float(force[i]), 3),
                      "lettre": int(np.argmax(Un[i]))})
    blocs.sort(key=lambda x: x["i"])
    return blocs, [i for i in range(len(Hn)) if i not in pris]


def main():
    chart = json.load(open(HERE / f"harmonia_min/state/charts/{FICHIER}.json",
                           encoding="utf-8"))
    grid, n = chart["barGrid"], chart["nBars"]
    stem = Path(chart["audio_url"]).stem
    from harmonia_min import harmonic_sections as HS
    from harmonia_min import musx as MX
    from sklearn.decomposition import NMF

    triad = MX.frame_posteriors(HERE / "docs" / "audio" / f"{stem}.m4a")[0]
    V12 = np.clip(HS.harmonic_vectors(triad, grid), 0, None)
    S = V12 @ V12.T

    bi = [(b, min(b + 2, n)) for b in range(0, n - 1, 2)]
    X = np.array([np.concatenate([V12[a], V12[min(a + 1, b - 1)]]) for a, b in bi])

    dur = alphabet_dur(S, bi)
    mou = alphabet_mou(X)
    f = NMF(n_components=min(M, len(X) - 1), init="nndsvda", max_iter=800,
            random_state=0)
    H = f.fit_transform(X)
    blocs, queues = liaison(H)

    d = {
        "titre": chart.get("title") or stem, "n": n,
        "audio_url": f"{BASE}/audio/{stem}.m4a",
        "retour": f"{BASE}/?open={FICHIER}",
        "uv": f"{BASE}/plots/uv_explication.html",
        "bi": [{"a": a + 1, "b": b, "t": round(float(grid[a]), 3)} for a, b in bi],
        "dur": dur, "mou": mou,
        "blocs": [{"a": bi[x["i"]][0] + 1, "b": bi[x["i"] + 1][1],
                   "t": round(float(grid[bi[x["i"]][0]]), 3),
                   "force": x["force"], "lettre": x["lettre"]} for x in blocs],
        "queues": [{"a": bi[i][0] + 1, "b": bi[i][1],
                    "t": round(float(grid[bi[i][0]]), 3)} for i in queues],
        "vraies_queues": [{"a": a, "b": b,
                           "t": round(float(grid[a - 1]), 3)} for a, b in VRAIES_QUEUES],
    }
    SORTIE.write_text(_G.replace("__D__", json.dumps(
        d, ensure_ascii=False, separators=(",", ":"))), encoding="utf-8")
    print(f"{BASE}/plots/uvw_demo.html")
    print(f"  alphabet dur {len(set(dur))} lettres · mou {len(set(mou))} lettres")
    print(f"  liaison : {len(blocs)} blocs, {len(queues)} queues")
    return 0


_G = r"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Les trois étages, à écouter</title>
<style>
 :root{--papier:#f7f3e9;--encre:#1c1c1c;--pale:#6f6a60;--trait:#ddd5c4;
       --carte:#fffdf7;--accent:#8a2b2b}
 @media(prefers-color-scheme:dark){:root{--papier:#17171a;--encre:#ece8e0;
  --pale:#918c83;--trait:#33323a;--carte:#1f1f24;--accent:#d4735e}}
 *{box-sizing:border-box}
 body{margin:0 auto;max-width:900px;padding:20px 16px 40px;background:var(--papier);
      color:var(--encre);font:15px/1.6 -apple-system,BlinkMacSystemFont,sans-serif}
 a{color:var(--accent)}
 h1{font:italic 600 25px/1.2 Georgia,serif;margin:0 0 4px}
 h2{font:600 11px sans-serif;letter-spacing:.09em;text-transform:uppercase;
    color:var(--pale);margin:28px 0 6px}
 .lede{font:italic 14.5px/1.65 Georgia,serif;color:var(--pale);margin:0 0 14px}
 .carte{background:var(--carte);border:1px solid var(--trait);border-radius:14px;
        padding:14px;margin:8px 0;overflow-x:auto}
 .nom{font:600 13px sans-serif;margin-bottom:1px}
 .note{font:italic 12.5px Georgia,serif;color:var(--pale);margin-bottom:5px}
 .bande{display:flex;gap:1px;min-width:680px;margin-bottom:12px}
 .c{flex:1 1 0;height:30px;border-radius:4px;cursor:pointer;display:flex;
    align-items:center;justify-content:center;color:#fff;font:700 10.5px sans-serif}
 .c:hover{outline:2px solid var(--encre);outline-offset:-2px}
 .lien{display:flex;gap:1px;min-width:680px;margin-bottom:4px}
 .lien div{height:16px;border-radius:4px;cursor:pointer;
           display:flex;align-items:center;justify-content:center;
           font:700 9px sans-serif;color:#fff}
 .regle{display:flex;gap:1px;min-width:680px;font:10px ui-monospace,Menlo,monospace;
        color:var(--pale)}
 .regle span{flex:1 1 0;text-align:center}
 .verdict{border-left:3px solid var(--accent);padding-left:12px;margin:12px 0;
          font-size:14.5px}
 button{font:600 14px sans-serif;border:1px solid var(--trait);background:var(--carte);
        color:var(--encre);border-radius:11px;padding:11px 16px;cursor:pointer;
        min-height:44px}
 button.p{background:var(--accent);color:#fff;border-color:transparent;min-width:104px}
 #bar{display:flex;align-items:center;gap:12px;margin:14px 0;position:sticky;
      bottom:0;background:var(--papier);padding:10px 0}
 #ou{font:600 13px ui-monospace,Menlo,monospace;color:var(--pale)}
</style></head><body>
<script>window.D = __D__;</script>
<h1 id="t"></h1>
<p class="lede">Les trois étages de <b>X ≈ U·V·W</b>, à écouter. Tape n'importe
quelle plaque : le son y va.</p>

<h2>1. L'alphabet des bi-mesures — dur contre mou</h2>
<div class="carte">
  <div class="nom">DUR — l'égalité chord-tone (ce qui est livré)</div>
  <div class="note">une couleur = une lettre ; deux bi-mesures identiques ont la même</div>
  <div class="bande" id="dur"></div>
  <div class="nom">MOU — la factorisation (l'étage W)</div>
  <div class="note">la même chose, apprise par NMF au lieu d'être comparée</div>
  <div class="bande" id="mou"></div>
  <div class="regle" id="r1"></div>
</div>
<div class="verdict" id="v1"></div>

<h2>2. La liaison — l'étage V</h2>
<div class="carte">
  <div class="nom">ce que V soude en blocs de 4 mesures</div>
  <div class="note">la hauteur de la plaque = la force de la soudure</div>
  <div class="lien" id="blocs"></div>
  <div class="nom" style="margin-top:10px">ce que le morceau fait vraiment</div>
  <div class="note">la seule queue interne : la liaison avant le couplet 2 (l'accord
    final, mesure 63, tombe hors de la grille de bi-mesures — elle s'arrête à 62)</div>
  <div class="lien" id="vraies"></div>
  <div class="regle" id="r2"></div>
</div>
<div class="verdict" id="v2"></div>

<div id="bar"><button class="p" id="jouer">Écouter</button><span id="ou"></span></div>
<p id="liens"></p>

<script>
(function(){
 "use strict";
 var D = window.D, n = D.n, NB = D.bi.length;
 var PAL = ["#8a2b2b","#2f6f8f","#7a6320","#4a7a4a","#6b4a7a","#a85a2a",
            "#3d6d6d","#8a4a6a","#5a6b2a","#7a3a5a"];
 document.getElementById("t").textContent = D.titre + " — les trois étages";

 function bande(id, lab){
   var h = document.getElementById(id);
   D.bi.forEach(function(b, i){
     var e = document.createElement("div");
     e.className = "c";
     e.style.background = PAL[lab[i] % PAL.length];
     e.textContent = String.fromCharCode(97 + (lab[i] % 26));
     e.title = "mes. " + b.a + "–" + b.b;
     e.onclick = function(){ vers(b.t); };
     h.appendChild(e);
   });
 }
 bande("dur", D.dur); bande("mou", D.mou);

 function regle(id){
   var r = document.getElementById(id);
   D.bi.forEach(function(b, i){
     var s = document.createElement("span");
     s.textContent = (i % 4 === 0) ? b.a : "";
     r.appendChild(s);
   });
 }
 regle("r1"); regle("r2");

 /* Les DEUX sens, comptes sur CE morceau — la page doit dire ce qu'elle
    montre, pas la moyenne d'un corpus qu'on ne voit pas ici. */
 var separe = 0, unis = 0, fond = 0, distincts = 0;
 for (var i = 0; i < NB; i++) for (var j = i+1; j < NB; j++){
   if (D.dur[i] === D.dur[j]){ unis++; if (D.mou[i] !== D.mou[j]) separe++; }
   else { distincts++; if (D.mou[i] === D.mou[j]) fond++; }
 }
 var nd = (new Set(D.dur)).size, nm = (new Set(D.mou)).size;
 document.getElementById("v1").innerHTML =
   "<b>Sur CE morceau la molle est plus GROSSIÈRE, pas plus fine</b> — " + nm +
   " lettres contre " + nd + ". Elle sépare " + separe + " des " + unis +
   " paires que la dure réunit, mais elle en <b>fond " + fond + " sur " + distincts +
   "</b> que la dure distingue : regarde le milieu, où elle peint d'une seule " +
   "couleur des bi-mesures que la dure sépare. " +
   "<br><br>C'est la même cause dans les deux sens : la NMF cherche à " +
   "RECONSTRUIRE le morceau avec m composantes, pas à dire « ces deux-là sont " +
   "la même musique ». Elle dépense ses composantes là où ça fait baisser le " +
   "résidu, pas là où ça se répète. Sur les 18 morceaux annotés ça coûte du " +
   "rappel (0,49 → 0,34) : le détail est dans <i>docs/uvw_trois_etages.md</i>, " +
   "parce qu'un morceau ne prouve pas une moyenne.";

 var W = 100 / (NB - 1);
 var hb = document.getElementById("blocs");
 var occupe = {};
 D.blocs.forEach(function(b){ occupe[b.a] = b; });
 var pos = 0;
 D.bi.forEach(function(b, i){
   var e = document.createElement("div");
   e.style.flex = "1 1 0";
   var bl = occupe[b.a];
   if (bl){ e.style.background = PAL[bl.lettre % PAL.length];
            e.style.height = (8 + bl.force * 16).toFixed(0) + "px";
            e.textContent = bl.force.toFixed(2); e.title = "soudure mes. " + bl.a + "–" + bl.b;
            e.onclick = function(){ vers(bl.t); }; }
   else { e.style.background = "transparent"; }
   hb.appendChild(e);
 });
 var hv = document.getElementById("vraies");
 D.bi.forEach(function(b){
   var e = document.createElement("div");
   e.style.flex = "1 1 0";
   var q = D.vraies_queues.filter(function(x){ return x.a <= b.a && b.a <= x.b; })[0];
   if (q){ e.style.background = "var(--accent)"; e.textContent = "queue";
           e.title = "vraie queue mes. " + q.a + "–" + q.b;
           e.onclick = function(){ vers(q.t); }; }
   else { e.style.background = "var(--trait)"; e.style.opacity = ".45"; }
   hv.appendChild(e);
 });
 /* Une queue de 2 mesures ratee d'une bi-mesure est RATEE : on rapporte donc
    l'ecart, jamais un "trouve" a la tolerance large. La premiere version de
    cette page annoncait « il en retrouve 2 » avec +-2 mesures de tolerance,
    c'est-a-dire la bi-mesure d'a cote -- exactement l'erreur qu'on mesure. */
 var appariees = D.vraies_queues.map(function(v){
   var best = null;
   D.queues.forEach(function(q){
     var d = Math.abs(q.a - v.a);
     if (best === null || d < best.d) best = {q: q, d: d};
   });
   return {v: v, q: best && best.q, d: best ? best.d : 999};
 });
 var exactes = appariees.filter(function(x){ return x.d === 0; });
 var enjambe = D.blocs.filter(function(b){
   return D.vraies_queues.some(function(v){ return b.a < v.a && v.a < b.b; }); });
 var txt = "<b>Les deux lignes ne se superposent pas.</b> V propose " +
   D.blocs.length + " soudures et " + D.queues.length + " queues. Sur les " +
   D.vraies_queues.length + " vraies queues du morceau, il en pose <b>" +
   (exactes.length || "aucune") + "</b> au bon endroit : " +
   appariees.map(function(x){
     return "la vraie mes. " + x.v.a + (x.q ? " → sa plus proche est mes. " + x.q.a +
       " (" + x.d + " mesures à côté)" : " → rien");
   }).join(", ") + ". Rater une queue de deux mesures d'une bi-mesure, c'est la rater.";
 if (enjambe.length){
   var e = enjambe[0];
   txt += " Et une soudure <b>ENJAMBE</b> la liaison : mes. " + e.a + "–" + e.b +
     ", notée " + e.force.toFixed(2).replace(".", ",") + " — tape-la, tu entends la fin du tag " +
     "collée au début du couplet 2.";
 }
 txt += "<br><br>La cause est la même qu'au-dessus. « Ces deux bi-mesures vont " +
   "ensemble » est un fait de <b>COMPTAGE</b> — cette paire revient-elle " +
   "ailleurs ? — et reconstruire ne compte rien. C'est exactement ce que BPE " +
   "fait déjà (<i>bpe_lab</i>, <i>phrases4</i>), et bien.";
 document.getElementById("v2").innerHTML = txt;

 document.getElementById("liens").innerHTML =
   '<a href="' + D.uv + '">le X ≈ UV à deux étages</a> · ' +
   '<a href="' + D.retour + '">le chart dans l\'app</a>';

 /* son : timeupdate, jamais rAF ; blob, jamais Range */
 var el=null, lien=null, joue=false;
 if (window.fetch) fetch(D.audio_url).then(function(r){return r.ok?r.blob():null;})
  .then(function(b){ if(b){ lien=URL.createObjectURL(b); if(el&&!joue) el.src=lien; }})
  ["catch"](function(){});
 function media(){
   if (el) return el;
   el=new Audio(); el.preload="auto"; el.playsInline=true;
   try{ el.setAttribute("playsinline",""); el.setAttribute("webkit-playsinline",""); }catch(e){}
   el.src = lien || D.audio_url;
   el.addEventListener("timeupdate", function(){
     var t=el.currentTime;
     document.getElementById("ou").textContent =
       Math.floor(t/60)+":"+String(Math.floor(t%60)).padStart(2,"0"); });
   document.body.appendChild(el); return el;
 }
 function vers(t){ var a=media(); try{ a.currentTime=t; }catch(e){}
   if(!joue){ a.play(); joue=true; document.getElementById("jouer").textContent="Pause"; } }
 document.getElementById("jouer").onclick=function(){
   var a=media();
   if(joue){ a.pause(); joue=false; this.textContent="Écouter"; }
   else { a.play(); joue=true; this.textContent="Pause"; } };
})();
</script></body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
