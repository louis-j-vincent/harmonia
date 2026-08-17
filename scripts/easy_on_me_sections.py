"""Easy On Me : quatre lectures de la forme, à écouter.

    .venv/bin/python scripts/easy_on_me_sections.py
    ->  /plots/easy_on_me_sections.html

Louis, 2026-08-17 : « étudie les sections que tu ferais toi en tant que LLM, et
vois comment tu peux corriger mon modèle […] j'imagine que la voix est un cue,
et les sections sont un peu mal branlées, le D est mal utilisé […] réduction
matricielle X = UV, avec V les sections et U la répartition ».

La page ne donne AUCUN score. Elle met quatre découpages du même morceau l'un
sous l'autre, chaque mesure cliquable, pour qu'il tranche à l'oreille :

    le modèle, avant   ce que l'app servait — le D fourre-tout
    le modèle, après   le même détecteur, orphelins séparés
    la factorisation   X ≈ UV sur des blocs de 4 mesures (argmax de U)
    ma lecture         ce que j'écris, moi, comme sections

…plus deux bandes de mesure : le chant (sa piste demucs, en secondes chantées
par mesure) et la ressemblance harmonique à la mesure d'avant. Ce sont les deux
indices dont on dispose, et la page montre lequel voit quoi.

Les chiffres du banc sont dans le commit et dans docs/known_issues.md ; ici on
ne met que ce qui s'écoute.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

STEM = "X-yIEMduRXk"
FICHIER = "min_X-yIEMduRXk"
BASE = "http://100.89.209.63:7772"
SORTIE = HERE / "docs" / "plots" / "easy_on_me_sections.html"

#: La grille de blocs, avec la COUTURE de 2 mesures à la 25 (index 24). Elle
#: n'est pas devinée : elle est LUE sur le morceau (la liaison vers le couplet
#: 2 fait deux mesures, pas quatre) et c'est précisément ce que la
#: factorisation ne sait pas trouver toute seule — voir la note de la page.
BORNES = [0, 4, 8, 12, 16, 20, 24, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 63]

#: Ma lecture, en mesures 1-indexées, fin incluse. `queue` = ni une section ni
#: une intro : un raccord qui ne se compte pas dans la forme (le mot est de
#: Louis, et c'est le résidu de son « = approximatif »).
MA_LECTURE = [
    ("intro", 1, 4),
    ("A", 5, 12),      ("B", 13, 20),   ("C", 21, 24),   ("queue", 25, 26),
    ("A", 27, 34),     ("D", 35, 42),   ("E", 43, 50),
    ("B", 51, 58),     ("C", 59, 62),   ("queue", 63, 63),
]

#: Ce que l'app servait AVANT le correctif (relevé sur le chart le 2026-08-17).
AVANT = [("intro", 1, 4), ("A", 5, 12), ("B", 13, 20), ("C", 21, 24),
         ("D", 25, 26), ("A", 27, 34), ("E", 35, 38), ("E", 39, 42),
         ("D", 43, 50), ("B", 51, 58), ("C", 59, 62), ("D", 63, 63)]

COULEURS = ["#8a2b2b", "#2f6f8f", "#7a6320", "#4a7a4a", "#6b4a7a", "#a85a2a",
            "#3d6d6d", "#8a4a6a"]
GRIS = "#b9b2a4"


def facteurs(V12, n):
    """(assignation par bloc, pureté) — X ≈ UV sur des blocs de 4 mesures."""
    from sklearn.decomposition import NMF
    X = np.clip(V12, 0, None)
    blocs = [(BORNES[i], BORNES[i + 1]) for i in range(len(BORNES) - 1)]
    L = 4

    def plat(a, b):
        v = np.zeros(L * 12)
        for i in range(L):
            v[i * 12:(i + 1) * 12] = X[min(a + i, b - 1)] if a + i < b else X[b - 1]
        return v

    Xb = np.array([plat(a, b) for a, b in blocs])
    m = NMF(n_components=6, init="nndsvda", max_iter=800, random_state=0)
    U = m.fit_transform(Xb)
    Un = U / np.clip(U.sum(1, keepdims=True), 1e-9, None)
    LET = "VWXYZT"
    out = [(LET[int(np.argmax(Un[i]))], a + 1, b, float(Un[i].max()))
           for i, (a, b) in enumerate(blocs)]
    return out, float(np.mean(Un.max(1)))


def apres_correctif(grid, triad, audio):
    from harmonia_min import voice_sections as VS
    segs = VS.detect_sections(grid, triad, bars=None, audio=audio)
    return [(s["label"], s["b0"] + 1, s["b1"] + 1) for s in segs]


def main() -> int:
    chart = json.load(open(HERE / f"harmonia_min/state/charts/{FICHIER}.json",
                           encoding="utf-8"))
    grid, n = chart["barGrid"], chart["nBars"]
    audio = HERE / "docs" / "audio" / f"{STEM}.m4a"

    from harmonia_min import harmonic_sections as HS
    from harmonia_min import musx as MX
    triad = MX.frame_posteriors(audio)[0]
    V12 = HS.harmonic_vectors(triad, grid)
    S = V12 @ V12.T

    sys.path.insert(0, str(HERE / "scripts"))
    import vocal_anchor as VA
    import vocal_melody as VM
    voc = VA.separate_vocals(audio)
    tt, ff, vv, rr = VM.track_f0(voc)
    notes, _ = VM.clean(VM.melody_notes(tt, ff, vv, rr)[0])
    chante = np.zeros(n)
    for t0, d, _m in notes:
        b = int(np.searchsorted(grid, t0) - 1)
        if 0 <= b < n:
            chante[b] += d

    nmf, purete = facteurs(V12, n)
    apres = apres_correctif(grid, triad, audio)

    # les accords, mesure par mesure, pour la ligne du haut
    NOMS = ["C", "D♭", "D", "E♭", "E", "F", "G♭", "G", "A♭", "A", "B♭", "B"]
    par = {}
    for s in chart["sections"]:
        bars = s.get("bars") or []
        if not bars:
            continue
        for b0, b1 in s.get("barRanges") or []:
            for b in range(b0, min(b1, n - 1) + 1):
                par[b] = bars[(b - b0) % len(bars)]

    def nom(c):
        if c.get("nc"):
            return "N.C."
        t = NOMS[c["root"]] + (c.get("q") or "")
        if c.get("bass", -1) >= 0 and c["bass"] != c["root"]:
            t += "/" + NOMS[c["bass"]]
        return t

    accords = [" ".join(nom(c) for c in par.get(b, [])) or "—" for b in range(n)]
    voisin = [1.0] + [float(S[b, b - 1]) for b in range(1, n)]

    d = {
        "titre": chart.get("title") or STEM,
        "n": n, "grid": [round(x, 3) for x in grid],
        "audio_url": f"{BASE}/audio/{STEM}.m4a",
        "retour": f"{BASE}/?open={FICHIER}",
        "ssm": f"{BASE}/ssm/{FICHIER}",
        "accords": accords,
        "chante": [round(float(x), 2) for x in chante],
        "voisin": [round(v, 3) for v in voisin],
        "purete": round(purete, 3),
        "pistes": [
            {"nom": "le modèle, AVANT", "note": "un seul D pour trois musiques",
             "segs": [[l, a, b] for l, a, b in AVANT]},
            {"nom": "le modèle, APRÈS", "note": "orphelins séparés (le correctif)",
             "segs": [[l, a, b] for l, a, b in apres]},
            {"nom": "X ≈ UV", "note": "argmax de U, blocs de 4 mesures",
             "segs": [[l, a, b] for l, a, b, _p in nmf]},
            {"nom": "ma lecture", "note": "ce que j'écris, moi",
             "segs": [[l, a, b] for l, a, b in MA_LECTURE]},
        ],
        "couleurs": COULEURS, "gris": GRIS,
    }
    SORTIE.write_text(
        _GABARIT.replace("__D__", json.dumps(d, ensure_ascii=False,
                                             separators=(",", ":"))),
        encoding="utf-8")
    print(f"{BASE}/plots/easy_on_me_sections.html")
    print(f"  pureté de U : {purete:.3f}  ({len(nmf)} blocs, k=6)")
    return 0


_GABARIT = r"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Easy On Me — quatre lectures de la forme</title>
<style>
 :root{--papier:#f7f3e9;--encre:#1c1c1c;--pale:#6f6a60;--trait:#ddd5c4;
       --carte:#fffdf7;--accent:#8a2b2b}
 @media(prefers-color-scheme:dark){:root{--papier:#17171a;--encre:#ece8e0;
   --pale:#918c83;--trait:#33323a;--carte:#1f1f24;--accent:#d4735e}}
 *{box-sizing:border-box}
 body{margin:0;padding:20px 16px calc(30px + env(safe-area-inset-bottom));
      background:var(--papier);color:var(--encre);max-width:900px;
      margin-inline:auto;
      font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
 a{color:var(--accent)}
 h1{font:italic 600 25px/1.2 Georgia,serif;margin:0 0 4px}
 h2{font:600 11px sans-serif;letter-spacing:.09em;text-transform:uppercase;
    color:var(--pale);margin:26px 0 8px}
 .sous{color:var(--pale);font-size:13.5px;margin:0 0 6px}
 .lede{font:italic 14.5px/1.65 Georgia,serif;color:var(--pale);margin:0 0 18px}
 .carte{background:var(--carte);border:1px solid var(--trait);
        border-radius:14px;padding:12px;overflow-x:auto}
 .piste{margin-bottom:12px}
 .nom{font:600 12.5px sans-serif;margin-bottom:1px}
 .note{font:italic 12px Georgia,serif;color:var(--pale);margin-bottom:4px}
 .bande{display:flex;gap:1px;min-width:640px}
 .seg{flex:0 0 auto;height:30px;border-radius:5px;display:flex;
      align-items:center;justify-content:center;color:#fff;cursor:pointer;
      font:700 11.5px sans-serif;overflow:hidden;white-space:nowrap}
 .seg:hover{outline:2px solid var(--encre);outline-offset:-2px}
 .jauge{display:flex;gap:1px;min-width:640px;align-items:flex-end;height:34px}
 .jauge div{flex:1 1 0;background:var(--accent);border-radius:2px 2px 0 0;
            min-height:1px;cursor:pointer}
 .regle{display:flex;gap:1px;min-width:640px;font:10px ui-monospace,Menlo,monospace;
        color:var(--pale);margin-top:3px}
 .regle span{flex:1 1 0;text-align:center;overflow:hidden}
 #tete{position:relative;height:3px;background:var(--trait);min-width:640px;
       margin-top:6px;border-radius:2px}
 #tete i{position:absolute;top:-3px;width:3px;height:9px;background:var(--accent);
         border-radius:2px}
 #bar{display:flex;align-items:center;gap:12px;margin:16px 0 4px;
      position:sticky;bottom:0;background:var(--papier);padding:10px 0}
 button{font:600 14px sans-serif;border:1px solid var(--trait);
        background:var(--carte);color:var(--encre);border-radius:11px;
        padding:11px 16px;cursor:pointer;min-height:44px}
 button.p{background:var(--accent);color:#fff;border-color:transparent;min-width:104px}
 #ou{font:600 13px ui-monospace,Menlo,monospace;color:var(--pale)}
 #alerte{display:none;margin-top:8px;padding:10px;border-radius:10px;
         background:var(--accent);color:#fff;font-size:13.5px}
 .dit{font:14px/1.6 -apple-system,sans-serif;margin:8px 0 0}
 .dit b{color:var(--accent)}
 ul{margin:6px 0 0;padding-left:20px} li{margin:5px 0}
</style></head><body>
<script>window.D = __D__;</script>

<h1 id="t"></h1>
<p class="sous" id="s"></p>
<p class="lede">Quatre découpages du même morceau, l'un sous l'autre. Tape
n'importe quelle plaque : le son y va. C'est à l'oreille que ça se tranche —
la page ne donne aucun score.</p>

<h2>Les quatre lectures</h2>
<div class="carte" id="pistes"></div>

<h2>Les deux indices</h2>
<div class="carte">
  <div class="piste"><div class="nom">le chant</div>
    <div class="note">secondes chantées par mesure (piste vocale séparée)</div>
    <div class="jauge" id="chant"></div></div>
  <div class="piste"><div class="nom">l'harmonie</div>
    <div class="note">ressemblance de chaque mesure à la précédente — bas = rupture</div>
    <div class="jauge" id="harm"></div></div>
  <div class="regle" id="regle"></div>
  <div id="tete"><i></i></div>
</div>

<div id="bar"><button class="p" id="jouer">Écouter</button><span id="ou"></span></div>
<div id="alerte"></div>

<h2>Ce que ça dit</h2>
<div id="dit"></div>

<p class="dit" style="margin-top:22px" id="liens"></p>

<script>
(function(){
 "use strict";
 var D = window.D, n = D.n, G = D.grid;
 document.getElementById("t").textContent = D.titre + " — quatre lectures de la forme";
 document.getElementById("s").textContent = n + " mesures · si♭ majeur · la grille vient du chart";

 /* une couleur par lettre, stable d'une piste a l'autre : c'est ce qui rend
    la comparaison lisible — la meme lettre doit avoir la meme couleur partout */
 var teinte = {};
 D.pistes.forEach(function(p){ p.segs.forEach(function(s){
   var l = s[0];
   if (l === "intro" || l === "outro" || l === "queue") return;
   if (!(l in teinte)) teinte[l] = D.couleurs[Object.keys(teinte).length % D.couleurs.length];
 }); });
 function coul(l){ return (l==="intro"||l==="outro"||l==="queue") ? D.gris : teinte[l]; }

 var hote = document.getElementById("pistes");
 D.pistes.forEach(function(p){
   var w = document.createElement("div"); w.className = "piste";
   var a = document.createElement("div"); a.className = "nom"; a.textContent = p.nom;
   var b = document.createElement("div"); b.className = "note"; b.textContent = p.note;
   var c = document.createElement("div"); c.className = "bande";
   p.segs.forEach(function(s){
     var e = document.createElement("div"); e.className = "seg";
     e.style.flexBasis = ((s[2]-s[1]+1)/n*100).toFixed(3) + "%";
     e.style.background = coul(s[0]);
     e.textContent = s[0];
     e.title = s[0] + " · mesures " + s[1] + "–" + s[2] + " (" + (s[2]-s[1]+1) + ")";
     e.onclick = function(){ vers(G[s[1]-1], true); };
     c.appendChild(e);
   });
   w.appendChild(a); w.appendChild(b); w.appendChild(c); hote.appendChild(w);
 });

 function jauge(id, vals, lo, hi){
   var h = document.getElementById(id);
   vals.forEach(function(v, i){
     var e = document.createElement("div");
     e.style.height = Math.max(1, (v-lo)/(hi-lo)*32).toFixed(1) + "px";
     e.style.opacity = 0.35 + 0.65*Math.min(1, Math.max(0,(v-lo)/(hi-lo)));
     e.title = "mesure " + (i+1) + " · " + v;
     e.onclick = function(){ vers(G[i], true); };
     h.appendChild(e);
   });
 }
 jauge("chant", D.chante, 0, Math.max.apply(null, D.chante));
 jauge("harm", D.voisin, 0, 1);
 var r = document.getElementById("regle");
 for (var i = 0; i < n; i++){
   var s = document.createElement("span");
   s.textContent = ((i+1) % 8 === 1) ? (i+1) : "";
   r.appendChild(s);
 }

 var texte = [
  ["Le D fourre-tout n'était pas la fusion de lettres, mais l'écriture des orphelins.",
   "Les mesures que le détecteur n'attribue à personne portaient toutes le numéro <b>-1</b>, et le re-lettrage traduisait cet unique entier en une seule lettre. D couvrait donc la liaison de 2 mesures (25-26), le pont de 8 (43-50) et l'accord final (63) — trois musiques, trois longueurs, une lettre. Chaque orphelin a maintenant la sienne."],
  ["La voix est bien l'indice, mais elle ne suffit pas seule.",
   "L'intro (mes. 1-4) et le début du couplet (mes. 5-8) sont <b>harmoniquement identiques</b> — 0,98 à 1,00 mesure par mesure. Rien dans l'harmonie ne peut les séparer ; seul le silence vocal des mesures 1-3 le fait. Mais le chant se tait aussi à la mesure 47, en plein milieu du pont : un silence vocal marque une couture, il ne la prouve pas."],
  ["X ≈ UV retrouve la forme — une fois qu'on lui donne les frontières.",
   "Sur des blocs de 4 mesures, k=6 : le refrain reçoit la même rangée aux mesures 13-20 et 51-58, le tag la même aux 21-24 et 59-62, et la liaison de 2 mesures sort <b>sa propre rangée à 0,98</b> — tes « queues » tombent toutes seules. Mais la couture de 2 mesures à la mesure 25, je la lui ai <b>donnée</b> : la factorisation résout le NOMMAGE, pas le DÉCOUPAGE. C'est la bonne nouvelle et la limite en même temps."],
  ["Là où le modèle reste faible : il compare en diagonale, à longueur égale.",
   "Deux occurrences d'une même section ne se reconnaissent que si elles ont la même longueur et le même alignement (<code>merge_letters</code> compare sur min(L₁,L₂), en diagonale). Sur Yesterday, deux passages que tu annotes B font 6 et 8 mesures : le modèle ne les rapproche pas. Une factorisation les mettrait sur la même rangée de V sans se soucier de l'alignement — c'est le vrai gain à aller chercher, et c'est exactement ton intuition."],
  ["Ce qui reste à trancher par toi.",
   "Le modèle écrit E aux mesures 35-38 <i>puis</i> 39-42, deux occurrences de 4. J'écris une seule section de 8 avec une boucle interne ×2. Écoute les deux : si tu entends une seule montée, c'est ma lecture ; si tu entends deux fois la même phrase, c'est la sienne."]
 ];
 var dh = document.getElementById("dit");
 texte.forEach(function(x){
   var w = document.createElement("div"); w.style.marginBottom = "14px";
   var a = document.createElement("div");
   a.style.cssText = "font:600 14px sans-serif;margin-bottom:2px";
   a.textContent = x[0];
   var b = document.createElement("p"); b.className = "dit"; b.innerHTML = x[1];
   b.style.margin = "0"; w.appendChild(a); w.appendChild(b); dh.appendChild(w);
 });
 var L = document.getElementById("liens");
 L.innerHTML = '<a href="' + D.ssm + '">la matrice SSM cliquable</a> · ' +
               '<a href="' + D.retour + '">le chart dans l\'app</a>';

 /* ── le son : timeupdate, jamais rAF ; blob, jamais Range (pièges iPhone) ── */
 var el = null, lien = null, joue = false, garde = null;
 function dit(m){ var a=document.getElementById("alerte");
   a.textContent=m; a.style.display=m?"block":"none"; }
 if (window.fetch) fetch(D.audio_url).then(function(r){return r.ok?r.blob():null;})
   .then(function(b){ if(!b) return; lien=URL.createObjectURL(b);
     if(el && !joue) el.src=lien; })["catch"](function(){});
 function media(){
   if (el) return el;
   el = new Audio(); el.preload="auto"; el.playsInline=true;
   try{ el.setAttribute("playsinline",""); el.setAttribute("webkit-playsinline",""); }catch(e){}
   el.src = lien || D.audio_url;
   el.addEventListener("timeupdate", tic);
   el.addEventListener("seeked", tic);
   el.addEventListener("error", function(){ dit("le fichier audio n'a pas pu être lu"); });
   document.body.appendChild(el);
   return el;
 }
 function mmss(t){ t=Math.max(0,Math.floor(t||0));
   return Math.floor(t/60)+":"+String(t%60).padStart(2,"0"); }
 function mesure(t){ var k=0; for (var i=0;i<n;i++) if (G[i]<=t) k=i; return k; }
 var der = -1;
 function tic(){
   if (!el) return;
   var b = mesure(el.currentTime);
   if (b === der) return;                       // on n'écrit que si ça bouge
   der = b;
   document.getElementById("ou").textContent =
     mmss(el.currentTime) + " · mesure " + (b+1) + " · " + D.accords[b];
   document.querySelector("#tete i").style.left = (b/n*100).toFixed(2) + "%";
 }
 function veille(k){
   clearTimeout(garde); if (!el) return;
   var d0 = el.currentTime, m = k||0;
   garde = setTimeout(function(){
     if (!joue || !el) return;
     if (el.currentTime > d0 + 0.05) return;
     if (lien && el.src !== lien){ var t=el.currentTime; el.src=lien;
       try{ el.currentTime=t; }catch(e){} el.play(); veille(m+1); return; }
     if (m < 2){ el.play(); veille(m+1); return; }
     dit("le son ne démarre pas sur cet appareil — l'horloge est restée à " + mmss(el.currentTime));
   }, 1500);
 }
 function vers(t, lance){
   var a = media();
   try{ a.currentTime = Math.max(0, t); }catch(e){}
   der = -1; tic();
   if (lance && !joue) bascule(true); else if (joue) veille(0);
 }
 function bascule(force){
   var a = media();
   if (joue && force !== true){ a.pause(); joue=false; clearTimeout(garde); }
   else { dit("");
     var p = a.play();
     if (p && p["catch"]) p["catch"](function(e){ joue=false;
       document.getElementById("jouer").textContent="Écouter";
       dit("le navigateur a refusé la lecture ("+(e&&e.name||e)+")"); });
     joue = true; veille(0); }
   document.getElementById("jouer").textContent = joue ? "Pause" : "Écouter";
 }
 document.getElementById("jouer").onclick = function(){ bascule(); };
 document.addEventListener("keydown", function(e){
   if (e.code === "Space"){ e.preventDefault(); bascule(); } });
})();
</script></body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
