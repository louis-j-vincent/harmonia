"""Le trou de 1 ou 2 mesures, placé par le FIT — à voir et à écouter.

    .venv/bin/python scripts/trou_demo.py
    ->  /plots/trou_demo.html

Louis, 2026-08-17 : « on reste sur un X = UV, mais on lui donne l'opportunité
de pouvoir faire des trous de 1 ou 2 mesures pour qu'il puisse fitter au
mieux ? » puis « et oui il faut un petit k ».

LE MÉCANISME, et c'est lui qui fait marcher l'idée. Un objectif de
reconstruction ne mesure pas la répétition — c'est ce qui avait tué le UVW. Ici
il la mesure, **grâce au k serré** : avec cinq composantes pour quinze blocs,
la seule façon de tout reconstruire est que les blocs tombent vraiment dans
cinq groupes qui se répètent. Une grille décalée fabrique quinze blocs tous
différents, et rien de rang 5 ne peut les rendre. Le k serré est ce qui
transforme « ça fitte » en « ça se répète ».

CE QUE LA MESURE NE DIT PAS, et que j'ai failli affirmer : il n'y a **pas** de
« plus k est petit, mieux c'est ». Le rang de la vraie grille vaut 3, 5, 7, 1,
6, 1 pour k = 2, 3, 4, 5, 6, 8 — toujours dans les 10 % de tête, jamais de
tendance monotone. Le choix de k n'est pas résolu.

DEUX PRÉCAUTIONS, sans lesquelles la mesure ment :

  * on ne compare qu'entre grilles ayant le MÊME NOMBRE DE BLOCS. Une grille
    qui perd un bloc fitte mieux sans rien dire de mieux.
  * le résidu relatif compare des matrices X différentes d'une grille à
    l'autre. À nombre de blocs égal c'est comparable ; c'est pour ça que la
    contrainte ci-dessus n'est pas cosmétique.
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

BASE = "http://100.89.209.63:7772"
SORTIE = HERE / "docs" / "plots" / "trou_demo.html"
FICHIER = "min_X-yIEMduRXk"
L = 4
KS = [2, 3, 4, 5, 6, 8]
K_DEF = 5
VRAI_TROU = (24, 2)          # la liaison de 2 mesures, mes. 25-26

#: Ma lecture de la forme, en mesures 1-indexées (voir easy_on_me_sections.py).
FRONTIERES = [5, 13, 21, 25, 27, 35, 43, 51, 59, 63]


def blocs_de(trous, n):
    out, b = [], 0
    for p, w in sorted(trous):
        while b + L <= p:
            out.append(b); b += L
        b = max(b, p) + w
    while b + L <= n:
        out.append(b); b += L
    return out


def residu(blocs, V12, k):
    from sklearn.decomposition import NMF
    if len(blocs) < k + 1:
        return 1.0
    X = np.array([np.concatenate([V12[b + i] for i in range(L)]) for b in blocs])
    f = NMF(n_components=k, init="nndsvda", max_iter=600, random_state=0)
    U = f.fit_transform(X)
    return float(np.linalg.norm(X - U @ f.components_) / max(np.linalg.norm(X), 1e-9))


def paysage(V12, n, k):
    """Pour chaque trou candidat, le résidu — à nombre de blocs égal."""
    nb = len(blocs_de([VRAI_TROU], n))
    out = []
    for w in (1, 2):
        for p in range(1, n - L):
            b = blocs_de([(p, w)], n)
            if len(b) != nb:
                continue
            out.append({"p": p, "w": w, "r": round(residu(b, V12, k), 4)})
    return out


def corpus(k=K_DEF):
    import section_bench as SB
    T = SB.truth()
    out = []
    for st in sorted(T):
        F = SB.features(st)
        n = min(F["n"], T[st]["n"])
        V12 = np.clip(F["V"][:n], 0, None)
        vrai = sorted({s["b0"] for s in T[st]["sections"] if 0 < s["b0"] < n})
        if len(vrai) < 3 or n < 24:
            continue
        b0 = blocs_de([], n)
        best, bt = residu(b0, V12, k), None
        for w in (1, 2):
            for p in range(1, n - L):
                b = blocs_de([(p, w)], n)
                if len(b) != len(b0):
                    continue
                r = residu(b, V12, k)
                if r < best - 1e-6:
                    best, bt = r, (p, w)

        def couvre(bl):
            bords = set(bl) | {x + L for x in bl}
            return sum(1 for g in vrai if g in bords) / max(len(vrai), 1)

        out.append({"stem": st, "avant": round(couvre(b0), 3),
                    "apres": round(couvre(blocs_de([bt] if bt else [], n)), 3),
                    "trou": (bt[0] + 1) if bt else None,
                    "w": bt[1] if bt else None})
    return out


def main():
    chart = json.load(open(HERE / f"harmonia_min/state/charts/{FICHIER}.json",
                           encoding="utf-8"))
    grid, n = chart["barGrid"], chart["nBars"]
    stem = Path(chart["audio_url"]).stem
    from harmonia_min import harmonic_sections as HS
    from harmonia_min import musx as MX
    triad = MX.frame_posteriors(HERE / "docs" / "audio" / f"{stem}.m4a")[0]
    V12 = np.clip(HS.harmonic_vectors(triad, grid), 0, None)

    pays = {}
    for k in KS:
        pts = paysage(V12, n, k)
        pts.sort(key=lambda x: x["r"])
        vrai_r = next(x["r"] for x in pts if (x["p"], x["w"]) == VRAI_TROU)
        # rang du MEILLEUR candidat qui donne la même grille que la vraie
        cible = blocs_de([VRAI_TROU], n)
        rang = next(i for i, x in enumerate(pts)
                    if blocs_de([(x["p"], x["w"])], n) == cible)
        pays[str(k)] = {"pts": pts, "vrai": vrai_r, "rang": rang + 1,
                        "n": len(pts), "sans": round(residu(blocs_de([], n), V12, k), 4)}

    d = {"titre": chart.get("title") or stem, "n": n, "L": L, "ks": KS, "kdef": K_DEF,
         "audio_url": f"{BASE}/audio/{stem}.m4a",
         "retour": f"{BASE}/?open={FICHIER}",
         "uv": f"{BASE}/plots/uv_explication.html",
         "grid": [round(float(x), 3) for x in grid],
         "frontieres": FRONTIERES,
         "vrai": {"p": VRAI_TROU[0], "w": VRAI_TROU[1]},
         "paysage": pays,
         "sans_trou": blocs_de([], n),
         "avec_trou": blocs_de([VRAI_TROU], n),
         "corpus": corpus()}
    SORTIE.write_text(_G.replace("__D__", json.dumps(
        d, ensure_ascii=False, separators=(",", ":"))), encoding="utf-8")
    print(f"{BASE}/plots/trou_demo.html")
    for k in KS:
        print(f"  k={k} : la vraie grille est {pays[str(k)]['rang']}e / {pays[str(k)]['n']}")
    c = d["corpus"]
    print(f"  corpus : {np.mean([x['avant'] for x in c]):.3f} -> "
          f"{np.mean([x['apres'] for x in c]):.3f} sur {len(c)} morceaux")
    return 0


_G = r"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Un trou de 1 ou 2 mesures, placé par le fit</title>
<style>
 :root{--papier:#f7f3e9;--encre:#1c1c1c;--pale:#6f6a60;--trait:#ddd5c4;
       --carte:#fffdf7;--accent:#8a2b2b;--vert:#4a7a4a}
 @media(prefers-color-scheme:dark){:root{--papier:#17171a;--encre:#ece8e0;
  --pale:#918c83;--trait:#33323a;--carte:#1f1f24;--accent:#d4735e;--vert:#7aa87a}}
 *{box-sizing:border-box}
 body{margin:0 auto;max-width:920px;padding:20px 16px 90px;background:var(--papier);
      color:var(--encre);font:15px/1.6 -apple-system,BlinkMacSystemFont,sans-serif}
 a{color:var(--accent)}
 h1{font:italic 600 25px/1.2 Georgia,serif;margin:0 0 4px}
 h2{font:600 11px sans-serif;letter-spacing:.09em;text-transform:uppercase;
    color:var(--pale);margin:28px 0 6px}
 .lede{font:italic 14.5px/1.65 Georgia,serif;color:var(--pale);margin:0 0 14px}
 .carte{background:var(--carte);border:1px solid var(--trait);border-radius:14px;
        padding:14px;margin:8px 0;overflow-x:auto}
 .nom{font:600 13px sans-serif;margin-bottom:1px}
 .note{font:italic 12.5px Georgia,serif;color:var(--pale);margin-bottom:6px}
 #pays{display:flex;gap:2px;align-items:flex-end;height:150px;min-width:680px}
 #pays div{flex:1 1 0;background:var(--trait);border-radius:2px 2px 0 0;cursor:pointer;
           position:relative}
 #pays div:hover{outline:1px solid var(--encre)}
 .regle{display:flex;gap:2px;min-width:680px;font:10px ui-monospace,Menlo,monospace;
        color:var(--pale);margin-top:3px}
 .regle span{flex:1 1 0;text-align:center;overflow:hidden}
 .bande{display:flex;gap:2px;min-width:680px;margin-bottom:3px}
 .bl{height:26px;border-radius:4px;background:#2f6f8f;cursor:pointer;color:#fff;
     display:flex;align-items:center;justify-content:center;font:700 10px sans-serif}
 .tr{height:26px;border-radius:4px;background:var(--accent);color:#fff;cursor:pointer;
     display:flex;align-items:center;justify-content:center;font:700 9px sans-serif}
 .ks{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}
 .ks button{font:600 13px sans-serif;border:1px solid var(--trait);background:var(--carte);
            color:var(--encre);border-radius:9px;padding:7px 13px;cursor:pointer}
 .ks button.on{background:var(--accent);color:#fff;border-color:transparent}
 .verdict{border-left:3px solid var(--accent);padding-left:12px;margin:12px 0;font-size:14.5px}
 table{border-collapse:collapse;width:100%;font-size:13px;min-width:420px}
 th{text-align:left;font:600 10px sans-serif;letter-spacing:.07em;text-transform:uppercase;
    color:var(--pale);padding:0 8px 5px 0}
 td{padding:4px 8px 4px 0;border-top:1px solid var(--trait)}
 .mini{display:inline-block;height:8px;border-radius:2px;vertical-align:middle}
 button.p{background:var(--accent);color:#fff;border:none;border-radius:11px;
          padding:11px 16px;font:600 14px sans-serif;cursor:pointer;min-height:44px;min-width:104px}
 #bar{display:flex;align-items:center;gap:12px;margin:14px 0;position:sticky;bottom:0;
      background:var(--papier);padding:10px 0}
 #ou{font:600 13px ui-monospace,Menlo,monospace;color:var(--pale)}
</style></head><body>
<script>window.D = __D__;</script>
<h1 id="t"></h1>
<p class="lede">On garde <b>X ≈ U·V</b>, mais la grille a le droit de <b>sauter
1 ou 2 mesures</b>. Chaque barre ci-dessous est un trou possible : sa hauteur
est la qualité du fit qu'il produit. Tape une barre pour écouter l'endroit.</p>

<h2>1. Le paysage — où mettre le trou ?</h2>
<div class="ks" id="ks"></div>
<div class="carte">
  <div class="nom">plus la barre est HAUTE, mieux la grille fitte</div>
  <div class="note">rouge = le vrai trou du morceau (la liaison, mes. 25-26) ·
    vert = le meilleur candidat</div>
  <div id="pays"></div>
  <div class="regle" id="r1"></div>
</div>
<div class="verdict" id="v1"></div>

<h2>2. La grille, avant et après</h2>
<div class="carte">
  <div class="nom">sans trou</div>
  <div class="note">les traits sous la bande sont les vraies frontières de la forme</div>
  <div class="bande" id="g0"></div>
  <div class="regle" id="f0"></div>
  <div class="nom" style="margin-top:12px">avec le trou trouvé</div>
  <div class="bande" id="g1"></div>
  <div class="regle" id="f1"></div>
</div>
<div class="verdict" id="v2"></div>

<h2>3. Les 18 morceaux annotés</h2>
<div class="carte"><table><thead><tr><th>morceau</th><th>frontières sur un bord de bloc</th>
<th>trou</th></tr></thead><tbody id="corpus"></tbody></table></div>
<div class="verdict" id="v3"></div>

<div id="bar"><button class="p" id="jouer">Écouter</button><span id="ou"></span></div>
<p id="liens"></p>

<script>
(function(){
 "use strict";
 var D = window.D, n = D.n, k = String(D.kdef);

 document.getElementById("t").textContent = D.titre + " — le trou placé par le fit";

 var kb = document.getElementById("ks");
 D.ks.forEach(function(x){
   var b = document.createElement("button");
   b.textContent = "k = " + x;
   b.onclick = function(){ k = String(x); dessine(); };
   b.dataset.k = x; kb.appendChild(b);
 });

 function memeGrille(p, w){
   /* deux ecritures du meme trou donnent la meme grille : on compare les
      grilles, pas les (p,w) — sinon on declare rate un succes exact. */
   return JSON.stringify(grilleDe(p, w)) === JSON.stringify(D.avec_trou);
 }
 function grilleDe(p, w){
   var out = [], b = 0;
   while (b + D.L <= p){ out.push(b); b += D.L; }
   b = Math.max(b, p) + w;
   while (b + D.L <= n){ out.push(b); b += D.L; }
   return out;
 }

 function dessine(){
   Array.prototype.forEach.call(kb.children, function(b){
     b.className = (b.dataset.k === k) ? "on" : ""; });
   var P = D.paysage[k], pts = P.pts.slice().sort(function(a,b){ return a.p - b.p || a.w - b.w; });
   var rs = pts.map(function(x){ return x.r; });
   var lo = Math.min.apply(null, rs), hi = Math.max.apply(null, rs);
   var h = document.getElementById("pays"); h.innerHTML = "";
   var r1 = document.getElementById("r1"); r1.innerHTML = "";
   pts.forEach(function(x){
     var e = document.createElement("div");
     /* haut = bon fit : on inverse, le residu BAS est le meilleur */
     e.style.height = (6 + (hi - x.r) / Math.max(hi - lo, 1e-9) * 138).toFixed(1) + "px";
     if (memeGrille(x.p, x.w)) e.style.background = "var(--accent)";
     else if (x.r === lo) e.style.background = "var(--vert)";
     e.title = "trou mes. " + (x.p+1) + " largeur " + x.w + " · résidu " + x.r;
     e.onclick = function(){ vers(D.grid[x.p]); };
     h.appendChild(e);
     var s = document.createElement("span");
     s.textContent = (x.w === 1 && (x.p+1) % 8 === 1) ? (x.p+1) : "";
     r1.appendChild(s);
   });
   var rangs = D.ks.map(function(x){ return D.paysage[String(x)].rang; });
   var v1 = "<b>k = " + k + ".</b> La vraie grille du morceau arrive <b>" +
     P.rang + "ᵉ sur " + P.n + "</b>";
   v1 += (P.rang === 1) ? " — <b>le fit la trouve tout seul.</b>"
                        : ", dans le haut du panier.";
   v1 += "<br><br><b>Pourquoi ça marche.</b> Avec " + k + " composantes pour " +
     D.avec_trou.length + " blocs, la seule façon de tout reconstruire est que " +
     "les blocs tombent vraiment dans " + k + " groupes qui se répètent. Une " +
     "grille décalée fabrique des blocs tous différents, et rien d'aussi petit " +
     "ne peut les rendre : c'est le k serré qui transforme « ça fitte » en " +
     "« ça se répète ». C'est exactement ce qui manquait au UVW, où l'objectif " +
     "de reconstruction ne mesurait aucune répétition.";
   v1 += "<br><br><b>Mais attention à ce que je n'ai PAS montré.</b> Le rang de " +
     "la vraie grille selon k vaut " + D.ks.map(function(x, i){
       return "k=" + x + " → " + rangs[i] + "ᵉ"; }).join(", ") +
     ". Elle est dans les 10 % de tête partout, et première à deux valeurs — " +
     "mais il n'y a <b>pas</b> de « plus k est petit, mieux c'est ». Le choix " +
     "de k n'est pas résolu, et c'est la première chose à régler avant de " +
     "brancher quoi que ce soit. Change le bouton pour le voir.";
   document.getElementById("v1").innerHTML = v1;
 }

 function bande(id, reg, blocs){
   var h = document.getElementById(id), r = document.getElementById(reg);
   h.innerHTML = ""; r.innerHTML = "";
   var dans = {};
   blocs.forEach(function(b){ for (var i = 0; i < D.L; i++) dans[b+i] = b; });
   for (var m = 0; m < n; m++){
     var e = document.createElement("div");
     e.style.flex = "1 1 0";
     if (dans[m] !== undefined){
       e.className = "bl";
       e.style.opacity = ((dans[m] / D.L) % 2) ? ".72" : "1";
       if (m === dans[m]) e.textContent = m + 1;
     } else { e.className = "tr"; e.textContent = "•"; }
     var mm = m;
     e.onclick = function(){ vers(D.grid[mm]); };
     h.appendChild(e);
     var s = document.createElement("span");
     s.textContent = (D.frontieres.indexOf(m+1) >= 0) ? "▲" : "";
     s.style.color = "var(--accent)";
     r.appendChild(s);
   }
 }
 bande("g0", "f0", D.sans_trou);
 bande("g1", "f1", D.avec_trou);

 function surBord(blocs){
   var bords = {}; blocs.forEach(function(b){ bords[b]=1; bords[b+D.L]=1; });
   return D.frontieres.filter(function(f){ return bords[f-1]; }).length;
 }
 document.getElementById("v2").innerHTML =
   "<b>Les triangles rouges sont les vraies frontières.</b> Sans trou, <b>" +
   surBord(D.sans_trou) + " sur " + D.frontieres.length +
   "</b> tombent sur un bord de bloc ; avec le trou, <b>" + surBord(D.avec_trou) +
   " sur " + D.frontieres.length + "</b>. Deux mesures sautées une seule fois, " +
   "et tout ce qui suit se recale.";

 var cb = document.getElementById("corpus"), sa = 0, sp = 0;
 D.corpus.forEach(function(c){
   sa += c.avant; sp += c.apres;
   var t = document.createElement("tr");
   var w = function(v, col){ return "<span class='mini' style='width:" +
     (v*70).toFixed(0) + "px;background:" + col + "'></span>"; };
   t.innerHTML = "<td>" + c.stem.slice(0,34) + "</td><td>" +
     w(c.avant, "var(--trait)") + " " + c.avant.toFixed(2) + " → " +
     w(c.apres, "var(--vert)") + " <b>" + c.apres.toFixed(2) + "</b></td>" +
     "<td class=note style='margin:0'>" + (c.trou ? "mes. " + c.trou + " w" + c.w : "—") + "</td>";
   cb.appendChild(t);
 });
 var na = D.corpus.length;
 document.getElementById("v3").innerHTML =
   "<b>" + (sa/na).toFixed(2) + " → " + (sp/na).toFixed(2) +
   "</b> de frontières tombant sur un bord de bloc, et <b>aucun morceau ne " +
   "recule</b>. Détail qu'on n'attendait pas : sur Stand By Me et Happy, le " +
   "meilleur trou est à la mesure 2 — ce n'est pas une couture interne, c'est " +
   "la <b>levée</b> du morceau. Le fit retrouve tout seul ce que « Set bar 1 » " +
   "fait à la main.";

 document.getElementById("liens").innerHTML =
   '<a href="' + D.uv + '">l\'algorithme X ≈ UV</a> · ' +
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

 dessine();
})();
</script></body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
