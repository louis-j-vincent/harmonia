"""X = 1 × nombre de mesures : la NMF CONVOLUTIVE, sans aucun découpage.

    .venv/bin/python scripts/nmfd_demo.py
    ->  /plots/nmfd_demo.html

Louis, 2026-08-17 : « pourquoi ne pas directement faire cette décomposition
matricielle sur le NNLS ? et il faudrait réussir à se débarrasser de la
rigidité du découpage en 4 mesures, donc il faudrait peut-être une autre
matrice, et que X au final soit une matrice de dimension 1 × nombre de mesures ».

LES DEUX DEMANDES TOMBENT SUR LE MÊME ALGORITHME. Si X est le morceau entier —
une colonne par mesure, aucun bloc — alors la matrice qui manque est celle qui
dit OÙ poser chaque motif. C'est la NMF convolutive (Smaragdis 2004) :

    X (F × n)  ≈  Σ_τ  W_τ · H décalée de τ

    X   le morceau, une colonne par mesure       AUCUN découpage
    W   k motifs, chacun long de T mesures       -> les sections
    H   (k × n) où chaque motif DÉMARRE          -> le placement, libre

La rigidité des blocs de 4 disparaît vraiment : H peut poser un motif à
n'importe quelle mesure, pas seulement sur un treillis. Les mesures que
personne ne couvre sont les queues, sans qu'on ait à les prévoir.

DEUX ENTRÉES COMPARÉES, c'est l'autre question :
  * les postérieures d'accords dépliées en hauteurs (12) — ce qu'on utilisait ;
  * le NNLS brut (24 : basse + harmonie) — ce qui SONNE vraiment.

CE QUI EST OUVERT, ET QU'IL FAUT LIRE AVANT DE CROIRE LA PAGE : l'effet de la
PARCIMONIE sur H n'est pas mesuré. La pénalité L1 est multiplicative, et le
détecteur de pics normalise chaque composante par son propre maximum — ce qui
l'annule exactement. Il faut un critère de pic à l'échelle absolue avant de
pouvoir dire quoi que ce soit là-dessus. Tout ce que la page montre est à
parcimonie nulle.
"""
from __future__ import annotations

import base64
import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))

BASE = "http://100.89.209.63:7772"
SORTIE = HERE / "docs" / "plots" / "nmfd_demo.html"
FICHIER = "min_X-yIEMduRXk"
K = 4
TS = [4, 8]
#: Ma lecture de la forme, en index de mesure 0-indexé.
FRONT = [4, 12, 20, 24, 26, 34, 42, 50, 58, 62]


def nmfd(X, k, T, iters=250, seed=0):
    """Smaragdis 2004, divergence KL, mises à jour multiplicatives.

    Les colonnes de W sont renormalisées à chaque tour : sans ça W enfle et H
    maigrit sans que rien n'ait changé, et toute lecture de H devient
    dépendante de l'itération où on s'arrête.
    """
    rng = np.random.default_rng(seed)
    F, N = X.shape
    W = rng.random((T, F, k)) + 0.1
    H = rng.random((k, N)) + 0.1
    V = np.maximum(X, 1e-9)
    un = np.ones((F, N))

    def shift(M, t):
        if t == 0:
            return M
        out = np.zeros_like(M)
        out[:, t:] = M[:, :-t]
        return out

    def lam():
        L = np.zeros((F, N))
        for t in range(T):
            L += W[t] @ shift(H, t)
        return np.maximum(L, 1e-9)

    for _ in range(iters):
        R = V / lam()
        Hn = np.zeros_like(H)
        Hd = np.zeros_like(H)
        for t in range(T):
            Rs = np.zeros_like(R)
            Rs[:, :N - t] = R[:, t:] if t else R
            Hn += W[t].T @ Rs
            Hd += W[t].T @ un
        H *= Hn / np.maximum(Hd, 1e-9)
        R = V / lam()
        for t in range(T):
            Ht = shift(H, t)
            W[t] *= (R @ Ht.T) / np.maximum(un @ Ht.T, 1e-9)
        nrm = np.sqrt(np.sum(W ** 2, axis=(0, 1))) + 1e-9
        W /= nrm[None, None, :]
        H *= nrm[:, None]
    return W, H


def poses_de(H, seuil=0.45):
    out = []
    for j in range(H.shape[0]):
        h = H[j] / max(H[j].max(), 1e-9)
        p = sorted(np.where(h > seuil)[0].tolist())
        p = [x for i, x in enumerate(p) if i == 0 or x - p[i - 1] > 1]
        out.append(p)
    return out


def oct64(M, hi=None):
    M = np.asarray(M, float)
    hi = float(M.max()) if hi is None else hi
    q = np.clip(M / max(hi, 1e-9), 0, 1)
    return base64.b64encode((q * 255 + 0.5).astype(np.uint8).tobytes()).decode()


def main():
    chart = json.load(open(HERE / f"harmonia_min/state/charts/{FICHIER}.json",
                           encoding="utf-8"))
    grid, n = chart["barGrid"], chart["nBars"]
    stem = Path(chart["audio_url"]).stem
    audio = HERE / "docs" / "audio" / f"{stem}.m4a"
    from harmonia_min import harmonic_sections as HS
    from harmonia_min import musx as MX
    from harmonia_min.nnls_features import extract_bothchroma

    Xa = np.clip(HS.harmonic_vectors(MX.frame_posteriors(audio)[0], grid), 0, None).T
    arr, times = extract_bothchroma(audio)
    Xb = np.zeros((24, n))
    for b in range(n):
        m = (times >= grid[b]) & (times < grid[b + 1])
        if m.any():
            Xb[:, b] = arr[m].mean(0)
    Xb = np.clip(Xb, 0, None)
    Xb /= max(Xb.max(), 1e-9)

    entrees = {}
    for nom, X in (("accords", Xa), ("nnls", Xb)):
        combos = {}
        for T in TS:
            W, H = nmfd(X, K, T)
            po = poses_de(H)
            plat = sorted({p for l in po for p in l})
            ok = sum(1 for p in plat if any(abs(p - f) <= 1 for f in FRONT))
            okf = sum(1 for f in FRONT if any(abs(p - f) <= 1 for p in plat))
            combos[str(T)] = {
                "H": oct64(H / max(H.max(), 1e-9), 1.0),
                "W": [oct64(W[:, :, j].T) for j in range(K)],
                "poses": [[int(x) for x in l] for l in po],
                "prec": round(ok / max(len(plat), 1), 2),
                "rappel": round(okf / len(FRONT), 2),
            }
        entrees[nom] = {"X": oct64(X), "F": X.shape[0], "k": combos,
                        "nom": "postérieures d'accords (12)" if nom == "accords"
                               else "NNLS brut (24 : basse + harmonie)"}

    d = {"titre": chart.get("title") or stem, "n": n, "K": K, "ts": TS,
         "front": FRONT, "entrees": entrees,
         "grid": [round(float(x), 3) for x in grid],
         "audio_url": f"{BASE}/audio/{stem}.m4a",
         "retour": f"{BASE}/?open={FICHIER}",
         "mat": f"{BASE}/plots/uv_matrices.html"}
    SORTIE.write_text(_G.replace("__D__", json.dumps(
        d, ensure_ascii=False, separators=(",", ":"))), encoding="utf-8")
    print(f"{BASE}/plots/nmfd_demo.html")
    for nom in entrees:
        for T in TS:
            c = entrees[nom]["k"][str(T)]
            print(f"  {nom:8} T={T} : précision {c['prec']} rappel {c['rappel']}")
    return 0


_G = r"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>X = 1 × nombre de mesures</title>
<style>
 :root{--papier:#f7f3e9;--encre:#1c1c1c;--pale:#6f6a60;--trait:#ddd5c4;
       --carte:#fffdf7;--accent:#8a2b2b}
 @media(prefers-color-scheme:dark){:root{--papier:#17171a;--encre:#ece8e0;
  --pale:#918c83;--trait:#33323a;--carte:#1f1f24;--accent:#d4735e}}
 *{box-sizing:border-box}
 body{margin:0 auto;max-width:1000px;padding:20px 16px 90px;background:var(--papier);
      color:var(--encre);font:15px/1.6 -apple-system,BlinkMacSystemFont,sans-serif}
 a{color:var(--accent)}
 h1{font:italic 600 25px/1.2 Georgia,serif;margin:0 0 4px}
 .lede{font:italic 14.5px/1.6 Georgia,serif;color:var(--pale);margin:0 0 12px}
 .eq{font:600 15px ui-monospace,Menlo,monospace;text-align:center;margin:14px 0;
     padding:12px;background:var(--carte);border:1px solid var(--trait);border-radius:12px}
 .ctl{display:flex;gap:18px;flex-wrap:wrap;align-items:center;margin:14px 0;
      position:sticky;top:0;background:var(--papier);padding:10px 0;z-index:5;
      border-bottom:1px solid var(--trait)}
 .grp{display:flex;gap:5px;align-items:center}
 .grp b{font:600 10px sans-serif;letter-spacing:.08em;text-transform:uppercase;
        color:var(--pale);margin-right:3px}
 .ctl button{font:600 13px sans-serif;border:1px solid var(--trait);background:var(--carte);
   color:var(--encre);border-radius:9px;padding:7px 12px;cursor:pointer;min-height:38px}
 .ctl button.on{background:var(--accent);color:#fff;border-color:transparent}
 .tit{font:600 14px sans-serif;margin:20px 0 1px}
 .exp{font:italic 12.5px/1.5 Georgia,serif;color:var(--pale);margin-bottom:6px}
 .wrap{display:flex;gap:9px;align-items:flex-start}
 .lab{flex:0 0 auto;font:10px ui-monospace,Menlo,monospace;color:var(--pale);text-align:right}
 .lab div{height:20px;line-height:20px;white-space:nowrap}
 canvas{display:block;border:1px solid var(--trait);border-radius:6px;
        image-rendering:pixelated;cursor:crosshair}
 .regle{display:flex;font:10px ui-monospace,Menlo,monospace;color:var(--pale);margin-top:2px}
 .regle span{flex:1 1 0;text-align:center}
 .verdict{border-left:3px solid var(--accent);padding-left:12px;margin:14px 0;font-size:14.5px}
 .ouvert{border-left:3px solid var(--pale);padding-left:12px;margin:14px 0;
         font-size:14px;color:var(--pale)}
 #bar{position:fixed;bottom:0;left:0;right:0;background:var(--papier);
      border-top:1px solid var(--trait);padding:10px 16px;display:flex;gap:12px;
      align-items:center;z-index:6}
 #bar button{background:var(--accent);color:#fff;border:none;border-radius:11px;
   padding:11px 16px;font:600 14px sans-serif;cursor:pointer;min-width:104px;min-height:44px}
 #ou{font:600 13px ui-monospace,Menlo,monospace;color:var(--pale)}
</style></head><body>
<script>window.D = __D__;</script>
<h1 id="t"></h1>
<p class="lede">Plus aucun découpage. X est le morceau entier — <b>une colonne
par mesure</b> — et c'est <b>H</b> qui dit où chaque motif se pose, à la mesure
près. Clique n'importe où pour écouter.</p>
<div class="eq">X (traits × mesures) &nbsp;≈&nbsp; Σ<sub>τ</sub> &nbsp;W<sub>τ</sub> &nbsp;·&nbsp; H décalée de τ</div>

<div class="ctl">
  <div class="grp"><b>entrée</b><span id="es"></span></div>
  <div class="grp"><b>longueur des motifs</b><span id="ts"></span></div>
</div>

<div class="tit">X — le morceau, une colonne par mesure</div>
<div class="exp" id="expx"></div>
<div class="wrap"><div class="lab" id="lx"></div><div style="flex:1">
  <canvas id="cx"></canvas><div class="regle" id="rx"></div></div></div>

<div class="tit">H — où chaque motif démarre</div>
<div class="exp">une ligne par motif · les traits rouges sous la bande sont les
vraies frontières de la forme · rien n'oblige H à tomber dessus</div>
<div class="wrap"><div class="lab" id="lh"></div><div style="flex:1">
  <canvas id="ch"></canvas><div class="regle" id="rh"></div></div></div>

<div class="tit">W — les motifs appris</div>
<div class="exp">chacun est une section candidate, longue de T mesures</div>
<div class="wrap" id="ws"></div>

<div class="verdict" id="v"></div>
<div class="ouvert" id="o"></div>

<div id="bar"><button id="jouer">Écouter</button><span id="ou"></span>
  <a href="#" id="lien" style="margin-left:auto;font-size:13px">← les matrices X U V</a></div>

<script>
(function(){
 "use strict";
 var D = window.D, e = "accords", T = "8", n = D.n;
 document.getElementById("t").textContent = D.titre + " — X = 1 × " + n + " mesures";
 document.getElementById("lien").href = D.mat;

 function bytes(b){ var s = atob(b), a = new Uint8Array(s.length);
   for (var i = 0; i < s.length; i++) a[i] = s.charCodeAt(i); return a; }
 function couleur(v){
   var st = [[251,247,236],[207,224,234],[132,179,207],[61,127,166],[27,74,107],[13,36,55]];
   var x = v/255*(st.length-1), i = Math.min(st.length-2, Math.floor(x)), f = x-i;
   return "rgb(" + st[i].map(function(c,j){ return Math.round(c+f*(st[i+1][j]-c)); }).join(",") + ")";
 }
 function peins(cv, arr, rows, cols, ch){
   var x = cv.getContext("2d");
   cv.width = cols*ch; cv.height = rows*20;
   cv.style.width = "100%"; cv.style.height = (rows*20) + "px";
   for (var i = 0; i < rows; i++) for (var j = 0; j < cols; j++){
     x.fillStyle = couleur(arr[i*cols+j]); x.fillRect(j*ch, i*20, ch, 20);
   }
 }
 function regle(id, cols){
   var h = document.getElementById(id); h.innerHTML = "";
   for (var i = 0; i < cols; i++){
     var s = document.createElement("span");
     s.textContent = (D.front.indexOf(i) >= 0) ? "▲" : ((i % 8 === 0) ? (i+1) : "");
     if (D.front.indexOf(i) >= 0) s.style.color = "var(--accent)";
     h.appendChild(s);
   }
 }
 function labs(id, items){
   var h = document.getElementById(id); h.innerHTML = "";
   items.forEach(function(t){ var d = document.createElement("div");
     d.textContent = t; h.appendChild(d); });
 }

 function dessine(){
   var E = D.entrees[e], C = E.k[T];
   Array.prototype.forEach.call(document.querySelectorAll("#es button"), function(b){
     b.className = (b.dataset.v === e) ? "on" : ""; });
   Array.prototype.forEach.call(document.querySelectorAll("#ts button"), function(b){
     b.className = (b.dataset.v === T) ? "on" : ""; });
   document.getElementById("expx").textContent =
     E.nom + " — " + E.F + " lignes, " + n + " colonnes";

   peins(document.getElementById("cx"), bytes(E.X), E.F, n, 11);
   labs("lx", Array.from({length: E.F}, function(_, i){
     return (E.F === 12) ? ["C","D♭","D","E♭","E","F","G♭","G","A♭","A","B♭","B"][i]
                         : (i < 12 ? "b " : "h ") + ["C","D♭","D","E♭","E","F","G♭","G","A♭","A","B♭","B"][i%12]; }));
   regle("rx", n);

   peins(document.getElementById("ch"), bytes(C.H), D.K, n, 11);
   labs("lh", C.poses.map(function(_p, j){ return "motif " + (j+1); }));
   regle("rh", n);

   var hw = document.getElementById("ws"); hw.innerHTML = "";
   C.W.forEach(function(w, j){
     var box = document.createElement("div");
     box.style.cssText = "flex:0 0 auto;margin-right:14px";
     var t = document.createElement("div");
     t.style.cssText = "font:600 11px sans-serif;color:var(--pale);margin-bottom:3px";
     t.textContent = "motif " + (j+1) + " · " + C.poses[j].length + " pose(s)";
     var c = document.createElement("canvas");
     box.appendChild(t); box.appendChild(c); hw.appendChild(box);
     peins(c, bytes(w), parseInt(T,10), E.F, 9);
     c.style.width = "auto";
   });

   document.getElementById("v").innerHTML =
     "<b>Regarde H.</b> Les motifs se posent où ils veulent, à la mesure près — " +
     "aucun treillis. Avec cette entrée et des motifs de " + T + " mesures, " +
     "<b>" + Math.round(C.prec*100) + " %</b> des poses tombent sur une vraie " +
     "frontière et <b>" + Math.round(C.rappel*100) + " %</b> des frontières sont " +
     "trouvées. Le témoin bête (« une frontière toutes les 4 mesures ») est à " +
     "25 % de précision.<br><br><b>Et la réponse à ta question :</b> bascule " +
     "entrée sur NNLS. C'est <b>moins bon</b>, pas mieux — " +
     Math.round(D.entrees.nnls.k[T].prec*100) + " % contre " +
     Math.round(D.entrees.accords.k[T].prec*100) + " % de précision. Le chroma " +
     "brut porte le timbre, le voicing et l'énergie ; la postérieure d'accords " +
     "a déjà jeté tout ça. Ce que je te présentais comme une limite — « c'est " +
     "déjà une interprétation » — est ici un débruitage.";

   document.getElementById("o").innerHTML =
     "<b>Ce qui reste ouvert, et qu'il faut savoir avant de me croire.</b> " +
     "L'effet de la PARCIMONIE sur H n'est pas mesuré : la pénalité est " +
     "multiplicative et mon détecteur de pics normalise chaque motif par son " +
     "propre maximum, ce qui l'annule exactement. J'ai fait le balayage, il " +
     "rend des résultats identiques à la troisième décimale — c'est le signe " +
     "d'un test invalide, pas d'un effet nul. Tout ce qui est montré ici est à " +
     "parcimonie zéro, et « peu de poses, longues » reste la piste à tester " +
     "avec un critère de pic à l'échelle absolue.";
 }

 var he = document.getElementById("es");
 [["accords","accords"],["nnls","NNLS"]].forEach(function(p){
   var b = document.createElement("button");
   b.textContent = p[1]; b.dataset.v = p[0];
   b.onclick = function(){ e = p[0]; dessine(); }; he.appendChild(b); });
 var ht = document.getElementById("ts");
 D.ts.forEach(function(x){
   var b = document.createElement("button");
   b.textContent = x + " mesures"; b.dataset.v = String(x);
   b.onclick = function(){ T = String(x); dessine(); }; ht.appendChild(b); });

 [document.getElementById("cx"), document.getElementById("ch")].forEach(function(c){
   c.addEventListener("click", function(ev){
     var r = c.getBoundingClientRect();
     var b = Math.max(0, Math.min(n-1, Math.floor((ev.clientX-r.left)/r.width*n)));
     vers(D.grid[b]); });
 });

 var el=null, lien=null, joue=false;
 if (window.fetch) fetch(D.audio_url).then(function(r){return r.ok?r.blob():null;})
  .then(function(b){ if(b){ lien=URL.createObjectURL(b); if(el&&!joue) el.src=lien; }})
  ["catch"](function(){});
 function media(){
   if (el) return el;
   el=new Audio(); el.preload="auto"; el.playsInline=true;
   try{ el.setAttribute("playsinline",""); el.setAttribute("webkit-playsinline",""); }catch(e){}
   el.src = lien || D.audio_url;
   el.addEventListener("timeupdate", function(){ var t=el.currentTime;
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
