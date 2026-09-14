"""Voir X, U et V — en fonction de k, avec et sans le trou.

    .venv/bin/python scripts/uv_matrices.py
    ->  /plots/uv_matrices.html

Louis, 2026-08-17 : « la page n'est pas claire, j'aimerais bien voir les
matrices X U et V en fonction de K ».

La page précédente (`trou_demo.html`) montrait un paysage de résidus — un
chiffre déguisé en barres. Celle-ci montre les matrices elles-mêmes :

    X        ce qu'on donne     (blocs × 48)
    U        où sont les sections (blocs × k)
    V        les sections        (k × 48)
    U·V      ce qu'on récupère  (blocs × 48)   ← à comparer à X, à l'œil

Deux boutons : **k**, et **avec / sans le trou**. Tout se redessine.

Les 48 colonnes sont 4 mesures × 12 hauteurs : les trois traits verticaux
séparent les mesures. Sans eux la matrice est illisible.
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
SORTIE = HERE / "docs" / "plots" / "uv_matrices.html"
FICHIER = "min_X-yIEMduRXk"
L = 4
KS = [2, 3, 4, 5, 6, 8]
TROU = (24, 2)                       # sauter les mesures 25-26

NOMS = ["C", "D♭", "D", "E♭", "E", "F", "G♭", "G", "A♭", "A", "B♭", "B"]
QUAL = [("", [0, 4, 7]), ("m", [0, 3, 7]), ("7", [0, 4, 7, 10]),
        ("m7", [0, 3, 7, 10]), ("maj7", [0, 4, 7, 11])]


def nomme(v12) -> str:
    v = np.asarray(v12, float)
    if v.sum() <= 1e-9:
        return "—"
    v = v / np.linalg.norm(v)
    best, nom = -1.0, "—"
    for r in range(12):
        for suf, iv in QUAL:
            t = np.zeros(12)
            for i in iv:
                t[(r + i) % 12] = 1.0
            c = float(v @ (t / np.linalg.norm(t)))
            if c > best:
                best, nom = c, NOMS[r] + suf
    return nom


def blocs_de(trous, n):
    out, b = [], 0
    for p, w in sorted(trous):
        while b + L <= p:
            out.append(b); b += L
        b = max(b, p) + w
    while b + L <= n:
        out.append(b); b += L
    return out


def octets(M, hi=None):
    """Une matrice -> base64 d'un octet par case, plus son échelle."""
    M = np.asarray(M, float)
    hi = float(M.max()) if hi is None else hi
    q = np.clip(M / max(hi, 1e-9), 0, 1)
    return base64.b64encode((q * 255 + 0.5).astype(np.uint8).tobytes()).decode(), hi


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

    sorties = {}
    for nom_g, trous in (("sans", []), ("avec", [TROU])):
        blocs = blocs_de(trous, n)
        X = np.array([np.concatenate([V12[b + i] for i in range(L)]) for b in blocs])
        xb, xhi = octets(X)
        combos = {}
        for k in KS:
            if len(blocs) < k + 1:
                continue
            f = NMF(n_components=k, init="nndsvda", max_iter=800, random_state=0)
            U = f.fit_transform(X)
            V = f.components_
            R = U @ V
            Un = U / np.clip(U.sum(1, keepdims=True), 1e-9, None)
            ub, _ = octets(Un, 1.0)
            vb, vhi = octets(V)
            rb, _ = octets(R, xhi)
            combos[str(k)] = {
                "U": ub, "V": vb, "UV": rb, "vhi": round(vhi, 4),
                "argmax": [int(x) for x in np.argmax(Un, axis=1)],
                "purete": [round(float(x), 3) for x in Un.max(axis=1)],
                "residu": round(float(np.linalg.norm(X - R) /
                                      max(np.linalg.norm(X), 1e-9)), 4),
                "accords": [[nomme(V[j][i * 12:(i + 1) * 12]) for i in range(L)]
                            for j in range(k)],
            }
        sorties[nom_g] = {
            "X": xb, "xhi": round(xhi, 4), "nb": len(blocs),
            "blocs": [{"a": b + 1, "b": b + L, "t": round(float(grid[b]), 3)}
                      for b in blocs],
            "saut": [{"a": p + 1, "b": p + w} for p, w in trous],
            "k": combos,
        }

    d = {"titre": chart.get("title") or stem, "n": n, "L": L, "ks": KS,
         "audio_url": f"{BASE}/audio/{stem}.m4a",
         "retour": f"{BASE}/?open={FICHIER}",
         "trou_page": f"{BASE}/plots/trou_demo.html",
         "sorties": sorties}
    SORTIE.write_text(_G.replace("__D__", json.dumps(
        d, ensure_ascii=False, separators=(",", ":"))), encoding="utf-8")
    print(f"{BASE}/plots/uv_matrices.html")
    for g in ("sans", "avec"):
        s = sorties[g]
        print(f"  {g} trou : X {s['nb']}×48 · " +
              " ".join(f"k{k}={s['k'][k]['residu']}" for k in s["k"]))
    return 0


_G = r"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>X, U et V — en fonction de k</title>
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
 .lede{font:italic 14.5px/1.6 Georgia,serif;color:var(--pale);margin:0 0 14px}
 .ctl{display:flex;gap:18px;flex-wrap:wrap;align-items:center;margin:12px 0 18px;
      position:sticky;top:0;background:var(--papier);padding:10px 0;z-index:5;
      border-bottom:1px solid var(--trait)}
 .grp{display:flex;gap:5px;align-items:center}
 .grp b{font:600 10px sans-serif;letter-spacing:.08em;text-transform:uppercase;
        color:var(--pale);margin-right:3px}
 .ctl button{font:600 13px sans-serif;border:1px solid var(--trait);
   background:var(--carte);color:var(--encre);border-radius:9px;padding:7px 12px;
   cursor:pointer;min-height:38px}
 .ctl button.on{background:var(--accent);color:#fff;border-color:transparent}
 .bloc{margin:22px 0}
 .tit{font:600 14px sans-serif;margin-bottom:1px}
 .exp{font:italic 12.5px/1.5 Georgia,serif;color:var(--pale);margin-bottom:7px}
 .wrap{display:flex;gap:10px;align-items:flex-start;overflow-x:auto}
 .lab{flex:0 0 auto;font:10px ui-monospace,Menlo,monospace;color:var(--pale);
      text-align:right;padding-top:1px}
 .lab div{height:18px;line-height:18px;white-space:nowrap;cursor:pointer}
 .lab div:hover{color:var(--accent);font-weight:700}
 canvas{display:block;border:1px solid var(--trait);border-radius:6px;
        image-rendering:pixelated;cursor:crosshair}
 .cols{font:10px ui-monospace,Menlo,monospace;color:var(--pale);margin-top:3px;
       display:flex}
 .cols span{text-align:center;flex:1 1 0}
 .acc{font:600 12.5px Georgia,serif;margin-left:8px}
 .accs div{height:18px;line-height:18px;white-space:nowrap}
 .res{font:600 12px ui-monospace,Menlo,monospace;color:var(--pale);margin-top:5px}
 #bar{position:fixed;bottom:0;left:0;right:0;background:var(--papier);
      border-top:1px solid var(--trait);padding:10px 16px;display:flex;gap:12px;
      align-items:center;z-index:6}
 #bar button{background:var(--accent);color:#fff;border:none;border-radius:11px;
   padding:11px 16px;font:600 14px sans-serif;cursor:pointer;min-width:104px;min-height:44px}
 #ou{font:600 13px ui-monospace,Menlo,monospace;color:var(--pale)}
</style></head><body>
<script>window.D = __D__;</script>
<h1 id="t"></h1>
<p class="lede">Les vraies matrices. Les 48 colonnes de X et de V sont
<b>4 mesures × 12 hauteurs</b> — les traits verticaux séparent les mesures.
Clique une ligne pour l'écouter.</p>

<div class="ctl">
  <div class="grp"><b>k</b><span id="ks"></span></div>
  <div class="grp"><b>grille</b><span id="gs"></span></div>
</div>

<div class="bloc">
  <div class="tit">X — ce qu'on donne</div>
  <div class="exp">une ligne = un bloc de 4 mesures du morceau, mis bout à bout</div>
  <div class="wrap"><div class="lab" id="lx"></div><div><canvas id="cx"></canvas>
    <div class="cols" id="colx"></div></div></div>
</div>

<div class="bloc">
  <div class="tit">U — où sont les sections</div>
  <div class="exp">une ligne = le même bloc ; une colonne = une section.
    Clair = « ce bloc EST cette section ». Idéalement une seule case allumée par ligne.</div>
  <div class="wrap"><div class="lab" id="lu"></div><div><canvas id="cu"></canvas></div>
    <div class="lab accs" id="au"></div></div>
</div>

<div class="bloc">
  <div class="tit">V — les sections elles-mêmes</div>
  <div class="exp">une ligne = une section, écrite une fois : ses 4 mesures d'accords</div>
  <div class="wrap"><div class="lab" id="lv"></div><div><canvas id="cv"></canvas>
    <div class="cols" id="colv"></div></div>
    <div class="lab accs" id="av"></div></div>
</div>

<div class="bloc">
  <div class="tit">U·V — ce qu'on récupère</div>
  <div class="exp">à comparer avec X, à l'œil, sur la même échelle de couleur.
    Ce qui manque ici est le « ≈ ».</div>
  <div class="wrap"><div class="lab" id="lr"></div><div><canvas id="cr"></canvas></div></div>
  <div class="res" id="res"></div>
</div>

<div id="bar"><button id="jouer">Écouter</button><span id="ou"></span>
  <a href="#" id="lien" style="margin-left:auto;font-size:13px"></a></div>

<script>
(function(){
 "use strict";
 var D = window.D, k = "5", g = "avec", CELL = 18;
 document.getElementById("t").textContent = D.titre + " — X, U et V";

 function bytes(b64){
   var s = atob(b64), a = new Uint8Array(s.length);
   for (var i = 0; i < s.length; i++) a[i] = s.charCodeAt(i);
   return a;
 }
 /* clair -> fonce, une seule teinte : une magnitude, pas une categorie */
 function couleur(v){
   var st = [[251,247,236],[207,224,234],[132,179,207],[61,127,166],
             [27,74,107],[13,36,55]];
   var x = v/255*(st.length-1), i = Math.min(st.length-2, Math.floor(x)), f = x-i;
   return "rgb(" + st[i].map(function(c,j){
     return Math.round(c + f*(st[i+1][j]-c)); }).join(",") + ")";
 }
 function peins(id, arr, rows, cols, sep, large){
   var c = document.getElementById(id), x = c.getContext("2d");
   /* U n'a que k colonnes : a 14 px elle sort en timbre-poste et on ne voit
      plus ce qui est justement le plus lisible du lot. */
   var w = large ? 34 : Math.max(4, Math.min(14, Math.floor(760/cols)));
   c.width = cols*w; c.height = rows*CELL;
   c.style.width = (cols*w) + "px"; c.style.height = (rows*CELL) + "px";
   for (var i = 0; i < rows; i++) for (var j = 0; j < cols; j++){
     x.fillStyle = couleur(arr[i*cols+j]);
     x.fillRect(j*w, i*CELL, w, CELL);
   }
   if (sep){
     x.strokeStyle = "rgba(138,43,43,.55)"; x.lineWidth = 1;
     for (var m = 1; m < D.L; m++){
       x.beginPath(); x.moveTo(m*12*w + .5, 0); x.lineTo(m*12*w + .5, rows*CELL);
       x.stroke();
     }
   }
   return {w: w, cols: cols};
 }
 function etiquettes(id, items, clic){
   var h = document.getElementById(id); h.innerHTML = "";
   items.forEach(function(txt, i){
     var e = document.createElement("div");
     e.textContent = txt;
     if (clic) e.onclick = function(){ clic(i); };
     h.appendChild(e);
   });
 }
 function colonnes(id, w, cols){
   var h = document.getElementById(id); h.innerHTML = "";
   h.style.width = (cols*w) + "px";
   for (var m = 0; m < D.L; m++){
     var s = document.createElement("span");
     s.textContent = "mesure " + (m+1);
     h.appendChild(s);
   }
 }

 function dessine(){
   var S = D.sorties[g], C = S.k[k];
   Array.prototype.forEach.call(document.querySelectorAll("#ks button"), function(b){
     b.className = (b.dataset.v === k) ? "on" : ""; });
   Array.prototype.forEach.call(document.querySelectorAll("#gs button"), function(b){
     b.className = (b.dataset.v === g) ? "on" : ""; });

   var nb = S.nb, K = parseInt(k, 10);
   var lignes = S.blocs.map(function(b){ return "mes. " + b.a + "–" + b.b; });
   var clic = function(i){ vers(S.blocs[i].t); };

   var mx = peins("cx", bytes(S.X), nb, 48, true);
   etiquettes("lx", lignes, clic); colonnes("colx", mx.w, 48);

   peins("cu", bytes(C.U), nb, K, false, true);
   etiquettes("lu", lignes, clic);
   etiquettes("au", C.argmax.map(function(j, i){
     return "→ section " + (j+1) + "  (" + C.purete[i].toFixed(2) + ")"; }), clic);

   var mv = peins("cv", bytes(C.V), K, 48, true);
   etiquettes("lv", C.accords.map(function(_a, j){ return "section " + (j+1); }));
   colonnes("colv", mv.w, 48);
   etiquettes("av", C.accords.map(function(a){ return "| " + a.join(" | ") + " |"; }));

   peins("cr", bytes(C.UV), nb, 48, true);
   etiquettes("lr", lignes, clic);
   document.getElementById("res").textContent =
     "résidu ‖X − U·V‖ / ‖X‖ = " + C.residu +
     "   ·   X " + nb + "×48   U " + nb + "×" + K + "   V " + K + "×48" +
     (S.saut.length ? "   ·   trou : mesures " + S.saut[0].a + "–" + S.saut[0].b
                    : "   ·   aucun trou");
 }

 var hk = document.getElementById("ks");
 D.ks.forEach(function(x){
   if (!D.sorties[g].k[String(x)]) return;
   var b = document.createElement("button");
   b.textContent = x; b.dataset.v = String(x);
   b.onclick = function(){ k = String(x); dessine(); };
   hk.appendChild(b);
 });
 var hg = document.getElementById("gs");
 [["avec", "avec le trou"], ["sans", "sans trou"]].forEach(function(p){
   var b = document.createElement("button");
   b.textContent = p[1]; b.dataset.v = p[0];
   b.onclick = function(){ g = p[0]; dessine(); };
   hg.appendChild(b);
 });
 var L2 = document.getElementById("lien");
 L2.href = D.trou_page; L2.textContent = "où placer le trou →";

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
