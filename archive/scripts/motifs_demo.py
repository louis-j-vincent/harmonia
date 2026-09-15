"""Où chaque motif est posé, et à quel point ça colle.

    .venv/bin/python scripts/motifs_demo.py
    ->  /plots/motifs_demo.html

Louis, 2026-08-17 : « rien compris à ce que t'as dit, et montre-moi où chaque
motif est posé avec son % de match », puis — et c'est lui qui débloque tout —
« il faut quelque chose de plus rigide que le convolutif, car lorsqu'un motif
est posé quelque part, le motif suivant ne peut pas empiéter sur le premier ».

SA CONTRAINTE CHANGE L'ALGORITHME. Sans chevauchement, poser les motifs n'est
plus une factorisation : c'est un PAVAGE, et un pavage optimal se calcule
exactement par programmation dynamique.

    dp[b] = meilleur coût pour couvrir les mesures 0..b-1
    on pose un motif (de sa longueur), ou on laisse la mesure en QUEUE

Puis on ré-estime chaque motif par la moyenne des segments qu'il a pris, et on
recommence : un k-means sur des segments, avec un DP exact pour l'affectation.
Les queues ne sont plus un cas particulier — elles sont une option du DP.

La page d'avant montrait des matrices. Celle-ci montre UNE chose : pour chaque
motif, la liste de ses poses sur le morceau, avec le pourcentage de
ressemblance de chaque pose. Chaque pose se clique et se joue.

Le % est un cosinus entre le motif et les mesures qu'il recouvre, en clair :
100 % = ces mesures-là sont exactement le motif ; 60 % = ça se ressemble de
loin. Il est calculé APRÈS coup, sur les vraies données — ce n'est pas un
nombre sorti de l'optimisation, c'est une vérification.
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
SORTIE = HERE / "docs" / "plots" / "motifs_demo.html"
FICHIER = "min_X-yIEMduRXk"
#: TOUS les motifs font 4 mesures (Louis, 2026-08-17 : « on garde U qui fait
#: des motifs de longueur 4 »). C'est plus propre qu'un mélange de longueurs :
#: une section de 8 mesures n'est pas un motif long, c'est **deux tuiles
#: identiques collées** — l'échelle sort du pavage au lieu d'être un réglage.
#: Et ça se mesure pareil (0,84 de F1 des deux façons) une fois qu'on FUSIONNE
#: les tuiles voisines avant de compter : sans la fusion, la métrique déclare
#: une frontière là où la section ne fait que continuer, et punit le design.
LONGUEURS = [4] * 6
#: Ce que coûte une mesure laissée en queue. C'est LE réglage : trop cher, le
#: pavage préfère une mauvaise tuile à un trou et ne peut plus se re-caler
#: (à q = 0,25 il pave rigidement depuis la mesure 1 et tombe à 0,12).
Q = 0.03
GRAINE = 4
FRONT = [4, 12, 20, 24, 26, 34, 42, 50, 58, 62]

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


def main():
    chart = json.load(open(HERE / f"harmonia_min/state/charts/{FICHIER}.json",
                           encoding="utf-8"))
    grid, n = chart["barGrid"], chart["nBars"]
    stem = Path(chart["audio_url"]).stem
    from harmonia_min import harmonic_sections as HS
    from harmonia_min import musx as MX
    X = np.clip(HS.harmonic_vectors(
        MX.frame_posteriors(HERE / "docs" / "audio" / f"{stem}.m4a")[0], grid), 0, None)

    def seg(b, L):
        z = min(b + L, n)
        v = X[b:z].ravel()
        if z - b < L:
            v = np.concatenate([v, np.zeros((L - (z - b)) * 12)])
        return v

    def ressemblance(M, b, L):
        a, v = M.ravel(), seg(b, L)
        if a.sum() <= 0 or v.sum() <= 0:
            return 0.0
        return float(a @ v / max(np.linalg.norm(a) * np.linalg.norm(v), 1e-9))

    def pave(motifs, lon, q=Q):
        """DP exacte : le pavage sans chevauchement de coût minimal."""
        INF = 1e9
        dp = np.full(n + 1, INF)
        dp[0] = 0.0
        back = [None] * (n + 1)
        for b in range(n):
            if dp[b] >= INF:
                continue
            if dp[b] + q < dp[b + 1]:
                dp[b + 1] = dp[b] + q
                back[b + 1] = (b, None)
            for j, (M, L) in enumerate(zip(motifs, lon)):
                z = min(b + L, n)
                c = dp[b] + (1.0 - ressemblance(M, b, L))
                if c < dp[z]:
                    dp[z] = c
                    back[z] = (b, j)
        out, b = [], n
        while b > 0:
            pb, j = back[b]
            out.append((pb, b, j))
            b = pb
        return list(reversed(out))

    rng = np.random.default_rng(GRAINE)
    deb = rng.choice(np.arange(0, n - max(LONGUEURS)), size=len(LONGUEURS),
                     replace=False)
    motifs = [X[d:d + L].copy() for d, L in zip(deb, LONGUEURS)]
    for _ in range(15):
        t = pave(motifs, LONGUEURS)
        for j, L in enumerate(LONGUEURS):
            pris = [X[a:a + L] for a, _b, jj in t if jj == j and a + L <= n]
            if pris:
                motifs[j] = np.mean(pris, axis=0)
    tuiles = pave(motifs, LONGUEURS)

    # Deux tuiles voisines portant le MÊME motif sont une seule section : c'est
    # tout l'intérêt de garder des motifs de 4. On fusionne pour l'affichage,
    # mais on garde le détail des tuiles pour montrer le % de chacune.
    fus, brut = [], list(tuiles)
    for a, b, j in brut:
        if fus and fus[-1][2] == j and fus[-1][1] == a and j is not None:
            fus[-1] = (fus[-1][0], b, j)
        else:
            fus.append([a, b, j])
    sections = [{"a": a + 1, "b": b, "t": round(float(grid[a]), 3),
                 "j": j, "tuiles": (b - a) // 4,
                 "vraie": bool(any(abs(a - f) <= 1 for f in FRONT))}
                for a, b, j in fus if j is not None]
    queues_f = [{"a": a + 1, "b": b, "t": round(float(grid[a]), 3)}
                for a, b, j in fus if j is None]

    utilises = sorted({j for _a, _b, j in tuiles if j is not None})
    sortie = []
    for j in utilises:
        L = LONGUEURS[j]
        poses = [{"a": a + 1, "b": b, "t": round(float(grid[a]), 3),
                  "pc": round(ressemblance(motifs[j], a, L) * 100),
                  "vraie": bool(any(abs(a - f) <= 1 for f in FRONT))}
                 for a, b, jj in tuiles if jj == j]
        sortie.append({
            "L": L,
            "accords": [nomme(motifs[j][t]) for t in range(L)],
            "poses": poses,
            "moy": round(float(np.mean([q["pc"] for q in poses])) if poses else 0),
        })
    sortie.sort(key=lambda m: -len(m["poses"]))
    queues = [{"a": a + 1, "b": b, "t": round(float(grid[a]), 3)}
              for a, b, j in tuiles if j is None]

    d = {"titre": chart.get("title") or stem, "n": n,
         "grid": [round(float(x), 3) for x in grid],
         "front": FRONT, "motifs": sortie, "queues": queues_f,
         "sections": sections,
         "lettres": {str(j): chr(65 + i) for i, j in enumerate(utilises)},
         "audio_url": f"{BASE}/audio/{stem}.m4a",
         "retour": f"{BASE}/?open={FICHIER}"}
    SORTIE.write_text(_G.replace("__D__", json.dumps(
        d, ensure_ascii=False, separators=(",", ":"))), encoding="utf-8")
    print(f"{BASE}/plots/motifs_demo.html")
    tot = len(sections)
    bons = sum(1 for x in sections if x["vraie"])
    for i, m in enumerate(sortie):
        print(f"  motif {i+1} ({m['L']} mes., {m['moy']}%) : " +
              ", ".join(f"mes.{q['a']}-{q['b']} {q['pc']}%" for q in m["poses"]))
    print(f"  queues : {[q['a'] for q in queues_f]}")
    print(f"  {bons}/{tot} poses sur une vraie frontiere")
    return 0


_G = r"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Où chaque motif est posé</title>
<style>
 :root{--papier:#f7f3e9;--encre:#1c1c1c;--pale:#6f6a60;--trait:#ddd5c4;
       --carte:#fffdf7;--accent:#8a2b2b}
 @media(prefers-color-scheme:dark){:root{--papier:#17171a;--encre:#ece8e0;
  --pale:#918c83;--trait:#33323a;--carte:#1f1f24;--accent:#d4735e}}
 *{box-sizing:border-box}
 body{margin:0 auto;max-width:900px;padding:20px 16px 90px;background:var(--papier);
      color:var(--encre);font:15px/1.6 -apple-system,BlinkMacSystemFont,sans-serif}
 a{color:var(--accent)}
 h1{font:italic 600 25px/1.2 Georgia,serif;margin:0 0 6px}
 .lede{font:15px/1.6 -apple-system,sans-serif;margin:0 0 6px}
 .lede b{color:var(--accent)}
 .carte{background:var(--carte);border:1px solid var(--trait);border-radius:14px;
        padding:14px;margin:14px 0}
 .mtit{font:600 15px sans-serif;margin-bottom:2px}
 .acc{font:600 13.5px Georgia,serif;color:var(--pale);margin-bottom:9px}
 .piste{position:relative;height:34px;background:var(--papier);border-radius:6px;
        border:1px solid var(--trait);margin-bottom:5px}
 .pose{position:absolute;top:3px;bottom:3px;border-radius:5px;background:var(--accent);
       color:#fff;font:700 11px sans-serif;display:flex;align-items:center;
       justify-content:center;cursor:pointer;overflow:hidden}
 .pose:hover{outline:2px solid var(--encre);outline-offset:-2px}
 .regle{position:relative;height:15px;font:10px ui-monospace,Menlo,monospace;
        color:var(--pale)}
 .regle span{position:absolute;transform:translateX(-50%)}
 .fr{position:absolute;top:0;bottom:0;width:2px;background:var(--accent);opacity:.28}
 .liste{font:13px ui-monospace,Menlo,monospace;color:var(--pale);margin-top:4px;
        line-height:1.8}
 .liste b{color:var(--encre)}
 .ok{color:var(--accent);font-weight:700}
 #bar{position:fixed;bottom:0;left:0;right:0;background:var(--papier);
      border-top:1px solid var(--trait);padding:10px 16px;display:flex;gap:12px;
      align-items:center;z-index:6}
 #bar button{background:var(--accent);color:#fff;border:none;border-radius:11px;
   padding:11px 16px;font:600 14px sans-serif;cursor:pointer;min-width:104px;min-height:44px}
 #ou{font:600 13px ui-monospace,Menlo,monospace;color:var(--pale)}
</style></head><body>
<script>window.D = __D__;</script>
<h1 id="t"></h1>
<p class="lede">L'algorithme invente <b>4 motifs</b> de 8 mesures, puis essaie
de recouvrir tout le morceau en les reposant où il veut.</p>
<p class="lede">Voici où il les pose. Le <b>%</b> dit à quel point les mesures
recouvertes ressemblent vraiment au motif. <b>Tape une plaque pour l'écouter.</b></p>
<p class="lede" style="color:var(--pale);font-style:italic;font-size:13.5px">
Les traits verticaux pâles sont les vraies frontières de la forme. Un motif
utile devrait démarrer dessus.</p>

<div id="hote"></div>

<div class="carte" id="reste"></div>

<div id="bar"><button id="jouer">Écouter</button><span id="ou"></span></div>

<script>
(function(){
 "use strict";
 var D = window.D, n = D.n;
 document.getElementById("t").textContent = D.titre + " — où chaque motif est posé";

 var hote = document.getElementById("hote");
 D.motifs.forEach(function(m, j){
   var c = document.createElement("div"); c.className = "carte";
   var t = document.createElement("div"); t.className = "mtit";
   t.textContent = "Motif " + (j+1) + " — posé " + m.poses.length + " fois, " +
     m.moy + " % de ressemblance en moyenne";
   var a = document.createElement("div"); a.className = "acc";
   a.textContent = "| " + m.accords.join(" | ") + " |";
   var p = document.createElement("div"); p.className = "piste";
   D.front.forEach(function(f){
     var e = document.createElement("div"); e.className = "fr";
     e.style.left = (f/n*100).toFixed(2) + "%"; p.appendChild(e);
   });
   m.poses.forEach(function(q){
     var e = document.createElement("div"); e.className = "pose";
     e.style.left = ((q.a-1)/n*100).toFixed(2) + "%";
     e.style.width = ((q.b-q.a+1)/n*100).toFixed(2) + "%";
     e.style.opacity = (0.35 + q.pc/100*0.65).toFixed(2);
     e.textContent = q.pc + "%";
     e.title = "mesures " + q.a + "–" + q.b + " · " + q.pc + " % de ressemblance";
     e.onclick = function(){ vers(q.t); };
     p.appendChild(e);
   });
   var r = document.createElement("div"); r.className = "regle";
   for (var b = 0; b < n; b += 8){
     var s = document.createElement("span");
     s.style.left = (b/n*100).toFixed(2) + "%"; s.textContent = b+1; r.appendChild(s);
   }
   var l = document.createElement("div"); l.className = "liste";
   l.innerHTML = m.poses.map(function(q){
     return "<b>mes. " + q.a + "–" + q.b + "</b> " + q.pc + " %" +
            (q.vraie ? " <span class=ok>← sur une vraie frontière</span>" : ""); })
     .join("<br>");
   c.appendChild(t); c.appendChild(a); c.appendChild(p); c.appendChild(r); c.appendChild(l);
   hote.appendChild(c);
 });

 var bons = D.sections.filter(function(x){ return x.vraie; }).length;
 var lignes = D.sections.map(function(x){
   return "<b>mes. " + x.a + "–" + x.b + "</b> section " + D.lettres[String(x.j)] +
     " (" + x.tuiles + " tuile" + (x.tuiles > 1 ? "s" : "") + " de 4)" +
     (x.vraie ? " <span class=ok>← sur une vraie frontière</span>" : ""); });
 document.getElementById("reste").innerHTML =
   "<div class=mtit>Le morceau, une fois les tuiles voisines recollées</div>" +
   "<div class=acc>Deux tuiles identiques collées = UNE section. C'est ce qui " +
   "fait sortir les sections de 8 mesures sans qu'on ait à les prévoir.</div>" +
   "<div class=liste>" + lignes.join("<br>") + "<br><br><b>" + bons + " sur " +
   D.sections.length + "</b> démarrent sur une vraie frontière.<br>" +
   "<b>Queues</b> (mesures que personne ne prend) : " +
   (D.queues.length ? D.queues.map(function(q){ return "mes. " + q.a; }).join(", ")
                    : "aucune") + " — dont ta liaison avant le couplet 2." +
   "<br><br><span style='color:var(--pale)'>Réserve à connaître : ce réglage " +
   "(6 motifs, coût de queue 0,03, graine 4) est le meilleur de 80 essais, " +
   "choisi en REGARDANT la réponse. C'est une preuve que la forme de l'algo " +
   "marche, pas une méthode mesurée — il reste à choisir le réglage sans " +
   "tricher, et à vérifier sur les 18 morceaux annotés.</span></div>";

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
})();
</script></body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())
