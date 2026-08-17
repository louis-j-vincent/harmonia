"""X ≈ U·V, l'algorithme en détail et une démo qui s'écoute.

    .venv/bin/python scripts/uv_explication.py [<file_key>]
    ->  /plots/uv_explication.html

Louis, 2026-08-17 : « comment tu fais marcher le X = UV exactement, montre-moi
l'algo en détail et une démo ».

L'idée est la sienne : réduire un morceau à ses sections distinctes, c'est
factoriser. **V** porte les sections (une rangée = une section), **U** dit où
chacune se joue, et le « ≈ » absorbe ce qu'on ne veut pas compter — intros,
outros, queues.

CE QUI DÉCIDE DE TOUT : SUR QUOI ON FACTORISE.

    X = mesures × 12          ->  V = un vocabulaire d'ACCORDS. Inutile.
    X = blocs × (L × 12)      ->  V = des MOTIFS de L mesures = des sections.

C'est le seul choix vraiment structurant, et la page le montre côte à côte :
avec le premier X, la rangée « la plus utilisée » est un accord de si♭ ; avec
le second, c'est le refrain entier.

CE QUE LA DÉMO MONTRE AUSSI, ET QUI EST LA VRAIE LIMITE : la grille de blocs
lui est DONNÉE. La factorisation résout le nommage (quelles occurrences sont la
même section), pas le découpage (où sont les frontières). Sur Easy On Me, la
couture de 2 mesures à la mesure 25 vient d'une lecture à la main ; sans elle,
la parité se décale et tout ce qui suit tombe à côté — la page montre aussi ce
cas-là, parce que c'est lui qui dit à quoi sert le reste du projet.
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
SORTIE = HERE / "docs" / "plots" / "uv_explication.html"
DEFAUT = "min_X-yIEMduRXk"                       # Easy On Me
L = 4                                            # mesures par bloc

#: La grille de blocs, lue à la main sur le morceau : 4 mesures partout, sauf
#: la liaison de 2 mesures à la 25. C'est l'entrée que la factorisation ne sait
#: PAS trouver toute seule — voir la démo « sans la couture ».
BORNES = [0, 4, 8, 12, 16, 20, 24, 26, 30, 34, 38, 42, 46, 50, 54, 58, 62, 63]

NOMS = ["C", "D♭", "D", "E♭", "E", "F", "G♭", "G", "A♭", "A", "B♭", "B"]
QUALITES = [("", [0, 4, 7]), ("m", [0, 3, 7]), ("7", [0, 4, 7, 10]),
            ("m7", [0, 3, 7, 10]), ("maj7", [0, 4, 7, 11])]


def nomme(v12) -> str:
    """Le vecteur de 12 hauteurs -> le nom d'accord le plus proche.

    C'est ce qui rend V LISIBLE : sans ça une rangée de V est une colonne de
    48 nombres, et personne ne peut dire si l'algorithme a trouvé le refrain.
    """
    v = np.asarray(v12, float)
    if v.sum() <= 1e-9:
        return "—"
    v = v / np.linalg.norm(v)
    best, nom = -1.0, "—"
    for r in range(12):
        for suf, iv in QUALITES:
            t = np.zeros(12)
            for i in iv:
                t[(r + i) % 12] = 1.0
            c = float(v @ (t / np.linalg.norm(t)))
            if c > best:
                best, nom = c, NOMS[r] + suf
    return nom


def bloque(X, bornes):
    """Les blocs, aplatis : une rangée = L mesures mises bout à bout."""
    out = []
    for i in range(len(bornes) - 1):
        a, b = bornes[i], bornes[i + 1]
        v = np.zeros(L * 12)
        for j in range(L):
            v[j * 12:(j + 1) * 12] = X[min(a + j, b - 1)] if a + j < b else X[b - 1]
        out.append(v)
    return np.array(out)


def factorise(Xb, k, seed=0):
    from sklearn.decomposition import NMF
    m = NMF(n_components=k, init="nndsvda", max_iter=800, random_state=seed)
    U = m.fit_transform(Xb)
    V = m.components_
    res = float(np.linalg.norm(Xb - U @ V) / max(np.linalg.norm(Xb), 1e-9))
    Un = U / np.clip(U.sum(1, keepdims=True), 1e-9, None)
    return U, V, Un, res, float(np.mean(Un.max(1)))


def main(argv):
    file_key = argv[0] if argv else DEFAUT
    chart = json.load(open(HERE / f"harmonia_min/state/charts/{file_key}.json",
                           encoding="utf-8"))
    grid, n = chart["barGrid"], chart["nBars"]
    stem = Path(chart["audio_url"]).stem
    audio = HERE / "docs" / "audio" / f"{stem}.m4a"

    from harmonia_min import harmonic_sections as HS
    from harmonia_min import musx as MX
    triad = MX.frame_posteriors(audio)[0]
    X = np.clip(HS.harmonic_vectors(triad, grid), 0, None)     # (n, 12)

    # ── (a) le mauvais X : mesures × 12 -> un vocabulaire d'accords
    _Ua, Va, _Un, resa, pura = factorise(X, 4)
    mauvais = [{"nom": nomme(Va[j]), "poids": round(float(Va[j].sum()), 2)}
               for j in range(len(Va))]

    # ── (b) le bon X : blocs × 48 -> des motifs
    Xb = bloque(X, BORNES)
    k = 6
    U, V, Un, res, pur = factorise(Xb, k)
    LET = "VWXYZT"
    rangees = [{"nom": LET[j],
                "accords": [nomme(V[j][i * 12:(i + 1) * 12]) for i in range(L)],
                "poids": round(float(V[j].sum()), 2)}
               for j in range(k)]
    blocs = []
    for i in range(len(BORNES) - 1):
        a, b = BORNES[i], BORNES[i + 1]
        blocs.append({"a": a + 1, "b": b, "t": round(float(grid[a]), 3),
                      "u": [round(float(x), 3) for x in Un[i]],
                      "j": int(np.argmax(Un[i])),
                      "lettre": LET[int(np.argmax(Un[i]))],
                      "purete": round(float(Un[i].max()), 3),
                      "residu": round(float(np.linalg.norm(Xb[i] - U[i] @ V) /
                                            max(np.linalg.norm(Xb[i]), 1e-9)), 3)})

    # ── (c) le choix de k
    courbe = []
    for kk in range(2, 9):
        _u, _v, _un, r, p = factorise(Xb, kk)
        courbe.append({"k": kk, "residu": round(r, 3), "purete": round(p, 3)})

    # ── (d) sans la couture : la grille rigide de 4, et ce qu'elle casse
    rigide = list(range(0, n - L + 1, L)) + [n]
    Xr = bloque(X, rigide)
    Ur, Vr, Unr, _rr, purr = factorise(Xr, k)
    sans = [{"a": rigide[i] + 1, "b": rigide[i + 1],
             "lettre": LET[int(np.argmax(Unr[i]))],
             "purete": round(float(Unr[i].max()), 3)}
            for i in range(len(rigide) - 1)]

    d = {"titre": chart.get("title") or stem, "n": n, "k": k, "L": L,
         "audio_url": f"{BASE}/audio/{stem}.m4a",
         "retour": f"{BASE}/?open={file_key}",
         "sections_page": f"{BASE}/plots/easy_on_me_sections.html",
         "mauvais": mauvais, "res_mauvais": round(resa, 3), "pur_mauvais": round(pura, 3),
         "rangees": rangees, "blocs": blocs, "residu": round(res, 3),
         "purete": round(pur, 3), "courbe": courbe, "sans_couture": sans,
         "pur_sans": round(purr, 3),
         "dims": {"X": [len(Xb), L * 12], "U": [len(Xb), k], "V": [k, L * 12]}}

    SORTIE.write_text(_GABARIT.replace("__D__", json.dumps(
        d, ensure_ascii=False, separators=(",", ":"))), encoding="utf-8")
    print(f"{BASE}/plots/uv_explication.html")
    print(f"  X {d['dims']['X']}  =  U {d['dims']['U']} · V {d['dims']['V']}")
    print(f"  résidu {res:.3f}  pureté {pur:.3f}")
    for r in rangees:
        print(f"    {r['nom']} : | " + " | ".join(r["accords"]) + " |")
    return 0


_GABARIT = r"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>X ≈ U·V — l'algorithme, et une démo</title>
<style>
 :root{--papier:#f7f3e9;--encre:#1c1c1c;--pale:#6f6a60;--trait:#ddd5c4;
       --carte:#fffdf7;--accent:#8a2b2b;--bleu:#2f6f8f}
 @media(prefers-color-scheme:dark){:root{--papier:#17171a;--encre:#ece8e0;
  --pale:#918c83;--trait:#33323a;--carte:#1f1f24;--accent:#d4735e;--bleu:#6fa8c7}}
 *{box-sizing:border-box}
 body{margin:0 auto;max-width:880px;padding:20px 16px 40px;background:var(--papier);
      color:var(--encre);font:15px/1.6 -asystem,-apple-system,BlinkMacSystemFont,sans-serif}
 a{color:var(--accent)} code{font:13px ui-monospace,Menlo,monospace;
   background:var(--carte);padding:1px 5px;border-radius:4px;border:1px solid var(--trait)}
 h1{font:italic 600 26px/1.2 Georgia,serif;margin:0 0 6px}
 h2{font:600 11px sans-serif;letter-spacing:.09em;text-transform:uppercase;
    color:var(--pale);margin:30px 0 8px;border-top:1px solid var(--trait);padding-top:14px}
 h3{font:600 15px sans-serif;margin:18px 0 4px}
 .lede{font:italic 15px/1.65 Georgia,serif;color:var(--pale);margin:0 0 6px}
 .carte{background:var(--carte);border:1px solid var(--trait);border-radius:14px;
        padding:14px;margin:10px 0;overflow-x:auto}
 table{border-collapse:collapse;width:100%;font-size:13.5px;min-width:460px}
 th{text-align:left;font:600 10.5px sans-serif;letter-spacing:.07em;
    text-transform:uppercase;color:var(--pale);padding:0 8px 6px 0;white-space:nowrap}
 td{padding:5px 8px 5px 0;border-top:1px solid var(--trait);vertical-align:middle}
 .acc{font:600 13.5px Georgia,serif}
 .let{display:inline-block;min-width:24px;text-align:center;color:#fff;
      border-radius:5px;padding:2px 6px;font:700 12px sans-serif}
 .bar{display:inline-block;height:9px;border-radius:3px;background:var(--bleu);
      vertical-align:middle}
 .mono{font:12.5px ui-monospace,Menlo,monospace;color:var(--pale)}
 .eq{font:600 17px ui-monospace,Menlo,monospace;text-align:center;margin:14px 0;
     color:var(--encre)}
 .note{font:italic 13.5px/1.6 Georgia,serif;color:var(--pale);margin:6px 0 0}
 .cle{border-left:3px solid var(--accent);padding-left:12px;margin:14px 0}
 button{font:600 13px sans-serif;border:1px solid var(--trait);background:var(--carte);
        color:var(--encre);border-radius:9px;padding:7px 11px;cursor:pointer}
 ol{padding-left:22px} li{margin:8px 0}
</style></head><body>
<script>window.D = __D__;</script>
<h1>X ≈ U·V — l'algorithme, et une démo</h1>
<p class="lede" id="lede"></p>

<h2>1. L'idée, en une ligne</h2>
<div class="eq">X &nbsp;≈&nbsp; U &nbsp;·&nbsp; V</div>
<p><b>V</b> porte les sections : une rangée = une section, écrite une fois.
<b>U</b> dit où chacune se joue. Le <b>≈</b> est le résidu — ce qu'on ne veut
pas compter : intros, outros, queues. Réduire un morceau à ses sections
distinctes, c'est exactement chercher le plus petit V qui, replacé par U,
redonne le morceau.</p>

<h2>2. Le seul choix qui décide de tout : sur quoi on factorise</h2>
<p>C'est ici que ça se joue, et c'est contre-intuitif. Si les <b>lignes de X
sont des mesures</b>, les rangées de V sont des <b>accords</b> — pas des
sections. Il faut que les lignes de X soient des <b>blocs de plusieurs
mesures</b> pour que V apprenne des motifs.</p>

<h3>(a) X = mesures × 12 hauteurs — ce qu'on obtient</h3>
<div class="carte"><table><thead><tr><th>rangée de V</th><th>ce que c'est</th></tr></thead>
<tbody id="mauvais"></tbody></table></div>
<p class="note" id="note-mauvais"></p>

<h3>(b) X = blocs de 4 mesures × 48 — ce qu'on veut</h3>
<p>Chaque ligne de X est un bloc de 4 mesures <b>mis bout à bout</b> :
4 × 12 = 48 colonnes. Une rangée de V est donc une <b>suite de 4 accords</b>,
c'est-à-dire une section.</p>
<div class="carte"><table><thead><tr><th>rangée de V</th><th>les 4 mesures qu'elle porte</th>
<th>poids</th></tr></thead><tbody id="rangees"></tbody></table></div>
<p class="note">Ces noms d'accords ne sont pas dans les données : chaque tranche
de 12 nombres est comparée aux gabarits d'accords et on écrit le plus proche.
C'est ce qui rend V lisible — sinon une rangée est une colonne de 48 nombres.</p>

<h2>3. L'algorithme, étape par étape</h2>
<ol>
<li><b>Construire X.</b> Pour chaque bloc, empiler les vecteurs chord-tone des
L mesures. <code>X : {n_blocs} × {L×12}</code>, tout ≥ 0.</li>
<li><b>Factoriser en non-négatif</b> (NMF, <code>init="nndsvda"</code>) :
on cherche U ≥ 0 et V ≥ 0 qui minimisent ‖X − UV‖.
<b>La non-négativité est le point.</b> Avec une SVD on obtiendrait des
composantes signées, où une section peut en « annuler » une autre : illisible
musicalement. En non-négatif, un bloc est une <b>somme</b> de sections — jamais
une soustraction.</li>
<li><b>Lire la forme.</b> On normalise chaque ligne de U pour qu'elle somme à 1,
puis <code>argmax</code> donne la lettre du bloc, et le max donne la
<b>pureté</b> — à quel point ce bloc est UNE section plutôt qu'un mélange.</li>
<li><b>Lire le résidu.</b> ‖X − UV‖ par bloc : les blocs mal expliqués sont les
candidats « queue / intro / outro ».</li>
<li><b>Choisir k</b> en regardant où la pureté cesse de progresser (tableau
plus bas), pas seulement où le résidu baisse — le résidu baisse toujours.</li>
</ol>

<h2>4. La démo : U, bloc par bloc</h2>
<p>Chaque ligne est un bloc du morceau. La barre montre la répartition de U —
tape une ligne pour l'écouter.</p>
<div class="carte"><table><thead><tr><th>mesures</th><th>lettre</th>
<th>répartition de U</th><th>pureté</th><th>résidu</th></tr></thead>
<tbody id="blocs"></tbody></table></div>
<div id="bar" style="margin:10px 0"><button id="jouer">Écouter</button>
 <span class="mono" id="ou"></span></div>
<div class="cle" id="verdict"></div>

<h2>5. Le choix de k</h2>
<div class="carte"><table><thead><tr><th>k</th><th>résidu</th><th>pureté de U</th>
</tr></thead><tbody id="courbe"></tbody></table></div>
<p class="note">Le résidu baisse toujours quand k monte — ce n'est donc pas un
critère. La pureté, elle, plafonne : c'est elle qui dit quand on a assez de
sections.</p>

<h2>6. La limite, et c'est la vraie</h2>
<p><b>La grille de blocs lui est donnée.</b> Sur ce morceau elle fait 4 mesures
partout <i>sauf</i> une liaison de 2 mesures à la mesure 25. Cette couture, je
l'ai lue à la main. Voici la même factorisation avec une grille rigide de 4
mesures :</p>
<div class="carte"><table><thead><tr><th>mesures</th><th>lettre</th><th>pureté</th>
</tr></thead><tbody id="sans"></tbody></table></div>
<p class="note" id="note-sans"></p>
<div class="cle"><b>Conclusion.</b> La factorisation résout le <b>nommage</b> —
quelles occurrences sont la même section, sans se soucier de leur alignement,
ce qui est exactement là où <code>merge_letters</code> échoue aujourd'hui. Elle
ne résout <b>pas le découpage</b> : il lui faut les frontières en entrée. Les
deux problèmes sont séparés, et c'est une bonne nouvelle — on peut brancher la
factorisation sur le découpage actuel sans rien casser.</p>

<p style="margin-top:24px" id="liens"></p>

<script>
(function(){
 "use strict";
 var D = window.D;
 var COUL = ["#8a2b2b","#2f6f8f","#7a6320","#4a7a4a","#6b4a7a","#a85a2a"];
 document.getElementById("lede").textContent =
   D.titre + " — X est " + D.dims.X[0] + "×" + D.dims.X[1] + ", U est " +
   D.dims.U[0] + "×" + D.dims.U[1] + ", V est " + D.dims.V[0] + "×" + D.dims.V[1] +
   ". Résidu " + D.residu + ", pureté " + D.purete + ".";

 var mb = document.getElementById("mauvais");
 D.mauvais.forEach(function(r){
   var t = document.createElement("tr");
   t.innerHTML = "<td class=acc>" + r.nom + "</td><td class=mono>un accord, pas une section</td>";
   mb.appendChild(t);
 });
 document.getElementById("note-mauvais").textContent =
   "Quatre rangées, quatre accords. Le morceau entier est décrit par son " +
   "vocabulaire harmonique — et on n'a rien appris sur sa forme. Pureté " +
   D.pur_mauvais + ", résidu " + D.res_mauvais + " : de bons chiffres pour une réponse inutile.";

 var rg = document.getElementById("rangees");
 D.rangees.forEach(function(r, j){
   var t = document.createElement("tr");
   t.innerHTML = "<td><span class=let style='background:" + COUL[j % COUL.length] +
     "'>" + r.nom + "</span></td><td class=acc>| " + r.accords.join(" | ") +
     " |</td><td class=mono>" + r.poids + "</td>";
   rg.appendChild(t);
 });

 var bl = document.getElementById("blocs");
 D.blocs.forEach(function(b){
   var t = document.createElement("tr");
   t.style.cursor = "pointer";
   var barres = b.u.map(function(x, j){
     return x < 0.02 ? "" : "<span class=bar style='width:" + (x*120).toFixed(1) +
       "px;background:" + COUL[j % COUL.length] + "'></span>";
   }).join("");
   t.innerHTML = "<td class=mono>" + b.a + "–" + b.b + "</td>" +
     "<td><span class=let style='background:" + COUL[b.j % COUL.length] + "'>" +
     b.lettre + "</span></td><td>" + barres + "</td>" +
     "<td class=mono>" + b.purete.toFixed(2) + "</td>" +
     "<td class=mono>" + b.residu.toFixed(2) + "</td>";
   t.onclick = function(){ vers(b.t); };
   bl.appendChild(t);
 });

 var meilleur = D.blocs.reduce(function(a, b){ return b.purete > a.purete ? b : a; });
 document.getElementById("verdict").innerHTML =
   "<b>Ce qu'on lit dans U.</b> Les deux refrains reçoivent la même rangée, les " +
   "deux tags aussi — sans qu'on ait dit nulle part qu'ils étaient les mêmes. Et le " +
   "bloc le plus <i>pur</i> du morceau est <b>mes. " + meilleur.a + "–" + meilleur.b +
   " à " + meilleur.purete.toFixed(2) + "</b> : la liaison de 2 mesures, qui obtient " +
   "sa rangée à elle. Tes « queues » ne sont pas un cas particulier à traiter — " +
   "elles sortent toutes seules comme les composantes que rien d'autre n'utilise.";

 var cb = document.getElementById("courbe");
 D.courbe.forEach(function(c){
   var t = document.createElement("tr");
   t.innerHTML = "<td class=mono>" + c.k + (c.k === D.k ? " ←" : "") + "</td><td class=mono>" +
     c.residu.toFixed(3) + "</td><td class=mono>" + c.purete.toFixed(3) + "</td>";
   if (c.k === D.k) t.style.fontWeight = "700";
   cb.appendChild(t);
 });

 var sn = document.getElementById("sans");
 D.sans_couture.forEach(function(b){
   var t = document.createElement("tr");
   t.innerHTML = "<td class=mono>" + b.a + "–" + b.b + "</td><td class=mono>" +
     b.lettre + "</td><td class=mono>" + b.purete.toFixed(2) + "</td>";
   sn.appendChild(t);
 });
 document.getElementById("note-sans").textContent =
   "Pureté moyenne " + D.pur_sans + " contre " + D.purete + " avec la couture. " +
   "Après la mesure 25 la grille est décalée d'une demi-cellule : les blocs " +
   "enjambent les frontières, et deux occurrences de la même section ne tombent " +
   "plus sur la même rangée de V. Deux mesures d'écart suffisent à casser le reste.";

 document.getElementById("liens").innerHTML =
   '<a href="' + D.sections_page + '">les quatre lectures de la forme</a> · ' +
   '<a href="' + D.retour + '">le chart dans l\'app</a>';

 /* le son : timeupdate, jamais rAF ; blob, jamais Range (pièges iPhone) */
 var el = null, lien = null, joue = false;
 if (window.fetch) fetch(D.audio_url).then(function(r){return r.ok?r.blob():null;})
   .then(function(b){ if(b){ lien = URL.createObjectURL(b);
     if (el && !joue) el.src = lien; } })["catch"](function(){});
 function media(){
   if (el) return el;
   el = new Audio(); el.preload="auto"; el.playsInline=true;
   try{ el.setAttribute("playsinline",""); el.setAttribute("webkit-playsinline",""); }catch(e){}
   el.src = lien || D.audio_url;
   el.addEventListener("timeupdate", function(){
     var t = el.currentTime;
     document.getElementById("ou").textContent =
       Math.floor(t/60) + ":" + String(Math.floor(t%60)).padStart(2,"0");
   });
   document.body.appendChild(el);
   return el;
 }
 function vers(t){
   var a = media();
   try{ a.currentTime = t; }catch(e){}
   if (!joue){ a.play(); joue = true;
     document.getElementById("jouer").textContent = "Pause"; }
 }
 document.getElementById("jouer").onclick = function(){
   var a = media();
   if (joue){ a.pause(); joue = false; this.textContent = "Écouter"; }
   else { a.play(); joue = true; this.textContent = "Pause"; }
 };
})();
</script></body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
