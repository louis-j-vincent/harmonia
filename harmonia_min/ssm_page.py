"""harmonia_min/ssm_page.py — la matrice SSM d'un morceau, avec tête de lecture.

Louis, 2026-08-16 : « je veux une matrice ssm avec playhead cliquable ».

Les pages SSM du projet (`/plots/ssm_zoo.html`, `/reports/ssm_harmonic.html`)
sont des images : on y voit la structure, on ne peut pas l'ENTENDRE. Celle-ci
est la même matrice, rendue dans un canvas, avec deux gestes :

    on clique n'importe où      -> le son saute à la colonne cliquée
    ça joue                     -> la croix de lecture suit la musique

et, sous la matrice, la LIGNE de la tête de lecture — « ce qui ressemble à
maintenant », mesure par mesure, cliquable elle aussi. C'est elle qui rend la
page utile plutôt que jolie : un pic à la mesure 41 quand on est à la 9 se
vérifie à l'oreille en un tap.

LA MATRICE EST CELLE DE LA PROD. `harmonic_sections.ssm` — les vecteurs
chord-tone, ceux où Si♭ est plus près de Sol m que de Fa (règle de Louis,
2026-07-30, catégorique : jamais de SSM fondamentale-seule). Pas de second
scorer de similarité : cette page doit pouvoir contredire le chart, pas
inventer une deuxième vérité (même règle que `section_tool.py`).

LE GRAIN EST LA DEMI-MESURE, celui que Louis a demandé pour le zoo le
2026-08-10. La grille de mesures vient du CHART, donc elle porte déjà son
« Set bar 1 » : la matrice et l'app comptent les mesures pareil.

L'ÉTIREMENT DE CONTRASTE EST DIT. Les valeurs sont ré-étalées entre le 5e et
le 99e centile des cases hors-diagonale, sinon un morceau à boucle unique
(Stand By Me) sort uniformément foncé et ne montre rien. La page affiche la
plage réelle sous la matrice : un étirement muet peut faire passer du bruit
pour de la structure.

CE QUE ÇA NE FAIT PAS. Aucune détection, aucun score, aucune frontière
proposée : les traits rouges sont ceux que Louis a lui-même validés
(`state/sections/<stem>.json`) ou, à défaut, ceux que le chart affiche
aujourd'hui — et la page dit lequel des deux. Elle ne modifie rien non plus :
c'est une page de lecture, pas un outil d'édition (pour éditer, c'est
`/soudure/<file>`).

Deux limites à connaître avant de lire un chiffre sur cette page. (1) Les
cases AU-DESSUS du 99e centile sont écrêtées : elles s'affichent toutes à la
borne haute, donc une valeur lue à cette borne veut dire « au moins ça », pas
« exactement ça ». (2) Un octet par case : la précision affichée est de
±(haut−bas)/255, soit ~0,004 — assez pour comparer deux cases, pas pour
publier un seuil.
"""
from __future__ import annotations

import base64
import json
import logging
from pathlib import Path

import numpy as np

log = logging.getLogger(__name__)

PKG = Path(__file__).resolve().parent
SECTIONS_DIR = PKG / "state" / "sections"

#: Une seule teinte, clair -> foncé : une similarité est une MAGNITUDE. Reprise
#: telle quelle de `scripts/ssm_zoo.py`, pour que les deux pages se lisent avec
#: le même œil. Un arc-en-ciel inventerait des frontières là où la valeur monte
#: régulièrement.
CMAP = ["#fbf7ec", "#cfe0ea", "#84b3cf", "#3d7fa6", "#1b4a6b", "#0d2437"]

#: Le rouge des frontières validées, celui du zoo.
COULEUR_FRONTIERE = "#b4472c"

#: Cases par mesure. 2 = la demi-mesure demandée pour le zoo.
PAR_MESURE = 2


def _demi_mesures(grid: list[float]) -> list[float]:
    """Les bornes de demi-mesures — le grain de la page."""
    e: list[float] = []
    for b in range(len(grid) - 1):
        e += [float(grid[b]), 0.5 * (float(grid[b]) + float(grid[b + 1]))]
    e.append(float(grid[-1]))
    return e


def _frontieres(chart: dict, stem: str) -> tuple[list[dict], str]:
    """Les frontières à tracer, et D'OÙ elles viennent.

    L'annotation à la main de Louis d'abord (`state/sections/`), le chart
    ensuite. La page l'affiche : lire « le substrat sépare tes sections » sur
    des frontières que le détecteur a lui-même posées serait circulaire.
    """
    f = SECTIONS_DIR / f"{stem}.json"
    if f.exists():
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
            secs = [s for s in (doc.get("sections") or [])
                    if isinstance(s, dict) and "b0" in s]
            if secs:
                secs.sort(key=lambda s: s["b0"])
                return ([{"label": str(s.get("label") or "?"),
                          "mesure": int(s["b0"])} for s in secs],
                        "tes annotations")
        except (OSError, ValueError):
            log.warning("ssm_page: %s illisible, on retombe sur le chart", f)
    out = [{"label": str(s.get("label") or "?"), "mesure": int(b0)}
           for s in (chart.get("sections") or [])
           for b0, _b1 in (s.get("barRanges") or [])]
    out.sort(key=lambda s: s["mesure"])
    return out, "le chart"


def donnees(chart: dict, audio_dir=None) -> dict | None:
    """Tout ce que la page affiche — ou None si le chart n'a pas de quoi.

    La matrice part en base64 d'un octet par case (n² octets) : 35 ko pour un
    morceau de 80 mesures, contre une image PNG qu'il faudrait ensuite
    réaligner au pixel près sur la tête de lecture. Un seul système de
    coordonnées, donc aucun décalage possible entre ce qu'on voit et ce qu'on
    clique.
    """
    from harmonia_min import harmonic_sections as HS
    from harmonia_min import musx as _musx

    grid = chart.get("barGrid") or []
    if len(grid) < 5:
        return None
    stem = Path(chart.get("audio_url") or "").stem
    audio = (audio_dir or Path("docs/audio")) / f"{stem}.m4a"
    if not stem or not audio.exists():
        return None

    bornes = _demi_mesures(grid)
    triad = _musx.frame_posteriors(audio)[0]
    S = np.asarray(HS.ssm(triad, bornes), dtype=float)
    n = len(S)

    # L'étirement se lit sur les cases HORS-diagonale : la diagonale vaut 1
    # par construction et tirerait le haut de l'échelle à elle toute seule.
    hors = S[~np.eye(n, dtype=bool)]
    lo, hi = (float(np.quantile(hors, 0.05)), float(np.quantile(hors, 0.99))) \
        if hors.size else (0.0, 1.0)
    if hi - lo < 1e-6:
        lo, hi = float(S.min()), max(float(S.max()), float(S.min()) + 1e-6)
    q = np.clip((S - lo) / (hi - lo), 0.0, 1.0)
    octets = (q * 255.0 + 0.5).astype(np.uint8).tobytes()

    front, source = _frontieres(chart, stem)
    n_mes = len(grid) - 1
    front = [f for f in front if 0 <= f["mesure"] < n_mes]

    mesure1 = None
    try:
        t1 = float(chart["bar1"])
        mesure1 = min(range(n_mes), key=lambda b: abs(float(grid[b]) - t1))
    except (KeyError, TypeError, ValueError):
        pass

    return {
        "titre": chart.get("title") or stem,
        "stem": stem,
        "file": chart.get("file") or "",
        "audio_url": chart.get("audio_url") or f"/audio/{stem}.m4a",
        "n": n,
        "par_mesure": PAR_MESURE,
        "n_mesures": n_mes,
        "t": [round(x, 3) for x in bornes],
        "m": base64.b64encode(octets).decode("ascii"),
        "plage": [round(float(S.min()), 3), round(float(S.max()), 3)],
        "etirement": [round(lo, 3), round(hi, 3)],
        "frontieres": front,
        "source_frontieres": source,
        "mesure1": mesure1,
        "cmap": CMAP,
        "couleur_frontiere": COULEUR_FRONTIERE,
    }


def page_html(chart: dict, audio_dir=None, *, base_url: str = "") -> str | None:
    """La page complète, autonome, prête à écrire sur disque ou à servir.

    `base_url` préfixe l'audio et le lien retour : vide quand la page est
    servie par le serveur lui-même, `http://100.89.209.63:7772` quand elle est
    écrite dans `docs/plots/` et ouverte depuis un téléphone (les liens
    file:// ne marchent pas pour Louis — 2026-08-07).
    """
    d = donnees(chart, audio_dir=audio_dir)
    if d is None:
        return None
    d["audio_url"] = base_url + d["audio_url"]
    d["retour"] = (base_url + "/?open=" + d["file"]) if d["file"] else ""
    charge = json.dumps(d, ensure_ascii=False, separators=(",", ":"))
    return _GABARIT.replace("__DONNEES__", charge).replace(
        "__TITRE__", _echappe(d["titre"]))


def _echappe(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;")
            .replace(">", "&gt;").replace('"', "&quot;"))


# ── la page ─────────────────────────────────────────────────────────────────
# Autonome : pas de CDN, pas de dépendance, tout tient dans le fichier — c'est
# la règle de toutes les pages de docs/plots/.
#
# LES DEUX PIÈGES IPHONE, tous les deux déjà payés (voir la note « iPhone
# no-sound bug » de app_shell.html et docs/plots/soudure.html) :
#   1. AUCUNE boucle requestAnimationFrame. Une boucle rAF qui écrit dans le
#      DOM empêche le moteur audio de WebKit de démarrer : `currentTime` reste
#      à 0 et le son ne sort qu'une fois l'onglet en arrière-plan. La tête de
#      lecture est donc battue par `timeupdate`, l'horloge du média.
#   2. Le fichier est téléchargé en entier et rejoué depuis un `blob:` — le
#      lecteur média d'iOS ne bufferise parfois jamais un fichier servi en
#      Range (rafale de 206 jetés).
# Et jamais de `play().catch(() => {})` muet : une veille dit à l'écran si
# l'horloge n'a pas bougé.
_GABARIT = r"""<!doctype html>
<html lang="fr"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>SSM — __TITRE__</title>
<style>
  :root{
    --papier:#f7f3e9; --encre:#1c1c1c; --pale:#6f6a60; --trait:#ddd5c4;
    --carte:#fffdf7; --accent:#8a2b2b;
  }
  @media (prefers-color-scheme: dark){
    :root{ --papier:#17171a; --encre:#ece8e0; --pale:#918c83; --trait:#33323a;
           --carte:#1f1f24; --accent:#d4735e; }
  }
  *{ box-sizing:border-box; }
  body{ margin:0; padding:18px 16px calc(28px + env(safe-area-inset-bottom));
        background:var(--papier); color:var(--encre);
        font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
        max-width:820px; margin-inline:auto; }
  a{ color:var(--accent); }
  h1{ font:italic 600 23px/1.25 Georgia,"Times New Roman",serif; margin:0 0 2px; }
  .sous{ color:var(--pale); font-size:13.5px; margin:0 0 14px; }
  .lede{ font:italic 14px/1.6 Georgia,serif; color:var(--pale);
         margin:0 0 16px; }
  .carte{ background:var(--carte); border:1px solid var(--trait);
          border-radius:14px; padding:12px; }
  #enveloppe{ position:relative; width:100%; }
  canvas{ display:block; width:100%; height:auto; border-radius:8px;
          touch-action:manipulation; cursor:crosshair; }
  canvas.regle{ border-radius:0; cursor:default; margin-top:3px; }
  #bulle{ position:absolute; pointer-events:none; opacity:0;
          transform:translate(-50%,-130%); background:var(--encre);
          color:var(--papier); font:600 11.5px ui-monospace,Menlo,monospace;
          padding:4px 7px; border-radius:6px; white-space:nowrap;
          transition:opacity .12s; }
  .legende{ display:flex; justify-content:space-between; gap:10px;
            font:500 11.5px ui-monospace,Menlo,monospace; color:var(--pale);
            margin-top:8px; flex-wrap:wrap; }
  .titraille{ font:600 10.5px sans-serif; letter-spacing:.08em;
              text-transform:uppercase; color:var(--pale); margin:18px 0 6px; }
  #transport{ display:flex; align-items:center; gap:12px; margin-top:14px; }
  button{ font:600 14px sans-serif; border:1px solid var(--trait);
          background:var(--carte); color:var(--encre); border-radius:11px;
          padding:11px 16px; cursor:pointer; min-height:44px; }
  button.principal{ background:var(--accent); color:#fff; border-color:transparent;
                    min-width:104px; }
  #horloge{ font:600 14px ui-monospace,Menlo,monospace; color:var(--pale);
            font-variant-numeric:tabular-nums; }
  #alerte{ display:none; margin-top:10px; padding:10px 12px; border-radius:10px;
           background:var(--accent); color:#fff; font-size:13.5px; }
  .pied{ margin-top:22px; font-size:13px; color:var(--pale); }
</style>
</head><body>
<script>window.SSM = __DONNEES__;</script>

<h1 id="titre"></h1>
<p class="sous" id="sous"></p>
<p class="lede">Clique n'importe où dans la matrice : le son saute à cette
mesure. Sous la matrice, la ligne de la tête de lecture — <b>ce qui ressemble
à maintenant</b> — se clique aussi.</p>

<div class="carte">
  <div id="enveloppe">
    <canvas id="mat"></canvas>
    <div id="bulle"></div>
  </div>
  <canvas class="regle" id="regle1"></canvas>
  <div class="legende">
    <span id="echelle"></span>
    <span id="src"></span>
  </div>
</div>

<div class="titraille">Ce qui ressemble à maintenant</div>
<div class="carte">
  <canvas id="ligne"></canvas>
  <canvas class="regle" id="regle2"></canvas>
  <div class="legende"><span id="ou"></span></div>
</div>

<div id="transport">
  <button class="principal" id="jouer">Écouter</button>
  <span id="horloge">0:00</span>
</div>
<div id="alerte"></div>
<p class="pied" id="pied"></p>

<script>
(function(){
  "use strict";
  var D = window.SSM, N = D.n, PM = D.par_mesure;

  /* la matrice, dépliée une fois */
  var brut = atob(D.m), M = new Uint8Array(N * N);
  for (var i = 0; i < M.length; i++) M[i] = brut.charCodeAt(i);

  /* Les octets sont la similarité ÉTIRÉE (5e–99e centile) : ce qu'on affiche
     doit être la vraie, sinon la page annonce 1.00 là où le cosinus vaut 0.96
     et l'étirement se met à mentir au lieu d'aider. */
  var LO = D.etirement[0], HI = D.etirement[1];
  function val(i, j){ return LO + M[i * N + j] / 255 * (HI - LO); }

  /* la palette, échantillonnée une fois en 256 teintes */
  function lit(h){ return [parseInt(h.substr(1,2),16), parseInt(h.substr(3,2),16),
                           parseInt(h.substr(5,2),16)]; }
  var stops = D.cmap.map(lit), PAL = new Uint8Array(256 * 3);
  for (var v = 0; v < 256; v++){
    var x = v / 255 * (stops.length - 1), k = Math.min(stops.length - 2, Math.floor(x)),
        f = x - k;
    for (var c = 0; c < 3; c++)
      PAL[v * 3 + c] = Math.round(stops[k][c] + f * (stops[k + 1][c] - stops[k][c]));
  }

  /* la matrice en pixels, une seule fois : ensuite on ne fait que la recopier
     agrandie, ce qui rend le redessin de la tête de lecture gratuit */
  var fond = document.createElement("canvas");
  fond.width = fond.height = N;
  (function(){
    var g = fond.getContext("2d"), im = g.createImageData(N, N), p = im.data;
    for (var i = 0; i < N * N; i++){
      var v = M[i] * 3;
      p[i*4] = PAL[v]; p[i*4+1] = PAL[v+1]; p[i*4+2] = PAL[v+2]; p[i*4+3] = 255;
    }
    g.putImageData(im, 0, 0);
  })();

  var mat = document.getElementById("mat"), gm = mat.getContext("2d");
  var lig = document.getElementById("ligne"), gl = lig.getContext("2d");
  var regles = [document.getElementById("regle1"), document.getElementById("regle2")];
  var bulle = document.getElementById("bulle");
  var cote = 0, DPR = Math.min(2, window.devicePixelRatio || 1);
  var tete = 0;                      // la case sous la tête de lecture
  var PAS = 8;                       // l'écart des repères, en mesures

  function css(n){ return getComputedStyle(document.documentElement)
                     .getPropertyValue(n).trim(); }

  /* Une tête de lecture doit se voir sur une case claire ET sur une case
     foncée : on la trace deux fois, un halo couleur papier dessous, le trait
     d'accent dessus. Sans le halo elle disparaît dans les colonnes claires —
     constaté sur This Love, où la mesure 61 est justement une bande pâle. */
  function trait(g, x0, y0, x1, y1, large){
    g.lineCap = "butt";
    g.strokeStyle = css("--papier"); g.lineWidth = large + DPR * 2.2;
    g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke();
    g.strokeStyle = css("--accent"); g.lineWidth = large;
    g.beginPath(); g.moveTo(x0, y0); g.lineTo(x1, y1); g.stroke();
  }

  function taille(){
    var l = document.getElementById("enveloppe").clientWidth;
    if (!l) return;
    cote = l;
    mat.width = Math.round(l * DPR); mat.height = Math.round(l * DPR);
    mat.style.height = l + "px";
    lig.width = Math.round(l * DPR); lig.height = Math.round(54 * DPR);
    lig.style.width = "100%"; lig.style.height = "54px";
    // assez de repères pour se situer, assez peu pour rester lisible à 390 px
    PAS = 8;
    while (D.n_mesures / PAS > Math.max(4, l / 58)) PAS *= 2;
    regles.forEach(function(r){
      r.width = Math.round(l * DPR); r.height = Math.round(15 * DPR);
      r.style.width = "100%"; r.style.height = "15px";
    });
    dessine();
  }

  /* La règle des mesures, partagée par la matrice et par la ligne : même axe
     des x, donc un pic repéré sur la ligne se retrouve sur la matrice. */
  function dessineRegle(){
    regles.forEach(function(r){
      var g = r.getContext("2d"), L = r.width, H = r.height, u = L / N;
      g.clearRect(0, 0, L, H);
      g.font = (10 * DPR) + "px ui-monospace,Menlo,monospace";
      g.textBaseline = "top";
      for (var m = 0; m < D.n_mesures; m += PAS){
        var x = m * PM * u;
        g.fillStyle = css("--trait");
        g.fillRect(x, 0, Math.max(1, DPR * 0.8), H * 0.42);
        g.fillStyle = css("--pale");
        g.textAlign = m === 0 ? "left" : "center";
        g.fillText(String(m + 1), m === 0 ? 2 : x, H * 0.44);
      }
      var p = (tete + 0.5) * u;
      g.fillStyle = css("--accent");
      g.beginPath(); g.moveTo(p, 0); g.lineTo(p - DPR * 3.2, H * 0.55);
      g.lineTo(p + DPR * 3.2, H * 0.55); g.closePath(); g.fill();
    });
  }

  function dessine(){
    if (!cote) return;
    var S = cote * DPR, u = S / N;
    gm.imageSmoothingEnabled = false;
    gm.clearRect(0, 0, S, S);
    gm.drawImage(fond, 0, 0, N, N, 0, 0, S, S);

    /* les frontières : deux traits fins par section, un par axe */
    gm.strokeStyle = D.couleur_frontiere; gm.lineWidth = Math.max(1, DPR * 0.8);
    gm.globalAlpha = 0.75;
    D.frontieres.forEach(function(f){
      var x = f.mesure * PM * u;
      gm.beginPath(); gm.moveTo(x, 0); gm.lineTo(x, S);
      gm.moveTo(0, x); gm.lineTo(S, x); gm.stroke();
    });
    gm.globalAlpha = 1;

    /* la mesure 1 de Louis, si elle est posée */
    if (D.mesure1){
      var x1 = D.mesure1 * PM * u;
      gm.strokeStyle = css("--encre"); gm.lineWidth = Math.max(1.5, DPR);
      gm.setLineDash([DPR * 5, DPR * 4]);
      gm.beginPath(); gm.moveTo(x1, 0); gm.lineTo(x1, S);
      gm.moveTo(0, x1); gm.lineTo(S, x1); gm.stroke();
      gm.setLineDash([]);
    }

    /* la croix de lecture */
    var p = (tete + 0.5) * u, e = Math.max(2, DPR * 1.6);
    trait(gm, p, 0, p, S, e);
    trait(gm, 0, p, S, p, e);
    gm.fillStyle = css("--papier");
    gm.beginPath(); gm.arc(p, p, Math.max(5, DPR * 4.4), 0, 6.284); gm.fill();
    gm.fillStyle = css("--accent");
    gm.beginPath(); gm.arc(p, p, Math.max(3, DPR * 2.8), 0, 6.284); gm.fill();

    dessineLigne();
    dessineRegle();
  }

  /* la LIGNE de la tête de lecture : S[tete, :], le profil « qu'est-ce qui
     ressemble à la mesure où j'en suis » */
  function dessineLigne(){
    var L = lig.width, H = lig.height, u = L / N;
    gl.clearRect(0, 0, L, H);
    for (var j = 0; j < N; j++){
      var v = M[tete * N + j] * 3;
      gl.fillStyle = "rgb(" + PAL[v] + "," + PAL[v+1] + "," + PAL[v+2] + ")";
      gl.fillRect(j * u, 0, Math.ceil(u), H);
    }
    gl.strokeStyle = D.couleur_frontiere; gl.lineWidth = Math.max(1, DPR * 0.8);
    gl.globalAlpha = 0.75;
    D.frontieres.forEach(function(f){
      var x = f.mesure * PM * u;
      gl.beginPath(); gl.moveTo(x, 0); gl.lineTo(x, H); gl.stroke();
    });
    gl.globalAlpha = 1;
    trait(gl, (tete + 0.5) * u, 0, (tete + 0.5) * u, H, Math.max(2, DPR * 1.6));
    /* la meilleure ressemblance ailleurs : le tap suivant, nommé */
    var loin = 4 * PM, best = -1, ou = -1;
    for (var k = 0; k < N; k++){
      if (Math.abs(k - tete) < loin) continue;
      if (M[tete * N + k] > best){ best = M[tete * N + k]; ou = k; }
    }
    document.getElementById("ou").textContent =
      "mesure " + (Math.floor(tete / PM) + 1) +
      (ou < 0 ? "" : " · elle ressemble le plus à la mesure " +
        (Math.floor(ou / PM) + 1) + " (" + val(tete, ou).toFixed(2) +
        ") — tape le pic pour l'écouter");
  }

  /* ── le son ─────────────────────────────────────────────────────────────
     Deux pièges iPhone, déjà payés ailleurs (voir l'en-tête du module) :
     la tête est battue par `timeupdate`, jamais par rAF, et le fichier est
     rejoué depuis un blob parce que le lecteur média d'iOS ne bufferise pas
     toujours un fichier servi en Range. */
  var el = null, lien = null, joue = false, garde = null;

  function dit(msg){
    var a = document.getElementById("alerte");
    a.textContent = msg; a.style.display = msg ? "block" : "none";
  }

  function precharge(){
    if (!window.fetch) return;
    fetch(D.audio_url).then(function(r){ return r.ok ? r.blob() : null; })
      .then(function(b){ if (!b) return;
        lien = URL.createObjectURL(b);
        if (el && !joue) el.src = lien; })
      ["catch"](function(){});
  }

  function media(){
    if (el) return el;
    el = new Audio();
    el.preload = "auto";
    el.playsInline = true;
    try { el.setAttribute("playsinline", "");
          el.setAttribute("webkit-playsinline", ""); } catch(e){}
    el.src = lien || D.audio_url;
    el.addEventListener("timeupdate", tic);
    el.addEventListener("seeked", tic);
    el.addEventListener("ended", function(){ joue = false; bouton(); });
    el.addEventListener("error", function(){
      dit("le fichier audio n'a pas pu être lu — " + D.audio_url); });
    document.body.appendChild(el);   // posé dans la page : inspectable
    return el;
  }

  function mmss(t){
    t = Math.max(0, Math.floor(t || 0));
    return Math.floor(t / 60) + ":" + String(t % 60).padStart(2, "0");
  }

  function caseDe(t){
    var T = D.t, lo = 0, hi = N - 1, k = 0;
    while (lo <= hi){ var mid = (lo + hi) >> 1;
      if (T[mid] <= t){ k = mid; lo = mid + 1; } else hi = mid - 1; }
    return Math.max(0, Math.min(N - 1, k));
  }

  function tic(){
    if (!el) return;
    document.getElementById("horloge").textContent = mmss(el.currentTime);
    var k = caseDe(el.currentTime);
    if (k !== tete){ tete = k; dessine(); }        // on n'écrit que si ça bouge
  }

  function bouton(){
    document.getElementById("jouer").textContent = joue ? "Pause" : "Écouter";
  }

  /* Si au bout d'1,5 s l'horloge n'a pas bougé, on le DIT — une page muette
     ne doit pas faire semblant de jouer. */
  function veille(essai){
    clearTimeout(garde);
    if (!el) return;
    var depart = el.currentTime, n = essai || 0;
    garde = setTimeout(function(){
      if (!joue || !el) return;
      if (el.currentTime > depart + 0.05) return;
      if (lien && el.src !== lien){                 // le blob est arrivé
        var t = el.currentTime;
        el.src = lien;
        try { el.currentTime = t; } catch(e){}
        el.play(); veille(n + 1); return;
      }
      if (n < 2){ el.play(); veille(n + 1); return; }
      dit("le son ne démarre pas sur cet appareil — l'horloge est restée à " +
          mmss(el.currentTime) + ".");
    }, 1500);
  }

  function vers(t, lance){
    var a = media();
    try { a.currentTime = Math.max(0, t); } catch(e){}
    tete = caseDe(t); dessine();
    document.getElementById("horloge").textContent = mmss(t);
    if (lance && !joue){ bascule(true); } else if (joue){ veille(0); }
  }

  function bascule(force){
    var a = media();
    if (joue && force !== true){ a.pause(); joue = false; clearTimeout(garde); }
    else {
      dit("");
      var p = a.play();
      if (p && p["catch"]) p["catch"](function(e){
        joue = false; bouton();
        dit("le navigateur a refusé la lecture (" + (e && e.name || e) + ")");
      });
      joue = true; veille(0);
    }
    bouton();
  }

  /* ── les gestes ─────────────────────────────────────────────────────── */
  function colonne(ev, cible){
    var r = cible.getBoundingClientRect();
    var x = (ev.touches ? ev.touches[0].clientX : ev.clientX) - r.left;
    return Math.max(0, Math.min(N - 1, Math.floor(x / r.width * N)));
  }

  mat.addEventListener("click", function(ev){ vers(D.t[colonne(ev, mat)], true); });
  lig.addEventListener("click", function(ev){ vers(D.t[colonne(ev, lig)], true); });

  mat.addEventListener("pointermove", function(ev){
    var r = mat.getBoundingClientRect();
    var j = colonne(ev, mat);
    var i = Math.max(0, Math.min(N - 1,
              Math.floor((ev.clientY - r.top) / r.height * N)));
    bulle.style.left = (ev.clientX - r.left) + "px";
    bulle.style.top = (ev.clientY - r.top) + "px";
    bulle.style.opacity = 1;
    bulle.textContent = "mes. " + (Math.floor(j / PM) + 1) + " ↔ " +
      (Math.floor(i / PM) + 1) + "  ·  " + val(i, j).toFixed(2);
  });
  mat.addEventListener("pointerleave", function(){ bulle.style.opacity = 0; });

  document.getElementById("jouer").addEventListener("click", function(){ bascule(); });
  document.addEventListener("keydown", function(ev){
    if (ev.code === "Space"){ ev.preventDefault(); bascule(); }
  });

  /* ── l'habillage ────────────────────────────────────────────────────── */
  document.getElementById("titre").textContent = D.titre;
  document.getElementById("sous").textContent =
    D.n_mesures + " mesures · grain demi-mesure · matrice chord-tone (celle de la prod)";
  document.getElementById("echelle").textContent =
    "contraste étiré " + D.etirement[0].toFixed(2) + "–" + D.etirement[1].toFixed(2) +
    " (valeurs réelles " + D.plage[0].toFixed(2) + "–" + D.plage[1].toFixed(2) + ")";
  document.getElementById("src").textContent =
    "traits rouges : " + D.source_frontieres +
    (D.mesure1 ? " · pointillés : ta mesure 1" : "");
  var pied = document.getElementById("pied");
  if (D.retour){
    var a = document.createElement("a");
    a.href = D.retour; a.textContent = "‹ ouvrir le chart dans l'app";
    pied.appendChild(a);
  }

  window.addEventListener("resize", taille);
  taille();
  precharge();
})();
</script>
</body></html>
"""
