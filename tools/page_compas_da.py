"""Le compas en cascade, avec la DA du compas de l'app.

Louis, 2026-09-17 : « c'est vraiment top, maintenant refais moi ca mais en
reprenant la DA du compass qu'on a actuellement ».

Ce qui est REPRIS de `annotate.buildCompass`, à la valeur près — la DA de ce
compas n'est pas une décoration, c'est une suite de décisions payées :

  * la PALETTE de l'app (`ui/kit.js`) : papier #f7f3e9, carte #fffdf6, encre
    #1c1c1c, filets #e5dcc6, accent #8a2b2b, bleu #2a6fb0 — et le jeu sombre ;
  * les ANGLES SONT LE CERCLE DES QUINTES, `pc = (i × 7) mod 12` à
    `-90 + i × 30` degrés. C'est « toute la revendication du compas » (le
    commentaire du code le dit) : l'encombrement se résout RADIALEMENT, jamais
    en tordant l'angle ;
  * la jante, ses douze lettres à `R + Sz×0,085`, ses petits points à r=2, et
    la tonique cerclée d'accent ;
  * le MOYEU porte l'accord courant et rien d'autre — « re-drawing it as an
    orb on the rim said nothing and stole a spoke » (Louis, 2026-08-08). Seules
    les VRAIES alternatives orbitent ;
  * l'aire de l'orbe suit √proba, et son rayon plancher est 22 px — le seuil
    TACTILE (44 px de diamètre), pas une valeur de style ;
  * jamais de chevauchement : deux orbes sur des rayons voisins se décalent le
    long de leur propre rayon, le plus gros contre le moyeu, le suivant vers la
    jante ;
  * le glyphe Georgia italique, qualité en petit calée sur la ligne de base.

CE QUE LA CASCADE AJOUTE, et pourquoi ça tient dans cette DA sans la tordre :
aux niveaux 2 à 5 les options ne sont plus des accords mais des NOTES qu'on
ajoute — la 7e, la 9e, la 11e, la 13e. Une note a une classe de hauteur, donc
un angle sur le cercle des quintes. Chaque orbe se pose donc à l'angle de la
note qu'il ajoute, et l'accord se construit en se déployant sur le cercle.
L'angle reste ce qu'il a toujours été : la place de la note dans le cycle.

Et « ne rien ajouter » n'est plus un orbe : c'est le MOYEU qu'on touche pour
valider. C'est la doctrine du compas existant appliquée à la lettre, et ça
supprime au passage les deux orbes qui portaient la même étiquette.

    .venv/bin/python -m tools.page_compas_da
"""
from __future__ import annotations

import html
import json

from harmonia.settings import SETTINGS

DATA = SETTINGS.repo / "scratchpad" / "cascade.json"
OUT = SETTINGS.repo / "docs" / "plots" / "compas_da.html"

#: repris de `span_rescore.SUG_FLOOR` — aucun seuil nouveau ici.
SUGGERE = 0.125


def page(D: dict) -> str:
    e = html.escape
    ac = D["accords"]
    rubans = "".join(
        f"""<button class="slot" data-i="{i}" type="button">
             <span class="sb">{a['bar']}.{a['beat']}</span>
             <span class="sq">{e(a['ecrit'])}</span></button>"""
        for i, a in enumerate(ac))

    return f"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Le Compas De L'App</title>
<style>
/* LA PALETTE DE L'APP, valeur pour valeur (`harmonia/static/ui/kit.js`) :
   LIGHT/DARK plus les quatre teintes partagées entre les deux thèmes. */
 :root{{--bg:#e7e0d0;--paper:#f7f3e9;--card:#fffdf6;--ink:#1c1c1c;--rule:#b9b09a;
  --faint:#8a8371;--line:#e5dcc6;--deep:#2a2622;
  --accent:#8a2b2b;--green:#1f8a5b;--amber:#c58a2e;--blue:#2a6fb0;}}
 @media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{
  --bg:#151210;--paper:#211c17;--card:#2a241d;--ink:#f2ebde;--rule:#5b5344;
  --faint:#a99f8c;--line:#39332a;--deep:#0e0c0a;}}}}
 :root[data-theme="dark"]{{--bg:#151210;--paper:#211c17;--card:#2a241d;--ink:#f2ebde;
  --rule:#5b5344;--faint:#a99f8c;--line:#39332a;--deep:#0e0c0a;}}
 *{{box-sizing:border-box}}
 body{{background:var(--paper);color:var(--ink);margin:0 auto;max-width:560px;
  font:400 15px/1.5 -apple-system,BlinkMacSystemFont,'Segoe UI',system-ui,sans-serif;
  padding:16px 14px 40px}}
 h1{{font:600 22px/1.2 Georgia,'Times New Roman',serif;margin:0;letter-spacing:-.01em}}
 .kick{{font:italic 13px/1.5 Georgia,'Times New Roman',serif;color:var(--faint);margin:3px 0 0}}
 .eyebrow{{font:600 10.5px/1 -apple-system,system-ui,sans-serif;letter-spacing:.1em;
  text-transform:uppercase;color:var(--faint);margin-bottom:7px}}

 button{{font:600 13px/1 -apple-system,system-ui,sans-serif;border-radius:9px;
  padding:9px 12px;min-height:40px;cursor:pointer;border:1px solid var(--rule);
  background:var(--card);color:var(--ink);-webkit-tap-highlight-color:transparent}}
 button:active{{transform:translateY(1px)}}
 button:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}

 .slots{{display:flex;gap:6px;overflow-x:auto;padding:4px 0 10px;-webkit-overflow-scrolling:touch}}
 .slot{{flex:0 0 auto;display:flex;flex-direction:column;align-items:flex-start;
  gap:1px;padding:6px 10px;min-width:64px}}
 .slot.on{{background:var(--accent);border-color:var(--accent);color:#f4eee2}}
 .sb{{font:500 9.5px/1 -apple-system,system-ui,sans-serif;opacity:.72}}
 .sq{{font:italic 600 14px/1 Georgia,serif}}

 .fil{{display:flex;gap:5px;align-items:center;flex-wrap:wrap;margin:8px 0 2px;min-height:32px}}
 .fil .pas{{padding:5px 10px;min-height:32px;font:italic 600 13px/1 Georgia,serif}}
 .fil .fleche{{color:var(--faint);font-size:13px}}
 .fil .niv{{margin-left:auto;font:500 11px/1 -apple-system,system-ui,sans-serif;color:var(--faint)}}

 .rond{{display:flex;justify-content:center;padding:4px 0 2px}}
 svg{{display:block;overflow:visible;touch-action:manipulation}}
 svg .orbe{{cursor:pointer}}
 svg .orbe:hover circle{{stroke-width:2.5}}

 .barre{{display:flex;gap:10px;align-items:center;flex-wrap:wrap;margin-top:6px}}
 .barre .av{{font:500 11.5px/1 -apple-system,system-ui,sans-serif;color:var(--faint)}}
 .barre .sp{{flex:1;min-width:4px}}
 .note{{background:var(--card);border:1px solid var(--line);border-radius:12px;
  padding:12px 14px;margin:16px 0}}
 .note p{{margin:7px 0;font-size:12.5px;color:var(--faint)}}
 .note b{{color:var(--ink)}}
 footer{{margin-top:20px;padding-top:12px;border-top:1px solid var(--line);
  font:500 11px/1.8 -apple-system,system-ui,sans-serif;color:var(--faint)}}
 code{{font:500 11.5px/1 ui-monospace,Menlo,monospace;background:var(--bg);
  border-radius:4px;padding:1px 4px}}
</style>

<div class="eyebrow">{e(D['titre'] or D['key'])} &middot; {e(D['tonalite'] or '')}</div>
<h1>Le compas</h1>
<p class="kick">on touche, on descend d'un niveau, et le choix passe au moyeu</p>

<div class="slots" id="slots">{rubans}</div>
<div class="fil" id="fil"></div>
<div class="rond" id="rond"></div>

<div class="barre">
  <span class="av" id="av"></span>
  <span class="sp"></span>
  <button id="play" type="button">&#9658; écouter</button>
  <button id="haut" type="button">&#8630; remonter</button>
</div>

<div class="note">
  <p><b>Le cercle des quintes vaut pour le premier cercle</b> — un orbe s'y
  pose à l'angle de sa fondamentale, et la jante s'allume sur les notes
  proposées. Aux niveaux suivants les options sont des degrés du même accord&nbsp;:
  l'angle n'a plus rien à dire, on les répartit à intervalles égaux.</p>
  <p><b>Le moyeu porte l'accord courant, et on le touche pour valider</b> —
  « ne rien ajouter » n'est pas une orbite. L'aire de l'orbe suit la
  probabilité&nbsp;; son rayon ne descend jamais sous 22&nbsp;px, qui est le seuil
  tactile et non une valeur de style. En pointillé&nbsp;: sous le seuil de
  suggestion, proposé quand même.</p>
  <p><b>Ce que la géométrie impose.</b> Quand un gros orbe et un petit tombent
  à 30° l'un de l'autre, tout rétrécit ensemble — c'est ce que fait déjà le
  compas de l'app, qui le dit dans son code&nbsp;: « deux orbes de 60&nbsp;px à
  30° dans un anneau de 115, c'est de la géométrie, pas un choix ». Un orbe peut
  alors passer sous les 44&nbsp;px tactiles. Et un rayon ne porte qu'UNE note&nbsp;:
  le <code>G-</code> à 5&nbsp;% masque le <code>G</code> à 3&nbsp;% — la cascade
  ne rattrape pas le passage majeur/mineur, c'est sa limite connue.</p>
</div>

<footer>
  palette, angles, moyeu, loi d'aire et plancher tactile repris de
  <code>annotate.buildCompass</code> &middot; têtes du modèle&nbsp;:
  <code>musx.frame_posteriors</code> &middot; version linéaire&nbsp;:
  <a href="/plots/cascade.html" style="color:var(--accent)">cascade</a>
</footer>

<script>
const D = {json.dumps(D, ensure_ascii=False, separators=(',', ':'))};
const SUGGERE = {SUGGERE};
const N = D.noms;
const TYPE_TAIL = ["", "-", "sus4", "sus2", "o", "+"];
const NS = "http://www.w3.org/2000/svg";
const mod = (n, m) => ((n % m) + m) % m;
const fifths = pc => mod(pc * 7, 12);          // `kit.js::fifthsIndex`
let sel = 0, chemin = [];

function sv(t, a){{ const n = document.createElementNS(NS, t);
  for (const k in a) n.setAttribute(k, a[k]); return n; }}
const css = v => getComputedStyle(document.documentElement).getPropertyValue(v).trim();
// l'aire suit la proba, donc le rayon suit sa racine (`kit.js::byArea`)
const byArea = (p, rMin, rMax) => rMin + Math.sqrt(Math.max(0, Math.min(1, p))) * (rMax - rMin);

function symbole(c){{
  const base = TYPE_TAIL[c.type] || "";
  const alt = [];
  if (c.neuf === 2) alt.push("#9"); if (c.neuf === 3) alt.push("b9");
  if (c.onze === 2) alt.push("#11");
  if (c.treize === 2) alt.push("b13");
  let q;
  if (c.toit === 0) q = base + (c.neuf === 1 ? "add9" : "");
  else if (c.toit === 3) q = base + (c.neuf === 1 ? "69" : "6");
  else {{
    let haut = 7;
    if (c.neuf === 1) haut = 9;
    if (c.onze === 1) haut = 11;
    if (c.treize === 1) haut = 13;
    if (c.type === 4) q = (c.toit === 2 ? "h7" : "o7");
    else if (c.toit === 1) q = (c.type === 1 ? "-^" : "^") + haut;
    else if (c.type === 2) q = haut + "sus4";
    else if (c.type === 3) q = haut + "sus2";
    else q = (c.type === 1 ? "-" : "") + haut;
  }}
  if (alt.length) q += alt.join("");
  return {{root: N.notes[c.root], q}};
}}

// le glyphe de l'app : Georgia italique, la qualité en petit sur la ligne de base
function glyphe(g, x, y, root, q, taille, couleur){{
  const t = sv("text", {{x, y, "text-anchor": "middle", "dominant-baseline": "central",
    "font-family": "Georgia,'Times New Roman',serif", "font-style": "italic",
    "font-weight": 600, "font-size": taille, fill: couleur}});
  t.appendChild(document.createTextNode(root));
  if (q) {{
    const s = sv("tspan", {{"font-size": Math.round(taille * 0.55), dy: taille * 0.10}});
    s.textContent = q; t.appendChild(s);
  }}
  g.appendChild(t);
}}

function etat(){{
  const c = {{root: 0, type: 0, toit: 0, neuf: 0, onze: 0, treize: 0}};
  for (const p of chemin) {{
    if (p.cle === 'base') {{ c.root = p.root; c.type = p.type; }} else c[p.cle] = p.i;
  }}
  return c;
}}

// L'INTERVALLE QUE CHAQUE OPTION AJOUTE — c'est lui qui donne son angle.
const IV = {{toit: [null, 11, 10, 9], neuf: [null, 2, 3, 1],
             onze: [null, 5, 6], treize: [null, 9, 8]}};

function niveau(){{
  const a = D.accords[sel], c = etat(), n = chemin.length;
  if (n === 0) return {{cle: 'base', titre: 'la base',
    // UN ORBE PAR CLASSE DE HAUTEUR, cinq au plus. Le compas dit lui-même
    // qu'« une lettre de la roue est une CLASSE DE HAUTEUR » : y poser trois
    // qualités du même SOL (`G-`, `G`, `Gsus4`) était mon extension, et elle
    // ne tient pas dans la géométrie — trois orbes TACTILES (44 px chacun)
    // empilés sur un rayon demandent 189 px quand la jante en offre 137, donc
    // la relaxation les rabotait à 27 px, sous le seuil tactile. On garde donc
    // la meilleure qualité par note.
    // CE QUE ÇA COÛTE, et il faut le dire : le `G-` à 5 % masque le `G` à 3 %
    // et le `Gsus4` à 2 %. La cascade ne les rattrape pas — son niveau 2
    // propose la 7e et la sixte, pas le passage majeur/mineur. C'est la limite
    // connue de cette démo.
    opts: Object.values(a.base.reduce((acc, b) => {{
      if (!acc[b.root] || b.c > acc[b.root].c) acc[b.root] = b;
      return acc;
    }}, {{}})).sort((x, y) => y.c - x.c).slice(0, 5)
      .map(b => ({{root: b.root, type: b.type, p: b.c, pc: b.root,
        sym: {{root: N.notes[b.root], q: TYPE_TAIL[b.type] || ''}}}}))}};
  if (n === 1) return {{cle: 'toit', titre: 'ce qui se pose dessus',
    opts: [3, 1, 2].map(i => ({{i, p: (i === 3 ? a.treize[1] : a.sev[i]),
      pc: mod(c.root + IV.toit[i], 12), sym: symbole({{...c, toit: i}})}}))}};
  if ((c.toit === 0 || c.toit === 3) && n >= 3) return null;
  const suite = [['neuf', 'la 9e', a.neuf], ['onze', "la 11e", a.onze],
                 ['treize', 'la 13e', a.treize]][n - 2];
  if (!suite) return null;
  return {{cle: suite[0], titre: suite[1],
    opts: suite[2].map((p, i) => i === 0 ? null : ({{i, p,
      pc: mod(c.root + IV[suite[0]][i], 12),
      sym: symbole(Object.assign({{...c}}, {{[suite[0]]: i}}))}})).filter(Boolean)}};
}}

function dessine(){{
  const rond = document.getElementById('rond');
  rond.innerHTML = '';
  const Sz = Math.min(318, (document.body.clientWidth || 360) - 40);
  const cx = Sz / 2, cy = Sz / 2, R = Sz * 0.4;
  const C = {{line: css('--line'), rule: css('--rule'), faint: css('--faint'),
             ink: css('--ink'), accent: css('--accent'), card: css('--card'),
             blue: css('--blue'), green: css('--green')}};
  const svg = sv("svg", {{width: Sz, height: Sz + 10, viewBox: `0 0 ${{Sz}} ${{Sz}}`}});
  svg.appendChild(sv("circle", {{cx, cy, r: R, fill: "none", stroke: C.line, "stroke-width": 1.5}}));

  const niv = niveau(), c = etat();
  // la jante ne désigne des notes qu'au PREMIER niveau ; ensuite elle reste
  // comme mobilier, sans rien prétendre.
  const cible = {{}};
  if (niv && niv.cle === 'base') niv.opts.forEach(o => {{ cible[o.pc] = o; }});
  const tonique = D.accords[sel].root;

  // LA JANTE : douze lettres au cercle des quintes, la tonique cerclée
  for (let i = 0; i < 12; i++) {{
    const ang = (-90 + i * 30) * Math.PI / 180, pc = mod(i * 7, 12);
    const tx = cx + (R + Sz * 0.085) * Math.cos(ang), ty = cy + (R + Sz * 0.085) * Math.sin(ang);
    svg.appendChild(sv("circle", {{cx: cx + R * Math.cos(ang), cy: cy + R * Math.sin(ang),
      r: 2, fill: C.rule}}));
    const vise = cible[pc];
    const lbl = sv("text", {{x: tx, y: ty, "text-anchor": "middle", "dominant-baseline": "central",
      "font-family": "-apple-system,system-ui,sans-serif", "font-size": Sz * 0.042,
      "font-weight": (pc === tonique || vise) ? 700 : 500,
      fill: pc === tonique ? C.accent : (vise ? C.blue : C.faint)}});
    lbl.textContent = N.notes[pc];
    svg.appendChild(lbl);
    if (pc === tonique) svg.appendChild(sv("circle", {{cx: cx + R * Math.cos(ang),
      cy: cy + R * Math.sin(ang), r: 5, fill: "none", stroke: C.accent, "stroke-width": 1.5}}));
  }}

  // LES ORBES : à l'angle de leur note, l'aire suit la proba, jamais sous 22 px
  const centerClear = Sz * 0.155, Rout = R + Sz * 0.03;
  const prMin = 22, prMax = Sz * 0.15;
  if (niv) {{
    const top = niv.opts.reduce((a, b) => b.p > a.p ? b : a, niv.opts[0]);
    // LE CERCLE DES QUINTES NE VAUT QUE POUR LE PREMIER CERCLE (Louis,
    // 2026-09-17 : « ils sont le cercle des quintes sur le premier cercle,
    // quand on passe à la granularité suivante une fois que l'accord est
    // trouvé, on peut passer les 7e 9e tout ça partout »).
    // Au niveau 1 une option EST une fondamentale, donc son angle veut dire
    // quelque chose. Aux niveaux suivants les options sont des degrés du même
    // accord : leur imposer l'angle de la note ajoutée faisait tomber deux
    // orbes presque au même endroit (la 7e majeure et la sixte sont voisines
    // sur le cycle) et ils se recouvraient — un clic atterrissait sur le
    // mauvais. On les répartit donc à intervalles égaux, ce qui est aussi
    // plus lisible : à ce niveau, l'angle n'a rien à dire.
    const parQuintes = niv.cle === 'base';
    const nodes = niv.opts.map((o, k) => ({{o, pr: byArea(o.p, prMin, prMax),
      a: parQuintes ? (-90 + fifths(o.pc) * 30) * Math.PI / 180
                    : (-90 + (360 / niv.opts.length) * k) * Math.PI / 180}}));
    // Jamais de chevauchement : deux orbes à moins de 45° se décalent le long
    // de LEUR rayon — le plus gros contre le moyeu, le suivant vers la jante.
    // JAMAIS DE CHEVAUCHEMENT (Louis, 2026-08-08 : « ils ne doivent jamais se
    // chevaucher, il faut laisser un tout petit interstice »). Le compas de
    // l'app décale DEUX orbes voisins en alternant deux rayons — ça suffit
    // chez lui, pas ici : au premier niveau, `G-`, `G` et `Gsus4` partagent
    // le MÊME rayon de quintes, et l'alternance à deux places en laissait un
    // sous un autre (constaté : un clic atterrissait sur le mauvais orbe).
    // On généralise donc la règle au lieu de la contourner — on empile le
    // long du rayon, du plus gros au plus petit, et si la pile dépasse la
    // jante tout le groupe rétrécit ensemble. C'est de la géométrie, pas un
    // choix : trois orbes sur un rayon, il faut bien la place.
    const mid = (centerClear + R) / 2, gapO = Sz * 0.012;
    const paquets = {{}};
    nodes.forEach(n => {{ const k = Math.round(n.a * 1000);
      (paquets[k] = paquets[k] || []).push(n); }});
    Object.values(paquets).forEach(g => {{
      if (g.length === 1) {{ const n = g[0];
        // LE BORD INTÉRIEUR PASSE AVANT LA JANTE. La formule de l'app borne le
        // rayon par `R - pr` pour garder l'orbe dedans ; sur un gros orbe (45 px
        // de rayon) cette borne le tire vers le centre et il mord le moyeu de
        // 1,6 px — mesuré. On garantit donc d'abord l'interstice avec le moyeu,
        // et on laisse l'orbe dépasser la jante si besoin : le SVG est en
        // `overflow:visible` et l'app l'autorise explicitement.
        n.r = Math.max(centerClear + n.pr, Math.min(mid, R - n.pr)); return; }}
      g.sort((x, y) => y.pr - x.pr);
      const besoin = g.reduce((s, n) => s + 2 * n.pr, 0) + gapO * (g.length - 1);
      const place = Rout - centerClear;
      if (besoin > place) {{ const f = place / besoin; g.forEach(n => {{ n.pr *= f; }}); }}
      let r = centerClear;
      g.forEach(n => {{ n.r = r + n.pr; r = n.r + n.pr + gapO; }});
    }});

    // ET SI ÇA NE DÉGAGE TOUJOURS PAS, TOUT RÉTRÉCIT ENSEMBLE — la règle du
    // compas de l'app, mot pour mot. Deux paquets voisins de 30° peuvent
    // encore se toucher par leurs orbes extérieurs (mesuré : 3 et 4 px au
    // premier niveau). On mesure le pire recouvrement, on met tout le monde à
    // l'échelle, on replace, et on recommence jusqu'à ce que ça dégage. Le rayon ne
    // descend jamais sous le seuil tactile.
    const pose = () => {{
      Object.values(paquets).forEach(g => {{
        if (g.length === 1) {{ const n = g[0];
          n.r = Math.max(centerClear + n.pr, Math.min(mid, R - n.pr)); return; }}
        let r = centerClear;
        g.forEach(n => {{ n.r = r + n.pr; r = n.r + n.pr + gapO; }});
      }});
    }};
    for (let tour = 0; tour < 10; tour++) {{
      let pire = 0;
      for (let i = 0; i < nodes.length; i++) for (let j = i + 1; j < nodes.length; j++) {{
        const a = nodes[i], b = nodes[j];
        const dx = a.r * Math.cos(a.a) - b.r * Math.cos(b.a);
        const dy = a.r * Math.sin(a.a) - b.r * Math.sin(b.a);
        const d = Math.hypot(dx, dy), somme = a.pr + b.pr + gapO;
        if (d < somme) pire = Math.max(pire, somme - d);
      }}
      // le moyeu compte comme un disque de plus
      nodes.forEach(n => {{ const bord = n.r - n.pr - Sz * 0.12;
        if (bord < 0) pire = Math.max(pire, -bord); }});
      if (pire < 0.5) break;
      const f = Math.max(0.90, 1 - pire / (2 * prMax));
      nodes.forEach(n => {{ n.pr = Math.max(prMin * 0.62, n.pr * f); }});
      pose();
    }}
    nodes.forEach(n => {{
      const x = cx + n.r * Math.cos(n.a), y = cy + n.r * Math.sin(n.a);
      const g = sv("g", {{class: "orbe"}});
      g.appendChild(sv("circle", {{cx: x, cy: y, r: n.pr, fill: C.card,
        stroke: n.o === top ? C.green : C.rule, "stroke-width": n.o === top ? 2 : 1.25,
        "stroke-dasharray": n.o.p < SUGGERE ? "3 3" : "none",
        "fill-opacity": n.o.p < SUGGERE ? 0.55 : 1}}));
      glyphe(g, x, y - n.pr * 0.08, n.o.sym.root, n.o.sym.q,
             Math.min(21, n.pr * 0.62), C.ink);
      const pct = sv("text", {{x, y: y + n.pr * 0.60, "text-anchor": "middle",
        "font-family": "-apple-system,system-ui,sans-serif", "font-size": 9.5, fill: C.faint}});
      pct.textContent = (n.o.p * 100).toFixed(0) + ' %';
      g.appendChild(pct);
      g.addEventListener('click', () => {{
        chemin.push(niv.cle === 'base'
          ? {{cle: 'base', root: n.o.root, type: n.o.type, lab: n.o.sym}}
          : {{cle: niv.cle, i: n.o.i, lab: n.o.sym}});
        dessine();
      }});
      svg.appendChild(g);
    }});
  }}

  // LE MOYEU : l'accord courant, et on le touche pour valider
  const hub = sv("g", {{class: "orbe"}});
  hub.appendChild(sv("circle", {{cx, cy, r: Sz * 0.12, fill: C.accent, stroke: C.accent}}));
  const sym = chemin.length ? symbole(c)
    : {{root: D.accords[sel].ecrit, q: ''}};
  glyphe(hub, cx, cy - Sz * 0.012, sym.root, sym.q, Sz * 0.105, "#f4eee2");
  const ok = sv("text", {{x: cx, y: cy + Sz * 0.072, "text-anchor": "middle",
    "font-family": "-apple-system,system-ui,sans-serif", "font-size": 9,
    "font-weight": 600, fill: "rgba(244,238,226,.78)"}});
  ok.textContent = niv ? "c'est celui-là" : "terminé";
  hub.appendChild(ok);
  hub.addEventListener('click', () => {{
    document.getElementById('av').textContent = 'retenu : ' + sym.root + sym.q;
  }});
  svg.appendChild(hub);
  rond.appendChild(svg);

  // le fil d'Ariane
  const fil = document.getElementById('fil');
  fil.innerHTML = '';
  const dep = document.createElement('button');
  dep.className = 'pas'; dep.type = 'button'; dep.textContent = 'départ';
  dep.addEventListener('click', () => {{ chemin = []; dessine(); }});
  fil.appendChild(dep);
  chemin.forEach((p, i) => {{
    const f = document.createElement('span'); f.className = 'fleche'; f.textContent = '›';
    fil.appendChild(f);
    const b = document.createElement('button');
    b.className = 'pas'; b.type = 'button'; b.textContent = p.lab.root + p.lab.q;
    b.addEventListener('click', () => {{ chemin = chemin.slice(0, i + 1); dessine(); }});
    fil.appendChild(b);
  }});
  const nn = document.createElement('span'); nn.className = 'niv';
  nn.textContent = niv ? `${{chemin.length + 1}} · ${{niv.titre}}` : 'accord complet';
  fil.appendChild(nn);

  document.getElementById('av').textContent = 'le chart écrit ' + D.accords[sel].ecrit;
  document.querySelectorAll('.slot').forEach((s, i) => s.classList.toggle('on', i === sel));
}}

document.querySelectorAll('.slot').forEach(s => s.addEventListener('click', () => {{
  sel = parseInt(s.dataset.i, 10); chemin = []; dessine();
  s.scrollIntoView({{inline: 'center', block: 'nearest', behavior: 'smooth'}});
}}));
document.getElementById('haut').addEventListener('click', () => {{ chemin.pop(); dessine(); }});

let audio = null, stopAt = null;
fetch(D.audio).then(r => r.blob()).then(b => {{
  audio = new Audio(URL.createObjectURL(b));
  audio.addEventListener('timeupdate', () => {{
    if (stopAt != null && audio.currentTime >= stopAt) {{ audio.pause(); stopAt = null; }}
  }});
}}).catch(() => {{}});
document.getElementById('play').addEventListener('click', () => {{
  if (!audio) return;
  const a = D.accords[sel];
  stopAt = a.t1 + 0.2; audio.currentTime = Math.max(0, a.t0);
  audio.play().catch(() => {{}});
}});

dessine();
addEventListener('resize', dessine);
</script>
"""


def main() -> int:
    D = json.loads(DATA.read_text(encoding="utf-8"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page(D), encoding="utf-8")
    print(f"→ {OUT}  ({len(D['accords'])} accords, {OUT.stat().st_size // 1024} Ko)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
