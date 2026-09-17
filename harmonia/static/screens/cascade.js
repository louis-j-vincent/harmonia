// Le compas en CASCADE — un quatrième onglet dans l'outil d'annotation.
//
// Louis, 2026-09-17 : « fais en sorte qu'on puisse avoir des extensions, et
// ensuite j'ai envie que tu me fasses une démo d'un compas amélioré, où je
// sélectionne d'abord l'accord en maj/min, ensuite dès qu'on le sélectionne on
// select la 7ème, puis la 9ème, puis la 11ème, puis la 13ème si elle est
// suggérée », puis « fais moi une vraie interface compass en quinconce, où dès
// qu'on clique on arrive au niveau suivant, avec à chaque fois le niveau
// sélectionné au centre ».
//
// POURQUOI UNE CASCADE ET PAS UN CHOIX UNIQUE. Les extensions ne peuvent pas
// CONCOURIR au décodage : mesuré le 2026-09-17, passer musx au vocabulaire
// complet (382 étiquettes) donne un décodage identique — la 9e ne gagne que
// 3 fois sur 1255, la 11e et la 13e jamais — parce que la classe « aucune »
// domine sa tête et rafle Viterbi. Les faire concourir est donc une impasse
// prouvée. Mais le modèle a bien une opinion sur chaque degré, dans une tête
// séparée. On ne les fait pas concourir : on les MONTRE, un étage par tête, et
// c'est l'oreille qui tranche. C'est exactement ce que cet écran est.
//
// CE QU'IL PARTAGE AVEC `buildCompass`, à la valeur près — cette DA n'est pas
// une décoration, c'est une suite de décisions payées :
//   * les angles SONT le cercle des quintes (`pc = (i×7) mod 12`), et
//     l'encombrement se résout RADIALEMENT, jamais en tordant l'angle ;
//   * l'aire de l'orbe suit √proba, plancher 22 px — le seuil TACTILE ;
//   * jamais de chevauchement (Louis, 2026-08-08) ;
//   * la teinte vient de la note via `rootHue`, la roue des couleurs est la
//     roue des quintes ; l'encre posée sur un pétale ne suit pas le thème.
//
// CE QUE LA CASCADE AJOUTE. Aux niveaux 2 à 5 les options ne sont plus des
// accords mais des NOTES qu'on ajoute — la 7e, la 9e, la 11e, la 13e. Une note
// a une classe de hauteur, donc un angle. Chaque orbe se pose à l'angle de la
// note qu'il ajoute, et l'accord se construit en se déployant sur le cercle.
// L'angle reste ce qu'il a toujours été : la place de la note dans le cycle.
// Et « ne rien ajouter » n'est pas un orbe : c'est le MOYEU qu'on touche pour
// valider — la doctrine du compas existant appliquée à la lettre.

import { S } from "../state.js";
import { FAINT_ON_PETAL, INK_ON_PETAL, SERIF, T, UI, confColor, el, fifthsIndex,
         glyph, mod, note, sv } from "../ui/kit.js";

//: repris de `span_rescore.SUG_FLOOR` élargi — aucun seuil nouveau ici, c'est
//: le plancher de suggestion du compas ordinaire (`BASS_SUG_FLOOR`).
const SUGGERE = 0.125;
//: les six familles de la tête de triade de musx, dans son ordre de colonnes.
const TYPE_TAIL = ["", "-", "sus4", "sus2", "o", "+"];

// ── LES COULEURS ───────────────────────────────────────────────────────────
// La TEINTE vient de la note et suit le cercle des quintes : la roue des
// couleurs EST la roue du compas, deux rayons voisins portent deux teintes
// voisines. La PROBABILITÉ, elle, ne dit que la TAILLE (Louis, 2026-09-17 :
// « la saturation ne devrait pas être affectée par le % de proba, juste la
// taille des orbes »). On garde donc les formules de `kit.js` mot pour mot,
// gelées sur une confiance unique : la couleur ne dit plus que « quelle note ».
const FIGE = 0.35;
const hueOf = pc => Math.round(fifthsIndex(pc) / 12 * 360);
const fill = pc => `hsl(${hueOf(pc)} ${Math.round(46 + FIGE * 26)}% ${Math.round(84 - FIGE * 34)}%)`;
const edge = pc => `hsl(${hueOf(pc)} ${Math.round(50 + FIGE * 26)}% ${Math.round(60 - FIGE * 22)}%)`;
const tint = pc => `hsl(${hueOf(pc)} 52% 90%)`;
// `confColor` veut dire UNE chose : « à quel point le modèle croit que c'est
// CET accord-là ». Ce sens n'existe qu'au premier niveau et au moyeu. Plus
// bas, les nombres sont des parts entre extensions (1 %, 19 %) — les peindre
// en rouge ferait crier « le modèle doute » à un compas qui ne doute pas, il
// répartit.
const encre = (p, premier) => premier ? confColor(p) : INK_ON_PETAL;
// l'aire suit la proba, donc le rayon suit sa racine
const byArea = (p, rMin, rMax) => rMin + Math.sqrt(Math.max(0, Math.min(1, p))) * (rMax - rMin);

// ── L'ACCORD QU'ON EST EN TRAIN DE CONSTRUIRE ──────────────────────────────
// L'INTERVALLE QUE CHAQUE OPTION AJOUTE — c'est lui qui donne son angle.
const IV = { toit: [null, 11, 10, 9], neuf: [null, 2, 3, 1],
             onze: [null, 5, 6], treize: [null, 9, 8] };

// Le degré le plus HAUT nomme l'accord : un accord avec 7e + 9e + 11e s'écrit
// `-11`, pas `-7add9add11`. C'est la convention des grilles, pas une
// simplification — et elle rend la cascade lisible, chaque étage renomme.
function symbole(c) {
  const base = TYPE_TAIL[c.type] || "";
  const alt = [];
  if (c.neuf === 2) alt.push("#9");
  if (c.neuf === 3) alt.push("b9");
  if (c.onze === 2) alt.push("#11");
  if (c.treize === 2) alt.push("b13");
  let q;
  if (c.toit === 0) q = base + (c.neuf === 1 ? "add9" : "");
  else if (c.toit === 3) q = base + (c.neuf === 1 ? "69" : "6");
  else {
    let haut = 7;
    if (c.neuf === 1) haut = 9;
    if (c.onze === 1) haut = 11;
    if (c.treize === 1) haut = 13;
    if (c.type === 4) q = (c.toit === 2 ? "h7" : "o7");
    else if (c.toit === 1) q = (c.type === 1 ? "-^" : "^") + haut;
    else if (c.type === 2) q = haut + "sus4";
    else if (c.type === 3) q = haut + "sus2";
    else q = (c.type === 1 ? "-" : "") + haut;
  }
  if (alt.length) q += alt.join("");
  return { root: c.root, q };
}

function noCascBox() {
  const b = el("div", `border:1px dashed ${T.rule};border-radius:12px;padding:16px;text-align:center;`);
  b.appendChild(el("div", `font:600 12.5px ${UI};color:${T.ink};margin-bottom:5px;`,
    "pas encore de cascade sur ce chart"));
  b.appendChild(el("div", `font:italic 12px ${SERIF};color:${T.faint};line-height:1.5;`,
    "les têtes d'extension de musx sont écrites à l'analyse — refais tourner la chanson pour les avoir"));
  return b;
}

/**
 * Le compas en cascade pour l'accord `idx`.
 *
 * `onPick({root, q}, btn)` est le MÊME rappel que `buildCompass` : toucher le
 * moyeu choisit l'accord construit, la prévisualisation le joue, « Lock »
 * s'arme. Un orbe ne choisit rien — il descend d'un étage.
 */
export function buildCascade(idx, onPick) {
  const chord = S.chords[idx];
  const casc = chord && chord.casc;
  if (!casc || !casc.base || !casc.base.length) return noCascBox();

  const box = el("div", "");
  box.appendChild(el("div", `text-align:center;font:600 9.5px ${UI};letter-spacing:.06em;text-transform:uppercase;color:${T.faint};margin-bottom:6px;`,
    "on descend un degré à la fois"));
  const fil = el("div", `display:flex;gap:5px;align-items:center;flex-wrap:wrap;min-height:30px;margin-bottom:2px;`);
  const rond = el("div", "display:flex;justify-content:center;");
  const pied = el("div", `text-align:center;font:italic 12px ${SERIF};color:${T.faint};margin-top:6px;min-height:18px;`);
  box.appendChild(fil); box.appendChild(rond); box.appendChild(pied);

  let chemin = [];

  const etat = () => {
    const c = { root: chord.root, type: 0, toit: 0, neuf: 0, onze: 0, treize: 0 };
    for (const p of chemin) {
      if (p.cle === "base") { c.root = p.root; c.type = p.type; } else c[p.cle] = p.i;
    }
    return c;
  };

  // QUEL ÉTAGE ON REGARDE. `null` = l'accord est complet, il n'y a plus rien
  // à demander. La descente s'arrête après la 9e quand il n'y a PAS de
  // septième : une 11e sans 7e n'est pas une 11e, et les deux orbes portaient
  // alors la même étiquette (constaté : `D-69` deux fois).
  const niveau = () => {
    const c = etat(), n = chemin.length;
    if (n === 0) return { cle: "base", titre: "la base",
      // UN ORBE PAR CLASSE DE HAUTEUR, cinq au plus. Le compas dit lui-même
      // qu'« une lettre de la roue est une CLASSE DE HAUTEUR » : y poser trois
      // qualités du même SOL (`G-`, `G`, `Gsus4`) ne tient pas dans la
      // géométrie — trois orbes tactiles empilés sur un rayon demandent 189 px
      // quand la jante en offre 137. On garde la meilleure qualité par note.
      // CE QUE ÇA COÛTE, et il faut le dire : le `G-` à 5 % masque le `G` à
      // 3 %. La cascade ne rattrape pas le passage majeur/mineur.
      opts: Object.values(casc.base.reduce((acc, b) => {
        if (!acc[b.root] || b.c > acc[b.root].c) acc[b.root] = b;
        return acc;
      }, {})).sort((x, y) => y.c - x.c).slice(0, 5)
        .map(b => ({ root: b.root, type: b.type, p: b.c, pc: b.root,
                     sym: { root: b.root, q: TYPE_TAIL[b.type] || "" } })) };
    if (n === 1) return { cle: "toit", titre: "ce qui se pose dessus",
      opts: [3, 1, 2].map(i => ({ i, p: (i === 3 ? casc.treize[1] : casc.sev[i]),
        pc: mod(c.root + IV.toit[i], 12), sym: symbole({ ...c, toit: i }) })) };
    if ((c.toit === 0 || c.toit === 3) && n >= 3) return null;
    const suite = [["neuf", "la 9e", casc.neuf], ["onze", "la 11e", casc.onze],
                   ["treize", "la 13e", casc.treize]][n - 2];
    if (!suite) return null;
    return { cle: suite[0], titre: suite[1],
      opts: suite[2].map((p, i) => i === 0 ? null : ({ i, p,
        pc: mod(c.root + IV[suite[0]][i], 12),
        sym: symbole(Object.assign({ ...c }, { [suite[0]]: i })) })).filter(Boolean) };
  };

  function dessine() {
    rond.textContent = ""; fil.textContent = "";
    const Sz = Math.min(318, (rond.clientWidth || box.clientWidth || 340) - 8);
    const cx = Sz / 2, cy = Sz / 2, R = Sz * 0.4;
    const svg = sv("svg", { width: Sz, height: Sz, viewBox: `0 0 ${Sz} ${Sz}`,
                            style: "display:block;overflow:visible" });
    svg.appendChild(sv("circle", { cx, cy, r: R, fill: "none", stroke: T.line, "stroke-width": 1.5 }));

    const niv = niveau(), c = etat();
    // la jante ne DÉSIGNE des notes qu'au premier niveau ; ensuite elle reste
    // comme mobilier, sans rien prétendre.
    const cible = {};
    if (niv && niv.cle === "base") niv.opts.forEach(o => { cible[o.pc] = o; });

    // LA JANTE : douze lettres au cercle des quintes, la fondamentale écrite cerclée
    for (let i = 0; i < 12; i++) {
      const ang = (-90 + i * 30) * Math.PI / 180, pc = mod(i * 7, 12);
      const ox = cx + R * Math.cos(ang), oy = cy + R * Math.sin(ang);
      svg.appendChild(sv("circle", { cx: ox, cy: oy, r: 2, fill: T.rule }));
      const lbl = sv("text", { x: cx + (R + Sz * 0.085) * Math.cos(ang),
        y: cy + (R + Sz * 0.085) * Math.sin(ang), "text-anchor": "middle",
        "dominant-baseline": "central", "font-family": UI, "font-size": Sz * 0.042,
        "font-weight": (pc === chord.root || cible[pc]) ? 700 : 500,
        fill: pc === chord.root ? T.accent : (cible[pc] ? T.blue : T.faint) });
      lbl.textContent = note(pc);
      svg.appendChild(lbl);
      if (pc === chord.root) svg.appendChild(sv("circle", { cx: ox, cy: oy, r: 5,
        fill: "none", stroke: T.accent, "stroke-width": 1.5 }));
    }

    // LE MOYEU A LA TAILLE DE SA PROBA (Louis, 2026-09-17). On ne la recalcule
    // pas : il GARDE le rayon qu'avait l'orbe qu'on vient de toucher, donc
    // cliquer, c'est voir le disque glisser au centre sans changer de taille.
    const Rout = R + Sz * 0.03, prMin = 22, prMax = Sz * 0.15;
    const dern = chemin.length ? chemin[chemin.length - 1] : null;
    const pMoy = dern ? (dern.p || 0)
      : ((casc.base.find(b => b.root === chord.root) || casc.base[0] || {}).c || 0);
    const rMoy = (dern && dern.pr) ? dern.pr : byArea(pMoy, prMin, prMax);
    // l'anneau libre autour du moyeu suit donc le moyeu, et laisse la place à
    // sa légende, qui est SOUS le disque — elle ne peut plus le contraindre.
    const centerClear = rMoy + Sz * 0.055;

    let nodes = [];
    if (niv) {
      const top = niv.opts.reduce((a, b) => b.p > a.p ? b : a, niv.opts[0]);
      // LE CERCLE DES QUINTES NE VAUT QUE POUR LE PREMIER CERCLE (Louis,
      // 2026-09-17). Au niveau 1 une option EST une fondamentale, son angle
      // veut dire quelque chose. Aux niveaux suivants les options sont des
      // degrés du même accord : leur imposer l'angle de la note ajoutée
      // faisait tomber deux orbes presque au même endroit (la 7e majeure et la
      // sixte sont voisines sur le cycle) et un clic atterrissait sur le
      // mauvais. On les répartit à intervalles égaux — à ce niveau, l'angle
      // n'a rien à dire.
      const parQuintes = niv.cle === "base";
      nodes = niv.opts.map((o, k) => ({ o, top: o === top, pr: byArea(o.p, prMin, prMax),
        a: parQuintes ? (-90 + fifthsIndex(o.pc) * 30) * Math.PI / 180
                      : (-90 + (360 / niv.opts.length) * k) * Math.PI / 180 }));

      // JAMAIS DE CHEVAUCHEMENT (Louis, 2026-08-08). Au premier niveau `G-`,
      // `G` et `Gsus4` partagent le MÊME rayon de quintes : on empile le long
      // du rayon, du plus gros au plus petit, et si la pile dépasse la jante
      // tout le groupe rétrécit ensemble. C'est de la géométrie, pas un choix.
      const mid = (centerClear + R) / 2, gapO = Sz * 0.012;
      const paquets = {};
      nodes.forEach(n => { const k = Math.round(n.a * 1000);
        (paquets[k] = paquets[k] || []).push(n); });
      Object.values(paquets).forEach(g => g.sort((x, y) => y.pr - x.pr));
      const pose = () => {
        Object.values(paquets).forEach(g => {
          if (g.length === 1) { const n = g[0];
            // LE BORD INTÉRIEUR PASSE AVANT LA JANTE : borner par `R - pr`
            // tire un gros orbe vers le centre et il mord le moyeu (mesuré
            // 1,6 px). On garantit d'abord l'interstice, et on laisse l'orbe
            // dépasser la jante — le SVG est en `overflow:visible`.
            n.r = Math.max(centerClear + n.pr, Math.min(mid, R - n.pr)); return; }
          const besoin = g.reduce((s, n) => s + 2 * n.pr, 0) + gapO * (g.length - 1);
          const place = Rout - centerClear;
          if (besoin > place) { const f = place / besoin; g.forEach(n => { n.pr *= f; }); }
          let r = centerClear;
          g.forEach(n => { n.r = r + n.pr; r = n.r + n.pr + gapO; });
        });
      };
      pose();
      // Deux paquets voisins de 30° peuvent encore se toucher par leurs orbes
      // extérieurs. On mesure le pire recouvrement — le moyeu compte comme un
      // disque de plus —, on met tout le monde à l'échelle, on replace.
      for (let tour = 0; tour < 10; tour++) {
        let pire = 0;
        for (let i = 0; i < nodes.length; i++) for (let j = i + 1; j < nodes.length; j++) {
          const a = nodes[i], b = nodes[j];
          const d = Math.hypot(a.r * Math.cos(a.a) - b.r * Math.cos(b.a),
                               a.r * Math.sin(a.a) - b.r * Math.sin(b.a));
          const somme = a.pr + b.pr + gapO;
          if (d < somme) pire = Math.max(pire, somme - d);
        }
        nodes.forEach(n => { const bord = n.r - n.pr - rMoy;
          if (bord < 0) pire = Math.max(pire, -bord); });
        if (pire < 0.5) break;
        const f = Math.max(0.90, 1 - pire / (2 * prMax));
        nodes.forEach(n => { n.pr = Math.max(prMin * 0.62, n.pr * f); });
        pose();
      }
      nodes.forEach(n => { n.x = cx + n.r * Math.cos(n.a); n.y = cy + n.r * Math.sin(n.a); });
      // le rayon jusqu'au moyeu, dans la teinte de l'orbe (`buildCompass` fait
      // pareil, `stroke-opacity` 0.4) : il dit d'où vient la proposition.
      nodes.forEach(n => svg.appendChild(sv("line", { x1: cx, y1: cy, x2: n.x, y2: n.y,
        stroke: edge(n.o.pc), "stroke-width": 1, "stroke-opacity": 0.4 })));
    }
    svg.appendChild(sv("circle", { cx, cy, r: rMoy, fill: tint(c.root),
      stroke: T.accent, "stroke-width": 2 }));

    const wrap = el("div", `position:relative;width:${Sz}px;margin:0 auto;`);
    const layer = el("div", "position:absolute;inset:0;pointer-events:none;");
    nodes.forEach((n, i) => {
      const b = el("button", `position:absolute;left:${n.x}px;top:${n.y}px;`
        + `transform:translate(-50%,-50%);width:${n.pr * 2}px;height:${n.pr * 2}px;`
        + `border-radius:50%;border:${n.top ? 2.5 : 1.5}px ${n.o.p < SUGGERE ? "dashed" : "solid"} ${edge(n.o.pc)};`
        + `background:${fill(n.o.pc)};cursor:pointer;pointer-events:auto;display:flex;`
        + `flex-direction:column;align-items:center;justify-content:center;gap:1px;padding:0;`
        + `box-shadow:0 2px 6px rgba(60,40,20,.14);animation:ap-orb .3s ${0.04 * i}s both;`);
      b.appendChild(glyph(n.o.sym.root, n.o.sym.q, Math.max(13, Math.round(n.pr * 0.58)),
                          "exact", encre(n.o.p, niv.cle === "base")));
      if (n.pr > Sz * 0.068) b.appendChild(el("span",
        `font:600 ${Math.max(9, Math.round(n.pr * 0.30))}px ${UI};font-style:normal;color:${FAINT_ON_PETAL};`,
        Math.round(n.o.p * 100) + "%"));
      b.onclick = () => {
        chemin.push(niv.cle === "base"
          ? { cle: "base", root: n.o.root, type: n.o.type, sym: n.o.sym, p: n.o.p, pr: n.pr }
          : { cle: niv.cle, i: n.o.i, sym: n.o.sym, p: n.o.p, pr: n.pr });
        dessine();
      };
      layer.appendChild(b);
    });

    // LE MOYEU EST L'ACCORD COURANT, et on le touche pour VALIDER — même
    // rappel qu'un orbe du compas ordinaire : la prévisualisation joue,
    // « Lock » s'arme. « Ne rien ajouter » n'est donc pas une orbite.
    const sym = chemin.length ? symbole(c) : { root: chord.root, q: chord.q || "" };
    const hub = el("button", `position:absolute;left:${cx}px;top:${cy}px;`
      + `transform:translate(-50%,-50%);width:${rMoy * 2}px;height:${rMoy * 2}px;`
      + `border-radius:50%;border:none;background:transparent;cursor:pointer;`
      + `pointer-events:auto;display:flex;align-items:center;justify-content:center;padding:0;`);
    // Le moyeu pouvant être petit, le symbole doit tenir dedans : largeur
    // estimée du Georgia italique, bornée par la corde du disque.
    const larg = 1 + (sym.q || "").length * 0.48;
    hub.appendChild(glyph(sym.root, sym.q,
      Math.max(12, Math.min(Math.round(Sz * 0.1), Math.round(1.7 * rMoy / larg))),
      "exact", confColor(chemin.length ? (chemin[0].p || 0) : pMoy)));
    hub.onclick = () => { onPick({ root: sym.root, q: sym.q }, hub); };
    layer.appendChild(hub);
    // la légende est SOUS le disque : dedans, elle imposait au moyeu une
    // taille minimale et un moyeu à 1 % ne pouvait pas être petit.
    layer.appendChild(el("div", `position:absolute;left:${cx}px;top:${cy + rMoy + 10}px;`
      + `transform:translate(-50%,-50%);font:600 9px ${UI};color:${T.faint};white-space:nowrap;`,
      niv ? "c'est celui-là" : "accord complet"));
    wrap.appendChild(svg); wrap.appendChild(layer);
    rond.appendChild(wrap);

    // LE FIL D'ARIANE — on remonte d'où on veut, l'état n'est jamais perdu.
    const pas = (txt, fn, actif) => {
      const b = el("button", `border:1px solid ${actif ? T.accent : T.rule};`
        + `background:${T.card};color:${T.ink};border-radius:8px;padding:5px 10px;`
        + `min-height:30px;font:italic 600 12.5px ${SERIF};cursor:pointer;`, txt);
      b.onclick = fn; return b;
    };
    fil.appendChild(pas("départ", () => { chemin = []; dessine(); }, !chemin.length));
    chemin.forEach((p, i) => {
      fil.appendChild(el("span", `color:${T.faint};font-size:13px;`, "›"));
      const b = pas("", () => { chemin = chemin.slice(0, i + 1); dessine(); },
                    i === chemin.length - 1);
      b.appendChild(glyph(p.sym.root, p.sym.q, 13, "exact", T.ink));
      fil.appendChild(b);
    });
    fil.appendChild(el("div", `margin-left:auto;font:500 11px ${UI};color:${T.faint};`,
      niv ? `${chemin.length + 1} · ${niv.titre}` : "rien à ajouter"));
    pied.textContent = niv
      ? "touche un orbe pour descendre, le moyeu pour garder l'accord du centre"
      : "touche le moyeu pour garder cet accord";
  }

  dessine();
  return box;
}
