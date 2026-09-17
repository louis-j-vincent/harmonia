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

import { bassList } from "./annotate.js";
import { collOf, keyTrackFor } from "./chart.js";
import { S } from "../state.js";
import { CE_IV, FAINT_ON_PETAL, INK_ON_PETAL, SERIF, T, UI, confColor, el,
         fifthsIndex, glyph, keyHue, mod, note, sv } from "../ui/kit.js";

//: repris de `span_rescore.SUG_FLOOR` élargi — aucun seuil nouveau ici, c'est
//: le plancher de suggestion du compas ordinaire (`BASS_SUG_FLOOR`).
const SUGGERE = 0.125;
//: les six familles de la tête de triade de musx, dans son ordre de colonnes.
const TYPE_TAIL = ["", "-", "sus4", "sus2", "o", "+"];

// ── LES COULEURS : LA GAMME, PAS LA NOTE ───────────────────────────────────
// Louis, 2026-09-17 : « je veux une couleur par gamme, mais pour les accords
// ils sont par défaut de la couleur de la gamme dans laquelle cette partie de
// la chanson est, sauf si l'accord lui même est en dehors de cette gamme
// auquel cas il est de la couleur de la gamme dans laquelle il projette via
// les notes qui sortent de la gamme, donc comme l'outil local keys dans
// analyse » — puis « mets en prod ce jeu de couleurs pour le compas en
// cascade ». Arbitré sur `docs/plots/gammes.html`.
//
// CE QU'EST UNE GAMME ICI : SEPT NOTES. Pas une tonique, pas un mode — un jeu
// de notes, parce que la règle travaille sur « les notes qui sortent ».
// Conséquence assumée : si♭ majeur et sol mineur sont la MÊME gamme et
// portent la même couleur. C'est exact (ce sont les mêmes sept notes) et
// c'est ce qui empêche un morceau en mineur de clignoter à chaque emprunt à
// sa relative. C'est aussi déjà la convention de la lentille « key » du chart
// en mode analyse, qui colore par `keyHue(collOf(...))` : le compas et le
// chart disent donc la même chose de la même couleur.
//
// CE QUE ÇA REMPLACE. La teinte venait de la FONDAMENTALE du candidat : cinq
// candidats faisaient cinq couleurs, et aucune ne disait rien d'autre que
// « ce sont cinq accords différents » — ce qu'on voyait déjà. Maintenant un
// orbe de la couleur du fond reste dans la tonalité, et un orbe d'une autre
// couleur en sort, vers la gamme que sa couleur nomme.
const DIATONIQUE = [0, 2, 4, 5, 7, 9, 11];
//: les sept notes de la gamme dont la tonique MAJEURE est `maj`.
const collection = maj => DIATONIQUE.map(i => mod(maj + i, 12));
const dansColl = (pcs, maj) => { const c = collection(maj);
  return pcs.every(p => c.indexOf(p) >= 0); };
const combienDans = (pcs, maj) => { const c = collection(maj);
  return pcs.reduce((n, p) => n + (c.indexOf(p) >= 0 ? 1 : 0), 0); };
//: combien de quintes séparent deux gammes, dans le sens le plus court.
const ecartQuintes = (a, b) => { const d = mod(fifthsIndex(a) - fifthsIndex(b), 12);
  return Math.min(d, 12 - d); };
// LA PROJECTION : la gamme la plus proche EN QUINTES qui contient l'accord
// tout entier. Si aucune ne le contient (un diminué, un altéré), celle qui en
// contient le plus — départagée par la proximité, pour ne pas envoyer un
// accord à l'autre bout du cercle quand deux gammes le servent aussi mal.
function gammeDe(pcs, regne) {
  if (dansColl(pcs, regne)) return regne;
  let meilleur = regne, best = [-1, -99];
  for (let k = 0; k < 12; k++) {
    const score = [dansColl(pcs, k) ? 100 : combienDans(pcs, k), -ecartQuintes(k, regne)];
    if (score[0] > best[0] || (score[0] === best[0] && score[1] > best[1])) {
      best = score; meilleur = k;
    }
  }
  return meilleur;
}

// La PROBABILITÉ ne dit que la TAILLE (Louis, 2026-09-17 : « la saturation ne
// devrait pas être affectée par le % de proba »). Saturation et clarté sont
// donc figées, et la teinte ne porte que l'identité de la gamme.
const hueOf = maj => Math.round(keyHue(maj));
const fill = maj => `hsl(${hueOf(maj)} 55% 72%)`;
const edge = maj => `hsl(${hueOf(maj)} 59% 52%)`;
const tint = maj => `hsl(${hueOf(maj)} 52% 90%)`;

// ── LES NOTES D'UN ACCORD DE LA CASCADE ────────────────────────────────────
// On ne relit pas le symbole : l'état de la cascade porte déjà la triade et
// chaque degré ajouté, donc les notes se lisent directement dessus.
const TYPE_IV = [[0, 4, 7], [0, 3, 7], [0, 5, 7], [0, 2, 7], [0, 3, 6], [0, 4, 8]];
function notesDe(c) {
  const iv = (TYPE_IV[c.type] || TYPE_IV[0]).slice();
  if (c.toit) iv.push(IV.toit[c.toit]);
  if (c.neuf) iv.push(IV.neuf[c.neuf]);
  if (c.onze) iv.push(IV.onze[c.onze]);
  if (c.treize) iv.push(IV.treize[c.treize]);
  const vus = {};
  return iv.map(i => mod(c.root + i, 12)).filter(p => vus[p] ? false : (vus[p] = 1));
}
// `confColor` veut dire UNE chose : « à quel point le modèle croit que c'est
// CET accord-là ». Ce sens n'existe qu'au premier niveau et au moyeu. Plus
// bas, les nombres sont des parts entre extensions (1 %, 19 %) — les peindre
// en rouge ferait crier « le modèle doute » à un compas qui ne doute pas, il
// répartit.
const encre = (p, premier) => premier ? confColor(p) : INK_ON_PETAL;
// ── LA TAILLE, ET RIEN QU'ELLE, DIT LA PROBABILITÉ ─────────────────────────
// Louis, 2026-09-17 : « il faut que la taille des cercles / orbes reflète
// vraiment les probas, je peux pas avoir 30 % plus grand que 20 % ».
//
// La formule de l'app est `rMin + √p × (rMax - rMin)` avec `rMin = 22` — le
// plancher TACTILE. Mesuré : sur un morceau où le modèle donne 63 / 24 / 3 /
// 2 / 1 %, elle rend 40 / 33 / 26 / 25 / 24 px de rayon. Un rapport de 63 en
// probabilité devient 1,7 en rayon et 2,7 en aire : le plancher a mangé toute
// l'échelle, et deux candidats que tout sépare ont l'air presque égaux.
//
// Ici l'aire est donc VRAIMENT proportionnelle à la probabilité : le rayon
// est `rMax × √p`, sans plancher, et 100 % remplit la bande. Le plancher
// tactile ne disparaît pas — il change de nature : il devient une zone de
// contact INVISIBLE de 44 px autour de l'orbe, comme les anneaux de basse le
// font déjà. Un orbe peut alors être un point, et rester touchable.
const R_VU_MIN = 4;          // en dessous, on ne verrait plus qu'il existe
const TACTILE = 44;          // le diamètre minimal d'une zone de contact
const rayonDe = (p, rMax) => Math.max(R_VU_MIN, rMax * Math.sqrt(Math.max(0, Math.min(1, p))));
//: l'orbe est-il assez grand pour porter son étiquette DEDANS ?
const TIENT = 19;
// l'aire suit la proba, donc le rayon suit sa racine (pour les anneaux de
// basse, qui eux gardent la convention de l'app)
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
 *
 * `bassOpts = {onPickBass, picked}` — mêmes anneaux de basse que le compas
 * ordinaire, autour des lettres de la jante, et la basse choisie s'écrit en
 * slash sur le moyeu (Louis, 2026-09-17 : « les basses autour du cercle en
 * slash chords »).
 */
export function buildCascade(idx, onPick, bassOpts) {
  bassOpts = bassOpts || {};
  const chord = S.chords[idx];
  const casc = chord && chord.casc;
  if (!casc || !casc.base || !casc.base.length) return noCascBox();

  const box = el("div", "");
  const entete = el("div", `text-align:center;font:600 9.5px ${UI};letter-spacing:.06em;text-transform:uppercase;color:${T.faint};margin-bottom:6px;`);
  box.appendChild(entete);
  const fil = el("div", `display:flex;gap:5px;align-items:center;flex-wrap:wrap;min-height:30px;margin-bottom:2px;`);
  const rond = el("div", "display:flex;justify-content:center;");
  const pied = el("div", `text-align:center;font:italic 12px ${SERIF};color:${T.faint};margin-top:6px;min-height:18px;`);
  box.appendChild(fil); box.appendChild(rond); box.appendChild(pied);

  let chemin = [];

  // LA GAMME QUI RÈGNE ICI. `keyTrackFor` est la piste de clés LOCALES que le
  // chart utilise déjà pour sa lentille « key » en mode analyse — celle que
  // Louis appelle « l'outil local keys dans analyse ». Elle est par accord,
  // pas par section : un II-V passager a la sienne. `collOf` en fait un jeu de
  // sept notes (une mineure rend sa relative majeure, ce sont les mêmes).
  let regne = collOf(S.key, S.keyMode || "major");
  try {
    const piste = keyTrackFor(S.chords, S.key, S.keyMode);
    if (piste && piste[idx]) regne = collOf(piste[idx].tonic, piste[idx].mode);
  } catch (e) { /* une piste absente ne doit pas fermer le compas */ }

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
                     gam: gammeDe(notesDe({ root: b.root, type: b.type }), regne),
                     sym: { root: b.root, q: TYPE_TAIL[b.type] || "" } })) };
    if (n === 1) return { cle: "toit", titre: "ce qui se pose dessus",
      opts: [3, 1, 2].map(i => ({ i, p: (i === 3 ? casc.treize[1] : casc.sev[i]),
        pc: mod(c.root + IV.toit[i], 12),
        gam: gammeDe(notesDe({ ...c, toit: i }), regne),
        sym: symbole({ ...c, toit: i }) })) };
    if ((c.toit === 0 || c.toit === 3) && n >= 3) return null;
    const suite = [["neuf", "la 9e", casc.neuf], ["onze", "la 11e", casc.onze],
                   ["treize", "la 13e", casc.treize]][n - 2];
    if (!suite) return null;
    return { cle: suite[0], titre: suite[1],
      opts: suite[2].map((p, i) => i === 0 ? null : ({ i, p,
        pc: mod(c.root + IV[suite[0]][i], 12),
        gam: gammeDe(notesDe(Object.assign({ ...c }, { [suite[0]]: i })), regne),
        sym: symbole(Object.assign({ ...c }, { [suite[0]]: i })) })).filter(Boolean) };
  };

  //: une gamme se nomme par sa majeure ET sa relative mineure — c'est le même
  //: jeu de notes, et le dire évite de croire qu'on a changé de monde.
  const nomGamme = maj => note(maj) + " / " + note(mod(maj + 9, 12)) + "-";

  function dessine() {
    rond.textContent = ""; fil.textContent = "";
    entete.textContent = "la gamme ici : " + nomGamme(regne);
    const Sz = Math.min(318, (rond.clientWidth || box.clientWidth || 340) - 8);
    const cx = Sz / 2, cy = Sz / 2, R = Sz * 0.4;
    const svg = sv("svg", { width: Sz, height: Sz, viewBox: `0 0 ${Sz} ${Sz}`,
                            style: "display:block;overflow:visible" });
    // LA ROUE REPOSE SUR LA GAMME QUI RÈGNE. C'est ce fond qui rend la règle
    // lisible sans légende : un orbe de la couleur du fond reste dans la
    // tonalité, un orbe d'une autre couleur en sort — et sa couleur nomme la
    // gamme vers laquelle il sort.
    // On peint avec `fill` (clarté 72 %) à faible opacité, PAS avec le `tint`
    // pâle du compas : un `tint` à 90 % de clarté composité sur la carte
    // sombre rend un gris boueux qui a perdu sa teinte — donc la règle « le
    // fond, c'est la gamme » ne se lisait plus du tout en thème sombre.
    svg.appendChild(sv("circle", { cx, cy, r: R, fill: fill(regne),
      "fill-opacity": 0.32, stroke: T.line, "stroke-width": 1.5 }));

    const niv = niveau(), c = etat();
    //: le pourcentage, écrit pour être lu — sous 1 %, « 0 % » ne distingue pas
    //: un candidat que le modèle a vu d'un candidat qu'il a écarté.
    const pct = p => p < 0.01 ? (p * 100).toFixed(1).replace(".", ",") + " %"
                              : Math.round(p * 100) + " %";
    // la jante ne DÉSIGNE des notes qu'au premier niveau ; ensuite elle reste
    // comme mobilier, sans rien prétendre.
    const cible = {};
    if (niv && niv.cle === "base") niv.opts.forEach(o => { cible[o.pc] = o; });

    // LA BASSE ENTENDUE, autour des lettres de la jante — repris tel quel de
    // `buildCompass`. Une lettre de la jante est une CLASSE DE HAUTEUR, pas un
    // accord : c'est exactement ce qu'une basse nomme, donc elle se pose là
    // sans rien inventer et sans voler de place aux orbes, qui vivent à
    // l'intérieur du cercle. Bleu, et PAS la teinte de quintes de la note :
    // la basse est une AUTRE voix, pas un autre accord — si les deux
    // partageaient la palette, un anneau se lirait comme une suggestion.
    const bass = bassList(idx);
    const bassBy = {};
    bass.forEach((b, i) => { bassBy[b.pc] = { b, isTop: i === 0 }; });
    const bMin = Sz * 0.042, bMax = Sz * 0.082, bassHits = [];

    // LA JANTE : douze lettres au cercle des quintes, la fondamentale écrite cerclée
    for (let i = 0; i < 12; i++) {
      const ang = (-90 + i * 30) * Math.PI / 180, pc = mod(i * 7, 12);
      const ox = cx + R * Math.cos(ang), oy = cy + R * Math.sin(ang);
      const tx = cx + (R + Sz * 0.085) * Math.cos(ang);
      const ty = cy + (R + Sz * 0.085) * Math.sin(ang);
      svg.appendChild(sv("circle", { cx: ox, cy: oy, r: 2, fill: T.rule }));
      // AVANT la lettre : en SVG le dernier peint gagne, et un disque posé
      // par-dessus rendrait illisible la note qu'on vient justement lire.
      const hit = bassBy[pc];
      if (hit) {
        const br = byArea(hit.b.c, bMin, bMax), choisie = bassOpts.picked === pc;
        svg.appendChild(sv("circle", { cx: tx, cy: ty, r: br, fill: T.blue,
          "fill-opacity": choisie ? 0.42 : 0.13, stroke: T.blue,
          "stroke-width": choisie ? 3 : (hit.isTop ? 2 : 1.25),
          "stroke-opacity": choisie ? 1 : (hit.isTop ? 0.95 : 0.6) }));
        bassHits.push({ pc, tx, ty, br, c: hit.b.c });
      }
      const lbl = sv("text", { x: tx, y: ty, "text-anchor": "middle",
        "dominant-baseline": "central", "font-family": UI, "font-size": Sz * 0.042,
        "font-weight": (pc === chord.root || cible[pc] || hit) ? 700 : 500,
        fill: pc === chord.root ? T.accent
            : (hit ? T.blue : (cible[pc] ? T.blue : T.faint)) });
      lbl.textContent = note(pc);
      svg.appendChild(lbl);
      if (pc === chord.root) svg.appendChild(sv("circle", { cx: ox, cy: oy, r: 5,
        fill: "none", stroke: T.accent, "stroke-width": 1.5 }));
    }

    // LE MOYEU A LA TAILLE DE SA PROBA (Louis, 2026-09-17). On ne la recalcule
    // pas : il GARDE le rayon qu'avait l'orbe qu'on vient de toucher, donc
    // cliquer, c'est voir le disque glisser au centre sans changer de taille.
    const Rout = R + Sz * 0.03, prMax = Sz * 0.155;
    const dern = chemin.length ? chemin[chemin.length - 1] : null;
    const pMoy = dern ? (dern.p || 0)
      : ((casc.base.find(b => b.root === chord.root) || casc.base[0] || {}).c || 0);
    const rMoy = (dern && dern.pr) ? dern.pr : rayonDe(pMoy, prMax);
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
      nodes = niv.opts.map((o, k) => ({ o, top: o === top, pr: rayonDe(o.p, prMax),
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
        // On rétrécit TOUT LE MONDE du même facteur : les aires restent
        // proportionnelles entre elles, c'est la seule chose qui compte pour
        // comparer. (L'ancien `Math.max(plancher, …)` écrasait cette
        // proportion dès qu'un orbe touchait le plancher.)
        const f = Math.max(0.90, 1 - pire / (2 * prMax));
        nodes.forEach(n => { n.pr = Math.max(R_VU_MIN, n.pr * f); });
        pose();
      }
      nodes.forEach(n => { n.x = cx + n.r * Math.cos(n.a); n.y = cy + n.r * Math.sin(n.a); });
      // le rayon jusqu'au moyeu, dans la teinte de l'orbe (`buildCompass` fait
      // pareil, `stroke-opacity` 0.4) : il dit d'où vient la proposition.
      nodes.forEach(n => svg.appendChild(sv("line", { x1: cx, y1: cy, x2: n.x, y2: n.y,
        stroke: edge(n.o.gam), "stroke-width": 1, "stroke-opacity": 0.4 })));
    }
    // LA GAMME DU MOYEU. Tant qu'on n'a rien choisi, le moyeu porte l'accord
    // ÉCRIT : ses notes viennent donc de sa queue (`CE_IV`, la table dont le
    // son de prévisualisation se sert déjà), pas d'une triade nue — sans quoi
    // un `D7` perdrait son fa♯, c'est-à-dire exactement la note qui le fait
    // sortir de la tonalité.
    const notesMoy = chemin.length ? notesDe(c)
      : (CE_IV[chord.q] || CE_IV[""]).map(i => mod(chord.root + i, 12));
    const gamMoy = gammeDe(notesMoy, regne);
    // Le moyeu prend la MÊME clarté que les orbes, pas le `keyTint` pâle du
    // compas ordinaire : sa couleur porte maintenant une information (sa
    // gamme), et à 90 % de clarté deux gammes voisines ne se distinguaient
    // plus. L'anneau d'accent suffit à dire que c'est le moyeu.
    svg.appendChild(sv("circle", { cx, cy, r: rMoy, fill: fill(gamMoy),
      stroke: T.accent, "stroke-width": 2 }));

    // Les anneaux de basse vivent DEHORS, à `R + 0,085·Sz` plus leur propre
    // rayon : celui du bas dépasse la boîte du SVG d'environ 7 % de sa taille
    // et venait mordre la légende. On réserve la place plutôt que de rentrer
    // les anneaux — leur distance à la jante est ce qui les distingue des
    // orbes.
    rond.style.paddingBottom = Math.round(Sz * 0.085) + "px";
    const wrap = el("div", `position:relative;width:${Sz}px;margin:0 auto;`);
    const layer = el("div", "position:absolute;inset:0;pointer-events:none;");
    // LE POINTILLÉ NE TIENT PAS SUR UN POINT. Il dit « sous le seuil de
    // suggestion », mais sur un disque de 4 px une bordure pointillée de
    // 2,5 px se rend en ÉTOILE : l'orbe cesse d'être un cercle et l'info
    // secondaire détruit l'info principale (sa taille). Sous 12 px de rayon
    // on repasse donc en trait plein, et l'épaisseur suit le disque.
    const pointille = n => n.pr >= 12 && n.o.p < SUGGERE;
    const trait = n => Math.min(n.top ? 2.5 : 1.5, Math.max(0.9, n.pr * 0.35));
    nodes.forEach((n, i) => {
      const dedans = n.pr >= TIENT;
      // LE DISQUE : sa taille est la probabilité, rien d'autre.
      const b = el("div", `position:absolute;left:${n.x}px;top:${n.y}px;`
        + `transform:translate(-50%,-50%);width:${n.pr * 2}px;height:${n.pr * 2}px;`
        + `border-radius:50%;border:${trait(n)}px ${pointille(n) ? "dashed" : "solid"} ${edge(n.o.gam)};`
        + `background:${fill(n.o.gam)};display:flex;flex-direction:column;`
        + `align-items:center;justify-content:center;gap:1px;padding:0;overflow:visible;`
        + `box-shadow:0 2px 6px rgba(60,40,20,.14);animation:ap-orb .3s ${0.04 * i}s both;`);
      if (dedans) {
        b.appendChild(glyph(n.o.sym.root, n.o.sym.q, Math.round(n.pr * 0.58),
                            "exact", encre(n.o.p, niv.cle === "base")));
        b.appendChild(el("span",
          `font:600 ${Math.min(11, Math.round(n.pr * 0.30))}px ${UI};`
          + `font-style:normal;line-height:1;color:${FAINT_ON_PETAL};`, pct(n.o.p)));
      }
      layer.appendChild(b);
      // L'ÉTIQUETTE SORT quand le disque est trop petit pour la porter. Elle
      // se pose vers l'EXTÉRIEUR, dans le prolongement du rayon : c'est la
      // seule direction où l'on est sûr de ne pas retomber sur le moyeu.
      if (!dedans) {
        const d = n.pr + 11, ex = n.x + d * Math.cos(n.a), ey = n.y + d * Math.sin(n.a);
        const droite = Math.cos(n.a) > 0.2, gauche = Math.cos(n.a) < -0.2;
        const lab = el("div", `position:absolute;left:${ex}px;top:${ey}px;`
          + `transform:translate(${droite ? "0" : gauche ? "-100%" : "-50%"},-50%);`
          + `display:flex;align-items:center;gap:4px;white-space:nowrap;`
          + `pointer-events:none;animation:ap-orb .3s ${0.04 * i}s both;`);
        lab.appendChild(glyph(n.o.sym.root, n.o.sym.q, 13, "exact",
                              encre(n.o.p, niv.cle === "base")));
        lab.appendChild(el("span",
          `font:600 9.5px ${UI};font-style:normal;color:${T.faint};`, pct(n.o.p)));
        layer.appendChild(lab);
      }
      // LA ZONE DE CONTACT, invisible et toujours d'au moins 44 px : c'est
      // elle qui porte le plancher tactile, plus la taille du disque. Un orbe
      // à 0,4 % peut donc être un point et rester touchable.
      const z = Math.max(TACTILE, n.pr * 2);
      const hit = el("button", `position:absolute;left:${n.x}px;top:${n.y}px;`
        + `transform:translate(-50%,-50%);width:${z}px;height:${z}px;border-radius:50%;`
        + `border:none;background:transparent;cursor:pointer;pointer-events:auto;padding:0;`);
      hit.title = `${pct(n.o.p)}`;
      hit.onclick = () => {
        chemin.push(niv.cle === "base"
          ? { cle: "base", root: n.o.root, type: n.o.type, sym: n.o.sym, p: n.o.p, pr: n.pr }
          : { cle: niv.cle, i: n.o.i, sym: n.o.sym, p: n.o.p, pr: n.pr });
        dessine();
      };
      layer.appendChild(hit);
    });

    // LE MOYEU EST L'ACCORD COURANT, et on le touche pour VALIDER — même
    // rappel qu'un orbe du compas ordinaire : la prévisualisation joue,
    // « Lock » s'arme. « Ne rien ajouter » n'est donc pas une orbite.
    const sym = chemin.length ? symbole(c) : { root: chord.root, q: chord.q || "" };
    // LA BASSE CHOISIE S'ÉCRIT EN SLASH SUR LE MOYEU. `glyph` sait déjà le
    // faire (son 6e argument) et n'écrit rien quand la basse EST la
    // fondamentale — un `D-7/D` n'existe pas.
    const basseChoisie = (bassOpts.picked == null) ? (chord.bass == null ? -1 : chord.bass)
                                                   : bassOpts.picked;
    const zMoy = Math.max(TACTILE, rMoy * 2);
    const hub = el("button", `position:absolute;left:${cx}px;top:${cy}px;`
      + `transform:translate(-50%,-50%);width:${zMoy}px;height:${zMoy}px;`
      + `border-radius:50%;border:none;background:transparent;cursor:pointer;`
      + `pointer-events:auto;display:flex;align-items:center;justify-content:center;padding:0;`);
    // Le moyeu suit la même loi que les orbes, donc il peut être minuscule :
    // son symbole sort alors sous le disque, là où vit déjà sa légende.
    const larg = 1 + (sym.q || "").length * 0.48;
    const tailleMoy = Math.min(Math.round(Sz * 0.1), Math.round(1.7 * rMoy / larg));
    const moyDedans = tailleMoy >= 12;
    const encreMoy = confColor(chemin.length ? (chemin[0].p || 0) : pMoy);
    if (moyDedans) hub.appendChild(glyph(sym.root, sym.q, tailleMoy, "exact",
                                         encreMoy, basseChoisie));
    hub.onclick = () => { onPick({ root: sym.root, q: sym.q }, hub); };
    layer.appendChild(hub);
    if (!moyDedans) {
      const g = el("div", `position:absolute;left:${cx}px;top:${cy + rMoy + 13}px;`
        + `transform:translate(-50%,-50%);display:flex;align-items:center;gap:5px;`
        + `white-space:nowrap;pointer-events:none;`);
      g.appendChild(glyph(sym.root, sym.q, 17, "exact", encreMoy, basseChoisie));
      g.appendChild(el("span", `font:600 9.5px ${UI};color:${T.faint};`,
                       pct(chemin.length ? (dern.p || 0) : pMoy)));
      layer.appendChild(g);
    }
    // Les anneaux de basse, tapables. Le disque bleu dessiné dans le SVG reste
    // le VISUEL (sa taille dit la probabilité) ; ce bouton n'est que sa zone
    // de contact, portée au minimum tactile de 44 px même quand l'anneau est
    // plus petit — une lecture à 13 % doit rester atteignable au doigt.
    if (bassOpts.onPickBass) bassHits.forEach(h => {
      const d = Math.max(44, h.br * 2);
      const bb = el("button", `position:absolute;left:${h.tx}px;top:${h.ty}px;`
        + `transform:translate(-50%,-50%);width:${d}px;height:${d}px;border-radius:50%;`
        + `border:none;background:transparent;cursor:pointer;pointer-events:auto;padding:0;`);
      bb.title = `${note(h.pc)} — ${Math.round(h.c * 100)} % de l'énergie grave à l'attaque`;
      bb.onclick = () => bassOpts.onPickBass(h.pc);
      layer.appendChild(bb);
    });
    // la légende est SOUS le disque : dedans, elle imposait au moyeu une
    // taille minimale et un moyeu à 1 % ne pouvait pas être petit.
    // La légende du moyeu ne tient que sous un moyeu assez gros : sous un
    // moyeu minuscule elle tomberait en travers de l'orbe voisin, et le pied
    // de page dit déjà la même chose.
    if (moyDedans) layer.appendChild(el("div",
      `position:absolute;left:${cx}px;top:${cy + rMoy + 10}px;`
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
    // LE PIED DIT LA COULEUR, une fois, et seulement quand elle a quelque
    // chose à dire : si tous les orbes sont dans la gamme, la phrase serait du
    // bruit. Sinon il NOMME la sortie, ce qui économise une légende.
    const sortants = nodes.filter(n => n.o.gam !== regne);
    if (sortants.length) {
      const noms = [...new Set(sortants.map(n => nomGamme(n.o.gam)))];
      pied.textContent = (noms.length === 1
        ? "l'orbe d'une autre couleur sort de la gamme, vers " + noms[0]
        : "les orbes d'une autre couleur sortent de la gamme, vers "
          + noms.slice(0, 2).join(" ou "))
        + (niv ? " — touche pour descendre" : "");
    } else {
      pied.textContent = niv
        ? "tout est dans la gamme — touche un orbe pour descendre, le moyeu pour garder le centre"
        : "touche le moyeu pour garder cet accord";
    }
  }

  dessine();
  return box;
}
