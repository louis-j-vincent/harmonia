// « POURQUOI ? » — l'écran qui montre comment les autres sections sont trouvées
// une fois qu'une brique est tracée (Louis, 2026-10-11 : « je veux comprendre
// tous les mécanismes derrière chaque étape qui mène à la détection automatique
// des autres sections une fois qu'une est annotée »).
//
// Tout vient de `/api/sections/expliquer/<file>`, qui rejoue LE MÊME calcul que
// « Valider les sections » (même corps `humain`, même code serveur, traces en
// plus — voir harmonia/sections/explication.py) : rien ici n'est un exemple
// figé, tout est recalculé pour ce morceau et ces briques. Le calcul est
// montré TEL QU'IL EST, défauts compris : c'est un outil de débogage.
//
// Plan approuvé : docs/plots/plan_explication_sections_2026-10-11.html (8
// chapitres + les suggestions pâles). Les 5 décisions où l'algo tranche à la
// place de Louis sont en rouge.
//
// SON : chaque plage de mesures se joue par `playSpan` (l'élément audio du
// chart, rien de neuf côté iOS) et le curseur suit l'événement `timeupdate`
// de l'élément média — jamais une boucle requestAnimationFrame (piège iPhone :
// un rendu rAF empêche WebKit de démarrer l'audio).

import { api } from "../api.js";
import { playSpan, stopSpan } from "../audio.js";
import { S } from "../state.js";
import { SERIF, SZ, T, UI, closeOverlay, el, haptic, kitButton, overlay, toast } from "../ui/kit.js";
import { sectColor } from "./sections_editor.js";

const root = document.getElementById("app");
const SVGNS = "http://www.w3.org/2000/svg";

// ── petits outils ──────────────────────────────────────────────────────────
const fr = (v, d = 2) => (v == null || !isFinite(v)) ? "—" : Number(v).toFixed(d).replace(".", ",");
const plage = (a, b) => a === b ? `${a + 1}` : `${a + 1}–${b + 1}`;     // 0-indexé → « 18–25 »
const svg = (t, a) => { const e = document.createElementNS(SVGNS, t); for (const k in (a || {})) e.setAttribute(k, a[k]); return e; };
function couleur(label) {
  const base = String(label || "").replace(/′/g, "").replace(/ \d+$/, "");
  const c = sectColor(base);
  return c === T.faint && /^[A-Z]$/.test(base) ? `hsl(${(base.charCodeAt(0) * 53) % 360} 45% 42%)` : c;
}
const lettreCouleur = l => `hsl(${((String(l).charCodeAt(0) - 97) * 67 + 20) % 360} 50% 58%)`;
const SRC = { humain: ["toi", T.ink], propage: ["retrouvée par ta brique", T.green],
              ressemble: ["devinée (en suspens)", T.amber], algo: ["nommée par l'algo", T.faint] };

let D = null, W = 320, box = null, body = null, lecteur = null;
const P = { tok: 0, curseurs: [], onTU: null };

// ── le son ─────────────────────────────────────────────────────────────────
function barreA(t) {               // position fractionnaire en mesures à l'instant t
  const g = D.barGrid; let lo = 0, hi = g.length - 2;
  if (t <= g[0]) return 0;
  while (lo < hi) { const m = (lo + hi + 1) >> 1; if (g[m] <= t) lo = m; else hi = m - 1; }
  const d = (g[lo + 1] - g[lo]) || 1;
  return lo + Math.min(1, Math.max(0, (t - g[lo]) / d));
}
function peindreCurseur(pos) { P.curseurs.forEach(f => { try { f(pos); } catch (e) {} }); }
function detacher() { if (P.onTU && S.audio) S.audio.removeEventListener("timeupdate", P.onTU); P.onTU = null; }
function jouer(b0, b1, then) {
  if (!D || !S.audio) { toast("Pas d'enregistrement ici"); return; }
  const g = D.barGrid, t0 = g[Math.max(0, b0)], t1 = g[Math.min(b1 + 1, g.length - 1)];
  const tok = ++P.tok;
  detacher();
  const ok = playSpan(t0, t1, () => {
    if (P.tok !== tok) return;                          // remplacé par une autre lecture
    detacher(); peindreCurseur(null); montrerLecteur(null);
    if (then) then();
  });
  if (!ok) { toast("Pas d'enregistrement ici"); return; }
  haptic(5);
  P.onTU = () => { if (P.tok === tok) { const p = barreA(S.audio.currentTime); peindreCurseur(p); montrerLecteur(b0, b1, p); } };
  S.audio.addEventListener("timeupdate", P.onTU);
  peindreCurseur(b0); montrerLecteur(b0, b1, b0);
}
function jouerSuite(plages) {            // [[b0,b1], …] l'une après l'autre
  const [a, ...reste] = plages; if (!a) return;
  jouer(a[0], a[1], reste.length ? () => jouerSuite(reste) : null);
}
function arreter() { P.tok++; detacher(); stopSpan(); peindreCurseur(null); montrerLecteur(null); }
function montrerLecteur(b0, b1, pos) {
  if (!lecteur) return;
  if (b0 == null) { lecteur.style.display = "none"; return; }
  lecteur.style.display = "flex";
  lecteur.firstChild.textContent = `▶ mesures ${plage(b0, b1)} · on entend la ${Math.floor(pos) + 1}`;
}
const btnEcoute = (txt, f) => { const b = kitButton(txt, f, { small: true, height: SZ.control }); b.style.cssText += "max-width:100%;"; return b; };

// ── les briques de dessin ──────────────────────────────────────────────────
function titreDessin(t) { return el("div", `font:600 12px ${UI};color:${T.faint};margin:10px 0 4px;`, t); }
function texte(t, css) { return el("div", `font:14px/1.5 ${UI};color:${T.ink};margin:6px 0;` + (css || ""), t); }
function petit(t) { return el("div", `font:italic 12.5px/1.45 ${SERIF};color:${T.faint};margin:4px 0;`, t); }
function decision(k, phrase) {
  const d = el("div", `border-left:3px solid ${T.accent};background:${T.accent}10;padding:7px 10px;margin:10px 0 4px;` +
    `border-radius:0 8px 8px 0;font:13.5px/1.5 ${UI};color:${T.ink};`);
  d.appendChild(el("b", `color:${T.accent};`, `Décision ${k} — `));
  d.appendChild(document.createTextNode(phrase));
  d.setAttribute("data-decision", String(k));
  return d;
}
/* Une BANDE : le morceau en largeur, des segments [{m0,m1,label,color,
   dashed,faded,onTap}] (0-indexés). Toucher un segment le joue. */
function bande(segs, opts) {
  opts = opts || {};
  const n = D.n, h = opts.h || 26;
  const w = el("div", `position:relative;height:${h}px;background:${T.paper};border:1px solid ${T.line};border-radius:6px;overflow:hidden;`);
  segs.forEach(s => {
    const c = s.color || couleur(s.label);
    const px = W * (s.m1 - s.m0 + 1) / n;
    const d = el("div", `position:absolute;top:0;bottom:0;left:${(100 * s.m0 / n).toFixed(3)}%;width:${(100 * (s.m1 - s.m0 + 1) / n).toFixed(3)}%;` +
      (s.dashed ? `border:1.5px dashed ${c};background:${c}22;` : `background:${c};border-right:1px solid ${T.card};`) +
      `opacity:${s.faded ? .35 : 1};display:flex;align-items:center;justify-content:center;overflow:hidden;cursor:pointer;` +
      `font:700 10px ${UI};color:${s.dashed ? T.ink : "#fff"};white-space:nowrap;box-sizing:border-box;`,
      px > 6 + 6 * String(s.label || "").length ? (s.label || "") : (px > 12 ? String(s.label || "").slice(0, 1) : ""));
    d.title = `${s.label || ""} · mesures ${plage(s.m0, s.m1)}`;
    d.onclick = e => { e.stopPropagation(); (s.onTap || (() => jouer(s.m0, s.m1)))(); };
    w.appendChild(d);
  });
  const cur = el("div", `position:absolute;top:0;bottom:0;width:2px;background:${T.accent};display:none;pointer-events:none;box-shadow:0 0 4px ${T.accent};`);
  w.appendChild(cur);
  P.curseurs.push(pos => { if (pos == null) { cur.style.display = "none"; return; } cur.style.display = "block"; cur.style.left = (100 * pos / n) + "%"; });
  if (!opts.ticks) return w;
  const out = el("div", ""); out.appendChild(w); out.appendChild(graduation()); return out;
}
function graduation() {
  const n = D.n, pas = n > 160 ? 32 : n > 80 ? 16 : 8;
  const g = el("div", `position:relative;height:14px;font:10px ${UI};color:${T.faint};`);
  for (let b = 0; b < n; b += pas) {
    const t = el("div", `position:absolute;left:${(100 * b / n).toFixed(2)}%;top:1px;`, String(b + 1));
    g.appendChild(t);
  }
  return g;
}
function legende(items) {
  const l = el("div", "display:flex;flex-wrap:wrap;gap:4px 12px;margin:4px 0;");
  items.forEach(([txt, c, dashed]) => {
    const it = el("div", `display:flex;align-items:center;gap:5px;font:12px ${UI};color:${T.faint};`);
    it.appendChild(el("span", `display:inline-block;width:14px;height:10px;border-radius:2px;` + (dashed ? `border:1.5px dashed ${c};` : `background:${c};`)));
    it.appendChild(document.createTextNode(txt)); l.appendChild(it);
  });
  return l;
}
/* Une COURBE alignée sur les mesures : la valeur d'indice p (un départ) est
   dessinée au-dessus de la mesure p. */
function courbe(series, opts) {
  opts = opts || {};
  const n = D.n, h = opts.h || 150, pad = 10;
  let lo = opts.ymin, hi = opts.ymax;
  if (lo == null) { lo = Math.min(...series.flatMap(s => s.vals.filter(v => v != null))); lo = Math.max(-1, Math.floor(lo * 10) / 10); }
  if (hi == null) hi = Math.max(1, ...series.flatMap(s => s.vals.filter(v => v != null)));
  const X = p => (p + 0.5) / n * W, Y = v => h - pad - (v - lo) / ((hi - lo) || 1) * (h - 2 * pad);
  const s = svg("svg", { width: W, height: h, viewBox: `0 0 ${W} ${h}` });
  s.style.cssText = `display:block;background:${T.paper};border:1px solid ${T.line};border-radius:6px;touch-action:manipulation;`;
  (opts.zones || []).forEach(z => s.appendChild(svg("rect", { x: z[0] / n * W, y: 0, width: (z[1] - z[0] + 1) / n * W, height: h, fill: z[2] || T.line, opacity: .7 })));
  [lo, (lo + hi) / 2, hi].forEach(v => {
    s.appendChild(svg("line", { x1: 0, x2: W, y1: Y(v), y2: Y(v), stroke: T.line, "stroke-width": 1 }));
    const t = svg("text", { x: 3, y: Y(v) - 2, "font-size": 9, fill: T.faint, "font-family": UI }); t.textContent = fr(v, 1); s.appendChild(t);
  });
  (opts.lignes || []).forEach(l => {
    s.appendChild(svg("line", { x1: 0, x2: W, y1: Y(l.v), y2: Y(l.v), stroke: l.color || T.accent, "stroke-width": 1, "stroke-dasharray": "4 3" }));
    const t = svg("text", { x: W - 3, y: Y(l.v) - 3, "font-size": 9.5, fill: l.color || T.accent, "text-anchor": "end", "font-family": UI }); t.textContent = l.label; s.appendChild(t);
  });
  series.forEach(se => {
    let d = "", pen = false;
    se.vals.forEach((v, p) => { if (v == null) { pen = false; return; } d += (pen ? "L" : "M") + X(p).toFixed(1) + " " + Y(v).toFixed(1); pen = true; });
    s.appendChild(svg("path", { d, fill: "none", stroke: se.color || T.ink, "stroke-width": se.width || 1.4, "stroke-dasharray": se.dash || "", opacity: se.opacity || 1 }));
  });
  (opts.points || []).forEach(pt => {
    const c = svg("circle", { cx: X(pt.p), cy: Y(pt.v), r: pt.r || 4.5, fill: pt.color || T.accent, stroke: T.card, "stroke-width": 1.5 });
    s.appendChild(c);
    if (pt.onTap) {
      const hit = svg("circle", { cx: X(pt.p), cy: Y(pt.v), r: 12, fill: "transparent" });
      hit.style.cursor = "pointer"; hit.addEventListener("click", e => { e.stopPropagation(); pt.onTap(); });
      hit.setAttribute("data-pic", String(pt.p));
      s.appendChild(hit);
    }
  });
  if (opts.onTapX) s.addEventListener("click", e => {
    const r = s.getBoundingClientRect(); const p = Math.round((e.clientX - r.left) / r.width * n - 0.5);
    opts.onTapX(Math.max(0, Math.min(n - 1, p)));
  });
  const cur = svg("line", { x1: 0, x2: 0, y1: 0, y2: h, stroke: T.accent, "stroke-width": 2, display: "none" });
  s.appendChild(cur);
  P.curseurs.push(pos => { if (pos == null) { cur.setAttribute("display", "none"); return; } const x = pos / n * W; cur.setAttribute("x1", x); cur.setAttribute("x2", x); cur.setAttribute("display", "inline"); });
  const out = el("div", ""); out.appendChild(s); out.appendChild(graduation()); return out;
}
/* Une MATRICE carrée en canvas, mise à l'échelle de l'écran (aucun
   débordement : 213 mesures tiennent à ~1,6 px la case). */
function matrice(M, opts) {
  opts = opts || {};
  const J = M.length, taille = Math.min(W, opts.taille || W), dpr = Math.min(2, window.devicePixelRatio || 1);
  const cv = el("canvas", `display:block;width:${taille}px;height:${taille}px;border:1px solid ${T.line};border-radius:4px;touch-action:manipulation;cursor:crosshair;`);
  cv.width = Math.round(taille * dpr); cv.height = Math.round(taille * dpr);
  const cx = cv.getContext("2d"), c = cv.width / J;
  const fond = [247, 243, 233], encre = [42, 30, 26];
  for (let i = 0; i < J; i++) for (let j = 0; j < J; j++) {
    const v = Math.max(0, Math.min(1, M[i][j])), t = Math.pow(v, opts.gamma || 2.2);
    cx.fillStyle = `rgb(${fond.map((f, k) => Math.round(f + (encre[k] - f) * t)).join(",")})`;
    cx.fillRect(j * c, i * c, Math.ceil(c), Math.ceil(c));
  }
  (opts.surligner || []).forEach(([a, b, col]) => {
    cx.strokeStyle = col || T.accent; cx.lineWidth = 1.5 * dpr;
    cx.strokeRect(0, a * c, cv.width, (b - a + 1) * c);
  });
  if (opts.onTap) cv.addEventListener("click", e => {
    const r = cv.getBoundingClientRect();
    const j = Math.floor((e.clientX - r.left) / r.width * J), i = Math.floor((e.clientY - r.top) / r.height * J);
    if (i >= 0 && j >= 0 && i < J && j < J) opts.onTap(i, j);
  });
  return cv;
}
function tableau(lignes, entete) {
  const t = el("table", `border-collapse:collapse;font:12.5px ${UI};color:${T.ink};width:100%;margin:6px 0;`);
  if (entete) { const tr = el("tr", ""); entete.forEach(h => tr.appendChild(el("th", `text-align:left;color:${T.faint};font-weight:600;padding:3px 4px;border-bottom:1px solid ${T.line};`, h))); t.appendChild(tr); }
  lignes.forEach(l => {
    const tr = el("tr", "");
    l.forEach(c => { const td = el("td", `padding:4px;border-bottom:1px solid ${T.line};vertical-align:top;`); if (c instanceof Node) td.appendChild(c); else td.textContent = c; tr.appendChild(td); });
    t.appendChild(tr);
  });
  const w = el("div", "overflow-x:auto;max-width:100%;"); w.appendChild(t); return w;
}
const lien = (txt, f) => { const a = el("button", `border:none;background:none;padding:0;color:${T.accent};font:600 12.5px ${UI};cursor:pointer;text-decoration:underline;min-height:28px;`, txt); a.onclick = e => { e.stopPropagation(); f(); }; return a; };

// ── les chapitres ──────────────────────────────────────────────────────────
function carte(id, num, titre, question) {
  const c = el("section", `background:${T.paper};border:1px solid ${T.line};border-radius:14px;padding:12px 12px 10px;margin:0 0 14px;`);
  c.id = "pq-" + id; c.setAttribute("data-chapitre", id);
  const h = el("div", `display:flex;align-items:center;gap:8px;font:700 15.5px ${UI};color:${T.ink};`);
  h.appendChild(el("span", `flex:0 0 auto;min-width:24px;height:24px;border-radius:12px;background:${T.accent};color:#fff;font:700 12px ${UI};display:inline-flex;align-items:center;justify-content:center;`, num));
  h.appendChild(document.createTextNode(titre)); c.appendChild(h);
  if (question) c.appendChild(el("div", `font:italic 13px ${SERIF};color:${T.accent};margin:3px 0 6px;`, question));
  return c;
}
const segsSections = (secs, opts) => secs.map(s => ({ m0: s.mesure_debut - 1, m1: s.mesure_fin - 1, label: s.label,
  dashed: s.source === "ressemble", color: opts && opts.parSource ? SRC[s.source][1] : null }));
const brique = () => D.briques[D._ib || 0];

function ch0() {
  const c = carte("0", "0", "Le point de départ", "Ce que tu as donné à la machine, et ce qu'elle en a fait.");
  const g = D.nettoyage.gardes;
  c.appendChild(titreDessin("tes briques (touche pour écouter)"));
  c.appendChild(bande(g.map(([a, b, l]) => ({ m0: a, m1: b, label: l }))));
  c.appendChild(titreDessin("ce que « Valider » en tire"));
  c.appendChild(bande(segsSections(D.valider.sections), { ticks: true }));
  const n = k => D.valider.sections.filter(s => s.source === k).length;
  c.appendChild(texte(`Tu as tracé ${g.length} brique${g.length > 1 ? "s" : ""} (${g.map(([a, b, l]) => `${l} ${plage(a, b)}`).join(", ")}). ` +
    `Valider rend ${D.valider.sections.length} sections : ${n("propage")} retrouvée${n("propage") > 1 ? "s" : ""} sous ton nom, ` +
    `${n("ressemble") ? n("ressemble") + " devinée(s), " : ""}${n("algo")} nommée${n("algo") > 1 ? "s" : ""} par l'algo.`));
  if (D.nettoyage.ecartes.length) c.appendChild(texte("⚠ Traits écartés : " + D.nettoyage.ecartes.map(e => `${e.label} (${e.raison})`).join(", "), `color:${T.accent};`));
  c.appendChild(petit(`Les chapitres suivent le calcul dans l'ordre où il se fait. Les encadrés rouges sont les 5 endroits où la machine tranche à ta place. Calculé à l'instant sur ce morceau (${D.durees_ms.total} ms).`));
  return c;
}

const NOTES = ["C", "D♭", "D", "E♭", "E", "F", "G♭", "G", "A♭", "A", "B♭", "B"];
function ch1() {
  const c = carte("1", "1", "Ce que la machine entend", "Chaque mesure devient « quelles notes d'accord sonnent ».");
  c.appendChild(texte("Le modèle d'accords écoute le son ; pour chaque mesure on moyenne ce qu'il entend et on le traduit en 12 notes (les notes des accords probables). Une colonne = une mesure, plus c'est foncé plus la note est présente."));
  const n = D.n, dpr = Math.min(2, window.devicePixelRatio || 1), gut = 24, hh = 12 * 10;
  const wrap = el("div", "position:relative;");
  const cv = el("canvas", `display:block;width:${W}px;height:${hh}px;border:1px solid ${T.line};border-radius:4px;cursor:pointer;`);
  cv.width = W * dpr; cv.height = hh * dpr;
  const cx = cv.getContext("2d"), cw = (W - gut) * dpr / n, rh = hh * dpr / 12;
  cx.fillStyle = T.paper; cx.fillRect(0, 0, cv.width, cv.height);
  cx.fillStyle = T.faint; cx.font = `${9 * dpr}px ${UI}`;
  for (let k = 0; k < 12; k++) cx.fillText(NOTES[11 - k], 2 * dpr, (k + 0.8) * rh);
  D.V.forEach((v, b) => v.forEach((x, k) => {
    const t = Math.max(0, Math.min(1, x / 0.6));
    cx.fillStyle = `rgba(42,30,26,${t.toFixed(3)})`;
    cx.fillRect(gut * dpr + b * cw, (11 - k) * rh, Math.ceil(cw), Math.ceil(rh));
  }));
  const info = petit("touche une colonne : la mesure, ses accords, ses notes fortes — et on l'écoute");
  cv.addEventListener("click", e => {
    const r = cv.getBoundingClientRect(); const b = Math.floor(((e.clientX - r.left) - gut) / (r.width - gut) * n);
    if (b < 0 || b >= n) return;
    const fortes = D.V[b].map((x, k) => [x, k]).filter(([x]) => x > 0.3).sort((a, b) => b[0] - a[0]).map(([, k]) => NOTES[k]);
    info.textContent = `mesure ${b + 1} : ${D.accords[b]} — notes fortes : ${fortes.join(", ") || "aucune"}`;
    jouer(b, b);
  });
  wrap.appendChild(cv);
  const cur = el("div", `position:absolute;top:0;height:${hh}px;width:2px;background:${T.accent};display:none;pointer-events:none;`);
  wrap.appendChild(cur);
  P.curseurs.push(pos => { if (pos == null) { cur.style.display = "none"; return; } cur.style.display = "block"; cur.style.left = (gut + pos / n * (W - gut)) + "px"; });
  c.appendChild(wrap); c.appendChild(info);
  c.appendChild(titreDessin(`ressemblance mesure × mesure (${n} × ${n}) — le cadre rouge = ta brique`));
  c.appendChild(texte("Chaque case compare deux mesures : 1 = les mêmes notes, 0 = rien en commun. Une diagonale sombre parallèle à la grande diagonale = un passage qui se rejoue plus loin."));
  const info2 = petit("touche une case : les deux mesures, leur score, et on les écoute l'une après l'autre");
  c.appendChild(matrice(D.S, {
    surligner: D.nettoyage.gardes.map(([a, b]) => [a, b, T.accent]),
    onTap: (i, j) => {
      info2.textContent = `mesure ${i + 1} (${D.accords[i]}) × mesure ${j + 1} (${D.accords[j]}) : ${fr(D.S[i][j])}`;
      jouerSuite([[i, i], [j, j]]);
    }
  }));
  c.appendChild(info2);
  return c;
}

function choixBrique(c, rerender) {
  if (D.briques.length < 2) return;
  const r = el("div", "display:flex;gap:6px;flex-wrap:wrap;margin:4px 0 6px;");
  D.briques.forEach((b, i) => {
    const on = (D._ib || 0) === i;
    const k = kitButton(`${b.label} ${plage(b.m0, b.m1)}`, () => { D._ib = i; rerender(); }, { small: true, state: on ? "on" : "off", height: 36 });
    r.appendChild(k);
  });
  c.appendChild(r);
}

function ch2() {
  const c = carte("2", "2", "Ta brique est-elle une boucle plus courte ?", "Avant de chercher ailleurs, on regarde si ta brique se répète elle-même.");
  choixBrique(c, () => remplacer("2", ch2));
  const b = brique(), L = b.m1 - b.m0 + 1, bm = b.boucle || {};
  // la brique, mesure par mesure
  const cols = Math.min(L, 8), grille = el("div", `display:grid;grid-template-columns:repeat(${cols},minmax(0,1fr));gap:2px;margin:6px 0;`);
  for (let i = 0; i < L; i++) {
    const m = b.m0 + i;
    const d = el("button", `border:1px solid ${T.line};background:${T.card};border-radius:4px;padding:3px 2px;font:600 10.5px ${UI};color:${T.ink};cursor:pointer;min-height:40px;overflow:hidden;text-overflow:ellipsis;`);
    d.appendChild(el("div", `font:9px ${UI};color:${T.faint};`, String(m + 1)));
    d.appendChild(el("div", "white-space:nowrap;overflow:hidden;text-overflow:ellipsis;", D.accords[m]));
    d.onclick = () => jouer(m, m); grille.appendChild(d);
  }
  c.appendChild(grille);
  if (b.info && b.info.raison === "role") { c.appendChild(texte(`« ${b.label} » ne se rejoue jamais (une intro ou une outro n'a pas de reprise) : rien n'est cherché.`)); return c; }
  if (L < 8) {
    c.appendChild(texte(`Ta brique fait ${L} mesure${L > 1 ? "s" : ""} — moins de 8 : on ne peut pas y voir une boucle de 4 jouée deux fois. Elle est cherchée entière, avec la règle des briques courtes (chapitre 3).`));
    return c;
  }
  c.appendChild(texte(`Seuil « même mesure » pour cette brique : τ = ${fr(bm.tau, 3)} — à mi-chemin entre 1 (identique) et la ressemblance médiane de ses mesures entre elles.`));
  c.appendChild(petit("Pour chaque décalage l, on compare chaque mesure à celle qui est l mesures plus loin, depuis le début de la brique. Il faut au moins l mesures qui collent d'affilée (deux passages), et ce qui reste après ne doit pas dépasser un passage. Le calcul s'arrête au premier décalage qui marche."));
  (b.periode || []).forEach(e => {
    const row = el("div", `margin:6px 0;`);
    row.appendChild(el("div", `font:600 12.5px ${UI};color:${T.ink};`, `décalage l = ${e.l} : ${e.r} mesure${e.r > 1 ? "s" : ""} collent d'affilée (il en faut ${e.l})  →  ${e.ok ? "OUI, c'est une boucle" : "non"}`));
    const cells = el("div", "display:flex;flex-wrap:wrap;gap:2px;margin-top:3px;");
    e.S.forEach((v, i) => {
      const col = i < e.r ? T.green : i === e.r ? T.accent : T.line;
      const k = el("button", `border:none;border-radius:4px;padding:2px 5px;min-height:30px;font:600 11px ${UI};cursor:pointer;background:${col}${i <= e.r ? "" : ""};color:${i <= e.r ? "#fff" : T.faint};`,
        `${b.m0 + i + 1}↔${b.m0 + i + e.l + 1} ${fr(v)}`);
      k.onclick = () => jouerSuite([[b.m0 + i, b.m0 + i], [b.m0 + i + e.l, b.m0 + i + e.l]]);
      cells.appendChild(k);
    });
    row.appendChild(cells); c.appendChild(row);
  });
  if (!(b.periode || []).length) c.appendChild(petit("Aucun décalage essayé."));
  const nb = Math.floor((bm.suite || L) / (bm.boucle || L));
  c.appendChild(texte(bm.boucle < L
    ? `→ Verdict : ta brique = ${nb} × une boucle de ${bm.boucle} mesures${bm.suite < L ? `, puis une fin de ${L - bm.suite} mesure(s)` : ""}. C'est la BOUCLE de ${bm.boucle} qu'on cherche ailleurs.`
    : `→ Verdict : pas de boucle plus courte — la brique est cherchée entière (${L} mesures).`, "font-weight:600;"));
  if (bm.sous_periode && bm.sous_periode < bm.boucle)
    c.appendChild(petit(`À l'intérieur de la boucle, l'harmonie se répète toutes les ${bm.sous_periode} mesures : un départ décalé de ${bm.sous_periode} ressemble autant que le bon, la machine ne peut pas choisir la phase à l'harmonie seule.`));
  return c;
}

function fiche(b, p, cible) {
  cible.innerHTML = "";
  const L = b.m1 - b.m0 + 1, Lb = (b.boucle && b.boucle.boucle) || L;
  const pk = (b.pics || []).find(x => x.p === p);
  const f = el("div", `background:${T.card};border:1px solid ${T.line};border-radius:10px;padding:8px 10px;margin-top:6px;`);
  f.setAttribute("data-fiche", String(p));
  f.appendChild(el("div", `font:700 13.5px ${UI};color:${T.ink};`, `Départ mesure ${p + 1} (mesures ${plage(p, p + Lb - 1)}) — ${D.accords[p]}…`));
  if (!pk) {
    f.appendChild(texte(p + Lb - 1 >= b.m0 && p <= b.m1
      ? "Ce départ chevauche ta brique : c'est ta brique comparée à elle-même, décalée. Jamais une reprise, et il ne compte pas dans le fond."
      : "Ce départ n'a pas de fond calculable (bord du morceau) : il n'est pas jugé."));
  } else {
    const suspens = (b.suspens || []).some(g => (g.departs || []).some(s => s.includes(p)));
    const r3 = pk.c >= D.constantes.plancher && pk.rho >= D.constantes.proeminence;
    const ligne = (ok, regle, question, chiffres) => [ok === null ? "—" : ok ? "✓" : "✗", regle, question, chiffres];
    const ref = b.reference;
    const rows = [
      ligne(pk.c >= D.constantes.plancher, "plancher", "ça ressemble assez ? (score ≥ 0,85)", fr(pk.c, 3)),
      ligne(pk.rho >= D.constantes.proeminence, "proéminence", `se détache de son fond ? fond ${fr(pk.fond)} ; chemin parcouru vers 1 : ${fr(pk.rho)} (≥ 0,6)`, fr(pk.rho)),
      ligne(!r3 ? null : !pk.sous_reference, "référence", ref ? `pas un étage nettement sous la meilleure reprise (${fr(ref.R, 3)}) ?` : "sans objet (moins de 2 candidats)", pk.sous_reference ? "sous le trou" : ""),
      ligne(pk.maximum_local === undefined || (!r3 && !pk.sous_reference) ? null : pk.maximum_local, "maximum local", `le plus haut des candidats à moins d'une boucle (${Lb} mes.) ? égalité à ${fr(b.egalite, 3)} près`, ""),
      ligne(!pk.ok && !pk.pris ? null : (pk.pris || suspens), "couverture", "posé sans chevaucher une autre reprise ?", pk.pris ? "posé" : suspens ? "en suspens (ch. 4)" : "écarté"),
    ];
    f.appendChild(tableau(rows, ["", "règle", "question", "ici"]));
    f.appendChild(texte(pk.pris ? "→ C'est une reprise de ta brique." : suspens ? "→ En suspens : calage rival d'une autre reprise, tranché plus tard (chapitre 4)." : "→ Pas une reprise.", "font-weight:600;"));
  }
  const btns = el("div", "display:flex;gap:6px;flex-wrap:wrap;");
  btns.appendChild(btnEcoute("Ta brique puis ce départ", () => jouerSuite([[b.m0, b.m0 + Lb - 1], [p, Math.min(D.n - 1, p + Lb - 1)]])));
  btns.appendChild(btnEcoute("Ce départ", () => jouer(p, Math.min(D.n - 1, p + Lb - 1))));
  f.appendChild(btns);
  cible.appendChild(f);
}

function ch3() {
  const c = carte("3", "3", "Faire glisser la brique sur tout le morceau", "À chaque départ possible : à quel point ça ressemble à ta brique ?");
  choixBrique(c, () => remplacer("3", ch3));
  const b = brique(), L = b.m1 - b.m0 + 1, Lb = (b.boucle && b.boucle.boucle) || L;
  if (b.info && b.info.raison === "role") { c.appendChild(texte(`« ${b.label} » ne se rejoue jamais : on ne fait rien glisser.`)); return c; }
  if (!b.courbe || !b.courbe.length) { c.appendChild(texte(`Rien à faire glisser (${b.info && b.info.raison || "pas de courbe"}).`)); return c; }
  const ficheBox = el("div", "");
  const pris = new Set((b.pics || []).filter(x => x.pris).map(x => x.p));
  const susp = new Set((b.suspens || []).flatMap(g => (g.departs || []).flat()));
  if (b.mode === "ancien") {
    c.appendChild(texte(`Ta brique fait ${L} mesures (moins de 8) : règle des briques courtes. Seuil du morceau = médiane ${fr(b.mediane, 3)} + 1,5 × écart-type ${fr(b.sigma, 3)} = ${fr(b.seuil, 3)} (borné entre 0,85 et 0,999). Tout départ au-dessus, sans chevauchement, est pris, du plus haut au plus bas.`));
    const reps = (b.reprises || []).filter(o => !o[3] && (o[1] < b.m0 || o[0] > b.m1));
    c.appendChild(courbe([{ vals: b.courbe, color: T.ink }], {
      ymin: 0, zones: [[Math.max(0, b.m0 - Lb + 1), b.m1, T.line]], lignes: [{ v: b.seuil, label: "seuil " + fr(b.seuil, 3) }],
      points: reps.map(o => ({ p: o[0], v: b.courbe[o[0]], color: T.green, onTap: () => jouerSuite([[b.m0, b.m1], [o[0], o[1]]]) })),
      onTapX: p => jouer(p, Math.min(D.n - 1, p + L - 1))
    }));
    c.appendChild(texte(`Reprises gardées : ${reps.map(o => plage(o[0], o[1])).join(", ") || "aucune"}.`));
    return c;
  }
  const fond = new Array(b.courbe.length).fill(null);
  (b.pics || []).forEach(x => { fond[x.p] = x.fond; });
  const cands = (b.pics || []).filter(x => x.c >= D.constantes.plancher || x.pris);
  c.appendChild(courbe([{ vals: b.courbe, color: T.ink }, { vals: fond, color: T.faint, dash: "3 3", width: 1.2 }], {
    ymin: 0, zones: [[Math.max(0, b.m0 - Lb + 1), b.m1, `${T.accent}22`]],
    lignes: [{ v: D.constantes.plancher, label: "plancher 0,85" }],
    points: cands.map(x => ({ p: x.p, v: x.c, r: 5, color: x.pris ? T.green : susp.has(x.p) ? T.amber : T.accent, onTap: () => fiche(b, x.p, ficheBox) })),
    onTapX: p => fiche(b, p, ficheBox)
  }));
  c.appendChild(legende([["courbe : ressemblance à ta brique", T.ink], ["pointillé : fond local", T.faint, true], ["zone rose : chevauche ta brique", `${T.accent}44`],
    ["point : gardé", T.green], ["en suspens", T.amber], ["refusé", T.accent]]));
  c.appendChild(petit(`Touche un point (ou n'importe où sur la courbe) pour la fiche du départ : les 5 règles, leurs chiffres, et l'écoute côte à côte. Boucle cherchée : ${Lb} mesures.`));
  c.appendChild(ficheBox);
  const premier = cands.find(x => x.pris) || cands[0];
  if (premier) fiche(b, premier.p, ficheBox);
  const ref = b.reference;
  c.appendChild(texte(ref
    ? `Référence du morceau : meilleure reprise ${fr(ref.R, 3)} ; plus grand trou entre deux hauteurs de pics ${fr(ref.trou, 3)} (tolérance ${fr(ref.tolerance, 3)}) — ${ref.coupe ? `COUPE : les ${ref.n_bas} pics sous le trou sont refusés` : "pas de coupe"}.`
    : "Référence du morceau : sans objet (moins de deux départs ont passé les règles)."));
  // les transposées
  const tr = (b.transposees || []).filter(t => t.meilleur != null);
  if (tr.length) {
    const best = tr.reduce((a, t) => t.score > a.score ? t : a, tr[0]);
    const trouv = tr.filter(t => (t.trouvees || []).length);
    c.appendChild(titreDessin("la même brique transposée (11 tonalités)"));
    const hh = 70, s = svg("svg", { width: W, height: hh, viewBox: `0 0 ${W} ${hh}` });
    s.style.cssText = `display:block;background:${T.paper};border:1px solid ${T.line};border-radius:6px;`;
    const bw = W / 11, seuil = tr[0].seuil;
    s.appendChild(svg("line", { x1: 0, x2: W, y1: hh - 14 - seuil * (hh - 20), y2: hh - 14 - seuil * (hh - 20), stroke: T.accent, "stroke-dasharray": "4 3" }));
    tr.forEach((t, i) => {
      const hv = Math.max(0, t.score) * (hh - 20);
      s.appendChild(svg("rect", { x: i * bw + 3, y: hh - 14 - hv, width: bw - 6, height: hv, fill: (t.trouvees || []).length ? T.green : T.rule }));
      const tx = svg("text", { x: i * bw + bw / 2, y: hh - 3, "font-size": 9, fill: T.faint, "text-anchor": "middle", "font-family": UI }); tx.textContent = "+" + t.k; s.appendChild(tx);
    });
    c.appendChild(s);
    c.appendChild(petit(`Meilleur score transposé : ${fr(best.score, 3)} (+${best.k} demi-ton${best.k > 1 ? "s" : ""}, départ mesure ${best.meilleur + 1}). Il faudrait au moins ${fr(seuil, 2)} et battre la version non transposée de 0,06 → ${trouv.length ? trouv.length + " trouvée(s)" : "aucune"}.`));
  }
  (b.fins || []).forEach(fn => c.appendChild(petit(`Fin de brique cherchée après ${plage(fn.apres[0], fn.apres[1])} : ressemble à la fin ${fr(fn.st, 3)}, à la boucle ${fr(fn.sl, 3)} → ${fn.prise ? "prise" : "non"}.`)));
  // décision 3
  const haut = (b.pics || []).filter(x => x.c >= D.constantes.plancher);
  const parFond = haut.filter(x => x.rho < D.constantes.proeminence).length;
  const parRef = haut.filter(x => x.sous_reference).length;
  const gard = (b.pics || []).filter(x => x.pris);
  const rhoMin = gard.length ? Math.min(...gard.map(x => x.rho)) : null;
  c.appendChild(decision(3, (parFond || parRef)
    ? `La proéminence (se détacher de son fond d'au moins 0,6) et la référence du morceau séparent une vraie reprise d'un « jumeau » qui joue les mêmes accords. Ici : ${haut.length} départs au-dessus de 0,85, ${parFond} refusé(s) parce qu'ils ne se détachent pas de leur fond, ${parRef} coupé(s) par la référence.`
    : `La proéminence et la référence du morceau séparent une vraie reprise d'un jumeau harmonique. Ici elles ne tranchent rien : ${gard.length} reprise(s) gardée(s), qui se détachent de leur fond d'au moins ${fr(rhoMin)} (il faut 0,6).`));
  return c;
}

const RAISON = { mangee: "une autre section en a mangé une partie", phase: "le plus de contacts avec les autres passages de la même brique (même phase)",
  fragments: "laisse le moins de bouts orphelins (1 à 3 mesures)", score: "au score, faute de mieux", tout_mange: "tout a été mangé par les autres" };
function ch4() {
  const c = carte("4", "4", "Quand plusieurs briques se disputent les mesures", "Qui gagne quand deux reprises veulent la même zone ?");
  const A = D.arbitrage, lab = i => D.nettoyage.gardes[i] ? D.nettoyage.gardes[i][2] : "?";
  D.nettoyage.gardes.forEach(([a, b, l], i) => {
    const segs = [];
    A.verdicts.filter(v => v.brique === i).forEach(v => segs.push({ m0: v.m0, m1: v.m1, label: l, faded: v.verdict === "chevauche", dashed: v.verdict === "retenue" }));
    A.en_suspens.filter(g => g.brique === i).forEach(g => (g.alternatives || []).forEach(alt => alt.forEach(o => segs.push({ m0: o[0], m1: o[1], label: "?", color: T.amber, dashed: true }))));
    c.appendChild(titreDessin(`brique ${l} ${plage(a, b)}`));
    c.appendChild(bande(segs));
  });
  c.appendChild(legende([["posée", couleur(D.nettoyage.gardes.length ? D.nettoyage.gardes[0][2] : "A")], ["refusée : chevauche (pâle)", T.rule], ["retenue / en suspens", T.amber, true]]));
  const ch = A.verdicts.filter(v => v.verdict === "chevauche");
  const rien = !A.en_suspens.length && !A.retenues.length && !ch.length;
  if (rien) c.appendChild(texte("Rien à arbitrer ici : aucune reprise ne dispute ses mesures à une autre, aucun calage en suspens. Ce chapitre ne joue que quand tu traces plusieurs briques qui se marchent dessus."));
  ch.forEach(v => c.appendChild(petit(`${lab(v.brique)} ${plage(v.m0, v.m1)} (score ${fr(v.score, 3)}) refusée : chevauche ${v.contre.map(x => plage(x[0], x[1])).join(", ")}, déjà posé (ton trait ou une reprise d'une brique tracée avant).`)));
  A.retenues.forEach(r => {
    const ret = A.retours.find(x => x.m0 === r[1] && x.m1 === r[2]);
    c.appendChild(petit(`${r[5]} ${plage(r[1], r[2])} (score ${fr(r[3], 3)}) retenue : la zone est tenue par un calage en suspens d'une autre brique, qu'elle ne dépasse pas nettement. ${ret ? (ret.revient ? "Elle revient à la fin (la zone est restée libre)." : "Elle ne revient pas (la zone a été prise).") : ""}`));
  });
  A.en_suspens.forEach(g => {
    c.appendChild(titreDessin(`${g.label} : ${g.alternatives.length} calages rivaux à égalité (tolérance ${fr(g.tolerance, 3)})`));
    c.appendChild(tableau((g.termes || []).map((t, k) => [k === g.choix ? "✓" : "", (g.alternatives[k] || []).map(o => plage(o[0], o[1])).join(" "),
      String(t.mangees), String(t.contacts), String(t.fragments), fr(t.score, 3)]), ["", "calage", "mangées", "contacts", "bouts", "score"]));
    c.appendChild(petit(`Choisi : ${RAISON[g.raison] || g.raison}. (Critères dans l'ordre : mangées, contacts, bouts orphelins, score — chacun ne départage que les ex æquo du précédent.)`));
  });
  c.appendChild(decision(4, rien
    ? "Les suspens et la règle « une reprise sûre ne mange un calage en suspens que si elle est nettement plus haute » : l'arbitrage le plus opaque du calcul. Il ne s'est pas déclenché sur ce morceau avec ces briques."
    : `Les suspens et la règle « une reprise sûre ne mange un calage en suspens que nettement » ont décidé ici : ${A.en_suspens.length} groupe(s) en suspens, ${A.retenues.length} reprise(s) retenue(s), ${ch.length} refusée(s) pour chevauchement.`));
  return c;
}

function bandeJetons(bornes, lettres, opts) {
  return bande(lettres.split("").map((L, j) => ({ m0: bornes[j], m1: bornes[j + 1] - 1, label: L, color: lettreCouleur(L) })), opts);
}
function ch5() {
  const c = carte("5", "5", "Le morceau en mots", "Pour nommer ce qui reste, chaque paire de mesures devient une lettre.");
  const M = D.mot;
  if (!M.bornes || !M.lettres) { c.appendChild(texte("Pas de mot pour ce morceau.")); return c; }
  const J = M.lettres.length;
  c.appendChild(texte(`Le morceau est découpé en ${J} « jetons » de 2 mesures, calés sur les bords de tes briques (et de leurs reprises). Deux jetons aux accords identiques ont toujours la même lettre ; deux jetons différents ne la partagent que s'ils sont presque identiques — aussi ressemblants que deux jetons identiques de ce morceau.`));
  c.appendChild(titreDessin("les jetons et leurs lettres"));
  c.appendChild(bandeJetons(M.bornes, M.lettres, { ticks: true }));
  // les lettres en clair (la bande est trop étroite pour les écrire) : un
  // groupe par tranche de 8 jetons, avec la mesure où il commence
  const mots = el("div", `font:600 12.5px ui-monospace,Menlo,monospace;line-height:1.7;margin:4px 0;display:flex;flex-wrap:wrap;gap:2px 10px;`);
  for (let j = 0; j < J; j += 8) {
    const g = el("button", `border:none;background:none;padding:0;font:inherit;cursor:pointer;color:${T.ink};min-height:28px;`);
    g.appendChild(el("span", `font:10px ${UI};color:${T.faint};margin-right:3px;`, String(M.bornes[j] + 1)));
    M.lettres.slice(j, j + 8).split("").forEach(L => g.appendChild(el("span", `color:${lettreCouleur(L)};`, L)));
    const j1 = Math.min(J, j + 8);
    g.onclick = () => jouer(M.bornes[j], M.bornes[j1] - 1);
    mots.appendChild(g);
  }
  c.appendChild(mots);
  const impairs = [];
  for (let j = 0; j < J; j++) { const w = M.bornes[j + 1] - M.bornes[j]; if (w !== 2) impairs.push([M.bornes[j], M.bornes[j + 1] - 1, w]); }
  const cpt = {}; M.lettres.split("").forEach(l => cpt[l] = (cpt[l] || 0) + 1);
  const [lmax, nmax] = Object.entries(cpt).sort((a, b) => b[1] - a[1])[0];
  c.appendChild(texte(`${Object.keys(cpt).length} lettres. La plus fréquente, « ${lmax} », couvre ${nmax} jetons sur ${J}.` +
    (impairs.length ? ` Jetons qui ne font pas 2 mesures : ${impairs.map(x => `${plage(x[0], x[1])} (${x[2]})`).join(", ")} — ils décalent d'une mesure tout ce qui suit jusqu'au prochain bord.` : "")));
  if (M.source !== "harmonie") c.appendChild(petit("⚠ Pas de ressemblance harmonique disponible : les lettres viennent de l'égalité stricte des basses."));
  if (M.B) {
    c.appendChild(titreDessin(`ressemblance jeton × jeton (${J} × ${J})`));
    const info = petit("touche une case : les deux jetons, leur score, et on les écoute");
    c.appendChild(matrice(M.B, { onTap: (i, j) => {
      info.textContent = `jeton ${i + 1} « ${M.lettres[i]} » (mes. ${plage(M.bornes[i], M.bornes[i + 1] - 1)}) × jeton ${j + 1} « ${M.lettres[j]} » (mes. ${plage(M.bornes[j], M.bornes[j + 1] - 1)}) : ${fr(M.B[i][j])}${M.seuil != null ? (M.B[i][j] >= M.seuil ? " ≥ seuil" : " < seuil") : ""}`;
      jouerSuite([[M.bornes[i], M.bornes[i + 1] - 1], [M.bornes[j], M.bornes[j + 1] - 1]]);
    } }));
    c.appendChild(info);
  }
  if (M.histogramme) {
    c.appendChild(titreDessin("toutes les ressemblances entre jetons, et le seuil mesuré sur les jetons identiques"));
    const h = M.histogramme, hh = 90, s = svg("svg", { width: W, height: hh, viewBox: `0 0 ${W} ${hh}` });
    s.style.cssText = `display:block;background:${T.paper};border:1px solid ${T.line};border-radius:6px;`;
    const lo = h.bords[0], hi = h.bords[h.bords.length - 1], X = v => (v - lo) / (hi - lo) * W, mx = Math.max(...h.comptes);
    const sl = M.seuil == null ? 2 : M.seuil;
    h.comptes.forEach((k, i) => { const hv = k / mx * (hh - 16); s.appendChild(svg("rect", { x: X(h.bords[i]) + 0.5, y: hh - 12 - hv, width: Math.max(1, X(h.bords[i + 1]) - X(h.bords[i]) - 1), height: hv, fill: h.bords[i] >= sl ? T.green : T.rule })); });
    if (M.seuil != null) s.appendChild(svg("line", { x1: X(M.seuil), x2: X(M.seuil), y1: 0, y2: hh - 12, stroke: T.accent, "stroke-width": 2 }));
    [lo, (lo + hi) / 2, hi].forEach(v => { const t = svg("text", { x: Math.min(W - 14, Math.max(2, X(v))), y: hh - 2, "font-size": 9, fill: T.faint, "font-family": UI }); t.textContent = fr(v, 1); s.appendChild(t); });
    c.appendChild(s);
    c.appendChild(petit(M.seuil != null
      ? `Le trait rouge : ${fr(M.seuil, 3)}, la ressemblance la plus BASSE entre deux jetons aux accords identiques de ce morceau (${M.n_identiques} paires). C'est ce que « pareil » veut dire ici ; seules les paires au-dessus (vert) peuvent rapprocher deux jetons différents.`
      : "Aucun jeton n'a d'accords identiques à un autre dans ce morceau : rien n'est rapproché, chaque jeton différent garde sa lettre."));
  }
  const fus = M.fusions || [];
  if (fus.length) {
    c.appendChild(titreDessin("jetons différents rapprochés (presque identiques)"));
    c.appendChild(tableau(fus.map(f => {
      const a = f.a[0], b = f.b[0];
      return [`${plage(M.bornes[a], M.bornes[a + 1] - 1)} ≈ ${plage(M.bornes[b], M.bornes[b + 1] - 1)}`, fr(f.lien, 3),
        lien("écouter", () => jouerSuite([[M.bornes[a], M.bornes[a + 1] - 1], [M.bornes[b], M.bornes[b + 1] - 1]]))];
    }), ["jetons", "ressemblance", ""]));
  }
  c.appendChild(decision(2, `Les lettres décident ce que l'algo considère comme « la même musique ». Accords identiques = même lettre ; sinon il faut être aussi ressemblant que deux passages identiques du morceau (${M.seuil != null ? fr(M.seuil, 3) : "aucun repère ici"}). Un passage que le décodeur entend un peu autrement reçoit donc une autre lettre. Et les jetons sont calés sur le début des trous${impairs.length ? ", pas sur la phrase : " + impairs.map(x => plage(x[0], x[1])).join(", ") + " font " + impairs.map(x => x[2]).join("/") + " mesures" : ""}.`));
  return c;
}

function decompose(nom) {         // coût = dictionnaire + partition + primes + restes (phrases4.cout)
  const types = new Set(nom.filter(x => !x.queue).map(x => x.type));
  const d = [...types].reduce((s, t) => s + t.length, 0), p = nom.length, pr = nom.filter(x => x.prime).length;
  const r = nom.filter(x => x.queue).reduce((s, x) => s + x.j1 - x.j0, 0);
  return { d, p, pr, r };
}
function ch6() {
  const c = carte("6", "6", "Souder les mots en phrases", "La paire de jetons voisins la plus fréquente est soudée, encore et encore.");
  const Sd = D.soudures, M = D.mot;
  if (!Sd.essais || !Sd.essais.length) { c.appendChild(texte("Pas de soudure pour ce morceau.")); return c; }
  const gel = new Set((Sd.geles || []).map(g => g.join(",")));
  c.appendChild(texte(`Tes briques et leurs reprises entrent GELÉES (${gel.size} blocs) : elles ne se soudent à rien. On soude la paire la plus fréquente (au moins 2 fois), sans dépasser une phrase de 4 jetons (8 mesures)… ou de 6 (12 mesures) ; on essaie les deux et on garde la moins chère à écrire.`));
  const ess = Sd.essais.find(e => e.cible === Sd.cible) || Sd.essais[0];
  const steps = ess.steps || [];
  let k = steps.length - 1;
  const titre = el("div", `font:600 13px ${UI};color:${T.ink};margin:8px 0 4px;`);
  const zone = el("div", "");
  const peindre = () => {
    const st = steps[k];
    titre.textContent = k === 0 ? `pas 0 : le départ (phrases de ${ess.cible})` :
      `pas ${k}/${steps.length - 1} : (${st.paire[0]}, ${st.paire[1]}) soudés ×${st.compte} → ${st.paire[0] + st.paire[1]}`;
    zone.innerHTML = "";
    zone.appendChild(bande(st.jetons.map(([j0, j1, t]) => ({ m0: M.bornes[j0], m1: M.bornes[j1] - 1,
      label: gel.has(j0 + "," + j1) ? (Sd.par_jeton[j0 + "," + j1] || "gelé") : t,
      color: gel.has(j0 + "," + j1) ? couleur(Sd.par_jeton[j0 + "," + j1]) : lettreCouleur(t[0]) })), { ticks: true, h: 30 }));
  };
  const nav = el("div", "display:flex;gap:6px;align-items:center;margin-top:6px;");
  let iv = null;
  const stop = () => { if (iv) clearInterval(iv); iv = null; };
  nav.appendChild(kitButton("◀", () => { stop(); k = Math.max(0, k - 1); peindre(); }, { small: true }));
  nav.appendChild(kitButton("▶", () => { stop(); k = Math.min(steps.length - 1, k + 1); peindre(); }, { small: true }));
  nav.appendChild(kitButton("Rejouer", () => { stop(); k = 0; peindre(); iv = setInterval(() => { if (k >= steps.length - 1 || !box || !box.isConnected) { stop(); return; } k++; peindre(); }, 900); }, { small: true }));
  c.appendChild(titre); c.appendChild(zone); c.appendChild(nav); peindre();
  c.appendChild(titreDessin("ce que coûte chaque hypothèse (ce qu'il faut écrire pour dire tout le morceau)"));
  c.appendChild(tableau(Sd.essais.map(e => { const x = decompose(e.nom || []); return [`phrases de ${e.cible}`, String(x.d), String(x.p), String(x.pr), String(x.r), String(e.cout), e.cible === Sd.cible ? "✓ gagne" : ""]; }),
    ["", "dico", "renvois", "primes", "restes", "coût", ""]));
  c.appendChild(petit("dico = longueur des phrases distinctes ; renvois = une par section ; primes = une fin qui change ; restes = ce qui n'est entré dans aucune phrase, payé en entier."));
  c.appendChild(titreDessin("nommer : même phrase = même lettre ; différence sur le seul dernier jeton = prime (A′)"));
  c.appendChild(bande((Sd.restes || []).map(x => ({ m0: M.bornes[x.j0], m1: M.bornes[x.j1] - 1, label: (x.label || "?") + (x.prime ? "′" : "") + (x.queue ? "·" : ""), dashed: !!x.queue }))));
  c.appendChild(petit("En pointillé : les restes trop courts pour une phrase, regroupés avec leurs voisins. Ces lettres sont celles de l'algo ; elles seront renommées pour ne pas reprendre les tiennes (chapitre 7)."));
  return c;
}

function ch7() {
  const c = carte("7", "7", "Les règles de nommage finales", "Les règles écrites à la main qui retouchent le résultat.");
  const N = D.nommage, M = D.mot;
  // intro de tête
  c.appendChild(titreDessin("① intro de tête"));
  const tete = N.tete;
  if ((N.intro_avale || []).length) {
    const av = N.intro_avale;
    c.appendChild(petit("avant la règle (ce que l'algo entendait avant ta première brique) :"));
    c.appendChild(bande(av.map(x => ({ m0: x.m0, m1: x.m1, label: x.label, color: lettreCouleur(String(x.label).toLowerCase()), dashed: x.queue }))));
    c.appendChild(petit("après :"));
    c.appendChild(bande([{ m0: av[0].m0, m1: av[av.length - 1].m1, label: "intro", color: couleur("intro") }]));
    const nonQ = av.filter(x => !x.queue);
    c.appendChild(decision(1, `Tout ce qui précède ta première brique (mesures 1–${tete}) devient UNE « intro »${nonQ.length ? `, alors que l'algo y entendait ${nonQ.map(x => `${x.label} (${plage(x.m0, x.m1)})`).join(" et ")}` : ""}. La règle ne regarde pas ce qui est joué.`));
  } else c.appendChild(texte(N.a_deja_une_intro ? "Non déclenchée : tu as tracé ta propre intro." : "Non déclenchée : rien avant ta première brique."));
  // B + queue
  c.appendChild(titreDessin("② section + queue (un bloc qui joue une autre section puis continue)"));
  const pq = (N.prefixe || []).filter(x => x.nom_fixe);
  c.appendChild(texte(pq.length ? "Déclenchée : " + pq.map(x => `${x.nom_fixe} ${plage(M.bornes[x.j0], M.bornes[x.j1] - 1)}`).join(", ") : "Non déclenchée."));
  // queue optionnelle
  c.appendChild(titreDessin("③ queue optionnelle (1–2 mesures que certaines occurrences ont à leur fin, pas les autres)"));
  const essais = (N.queue_opt || []);
  const gardes = essais.filter(e => (e.retenus || []).some(r => r.garde));
  if (!gardes.length) c.appendChild(texte("Non déclenchée."));
  gardes.forEach(e => (e.retenus || []).filter(r => r.garde).forEach(r => {
    const g = (e.groupes || []).find(x => JSON.stringify(x.avec) === JSON.stringify(r.avec)) || {};
    const x0 = g.x ? g.x[0] : null, x1 = g.x ? g.x[1] : null;
    c.appendChild(texte(`${e.label} est suivi ${r.avec.length} fois de la même mesure ${x0 != null ? "(" + D.accords[x0] + ")" : ""}, et ${r.sans.length} fois directement de la suite (${g.f != null ? D.accords[g.f] : "?"}) : cette mesure est une queue de ${e.label}, pas le début de la section d'après.`));
    const rows = [];
    r.avec.forEach(o => { const k = e.k; rows.push(["avec", `${e.label} ${plage(o[0], o[1])} → ${D.accords[o[1] + 1]} → ${D.accords[o[1] + 1 + k] || "fin"}`, lien("écouter", () => jouer(Math.max(0, o[1] - 1), Math.min(D.n - 1, o[1] + 1 + k)))]); });
    r.sans.forEach(o => rows.push(["sans", `${e.label} ${plage(o[0], o[1])} → ${D.accords[o[1] + 1] || "fin"}`, lien("écouter", () => jouer(Math.max(0, o[1] - 1), Math.min(D.n - 1, o[1] + 1)))]));
    c.appendChild(tableau(rows, ["", "fin de section → mesure suivante", ""]));
  }));
  // les candidats qu'UNE seule occurrence a ne sont jamais des queues : on
  // les compte sans les lister (ils noieraient les vrais refus)
  const tous = essais.flatMap(e => (e.groupes || []).filter(g => g.ecarte).map(g => [e, g]));
  const seuls = tous.filter(([, g]) => g.avec.length < 2).length;
  const ecartes = tous.filter(([, g]) => g.avec.length >= 2).map(([e, g]) =>
    `${e.label} suivi ${g.avec.length} fois de la mesure ${plage(g.x[0], g.x[1])} (${D.accords[g.x[0]]}) : ${g.ecarte}`);
  if (ecartes.length || seuls) c.appendChild(petit((ecartes.length ? "Essais écartés : " + ecartes.join(" ; ") + ". " : "") +
    (seuls ? `${seuls} autre(s) fin(s) n'arrivent qu'une fois : pas une queue.` : "")));
  const props = (D.valider.queues_proposees || []);
  if (props.length) c.appendChild(petit("Queues seulement PROPOSÉES (elles tomberaient dans un bloc que tu as tracé, qu'on ne coupe jamais) : " + props.map(q => `${q.label} ${q.mesure_debut}–${q.mesure_fin}`).join(", ") + ". L'écran ne les montre pas encore."));
  c.appendChild(decision(5, gardes.length
    ? `La queue optionnelle exige au moins 2 « avec » et 1 « sans » ; ici elle détache ${gardes.reduce((s, e) => s + e.retenus.filter(r => r.garde).reduce((t, r) => t + r.avec.length, 0), 0)} mesure(s), ce qui change la longueur des sections d'après.`
    : "La queue optionnelle (au moins 2 « avec », au moins 1 « sans ») ne s'est pas déclenchée ici."));
  // dernier tour
  c.appendChild(titreDessin("④ dernier tour (la fin du morceau rejoue le début d'une lettre puis s'éteint)"));
  const dt = N.dernier_tour || {};
  c.appendChild(texte(dt.raison ? `Dernier bloc ${dt.dernier ? `${dt.dernier[2]} ${plage(dt.dernier[0], dt.dernier[1])}` : ""} : ${dt.raison}.` : "Non examiné."));
  c.appendChild(titreDessin("avant ces règles → après"));
  c.appendChild(bande((N.prefixe || []).map(x => ({ m0: M.bornes[x.j0], m1: M.bornes[x.j1] - 1, label: (x.label || "?") + (x.prime ? "′" : ""), color: Sd_couleur(x) }))));
  c.appendChild(bande((N.final || []).map(x => ({ m0: x.m0, m1: x.m1, label: x.label })), { ticks: true }));
  return c;
}
function Sd_couleur(x) {
  const k = D.soudures.par_jeton[x.j0 + "," + x.j1];
  return k ? couleur(k) : lettreCouleur(String(x.label || "?").toLowerCase());
}

const RAISON_PILE = r => {
  if (!r) return "";
  if (/stack incoherent/.test(r)) return "les passages empilés ne disent pas la même chose";
  if (/unequal length/.test(r)) return "pas de boucle sûre (passages de longueurs inégales) : rien à empiler";
  if (/single occurrence|only one/.test(r)) return "une seule occurrence";
  return r;
};
function ch8() {
  const c = carte("8", "8", "Du résultat au chart", "D'où vient chaque section, et comment elle est écrite.");
  const V = D.valider.sections, E = D.ecriture;
  c.appendChild(titreDessin("les sections"));
  c.appendChild(bande(segsSections(V)));
  c.appendChild(titreDessin("leur origine"));
  c.appendChild(bande(segsSections(V, { parSource: true }), { ticks: true }));
  c.appendChild(legende(Object.values(SRC).map(([t, col]) => [t, col])));
  const ajout = E.combler.filter(x => !V.some(v => v.mesure_debut === x.mesure_debut && v.mesure_fin === x.mesure_fin));
  if (ajout.length) c.appendChild(petit("Trous comblés par l'écran avant l'écriture : " + ajout.map(x => `${x.label} ${x.mesure_debut}–${x.mesure_fin}`).join(", ") + "."));
  if (E.scindees.length !== E.combler.length) c.appendChild(petit(`Sections tracées sur plusieurs fois leur longueur habituelle, coupées : ${E.combler.length} → ${E.scindees.length}.`));
  c.appendChild(titreDessin(`empilement : les passages d'une même lettre sont superposés et réécrits ensemble si leur cohérence ≥ ${fr(D.constantes.coherence_pile)}`));
  const rows = Object.entries(E.rapport).map(([L, r]) => {
    const coh = r.coh && r.coh.length ? Math.min(...r.coh) : E.coh_min[L];
    const ok = !r.reason;
    return [L, r.period ? r.period + " mes." : "—", coh != null ? fr(coh) : "—", ok ? "✓ empilé" : "✗ " + RAISON_PILE(r.reason), ok ? String((r.changed || []).length) : "0"];
  });
  c.appendChild(tableau(rows, ["lettre", "longueur", "cohérence", "verdict", "réécrites"]));
  c.appendChild(texte(E.empile.ok ? `${E.empile.n_reecrites} mesure(s) réécrite(s) par l'empilement.` : `Empilement refusé : ${E.empile.raison || "?"}.`));
  c.appendChild(titreDessin("ce que le chart écrirait (rien n'a été écrit)"));
  c.appendChild(tableau((E.sections_chart || []).map(s => [s.label || "?", (s.barRanges || []).map(r => plage(r[0], r[1])).join(", "), "×" + (s.barRanges || []).length]), ["section", "passages", ""]));
  return c;
}

const RAISON_PALE = { parite: "à une distance IMPAIRE de ta brique : la recherche pâle n'accepte que les départs à un nombre pair de mesures",
  chevauche_la_brique: "chevauche ta brique", deborde_la_fin: "déborde la fin du morceau", deja_pris: "mesures déjà à toi",
  voisin_plus_fort: "un départ plus fort juste à côté l'a pris" };
function chBonus() {
  // Depuis le 2026-10-11 (Evening Star) les pâles passent par le MÊME calcul que
  // Valider (`simulation.propositions_pales` → `cherche_brique`) : ce chapitre
  // ne montre plus deux moteurs qui divergent, il montre que les deux listes
  // sont la même — et dit ce qui peut encore les séparer (l'arbitrage).
  const c = carte("p", "+", "À part : les suggestions pâles", "Ce que l'outil propose en pâle dès que tu traces une brique.");
  choixBrique(c, () => remplacer("p", chBonus));
  const b = brique(), p = (D.pales || []).find(x => x.m0 === b.m0 && x.m1 === b.m1);
  if (!p || p.erreur) { c.appendChild(texte("Pas de suggestions pâles pour cette brique" + (p && p.erreur ? " : " + p.erreur : "."))); return c; }
  const prop = p.proposees || [];
  c.appendChild(texte(`Les pâles font maintenant exactement le calcul de Valider (chapitre 3) pour ta brique seule : ${prop.length} proposition(s)${prop.length ? " — " + prop.map(x => `mesures ${x.b0 + 1}–${x.b1 + 1}`).join(", ") : ""}.`, "font-weight:600;"));
  if (prop.length) c.appendChild(tableau(prop.map(x => [`${x.b0 + 1}–${x.b1 + 1}`, fr(x.score, 3), lien("écouter", () => jouer(x.b0, x.b1))]), ["mesures", "ressemblance", ""]));
  c.appendChild(petit("Ce qui peut encore les séparer de la décision finale : Valider arbitre ensuite entre TOUTES tes briques (chapitre 4) — une pâle peut céder sa place à la reprise d'une autre brique. Avant le 2026-10-11 les pâles cherchaient la plus petite « cellule » cohérente de ta brique (2 mesures sur Evening Star) : d'où des bouts de 2 mesures."));
  return c;
}

// ── la feuille ─────────────────────────────────────────────────────────────
const CHAPITRES = [["0", "Départ", ch0], ["1", "Écoute", ch1], ["2", "Boucle", ch2], ["3", "Glisser", ch3], ["4", "Arbitrage", ch4],
  ["5", "Mots", ch5], ["6", "Phrases", ch6], ["7", "Règles", ch7], ["8", "Chart", ch8], ["p", "+ Pâles", chBonus]];
// Changer de brique redessine TOUT (les chapitres 2, 3 et « pâles » en
// dépendent, et chaque dessin enregistre son curseur de lecture : tout
// reconstruire est la façon simple de ne garder aucun curseur mort).
function remplacer() { rebatir(false); }
function rebatir(premier) {
  const y = body.scrollTop;
  P.curseurs = [];
  const frag = document.createDocumentFragment();
  CHAPITRES.forEach(([id, , f]) => {
    try { frag.appendChild(f()); }
    catch (e) { const c = carte(id, id, "Chapitre impossible à dessiner", ""); c.appendChild(texte(String(e && e.message || e), `color:${T.accent};`)); frag.appendChild(c); console.warn("pourquoi", id, e); }
  });
  body.innerHTML = ""; body.appendChild(frag);
  if (!premier) body.scrollTop = y;
}

/* Ouvre l'écran pour les traits `humain` (le MÊME corps que Valider). */
export async function openPourquoi(humain) {
  const m = S.model; if (!m || !m.file) return;
  const back = overlay();
  back.style.alignItems = "stretch";
  box = el("div", `width:100%;max-width:560px;height:100%;background:${T.card};display:flex;flex-direction:column;overflow:hidden;` +
    `padding-top:env(safe-area-inset-top);box-sizing:border-box;animation:ap-sheet .3s cubic-bezier(.32,.9,.35,1);`);
  box.setAttribute("data-pourquoi", "1");
  const close = () => { arreter(); P.curseurs = []; D = null; closeOverlay(back); box = null; };
  back.onclick = e => { if (e.target === back) close(); };
  const head = el("div", `flex:0 0 auto;display:flex;align-items:center;gap:8px;padding:10px 12px 6px 16px;`);
  head.appendChild(el("div", `flex:1;min-width:0;font:700 16px ${UI};color:${T.ink};white-space:nowrap;overflow:hidden;text-overflow:ellipsis;`, "Pourquoi ces sections ?"));
  const x = el("button", `flex:0 0 auto;min-height:44px;padding:0 12px;border:none;background:transparent;color:${T.accent};font:600 14px ${UI};cursor:pointer;`, "Fermer");
  x.onclick = close; head.appendChild(x);
  box.appendChild(head);
  const nav = el("div", `flex:0 0 auto;display:flex;gap:5px;overflow-x:auto;padding:0 12px 8px;border-bottom:1px solid ${T.line};`);
  nav.className = "ap-nobar";
  box.appendChild(nav);
  body = el("div", `flex:1 1 auto;overflow-y:auto;overflow-x:hidden;padding:12px 16px calc(80px + env(safe-area-inset-bottom));-webkit-overflow-scrolling:touch;`);
  box.appendChild(body);
  lecteur = el("div", `position:absolute;left:12px;right:12px;bottom:calc(12px + env(safe-area-inset-bottom));display:none;align-items:center;gap:8px;` +
    `background:${T.deep};color:#fff;border-radius:22px;padding:4px 6px 4px 14px;box-shadow:0 8px 22px -8px rgba(0,0,0,.5);z-index:3;`);
  lecteur.appendChild(el("div", `flex:1;min-width:0;font:600 12.5px ${UI};white-space:nowrap;overflow:hidden;text-overflow:ellipsis;`, ""));
  const st = el("button", `flex:0 0 auto;min-height:40px;padding:0 12px;border:none;border-radius:18px;background:${T.accent};color:#fff;font:600 13px ${UI};cursor:pointer;`, "■ Arrêter");
  st.onclick = arreter; lecteur.appendChild(st);
  box.style.position = "relative"; box.appendChild(lecteur);
  back.appendChild(box); root.appendChild(back);
  body.appendChild(el("div", `font:italic 14px ${SERIF};color:${T.faint};padding:30px 0;text-align:center;`, "je refais le calcul de « Valider », étape par étape…"));
  try {
    const r = await api.post("/api/sections/expliquer/" + m.file, { humain });
    if (!box) return;                                       // fermé entre-temps
    if (r.error) throw new Error(r.error);
    D = r; D._ib = 0;
    // largeur utile d'un dessin : le corps moins ses marges (2×16) et celles
    // de la carte (2×12 + bordures)
    W = Math.max(220, body.clientWidth - 32 - 26);
    CHAPITRES.forEach(([id, nom]) => {
      const b = el("button", `flex:0 0 auto;min-height:36px;padding:0 10px;border-radius:18px;border:1px solid ${T.line};background:${T.paper};color:${T.ink};font:600 12.5px ${UI};cursor:pointer;white-space:nowrap;`,
        (id === "p" ? "" : id + " ") + nom);
      b.onclick = () => { const c = body.querySelector(`[data-chapitre="${id}"]`); if (c) body.scrollTo({ top: c.offsetTop - body.offsetTop - 8, behavior: "smooth" }); };
      nav.appendChild(b);
    });
    rebatir(true);
  } catch (e) {
    if (!box) return;
    body.innerHTML = "";
    let msg = String((e && e.message) || e); try { const j = JSON.parse(msg); if (j && j.error) msg = j.error; } catch (_) {}
    body.appendChild(texte("L'explication a échoué : " + msg, `color:${T.accent};`));
  }
}
