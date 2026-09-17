"""Le compas radial : on descend d'un niveau à chaque clic, le choix au centre.

Louis, 2026-09-17 : « fais moi une vraie interface compass en quinconce, ou dès
qu'on clique on arrive au niveau suivant, avec à chaque fois le niveau
sélectionné au centre. Donc premier niveau j'ai le choix entre par exemple
Dm D Gm, si je clique sur Dm j'ai Dm au milieu et comme suggestions Dm7 Dm Dm6,
si je clique sur Dm6 j'ai Dm6/9 ».

SON EXEMPLE DIT LA STRUCTURE, et elle n'est pas celle qu'on aurait devinée :
`Dm6` arrive au MÊME niveau que `Dm7`. Or le 6 n'est pas une septième — c'est
la TREIZIÈME sans septième. Le deuxième niveau n'est donc pas « la 7e » mais
« ce qui se pose au-dessus de la triade » : rien, la sixte, la septième
majeure, la septième mineure. Et `Dm6/9` au troisième confirme la suite : une
fois ce toit posé, on ajoute la 9e, puis la 11e, puis la 13e.

Chaque niveau lit UNE tête du modèle (`musx.frame_posteriors` en rend six,
entraînées séparément), sauf le deuxième qui en croise deux — septième et
treizième — parce que c'est ce que la musique fait.

L'ORBE DIT LA PROBABILITÉ PAR SON AIRE, pas par son rayon : la convention déjà
posée pour le compas de l'app le 2026-08-08 (« area proportional to proba, the
honest encoding »). Un rayon proportionnel exagérerait les gros.

EN QUINCONCE : deux rayons alternés. Avec un seul, six étiquettes d'accord se
chevauchent dès qu'on dépasse quatre options — mesuré à 390 px, c'est le cas
dès le deuxième niveau.

    .venv/bin/python -m tools.page_compas
"""
from __future__ import annotations

import html
import json

from harmonia.settings import SETTINGS

DATA = SETTINGS.repo / "scratchpad" / "cascade.json"
OUT = SETTINGS.repo / "docs" / "plots" / "compas.html"

#: repris de `span_rescore.SUG_FLOOR` — aucun seuil nouveau dans cette page.
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
<title>Le Compas</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=IBM+Plex+Mono:wght@400;500;600&family=Public+Sans:wght@400;500&display=swap">
<style>
 :root{{--bg:#F5F1E7;--surface:#fff;--surface-2:#FBF7EE;--rule:#E3DAC4;--rule-strong:#C9BE9F;
  --ink:#211C14;--ink-dim:#6F6555;--ink-faint:#A79C86;--accent:#B4632A;--accent-soft:#EFDFC5;
  --accent-ink:#5A3315;--mark:#1F6E5C;--mark-soft:#DCEDE7;--warn:#B23A33;--warn-soft:#F6E0DD;}}
 @media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{
  --bg:#14120D;--surface:#1B1812;--surface-2:#211D16;--rule:#332D22;--rule-strong:#4A4130;
  --ink:#ECE4D3;--ink-dim:#A89D89;--ink-faint:#726858;--accent:#DE9251;--accent-soft:#3B2C1B;
  --accent-ink:#F3D2AC;--mark:#59B39C;--mark-soft:#1B302B;--warn:#E1837B;--warn-soft:#3A2220;}}}}
 :root[data-theme="dark"]{{--bg:#14120D;--surface:#1B1812;--surface-2:#211D16;--rule:#332D22;
  --rule-strong:#4A4130;--ink:#ECE4D3;--ink-dim:#A89D89;--ink-faint:#726858;--accent:#DE9251;
  --accent-soft:#3B2C1B;--accent-ink:#F3D2AC;--mark:#59B39C;--mark-soft:#1B302B;
  --warn:#E1837B;--warn-soft:#3A2220;}}
 *{{box-sizing:border-box}}
 body{{background:var(--bg);color:var(--ink);margin:0 auto;max-width:640px;
  font-family:'Public Sans',-apple-system,sans-serif;padding:18px 14px 40px;font-size:14.5px;line-height:1.5}}
 h1{{font-family:'Fraunces',Georgia,serif;font-size:24px;font-weight:600;margin:0;text-wrap:balance}}
 .eyebrow{{font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:.09em;text-transform:uppercase;
  color:var(--accent-ink);background:var(--accent-soft);display:inline-block;padding:2px 7px;
  border-radius:3px;margin-bottom:8px}}
 .sub{{color:var(--ink-dim);font-size:13px;margin:5px 0 0}}
 code{{font-family:'IBM Plex Mono',monospace;font-size:12.5px;background:var(--surface-2);
  border:1px solid var(--rule);border-radius:4px;padding:1px 5px}}
 button{{font-family:'IBM Plex Mono',monospace;font-size:12.5px;border-radius:7px;padding:6px 10px;
  cursor:pointer;border:1px solid var(--rule);background:var(--surface-2);color:var(--ink-dim)}}
 button:hover{{border-color:var(--accent);color:var(--accent-ink);background:var(--accent-soft)}}
 button:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}

 .slots{{display:flex;gap:6px;overflow-x:auto;padding:4px 0 10px;-webkit-overflow-scrolling:touch}}
 .slot{{flex:0 0 auto;display:flex;flex-direction:column;align-items:flex-start;padding:5px 9px;min-width:66px}}
 .slot.on{{background:var(--accent);border-color:var(--accent);color:var(--surface)}}
 .sb{{font-size:9.5px;opacity:.75}} .sq{{font-size:13px;font-weight:600}}

 /* le fil d'ariane : d'où on vient, et on peut y remonter */
 .fil{{display:flex;gap:5px;align-items:center;flex-wrap:wrap;margin:6px 0 2px;
  font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--ink-faint);min-height:30px}}
 .fil .pas{{border:1px solid var(--rule);border-radius:6px;padding:3px 8px;cursor:pointer;
  background:var(--surface-2);color:var(--ink-dim)}}
 .fil .pas:hover{{border-color:var(--accent);color:var(--accent-ink)}}
 .fil .fleche{{color:var(--ink-faint)}}
 .fil .niv{{margin-left:auto;font-size:11px}}

 /* le compas */
 .rond{{position:relative;width:100%;aspect-ratio:1/1;max-width:440px;margin:4px auto 0;
  touch-action:manipulation}}
 .orbe{{position:absolute;display:flex;align-items:center;justify-content:center;
  border-radius:50%;border:1.5px solid var(--rule-strong);background:var(--surface);
  cursor:pointer;transform:translate(-50%,-50%);transition:left .32s cubic-bezier(.4,0,.2,1),
  top .32s cubic-bezier(.4,0,.2,1),width .32s,height .32s,opacity .2s;
  font-family:'IBM Plex Mono',monospace;font-weight:600;color:var(--ink);text-align:center;
  line-height:1.05;padding:2px;overflow:hidden}}
 .orbe:hover{{border-color:var(--accent);color:var(--accent-ink)}}
 .orbe .lb{{padding:0 3px;word-break:keep-all}}
 .orbe .pc{{position:absolute;bottom:5%;font-size:8.5px;font-weight:500;color:var(--ink-faint)}}
 .orbe.top{{border-color:var(--mark)}}
 .orbe.faible{{opacity:.55;border-style:dashed}}
 .centre{{background:var(--accent);border-color:var(--accent);color:var(--surface);cursor:default;
  box-shadow:0 2px 14px rgba(0,0,0,.10)}}
 .centre .pc{{color:rgba(255,255,255,.8)}}
 .rayon{{position:absolute;border:1px dashed var(--rule);border-radius:50%;
  transform:translate(-50%,-50%);pointer-events:none}}

 .barre{{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin-top:10px}}
 .barre .sym{{font-family:'IBM Plex Mono',monospace;font-size:20px;font-weight:600;color:var(--accent-ink)}}
 .barre .av{{font-family:'IBM Plex Mono',monospace;font-size:11.5px;color:var(--ink-faint)}}
 .barre .sp{{flex:1;min-width:4px}}
 .card{{background:var(--surface);border:1px solid var(--rule);border-radius:11px;padding:12px 14px;margin:16px 0}}
 .card p{{margin:7px 0;font-size:12.5px;color:var(--ink-dim)}} .card b{{color:var(--ink)}}
 footer{{margin-top:22px;padding-top:12px;border-top:1px solid var(--rule);font-size:11.5px;
  color:var(--ink-faint);font-family:'IBM Plex Mono',monospace;line-height:1.8}}
</style>

<span class="eyebrow">{e(D['titre'] or D['key'])} &middot; {e(D['tonalite'] or '')}</span>
<h1>Le compas</h1>
<p class="sub">On clique, on descend d'un niveau, et le choix passe au centre.</p>

<div class="slots" id="slots">{rubans}</div>

<div class="fil" id="fil"></div>
<div class="rond" id="rond"></div>

<div class="barre">
  <span class="av" id="av"></span>
  <span class="sym" id="sym">—</span>
  <span class="sp"></span>
  <button id="play" type="button">&#9658; écouter</button>
  <button id="haut" type="button">&#8630; remonter</button>
</div>

<div class="card">
  <p><b>Ce que chaque niveau lit.</b> 1&nbsp;: la triade (racine et type).
  2&nbsp;: ce qui se pose au-dessus — rien, la sixte, la 7<sup>e</sup> majeure, la
  7<sup>e</sup> mineure. 3&nbsp;: la 9<sup>e</sup>. 4&nbsp;: la 11<sup>e</sup>.
  5&nbsp;: la 13<sup>e</sup>. Chacun est une tête du modèle, sauf le 2<sup>e</sup> qui
  en croise deux — un <code>-6</code> est la 13<sup>e</sup> SANS septième, c'est
  pour ça qu'il arrive au même niveau qu'un <code>-7</code>.</p>
  <p><b>La taille de l'orbe est son AIRE</b>, pas son rayon — un rayon
  proportionnel exagérerait les gros. Cerclé de vert&nbsp;: ce que le modèle
  choisirait. En pointillé&nbsp;: sous le seuil de suggestion, proposé quand même,
  parce que ton oreille a le dernier mot.</p>
</div>

<footer>
  têtes : <code>harmonia.musx.frame_posteriors</code> &middot; seuil « suggéré »
  {SUGGERE:.3f}, repris de <code>span_rescore.SUG_FLOOR</code><br>
  démo linéaire des mêmes données : <a href="/plots/cascade.html">cascade.html</a>
</footer>

<script>
const D = {json.dumps(D, ensure_ascii=False, separators=(',', ':'))};
const SUGGERE = {SUGGERE};
const N = D.noms;
const TYPE_TAIL = ["", "-", "sus4", "sus2", "o", "+"];
let sel = 0, chemin = [];     // chemin = [{{cle, i, lab}}] — les choix faits

// ── le symbole ────────────────────────────────────────────────────────────
// Le degré le plus haut nomme l'accord, les inférieurs sont implicites : la
// convention des lead sheets. Les altérations se disent, sans parenthèses —
// `ui/kit.js` a un vrai glyphe pour `7b9`, `7#9`, `7#11`.
function symbole(c){{
  const base = TYPE_TAIL[c.type] || "";
  const alt = [];
  if (c.neuf === 2) alt.push("#9"); if (c.neuf === 3) alt.push("b9");
  if (c.onze === 2) alt.push("#11");
  if (c.treize === 2) alt.push("b13");
  let q;
  if (c.toit === 0) {{                       // rien au-dessus de la triade
    q = base + (c.neuf === 1 ? "add9" : "");
  }} else if (c.toit === 3) {{                // la sixte
    q = base + (c.neuf === 1 ? "69" : "6");
  }} else {{
    let haut = 7;
    if (c.neuf === 1) haut = 9;
    if (c.onze === 1) haut = 11;
    if (c.treize === 1) haut = 13;
    if (c.type === 4) q = (c.toit === 2 ? "h7" : "o7");           // dim
    else if (c.toit === 1) q = (c.type === 1 ? "-^" : "^") + haut; // 7e majeure
    else if (c.type === 2) q = haut + "sus4";
    else if (c.type === 3) q = haut + "sus2";
    else q = (c.type === 1 ? "-" : "") + haut;                     // 7e mineure
  }}
  if (alt.length) q += alt.join("");
  return N.notes[c.root] + q;
}}

// ── l'état courant, reconstruit depuis le chemin ──────────────────────────
function etat(){{
  const c = {{root: 0, type: 0, toit: 0, neuf: 0, onze: 0, treize: 0}};
  for (const p of chemin) {{
    if (p.cle === 'base') {{ c.root = p.root; c.type = p.type; }}
    else c[p.cle] = p.i;
  }}
  return c;
}}

// ── ce que propose le niveau courant ──────────────────────────────────────
// `toit` croise deux têtes : la septième (aucune / maj7 / b7) et la
// treizième, dont `add_13` SANS septième est la sixte.
function niveau(){{
  const a = D.accords[sel], c = etat(), n = chemin.length;
  if (n === 0) return {{cle: 'base', titre: 'la base',
    opts: a.base.slice(0, 7).map(b => ({{
      root: b.root, type: b.type, p: b.c,
      lab: N.notes[b.root] + (TYPE_TAIL[b.type] || '')}}))}};
  if (n === 1) return {{cle: 'toit', titre: 'ce qui se pose dessus',
    opts: [0, 3, 1, 2].map(i => {{
      const p = (i === 3) ? a.treize[1] : a.sev[i];
      const t = {{...c, toit: i}};
      return {{i, p, lab: symbole(t)}};
    }})}};
  // SANS SEPTIÈME, ON S'ARRÊTE APRÈS LA 9e. Une 11e posée sur une triade n'est
  // pas une 11e mais un `add11`, et sur une sixte le symbole n'existe pas — les
  // deux orbes portaient alors la MÊME étiquette (`D-69` deux fois), ce qui est
  // le signe qu'on propose un choix qui n'en est pas un.
  if ((c.toit === 0 || c.toit === 3) && n >= 3) return null;
  const suite = [['neuf', 'la 9e', a.neuf], ['onze', "la 11e", a.onze],
                 ['treize', 'la 13e', a.treize]][n - 2];
  if (!suite) return null;
  return {{cle: suite[0], titre: suite[1],
    opts: suite[2].map((p, i) => {{
      const t = {{...c}}; t[suite[0]] = i;
      return {{i, p, lab: symbole(t)}};
    }})}};
}}

// ── dessin ────────────────────────────────────────────────────────────────
function dessine(){{
  const rond = document.getElementById('rond');
  const W = rond.clientWidth || 360, R = W / 2;
  rond.innerHTML = '';
  const niv = niveau();
  const c = etat();

  // les deux cercles guides du quinconce
  [0.58, 0.76].forEach(f => {{
    const g = document.createElement('div');
    g.className = 'rayon';
    g.style.cssText += `left:50%;top:50%;width:${{W * f}}px;height:${{W * f}}px`;
    rond.appendChild(g);
  }});

  // le centre : le choix courant (ou l'accord écrit tant qu'on n'a rien choisi)
  const dc = Math.max(72, W * 0.28);
  const ce = document.createElement('div');
  ce.className = 'orbe centre';
  ce.style.cssText += `left:50%;top:50%;width:${{dc}}px;height:${{dc}}px;font-size:${{Math.round(dc * 0.26)}}px`;
  ce.textContent = chemin.length ? symbole(c) : D.accords[sel].ecrit;
  rond.appendChild(ce);

  if (!niv) {{
    const fin = document.createElement('div');
    fin.className = 'orbe';
    fin.style.cssText += `left:50%;top:86%;width:${{W * 0.34}}px;height:${{W * 0.14}}px;`
      + `border-radius:12px;font-size:12px;font-weight:500`;
    fin.textContent = 'accord complet';
    rond.appendChild(fin);
  }} else {{
    // EN QUINCONCE : un rayon sur deux. Avec un seul rayon, six étiquettes se
    // chevauchent dès 390 px de large — mesuré, c'est le cas dès le niveau 2.
    const n = niv.opts.length;
    const top = niv.opts.reduce((b, o, i, arr) => o.p > arr[b].p ? i : b, 0);
    niv.opts.forEach((o, k) => {{
      const ang = -Math.PI / 2 + (2 * Math.PI * k) / n;
      // GÉOMÉTRIE : le centre fait 0,28 R de rayon, l'orbe au plus 0,22 R.
      // Les deux couronnes sont donc à 0,58 et 0,76 R — la première dégage le
      // centre (0,58 - 0,22 = 0,36 > 0,28) et la seconde reste dans le disque
      // (0,76 + 0,22 = 0,98). La version d'avant les posait à 0,30 et 0,43,
      // c'est-à-dire DANS le centre : les orbes le recouvraient.
      const rad = R * (k % 2 ? 0.76 : 0.58);
      const d = Math.max(44, Math.min(W * 0.22, 34 + Math.sqrt(Math.max(0, o.p)) * W * 0.17));
      const b = document.createElement('div');
      b.className = 'orbe' + (k === top ? ' top' : '') + (o.p < SUGGERE ? ' faible' : '');
      b.style.cssText += `left:${{50 + Math.cos(ang) * rad / R * 50}}%;`
        + `top:${{50 + Math.sin(ang) * rad / R * 50}}%;width:${{d}}px;height:${{d}}px;`
        + `font-size:${{Math.round(Math.min(15, d * 0.26))}}px`;
      b.innerHTML = `<span class="lb">${{o.lab}}</span>`
                  + `<span class="pc">${{(o.p * 100).toFixed(0)}} %</span>`;
      b.addEventListener('click', () => {{
        chemin.push(niv.cle === 'base'
          ? {{cle: 'base', root: o.root, type: o.type, lab: o.lab}}
          : {{cle: niv.cle, i: o.i, lab: o.lab}});
        dessine();
      }});
      rond.appendChild(b);
    }});
  }}

  // le fil d'ariane
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
    b.className = 'pas'; b.type = 'button'; b.textContent = p.lab;
    b.addEventListener('click', () => {{ chemin = chemin.slice(0, i + 1); dessine(); }});
    fil.appendChild(b);
  }});
  const nn = document.createElement('span'); nn.className = 'niv';
  nn.textContent = niv ? `niveau ${{chemin.length + 1}} — ${{niv.titre}}` : 'terminé';
  fil.appendChild(nn);

  document.getElementById('sym').textContent = chemin.length ? symbole(c) : '—';
  document.getElementById('av').textContent = 'le chart écrit ' + D.accords[sel].ecrit + ' \\u2192';
  document.querySelectorAll('.slot').forEach((s, i) => s.classList.toggle('on', i === sel));
}}

document.querySelectorAll('.slot').forEach(s => s.addEventListener('click', () => {{
  sel = parseInt(s.dataset.i, 10); chemin = []; dessine();
  s.scrollIntoView({{inline: 'center', block: 'nearest', behavior: 'smooth'}});
}}));
document.getElementById('haut').addEventListener('click', () => {{ chemin.pop(); dessine(); }});

// audio : fetch + blob, arrêt piloté par `timeupdate` (les deux pièges iOS)
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
