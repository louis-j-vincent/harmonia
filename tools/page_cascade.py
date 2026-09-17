"""Le compas en cascade : la base, puis la 7e, la 9e, la 11e, la 13e.

Louis, 2026-09-17 : « fais en sorte qu'on puisse avoir des extensions, et
ensuite j'ai envie que tu me fasses une demo d'un compas ameliore, ou je
selectionne d'abord l'accord en maj/min, ensuite des qu'on le selectionne on
select la 7eme, puis la 9eme, puis la 11eme, puis la 13eme si elle est
suggeree ».

POURQUOI CETTE FORME EST LA BONNE. Le modèle est déjà fait comme ça :
`frame_posteriors` rend SIX tableaux — triade (73 = N + 12 racines × 6 types),
basse (13), septième (4), neuvième (4), onzième (3), treizième (3) — entraînés
comme des têtes catégorielles séparées. Le compas en cascade ne fait donc pas
que ranger les boutons autrement : il lit la structure du modèle, un étage par
tête. Chaque étage montre SA distribution, pas une opinion recalculée.

CE QUE ÇA DÉBLOQUE, et c'est le point. Mesuré la veille : dans le décodage, les
extensions ne gagnent jamais — la 9e trois fois sur 1255 accords, la 11e et la
13e zéro — parce qu'une extension multiplie le score par la probabilité de sa
classe et que « aucune » domine. Elles sont pourtant LÀ : sur Lost Without U,
23 accords portent une 9e au-dessus de 25 %, jusqu'à 54 % sur un `G7sus4`, ce
qui en fait un `G9sus`. En cascade, l'extension n'a plus à GAGNER contre
« aucune » dans un Viterbi — elle est proposée, et l'oreille tranche. C'est la
piste notée la veille comme la seule sortie de l'impasse.

    .venv/bin/python -m tools.page_cascade
"""
from __future__ import annotations

import html
import json

from harmonia.settings import SETTINGS

DATA = SETTINGS.repo / "scratchpad" / "cascade.json"
OUT = SETTINGS.repo / "docs" / "plots" / "cascade.html"

#: au-dessus de cette part, un degré est « suggéré » et l'étage s'ouvre tout
#: seul dessus. En dessous, l'étage dit « rien de suggéré » et propose
#: « aucune » — sans jamais l'interdire : Louis peut forcer.
#: Valeur reprise de `span_rescore.SUG_FLOOR` (le plancher d'affichage déjà
#: arbitré), PAS inventée ici — un seuil de plus aurait été un seuil posé.
SUGGERE = 0.125


def page(D: dict) -> str:
    e = html.escape
    ac = D["accords"]
    forts = sum(1 for a in ac if max(a["neuf"][1:] + a["onze"][1:] + a["treize"][1:]) >= 0.25)

    lignes = "".join(
        f"""<button class="slot" data-i="{i}" type="button">
             <span class="sb">{a['bar']}.{a['beat']}</span>
             <span class="sq">{e(a['ecrit'])}</span>
             {"<span class='sx'>9e " + str(round(max(a['neuf'][1:]) * 100)) + " %</span>"
              if max(a['neuf'][1:]) >= 0.25 else ""}
           </button>""" for i, a in enumerate(ac))

    return f"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Le Compas En Cascade</title>
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
 body{{background:var(--bg);color:var(--ink);margin:0 auto;max-width:900px;
  font-family:'Public Sans',-apple-system,sans-serif;padding:20px 14px 90px;font-size:14.5px;line-height:1.5}}
 h1{{font-family:'Fraunces',Georgia,serif;font-size:25px;font-weight:600;margin:0;text-wrap:balance}}
 h2{{font-family:'Fraunces',Georgia,serif;font-size:16px;margin:26px 0 8px}}
 .eyebrow{{font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:.09em;text-transform:uppercase;
  color:var(--accent-ink);background:var(--accent-soft);display:inline-block;padding:2px 7px;
  border-radius:3px;margin-bottom:8px}}
 .sub{{color:var(--ink-dim);font-size:13px;margin:5px 0 0}}
 code{{font-family:'IBM Plex Mono',monospace;font-size:12.5px;background:var(--surface-2);
  border:1px solid var(--rule);border-radius:4px;padding:1px 5px}}
 .card{{background:var(--surface);border:1px solid var(--rule);border-radius:11px;padding:13px 15px;margin:14px 0}}
 .card p{{margin:8px 0;font-size:13px;color:var(--ink-dim)}} .card b{{color:var(--ink)}}
 button{{font-family:'IBM Plex Mono',monospace;font-size:12.5px;border-radius:7px;padding:7px 11px;
  cursor:pointer;border:1px solid var(--rule);background:var(--surface-2);color:var(--ink-dim)}}
 button:hover{{border-color:var(--accent);color:var(--accent-ink);background:var(--accent-soft)}}
 button:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}

 .slots{{display:flex;gap:6px;overflow-x:auto;padding:4px 0 8px;-webkit-overflow-scrolling:touch}}
 .slot{{flex:0 0 auto;display:flex;flex-direction:column;align-items:flex-start;gap:1px;
  padding:6px 9px;min-width:74px}}
 .slot.on{{background:var(--accent);border-color:var(--accent);color:var(--surface)}}
 .sb{{font-size:9.5px;opacity:.75}} .sq{{font-size:13.5px;font-weight:600}}
 .sx{{font-size:9px;color:var(--mark);background:var(--mark-soft);border-radius:3px;padding:0 4px}}
 .slot.on .sx{{color:var(--surface);background:rgba(255,255,255,.22)}}

 .etage{{border:1px solid var(--rule);border-radius:11px;background:var(--surface);
  padding:11px 13px;margin-bottom:8px}}
 .etage.dort{{opacity:.45}}
 .eh{{display:flex;gap:8px;align-items:baseline;flex-wrap:wrap;margin-bottom:8px}}
 .num{{font-family:'IBM Plex Mono',monospace;font-size:10.5px;color:var(--accent-ink);
  background:var(--accent-soft);border-radius:4px;padding:1px 7px}}
 .eh b{{font-size:13.5px}} .eh .note{{font-size:11.5px;color:var(--ink-faint);margin-left:auto}}
 .opts{{display:flex;gap:6px;flex-wrap:wrap}}
 .opt{{position:relative;overflow:hidden;min-width:70px;text-align:left}}
 .opt .lab{{position:relative;z-index:2;font-weight:600}}
 .opt .pc{{position:relative;z-index:2;font-size:10px;opacity:.8;margin-left:6px}}
 .opt .fill{{position:absolute;left:0;top:0;bottom:0;background:var(--accent-soft);z-index:1}}
 .opt.on{{background:var(--accent);border-color:var(--accent);color:var(--surface)}}
 .opt.on .fill{{background:rgba(255,255,255,.22)}}
 .opt.top{{border-color:var(--mark)}}
 .opt.faible{{color:var(--ink-faint)}}

 .res{{position:sticky;bottom:0;z-index:20;background:var(--surface);border:1px solid var(--rule-strong);
  border-radius:11px;padding:11px 14px;margin-top:14px;display:flex;gap:12px;align-items:center;flex-wrap:wrap}}
 .res .sym{{font-family:'IBM Plex Mono',monospace;font-size:22px;font-weight:600;color:var(--accent-ink)}}
 .res .av{{font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--ink-faint)}}
 .res .sp{{flex:1;min-width:4px}}
 footer{{margin-top:26px;padding-top:12px;border-top:1px solid var(--rule);font-size:11.5px;
  color:var(--ink-faint);font-family:'IBM Plex Mono',monospace;line-height:1.8}}
</style>

<span class="eyebrow">{e(D['titre'] or D['key'])} &middot; {e(D['tonalite'] or '')} &middot; démo</span>
<h1>Le compas en cascade</h1>
<p class="sub">La base, puis la 7<sup>e</sup>, la 9<sup>e</sup>, la 11<sup>e</sup>, la 13<sup>e</sup> —
un étage par tête du modèle, avec sa vraie distribution.</p>

<div class="card">
  <p><b>Pourquoi cette forme.</b> Le modèle est déjà fait comme ça&nbsp;: six têtes
  entraînées séparément — triade, basse, 7<sup>e</sup>, 9<sup>e</sup>, 11<sup>e</sup>,
  13<sup>e</sup>. La cascade ne range pas les boutons autrement, elle lit la structure
  du modèle.</p>
  <p><b>Ce que ça débloque.</b> Dans le décodage, les extensions ne gagnent jamais&nbsp;:
  la 9<sup>e</sup> trois fois sur 1255 accords, la 11<sup>e</sup> et la 13<sup>e</sup> zéro
  — une extension doit battre « aucune » dans le Viterbi, et « aucune » domine.
  Elles sont pourtant là&nbsp;: <b>{forts} accords de ce morceau</b> portent une extension
  au-dessus de 25&nbsp;%, jusqu'à 54&nbsp;% sur un <code>G7sus4</code>, ce qui en fait un
  <code>G9sus</code>. En cascade, l'extension n'a plus à gagner&nbsp;: elle est proposée.</p>
</div>

<h2>Choisis un accord</h2>
<p class="sub" style="margin-bottom:6px">Le badge vert marque ceux où une 9<sup>e</sup> est forte.</p>
<div class="slots" id="slots">{lignes}</div>

<div id="cascade"></div>

<div class="res">
  <span class="av" id="av"></span>
  <span class="sym" id="sym">—</span>
  <span class="sp"></span>
  <button id="play" type="button">&#9658; écouter</button>
  <button id="reset" type="button">repartir du modèle</button>
</div>

<footer>
  têtes : <code>harmonia.musx.frame_posteriors</code> — triade 73 · basse 13 · 7<sup>e</sup> 4 ·
  9<sup>e</sup> 4 · 11<sup>e</sup> 3 · 13<sup>e</sup> 3<br>
  seuil « suggéré » : {SUGGERE:.3f}, repris de <code>span_rescore.SUG_FLOOR</code> —
  aucun seuil nouveau &middot; données : <code>scratchpad/cascade_data.py</code><br>
  le rendu de l'app connaît 29 queues (<code>ui/kit.js</code>) dont <code>^9</code>,
  <code>-9</code>, <code>13</code>, <code>-^7</code>, <code>7b9</code>, <code>7#11</code>,
  <code>69</code> ; celles que la cascade peut produire sans glyphe encore
  (<code>11</code>, <code>^11</code>, <code>^13</code>, <code>-11</code>,
  <code>-13</code>, <code>9sus4</code>, <code>add9</code>) s'affichent en texte brut —
  lisibles, mais à typographier si on garde la cascade
</footer>

<script>
const D = {json.dumps(D, ensure_ascii=False, separators=(',', ':'))};
const SUGGERE = {SUGGERE};
const N = D.noms;
const TYPE_TAIL = ["", "-", "sus4", "sus2", "o", "+"];
let sel = 0, ch = null;

// ── le symbole : c'est le degré le PLUS HAUT retenu qui nomme l'accord, les
// autres étant implicites — la convention des lead sheets. Les altérations,
// elles, se disent entre parenthèses parce qu'elles ne sont pas implicites.
function symbole(c){{
  const base = TYPE_TAIL[c.type] || "";
  const sev = c.sev, n9 = c.neuf, n11 = c.onze, n13 = c.treize;
  const alt = [];
  if (n9 === 2) alt.push("#9"); if (n9 === 3) alt.push("b9");
  if (n11 === 2) alt.push("#11");
  if (n13 === 2) alt.push("b13");
  // le plus haut degré NATUREL retenu
  let haut = 0;
  if (sev) haut = 7;
  if (n9 === 1) haut = 9;
  if (n11 === 1) haut = 11;
  if (n13 === 1) haut = 13;
  let q;
  if (!sev){{
    // sans septième, une extension seule se dit « add »
    q = base + (haut ? "add" + haut : "");
  }} else if (c.type === 4 && sev === 3) {{
    q = "o7";                                  // dim + bb7
  }} else if (c.type === 4 && sev === 2) {{
    q = "h7";                                  // dim + b7 = demi-diminué
  }} else if (sev === 1) {{
    q = (c.type === 1 ? "-^" : "^") + (haut > 7 ? haut : 7);   // maj7 et au-delà
  }} else {{
    // b7 : dominante, ou mineure septième
    q = (c.type === 1 ? "-" : "") + (haut > 7 ? haut : 7);
    if (c.type === 2) q = (haut > 7 ? haut : 7) + "sus4";
    if (c.type === 3) q = (haut > 7 ? haut : 7) + "sus2";
  }}
  // SANS PARENTHÈSES : le rendu de l'app connaît `7b9`, `7#9`, `7#11` et leur
  // donne un vrai glyphe (`ui/kit.js`, tables TOK/CTOK, 29 queues). Écrire
  // `7(b9)` produirait du texte brut là où il existe une typographie.
  if (alt.length) q += alt.join("");
  return N.notes[c.root] + q;
}}

function argmax(v, from){{ let b = from || 0; for (let i = from || 0; i < v.length; i++) if (v[i] > v[b]) b = i; return b; }}

function depuisModele(a){{
  const b = a.base[0];
  return {{root: b.root, type: b.type, sev: argmax(a.sev), neuf: argmax(a.neuf),
           onze: argmax(a.onze), treize: argmax(a.treize)}};
}}

function etage(num, titre, cle, options, valeurs, actif){{
  const d = document.createElement('div');
  d.className = 'etage' + (actif ? '' : ' dort');
  const top = argmax(valeurs);
  const suggere = valeurs.slice(1).some(v => v >= SUGGERE);
  d.innerHTML = `<div class="eh"><span class="num">${{num}}</span><b>${{titre}}</b>` +
    `<span class="note">${{suggere ? "le modèle en suggère une" : "rien de suggéré ici"}}</span></div>`;
  const row = document.createElement('div'); row.className = 'opts';
  options.forEach((lab, i) => {{
    const b = document.createElement('button');
    const v = valeurs[i] || 0;
    b.className = 'opt' + (ch[cle] === i ? ' on' : '') + (i === top ? ' top' : '')
                + (i && v < SUGGERE ? ' faible' : '');
    b.type = 'button';
    b.innerHTML = `<span class="fill" style="width:${{Math.round(v * 100)}}%"></span>` +
      `<span class="lab">${{lab}}</span><span class="pc">${{(v * 100).toFixed(0)}} %</span>`;
    b.addEventListener('click', () => {{
      ch[cle] = i;
      // choisir « aucune » à un étage efface les étages du dessus : un accord
      // ne porte pas une 13e sans 9e retenue quand on vient de tout couper.
      if (cle === 'sev' && i === 0) {{ ch.neuf = 0; ch.onze = 0; ch.treize = 0; }}
      if (cle === 'neuf' && i === 0) {{ ch.onze = 0; ch.treize = 0; }}
      if (cle === 'onze' && i === 0) {{ ch.treize = 0; }}
      rendre();
    }});
    row.appendChild(b);
  }});
  d.appendChild(row);
  return d;
}}

function rendre(){{
  const a = D.accords[sel];
  const host = document.getElementById('cascade');
  host.innerHTML = '';
  // ÉTAGE 1 : la base, les huit meilleures (racine + type) de la tête triade
  const d1 = document.createElement('div'); d1.className = 'etage';
  d1.innerHTML = '<div class="eh"><span class="num">1</span><b>la base</b>' +
    '<span class="note">racine et type, tête triade</span></div>';
  const r1 = document.createElement('div'); r1.className = 'opts';
  a.base.forEach((b, k) => {{
    const btn = document.createElement('button');
    const on = ch.root === b.root && ch.type === b.type;
    btn.className = 'opt' + (on ? ' on' : '') + (k === 0 ? ' top' : '');
    btn.type = 'button';
    btn.innerHTML = `<span class="fill" style="width:${{Math.round(b.c * 100)}}%"></span>` +
      `<span class="lab">${{N.notes[b.root]}}${{TYPE_TAIL[b.type] || ''}}</span>` +
      `<span class="pc">${{(b.c * 100).toFixed(0)}} %</span>`;
    btn.addEventListener('click', () => {{ ch.root = b.root; ch.type = b.type; rendre(); }});
    r1.appendChild(btn);
  }});
  d1.appendChild(r1); host.appendChild(d1);

  // ÉTAGES 2 à 5 : chacun s'ouvre quand le précédent est choisi (la cascade
  // que Louis décrit) ; un étage endormi reste lisible mais grisé.
  host.appendChild(etage(2, "la 7<sup>e</sup>", 'sev', N.sev, a.sev, true));
  host.appendChild(etage(3, "la 9<sup>e</sup>", 'neuf', N.neuf, a.neuf, ch.sev !== 0));
  host.appendChild(etage(4, "la 11<sup>e</sup>", 'onze', N.onze, a.onze, ch.neuf !== 0));
  host.appendChild(etage(5, "la 13<sup>e</sup>", 'treize', N.treize, a.treize, ch.onze !== 0));

  document.getElementById('sym').textContent = symbole(ch);
  document.getElementById('av').textContent = 'le chart écrit ' + a.ecrit + ' \\u2192';
  document.querySelectorAll('.slot').forEach((s, i) => s.classList.toggle('on', i === sel));
}}

document.querySelectorAll('.slot').forEach(s => s.addEventListener('click', () => {{
  sel = parseInt(s.dataset.i, 10);
  ch = depuisModele(D.accords[sel]);
  rendre();
  s.scrollIntoView({{inline: 'center', block: 'nearest', behavior: 'smooth'}});
}}));
document.getElementById('reset').addEventListener('click', () => {{
  ch = depuisModele(D.accords[sel]); rendre();
}});

// audio : fetch + blob, arrêt sur `timeupdate` (les deux pièges iOS du projet)
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
  stopAt = a.t1 + 0.2;
  audio.currentTime = Math.max(0, a.t0);
  audio.play().catch(() => {{}});
}});

ch = depuisModele(D.accords[0]);
rendre();
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
