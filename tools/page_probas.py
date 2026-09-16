"""La page des probabilités qui défilent : 16 mesures, les accords, le curseur.

Louis, 2026-09-16 : « cette vue en probas est vraiment bien, fais moi une page
html avec ca sur les 16 premieres barres, avec les accords marqués et le curseur
qui défile dessus pour voir ou on en est ».

Quatre bandes de quatre mesures — une bande tient dans un écran de téléphone —
partageant les mêmes quatorze lignes, donc comparables entre elles. Sur chaque
bande : ce que musx donne image par image (23 ms), les accords écrits posés à
leur place, et le curseur de lecture.

Deux choix de fond, pris pour que la carte ne mente pas :

  * rampe SÉQUENTIELLE à teinte unique, clair vers foncé. Une probabilité est
    une magnitude, pas une catégorie : un arc-en-ciel inventerait des frontières
    là où il n'y en a pas. Un gamma ouvre le bas de l'échelle, sans quoi tout ce
    qui est sous 20 % se confond avec le fond.
  * quatorze lignes au plus. Les 59 autres accords du vocabulaire pèsent 2,9 %
    en moyenne — ce chiffre est ÉCRIT sur la page plutôt que dessiné, parce
    qu'une ligne quasi vide lue quatorze fois fatigue l'œil pour rien.

    .venv/bin/python -m tools.page_probas
"""
from __future__ import annotations

import html
import json

from harmonia.settings import SETTINGS

DATA = SETTINGS.repo / "scratchpad" / "heatmap16.json"
OUT = SETTINGS.repo / "docs" / "plots" / "probas16.html"


def page(D: dict) -> str:
    e = html.escape
    bandes = "".join(
        f"""
<div class="bande" data-i="{i}">
  <div class="bl">mesures {b['bar0']}&ndash;{b['bar0'] + 3}
    <span class="tt">{b['t0']:.2f}&ndash;{b['t1']:.2f} s</span></div>
  <div class="cw">
    <canvas class="hm" data-i="{i}"></canvas>
    <div class="cur" data-i="{i}"></div>
  </div>
</div>""" for i, b in enumerate(D["bandes"]))

    return f"""<!doctype html>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Les Probabilités Qui Défilent</title>
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
 body{{background:var(--bg);color:var(--ink);margin:0 auto;max-width:960px;
  font-family:'Public Sans',-apple-system,sans-serif;padding:20px 14px 80px;font-size:14.5px;line-height:1.5}}
 h1{{font-family:'Fraunces',Georgia,serif;font-size:25px;font-weight:600;margin:0;text-wrap:balance}}
 .eyebrow{{font-family:'IBM Plex Mono',monospace;font-size:11px;letter-spacing:.09em;text-transform:uppercase;
  color:var(--accent-ink);background:var(--accent-soft);display:inline-block;padding:2px 7px;
  border-radius:3px;margin-bottom:8px}}
 .sub{{color:var(--ink-dim);font-size:13px;margin:5px 0 0}}
 code{{font-family:'IBM Plex Mono',monospace;font-size:12.5px;background:var(--surface-2);
  border:1px solid var(--rule);border-radius:4px;padding:1px 5px}}

 .tr{{position:sticky;top:0;z-index:30;background:var(--surface);border:1px solid var(--rule);
  border-radius:10px;padding:8px 12px;margin:14px 0 16px;display:flex;gap:11px;flex-wrap:wrap;
  align-items:center;font-family:'IBM Plex Mono',monospace;font-size:12px;color:var(--ink-dim)}}
 button{{font-family:'IBM Plex Mono',monospace;font-size:12px;border-radius:6px;padding:6px 11px;
  cursor:pointer;border:1px solid var(--rule);background:var(--surface-2);color:var(--ink-dim)}}
 button:hover{{border-color:var(--accent);color:var(--accent-ink);background:var(--accent-soft)}}
 button:focus-visible{{outline:2px solid var(--accent);outline-offset:2px}}
 #play{{min-width:92px;background:var(--accent);border-color:var(--accent);color:var(--surface)}}
 .opt{{display:inline-flex;align-items:center;gap:5px;cursor:pointer;white-space:nowrap}}
 .opt input{{accent-color:var(--accent);width:14px;height:14px}}
 #clock{{font-variant-numeric:tabular-nums;color:var(--ink)}}
 #read{{margin-left:auto;font-size:11.5px;color:var(--ink-faint)}}
 #read b{{color:var(--ink)}}

 .bande{{margin-bottom:16px}}
 .bl{{font-family:'IBM Plex Mono',monospace;font-size:10.5px;text-transform:uppercase;
  letter-spacing:.05em;color:var(--ink-faint);margin-bottom:5px}}
 .bl .tt{{margin-left:8px;text-transform:none;letter-spacing:0}}
 .cw{{position:relative;line-height:0}}
 canvas.hm{{display:block;width:100%;height:auto;border:1px solid var(--rule);border-radius:7px;
  cursor:crosshair}}
 .cur{{position:absolute;top:0;bottom:0;width:2px;background:var(--warn);display:none;
  pointer-events:none;z-index:5}}

 .card{{background:var(--surface);border:1px solid var(--rule);border-radius:11px;
  padding:13px 15px;margin:18px 0}}
 .card p{{margin:8px 0;font-size:13px;color:var(--ink-dim)}}
 .card b{{color:var(--ink)}}
 .ech{{display:flex;align-items:center;gap:8px;margin-top:10px;font-family:'IBM Plex Mono',monospace;
  font-size:11px;color:var(--ink-faint);flex-wrap:wrap}}
 .ramp{{height:11px;width:150px;border-radius:3px;border:1px solid var(--rule)}}
 footer{{margin-top:26px;padding-top:12px;border-top:1px solid var(--rule);font-size:11.5px;
  color:var(--ink-faint);font-family:'IBM Plex Mono',monospace;line-height:1.8}}
</style>

<span class="eyebrow">{e(D['titre'] or D['key'])} &middot; {e(D['tonalite'] or '')} &middot; 16 mesures</span>
<h1>Ce que le modèle entend</h1>
<p class="sub">{D['n_images']} images à 23&nbsp;ms, les {len(D['lignes'])} accords les plus présents.
Les {73 - len(D['lignes'])} autres pèsent {D['reste_moyen'] * 100:.1f}&nbsp;% en moyenne.
En pointillé rouge, l'accord qui a été écrit.</p>

<div class="tr">
  <button id="play" type="button">&#9658; jouer</button>
  <span id="clock">mes. — &middot; 0,0 s</span>
  <label class="opt"><input type="checkbox" id="slow"> ralenti 60&nbsp;%</label>
  <label class="opt"><input type="checkbox" id="foll" checked> suivre</label>
  <span id="read">survole la carte</span>
</div>

{bandes}

<div class="card">
  <p><b>Comment la lire.</b> Une colonne = 23&nbsp;ms. Une ligne = un accord du
  vocabulaire. Plus c'est foncé, plus le modèle y croit à cet instant. Les traits
  pleins sont les barres de mesure, les pointillés rouges les accords écrits.</p>
  <div class="ech"><span>0&nbsp;%</span><canvas id="ramp" class="ramp"></canvas><span>100&nbsp;%</span>
    <span style="margin-left:10px">échelle à teinte unique, non linéaire (le bas est ouvert)</span></div>
  <p style="margin-top:12px"><b>Ce qu'elle ne peut pas montrer.</b> Un
  <code>A♭^9</code> sans tierce est un accord de <code>E♭</code> posé sur un
  <code>A♭</code>&nbsp;: au-dessus, c'est la même chose. Aucune ligne de cette
  carte ne les sépare — c'est la basse qui le fait, un étage plus bas.</p>
</div>

<footer>
  postérieures : <code>harmonia.musx.frame_posteriors</code>, 5 folds ISMIR 2019 moyennés
  &middot; accords écrits : le chart en bibliothèque<br>
  données : <code>scratchpad/heatmap16_data.py</code> &middot; page : <code>tools/page_probas.py</code>
</footer>

<script>
const D = {json.dumps(D, ensure_ascii=False, separators=(',', ':'))};
const AUDIO = {json.dumps('../audio/' + D['key'] + '.m4a')};
const LAB_W = 52, TOP = 15, ROW = 15, BAS = 15;
const NL = D.lignes.length;
const HH = TOP + NL * ROW + BAS;

const sombre = () => {{
  const t = document.documentElement.getAttribute('data-theme');
  if (t === 'dark') return true;
  if (t === 'light') return false;
  return matchMedia('(prefers-color-scheme: dark)').matches;
}};
const CLAIR = [[251,247,238],[239,223,197],[222,146,81],[180,99,42],[90,51,21]];
const SOMBRE = [[27,24,18],[59,44,27],[180,99,42],[222,146,81],[243,210,172]];
function couleur(v, st){{
  const x = Math.max(0, Math.min(1, Math.pow(v, 0.55)));
  const n = st.length - 1, i = Math.min(n - 1, Math.floor(x * n));
  const f = x * n - i, a = st[i], b = st[i + 1];
  return `rgb(${{Math.round(a[0]+(b[0]-a[0])*f)}},${{Math.round(a[1]+(b[1]-a[1])*f)}},${{Math.round(a[2]+(b[2]-a[2])*f)}})`;
}}

const cvs = [...document.querySelectorAll('canvas.hm')];
const curs = [...document.querySelectorAll('.cur')];
let cellW = [];

function dessine(){{
  const dark = sombre(), st = dark ? SOMBRE : CLAIR;
  const encre = dark ? '#A89D89' : '#6F6555';
  const trait = dark ? '#4A4130' : '#C9BE9F';
  const rouge = dark ? '#E1837B' : '#B23A33';
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  cvs.forEach((cv, i) => {{
    const b = D.bandes[i];
    const W = cv.parentNode.clientWidth;
    cv.width = W * dpr; cv.height = HH * dpr;
    cv.style.height = HH + 'px';
    const g = cv.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    g.clearRect(0, 0, W, HH);
    const COLS = b.p[0].length;
    const cw = (W - LAB_W - 4) / COLS;
    cellW[i] = cw;
    g.font = '9.5px "IBM Plex Mono", monospace';
    g.textBaseline = 'middle';
    for (let r = 0; r < NL; r++){{
      const y = TOP + r * ROW, row = b.p[r];
      for (let c = 0; c < COLS; c++){{
        g.fillStyle = couleur(row[c], st);
        g.fillRect(LAB_W + c * cw, y, Math.ceil(cw) + 0.5, ROW - 1.5);
      }}
      g.fillStyle = encre; g.textAlign = 'right';
      g.fillText(D.lignes[r], LAB_W - 6, y + (ROW - 1.5) / 2);
    }}
    const x = t => LAB_W + (t - b.t0) / D.dt * cw;
    g.textAlign = 'center';
    b.barres.forEach((t, k) => {{
      g.strokeStyle = trait; g.lineWidth = k === 0 ? 2 : 1;
      g.beginPath(); g.moveTo(x(t), TOP - 2); g.lineTo(x(t), TOP + NL * ROW); g.stroke();
      if (k < 4){{ g.fillStyle = encre; g.fillText(b.bar0 + k, x(t) + 11, TOP - 7); }}
    }});
    g.textAlign = 'left';
    D.ecrits.forEach(c => {{
      if (c.t0 < b.t0 - 1e-6 || c.t0 >= b.t1) return;
      g.strokeStyle = rouge; g.lineWidth = 1.5; g.setLineDash([3, 3]);
      g.beginPath(); g.moveTo(x(c.t0), TOP); g.lineTo(x(c.t0), TOP + NL * ROW); g.stroke();
      g.setLineDash([]);
      g.fillStyle = rouge; g.fillText(c.lab, x(c.t0) + 3, TOP + NL * ROW + 8);
    }});
  }});
  const rp = document.getElementById('ramp');
  if (rp){{
    const w = rp.clientWidth || 150;
    rp.width = w; rp.height = 11;
    const g = rp.getContext('2d');
    for (let i = 0; i < w; i++){{ g.fillStyle = couleur(i / (w - 1), st); g.fillRect(i, 0, 1, 11); }}
  }}
}}

function bandeDe(t){{
  for (let i = 0; i < D.bandes.length; i++) if (t < D.bandes[i].t1) return i;
  return D.bandes.length - 1;
}}
function place(t){{ const b = D.bandes[bandeDe(t)]; return (t - b.t0) / D.dt; }}

// ── lecture ──
// fetch + blob (Safari ne bufferise pas les 206) et tick sur `timeupdate`,
// jamais une boucle rAF (elle empêche le moteur audio de démarrer sur iOS).
let audio = null, playing = false, curBande = -1;
const clock = document.getElementById('clock'), playBtn = document.getElementById('play');
fetch(AUDIO).then(r => r.blob()).then(bl => {{
  audio = new Audio(URL.createObjectURL(bl));
  audio.preservesPitch = true;
  audio.addEventListener('timeupdate', tick);
  audio.addEventListener('play', () => {{ playing = true; playBtn.innerHTML = '&#10073;&#10073; pause'; }});
  audio.addEventListener('pause', () => {{ playing = false; playBtn.innerHTML = '&#9658; jouer'; }});
  audio.addEventListener('ended', () => {{ playing = false; playBtn.innerHTML = '&#9658; jouer'; }});
}}).catch(() => {{ clock.textContent = 'audio indisponible'; }});

function tick(){{
  if (!audio) return;
  const t = audio.currentTime;
  const i = (t < D.t0 || t > D.t1) ? -1 : bandeDe(t);
  if (i !== curBande){{
    curs.forEach((c, k) => {{ c.style.transition = 'none'; c.style.display = k === i ? 'block' : 'none'; }});
    curBande = i;
    if (i >= 0 && document.getElementById('foll').checked)
      cvs[i].scrollIntoView({{block: 'nearest', behavior: 'smooth'}});
  }}
  if (i >= 0){{
    const b = D.bandes[i], cur = curs[i];
    const px = u => LAB_W + (u - b.t0) / D.dt * cellW[i];
    cur.style.transition = 'none'; cur.style.left = px(t) + 'px';
    void cur.offsetWidth;
    const ahead = t + 0.3;
    if (playing && ahead < b.t1){{
      cur.style.transition = 'left 300ms linear'; cur.style.left = px(ahead) + 'px';
    }}
  }}
  let bar = '—', beat = '';
  for (const b of D.bandes) for (let k = 0; k < 4; k++){{
    const a = b.barres[k], z = b.barres[k + 1];
    if (t >= a && t < z){{ bar = b.bar0 + k; beat = '.' + (Math.floor((t - a) / ((z - a) / 4)) + 1); }}
  }}
  clock.textContent = `mes. ${{bar}}${{beat}} · ${{t.toFixed(1).replace('.', ',')}} s`;
}}

playBtn.addEventListener('click', () => {{ if (audio) playing ? audio.pause() : audio.play().catch(()=>{{}}); }});
document.getElementById('slow').addEventListener('change', ev => {{
  if (audio) audio.playbackRate = ev.target.checked ? 0.6 : 1;
}});
document.addEventListener('keydown', ev => {{
  if (ev.code === 'Space' && ev.target === document.body){{ ev.preventDefault(); playBtn.click(); }}
}});

const read = document.getElementById('read');
cvs.forEach((cv, i) => {{
  const pos = ev => {{
    const r = cv.getBoundingClientRect();
    const px = (ev.touches ? ev.touches[0].clientX : ev.clientX) - r.left;
    const py = (ev.touches ? ev.touches[0].clientY : ev.clientY) - r.top;
    return {{c: Math.floor((px - LAB_W) / cellW[i]), r: Math.floor((py - TOP) / ROW),
             t: D.bandes[i].t0 + Math.max(0, (px - LAB_W) / cellW[i]) * D.dt}};
  }};
  cv.addEventListener('mousemove', ev => {{
    const {{c, r}} = pos(ev), b = D.bandes[i];
    if (c < 0 || c >= b.p[0].length || r < 0 || r >= NL){{ read.textContent = 'survole la carte'; return; }}
    read.innerHTML = `<b>${{D.lignes[r]}}</b> · ${{(b.p[r][c]*100).toFixed(1)}} %`;
  }});
  cv.addEventListener('mouseleave', () => read.textContent = 'survole la carte');
  cv.addEventListener('click', ev => {{
    const {{t}} = pos(ev);
    if (!audio) return;
    audio.currentTime = Math.max(0, t); tick();
    if (!playing) audio.play().catch(()=>{{}});
  }});
}});

dessine();
addEventListener('resize', () => {{ dessine(); tick(); }});
matchMedia('(prefers-color-scheme: dark)').addEventListener('change', dessine);
</script>
"""


def main() -> int:
    D = json.loads(DATA.read_text(encoding="utf-8"))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(page(D), encoding="utf-8")
    ko = OUT.stat().st_size // 1024
    print(f"→ {OUT}  ({len(D['bandes'])} bandes, {len(D['lignes'])} lignes, {ko} Ko)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
