"""SSM viewer: chord-tone self-similarity matrix + aligned playhead + sections.

One page per song. The matrix is rendered at exactly n_bars x n_bars pixels
(no axes, no margins) so bar b maps to x = (b + 0.5) / n_bars of the element
width — the playhead, the section strip and the matrix therefore share one
time axis by construction, not by eyeballing.
"""
import base64
import io
import json
import sys

import numpy as np

sys.path.insert(0, "/Users/vincente/Documents/Projets Perso/Code/harmonia")
from harmonia_min import musx as _musx
from harmonia_min.harmonic_sections import ssm

REPO = "/Users/vincente/Documents/Projets Perso/Code/harmonia"
CHARTS = f"{REPO}/harmonia_min/state/charts"
AUDIO = f"{REPO}/docs/audio"

SONGS = [
    ("min_aretha_franklin_chain_of_fools_official_lyric_video",
     "aretha_franklin_chain_of_fools_official_lyric_video.m4a"),
    ("min_ben_e_king_stand_by_me_audio", "ben_e_king_stand_by_me_audio.m4a"),
    ("min_maroon_5_this_love", "maroon_5_this_love.m4a"),
]

# 5 section colours, colourblind-safe on both surfaces (validated palette)
COLS = {"intro": "#8a8f98", "A": "#3b7dd8", "B": "#c8722a",
        "C": "#4a9d6f", "D": "#9b5fc0"}


def matrix_png(S):
    """n x n pixels, blue->white ramp, no axes. Returned as a data: URI."""
    from PIL import Image
    x = np.clip((S - np.quantile(S, .05)) /
                max(np.quantile(S, .98) - np.quantile(S, .05), 1e-6), 0, 1)
    # single-hue sequential ramp (light = dissimilar, dark = similar)
    r = (255 - x * (255 - 0x1b)).astype(np.uint8)
    g = (255 - x * (255 - 0x3a)).astype(np.uint8)
    b = (255 - x * (255 - 0x8c)).astype(np.uint8)
    img = Image.fromarray(np.dstack([r, g, b]))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


parts = []
for key, audio in SONGS:
    d = json.load(open(f"{CHARTS}/{key}.json"))
    grid = d["barGrid"]
    n = len(grid) - 1
    probs = _musx.frame_posteriors(f"{AUDIO}/{audio}")
    S = ssm(probs[0], grid)
    png = matrix_png(S)

    # musx's own certainty, bar by bar: the mean height of the winning
    # posterior over the bar's frames. Flat posterior = unsure = blurry SSM.
    FD = _musx.FRAME_DT
    conf = []
    for b in range(n):
        a, z = int(grid[b] / FD), int(grid[b + 1] / FD)
        seg = probs[0][a:min(z, len(probs[0]))]
        conf.append(float(seg.max(1).mean()) if len(seg) else 0.0)
    bar_s = np.diff(np.array(grid))
    med = float(np.median(bar_s))
    conf_pts = " ".join(f"{i + .5:.2f},{1 - c:.4f}" for i, c in enumerate(conf))
    # bar length vs the median bar — the "is the grid drifting?" curve
    grid_pts = " ".join(
        f"{i + .5:.2f},{0.5 - max(-0.5, min(0.5, (x - med) / med)):.4f}"
        for i, x in enumerate(bar_s))

    occ = sorted((b0, s["label"], b1)
                 for s in d["sections"] for b0, b1 in s["barRanges"])
    blocks = "".join(
        f'<div class="blk" style="left:{100*b0/n:.4f}%;'
        f'width:{100*(b1-b0+1)/n:.4f}%;background:{COLS.get(lab, "#888")}">'
        f'<span>{lab}<sub>{b1-b0+1}</sub></span></div>'
        for b0, lab, b1 in occ)
    lengths = {}
    for b0, lab, b1 in occ:
        lengths.setdefault(lab, []).append(b1 - b0 + 1)
    summary = " · ".join(
        f"<b>{lab}</b> ×{len(v)} : {'toutes ' + str(v[0]) + ' mes.' if len(set(v)) == 1 else 'longueurs ' + '/'.join(map(str, v))}"
        for lab, v in lengths.items())

    parts.append(f"""
<section data-grid='{json.dumps(grid)}' data-n="{n}">
<h2>{d.get('title', key)}</h2>
<div class="wrap">
  <img class="ssm" src="{png}" alt="matrice de similarité">
  <div class="ph vert"></div>
  <div class="ph horz"></div>
</div>
<div class="strip">{blocks}<div class="ph vert strip-ph"></div></div>
<div class="curvelab">certitude de musx, mesure par mesure — haut = sûr</div>
<div class="curve">
  <svg viewBox="0 0 {n} 1" preserveAspectRatio="none">
    <polyline points="{conf_pts}" fill="none" stroke="#3b7dd8" stroke-width="0.012"
      vector-effect="non-scaling-stroke" style="stroke-width:2px"/>
  </svg>
  <div class="ph vert strip-ph"></div>
</div>
<div class="curvelab">durée des mesures vs la mesure médiane — plat = tempo stable</div>
<div class="curve short">
  <svg viewBox="0 0 {n} 1" preserveAspectRatio="none">
    <line x1="0" y1="0.5" x2="{n}" y2="0.5" stroke="#8a8f98" stroke-width="1"
      vector-effect="non-scaling-stroke" style="stroke-width:1px" stroke-dasharray="4 4"/>
    <polyline points="{grid_pts}" fill="none" stroke="#c8722a"
      vector-effect="non-scaling-stroke" style="stroke-width:2px"/>
  </svg>
  <div class="ph vert strip-ph"></div>
</div>
<div class="sub">{summary}</div>
<audio controls preload="none" src="http://100.89.209.63:7772/audio/{audio}"></audio>
<a class="full" href="http://100.89.209.63:7772/?open={key}">ouvrir le chart dans l'app ↗</a>
</section>""")

html = f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Matrice de similarité, tête de lecture, sections</title>
<style>
:root {{ color-scheme: light dark; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; padding:14px 12px 40px; max-width:640px; margin-inline:auto;
  font-family:-apple-system,"Segoe UI",sans-serif; background:#1e1c1a; color:#ece8e2; }}
@media (prefers-color-scheme: light) {{ body {{ background:#f7f6f3; color:#1a1a1a; }} }}
h1 {{ font-size:1.24rem; margin:4px 0 8px; }}
h2 {{ font-size:1.05rem; margin:0 0 10px; }}
section {{ border-top:1px solid rgba(128,128,128,.3); padding-top:20px; margin-top:26px; }}
.intro {{ font-size:.87rem; line-height:1.55; }}
.sub {{ opacity:.75; font-size:.8rem; line-height:1.5; margin:8px 0 10px; }}
.wrap {{ position:relative; width:100%; aspect-ratio:1; border-radius:8px;
  overflow:hidden; border:1px solid rgba(128,128,128,.35); }}
.ssm {{ width:100%; height:100%; display:block; image-rendering:pixelated; }}
.ph {{ position:absolute; background:#d4453a; pointer-events:none; opacity:.9; }}
.vert {{ top:0; bottom:0; width:2px; left:-10px; }}
.horz {{ left:0; right:0; height:2px; top:-10px; }}
.strip {{ position:relative; height:30px; margin-top:4px; border-radius:5px;
  overflow:hidden; background:rgba(128,128,128,.15); }}
.blk {{ position:absolute; top:0; bottom:0; display:flex; align-items:center;
  justify-content:center; border-right:1px solid rgba(0,0,0,.35); }}
.blk span {{ font:700 11px -apple-system; color:#fff; text-shadow:0 1px 2px rgba(0,0,0,.5); }}
.blk sub {{ font-weight:400; opacity:.85; }}
.strip-ph {{ z-index:5; }}
.curvelab {{ font-size:.68rem; opacity:.55; margin:9px 0 2px; font-weight:600;
  letter-spacing:.03em; text-transform:uppercase; }}
.curve {{ position:relative; height:56px; border-radius:5px;
  background:rgba(128,128,128,.10); border:1px solid rgba(128,128,128,.25); }}
.curve.short {{ height:34px; }}
.curve svg {{ width:100%; height:100%; display:block; }}
audio {{ width:100%; margin-top:10px; }}
.full {{ display:inline-block; margin-top:8px; font-size:.85rem; font-weight:600; }}
.box {{ margin-top:14px; padding:11px 13px; border-radius:8px; font-size:.83rem;
  line-height:1.5; background:rgba(196,138,32,.12);
  border:1px solid rgba(196,138,32,.35); }}
</style></head><body>

<h1>Matrice de similarité, tête de lecture, sections</h1>
<div class="intro">
Chaque carré est la <b>matrice de similarité harmonique</b> du morceau : une
case (i, j) est sombre quand la mesure i sonne comme la mesure j. Les carrés
sombres hors diagonale sont les répétitions — c'est de là que sortent les
sections. Lance la lecture : le trait rouge suit la mesure en cours, sur la
matrice <i>et</i> sur la bande des sections, dessous. Les deux partagent le
même axe de temps par construction (la matrice fait exactement une mesure par
pixel).
</div>
<div class="box"><b>Tes deux hypothèses sur Stand By Me, tranchées.</b> Tu as vu
le damier plus flou en haut à gauche qu'en bas à droite. Les deux courbes
ci-dessous répondent : la <b>durée des mesures est plate</b> (120,0 BPM du
début à la fin, seules les 4 toutes premières mesures sont étirées par le
fondu) — donc pas de décalage sonore ; mais la <b>certitude de musx monte de
0,65 à 0,85</b> entre le premier quart et le reste. C'est bien un flou
d'accords : au début, seule la basse porte l'harmonie, musx hésite, ses
vecteurs sont plats, la matrice pâlit. Mesuré : similarité d'une mesure avec
sa jumelle 8 mesures plus loin = 0,904 sur le premier quart contre 0,994
ensuite, et cette similarité corrèle à 0,50 avec la certitude.</div>
<div class="box"><b>Chain Of Fools : ce n'est pas le tempo.</b> Sa grille est
la plus régulière des trois (116,5 BPM, écart-type 26 ms). C'est l'acoustique :
sa certitude médiane est de <b>0,57</b>, contre 0,84 pour Stand By Me. musx
doute sur tout le morceau — d'où un découpage qui part dans tous les sens
(A de 8, 2, 8, 2, 4, 6, 15 mesures) alors que la matrice montre un damier
parfaitement régulier.</div>
{''.join(parts)}

<script>
document.querySelectorAll('section').forEach(sec => {{
  const grid = JSON.parse(sec.dataset.grid), n = +sec.dataset.n;
  const audio = sec.querySelector('audio');
  const phs = sec.querySelectorAll('.vert'), horz = sec.querySelector('.horz');
  let raf = null;
  const draw = () => {{
    const t = audio.currentTime;
    let b = 0;
    while (b + 1 < grid.length && grid[b + 1] <= t) b++;
    const frac = (b + (grid[b+1] > grid[b] ? (t - grid[b]) / (grid[b+1] - grid[b]) : 0)) / n;
    const pct = Math.max(0, Math.min(1, frac)) * 100;
    phs.forEach(p => p.style.left = pct + '%');
    horz.style.top = pct + '%';
    raf = requestAnimationFrame(draw);
  }};
  audio.addEventListener('play', () => {{ if (!raf) draw(); }});
  audio.addEventListener('pause', () => {{ cancelAnimationFrame(raf); raf = null; }});
  audio.addEventListener('seeked', draw);
}});
</script>
</body></html>"""

out = f"{REPO}/docs/plots/formprior_ssm_sections.html"
open(out, "w").write(html)
print("wrote", out)
