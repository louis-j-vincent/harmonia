#!/usr/bin/env python3
"""scripts/demo_basse.py — accords contre basse, distances à l'écran.

Louis, 2026-08-16 : « sur let it be on chope mal les différences harmoniques,
il y a un A et un B et on n'a que le A. Je me demande si utiliser la matrice
SSM de la basse ne pourrait pas aider » — puis « montre-moi les distances avec
la basse et avec l'harmonique, et laisse-MOI juger ».

Cette page ne décide rien et ne moyenne rien. Pour chaque morceau elle met
côte à côte, sur SES sections à lui :

  * la matrice de ressemblance, une par substrat, avec ses frontières dessus ;
  * la distance entre chaque paire de ses sections, terme par terme, sous
    chaque substrat — donc « A↔A vaut ça, A↔B vaut ça » et non une moyenne ;
  * mesure par mesure, l'accord entendu et la basse entendue, pour qu'on voie
    d'où vient le chiffre.

    .venv/bin/python scripts/demo_basse.py
"""
from __future__ import annotations

import base64
import html
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harmonia_min import retour as R          # noqa: E402
from demo_retour import (MORCEAUX, accords_par_mesure,           # noqa: E402
                         annotation, couleurs_louis, e)

CHARTS = REPO / "harmonia_min" / "state" / "charts"
SORTIE = REPO / "docs" / "plots" / "retour_basse_vs_accords.html"

SUBSTRATS = ("accords", "basse", "accords*basse")

#: La rampe de `ssm_page.CMAP`, pour que les matrices du projet se lisent avec
#: le même œil.
CMAP = ["#fbf7ec", "#cfe0ea", "#84b3cf", "#3d7fa6", "#1b4a6b", "#0d2437"]
NOTES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]

#: Sur combien de mesures on compare deux sections. 4 = la carrure ; c'est
#: exactement ce que `_compare` fait quand l'algorithme cherche une occurrence.
FENETRE = 4


def basse_notes(chart: dict, audio_dir=None) -> list[str]:
    """La note de basse dominante de chaque mesure, telle que musx l'entend."""
    from harmonia_min import musx as _musx
    grid = chart["barGrid"]
    stem = Path(chart["audio_url"]).stem
    bass = _musx.frame_posteriors(Path(audio_dir or "docs/audio") /
                                  f"{stem}.m4a")[1]
    out = []
    for b in range(len(grid) - 1):
        a = int(grid[b] / _musx.FRAME_DT)
        z = max(a + 1, int(grid[b + 1] / _musx.FRAME_DT))
        seg = bass[a:min(z, len(bass))]
        if not len(seg):
            out.append("·")
            continue
        k = int(seg.mean(0).argmax())
        out.append("N" if k == 0 else NOTES[(k - 1) % 12])
    return out


def matrice_png(S: np.ndarray) -> str:
    """La matrice en base64, un octet par case — même transport que ssm_page."""
    n = len(S)
    hors = S[~np.eye(n, dtype=bool)]
    lo, hi = ((float(np.quantile(hors, 0.05)), float(np.quantile(hors, 0.99)))
              if hors.size else (0.0, 1.0))
    if hi - lo < 1e-6:
        lo, hi = float(S.min()), max(float(S.max()), float(S.min()) + 1e-6)
    q = np.clip((S - lo) / (hi - lo), 0.0, 1.0)
    return base64.b64encode((q * 255.0 + 0.5).astype(np.uint8).tobytes()).decode()


def paires(S: np.ndarray, secs: list[dict], n: int) -> list[dict]:
    """La distance entre chaque paire de sections de Louis, terme par terme.

    Chaque section est représentée par ses `FENETRE` premières mesures — le mot
    que l'algorithme comparerait. Rien n'est moyenné entre paires.
    """
    out = []
    for i, a in enumerate(secs):
        for b in secs[i + 1:]:
            if a["b0"] + FENETRE > n or b["b0"] + FENETRE > n:
                continue
            sims = [float(S[a["b0"] + t, b["b0"] + t]) for t in range(FENETRE)]
            out.append({"a": a, "b": b, "sims": sims,
                        "moyenne": float(np.mean(sims)),
                        "meme": a["label"] == b["label"]})
    return out


def bloc(fichier: str, titre: str) -> str:
    chart = json.loads((CHARTS / f"{fichier}.json").read_text(encoding="utf-8"))
    grid = chart["barGrid"]
    n = len(grid) - 1
    stem = Path(chart["audio_url"]).stem

    S = {}
    for sub in SUBSTRATS:
        M = R.ssm_mesures(chart, substrat=sub)
        if M is None:
            return f'<section><h2>{e(titre)}</h2><p>pas de matrice.</p></section>'
        S[sub] = M

    louis = annotation(stem, n)
    coul = couleurs_louis(louis)
    chords = accords_par_mesure(chart)
    basses = basse_notes(chart)

    # Les sections de Louis, dans l'ordre, dédupliquées sur leur mesure de début.
    vus, secs = set(), []
    for b in range(n):
        if louis[b] and (b == 0 or louis[b - 1] != louis[b]):
            if b not in vus:
                vus.add(b)
                secs.append({"label": louis[b], "b0": b})

    # ── les matrices ────────────────────────────────────────────────────────
    front = [{"m": s["b0"], "l": s["label"]} for s in secs]
    mats = "".join(
        f'<figure><figcaption>{e(sub)}</figcaption>'
        f'<canvas data-m="{matrice_png(S[sub])}" data-n="{n}" '
        f'data-front=\'{json.dumps(front)}\'></canvas></figure>'
        for sub in SUBSTRATS)

    # ── les distances, paire par paire ──────────────────────────────────────
    P = {sub: paires(S[sub], secs, n) for sub in SUBSTRATS}
    lignes = []
    ordre = sorted(range(len(P["accords"])),
                   key=lambda i: (not P["accords"][i]["meme"],
                                  P["accords"][i]["a"]["b0"]))
    for i in ordre:
        p = P["accords"][i]
        a, b = p["a"], p["b"]
        cells = "".join(
            f'<td class="v"><b>{P[sub][i]["moyenne"]:.3f}</b>'
            f'<span>{" ".join(f"{x:.2f}" for x in P[sub][i]["sims"])}</span></td>'
            for sub in SUBSTRATS)
        lignes.append(
            f'<tr class="{"meme" if p["meme"] else "diff"}">'
            f'<td class="pa"><b class="lab" style="--c:{coul.get(a["label"],"#ccc")}">'
            f'{e(a["label"])}</b> <i>m.{a["b0"]}</i> ↔ '
            f'<b class="lab" style="--c:{coul.get(b["label"],"#ccc")}">'
            f'{e(b["label"])}</b> <i>m.{b["b0"]}</i></td>'
            f'<td class="q">{"même" if p["meme"] else "différentes"}</td>'
            f'{cells}</tr>')

    entetes = "".join(f'<th>{e(s)}</th>' for s in SUBSTRATS)

    # ── mesure par mesure ───────────────────────────────────────────────────
    cases = "".join(
        f'<b class="case" style="--c:{coul.get(louis[b], "#d8d2c6")}" '
        f'data-t="{grid[b]:.3f}"><u>{e(louis[b] or "·")}</u>'
        f'<i>{e(chords[b])}</i><em>{e(basses[b])}</em><s>{b}</s></b>'
        for b in range(n))

    return f"""<section data-audio="/audio/{e(stem)}.m4a">
<h2>{e(titre)}</h2>
<audio controls preload="none"></audio>
<div class="mats">{mats}</div>
<p class="leg">Les traits rouges sont TES frontières. Plus c'est foncé, plus ça
se ressemble.</p>
<div class="tbl"><table class="paires">
<tr><th>deux sections à toi</th><th></th>{entetes}</tr>
{"".join(lignes)}
</table></div>
<p class="leg">Gros chiffre = la ressemblance des {FENETRE} premières mesures de
l'une contre l'autre. Petits chiffres = mesure par mesure, sans rien moyenner.</p>
<details><summary>mesure par mesure : ta section, l'accord entendu, la basse
entendue</summary><div class="cases">{cases}</div></details>
</section>"""


CSS = """
:root{--fg:#1b1b19;--fond:#fbf7ec;--trait:#ddd6c7;--doux:#6b6558}
*{box-sizing:border-box}
body{margin:0 auto;padding:24px 18px 80px;background:var(--fond);color:var(--fg);
 font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif;
 max-width:1180px}
h1{font-size:24px;margin:0 0 6px}
h2{font-size:20px;margin:0 0 10px}
section{margin:0 0 46px;padding:0 0 10px;border-bottom:2px solid var(--trait)}
.chapo{color:var(--doux);max-width:78ch;margin:0 0 26px}
audio{width:100%;max-width:420px;height:32px;margin:0 0 14px;display:block}
.mats{display:flex;gap:16px;flex-wrap:wrap}
figure{margin:0;flex:1 1 240px;min-width:0}
figcaption{font-size:11px;color:var(--doux);text-transform:uppercase;
 letter-spacing:.04em;font-weight:600;margin:0 0 4px}
canvas{width:100%;aspect-ratio:1;display:block;border:1px solid var(--trait);
 border-radius:4px;background:#fbf7ec}
.leg{font-size:12px;color:var(--doux);margin:8px 0 14px}
.tbl{overflow-x:auto}
table.paires{border-collapse:collapse;font-size:13px;width:100%;min-width:560px}
table.paires th{text-align:left;font-size:11px;color:var(--doux);font-weight:600;
 padding:0 10px 6px 0;text-transform:uppercase;letter-spacing:.03em}
table.paires td{padding:5px 10px 5px 0;border-top:1px solid #efe9db;
 vertical-align:top}
tr.meme td{background:#eef4ea}
tr.diff td{background:#fbf1e8}
td.pa i{font-style:normal;color:var(--doux);font-size:11px}
td.q{font-size:11.5px;color:var(--doux);white-space:nowrap}
td.v b{font-size:14px;font-variant-numeric:tabular-nums}
td.v span{display:block;font-size:10px;color:var(--doux);
 font-variant-numeric:tabular-nums}
.lab{--c:#ccc;display:inline-block;min-width:18px;text-align:center;padding:0 5px;
 border-radius:4px;color:#fff;background:var(--c);font-size:12px}
details{margin:6px 0 0}
summary{font-size:12.5px;color:var(--doux);cursor:pointer}
.cases{display:flex;flex-wrap:wrap;gap:2px;margin:10px 0 0}
.case{--c:#d8d2c6;width:44px;padding:3px 2px;border-radius:4px;cursor:pointer;
 text-align:center;line-height:1.25;display:flex;flex-direction:column;
 background:color-mix(in srgb,var(--c) 20%,white);
 border:1px solid color-mix(in srgb,var(--c) 50%,white)}
.case u{text-decoration:none;font-size:11px;font-weight:700}
.case i{font-style:normal;font-size:10px;color:var(--doux)}
.case em{font-style:normal;font-size:11px;font-weight:700;color:#8a5a2b}
.case s{text-decoration:none;font-size:9px;color:#a49c8c}
"""

JS = """
var CMAP = __CMAP__;
document.querySelectorAll("canvas[data-m]").forEach(function(cv){
  var bin = atob(cv.dataset.m), n = +cv.dataset.n, front = JSON.parse(cv.dataset.front);
  var DPR = window.devicePixelRatio || 1, cote = cv.clientWidth || 260;
  cv.width = cv.height = Math.round(cote * DPR);
  var ctx = cv.getContext("2d"), u = cv.width / n;
  for (var i = 0; i < n; i++) for (var j = 0; j < n; j++){
    var v = bin.charCodeAt(i * n + j) / 255;
    ctx.fillStyle = CMAP[Math.min(CMAP.length - 1, Math.floor(v * CMAP.length))];
    ctx.fillRect(j * u, i * u, Math.ceil(u), Math.ceil(u));
  }
  ctx.strokeStyle = "#b4472c"; ctx.lineWidth = Math.max(1, DPR);
  front.forEach(function(f){
    var p = Math.round(f.m * u) + 0.5;
    ctx.beginPath(); ctx.moveTo(p, 0); ctx.lineTo(p, cv.height);
    ctx.moveTo(0, p); ctx.lineTo(cv.width, p); ctx.stroke();
  });
});
/* Clic sur une mesure = tête de lecture. Le disque passe par un blob : dans
   Safari iOS le lecteur ne bufferise jamais un fichier servi en direct. Aucune
   boucle de rendu — un requestAnimationFrame empêcherait le moteur audio de
   WebKit de démarrer. */
document.querySelectorAll("section[data-audio]").forEach(function(sec){
  var el = sec.querySelector("audio"), url = sec.dataset.audio;
  el.src = url;
  if (window.fetch) fetch(url).then(function(r){ return r.ok ? r.blob() : null; })
    .then(function(b){ if (!b) return;
      var t = el.currentTime, j = !el.paused;
      el.src = URL.createObjectURL(b); el.currentTime = t; if (j) el.play();
    }).catch(function(){});
  sec.addEventListener("click", function(ev){
    var c = ev.target.closest("[data-t]"); if (!c) return;
    try { el.currentTime = parseFloat(c.dataset.t); el.play(); } catch(e){}
  });
});
"""


def main() -> None:
    blocs = [bloc(f, t) for f, t in MORCEAUX]
    page = f"""<!doctype html><html lang="fr"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Accords contre basse — les distances</title>
<style>{CSS}</style>
<h1>Accords contre basse — les distances, sur tes sections</h1>
<p class="chapo">Pour chaque morceau : la matrice de ressemblance sous trois
substrats, tes frontières dessinées dessus, puis la distance entre chaque paire
de TES sections sous chacun des trois. Rien n'est moyenné entre morceaux ni
entre paires — chaque ligne est une comparaison que tu peux écouter en cliquant
une mesure. Les lignes vertes sont deux passages de la <b>même</b> section, les
oranges deux sections <b>différentes</b>. Un bon substrat, c'est celui qui garde
le vert haut et fait descendre l'orange.</p>
{"".join(blocs)}
<script>{JS.replace("__CMAP__", json.dumps(CMAP))}</script>
</html>"""
    SORTIE.write_text(page, encoding="utf-8")
    print(f"écrit {SORTIE} ({len(page) / 1024:.0f} ko)")


if __name__ == "__main__":
    main()
