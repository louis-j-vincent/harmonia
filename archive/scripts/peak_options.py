"""Les deux pistes pour récupérer les pics manquants, illustrées morceau par morceau.

    .venv/bin/python scripts/peak_options.py  ->  /plots/peak_options.html

Louis, 2026-08-12, après le diagnostic de Blue Lights (le pic de la mesure 49
existe, juste sous le seuil, et l'ajouter fait passer le morceau de 0,696 à
0,813) : « piste : un seuil de rétention par morceau, ou une passe qui rouvre les
suivants là où un bloc de 8 recouvre deux sections de 4 → fais-moi illustrations
pour les deux ».

LES DEUX PISTES, telles qu'implémentées :

  **A — seuil par morceau.** Au lieu d'un seuil global sur la proéminence
  (`peak_profile.KEEP_PROM = 0,25`), on garde autant de pics que le morceau a de
  tranches de huit mesures : une chanson pop change de section environ toutes les
  huit mesures, donc un morceau de 84 mesures a droit à ~10 frontières. C'est un
  seuil de DÉBIT, pas de niveau, donc il ne dépend plus de l'échelle du profil.

  **B — réouverture.** On lance la détection une première fois ; pour chaque
  section de huit mesures ou plus, on regarde s'il existe un « suivant » à sa
  moitié (±1 mesure). Si oui il est promu en pic dur et on relance. C'est
  exactement le cas Blue Lights : un bloc de 8 posé sur deux sections de 4.

CE QUE ÇA DONNE, et c'est un résultat négatif pour les deux (médiane du score
`section_metric` sur les douze morceaux annotés) :

    aujourd'hui            0,789
    A — seuil par morceau  0,615
    B — réouverture        0,768

Mais la médiane cache l'essentiel, et c'est pour ça que cette page existe :

  * **A** sauve Let It Be (0,716 → 0,821) et aide Blue Lights (0,696 → 0,739),
    et démolit tout le reste — Stand By Me 0,795 → 0,445, Every Breath 0,835 →
    0,527. Le débit « une section toutes les huit mesures » est faux dès qu'un
    morceau a de longues sections.
  * **B** répare exactement le cas pour lequel elle a été écrite (Blue Lights
    0,696 → 0,813) et casse les morceaux déjà justes : This Love 1,000 → 0,923,
    Bein' Green 0,993 → 0,845, Don't Know Why 0,974 → 0,854. Elle coupe des blocs
    de 8 qui sont de VRAIS blocs de 8.

UNE GARDE A ÉTÉ TESTÉE ET NE DISCRIMINE PAS : n'accepter la coupe que si les
deux moitiés du bloc ne se ressemblent pas (au-dessus du 3e quartile de la
matrice d'accords). Elle n'écarte que deux coupes sur les douze morceaux (Sunny,
et une de Chain of Fools) et laisse passer toutes celles qui cassent. Ce qui
sépare Blue Lights de This Love n'est donc pas la cohérence interne du bloc — ça
reste à trouver, et c'est la question que cette page pose.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
sys.path.insert(0, str(HERE / "scratchpad"))

from ssm_zoo import SONGS, AUDIO, GT_LINE, gt_sections, fig2b64   # noqa: E402
from peak_profile import fused_profile                            # noqa: E402
from hard_prior_sections import voice_with_hard, score, colourmap  # noqa: E402
from scipy.signal import find_peaks, peak_prominences             # noqa: E402

OUT = HERE / "docs" / "plots" / "peak_options.html"
INK = "#1c1c1c"
COL = {"actuel": "#0d2437", "A": "#8a2b2b", "B": "#1f8a5b"}
PER = 8          # A : une frontière par tranche de tant de mesures


def _bars(cuts, n, m):
    return [h for h in sorted({int(round(c * n / m)) for c in cuts}) if 0 < h < n]


def option_a(P, n, m, per=PER):
    x = np.nan_to_num(P)
    idx, _ = find_peaks(x, distance=8)
    if not len(idx):
        return []
    pr = peak_prominences(x, idx)[0]
    idx = idx[np.argsort(-pr)]
    return _bars(idx[:max(1, int(round(n / per)))], n, m)


def option_b(stem, kept, rest, n, m, extra):
    base = _bars(kept, n, m)
    secs = voice_with_hard(stem, base, n, extra, triad=extra["triad"])
    restb = _bars(rest, n, m)
    add = [r for s in secs if s["b1"] - s["b0"] + 1 >= 8
           for r in restb if abs(r - (s["b0"] + (s["b1"] - s["b0"] + 1) // 2)) <= 1]
    return sorted(set(base) | set(add)), sorted(set(add))


def song_html(stem: str, title: str) -> str:
    P, kept, rest, lines, n, extra = fused_profile(stem)
    m = len(P)
    tri = extra["triad"]
    gt = gt_sections(stem)
    gtb = [s["b0"] for s in gt["sections"][1:]] if gt else []

    cur = _bars(kept, n, m)
    A = option_a(P, n, m)
    B, addB = option_b(stem, kept, rest, n, m, extra)
    secs = {k: voice_with_hard(stem, hh, n, extra, triad=tri)
            for k, hh in (("actuel", cur), ("A", A), ("B", B))}
    sc = {k: (score(v, gt["sections"], n) if gt else float("nan"))
          for k, v in secs.items()}

    strips = [("aujourd'hui", secs["actuel"], cur, "actuel"),
              ("A — seuil par morceau", secs["A"], A, "A"),
              ("B — réouverture", secs["B"], B, "B")]
    if gt:
        strips.append(("toi", gt["sections"], [], None))

    fig, axs = plt.subplots(1 + len(strips), 1, figsize=(12.6, 1.05 * len(strips) + 1.5),
                            facecolor="#fffdf6",
                            gridspec_kw={"height_ratios": [2.1] + [1] * len(strips),
                                         "hspace": 0.12})
    x = (np.arange(m) + 0.5) / 2.0

    ax = axs[0]
    ax.fill_between(x, P, color="#cdd8df", lw=0)
    for b in gtb:
        ax.axvline(b, color=GT_LINE, lw=0.8, alpha=0.7)
    for c in _bars(rest, n, m):                     # les suivants, non retenus
        ax.plot([c], [1.08], marker="v", ms=7, markerfacecolor="none",
                markeredgecolor="#9aa3a9", markeredgewidth=1.1, clip_on=False)
    for c in cur:
        ax.plot([c], [1.08], marker="v", ms=9, color=COL["actuel"], clip_on=False)
    for c in [c for c in A if c not in cur]:
        ax.plot([c], [1.08], marker="v", ms=9, color=COL["A"], clip_on=False)
    for c in addB:
        ax.plot([c], [1.08], marker="v", ms=9, color=COL["B"], clip_on=False)
    ax.set_xlim(0, n); ax.set_ylim(0, 1.18)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylabel("profil", rotation=0, ha="right", va="center", fontsize=10)
    for s in ax.spines.values():
        s.set_color("#e0d7c2")

    cm = colourmap()
    for i, (ax, (lab, ss, hh, key)) in enumerate(zip(axs[1:], strips)):
        for s in ss:
            w = s["b1"] - s["b0"] + 1
            ax.add_patch(plt.Rectangle((s["b0"], 0.06), w, 0.88,
                                       facecolor=cm(s["label"]),
                                       edgecolor="#fffdf6", lw=1.2))
            if w >= max(2, n * 0.04):
                ax.text(s["b0"] + w / 2, 0.5, str(s["label"]), ha="center",
                        va="center", fontsize=8.5, color=INK)
        for h in hh:
            ax.axvline(h, color=COL[key], lw=1.5, alpha=0.9)
        for b in gtb:
            ax.axvline(b, color=GT_LINE, lw=0.7, alpha=0.6)
        ax.set_xlim(0, n); ax.set_ylim(0, 1)
        ax.set_yticks([])
        ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=10)
        for s in ax.spines.values():
            s.set_visible(False)
        if i == len(strips) - 1:
            step = 8 if n <= 120 else 16
            ax.set_xticks(np.arange(0, n + 1, step))
            ax.set_xticklabels([str(i + 1) for i in np.arange(0, n + 1, step)],
                               fontsize=8.5, color="#8a8371")
            ax.tick_params(length=2, colors="#c9c0aa")
        else:
            ax.set_xticks([])
    img = fig2b64(fig)

    def cell(k, lab):
        d = sc[k] - sc["actuel"]
        cls = "up" if d > 0.005 else ("down" if d < -0.005 else "flat")
        return (f"<span class='pill {cls}'>{lab} {sc[k]:.3f}"
                + (f" ({d:+.3f})" if k != "actuel" else "") + "</span>")

    return f"""<section><h2>{title} <span class=sub>{n} mesures</span></h2>
<img src="data:image/png;base64,{img}" alt="{title}">
<div class=row>{cell('actuel', 'aujourd’hui')}{cell('A', 'A seuil/morceau')}
{cell('B', 'B réouverture')}
<a class=listen href="lab_{stem}.html">écouter →</a></div>
<div class=note>A retient {len(A)} pics · B en ajoute
{', '.join('mes. ' + str(b + 1) for b in addB) or 'aucun'}</div></section>"""


def main():
    body = ""
    for stem, title in SONGS:
        if not (AUDIO / f"{stem}.m4a").exists():
            continue
        body += song_html(stem, title)
        print(f"  ok {title}")
    OUT.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Deux pistes pour les pics manquants</title><style>
body{{margin:0;background:#e7e0d0;font:15px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1250px;margin:0 auto;padding:22px 15px 70px}}
h1{{font:italic 600 25px Georgia,serif;margin:0 0 6px}}
.lede{{color:#6f6857;font-size:13.5px;margin-bottom:18px;max-width:930px}}
.lede b{{color:{INK}}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:14px 16px;margin-bottom:13px}}
h2{{font:700 17px system-ui;margin:0 0 8px;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371}}
img{{width:100%;border-radius:8px;display:block}}
.row{{display:flex;gap:8px;align-items:center;margin-top:9px;flex-wrap:wrap}}
.pill{{font:600 12.5px system-ui;padding:4px 9px;border-radius:20px;background:#f2ede0;color:#4a4438}}
.pill.up{{background:#dff0e6;color:#17603f}} .pill.down{{background:#f7e2df;color:#8a2b2b}}
.listen{{margin-left:auto;font:600 12.5px system-ui;color:#8a2b2b;text-decoration:none}}
.note{{font-size:12px;color:#8a8371;margin-top:6px}}
</style></head><body><div class=wrap>
<h1>Deux pistes pour les pics manquants</h1>
<div class=lede>Le cas de départ : sur Blue Lights le pic de la mesure 49 existe
mais reste sous le seuil, et l'ajouter fait passer le morceau de 0,696 à 0,813.
Deux façons d'aller le chercher — <b class=a>A</b>, un seuil par morceau (garder
autant de pics que le morceau a de tranches de huit mesures) ; <b class=b>B</b>,
une réouverture (relancer et promouvoir tout « suivant » qui tombe au milieu
d'un bloc de huit).<br><br>
<b>Les deux perdent en médiane</b> — 0,615 pour A et 0,768 pour B, contre 0,789
aujourd'hui — et c'est le détail qui compte. Sur le profil : triangles
<b style="color:#0d2437">noirs</b> = les pics d'aujourd'hui,
<b style="color:#8a2b2b">rouges</b> = ceux qu'ajoute A,
<b style="color:#1f8a5b">verts</b> = ceux qu'ajoute B, creux = les suivants que
personne ne retient. Les traits rouges fins sont tes frontières.<br><br>
<b>Ce qu'il faut regarder</b> : B répare exactement Blue Lights et casse les
morceaux déjà justes (This Love, Bein' Green, Don't Know Why) — elle coupe des
blocs de 8 qui sont de vrais blocs de 8. Une garde a été testée (ne couper que si
les deux moitiés ne se ressemblent pas) : elle n'écarte que deux coupes sur douze
morceaux et laisse passer toutes celles qui cassent. <b>Ce qui sépare Blue Lights
de This Love n'est donc pas la cohérence interne du bloc</b> — c'est la question
que cette page pose.</div>
{body}</div></body></html>""")
    print(f"wrote docs/plots/peak_options.html ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
