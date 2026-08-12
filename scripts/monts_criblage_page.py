"""Les monts de basse+harmonie, et ce que le modèle appris trouve en plus.

    .venv/bin/python scripts/monts_criblage_page.py  ->  /plots/monts_criblage.html

Deux résultats du 2026-08-12, mis en images sur le même axe de mesures :

  1. **Les monts.** Louis : « le profil basse + harmonie fait souvent des pics
     allongés pseudo-symétriques (ex. mesures 19 à 33 sur The Walk), et ceux-là
     sont TOUJOURS marqueurs d'une section. » Vérifié : 21 monts sur les douze
     morceaux, 21 contiennent une de ses frontières. Quatre portent même une
     frontière qu'aucun de nos pics ne trouve. Ils sont ombrés sur la première
     bande, avec leur sommet marqué.

  2. **Le criblage.** Une régression logistique sur 110 features par mesure,
     entraînée un-contre-tous (jamais le morceau qu'on regarde), contre notre
     profil fusionné. Le profil est parfaitement précis et sourd (précision 1,00,
     rappel 0,40) ; le modèle trouve plus, moins proprement (0,62 / 0,49).
     Deuxième bande : les deux courbes et ce que chacune retient.

Un mont = suite contiguë de la nouveauté basse+harmonie au-dessus de la médiane
du morceau, large d'au moins 6 mesures. C'est la définition la plus simple qui
capture ce que Louis décrit ; elle n'a pas été optimisée.
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

from ssm_zoo import SONGS, GT_LINE, gt_sections, fig2b64    # noqa: E402
from peak_profile import fused_profile, select_peaks        # noqa: E402
import section_features as SF                               # noqa: E402
from section_screen import prf, pick                        # noqa: E402

OUT = HERE / "docs" / "plots" / "monts_criblage.html"
INK = "#1c1c1c"
BUMP = "#c9a227"
MODEL = "#8a2b2b"
WMIN = 6          # un mont fait au moins tant de mesures
QUANT = 0.5       # …et se tient au-dessus de la médiane du morceau


def bumps(nov, hb, wmin=WMIN, q=QUANT):
    """[(b0, b1, sommet)] en mesures — les monts larges."""
    x = np.nan_to_num(nov)
    pos = x[x > 0]
    thr = float(np.quantile(pos, q)) if pos.size else 0.0
    out, i, m = [], 0, len(x)
    while i < m:
        if x[i] >= thr:
            j = i
            while j + 1 < m and x[j + 1] >= thr:
                j += 1
            if (j - i + 1) / hb >= wmin:
                top = (i + int(np.argmax(x[i:j + 1]))) / hb
                out.append((i / hb, (j + 1) / hb, top))
            i = j + 1
        else:
            i += 1
    return out


def loso_probs():
    """La probabilité par mesure, un-contre-tous. Jamais le morceau évalué."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    d = SF.load()
    stems = [s for s, _ in SONGS]
    X = {s: d[f"X:{s}"] for s in stems}
    Y = {s: d[f"y:{s}"] for s in stems}
    out = {}
    for s in stems:
        tr = [t for t in stems if t != s]
        mdl = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=2000, C=0.1,
                                               class_weight="balanced"))
        mdl.fit(np.vstack([X[t] for t in tr]), np.concatenate([Y[t] for t in tr]))
        out[s] = mdl.predict_proba(X[s])[:, 1]
    return out


def song_html(stem, title, prob) -> str:
    P, kept, rest, lines, n, extra = fused_profile(stem)
    m = len(P); hb = m // n
    bh = np.nan_to_num(next(d for d in lines if d["nom"] == "basse + harmonie")["nov"])
    gt = [s["b0"] for s in gt_sections(stem)["sections"][1:]]
    bs = bumps(bh, hb)
    prof_pk = [int(round(c * n / m)) for c in kept]
    model_pk = pick(prob)

    fig, axs = plt.subplots(2, 1, figsize=(12.6, 3.5), facecolor="#fffdf6",
                            gridspec_kw={"hspace": 0.32})
    x = (np.arange(m) + 0.5) / hb

    ax = axs[0]
    ax.fill_between(x, bh, color="#cdd8df", lw=0)
    for b0, b1, top in bs:
        ax.axvspan(b0, b1, color=BUMP, alpha=0.28, lw=0)
        ax.plot([top], [1.06], marker="v", ms=8, color="#8a6d1f", clip_on=False)
    for g in gt:
        ax.axvline(g, color=GT_LINE, lw=0.9, alpha=0.85)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("basse\n+ harmonie", rotation=0, ha="right", va="center", fontsize=9.5)

    ax = axs[1]
    ax.fill_between(x, P, color="#dfe6ea", lw=0, label="profil")
    pb = np.asarray(prob) / max(1e-9, np.max(prob))
    ax.plot(np.arange(n) + 0.5, pb, color=MODEL, lw=1.4)
    for g in gt:
        ax.axvline(g, color=GT_LINE, lw=0.9, alpha=0.85)
    for c in prof_pk:
        ax.plot([c], [1.06], marker="v", ms=8, color="#0d2437", clip_on=False)
    for c in model_pk:
        ax.plot([c], [-0.09], marker="^", ms=8, color=MODEL, clip_on=False)
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("profil (gris)\nmodèle (rouge)", rotation=0, ha="right",
                  va="center", fontsize=9.5)

    for i, ax in enumerate(axs):
        ax.set_xlim(0, n); ax.set_yticks([])
        step = 8 if n <= 120 else 16
        ax.set_xticks(np.arange(0, n + 1, step))
        ax.set_xticklabels([str(k + 1) for k in np.arange(0, n + 1, step)],
                           fontsize=8, color="#8a8371")
        ax.tick_params(length=2, colors="#c9c0aa")
        for s_ in ax.spines.values():
            s_.set_color("#e0d7c2")
    img = fig2b64(fig)

    ok = sum(1 for b0, b1, _ in bs if any(b0 - 1 <= g <= b1 + 1 for g in gt))
    neuf = sum(1 for b0, b1, _ in bs
               if any(b0 - 1 <= g <= b1 + 1 and not any(abs(p - g) <= 1 for p in prof_pk)
                      for g in gt))
    pp = prf(prof_pk, gt); pm = prf(model_pk, gt)
    return f"""<section><h2>{title} <span class=sub>{n} mesures</span></h2>
<img src="data:image/png;base64,{img}" alt="{title}">
<div class=row>
<span class="pill {'up' if bs and ok == len(bs) else 'flat'}">{len(bs)} mont(s) ·
{ok} avec une frontière{f' · {neuf} apporte(nt) du neuf' if neuf else ''}</span>
<span class=pill>profil {pp[0]:.2f}/{pp[1]:.2f}</span>
<span class="pill {'up' if pm[2] > pp[2] else 'down' if pm[2] < pp[2] else 'flat'}">
modèle {pm[0]:.2f}/{pm[1]:.2f}</span>
<a class=listen href="lab_{stem}.html">écouter →</a></div></section>"""


def main():
    probs = loso_probs()
    body = "".join(song_html(s, t, probs[s]) for s, t in SONGS)
    OUT.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Les monts, et ce que le modèle trouve en plus</title><style>
body{{margin:0;background:#e7e0d0;font:15px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1250px;margin:0 auto;padding:22px 15px 70px}}
h1{{font:italic 600 25px Georgia,serif;margin:0 0 6px}}
.lede{{color:#6f6857;font-size:13.5px;margin-bottom:18px;max-width:940px}}
.lede b{{color:{INK}}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:14px 16px;margin-bottom:13px}}
h2{{font:700 17px system-ui;margin:0 0 8px;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371}}
img{{width:100%;border-radius:8px;display:block}}
.row{{display:flex;gap:8px;align-items:center;margin-top:9px;flex-wrap:wrap}}
.pill{{font:600 12.5px system-ui;padding:4px 9px;border-radius:20px;background:#f2ede0;color:#4a4438}}
.pill.up{{background:#dff0e6;color:#17603f}} .pill.down{{background:#f7e2df;color:#8a2b2b}}
.listen{{margin-left:auto;font:600 12.5px system-ui;color:#8a2b2b;text-decoration:none}}
</style></head><body><div class=wrap>
<h1>Les monts, et ce que le modèle trouve en plus</h1>
<div class=lede><b>Bande du haut — ta règle.</b> La nouveauté de
<b>basse + harmonie</b>, avec ses <b style="color:#8a6d1f">monts</b> ombrés : une
suite large d'au moins six mesures au-dessus de la médiane du morceau. Sur les
douze morceaux, <b>21 monts et 21 contiennent une de tes frontières</b> (les
traits rouges), et quatre portent une frontière qu'aucun de nos pics ne trouve.
Le triangle marque le sommet du mont — c'est le meilleur repère à l'intérieur
(±2 mesures dans 86 % des cas).<br><br>
<b>Bande du bas — le criblage.</b> En gris notre profil fusionné et ses pics
(triangles noirs, au-dessus) ; en rouge la probabilité d'une régression
logistique sur 110 features par mesure, <b>entraînée sans jamais voir ce
morceau</b>, et ses pics (triangles rouges, en dessous). Les pastilles donnent
précision/rappel de chacun, à ±1 mesure.<br><br>
Ce qu'on lit : le profil ne se trompe presque jamais et <b>rate 60 % de tes
frontières</b> ; le modèle en trouve plus, moins proprement. Le problème n'est
pas de bien placer un pic, c'est d'en trouver assez.</div>
{body}</div></body></html>""")
    print(f"wrote docs/plots/monts_criblage.html ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
