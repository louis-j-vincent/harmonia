"""« Pourquoi on ne prend pas juste une ligne de la matrice ? » — la réponse.

    python scripts/period_rows.py [<stem> ...]
      -> /reports/period_rows_<stems>.html

Louis, 2026-08-05, en lisant l'étape 4 sur Grenade : « la matrice SSM est ultra
claire. Et quand on en est à la détection du décalage, pour savoir si ça se
répète tous les deux ou tous les quatre, c'est ultra pas clair, et je ne
comprends pas pourquoi. Pour moi ce qu'on devrait juste faire c'est prendre une
ligne de la matrice et la regarder. »

Il a raison sur le diagnostic, et il y a DEUX problèmes derrière, dont un qui
est de ma faute :

1. **Le graphique.** Je le traçais depuis zéro. Toutes les barres valent entre
   0,5 et 0,9, donc elles se ressemblent toutes à l'œil alors que l'écart est
   parfois net. Corrigé ici : l'axe part du minimum, et la deuxième version
   trace la MARGE au-dessus de la médiane, ce qui est la quantité qui décide.

2. **La quantité elle-même, et là c'est structurel.** La moyenne par décalage
   ne peut pas séparer une période de ses multiples : un morceau qui se répète
   toutes les 4 mesures se répète AUSSI toutes les 8, 12, 16. Et si la cellule
   de 4 est faite de deux moitiés qui se ressemblent, le décalage 2 monte
   aussi. Mesuré : The Walk donne 0,900 à 4 mesures et 0,896 à 2 — 0,4 %
   d'écart. Let It Be, 0,940 à 4 et 0,929 à 12 — 1,2 %.

Sa proposition, mesurée honnêtement : lire UNE ligne, y détecter les pics, et
prendre l'écart entre pics consécutifs. C'est plus direct, et ça donne une
image lisible. Mais ça a son propre biais, dans l'autre sens : l'écart entre
deux pics CONSÉCUTIFS est la plus petite unité qui se répète, donc une boucle
de 4 mesures faite de deux moitiés proches vote « 2 ». Sur Grenade le vote des
lignes donne 2 mesures à 43 % contre 4 à 31 % — et c'est bien 4.

La page montre les deux, sur les mêmes morceaux, sans trancher : les lignes
brutes sont là pour être lues à l'œil.
"""
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from scipy.signal import find_peaks      # noqa: E402

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from ssm_rows_plot import fig2b64                                  # noqa: E402
from harmonia_min import sections as hs, musx as mx                # noqa: E402
import harmonia_min.harmonic_sections as HS                        # noqa: E402

INK = "#1c1c1c"
RED, BLUE, GREEN = "#8a2b2b", "#2a6fb0", "#1f8a5b"
DEFAULT = ["bruno_mars_grenade_official_music_video",
           "let_it_be_remastered_2009", "maroon_5_this_love",
           "norah_jones_don_t_know_why", "mayer_hawthorne_the_walk"]


def load(stem):
    from harmonia_min import pipeline as _pl
    real = hs.detect_sections
    c = {}

    def spy(g, a, t, bars=None, **k):
        c.update(grid=g)
        return real(g, a, t, bars, **k)

    hs.detect_sections = spy
    try:
        _pl.analyze(HERE / f"docs/audio/{stem}.m4a", title="x", file_key="x",
                    audio_url="")
    finally:
        hs.detect_sections = real
    grid = c["grid"]
    V = HS.harmonic_vectors(mx.frame_posteriors(HERE / f"docs/audio/{stem}.m4a")[0], grid)
    return V @ V.T, len(grid) - 1


def row_peaks(S, b, thr):
    row = S[b]
    pk, _ = find_peaks(row, height=thr)
    return [int(p) for p in pk if abs(p - b) >= HS.LAG_MIN]


def song_html(stem):
    S, n = load(stem)
    thr = float(np.quantile(HS.off_diagonal(S), 0.90))
    lags = list(range(HS.LAG_MIN, min(HS.LAG_MAX, n - 2) + 1))
    means = np.array([np.mean([S[b, b + L] for b in range(n - L)]) for L in lags])
    won = lags[int(means.argmax())]

    # ── A. le graphique actuel, et le même lisible ──────────────────────────
    fig, axs = plt.subplots(1, 3, figsize=(13, 2.9))
    cols = [RED if L == won else "#9db4cc" for L in lags]
    axs[0].bar(lags, means, color=cols)
    axs[0].set_ylim(0, 1)
    axs[0].set_title("ce que je te montrais — axe depuis 0", fontsize=8.5, loc="left")
    axs[1].bar(lags, means, color=cols)
    axs[1].set_ylim(means.min() * .985, means.max() * 1.004)
    axs[1].set_title("le même, axe sur la vraie plage", fontsize=8.5, loc="left")
    marg = means - np.median(means)
    axs[2].bar(lags, marg, color=cols)
    axs[2].axhline(0, color=INK, lw=.8)
    axs[2].set_title("la marge au-dessus de la médiane", fontsize=8.5, loc="left")
    for ax in axs:
        ax.set_xlabel("décalage testé (mesures)", fontsize=8)
        ax.set_xticks([L for L in lags if L % 2 == 0])
    pA = fig2b64(fig)

    order = np.argsort(means)[::-1]
    gap = (means[order[0]] - means[order[1]]) / means[order[0]] * 100

    # ── B. des lignes brutes, sa proposition ────────────────────────────────
    picks = [b for b in range(n) if len(row_peaks(S, b, thr)) >= 2][:200]
    picks = picks[::max(1, len(picks) // 6)][:6] or list(range(min(6, n)))
    fig, axs = plt.subplots(len(picks), 1, figsize=(13, 1.5 * len(picks)),
                            gridspec_kw={"hspace": .75})
    axs = np.atleast_1d(axs)
    for ax, b in zip(axs, picks):
        ax.plot(range(n), S[b], lw=1.1, color=INK)
        ax.axhline(thr, color=BLUE, lw=1, ls="--")
        ax.axvline(b, color=RED, lw=1.6)
        pk = row_peaks(S, b, thr)
        ax.plot(pk, S[b][pk], "o", color=RED, ms=6, mfc="none", mew=1.7)
        d = [y - x for x, y in zip(sorted(pk), sorted(pk)[1:])]
        ax.set_ylabel(f"mesure {b+1}", fontsize=8, rotation=0, ha="right", va="center")
        ax.set_title("écarts entre pics : " + (", ".join(map(str, d)) or "—"),
                     fontsize=7.5, loc="left", color=RED)
        ax.set_ylim(0, 1.05)
        ax.set_yticks([])
    axs[-1].set_xlabel("mesure", fontsize=8)
    pB = fig2b64(fig)

    # ── C. le vote de toutes les lignes ─────────────────────────────────────
    votes = {}
    for b in range(n):
        pk = sorted(row_peaks(S, b, thr))
        for x, y in zip(pk, pk[1:]):
            if HS.LAG_MIN <= y - x <= HS.LAG_MAX:
                votes[y - x] = votes.get(y - x, 0) + 1
    tot = sum(votes.values()) or 1
    ks = sorted(votes)
    fig, ax = plt.subplots(figsize=(13, 2.4))
    top = max(votes, key=votes.get) if votes else None
    ax.bar(ks, [votes[k] for k in ks],
           color=[GREEN if k == top else "#9db4cc" for k in ks])
    ax.set_xlabel("écart entre deux pics consécutifs d'une même ligne (mesures)",
                  fontsize=8)
    ax.set_ylabel("nombre de\nlignes", fontsize=8)
    ax.set_xticks(ks)
    pC = fig2b64(fig)

    rank = " · ".join(f"{k} mes {100*votes[k]/tot:.0f} %"
                      for k in sorted(votes, key=votes.get, reverse=True)[:4])
    title = stem.replace("_", " ").title()
    return f"""<section><h2>{title} <span class=sub>{n} mesures</span></h2>

<h3>A · la moyenne par décalage — ce que fait le code</h3>
<p class=ex>Pour chaque distance, la similarité moyenne entre une mesure et
celle qui la suit de cette distance. <b>Le premier graphique est celui que je te
montrais, et il est mal fait</b> : l'axe part de zéro, donc tout se ressemble.
Les deux autres montrent la même chose lisiblement.</p>
<img src="data:image/png;base64,{pA}">
<p class=ex>Verdict : <b>{won} mesures</b>, mais l'écart avec le deuxième
({lags[int(order[1])]} mesures) n'est que de <b>{gap:.1f} %</b>. C'est la limite
de fond de cette quantité : un morceau qui se répète toutes les 4 mesures se
répète aussi toutes les 8, 12, 16 — <b>tous les multiples marquent haut</b>, et
si la cellule de 4 est faite de deux moitiés qui se ressemblent, 2 marque haut
aussi. La moyenne ne peut pas les départager.</p>

<h3>B · ta proposition — lire une ligne</h3>
<p class=ex>Une ligne de la matrice, c'est « cette mesure-là, contre toutes les
autres ». Le trait rouge est sa propre position, les cercles sont ses pics, le
pointillé bleu est le q90 du morceau. Les écarts entre pics sont écrits
au-dessus.</p>
<img src="data:image/png;base64,{pB}">

<h3>C · ce que voteraient toutes les lignes</h3>
<img src="data:image/png;base64,{pC}">
<p class=ex>Chaque ligne vote pour l'écart entre ses pics consécutifs.
Classement : <b>{rank}</b>. <b>Le biais de cette lecture, dans l'autre sens :
l'écart entre deux pics CONSÉCUTIFS est la plus petite unité qui se répète.</b>
Une boucle de 4 mesures faite de deux moitiés proches vote « 2 ». C'est direct
et lisible — mais ça ne donne pas la bonne réponse tout seul non plus.</p>
</section>"""


def main():
    stems = sys.argv[1:] or DEFAULT
    body = ""
    for st in stems:
        if not (HERE / f"docs/audio/{st}.m4a").exists():
            print(f"  !! {st} introuvable")
            continue
        body += song_html(st)
        print(f"  ok {st}")
    out = HERE / ("harmonia_min/state/reports/period_rows_"
                  + "_".join(s[:20] for s in stems[:3]) + ".html")
    out.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Pourquoi pas juste une ligne de la matrice ?</title><style>
*{{box-sizing:border-box;-webkit-tap-highlight-color:transparent}}
body{{margin:0;background:#e7e0d0;font:16px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:960px;margin:0 auto;padding:20px 12px calc(40px + env(safe-area-inset-bottom))}}
h1{{font:italic 600 24px Georgia,serif;margin:0 0 3px}}
.lede{{color:#6f6858;font-size:13.5px;line-height:1.55;margin-bottom:20px}}
.lede b{{color:{INK}}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:14px 13px;margin-bottom:16px}}
h2{{font:700 19px system-ui;margin:0 0 10px;color:{RED}}}
h3{{font:700 13px system-ui;margin:18px 0 5px}}
.sub{{font:500 12px system-ui;color:#8a8371}}
.ex{{font:400 13px/1.5 system-ui;color:#6f6858;margin:0 0 9px}}
.ex b{{color:{INK}}}
img{{max-width:100%;border-radius:8px;display:block}}
</style></head><body><div class=wrap>
<h1>Pourquoi pas juste une ligne de la matrice ?</h1>
<div class=lede>Tu as raison, et il y a deux problèmes derrière, dont un qui est
de ma faute.<br><br><b>1. Le graphique était mal fait.</b> Je traçais la moyenne
par décalage depuis zéro : toutes les barres valent entre 0,5 et 0,9, donc elles
se ressemblent toutes à l'œil alors que l'écart est parfois net. Corrigé ici.
<br><br><b>2. La quantité elle-même ne peut pas trancher, et c'est structurel.</b>
Un morceau qui se répète toutes les 4 mesures se répète AUSSI toutes les 8, 12,
16 : tous les multiples marquent haut. Et si la cellule de 4 est faite de deux
moitiés qui se ressemblent, 2 marque haut aussi. Mesuré : The Walk donne 0,900 à
4 mesures et 0,896 à 2 — <b>0,4 % d'écart</b>. Let It Be, 0,940 à 4 et 0,929 à
12 — 1,2 %.
<br><br><b>Ta proposition, mesurée honnêtement :</b> lire une ligne, ses pics,
l'écart entre pics. C'est plus direct et bien plus lisible. Mais elle a son
propre biais, en sens inverse : l'écart entre deux pics <i>consécutifs</i> est la
plus petite unité qui se répète, donc une boucle de 4 faite de deux moitiés
proches vote « 2 ». Sur Grenade, le vote des lignes donne 2 mesures à 43 % contre
4 à 31 % — et c'est bien 4.
<br><br>Les deux sont ci-dessous, sur les mêmes morceaux. Les lignes brutes sont
là pour être lues à l'œil : c'est toi qui tranches.</div>
{body}</div></body></html>""")
    print(f"wrote {out.relative_to(HERE)} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
