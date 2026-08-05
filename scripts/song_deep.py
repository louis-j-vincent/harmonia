"""Tout, en détail, sur un morceau — de la grille de mesures aux sections.

    python scripts/song_deep.py <stem> [<stem> ...]
      -> /reports/song_deep_<stems>.html

Louis, 2026-08-05 : « pour Grenade et Let It Be, tu me ressors tout en détail,
la matrice SSM, les pics, comment on déduit le dictionnaire… je veux quelque
chose d'extensive, et cliquable depuis mon tél. »

Dix étapes, dans l'ordre où le code les exécute, chacune avec son image :

  1. les mesures et leurs accords
  2. la matrice de similarité harmonique
  3. la distribution de cette matrice, et les seuils qui en sortent
  4. la recherche de période — quelle distance se répète le plus
  5. par entrée : la plus longue série de mesures fortes, à quel décalage
  6. par entrée : le motif glissé, et les deux critères qui gardent un pic
  7. qui possède quelle mesure
  8. les quatre passes qui mènent aux sections (brut → absorbé → fusionné → collé)
  9. chaque section contre chaque section, lue mesure à mesure
 10. les accords de chaque lettre, et les boutons pour écouter

Tout importe `harmonia_min.harmonic_sections` — le code que l'app exécute.
La page est faite pour le téléphone : une colonne, images à l'échelle.
"""
from __future__ import annotations

import copy
import itertools
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
from ssm_rows_plot import fig2b64                                  # noqa: E402
from harmonia_min import sections as hs, musx as mx                # noqa: E402
import harmonia_min.harmonic_sections as HS                        # noqa: E402

INK = "#1c1c1c"
COLS = ["#8a2b2b", "#1f8a5b", "#2a6fb0", "#c58a2e", "#7c3aed", "#0f766e",
        "#be123c", "#0369a1"]
NAMES = "C Db D Eb E F Gb G Ab A Bb B".split()


def capture(stem):
    from harmonia_min import pipeline as _pl
    real = hs.detect_sections
    cap = {}

    def spy(grid, arr, times, bars=None, **kw):
        cap.update(grid=grid, bars=copy.deepcopy(bars))
        return real(grid, arr, times, bars, **kw)

    hs.detect_sections = spy
    try:
        _pl.analyze(HERE / f"docs/audio/{stem}.m4a", title=stem, file_key="x",
                    audio_url="")
    finally:
        hs.detect_sections = real
    return cap


def bar_text(bar):
    if not bar:
        return "—"
    return " ".join("N.C." if c.get("nc") else NAMES[c["root"]] + c["q"]
                    for c in bar)


def strip(ax, secs, n, key="letter"):
    seen = {}
    for s in secs:
        seen.setdefault(s[key], COLS[len(seen) % len(COLS)])
        ax.add_patch(plt.Rectangle((s["b0"], 0), s["b1"] - s["b0"] + 1, 1,
                                   color=seen[s[key]]))
        if s["b1"] - s["b0"] >= 1:
            ax.text((s["b0"] + s["b1"] + 1) / 2, .5, s[key], ha="center",
                    va="center", color="#fff", fontsize=8, fontweight="bold")
        ax.axvline(s["b0"], color="#fff", lw=1.2)
    ax.set_xlim(0, n); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])


def song_html(stem):
    cap = capture(stem)
    grid, bars = cap["grid"], cap["bars"]
    n = len(grid) - 1
    V = HS.harmonic_vectors(mx.frame_posteriors(HERE / f"docs/audio/{stem}.m4a")[0], grid)
    S = V @ V.T
    off = HS.off_diagonal(S)
    q80 = float(np.quantile(off, HS.CONT_QUANTILE))
    q90 = float(np.quantile(off, HS.PHASE_QUANTILE))
    entries, _ = HS.build_dictionary(S, n)
    raw = HS.sections_from(S, n, entries, post_process=False)
    absorbed = HS.absorb_short(S, [dict(s) for s in raw])
    merged = HS.merge_same_letters(S, [dict(s) for s in absorbed])
    final = HS.coalesce_adjacent([dict(s) for s in merged])

    P = []          # (titre, explication, image)

    # 1 — the bars
    rows = "".join(f"<tr><td class=num>{b+1}</td><td class=ch>{bar_text(bars[b])}</td>"
                   f"<td class=num>{grid[b]:.1f}s</td></tr>" for b in range(n))
    bars_tbl = f"<div class=scroll><table class=bars>{rows}</table></div>"

    # 2 — the matrix
    fig, ax = plt.subplots(figsize=(6.4, 6.4))
    ax.imshow(S, cmap="RdYlBu_r", origin="lower",
              vmin=np.percentile(S, 5), vmax=np.percentile(S, 99))
    ax.set_xticks([]); ax.set_yticks([])
    P.append(("2 · la matrice de similarité harmonique",
              "Chaque case (i, j) dit à quel point la mesure i et la mesure j "
              "ont la même harmonie. Rouge = pareil, bleu = différent. Les "
              "diagonales rouges décalées de la grande diagonale sont les "
              "répétitions : une ligne rouge à 8 mesures de distance veut dire "
              "que le morceau se répète toutes les 8 mesures.", fig2b64(fig)))

    # 3 — the distribution
    fig, axs = plt.subplots(1, 2, figsize=(11, 2.9))
    for ax, lg in ((axs[0], False), (axs[1], True)):
        ax.hist(off, bins=80, color="#9db4cc")
        for v, c, lab in ((float(np.median(off)), "#8a8371", "médiane"),
                          (q80, "#2a6fb0", "q80 · une série continue"),
                          (q90, "#8a2b2b", "q90 · une série démarre")):
            ax.axvline(v, color=c, lw=1.5)
            ax.text(v, ax.get_ylim()[1] * .88, f" {lab.split(' ·')[0]}={v:.3f}",
                    color=c, fontsize=7.5)
        if lg:
            ax.set_yscale("log")
        ax.set_xlabel("similarité entre deux mesures", fontsize=8)
    P.append(("3 · la distribution, et les deux seuils qui en sortent",
              f"Toutes les paires de mesures distantes d'au moins {HS.LAG_MIN} "
              f"mesures. Médiane {float(np.median(off)):.3f}. Une série de "
              f"mesures fortes DÉMARRE au-dessus de q90 ({q90:.3f}) et "
              f"CONTINUE tant qu'on reste au-dessus de q80 ({q80:.3f}) — c'est "
              "l'hystérésis, elle empêche une mesure un peu faible de couper un "
              "motif en deux. À droite la même chose en échelle log, pour voir "
              "la queue où tout se joue.", fig2b64(fig)))

    # 4 — the period search.
    # Drawn on the REAL range and as the margin above the median, never from
    # zero. Louis, 2026-08-05: from zero, values between 0.5 and 0.9 all look
    # alike and the step reads as undecided when it is not — « tu me montrais
    # le mauvais graphique, avec le vrai la méthode A est la bonne ».
    lags = list(range(HS.LAG_MIN, min(HS.LAG_MAX, n - 2) + 1))
    means = np.array([float(np.mean([S[b, b + L] for b in range(n - L)]))
                      for L in lags])
    won = lags[int(means.argmax())]
    order = np.argsort(means)[::-1]
    gap = (means[order[0]] - means[order[1]]) / means[order[0]] * 100
    cols = ["#8a2b2b" if L == won else "#9db4cc" for L in lags]
    fig, axs = plt.subplots(1, 2, figsize=(11, 2.7))
    axs[0].bar(lags, means, color=cols)
    axs[0].set_ylim(means.min() * .985, means.max() * 1.004)
    axs[0].set_ylabel("similarité\nmoyenne", fontsize=8)
    axs[0].set_title("sur la vraie plage de valeurs", fontsize=8.5, loc="left")
    axs[1].bar(lags, means - np.median(means), color=cols)
    axs[1].axhline(0, color=INK, lw=.8)
    axs[1].set_ylabel("marge sur\nla médiane", fontsize=8)
    axs[1].set_title("la marge au-dessus de la médiane", fontsize=8.5, loc="left")
    for ax in axs:
        ax.set_xlabel("distance testée, en mesures", fontsize=8)
        ax.set_xticks([L for L in lags if L % 2 == 0])
    P.append(("4 · la recherche de période",
              f"Pour chaque distance, la similarité moyenne entre une mesure et "
              f"celle qui la suit de cette distance. Verdict : "
              f"<b>{won} mesures</b>, devant {lags[int(order[1])]} de "
              f"<b>{gap:.1f} %</b>. Cette période donne le PREMIER motif ; les "
              f"entrées suivantes sont cherchées autrement (étape 5, la plus "
              f"longue série), parce que la moyenne rate un bloc qui se répète "
              f"une seule fois mais parfaitement. "
              f"<br>Les deux vues sont la même mesure : la gauche sur la plage "
              f"réelle, la droite en écart à la médiane. Tracées depuis zéro "
              f"elles paraissent indécises alors qu'elles ne le sont pas — "
              f"c'est ce que montre "
              f"<a href='/reports/period_rows_bruno_mars_grenade_o_let_it_be_"
              f"remastered_maroon_5_this_love.html'>la page sur les lignes de "
              f"la matrice</a>.", fig2b64(fig)))

    # 5+6 — per entry
    owner = -np.ones(n, int)
    for ei, e in enumerate(entries):
        col = COLS[ei % len(COLS)]
        L, b0, curve = e["L"], e["b0"], e["curve"]
        for b in sorted(set([b0] + list(e["occ"]))):
            owner[b:min(n, b + L)] = ei
        fig, axx = plt.subplots(2, 1, figsize=(11, 5.2),
                                gridspec_kw={"height_ratios": [1, 1.15]})
        # the run at the entry's own lag, if any
        d = L
        vals = [float(S[b, b + d]) for b in range(n - d)]
        axx[0].plot(range(len(vals)), vals, lw=1.2, color=INK)
        axx[0].axhline(q90, color="#8a2b2b", lw=1.2, ls="--")
        axx[0].axhline(q80, color="#2a6fb0", lw=1.2, ls=":")
        axx[0].axvspan(b0, min(b0 + L, len(vals)), color=col, alpha=.18)
        axx[0].set_title(f"la mesure b comparée à la mesure b+{d} — "
                         f"la série retenue démarre en {b0+1}",
                         fontsize=8.5, loc="left", color=col)
        axx[0].set_xlabel("mesure b", fontsize=8)
        # the slide curve + the two peak criteria
        axx[1].plot(np.arange(len(curve)), curve, lw=1.3, color=INK)
        axx[1].axhline(HS.INITIAL_PEAK_FRAC * curve[b0], color=col, ls="--", lw=1.2)
        axx[1].text(0, HS.INITIAL_PEAK_FRAC * curve[b0],
                    f" {int(HS.INITIAL_PEAK_FRAC*100)} % du pic initial",
                    color=col, fontsize=7.5, va="bottom")
        axx[1].axvline(b0, color=col, lw=1.8)
        axx[1].plot(e["occ"], curve[e["occ"]], "o", color=col, ms=9,
                    mfc="none", mew=2)
        axx[1].set_xlabel("on pose le motif à partir de cette mesure", fontsize=8)
        axx[1].set_title("le motif glissé le long du morceau — les cercles sont "
                         "les occurrences gardées", fontsize=8.5, loc="left")
        P.append((f"5–6 · entrée {ei+1} — un motif de {L} mesures pris mesure {b0+1}",
                  f"En haut : chaque mesure comparée à celle située {d} mesures "
                  f"plus loin. La plus longue suite au-dessus des seuils donne "
                  f"la longueur du motif. En bas : ce motif est glissé sur tout "
                  f"le morceau et on lit la DIAGONALE (mesure 1 contre mesure 1, "
                  f"mesure 2 contre mesure 2…). Un pic n'est gardé que s'il "
                  f"dépasse son voisinage local ET atteint "
                  f"{int(HS.INITIAL_PEAK_FRAC*100)} % du pic initial. "
                  f"Résultat : {len(e['occ'])} occurrences, mesures "
                  f"{', '.join(str(o+1) for o in e['occ'])}.", fig2b64(fig)))

    # 7 — coverage
    fig, ax = plt.subplots(figsize=(11, 1.1))
    for b in range(n):
        c = COLS[owner[b] % len(COLS)] if owner[b] >= 0 else "#e5dcc6"
        ax.add_patch(plt.Rectangle((b, 0), 1, 1, color=c))
    ax.set_xlim(0, n); ax.set_ylim(0, 1); ax.set_yticks([])
    ax.set_xlabel("mesure", fontsize=8)
    P.append(("7 · qui possède quelle mesure",
              "Une couleur par entrée du dictionnaire, gris = aucune entrée ne "
              "la revendique. Le gris devient une section à part, qu'on essaie "
              "ensuite d'apparier avec les autres gris.", fig2b64(fig)))

    # 8 — the four passes
    fig, axs = plt.subplots(4, 1, figsize=(11, 3.4), gridspec_kw={"hspace": .9})
    for ax, (lab, secs) in zip(axs, (("brut", raw), ("absorbé", absorbed),
                                     ("fusionné", merged), ("collé", final))):
        strip(ax, secs, n)
        ax.set_ylabel(lab, fontsize=8, rotation=0, ha="right", va="center")
    axs[-1].set_xlabel("mesure", fontsize=8)
    P.append(("8 · les quatre passes qui mènent aux sections",
              f"<b>brut</b> : une lettre par entrée du dictionnaire, plus une "
              f"par trou. <b>absorbé</b> : une section de moins de "
              f"{HS.MIN_SECTION_BARS} mesures rejoint sa voisine. "
              f"<b>fusionné</b> : deux lettres deviennent une quand une de "
              f"leurs sections en recouvre une autre à "
              f"{HS.SAME_SECTION:.2f} (décalage jusqu'à {HS.MAX_SHIFT} mesures "
              f"autorisé). <b>collé</b> : deux sections voisines de même lettre "
              f"n'en font qu'une.", fig2b64(fig)))

    # 9 — section against section
    k = len(final)
    D = np.array([[HS.section_match(S, a, b, allow_shift=True) for b in final]
                  for a in final])
    fig, ax = plt.subplots(figsize=(0.66 * k + 2.2, 0.66 * k + 1.9))
    ax.imshow(D, cmap="RdYlBu_r", vmin=0, vmax=1)
    for a in range(k):
        for b in range(k):
            ax.text(b, a, f"{D[a, b]:.2f}", ha="center", va="center", fontsize=7,
                    color="#fff" if D[a, b] > .72 or D[a, b] < .18 else INK)
    lab = [f"{s['letter']} {s['b0']+1}" for s in final]
    ax.set_xticks(range(k)); ax.set_xticklabels(lab, fontsize=7, rotation=90)
    ax.set_yticks(range(k)); ax.set_yticklabels(lab, fontsize=7)
    P.append(("9 · chaque section contre chaque section",
              f"Lue mesure à mesure, avec un décalage jusqu'à {HS.MAX_SHIFT} "
              f"mesures. 1,00 = exactement la même harmonie. C'est cette table "
              f"qui décide des fusions : au-dessus de {HS.SAME_SECTION:.2f}, "
              "deux lettres n'en font qu'une.", fig2b64(fig)))

    # 10 — the letters
    firsts = {}
    for s in final:
        firsts.setdefault(s["letter"], s)
    letters = sorted(firsts)
    ltr = ""
    for i, L in enumerate(letters):
        s = firsts[L]
        spans = ", ".join(f"{x['b0']+1}–{x['b1']+1}" for x in final if x["letter"] == L)
        body = "".join(f"<tr><td class=num>{b - s['b0'] + 1}</td>"
                       f"<td class=ch>{bar_text(bars[b])}</td></tr>"
                       for b in range(s["b0"], s["b1"] + 1))
        ltr += (f"<div class=letter><div class=lh style='background:{COLS[i%len(COLS)]}'>"
                f"{L}</div><div class=ls>mesures {spans}</div>"
                f"<div class=scroll><table class=bars>{body}</table></div></div>")

    panels = "".join(
        f"<div class=step><h3>{t}</h3><p class=ex>{ex}</p>"
        f"<img src='data:image/png;base64,{img}'></div>" for t, ex, img in P)
    ug = HERE / f"harmonia_min/state/charts/min_{stem}__ug.json"
    btns = (f"<a class='b us' href='/?open=min_{stem}'>▶ écouter notre chart</a>"
            + (f"<a class='b tab' href='/?open=min_{stem}__ug'>▶ écouter le chart du TAB</a>"
               if ug.exists() else ""))
    title = stem.replace("_", " ").title()
    return f"""<section><h2>{title}</h2>
<div class=meta>{n} mesures · {len(entries)} entrées au dictionnaire ·
{len(final)} sections · {len(letters)} lettres</div>
<div class=btns>{btns}</div>
<div class=step><h3>1 · les mesures et leurs accords</h3>
<p class=ex>La grille vient des vraies mesures détectées dans l'audio. Tout ce
qui suit est calculé là-dessus : si cette colonne est fausse, rien en dessous
ne peut être juste.</p>{bars_tbl}</div>
{panels}
<div class=step><h3>10 · ce que joue chaque lettre</h3>
<p class=ex>La première occurrence de chaque section, écrite en entier.</p>
{ltr}</div>
</section>"""


def main():
    stems = sys.argv[1:] or ["bruno_mars_grenade_official_music_video",
                             "let_it_be_remastered_2009"]
    body = ""
    for st in stems:
        if not (HERE / f"docs/audio/{st}.m4a").exists():
            print(f"  !! docs/audio/{st}.m4a introuvable")
            continue
        body += song_html(st)
        print(f"  ok {st}")
    out = HERE / ("harmonia_min/state/reports/song_deep_"
                  + "_".join(s[:24] for s in stems) + ".html")
    out.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Tout en détail — {', '.join(s.replace('_',' ').title() for s in stems)}</title><style>
*{{box-sizing:border-box;-webkit-tap-highlight-color:transparent}}
body{{margin:0;background:#e7e0d0;font:16px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:900px;margin:0 auto;padding:20px 12px calc(40px + env(safe-area-inset-bottom))}}
h1{{font:italic 600 24px Georgia,serif;margin:0 0 3px}}
.lede{{color:#8a8371;font-size:13px;margin-bottom:20px}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:14px 13px;margin-bottom:16px}}
h2{{font:700 20px system-ui;margin:0 0 3px;color:#8a2b2b}}
.meta{{font:500 12px system-ui;color:#8a8371;margin-bottom:12px}}
.step{{border-top:1px solid #eee4cd;padding-top:14px;margin-top:16px}}
h3{{font:700 13px system-ui;margin:0 0 5px;color:{INK}}}
.ex{{font:400 13px/1.5 system-ui;color:#6f6858;margin:0 0 10px}}
.ex b{{color:{INK}}}
img{{max-width:100%;border-radius:8px;display:block}}
.scroll{{max-height:340px;overflow:auto;border:1px solid #e5dcc6;border-radius:8px}}
table.bars{{border-collapse:collapse;font-size:13px;width:100%}}
table.bars td{{border-bottom:1px solid #f0ead9;padding:2px 8px}}
td.num{{color:#8a8371;text-align:right;width:52px;font-size:11.5px}}
td.ch{{font-family:ui-monospace,Menlo,monospace;font-size:12.5px}}
.letter{{margin:10px 0}}
.lh{{display:inline-block;color:#fff;font:700 14px system-ui;border-radius:7px;padding:2px 11px}}
.ls{{display:inline-block;font:500 12px system-ui;color:#8a8371;margin-left:8px}}
.btns{{display:flex;gap:8px;margin-bottom:6px;flex-wrap:wrap}}
.b{{flex:1;min-width:150px;text-align:center;text-decoration:none;border-radius:10px;
   padding:12px 8px;font:700 13px system-ui;color:#fff}}
.b.tab{{background:#2a6fb0}} .b.us{{background:#8a2b2b}} .b:active{{opacity:.75}}
</style></head><body><div class=wrap>
<h1>Tout, en détail</h1>
<div class=lede>De la grille de mesures aux sections, dans l'ordre où le code
les exécute. Dix étapes, chacune avec son image et ce qu'elle décide.</div>
{body}</div></body></html>""")
    print(f"wrote {out.relative_to(HERE)} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
