"""Comment la longueur d'un motif est choisie, et de quoi dépend l'arbitrage.

    python scripts/length_choice.py [<stem> ...]  ->  /reports/length_choice.html

Louis, 2026-08-05 : « Tu as mis la règle que si on n'est pas sûr de la longueur
du pattern, on prend la plus petite longueur ? Il faudrait que je regarde ce que
ça donne, les plots de décision de longueur, pour pouvoir arbitrer de la règle
qui fait qu'on dit qu'on hésite entre plusieurs longueurs de mesures pour les
patterns de répétition. »

La décision, en une phrase : pour chaque distance d, on mesure « les d mesures
qui commencent ici reviennent-elles, mesure pour mesure, d mesures plus loin ? »
(`diag_match`), puis on garde les distances à LEN_TOL près de la meilleure et
au-dessus de LEN_FLOOR, et on prend la plus petite.

Deux réglages, et un seul est vraiment discutable :

  * LEN_FLOOR = 0,90 — en dessous, rien ne revient vraiment ici. Il est sur la
    même échelle que GAP_MATCH et la règle de pic : 1,00 = identique mesure pour
    mesure. Peu discutable.
  * LEN_TOL = 0,02 — c'est LUI l'arbitrage : « à quel point deux longueurs
    doivent-elles être proches pour qu'on dise qu'on hésite ». La page le montre
    de trois façons : le profil de chaque ancre, ce que chaque valeur de TOL
    choisirait, et l'effet corpus.
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
from pattern_lanes import load, fig2b64_fixed                       # noqa: E402
import harmonia_min.harmonic_sections as HS                         # noqa: E402

INK, ACC, GRN, GREY = "#1c1c1c", "#b3261e", "#1f8a5b", "#a89f8c"
TOLS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.20]
SHOWN = ["mayer_hawthorne_the_walk", "bruno_mars_grenade_official_music_video",
         "maroon_5_this_love", "norah_jones_don_t_know_why",
         "let_it_be_remastered_2009", "yesterday_remastered_2009",
         "ben_e_king_stand_by_me_audio",
         "maroon_5_she_will_be_loved_official_music_video"]


def choose(prof, tol, floor=None):
    """La règle, isolée pour qu'on puisse la faire varier."""
    floor = HS.LEN_FLOOR if floor is None else floor
    if not prof or max(prof.values()) < floor:
        return None
    keep = max(floor, max(prof.values()) - tol)
    return min(d for d, v in prof.items() if v >= keep)


def profile_fig(trace):
    """Un panneau par ancre : le profil, le plancher, la bande, le choix."""
    k = len(trace)
    fig, axs = plt.subplots(k, 1, figsize=(11.4, 1.62 * k + .5), sharex=True,
                            gridspec_kw={"hspace": .30})
    H = 1.62 * k + .5
    fig.subplots_adjust(left=.075, right=.995, top=1 - .18 / H, bottom=.52 / H)
    for ax, t in zip(np.atleast_1d(axs), trace):
        prof, L = t["prof"], t["L"]
        ds = sorted(prof)
        vs = [prof[d] for d in ds]
        top = max(vs)
        band = max(HS.LEN_FLOOR, top - HS.LEN_TOL)
        ax.axhspan(band, 1.0, color=GRN, alpha=.13, zorder=0)
        ax.axhline(HS.LEN_FLOOR, color=GREY, lw=1, ls=(0, (4, 3)), zorder=1)
        cols = [ACC if d == L else (GRN if prof[d] >= band else "#cfc7b2")
                for d in ds]
        ax.bar(ds, vs, width=.72, color=cols, zorder=2)
        for d, v in zip(ds, vs):
            if v >= HS.LEN_FLOOR or d == L:   # tout ce qui passe le plancher
                ax.text(d, v + .012, f"{v:.3f}", ha="center", fontsize=5.8,
                        color=ACC if d == L else "#5a6f5f")
        ax.set_ylim(0, 1.10)
        ax.set_yticks([0, .5, HS.LEN_FLOOR, 1])
        ax.set_yticklabels(["0", "0,5", "0,90", "1"], fontsize=6.2)
        ax.set_ylabel(f"ancre\nmes. {t['b0']+1}", fontsize=7, rotation=0,
                      ha="right", va="center", color=INK)
        n_in = sum(1 for d in ds if prof[d] >= band)
        ax.text(.998, .93, f"retenu {L} mes."
                + (f"  ·  {n_in} candidates à égalité" if n_in > 1
                   else "  ·  une seule candidate"),
                transform=ax.transAxes, ha="right", va="top", fontsize=7,
                color=ACC, fontweight="bold")
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    a = np.atleast_1d(axs)[-1]
    a.set_xlabel("longueur candidate (mesures)", fontsize=8)
    a.set_xticks(range(HS.LAG_MIN, HS.LAG_MAX + 1))
    a.tick_params(labelsize=7)
    return fig2b64_fixed(fig)


def corpus_fig(all_traces):
    """L'effet de TOL sur tout le corpus — c'est ça, l'arbitrage."""
    fig, axs = plt.subplots(1, 3, figsize=(11.4, 2.9))
    fig.subplots_adjust(left=.06, right=.985, top=.86, bottom=.20, wspace=.28)

    grid = np.arange(0, .2001, .005)
    n_hes = [np.mean([sum(1 for d in t["prof"]
                          if t["prof"][d] >= max(HS.LEN_FLOOR,
                                                 max(t["prof"].values()) - g)) > 1
                      for t in all_traces]) for g in grid]
    axs[0].plot(grid, n_hes, color=ACC, lw=1.8)
    axs[0].axvline(HS.LEN_TOL, color=INK, lw=1, ls=(0, (3, 2)))
    axs[0].set_title("part des ancres où l'on hésite\n(≥ 2 longueurs à égalité)",
                     fontsize=8, color="#6f6858")
    axs[0].set_xlabel("TOL", fontsize=7.5)
    axs[0].yaxis.set_major_formatter(lambda v, _: f"{v:.0%}")

    med = [np.median([choose(t["prof"], g) for t in all_traces]) for g in grid]
    axs[1].plot(grid, med, color=GRN, lw=1.8)
    axs[1].axvline(HS.LEN_TOL, color=INK, lw=1, ls=(0, (3, 2)))
    axs[1].set_title("longueur médiane retenue\n(mesures)", fontsize=8, color="#6f6858")
    axs[1].set_xlabel("TOL", fontsize=7.5)

    # ce que coûte « prendre la plus petite » : l'écart de score consenti
    cost = [max(t["prof"].values()) - t["prof"][t["L"]] for t in all_traces]
    axs[2].hist(cost, bins=np.arange(0, .205, .01), color="#2a6fb0", alpha=.85)
    axs[2].axvline(HS.LEN_TOL, color=INK, lw=1, ls=(0, (3, 2)))
    axs[2].set_title("écart de score consenti\n(meilleure − retenue)",
                     fontsize=8, color="#6f6858")
    axs[2].set_xlabel("écart", fontsize=7.5)
    for ax in axs:
        ax.tick_params(labelsize=6.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    return fig2b64_fixed(fig), float(np.mean([c > 0 for c in cost])), cost


def song_section(stem):
    S, n, grid = load(stem)
    tr = []
    HS.build_cells(S, n, trace=tr)
    tried = len(tr)
    tr = [t for t in tr if t["kept"]]      # les ancres devenues des motifs
    if not tr:
        return "", []
    img = profile_fig(tr)
    rows = ""
    for t in tr:
        cells = "".join(
            f"<td class='{'now' if abs(g - HS.LEN_TOL) < 1e-9 else ''}'>"
            f"{choose(t['prof'], g)}</td>" for g in TOLS)
        best = max(t["prof"], key=t["prof"].get)
        rows += (f"<tr><td>mes. {t['b0']+1}</td>{cells}"
                 f"<td>{best} ({t['prof'][best]:.3f})</td></tr>")
    head = "".join(f"<th class='{'now' if abs(g - HS.LEN_TOL) < 1e-9 else ''}'>"
                   f"{g:.2f}</th>" for g in TOLS)
    return (f"""<section><h2>{stem.replace('_',' ').title()}
<span class=sub>{n} mesures · {len(tr)} motifs retenus sur {tried} ancres essayées</span></h2>
<img src="data:image/png;base64,{img}">
<table><tr><th>ancre</th><th colspan={len(TOLS)}>longueur retenue selon TOL</th>
<th rowspan=2>meilleure<br>longueur</th></tr><tr><th></th>{head}</tr>{rows}</table>
</section>""", tr)


def main():
    stems = sys.argv[1:] or SHOWN
    body, shown_traces = "", []
    for st in stems:
        if not (HERE / f"docs/audio/{st}.m4a").exists():
            print(f"  !! {st} introuvable")
            continue
        try:
            html, tr = song_section(st)
        except Exception as exc:
            print(f"  !! {st} — {type(exc).__name__}: {exc}")
            continue
        body += html
        shown_traces += tr
        print(f"  ok {st}  ({len(tr)} ancres)")

    # l'effet corpus se mesure sur TOUTES les chansons, pas seulement celles
    # qui sont dessinées — sinon le réglage est choisi sur son propre échantillon
    allt = list(shown_traces)
    others = [p.stem for p in sorted((HERE / "docs/audio").glob("*.m4a"))
              if p.stem not in stems]
    for st in others:
        try:
            S, n, _ = load(st)
            tr = []
            HS.build_cells(S, n, trace=tr)
            allt += [t for t in tr if t["kept"]]
        except Exception:
            continue
    print(f"  corpus : {len(allt)} ancres au total")
    cimg, share, cost = corpus_fig(allt)

    out = HERE / "harmonia_min/state/reports/length_choice.html"
    out.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Choisir la longueur d'un motif</title><style>
*{{box-sizing:border-box;-webkit-tap-highlight-color:transparent}}
body{{margin:0;background:#e7e0d0;font:16px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1050px;margin:0 auto;padding:20px 12px calc(40px + env(safe-area-inset-bottom))}}
h1{{font:italic 600 24px Georgia,serif;margin:0 0 3px}}
.lede{{color:#6f6858;font-size:13.5px;line-height:1.55;margin-bottom:18px}}
.lede b{{color:{INK}}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:14px 13px;margin-bottom:16px}}
h2{{font:700 18px system-ui;margin:0 0 10px;color:{ACC}}}
.sub{{font:500 12px system-ui;color:#8a8371}}
img{{width:100%;border-radius:8px;display:block;margin-bottom:8px}}
table{{border-collapse:collapse;font-size:12.5px;width:100%}}
th,td{{border:1px solid #e5dcc6;padding:3px 8px;text-align:left}}
th{{background:#f7f3e9;font-size:11px}}
th.now,td.now{{background:#e4f0e8;font-weight:700}}
</style></head><body><div class=wrap>
<h1>Choisir la longueur d'un motif</h1>
<div class=lede>Pour chaque distance <b>d</b>, une seule question : « les <b>d</b>
mesures qui commencent ici reviennent-elles, mesure pour mesure, <b>d</b> mesures
plus loin ? ». Ça donne le profil dessiné ci-dessous, comparable d'une distance à
l'autre parce que c'est toujours un cosinus.<br><br>
La règle ensuite : <b>plancher à {HS.LEN_FLOOR:.2f}</b> (en dessous, rien ne
revient vraiment ici — trait pointillé), puis on garde tout ce qui est à
<b>TOL = {HS.LEN_TOL:.2f}</b> près de la meilleure (bande verte) et
<b>on prend la plus petite</b> (barre rouge).<br><br>
<b>C'est TOL qu'il faut arbitrer</b> — c'est lui qui dit à partir de quand on
« hésite ». Le tableau sous chaque morceau donne la longueur que chaque valeur de
TOL choisirait, ancre par ancre ; la figure du bas donne l'effet sur les
{len(allt)} ancres du corpus entier. Aujourd'hui, <b>{share:.0%} des ancres</b>
paient quelque chose pour la règle du plus petit, et l'écart médian consenti est
de <b>{np.median([c for c in cost if c > 0]) if any(c > 0 for c in cost) else 0:.3f}</b>.</div>

<section><h2>Effet sur tout le corpus <span class=sub>{len(allt)} ancres</span></h2>
<img src="data:image/png;base64,{cimg}">
<p class=sub>Le trait noir vertical est le réglage actuel. À gauche : plus TOL est
grand, plus souvent plusieurs longueurs sont à égalité — donc plus souvent la
règle du plus petit tranche. Au milieu : la longueur médiane retenue s'effondre
quand TOL grandit. À droite : ce que la règle coûte réellement, en écart de score
entre la meilleure longueur et celle qu'on garde.</p></section>
{body}</div></body></html>""")
    print(f"wrote {out.relative_to(HERE)} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
