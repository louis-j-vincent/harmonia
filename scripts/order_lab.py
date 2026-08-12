"""« Petites sections d'abord » : ce que ça change, morceau par morceau, à écouter.

    .venv/bin/python scripts/order_lab.py [<stem> ...]
        -> /plots/order_lab.html + une page par morceau (audio + tête de lecture)

Louis, 2026-08-12, sur Blue Lights : « clairement un B est bien détecté, et
ensuite il y a sûrement une hésitation après ce B entre est-ce qu'on passe un A
ou un autre B… il faudrait peut-être commencer par privilégier les petites
sections puis les grandes, donc 4 d'abord puis 8 après. »

CE QUI A ÉTÉ MESURÉ, ET POURQUOI L'INVERSION SIMPLE ÉCHOUE. Lancer la passe de 4
avant celle de 8 rend 0,566 de médiane contre 0,789. Ce n'est PAS un problème de
seuil : les scores de bloc à 4 et à 8 mesures vivent sur la même échelle (médiane
0,36 contre 0,35 ; 4,2 % contre 3,2 % de positions au-dessus du seuil), parce que
`block_score` divise déjà par le score de l'ancre. La passe de 4 lancée en
premier mesure la **période de la boucle harmonique**, pas l'**échelle de la
section** : elle découpe les A de 8 et 12 mesures en tranches de 4 qui reçoivent
chacune une lettre différente.

CE QUI MARCHE. Un seul parcours du curseur, où les deux longueurs sont proposées
à chaque ancre :

  * le bloc de 8 garde la priorité quand il existe ;
  * le bloc de 4 ne passe devant QUE là où aucun 8 n'est possible (un pic dur le
    barre, ou il n'a pas de reprise) ET que ses reprises valent ≥ 0,88 de l'ancre
    — c'est-à-dire qu'il se rejoue à l'IDENTIQUE. À 0,75, c'est la boucle du
    morceau, et elle attend son tour ;
  * une coupure sur pic qui laisserait une section d'UNE mesure est annulée (les
    pics ne sont justes qu'à ±1 mesure ; la recherche le savait déjà, l'écriture
    non).

Le gain vient d'un seul mécanisme, visible sur les runs de Blue Lights : la passe
de 8 lancée seule arrive mesure 45 et **vole la 3e occurrence de la famille B**,
si bien que la passe de 4 qui suit ne retrouve que 13/29/61. Dans le parcours
unique, le bloc de 4 de la mesure 13 est proposé pendant que sa famille est
encore entière et prend 13/29/45/61/77 d'un coup.

Médiane sur les douze annotés : **0,805** contre 0,789 (moyenne 0,805 contre
0,786), Blue Lights **0,813** contre 0,696, et aucun morceau en recul. Six
morceaux annotés tenus hors de la conception : moyenne 0,636 contre 0,620.
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

from ssm_zoo import SONGS, AUDIO, gt_sections, fig2b64          # noqa: E402
from peak_profile import page, PLOT_L, PLOT_R                   # noqa: E402
from hard_prior_sections import _strip, colourmap, score        # noqa: E402
import order_bundle                                             # noqa: E402
import order_search as OS                                       # noqa: E402
import order_multi as MU                                        # noqa: E402

OUTDIR = HERE / "docs" / "plots"

LITERAL = 0.88     # plateau mesuré 0,86–1,02 ; l'un-contre-tous choisit 0,88
STAT = "moy"       # sur 11 replis sur 12
MIN_CUT = 2        # une coupure ne peut pas laisser une section d'1 mesure
LENGTHS = (8, 4)   # 12 et 16 mesurés et rejetés (médiane 0,630 et 0,715)

# Les six morceaux annotés qui n'ont servi à AUCUNE décision de conception.
HOLDOUT = [
    ("bruno_mars_the_lazy_song_official_music_video", "The Lazy Song"),
    ("pharrell_williams_happy_official_video", "Happy"),
    ("the_commodores_easy_1977", "Easy"),
    ("the_jackson_5_abc", "ABC"),
    ("the_ronettes_be_my_baby_music_video", "Be My Baby"),
    ("yesterday_remastered_2009", "Yesterday"),
]


def new_sections(b):
    runs, _ = MU.multi_runs(b, b["hard"], lengths=LENGTHS, literal=LITERAL,
                            stat=STAT)
    return OS.assemble(b, runs, b["hard"], tail_unit=4, min_cut=MIN_CUT), runs


def old_sections(b):
    runs, _ = OS._passes(b, b["hard"], [(8, 0.66), (4, 0.70)])
    return OS.assemble(b, runs, b["hard"], tail_unit=4), runs


def song_page(stem: str, title: str) -> str:
    b = order_bundle.get(stem)
    n, grid = b["n"], b["grid"]
    new, runs_new = new_sections(b)
    old, runs_old = old_sections(b)
    gt = gt_sections(stem)
    theirs = gt["sections"] if gt else None

    strips = [("petites d'abord", new), ("la prod", old)]
    if theirs:
        strips.append(("toi", theirs))

    fig, axs = plt.subplots(len(strips), 1,
                            figsize=(12.6, 0.62 * len(strips) + 0.5),
                            facecolor="#fffdf6", gridspec_kw={"hspace": 0.45})
    cm = colourmap()
    for ax, (lab, secs) in zip(np.atleast_1d(axs), strips):
        _strip(ax, secs, n, cm)
        for h in b["hard"]:
            ax.axvline(h, color="#0d2437", lw=2.0)
        ax.set_xlim(0, n); ax.set_ylim(0, 1)
        ax.set_yticks([]); ax.set_xticks([])
        ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=10)
        for s in ax.spines.values():
            s.set_visible(False)
    img = fig2b64(fig)

    btns = "".join(
        f'<button class=blk data-p="[{s["b0"]},{s["b1"] + 1}]">{s["label"]}'
        f'<small>mes. {s["b0"] + 1}–{s["b1"] + 1}</small></button>' for s in new)

    def runs_txt(rr):
        return " · ".join(f"{r['block']}@{r['b0'] + 1}"
                          f"→{','.join(str(o + 1) for o in r['occ'])}" for r in rr)

    lines = [f"<b>blocs posés — petites d'abord :</b> {runs_txt(runs_new)}",
             f"<b>blocs posés — la prod :</b> {runs_txt(runs_old)}"]
    if theirs:
        a, c = score(new, theirs, n), score(old, theirs, n)
        arrow = ("gagne" if a > c + 0.005 else
                 ("perd" if a < c - 0.005 else "égalité"))
        lines.insert(0, f"contre ton découpage : <b>petites d'abord {a:.3f}</b> · "
                        f"prod {c:.3f} — <b>{arrow}</b> ici")
    verdict = "<br>".join(lines)

    body = f"""<section><h2>{title} <span class=sub>{n} mesures</span></h2>
<div class=plot><img src="data:image/png;base64,{img}">
<div class=cur></div><div class=hit></div></div>
<div class=bar><button class=pp>▶</button><span class=pos>mes. 1 · 0:00</span>
<span class=hint>touche le graphique pour te déplacer</span></div>
<div class=lane><span class=lab>écouter</span>{btns}</div>
<div class=verdict>{verdict}</div></section>
<audio id=au preload=metadata playsinline src="/audio/{stem}.m4a"></audio>
<script>window.GRID={[round(t, 3) for t in grid]};
window.PLOT=[{PLOT_L},{PLOT_R}]; window.U=1;</script>"""
    return page(f"{title} — petites sections d'abord", body, back=True,
                lede_html=LEDE, back_href="order_lab.html")


LEDE = """<div class=lede>Trois bandes : <b>petites d'abord</b> (les deux
longueurs proposées à chaque ancre, le bloc de 4 passant devant quand il se
rejoue à l'identique), <b>la prod</b> (les 8 d'abord, les 4 ensuite), et
<b>toi</b>. Les traits noirs sont les pics durs.<br><br>
<b>L'inversion simple ne marche pas</b> : lancer la passe de 4 avant celle de 8
donne 0,566 de médiane contre 0,789 — et ce n'est pas une histoire de seuil, les
deux longueurs ont exactement la même échelle de score. La passe de 4 lancée en
premier mesure la <b>période de la boucle</b>, pas l'<b>échelle de la
section</b>.<br><br>
<b>Ce qui marche</b> : un seul parcours, le 8 prioritaire, et le 4 qui passe
devant seulement là où aucun 8 n'est possible et où ses reprises valent ≥ 0,88 de
l'ancre. Sur Blue Lights c'est ce qui donne tes cinq B de 4 mesures d'un coup —
la passe de 8, lancée seule, arrivait mesure 45 et volait le troisième.<br><br>
<b>Touche une section pour l'écouter</b> : le jugement qui compte est à
l'oreille.</div>"""


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    todo = list(SONGS) + HOLDOUT
    if argv:
        todo = [(s, t) for s, t in todo if s in argv] or [(s, s) for s in argv]
    rows, sc_new, sc_old = [], {}, {}
    for stem, title in todo:
        if not (AUDIO / f"{stem}.m4a").exists():
            continue
        (OUTDIR / f"order_{stem}.html").write_text(song_page(stem, title))
        b = order_bundle.get(stem)
        gt = gt_sections(stem)
        if gt:
            new, _ = new_sections(b)
            old, _ = old_sections(b)
            sc_new[stem] = score(new, gt["sections"], b["n"])
            sc_old[stem] = score(old, gt["sections"], b["n"])
        hold = " <span class=hint>(hors conception)</span>" if any(
            stem == h for h, _ in HOLDOUT) else ""
        d = sc_new.get(stem, 0) - sc_old.get(stem, 0)
        col = "#2c7a4b" if d > 0.005 else ("#8a2b2b" if d < -0.005 else "#8a8371")
        rows.append(
            f'<tr><td><a href="order_{stem}.html">{title}</a>{hold}</td>'
            f'<td>{sc_old.get(stem, float("nan")):.3f}</td>'
            f'<td><b>{sc_new.get(stem, float("nan")):.3f}</b></td>'
            f'<td style="color:{col}">{d:+.3f}</td></tr>')
        print(f"  ok {title}")

    zoo = [s for s, _ in SONGS if s in sc_new]
    hol = [s for s, _ in HOLDOUT if s in sc_new]

    def line(lab, ss):
        return (f"<tr><td><b>{lab}</b></td>"
                f"<td>{np.median([sc_old[s] for s in ss]):.3f}</td>"
                f"<td><b>{np.median([sc_new[s] for s in ss]):.3f}</b></td>"
                f"<td>moyenne {np.mean([sc_old[s] for s in ss]):.3f} → "
                f"{np.mean([sc_new[s] for s in ss]):.3f}</td></tr>")

    (OUTDIR / "order_lab.html").write_text(page(
        "Petites sections d'abord",
        "<section><table class=idx>"
        "<tr><td>morceau</td><td>prod</td><td>petites d'abord</td><td>écart</td></tr>"
        + "".join(rows) + line("médiane des 12", zoo)
        + (line("médiane des 6 hors conception", hol) if hol else "")
        + "</table></section>", lede_html=LEDE))
    print(f"wrote docs/plots/order_lab.html + {len(rows)} pages")


if __name__ == "__main__":
    main()
