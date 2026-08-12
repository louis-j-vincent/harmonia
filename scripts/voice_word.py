"""Le mot des accords et le mot de la VOIX, côte à côte — et ce qu'ils tranchent.

    .venv/bin/python scripts/voice_word.py [<stem> ...]
        -> /plots/voice_word.html + une page par morceau (audio + tête de lecture)

Louis, 2026-08-12 : « fais-moi aussi les bi-barres issues de la matrice SSM de la
voix, et montre-moi les bi-barres de voix qui concordent avec les bi-barres de la
matrice SSM classique (accords) qu'on utilise pour les bigrammes — une visu pour
que je puisse voir si on peut trancher avec cela. »

POURQUOI C'EST LA BONNE QUESTION. Le problème qui bloque depuis le début de ce
chantier est que **deux sections identiques collées sont indiscernables par
l'harmonie** : un A de 8 mesures et deux A de 4 mesures ont exactement la même
matrice harmonique. La voix, elle, ne rejoue pas la même mélodie sur le deuxième —
c'est le seul indice qui puisse trancher, et il n'était jamais mis en face du
mot des accords.

CE QUI A ÉTÉ ESSAYÉ ET QUI NE MARCHE PAS — et c'est le résultat du jour. Fabriquer
un MOT de la voix, exactement comme celui des accords, ne donne rien : à tous les
seuils testés, This Love sort 18 à 21 lettres pour 40 bi-mesures (7 côté accords)
et Blue Lights 18 pour 42 (3 côté accords). Chaque bi-mesure chantée est unique.
C'est musicalement juste : une mélodie ne se rejoue pas note pour note d'un
couplet à l'autre, une grille d'accords si. **La voix ne sait pas dire « c'est la
même chose ».** Le mot est quand même affiché sur la page, pour que ça se voie.

CE QU'ELLE SAIT DIRE, et qu'on lit ici : « ça continue » ou « ça repart ». La
bande VOIX donne la ressemblance de chaque bi-mesure à la précédente ; sous le
premier tiers du morceau (trait pointillé), la voix repart.

LA BANDE DES JONCTIONS, entre deux bi-mesures voisines :

  * **trait noir plein** — les deux changent : frontière franche, rien à
    arbitrer ;
  * **trait bleu** — la basse seule change : la boucle harmonique tourne,
    la voix continue sa phrase. C'est typiquement un milieu de section ;
  * **trait doré** — la voix seule change : MÊME harmonie, NOUVELLE mélodie. Le
    cas qui nous manquait — deux sections identiques collées se distinguent là,
    et nulle part ailleurs ;
  * **rien** — ni l'un ni l'autre : on est à l'intérieur d'une section.

À REGARDER : les traits dorés tombent-ils sur les frontières de Louis que
l'harmonie seule rate ? Si oui, la règle d'arrêt du merging s'écrit toute seule —
« ne jamais souder par-dessus un changement de voix ».
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

from ssm_zoo import SONGS, AUDIO, GT_LINE, gt_sections     # noqa: E402
from peak_profile import page, fused_profile               # noqa: E402
from hard_prior_sections import colourmap                  # noqa: E402
from change_curves import load_curves, draw_curves         # noqa: E402
from vote_fill import (fill, fig2b64_fixed, SYMCOL, SYM,   # noqa: E402
                       PLOT_L, PLOT_R, INK, HARD)
import bpe_lab as BP                                       # noqa: E402
import order_bundle                                        # noqa: E402
import order_lab as OL                                     # noqa: E402

OUTDIR = HERE / "docs" / "plots"
ACC = "#2f7dbd"      # les accords seuls changent
VOI = "#c07a1e"      # la voix seule change
MUET_MIN = 0.6       # au-delà de tant de silence, la bi-mesure n'a pas de voix
VOIX_THR = 0.70      # le seuil de groupage des bi-mesures de voix
                     #
                     # LA MATRICE COMPTE PLUS QUE LE SEUIL. Louis, 2026-08-12 :
                     # « il y a quand même des répétitions dans les chansons, je
                     # suis surpris que tu n'en trouves pas, montre-moi la matrice
                     # SSM de la voix. » Il avait raison. La première version
                     # comparait les bi-mesures sur la matrice de voix du zoo, qui
                     # est en DEMI-MESURES : un demi-mesure de chant tient deux ou
                     # trois notes, son profil de 12 hauteurs est presque vide, et
                     # deux cases voisines se ressemblent au hasard. Résultat, une
                     # matrice mouchetée où rien ne se groupe. La matrice de la
                     # PROD (`melody_bars`, par mesure) est la même chaîne agrégée
                     # une mesure entière : elle est lisse et ses blocs se voient
                     # à l'œil (le pont de This Love, mesures 49-56, saute aux
                     # yeux). Les trois matrices sont sur la page pour comparer.


def bibar_from(S, spans, res: int, thr=VOIX_THR, muets=None):
    """(mot, sim) — une lettre par bi-mesure, depuis N'IMPORTE quelle matrice.

    `res` est le nombre de cases de la matrice par mesure (1 = par mesure,
    2 = par demi-mesure). Les bi-mesures listées dans `muets` reçoivent `·` et
    sont exclues du groupage : une bi-mesure sans chant ne ressemble à rien, et
    la laisser voter ferait un faux groupe « silence » qui absorberait tout.
    """
    J = len(spans)
    muets = set(muets or ())
    D = np.zeros((J, J))
    for i, (p, _) in enumerate(spans):
        for k, (q, _) in enumerate(spans):
            a, b = p * res, q * res
            w = 2 * res
            v = [S[a + t, b + t] for t in range(w)
                 if a + t < S.shape[0] and b + t < S.shape[0]]
            D[i, k] = float(np.mean(v)) if v else 0.0
    d = np.sqrt(np.clip(np.diag(D), 1e-9, None))
    D = D / np.outer(d, d)

    vivants = [j for j in range(J) if j not in muets]
    used, lab, k = set(), [-1] * J, 0
    for j in sorted(vivants, key=lambda j: -(D[j, vivants] >= thr).sum()):
        if j in used:
            continue
        mem = [j]
        for q in vivants:
            if q in used or q == j:
                continue
            if all(D[q, x] >= thr for x in mem):
                mem.append(q)
        for q in mem:
            lab[q] = k
        used.update(mem)
        k += 1
    ren, k = {}, 0
    for j in range(J):
        if lab[j] >= 0 and lab[j] not in ren:
            ren[lab[j]] = k; k += 1
    mot = "".join("·" if lab[j] < 0 else SYM[ren[lab[j]] % 26] for j in range(J))
    return mot, D


VOIX_Q = 0.30      # sous ce quantile des continuités, la voix « change »


def continuite(D, muets, J):
    """[(j, valeur | None)] — à quel point la voix de la bi-mesure j prolonge
    celle de j−1. `None` quand l'une des deux est muette.

    POURQUOI PAS DES LETTRES, comme pour les accords. Essayé, et c'est le
    résultat de la journée : **le mot de la voix ne se groupe jamais**. À tous
    les seuils, This Love sort 18 à 21 lettres pour 40 bi-mesures contre 7 pour
    les accords ; Blue Lights 18 contre 3. C'est musicalement juste — une mélodie
    ne se rejoue pas note pour note d'un couplet à l'autre, alors qu'une grille
    d'accords, si. La voix ne sait donc pas dire « c'est la même chose » ; elle
    sait dire « ça continue » ou « ça repart », et c'est ça qu'on lit ici.
    """
    out = []
    for j in range(1, J):
        out.append((j, None if (j in muets or j - 1 in muets)
                    else float(D[j - 1, j])))
    return out


def jonctions(acc, cont, seuil):
    """[(j, cas)] pour chaque jonction. `cas` ∈ {2, 'a', 'v', 0}.

    La voix « repart » quand sa continuité est sous le seuil ET qu'elle est plus
    basse que ses deux voisines. Le creux local est indispensable : un seuil par
    quantile marque mécaniquement 30 % des jonctions, donc du bruit garanti au
    milieu des sections. Un creux, lui, est un événement.
    """
    val = {j: v for j, v in cont}
    out = []
    for j, v in cont:
        ca = acc[j] != acc[j - 1]
        voisines = [val.get(j - 1), val.get(j + 1)]
        cv = (v is not None and v <= seuil
              and all(w is None or v <= w for w in voisines))
        out.append((j, 2 if (ca and cv) else "a" if ca else "v" if cv else 0))
    return out


def matrices_png(Szoo, M, Sacc, n, gtb) -> str:
    """Les trois matrices côte à côte, sur le même axe de mesures."""
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("h", ["#faf6ec", "#9fc0d4", "#1d4d69"])
    fig, axs = plt.subplots(1, 3, figsize=(13.0, 4.6), facecolor="#fffdf6",
                            gridspec_kw={"wspace": 0.14})
    for ax, S, lab in ((axs[0], Szoo, "voix — par DEMI-mesure"),
                       (axs[1], M, "voix — par mesure (prod)"),
                       (axs[2], Sacc, "harmonie — par mesure")):
        S = np.nan_to_num(np.asarray(S, float))
        ax.imshow(S, cmap=cmap, vmin=0, vmax=1, extent=[0, n, n, 0],
                  interpolation="nearest")
        for g in gtb:
            ax.axvline(g, color=GT_LINE, lw=0.7, alpha=0.8)
            ax.axhline(g, color=GT_LINE, lw=0.7, alpha=0.8)
        step = 16 if n <= 120 else 32
        ax.set_xticks(np.arange(0, n + 1, step)); ax.set_yticks(np.arange(0, n + 1, step))
        ax.set_xticklabels([str(i + 1) for i in np.arange(0, n + 1, step)], fontsize=7.5)
        ax.set_yticklabels([str(i + 1) for i in np.arange(0, n + 1, step)], fontsize=7.5)
        ax.tick_params(length=2, colors="#8a8371")
        ax.set_title(lab, fontsize=10, color="#4a4438", pad=6)
        for sp in ax.spines.values():
            sp.set_color("#e0d7c2")
    fig.subplots_adjust(left=0.05, right=0.99, top=0.92, bottom=0.08)
    return fig2b64_fixed(fig)


def song_page(stem: str, title: str) -> str:
    b = order_bundle.get(stem)
    n, grid = b["n"], b["grid"]
    R = fill(b, stem)
    spans, x0, word = R["spans"], R["x0"], R["mot"]
    C = load_curves(stem, n)
    _P, _k, _r, lines, _n, extra = fused_profile(stem)
    Szoo = next(d["S"] for d in lines if d["nom"] == "voix")
    M = np.nan_to_num(np.asarray(b["M"], float))     # la matrice de la PROD
    mute = np.asarray(b["mute"], bool)
    hb = C["hb"]

    muets = {j for j, (p, q) in enumerate(spans)
             if mute[p:q].mean() >= MUET_MIN}
    voi, Dv = bibar_from(M, spans, 1, muets=muets)
    cont = continuite(Dv, muets, len(spans))
    vals = [v for _j, v in cont if v is not None]
    seuil = float(np.quantile(vals, VOIX_Q)) if vals else 0.0
    jj = jonctions(word, cont, seuil)

    st = BP.merges(word)
    ents = BP.entites(st, x0, n, BP.arret(st, R["cuts"]))
    today = OL.new_sections(b)[0]
    gt = gt_sections(stem)
    gtb = [s["b0"] for s in gt["sections"][1:]] if gt else []
    strips = [("les entités", ents), ("ce qu'on écrit", today)]
    if gt:
        strips.append(("toi", gt["sections"]))

    rows = 4 + len(strips)              # courbes · accords · voix · jonctions
    fig, axs = plt.subplots(
        rows, 1, figsize=(13.0, 3.9 + 0.62 * len(strips)), facecolor="#fffdf6",
        gridspec_kw={"height_ratios": [5.0, 0.95, 0.95, 1.1] + [1.15] * len(strips),
                     "hspace": 0.0})

    def deco(ax, lab, colour=INK, last=False, gtl=True):
        ax.set_xlim(0, n); ax.set_ylim(0, 1); ax.set_yticks([])
        ax.set_ylabel(lab, rotation=0, ha="right", va="center", fontsize=9.5,
                      color=colour)
        for s in ax.spines.values():
            s.set_visible(False)
        if gtl:
            for g in gtb:
                ax.axvline(g, color=GT_LINE, lw=0.8, alpha=0.7, zorder=7)
        if last:
            step = 8 if n <= 120 else 16
            ax.set_xticks(np.arange(0, n + 1, step))
            ax.set_xticklabels([str(i + 1) for i in np.arange(0, n + 1, step)],
                               fontsize=8, color="#8a8371")
            ax.tick_params(length=2, colors="#c9c0aa")
        else:
            ax.set_xticks([])

    deco(axs[0], "profils de\nchangement", INK)
    draw_curves(axs[0], C, n)

    ax = axs[1]
    for j, ch in enumerate(word):
        ax.add_patch(plt.Rectangle(
            (x0[j], 0.15), x0[j + 1] - x0[j], 0.7,
            facecolor=SYMCOL[(ord(ch) - 97) % len(SYMCOL)],
            edgecolor="#fffdf6", lw=0.9))
        if n <= 120:
            ax.text((x0[j] + x0[j + 1]) / 2, 0.5, ch, ha="center", va="center",
                    fontsize=7.5, color="#4a4438")
    deco(ax, "mot BASSE", "#4a4438")

    # la VOIX : sa continuité d'une bi-mesure à la suivante, pas ses lettres
    ax = axs[2]
    for j in muets:
        ax.axvspan(x0[j], x0[j + 1], color="#f2e3c4", lw=0, zorder=0)
    for j, v in cont:
        if v is None:
            continue
        bas = v <= seuil
        ax.plot([x0[j], x0[j]], [0.08, 0.08 + 0.84 * max(0.0, min(1.0, v))],
                color=VOI if bas else "#e0cba6", lw=3.0, solid_capstyle="butt",
                zorder=3)
    ax.axhline(0.08 + 0.84 * seuil, color=VOI, lw=0.8, ls=(0, (3, 3)), alpha=0.8)
    deco(ax, "voix : continuité\n(doré = ça repart)", "#8a6d1f")

    ax = axs[3]
    for j, cas in jj:
        if not cas:
            continue
        col = INK if cas == 2 else (ACC if cas == "a" else VOI)
        ax.plot([x0[j], x0[j]], [0.08, 0.92], color=col,
                lw=3.4 if cas == 2 else 2.6, solid_capstyle="butt", zorder=3)
    deco(ax, "jonctions", "#4a4438")

    cm = colourmap()
    for k, (ax, (lab, ss)) in enumerate(zip(axs[4:], strips)):
        for s in ss:
            w = s["b1"] - s["b0"] + 1
            ax.add_patch(plt.Rectangle((s["b0"], 0.08), w, 0.84,
                                       facecolor=cm(s["label"]),
                                       edgecolor="#fffdf6", lw=1.1))
            if w >= max(4, n * 0.04):
                ax.text(s["b0"] + w / 2, 0.5, str(s["label"]), ha="center",
                        va="center", fontsize=8.5, color=INK)
        deco(ax, lab, INK, last=(k == len(strips) - 1))

    fig.subplots_adjust(left=PLOT_L, right=PLOT_R, top=0.99, bottom=0.07)
    img = fig2b64_fixed(fig)
    mats = matrices_png(Szoo, M, b["S"], n, gtb)

    nvoi = len(set(voi) - {"·"})
    dore = [x0[j] for j, c in jj if c == "v"]
    btns = "".join(
        f'<button class=blk data-p="[{max(0, p - 2)},{min(n, p + 2)}]">mes. {p + 1}'
        "</button>" for p in dore) or "<span class=hint>aucune</span>"

    body = f"""<section><h2>{title} <span class=sub>{n} mesures</span></h2>
<div class=plot><img src="data:image/png;base64,{img}">
<div class=cur></div><div class=hit></div></div>
<div class=bar><button class=pp>▶</button><span class=pos>mes. 1 · 0:00</span>
<span class=hint>touche le graphique pour te déplacer</span></div>
<div class=lane><span class=lab>voix seule</span>{btns}</div>
<img src="data:image/png;base64,{mats}" alt="matrices de voix">
<div class=votes><b>Les trois matrices, mêmes axes de mesures.</b> À gauche la
voix en <b>demi-mesures</b> : mouchetée, presque vide — un demi-mesure de chant
ne tient que deux ou trois notes, son profil de hauteurs est trop maigre pour que
deux cases se ressemblent autrement que par hasard. Au milieu la même chaîne
agrégée <b>par mesure</b> (celle de la prod) : lisse, et ses blocs se voient (le
pont de This Love, mesures 49-56). À droite les accords, pour comparer. C'est la
matrice du milieu qui sert maintenant à construire le mot de la voix.</div>
<div class=votes><b>mot BASSE —</b> <code>{word}</code><br>
<b>mot VOIX —</b> <code>{voi}</code> — {nvoi} lettres pour
{len(word)} bi-mesures, contre {len(set(word))} côté accords : <b>le mot de la
voix ne se groupe pas</b>. Une mélodie ne se rejoue pas note pour note d'un
couplet à l'autre, une grille d'accords si. C'est pourquoi on lit la voix en
CONTINUITÉ (« ça continue » / « ça repart ») et non en lettres.</div>
</section>
<audio id=au preload=metadata playsinline src="/audio/{stem}.m4a"></audio>
<script>window.GRID={[round(t, 3) for t in grid]};
window.PLOT=[{PLOT_L},{PLOT_R}]; window.U=1;</script>"""
    return page(f"{title} — accords contre voix", body, back=True,
                lede_html=LEDE, back_href="voice_word.html",
                back_label="tous les morceaux")


LEDE = f"""<div class=lede>Deux mots sur la même grille de bi-mesures : celui de la
<b>basse</b> (celui qu'on utilise pour les bigrammes) et celui de la
<b>voix</b>. Une bi-mesure où personne ne chante porte <code>·</code> — un trou,
pas une lettre.<br><br>
La bande <b>jonctions</b> compare les deux, entre chaque paire de bi-mesures
voisines :<br>
<b style="color:#1c1c1c">▮ trait noir</b> — les deux changent : frontière
franche.<br>
<b style="color:{ACC}">▮ trait bleu</b> — la basse seule change : la boucle
harmonique tourne, la voix poursuit sa phrase. En général un milieu de
section.<br>
<b style="color:{VOI}">▮ trait doré</b> — <b>la voix seule change</b> : même
harmonie, nouvelle mélodie. C'est le cas qui nous manquait — deux sections
identiques collées ne se distinguent QUE là.<br>
rien — on est à l'intérieur d'une section.<br><br>
Traits rouges : tes frontières. <b>La question à trancher :</b> les traits dorés
tombent-ils sur celles que l'harmonie seule rate ? Touche-en un pour l'écouter.
</div>"""


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    todo = [(s, t) for s, t in SONGS if s in args] or (
        [(s, s) for s in args] if args else list(SONGS))
    rows = []
    for stem, title in todo:
        if not (AUDIO / f"{stem}.m4a").exists():
            continue
        (OUTDIR / f"voix_{stem}.html").write_text(song_page(stem, title))
        rows.append(f'<tr><td><a href="voix_{stem}.html">{title}</a></td></tr>')
        print(f"  ok {title}")
    (OUTDIR / "voice_word.html").write_text(page(
        "Accords contre voix",
        "<section><table class=idx>" + "".join(rows) + "</table></section>",
        lede_html=LEDE, back_href="voice_word.html"))
    print(f"wrote docs/plots/voice_word.html + {len(rows)} pages")


if __name__ == "__main__":
    main()
