"""Le rythme dit-il où les sections changent ? — la courbe de nouveauté, à voir.

    python scripts/rhythm_boundaries.py [<stem> ...]
      -> /reports/rhythm_boundaries.html   (port 7772)

Louis, 2026-08-05 : « la matrice ssm rythmique cosinus apporte une belle info
complémentaire, il faut juste savoir comment l'exploiter. D'abord commence par
l'agréger à l'échelle 1 ou 2 barres et tu me montres ce que ça veut pour
détecter les changements de section. »

La page ne tranche rien. Elle met sur le MÊME axe de mesures, chanson par
chanson :

  1. la matrice rythmique au cosinus, mesure contre mesure ;
  2. la même agrégée à DEUX mesures ;
  3. la matrice harmonique, celle qu'on utilise aujourd'hui ;
  4. la courbe de nouveauté de Foote lue sur le rythme (1 mesure, puis
     2 mesures), à trois largeurs de noyau ;
  5. la même courbe lue sur l'harmonie, juste en dessous, pour comparer à l'œil ;
  6. les sections écrites aujourd'hui, en traits verticaux sur TOUT.

Les features de batterie viennent telles quelles de `scripts/rhythm_vs_harmony.py`
(stem demucs -> 3 bandes × 16 doubles-croches par mesure) : rien n'est réinventé
ici, seule la lecture change.

Foote 2000, « Automatic audio segmentation using a measure of audio novelty »,
IEEE ICME — https://doi.org/10.1109/ICME.2000.869637
"""
from __future__ import annotations

import json
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
sys.path.insert(0, str(HERE / "scratchpad"))
from pattern_lanes import load, fig2b64_fixed, PLOT_L, PLOT_R      # noqa: E402
from rhythm_vs_harmony import bar_patches                          # noqa: E402
import harmonia_min.harmonic_sections as HS                        # noqa: E402

INK, ACC = "#1c1c1c", "#8a2b2b"
MUTED = "#6f6858"
# largeur du noyau = nombre de mesures comparées DE CHAQUE CÔTÉ de la frontière
HALFWIDTHS = (2, 4, 8)
RCOL = ["#e0a49c", "#cf5a4c", "#8a2b2b"]     # rythme, du plus étroit au plus large
HCOL = ["#a8c6e2", "#4a8cc4", "#1f4e79"]     # harmonie, idem
NEAR = 2            # « en face » d'une frontière écrite = à ± NEAR mesures

# Une phrase par chanson, écrite en regardant la figure — pas de verdict, ce que
# l'œil voit. Les mesures citées sont les pics de la largeur ± 4 mesures.
NOTES = {
 "bein_green":
   "Il n'y a pas de batterie. La matrice rythmique est bleue de bout en bout — "
   "aucune mesure ne ressemble à une autre — et la courbe qui en sort n'est que "
   "du bruit. C'est le cas à retenir : cette lane suppose une batterie.",
 "maroon_5_she_will_be_loved_official_music_video":
   "À deux mesures, la matrice rythmique se découpe en grands blocs "
   "d'ARRANGEMENT que l'harmonie ne montre pas : une intro dépouillée, puis un "
   "bloc dense, puis un autre. Les gros pics du rythme sont des changements de "
   "texture, pas d'accords — les boutons du bas les font écouter.",
 "maroon_5_this_love":
   "Les deux lanes se répartissent le travail : le rythme marque proprement les "
   "grandes frontières du début (16, 36) et reste calme à l'intérieur des "
   "sections, l'harmonie s'agite surtout dans la seconde moitié (44–56).",
 "the_police_every_breath_you_take_official_music_video":
   "L'harmonie bouge partout ; le rythme reste PLAT pendant tout le premier A "
   "puis monte d'un coup à l'entrée du B, et il repère les breaks de batterie "
   "(~30, ~90–96) qui ne changent aucun accord. En face, ni l'une ni l'autre "
   "lane ne retrouve les frontières écrites mieux que le hasard.",
 "mayer_hawthorne_the_walk":
   "Le cas le plus net. L'harmonie est une boucle de deux mesures d'un bout à "
   "l'autre : sa courbe de nouveauté est quasi PLATE jusqu'à la mesure 48. Le "
   "rythme, lui, a des pics francs bien avant. C'est là que la lane rythmique "
   "apporte de l'information qui n'existe pas dans l'harmonie.",
 "bruno_mars_grenade_official_music_video":
   "Le rythme marque l'entrée de la batterie, puis des repères espacés d'une "
   "douzaine de mesures. C'est ici que l'agrégation à deux mesures aide le "
   "plus : elle passe de 10 à 15 frontières touchées sur 17, contre 10 "
   "attendues au hasard.",
}
DEFAULT = ["bein_green",
           "maroon_5_she_will_be_loved_official_music_video",
           "maroon_5_this_love",
           "the_police_every_breath_you_take_official_music_video",
           "mayer_hawthorne_the_walk",
           "bruno_mars_grenade_official_music_video"]


# ── agrégation à deux mesures ──────────────────────────────────────────────
def cosine_ssm(V: np.ndarray) -> np.ndarray:
    """Cosinus sur des motifs aplatis, non négatifs -> déjà dans [0,1]."""
    X = V.reshape(len(V), -1).astype(np.float64)
    X = X / np.clip(np.linalg.norm(X, axis=1, keepdims=True), 1e-9, None)
    return np.clip(X @ X.T, 0, 1)


def pair_bars(P: np.ndarray) -> np.ndarray:
    """(n,3,16) -> (n//2, 3, 32) : deux mesures CONCATÉNÉES, pas moyennées.

    Moyenner les deux mesures effacerait justement ce qui distingue un groove
    de deux mesures de sa propre première moitié (la relance de la mesure 2,
    le fill). Concaténer garde l'ordre : la case compare « les deux mesures
    telles qu'elles se jouent » à « les deux mesures d'ailleurs ».

    Ce que ça ne règle PAS (règle #4) : la PHASE. Les mesures sont appariées
    0-1, 2-3, … ; si l'hypermesure de deux temps commence sur une mesure
    impaire, chaque paire est à cheval sur deux motifs. Rien ici ne le détecte.
    """
    m = len(P) // 2
    return P[:2 * m].reshape(m, 2, P.shape[1], P.shape[2]) \
                    .transpose(0, 2, 1, 3).reshape(m, P.shape[1], 2 * P.shape[2])


def block_average(S: np.ndarray, k: int = 2) -> np.ndarray:
    """L'autre agrégation : moyenner la MATRICE par blocs k×k."""
    m = len(S) // k
    return S[:k * m, :k * m].reshape(m, k, m, k).mean(axis=(1, 3))


# ── le noyau de Foote ──────────────────────────────────────────────────────
def foote_kernel(M: int) -> np.ndarray:
    """Le damier : + sur les deux carrés « avant×avant » et « après×après »,
    − sur les deux carrés croisés, adouci par une gaussienne pour que les
    coins pèsent moins que le centre. Normalisé pour que deux largeurs
    différentes se lisent sur la même échelle."""
    K = np.ones((2 * M, 2 * M))
    K[:M, M:] = -1.0
    K[M:, :M] = -1.0
    g = np.exp(-0.5 * (np.linspace(-2.0, 2.0, 2 * M)) ** 2)
    K = K * np.outer(g, g)
    return K / np.abs(K).sum()


def novelty(S: np.ndarray, M: int) -> np.ndarray:
    """Nouveauté de Foote (2000) : le damier glissé le long de la diagonale.

    `out[i]` se lit « frontière juste AVANT la case i » : le noyau compare les
    M cases qui précèdent aux M cases qui suivent. Il est haut quand chacun des
    deux côtés se ressemble à l'intérieur ET que les deux ne se ressemblent pas
    entre eux — c'est exactement la définition musicale d'un changement de
    section. Les bords sont prolongés par recopie (`mode="edge"`), donc les
    M premières et M dernières cases sont à ignorer.
    """
    n = len(S)
    if n < 2 * M + 1:
        return np.zeros(n)
    K = foote_kernel(M)
    Pd = np.pad(S, M, mode="edge")
    return np.array([float((Pd[i:i + 2 * M, i:i + 2 * M] * K).sum())
                     for i in range(n)])


def unit(c: np.ndarray, edge: int = 0) -> np.ndarray:
    """Chaque courbe ramenée à son propre maximum — les largeurs de noyau ne
    produisent pas la même amplitude, seule la FORME est comparable.

    Le maximum est cherché à l'INTÉRIEUR seulement. Les `edge` premières et
    dernières cases sont un artefact du prolongement des bords : le noyau y
    compare la musique à sa propre recopie, ce qui donne un pic énorme mesure 1
    qui écrasait tout le reste de la courbe (vu sur This Love, harmonie).
    """
    core = c[edge:len(c) - edge] if edge and len(c) > 2 * edge else c
    lo, hi = float(np.min(core)), float(np.max(core))
    if hi <= lo:
        return np.zeros_like(c)
    return np.clip((c - lo) / (hi - lo), -0.05, 1.05)


def top_peaks(c: np.ndarray, k: int, edge: int, step: int = 1) -> list[int]:
    """Les k pics les plus hauts, en index de MESURE. Aucun seuil : k est
    imposé de l'extérieur (le nombre de frontières écrites), pour que rythme et
    harmonie soient jugés à nombre de propositions égal."""
    idx, _ = find_peaks(c)
    idx = [int(i) for i in idx if edge <= i < len(c) - edge]
    idx.sort(key=lambda i: -c[i])
    return sorted(i * step for i in idx[:max(0, k)])


# ── la figure ──────────────────────────────────────────────────────────────
def figure(n, S_h, S_r1, S_r2, nov_r1, nov_r2, nov_r2b, nov_h, secs,
           pk_r, pk_r2, pk_h, how):
    ratios = [2.8, 2.8, 2.8, 1.4, 1.4, 1.4, 0.72]
    H = sum(ratios) * 1.02 + 1.1
    fig, axs = plt.subplots(len(ratios), 1, sharex=True, figsize=(12.6, H),
                            gridspec_kw={"height_ratios": ratios, "hspace": 0.17})
    fig.subplots_adjust(left=PLOT_L, right=PLOT_R, top=1 - 0.34 / H, bottom=0.55 / H)
    cuts = [s["b0"] for s in secs if s["b0"] > 0]

    mats = ((S_r1, 1, f"RYTHME — cosinus, 1 mesure  ·  {how}", ACC),
            (S_r2, 2, "RYTHME — cosinus, 2 mesures (les deux mesures concaténées)", ACC),
            (S_h, 1, "HARMONIE — la matrice qu'on utilise aujourd'hui", "#1f4e79"))
    for ax, (M, step, name, col) in zip(axs[:3], mats):
        e = len(M) * step
        ax.imshow(M, origin="lower", extent=(0, e, 0, e), cmap="RdYlBu_r",
                  vmin=float(np.percentile(M, 5)), vmax=float(np.percentile(M, 99)),
                  aspect="auto", interpolation="nearest")
        ax.set_ylabel("mesure", fontsize=7.5)
        ax.set_ylim(0, n)
        ax.tick_params(labelsize=6.4)
        ax.set_title(name, fontsize=8.2, color=col, loc="left", pad=3)

    curves = (
        (axs[3], nov_r1, 1, RCOL, "NOUVEAUTÉ — rythme, matrice 1 mesure", ACC, pk_r),
        (axs[4], nov_r2, 2, RCOL, "NOUVEAUTÉ — rythme, matrice 2 mesures", ACC, pk_r2),
        (axs[5], nov_h, 1, HCOL, "NOUVEAUTÉ — harmonie", "#1f4e79", pk_h),
    )
    for ax, novs, step, cols, name, col, pk in curves:
        def draw(c, e, **kw):
            """Les `e` cases de chaque bord ne sont pas tracées : le noyau y
            compare la musique à la recopie du bord, pas à de la musique."""
            y = unit(c, e)
            y[:e] = np.nan
            y[len(y) - e:] = np.nan
            ax.plot(np.arange(len(c)) * step, y, **kw)

        for j, M in enumerate(HALFWIDTHS):
            draw(novs[M], max(1, M // step), color=cols[j],
                 lw=1.15 + 0.25 * j, label=f"± {M} mes.", zorder=3 + j)
        if novs is nov_r2 and nov_r2b is not None:
            draw(nov_r2b[HALFWIDTHS[1]], max(1, HALFWIDTHS[1] // 2),
                 color="#8a8371", lw=1.0, ls=(0, (3, 2)), zorder=2,
                 label="± 4 mes., matrice moyennée")
        if pk:
            ax.plot(pk, [1.11] * len(pk), marker="v", ls="none", ms=5.4,
                    color=col, clip_on=False, zorder=6)
        ax.set_ylim(-0.06, 1.10)
        ax.set_yticks([])
        ax.set_ylabel("nouveauté", fontsize=6.8, color=col)
        ax.legend(fontsize=5.8, loc="lower right", bbox_to_anchor=(1.0, 1.0),
                  ncol=4, frameon=False, handlelength=1.6, columnspacing=1.0,
                  borderpad=0.1)
        ax.set_title(name, fontsize=8.2, color=col, loc="left", pad=3)
        for sp in ax.spines.values():
            sp.set_color("#ddd5c0")

    ax = axs[-1]
    for s in secs:
        ax.add_patch(plt.Rectangle((s["b0"], .10), s["b1"] - s["b0"] + 1, .68,
                                   facecolor="#efe8d6", edgecolor="#b9b09a", lw=.7))
        ax.text((s["b0"] + s["b1"] + 1) / 2, .44, s["letter"], ha="center",
                va="center", fontsize=6.6, color=MUTED, fontweight="bold")
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel("sections\nécrites", fontsize=6.8, rotation=0, ha="right",
                  va="center", color=INK)
    ax.set_xlim(0, n)
    ax.set_xticks(range(0, n + 1, 4 if n <= 80 else 8))
    ax.tick_params(labelsize=6.5)
    ax.set_xlabel("mesure", fontsize=8)

    for ax in axs:                     # les frontières écrites, sur TOUT
        for b in cuts:
            ax.axvline(b, color="#111", lw=0.8, alpha=0.42, zorder=8)
    return fig2b64_fixed(fig)


# ── une chanson ────────────────────────────────────────────────────────────
def song(stem):
    S_h, n, grid = load(stem)
    cells, _ = HS.build_cells(S_h, n)
    secs = HS.sections_from(S_h, n, cells)
    P, how = bar_patches(stem, grid)
    n = min(n, len(P))
    S_h, P = S_h[:n, :n], P[:n]

    S_r1 = cosine_ssm(P)
    S_r2 = cosine_ssm(pair_bars(P))
    S_r2b = block_average(S_r1, 2)

    nov_r1 = {M: novelty(S_r1, M) for M in HALFWIDTHS}
    nov_h = {M: novelty(S_h, M) for M in HALFWIDTHS}
    nov_r2 = {M: novelty(S_r2, max(1, M // 2)) for M in HALFWIDTHS}
    nov_r2b = {M: novelty(S_r2b, max(1, M // 2)) for M in HALFWIDTHS}

    cuts = [s["b0"] for s in secs if s["b0"] > 0]
    k = len(cuts)
    mid = HALFWIDTHS[1]
    pk_r = top_peaks(nov_r1[mid], k, mid)
    pk_h = top_peaks(nov_h[mid], k, mid)
    pk_r2 = top_peaks(nov_r2[mid], k, max(1, mid // 2), step=2)

    img = figure(n, S_h, S_r1, S_r2, nov_r1, nov_r2, nov_r2b, nov_h, secs,
                 pk_r, pk_r2, pk_h, how)

    def near(p, xs):
        return min((abs(p - x) for x in xs), default=None)

    rows = ""
    for b in cuts:
        dr, dh, d2 = near(b, pk_r), near(b, pk_h), near(b, pk_r2)
        def cell(d):
            if d is None:
                return "<td class=no>—</td>"
            klass = "yes" if d <= NEAR else "no"
            return f"<td class={klass}>{'à ' + str(d) + ' mes.' if d else 'pile'}</td>"
        rows += (f"<tr><td><b>mes. {b+1}</b></td>{cell(dr)}{cell(d2)}{cell(dh)}</tr>")

    orphans = [p for p in pk_r if near(p, cuts) is None or near(p, cuts) > NEAR]
    btns = "".join(
        f"<button class=blk data-p='[{max(0, p-2)},{min(n, p+2)}]'>▶ mes. {p+1}</button>"
        for p in orphans)

    wsec = " ".join(f"{s['letter']}[{s['b0']+1}-{s['b1']+1}]" for s in secs)
    note = NOTES.get(stem)
    html = f"""<section data-stem="{stem}" data-grid='{json.dumps([round(float(t),3) for t in grid])}'>
<h2>{stem.replace('_',' ').title()} <span class=sub>{n} mesures ·
{len(secs)} sections écrites · {k} frontières</span></h2>
{f'<p class=obs>{note}</p>' if note else ''}
<img src="data:image/png;base64,{img}">
<p class=sub>Sections écrites aujourd'hui : {wsec}</p>
<table><tr><th>frontière écrite</th><th>pic rythme (1 mes.)</th>
<th>pic rythme (2 mes.)</th><th>pic harmonie</th></tr>{rows}</table>
<p class=sub>Chaque colonne propose exactement {k} pics — autant que de
frontières écrites — donc les trois lanes sont jugées à nombre de propositions
égal. Vert = le pic tombe à {NEAR} mesures ou moins de la frontière.</p>
{'<div class=lane><span class=lab>pics du rythme SANS frontière en face — écoute 2 mesures avant / 2 après</span>' + btns + '</div>' if btns else ''}
<audio preload=none playsinline src="../audio/{stem}.m4a"></audio></section>"""
    # Repère de lecture : combien de frontières un tirage AU HASARD de k pics
    # attraperait à ± NEAR mesures. Sur une chanson découpée en 26 sections,
    # c'est presque tout — la colonne « touchées » n'y veut plus rien dire.
    p = min(1.0, (2 * NEAR + 1) / max(n, 1))
    chance = k * (1 - (1 - p) ** k) if k else 0.0
    return html, {"stem": stem, "n": n, "k": k, "chance": chance,
                  "pk_r": pk_r, "pk_h": pk_h,
                  "hit_r": sum(1 for b in cuts if near(b, pk_r) is not None
                               and near(b, pk_r) <= NEAR),
                  "hit_r2": sum(1 for b in cuts if near(b, pk_r2) is not None
                                and near(b, pk_r2) <= NEAR),
                  "hit_h": sum(1 for b in cuts if near(b, pk_h) is not None
                               and near(b, pk_h) <= NEAR)}


LEDE = f"""<div class=lede>
<b>La matrice rythmique</b> compare la batterie mesure contre mesure : le stem
de batterie (demucs) est découpé en trois bandes — grosse caisse, caisse claire
et médium, cymbales — échantillonnées en seize doubles-croches par mesure, et
deux mesures se ressemblent au cosinus de ces deux petits tableaux. Ce sont
exactement les features de <i>rythme et harmonie côte à côte</i> ; seule la
lecture change ici.<br><br>

<b>L'agrégation à deux mesures.</b> Deux façons de faire, les deux calculées :
concaténer les deux mesures avant de comparer (une case = un motif de deux
mesures, <b>c'est celle qui est dessinée</b>), ou moyenner la matrice par blocs
2×2 (dessinée en pointillé gris sur la courbe). Concaténer garde l'ordre des
deux mesures ; moyenner efface la différence entre la mesure 1 et la mesure 2
d'un même groove. Les deux apparient les mesures 0-1, 2-3, … : si le motif de
deux mesures commence sur une mesure impaire, la paire est à cheval — rien ici
ne le détecte.<br><br>

<b>La courbe de nouveauté</b> — <i>noyau de Foote</i> : un damier qu'on fait
glisser le long de la diagonale de la matrice. Il vaut « + » sur le carré
avant×avant et sur le carré après×après, « − » sur les deux carrés croisés.
Il est donc haut quand ce qui précède se ressemble, ce qui suit se ressemble,
et que les deux ne se ressemblent pas entre eux : la définition d'un changement
de section. Foote 2000, <a href="https://doi.org/10.1109/ICME.2000.869637">
« Automatic audio segmentation using a measure of audio novelty »</a>.<br><br>

<b>La largeur du noyau décide de l'échelle.</b> ± 2 mesures voit les fills et
les relances ; ± 8 mesures ne voit que les grands blocs. Les trois largeurs sont
tracées ensemble, chacune ramenée à son propre maximum — seule la forme compte,
pas l'amplitude.<br><br>

<b>Comment lire.</b> Trois lanes de nouveauté empilées sur le même axe de
mesures : rythme à 1 mesure, rythme à 2 mesures, harmonie. Les traits verticaux
noirs sont les frontières de sections écrites aujourd'hui. Les triangles sont
les pics les plus forts de la largeur ± 4 mesures, <b>en nombre égal au nombre
de frontières écrites</b> — pas de seuil, chaque lane a droit au même nombre de
propositions. La question est simple : <b>est-ce que les triangles rouges
tombent là où les bleus ne tombent pas ?</b><br><br>

Rien n'est tranché ici, et aucun chiffre ne décide : les matrices sont empilées
plutôt que côte à côte précisément pour qu'elles partagent l'axe des mesures.
</div>"""


def main():
    stems = sys.argv[1:] or DEFAULT
    body, stats = "", []
    for st in stems:
        if not (HERE / f"docs/audio/{st}.m4a").exists():
            print(f"  !! {st} introuvable")
            continue
        try:
            h, s = song(st)
            body += h
            stats.append(s)
            print(f"  ok {st}  ({s['hit_r']}/{s['k']} rythme, "
                  f"{s['hit_r2']}/{s['k']} rythme-2, {s['hit_h']}/{s['k']} harmonie, "
                  f"hasard {s['chance']:.1f})")
            print(f"       pics rythme {s['pk_r']}")
            print(f"       pics harmo. {s['pk_h']}")
        except Exception as exc:
            print(f"  !! {st} — {type(exc).__name__}: {exc}")

    rec = "".join(
        f"<tr><td>{s['stem'].replace('_',' ').title()}</td><td>{s['n']}</td>"
        f"<td>{s['k']}</td><td>{s['hit_r']}</td><td>{s['hit_r2']}</td>"
        f"<td>{s['hit_h']}</td><td class=no>{s['chance']:.1f}</td></tr>"
        for s in stats)
    recap = (f"""<section><h2>Récapitulatif</h2>
<p class=sub>Combien des frontières écrites tombent à {NEAR} mesures ou moins
d'un pic, chaque lane ayant droit au même nombre de pics que de frontières.
<b>C'est un repère de lecture, pas un verdict</b> — la page se juge sur les
courbes. La dernière colonne dit combien de frontières on toucherait en tirant
les pics AU HASARD : sur les chansons découpées en 26 ou 31 sections, elle est
presque égale au total, donc les colonnes du milieu n'y veulent rien dire. Les
seules lignes où le chiffre porte une information sont celles où le hasard reste
bas.</p>
<table><tr><th>chanson</th><th>mesures</th><th>frontières</th>
<th>rythme 1 mes.</th><th>rythme 2 mes.</th><th>harmonie</th>
<th>au hasard</th></tr>{rec}</table></section>"""
             if stats else "")

    out = HERE / "harmonia_min/state/reports/rhythm_boundaries.html"
    out.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Le rythme dit-il où ça change ?</title><style>
*{{box-sizing:border-box;-webkit-tap-highlight-color:transparent}}
body{{margin:0;background:#e7e0d0;font:16px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1150px;margin:0 auto;padding:20px 12px calc(40px + env(safe-area-inset-bottom))}}
h1{{font:italic 600 24px Georgia,serif;margin:0 0 3px}}
.lede{{color:{MUTED};font-size:13.5px;line-height:1.55;margin-bottom:18px}}
.lede b{{color:{INK}}} .lede a{{color:{ACC}}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:14px 13px;margin-bottom:16px}}
h2{{font:700 18px system-ui;margin:0 0 10px;color:{ACC}}}
.sub{{font:500 12px system-ui;color:#8a8371;line-height:1.5}}
.sub b{{color:{INK}}}
.obs{{font-size:13.5px;background:#f7f3e9;border-left:3px solid {ACC};
  border-radius:0 8px 8px 0;padding:9px 11px;margin:0 0 10px}}
img{{width:100%;border-radius:8px;display:block;margin-bottom:8px}}
table{{border-collapse:collapse;font-size:12.5px;width:100%;margin:8px 0}}
th,td{{border:1px solid #e5dcc6;padding:3px 8px;text-align:left}}
th{{background:#f7f3e9;font-size:11px}}
td.yes{{background:#eaf3e6;color:#2f6b31;font-weight:600}}
td.no{{color:#a89f8c}}
.lane{{display:flex;flex-wrap:wrap;gap:4px;align-items:center;margin:8px 0 0}}
.lab{{font:600 11px system-ui;color:#8a8371;width:100%;margin-bottom:3px}}
button.blk{{border:1px solid #d8cfb4;background:#f7f3e9;border-radius:7px;
  padding:5px 8px;cursor:pointer;font:600 12px system-ui;color:{MUTED}}}
button.blk.on{{background:{ACC};border-color:{ACC};color:#fff}}
</style></head><body><div class=wrap>
<h1>Le rythme dit-il où ça change ?</h1>
{LEDE}{body}{recap}</div>
<script>
document.querySelectorAll("section[data-stem]").forEach(function(sec){{
  var au=sec.querySelector("audio"); if(!au) return;
  var G=JSON.parse(sec.dataset.grid), stop=null, on=null;
  au.addEventListener("timeupdate",function(){{
    if(stop!=null && au.currentTime>=stop){{ au.pause(); stop=null;
      if(on){{on.classList.remove("on"); on=null;}} }} }});
  sec.querySelectorAll("button[data-p]").forEach(function(b){{
    b.onclick=function(){{
      document.querySelectorAll("audio").forEach(function(a){{if(a!==au)a.pause();}});
      document.querySelectorAll("button.on").forEach(function(x){{x.classList.remove("on");}});
      var d=JSON.parse(b.dataset.p);
      stop=G[Math.min(d[1],G.length-1)];
      try{{au.currentTime=G[d[0]];}}catch(e){{}}
      on=b; b.classList.add("on"); au.play().catch(function(){{}});
    }};
  }});
}});
</script></body></html>""")
    print(f"\nwrote {out.relative_to(HERE)} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
