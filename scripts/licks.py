"""Les licks : une matrice de distance qui garde l'ORDRE et le RYTHME du chant.

    .venv/bin/python scripts/licks.py [<stem> ...]
        -> docs/plots/licks.html   (local, jouable, une section par morceau)

Louis, 2026-08-12 : « j'aimerais pouvoir détecter les licks des refrains dans les
chansons, tu peux m'aider à trouver la bonne matrice de distance voix pour ce
faire ? comment la computes-tu pour l'instant ? »

CE QUI EXISTAIT, ET POURQUOI ÇA NE PEUT PAS MARCHER POUR UN LICK. Les deux
matrices de voix du projet (`ssm_zoo.sub_voix` par demi-mesure,
`melody_ssm.melody_bars` par mesure) font la même chose : chaque note est versée
dans la case où elle COMMENCE, pondérée par sa durée, dans un vecteur de douze
demi-tons ; puis cosinus. C'est un **sac de notes par case**. Quatre
conséquences, toutes fatales à un lick :

  1. **l'ordre est perdu** — `mi ré do` et `do ré mi` donnent le même vecteur,
     alors qu'un lick EST un ordre ;
  2. **le rythme est perdu** — seule la durée totale par classe compte, le
     placement dans la mesure disparaît ;
  3. **mod 12 et pas d'intervalles** — un lick repris une tierce plus haut ne se
     reconnaît pas, et Sunny module d'un demi-ton à chaque reprise ;
  4. **une case vaut une mesure entière** — or un lick tient 6 à 15 notes sur une
     ou deux mesures, tout y est moyenné.

Et un cinquième, plus sournois : une note tenue ne compte que dans sa case de
départ, donc les cases suivantes paraissent MUETTES.

CE QU'ON MET À LA PLACE. On ne quantifie plus le chant par case : on garde la
suite des notes, et on décrit chaque note par deux nombres relatifs à la
précédente —

    (intervalle en demi-tons,  écart d'attaque en fractions de temps)

Les intervalles rendent la transposition GRATUITE (un lick monté d'un ton a les
mêmes intervalles) ; les écarts d'attaque gardent la figure rythmique. Une
fenêtre de `FEN` temps est alors une petite séquence, et la distance entre deux
fenêtres est un **alignement** (programmation dynamique type distance d'édition,
avec insertions et suppressions) — pas un produit scalaire, parce qu'un lick
rejoué a rarement exactement le même nombre de notes.

La matrice obtenue est indexée par les temps du morceau. Ses **diagonales
brillantes** sont les endroits où le même lick revient. La page les dessine, et
liste les licks les plus repris avec un bouton pour les écouter d'affilée.
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
from vote_fill import INK                                          # noqa: E402
import order_bundle                                                # noqa: E402

OUT = HERE / "docs" / "plots" / "licks.html"
PLOT_L, PLOT_R = 0.08, 0.995
FEN = 8.0          # une fenêtre de lick : 8 temps, soit deux mesures à 4/4
PAS = 1.0          # on avance d'un temps
COUT_TROU = 1.0    # ce que coûte une note en trop ou en moins dans l'alignement
POIDS_RYTH = 0.6   # le rythme compte un peu moins que les intervalles
NMIN = 4           # une fenêtre de moins de 4 notes n'est pas un lick


def notes_de(stem):
    """[(début en s, durée, hauteur MIDI)] — la mélodie chantée, telle quelle."""
    import vocal_anchor as VA
    import vocal_melody as VM
    voc = VA.separate_vocals(AUDIO / f"{stem}.m4a")
    tt, ff, vv, rr = VM.track_f0(voc)
    notes, _ = VM.melody_notes(tt, ff, vv, rr)
    notes, _ = VM.clean(notes)
    return sorted(notes, key=lambda x: x[0])


def temps_de(grid, n):
    """Les temps du morceau, quatre par mesure, depuis la grille de mesures."""
    t = []
    for b in range(n):
        t0, t1 = grid[b], grid[b + 1]
        t += [t0 + (t1 - t0) * k / 4 for k in range(4)]
    t.append(grid[n])
    return np.asarray(t)


def fenetres(notes, temps, fen=FEN, pas=PAS, nmin=NMIN):
    """[(temps de départ, séquence)] — le chant découpé en fenêtres glissantes.

    Une séquence est la liste des `(intervalle, écart d'attaque)` de la fenêtre,
    l'écart mesuré en TEMPS (pas en secondes) pour être indépendant du tempo.
    """
    idx = np.searchsorted(temps, [nt[0] for nt in notes]) - 1
    out = []
    for d in range(0, len(temps) - int(fen), int(pas)):
        f = int(d + fen)
        mem = [(i, notes[k]) for k, i in enumerate(idx) if d <= i < f]
        if len(mem) < nmin:
            continue
        seq = []
        for (i0, a), (i1, b) in zip(mem, mem[1:]):
            seq.append((int(round(b[2] - a[2])), float(i1 - i0)))
        if seq:
            out.append((d, seq))
    return out


def distance(u, v, trou=COUT_TROU, w=POIDS_RYTH):
    """Alignement de deux séquences (intervalle, écart) — distance d'édition.

    Chaque substitution coûte `|Δintervalle| / 12 + w · |Δécart| / 4`, bornée à 1
    (au-delà, deux notes n'ont plus rien à voir et la borne évite qu'un seul
    grand écart écrase tout le reste). Insertions et suppressions coûtent `trou`.
    Le résultat est ramené à la longueur de la plus longue des deux séquences,
    donc il vit entre 0 (le même lick) et ~1 (rien à voir).
    """
    nu, nv = len(u), len(v)
    D = np.zeros((nu + 1, nv + 1))
    D[:, 0] = np.arange(nu + 1) * trou
    D[0, :] = np.arange(nv + 1) * trou
    for i in range(1, nu + 1):
        ai, ri = u[i - 1]
        for j in range(1, nv + 1):
            aj, rj = v[j - 1]
            c = min(1.0, abs(ai - aj) / 12.0 + w * abs(ri - rj) / 4.0)
            D[i, j] = min(D[i - 1, j - 1] + c, D[i - 1, j] + trou,
                          D[i, j - 1] + trou)
    return float(D[nu, nv] / max(nu, nv))


def matrice(fens):
    """La matrice de distance entre toutes les fenêtres."""
    K = len(fens)
    M = np.ones((K, K))
    for i in range(K):
        M[i, i] = 0.0
        for j in range(i + 1, K):
            d = distance(fens[i][1], fens[j][1])
            M[i, j] = M[j, i] = d
    return M


def licks(fens, M, temps, seuil=None, ecart=8):
    """Les licks les plus repris : [(départ, [reprises], distance moyenne)].

    Une fenêtre est un lick si elle a au moins deux reprises à plus de `ecart`
    temps d'elle (sinon on retrouve son propre voisinage, qui se ressemble
    forcément) et sous le seuil. Le seuil est le premier décile des distances du
    morceau : ce qui est proche POUR CE MORCEAU, pas une constante.
    """
    K = len(fens)
    if K < 3:
        return [], 1.0
    off = M[~np.eye(K, dtype=bool)]
    seuil = float(np.quantile(off, 0.02)) if seuil is None else seuil
    out = []
    for i in range(K):
        rep = [j for j in range(K)
               if abs(fens[j][0] - fens[i][0]) >= ecart and M[i, j] <= seuil]
        if len(rep) >= 2:
            out.append((i, rep, float(np.mean([M[i, j] for j in rep]))))
    out.sort(key=lambda x: (-len(x[1]), x[2]))
    # LES LICKS RETENUS SONT DISJOINTS, occurrences comprises. Sans ça les
    # quatre premiers sont le MÊME lick vu depuis quatre fenêtres voisines : ils
    # ont des départs différents mais exactement les mêmes reprises, et la page
    # dessinait quatre bandes superposées. On exclut donc, après chaque lick
    # retenu, toutes les fenêtres qui chevauchent l'une de ses occurrences.
    gardes, pris = [], set()
    for i, rep, d in out:
        occ = [i] + rep
        if any(any(abs(fens[j][0] - fens[k][0]) < FEN for k in pris) for j in occ):
            continue
        gardes.append((i, rep, d))
        pris.update(occ)
        if len(gardes) >= 4:
            break
    return gardes, seuil


def song_png(stem, title):
    b = order_bundle.get(stem)
    n, grid = b["n"], np.asarray(b["grid"])
    temps = temps_de(grid, n)
    fens = fenetres(notes_de(stem), temps)
    if len(fens) < 4:
        return None
    M = matrice(fens)
    lk, seuil = licks(fens, M, temps)
    gt = gt_sections(stem)
    gtb = [s["b0"] for s in gt["sections"][1:]] if gt else []
    x = np.asarray([temps[f[0]] for f in fens])          # en secondes
    mes = np.asarray([f[0] / 4.0 for f in fens])         # en mesures

    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("h", ["#1d4d69", "#9fc0d4", "#faf6ec"])
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(12.6, 8.4), facecolor="#fffdf6",
        gridspec_kw={"height_ratios": [4.2, 1.0], "hspace": 0.16})

    ax1.imshow(M, cmap=cmap, vmin=0, vmax=float(np.quantile(M, 0.6)),
               extent=[mes[0], mes[-1], mes[-1], mes[0]], interpolation="nearest")
    for g in gtb:
        ax1.axvline(g, color=GT_LINE, lw=0.6, alpha=0.55)
        ax1.axhline(g, color=GT_LINE, lw=0.6, alpha=0.55)
    ax1.set_title("distance entre licks — sombre = le même lick", fontsize=10.5,
                  color="#4a4438", pad=8)
    ax1.set_xlabel("mesure", fontsize=9, color="#8a8371")

    COL = ["#2f7dbd", "#c07a1e", "#8155c6", "#2f8f6b"]
    ax2.set_xlim(0, n); ax2.set_ylim(0, 1); ax2.set_yticks([])
    for k, (i, rep, d) in enumerate(lk):
        for j in [i] + rep:
            m0 = fens[j][0] / 4.0
            ax2.add_patch(plt.Rectangle((m0, 0.12 + 0.2 * k), FEN / 4.0, 0.17,
                                        facecolor=COL[k % len(COL)],
                                        edgecolor="none", alpha=0.9))
    for g in gtb:
        ax2.axvline(g, color=GT_LINE, lw=0.7, alpha=0.6)
    ax2.set_xlabel("mesure", fontsize=9, color="#8a8371")
    ax2.set_title(f"les licks les plus repris (seuil {seuil:.2f})", fontsize=10.5,
                  color="#4a4438", pad=6)
    for ax in (ax1, ax2):
        ax.tick_params(labelsize=7.5, colors="#8a8371")
        for sp in ax.spines.values():
            sp.set_color("#e0d7c2")
    fig.subplots_adjust(left=PLOT_L, right=PLOT_R, top=0.95, bottom=0.07)
    img = fig2b64(fig)

    btns = "".join(
        f'<button class=blk data-t=\'{[round(float(temps[fens[j][0]]), 2) for j in [i] + rep]}\''
        f' style="border-color:{COL[k % len(COL)]}">lick {k + 1}'
        f'<small>mes. {fens[i][0] // 4 + 1} · {len(rep) + 1} fois</small></button>'
        for k, (i, rep, d) in enumerate(lk))
    return img, btns, len(fens), len(lk)


def main():
    args = [x for x in sys.argv[1:] if not x.startswith("--")]
    todo = [(s, t) for s, t in SONGS if s in args] or (
        [(s, s) for s in args] if args else list(SONGS))
    body = []
    for stem, title in todo:
        if not (AUDIO / f"{stem}.m4a").exists():
            continue
        r = song_png(stem, title)
        if r is None:
            print(f"  ?? {title} : pas assez de chant")
            continue
        img, btns, nf, nl = r
        body.append(
            f'<section><h2>{title} <span class=sub>{nf} fenêtres · {nl} lick(s)'
            f'</span></h2><img src="data:image/png;base64,{img}" alt="{title}">'
            f'<div class=lane>{btns or "<span class=hint>aucun lick repris</span>"}</div>'
            f'<audio preload=metadata playsinline src="../audio/{stem}.m4a"></audio>'
            '</section>')
        print(f"  ok {title} : {nl} lick(s)")
    OUT.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Les licks du chant</title><style>
body{{margin:0;background:#e7e0d0;font:15px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1280px;margin:0 auto;padding:22px 14px 70px}}
h1{{font:italic 600 25px Georgia,serif;margin:0 0 6px}}
.lede{{color:#6f6857;font-size:13.5px;margin-bottom:18px;max-width:980px}}
.lede b{{color:{INK}}} .lede code{{background:#f2ede0;padding:1px 4px;border-radius:4px}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:12px 14px;margin-bottom:12px}}
h2{{font:700 17px system-ui;margin:0 0 8px;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371}}
img{{width:100%;border-radius:8px;display:block}}
.lane{{display:flex;flex-wrap:wrap;gap:6px;margin-top:10px}}
button.blk{{border:2px solid #d8cfb4;background:#f7f3e9;border-radius:8px;
  padding:6px 10px;cursor:pointer;font:600 12.5px system-ui;color:#4a4438}}
button.blk small{{display:block;font:500 10px ui-monospace,monospace;color:#a89f8c}}
button.blk.on{{background:#0d2437;color:#fff}}
.hint{{font:500 11.5px system-ui;color:#a89f8c}}
audio{{display:none}}
</style></head><body><div class=wrap>
<h1>Les licks du chant</h1>
<div class=lede>Les matrices de voix du projet décrivent chaque mesure par un
<b>sac de notes</b> : douze demi-tons pondérés par la durée, puis cosinus. Pour
des sections ça suffit ; pour un lick c'est aveugle — <b>l'ordre</b> et
<b>le rythme</b> y disparaissent, <code>mi ré do</code> et <code>do ré mi</code>
donnent le même vecteur, et une transposition d'un ton casse tout.<br><br>
Ici chaque note est décrite par deux nombres relatifs à la précédente :
<b>(intervalle en demi-tons, écart d'attaque en temps)</b>. Les intervalles
rendent la transposition gratuite, les écarts gardent la figure rythmique. Une
fenêtre de deux mesures est une petite séquence, et deux fenêtres se comparent
par <b>alignement</b> — insertions et suppressions permises, parce qu'un lick
rejoué a rarement le même nombre de notes.<br><br>
<b>En haut</b> la matrice : sombre = le même lick. Ses <b>diagonales</b> sont les
reprises. <b>En bas</b> les licks les plus repris, une couleur chacun. Traits
rouges : tes frontières de sections. <b>Touche un lick pour entendre ses reprises
à la suite.</b></div>
{''.join(body)}</div>
<script>
document.querySelectorAll("section").forEach(function(sec){{
  var au = sec.querySelector("audio"); if(!au) return;
  var file = null, stop = null, raf = null;
  function jouer(ts, btn){{
    sec.querySelectorAll("button.blk").forEach(function(b){{ b.classList.remove("on"); }});
    btn.classList.add("on");
    var k = 0;
    function suivant(){{
      if(k >= ts.length){{ au.pause(); btn.classList.remove("on"); return; }}
      var t0 = ts[k], t1 = t0 + 4.0; k++;
      try {{ au.currentTime = t0; }} catch(e) {{}}
      au.play().catch(function(){{}});
      clearInterval(stop);
      stop = setInterval(function(){{
        if(au.currentTime >= t1 || au.paused){{ clearInterval(stop); suivant(); }}
      }}, 60);
    }}
    suivant();
  }}
  sec.querySelectorAll("button.blk").forEach(function(b){{
    b.onclick = function(){{ jouer(JSON.parse(b.dataset.t), b); }};
  }});
}});
</script></body></html>""")
    print(f"wrote {OUT}  ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
