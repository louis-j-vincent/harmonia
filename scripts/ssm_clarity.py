"""Quelle matrice a des pics NETS, morceau par morceau.

    .venv/bin/python scripts/ssm_clarity.py [<stem> ...]  ->  /plots/ssm_clarity.html

Louis, 2026-08-10, après le zoo (`scripts/ssm_zoo.py`) :

  « Ce que je vois, c'est qu'à chaque fois ce sont des matrices différentes qui
    donnent la réponse. Fais-moi un outil pour détecter quelles matrices ont des
    pics significatifs (clairs) — sur chaque chanson il suffirait d'une ou deux
    matrices pour trouver des sections claires. »

DEUX MESURES, ET ELLES NE FONT PAS LE MÊME TRAVAIL.

1. La **netteté** choisit COMBIEN de coupures la matrice propose. Une courbe de
   nouveauté intéressante a un petit nombre de pics qui se détachent du reste :

       netteté = proéminence du k-ième pic / (proéminence du (k+1)-ième + bruit)

   maximisée sur k, et le k gagnant EST le nombre de coupures — on ne le fixe pas
   d'avance. Le « bruit » est la proéminence médiane de tous les maxima locaux,
   ce qui rend la mesure sans échelle : chaque courbe est jugée à son propre
   étalon (la leçon du seuil TILE fixé à 0,80, docs/known_issues.md 2026-08-05).

2. Le **contraste** décide QUELLE matrice écouter. On coupe aux pics retenus et
   on demande si ça sépare vraiment :

       contraste = (similarité moyenne DANS les blocs − ENTRE les blocs)
                   / écart-type hors diagonale

CE QUI A CHANGÉ D'AVIS EN COURS DE ROUTE, et c'est le point important. Ma
première version classait par NETTETÉ. Mesuré sur les douze morceaux, elle classe
mal : sur This Love elle met `basse` (2,2) et `rythme` (2,2) devant `timbre`
(2,1) alors que l'œil voit l'inverse sans hésiter, et elle explose sur des cas
sans intérêt (29,5 sur un `rythme` à deux pics). Le contraste, lui, met `timbre`
en tête avec 0,81 contre 0,18 et 0,27 — et sur onze morceaux sur douze, les
coupures de la matrice la mieux classée tombent toutes sur une frontière validée
par Louis. Donc : **la netteté choisit k, le contraste classe.** Un peigne
régulier (Blue Lights sur les accords : huit pics égaux, la période de la boucle
et pas la forme) est éliminé par le contraste, pas par la netteté — couper une
boucle en tranches identiques ne sépare rien.

CE QUE ÇA NE FAIT PAS (règle n°4). L'outil ne pose pas de sections : il classe
les matrices et rend leurs coupures. Il en rend PEU — trois sur This Love quand
Louis en a neuf. La précision est haute, le rappel non, et c'est cohérent avec ce
que disait le zoo : le timbre et le rythme donnent la grande échelle, le grain
fin reste à la charge des matrices harmoniques. La fusion des deux n'est pas
écrite, et rien n'est branché sur `harmonia_min/sections.py`.
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

from ssm_zoo import (SONGS, AUDIO, CMAP, GT_LINE, _novelty,   # noqa: E402
                     fig2b64, gt_sections, substrates)

OUT = HERE / "docs" / "plots" / "ssm_clarity.html"

MIN_GAP_HB = 8      # deux coupures ne peuvent pas être à moins de 4 mesures
K_MAX = 8           # au-delà, ce n'est plus une forme, c'est une grille
KEEP_FRAC = 0.5     # on retient les matrices à au moins la moitié du meilleur
KEEP_FLOOR = 0.15   # …et jamais en dessous de ça, même si tout est mauvais
KEEP_MAX = 3        # « une ou deux matrices suffisent » — trois est déjà large
# Le seuil est RELATIF au meilleur du morceau, pas absolu : le contraste vit à
# 0,29 sur Don't Know Why et à 1,62 sur Let It Be, et dans les deux cas la
# meilleure matrice est juste. Un seuil commun aurait déclaré Don't Know Why
# « sans réponse » alors que ses cinq coupures tombent toutes justes.


# ── les deux mesures ────────────────────────────────────────────────────────

def peak_report(nov: np.ndarray) -> dict:
    """{'cuts', 'nettete', 'k', 'prom', 'back'} — les pics qui se détachent.

    `cuts` est en cases de demi-mesure, dans l'ordre. `k` est choisi par la
    mesure elle-même, pas imposé.
    """
    from scipy.signal import find_peaks, peak_prominences
    x = np.nan_to_num(nov, nan=0.0)
    ok = np.isfinite(nov)

    # le BRUIT de cette courbe : la proéminence médiane de tous ses maxima
    allp, _ = find_peaks(x)
    back = float(np.median(peak_prominences(x, allp)[0])) if len(allp) else 0.0
    back = max(back, 1e-3)

    idx, _ = find_peaks(x, distance=MIN_GAP_HB)
    idx = np.array([i for i in idx if ok[i]], dtype=int)
    if len(idx) == 0:
        return {"cuts": [], "nettete": 0.0, "k": 0, "prom": [], "back": back}
    prom = peak_prominences(x, idx)[0]
    order = np.argsort(-prom)
    idx, prom = idx[order], prom[order]

    if len(prom) == 1:
        return {"cuts": [int(idx[0])], "nettete": float(prom[0] / back),
                "k": 1, "prom": prom.tolist(), "back": back}

    ratios = [prom[k - 1] / (prom[k] + back) for k in range(1, min(K_MAX, len(prom)))]
    k = int(np.argmax(ratios)) + 1
    return {"cuts": sorted(int(i) for i in idx[:k]),
            "nettete": float(ratios[k - 1]), "k": k,
            "prom": prom.tolist(), "back": back}


def block_contrast(S: np.ndarray, cuts: list[int], *, skip: int = 2) -> float:
    """Combien les blocs découpés par `cuts` se distinguent, en écarts-types de
    la matrice elle-même. `skip` écarte la bande diagonale, trivialement
    similaire — sans ça toute découpe paraît bonne."""
    m = len(S)
    if not cuts:
        return 0.0
    lab = np.zeros(m, int)
    for c in cuts:
        lab[c:] += 1
    i, j = np.indices((m, m))
    far = np.abs(i - j) >= skip
    same = (lab[:, None] == lab[None, :]) & far
    diff = (lab[:, None] != lab[None, :]) & far
    if not same.any() or not diff.any():
        return 0.0
    sd = float(S[far].std())
    return float((S[same].mean() - S[diff].mean()) / max(sd, 1e-9))


def rank_substrates(subs) -> list[dict]:
    """[{nom, S, nov, cuts, nettete, k, contraste, retenu}] — classé par
    CONTRASTE décroissant, et `retenu` marque les matrices à écouter."""
    out = []
    for nm, gloss, S in subs:
        nov = _novelty(S)
        r = peak_report(nov)
        out.append({"nom": nm, "gloss": gloss, "S": S, "nov": nov,
                    "cuts": r["cuts"], "nettete": r["nettete"], "k": r["k"],
                    "contraste": block_contrast(S, r["cuts"])})
    out.sort(key=lambda d: -d["contraste"])
    best = out[0]["contraste"] if out else 0.0
    thr = max(KEEP_FRAC * best, KEEP_FLOOR)
    for i, d in enumerate(out):
        d["retenu"] = bool(i < KEEP_MAX and d["contraste"] >= thr and d["cuts"])
    return out


# ── la page ─────────────────────────────────────────────────────────────────

def _bars(cuts, m, n):
    """Les coupures en numéros de MESURE (1-indexé), pour l'affichage."""
    return [1 + int(round(c * n / max(1, m))) for c in cuts]


def song_html(stem: str, title: str) -> str:
    subs, n, _T, _x = substrates(stem)
    ranked = rank_substrates(subs)
    gt = gt_sections(stem)
    gtb = [sg["b0"] for sg in gt["sections"][1:]] if gt else []

    k = len(ranked)
    fig, axs = plt.subplots(k, 1, figsize=(13.5, 0.92 * k), facecolor="#fffdf6",
                            gridspec_kw={"hspace": 0.55})
    for ax, d in zip(np.atleast_1d(axs), ranked):
        m = len(d["nov"])
        net = d["retenu"]
        ax.fill_between(np.arange(m), np.nan_to_num(d["nov"]),
                        color="#3d7fa6" if net else "#c3cdd4", lw=0)
        for b in gtb:                                  # tes frontières
            ax.axvline(b * m / max(1, n), color=GT_LINE, lw=0.8, alpha=0.8)
        for c in d["cuts"]:                            # ce que la matrice propose
            ax.plot([c], [1.13], marker="v", ms=7, color="#1b4a6b" if net else "#aab4bb",
                    clip_on=False)
        ax.set_xlim(0, m - 1); ax.set_ylim(0, 1.2)
        ax.set_yticks([]); ax.set_xticks([])
        ax.set_ylabel(d["nom"], rotation=0, ha="right", va="center", fontsize=11,
                      color="#1c1c1c" if net else "#9aa3a9")
        for s in ax.spines.values():
            s.set_color("#d8cfb8")
    img = fig2b64(fig)

    def verif(d):
        cb = _bars(d["cuts"], len(d["nov"]), n)
        if not gtb:
            return "—"
        just = sum(1 for b in cb if any(abs(b - 1 - g) <= 1 for g in gtb))
        found = sum(1 for g in gtb if any(abs(b - 1 - g) <= 1 for b in cb))
        return f"{just}/{len(cb)} · {found}/{len(gtb)}"

    rows = "".join(
        f"<tr class={'net' if d['retenu'] else 'flat'}>"
        f"<td>{d['nom']}</td><td>{d['contraste']:.2f}</td><td>{d['k']}</td>"
        f"<td>{d['nettete']:.1f}</td>"
        f"<td class=cutlist>{' · '.join(str(b) for b in _bars(d['cuts'], len(d['nov']), n)) or '—'}</td>"
        f"<td>{verif(d)}</td></tr>" for d in ranked)

    best = [d for d in ranked if d["retenu"]]
    verdict = ("aucune matrice ne sépare quoi que ce soit ici"
               if not best else
               "ici, écoute " + " et ".join(f"<b>{d['nom']}</b>" for d in best))
    yours = " · ".join(str(b + 1) for b in gtb) or "—"

    return f"""<section id="{stem}"><h2>{title}
<span class=sub>{n} mesures</span></h2>
<div class=verdict>{verdict}</div>
<img src="data:image/png;base64,{img}" alt="courbes de nouveauté de {title}">
<div class=legend>▼ = les coupures que la matrice propose · traits rouges = les
tiennes (mesures {yours}) · en bleu les matrices retenues, en gris les autres ·
classement par <b>contraste</b></div>
<table><tr><th>matrice</th><th>contraste</th><th>coupures</th><th>netteté</th>
<th>aux mesures</th><th>justes · tes frontières trouvées</th></tr>{rows}</table></section>"""


def main():
    stems = sys.argv[1:] or [s for s, _ in SONGS]
    todo = [(s, t) for s, t in SONGS if s in stems] or [(s, s) for s in stems]
    body = ""
    for stem, title in todo:
        if not (AUDIO / f"{stem}.m4a").exists():
            print(f"  ?? pas d'audio pour {stem}")
            continue
        body += song_html(stem, title)
        print(f"  ok {title}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Quelle matrice a des pics nets</title><style>
body{{margin:0;background:#e7e0d0;font:15px/1.55 -apple-system,system-ui,sans-serif;color:#1c1c1c}}
.wrap{{max-width:1250px;margin:0 auto;padding:22px 15px 70px}}
h1{{font:italic 600 26px Georgia,serif;margin:0 0 6px}}
.lede{{color:#6f6857;font-size:13.5px;margin-bottom:20px;max-width:900px}}
.lede b{{color:#1c1c1c}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:16px 18px;margin-bottom:14px}}
h2{{font:700 18px system-ui;margin:0 0 4px;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371}}
.verdict{{font-size:14.5px;margin:2px 0 10px}}
img{{max-width:100%;display:block}}
.legend{{font-size:11.5px;color:#8a8371;margin:6px 0 10px}}
table{{border-collapse:collapse;font-size:12.5px;width:100%;max-width:760px}}
th,td{{padding:3px 10px;text-align:right;border-bottom:1px solid #ece4d2}}
th:first-child,td:first-child{{text-align:left}}
.cutlist{{text-align:left;color:#6f6857}}
th{{color:#8a8371;font-weight:600;font-size:11px;text-transform:uppercase;letter-spacing:.04em}}
tr.net td{{color:#1c1c1c;font-weight:600}}
tr.flat td{{color:#a8a294}}
ul.how{{font-size:13.5px;max-width:960px;padding-left:20px}}
ul.how li{{margin-bottom:7px}}
</style></head><body><div class=wrap>
<h1>Quelle matrice a des pics nets</h1>
<div class=lede>Suite du <a href="ssm_zoo.html">zoo</a> : à chaque morceau ce
n'est pas la même matrice qui répond, donc l'outil dit <b>laquelle écouter</b>.
Il ne pose aucune section — il classe les huit courbes de nouveauté et rend les
coupures de celles qui se détachent.</div>
<section><h2>Les deux mesures <span class=sub>et laquelle décide</span></h2>
<ul class=how>
<li><b>La netteté choisit COMBIEN de coupures</b>, pas lesquelles écouter :
l'écart entre le k-ième pic et le suivant, maximisé sur k. Le k gagnant est le
nombre de coupures — rien n'est fixé d'avance.</li>
<li><b>Le CONTRASTE décide quelle matrice écouter</b> : on coupe aux pics, et on
demande si ça sépare vraiment — similarité moyenne DANS les blocs moins ENTRE
les blocs, en écarts-types de la matrice.</li>
<li><b>J'ai changé d'avis en cours de route, et c'est le point.</b> Ma première
version classait par netteté. Sur This Love elle mettait <i>basse</i> (2,2) et
<i>rythme</i> (2,2) devant <i>timbre</i> (2,1) — l'inverse de ce que l'œil voit.
Le contraste met <i>timbre</i> à 0,81 contre 0,18 et 0,27. C'est aussi lui qui
élimine le peigne régulier (Blue Lights sur les accords : huit pics égaux =
la période de la boucle) — couper une boucle en tranches identiques ne sépare
rien, alors que ses pics sont parfaitement nets.</li>
<li>Les deux sont <b>sans échelle</b> : chaque courbe est jugée à son propre
bruit, jamais à un seuil commun entre substrats. Le seuil de sélection est
lui aussi relatif au meilleur du morceau (le contraste vit à 0,29 sur Don't Know
Why et à 1,62 sur Let It Be, et les deux fois la meilleure matrice tombe juste).</li>
<li><b>Tes frontières ne servent à rien dans le calcul.</b> Elles sont tracées en
rouge, et la dernière colonne dit « coupures justes · tes frontières
retrouvées » (± 1 mesure) — pour vérifier, pas pour choisir.</li>
</ul></section>
{body}</div></body></html>""")
    print(f"wrote {OUT.relative_to(HERE)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
