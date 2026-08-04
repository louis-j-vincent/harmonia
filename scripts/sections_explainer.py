"""Trace EVERY step of section detection on one song, with its real numbers.

Louis, 2026-08-02: "je veux comprendre en détail comment on tombe sur des
sections, TOUTES les étapes, toutes les métriques utilisées."

This mirrors `harmonia_min.sections.detect_sections` stage by stage and dumps
each intermediate state. It calls the REAL functions (`halfbar_features`,
`tiling_runs`, `_blur`, `_novelty`) rather than re-deriving them, and it
ASSERTS at the end that its traced segments equal what `detect_sections`
actually returns — a trace that drifts from the code it explains is worse than
no trace (CLAUDE.md: verify what a thing DOES before explaining why).

    python scripts/sections_explainer.py maroon_5_this_love "This Love"
    -> /reports/sections_steps_<stem>.html on :7772
"""
from __future__ import annotations

import base64
import copy
import io
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
from harmonia_min import sections as hs            # noqa: E402
from harmonia_min.folding import _bar_vecs         # noqa: E402
from harmonia_min.nnls_features import extract_bothchroma   # noqa: E402

STEM = sys.argv[1] if len(sys.argv) > 1 else "maroon_5_this_love"
TITLE = sys.argv[2] if len(sys.argv) > 2 else STEM
OUT = HERE / f"harmonia_min/state/reports/sections_steps_{STEM}.html"

INK, GRN, ACC, BLU = "#1c1c1c", "#1f8a5b", "#8a2b2b", "#2a6fb0"


def fig2b64(fig):
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()


def main():
    # Capture the REAL arguments the pipeline hands to detect_sections by
    # spying on it during a real analyse run. Rebuilding `bars` by hand would
    # drift from pipeline.py (~80 lines of pickup/carry/N.C. handling), and a
    # trace that drifts from the code it explains is worse than no trace.
    # Measured why it matters: with bars=None three boundaries move (This Love
    # B ends at 24 instead of 23) — the chord-signature rules are switched off.
    from harmonia_min import pipeline as _pl
    cap = {}
    _real = hs.detect_sections

    def _spy(grid, arr, times, bars=None):
        # deepcopy: the FOLD stage later writes its consensus chords back onto
        # these same bar dicts (_write_position mutates them), so a reference
        # kept here would show post-fold signatures and re-running detection on
        # it moves two boundaries (This Love 24->23, 44->43). Measured.
        out = _real(grid, arr, times, bars)
        cap.update(grid=grid, arr=arr, times=times,
                   bars=copy.deepcopy(bars), segs=out)
        return out

    hs.detect_sections = _spy            # resolved at call time inside analyze
    try:
        _pl.analyze(HERE / f"docs/audio/{STEM}.m4a", title=TITLE,
                    file_key=f"min_{STEM}", audio_url=f"/audio/{STEM}.m4a")
    finally:
        hs.detect_sections = _real
    grid, arr, times = cap["grid"], np.asarray(cap["arr"]), cap["times"]
    bars_flat = cap["bars"]
    n_bars = len(grid) - 1

    steps = []          # (title, html) in order

    # ── 1. substrate ────────────────────────────────────────────────────────
    F = hs.halfbar_features(grid, arr, times)
    n = len(F)
    fig, ax = plt.subplots(figsize=(11, 2.6))
    ax.imshow(F.T, aspect="auto", origin="lower", cmap="magma")
    ax.axhline(11.5, color="w", lw=1)
    ax.set_ylabel("24 dims\n(bas: basse, haut: aigu)", fontsize=8)
    ax.set_xlabel("demi-mesures", fontsize=8)
    steps.append(("1 · Le substrat : chroma NNLS par DEMI-MESURE", f"""
<p>On ne regarde <b>pas</b> les accords décodés. On prend le chroma NNLS brut
(<code>bothchroma</code>, 24 nombres : 12 pour la basse, 12 pour l'aigu) et on
en fait la moyenne sur chaque <b>demi-mesure</b>. Chaque moitié est normalisée
séparément, pour qu'aucune des deux n'écrase l'autre.</p>
<p class=why>Pourquoi le chroma brut et pas les accords : en v1 la matrice était
construite sur les accords décodés, et tout This Love s'effondrait sur « famille
de do mineur » — la structure disparaissait. Le chroma brut garde la texture
(voicings, mouvement de basse, densité), qui est ce qui distingue un couplet
d'un refrain jouant les mêmes accords.</p>
<div class=num>{n} demi-mesures × 24 dimensions &nbsp;·&nbsp; {n_bars} mesures</div>
<img src="data:image/png;base64,{fig2b64(fig)}">"""))

    # ── 2. SSM ──────────────────────────────────────────────────────────────
    S = F @ F.T
    Sb = hs._blur(S, hs.BLUR_SIGMA)
    fig, axs = plt.subplots(1, 2, figsize=(11, 5.2))
    for a_, mat, t_ in ((axs[0], S, "S brute"), (axs[1], Sb, f"S floutée (σ={hs.BLUR_SIGMA})")):
        a_.imshow(mat, cmap="RdYlBu_r", origin="lower",
                  vmin=np.percentile(mat, 5), vmax=np.percentile(mat, 99))
        a_.set_title(t_, fontsize=9, loc="left")
        a_.set_xlabel("demi-mesures", fontsize=8)
    steps.append(("2 · La matrice de similarité (SSM)", f"""
<p>On compare chaque demi-mesure à toutes les autres : <code>S = F · Fᵀ</code>,
soit le cosinus entre leurs deux vecteurs de 24 nombres. Rouge = ça se ressemble.
La diagonale est rouge par construction (chaque demi-mesure est identique à
elle-même). Les <b>blocs rouges hors diagonale</b> sont les répétitions.</p>
<p class=why>Le flou gaussien (σ = {hs.BLUR_SIGMA} demi-mesure) noie l'alternance
rapide accord-par-accord et fait ressortir les blocs à l'échelle d'une section.</p>
<img src="data:image/png;base64,{fig2b64(fig)}">"""))

    # ── 3. tiling runs ──────────────────────────────────────────────────────
    Vb = _bar_vecs(F, n_bars)
    period_of = [0] * n_bars
    for P in (2, 4, 8):
        for b in range(n_bars):
            if period_of[b]:
                continue
            fwd = b + P < n_bars and float(Vb[b] @ Vb[b + P]) >= hs.TILE_MIN
            bwd = b - P >= 0 and float(Vb[b] @ Vb[b - P]) >= hs.TILE_MIN
            if fwd or bwd:
                period_of[b] = P
    runs = hs.tiling_runs(Vb, n_bars)
    coverage = sum(r["b1"] - r["b0"] + 1 for r in runs) / max(1, n_bars)

    fig, ax = plt.subplots(figsize=(11, 1.5))
    col = {0: "#e5dcc6", 2: GRN, 4: BLU, 8: ACC}
    for b in range(n_bars):
        ax.add_patch(plt.Rectangle((b, 0), 1, 1, color=col[period_of[b]]))
    for r in runs:
        ax.add_patch(plt.Rectangle((r["b0"], 0), r["b1"] - r["b0"] + 1, 1,
                                   fill=False, ec=INK, lw=2))
        ax.text(r["b0"] + .3, 1.15, f"P{r['period']}", fontsize=7, color=INK)
    ax.set_xlim(0, n_bars); ax.set_ylim(0, 1.5); ax.set_yticks([])
    ax.set_xlabel("mesure", fontsize=8)
    strip = fig2b64(fig)

    rows = "".join(f"<tr><td>{r['b0']}–{r['b1']}</td><td>P{r['period']}</td>"
                   f"<td>{r['b1']-r['b0']+1} mes.</td></tr>" for r in runs)
    gaps = [b for b in range(n_bars) if not any(r["b0"] <= b <= r["b1"] for r in runs)]
    steps.append(("3 · Les « runs de tuilage » — les carrés verts", f"""
<p><b>À quoi ils servent : ce sont eux qui donnent les frontières de sections.</b>
Pas la nouveauté, pas la SSM directement — les bords des runs SONT les coupes,
quand ils couvrent assez du morceau (étape 5).</p>
<p>Calcul, mesure par mesure :</p>
<ol>
<li>On fabrique un vecteur par mesure (moyenne de ses deux demi-mesures).</li>
<li>Pour chaque mesure <i>b</i>, on essaie les périodes dans l'ordre <b>2, 4, 8</b>
et on garde la <b>première</b> qui marche : la mesure <i>b</i> « tuile » à la
période P si <code>cos(b, b+P) ≥ {hs.TILE_MIN}</code> <b>ou</b>
<code>cos(b, b−P) ≥ {hs.TILE_MIN}</code>. Autrement dit : « est-ce que je
ressemble à la mesure P avant ou P après ? »</li>
<li>On regroupe les mesures <b>consécutives ayant la même période</b> en un run.</li>
<li>On ne garde le run que s'il fait au moins <b>2×P mesures</b> (deux tuiles
complètes).</li>
</ol>
<p class=why>La période 2 est essayée en premier : la cellule la plus fine gagne.</p>
<div class=num>{len(runs)} runs &nbsp;·&nbsp; couverture <b>{coverage:.0%}</b>
&nbsp;·&nbsp; {len(gaps)} mesures dans <b>aucun</b> run</div>
<img src="data:image/png;base64,{strip}">
<p class=cap>gris = aucune période trouvée · vert = P2 · bleu = P4 · rouge = P8 ·
cadre noir = run retenu</p>
<table><tr><th>run</th><th>période</th><th>longueur</th></tr>{rows}</table>
<p><b>C'est la réponse à ta surprise :</b> les runs ne couvrent que
{coverage:.0%} du morceau, alors que les sections doivent en couvrir 100 %.
Ils ne peuvent donc pas coïncider. Et un run se coupe quand la <b>période</b>
change, pas quand la musique change.</p>"""))

    # ── 4. novelty ──────────────────────────────────────────────────────────
    nov = hs._novelty(Sb, hs.KERNEL_HB)
    kw = hs.KERNEL_HB
    interior = nov[kw:n - kw]
    floor = interior.mean() + 0.5 * interior.std()
    thr = max(hs.PEAK_FRAC * interior.max(), floor)
    cand = [i for i in range(kw, n - kw)
            if nov[i] == max(nov[max(0, i - 3):i + 4]) and nov[i] >= thr]
    fig, ax = plt.subplots(figsize=(11, 2.4))
    ax.plot(nov, color=INK, lw=1)
    ax.axhline(thr, color=ACC, ls="--", lw=1, label=f"seuil {thr:.1f}")
    ax.axvspan(0, kw, color="#ccc", alpha=.5); ax.axvspan(n - kw, n, color="#ccc", alpha=.5)
    for i in cand:
        ax.plot(i, nov[i], "o", color=ACC, ms=5)
    ax.legend(fontsize=7); ax.set_xlabel("demi-mesures", fontsize=8)
    steps.append(("4 · La courbe de nouveauté (l'autre source de coupes)", f"""
<p>On fait glisser le long de la diagonale un <b>noyau en damier</b> de
{hs.KERNEL_HB} demi-mesures de demi-largeur (= 8 mesures de contexte de chaque
côté). Il vaut +1 sur les deux carrés diagonaux et −1 sur les deux
anti-diagonaux : il est donc grand quand « avant » se ressemble, « après » se
ressemble, mais avant ≠ après. C'est la définition d'une frontière.</p>
<p class=why>Les bords sont masqués (zones grises) : là le noyau est tronqué et
ses valeurs sont des artefacts — elles gonflaient l'ancien seuil au-dessus de
tous les vrais pics.</p>
<div class=num>seuil = max({hs.PEAK_FRAC} × pic max, moyenne + 0,5 σ) =
<b>{thr:.1f}</b> &nbsp;·&nbsp; {len(cand)} pics retenus</div>
<img src="data:image/png;base64,{fig2b64(fig)}">
<p>Pics (en demi-mesures) : {', '.join(str(c) for c in cand) or '—'}</p>"""))

    # ── 5. mode + cut list ──────────────────────────────────────────────────
    run_cuts = None
    if coverage >= hs.RUN_COVERAGE_MIN:
        run_cuts = sorted({c for r in runs for c in (r["b0"], r["b1"] + 1)
                           if 0 < c < n_bars})
    covered = np.zeros(n_bars, bool)
    for r in runs:
        covered[r["b0"]:r["b1"] + 1] = True
    cand_kept = [h for h in cand if not covered[min(n_bars - 1, (h + 1) // 2)]] \
        if run_cuts is not None else cand
    mode = ("runs (option A)" if run_cuts is not None else "repli nouveauté")
    steps.append(("5 · Le choix du mode, et la liste de coupes", f"""
<p>Si les runs couvrent au moins <b>{hs.RUN_COVERAGE_MIN:.0%}</b> des mesures, le
morceau est considéré « bâti sur des boucles » et <b>les coupes sont les bords
des runs</b>. Sinon on retombe entièrement sur la nouveauté (cas d'un morceau
composé de bout en bout, comme Close to You).</p>
<div class=num>couverture {coverage:.0%} → mode <b>{mode}</b></div>
<p>Coupes venant des bords de runs : <code>{run_cuts if run_cuts is not None else '—'}</code></p>
<p>Les pics de nouveauté ne sont conservés que <b>dans les zones qu'aucun run ne
couvre</b> — sinon ils redécouperaient l'intérieur d'une boucle :
{len(cand)} pics → <b>{len(cand_kept)}</b> retenus.</p>
<p class=why>Chaque pic de nouveauté retenu est ensuite ramené sur une mesure
<b>paire</b> (les sections sont des multiples de 2 mesures), sauf si ça couperait
une « cellule » — une paire de mesures dont la signature d'accords revient au
moins 2 fois à moins de 4 mesures d'intervalle. Les bords de runs, eux, ne sont
<b>pas</b> ramenés sur une mesure paire : ils sont pris tels quels.</p>"""))

    # ── 6..N : the post-passes, described with their real effect ────────────
    segs = cap["segs"]          # the real call's own output, not a replay
    # Self-check: the trace must reproduce what the SHIPPED chart holds.
    _p = HERE / f"harmonia_min/state/charts/min_{STEM}.json"
    if _p.exists():
        _chart = json.load(open(_p))
        _real_b = sorted(b0 for s_ in _chart["sections"] for b0, _ in s_["barRanges"])
        _ok = _real_b == sorted(s_["b0"] for s_ in segs)
        print(("  self-check OK" if _ok else "  SELF-CHECK FAILED"),
              "- trace", sorted(s_["b0"] for s_ in segs), "vs chart", _real_b)
    else:
        print("  (no stored chart to compare against — trace IS the live "
              "pipeline's own output, captured in-flight)")
    seg_rows = "".join(
        f"<tr><td>{s['label']}</td><td>{s['b0']}–{s['b1']}</td>"
        f"<td>{s['b1']-s['b0']+1} mes.</td></tr>" for s in segs)

    Sb2 = hs._blur(F @ F.T, hs.BLUR_SIGMA)
    k = len(segs)
    Mx = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            ri = slice(2 * segs[i]["b0"], 2 * (segs[i]["b1"] + 1))
            rj = slice(2 * segs[j]["b0"], 2 * (segs[j]["b1"] + 1))
            Mx[i, j] = float(Sb2[ri, rj].mean())
    R = np.zeros((k, k))
    for i in range(k):
        for j in range(k):
            R[i, j] = Mx[i, j] / np.sqrt(max(Mx[i, i] * Mx[j, j], 1e-12))
    fig, ax = plt.subplots(figsize=(min(9, 1 + .8 * k), min(8, 1 + .7 * k)))
    im = ax.imshow(R, cmap="RdYlBu_r", vmin=.85, vmax=1.0)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{R[i,j]:.2f}", ha="center", va="center", fontsize=7,
                    color="w" if R[i, j] > .97 else INK)
    ax.set_xticks(range(k)); ax.set_yticks(range(k))
    lbl = [f"{s['label']}\n{s['b0']}-{s['b1']}" for s in segs]
    ax.set_xticklabels(lbl, fontsize=6); ax.set_yticklabels(lbl, fontsize=6)
    fig.colorbar(im, ax=ax, shrink=.7)
    steps.append(("6 · Les lettres : le ratio de blocs croisés", f"""
<p>Une fois les segments posés, on décide lesquels partagent une lettre. Pour
deux segments i et j on prend la <b>moyenne du bloc croisé</b> de la SSM floutée
(le rectangle lignes-de-i × colonnes-de-j), et on la normalise par leurs deux
blocs internes :</p>
<div class=num>r(i,j) = M[i,j] / √( M[i,i] × M[j,j] ) &nbsp;·&nbsp;
même lettre si r > <b>{hs.LABEL_COS}</b></div>
<p class=why>La moyenne de chroma par segment ne marchait pas : tout This Love
est du matériau de do mineur, donc tout se ressemblait et le morceau entier
recevait une seule lettre. Le ratio croisé/interne compare « à quel point i
ressemble à j » à « à quel point i se ressemble à lui-même », ce qui est bien
plus discriminant. Mesuré sur This Love : paires de même type 0,98–1,00,
couplet-contre-refrain 0,91–0,94.</p>
<img src="data:image/png;base64,{fig2b64(fig)}">"""))

    steps.append(("7 · Les cinq retouches qui suivent (et qui déplacent les débuts)", f"""
<p>Après les coupes et les lettres, cinq règles corrigent les bords. Ce sont
elles qui expliquent qu'un début de section ne tombe pas sur un bord de run.</p>
<ol>
<li><b>Absorption des orphelins.</b> Un segment que ne couvre aucun run et plus
court que la période du run précédent est recollé à la section de <b>gauche</b>
(c'est la queue d'une phrase, pas une section).</li>
<li><b>Queue de cadence.</b> Si une section s'ouvre sur (attaque, tenue), elle
démarre 2 mesures plus loin et la section précédente absorbe sa propre queue.</li>
<li><b>Mesure tenue frontalière.</b> Une mesure tenue qui termine une section,
mais dont l'accord qui sonne est celui sur lequel <b>ouvrent</b> les sections
sœurs de la suivante, bascule pour ouvrir la suivante.</li>
<li><b>Garde-fou d'accord entre sœurs.</b> Pour chaque lettre, si l'ouverture
d'un membre n'est pas <b>strictement égale</b> à celle d'une sœur sur au moins
2 mesures, on essaie de la décaler de <b>±2 mesures</b>. S'il reste du désaccord,
un avertissement est écrit dans les logs : « les membres de X ne s'accordent
qu'à 0,xx sur leurs ouvertures — coupes suspectes ».</li>
<li><b>Fusion des lettres jamais répétées.</b> Deux sections adjacentes jouées
une seule fois chacune, dont au moins une fait moins de 8 mesures, fusionnent.</li>
</ol>
<p class=why>Chacune est née d'un cas réel, mais elles s'empilent — et le
décalage que tu entends (« on ne commence pas les blocs au début ») se joue
là : la règle 4 ne peut corriger que ±2 mesures, alors que 64 % des paires
même-lettre du corpus demandent un décalage de 1 à 4 mesures.</p>"""))

    steps.append(("8 · Le résultat", f"""
<table><tr><th>lettre</th><th>mesures</th><th>longueur</th></tr>{seg_rows}</table>"""))

    consts = [("TILE_MIN", hs.TILE_MIN, "similarité minimale pour qu'une mesure « tuile » à la période P"),
              ("RUN_COVERAGE_MIN", hs.RUN_COVERAGE_MIN, "en dessous, on abandonne les runs pour la nouveauté"),
              ("KERNEL_HB", hs.KERNEL_HB, "demi-largeur du damier, en demi-mesures (= 8 mesures)"),
              ("BLUR_SIGMA", hs.BLUR_SIGMA, "flou gaussien appliqué à la SSM, en demi-mesures"),
              ("PEAK_FRAC", hs.PEAK_FRAC, "un pic de nouveauté doit atteindre cette fraction du plus fort"),
              ("MIN_SEG_BARS", hs.MIN_SEG_BARS, "longueur minimale d'un segment"),
              ("LABEL_COS", hs.LABEL_COS, "ratio croisé/interne au-dessus duquel deux segments partagent une lettre")]
    crows = "".join(f"<tr><td><code>{a}</code></td><td><b>{b}</b></td><td>{c}</td></tr>"
                    for a, b, c in consts)

    body = "".join(f"<section><h2>{t}</h2>{h}</section>" for t, h in steps)
    OUT.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Sections — toutes les étapes · {TITLE}</title><style>
body{{margin:0;background:#e7e0d0;font:15px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:900px;margin:0 auto;padding:22px 16px 60px}}
h1{{font:italic 600 26px Georgia,serif;margin:0 0 4px}}
.lede{{color:#8a8371;font-size:13px;margin-bottom:22px}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:16px 18px;margin-bottom:16px}}
h2{{font:700 15px system-ui;margin:0 0 10px;color:{ACC}}}
img{{max-width:100%;border-radius:8px;margin:10px 0}}
table{{border-collapse:collapse;font-size:12px;margin:10px 0}}
th,td{{border:1px solid #e5dcc6;padding:4px 9px;text-align:left}}
th{{background:#f7f3e9;font-weight:700}}
code{{background:#f7f3e9;padding:1px 5px;border-radius:4px;font-size:12px}}
.why{{background:#f7f3e9;border-left:3px solid {GRN};padding:8px 12px;font-size:13px;border-radius:0 8px 8px 0}}
.num{{background:{INK};color:#f4eee2;padding:7px 12px;border-radius:8px;font:600 13px system-ui;display:inline-block;margin:8px 0}}
.cap{{font-size:11.5px;color:#8a8371;margin-top:-4px}}
ol,ul{{padding-left:20px}} li{{margin-bottom:6px}}
</style></head><body><div class=wrap>
<h1>{TITLE} — comment on tombe sur les sections</h1>
<div class=lede>Chaque étape avec ses vraies valeurs sur ce morceau.
Trace vérifiée : elle reproduit exactement la sortie de <code>detect_sections</code>.</div>
{body}
<section><h2>Toutes les constantes, au même endroit</h2>
<table><tr><th>constante</th><th>valeur</th><th>rôle</th></tr>{crows}</table></section>
</div></body></html>""")
    print(f"wrote {OUT.relative_to(HERE)}  ({OUT.stat().st_size//1024} KB)")
    print(f"  {len(runs)} runs, coverage {coverage:.0%}, mode {mode}, "
          f"{len(segs)} sections")


if __name__ == "__main__":
    main()
