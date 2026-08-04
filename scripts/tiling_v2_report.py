"""Build /reports/tiling_v2.html — Louis's four directives on the tiling runs,
seen on This Love, Don't Know Why and Sunny, and scored on 285 Billboard tracks.

Shows per song: the UN-BLURRED SSM, the runs today vs the runs after the
minimal-loop snap, where the novelty peaks moved a boundary, and the resulting
sections side by side with what `harmonia_min/sections.py` produces.

Captures the pipeline's REAL grid/bars/segments by spying on `detect_sections`
during a live `analyze()` (same trick as `scripts/bibar_report.py`).

    python scripts/tiling_v2_report.py
"""
from __future__ import annotations

import base64
import copy
import io
import json
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
from harmonia_min import sections as hs                     # noqa: E402
from harmonia_min.folding import _bar_vecs                  # noqa: E402
from tiling_v2 import TILE_MIN, chg, minimal_loop, tiling_runs_v2  # noqa: E402

CACHE = Path("/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-"
             "Code-harmonia/29e6c8ff-69c3-4685-aae3-2c46f129bcde/scratchpad")
OUT = HERE / "harmonia_min/state/reports/tiling_v2.html"
SONGS = [("maroon_5_this_love", "Maroon 5 — This Love"),
         ("norah_jones_don_t_know_why", "Norah Jones — Don't Know Why"),
         ("bobby_hebb_sunny_official_audio", "Bobby Hebb — Sunny")]
INK, GRN, ACC, BLU, SAND = "#1c1c1c", "#1f8a5b", "#8a2b2b", "#2a6fb0", "#8a8371"


def fig2b64(fig):
    b = io.BytesIO()
    fig.savefig(b, format="png", dpi=108, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(b.getvalue()).decode()


def capture(stem, title):
    from harmonia_min import pipeline as _pl
    cap, real = {}, hs.detect_sections

    def spy(grid, arr, times, bars=None):
        out = real(grid, arr, times, bars)
        cap.update(grid=grid, arr=np.asarray(arr), times=times,
                   bars=copy.deepcopy(bars), segs=copy.deepcopy(out))
        return out

    hs.detect_sections = spy
    try:
        _pl.analyze(HERE / f"docs/audio/{stem}.m4a", title=title,
                    file_key=f"min_{stem}", audio_url=f"/audio/{stem}.m4a")
    finally:
        hs.detect_sections = real
    return cap


def nov_bars(S, blur_sigma):
    Sx = hs._blur(S, blur_sigma) if blur_sigma > 0 else S
    return hs._novelty(Sx, hs.KERNEL_HB)[::2]


def peaks(nv, nb, edge=8):
    """`sections.py`'s own peak rule on BAR grain. `edge` = the half-kernel
    (KERNEL_HB/2 bars): outside it the kernel is truncated and the values are
    artifacts, so they are masked exactly as the shipped code masks them."""
    if nb < 2 * edge + 4:
        return []
    inner = nv[edge:nb - edge]
    thr = max(0.5 * inner.max(), inner.mean() + 0.5 * inner.std())
    return [i for i in range(edge, nb - edge)
            if nv[i] == nv[max(0, i - 2):i + 3].max() and nv[i] >= thr]


def run_strip(ax, runs, nb, key, colour, label):
    for r in runs:
        a, b = r["b0"], r["b1"]              # v1 dicts carry b0/b1 only
        ax.add_patch(plt.Rectangle((a - .5, 0), b - a + 1, 1,
                                   fc=colour, ec="white", lw=.8, alpha=.85))
        if b - a >= 3:
            ax.text((a + b) / 2, .5,
                    f"P{r['period']}"
                    + (f"/L{r['L']}" if r.get("L") else ""),
                    ha="center", va="center", color="white",
                    fontsize=7.5, weight="bold")
    ax.set_xlim(-.5, nb - .5)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=8.5)


def one_song(stem, title):
    cap = capture(stem, title)
    grid, arr, times = cap["grid"], cap["arr"], cap["times"]
    segs, nb = cap["segs"], len(cap["grid"]) - 1
    F = hs.halfbar_features(grid, arr, times)
    S = F @ F.T
    Vb = _bar_vecs(F, nb)
    nv_raw, nv_blur = nov_bars(S, 0.0), nov_bars(S, hs.BLUR_SIGMA)
    pk = peaks(nv_raw, nb)
    v1 = hs.tiling_runs(Vb, nb)
    v2 = tiling_runs_v2(Vb, nb, nv_raw, 0.0, TILE_MIN)

    # ── fig 1: un-blurred vs blurred SSM, half-bar grain ────────────────────
    fig, axs = plt.subplots(1, 2, figsize=(11.4, 5.5))
    for ax, M, t in ((axs[0], S, "SSM brute (non floutée)"),
                     (axs[1], hs._blur(S, hs.BLUR_SIGMA),
                      f"SSM floutée σ={hs.BLUR_SIGMA} (ce qu'on utilise)")):
        ax.imshow(M, cmap="magma", origin="lower", vmin=np.percentile(M, 5),
                  vmax=1.0, interpolation="nearest")
        ax.set_title(t, fontsize=10)
        ax.set_xticks(np.arange(0, 2 * nb, 16))
        ax.set_xticklabels((np.arange(0, 2 * nb, 16) // 2).astype(int),
                           fontsize=7)
        ax.set_yticks([])
        for s in segs[1:]:
            ax.axhline(2 * s["b0"], color="#7fdcff", lw=.7, alpha=.8)
            ax.axvline(2 * s["b0"], color="#7fdcff", lw=.7, alpha=.8)
    f_ssm = fig2b64(fig)

    # ── fig 2: runs v1 vs v2, novelty, sections ─────────────────────────────
    fig, axs = plt.subplots(5, 1, figsize=(11.4, 5.4), sharex=True,
                            gridspec_kw=dict(height_ratios=[1, 1, 1.9, 1, 1],
                                             hspace=.34))
    run_strip(axs[0], v1, nb, "v1", BLU, "runs\naujourd'hui")
    run_strip(axs[1], v2, nb, "v2", GRN, "runs après\nla boucle")
    ax = axs[2]
    # the first/last 8 bars see a TRUNCATED checkerboard kernel — their values
    # are artifacts, `sections.py` masks them, and left in they flatten the
    # whole interior curve to a line. Masked here for the same reason.
    m = slice(8, max(9, nb - 8))
    x = np.arange(nb)[m]

    def _zi(v):
        w = v[m]
        return (w - w.mean()) / max(w.std(), 1e-9)
    z = np.full(nb, np.nan)
    z[m] = _zi(nv_raw)
    ax.plot(x, _zi(nv_blur), color=SAND, lw=1.1, label="nouveauté floutée")
    ax.plot(x, z[m], color=ACC, lw=1.5, label="nouveauté brute")
    pin = [p for p in pk if 8 <= p < nb - 8]
    ax.plot(pin, z[pin], "o", ms=5, color=ACC)
    ax.legend(fontsize=7, frameon=False, loc="upper right", ncol=2)
    ax.set_ylabel("pics", rotation=0, ha="right", va="center", fontsize=8.5)
    ax.tick_params(labelsize=7)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for name, src, colour, axi in (("sections.py", segs, "#c9b98a", 3),
                                   ("runs recalés\n+ pics", None, "#a8cbb0", 4)):
        a = axs[axi]
        if src is None:
            # the RAW union of the two cut sources — Louis's directive (1) with
            # nothing else applied. Shown as-is: it is what scores best on the
            # corpus (boundary F 0.281 vs 0.232 runs-only), and it visibly
            # over-segments. Both facts belong in the picture.
            cuts = sorted({0, nb} | {int(r["b0"]) for r in v2}
                          | {int(r["b1"]) + 1 for r in v2} | set(pk))
            cuts = [c for c in cuts if 0 <= c <= nb]
            keep = [cuts[0]]
            for c in cuts[1:]:
                if c - keep[-1] >= 2:
                    keep.append(c)
            src = [{"b0": x, "b1": y - 1, "label": ""}
                   for x, y in zip(keep, keep[1:])]
        for i, s in enumerate(src):
            a.add_patch(plt.Rectangle((s["b0"] - .5, 0),
                                      s["b1"] - s["b0"] + 1, 1,
                                      fc=colour if i % 2 == 0 else "#e7e0d0",
                                      ec="white", lw=.9))
            a.text((s["b0"] + s["b1"]) / 2, .5,
                   s.get("label") or str(s["b1"] - s["b0"] + 1),
                   ha="center", va="center", fontsize=8, color=INK)
        a.set_xlim(-.5, nb - .5)
        a.set_ylim(0, 1)
        a.set_yticks([])
        a.set_ylabel(name, rotation=0, ha="right", va="center", fontsize=8.5)
    axs[-1].set_xlabel("mesure", fontsize=8.5)
    axs[-1].tick_params(labelsize=7)
    f_runs = fig2b64(fig)

    # ── the per-run table: what the minimal-loop rule actually did ──────────
    rows = []
    for r in v2:
        Lv1 = r["b1_v1"] - r["b0_v1"] + 1
        ok = r["L"] >= 2 and Lv1 % r["L"] == 0
        rows.append(dict(
            v1=f"{r['b0_v1']}–{r['b1_v1']} ({Lv1} mes.)", P=r["period"],
            L=r["L"] or "—", mult="oui" if ok else "NON",
            phase="—" if r["phase"] is None else r["phase"],
            v2=f"{r['b0']}–{r['b1']} ({r['b1']-r['b0']+1} mes.)",
            moved=(r["b0"], r["b1"]) != (r["b0_v1"], r["b1_v1"])))
    gl, gs = minimal_loop(Vb, 0, nb - 1)
    return dict(title=title, nb=nb, f_ssm=f_ssm, f_runs=f_runs, rows=rows,
                n_v1=len(v1), n_v2=len(v2), pk=pk, segs=len(segs),
                gl=gl, gs=gs,
                mult_ok=sum(r["mult"] == "oui" for r in rows),
                mult_n=sum(r["L"] != "—" for r in rows))


CSS = """
*{box-sizing:border-box} body{margin:0;background:#e7e0d0;color:#1c1c1c;
 font:15.5px/1.62 -apple-system,BlinkMacSystemFont,system-ui,sans-serif}
.wrap{max-width:1000px;margin:0 auto;padding:26px 16px 60px}
h1{font:italic 600 28px Georgia,serif;margin:0 0 4px}
.lede{color:#8a8371;font-size:13.5px;margin-bottom:26px}
h2{font:italic 600 21px Georgia,serif;margin:34px 0 8px;
   border-top:1px solid #d8cfb8;padding-top:18px}
h3{font:700 10.5px system-ui;letter-spacing:.08em;text-transform:uppercase;
   color:#8a8371;margin:22px 0 8px}
.card{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;
      padding:16px 18px;margin:12px 0}
img{max-width:100%;display:block;border-radius:8px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}
th{text-align:left;font:700 9.5px system-ui;letter-spacing:.06em;
   text-transform:uppercase;color:#8a8371;border-bottom:1px solid #d8cfb8;
   padding:5px 8px}
td{padding:5px 8px;border-bottom:1px solid #efe8d6}
tr:hover td{background:#faf6ea}
.win{color:#1f8a5b;font-weight:700} .lose{color:#8a2b2b;font-weight:700}
.big{font:700 30px Georgia,serif}
.verdict{border-left:4px solid #8a8371;padding:2px 0 2px 14px;margin:14px 0}
.verdict.t{border-color:#1f8a5b} .verdict.f{border-color:#8a2b2b}
.verdict .h{font:700 11px system-ui;letter-spacing:.07em;text-transform:uppercase}
.t .h{color:#1f8a5b} .f .h{color:#8a2b2b}
.scroll{overflow-x:auto}
code{background:#f2ecdc;padding:1px 5px;border-radius:4px;font-size:12.5px}
"""


def main():
    fin = json.load(open(CACHE / "tiling_v2.json"))
    bnd = json.load(open(CACHE / "tiling_v2_bound.json"))
    rowsd = fin["rows"]

    class _P(dict):
        def __missing__(self, k):
            hit = [r for r in rowsd if k in r["name"]]
            if not hit:
                raise KeyError(k)
            return hit[0]
    P = _P()
    songs = [one_song(s, t) for s, t in SONGS]

    def row(label, key, cls=""):
        r = P[key]
        return (f"<tr><td>{label}</td><td class='{cls}'>{r['exact']:.1f}%</td>"
                f"<td>{r['pm1']:.1f}%</td><td>{r['med']:.0f}</td></tr>")

    h = [f"<!DOCTYPE html><html lang='fr'><head><meta charset='utf-8'>"
         f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
         f"<title>Tuilage v2 — tes quatre consignes, mesurées</title>"
         f"<style>{CSS}</style></head><body><div class='wrap'>",
         "<h1>Tuilage v2 — tes quatre consignes, mesurées</h1>",
         "<div class='lede'>285 morceaux Billboard, 2781 débuts de section "
         "annotés, égalités tranchées AU HASARD. Le témoin « ne bouge jamais » "
         "est à côté de chaque chiffre : c'est lui qui avait donné le faux "
         "75,2 % d'hier.</div>"]

    # ── headline ────────────────────────────────────────────────────────────
    h.append("<div class='card'><h3>Le chiffre</h3>"
             "<div class='scroll'><table>"
             "<tr><th>indice de placement</th><th>bonne mesure</th>"
             "<th>±1 mesure</th><th>err. médiane</th></tr>"
             + row("témoin : ne bouge jamais",
                   "trivial: never move (flat score)")
             + row("nouveauté chroma floutée — <b>ce qu'on livre</b>",
                   "chroma novelty, BLURRED sigma=1.5 (SHIPPED)")
             + row("séries de tuilage seules (hier)",
                   "lag-runs, continuous (yesterday)")
             + row("séries + pics, nouveauté floutée (hier)",
                   "FUSION lag-runs + blurred novelty (yesterday, 40.2%)")
             + row("<b>séries + pics, nouveauté NON floutée</b> (consignes 1+3)",
                   "FUSION lag-runs + UN-blurred novelty", "win")
             + row("… + contrainte de boucle minimale (consigne 2)",
                   "v2  fusion + 0.25*loop-congruence (NO blur)")
             + "</table></div>"
             "<p style='margin:10px 0 0;font-size:13.5px'>Deux de tes quatre "
             "consignes paient (1 et 3) : <b>41,8 %</b> contre 40,2 % hier et "
             "27,6 % aujourd'hui en production, pour un hasard à 10,8 %. "
             "Les deux autres (2 et 4) ne paient pas — détail plus bas, "
             "avec les chiffres.</p></div>")

    # ── the four verdicts ───────────────────────────────────────────────────
    h.append("<h2>Tes quatre consignes, une par une</h2>")

    h.append("<div class='card'><div class='verdict t'><div class='h'>"
             "Consigne 1 — les séries ET les pics ensemble · VRAIE</div>"
             "« pas l'un ou l'autre mais les deux ensemble »</div>"
             "<p>Mesuré des deux façons. En <b>placement</b> : séries seules "
             f"{P['lag-runs, continuous (yesterday)']['exact']:.1f} %, pics "
             "seuls "
             f"{P['chroma novelty, UN-BLURRED (Louis 3)']['exact']:.1f} %, "
             "les deux additionnés "
             f"<span class='win'>{P['FUSION lag-runs + UN-blurred novelty']['exact']:.1f} %</span>"
             " — la somme bat chacun des deux, de 6 points. En "
             "<b>détection de frontières</b> (F, tolérance 0 mesure) : arêtes "
             f"de séries {bnd['bound']['0']['v1']:.3f}, pics "
             f"{bnd['bound']['0']['pk']:.3f}, réunion "
             f"<span class='win'>{bnd['bound']['0']['v1+pk']:.3f}</span>. "
             "Aujourd'hui <code>detect_sections</code> choisit un mode et jette "
             "les pics là où un run couvre : c'est cette réunion qu'il jette."
             "</p></div>")

    g, gn = fin["multiple_0.0"]["gt"]
    r_, rn = fin["multiple_0.0"]["runs"]
    ph = bnd["phase"]
    h.append("<div class='card'><div class='verdict f'><div class='h'>"
             "Consigne 2 — la boucle minimale · PRÉMISSE VRAIE, RÈGLE NEUTRE "
             "À NÉGATIVE</div>"
             "« un carré de tuilage est forcément un multiple de sa boucle "
             "minimale »</div>"
             f"<p><b>Tu as raison sur les faits.</b> Sur les 285 morceaux, "
             f"une section annotée a une longueur multiple de sa propre boucle "
             f"minimale <span class='win'>{100*g/gn:.1f} %</span> du temps "
             f"({g}/{gn}) — alors que nos runs actuels ne respectent la règle "
             f"que <span class='lose'>{100*r_/rn:.1f} %</span> du temps "
             f"({r_}/{rn}). Tes 22/40 sur nos 6 grilles se reproduisent à "
             f"l'échelle du corpus. La règle décrit bien la vérité, et notre "
             f"code la viole.</p>"
             f"<p><b>Mais l'appliquer ne fait pas gagner.</b> Recaler chaque "
             f"carré sur un multiple de sa boucle déplace 49 % des runs et "
             f"fait <i>baisser</i> le F des frontières : "
             f"{bnd['bound']['0']['v1']:.3f} → "
             f"<span class='lose'>{bnd['bound']['0']['v2']:.3f}</span> "
             f"(tolérance 0), {bnd['bound']['1']['v1']:.3f} → "
             f"{bnd['bound']['1']['v2']:.3f} (±1). En prior de placement, elle "
             f"est au mieux neutre : "
             f"{P['FUSION lag-runs + UN-blurred novelty']['exact']:.1f} % sans, "
             f"{P['v2  fusion + 0.25*loop-congruence (NO blur)']['exact']:.1f} % "
             f"avec un poids faible, et ça descend continûment quand on "
             f"augmente le poids "
             f"({P['v2  fusion + 2.0*loop-congruence (NO blur)']['exact']:.1f} % "
             f"à poids 2).</p>"
             f"<p><b>Pourquoi.</b> La phase de boucle est du vrai signal : un "
             f"début de section annoté tombe sur la phase qu'on détecte "
             f"<b>{100*ph['hit']/ph['tot']:.1f} %</b> du temps, contre "
             f"{100*ph['chance']:.1f} % au hasard. Mais l'information est "
             f"<i>déjà</i> dans les séries + pics : même avec une phase "
             f"ORACLE lue sur la vraie annotation, on plafonne à "
             f"{P['ORACLE phase']['exact']:.1f} %"
             f" — <i>en dessous</i> des 41,8 % sans contrainte. La contrainte "
             f"n'ajoute rien parce qu'elle redit ce que les séries disent "
             f"déjà, en plus grossier.</p>"
             f"<p>Les boucles détectées aux débuts de section : "
             + ", ".join(f"L={k} → {v} fois" for k, v in ph["L"].items())
             + " — donc surtout des boucles de 2 mesures, où la contrainte "
               "n'élimine qu'un candidat sur deux.</p></div>")

    b0 = fin["blur"]["0.0"]["auc"]
    b15 = fin["blur"]["1.5"]["auc"]
    h.append("<div class='card'><div class='verdict t'><div class='h'>"
             "Consigne 3 — enlever le flou gaussien · VRAIE (mais pas par le "
             "mécanisme que tu donnes)</div>"
             "« le flou nous empêche de distinguer deux sections similaires "
             "mais différentes »</div>"
             f"<p><b>Enlever le flou paie, nettement, sur le placement.</b> "
             f"Nouveauté seule : "
             f"{P['chroma novelty, BLURRED sigma=1.5 (SHIPPED)']['exact']:.1f} % "
             f"floutée → <span class='win'>"
             f"{P['chroma novelty, UN-BLURRED (Louis 3)']['exact']:.1f} %</span> "
             f"brute (+2,5 points). En fusion : 40,2 % → "
             f"<span class='win'>41,8 %</span> (+1,6). C'est la moitié du gain "
             f"total de la journée, pour la suppression d'une constante.</p>"
             f"<p><b>Ton mécanisme, lui, ne se vérifie pas.</b> Tu dis que le "
             f"flou casse la distinction entre deux sections similaires mais "
             f"différentes. Mesuré directement — AUC qui sépare les paires de "
             f"sections de MÊME lettre des paires de lettres différentes, sur "
             f"la statistique de lettre de <code>sections.py</code> "
             f"elle-même : {b15:.3f} floutée → {b0:.3f} brute. "
             f"<b>+0,007.</b> Le flou ne détruit pas les lettres ; il floute "
             f"la position du pic, et c'est là qu'il coûte. Bonne conclusion, "
             f"autre cause.</p>"
             "<div class='scroll'><table><tr><th>σ du flou</th>"
             "<th>AUC lettres (même vs différente)</th></tr>"
             + "".join(f"<tr><td>{k}</td><td>{v['auc']:.3f}</td></tr>"
                       for k, v in fin["blur"].items())
             + "</table></div></div>")

    dec = fin["decay"]
    h.append("<div class='card'><div class='verdict f'><div class='h'>"
             "Consigne 4 — similarité qui se relâche avec la distance · "
             "FAUSSE</div>"
             "« plus on s'éloigne dans le temps entre les répétitions, moins "
             "elles se ressemblent »</div>"
             "<p><b>La décroissance existe mais elle est minuscule.</b> "
             "Mesurée sur les vraies répétitions (mesures de même position "
             "dans deux sections de même lettre annotée) :</p>"
             "<div class='scroll'><table><tr><th>écart entre les deux "
             "répétitions</th><th>cosinus moyen</th><th>n</th></tr>"
             + "".join(f"<tr><td>{2**int(k)}–{2**(int(k)+1)-1} mesures</td>"
                       f"<td>{v[0]:.3f}</td><td>{v[1]}</td></tr>"
                       for k, v in dec.items())
             + "</table></div>"
             "<p>De 0,896 à 0,862 sur un facteur 16 de distance : 3,4 points, "
             "et non monotone au bout. Le seuil <code>TILE_MIN = 0,80</code> "
             "est très en dessous de <i>toutes</i> ces valeurs — il ne coupe "
             "pas les répétitions lointaines, donc il n'y a rien à relâcher."
             "</p><p><b>Et le relâcher fait perdre.</b> Avec un seuil qui "
             "décroît en log de l'écart :</p><div class='scroll'><table>"
             "<tr><th>β (relâchement)</th><th>bonne mesure</th></tr>"
             + "".join(f"<tr><td>{b}</td><td>{P[k]['exact']:.1f}%</td></tr>"
                       for b, k in ((0.0, "v2  fusion + 2.0*loop-congruence "
                                          "(NO blur)"),
                                    (0.02, "v2  w=2.0, beta=0.02 (floor "
                                           "relaxes with lag)"),
                                    (0.04, "v2  w=2.0, beta=0.04 (floor "
                                           "relaxes with lag)"),
                                    (0.08, "v2  w=2.0, beta=0.08 (floor "
                                           "relaxes with lag)")))
             + "</table></div></div>")

    # ── the three songs ─────────────────────────────────────────────────────
    h.append("<h2>Les trois morceaux</h2>")
    for s in songs:
        h.append(f"<div class='card'><h3>{s['title']} — {s['nb']} mesures, "
                 f"{s['segs']} sections aujourd'hui</h3>"
                 f"<p style='font-size:13.5px;color:#8a8371;margin:0 0 10px'>"
                 f"Boucle minimale du morceau entier : "
                 f"{'L=' + str(s['gl']) if s['gl'] else 'aucune'} "
                 f"(score {s['gs']:.3f}). Runs : {s['n_v1']} aujourd'hui, "
                 f"{s['n_v2']} en v2, dont {s['mult_ok']}/{s['mult_n']} "
                 f"étaient déjà un multiple de leur boucle.</p>"
                 f"<img src='data:image/png;base64,{s['f_ssm']}'>"
                 f"<p style='font-size:12.5px;color:#8a8371;margin:8px 0 14px'>"
                 f"À gauche la SSM brute que tu demandes, à droite la floutée "
                 f"qu'on utilise. Les traits bleus sont les sections actuelles."
                 f"</p>"
                 f"<img src='data:image/png;base64,{s['f_runs']}'>"
                 f"<p style='font-size:12.5px;color:#8a8371;margin:8px 0 0'>"
                 f"Bandeaux : runs d'aujourd'hui (bleu), runs après recalage "
                 f"sur la boucle minimale (vert, <code>P</code> = période du "
                 f"run, <code>L</code> = boucle trouvée dedans), puis la "
                 f"nouveauté brute (rouge, pics marqués) contre la floutée, "
                 f"puis les sections de <code>sections.py</code> et l'union "
                 f"brute runs-recalés + pics. Deux honnêtetés : les deux "
                 f"courbes de nouveauté se superposent presque — le flou coûte "
                 f"en <i>position</i> du pic, pas en forme, et ça ne se voit "
                 f"pas à l'œil ; et l'union brute sur-segmente visiblement, "
                 f"alors qu'elle a le meilleur F du corpus (0,281 contre 0,232 "
                 f"pour les runs seuls). Les deux faits sont dans l'image.</p>"
                 "<div class='scroll'><table><tr><th>run aujourd'hui</th>"
                 "<th>P</th><th>boucle L</th><th>déjà multiple ?</th>"
                 "<th>phase</th><th>run recalé</th></tr>"
                 + "".join(
                     f"<tr><td>{r['v1']}</td><td>{r['P']}</td><td>{r['L']}</td>"
                     f"<td class=\"{'win' if r['mult']=='oui' else 'lose'}\">"
                     f"{r['mult']}</td><td>{r['phase']}</td>"
                     f"<td>{r['v2']}{' ←' if r['moved'] else ''}</td></tr>"
                     for r in s["rows"])
                 + "</table></div></div>")

    h.append("<h2>Ce que je ferais</h2><div class='card'>"
             "<p><b>À prendre :</b> enlever le flou (consigne 3) et additionner "
             "séries + pics au lieu de choisir un mode (consigne 1). Ensemble : "
             "27,6 % → 41,8 % sur la bonne mesure. C'est une constante à "
             "supprimer et un <code>if</code> à remplacer par une somme.</p>"
             "<p><b>À laisser :</b> le recalage sur la boucle minimale "
             "(consigne 2) et le seuil qui se relâche (consigne 4). Le premier "
             "décrit correctement la vérité (84,6 % des sections annotées) mais "
             "n'ajoute rien à ce que les séries disent déjà — même avec une "
             "phase oracle. Le second n'a rien à corriger : la décroissance "
             "réelle est de 3 points de cosinus, très au-dessus du seuil.</p>"
             "<p><b>Ce qui reste ouvert :</b> la consigne 2 pourrait payer "
             "ailleurs qu'en placement — comme garde-fou de <i>longueur</i> "
             "quand on écrit la grille (une section de 7 mesures avec une "
             "boucle de 2, c'est presque sûrement une erreur d'écriture, même "
             "si la corriger ne déplace pas le début).</p></div>")

    h.append("</div></body></html>")
    OUT.write_text("\n".join(h), encoding="utf-8")
    print("wrote", OUT, f"({OUT.stat().st_size/1024:.0f} kB)")


if __name__ == "__main__":
    main()
