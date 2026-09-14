"""Louis's rule: local-baseline margin AND ≥ X% of the initial peak.

« Je pense que la bonne métrique est marge sur le fond local ET il faut que le
pic soit au-dessus d'un certain pourcentage du pic initial → tester plusieurs
seuils, 70 %, 80 %, 90 %… »

The "initial peak" is the curve's value at the reference block itself (c = b0):
the block matched against itself, which is the ceiling by construction. A genuine
other occurrence should reach a good fraction of it. The rule is therefore

    peak survives  ⇔  local-baseline margin  AND  raw[c] ≥ frac · raw[b0]

Both halves matter and do different jobs: the margin says "this is a peak, not a
plateau", the fraction says "this is a strong enough match to be the same music".
Swept over frac, per song, plots and SSM for each.

    python scripts/peak_rule_sweep.py   ->  /reports/peak_rule_sweep.html
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

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from harmonia_min import beats as bts                       # noqa: E402
from ssm_rows_plot import bar_ssm, fig2b64                  # noqa: E402
from pattern_slide import off_diag, dominant_lag            # noqa: E402
from peak_selectors import curves, sel_margin, SONGS        # noqa: E402

OUT = HERE / "harmonia_min/state/reports/peak_rule_sweep.html"
INK = "#1c1c1c"
# 0.90 VALIDATED by Louis on these very plots (2026-08-05); the others
# stay in the sweep so the choice remains visible and re-checkable.
CHOSEN = 0.90
FRACS = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95]
COLS = ["#8a2b2b", "#c58a2e", "#1f8a5b", "#2a6fb0", "#7c3aed", "#0f766e"]


def apply_rule(raw, L, b0, frac, margin=0.5):
    """margin over the local baseline AND >= frac of the initial peak."""
    ref = float(raw[b0])
    cand = sel_margin(raw, L, margin=margin)
    return np.array([c for c in cand if raw[c] >= frac * ref], int), ref


def song(stem, title):
    S, n, segs = bar_ssm(stem)
    raw, cos, cen, L, b0 = curves(S, n)
    x = np.arange(len(raw))
    ref = float(raw[b0])

    # ── the curve with every threshold drawn, peaks coloured by survival ────
    fig, ax = plt.subplots(figsize=(12, 4.0))
    ax.plot(x, raw, lw=1.4, color=INK, zorder=3)
    ax.axvline(b0, color="#8a2b2b", lw=1.6, zorder=2)
    for sg in segs:
        ax.axvline(sg["b0"], color="#1f8a5b", lw=.8, alpha=.30, zorder=1)
    for f, c in zip(FRACS, COLS):
        ax.axhline(f * ref, color=c, ls="--", lw=.9, alpha=.75)
        ax.text(len(raw) * 1.005, f * ref, f"{int(f*100)}%", color=c,
                fontsize=7, va="center")
    kept_by = {}
    base = sel_margin(raw, L, margin=0.5)
    for c_ in base:
        surv = max([f for f in FRACS if raw[c_] >= f * ref], default=None)
        kept_by[int(c_)] = surv
    for c_, surv in kept_by.items():
        if surv is None:
            ax.plot(c_, raw[c_], "x", color="#b9b09a", ms=7, zorder=4)
        else:
            ax.plot(c_, raw[c_], "o", color=COLS[FRACS.index(surv)], ms=8,
                    mfc="none", mew=1.8, zorder=5)
    ax.set_xlabel("décalage sur l'axe x, en mesures  (rouge = le motif de "
                  "référence, verts pâles = sections actuelles)", fontsize=9)
    ax.set_ylabel("produit scalaire brut", fontsize=9)
    ax.set_title("cercle = pic retenu par la marge locale, couleur = seuil le "
                 "plus exigeant qu'il franchit ·  × = rejeté par la marge",
                 fontsize=8.5, loc="left")
    curve = fig2b64(fig)

    # ── one SSM per threshold ───────────────────────────────────────────────
    fig, axs = plt.subplots(1, len(FRACS), figsize=(3.1 * len(FRACS), 3.4))
    counts = []
    for a_, f, c in zip(axs, FRACS, COLS):
        pk, _ = apply_rule(raw, L, b0, f)
        counts.append(len(pk))
        a_.imshow(S, cmap="RdYlBu_r", origin="lower",
                  vmin=np.percentile(S, 5), vmax=np.percentile(S, 99))
        a_.add_patch(plt.Rectangle((b0 - .5, b0 - .5), L, L, fill=False,
                                   ec=INK, lw=2.0))
        for cc in pk:
            a_.add_patch(plt.Rectangle((cc - .5, b0 - .5), L, L, fill=False,
                                       ec=c, lw=1.6, ls="--"))
        a_.set_title(f"≥ {int(f*100)}% — {len(pk)} pics", fontsize=8.5,
                     color=c, loc="left")
        a_.set_xticks([]); a_.set_yticks([])
    mats = fig2b64(fig)

    starts = {s["b0"] for s in segs}
    rows = ""
    for f, c, k in zip(FRACS, COLS, counts):
        pk, _ = apply_rule(raw, L, b0, f)
        onstart = sum(1 for cc in pk if any(abs(cc - s) <= 1 for s in starts))
        gaps = np.diff(sorted(pk)) if len(pk) > 1 else np.array([])
        mult = (all(g % L == 0 for g in gaps) if len(gaps) else None)
        mark = " ← retenu" if abs(f - CHOSEN) < 1e-9 else ""
        rows += (f"<tr><td style='color:{c}'><b>≥ {int(f*100)}%{mark}</b></td>"
                 f"<td>{float(f*ref):.4f}</td><td>{k}</td>"
                 f"<td>{onstart}</td>"
                 f"<td>{sorted(int(g) for g in gaps) if len(gaps) else '—'}</td>"
                 f"<td>{'oui' if mult else ('—' if mult is None else 'non')}</td></tr>")

    return f"""<section><h2>{title}</h2>
<div class=num>{n} mesures &nbsp;·&nbsp; motif de <b>{L} mesures</b> à la mesure
<b>{b0}</b> &nbsp;·&nbsp; pic initial = <b>{ref:.4f}</b></div>
<img src="data:image/png;base64,{curve}">
<h3>Un seuil par colonne</h3>
<img src="data:image/png;base64,{mats}">
<h3>Ce que chaque seuil donne</h3>
<table><tr><th>seuil</th><th>valeur absolue</th><th>pics</th>
<th>dont sur un début de section actuel</th><th>écarts entre pics</th>
<th>tous multiples de {L} ?</th></tr>{rows}</table>
<p class=cap>La colonne « tous multiples de {L} » est un contrôle interne : si le
motif fait {L} mesures, des occurrences réelles devraient s'espacer d'un multiple
de {L}. Un seuil qui casse cette régularité a probablement laissé passer du
bruit.</p></section>"""


def main():
    body, done = "", []
    for stem, title in SONGS:
        if not (HERE / f"docs/audio/{stem}.m4a").exists():
            continue
        cache = HERE / f"harmonia_min/state/beats/{stem}.json"
        if cache.exists():
            d = json.loads(cache.read_text())
            q = bts.grid_quality(d.get("beats") or [], d.get("downbeats") or [])
            if q["n_bars"] >= bts.GRID_MIN_BARS and (
                    q["metre"] != 4 or q["consistency"] < bts.GRID_MIN_CONSISTENCY):
                continue
        body += song(stem, title)
        done.append(title)
        print(f"  ok {title}")
    OUT.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Marge locale + seuil sur le pic initial</title><style>
body{{margin:0;background:#e7e0d0;font:15px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1250px;margin:0 auto;padding:22px 15px 60px}}
h1{{font:italic 600 25px Georgia,serif;margin:0 0 4px}}
.lede{{color:#8a8371;font-size:13px;margin-bottom:22px;max-width:860px}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:16px 18px;margin-bottom:16px}}
h2{{font:700 16px system-ui;margin:0 0 10px;color:#8a2b2b}}
h3{{font:700 12px system-ui;margin:18px 0 6px;color:#8a8371;text-transform:uppercase;letter-spacing:.05em}}
img{{max-width:100%;border-radius:8px}}
.cap{{font-size:12px;color:#8a8371;margin:6px 0 0}}
.num{{background:{INK};color:#f4eee2;padding:6px 11px;border-radius:8px;font:600 12.5px system-ui;display:inline-block;margin-bottom:10px}}
table{{border-collapse:collapse;font-size:12.5px}}
th,td{{border:1px solid #e5dcc6;padding:3px 10px;text-align:left}}
th{{background:#f7f3e9}}
</style></head><body><div class=wrap>
<h1>Marge locale <span style="color:#8a2b2b">ET</span> seuil sur le pic initial</h1>
<div class=lede>Un pic est retenu s'il dépasse d'au moins 0,5 σ la médiane de son
voisinage <b>et</b> s'il atteint au moins X % du pic initial — la valeur du motif
comparé à lui-même, qui est le plafond par construction. Six valeurs de X, de
70 % à 95 %. Les deux moitiés font des choses différentes : la marge dit « c'est
un pic et pas un plateau », le pourcentage dit « la ressemblance est assez forte
pour que ce soit la même musique ». Morceaux : {', '.join(done)}.</div>
{body}</div></body></html>""")
    print(f"wrote {OUT.relative_to(HERE)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
