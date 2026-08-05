"""Diagonal reading vs the sliding square — the same block, the same rule.

« Montre-moi une démo du diagonal match comparé à ce qu'on fait actuellement. »

Same reference block, same peak rule (local margin AND ≥ 90 % of the initial
value), two ways of comparing it to the rest of the song:

  SQUARE  (what runs today)  f(c) = mean over the whole L×L block of
                             S[b0+i, c+j] × S[b0+i, b0+j]
  DIAGONAL (the addition)    f(c) = mean over i of S[b0+i, c+i]

The square asks "does the material at c have the same internal texture as the
motif". The diagonal asks "is the material at c the same music, bar for bar".
Those differ most exactly where it matters: a THROUGH-COMPOSED block, whose
internal texture is empty, is invisible to the square and obvious to the
diagonal.

    python scripts/diag_vs_square.py   ->  /reports/diag_vs_square.html
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
from ssm_rows_plot import fig2b64                        # noqa: E402
import harmonic_method as HM                             # noqa: E402
from dictionary_harmonic import bar_grid                 # noqa: E402

OUT = HERE / "harmonia_min/state/reports/diag_vs_square.html"
INK, SQ, DG = "#1c1c1c", "#8a2b2b", "#2a6fb0"
PC = ["C", "D♭", "D", "E♭", "E", "F", "G♭", "G", "A♭", "A", "B♭", "B"]

# (stem, title, [(b0, L, "why this block")])
CASES = [
    ("maroon_5_this_love", "This Love",
     [(0, 4, "le cycle de 4 mesures — se répète À L'INTÉRIEUR de lui-même"),
      (16, 8, "le refrain de 8 mesures")]),
    ("norah_jones_don_t_know_why", "Don't Know Why",
     [(0, 4, "le cycle de 4 mesures"),
      (22, 8, "LE cas : le B, composé de bout en bout, "
              "que le carré ne voit pas")]),
]


def square_curve(S, b0, L):
    n = len(S)
    P = S[b0:b0 + L, b0:b0 + L]
    return np.array([float((P * S[b0:b0 + L, c:c + L]).sum()) / (L * L)
                     for c in range(0, n - L + 1)])


def diag_curve(S, b0, L):
    n = len(S)
    return np.array([HM.diag_match(S, b0, c, L) for c in range(0, n - L + 1)])


def rule(curve, b0, L):
    """The locked rule, applied to whichever curve."""
    return HM.peaks(curve, L, b0)


def case_html(S, n, b0, L, why, bars):
    sq, dg = square_curve(S, b0, L), diag_curve(S, b0, L)
    psq, pdg = rule(sq, b0, L), rule(dg, b0, L)

    fig, axs = plt.subplots(2, 1, figsize=(12.5, 5.0), sharex=True)
    for ax, cv, pk, col, nm in ((axs[0], sq, psq, SQ, "CARRÉ — ce qui tourne"),
                                (axs[1], dg, pdg, DG, "DIAGONALE — la proposition")):
        ax.plot(np.arange(len(cv)), cv, lw=1.4, color=INK)
        ax.axhline(HM.INITIAL_PEAK_FRAC * cv[b0], color=col, ls="--", lw=1.1)
        ax.axvline(b0, color=col, lw=1.8)
        ax.plot(pk, cv[pk], "o", color=col, ms=9, mfc="none", mew=2)
        ax.set_ylabel(nm, fontsize=8.5, color=col)
        ax.set_title(f"{len(pk)} occurrences retenues : {list(pk)}",
                     fontsize=8.5, loc="left", color=col)
    axs[1].set_xlabel("décalage en mesures", fontsize=9)
    curves = fig2b64(fig)

    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.9))
    for ax, pk, col, nm in ((axs[0], psq, SQ, "carré"), (axs[1], pdg, DG, "diagonale")):
        ax.imshow(S, cmap="RdYlBu_r", origin="lower",
                  vmin=np.percentile(S, 5), vmax=np.percentile(S, 99))
        ax.add_patch(plt.Rectangle((b0 - .5, b0 - .5), L, L, fill=False, ec=INK, lw=2.2))
        for c in pk:
            ax.add_patch(plt.Rectangle((c - .5, b0 - .5), L, L, fill=False,
                                       ec=col, lw=1.7, ls="--"))
        ax.set_title(f"{nm} — {len(pk)} trouvés", fontsize=9, color=col, loc="left")
        ax.set_xticks([]); ax.set_yticks([])
    mats = fig2b64(fig)

    def chords(a):
        out = []
        for i in range(a, min(a + L, len(bars))):
            bb = bars[i]
            out.append("%" if not bb else " ".join(
                "N.C." if c["nc"] else PC[c["root"]] + c["q"] for c in bb))
        return " | ".join(out)

    rows = "".join(
        f"<tr><td>mes. {c+1}–{c+L}</td><td>{sq[c]:.3f}</td><td>{dg[c]:.3f}</td>"
        f"<td>{'carré' if c in psq else ''} {'diagonale' if c in pdg else ''}</td>"
        f"<td class=ch>{chords(c)}</td></tr>"
        for c in sorted(set(list(psq) + list(pdg))))

    return f"""<h3>Motif de {L} mesures pris en mesure {b0+1} — {why}</h3>
<div class=ch style="margin-bottom:8px">{chords(b0)}</div>
<img src="data:image/png;base64,{curves}">
<img src="data:image/png;base64,{mats}">
<table><tr><th>bloc</th><th>score carré</th><th>score diagonale</th>
<th>retenu par</th><th>accords</th></tr>{rows}</table>"""


def main():
    body = ""
    for stem, title, cases in CASES:
        cap = bar_grid(stem)
        grid = cap["grid"]
        n = len(grid) - 1
        S = HM.ssm(HERE / f"docs/audio/{stem}.m4a", grid)
        from harmonia_min import sections as hs
        import copy
        from harmonia_min import pipeline as _pl
        real = hs.detect_sections
        cp = {}

        def spy(g, a, t, bars=None):
            cp["bars"] = copy.deepcopy(bars)
            return real(g, a, t, bars)

        hs.detect_sections = spy
        try:
            _pl.analyze(HERE / f"docs/audio/{stem}.m4a", title="x",
                        file_key="x", audio_url="")
        finally:
            hs.detect_sections = real
        bars = cp.get("bars") or [[] for _ in range(n)]
        inner = "".join(case_html(S, n, b0, L, why, bars) for b0, L, why in cases)
        body += f"<section><h2>{title}</h2>{inner}</section>"
        print(f"  ok {title}")
    OUT.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Diagonale contre carré</title><style>
body{{margin:0;background:#e7e0d0;font:15px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1250px;margin:0 auto;padding:22px 15px 60px}}
h1{{font:italic 600 25px Georgia,serif;margin:0 0 4px}}
.lede{{color:#8a8371;font-size:13px;margin-bottom:22px;max-width:900px}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:16px 18px;margin-bottom:16px}}
h2{{font:700 17px system-ui;margin:0 0 10px;color:#8a2b2b}}
h3{{font:700 12px system-ui;margin:20px 0 4px;color:#1c1c1c}}
img{{max-width:100%;border-radius:8px;margin-bottom:8px}}
table{{border-collapse:collapse;font-size:12.5px;margin-top:6px}}
th,td{{border:1px solid #e5dcc6;padding:3px 9px;text-align:left}}
th{{background:#f7f3e9}}
.ch{{font:500 11.5px ui-monospace,monospace;color:#8a8371}}
</style></head><body><div class=wrap>
<h1>La diagonale contre le carré</h1>
<div class=lede>Même bloc de référence, même règle de pics (marge locale
<b>et</b> 90 % de la valeur initiale), deux façons de le comparer au reste du
morceau. Le <b>carré</b> demande « ce passage a-t-il la même texture interne que
le motif ». La <b>diagonale</b> demande « est-ce la même musique, mesure par
mesure ». Ils divergent exactement là où ça compte : un bloc composé de bout en
bout, dont la texture interne est vide, est invisible au carré et évident à la
diagonale.</div>
{body}</div></body></html>""")
    print(f"wrote {OUT.relative_to(HERE)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
