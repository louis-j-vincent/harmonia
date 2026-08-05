"""The repetition dictionary, built on the locked method. Pictures first.

« Matrice SSM accords harmonie totale + détection de pics, montre-moi le
dictionnaire de répétitions construit avec ça. »

Louis's algorithm, on the substrate he chose and with the rule he validated:

  1. period + phase → the first motif, an L×L block on the diagonal
  2. slide it along X, keep the peaks (local margin AND ≥ 90 % of the initial
     peak) → the other occurrences of that motif → dictionary entry #1
  3. the blocks found are NOT deleted from the matrix. They go to a SEPARATE
     BOX and stay candidates for the later entries.
  4. look for the next motif among the bars no entry has claimed → entry #2, …
  5. at the end, each boxed block goes to the entry that scored it highest —
     but only if that best score clearly stands out. Otherwise it becomes a
     section of its own.

Everything imports `harmonic_method`; nothing is re-implemented here.

    python scripts/dictionary_harmonic.py   ->  /reports/dictionary.html
"""
from __future__ import annotations

import copy
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
from harmonia_min import sections as hs                  # noqa: E402
from ssm_rows_plot import fig2b64                        # noqa: E402
import harmonic_method as HM                             # noqa: E402

OUT = HERE / "harmonia_min/state/reports/dictionary.html"
INK = "#1c1c1c"
ENTRY_COLS = ["#8a2b2b", "#1f8a5b", "#2a6fb0", "#c58a2e", "#7c3aed", "#0f766e"]
CLEAR_MARGIN = 0.15   # "clairement au-dessus" = the winner beats the runner-up
                      # by this much, expressed as a fraction of the winner's
                      # own initial peak. Sensitivity is plotted, not hidden.
SONGS = [("maroon_5_this_love", "This Love"),
         ("norah_jones_don_t_know_why", "Don't Know Why")]


def bar_grid(stem):
    from harmonia_min import pipeline as _pl
    real = hs.detect_sections
    cap = {}

    def spy(grid, arr, times, bars=None):
        out = real(grid, arr, times, bars)
        cap.update(grid=grid, arr=arr, times=times, segs=copy.deepcopy(out))
        return out

    hs.detect_sections = spy
    try:
        _pl.analyze(HERE / f"docs/audio/{stem}.m4a", title="x", file_key="x",
                    audio_url="")
    finally:
        hs.detect_sections = real
    return cap


def build(S, n, max_entries=6, criterion="hybrid"):
    """Louis's loop. Returns (entries, boxed) where an entry is
    {L, b0, curve, occ} and `boxed` lists every occurrence with the score every
    entry gave it."""
    entries, claimed = [], np.zeros(n, bool)
    for _ in range(max_entries):
        free = np.flatnonzero(~claimed)
        if len(free) < 2 * HM.LAG_MIN + 2:
            break
        # Motif search on the free bars — by the LONGEST RUN of consecutive
        # strong bar-matches at some lag, not by the mean and not by a total.
        #
        # Two dead ends recorded so they are not retried (2026-08-05):
        #   * the MEAN over a lag: a lag where 8 bars agree at 0.98 loses to one
        #     where 20 bars average 0.5 — Don't Know Why's B block sat at lag 16
        #     (0.894) and lost to lag 4 (0.981), so it was never proposed;
        #   * the TOTAL evidence above a threshold: biased toward SHORT lags,
        #     which simply have more pairs. It turned This Love's 8-bar chorus
        #     into a 2-bar motif with 15 occurrences and cost 6 bars of coverage.
        #
        # A real block repeat shows as a CONSECUTIVE run of strong matches along
        # one diagonal. That run's LENGTH is the motif length and the lag is the
        # distance to its copy — the two are different quantities, which the old
        # "L = lag" formulation conflated.
        i_all = np.arange(n)
        off_all = S[np.abs(i_all[:, None] - i_all[None, :]) >= HM.LAG_MIN]
        strong = float(np.quantile(off_all, HM.PHASE_QUANTILE))
        use_mean = (criterion == "mean") or (criterion == "hybrid" and not entries)
        if use_mean:
            # ROUND 1 (hybrid) or always (mean): the mean-based period + phase.
            L, b0 = HM.period_and_phase(S)
            best = (L, b0, L)
        elif criterion == "total":
            bt, bL = -1.0, None
            for Lx in range(HM.LAG_MIN, min(HM.LAG_MAX, n - 2) + 1):
                v = [S[b, b + Lx] for b in free
                     if b + Lx < n and not claimed[b + Lx]]
                if len(v) < 2:
                    continue
                sc = float(sum(max(0.0, x - strong) for x in v))
                if sc > bt:
                    bt, bL = sc, Lx
            if bL is None or bt <= 0:
                break
            b0m = next((b for b in free if b + bL < n and S[b, b + bL] >= strong), None)
            if b0m is None:
                break
            best = (bL, b0m, bL)
        else:
            best = (0, None, None)                  # (run length, start, lag)
            for d in range(HM.LAG_MIN, n - HM.LAG_MIN):
                run, start = 0, None
                for b in range(n - d):
                    ok = (not claimed[b] and not claimed[b + d]
                          and S[b, b + d] >= strong)
                    if ok:
                        if run == 0:
                            start = b
                        run += 1
                        if run > best[0]:
                            best = (run, start, d)
                    else:
                        run = 0
        run_len, b0, lag = best
        if run_len < HM.LAG_MIN or b0 is None:
            break
        L = int(run_len)
        curve = HM.slide(S, L, b0)
        # Occurrences must not overlap each other either: a motif of L bars
        # cannot start again L/2 bars later (This Love's 8-bar entry was
        # returning 56, 58, 60, 62 … as separate occurrences). Take them
        # strongest-first and drop anything that collides with a kept one or
        # with a block an earlier entry already owns.
        cand = sorted((int(o) for o in HM.peaks(curve, L, b0)),
                      key=lambda o: -curve[o])
        occ, taken = [], claimed.copy()
        for o in cand:
            if taken[o:min(n, o + L)].any():
                continue
            occ.append(o)
            taken[o:min(n, o + L)] = True
        occ.sort()
        if not occ:
            break
        entries.append({"L": L, "b0": int(b0), "curve": curve, "occ": occ})
        # A detected block leaves the game (Louis, 2026-08-05, revising his
        # earlier "keep them as candidates"): « lorsqu'on a détecté un block,
        # il ne devrait plus être considéré par les autres blocks, il est
        # maintenant affecté à une section et mis dans le dictionnaire donc il
        # n'apparaît plus ». The motif's OWN block is claimed too — without
        # that, b0 stays free and every round rediscovers the same motif
        # (measured: 6 identical entries per song).
        claimed[b0:min(n, b0 + L)] = True
        for o in occ:
            claimed[o:min(n, o + L)] = True
        if claimed.all():
            break

    # every occurrence, scored by EVERY entry — this is the separate box
    boxed = []
    for ei, e in enumerate(entries):
        for o in e["occ"]:
            scores = []
            for ej, f in enumerate(entries):
                c = f["curve"]
                k = min(max(o, 0), len(c) - 1)
                scores.append(float(c[k]) / max(float(c[f["b0"]]), 1e-9))
            order = np.argsort(scores)[::-1]
            win, run = int(order[0]), (int(order[1]) if len(order) > 1 else None)
            margin = scores[win] - (scores[run] if run is not None else 0.0)
            boxed.append({"bar": o, "found_by": ei, "scores": scores,
                          "winner": win if (run is None or margin >= CLEAR_MARGIN) else None,
                          "margin": margin})
    return entries, boxed


def song_html(stem, title):
    cap = bar_grid(stem)
    grid, segs = cap["grid"], cap["segs"]
    n = len(grid) - 1
    S = HM.ssm(HERE / f"docs/audio/{stem}.m4a", grid)
    entries, boxed = build(S, n)

    # one panel per entry: its curve, and the SSM with its occurrences boxed
    figs = []
    for ei, e in enumerate(entries):
        col = ENTRY_COLS[ei % len(ENTRY_COLS)]
        L, b0, curve = e["L"], e["b0"], e["curve"]
        fig, axs = plt.subplots(1, 2, figsize=(13, 3.9),
                                gridspec_kw={"width_ratios": [2.1, 1]})
        axs[0].plot(np.arange(len(curve)), curve, lw=1.3, color=INK)
        axs[0].axhline(HM.INITIAL_PEAK_FRAC * curve[b0], color=col, ls="--", lw=1)
        axs[0].axvline(b0, color=col, lw=1.8)
        axs[0].plot(e["occ"], curve[e["occ"]], "o", color=col, ms=8, mfc="none", mew=1.9)
        for sg in segs:
            axs[0].axvline(sg["b0"], color="#1f8a5b", lw=.7, alpha=.25)
        axs[0].set_xlabel("décalage (mesures)", fontsize=8)
        axs[0].set_title(f"entrée {ei+1} — motif de {L} mesures pris en {b0}, "
                         f"{len(e['occ'])} occurrences", fontsize=9, loc="left", color=col)
        axs[1].imshow(S, cmap="RdYlBu_r", origin="lower",
                      vmin=np.percentile(S, 5), vmax=np.percentile(S, 99))
        axs[1].add_patch(plt.Rectangle((b0 - .5, b0 - .5), L, L, fill=False, ec=INK, lw=2.2))
        for o in e["occ"]:
            axs[1].add_patch(plt.Rectangle((o - .5, b0 - .5), L, L, fill=False,
                                           ec=col, lw=1.7, ls="--"))
        axs[1].set_xticks([]); axs[1].set_yticks([])
        figs.append(fig2b64(fig))

    # coverage strip: which entry owns which bar
    fig, ax = plt.subplots(figsize=(13, 1.5))
    owner = -np.ones(n, int)
    for ei, e in enumerate(entries):
        owner[e["b0"]:min(n, e["b0"] + e["L"])] = ei
        for o in e["occ"]:
            owner[o:min(n, o + e["L"])] = ei
    for b in range(n):
        c = ENTRY_COLS[owner[b] % len(ENTRY_COLS)] if owner[b] >= 0 else "#e5dcc6"
        ax.add_patch(plt.Rectangle((b, 0), 1, 1, color=c))
    for sg in segs:
        ax.axvline(sg["b0"], color=INK, lw=1.0)
    ax.set_xlim(0, n); ax.set_ylim(0, 1); ax.set_yticks([])
    ax.set_xlabel("mesure — couleur = entrée du dictionnaire, gris = non couvert, "
                  "traits noirs = sections actuelles", fontsize=8)
    strip = fig2b64(fig)

    rows = ""
    for bx in boxed:
        sc = " · ".join(f"e{j+1} {s:.2f}" for j, s in enumerate(bx["scores"]))
        verdict = (f"entrée {bx['winner']+1}" if bx["winner"] is not None
                   else "<b>section à part</b> (rien ne se détache)")
        rows += (f"<tr><td>{bx['bar']}</td><td>entrée {bx['found_by']+1}</td>"
                 f"<td>{sc}</td><td>{bx['margin']:.2f}</td><td>{verdict}</td></tr>")

    cover = int((owner >= 0).sum())
    return f"""<section><h2>{title} <span class=sub>{n} mesures ·
{len(entries)} entrée(s) · {cover}/{n} mesures couvertes</span></h2>
{''.join(f'<img src="data:image/png;base64,{g}">' for g in figs)}
<h3>Ce que chaque entrée possède</h3>
<img src="data:image/png;base64,{strip}">
<h3>La case séparée — chaque bloc noté par TOUTES les entrées</h3>
<table><tr><th>mesure</th><th>trouvé par</th><th>scores</th><th>marge</th>
<th>attribué à</th></tr>{rows}</table>
<p class=cap>Un bloc n'est retiré de la matrice par personne : il est noté par
chaque entrée, et attribué à la meilleure seulement si elle devance la suivante
de {CLEAR_MARGIN:.2f} (en fraction du pic initial de l'entrée gagnante).</p>
</section>"""


def main():
    body = ""
    for stem, title in SONGS:
        body += song_html(stem, title)
        print(f"  ok {title}")
    OUT.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Le dictionnaire de répétitions</title><style>
body{{margin:0;background:#e7e0d0;font:15px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1400px;margin:0 auto;padding:22px 15px 60px}}
h1{{font:italic 600 25px Georgia,serif;margin:0 0 4px}}
.lede{{color:#8a8371;font-size:13px;margin-bottom:22px;max-width:900px}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:16px 18px;margin-bottom:16px}}
h2{{font:700 17px system-ui;margin:0 0 10px;color:#8a2b2b}}
.sub{{font:500 12px system-ui;color:#8a8371}}
h3{{font:700 11px system-ui;margin:18px 0 6px;color:#8a8371;text-transform:uppercase;letter-spacing:.05em}}
img{{max-width:100%;border-radius:8px;margin-bottom:8px}}
.cap{{font-size:12px;color:#8a8371}}
table{{border-collapse:collapse;font-size:12.5px}}
th,td{{border:1px solid #e5dcc6;padding:3px 10px;text-align:left}}
th{{background:#f7f3e9}}
</style></head><body><div class=wrap>
<h1>Le dictionnaire de répétitions</h1>
<div class=lede>Sur la matrice que tu as choisie — accords projetés sur les 12
notes, distribution complète — et avec la règle verrouillée : marge locale
<b>et</b> 90 % du pic initial. Le motif est cherché parmi les mesures qu'aucune
entrée n'a encore revendiquées ; les blocs trouvés ne sont <b>jamais effacés</b>
de la matrice, ils restent candidats pour les entrées suivantes.
Les deux morceaux dont la matrice est bonne.</div>
{body}</div></body></html>""")
    print(f"wrote {OUT.relative_to(HERE)} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
