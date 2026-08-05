"""Comment décider que deux sections sont LA MÊME — les règles candidates.

    python scripts/merge_rules.py   ->  /reports/merge_rules.html

Louis, 2026-08-05, avec sa vérité terrain :
  * The Walk        → 2 sections (le vamp A/Bm7, et le C#m7/Bm7)
  * Don't Know Why  → 2 sections
  * This Love       → 3 sections (B et D sont la même)

Ce que la mesure dit, avant toute proposition :

1. **Le quantile n'y est pour rien.** Les trois erreurs sont des LETTRES, pas
   des frontières — les frontières sont déjà bonnes. Rien dans le dictionnaire
   ne vérifie, à la fin, si deux lettres désignent la même musique. Il n'y a
   pas de passe de fusion.
2. **Otsu / la vallée entre les deux modes (l'idée de Louis) : mesurée, elle
   échoue.** La vallée tombe à 0,576 / 0,618 / 0,634 sur les trois morceaux.
   Ce seuil-là sépare « sans rapport » de « apparenté » — pas « la même » de
   « pas la même ». Fusionner à 0,6 écrase les trois morceaux en une lettre.
   Les deux modes existent, mais le second contient À LA FOIS les vraies
   répétitions et tout ce qui partage simplement la tonalité.
3. **Comparer le CONTENU (la moyenne des notes de la section) sur-fusionne.**
   C'est exactement le mode d'échec de l'ancien détecteur chroma, déjà
   documenté : « chaque segment de This Love est du do mineur → une lettre
   pour tout le morceau ».

Les règles sont donc croisées sur deux axes : comment on absorbe les restes
trop courts, et comment on compare deux sections.
"""
from __future__ import annotations

import copy
import itertools
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
from ssm_rows_plot import fig2b64                                  # noqa: E402
import harmonic_method as HM                                       # noqa: E402
from harmonia_min import sections as hs, musx as mx                # noqa: E402
from harmonia_min.harmonic_sections import (                       # noqa: E402
    build_dictionary, sections_from, off_diagonal, harmonic_vectors)

INK = "#1c1c1c"
COLS = ["#8a2b2b", "#1f8a5b", "#2a6fb0", "#c58a2e", "#7c3aed", "#0f766e",
        "#be123c", "#0369a1"]
NAMES = "C Db D Eb E F Gb G Ab A Bb B".split()
TARGET = {"mayer_hawthorne_the_walk": (2, "le vamp A/Bm7, puis le C#m7/Bm7"),
          "norah_jones_don_t_know_why": (2, "deux sections"),
          "maroon_5_this_love": (3, "trois sections — B et D sont la même")}


# ── les deux axes ───────────────────────────────────────────────────────────
def sc_strict(S, V, a, b):
    """Mesure à mesure, sans décalage : la 1re mesure de l'une contre la 1re de
    l'autre. 1,00 = exactement la même harmonie, mesure pour mesure."""
    L = min(a["b1"] - a["b0"] + 1, b["b1"] - b["b0"] + 1)
    return HM.diag_match(S, a["b0"], b["b0"], L)


def sc_shift(S, V, a, b):
    """Le meilleur alignement possible entre les deux, le décalage restant À
    L'INTÉRIEUR des deux sections. La contrainte n'est pas cosmétique : un
    décalage libre sort de la section, mord sur la voisine et invente des 1,00
    (The Walk : B contre D montait à 1,00 comme ça)."""
    La, Lb = a["b1"] - a["b0"] + 1, b["b1"] - b["b0"] + 1
    L = min(La, Lb)
    return max(HM.diag_match(S, a["b0"] + sa, b["b0"] + sb, L)
               for sa in range(La - L + 1) for sb in range(Lb - L + 1))


def sc_content(S, V, a, b):
    """Les notes moyennes de chaque section, comparées — l'ordre disparaît."""
    ca, cb = V[a["b0"]:a["b1"] + 1].mean(0), V[b["b0"]:b["b1"] + 1].mean(0)
    return float(ca @ cb / max(np.linalg.norm(ca) * np.linalg.norm(cb), 1e-9))


SCORERS = {"mesure à mesure": sc_strict, "meilleur alignement": sc_shift,
           "contenu (notes moyennes)": sc_content}


def otsu(x, bins=256):
    """La vallée entre les deux modes de la distribution (Otsu 1979)."""
    h, e = np.histogram(x, bins=bins)
    c, p = 0.5 * (e[1:] + e[:-1]), h / h.sum()
    w0 = np.cumsum(p); w1 = 1 - w0
    cm = np.cumsum(p * c)
    m0 = cm / np.maximum(w0, 1e-12)
    m1 = (np.sum(p * c) - cm) / np.maximum(w1, 1e-12)
    return float(c[int(np.argmax(w0 * w1 * (m0 - m1) ** 2))])


def absorb_short(S, V, secs, min_bars):
    """Une section plus courte que min_bars n'est pas une section : elle rejoint
    la voisine avec laquelle elle s'aligne le mieux. Norah garde sinon un
    « B » de 2 mesures qui est le turnaround du A."""
    out = [dict(s) for s in secs]
    while True:
        for i, s in enumerate(out):
            if s["b1"] - s["b0"] + 1 >= min_bars or len(out) < 2:
                continue
            j = max([k for k in (i - 1, i + 1) if 0 <= k < len(out)],
                    key=lambda k: sc_shift(S, V, s, out[k]))
            out[j]["b0"] = min(out[j]["b0"], s["b0"])
            out[j]["b1"] = max(out[j]["b1"], s["b1"])
            out.pop(i)
            break
        else:
            return out


def merge_letters(S, V, secs, scorer, thr):
    """Deux lettres fusionnent dès qu'une de leurs sections en atteint une
    autre. Union-find, donc la fusion est transitive."""
    letters = sorted({s["letter"] for s in secs})
    par = {L: L for L in letters}

    def find(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    for a, b in itertools.combinations(secs, 2):
        if a["letter"] != b["letter"] and scorer(S, V, a, b) >= thr:
            ra, rb = find(a["letter"]), find(b["letter"])
            if ra != rb:
                par[ra] = rb
    ren, out = {}, []
    for s in secs:
        g = find(s["letter"])
        ren.setdefault(g, chr(ord("A") + len(ren)))
        out.append(dict(s, letter=ren[g]))
    return out


RULES = [("aucune fusion — l'état d'avant", None, None, None),
         ("CE QUI TOURNE MAINTENANT : absorbe < 6, fusion mesure à mesure ≥ 0,95",
          6, "mesure à mesure", 0.95)]
for mb in (0, 6, 8):
    for nm in SCORERS:
        for T in (0.90, 0.95, 0.98):
            tag = "sans absorption" if not mb else f"absorbe < {mb} mesures"
            RULES.append((f"{tag}, puis fusion {nm} ≥ {T:.2f}", mb, nm, T))
for nm in SCORERS:
    RULES.append((f"absorbe < 6, puis fusion {nm} ≥ vallée du morceau (Otsu)",
                  6, nm, "otsu"))


def apply_rule(d, mb, nm, T):
    if nm is None:
        return d["secs"]
    secs = absorb_short(d["S"], d["V"], d["secs"], mb) if mb else d["secs"]
    thr = otsu(off_diagonal(d["S"])) if T == "otsu" else T
    return merge_letters(d["S"], d["V"], secs, SCORERS[nm], thr)


# ── données ─────────────────────────────────────────────────────────────────
def load(stem):
    from harmonia_min import pipeline as _pl
    real = hs.detect_sections
    c = {}

    def spy(grid, arr, times, bars=None, **kw):
        c.update(grid=grid, bars=copy.deepcopy(bars))
        return real(grid, arr, times, bars, **kw)

    hs.detect_sections = spy
    try:
        _pl.analyze(HERE / f"docs/audio/{stem}.m4a", title=stem, file_key="x",
                    audio_url="")
    finally:
        hs.detect_sections = real
    grid, n = c["grid"], len(c["grid"]) - 1
    V = harmonic_vectors(mx.frame_posteriors(HERE / f"docs/audio/{stem}.m4a")[0], grid)
    S = V @ V.T
    ent, _ = build_dictionary(S, n)
    return dict(stem=stem, n=n, V=V, S=S, bars=c["bars"], ent=ent,
                secs=sections_from(S, n, ent, post_process=False))


def strip_img(rows, n):
    fig, axs = plt.subplots(len(rows), 1, figsize=(12.4, 0.42 * len(rows) + .6),
                            gridspec_kw={"hspace": 0.55})
    axs = np.atleast_1d(axs)
    for ax, (label, secs, hit) in zip(axs, rows):
        seen = {}
        for s in secs:
            seen.setdefault(s["letter"], COLS[len(seen) % len(COLS)])
            ax.add_patch(plt.Rectangle((s["b0"], 0), s["b1"] - s["b0"] + 1, 1,
                                       color=seen[s["letter"]]))
            if s["b1"] - s["b0"] >= 2:
                ax.text((s["b0"] + s["b1"] + 1) / 2, .5, s["letter"], ha="center",
                        va="center", color="#fff", fontsize=8, fontweight="bold")
        ax.set_xlim(0, n); ax.set_ylim(0, 1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_ylabel(("✓ " if hit else "") + label, fontsize=7.2, rotation=0,
                      ha="right", va="center",
                      color="#1f8a5b" if hit else "#8a8371")
        for sp in ax.spines.values():
            sp.set_visible(False)
    axs[-1].set_xlabel("mesure", fontsize=8)
    return fig2b64(fig)


def main():
    data = {st: load(st) for st in TARGET}
    for st in TARGET:
        print(f"  chargé {st}")

    body = ""
    # per-song: the score matrices, then every rule as a strip
    for st, (tgt, why) in TARGET.items():
        d = data[st]
        first = {}
        for s in d["secs"]:
            first.setdefault(s["letter"], s)
        Ls = sorted(first)
        tabs = ""
        for nm, f in SCORERS.items():
            cells = "".join(
                f"<td class='{'hi' if f(d['S'], d['V'], first[a], first[b]) >= .95 else ''}'>"
                f"{a}–{b}<br><b>{f(d['S'], d['V'], first[a], first[b]):.2f}</b></td>"
                for a, b in itertools.combinations(Ls, 2))
            tabs += f"<tr><th>{nm}</th>{cells}</tr>"
        rows = []
        for label, mb, nm, T in RULES:
            secs = apply_rule(d, mb, nm, T)
            rows.append((label, secs, len({s["letter"] for s in secs}) == tgt))
        img = strip_img(rows, d["n"])
        chords = "".join(
            f"<tr><td style='color:{COLS[i % len(COLS)]}'><b>{L}</b></td><td class=ch>"
            + " | ".join(" ".join("N.C." if c.get("nc") else NAMES[c["root"]] + c["q"]
                                  for c in d["bars"][b]) or "—"
                         for b in range(first[L]["b0"],
                                        min(first[L]["b1"] + 1, first[L]["b0"] + 8)))
            + "</td></tr>" for i, L in enumerate(Ls))
        body += f"""<section><h2>{st.replace('_',' ').title()}
<span class=sub>cible : {tgt} sections — {why}</span></h2>
<h3>ce que jouent les lettres actuelles</h3>
<table><tr><th>lettre</th><th>accords</th></tr>{chords}</table>
<h3>ce que donne chaque façon de comparer deux sections</h3>
<table class=sc>{tabs}</table>
<p class=cap>En vert : ≥ 0,95. Vallée du morceau (Otsu) =
<b>{otsu(off_diagonal(d['S'])):.3f}</b> — bien trop bas pour dire « la même ».</p>
<h3>chaque règle, en bandes</h3>
<img src="data:image/png;base64,{img}">
<p class=cap>Un ✓ vert = cette règle donne le nombre de sections que tu as
demandé sur ce morceau.</p></section>"""
        print(f"  ok {st}")

    # the cross-song verdict table
    hdr = "".join(f"<th>{s.replace('_',' ').title()[:22]}<br>"
                  f"<span class=sub>cible {TARGET[s][0]}</span></th>" for s in TARGET)
    trs = ""
    for label, mb, nm, T in RULES:
        cells, allok = "", True
        for st in TARGET:
            k = len({s["letter"] for s in apply_rule(data[st], mb, nm, T)})
            ok = k == TARGET[st][0]
            allok &= ok
            cells += f"<td class='{'ok' if ok else 'no'}'>{k}</td>"
        trs += (f"<tr class='{'win' if allok else ''}'><td>{label}</td>{cells}"
                f"<td>{'les trois' if allok else ''}</td></tr>")

    out = HERE / "harmonia_min/state/reports/merge_rules.html"
    out.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Quand deux sections sont-elles la même ?</title><style>
body{{margin:0;background:#e7e0d0;font:15px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1250px;margin:0 auto;padding:22px 15px 60px}}
h1{{font:italic 600 25px Georgia,serif;margin:0 0 4px}}
.lede{{color:#8a8371;font-size:13px;margin-bottom:22px;max-width:940px}}
.lede b{{color:{INK}}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:16px 18px;margin-bottom:16px}}
h2{{font:700 17px system-ui;margin:0 0 10px;color:#8a2b2b}}
h3{{font:700 11px system-ui;margin:18px 0 6px;color:#8a8371;text-transform:uppercase;letter-spacing:.05em}}
.sub{{font:500 12px system-ui;color:#8a8371}}
img{{max-width:100%;border-radius:8px}}
.cap{{font-size:12px;color:#8a8371;margin:4px 0 10px;max-width:900px}}
table{{border-collapse:collapse;font-size:12.5px;margin-top:4px}}
th,td{{border:1px solid #e5dcc6;padding:4px 10px;text-align:left;vertical-align:top}}
th{{background:#f7f3e9;font-weight:700}}
table.sc td{{text-align:center;font-size:11.5px}}
td.hi{{background:#dff0e6;color:#0f5132}}
td.ok{{background:#dff0e6;color:#0f5132;text-align:center;font-weight:700}}
td.no{{background:#f7e7e7;color:#8a2b2b;text-align:center}}
tr.win td:first-child{{font-weight:700}}
.ch{{font:11.5px ui-monospace,SFMono-Regular,Menlo,monospace}}
</style></head><body><div class=wrap>
<h1>Quand deux sections sont-elles la même ?</h1>
<div class=lede>Les trois erreurs que tu as relevées sont des <b>lettres</b>, pas
des frontières : les frontières sont déjà au bon endroit. Rien, à la fin du
dictionnaire, ne vérifie si deux lettres désignent la même musique — il manque
une passe de fusion, et c'est tout ce qui manque.
<br><br><b>Ton idée des modes, mesurée : elle échoue.</b> La vallée entre les
deux modes tombe à 0,58 / 0,62 / 0,63 sur les trois morceaux. Ce seuil sépare
« sans rapport » de « apparenté », pas « la même » de « pas la même » — le
second mode contient à la fois les vraies reprises et tout ce qui partage
simplement la tonalité. Fusionner à 0,6 écrase chaque morceau en une lettre.
<br><br>Deux axes, croisés ci-dessous : <b>absorber les restes trop courts</b>
(le « B » de 2 mesures de Norah est le turnaround du A, pas une section) et
<b>comment comparer deux sections</b>.</div>

<section><h2>Le verdict, sur tes trois structures</h2>
<table><tr><th>règle</th>{hdr}<th></th></tr>{trs}</table>
<p class=cap>Le nombre de sections obtenu. Vert = c'est ta cible.</p></section>
{body}</div></body></html>""")
    print(f"wrote {out.relative_to(HERE)} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
