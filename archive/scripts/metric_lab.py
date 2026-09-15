"""Comment le score de matching des sections est fabriqué — l'addition, dessinée.

    .venv/bin/python scripts/metric_lab.py [<stem> ...]
        -> /plots/metric_lab.html  (les cinq cas d'école)
           /plots/metric_<stem>.html  (l'addition mesure par mesure, à écouter)

Louis, 2026-08-12 : « montre-moi une démo de comment le score est fait pour le
matching des sections. »

Le principe de la page : elle ne PARAPHRASE pas `section_metric.compare`, elle
la fait parler. `compare()` rend depuis aujourd'hui le coût de CHAQUE mesure et
sa raison (`bars_p2r`, `bars_r2p`) ; la bande « où partent les points » est
tracée à partir de ça. Si la métrique change, la page change avec elle — pas
moyen qu'elles divergent, ce qui est arrivé assez souvent ailleurs.

Les couleurs du coût sont une rampe d'une seule teinte (0 → 0,35 → 1), pas un
feu tricolore : le coût est une grandeur ordonnée, pas trois états.
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

from ssm_zoo import SONGS, AUDIO, gt_sections, fig2b64          # noqa: E402
from peak_profile import page, PLOT_L, PLOT_R                   # noqa: E402
from hard_prior_sections import _strip, colourmap               # noqa: E402
from section_metric import compare, TAIL_W                      # noqa: E402
import order_bundle                                             # noqa: E402
from order_lab import new_sections, HOLDOUT                     # noqa: E402

OUTDIR = HERE / "docs" / "plots"

# Rampe d'une seule teinte : rien / remise / plein tarif.
FREE, PART, FULL = "#eae4d4", "#9fb9cc", "#2f5d80"
PAPER, INK = "#fffdf6", "#0d2437"


def cost_colour(c: float) -> str:
    return FREE if c <= 0 else (PART if c < 0.9 else FULL)


# ── les bandes ──────────────────────────────────────────────────────────────

def _cost_lane(ax, trace, n, y=0.0, h=1.0):
    """Une case par mesure, teintée par ce qu'elle coûte."""
    for i in range(n):
        c = trace.get(i, (0.0, ""))[0]
        ax.add_patch(plt.Rectangle((i, y), 1, h, facecolor=cost_colour(c),
                                   edgecolor=PAPER, lw=0.7))
    if not any(v[0] > 0 for v in trace.values()):
        ax.text(n / 2, y + h / 2, "rien à payer dans ce sens", ha="center",
                va="center", fontsize=9, color="#a89f8c", style="italic")


def _blank(ax, n, label):
    ax.set_xlim(0, n)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)


def song_fig(ours, theirs, r, n, hard=()):
    lanes = [("toi", "sec", theirs), ("nous", "sec", ours),
             ("coût nous→toi", "cost", r["bars_p2r"]),
             ("coût toi→nous", "cost", r["bars_r2p"])]
    fig, axs = plt.subplots(len(lanes), 1, figsize=(9.2, 0.82 * len(lanes) + 0.7),
                            facecolor=PAPER, gridspec_kw={"hspace": 0.55})
    cm = colourmap()
    for ax, (lab, kind, data) in zip(axs, lanes):
        if kind == "sec":
            _strip(ax, data, n, cm)
        else:
            _cost_lane(ax, data, n)
        for h in hard:
            ax.axvline(h, color=INK, lw=1.6, alpha=.55)
        _blank(ax, n, lab)
        ax.set_xticks([])
    step = 8 if n <= 120 else 16
    axs[-1].set_xticks(np.arange(0, n + 1, step))
    axs[-1].set_xticklabels([str(int(t) + 1) for t in np.arange(0, n + 1, step)],
                            fontsize=8, color="#8a8371")
    axs[-1].tick_params(length=0, pad=2)
    return fig2b64(fig)


def mini_fig(a, b, n, labs=("A", "B")):
    """Deux bandes nues, pour un cas d'école."""
    fig, axs = plt.subplots(2, 1, figsize=(7.4, 1.15), facecolor=PAPER,
                            gridspec_kw={"hspace": 0.5})
    for ax, (lab, secs) in zip(axs, [(labs[0], a), (labs[1], b)]):
        # Une palette NEUVE par bande : les deux grilles se colorient dans leur
        # propre ordre d'apparition, donc « X Y » et « A B » se dessinent
        # pareil. C'est le premier cas d'école — renommer ne coûte rien.
        _strip(ax, secs, n, colourmap())
        # Deux blocs adjacents de MÊME lettre sont le cas qu'on veut voir : sans
        # ce trait, « un A de 16 » et « deux A de 8 » se dessinent pareil.
        for s in secs[1:]:
            ax.axvline(s["b0"], color=PAPER, lw=3.2)
            ax.axvline(s["b0"], color="#b9ae94", lw=1.0)
        _blank(ax, n, lab)
        ax.set_xticks([])
    return fig2b64(fig)


# ── l'addition, en toutes lettres ───────────────────────────────────────────

def sums_html(r) -> str:
    def h(a, b):
        return f"2×{a:.3f}×{b:.3f} ÷ ({a:.3f}+{b:.3f})"
    return f"""<table class=sum>
<tr><td class=k>nous → toi</td><td class=v>{r['span_p2r']:.3f}</td>
    <td class=c>la part de NOS mesures que leur partenaire chez toi couvre aussi
    — ce qui chute quand on sur-découpe</td></tr>
<tr><td class=k>toi → nous</td><td class=v>{r['span_r2p']:.3f}</td>
    <td class=c>l'inverse — ce qui chute quand on sous-découpe</td></tr>
<tr class=tot><td class=k>segments</td><td class=v>{r['spans']:.3f}</td>
    <td class=c>moyenne harmonique : {h(r['span_p2r'], r['span_r2p'])}</td></tr>
<tr><td class=k>lettres nous → toi</td><td class=v>{r['letter_p2r']:.3f}</td>
    <td class=c>la part de nos mesures dont la lettre traduit TOUJOURS la même
    chose chez toi</td></tr>
<tr><td class=k>lettres toi → nous</td><td class=v>{r['letter_r2p']:.3f}</td>
    <td class=c>l'inverse</td></tr>
<tr class=tot><td class=k>lettres</td><td class=v>{r['letters']:.3f}</td>
    <td class=c>moyenne harmonique : {h(r['letter_p2r'], r['letter_r2p'])}</td></tr>
<tr class=fin><td class=k>score</td><td class=v>{r['score']:.3f}</td>
    <td class=c><b>segments × lettres</b> = {r['spans']:.3f} × {r['letters']:.3f}
    — un produit, pour qu'une moitié ne puisse pas racheter l'autre</td></tr>
</table>"""


def runs_of(trace, n):
    """Les tronçons contigus de même coût et même raison, du plus cher au moins."""
    out, cur = [], None
    for i in range(n):
        c, why = trace.get(i, (0.0, ""))
        if c <= 0:
            cur = None
            continue
        if cur and cur["why"] == why and cur["c"] == c and cur["b1"] == i - 1:
            cur["b1"] = i
        else:
            cur = {"b0": i, "b1": i, "c": c, "why": why}
            out.append(cur)
    for r in out:
        r["tot"] = r["c"] * (r["b1"] - r["b0"] + 1)
    return sorted(out, key=lambda r: -r["tot"])


# Ce que chaque raison veut dire, dans le sens où elle est comptée.
WHY = {
    ("nous→toi", "sans partenaire"): "un bloc à nous ne touche aucun bloc à toi",
    ("toi→nous", "sans partenaire"): "un bloc à toi ne touche aucun bloc à nous",
    ("nous→toi", "section en trop"): "nos mesures tombent dans un bloc à toi que rien n'apparie",
    ("toi→nous", "section en trop"): "tes mesures tombent dans un bloc à nous que rien n'apparie",
    ("nous→toi", "chez le voisin apparié"): "frontière décalée — ça déborde sur le bloc voisin",
    ("toi→nous", "chez le voisin apparié"): "frontière décalée — ça déborde sur le bloc voisin",
    ("nous→toi", "queue"): "queue en litige — remise",
    ("toi→nous", "queue"): "queue en litige — remise",
    ("nous→toi", "sur-découpage constant"): "sur-découpage fait pareil partout — remise",
    ("toi→nous", "sur-découpage constant"): "sur-découpage fait pareil partout — remise",
}


def _letters_in(lab, b0, b1):
    """Les lettres écrites sur ces mesures, dans l'ordre, sans répétition
    consécutive — « A, B » se lit tout de suite, « A A A B » non."""
    out = []
    for i in range(b0, b1 + 1):
        l = lab[i] if lab[i] is not None else "—"
        if not out or out[-1] != l:
            out.append(l)
    return " ".join(out)


def leaks(r, n, ours, theirs, k=4):
    from section_metric import bars as _bars
    lp, lb = _bars(ours, n), _bars(theirs, n)
    out = []
    for sens, key in (("nous→toi", "bars_p2r"), ("toi→nous", "bars_r2p")):
        for x in runs_of(r[key], n)[:k]:
            out.append({**x, "sens": sens,
                        "why_txt": WHY.get((sens, x["why"]), x["why"]),
                        "you": _letters_in(lb, x["b0"], x["b1"]),
                        "us": _letters_in(lp, x["b0"], x["b1"])})
    return sorted(out, key=lambda x: -x["tot"])


def letter_leaks(r):
    """Ce que perd la moitié LETTRES : les reprises dont la traduction n'est
    pas celle que leur lettre a d'habitude."""
    out = []
    for sens, key in (("nous→toi", "letter_odd_p2r"), ("toi→nous", "letter_odd_r2p")):
        who, other = ("notre", "ton") if sens == "nous→toi" else ("ton", "notre")
        for x in r[key]:
            out.append({**x, "sens": sens, "n_bars": x["b1"] - x["b0"] + 1,
                        "txt": f"{who} {x['label']} de ces mesures se traduit par "
                               f"« {other} {x['sig']} », alors que les autres "
                               f"{x['label']} donnent « {other} {x['kept']} »"})
    return sorted(out, key=lambda x: -x["n_bars"])


def leak_html(items, n, letters=(), playable=True):
    cards = []
    for x in letters:
        btn = (f'<button class=blk data-p="[{x["b0"]},{x["b1"] + 1}]">écouter</button>'
               if playable else "")
        cards.append(
            f'<div class=leak><div class=lk1><b>mes. {x["b0"] + 1}–{x["b1"] + 1}</b>'
            f'<span class=cost>lettres</span></div>'
            f'<div class=lk2>{x["txt"]}</div>'
            f'<div class=lk3><span>{x["sens"]}</span>'
            f'<span>{x["n_bars"]} mes.</span>{btn}</div></div>')
    if not items and not cards:
        return "<p class=hint>aucune mesure ne coûte quoi que ce soit.</p>"
    for x in items:
        nb = x["b1"] - x["b0"] + 1
        btn = (f'<button class=blk data-p="[{x["b0"]},{x["b1"] + 1}]">écouter</button>'
               if playable else "")
        cards.append(
            f'<div class=leak><div class=lk1><b>mes. {x["b0"] + 1}–{x["b1"] + 1}</b>'
            f'<span class=cost>−{x["tot"] / n:.3f}</span></div>'
            f'<div class=lk2>tu écris <b>{x["you"]}</b>, on écrit '
            f'<b>{x["us"]}</b> — {x["why_txt"]}</div>'
            f'<div class=lk3><span>{x["sens"]}</span>'
            f'<span>{nb} mes. × {x["c"]:.2f}</span>{btn}</div></div>')
    return "".join(cards)


def reading(items, r, n, letters=()) -> str:
    """Une phrase : ce que ce score dit du morceau, pas de la métrique."""
    if not items and not letters:
        return "Découpage identique au tien."
    if not items:
        x = letters[0]
        return (f"Les frontières sont justes — tout le coût est dans les noms : "
                f"{x['txt']} (mes. {x['b0'] + 1}–{x['b1'] + 1}).")
    big = items[0]
    where = f"mes. {big['b0'] + 1}–{big['b1'] + 1}"
    if big["why"] == "chez le voisin apparié":
        what = f"une frontière posée à côté ({where})"
    elif big["you"] == big["us"]:
        what = (f"ton <b>{big['you']}</b> de {where} qu'on a coupé en morceaux "
                "au lieu de l'écrire d'un bloc")
    else:
        what = (f"{where} : tu écris <b>{big['you']}</b>, on écrit "
                f"<b>{big['us']}</b>")
    part = big["tot"] / n / max(1e-9, 1 - r["spans"]) if r["spans"] < 1 else 0
    return (f"Ce qui coûte le plus : {what} — à lui seul "
            f"{part:.0%} de ce qu'on perd sur les segments.")


def mapping_html(r) -> str:
    a = " · ".join(f"notre {k} → ton {v}" for k, v in sorted(r["mapping"].items()))
    b = " · ".join(f"ton {k} → notre {v}" for k, v in sorted(r["mapping_ref"].items()))
    return (f"<p class=map><b>la correspondance trouvée</b><br>{a}<br>{b}</p>")


# ── page par morceau ────────────────────────────────────────────────────────

def song_page(stem: str, title: str) -> str:
    b = order_bundle.get(stem)
    n, grid = b["n"], b["grid"]
    ours, _ = new_sections(b)
    gt = gt_sections(stem)
    theirs = gt["sections"]
    r = compare(ours, theirs, n)
    img = song_fig(ours, theirs, r, n, b["hard"])
    items = leaks(r, n, ours, theirs)
    lets = letter_leaks(r)

    body = f"""<section><h2>{title} <span class=sub>{n} mesures ·
score {r['score']:.3f}</span></h2>
<div class=plot><img src="data:image/png;base64,{img}">
<div class=cur></div><div class=hit></div></div>
<div class=bar><button class=pp>▶</button><span class=pos>mes. 1 · 0:00</span>
<span class=hint>touche le graphique pour te déplacer</span></div>
<p class=leg><span class=sw style="background:{FREE}"></span> gratuit
<span class=sw style="background:{PART}"></span> remise ({TAIL_W:.2f})
<span class=sw style="background:{FULL}"></span> plein tarif (1,00)</p>
<p class=read>{reading(items, r, n, lets)}</p>
{sums_html(r)}
{mapping_html(r)}
<h3>où partent les points</h3>
{leak_html(items, n, lets)}
</section>
<audio id=au preload=metadata playsinline src="../audio/{stem}.m4a"></audio>
<script>window.GRID={[round(t, 3) for t in grid]};
window.PLOT=[{PLOT_L},{PLOT_R}]; window.U=1;</script>"""
    return page(f"{title} — d'où vient le score", body, back=True,
                lede_html=LEDE_SONG, back_href="metric_lab.html")


# ── les cas d'école, calculés en vrai ───────────────────────────────────────

def S(*spans):
    return [{"b0": a, "b1": b, "label": l} for a, b, l in spans]


CASES = [
    ("Changer les noms ne coûte rien",
     32, S((0, 15, "A"), (16, 31, "B")), S((0, 15, "X"), (16, 31, "Y")),
     "On apparie avant de comparer, dans les deux sens. Ton A peut être notre B."),
    ("Tes deux A écrits comme un seul A de 16",
     32, S((0, 31, "A")), S((0, 15, "A"), (16, 31, "A")),
     "<b>Gratuit</b> (ta décision du 12/08). Les deux écritures désignent la "
     "même musique sous le même nom ; combien d'occurrences on écrit est une "
     "convention. En échange, le score ne distingue plus « un A de 16 » de "
     "« A A » — ça se juge au repli, plus ici."),
    ("Ton A et ton B collés en un seul bloc",
     32, S((0, 31, "A")), S((0, 15, "A"), (16, 31, "B")),
     "La contrepartie, et ce qui empêche la remise de dégénérer : la fusion "
     "n'est gratuite qu'entre occurrences de la MÊME lettre. Avaler une autre "
     "section reste une faute pleine."),
    ("Une frontière posée deux mesures trop loin",
     32, S((0, 17, "A"), (18, 31, "B")), S((0, 15, "A"), (16, 31, "B")),
     "Le coût est proportionnel au nombre de mesures déplacées, et il se paie "
     "des deux côtés."),
    ("Une queue de deux mesures en litige",
     32, S((0, 15, "A"), (16, 29, "B"), (30, 31, "Q")),
     S((0, 15, "A"), (16, 31, "B")),
     f"Deux mesures de fin dont personne ne veut vraiment : {TAIL_W:.2f} au lieu "
     "de 1,00. C'est le désaccord de goût, pas une erreur de frontière."),
    ("Ton B toujours écrit « B puis C » chez nous",
     32, S((0, 15, "A"), (16, 23, "B"), (24, 31, "C")),
     S((0, 15, "A"), (16, 31, "B")),
     "Sur-découper ne coûte presque rien TANT QUE c'est fait pareil à chaque "
     "reprise — la traduction reste une fonction. Découper ici et pas là, si."),
    ("Tout le morceau sous une seule lettre",
     32, S((0, 31, "A")), S((0, 7, "A"), (8, 15, "B"), (16, 23, "A"), (24, 31, "C")),
     "Le garde-fou : les lettres semblent « constantes » (une seule, rien pour "
     "la contredire). Le produit tient le score à {score}, là où leur moyenne "
     "aurait donné {mean} à un découpage qui ne dit rien."),
]


def case_html(title, n, ours, theirs, note) -> str:
    r = compare(ours, theirs, n)
    img = mini_fig(theirs, ours, n, labs=("toi", "nous"))
    note = note.format(score=f"{r['score']:.3f}".replace(".", ","),
                       mean=f"{(r['spans'] + r['letters']) / 2:.3f}".replace(".", ","))
    return f"""<section class=case><h2>{title}</h2>
<img src="data:image/png;base64,{img}">
<p class=note>{note}</p>
<p class=chip><span>segments <b>{r['spans']:.3f}</b></span>
<span>lettres <b>{r['letters']:.3f}</b></span>
<span class=hi>score <b>{r['score']:.3f}</b></span></p></section>"""


LEDE_COMMON = """<div class=lede>Le score répond à une seule question :
<b>est-ce que notre découpage dit la même chose que le tien ?</b> Il vaut 1 quand
c'est identique.<br><br>
Il se fabrique en deux moitiés, et le score est leur <b>produit</b>.
<b>Segments</b> apparie les blocs et regarde s'ils couvrent le même espace — il
voit les longueurs, les frontières, et tes deux A qu'on aurait écrits en un seul.
<b>Lettres</b> regarde si la traduction de tes noms vers les nôtres est la même
d'un bout à l'autre — renommer ne coûte rien, être incohérent coûte.<br><br>
Chaque moitié est calculée <b>dans les deux sens</b> (nous→toi, toi→nous) puis
moyennée en harmonique : sur-découper et sous-découper ne sont pas la même faute,
et aucune des deux ne peut se rattraper sur l'autre.<br><br>
Trois remises, parce que les deux lectures se défendent : une <b>queue</b> de
deux mesures en litige et un <b>sur-découpage constant</b> coûtent 0,35 au lieu
de 1 — et depuis le 12/08, écrire tes deux A collés comme un seul A de 16 ne
coûte <b>rien</b>.</div>"""

LEDE_SONG = """<div class=lede>Ton découpage, le nôtre, puis <b>ce que chaque
mesure coûte</b>, dans les deux sens. Touche le graphique pour te déplacer,
touche un tronçon pour l'écouter.<br>
<a href="metric_lab.html">Comment le score est fabriqué →</a></div>"""

EXTRA_CSS = """<style>
h3{font:700 14px system-ui;margin:16px 0 6px;color:#6f6858}
table.sum,table.idx{width:100%;border-collapse:collapse;
  font:500 12.5px system-ui;margin:8px 0;table-layout:fixed}
table.sum td,table.idx td{padding:5px 6px;border-top:1px solid #efe7d4;
  vertical-align:top;overflow-wrap:anywhere}
table.sum td.k{width:33%;color:#6f6858}
table.sum td.v{width:19%;font:700 13px ui-monospace,monospace;text-align:right}
table.sum td.c{color:#8a8371;font-size:11.5px}
table.idx td:first-child{width:40%}
table.idx td:not(:first-child){width:15%;text-align:right;
  font:600 12px ui-monospace,monospace}
table.idx .hint{display:block;font-size:10px}
tr.tot td{background:#faf6ea}
tr.fin td{background:#f2ecdc;font-size:13.5px}
tr.fin td.v{font-size:16px;color:#8a2b2b}
div.leak{border-top:1px solid #efe7d4;padding:7px 2px}
.lk1{display:flex;justify-content:space-between;align-items:baseline;
  font:600 13px system-ui}
.lk1 .cost{font:700 13px ui-monospace,monospace;color:#8a2b2b}
.lk2{font:500 12px system-ui;color:#6f6858;margin:1px 0 5px}
.lk3{display:flex;gap:6px;align-items:center;flex-wrap:wrap}
.lk3 span{font:600 10.5px ui-monospace,monospace;color:#8a8371;
  background:#f7f3e9;border:1px solid #e5dcc6;border-radius:6px;padding:2px 6px}
p.read{font:500 13px system-ui;color:#4a4437;background:#faf6ea;
  border-left:3px solid #c9a227;border-radius:0 8px 8px 0;padding:8px 10px;
  margin:6px 0 12px}
p.leg{font:600 11.5px system-ui;color:#6f6858;margin:2px 0 10px}
.sw{display:inline-block;width:13px;height:13px;border-radius:3px;
  vertical-align:-2px;margin:0 4px 0 12px;border:1px solid #d8cfb4}
p.map{font:500 12px system-ui;color:#6f6858;background:#faf6ea;border-radius:8px;
  padding:8px 10px}
section.case h2{font-size:15.5px}
p.note{font:500 12.5px system-ui;color:#6f6858;margin:6px 0 8px}
p.chip{margin:0;display:flex;gap:6px;flex-wrap:wrap}
p.chip span{font:600 11.5px system-ui;color:#6f6858;background:#f7f3e9;
  border:1px solid #e5dcc6;border-radius:7px;padding:4px 8px}
p.chip span.hi{background:#f2ecdc;color:#8a2b2b}
p.chip b{font:700 12.5px ui-monospace,monospace}
</style>"""


def main():
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    todo = list(SONGS) + HOLDOUT
    if argv:
        todo = [(s, t) for s, t in todo if s in argv] or [(s, s) for s in argv]

    rows = []
    for stem, title in todo:
        if not gt_sections(stem) or not (AUDIO / f"{stem}.m4a").exists():
            continue
        html = song_page(stem, title).replace("</head>", EXTRA_CSS + "</head>")
        (OUTDIR / f"metric_{stem}.html").write_text(html)
        b = order_bundle.get(stem)
        r = compare(new_sections(b)[0], gt_sections(stem)["sections"], b["n"])
        hold = " <span class=hint>(hors conception)</span>" if any(
            stem == h for h, _ in HOLDOUT) else ""
        rows.append(f'<tr><td><a href="metric_{stem}.html">{title}</a>{hold}</td>'
                    f'<td>{r["spans"]:.3f}</td><td>{r["letters"]:.3f}</td>'
                    f'<td><b>{r["score"]:.3f}</b></td>'
                    f'<td>{r["forgiven"]}</td></tr>')
        print(f"  ok {title}  {r['score']:.3f}")

    cases = "".join(case_html(*c) for c in CASES)
    idx = ("<section><h2>Les dix-huit morceaux</h2><table class=idx>"
           "<tr><td>morceau</td><td>segm.</td><td>lettr.</td><td>score</td>"
           "<td>remises</td></tr>" + "".join(rows) + "</table>"
           "<p class=hint>Ouvre un morceau pour voir l'addition mesure par "
           "mesure et l'écouter.</p></section>")
    html = page("D'où vient le score des sections", cases + idx,
                lede_html=LEDE_COMMON).replace("</head>", EXTRA_CSS + "</head>")
    (OUTDIR / "metric_lab.html").write_text(html)
    print(f"wrote docs/plots/metric_lab.html + {len(rows)} pages")


if __name__ == "__main__":
    main()
