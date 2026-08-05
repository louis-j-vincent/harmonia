"""/reports/ug_reference.html — la structure PUBLIÉE contre la nôtre.

Reads the per-song records written by `scripts/ug_reference_align.py`
(`scratchpad/ugref/<stem>.json`) and draws, per song:

  1. two letter strips stacked — what the TAB says the structure is, and what
     the shipped detector (`harmonic_sections.detect_sections`) produces. Our
     letters are coloured by the tab section they overlap most, so "same
     colour = the two agree here" is readable without counting;
  2. the tab's chords beside ours, bar by bar, for the FIRST occurrence of each
     tab section — the harmony question, not just the structure question;
  3. one count line: how many sections the tab says vs how many we produce;
  4. the source URL, its rating and its vote count, plus every alignment
     diagnostic (they are what says whether line 1 can be believed at all).

Then one summary table across all songs.

    .venv/bin/python scripts/ug_reference_report.py
"""
from __future__ import annotations

import html
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt      # noqa: E402
import numpy as np                   # noqa: E402

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))
from ssm_rows_plot import fig2b64                                # noqa: E402
from ug_reference_align import SONGS, UNCOVERED                  # noqa: E402

SRC = HERE / "scratchpad/ugref"
OUT = HERE / "harmonia_min/state/reports/ug_reference.html"

INK = "#1c1c1c"
MUTED = "#8a8371"
PAL = ["#2a6fb0", "#8a2b2b", "#c58a2e", "#1f8a5b", "#7c3aed",
       "#b0552a", "#2f7d7d", "#96407f"]
GREY = "#c9c2ae"


# ── tab section ↔ our letter correspondence ────────────────────────────────

def colour_map(rec) -> tuple[dict, dict, dict]:
    """Give every tab section a colour, then give every OUR letter the colour
    of the tab section it overlaps most (in bars). Where the two agree, the two
    strips are the same colour at the same place — that is the whole point of
    stacking them."""
    names = [n for n in rec["tab_names"]]
    tabcol = {n: PAL[i % len(PAL)] for i, n in enumerate(names)}
    tabcol[UNCOVERED] = GREY

    labels = rec["tab_labels"]
    ourcol, ourmatch = {}, {}
    for s in rec["our_segs"]:
        cnt: dict[str, int] = {}
        for b in range(s["b0"], s["b1"] + 1):
            nm = labels[b] if b < len(labels) else None
            if nm:
                cnt[nm] = cnt.get(nm, 0) + 1
        if not cnt:
            continue
        best = max(cnt, key=cnt.get)
        # a letter can appear several times; the majority over ALL its bars wins
        agg = ourmatch.setdefault(s["label"], {})
        for k, v in cnt.items():
            agg[k] = agg.get(k, 0) + v
    for lab, agg in ourmatch.items():
        ourcol[lab] = tabcol.get(max(agg, key=agg.get), GREY)
    for s in rec["our_segs"]:
        ourcol.setdefault(s["label"], GREY)
    return tabcol, ourcol, ourmatch


def strips_fig(rec, tabcol, ourcol):
    n = rec["n_bars"]
    labels = rec["tab_labels"]
    fig, ax = plt.subplots(figsize=(13.2, 2.5))

    seen: dict[str, int] = {}
    for name, s, e in rec["tab_occ"]:
        seen[name] = seen.get(name, 0) + 1
        ax.add_patch(plt.Rectangle((s, 0.56), e - s, 0.40,
                                   color=tabcol.get(name, GREY), alpha=.92))
        if e - s >= 2:
            ax.text((s + e) / 2, 0.76, f"{name} #{seen[name]}", ha="center",
                    va="center", fontsize=7.2, color="white", fontweight="bold")
    for b in range(n):
        if labels[b] is None:
            ax.add_patch(plt.Rectangle((b, 0.56), 1, 0.40, facecolor="#efe9db",
                                       hatch="////", edgecolor="#cfc6ae", lw=0))

    seen2: dict[str, int] = {}
    for s in rec["our_segs"]:
        b0, b1 = s["b0"], s["b1"] + 1
        seen2[s["label"]] = seen2.get(s["label"], 0) + 1
        ax.add_patch(plt.Rectangle((b0, 0.06), b1 - b0, 0.40,
                                   color=ourcol.get(s["label"], GREY), alpha=.92))
        ax.add_patch(plt.Rectangle((b0, 0.06), b1 - b0, 0.40, fill=False,
                                   edgecolor="white", lw=1.4))
        if b1 - b0 >= 2:
            ax.text((b0 + b1) / 2, 0.26, s["label"], ha="center", va="center",
                    fontsize=8.5, color="white", fontweight="bold")

    ax.set_xlim(0, n)
    ax.set_ylim(0, 1.02)
    ax.set_yticks([0.26, 0.76])
    ax.set_yticklabels(["NOUS\n(détecteur livré)", "TAB\n(structure publiée)"],
                       fontsize=8)
    ax.set_xticks(range(0, n + 1, 4))
    ax.tick_params(labelsize=7)
    ax.set_xlabel("mesure — sur notre grille", fontsize=8.5)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    return fig


# ── chords, bar by bar, first occurrence of each tab section ───────────────

def _root(tok: str) -> str:
    if not tok or tok in ("·", "N.C."):
        return tok
    return tok[:2] if len(tok) > 1 and tok[1] in "b#" else tok[:1]


def chord_table(rec) -> str:
    """First occurrence of each tab section, bar by bar: tab chords vs ours.

    « d'accord » = the two write the same ROOT in that bar (the first chord of
    the bar on each side). It is deliberately root-level: writing Cm7 where the
    tab writes Cm is not a disagreement about the harmony.
    """
    firsts: dict[str, tuple[int, int]] = {}
    for name, s, e in rec["tab_occ"]:
        firsts.setdefault(name, (s, e))
    rows = ""
    for name, (s, e) in firsts.items():
        span = min(e - s, 16)
        rows += (f"<tr class=sec><td colspan=4><b>{html.escape(name)}</b> "
                 f"<span class=cap>1<sup>re</sup> occurrence, mesures "
                 f"{s+1}–{e} ({e-s} mesures"
                 f"{', 16 premières affichées' if e - s > 16 else ''})</span></td></tr>")
        for k in range(span):
            b = s + k
            t = rec["tab_bar_chords"][b]
            o = rec["our_bar_chords"][b]
            ok = _root(t.split(" ")[0]) == _root(o.split(" ")[0])
            rows += (f"<tr><td class=num>{b+1}</td>"
                     f"<td class=ch>{html.escape(t)}</td>"
                     f"<td class=ch>{html.escape(o)}</td>"
                     f"<td class='{'yes' if ok else 'no'}'>"
                     f"{'=' if ok else '≠'}</td></tr>")
    return (f"<table class=chords><tr><th>mes.</th><th>TAB</th><th>NOUS</th>"
            f"<th></th></tr>{rows}</table>")


# ── how far apart the two readings are ─────────────────────────────────────

SHARE = 0.25    # a letter/name has to own this share of a block to count


def split_stats(rec) -> dict:
    """Two counts, both defined here and nowhere else on this page.

    * ``splits`` — tab passages our detector cuts into two or more of its own
      letters.
    * ``merges`` — our sections that swallow two or more different tab
      sections. The opposite error.

    Both use a SHARE floor: a letter only counts against a tab passage if it
    owns at least 25 % of that passage's bars. Without the floor a one-bar
    boundary offset — which the measured alignment error (0.5–2 s median, i.e.
    well under a bar on most of these songs, but not on all) can produce on its
    own — reads as a "split", and the number would be mostly alignment noise.
    """
    labels = rec["tab_labels"]
    splits = 0
    for name, s, e in rec["tab_occ"]:
        L = e - s
        share: dict[str, int] = {}
        for sg in rec["our_segs"]:
            k = len(set(range(sg["b0"], sg["b1"] + 1)) & set(range(s, e)))
            if k:
                share[sg["label"]] = share.get(sg["label"], 0) + k
        if sum(1 for v in share.values() if v >= SHARE * L) >= 2:
            splits += 1
    merges = 0
    for sg in rec["our_segs"]:
        L = sg["b1"] - sg["b0"] + 1
        share = {}
        for b in range(sg["b0"], sg["b1"] + 1):
            nm = labels[b] if b < len(labels) else None
            if nm:
                share[nm] = share.get(nm, 0) + 1
        if sum(1 for v in share.values() if v >= SHARE * L) >= 2:
            merges += 1
    return {"splits": splits, "merges": merges}


# ── page ───────────────────────────────────────────────────────────────────

def trust_box(rec) -> str:
    lm = rec.get("loo") or {}
    med = lm.get("median")
    p90 = lm.get("p90")
    verdict = rec["verdict"]
    vtxt = ("le coût harmonique seul suffit à placer le tab"
            if verdict == "determinate"
            else "le coût harmonique seul ne suffit PAS — ce sont les ancrages "
                 "lyriques qui portent cet alignement")
    tier = ("solide" if (med is not None and med <= 2.0 and rec["n_anchors"] >= 12)
            else "moyen" if (med is not None and med <= 5.0 and rec["n_anchors"] >= 6)
            else "fragile")
    tcol = {"solide": "#1f8a5b", "moyen": "#c58a2e", "fragile": "#8a2b2b"}[tier]
    return f"""<div class=trust>
<div class=trow><b>Source</b> <a href="{rec['url']}">{html.escape(rec['url'])}</a>
— Ultimate Guitar, <b>{rec['rating']:.2f}★ / {rec['votes']} votes</b>,
capo {rec['capo']} → transposition +{rec['shift']} demi-tons ({rec['shift_source']}).</div>
<div class=trow><b>Alignement</b> {rec['n_anchors']} ancrages lyriques
(transcription whisper des voix isolées), couvrant
{rec['anchor_span']*100:.0f}&nbsp;% de la durée. Test de contraste DTW :
<i>{verdict}</i> — {vtxt}.</div>
<div class=trow><b>Erreur d'alignement</b> (on cache un ancrage, on réaligne
sans lui, on regarde de combien de secondes il tombe à côté) :
médiane <b>{('%.1f s' % med) if med is not None else '—'}</b>,
90<sup>e</sup> centile {('%.1f s' % p90) if p90 is not None else '—'}
sur {lm.get('n', 0)} ancrages.
<span class=tier style="background:{tcol}">confiance {tier}</span></div>
<div class=trow><b>Accord des fondamentales</b> {rec['agree_rate']*100:.0f}&nbsp;%
sur {rec['n_compared']} accords — mesure JOINTE de l'alignement et de nos
accords, pas une note donnée au tab.</div>
</div>"""


def song_html(rec) -> str:
    tabcol, ourcol, _ = colour_map(rec)
    strip = fig2b64(strips_fig(rec, tabcol, ourcol))
    st = split_stats(rec)
    n_tab_names = len(rec["tab_names"])
    n_tab_blocks = len(rec["tab_occ"])
    our_letters = sorted({s["label"] for s in rec["our_segs"]})
    n_our = len(our_letters)
    n_our_blocks = len(rec["our_segs"])
    uncovered = sum(1 for l in rec["tab_labels"] if l is None)

    verdict_line = (
        f"<b>Le tab dit {n_tab_names} sections différentes "
        f"({n_tab_blocks} passages) ; nous en écrivons {n_our} "
        f"({n_our_blocks} passages).</b> "
        + ("Même compte." if n_our == n_tab_names else
           f"Nous en produisons {n_our - n_tab_names:+d}.")
        + f" Nous coupons en deux (ou plus) {st['splits']} des {n_tab_blocks} "
          f"passages du tab ; {st['merges']} de nos sections avalent au moins "
          f"deux sections différentes du tab.")

    notes = ""
    if rec.get("parse_notes"):
        notes = ("<p class=cap>Lecture du tab : "
                 + " ; ".join(html.escape(n) for n in rec["parse_notes"]) + ".</p>")

    return f"""<section id="{rec['stem']}">
<h2>{html.escape(rec['title'])}
<span class=sub>{rec['n_bars']} mesures · {uncovered} non couverte(s) par le tab</span></h2>
{trust_box(rec)}
{notes}
<h3>1 — les deux structures, l'une sur l'autre</h3>
<p class=cap>Bande du haut : ce que dit le tab. Bande du bas : ce que produit le
détecteur livré. <b>Une de nos lettres prend la couleur de la section du tab
qu'elle recouvre le plus</b> — deux blocs de la même couleur au même endroit
veulent dire que les deux lectures sont d'accord. Hachuré = mesures que le tab
ne couvre pas (intro/outro non transcrite) ; elles ne sont comparées nulle part.</p>
<img src="data:image/png;base64,{strip}">
<h3>2 — l'harmonie, mesure par mesure</h3>
<p class=cap>Première occurrence de chaque section du tab. « = » signifie même
fondamentale au début de la mesure (Cm7 contre Cm compte comme d'accord).
Les accords du tab sont recalés sur la <b>demi-mesure</b> la plus proche de
notre grille — mesuré sur les quatre premiers morceaux, ils y tombent déjà
(distribution nettement bimodale à 0 et 0,5 de mesure, gigue ≈ 0,15 mesure), et
le recalage déplace un accord d'un quart de mesure au maximum, bien moins que
l'erreur d'alignement mesurée ci-dessus.</p>
{chord_table(rec)}
<h3>3 — le compte</h3>
<p class=reco>{verdict_line}</p>
<p class=cap>Le chart tiré du tab est ouvrable dans l'app :
<a href="/?open=min_{rec['stem']}__ug">/?open=min_{rec['stem']}__ug</a></p>
</section>"""


def summary_table(recs) -> str:
    rows = ""
    tot_tab = tot_our = tot_split = 0
    for r in recs:
        st = split_stats(r)
        nt = len(r["tab_names"])
        no = len({s["label"] for s in r["our_segs"]})
        tot_tab += nt
        tot_our += no
        tot_split += st["splits"]
        d = no - nt
        cls = "same" if d == 0 else ("over" if d > 0 else "under")
        lm = r.get("loo") or {}
        med = lm.get("median")
        rows += (f"<tr><td><a href='#{r['stem']}'>{html.escape(r['title'])}</a></td>"
                 f"<td>{r['rating']:.2f}★ / {r['votes']}</td>"
                 f"<td>{nt}</td><td>{len(r['tab_occ'])}</td>"
                 f"<td>{no}</td><td>{len(r['our_segs'])}</td>"
                 f"<td class={cls}>{d:+d}</td>"
                 f"<td>{st['splits']}</td><td>{st['merges']}</td>"
                 f"<td>{r['agree_rate']*100:.0f}%</td>"
                 f"<td>{('%.1f' % med) if med is not None else '—'}</td></tr>")
    rows += (f"<tr class=tot><td><b>total</b></td><td></td><td><b>{tot_tab}</b></td>"
             f"<td></td><td><b>{tot_our}</b></td><td></td>"
             f"<td class={'over' if tot_our > tot_tab else 'same'}>"
             f"<b>{tot_our-tot_tab:+d}</b></td><td><b>{tot_split}</b></td>"
             f"<td></td><td></td><td></td></tr>")
    return f"""<table class=sum><tr>
<th>morceau</th><th>note du tab</th>
<th>sections<br>TAB</th><th>passages<br>TAB</th>
<th>sections<br>NOUS</th><th>passages<br>NOUS</th>
<th>écart</th><th>passages<br>coupés</th><th>sections<br>fusionnées</th>
<th>accord<br>fondamentales</th><th>err. align.<br>médiane (s)</th></tr>{rows}</table>"""


def dropped_html() -> str:
    p = SRC / "_dropped.json"
    if not p.exists():
        return ""
    rows = json.loads(p.read_text())
    if not rows:
        return ""
    items = "".join(
        f"<li><b>{html.escape(d.get('title', d['__dropped__']))}</b> — "
        f"{html.escape('; '.join(d['why']))}."
        + (f" <a href='{d['url']}'>tab</a>" if d.get("url") else "") + "</li>"
        for d in rows)
    return f"""<section><h2>Morceaux écartés</h2>
<p class=cap>Un alignement faux produirait un chart de référence faux, ce qui
est pire que pas de référence du tout. Les trois tests ci-dessous n'utilisent
QUE le tab et la transcription des voix — jamais nos accords — donc écarter sur
eux ne peut pas nous avantager : au moins 6 ancrages lyriques, couvrant au
moins 45&nbsp;% de la durée, avec une erreur laisser-un-de-côté médiane sous
4&nbsp;secondes (≈ deux mesures à 120&nbsp;BPM).</p>
<ul class=drop>{items}</ul></section>"""


def headline(recs) -> str:
    """The one paragraph that says what the table says, before the table."""
    over = under = same = 0
    tot_tab = tot_our = tot_split = tot_merge = 0
    for r in recs:
        nt = len(r["tab_names"])
        no = len({s["label"] for s in r["our_segs"]})
        tot_tab += nt
        tot_our += no
        st = split_stats(r)
        tot_split += st["splits"]
        tot_merge += st["merges"]
        over += no > nt
        under += no < nt
        same += no == nt
    n_pass = sum(len(r["tab_occ"]) for r in recs)
    return (
        f"<b>Nous n'écrivons PAS trop de lettres — nous en écrivons trop peu.</b> "
        f"Sur ces {len(recs)} morceaux le tab distingue {tot_tab} sections, nous "
        f"en produisons {tot_our} ({tot_our - tot_tab:+d}) : moins que le tab sur "
        f"{under} morceaux, plus sur {over}, autant sur {same}. C'est l'inverse du "
        f"défaut inscrit dans <code>docs/known_issues.md</code> le 2026-08-05 "
        f"(« nous produisons trop de lettres »), qui avait été établi sur trois "
        f"morceaux ; la passe d'absorption + fusion ajoutée depuis "
        f"(<code>MIN_SECTION_BARS</code>, <code>SAME_SECTION</code>) a corrigé "
        f"cela et est allée un cran plus loin. Sur les trois morceaux dont Louis "
        f"a donné la vérité (This Love 3, Don't Know Why 2, The Walk 2) le compte "
        f"est maintenant exact. "
        f"<b>Ce qui reste faux, c'est OÙ tombent les frontières :</b> "
        f"{tot_split} des {n_pass} passages du tab sont coupés en deux lettres ou "
        f"plus, et {tot_merge} de nos sections avalent deux sections différentes "
        f"du tab. <i>Réserve :</i> le découpage d'un tab est une convention "
        f"d'écriture, pas une vérité — un tab qui note « Intro » là où l'intro "
        f"joue la même harmonie que le couplet compte une section de plus que "
        f"nous, sans que personne ait tort.")


def main() -> None:
    recs = []
    for stem in SONGS:
        p = SRC / f"{stem}.json"
        if p.exists():
            recs.append(json.loads(p.read_text()))
    if not recs:
        raise SystemExit("no scratchpad/ugref/*.json — run ug_reference_align.py first")
    body = "".join(song_html(r) for r in recs)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>La structure publiée contre la nôtre</title><style>
body{{margin:0;background:#e7e0d0;font:15px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1320px;margin:0 auto;padding:22px 15px 60px}}
h1{{font:italic 600 25px Georgia,serif;margin:0 0 4px}}
.lede{{color:{MUTED};font-size:13px;margin-bottom:22px;max-width:960px}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:16px 18px;margin-bottom:18px}}
h2{{font:700 17px system-ui;margin:0 0 10px;color:#8a2b2b}}
h3{{font:700 11px system-ui;margin:18px 0 6px;color:{MUTED};text-transform:uppercase;letter-spacing:.05em}}
.sub{{font:500 12px system-ui;color:{MUTED}}}
img{{max-width:100%;border-radius:8px}}
.cap{{font-size:11.5px;color:{MUTED};max-width:960px}}
.reco{{font-size:14px;background:#f7f3e9;border-radius:8px;padding:10px 12px;max-width:960px}}
.trust{{background:#f7f3e9;border-radius:8px;padding:9px 12px;font-size:12.5px;max-width:1000px}}
.trow{{margin:3px 0}}
.tier{{color:white;border-radius:4px;padding:1px 7px;font-size:11px;font-weight:700;margin-left:6px}}
table{{border-collapse:collapse;font-size:12.5px;margin-top:8px}}
th,td{{border:1px solid #e5dcc6;padding:3px 9px;text-align:left}}
th{{background:#f7f3e9;font-size:11.5px}}
table.chords{{max-width:620px}}
table.chords td.num{{color:{MUTED};text-align:right;width:44px}}
table.chords td.ch{{font-family:ui-monospace,Menlo,monospace;font-size:12px}}
tr.sec td{{background:#f0ead9}}
td.yes{{color:#1f8a5b;font-weight:700;text-align:center}}
td.no{{color:#8a2b2b;font-weight:700;text-align:center}}
table.sum td.over{{background:#f6e3e3;font-weight:700}}
table.sum td.under{{background:#e3eef6;font-weight:700}}
table.sum td.same{{background:#e4f0e8;font-weight:700}}
tr.tot td{{background:#f0ead9}}
a{{color:#2a6fb0}}
.scroll{{overflow-x:auto}}
ul.drop{{font-size:13px;max-width:960px}}
ul.drop li{{margin:4px 0}}
</style></head><body><div class=wrap>
<h1>La structure publiée contre la nôtre</h1>
<div class=lede>Chaque morceau ci-dessous a un tab Ultimate&nbsp;Guitar noté
<b>≥ 4,7★ avec ≥ 100 votes</b> — le deuxième rang de l'ordre de confiance du
projet (iReal&nbsp;Pro &gt; UG&nbsp;≥4,7★ &gt; autres tabs &gt; sortie du
modèle). Le tab donne les accords ET le découpage en sections, mais sans
aucune horloge : un accord est écrit au-dessus d'une syllabe, pas au-dessus
d'une seconde. On lui en donne une en isolant les voix (demucs), en les
transcrivant avec des horodatages au MOT (whisper), puis en forçant un
alignement temporel à passer par les lignes de paroles reconnues. La grille de
mesures, elle, vient d'un vrai passage de <code>pipeline.analyze()</code> —
c'est donc bien notre grille, pas celle du tab.<br><br>
<b>Ce que la page ne prouve pas.</b> L'alignement peut être faux ; chaque
morceau porte donc son propre encadré de diagnostics, dont une mesure d'erreur
qui n'utilise NI nos accords NI nos sections (« on cache un ancrage lyrique, on
réaligne sans lui, de combien de secondes tombe-t-il à côté ? »). Un morceau
dont le tab ne pouvait pas être aligné a été écarté, pas rafistolé.<br><br>
Le chart tiré du tab est écrit comme un vrai chart de l'app
(<code>/?open=min_&lt;morceau&gt;__ug</code>, titre « — TAB ») : notre grille et
nos temps, les accords et les sections du tab.</div>
{body}
<section><h2>Toutes les chansons</h2>
<p class=reco>{headline(recs)}</p>
<p class=cap><b>sections</b> = noms distincts (un tab qui écrit « Verse 1 » et
« Verse 2 » compte UNE section) ; <b>passages</b> = occurrences consécutives
regroupées, donc un tab qui écrit six couplets à la suite ne compte qu'UN
passage — la colonne « passages » dépend de la façon d'écrire du tab et n'est
là que pour situer, la comparaison qui compte est celle des sections.
<b>passages coupés</b> = passages du tab que notre détecteur découpe en deux
lettres ou plus, chacune possédant au moins 25&nbsp;% des mesures du passage
(le plancher de 25&nbsp;% évite de compter un décalage d'une mesure comme une
coupure).
<b>sections fusionnées</b> = nos sections qui recouvrent, à 25&nbsp;% au moins
chacune, deux sections différentes du tab — l'erreur inverse.
<b>err. align.</b> = médiane de l'erreur laisser-un-ancrage-de-côté, en secondes.</p>
<div class=scroll>{summary_table(recs)}</div></section>
{dropped_html()}
</div></body></html>""")
    print(f"wrote {OUT.relative_to(HERE)} ({OUT.stat().st_size // 1024} KB, "
          f"{len(recs)} songs)")


if __name__ == "__main__":
    main()
