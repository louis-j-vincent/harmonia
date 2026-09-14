"""Deux étages : un dictionnaire de CELLULES, puis les répétitions de SÉQUENCES.

    python scripts/two_level_sections.py [<stem> ...]
      -> /reports/two_level_sections.html

Louis, 2026-08-05 : « Il faut repartir du premier temps pas couvert par les
blocs déjà dans le dictionnaire et faire le même exercice, et ainsi de suite…
et si on ne trouve rien de récurrent on continue d'avancer », puis, une fois
mesuré que cette règle sort des cellules de 2 mesures et non des sections :
« Je construis le deuxième étage (sections = répétitions de séquences de
cellules) — Go ».

    ÉTAGE 1 · l'alphabet
      on part de la première mesure que rien ne couvre ; on cherche à quelle
      distance ce qui commence là se répète ; on garde la cellule et toutes ses
      occurrences ; on avance. Rien de récurrent ici → on avance d'une mesure.
      Résultat : chaque mesure porte une lettre de cellule (ou un trou).

    ÉTAGE 2 · les sections
      le morceau devient une SUITE de cellules — a a a a b b c b b c … — et on
      relance exactement le même exercice sur cette suite, avec l'égalité de
      symboles au lieu de la similarité harmonique. Une section est une
      séquence de cellules qui se répète.

Pourquoi deux étages : mesuré le 2026-08-05, la règle de Louis appliquée seule
sort bien la cellule qu'on ratait (Grenade : `Dm Bb | F C`, 12 occurrences) mais
`sections_from` fabrique alors une section par chaîne d'occurrences d'UNE
cellule, ce qui hache Grenade en 19 sections A B C D B A B C D B… La cellule est
l'alphabet, pas la section.
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
from ssm_rows_plot import fig2b64                                  # noqa: E402
from harmonia_min import sections as hs, musx as mx                # noqa: E402
import harmonia_min.harmonic_sections as HS                        # noqa: E402

INK = "#1c1c1c"
COLS = ["#8a2b2b", "#1f8a5b", "#2a6fb0", "#c58a2e", "#7c3aed", "#0f766e",
        "#be123c", "#0369a1", "#a16207", "#4338ca"]
NAMES = "C Db D Eb E F Gb G Ab A Bb B".split()
DEFAULT = ["bruno_mars_grenade_official_music_video", "maroon_5_this_love",
           "norah_jones_don_t_know_why", "mayer_hawthorne_the_walk",
           "let_it_be_remastered_2009"]


# ── étage 1 : l'alphabet de cellules ────────────────────────────────────────
def cell_alphabet(S, n, max_cells=10):
    """La règle de Louis, à la lettre. Retourne [{L, b0, occ, lag, run}]."""
    off = HS.off_diagonal(S)
    strong = float(np.quantile(off, HS.PHASE_QUANTILE))
    cont = float(np.quantile(off, HS.CONT_QUANTILE))
    cells, claimed, cursor = [], np.zeros(n, bool), 0
    while cursor < n and len(cells) < max_cells:
        if claimed[cursor]:
            cursor += 1
            continue
        # LE MOTIF EST LE PLUS PETIT PATTERN QUI SE RÉPÈTE (Louis, 2026-08-05 :
        # « ils devraient être égaux au plus petit pattern de répétition »).
        # On balaie les distances de la plus PETITE à la plus grande et on
        # s'arrête à la première où la musique qui commence ici se retrouve.
        #
        # ET LE MOTIF VAUT CETTE DISTANCE, pas la longueur de la suite vérifiée.
        # C'est ce qui fait PAVER les cellules. Prendre la suite donnait des
        # cellules de 3 mesures pour des boucles de 4 (Norah, Let It Be) parce
        # qu'une mesure sur quatre passe sous le seuil — et la suite de
        # cellules devenait « a · a · a · », un trou entre chaque, que l'étage 2
        # ne peut plus lire. La distance EST la période ; la suite dit seulement
        # combien de ses mesures on a pu vérifier.
        b0, best = cursor, None
        for d in range(HS.LAG_MIN, min(HS.LAG_MAX, n - b0 - 1) + 1):
            r = 0
            while (b0 + r + d < n and not claimed[b0 + r]
                   and (S[b0 + r, b0 + r + d] >= strong
                        or (r > 0 and S[b0 + r, b0 + r + d] >= cont))):
                r += 1
            if r >= HS.LAG_MIN:
                best = (r, d)
                break                              # la première = la plus petite
        if best is None:
            cursor += 1                # rien de récurrent ici, on avance
            continue
        run, lag = best
        L = int(lag)
        curve = HS.slide(S, L, b0)
        cand = sorted((int(o) for o in HS.peaks(curve, L, b0)),
                      key=lambda o: -curve[o])
        occ, taken = [], claimed.copy()
        for o in cand:
            if taken[o:min(n, o + L)].any():
                continue
            occ.append(o)
            taken[o:min(n, o + L)] = True
        if not occ:
            cursor += 1
            continue
        occ.sort()
        cells.append({"L": L, "b0": b0, "occ": occ, "lag": lag, "run": run})
        for s in sorted(set([b0]) | set(occ)):
            claimed[s:min(n, s + L)] = True
        cursor = b0 + 1
    return cells, claimed


def cell_string(cells, claimed, n):
    """Le morceau comme SUITE de cellules dans le temps.

    Chaque élément est (bar0, bar1, symbole). Les mesures que rien ne couvre
    deviennent un symbole `·` — elles doivent rester dans la suite, sinon la
    section qui les contient perdrait de la musique.
    """
    spans = []
    for i, e in enumerate(cells):
        for s in sorted(set([e["b0"]]) | set(e["occ"])):
            spans.append((s, min(n - 1, s + e["L"] - 1), chr(ord("a") + i)))
    spans.sort()
    out, b = [], 0
    for s0, s1, sym in spans:
        if s0 > b:
            out.append((b, s0 - 1, "·"))
        if s1 >= b:
            out.append((max(s0, b), s1, sym))
            b = s1 + 1
    if b < n:
        out.append((b, n - 1, "·"))
    return out


# ── étage 2 : les répétitions de la SUITE ───────────────────────────────────
def sequence_motifs(seq, max_entries=8):
    """Le même exercice, sur la suite de symboles, avec l'égalité stricte.

    Retourne [{L, p0, occ}] en INDICES DE LA SUITE, pas en mesures.
    """
    m = len(seq)
    ent, claimed, cursor = [], np.zeros(m, bool), 0
    while cursor < m and len(ent) < max_entries:
        if claimed[cursor]:
            cursor += 1
            continue
        p0, best = cursor, None
        for d in range(1, m - p0):
            r = 0
            while (p0 + r + d < m and not claimed[p0 + r]
                   and seq[p0 + r] == seq[p0 + r + d]):
                r += 1
            if r >= 1 and (best is None or r > best[0]):
                best = (r, d)
        if best is None:
            cursor += 1
            continue
        run, lag = best
        L = int(min(run, lag))
        pat = seq[p0:p0 + L]
        occ, taken = [], claimed.copy()
        for p in range(m - L + 1):
            if taken[p:p + L].any() or seq[p:p + L] != pat:
                continue
            occ.append(p)
            taken[p:p + L] = True
        if len(occ) < 2:
            cursor += 1
            continue
        ent.append({"L": L, "p0": p0, "occ": occ})
        for p in occ:
            claimed[p:p + L] = True
        cursor = p0 + 1
    return ent


def two_level_sections(S, n):
    """[{b0, b1, letter, why}] — contiguës et couvrantes."""
    cells, claimed = cell_alphabet(S, n)
    spans = cell_string(cells, claimed, n)
    seq = [s[2] for s in spans]
    motifs = sequence_motifs(seq)

    owner = [None] * len(spans)
    for mi, e in enumerate(motifs):
        for p in e["occ"]:
            for k in range(e["L"]):
                owner[p + k] = (mi, p)

    # une section = une occurrence entière d'un motif de séquence ; ce qu'aucun
    # motif ne prend reste tel quel et sera groupé par son contenu
    secs, i = [], 0
    while i < len(spans):
        if owner[i] is None:
            j = i
            while j + 1 < len(spans) and owner[j + 1] is None:
                j += 1
            secs.append({"b0": spans[i][0], "b1": spans[j][1],
                         "key": ("seq", "".join(seq[i:j + 1])),
                         "why": f"suite « {' '.join(seq[i:j+1])} », non répétée"})
            i = j + 1
        else:
            mi, p = owner[i]
            L = motifs[mi]["L"]
            secs.append({"b0": spans[p][0], "b1": spans[p + L - 1][1],
                         "key": ("motif", mi),
                         "why": f"motif de séquence {mi+1} : "
                                f"« {' '.join(seq[p:p+L])} », "
                                f"{len(motifs[mi]['occ'])} fois"})
            i = p + L

    letters, nxt = {}, 0
    for s in secs:
        if s["key"] not in letters:
            letters[s["key"]] = chr(ord("A") + nxt) if nxt < 26 else f"S{nxt}"
            nxt += 1
        s["letter"] = letters[s["key"]]
    # Les deux règles de Louis, appliquées ici aussi : jamais de section d'une
    # seule mesure, et un bout de section rejoint TOUJOURS ce qui le précède,
    # jamais ce qui le suit.
    secs = HS.absorb_short(S, secs)
    return HS.coalesce_adjacent(secs), cells, spans, motifs


# ── mesure + page ───────────────────────────────────────────────────────────
def load(stem):
    from harmonia_min import pipeline as _pl
    real = hs.detect_sections
    c = {}

    def spy(g, a, t, bars=None, **k):
        c.update(grid=g, bars=copy.deepcopy(bars))
        return real(g, a, t, bars, **k)

    hs.detect_sections = spy
    try:
        _pl.analyze(HERE / f"docs/audio/{stem}.m4a", title="x", file_key="x",
                    audio_url="")
    finally:
        hs.detect_sections = real
    grid = c["grid"]
    V = HS.harmonic_vectors(mx.frame_posteriors(HERE / f"docs/audio/{stem}.m4a")[0], grid)
    return V @ V.T, len(grid) - 1, c["bars"]


def strip(ax, items, n, label):
    seen = {}
    for b0, b1, key in items:
        seen.setdefault(key, "#e5dcc6" if key == "·" else COLS[len(seen) % len(COLS)])
        ax.add_patch(plt.Rectangle((b0, 0), b1 - b0 + 1, 1, color=seen[key]))
        if b1 - b0 >= 1:
            ax.text((b0 + b1 + 1) / 2, .5, key, ha="center", va="center",
                    color="#fff" if key != "·" else "#8a8371",
                    fontsize=8, fontweight="bold")
        ax.axvline(b0, color="#fff", lw=1.1)
    ax.set_xlim(0, n); ax.set_ylim(0, 1)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")


def song_html(stem):
    S, n, bars = load(stem)
    secs, cells, spans, motifs = two_level_sections(S, n)
    now = HS.sections_from(S, n, HS.build_dictionary(S, n)[0])

    fig, axs = plt.subplots(3, 1, figsize=(12.5, 3.0), gridspec_kw={"hspace": .9})
    strip(axs[0], spans, n, "étage 1\ncellules")
    strip(axs[1], [(s["b0"], s["b1"], s["letter"]) for s in secs], n,
          "étage 2\nsections")
    strip(axs[2], [(s["b0"], s["b1"], s.get("label", s.get("letter"))) for s in now], n,
          "ce qui tourne\naujourd'hui")
    axs[-1].set_xlabel("mesure", fontsize=8)
    img = fig2b64(fig)

    crows = "".join(
        f"<tr><td style='color:{COLS[i%len(COLS)]}'><b>{chr(ord('a')+i)}</b></td>"
        f"<td>{e['L']} mes</td><td>{len(set([e['b0']])|set(e['occ']))}×</td>"
        f"<td class=ch>" + " | ".join(
            " ".join("N.C." if x.get("nc") else NAMES[x["root"]] + x["q"]
                     for x in bars[e["b0"] + k]) or "—"
            for k in range(e["L"])) + "</td></tr>"
        for i, e in enumerate(cells))
    srows = "".join(
        f"<tr><td><b>{s['letter']}</b></td><td>mes. {s['b0']+1}–{s['b1']+1}</td>"
        f"<td>{s['b1']-s['b0']+1}</td><td>{s['why']}</td></tr>" for s in secs)
    seqtxt = " ".join(x[2] for x in spans)
    return f"""<section><h2>{stem.replace('_',' ').title()}
<span class=sub>{n} mesures · {len(cells)} cellules · {len(motifs)} motifs de
séquence · {len({s['letter'] for s in secs})} lettres</span></h2>
<img src="data:image/png;base64,{img}">
<h3>étage 1 — l'alphabet de cellules</h3>
<table><tr><th></th><th>taille</th><th>fois</th><th>accords</th></tr>{crows}</table>
<h3>le morceau comme suite de cellules</h3>
<p class=seq>{seqtxt}</p>
<h3>étage 2 — les sections</h3>
<table><tr><th>lettre</th><th>mesures</th><th>long.</th><th>d'où elle vient</th></tr>
{srows}</table></section>"""


def main():
    stems = sys.argv[1:] or DEFAULT
    body = ""
    for st in stems:
        if not (HERE / f"docs/audio/{st}.m4a").exists():
            print(f"  !! {st} introuvable")
            continue
        body += song_html(st)
        print(f"  ok {st}")
    out = HERE / "harmonia_min/state/reports/two_level_sections.html"
    out.write_text(f"""<!DOCTYPE html><html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Deux étages — cellules, puis séquences</title><style>
*{{box-sizing:border-box;-webkit-tap-highlight-color:transparent}}
body{{margin:0;background:#e7e0d0;font:16px/1.55 -apple-system,system-ui,sans-serif;color:{INK}}}
.wrap{{max-width:1000px;margin:0 auto;padding:20px 12px calc(40px + env(safe-area-inset-bottom))}}
h1{{font:italic 600 24px Georgia,serif;margin:0 0 3px}}
.lede{{color:#6f6858;font-size:13.5px;line-height:1.55;margin-bottom:20px}}
.lede b{{color:{INK}}}
section{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:14px 13px;margin-bottom:16px}}
h2{{font:700 18px system-ui;margin:0 0 10px;color:#8a2b2b}}
h3{{font:700 11px system-ui;margin:16px 0 5px;color:#8a8371;text-transform:uppercase;letter-spacing:.05em}}
.sub{{font:500 12px system-ui;color:#8a8371}}
img{{max-width:100%;border-radius:8px;display:block}}
table{{border-collapse:collapse;font-size:12.5px;width:100%}}
th,td{{border:1px solid #e5dcc6;padding:3px 9px;text-align:left;vertical-align:top}}
th{{background:#f7f3e9;font-size:11.5px}}
.ch{{font:11.5px ui-monospace,Menlo,monospace}}
.seq{{font:13px ui-monospace,Menlo,monospace;background:#f7f3e9;border-radius:8px;
      padding:9px 11px;word-break:break-all;letter-spacing:.12em}}
</style></head><body><div class=wrap>
<h1>Deux étages — cellules, puis séquences</h1>
<div class=lede><b>Étage 1, ta règle :</b> on part de la première mesure que rien
ne couvre, on cherche à quelle distance ce qui commence là se répète, on garde la
cellule et toutes ses occurrences, on avance. Rien de récurrent ici → on avance
d'une mesure. Chaque mesure porte alors une lettre de cellule.
<br><br><b>Étage 2 :</b> le morceau devient une SUITE de cellules — a a a a b b c
b b c… — et on relance exactement le même exercice sur cette suite, avec
l'égalité de symboles au lieu de la similarité harmonique. Une section est une
séquence de cellules qui se répète.
<br><br>Trois bandes par morceau : les cellules, les sections à deux étages, et
ce que le pipeline écrit aujourd'hui. Rien n'est branché — c'est une mesure.</div>
{body}</div></body></html>""")
    print(f"wrote {out.relative_to(HERE)} ({out.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
