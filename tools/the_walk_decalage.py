"""The Walk — « après les 3 premiers mini A, on est décalé de deux mesures ».

Louis, 2026-09-30. Page de diagnostic : pour chaque hypothèse (grille de
battues, accords, ses sections, recherche de reprises, affichage), la mesure
qui la confirme ou l'écarte, sur le même axe de mesures, avec l'audio
cliquable. Rien d'agrégé : un morceau, terme par terme.

LECTURE SEULE sous `state/`.

    .venv/bin/python -m tools.the_walk_decalage
    -> docs/plots/the_walk_decalage.html
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from harmonia import musx
from harmonia.sections import similarity as VS
from harmonia.sections import simulation as SIM
from harmonia.settings import SETTINGS
from tools.occurrences_brique_page import CSS, JS

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "docs" / "plots" / "the_walk_decalage.html"
STEM = "mayer_hawthorne_the_walk"
NOMS = ["C", "D♭", "D", "E♭", "E", "F", "G♭", "G", "A♭", "A", "B♭", "B"]
X0, W = 130, 960           # marge gauche, largeur utile (px) pour 100 mesures


def _nom(c):
    if c["nc"]:
        return "N.C."
    q = {"-7": "m7", "^7": "maj7"}.get(c["q"], c["q"])
    return NOMS[c["root"]] + q


def _couleur(c):
    if c["nc"]:
        return "#777"
    return {9: "#3f9e5a", 11: "#4d8fe0", 1: "#c9962b", 4: "#c0453a"}.get(c["root"], "#999")


def charge():
    chart = json.loads((SETTINGS.repo / "state/cache/charts" / f"min_{STEM}.json").read_text())
    beats = json.loads(next((SETTINGS.repo / "state/cache/beats").glob(f"{STEM}__[0-9]*.json")).read_text())
    secs = json.loads((SETTINGS.repo / "state/human/sections" / f"{STEM}.json").read_text())["sections"]
    g = np.array(chart["barGrid"])
    n = len(g) - 1
    # un accord par mesure : celui qui couvre le plus de la mesure
    par_mesure = []
    for i in range(n):
        best = max((c for c in chart["prompter"]["chords"]),
                   key=lambda c: min(c["t1"], g[i + 1]) - max(c["t0"], g[i]))
        par_mesure.append(best)
    tri = musx.frame_posteriors(SETTINGS.audio_dir / f"{STEM}.m4a")[0]
    S = np.asarray(VS.ssm(tri, g))
    V = np.asarray(VS.harmonic_vectors(tri, g))
    return chart, beats, secs, g, n, par_mesure, S, V


def ancien_occurrences(chart, traits):
    """`simulation.occurrences_de_tous` AVANT le correctif du 2026-09-30 : une
    reprise pouvait recouvrir un trait de Louis posé plus loin."""
    from harmonia.soudure import traits_propres
    gardes, _ = traits_propres(traits, len(chart["barGrid"]) - 1)
    pris, out = [], []
    for b0, b1, lbl in gardes:
        rep, _ = SIM.cherche_brique(chart, b0, b1, SETTINGS.audio_dir, label=lbl)
        for x, y, _sc, _r in rep:
            if any(not (y < c or d < x) for c, d in pris):
                continue
            pris.append((x, y))
            out.append({"m0": x, "m1": y, "label": lbl})
    return sorted(out, key=lambda o: o["m0"])


def corpus_traits():
    """Sur chaque découpage validé de Louis, ses premières occurrences de
    chaque lettre comme traits : ce que l'outil rendait avant / après."""
    rows = []
    for f in sorted((SETTINGS.repo / "state/human/sections").glob("*.json")):
        d = json.loads(f.read_text())
        p = SETTINGS.repo / "state/cache/charts" / f"min_{d.get('stem')}.json"
        if not d.get("validated") or not p.exists():
            continue
        ch = json.loads(p.read_text())
        vus, tr = set(), []
        for s in d["sections"]:
            if s["label"] not in vus:
                vus.add(s["label"])
                tr.append({"mesure_debut": s["b0"] + 1, "mesure_fin": s["b1"] + 1, "label": s["label"]})
        lab = {i: s["label"] for s in d["sections"] for i in range(s["b0"], s["b1"] + 1)}
        a = {(o["m0"], o["m1"], o["label"]) for o in ancien_occurrences(ch, tr)}
        b = {(o["m0"], o["m1"], o["label"]) for o in SIM.occurrences_de_tous(ch, tr, SETTINGS.audio_dir)[0]}
        if a == b:
            continue
        just = lambda S: sum(1 for x, y, lb in S for i in range(x, y + 1) if lab.get(i) == lb)  # noqa: E731
        mangés = sorted({lb for x, y, lb in b - a})
        rows.append((d["stem"], just(a), just(b), mangés))
    return rows


def xb(i, n):
    return X0 + W * i / n


def play(t0, t1):
    return f'class="play" data-stem="{STEM}" data-t0="{t0:.2f}" data-t1="{t1:.2f}"'


def svg_grille(g, beats, n):
    """Panneau 1 : les premiers temps bruts du traqueur contre la grille."""
    db = np.array(beats["downbeats"])
    res = [(i, 1000 * (db[np.argmin(abs(db - g[i]))] - g[i])) for i in range(n)]
    dur = np.diff(g)
    H, y0 = 150, 75
    out = [f'<svg class="plot" viewBox="0 0 {X0 + W + 20} {H + 30}">']
    for v in (-40, -20, 0, 20, 40):
        y = y0 - v * 1.5
        out.append(f'<line class="grid" x1="{X0}" x2="{X0 + W}" y1="{y}" y2="{y}"/>'
                   f'<text class="tick" x="{X0 - 6}" y="{y + 3}" text-anchor="end">{v:+d} ms</text>')
    for i, r in res:
        out.append(f'<circle {play(g[i], g[i + 1])} class="dot" cx="{xb(i + .5, n):.1f}" '
                   f'cy="{y0 - r * 1.5:.1f}" r="3"><title>mesure {i + 1} : {r:+.0f} ms</title></circle>')
    for i in range(0, n + 1, 8):
        out.append(f'<text class="tick" x="{xb(i, n):.1f}" y="{H + 12}" text-anchor="middle">{i + 1}</text>')
    out.append(f'<text class="lab" x="{X0}" y="{H + 26}">mesure → · durée de mesure : '
               f'{dur.min():.3f}–{dur.max():.3f} s (tempo fixe {60 * 4 / dur.mean():.1f} bpm)</text>')
    out.append('</svg>')
    return "".join(out)


def _bande(y, h, blocs, n, g, titre):
    """Une rangée de blocs (m0, m1, texte, couleur, contour)."""
    out = [f'<text class="lab" x="{X0 - 6}" y="{y + h * .7}" text-anchor="end">{titre}</text>']
    for m0, m1, txt, col, extra in blocs:
        x, w = xb(m0, n), xb(m1 + 1, n) - xb(m0, n)
        out.append(f'<g {play(g[m0], g[min(m1 + 1, n)])}><rect x="{x + .5:.1f}" y="{y}" width="{w - 1:.1f}" '
                   f'height="{h}" rx="2" fill="{col}" {extra}><title>mes. {m0 + 1}–{m1 + 1} {txt}</title></rect>')
        if w > 14:
            out.append(f'<text class="seclab" x="{x + w / 2:.1f}" y="{y + h * .7}" '
                       f'text-anchor="middle">{txt}</text>')
        out.append('</g>')
    return "".join(out)


def _phase_louis(m0, secs):
    """Un bloc de 4 proposé est « à ta phase » s'il commence là où une de tes
    sections commence, ou 4 mesures après (le milieu d'un A de 8)."""
    departs = set()
    for s in secs:
        departs.update(range(s["b0"], s["b1"] + 1, 4))
    return m0 in departs


def svg_bandes(g, n, par_mesure, secs, occ_seul, occ_trois, lo=0, hi=None):
    hi = n if hi is None else hi
    rows = []
    rows.append(("accords entendus", [(i, i, _nom(c).replace("maj7", "△").replace("m7", "m"),
                                       _couleur(c), "") for i, c in enumerate(par_mesure)], 18))
    rows.append(("tes sections", [(s["b0"], s["b1"], s["label"], "rgba(128,128,128,.35)",
                                   'stroke="var(--fg)" stroke-width=".6"') for s in secs], 18))

    def blocs(occ):
        return [(o["m0"], o["m1"], o["label"],
                 "rgba(63,158,90,.55)" if _phase_louis(o["m0"], secs) else "rgba(192,69,58,.75)", "")
                for o in occ]
    rows.append(("outil : tu donnes A", blocs(occ_seul), 18))
    rows.append(("outil : intro+A+B, avant", blocs(occ_trois[0]), 18))
    rows.append(("intro+A+B, corrigé", blocs(occ_trois[1]), 18))
    H = 30 + 26 * len(rows) + 20
    out = [f'<svg class="plot" viewBox="0 0 {X0 + W + 20} {H}">']
    for i in range(0, n + 1, 4):
        out.append(f'<line class="grid" x1="{xb(i, n):.1f}" x2="{xb(i, n):.1f}" y1="8" y2="{H - 18}"/>')
    for i in range(0, n, 4):
        out.append(f'<text class="tick" x="{xb(i, n) + 2:.1f}" y="{H - 6}">{i + 1}</text>')
    y = 12
    for titre, bl, h in rows:
        out.append(_bande(y, h, bl, n, g, titre))
        y += 26
    out.append('</svg>')
    return "".join(out)


def svg_courbe(g, n, c, pris, secs):
    """Panneau 3 : ressemblance de la boucle de 4 (mes. 9–12) posée à chaque mesure."""
    H, top, bot = 230, 15, 190
    N = len(c)

    def yv(v):
        return bot - (v - 0.0) / 1.0 * (bot - top)
    out = [f'<svg class="plot" viewBox="0 0 {X0 + W + 20} {H}">']
    for v in (0, .25, .5, .75, .85, 1.0):
        out.append(f'<line class="{"thr" if v == .85 else "grid"}" x1="{X0}" x2="{X0 + W}" y1="{yv(v):.1f}" '
                   f'y2="{yv(v):.1f}"/><text class="tick" x="{X0 - 6}" y="{yv(v) + 3:.1f}" '
                   f'text-anchor="end">{v:.2f}</text>')
    pts = " ".join(f"{xb(q + .5, n):.1f},{yv(c[q]):.1f}" for q in range(N))
    out.append(f'<polyline class="curve" points="{pts}"/>')
    for q in range(N):
        col = "var(--curve)"
        r = 2
        if q in pris:
            col = "#3f9e5a" if _phase_louis(q, secs) else "#c0453a"
            r = 5
        out.append(f'<circle {play(g[q], g[min(q + 4, n)])} cx="{xb(q + .5, n):.1f}" cy="{yv(c[q]):.1f}" '
                   f'r="{r}" fill="{col}"><title>départ mes. {q + 1} : {c[q]:.3f}'
                   f'{" — CHOISI" if q in pris else ""}</title></circle>')
    for q in (26, 28, 50, 48):
        out.append(f'<text class="thrlab" x="{xb(q + .5, n):.1f}" y="{yv(c[q]) - 9:.1f}" '
                   f'text-anchor="middle" style="fill:var(--fg)">{q + 1}: {c[q]:.2f}</text>')
    for i in range(0, n + 1, 4):
        out.append(f'<text class="tick" x="{xb(i, n) + 2:.1f}" y="{H - 22}">{i + 1}</text>')
    out.append(f'<text class="lab" x="{X0}" y="{H - 6}">départ de la boucle (mesure) → · gros point = '
               f'départ retenu (vert : à ta phase, rouge : décalé de 2) · pointillé rouge = plancher 0,85</text>')
    out.append('</svg>')
    return "".join(out)


def table_zoom(g, par_mesure, secs, occ_seul, lo, hi):
    lab = {}
    for s in secs:
        for i in range(s["b0"], s["b1"] + 1):
            lab[i] = s["label"] + (" (début)" if i == s["b0"] else "")
    outil = {}
    for o in occ_seul:
        for i in range(o["m0"], o["m1"] + 1):
            outil[i] = f'A (bloc {o["m0"] + 1}–{o["m1"] + 1})' + (" ← début" if i == o["m0"] else "")
    rows = ['<table><tr><th>mesure</th><th>temps</th><th>accord entendu</th><th>ta section</th>'
            '<th>l\'outil (tu donnes A)</th><th></th></tr>']
    for i in range(lo, hi + 1):
        o = outil.get(i, "—")
        cls = ""
        if "début" in o:
            cls = "ok" if _phase_louis(int(o.split("bloc ")[1].split("–")[0]) - 1, secs) else "ko"
        rows.append(f'<tr><td class="num">{i + 1}</td><td class="num">{g[i]:.1f} s</td>'
                    f'<td>{_nom(par_mesure[i])}</td><td>{lab.get(i, "")}</td><td class="{cls}">{o}</td>'
                    f'<td><span class="lnk" {play(g[max(i - 1, 0)], g[i + 3])}>▶ écouter</span></td></tr>')
    rows.append('</table>')
    return "".join(rows)


def main():
    chart, beats, secs, g, n, par_mesure, S, V = charge()
    T = lambda a, b, lbl: {"mesure_debut": a + 1, "mesure_fin": b + 1, "label": lbl}  # noqa: E731
    occ_seul, _, _ = SIM.occurrences_de_tous(chart, [T(8, 15, "A")], SETTINGS.audio_dir)
    trois = [T(0, 7, "intro"), T(8, 15, "A"), T(24, 35, "B")]
    occ_trois = (ancien_occurrences(chart, trois), SIM.occurrences_de_tous(chart, trois, SETTINGS.audio_dir)[0])
    corpus = corpus_traits()
    _out, det = SIM.cherche_sur_matrices(S, V, 8, 15)
    c = det["courbe"]
    pris = {p["p"] for p in det["pics"] if p["pris"]}
    pk = {p["p"]: p for p in det["pics"]}

    corps = [f"""
<h1>The Walk — d'où vient le décalage de 2 mesures</h1>
<p class="sub">Louis, 2026-09-30 : « après les 3 premiers mini A plus ou moins, on commence à être décalé de deux
barres ». Cinq hypothèses, une mesure chacune. Clique n'importe quel point, bloc ou ▶ pour écouter ce passage.</p>
<div id="player">clique un bloc pour écouter</div>
<div class="say"><b>Réponse.</b> La musique, la grille et tes sections sont justes d'un bout à l'autre. Le décalage
vient de l'outil d'annotation : quand tu donnes A (mes. 9–16), il cherche sa boucle de 4 mesures (A–Bm–A–Bm) ailleurs.
À partir de la mesure 27 il la pose <b>2 mesures trop tôt</b> (27, 31, 35… au lieu de 29, 33, 37…) — parce qu'en
commençant à 27 il en case <b>7</b> dans le passage 25–56 au lieu de 6, et il préfère le plus grand nombre de
boucles à la ressemblance. Les deux boucles de bord (27–30 et 51–54) commencent sur le C♯m du B : elles ne
ressemblent qu'à 0,91, contre 0,99+ pour les bonnes.</div>
""",
             "<h2>1. La grille de mesures — écartée</h2>",
             '<div class="say">Ce que ça montre : chaque premier temps détecté par le traqueur (beatthis) tombe à '
             '±20 ms de la ligne de mesure du chart, de la mesure 1 à la 100. Le morceau est à tempo fixe '
             '(110 bpm, mesures de 2,182 s partout). Aucune battue perdue ni ajoutée : rien ne dérive.</div>',
             '<div class="tablewrap">' + svg_grille(g, beats, n) + '</div>',
             "<h2>2. Accords entendus, tes sections, ce que l'outil propose</h2>",
             '<div class="say">Ce que ça montre : les accords (vert A, bleu Bm, orange C♯m, rouge E7) alternent '
             'A / Bm sans jamais changer de parité — le A tombe toujours sur une mesure impaire (1, 3, 5… en '
             'numérotation de l\'app). Tes sections sont calées dessus : chaque B commence sur son C♯m (25, 53, 81), '
             'le pont sur 65. L\'outil, lui, pose ses boucles A en vert tant qu\'il est à ta phase, puis en '
             '<b>rouge = décalé de 2</b> à partir de 27, de nouveau juste à 57, puis décalé encore à 83. '
             'Dernière rangée : si tu as aussi tracé B (25–36), les A décalés le recouvrent et <b>ton B disparaît</b> '
             'de la réponse (deuxième défaut, corrigé : dernière rangée, et section 6).</div>',
             '<div class="legend"><span style="background:#3f9e5a"></span>A <span style="background:#4d8fe0"></span>Bm'
             '<span style="background:#c9962b"></span>C♯m <span style="background:#c0453a"></span>E7 '
             '<span style="background:rgba(63,158,90,.55)"></span>bloc proposé à ta phase '
             '<span style="background:rgba(192,69,58,.75)"></span>bloc proposé décalé</div>',
             '<div class="tablewrap">' + svg_bandes(g, n, par_mesure, secs, occ_seul, occ_trois) + '</div>',
             "<h2>3. Pourquoi l'outil choisit la mauvaise phase</h2>",
             f'<div class="say">Ce que ça montre : la courbe dit « à quel point la boucle A–Bm–A–Bm (mes. 9–12) '
             f'ressemble aux 4 mesures qui commencent ici ». Un départ sur deux vaut ~0,99 (deux accords qui '
             f'alternent : décaler de 2 mesures donne la même chose). Au bord d\'un B, le départ 27 '
             f'(C♯m–Bm–A–Bm) ne vaut que {c[26]:.3f}, contre {c[28]:.3f} au 29. Mais les deux passent les règles : '
             f'plancher 0,85, et hauteur relative {pk[26]["rho"]:.2f} ≥ 0,6 (le « fond » d\'un départ est la courbe '
             f'aux décalages impairs, ~0,2). La règle 4 (trou entre les hauteurs de pics) ne coupe pas : il n\'y a '
             f'que 7 pics bas contre 29 hauts. Reste le choix final : « le plus de boucles sans recouvrement, à '
             f'égalité les mieux notées ». 7 boucles depuis 27 battent 6 depuis 29 — le nombre gagne toujours sur '
             f'la ressemblance (+1 par boucle, contre au plus +0,09 de score).</div>',
             '<div class="tablewrap">' + svg_courbe(g, n, c, pris, secs) + '</div>',
             "<h2>4. Zoom : là où ça bascule (mes. 21–38) et où ça revient (mes. 49–58)</h2>",
             '<div class="say">Ce que ça montre : mesure par mesure, l\'accord entendu, ta section et le bloc '
             'proposé. Écoute 25–28 : c\'est le C♯m du B, et l\'outil y fait commencer un A à 27. '
             'Il revient à ta phase à 57 parce que le C♯m de 53–56 lui barre la route une seconde fois.</div>',
             '<div class="tablewrap">' + table_zoom(g, par_mesure, secs, occ_seul, 20, 37) + '</div>',
             '<div class="tablewrap">' + table_zoom(g, par_mesure, secs, occ_seul, 48, 57) + '</div>',
             "<h2>5. L'affichage — écarté</h2>",
             '<div class="say">Ce que ça montre : l\'intro (mes. 1–8) joue des accords (A / Bm), donc le '
             'masquage des mesures muettes de tête ajouté aujourd\'hui (commit e2d26cb) ne la cache pas ; la '
             'numérotation et les rangées de 4 ne bougent pas. Ta marque « mesure 1 » (18,23 s) et la grille '
             f'(1re mesure à {g[0]:.2f} s) sont à 0,08 s l\'une de l\'autre. Le chart servi est tes sections, telles '
             'quelles : ' + ", ".join(f'{s["label"]} {s["b0"] + 1}–{s["b1"] + 1}' for s in secs) + '.</div>',
             "<h2>6. Le deuxième défaut, corrigé : un trait de Louis mangé par les reprises d'un autre</h2>",
             '<div class="say">Ce que ça montre : sur tes découpages validés, en traçant la première occurrence de '
             'chaque lettre, les reprises d\'une lettre pouvaient recouvrir un trait que tu avais posé PLUS LOIN '
             'dans le morceau — ton trait disparaissait alors de la réponse (The Walk : ton B 25–36 sous les A '
             'décalés). Correctif : une reprise ne recouvre plus aucun de tes traits. Mesures bien étiquetées, '
             'avant → après ; aucune ne baisse. Les charts ne bougent pas (rapport d\'or : 0 mesure changée) — '
             'seules les propositions de l\'outil d\'annotation changent.</div>',
             '<div class="tablewrap"><table><tr><th>morceau</th><th>mesures justes avant</th><th>après</th>'
             '<th>lettres de tes traits qui revenaient</th></tr>' + "".join(
                 f'<tr><td>{st}</td><td class="num">{a}</td><td class="num ok">{b}</td><td>{", ".join(m)}</td></tr>'
                 for st, a, b, m in corpus) + '</table></div>',
             ]
    page = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>The Walk décalage</title>
<style>{CSS}</style></head><body>{"".join(corps)}<script>{JS}</script></body></html>"""
    PAGE.write_text(page, encoding="utf-8")
    print(PAGE)
    print("tu donnes A :", [(o["m0"] + 1, o["m1"] + 1) for o in occ_seul])
    print("intro+A+B   :", [(o["m0"] + 1, o["m1"] + 1, o["label"]) for o in occ_trois[1]])
    for r in corpus:
        print(r)


if __name__ == "__main__":
    main()
