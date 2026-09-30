"""Occurrences d'une brique, v2 : les trois règles de Louis, terme par terme.

Louis, 2026-09-30, sur `docs/plots/occurrences_brique_2026-09-30.html` :
« le détecteur avec pics est bon », trois changements —
  1. pas de pic dans la zone donnée ;
  2. chercher la BOUCLE MINIMALE de la brique, pas la brique ;
  3. juger un pic par rapport au reste de la courbe, pas par sa hauteur.
Implémentées dans `harmonia.sections.simulation.cherche_sur_matrices` (le
chemin live de « Valider les sections »). Cette page compare AVANT (la loi du
matin, réécrite à l'identique dans `tools.occurrences_brique.cherche_fixe`) et
APRÈS (le code live), sur ses découpages validés, avec ses deux verdicts :
  * Stand By Me : tout le morceau peut être UNE section (A et C = même famille) ;
  * The Walk : B peut être plus court — ce que A trouve dans l'intro et dans B
    est accepté.

LECTURE SEULE. Ne modifie rien sous `state/`.

    .venv/bin/python -m tools.occurrences_brique_v2            # texte, corpus
    .venv/bin/python -m tools.occurrences_brique_v2 --page     # + la page
"""
from __future__ import annotations

import argparse
import glob
import html
import json
from pathlib import Path

import numpy as np

from harmonia import musx
from harmonia.sections import similarity as VS
from harmonia.sections import simulation as SIM
from harmonia.settings import SETTINGS
from tools import occurrences_brique as OB
from tools.occurrences_brique_page import CSS, JS

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "docs" / "plots" / "occurrences_brique_v2_2026-09-30.html"

#: les verdicts de Louis du 2026-09-30 : lettre → familles qui comptent juste
MEME_FAMILLE = {"ben_e_king_stand_by_me_audio": {"A": {"A", "C"}, "C": {"A", "C"}}}
#: ... et ce qui, trouvé en trop, est accepté
TOLERE = {"mayer_hawthorne_the_walk": {"A": {"intro", "B"}}}

#: (stem, titre) des morceaux détaillés — ceux de la page du matin
DETAIL = [
    ("9zEl-FQLI4A", "San Francisco Street — Sun Rai"),
    ("ben_e_king_stand_by_me_audio", "Stand By Me — Ben E. King"),
    ("let_it_be_remastered_2009", "Let It Be — The Beatles"),
    ("mayer_hawthorne_the_walk", "The Walk — Mayer Hawthorne"),
    ("aretha_franklin_chain_of_fools_official_lyric_video", "Chain of Fools — Aretha Franklin"),
    ("the_police_every_breath_you_take_official_music_video",
     "Every Breath You Take — The Police"),
]

#: « ce que ça montre », une ligne par lettre détaillée — écrit en lisant les
#: chiffres de `python -m tools.occurrences_brique_v2` (2026-09-30)
DIRES = {
    ("9zEl-FQLI4A", "A"):
        "Ta brique de 24 mesures, c'est la boucle de 4 jouée 5 fois, puis une autre fin "
        "(mes. 29–32). On cherche la <b>boucle</b> (règle 2) : ta reprise de 16 mesures "
        "(42–57) est retrouvée en entier — 3 boucles puis la même fin. Avant, elle était ratée. "
        "Il ne manque que la mesure 71 (une mesure seule). 89–92 tombe dans la partie du "
        "morceau que tu n'as pas encore annotée.",
    ("ben_e_king_stand_by_me_audio", "A"):
        "Tes reprises (A et C : ton verdict, c'est une seule section) sont des pics à "
        "0,90–0,94 sur un fond à 0,66–0,70. Leur proéminence (0,70–0,81) passe la barre de 0,6 "
        "(règle 3). L'ancien seuil fixe (0,918) coupait ce plateau au hasard : 4 passages sur 10. "
        "Maintenant : les 10, rien en trop.",
    ("ben_e_king_stand_by_me_audio", "C"):
        "Vu depuis C : 8 passages sur 9. 15–22 est refusé de justesse (score 0,866, "
        "proéminence 0,57 &lt; 0,6) — le passage le moins net du morceau vu depuis C. "
        "Avant, ils étaient tous trouvés : le seuil était au plancher 0,85.",
    ("let_it_be_remastered_2009", "A"):
        "Ton couplet de 8 mesures = 2 fois la même boucle de 4. Tes 3 reprises (0,99–1,0) "
        "sont retrouvées ; avant : aucune, le seuil butait au plafond 0,999. <b>Mais</b> "
        "cette boucle de 4 est aussi jouée dans tes B — le refrain finit sur les mêmes accords "
        "(0,90–0,91, proéminence 0,63–0,73) : 24 mesures en trop. Des jumeaux harmoniques : "
        "une boucle de 4 ressemble à plus de choses qu'une brique de 8.",
    ("mayer_hawthorne_the_walk", "A"):
        "Deux accords qui alternent (ils tournent toutes les 2 mesures), donc boucle de 4. "
        "Tes 4 reprises sont trouvées (avant : une seule, décalée de 4). Ce qui est en plus tombe "
        "dans l'intro et dans la 2e moitié de tes B — accepté par ton verdict. Seules "
        "6 mesures sortent : l'outro (93–98) rejoue la boucle.",
    ("mayer_hawthorne_the_walk", "B"):
        "Ton B de 12 = la boucle de 4 jouée 3 fois (fin un peu différente). Dans chaque reprise "
        "de B (53–64, 81–92), 2 passages sont trouvés, mais <b>décalés de 2 mesures</b> "
        "(51–58, 79–86), et le 3e manque. Deux accords qui alternent : l'harmonie ne peut pas "
        "dire sur lequel la boucle commence. Tu as dit que B peut être plus court — à toi de "
        "dire si ce décalage te gêne.",
    ("aretha_franklin_chain_of_fools_official_lyric_video", "A"):
        "Un vamp : tout ressemble à tout (0,99 partout). Le fond local est lui-même à 0,99, "
        "aucun départ ne s'en détache (proéminence ≤ 0,06) : on s'abstient, comme avant, et on "
        "le dit. L'harmonie ne peut pas dire où ton A recommence ; il faudrait un autre canal "
        "(batterie, voix).",
    ("the_police_every_breath_you_take_official_music_video", "A"):
        "Pas de boucle dans ta brique (elle reste de 8). Les 4 reprises exactes et 59–66 "
        "(fin différente, 0,90) sont gardées ; l'intro (1–8) n'est plus prise — la règle 1 "
        "retire ta brique du fond, et l'intro ne se détache plus. Restent ratés : 83–84 "
        "(ton A de 10 mesures) et 89–111, la boucle de fin (ta question du matin : A ou A′ ?).",
    ("the_police_every_breath_you_take_official_music_video", "C"):
        "Ta brique de 10 = une boucle de 4 jouée 2 fois, puis 2 mesures qui la recommencent. "
        "Elle est découpée (41–44, 45–48, 49–50) ; aucune autre reprise, et c'est juste : tu "
        "n'as mis C qu'une fois.",
}

INTRO = """<div class="sub">Ce matin, tu as dit « le détecteur avec pics est bon » et donné trois règles.
Elles sont <b>branchées dans « Valider les sections »</b> (actives au prochain redémarrage du serveur).
Même simulation que ce matin : pour chaque lettre de tes découpages validés, sa première occurrence
devient la brique. Tes deux verdicts comptent comme vérité : Stand By Me = une seule section ;
The Walk, B peut être plus court (ce que A trouve dans l'intro et dans B est accepté).</div>
<div class="say"><b>Les trois règles, telles qu'implémentées.</b>
(1) <b>Zone donnée</b> : un départ dont la fenêtre chevauche ta brique, même d'une mesure, n'est jamais
une reprise — et il ne compte pas non plus dans le fond qui sert à juger les autres.
(2) <b>Boucle minimale</b> : si ta brique (8 mesures ou plus) est k fois une boucle d'au moins 4 mesures,
chaque mesure collant à celle d'une boucle plus loin, avec une dernière fois qui peut finir autrement,
elle devient ses k passages et c'est la boucle qu'on cherche ; sa fin est recherchée juste après les boucles
retrouvées. « Coller » = à mi-chemin entre identique et la ressemblance médiane des mesures de ta brique
entre elles (relatif à ta brique, pas un chiffre fixe).
(3) <b>Hauteur relative</b> : pour chaque départ, le fond local = la médiane de la courbe aux départs
décalés de 1 à L−1 mesures (la boucle mal calée). Un départ est une reprise s'il a parcouru au moins
<b>60 % du chemin entre ce fond et 1,0</b> (et reste au-dessus de 0,85). Plus de seuil global, plus de
plafond 0,999. Une brique de moins de 8 mesures ne peut pas montrer une boucle jouée deux fois : elle
garde l'ancienne loi (seuil du morceau), calculée hors de la zone donnée.</div>"""

RESUME = """<div class="say">Sur tes 20 découpages validés, en mesures : <b>justes 1270 → 1402</b>,
<b>ratées 260 → 128</b>, <b>en trop 278 → 408</b>. Presque tout ce qui est en plus en trop, ce sont des
jumeaux harmoniques — une autre section qui joue la même boucle : Let It Be A (le refrain), She Will Be
Loved A et D (la même grille presque tout le morceau, comme Stand By Me), This Love A (l'intro),
X-yIEMduRXk B. Chercher une boucle de 4 au lieu d'une brique de 8 les rend plus fréquents.
Le tableau ne montre que les lettres qui changent.</div>"""


def _esc(x) -> str:
    return html.escape(str(x))


def _mesures(occ) -> set[int]:
    out: set[int] = set()
    for a, b, *_r in occ:
        out |= set(range(a, b + 1))
    return out


def analyse_morceau(stem: str) -> dict | None:
    d = json.loads((SETTINGS.sections_dir / f"{stem}.json").read_text(encoding="utf-8"))
    if not d.get("validated"):
        return None
    c = OB._chart_for(stem, d["n"])
    if c is None:
        return None
    secs = sorted(d["sections"], key=lambda s: s["b0"])
    audio = SETTINGS.audio_dir / (Path(c["audio_url"]).stem + ".m4a")
    grid = c["barGrid"]
    V = VS.harmonic_vectors(musx.frame_posteriors(audio)[0], grid)
    S = V @ V.T
    n = S.shape[0]
    fin_annot = max(s["b1"] for s in secs)
    lettres, vues = [], set()
    for s in secs:
        lab = OB._fam(s["label"])
        if lab.lower() in OB.SANS_REPRISE or lab in vues:
            continue
        vues.add(lab)
        fam = MEME_FAMILLE.get(stem, {}).get(lab, {lab})
        vraies = [(x["b0"], x["b1"], x["label"]) for x in secs if OB._fam(x["label"]) in fam]
        if len([v for v in vraies if OB._fam(v[2]) == lab]) < 2 and len(vraies) < 2:
            # une lettre jouée une seule fois : on la montre quand même si la
            # brique contient une boucle (règle 2), sinon rien à retrouver
            pass
        b0, b1 = s["b0"], s["b1"]
        verite = _mesures(vraies)
        tol = set(range(fin_annot + 1, n))            # zone que Louis n'a pas annotée
        tol_verdict = set()
        for x in secs:
            if OB._fam(x["label"]) in TOLERE.get(stem, {}).get(lab, set()):
                tol_verdict |= set(range(x["b0"], x["b1"] + 1))
        avant, _thr = OB.cherche_fixe(S, V, b0, b1)
        avant = [(o["m0"], o["m1"], o.get("score", 1.0), o["rot"]) for o in avant]
        apres, det = SIM.cherche_sur_matrices(S, V, b0, b1)

        def compte(occ):
            m = _mesures(occ)
            return {"justes": len(m & verite), "ratees": len(verite - m),
                    "trop": len(m - verite - tol - tol_verdict),
                    "tolere": len((m - verite) & tol_verdict),
                    "non_annote": len((m - verite) & tol - tol_verdict)}
        lettres.append({"label": lab, "b0": b0, "b1": b1, "L": b1 - b0 + 1,
                        "vraies": vraies, "famille": sorted(fam),
                        "avant": avant, "apres": apres, "det": det,
                        "n_avant": compte(avant), "n_apres": compte(apres),
                        "verite": verite, "tol": tol, "tol_verdict": tol_verdict})
    return {"stem": stem, "n": n, "secs": secs, "grid": grid, "lettres": lettres}


def texte(res: dict) -> str:
    out = []
    for let in res["lettres"]:
        det = let["det"]
        a, p = let["n_avant"], let["n_apres"]
        chg = "" if a == p else "  <<"
        out.append(f"{res['stem'][:28]:28s} {let['label']:6s} brique {let['b0'] + 1}-{let['b1'] + 1} "
                   f"(L={let['L']}) boucle={det['boucle']} p0={det['sous_periode']} loi={det['mode']} "
                   f"vérité={len(let['verite'])}  avant j/r/t={a['justes']}/{a['ratees']}/{a['trop']}"
                   f"  après={p['justes']}/{p['ratees']}/{p['trop']} (+{p['tolere']} verdict, "
                   f"+{p['non_annote']} non annoté){chg}")
        out.append("      après : " + " ".join(f"{x + 1}-{y + 1}" + (f"↑{r}" if r else "")
                                          for x, y, _s, r in let["apres"]))
    return "\n".join(out)


# ── la page ─────────────────────────────────────────────────────────────────
def _classe_bloc(m0, m1, let, est_brique) -> str:
    if est_brique:
        return "r-brick"
    m = set(range(m0, m1 + 1))
    if len(m & let["verite"]) * 2 >= len(m):
        return "r-ok"
    if len(m & let["tol_verdict"]) * 2 >= len(m):
        return "r-tol"
    if len(m & let["tol"]) * 2 >= len(m):
        return "r-na"
    return "r-ko"


def _svg(res: dict, let: dict) -> str:
    n = res["n"]
    det = let["det"]
    Lb = det["boucle"]
    W, padL, padR = 1000, 70, 8
    bw = (W - padL - padR) / n
    x = lambda b: padL + b * bw
    y0, hC = 40, 190

    def yv(v):
        v = max(0.0, min(1.0, v))
        f = 0.4 * v / 0.85 if v < 0.85 else 0.4 + 0.6 * (v - 0.85) / 0.15
        return y0 + hC * (1 - f)
    rows = {"toi": y0 + hC + 18, "avant": y0 + hC + 40, "après": y0 + hC + 62}
    H = rows["après"] + 30
    grid, stem = res["grid"], res["stem"]
    b0, b1 = let["b0"], let["b1"]
    o = [f'<svg viewBox="0 0 {W} {H}" class="plot" role="img">']
    for s in res["secs"]:
        fam = OB._fam(s["label"])
        cls = "sec me" if fam in let["famille"] else "sec"
        t0, t1 = grid[s["b0"]], grid[min(len(grid) - 1, s["b1"] + 1)]
        o.append(f'<g class="play" data-stem="{_esc(stem)}" data-t0="{t0:.2f}" data-t1="{t1:.2f}">'
                 f'<rect class="{cls}" x="{x(s["b0"]):.1f}" y="6" width="{bw * (s["b1"] - s["b0"] + 1) - 1:.1f}" height="22" rx="3"/>'
                 f'<text x="{x(s["b0"]) + 3:.1f}" y="21" class="seclab">{_esc(s["label"])}</text>'
                 f'<title>{_esc(s["label"])} · mes. {s["b0"] + 1}–{s["b1"] + 1} · clic = écouter</title></g>')
    o.append(f'<rect class="brick" x="{x(b0):.1f}" y="4" width="{bw * (b1 - b0 + 1) - 1:.1f}" height="26" rx="3"/>')
    o.append('<text x="4" y="21" class="lab">tes sections</text>')
    # règle 1 : la zone interdite (départs dont la fenêtre chevauche la brique)
    z0 = max(0, b0 - Lb + 1)
    o.append(f'<rect class="zone" x="{x(z0):.1f}" y="{y0}" width="{bw * (b1 - z0 + 1):.1f}" height="{hC}">'
             f'<title>règle 1 : un départ ici chevauche ta brique — ni pic, ni fond</title></rect>')
    for v in (0.0, 0.5, 0.85, 0.9, 0.95, 1.0):
        o.append(f'<line class="grid" x1="{padL}" x2="{W - padR}" y1="{yv(v):.1f}" y2="{yv(v):.1f}"/>'
                 f'<text x="{padL - 4}" y="{yv(v) + 3:.1f}" class="tick" text-anchor="end">{v:.2f}</text>')
    for g0, _g1, _l in let["vraies"]:
        o.append(f'<rect class="truthband" x="{x(g0):.1f}" y="{y0}" width="{bw:.1f}" height="{hC}"/>')
    c = det["courbe"]
    cx = lambda i: x(i) + bw / 2
    o.append('<polyline class="curve" points="' + " ".join(f"{cx(i):.1f},{yv(v):.1f}" for i, v in enumerate(c)) + '"/>')
    pics = det.get("pics") or []
    if pics:
        # le fond local, et la barre à franchir fond + 0,6 (1 − fond)
        # une polyligne par segment continu (la zone donnée fait un trou)
        segs, cur = [], []
        for q in pics:
            if cur and q["p"] != cur[-1]["p"] + 1:
                segs.append(cur); cur = []
            cur.append(q)
        segs.append(cur)
        for seg in segs:
            o.append('<polyline class="fond" points="' + " ".join(f"{cx(q['p']):.1f},{yv(q['fond']):.1f}" for q in seg) + '"/>')
            barre = [(q["p"], q["fond"] + SIM.PROEMINENCE_MIN * max(1 - q["fond"], SIM.ECART_FOND_MIN)) for q in seg]
            o.append('<polyline class="barre" points="' + " ".join(f"{cx(p):.1f},{yv(max(v, SIM.PLANCHER_REPRISE)):.1f}" for p, v in barre) + '"/>')
        for q in pics:
            if q["c"] < SIM.PLANCHER_REPRISE:
                continue
            cls = "pk-pris" if q["pris"] else ("pk-ok" if q["ok"] else "pk-non")
            o.append(f'<circle class="{cls}" cx="{cx(q["p"]):.1f}" cy="{yv(q["c"]):.1f}" r="{4 if q["pris"] else 3}">'
                     f'<title>départ mes. {q["p"] + 1} · score {q["c"]:.3f} · fond {q["fond"]:.3f} · '
                     f'proéminence {q["rho"]:.2f} → {"GARDÉ" if q["pris"] else ("passe, non retenu (recouvrement)" if q["ok"] else "refusé")}</title></circle>')
            if q["pris"] or (q["c"] >= 0.85 and not q["ok"]):
                o.append(f'<text x="{cx(q["p"]):.1f}" y="{yv(q["c"]) - 6:.1f}" class="rho" text-anchor="middle">{q["rho"]:.2f}</text>')
    elif det.get("seuil"):
        s = det["seuil"]
        o.append(f'<line class="thr" x1="{padL}" x2="{W - padR}" y1="{yv(s):.1f}" y2="{yv(s):.1f}"/>'
                 f'<text x="{W - padR - 2}" y="{yv(s) - 3:.1f}" class="thrlab" text-anchor="end">seuil du morceau {s:.3f} (brique &lt; 8 mes. : ancienne loi)</text>')
    o.append(f'<text x="4" y="{y0 + 12}" class="lab">ressemb.</text>')
    o.append(f'<text x="4" y="{y0 + 26}" class="lab">boucle {Lb}</text>')

    def ligne(nom, occs, brique_fn):
        yy = rows[nom]
        o.append(f'<text x="4" y="{yy + 11}" class="lab">{nom}</text>')
        for m0, m1, sc, rot in occs:
            cls = brique_fn(m0, m1)
            t0, t1 = grid[m0], grid[min(len(grid) - 1, m1 + 1)]
            r = f" ↑{rot}" if rot else ""
            o.append(f'<g class="play" data-stem="{_esc(stem)}" data-t0="{t0:.2f}" data-t1="{t1:.2f}">'
                     f'<rect class="{cls}" x="{x(m0):.1f}" y="{yy}" width="{bw * (m1 - m0 + 1) - 1:.1f}" height="15" rx="2"/>'
                     + (f'<text x="{x(m0) + 2:.1f}" y="{yy + 11}" class="rot">{r}</text>' if r else "")
                     + f'<title>{nom} · mes. {m0 + 1}–{m1 + 1}{r} · {sc:.3f} · clic = écouter</title></g>')
    ligne("toi", [(g0, g1, 1.0, 0) for g0, g1, _l in let["vraies"]],
          lambda m0, m1: "r-brick" if b0 <= m0 and m1 <= b1 else "r-truth")
    ligne("avant", let["avant"], lambda m0, m1: _classe_bloc(m0, m1, let, (m0, m1) == (b0, b1)))
    ligne("après", let["apres"], lambda m0, m1: _classe_bloc(m0, m1, let, b0 <= m0 and m1 <= b1))
    for b in range(0, n, 8):
        o.append(f'<text x="{x(b):.1f}" y="{H - 4}" class="tick">{b + 1}</text>')
    o.append("</svg>")
    return "".join(o)


def _pics_table(let: dict) -> str:
    pics = [q for q in let["det"].get("pics") or [] if q["c"] >= SIM.PLANCHER_REPRISE]
    if not pics:
        return ""
    rows = []
    for q in pics:
        m = set(range(q["p"], q["p"] + let["det"]["boucle"]))
        ou = ("dans ta vérité" if len(m & let["verite"]) * 2 >= len(m) else
              "accepté (ton verdict)" if len(m & let["tol_verdict"]) * 2 >= len(m) else
              "zone non annotée" if len(m & let["tol"]) * 2 >= len(m) else "hors vérité")
        etat = "gardé" if q["pris"] else ("passe, recouvre un autre" if q["ok"] else "refusé")
        cls = "ok" if q["pris"] else ("part" if q["ok"] else "ko")
        rows.append(f'<tr><td class="num">{q["p"] + 1}</td><td class="num">{q["c"]:.3f}</td>'
                    f'<td class="num">{q["fond"]:.3f}</td><td class="num">{q["rho"]:.2f}</td>'
                    f'<td class="{cls}">{etat}</td><td>{ou}</td></tr>')
    return ('<details><summary>les départs au-dessus de 0,85, un par un '
            f'({len(pics)})</summary><table><tr><th class="num">départ</th><th class="num">score</th>'
            '<th class="num">fond local</th><th class="num">proéminence</th><th>décision</th>'
            '<th>où ça tombe</th></tr>' + "".join(rows) + '</table></details>')


def _chiffres(let: dict) -> str:
    a, p = let["n_avant"], let["n_apres"]
    return (f'<table class="mini"><tr><th></th><th class="num">mesures justes</th><th class="num">ratées</th>'
            f'<th class="num">en trop</th><th class="num">acceptées par ton verdict</th><th class="num">zone non annotée</th></tr>'
            f'<tr><td>avant</td><td class="num">{a["justes"]}</td><td class="num">{a["ratees"]}</td><td class="num">{a["trop"]}</td>'
            f'<td class="num">{a["tolere"]}</td><td class="num">{a["non_annote"]}</td></tr>'
            f'<tr><td>après</td><td class="num">{p["justes"]}</td><td class="num">{p["ratees"]}</td><td class="num">{p["trop"]}</td>'
            f'<td class="num">{p["tolere"]}</td><td class="num">{p["non_annote"]}</td></tr></table>'
            f'<div class="sub">sur {len(let["verite"])} mesures de ta vérité pour cette lettre</div>')


CSS2 = """
code { overflow-wrap:anywhere; }
.zone { fill:rgba(128,128,128,.18); }
.fond { fill:none; stroke:#4d8fe0; stroke-width:1.2; }
.barre { fill:none; stroke:#e0584b; stroke-width:1.1; stroke-dasharray:5 3; }
.pk-pris { fill:var(--ok); stroke:var(--fg); stroke-width:1; }
.pk-ok { fill:var(--mid); } .pk-non { fill:none; stroke:var(--ko); stroke-width:1.4; }
.rho { font-size:8.5px; fill:var(--mut); }
.r-tol { fill:#4d8fe0; } .r-na { fill:rgba(128,128,128,.55); }
table.mini { width:auto; } table.mini td, table.mini th { padding:3px 10px; }
details { font-size:.82rem; margin:4px 0 10px; } summary { cursor:pointer; color:var(--mut); }
"""


def ecrire_page(tous: list[dict], chemin: Path, dires: dict, intro: str, resume: str) -> None:
    par = {r["stem"]: r for r in tous}
    corps = []
    for stem, titre in DETAIL:
        r = par.get(stem)
        if not r:
            continue
        corps.append(f'<h2>{_esc(titre)}</h2>')
        for let in r["lettres"]:
            cle = (stem, let["label"])
            if cle not in dires:
                continue
            det = let["det"]
            boucle = (f'boucle minimale trouvée : <b>{det["boucle"]} mesures</b>'
                      + (f', jouée {det["suite"] // det["boucle"]} fois'
                         + (" + une fin" if det["suite"] // det["boucle"] * det["boucle"] < let["L"] else "")
                         if det["boucle"] < let["L"] else " (la brique ne se répète pas en elle-même)")
                      + (f' · les accords tournent toutes les {det["sous_periode"]} mesures'
                         if det["sous_periode"] < det["boucle"] else ""))
            corps.append(f'<h3>Tu donnes « {_esc(let["label"])} » = mesures {let["b0"] + 1}–{let["b1"] + 1} '
                         f'({let["L"]} mes.) — {boucle}</h3>')
            corps.append(f'<div class="say"><b>Ce que ça montre.</b> {dires[cle]}</div>')
            corps.append('<div class="tablewrap">' + _svg(r, let) + '</div>')
            corps.append('<div class="tablewrap">' + _chiffres(let) + _pics_table(let) + '</div>')
    # le tableau de tout le corpus, par lettre
    lignes = []
    for r in tous:
        for let in r["lettres"]:
            a, p = let["n_avant"], let["n_apres"]
            if a == p:
                continue
            lignes.append(f'<tr><td>{_esc(r["stem"][:34])}</td><td>{_esc(let["label"])}</td>'
                          f'<td class="num">{let["L"]}</td><td class="num">{let["det"]["boucle"]}</td>'
                          f'<td class="num">{len(let["verite"])}</td>'
                          f'<td class="num">{a["justes"]} → {p["justes"]}</td>'
                          f'<td class="num">{a["trop"]} → {p["trop"]}</td>'
                          f'<td class="num">{p["tolere"] + p["non_annote"]}</td></tr>')
    page = f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Occurrences d'une brique v2</title>
<style>{CSS}{CSS2}</style></head><body>
<h1>Tes trois règles, branchées — avant / après</h1>
{intro}
<div class="legend">
Courbe : ressemblance de ta <b>boucle</b> posée à partir de chaque mesure (trait plein).
<span style="background:rgba(128,128,128,.3)"></span>zone donnée (règle 1)
<span style="background:#4d8fe0"></span>fond local
<span style="background:#e0584b"></span>barre à franchir (fond + 60 % du chemin jusqu'à 1,0 ; jamais sous 0,85)
· points : <span style="background:var(--ok)"></span>gardé
<span style="background:var(--mid)"></span>passe mais recouvre un autre · cercle rouge = refusé ; le chiffre = la proéminence.
Bandes vertes = départs de tes vraies occurrences. Lignes : <span style="background:var(--ok)"></span>juste
<span style="background:#4d8fe0"></span>accepté par ton verdict <span style="background:rgba(128,128,128,.55)"></span>zone
que tu n'as pas annotée <span style="background:var(--ko)"></span>en trop ; cadre = ta brique (ou ses passages).
<b>Clic sur n'importe quel bloc = écouter.</b></div>
<div id="player">clic sur un bloc pour écouter</div>
{"".join(corps)}
<h2>Tout le corpus : les lettres qui changent</h2>
{resume}
<div class="tablewrap"><table><tr><th>morceau</th><th>lettre</th><th class="num">brique</th>
<th class="num">boucle</th><th class="num">mesures vraies</th><th class="num">justes avant → après</th>
<th class="num">en trop avant → après</th><th class="num">verdict / non annoté</th></tr>{"".join(lignes)}</table></div>
<p class="sub">Généré par <code>python -m tools.occurrences_brique_v2 --page</code> ; « après » = le code live
(<code>harmonia.sections.simulation.cherche_sur_matrices</code>), « avant » = la loi du matin.</p>
<script>{JS}</script>
</body></html>"""
    chemin.write_text(page, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", action="store_true")
    ap.add_argument("stems", nargs="*")
    a = ap.parse_args()
    stems = a.stems or [Path(p).stem for p in sorted(glob.glob(str(SETTINGS.sections_dir / "*.json")))]
    tous = []
    tot = np.zeros((2, 3), int)
    for s in stems:
        r = analyse_morceau(s)
        if r is None:
            continue
        tous.append(r)
        print(texte(r))
        for let in r["lettres"]:
            for i, k in enumerate(("n_avant", "n_apres")):
                tot[i] += [let[k]["justes"], let[k]["ratees"], let[k]["trop"]]
    print("TOTAL justes/ratées/en trop  avant", tot[0], " après", tot[1])
    if a.page:
        ecrire_page(tous, PAGE, DIRES, INTRO, RESUME)
        print("page :", PAGE)


if __name__ == "__main__":
    main()
