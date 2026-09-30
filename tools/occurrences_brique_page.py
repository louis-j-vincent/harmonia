"""La page de `tools.occurrences_brique` : terme par terme, jamais un score global.

Pour chaque morceau détaillé : tes sections, la courbe « ressemblance de ta
brique posée à partir de cette mesure », les seuils, et trois lignes de
résultats (prod, candidat 1, candidat 2) sur le même axe de mesures. Un clic
sur n'importe quel bloc joue ce passage (fetch + blob, pas de boucle rAF —
les deux pièges iPhone de la mémoire du projet).
"""
from __future__ import annotations

import html
import json
from pathlib import Path

from harmonia.settings import SETTINGS
from tools import occurrences_brique as OB

APP = "http://100.89.209.63:7772"

#: (stem, lettres, titre, ce que ça montre — par lettre)
DETAIL = [
    ("let_it_be_remastered_2009", "Let It Be — The Beatles", {
        "A": "Les trois reprises de ton couplet sont les trois pics les plus hauts "
             "du morceau (0,994 à 0,999). La prod exige 0,999 : son seuil a buté "
             "contre le plafond et elle ne rend RIEN. Le candidat 1 les rend toutes "
             "les trois, sans rien en trop. Type : <b>seuil trop haut</b>.",
    }),
    ("mayer_hawthorne_the_walk", "The Walk — Mayer Hawthorne", {
        "A": "Le morceau est presque une seule boucle de 2 mesures : un départ sur "
             "deux vaut ~1,0. Le seuil bute au plafond, la prod ne garde qu'un pic, "
             "et il est décalé de 4 mesures. Candidat 1 : 3 reprises justes + 1 "
             "décalée, mais 3 en trop — l'intro et la 2e moitié de chaque B jouent "
             "exactement ta boucle. Types : <b>seuil trop haut</b>, puis "
             "<b>jumeaux harmoniques</b> et <b>phase</b>.",
        "B": "Même mur : les deux reprises de B scorent 0,998–0,999 et la prod "
             "n'en rend aucune (« sans contraste »). Candidat 1 : les deux, exactes.",
    }),
    ("9zEl-FQLI4A", "San Francisco Street — Sun Rai", {
        "A": "Ta brique fait 24 mesures, la reprise en fait 16. La prod compare "
             "toujours 24 mesures d'un bloc : les 8 dernières tombent sur autre "
             "chose, la moyenne descend à 0,84 (&lt; 0,85) et elle rate une reprise "
             "dont les 16 mesures collent à 0,97–1,0 quand on la cale sur la 2e "
             "boucle de ta brique (calée sur son début, comme dans le tableau : 12 "
             "collent, les 4 dernières non). Type : <b>longueur différente</b>. Le candidat 2 "
             "n'en attrape que 12 mesures sur 16 : il ne sait partir que du début "
             "de ta brique. (Ton découpage de ce morceau est encore partiel.)",
    }),
    ("ben_e_king_stand_by_me_audio", "Stand By Me — Ben E. King", {
        "A": "A et C ont la même grille d'accords. Toutes les reprises, A comme C, "
             "scorent entre 0,90 et 0,94 : le seuil du morceau (0,918) coupe au "
             "hasard dans ce plateau — 2 A trouvés, 3 ratés, 1 C pris pour un A. "
             "Aucun seuil ne sépare ce que tu entends (arrangement, cordes, chant) : "
             "type <b>jumeaux harmoniques</b>, il faut un autre canal que les accords.",
        "C": "Vu depuis C, le même plateau : les 3 vrais C sont trouvés (0,99+), "
             "mais les 6 A aussi — ils ont les mêmes accords.",
    }),
]


def _esc(x) -> str:
    return html.escape(str(x))


def _col_score(v: float) -> str:
    if v >= OB.TAU:
        return "var(--ok)"
    if v >= 0.7:
        return "var(--mid)"
    return "var(--ko)"


def _etat_css(etat: str) -> str:
    if etat.startswith("trouvée"):
        return "part" if "au lieu de" in etat else "ok"
    if etat.startswith("décalée"):
        return "part"
    return "ko"


def _svg(res: dict, let: dict) -> str:
    """Bandes de sections + courbe + trois lignes de résultats, un seul axe."""
    n = res["n"]
    W, padL, padR = 1000, 70, 8
    bw = (W - padL - padR) / n
    x = lambda b: padL + b * bw
    y0, hC = 40, 170           # courbe : y0 .. y0+hC
    # échelle en deux morceaux : 0,2 → 0,85 sur le bas (40 %), 0,85 → 1,0 sur
    # le haut (60 %) — tout ce qui se décide se passe au-dessus de 0,85

    def yv(v):
        v = max(0.2, min(1.0, v))
        f = 0.4 * (v - 0.2) / 0.65 if v < 0.85 else 0.4 + 0.6 * (v - 0.85) / 0.15
        return y0 + hC * (1 - f)
    rows_y = {"vérité": y0 + hC + 18, "prod": y0 + hC + 40,
              "candidat 1": y0 + hC + 62, "candidat 2": y0 + hC + 84}
    H = rows_y["candidat 2"] + 30
    grid = res["grid"]
    stem = res["stem"]
    out = [f'<svg viewBox="0 0 {W} {H}" class="plot" role="img">']
    # sections de Louis
    for s in res["secs"]:
        lab = s["label"]
        fam = OB._fam(lab)
        cls = "sec me" if fam == let["label"] else "sec"
        t0, t1 = grid[s["b0"]], grid[min(len(grid) - 1, s["b1"] + 1)]
        out.append(f'<g class="play" data-stem="{_esc(stem)}" data-t0="{t0:.2f}" data-t1="{t1:.2f}">'
                   f'<rect class="{cls}" x="{x(s["b0"]):.1f}" y="6" width="{bw * (s["b1"] - s["b0"] + 1) - 1:.1f}" height="22" rx="3"/>'
                   f'<text x="{x(s["b0"]) + 3:.1f}" y="21" class="seclab">{_esc(lab)}</text>'
                   f'<title>{_esc(lab)} · mes. {s["b0"] + 1}–{s["b1"] + 1} · clic = écouter</title></g>')
    b0, b1 = let["b0"], let["b1"]
    out.append(f'<rect class="brick" x="{x(b0):.1f}" y="4" width="{bw * (b1 - b0 + 1) - 1:.1f}" height="26" rx="3"/>')
    out.append(f'<text x="4" y="21" class="lab">toi</text>')
    # axes de la courbe
    for v in (0.2, 0.5, 0.85, 0.9, 0.95, 1.0):
        out.append(f'<line class="grid" x1="{padL}" x2="{W - padR}" y1="{yv(v):.1f}" y2="{yv(v):.1f}"/>'
                   f'<text x="{padL - 4}" y="{yv(v) + 3:.1f}" class="tick" text-anchor="end">{v:.2f}</text>')
    # vraies occurrences : bandes vertes derrière la courbe
    for g0, g1, _ in let["vraies"]:
        out.append(f'<rect class="truthband" x="{x(g0):.1f}" y="{y0}" width="{bw:.1f}" height="{hC}"/>')
    # courbe transposée (meilleure des 11) puis droite
    c = let["courbe"]
    cr = let["courbe_rot"]
    if let["L"] >= 4:
        pts = " ".join(f"{x(i) + bw / 2:.1f},{yv(v):.1f}" for i, v in enumerate(cr))
        out.append(f'<polyline class="crot" points="{pts}"/>')
    pts = " ".join(f"{x(i) + bw / 2:.1f},{yv(v):.1f}" for i, v in enumerate(c))
    out.append(f'<polyline class="curve" points="{pts}"/>')
    for i, v in enumerate(c):
        out.append(f'<circle class="dot" cx="{x(i) + bw / 2:.1f}" cy="{yv(v):.1f}" r="2.2">'
                   f'<title>départ mes. {i + 1} : {v:.3f}' + (f' · transposée {cr[i]:.3f}' if let["L"] >= 4 else '') + '</title></circle>')
    # seuils
    thr_p = let["info"]["seuil"] or 0.85
    thr_c = let["thr_plafond"]
    out.append(f'<line class="thr" x1="{padL}" x2="{W - padR}" y1="{yv(thr_p):.1f}" y2="{yv(thr_p):.1f}"/>'
               f'<text x="{W - padR - 2}" y="{yv(thr_p) - 3:.1f}" class="thrlab" text-anchor="end">seuil prod {thr_p:.3f}</text>')
    if abs(thr_c - thr_p) > 1e-3:
        out.append(f'<line class="thr2" x1="{padL}" x2="{W - padR}" y1="{yv(thr_c):.1f}" y2="{yv(thr_c):.1f}"/>'
                   f'<text x="{W - padR - 2}" y="{yv(thr_c) + 11:.1f}" class="thrlab2" text-anchor="end">seuil candidat 1 {thr_c:.3f}</text>')
    out.append(f'<text x="4" y="{y0 + 12}" class="lab">ressemb.</text>')

    def ligne(nom, occs):
        yy = rows_y[nom]
        out.append(f'<text x="4" y="{yy + 11}" class="lab">{nom}</text>')
        vraies = [(g0, g1) for g0, g1, _ in let["vraies"]]
        for o in occs:
            m0, m1 = o["m0"], o["m1"]
            if nom == "vérité":
                cls = "r-brick" if (m0, m1) == (b0, b1) else "r-truth"
            elif o.get("src") == "brique" or (m0, m1) == (b0, b1):
                cls = "r-brick"
            elif (m0, m1) in vraies:
                cls = "r-ok"
            elif any(OB._chevauche(g0, g1, m0, m1) * 2 >= (m1 - m0 + 1) for g0, g1 in vraies):
                cls = "r-part"
            else:
                cls = "r-ko"
            t0, t1 = grid[m0], grid[min(len(grid) - 1, m1 + 1)]
            rot = f" ↑{o['rot']}" if o.get("rot") else ""
            out.append(f'<g class="play" data-stem="{_esc(stem)}" data-t0="{t0:.2f}" data-t1="{t1:.2f}">'
                       f'<rect class="{cls}" x="{x(m0):.1f}" y="{yy}" width="{bw * (m1 - m0 + 1) - 1:.1f}" height="15" rx="2"/>'
                       + (f'<text x="{x(m0) + 2:.1f}" y="{yy + 11}" class="rot">{rot}</text>' if rot else "")
                       + f'<title>{nom} · mes. {m0 + 1}–{m1 + 1}{rot} · clic = écouter</title></g>')
            for a in o.get("alt") or []:
                out.append(f'<rect class="r-alt" x="{x(a):.1f}" y="{yy}" width="{bw - 1:.1f}" height="15"/>')
    ligne("vérité", [{"m0": g0, "m1": g1} for g0, g1, _ in let["vraies"]])
    ligne("prod", [{"m0": o["m0"], "m1": o["m1"], "rot": o["rot"]} for o in let["prod"]])
    ligne("candidat 1", let["plafond"])
    ligne("candidat 2", let["cand"])
    # graduation des mesures
    for b in range(0, n, 8):
        out.append(f'<text x="{x(b):.1f}" y="{H - 4}" class="tick">{b + 1}</text>')
    out.append("</svg>")
    return "".join(out)


def _strip(pm: list[float]) -> str:
    return "".join(f'<i style="background:{_col_score(v)}" title="{v:.2f}"></i>' for v in pm)


def _table(res: dict, let: dict) -> str:
    n = res["n"]
    prod = [{**o, "src": "brique" if (o["m0"], o["m1"]) == (let["b0"], let["b1"]) else "fixe"}
            for o in let["prod"]]
    rp, fp = OB.classe(let, n, prod, res["secs"])
    r1, f1 = OB.classe(let, n, let["plafond"], res["secs"])
    r2, f2 = OB.classe(let, n, let["cand"], res["secs"])
    h = ['<table><tr><th>ta vraie occurrence</th><th>mesure par mesure</th>'
         '<th class="num">score au départ</th><th>prod</th><th>pourquoi</th>'
         '<th>cand. 1</th><th>cand. 2</th></tr>']
    grid = res["grid"]
    for a, b, c in zip(rp, r1, r2):
        t0, t1 = grid[a["g0"]], grid[min(len(grid) - 1, a["g1"] + 1)]
        sc = f'{a["score"]:.3f}' if a["score"] is not None else "—"
        h.append(f'<tr><td><span class="play lnk" data-stem="{_esc(res["stem"])}" data-t0="{t0:.2f}" data-t1="{t1:.2f}">'
                 f'▶ {_esc(a["label"])} {a["g0"] + 1}–{a["g1"] + 1}</span> <small>({a["Lg"]} mes.)</small></td>'
                 f'<td class="strip">{_strip(a["par_mesure"])}</td><td class="num">{sc}</td>'
                 f'<td class="{_etat_css(a["etat"])}">{_esc(a["etat"])}</td><td><small>{_esc(a["cause"])}</small></td>'
                 f'<td class="{_etat_css(b["etat"])}">{_esc(b["etat"])}</td>'
                 f'<td class="{_etat_css(c["etat"])}">{_esc(c["etat"])}</td></tr>')
    h.append("</table>")
    par_plage: dict = {}
    for nom, fs in (("prod", fp), ("cand. 1", f1), ("cand. 2", f2)):
        for f in fs:
            k = (f["m0"], f["m1"])
            par_plage.setdefault(k, [f, []])[1].append(nom)
    trop = []
    for (m0, m1), (f, noms) in sorted(par_plage.items()):
        t0, t1 = grid[m0], grid[min(len(grid) - 1, m1 + 1)]
        trop.append(f'<li><span class="play lnk" data-stem="{_esc(res["stem"])}" '
                    f'data-t0="{t0:.2f}" data-t1="{t1:.2f}">▶ mes. {m0 + 1}–{m1 + 1}</span> '
                    f'<small>({", ".join(noms)})</small> — {_esc(f["pourquoi"])}</li>')
    if trop:
        h.append('<div class="trop"><b>Trouvé en trop</b> (là où tu n\'as pas mis cette lettre)<ul>'
                 + "".join(trop) + "</ul></div>")
    return "".join(h)


def _compact(res: dict) -> str:
    """Une ligne par lettre : chaque vraie occurrence, prod et candidat 1."""
    out = []
    for let in res["lettres"]:
        prod = [{**o, "src": "brique" if (o["m0"], o["m1"]) == (let["b0"], let["b1"]) else "fixe"}
                for o in let["prod"]]
        for nom, cands in (("prod", prod), ("cand. 1", let["plafond"])):
            rows, fs = OB.classe(let, res["n"], cands, res["secs"])
            chips = "".join(
                f'<span class="chip {_etat_css(r["etat"])}" title="{_esc(r["etat"])} — {_esc(r["cause"])}">'
                f'{r["g0"] + 1}–{r["g1"] + 1}</span>' for r in rows)
            chips += "".join(f'<span class="chip ko2" title="{_esc(f["pourquoi"])}">+{f["m0"] + 1}–{f["m1"] + 1}</span>'
                             for f in fs)
            lab = (f'<b>{_esc(let["label"])}</b> <small>brique {let["b0"] + 1}–{let["b1"] + 1}</small>'
                   if nom == "prod" else "")
            out.append(f'<tr><td>{lab}</td><td><small>{nom}</small></td><td>{chips}</td></tr>')
    return "".join(out)


CSS = """
:root { color-scheme: light dark;
  --bg:#1e1c1a; --fg:#ece8e2; --mut:rgba(236,232,226,.65); --line:rgba(128,128,128,.25);
  --ok:#3f9e5a; --mid:#c9962b; --ko:#c0453a; --truth:rgba(63,158,90,.16); --acc:#b03a2e;
  --sec:rgba(128,128,128,.22); --me:rgba(63,158,90,.45); --curve:#e8e2d8; --rot:rgba(160,160,160,.7);
  --thr:#e0584b; --thr2:#4d8fe0; }
@media (prefers-color-scheme: light) { :root:not([data-theme="dark"]) {
  --bg:#f7f6f3; --fg:#1a1a1a; --mut:rgba(26,26,26,.62); --curve:#222; --rot:rgba(90,90,90,.6);
  --truth:rgba(63,158,90,.14); } }
* { box-sizing: border-box; }
body { margin:0 auto; padding:14px 16px 60px; font-family:-apple-system,"Segoe UI",sans-serif;
  background:var(--bg); color:var(--fg); max-width:1100px; line-height:1.45; }
h1 { font-size:1.3rem; margin:4px 0 8px; } h2 { font-size:1.1rem; margin:34px 0 4px; }
h3 { font-size:.98rem; margin:18px 0 4px; }
.sub { color:var(--mut); font-size:.88rem; }
.say { font-size:.92rem; margin:10px 0; padding:10px 12px; border-left:3px solid var(--acc);
  background:rgba(176,58,46,.08); border-radius:0 8px 8px 0; }
.plot { width:100%; min-width:760px; height:auto; display:block; margin:6px 0; }
.sec { fill:var(--sec); } .sec.me { fill:var(--me); }
.brick { fill:none; stroke:var(--fg); stroke-width:2; }
.seclab { font-size:10px; fill:var(--fg); } .lab { font-size:11px; fill:var(--mut); }
.tick { font-size:9px; fill:var(--mut); } .grid { stroke:var(--line); }
.truthband { fill:var(--truth); }
.curve { fill:none; stroke:var(--curve); stroke-width:1.4; } .dot { fill:var(--curve); }
.crot { fill:none; stroke:var(--rot); stroke-width:1; stroke-dasharray:2 3; }
.thr { stroke:var(--thr); stroke-dasharray:6 4; stroke-width:1.3; } .thrlab { font-size:10px; fill:var(--thr); }
.thr2 { stroke:var(--thr2); stroke-dasharray:6 4; stroke-width:1.3; } .thrlab2 { font-size:10px; fill:var(--thr2); }
.r-truth { fill:var(--ok); opacity:.9; } .r-brick { fill:none; stroke:var(--fg); stroke-width:1.5; }
.r-ok { fill:var(--ok); } .r-part { fill:var(--mid); } .r-ko { fill:var(--ko); }
.r-alt { fill:none; stroke:var(--fg); stroke-dasharray:2 2; }
.rot { font-size:9px; fill:#fff; }
.play { cursor:pointer; } .lnk { text-decoration:underline dotted; white-space:nowrap; }
.tablewrap { overflow-x:auto; }
table { width:100%; border-collapse:collapse; font-size:.8rem; margin:8px 0; }
th { text-align:left; color:var(--mut); font-weight:600; padding:4px 6px; }
td { padding:5px 6px; border-top:1px solid var(--line); vertical-align:top; }
td.num { text-align:right; font-variant-numeric:tabular-nums; }
td.ok { color:var(--ok); font-weight:600; } td.part { color:var(--mid); font-weight:600; }
td.ko { color:var(--ko); font-weight:600; }
.strip i { display:inline-block; width:9px; height:14px; margin-right:1px; border-radius:2px; }
.trop { font-size:.82rem; margin:6px 0 0; } .trop ul { margin:4px 0; padding-left:18px; }
.chip { display:inline-block; font-size:.72rem; padding:1px 5px; margin:1px 2px; border-radius:4px;
  font-variant-numeric:tabular-nums; }
.chip.ok { background:rgba(63,158,90,.3); } .chip.part { background:rgba(201,150,43,.35); }
.chip.ko { background:none; outline:1.5px solid var(--ko); } .chip.ko2 { background:rgba(192,69,58,.35); }
.legend { font-size:.82rem; color:var(--mut); margin:8px 0; }
.legend span { display:inline-block; width:12px; height:10px; border-radius:2px; vertical-align:middle; margin:0 3px 0 10px; }
#player { position:sticky; top:0; background:var(--bg); padding:6px 0; font-size:.8rem; color:var(--mut); z-index:2; }
"""

JS = """
const audio = new Audio(); let stopAt = null; const blobs = {};
const msg = t => { document.getElementById('player').textContent = t; };
audio.addEventListener('timeupdate', () => { if (stopAt !== null && audio.currentTime >= stopAt) { audio.pause(); stopAt = null; } });
async function jouer(stem, t0, t1) {
  try {
    if (!blobs[stem]) { msg('chargement de l\\'audio…');
      const r = await fetch('/audio/' + stem + '.m4a'); blobs[stem] = URL.createObjectURL(await r.blob()); }
    if (audio.dataset.stem !== stem) { audio.src = blobs[stem]; audio.dataset.stem = stem; }
    audio.currentTime = t0; stopAt = t1; await audio.play();
    msg('▶ ' + t0.toFixed(1) + ' s → ' + t1.toFixed(1) + ' s (reclique ailleurs pour changer)');
  } catch (e) { msg('audio indisponible : ' + e); }
}
document.addEventListener('click', ev => {
  const g = ev.target.closest('.play'); if (!g) return;
  jouer(g.dataset.stem, parseFloat(g.dataset.t0), parseFloat(g.dataset.t1));
});
"""


def ecrire_page(tous: list[dict], chemin: Path) -> None:
    par_stem = {r["stem"]: r for r in tous}
    corps = []
    for stem, titre, dires in DETAIL:
        r = par_stem.get(stem)
        if not r:
            continue
        corps.append(f'<h2>{_esc(titre)}</h2>')
        for let in r["lettres"]:
            if let["label"] not in dires:
                continue
            corps.append(f'<h3>Tu donnes « {_esc(let["label"])} » = mesures {let["b0"] + 1}–{let["b1"] + 1} '
                         f'({let["L"]} mes.)</h3>')
            corps.append(f'<div class="say">{dires[let["label"]]}</div>')
            corps.append('<div class="tablewrap">' + _svg(r, let) + '</div>')
            corps.append('<div class="tablewrap">' + _table(r, let) + '</div>')
    detail_stems = {d[0] for d in DETAIL}
    try:
        meta = json.loads(SETTINGS.chart_meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        meta = {}

    def titre(stem):
        m = meta.get(f"min_{stem}") or {}
        return " — ".join(x for x in (m.get("title"), m.get("artist")) if x) or stem
    reste = "".join(
        f'<tr><td colspan="3" style="padding-top:12px"><b>{_esc(titre(r["stem"]))}</b></td></tr>' + _compact(r)
        for r in tous if r["stem"] not in detail_stems and r["lettres"])
    page = f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Occurrences d'une brique</title>
<style>{CSS}</style></head><body>
<h1>Tu donnes une section — retrouve-t-on ses reprises ?</h1>
<div class="sub">
On rejoue ton geste sur tes découpages validés : pour chaque lettre, sa <b>première</b>
occurrence devient la brique, et on regarde où la machine retrouve les autres.
<b>Prod</b> = <code>cherche_brique</code>, ce que fait « Valider les sections » aujourd'hui :
ta brique glisse le long du morceau, on moyenne la ressemblance d'accords (chord-tone) mesure à mesure,
et on garde les pics au-dessus d'un seuil propre au morceau (médiane + 1,5 écart-type, entre 0,85 et 0,999).
Rien de ce qui est ici n'est branché dans l'app.
</div>
<div class="say"><b>Ce que la page montre, en trois phrases.</b>
(1) Quand ça rate franchement, c'est d'abord le <b>seuil qui bute au plafond de 0,999</b> :
Let It Be A, The Walk A et B, Chain of Fools — des reprises à 0,99 recalées.
(2) Ensuite la <b>longueur</b> : la brique est cherchée à sa longueur exacte, donc une reprise
plus courte (San Francisco Street, 16 mesures contre 24) ou une fin alternative est ratée même quand chaque
mesure colle. (3) Les fausses trouvailles sont presque toutes des <b>jumeaux harmoniques</b> —
une autre section, ou l'intro, qui joue les mêmes accords ; aucun seuil ne les séparera.
Les décalages d'une mesure et les reprises transposées, eux, ne posent presque pas de problème.
<br><b>Une question pour toi</b> (en bas de page) : dans Every Breath You Take, tes trois derniers « A »
(mes. 89–111) n'ont que 4 mesures sur 8 en commun avec le premier — c'est la boucle de fin. A, ou A′ ?</div>
<div class="legend">
Courbe : ressemblance de ta brique <i>posée à partir de chaque mesure</i> (trait plein = telle quelle,
pointillé gris = meilleure transposition). Bandes vertes = départs de tes vraies occurrences.
Lignes de résultats :<span style="background:var(--ok)"></span>juste
<span style="background:var(--mid)"></span>décalé / mauvaise longueur
<span style="background:var(--ko)"></span>en trop ; cadre = ta brique ; pointillé = mesures de fin alternative.
Tableau, « mesure par mesure » : chaque carré = la ressemblance d'une mesure de l'occurrence à la mesure
de même rang dans ta brique (<span style="background:var(--ok)"></span>≥ 0,85
<span style="background:var(--mid)"></span>0,70–0,85 <span style="background:var(--ko)"></span>&lt; 0,70).
<b>Clic sur n'importe quel bloc ou ▶ = écouter ce passage.</b>
</div>
<div class="say"><b>Les candidats.</b> <b>Candidat 1</b> : même loi, mais le plafond du seuil passe de 0,999 à
0,97 (pour une brique d'au moins 8 mesures ; en dessous, une brique courte ressemble à trop de choses).
Même résultat de 0,96 à 0,985 : c'est un plateau, pas un réglage.
<b>Candidat 2</b> : candidat 1 + une seconde passe qui accepte des reprises plus courtes que la brique
(au moins 4 mesures et la moitié de la brique), et garde la longueur de la brique quand seule la fin change
(fin alternative). Il rattrape peu et ajoute des fragments en trop : l'idée est juste, pas cette version.</div>
<div id="player">clic sur un bloc pour écouter</div>
{"".join(corps)}
<h2>Le reste de tes morceaux validés</h2>
<div class="sub">Une ligne par lettre : chaque vraie occurrence (vert = trouvée, orange = décalée ou
mauvaise longueur, cadre rouge = ratée ; survol = pourquoi), puis en rouge plein ce qui est trouvé en trop.
Première ligne = prod, seconde = candidat 1.</div>
<div class="tablewrap"><table>{reste}</table></div>
<p class="sub">Généré par <code>python -m tools.occurrences_brique --page</code>. Grilles de mesures :
sauvegardes de chart dont le nombre de mesures égale celui de ton découpage (les charts <code>min_*</code>
ne sont plus dans <code>state/cache/charts/</code> depuis le 2026-09-30).</p>
<script>{JS}</script>
</body></html>"""
    chemin.write_text(page, encoding="utf-8")
