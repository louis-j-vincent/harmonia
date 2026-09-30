"""Occurrences d'une brique, v3 : la règle 4 de Louis, terme par terme.

Louis, 2026-09-30 soir, sur `docs/plots/occurrences_brique_v2_2026-09-30.html` :
« beaucoup beaucoup beaucoup mieux », sauf Let It Be — le refrain B, vraiment
différent, sort comme des A. Règle 4 :

    « Prends le score du pic le + grand trouvé en dehors de la section déjà
    identifiée — ça nous donne une idée de la variabilité inter-chanson.
    Ensuite utilise cet indice de variabilité + la taille de tous les autres
    pics pour décider d'un score qui permet de différencier le A du B dans
    Let It Be, tout en gardant Stand By Me. »

Implémentée dans `harmonia.sections.simulation.reference_du_morceau` (commit
eb72b62). Cette page compare v2 (règles 1–3 : le même code avec la règle 4
neutralisée, `TROU_K = ∞`) et v3 (le code live), et dessine pour chaque lettre
l'échelle des pics : R, chaque hauteur triée, le plus grand trou, la
tolérance, ce qui est coupé.

LECTURE SEULE. Ne modifie rien sous `state/`.

    .venv/bin/python -m tools.occurrences_brique_v3            # texte, corpus
    .venv/bin/python -m tools.occurrences_brique_v3 --page     # + la page
"""
from __future__ import annotations

import argparse
import glob
from pathlib import Path

from harmonia.sections import simulation as SIM
from harmonia.settings import SETTINGS
from tools import occurrences_brique_v2 as V2
from tools.occurrences_brique_page import CSS, JS

REPO = Path(__file__).resolve().parents[1]
PAGE = REPO / "docs" / "plots" / "occurrences_brique_v3_2026-09-30.html"
_esc = V2._esc

DETAIL = [
    ("let_it_be_remastered_2009", "Let It Be — The Beatles"),
    ("ben_e_king_stand_by_me_audio", "Stand By Me — Ben E. King"),
    ("9zEl-FQLI4A", "San Francisco Street — Sun Rai"),
    ("mayer_hawthorne_the_walk", "The Walk — Mayer Hawthorne"),
    ("maroon_5_she_will_be_loved_official_music_video", "She Will Be Loved — Maroon 5"),
    ("the_police_every_breath_you_take_official_music_video",
     "Every Breath You Take — The Police"),
    ("aretha_franklin_chain_of_fools_official_lyric_video", "Chain of Fools — Aretha Franklin"),
    ("X-yIEMduRXk", "X-yIEMduRXk"),
]

#: « ce que ça montre », écrit en lisant `python -m tools.occurrences_brique_v3`
DIRES = {
    ("let_it_be_remastered_2009", "A"):
        "<b>Le cas pour lequel la règle a été faite.</b> Tes 6 passages de A sont tous entre 0,994 et "
        "1,000 : ce morceau se rejoue presque à l'identique. Puis un trou de 0,066, et 15 départs entre "
        "0,90 et 0,93 — les fins de refrain (B) qui rejouent la boucle, et des décalages. Le trou vaut "
        "3 fois la tolérance (0,02) et il y a plus de pics en bas (15) qu'en haut (6) : on coupe. "
        "Refrain : 24 mesures en trop → 0, tes A restent tous trouvés.",
    ("ben_e_king_stand_by_me_audio", "A"):
        "<b>Le cas à garder.</b> Ici la meilleure reprise hors zone n'est qu'à 0,943 : le morceau varie "
        "à chaque tour (1 − R = 0,057). La tolérance est donc large (2 × 0,057 = 0,114), et les 9 pics "
        "descendent en pente douce (plus grand trou 0,020). Rien n'est coupé : les 10 passages restent.",
    ("ben_e_king_stand_by_me_audio", "C"):
        "Vu depuis C, la meilleure reprise est à 0,998 et le plus grand trou (0,073) dépasse la tolérance "
        "(0,022). Mais sous le trou il n'y a qu'<b>un</b> pic contre 7 au-dessus : c'est une reprise à fin "
        "différente, pas le niveau du reste du morceau. Pas de coupe, rien ne bouge (15–22 reste refusé "
        "par la règle 3, comme en v2).",
    ("9zEl-FQLI4A", "A"):
        "Trou de 0,065 plus grand que la tolérance (0,023), mais 1 pic dessous contre 3 au-dessus : pas de "
        "coupe. Identique à v2 : ta reprise 42–57 entière, il manque la mesure 71.",
    ("mayer_hawthorne_the_walk", "A"):
        "36 départs passent la règle 3 (deux accords qui alternent : tout colle un peu). Le plus grand "
        "trou (0,055) est à peine au-dessus de la tolérance (0,049), et il y a 29 pics au-dessus contre 7 : "
        "pas de coupe. Identique à v2.",
    ("mayer_hawthorne_the_walk", "B"):
        "<b>Change, à toi de trancher.</b> Deux départs parfaits (53 et 81, 0,9996–0,9999), puis 4 à 0,92 "
        "(51, 55, 79, 83) : les mêmes boucles <b>décalées de 2 mesures</b>. Trou 0,075 &gt; 0,02, 4 &gt; 2 : "
        "coupé. Résultat : le décalage disparaît (53–56 et 81–84 au bon endroit, 0 en trop au lieu de 4), "
        "mais on perd 4 mesures justes (57–58, 85–86 que le décalé couvrait). Le 2e et le 3e passage de "
        "chaque B restent ratés, comme en v2.",
    ("maroon_5_she_will_be_loved_official_music_video", "A"):
        "Ce que la règle ne résout pas. 27 pics, trou 0,050 contre tolérance 0,043, mais 14 au-dessus "
        "et 13 dessous : pas de coupe. La même grille tourne presque tout le morceau ; les jumeaux sont "
        "aussi hauts que tes vrais A (0,98–1,0). 28 mesures en trop, comme en v2.",
    ("maroon_5_she_will_be_loved_official_music_video", "D"):
        "Même histoire : 17 au-dessus, 13 dessous, pas de coupe. 60 mesures en trop, comme en v2 : les "
        "jumeaux sont au niveau des vraies reprises, aucun trou ne les sépare.",
    ("the_police_every_breath_you_take_official_music_video", "A"):
        "Trou de 0,072 entre tes 3 A exacts (0,992–0,999) et tes 2 A à autre fin (0,919 et 0,902). "
        "C'est exactement la hauteur du refrain de Let It Be — seul le compte les sauve : 3 dessous "
        "contre 3 au-dessus, pas plus nombreux, pas de coupe. Identique à v2.",
    ("aretha_franklin_chain_of_fools_official_lyric_video", "A"):
        "Un vamp : aucun départ ne passe la règle 3, la règle 4 n'a rien à juger. Identique à v2.",
    ("X-yIEMduRXk", "A"):
        "<b>Régression, à toi de trancher.</b> En haut : ta reprise 23 (0,994) et 44 (0,991) — mais 44 est "
        "dans ton C, un jumeau aussi ressemblant que la vraie reprise. En bas : 27 et 31 (tes A, 0,87–0,90) "
        "et 48 (jumeau). Trou 0,092, 3 dessous contre 2 au-dessus : coupé. 27–30 et 48–51 reviennent par la "
        "recherche de la fin de brique ; on perd 31–34 (un vrai passage de A) et le jumeau 44–51 reste. Le jumeau gonfle le groupe du haut : la règle suppose que la "
        "meilleure trouvaille est une vraie reprise.",
}

INTRO = """<div class="sub">Ta règle 4, sur la page v2 : « prends le score du pic le plus grand trouvé en dehors de la
section déjà identifiée — ça donne la variabilité du morceau ; utilise-la avec la taille de tous les autres pics
pour séparer le A du B dans Let It Be, tout en gardant Stand By Me ». Branchée dans « Valider les sections »
(active au prochain redémarrage du serveur). v2 = tes règles 1–3 ; v3 = v2 + la règle 4.</div>
<div class="say"><b>La règle 4, telle qu'implémentée.</b> On prend les départs qui passent la règle 3 et on les trie
par score. <b>R</b> = le plus haut (ta meilleure reprise hors zone). On cherche le <b>plus grand trou</b> entre deux
scores consécutifs. <b>Tolérance</b> = 2 × la plus grande de trois choses : 1 − R (même la meilleure reprise du
morceau est loin d'une copie : Stand By Me 0,057, Let It Be 0,000), l'écart à l'intérieur du groupe du haut, et
un plancher de 0,01 (bruit). On coupe — tout ce qui est sous le trou est refusé — si le trou dépasse la tolérance
<b>et</b> si le groupe du bas est plus nombreux que celui du haut : c'est alors le niveau de « tout le reste du
morceau », pas deux reprises à fin différente. Sans cette seconde condition, Every Breath You Take perdait ses
deux A à autre fin (exactement la hauteur du refrain de Let It Be).</div>"""


def analyse(stem: str):
    """(v2, v3) : le même morceau, règle 4 neutralisée puis live."""
    k = SIM.TROU_K
    try:
        SIM.TROU_K = float("inf")
        r2 = V2.analyse_morceau(stem)
    finally:
        SIM.TROU_K = k
    r3 = V2.analyse_morceau(stem)
    if r3 is None:
        return None
    for l2, l3 in zip(r2["lettres"], r3["lettres"]):
        l3["avant"], l3["n_avant"] = l2["apres"], l2["n_apres"]
    return r3


def _ou(let, p) -> str:
    Lb = let["det"]["boucle"]
    m = set(range(p, p + Lb))
    if len(m & let["verite"]) * 2 >= Lb:
        return "vrai"
    if len(m & (let["tol_verdict"] | let["tol"])) * 2 >= Lb:
        return "toléré"
    return "faux"


def _echelle(let: dict) -> str:
    """Les pics qui ont passé la règle 3, triés : R, le trou, la tolérance."""
    det = let["det"]
    ref = det.get("reference")
    pics = sorted((q for q in det.get("pics") or [] if q["ok"] or q.get("sous_reference")),
                  key=lambda q: -q["c"])
    if ref is None or len(pics) < 2:
        return ('<div class="say r4">Règle 4 : moins de deux départs passent la règle 3 — rien à '
                'comparer, elle ne s\'applique pas.</div>')
    W, padL, padR, H, y0, y1 = 1000, 70, 150, 220, 18, 190
    lo = min(min(q["c"] for q in pics), ref["bas_du_haut"] - ref["tolerance"]) - 0.01
    lo = max(0.0, lo)
    yv = lambda v: y0 + (y1 - y0) * (1.0 - v) / max(1.0 - lo, 1e-6)
    n = len(pics)
    step = (W - padL - padR) / max(n, 1)
    xi = lambda i: padL + step * (i + 0.5)
    o = [f'<svg viewBox="0 0 {W} {H}" class="plot" role="img">']
    for v in sorted({round(lo + (1 - lo) * f, 3) for f in (0, .25, .5, .75, 1)}):
        o.append(f'<line class="grid" x1="{padL}" x2="{W - padR}" y1="{yv(v):.1f}" y2="{yv(v):.1f}"/>'
                 f'<text x="{padL - 4}" y="{yv(v) + 3:.1f}" class="tick" text-anchor="end">{v:.3f}</text>')
    # R
    o.append(f'<line class="rline" x1="{padL}" x2="{W - padR}" y1="{yv(ref["R"]):.1f}" y2="{yv(ref["R"]):.1f}"/>'
             f'<text x="{W - padR + 4}" y="{yv(ref["R"]) + 3:.1f}" class="lab">R = {ref["R"]:.3f}</text>')
    ih = ref["n_haut"] - 1
    # tolérance : bande sous le bas du groupe du haut
    yt0, yt1 = yv(ref["bas_du_haut"]), yv(ref["bas_du_haut"] - ref["tolerance"])
    o.append(f'<rect class="tolband" x="{padL}" y="{yt0:.1f}" width="{W - padL - padR}" height="{yt1 - yt0:.1f}">'
             f'<title>tolérance {ref["tolerance"]:.3f} sous le groupe du haut</title></rect>'
             f'<text x="{W - padR + 4}" y="{(yt0 + yt1) / 2 + 3:.1f}" class="lab tollab">tolérance {ref["tolerance"]:.3f}</text>')
    # le trou
    if ih + 1 < n:
        xg = (xi(ih) + xi(ih + 1)) / 2
        ya, yb = yv(pics[ih]["c"]), yv(pics[ih + 1]["c"])
        cls = "gap cut" if ref["coupe"] else "gap"
        o.append(f'<line class="{cls}" x1="{xg:.1f}" x2="{xg:.1f}" y1="{ya:.1f}" y2="{yb:.1f}"/>'
                 f'<line class="{cls}" x1="{xg - 5:.1f}" x2="{xg + 5:.1f}" y1="{ya:.1f}" y2="{ya:.1f}"/>'
                 f'<line class="{cls}" x1="{xg - 5:.1f}" x2="{xg + 5:.1f}" y1="{yb:.1f}" y2="{yb:.1f}"/>'
                 f'<text x="{W - padR + 4}" y="{yb + 3:.1f}" class="lab gaplab">trou {ref["trou"]:.3f}</text>')
        if ref["coupe"]:
            o.append(f'<line class="cutline" x1="{xg:.1f}" x2="{xg:.1f}" y1="{y0 - 6}" y2="{y1 + 4}"/>'
                     f'<text x="{xg + 4:.1f}" y="{y1 - 4}" class="lab cutlab">coupé ici</text>')
    stem = let.get("_stem", "")
    grid = let.get("_grid") or []
    Lb = det["boucle"]
    for i, q in enumerate(pics):
        ou = _ou(let, q["p"])
        cls = {"vrai": "d-vrai", "toléré": "d-tol", "faux": "d-faux"}[ou]
        cut = " d-cut" if q.get("sous_reference") else ""
        t0 = grid[q["p"]] if q["p"] < len(grid) else 0
        t1 = grid[min(len(grid) - 1, q["p"] + Lb)] if grid else 0
        o.append(f'<g class="play" data-stem="{_esc(stem)}" data-t0="{t0:.2f}" data-t1="{t1:.2f}">'
                 f'<circle class="{cls}{cut}" cx="{xi(i):.1f}" cy="{yv(q["c"]):.1f}" r="5"/>'
                 f'<title>départ mes. {q["p"] + 1} · score {q["c"]:.4f} · {ou} dans ta vérité · '
                 f'{"COUPÉ par la règle 4" if q.get("sous_reference") else "au-dessus du trou"} · clic = écouter</title></g>')
        if n <= 24:
            o.append(f'<text x="{xi(i):.1f}" y="{y1 + 16}" class="tick" text-anchor="middle">{q["p"] + 1}</text>')
    o.append(f'<text x="4" y="{y1 + 16}" class="lab">départ</text>')
    o.append("</svg>")
    verdict = ("<b>coupé</b>" if ref["coupe"] else "pas de coupe")
    pourquoi = (f'trou {ref["trou"]:.3f} {">" if ref["trou"] > ref["tolerance"] else "≤"} tolérance '
                f'{ref["tolerance"]:.3f} · {ref["n_bas"]} pic(s) dessous '
                f'{">" if ref["n_bas"] > ref["n_haut"] else "≤"} {ref["n_haut"]} au-dessus')
    tete = (f'<div class="sub r4h">Règle 4 — les {n} départs qui passent la règle 3, du plus haut au plus bas. '
            f'1 − R = {1 - ref["R"]:.3f}, écart dans le groupe du haut = {ref["R"] - ref["bas_du_haut"]:.3f} '
            f'→ tolérance = 2 × {ref["tolerance"] / SIM.TROU_K:.3f} = {ref["tolerance"]:.3f}. '
            f'{pourquoi} → {verdict}.</div>')
    rows = "".join(
        f'<tr><td class="num">{i + 1}</td><td class="num">{q["p"] + 1}</td><td class="num">{q["c"]:.4f}</td>'
        f'<td class="num">{(q["c"] - pics[i + 1]["c"]) if i + 1 < n else 0:.4f}</td>'
        f'<td>{_ou(let, q["p"])}</td>'
        f'<td class="{"ko" if q.get("sous_reference") else "ok"}">{"coupé" if q.get("sous_reference") else "gardé par la règle 4"}</td></tr>'
        for i, q in enumerate(pics))
    table = (f'<details><summary>les {n} hauteurs, une par une</summary><table><tr><th class="num">rang</th>'
             '<th class="num">départ</th><th class="num">score</th><th class="num">écart au suivant</th>'
             '<th>dans ta vérité ?</th><th>règle 4</th></tr>' + rows + '</table></details>')
    return tete + '<div class="tablewrap">' + "".join(o) + '</div>' + table


def _svg_v(res, let) -> str:
    s = V2._svg(res, let)
    return s.replace('class="lab">avant</text>', 'class="lab">v2</text>') \
            .replace('class="lab">après</text>', 'class="lab">v3</text>')


def _chiffres(let) -> str:
    return (V2._chiffres(let).replace("<td>avant</td>", "<td>v2 (règles 1–3)</td>")
            .replace("<td>après</td>", "<td>v3 (+ règle 4)</td>"))


CSS3 = """
.rline { stroke:var(--fg); stroke-dasharray:2 3; }
.tolband { fill:rgba(77,143,224,.18); } .tollab { fill:#4d8fe0; }
.gap { stroke:var(--mid); stroke-width:2; } .gap.cut { stroke:var(--ko); } .gaplab { fill:var(--mid); }
.cutline { stroke:var(--ko); stroke-width:1.2; stroke-dasharray:4 3; } .cutlab { fill:var(--ko); }
.d-vrai { fill:var(--ok); } .d-tol { fill:#4d8fe0; } .d-faux { fill:var(--ko); }
.d-cut { fill-opacity:.25; stroke-width:1.6; }
.d-vrai.d-cut { stroke:var(--ok); } .d-tol.d-cut { stroke:#4d8fe0; } .d-faux.d-cut { stroke:var(--ko); }
.r4h { margin-top:10px; }
"""


def ecrire_page(tous: list[dict]) -> None:
    par = {r["stem"]: r for r in tous}
    corps = []
    for stem, titre in DETAIL:
        r = par.get(stem)
        if not r:
            continue
        corps.append(f'<h2>{_esc(titre)}</h2>')
        for let in r["lettres"]:
            cle = (stem, let["label"])
            if cle not in DIRES:
                continue
            let["_stem"], let["_grid"] = stem, r["grid"]
            corps.append(f'<h3>« {_esc(let["label"])} » = mesures {let["b0"] + 1}–{let["b1"] + 1} '
                         f'({let["L"]} mes.), boucle cherchée {let["det"]["boucle"]} mes.</h3>')
            corps.append(f'<div class="say"><b>Ce que ça montre.</b> {DIRES[cle]}</div>')
            corps.append(_echelle(let))
            corps.append('<div class="tablewrap">' + _svg_v(r, let) + '</div>')
            corps.append('<div class="tablewrap">' + _chiffres(let) + V2._pics_table(let) + '</div>')
    lignes = []
    for r in tous:
        for let in r["lettres"]:
            ref = let["det"].get("reference")
            if ref is None:
                continue
            a, p = let["n_avant"], let["n_apres"]
            chg = a != p
            lignes.append(
                f'<tr{" class=chg" if chg else ""}><td>{_esc(r["stem"][:34])}</td><td>{_esc(let["label"])}</td>'
                f'<td class="num">{ref["R"]:.3f}</td><td class="num">{ref["trou"]:.3f}</td>'
                f'<td class="num">{ref["tolerance"]:.3f}</td><td class="num">{ref["n_haut"]} / {ref["n_bas"]}</td>'
                f'<td class="{"ko" if ref["coupe"] else ""}">{"coupé" if ref["coupe"] else "—"}</td>'
                f'<td class="num">{a["justes"]} → {p["justes"]}</td><td class="num">{a["ratees"]} → {p["ratees"]}</td>'
                f'<td class="num">{a["trop"]} → {p["trop"]}</td></tr>')
    page = f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Occurrences d'une brique v3</title>
<style>{CSS}{V2.CSS2}{CSS3} tr.chg td {{ font-weight:600; }}</style></head><body>
<h1>Ta règle 4 — la meilleure reprise calibre le morceau</h1>
{INTRO}
<div class="say"><b>En bref.</b> Sur tes 20 découpages validés, la règle 4 ne coupe que sur 3 lettres.
<b>Let It Be A</b> : refrain 24 mesures en trop → 0, tes A intacts. <b>Stand By Me</b> : rien ne bouge.
<b>The Walk B</b> : le décalage de 2 mesures disparaît, mais 4 mesures justes aussi (à toi de dire).
<b>X-yIEMduRXk A</b> : 4 mesures justes perdues (31–34) — un jumeau dans ton C est aussi haut que la vraie
reprise et gonfle le groupe du haut. Tout le reste : identique à v2.</div>
<div class="legend">
<b>Échelle (règle 4)</b> : un point par départ qui a passé la règle 3, trié du plus haut au plus bas.
<span style="background:var(--ok)"></span>dans ta vérité <span style="background:#4d8fe0"></span>accepté / non annoté
<span style="background:var(--ko)"></span>faux · point pâle = coupé par la règle 4 · pointillé = R ·
bande bleue = tolérance sous le groupe du haut · trait orange (rouge si coupé) = le plus grand trou.
<b>Courbe</b> : comme en v2 ; lignes « v2 » et « v3 » = les occurrences trouvées.
<b>Clic sur n'importe quel bloc ou point = écouter.</b></div>
<div id="player">clic sur un bloc pour écouter</div>
{"".join(corps)}
<h2>Tout le corpus : chaque lettre où la règle 4 a quelque chose à juger</h2>
<div class="tablewrap"><table><tr><th>morceau</th><th>lettre</th><th class="num">R</th><th class="num">trou</th>
<th class="num">tolérance</th><th class="num">haut / bas</th><th>règle 4</th><th class="num">justes v2 → v3</th>
<th class="num">ratées v2 → v3</th><th class="num">en trop v2 → v3</th></tr>{"".join(lignes)}</table></div>
<p class="sub">En gras : les lettres qui changent. Généré par <code>python -m tools.occurrences_brique_v3 --page</code> ;
v3 = le code live (<code>harmonia.sections.simulation.reference_du_morceau</code>), v2 = le même avec la règle 4
neutralisée.</p>
<script>{JS}</script>
</body></html>"""
    PAGE.write_text(page, encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", action="store_true")
    ap.add_argument("stems", nargs="*")
    a = ap.parse_args()
    stems = a.stems or [Path(p).stem for p in sorted(glob.glob(str(SETTINGS.sections_dir / "*.json")))]
    tous = []
    for s in stems:
        r = analyse(s)
        if r is None:
            continue
        tous.append(r)
        for let in r["lettres"]:
            x, y, ref = let["n_avant"], let["n_apres"], let["det"].get("reference")
            print(f"{s[:30]:30s} {let['label']:6s} v2 j/r/t={x['justes']}/{x['ratees']}/{x['trop']}  "
                  f"v3={y['justes']}/{y['ratees']}/{y['trop']}{'  <<' if x != y else ''}  ref={ref}")
    if a.page:
        ecrire_page(tous)
        print("page :", PAGE)


if __name__ == "__main__":
    main()
