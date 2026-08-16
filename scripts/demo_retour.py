#!/usr/bin/env python3
"""scripts/demo_retour.py — la démo pas-à-pas de `harmonia_min.retour`.

Écrit `docs/plots/retour_10morceaux.html` : pour chacun des dix morceaux, la
grille de mesures avec les accords, l'annotation à la main de Louis en regard,
puis CHAQUE étape de l'algorithme — les retours testés, ceux qui ont été jugés
forts, le mot proposé, le contrôle de répétition mesure par mesure, et les
occurrences retenues. Tout est cliquable : un clic sur une mesure place la tête
de lecture du disque.

Les dix morceaux sont pris dans `state/sections/`, donc parmi ceux dont Louis
a lui-même posé les frontières — l'algo se lit contre son oreille, pas contre
une vérité de corpus (mémoire : les GT brick0 sont condamnées).

    .venv/bin/python scripts/demo_retour.py
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harmonia_min import retour as R          # noqa: E402

CHARTS = REPO / "harmonia_min" / "state" / "charts"
ANNOT = REPO / "harmonia_min" / "state" / "sections"
SORTIE = REPO / "docs" / "plots" / "retour_10morceaux.html"

MORCEAUX = [
    ("min_maroon_5_this_love", "Maroon 5 — This Love"),
    ("min_let_it_be_remastered_2009", "The Beatles — Let It Be"),
    ("min_ben_e_king_stand_by_me_audio", "Ben E. King — Stand By Me"),
    ("min_bobby_hebb_sunny_official_audio", "Bobby Hebb — Sunny"),
    ("min_bruno_mars_grenade_official_music_video", "Bruno Mars — Grenade"),
    ("min_maroon_5_she_will_be_loved_official_music_video",
     "Maroon 5 — She Will Be Loved"),
    ("min_norah_jones_don_t_know_why", "Norah Jones — Don't Know Why"),
    ("min_the_police_every_breath_you_take_official_music_video",
     "The Police — Every Breath You Take"),
    ("min_yesterday_remastered_2009", "The Beatles — Yesterday"),
    ("min_the_ronettes_be_my_baby_music_video", "The Ronettes — Be My Baby"),
]

#: Une teinte par lettre de section. Reprises du bleu de `ssm_page.CMAP` pour
#: que les pages du projet se lisent avec le même œil, plus deux teintes
#: chaudes pour les lettres suivantes.
TEINTES = ["#3d7fa6", "#b4472c", "#5b8c5a", "#8a6bab", "#c08a2e", "#4a7b8c"]
GRIS = "#d8d2c6"

NOTES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def accords_par_mesure(chart: dict) -> list[str]:
    """Le nom de l'accord qui dure le plus longtemps dans chaque mesure."""
    grid = chart.get("barGrid") or []
    accords = (chart.get("prompter") or {}).get("chords") or []
    out = []
    for b in range(len(grid) - 1):
        t0, t1 = grid[b], grid[b + 1]
        duree: dict = {}
        for c in accords:
            chev = min(t1, c.get("t1", 0.0)) - max(t0, c.get("t0", 0.0))
            if chev <= 0:
                continue
            if c.get("nc"):
                nom = "N.C."
            else:
                root = int(c.get("root", -1))
                nom = (NOTES[root % 12] + str(c.get("q") or "")) if root >= 0 else "?"
                bass = c.get("bass", -1)
                if bass is not None and bass >= 0 and bass % 12 != root % 12:
                    nom += "/" + NOTES[bass % 12]
            duree[nom] = duree.get(nom, 0.0) + chev
        out.append(max(duree, key=duree.get) if duree else "")
    return out


def annotation(stem: str, n: int) -> list[str | None]:
    """L'étiquette de Louis pour chaque mesure, None s'il n'a rien posé."""
    f = ANNOT / f"{stem}.json"
    out: list[str | None] = [None] * n
    if not f.exists():
        return out
    for s in json.loads(f.read_text(encoding="utf-8")).get("sections") or []:
        for b in range(int(s["b0"]), min(int(s["b1"]) + 1, n)):
            out[b] = str(s.get("label") or "?")
    return out


def couleurs_louis(labels: list[str | None]) -> dict:
    """Une teinte par étiquette de Louis, dans l'ordre d'apparition."""
    vus, coul = [], {}
    for x in labels:
        if x and x not in vus:
            vus.append(x)
    for i, x in enumerate(vus):
        coul[x] = TEINTES[i % len(TEINTES)]
    return coul


# ── le rendu ────────────────────────────────────────────────────────────────

def e(s) -> str:
    return html.escape(str(s), quote=True)


#: Au-dessus, une occurrence est manifestement la même chose ; en dessous, elle
#: passe le seuil de justesse et c'est là que le découpage dérive. Le trait est
#: posé À VUE sur les trois morceaux (les bonnes occurrences sortent à 0,88–1,00
#: et les douteuses à 0,69–0,77) — il sert à MONTRER la coupure, pas à trancher.
NET = 0.85


def bandeau(labels, coul, grid, titre, chords=None, faibles=None) -> str:
    """Une bande de mesures cliquables, coloriée par étiquette."""
    faibles = faibles or {}
    cases = []
    for b, lab in enumerate(labels):
        c = coul.get(lab, GRIS) if lab else GRIS
        txt = e(lab or "·")
        acc = f'<i>{e(chords[b])}</i>' if chords else ""
        cl = "case faible" if b in faibles else "case"
        t = (f"mesure {b} — occurrence faible ({faibles[b]:.2f})"
             if b in faibles else f"mesure {b}")
        cases.append(
            f'<b class="{cl}" style="--c:{c}" data-t="{grid[b]:.3f}" '
            f'title="{t}">{acc}<u>{txt}</u><s>{b}</s></b>')
    return (f'<div class="bande"><h4>{e(titre)}</h4>'
            f'<div class="cases">{"".join(cases)}</div></div>')


def strip_sims(etape, seuil, n, grid) -> str:
    """La ressemblance de chaque mesure à la mesure de départ, en barres."""
    d = etape["depart"]
    cells = []
    for c in etape["candidats"]:
        k, v = c["barre"], c["sim"]
        if k >= n or not np.isfinite(v):
            continue
        h = max(2, int(round(38 * max(0.0, min(1.0, (v - 0.3) / 0.7)))))
        cl = "fort" if c["fort"] else ""
        if c is etape["retenu"]:
            cl += " retenu"
        cells.append(f'<b class="{cl}" style="height:{h}px" data-t="{grid[k]:.3f}" '
                     f'title="mesure {k} — ressemblance {v:.3f}"></b>')
    y = max(2, int(round(38 * max(0.0, min(1.0, (seuil - 0.3) / 0.7)))))
    return (f'<div class="sims"><span class="depart">mesure {d}</span>'
            f'<div class="barres">{"".join(cells)}'
            f'<i class="seuil" style="bottom:{y}px"></i></div></div>')


def bloc_recousu(et, grid, chords) -> str:
    """La chanson privée des sections déjà trouvées — avec ses coutures."""
    rest, cout = et["restants"], set(et["coutures"])
    if not cout:
        return ""
    cases = []
    for i, b in enumerate(rest):
        cases.append(f'<b class="case mini" data-t="{grid[b]:.3f}" '
                     f'title="mesure {b}"><i>{e(chords[b])}</i><s>{b}</s></b>')
        if i in cout:
            cases.append('<b class="couture" title="ici on a retiré une '
                         'section déjà trouvée">✂</b>')
    return (f'<div class="recousu"><h5>La chanson recousue '
            f'<span>{len(rest)} mesures restantes, {len(cout)} couture(s) — '
            f'c\'est LÀ-DEDANS qu\'on cherche, donc les deux mesures de part et '
            f'd\'autre d\'un ✂ sont voisines</span></h5>'
            f'<div class="cases">{"".join(cases)}</div></div>')


def table_candidats(etape, seuil, recousu: bool = False) -> str:
    """Les retours forts qui sont allés jusqu'au test, un par ligne.

    `recousu` ajoute le RANG de la mesure dans la chanson recousue : sans lui,
    « retour mesure 36, mot de 8 mesures » depuis la mesure 16 se lit comme une
    erreur d'arithmétique, alors que 8 est bien la distance une fois la section
    A retirée.
    """
    lignes = []
    for c in etape["candidats"]:
        if not c["fort"]:
            continue
        montre = (c["repet"] is not None and c["L"] >= R.LONGUEUR_MIN) or c["litteral"]
        if not montre:
            continue
        rep = f'{c["repet"]["moyenne"]:.3f}' if c["repet"] else "—"
        cls = []
        if c is etape["retenu"]:
            cls.append("ok")
        elif c["passe"]:
            cls.append("aurait")
        marques = []
        if c["litteral"]:
            marques.append('<em title="le premier retour fort rencontré">'
                           '1er retour</em>')
        if c is etape.get("premier_passant") and c is not etape["retenu"]:
            marques.append('<em class="vieux" title="ce que la règle « le '
                           'premier qui passe » aurait pris">l\'ancien choix'
                           '</em>')
        if c["passe"] and c.get("hors_carrure"):
            marques.append(f'<em class="cf" title="sa longueur n\'est pas un '
                           f'multiple de {R.CARRURE}">hors carrure</em>')
        barre = (f'{c["barre"]} <i class="jx">{c["j"]}<sup>e</sup></i>'
                 if recousu else str(c["barre"]))
        lignes.append(
            f'<tr class="{" ".join(cls)}"><td>{barre}</td><td>{c["L"]}</td>'
            f'<td>{c["sim"]:.3f}</td><td>{rep}</td>'
            f'<td>{e(c["verdict"])} {" ".join(marques)}</td></tr>')
    if not lignes:
        return '<p class="rien">aucun retour fort assez loin pour faire un mot.</p>'
    col1 = ('retour<br>mesure <i class="jx">rang</i>' if recousu
            else 'retour<br>mesure')
    return ('<div class="tbl"><table class="cand">'
            f'<tr><th>{col1}</th><th>mot<br>mesures</th>'
            '<th>ressemblance<br>à la mesure de départ</th>'
            '<th>le mot suivant<br>est-il le même ?</th><th></th></tr>'
            + "".join(lignes) + "</table></div>")


def detail_repet(cand, seuil, depart, grid) -> str:
    """Le contrôle de répétition, mesure par mesure — la queue en gris."""
    if not cand or not cand["repet"]:
        return ""
    L, sims = cand["L"], cand["repet"]["sims"]
    cases = []
    for t, v in enumerate(sims):
        a, b = depart + t, cand["barre"] + t
        cl = "oui" if v >= seuil else "non"
        cases.append(f'<b class="{cl}" data-t="{grid[a]:.3f}">'
                     f'<u>{a}↔{b}</u><s>{v:.2f}</s></b>')
    for t in range(len(sims), L):
        a, b = depart + t, cand["barre"] + t
        cases.append(f'<b class="libre" data-t="{grid[a]:.3f}">'
                     f'<u>{a}↔{b}</u><s>libre</s></b>')
    return (f'<div class="repet"><h5>Le mot suivant est-il le même mot ? '
            f'<span>les {R.QUEUE_LIBRE} dernières mesures ne comptent pas '
            f'— c\'est la cadence qui a le droit de changer</span></h5>'
            f'<div class="paires">{"".join(cases)}</div>'
            f'<p>moyenne <b>{cand["repet"]["moyenne"]:.3f}</b> '
            f'contre un seuil de {seuil:.3f}</p></div>')


def bloc_boucle(et, seuil, grid) -> str:
    """Le mot boucle-t-il sur lui-même ? — chaque période testée, et laquelle
    devient le modèle."""
    b = et.get("boucle")
    d, L = et["depart"], et["retenu"]["L"]
    if not b:
        return ""
    if not b["periodes"]:
        return (f'<div class="boucle"><h5>Le mot boucle-t-il sur lui-même ?</h5>'
                f'<p class="rien">un mot de {L} mesures ne peut pas boucler sur '
                f'{R.PERIODE_MIN} : il faut au moins {2 * R.PERIODE_MIN} mesures '
                f'pour qu\'une boucle de {R.PERIODE_MIN} se referme une fois. '
                f'Le modèle reste le mot entier.</p></div>')
    lignes = []
    for c in b["periodes"]:
        pris = c is b["retenue"]
        cases = "".join(
            f'<b class="{"oui" if v >= seuil else "non"}" '
            f'data-t="{grid[d + t]:.3f}">'
            f'<u>{d + t}↔{d + t + c["p"]}</u><s>{v:.2f}</s></b>'
            for t, v in enumerate(c["sims"]))
        lignes.append(
            f'<div class="per {"pris" if pris else ""}">'
            f'<span class="p">{c["p"]} mesures</span>'
            f'<div class="paires">{cases}</div>'
            f'<span class="m">{c["moyenne"]:.3f}'
            f'{" → modèle" if pris else (" ✓" if c["passe"] else " ✗")}</span></div>')
    if b["retenue"]:
        mot = (f'<p>Le mot de {L} mesures est la boucle de '
               f'<b>{b["retenue"]["p"]} mesures</b> jouée deux fois. C\'est la '
               f'boucle qui devient le modèle — c\'est elle qu\'on cherchera '
               f'dans la suite du morceau.</p>')
    else:
        mot = (f'<p>Aucune boucle interne d\'au moins {R.PERIODE_MIN} mesures. '
               f'Le modèle reste le mot de {L} mesures.</p>')
    return (f'<div class="boucle"><h5>Le mot boucle-t-il sur lui-même ? '
            f'<span>une boucle d\'au moins {R.PERIODE_MIN} mesures devient le '
            f'modèle à la place du mot</span></h5>'
            f'{"".join(lignes)}{mot}</div>')


def bloc_morceau(fichier: str, titre: str) -> str:
    chart = json.loads((CHARTS / f"{fichier}.json").read_text(encoding="utf-8"))
    stem = Path(chart.get("audio_url") or "").stem
    grid = chart["barGrid"]
    S = R.ssm_mesures(chart)
    if S is None:
        return f'<section><h2>{e(titre)}</h2><p>pas de SSM.</p></section>'
    res = R.sections(S)
    n = res["n_mesures"]
    seuil = res["seuil"]

    chords = accords_par_mesure(chart)
    algo = R.par_mesure(res)
    louis = annotation(stem, n)

    coul_algo = {s["label"]: TEINTES[i % len(TEINTES)]
                 for i, s in enumerate(res["sections"])}
    coul_louis = couleurs_louis(louis)
    faibles = {b: o["moyenne"] for s in res["sections"] for o in s["occurrences"]
               if o["moyenne"] < NET for b in range(o["b0"], o["b1"] + 1)}

    etapes = []
    for i, et in enumerate(res["etapes"], 1):
        d = et["depart"]
        rec = bool(et["coutures"])
        tete = bloc_recousu(et, grid, chords)
        if et["action"] == "avance":
            corps = (f'{tete}{strip_sims(et, seuil, n, grid)}'
                     f'{table_candidats(et, seuil, rec)}'
                     f'<p class="issue avance">Aucun retour ne donne de section '
                     f'depuis la mesure {d}. On avance d\'une mesure ; la mesure '
                     f'{d} restera sans section.</p>')
        else:
            sec, ret = et["section"], et["retenu"]
            occ = " ".join(
                f'<b class="occ {"net" if o["moyenne"] >= NET else "flou"}">'
                f'{o["b0"]}–{o["b1"]} <i>{o["moyenne"]:.2f}</i>'
                f'{" ✱" if o["variante"] else ""}</b>'
                for o in sec["occurrences"])
            quoi = (f'la boucle de {sec["L"]} mesures trouvée dans le mot de '
                    f'{sec["mot"]}' if sec["boucle"]
                    else f'le mot de {sec["L"]} mesures')
            # LE MÉMO DU REPLIEMENT (Louis, 2026-08-16 : « attention à noter
            # quelque part ces exemptions, car lors du repliement du chart il
            # faudra les noter sur le chart »). Une mesure exemptée qui ne
            # ressemble PAS au modèle est une mesure que cette occurrence-là
            # joue autrement : le repliement doit l'écrire, pas recopier le
            # modèle par-dessus.
            vs = [(o, x) for o in sec["occurrences"] for x in o["variante"]]
            memo = ""
            if vs:
                lignes_v = " ".join(
                    f'<b class="occ flou">mesure {x["mesure"]} '
                    f'<i>≠ m.{x["mesure_modele"]} du modèle ({x["sim"]:.2f})</i>'
                    f'</b>' for _o, x in vs)
                solos_v = ""
                memo = (f'<p class="issue memo"><b>À écrire sur le chart au '
                        f'repliement</b> — ces mesures-là ne ressemblent pas au '
                        f'modèle : l\'occurrence y joue autre chose, et le '
                        f'repliement doit l\'écrire au lieu de recopier le '
                        f'modèle. {lignes_v}</p>')
            solo = ""
            if sec["solos"]:
                liste_s = " ".join(f'<b class="occ flou">{o["b0"]}–{o["b1"]} '
                                   f'<i>{o["moyenne"]:.2f}</i></b>'
                                   for o in sec["solos"])
                solo = (f'<p class="issue solo">Jeté — boucle de {sec["L"]} '
                        f'mesures toute seule : {liste_s}. Une répétition de '
                        f'section doit faire au moins {R.REPETITION_MIN} mesures ; '
                        f'des occurrences collées s\'additionnent, un bloc isolé '
                        f'non.</p>')
            ecart = ""
            if sec["ecartees"]:
                liste = " ".join(f'<b class="occ flou">{o["b0"]}–{o["b1"]} '
                                 f'<i>{o["moyenne"]:.2f}</i></b>'
                                 for o in sec["ecartees"])
                ecart = (f'<p class="issue ecarte">Écartées : {liste} — '
                         f'au-dessus du seuil de retour ({seuil:.3f}) mais en '
                         f'dessous du seuil d\'occurrence '
                         f'(<b>{sec["seuil_occ"]:.3f}</b>). Ces mesures restent '
                         f'sans section.</p>')
            corps = (f'{tete}{strip_sims(et, seuil, n, grid)}'
                     f'{table_candidats(et, seuil, rec)}'
                     f'{detail_repet(ret, seuil, d, grid)}'
                     f'{bloc_boucle(et, seuil, grid)}'
                     f'<p class="issue ok">Section <b class="lettre" '
                     f'style="--c:{coul_algo[sec["label"]]}">{sec["label"]}</b> '
                     f'= {quoi}, qui commence mesure {d}. '
                     f'Ses {len(sec["occurrences"])} occurrences dans le morceau : '
                     f'{occ}</p>{memo}{solo}{ecart}')
        etapes.append(
            f'<div class="etape"><h3>Étape {i} — on repart de la mesure {d}'
            f'<span>{"section trouvée" if et["action"] == "section" else "rien ici"}'
            f'</span></h3>{corps}</div>')

    resume = " · ".join(
        f'<b class="lettre" style="--c:{coul_algo[s["label"]]}">{s["label"]}</b> '
        f'{s["L"]} mesures × {len(s["occurrences"])}'
        f'{f" (boucle du mot de {s['mot']})" if s["boucle"] else ""}'
        for s in res["sections"]) or "aucune section"

    return f"""<section data-audio="/audio/{e(stem)}.m4a">
<h2>{e(titre)}</h2>
<p class="meta">{n} mesures · seuil de « ressemblance forte » <b>{seuil:.3f}</b>
(Otsu sur les cases hors-diagonale) · sortie : {resume}
· <span class="reste">{len(res["reste"])} mesures sans section</span></p>
<audio controls preload="none"></audio>
{bandeau(algo, coul_algo, grid, "Ce que l'algo trouve", chords, faibles)}
{bandeau(louis, coul_louis, grid, "Ce que Louis a annoté à la main")}
<div class="etapes">{"".join(etapes)}</div>
</section>"""


CSS = """
:root{--fg:#1b1b19;--fond:#fbf7ec;--trait:#ddd6c7;--doux:#6b6558;}
*{box-sizing:border-box}
body{margin:0;padding:24px 18px 80px;background:var(--fond);color:var(--fg);
 font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif;
 max-width:1180px;margin-inline:auto}
h1{font-size:24px;margin:0 0 6px}
h2{font-size:20px;margin:0 0 4px}
h3{font-size:15px;margin:0 0 10px;display:flex;justify-content:space-between;
 align-items:baseline;border-bottom:1px solid var(--trait);padding-bottom:6px}
h3 span{font-size:12px;color:var(--doux);font-weight:400}
h4{font-size:12px;margin:0 0 4px;color:var(--doux);font-weight:600;
 text-transform:uppercase;letter-spacing:.04em}
h5{font-size:13px;margin:14px 0 6px;font-weight:600}
h5 span{font-weight:400;color:var(--doux);font-size:12px}
section{margin:0 0 46px;padding:0 0 8px;border-bottom:2px solid var(--trait)}
.chapo{color:var(--doux);max-width:76ch;margin:0 0 26px}
.chapo code{background:#efe9db;padding:1px 5px;border-radius:4px}
.meta{color:var(--doux);font-size:13px;margin:0 0 10px}
.reste{color:#a4462b}
audio{width:100%;max-width:420px;height:32px;margin:0 0 14px;display:block}
.bande{margin:0 0 12px}
.cases{display:flex;flex-wrap:wrap;gap:2px}
.case{--c:#d8d2c6;width:46px;padding:3px 2px;border-radius:4px;cursor:pointer;
 background:color-mix(in srgb,var(--c) 22%,white);
 border:1px solid color-mix(in srgb,var(--c) 55%,white);text-align:center;
 display:flex;flex-direction:column;gap:0;line-height:1.25}
.case:hover{background:color-mix(in srgb,var(--c) 42%,white)}
.case.faible{border-style:dashed;border-color:#c98a55;background:#fbf1e4}
.occ{display:inline-block;padding:1px 6px;border-radius:5px;margin:0 3px 3px 0}
.occ i{font-style:normal;font-weight:400;opacity:.65;font-size:11.5px}
.occ.net{background:#dcebd9}
.occ.flou{background:#fae7d2}
.case i{font-style:normal;font-size:10px;color:var(--doux);
 white-space:nowrap;overflow:hidden}
.case u{text-decoration:none;font-size:12px;font-weight:700;
 color:color-mix(in srgb,var(--c) 80%,black)}
.case s{text-decoration:none;font-size:9px;color:#a49c8c}
.etape{margin:0 0 22px;padding:12px 14px;background:#fff;border-radius:8px;
 border:1px solid var(--trait)}
.sims{display:flex;align-items:flex-end;gap:8px;margin:0 0 12px}
.sims .depart{font-size:11px;color:var(--doux);white-space:nowrap;
 padding-bottom:2px}
.barres{position:relative;display:flex;align-items:flex-end;gap:1px;
 height:40px;flex:1;border-bottom:1px solid var(--trait)}
.barres b{flex:1 1 0;min-width:1px;max-width:9px;background:#e3ddcf;
 cursor:pointer;border-radius:1px 1px 0 0}
.barres b.fort{background:#84b3cf}
.barres b.retenu{background:#b4472c}
.barres b:hover{outline:1px solid var(--fg)}
.seuil{position:absolute;left:0;right:0;height:0;border-top:1px dashed #b4472c}
.tbl{overflow-x:auto}
table.cand{border-collapse:collapse;font-size:12.5px;margin:0 0 4px;width:100%;
 min-width:440px}
table.cand th{text-align:left;font-weight:600;color:var(--doux);
 padding:2px 10px 6px 0;font-size:11px;vertical-align:bottom}
table.cand td{padding:3px 10px 3px 0;border-top:1px solid #f0ebde}
table.cand tr.ok td{background:#f4e7e2;font-weight:600}
table.cand tr.aurait td{background:#eef3f6}
table.cand em{font-style:normal;font-size:10px;padding:1px 5px;border-radius:9px;
 background:#e6dfd0;color:var(--doux)}
table.cand em.cf{background:#d5e3ec;color:#1b4a6b}
table.cand em.vieux{background:#f2ddd6;color:#8c3a22}
.issue.ecarte{background:#fbf1e4;color:#6b5a3a;margin-top:8px}
.issue.memo{background:#f2ede0;color:#5a4f38;margin-top:8px;
 border-left:3px solid #c08a2e}
.issue.solo{background:#f5eef0;color:#6b4a55;margin-top:8px;
 border-left:3px solid #a4607a}
.repet{margin:12px 0 4px}
.paires{display:flex;flex-wrap:wrap;gap:2px}
.paires b{width:52px;padding:2px;border-radius:4px;text-align:center;
 cursor:pointer;font-weight:400;display:flex;flex-direction:column}
.paires b u{text-decoration:none;font-size:10px;color:var(--doux)}
.paires b s{text-decoration:none;font-size:11px;font-weight:700}
.paires b.oui{background:#dcebd9;border:1px solid #9cc294}
.paires b.non{background:#f6e0da;border:1px solid #d09b8a}
.paires b.libre{background:#efeade;border:1px dashed #c8c0af}
.repet p{font-size:12.5px;color:var(--doux);margin:6px 0 0}
.recousu{margin:0 0 14px;padding:10px 12px;background:#f4f1e6;border-radius:6px;
 border:1px solid #e7e0cf}
.recousu h5{margin:0 0 8px}
.case.mini{width:36px}
.case.mini i{font-size:9.5px}
.couture{width:16px;display:flex;align-items:center;justify-content:center;
 color:#b4472c;font-size:13px;font-weight:700;cursor:default}
.jx{font-style:normal;font-size:10px;color:#7d94a4;background:#e6eef3;
 padding:0 4px;border-radius:6px;margin-left:3px}
.jx sup{font-size:8px}
.boucle{margin:14px 0 4px;padding:10px 12px;background:#f7f4ea;
 border-radius:6px;border:1px solid #e7e0cf}
.boucle p{font-size:12.5px;color:var(--doux);margin:8px 0 0}
.boucle p b{color:var(--fg)}
.per{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:0 0 5px;
 padding:4px 6px;border-radius:5px}
.per.pris{background:#e6efd9;box-shadow:inset 0 0 0 1px #a8c188}
.per .p{font-size:12px;font-weight:600;width:80px;flex:none}
.per .m{font-size:12px;color:var(--doux);white-space:nowrap}
.per.pris .m{color:#42631f;font-weight:600}
.issue{margin:12px 0 0;padding:8px 10px;border-radius:6px;font-size:13.5px}
.issue.ok{background:#eef3f6}
.issue.avance{background:#f7f1e4;color:#6b5a3a}
.lettre{--c:#3d7fa6;display:inline-block;min-width:20px;text-align:center;
 padding:0 5px;border-radius:4px;color:#fff;background:var(--c)}
.rien{font-size:12.5px;color:var(--doux);margin:0}
.pied{color:var(--doux);font-size:13px;max-width:78ch}
.pied li{margin:0 0 6px}
"""

JS = """
/* Un clic sur une mesure place la tête de lecture. Le disque est rejoué depuis
   un blob : dans Safari iOS le lecteur média ne bufferise jamais un fichier
   servi en direct (206 en boucle), et chaque tap replace la tête
   instantanément une fois le blob arrivé. Aucune boucle de rendu ici — un
   requestAnimationFrame empêcherait le moteur audio de WebKit de démarrer. */
document.querySelectorAll("section[data-audio]").forEach(function(sec){
  var el = sec.querySelector("audio"), url = sec.dataset.audio, lien = null;
  el.src = url;
  if (window.fetch) fetch(url).then(function(r){ return r.ok ? r.blob() : null; })
    .then(function(b){ if (!b) return;
      lien = URL.createObjectURL(b);
      var t = el.currentTime, jouait = !el.paused;
      el.src = lien; el.currentTime = t; if (jouait) el.play();
    }).catch(function(){});
  sec.addEventListener("click", function(ev){
    var c = ev.target.closest("[data-t]"); if (!c) return;
    try { el.currentTime = parseFloat(c.dataset.t); el.play(); } catch(e){}
  });
});
"""


def main() -> None:
    blocs = [bloc_morceau(f, t) for f, t in MORCEAUX]
    page = f"""<!doctype html><html lang="fr"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>L'algo du retour — trois morceaux, pas à pas</title>
<style>{CSS}</style>
<h1>L'algo du retour, pas à pas</h1>
<p class="chapo">On part de la première mesure et on cherche <b>où elle
revient</b>. La première mesure qui lui ressemble fortement ferme un mot ; si ce
mot fait plus de 6 mesures et que le mot d'après est le même (les
{R.QUEUE_LIBRE} dernières mesures exemptées, c'est la cadence), c'est une
section. Avant d'aller chercher ses répétitions, on regarde si <b>le mot boucle
sur lui-même</b> : s'il contient une boucle d'au moins {R.PERIODE_MIN} mesures,
c'est la boucle qui devient le modèle, pas le mot — c'est ce qui rattrape les
morceaux dont l'intro fait déjà tourner un bout du A. On retire alors toutes
ses occurrences, et on <b>recoud le morceau</b> : ce qui reste devient la
chanson qu'on analyse à l'étape suivante, où deux passages séparés par une
section retirée sont désormais voisins. Chaque étape ci-dessous montre tous les
retours
testés, et lequel a été choisi : parmi ceux qui passent, on prend le plus court
dont la longueur est un <b>multiple de {R.CARRURE} mesures</b> — la carrure.
Enfin, reconnaître une occurrence est plus sévère que repérer un retour : le
seuil d'occurrence se lit sur les scores du modèle lui-même, et ce qui tombe
entre les deux est <em>écarté</em> et reste sans section.
Cliquez n'importe quelle mesure pour l'écouter.</p>
{"".join(blocs)}
<h2>Ce que la démo montre à régler</h2>
<ul class="pied">
<li><b>Réglé — la carrure choisit le retour.</b> Stand By Me prenait le retour
de la mesure 7, donc un mot de 7 mesures : une longueur qui n'existe pas dans ce
morceau. Parmi les retours qui passent, on prend maintenant le plus court dont
la longueur est un multiple de 4 : le mot fait 8, et les occurrences tombent
enfin sur une grille de 8. Sur les 15 morceaux annotés, le nombre de ceux qui ne
rendaient qu'UNE section est passé de 11 à 8, cette règle et le seuil
d'occurrence comptant chacun pour moitié.</li>
<li><b>Réglé — reconnaître n'est pas repérer.</b> This Love 50–53 entrait dans le
A à 0,726 pour un seuil de 0,672, en plein milieu du pont, alors que les vraies
occurrences du A sortent entre 0,91 et 1,00. Le seuil d'occurrence se lit
désormais sur les scores du modèle lui-même (Otsu), jamais en dessous du seuil
de retour : 50–53 est écarté, et le pont 48–55 ressort entier comme « sans
section » — exactement le pont de Louis.</li>
<li><b>La boucle de Stand By Me rate d'un cheveu, et c'est la cadence.</b> Le
mot de 8 boucle à 4 avec des ressemblances 0,72 · 0,79 · 0,86 · <b>0,55</b> —
moyenne 0,732 contre un seuil de 0,732. C'est la 4<sup>e</sup> mesure, celle qui
cadence, qui fait tout tomber. La tolérance « modulo les 2 dernières mesures »
existe pour exactement ça, mais elle ne s'applique qu'au mot, pas à la boucle.
Question ouverte : faut-il l'étendre ? (Elle a été mise à zéro pour les
occurrences d'une boucle, pour une raison inverse — voir CHOIX 4.)</li>
<li><b>Rien n'ancre la phase.</b> L'algo démarre mesure 0 ; Louis fait commencer
Let It Be mesure 4 et Stand By Me mesure 6. Tout le découpage est décalé
d'autant. La marque « mesure 1 » (<code>chart["bar1"]</code>) est faite pour ça
et n'est pas encore lue ici.</li>
<li><b>La boucle interne rend le A et le B séparables, mais elle ne dit pas
où le A s'arrête.</b> Sur This Love elle fait tomber les cinq B exactement sur
ceux de Louis (16–23, 36–43, 56–63, 64–71, 72–79). En échange, A ne fait plus
que 4 mesures et avale l'intro, la queue et une moitié du pont : le modèle est
juste, mais rien ne regroupe deux boucles voisines en une phrase de 8. Il
manque une passe de recollement au-dessus.</li>
<li><b>Le recousu déplace la découverte du B là où il commence vraiment.</b>
Sur This Love, le B se trouve maintenant à l'étape 2 depuis la mesure 16 : une
fois les A retirés, la mesure 23 est directement suivie de la 36, le mot 16–23
et sa répétition 36–43 sont voisins, ressemblance 0,997. Sans le recousu,
l'étape butait sur « pas la place pour la répétition qui suit » et ne
retrouvait le B que bien plus loin, mesure 58, par raccroc. Le résultat final
est le même ici — mais il n'était plus trouvé pour la bonne raison, et sur un
morceau où la deuxième section ne revient pas trois fois de plus, il ne serait
pas trouvé du tout.</li>
<li><b>Une occurrence ne peut pas enjamber une couture</b> — sinon on écrirait
une section qui saute un trou et n'existe pas en musique. C'est le seul
garde-fou du recousu, et il se lit dans les verdicts « le mot enjamberait une
section déjà retirée ».</li>
<li><b>Une ressemblance purement harmonique ne sépare pas couplet et refrain</b>
quand ils partagent la grille : sur Let It Be un seul mot de 8 mesures pave tout
le morceau, sur Stand By Me un seul aussi. Il faudra un second signal (le chant,
l'énergie) ou une règle de longueur pour couper.</li>
<li><b>Les occurrences dérivent encore d'une mesure.</b> Elles sont posées de
gauche à droite sans contrainte de phase : sur Let It Be, une occurrence tombe
maintenant en 33–36 (0,87) alors que 32–35 est juste en dessous du seuil (0,76).
Le seuil plus sévère n'a pas causé ce décalage, il l'a révélé — c'est le
problème de phase du point ci-dessus. Une occurrence devrait préférer la phase
du modèle plutôt que la position la plus à gauche.</li>
<li><b>Vérifié sur les 15 morceaux que Louis a annotés</b> (règle #5 : un
résultat sur un morceau est une hypothèse) : rien ne plante, et le nombre de
morceaux qui ne rendent qu'UNE section est tombé de <b>11 à 8</b>. Ce qui
reste bloqué là-dessus, ce n'est plus un seuil : c'est que l'harmonie seule ne
sépare pas deux sections qui partagent la grille.</li>
</ul>
<script>{JS}</script>
</html>"""
    SORTIE.write_text(page, encoding="utf-8")
    print(f"écrit {SORTIE} ({len(page) / 1024:.0f} ko)")


if __name__ == "__main__":
    main()
