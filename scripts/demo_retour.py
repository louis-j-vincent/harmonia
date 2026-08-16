#!/usr/bin/env python3
"""scripts/demo_retour.py — la démo pas-à-pas de `harmonia_min.retour`.

Écrit `docs/plots/retour_3morceaux.html` : pour chacun des trois morceaux, la
grille de mesures avec les accords, l'annotation à la main de Louis en regard,
puis CHAQUE étape de l'algorithme — les retours testés, ceux qui ont été jugés
forts, le mot proposé, le contrôle de répétition mesure par mesure, et les
occurrences retenues. Tout est cliquable : un clic sur une mesure place la tête
de lecture du disque.

Les trois morceaux sont pris dans `state/sections/`, donc parmi ceux dont Louis
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
SORTIE = REPO / "docs" / "plots" / "retour_3morceaux.html"

MORCEAUX = [
    ("min_maroon_5_this_love", "Maroon 5 — This Love"),
    ("min_let_it_be_remastered_2009", "The Beatles — Let It Be"),
    ("min_ben_e_king_stand_by_me_audio", "Ben E. King — Stand By Me"),
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


def table_candidats(etape, seuil) -> str:
    """Les retours forts qui sont allés jusqu'au test, un par ligne."""
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
            marques.append('<em title="le premier retour fort : ce que la '
                           'règle littérale prend">1er</em>')
        if c["passe"] and c is not etape["retenu"]:
            marques.append('<em class="cf" title="l\'algo ne l\'a pas testé : '
                           'il s\'était déjà arrêté avant">contrefactuel</em>')
        lignes.append(
            f'<tr class="{" ".join(cls)}"><td>{c["barre"]}</td><td>{c["L"]}</td>'
            f'<td>{c["sim"]:.3f}</td><td>{rep}</td>'
            f'<td>{e(c["verdict"])} {" ".join(marques)}</td></tr>')
    if not lignes:
        return '<p class="rien">aucun retour fort assez loin pour faire un mot.</p>'
    return ('<div class="tbl"><table class="cand">'
            '<tr><th>retour<br>mesure</th><th>mot<br>mesures</th>'
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
        if et["action"] == "avance":
            corps = (f'{strip_sims(et, seuil, n, grid)}'
                     f'{table_candidats(et, seuil)}'
                     f'<p class="issue avance">Aucun retour ne donne de section '
                     f'depuis la mesure {d}. On avance d\'une mesure ; la mesure '
                     f'{d} restera sans section.</p>')
        else:
            sec, ret = et["section"], et["retenu"]
            occ = " ".join(
                f'<b class="occ {"net" if o["moyenne"] >= NET else "flou"}">'
                f'{o["b0"]}–{o["b1"]} <i>{o["moyenne"]:.2f}</i></b>'
                for o in sec["occurrences"])
            corps = (f'{strip_sims(et, seuil, n, grid)}'
                     f'{table_candidats(et, seuil)}'
                     f'{detail_repet(ret, seuil, d, grid)}'
                     f'<p class="issue ok">Section <b class="lettre" '
                     f'style="--c:{coul_algo[sec["label"]]}">{sec["label"]}</b> '
                     f'= le mot de {sec["L"]} mesures qui commence mesure {d}. '
                     f'Ses {len(sec["occurrences"])} occurrences dans le morceau : '
                     f'{occ}</p>')
        etapes.append(
            f'<div class="etape"><h3>Étape {i} — on repart de la mesure {d}'
            f'<span>{"section trouvée" if et["action"] == "section" else "rien ici"}'
            f'</span></h3>{corps}</div>')

    resume = " · ".join(
        f'<b class="lettre" style="--c:{coul_algo[s["label"]]}">{s["label"]}</b> '
        f'{s["L"]} mesures × {len(s["occurrences"])}' for s in res["sections"]) \
        or "aucune section"

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
.barres b{flex:1 1 0;min-width:1px;background:#e3ddcf;cursor:pointer;
 border-radius:1px 1px 0 0}
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
section, et on va chercher toutes ses répétitions. Puis on recommence à la
première mesure encore libre. Chaque étape ci-dessous montre tous les retours
testés, celui que la règle littérale prend (<em>1er</em>), et — en bleu — ceux
qui <em>auraient marché aussi</em> mais que l'algo n'a jamais testés parce
qu'il s'était déjà arrêté. C'est là que se trouvent les réglages à trancher.
Cliquez n'importe quelle mesure pour l'écouter.</p>
{"".join(blocs)}
<h2>Ce que la démo montre à régler</h2>
<ul class="pied">
<li><b>Le premier retour n'est pas le bon retour.</b> Sur Stand By Me l'algo
s'arrête au retour de la mesure 7 (mot de 7 mesures, répétition 0,916) alors que
le retour de la mesure 8 donne 0,946 et tombe sur les phrases de 8 mesures de
Louis. Prendre le <i>meilleur</i> retour, ou le plus long parmi ceux qui
passent, plutôt que le premier.</li>
<li><b>Rien n'ancre la phase.</b> L'algo démarre mesure 0 ; Louis fait commencer
Let It Be mesure 4 et Stand By Me mesure 6. Tout le découpage est décalé
d'autant. La marque « mesure 1 » (<code>chart["bar1"]</code>) est faite pour ça
et n'est pas encore lue ici.</li>
<li><b>Une ressemblance purement harmonique ne sépare pas couplet et refrain</b>
quand ils partagent la grille : sur Let It Be un seul mot de 8 mesures pave tout
le morceau, sur Stand By Me un seul aussi. Il faudra un second signal (le chant,
l'énergie) ou une règle de longueur pour couper.</li>
<li><b>Les occurrences dérivent, et elles se dénoncent toutes seules.</b> Elles
sont posées de gauche à droite sans contrainte de phase : sur This Love une
occurrence tombe mesure 20 là où Louis attend 16 et enjambe sa frontière. Mais
regardez leurs scores — les bonnes sortent à 0,93–1,00 et les trois qui dérivent
à 0,69–0,71 (encadré orange). Sur les trois morceaux la coupure est nette :
0,88–1,00 d'un côté, 0,69–0,77 de l'autre. Le seuil qui accepte une occurrence
est le même que celui qui détecte un retour ; il devrait être bien plus
sévère.</li>
<li><b>Un mot de 7 mesures sur un morceau en 8 est un aveu.</b> Sur Stand By Me
toutes les occurrences sortent à 0,74–0,77, c'est-à-dire « faibles partout » :
quand aucune occurrence n'est nette, c'est la longueur du mot qui est fausse, pas
le morceau qui est flou. Il y a là un critère de rejet gratuit.</li>
</ul>
<script>{JS}</script>
</html>"""
    SORTIE.write_text(page, encoding="utf-8")
    print(f"écrit {SORTIE} ({len(page) / 1024:.0f} ko)")


if __name__ == "__main__":
    main()
