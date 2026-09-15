#!/usr/bin/env python3
"""scripts/page_repli_musx.py — redonner les sections empilées à manger à musx.

Louis, 2026-08-19 : « montres-moi une démo sur bora bora de ce que donne la
réinférence des accords une fois le repliement par sections fait → on doit
redonner les sections empilées à manger à musx ».

CE QUE LA PAGE MONTRE, mesure par mesure et sans un seul score. Pour chaque
lettre du chart :

  * une LIGNE PAR OCCURRENCE — ce que musx a entendu la première fois, sur ce
    passage-là, seul ;
  * une COLONNE PAR POSITION dans la boucle de la section ;
  * une DERNIÈRE LIGNE : ce que musx entend quand on lui redonne les
    occurrences EMPILÉES (leurs CQT moyennés, une seule inférence sur la pile).

Une case qui change entre sa ligne et la ligne du bas est marquée. Chaque case
se joue d'un tap : c'est l'oreille qui tranche si l'empilement a eu raison.

ET SURTOUT LES REFUS. Le repli a un garde-fou : si les passages d'une lettre ne
se ressemblent pas assez, il refuse de les moyenner et chacun garde son premier
décodage. Sur Bora Bora il refuse TOUT. La page montre donc aussi, pour chaque
lettre refusée, ce que la pile AURAIT écrit — garde-fou débranché — pour que
Louis puisse trancher à l'oreille qui avait raison, le garde-fou ou la pile.

    .venv/bin/python scripts/page_repli_musx.py [--chart min_T64BgKEL-Sw]
"""
from __future__ import annotations

import copy
import html
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

ETAT = REPO / "harmonia_min" / "state"
AUD = REPO / "docs" / "audio"
SORTIE = REPO / "docs" / "plots" / "repli_musx.html"

NOTES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
LET = ["#3f6f8f", "#a4462b", "#3d6b47", "#8a6d1f", "#6b5b8a", "#b07a3a",
       "#4a7c8c", "#8a4a6d"]


def txt(c) -> str:
    if c is None:
        return "·"
    if c.get("nc"):
        return "N.C."
    t = NOTES[c["root"] % 12] + (c.get("q") or "")
    if c.get("bass", -1) >= 0 and c["bass"] != c["root"]:
        t += "/" + NOTES[c["bass"] % 12]
    return t


def mesure_txt(bar) -> str:
    """Une mesure = ses accords, séparés par une espace fine."""
    if not bar:
        return "·"
    return " ".join(txt(c) for c in bar)


INTERVALLES = {1: "un demi-ton", 2: "un ton", 3: "une tierce mineure",
               4: "une tierce majeure", 5: "une quarte", 7: "une quinte",
               8: "une sixte mineure", 9: "une sixte majeure",
               10: "une septi\u00e8me mineure", 11: "une septi\u00e8me majeure"}


def ecart_de_hauteur(brut, ref0, autre0, P):
    """L'intervalle constant entre deux passages, ou None s'il n'y en a pas.

    Deux passages peuvent etre LE MEME passage joue plus haut — c'est le cas
    d'un morceau qui module. Le garde-fou du repli les voit alors comme deux
    musiques differentes et refuse la pile, a juste titre : moyenner un passage
    en Re avec le meme en Mi bemol ecrirait n'importe quoi. Mais l'information
    utile est la : ce sont bien les memes accords, decales.

    On compare les fondamentales position par position et on retient l'ecart
    s'il est le MEME partout ou les deux passages ont un accord.
    """
    from collections import Counter
    ec = Counter()
    for k in range(P):
        a = (brut[ref0 + k] or [None])[0]
        b = (brut[autre0 + k] or [None])[0]
        if not a or not b or a.get("nc") or b.get("nc"):
            continue
        ec[(b["root"] - a["root"]) % 12] += 1
    if not ec:
        return None
    top, n = ec.most_common(1)[0]
    if top == 0 or n < 0.7 * sum(ec.values()):
        return None
    return top


def decoupages(chart, grid, audio):
    """[(nom, segs)] — les DEUX découpages, celui du modèle et celui du chart.

    Les montrer côte à côte n'est pas un luxe : c'est le découpage qui décide
    si l'empilement est seulement POSSIBLE. Deux passages ne s'empilent que
    s'ils font le même nombre de mesures, et les frontières de SongFormer
    tombent dans le temps, pas sur nos barres.
    """
    out = []
    try:
        from harmonia_min import songformer as SF
        out.append(("SongFormer (le détecteur en prod)",
                    SF.detect_sections(grid, audio)))
    except Exception as exc:
        print(f"  songformer indisponible : {exc}")
    segs = []
    for sec in chart.get("sections") or []:
        for b0, b1 in sec.get("barRanges") or []:
            segs.append({"b0": int(b0), "b1": int(b1),
                         "label": str(sec.get("label") or "?")})
    if segs:
        out.append(("le découpage déjà écrit dans le chart",
                    sorted(segs, key=lambda x: x["b0"])))
    return out


def groupe(nom, segs, chart, grid, audio, brut, probs, arr, times, cqt):
    """Le HTML d'un découpage : une section par lettre, refus compris."""
    import copy

    from harmonia_min import folding as _F
    from harmonia_min.folding import fold_letter_groups

    bpb = int(chart.get("bpb") or 4)
    boucle = os.environ.get("HARMONIA_FOLD_LOOP", "occurrence")
    sections = [{"label": s["label"], "barRanges": [[s["b0"], s["b1"]]]}
                for s in segs]
    replie = copy.deepcopy(brut)
    rapport = fold_letter_groups(sections, replie, grid, probs, bpb, arr=arr,
                                 times=times, combine="cqt", cqt=cqt,
                                 loop=boucle)
    # DEUXIÈME PASSE, GARDE-FOU DÉBRANCHÉ. `STACK_COHERENCE` est le seuil
    # au-dessous duquel le repli refuse de moyenner des passages jugés trop
    # différents. On le met à zéro le temps d'une passe pour voir CE QUE LA
    # PILE AURAIT ÉCRIT — jamais pour l'écrire dans un chart, seulement pour
    # que Louis arbitre à l'oreille si le refus était le bon appel.
    forcee = copy.deepcopy(brut)
    seuil = _F.STACK_COHERENCE
    _F.STACK_COHERENCE = -1.0
    try:
        rap_f = fold_letter_groups(sections, forcee, grid, probs, bpb, arr=arr,
                                   times=times, combine="cqt", cqt=cqt,
                                   loop=boucle)
    finally:
        _F.STACK_COHERENCE = seuil

    lettres = sorted({s["label"] for s in segs})
    blocs, n_chg, n_pris = [], 0, 0
    for L in dict.fromkeys(s["label"] for s in segs):
        occ = sorted((s["b0"], s["b1"]) for s in segs if s["label"] == L)
        rep, forc = rapport.get(L) or {}, rap_f.get(L) or {}
        coul = LET[lettres.index(L) % len(LET)]
        titre_l = (f'<h3><b style="background:{coul}">{html.escape(L)}</b>'
                   f' — {len(occ)} passage{"s" if len(occ) > 1 else ""}</h3>')
        P = rep.get("period") or forc.get("period")
        if not P or len(occ) < 2:
            blocs.append(
                f'<div class="bloc">{titre_l}<p class="refus">Rien à empiler : '
                + ("un seul passage." if len(occ) < 2
                   else "les passages n\u2019ont pas la même longueur.")
                + " Les mesures gardent ce que musx a entendu la première "
                  "fois.</p></div>")
            continue
        P = int(P)
        pris = bool(rep.get("period")) and not rep.get("reason")
        n_pris += 1 if pris else 0
        gard = [(b0, b1) for b0, b1 in occ if b1 - b0 + 1 == P]
        chgs = set(rep.get("changed") or [])
        vars_ = set((rep.get("variants") or []) + (forc.get("variants") or []))
        lignes = []
        for oi, (b0, _b1) in enumerate(gard, start=1):
            cells = []
            for k in range(P):
                bb = b0 + k
                chg = bb in chgs
                n_chg += 1 if chg else 0
                cls = "chg" if chg else ("var" if bb in vars_ else "")
                cells.append(
                    f'<td class="{cls}" data-t="{grid[bb]:.2f}">'
                    f'{html.escape(mesure_txt(brut[bb]))}'
                    + ('<i title="passage écarté de la pile">\u25b3</i>'
                       if bb in vars_ else "") + "</td>")
            lignes.append(f'<tr><th>passage {oi}<span>mes. {b0 + 1}</span>'
                          f'</th>{"".join(cells)}</tr>')
        src = replie if pris else forcee
        bas = "".join(
            f'<td class="pile" data-t="{grid[gard[0][0] + k]:.2f}">'
            f'{html.escape(mesure_txt(src[gard[0][0] + k]))}</td>'
            for k in range(P))
        cls_b = "basse" if pris else "basse refusee"
        etiq = "musx sur la pile" if pris else "ce que la pile aurait dit"
        sous = (f"{len(gard)} passages empilés" if pris
                else "garde-fou débranché")
        lignes.append(f'<tr class="{cls_b}"><th>{etiq}<span>{sous}</span>'
                      f"</th>{bas}</tr>")

        transp = []
        for oi, (b0, _b1) in enumerate(gard[1:], start=2):
            e = ecart_de_hauteur(brut, gard[0][0], b0, P)
            if e:
                transp.append("le passage %d est le m\u00eame, mont\u00e9 d\u2019%s"
                              % (oi, INTERVALLES.get(e, "%d demi-tons" % e)))
        mot = ('<p class="note transp">\u266b ' + " ; ".join(transp)
               + ". Le garde-fou a donc raison de refuser la pile : moyenner "
                 "deux hauteurs diff\u00e9rentes \u00e9crirait la mauvaise "
                 "musique. Ce qu'il faudrait, c'est <b>recaler la hauteur "
                 "avant d'empiler</b> — on y gagnerait une observation au lieu "
                 "d'en perdre une.</p>") if transp else ""
        verdict = mot + (
            '<p class="note">Le repli a écrit cette ligne dans le chart.</p>'
            if pris else
            '<p class="refus"><b>Le garde-fou a refusé cette pile</b> : les '
            'passages ne se ressemblent pas assez pour être moyennés. Chacun '
            'garde donc ce que musx a entendu sur lui seul. La ligne du bas '
            'est ce que la pile aurait écrit — à toi de dire qui avait '
            'raison.</p>')
        entete = "".join(f'<th class="pos">{k + 1}</th>' for k in range(P))
        blocs.append(f'<div class="bloc">{titre_l}<div class="tw"><table>'
                     f'<tr><th></th>{entete}</tr>{"".join(lignes)}</table>'
                     f'</div>{verdict}</div>')
    return (f'<section><h2>{html.escape(nom)}</h2>{"".join(blocs)}</section>',
            n_chg, n_pris)


def main() -> None:
    stem = "min_T64BgKEL-Sw"
    if "--chart" in sys.argv:
        stem = sys.argv[sys.argv.index("--chart") + 1]
    p = ETAT / "charts" / f"{stem}.json"
    chart = json.loads(p.read_text(encoding="utf-8"))
    grid = chart["barGrid"]
    audio = AUD / Path(chart["audio_url"]).name
    titre = chart.get("title") or stem

    from harmonia_min import musx as _musx
    from harmonia_min.nnls_features import extract_bothchroma
    from harmonia_min.soudure import accords_par_mesure
    brut = accords_par_mesure(chart)          # le brut : la vérité terrain
    probs = _musx.frame_posteriors(audio)
    arr, times = extract_bothchroma(audio)
    cqt = _musx.song_cqt(audio)

    blocs, n_chg, n_pris = [], 0, 0
    for nom, segs in decoupages(chart, grid, audio):
        h, c, pr = groupe(nom, segs, chart, grid, audio, brut, probs, arr,
                          times, cqt)
        blocs.append(h)
        n_chg += c
        n_pris += pr

    doc = f"""<!doctype html><html lang="fr"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ce que musx entend sur la pile</title>
<style>
:root{{--fg:#1b1b19;--fond:#fbf7ec;--trait:#ddd6c7;--doux:#6b6558}}
*{{box-sizing:border-box}}
body{{margin:0 auto;padding:18px 14px 60px;background:var(--fond);color:var(--fg);
 font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif;
 max-width:900px}}
h1{{font-size:20px;margin:0 0 6px}}
h2{{font-size:13px;margin:0 0 10px;font-weight:600;color:var(--doux);text-transform:uppercase;letter-spacing:.05em}}
h3{{font-size:14px;margin:0 0 9px;font-weight:600}}
.bloc{{margin:0 0 18px}}
.bloc:last-child{{margin-bottom:0}}
h3 b{{color:#fff;border-radius:4px;padding:1px 7px;margin-right:6px}}
.chapo{{color:var(--doux);margin:0 0 14px;font-size:13.5px}}
section{{background:#fff;border:1px solid var(--trait);border-radius:9px;
 padding:11px 13px;margin:0 0 12px}}
audio{{width:100%;height:34px;margin:0 0 12px;display:block}}
.tw{{overflow-x:auto;-webkit-overflow-scrolling:touch}}
table{{border-collapse:collapse;font-size:12.5px;min-width:100%}}
th,td{{border:1px solid #ece5d6;padding:5px 7px;text-align:left;
 white-space:nowrap}}
th{{color:var(--doux);font-weight:600;font-size:11px;text-align:right;
 background:#faf6ec}}
th span{{display:block;font-weight:400;opacity:.65;font-size:10px}}
th.pos{{text-align:center;width:1%}}
td{{cursor:pointer;font-variant-numeric:tabular-nums}}
td:active{{background:#f3ead8}}
td.vide{{background:#faf7f0;cursor:default}}
td.chg{{background:#fdf0e6;font-weight:600}}
td.var{{color:var(--doux)}}
td i{{font-style:normal;opacity:.5;margin-left:3px;font-size:10px}}
tr.basse th{{background:#f1ece0;color:var(--fg)}}
td.pile{{background:#eef3ee;font-weight:600}}
tr.refusee th{{background:#f7efe6}}
tr.refusee td.pile{{background:#faf2e8;font-weight:400;font-style:italic}}
.refus,.note{{margin:0 0 4px;font-size:12.5px;color:var(--doux)}}
.transp{{background:#f3f6f1;border-left:3px solid #3d6b47;padding:7px 9px;border-radius:0 5px 5px 0;color:var(--fg)}}
.lex{{font-size:12.5px;color:var(--doux);margin:0 0 14px}}
.lex b{{color:var(--fg)}}
.regle{{position:relative;height:4px;margin:9px 0 0;background:#efe9db;
 border-radius:2px}}
.regle i{{position:absolute;top:-2px;width:2px;height:8px;background:#a4462b;
 border-radius:1px;left:0;opacity:0}}
</style>
<h1>{html.escape(titre)} — ce que musx entend sur la pile</h1>

<p class="chapo">Chaque lettre est jouée plusieurs fois. On empile ses passages
(leurs CQT moyennés) et on redonne la pile à musx : une seule écoute, faite de
toutes les autres. Tape n'importe quelle case pour l'entendre.</p>
<p class="lex"><b>Une ligne = un passage</b>, tel que musx l'a entendu seul.
<b>La ligne du bas</b> = ce qu'il entend sur les passages empilés.
<b>Fond orange</b> = l'empilement a changé cette mesure.
<b>Ligne du bas en italique</b> = le garde-fou a refusé cette pile ; c'est ce
qu'elle aurait écrit, et rien n'a été écrit dans le chart.</p>
<section id="lect">
  <audio controls preload="none" src="{chart['audio_url']}"></audio>
  <div class="regle"><i></i></div>
</section>
{''.join(blocs)}
<script>
var el = document.querySelector("audio"), tete = document.querySelector(".regle i");
var dur = {float(grid[-1]):.2f};
if (window.fetch) fetch(el.getAttribute("src"))
  .then(function(r){{ return r.ok ? r.blob() : null; }})
  .then(function(b){{ if (b) el.src = URL.createObjectURL(b); }})
  .catch(function(){{}});
document.addEventListener("click", function(ev){{
  var c = ev.target.closest("[data-t]"); if (!c) return;
  try {{ el.currentTime = parseFloat(c.dataset.t); el.play(); }} catch(e){{}}
}});
/* timeupdate, jamais requestAnimationFrame (Safari iOS). */
el.addEventListener("timeupdate", function(){{
  tete.style.left = (el.currentTime/dur*100) + "%"; tete.style.opacity = 1;
}});
</script>
</html>
"""
    SORTIE.write_text(doc, encoding="utf-8")
    print(f"{n_pris} lettre(s) empilée(s), {n_chg} mesure(s) réécrites")
    print(f"  http://100.89.209.63:7772/plots/{SORTIE.name}")


if __name__ == "__main__":
    main()
