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

Les refus sont montrés aussi, avec leur raison — une lettre que le repli refuse
d'empiler garde son premier décodage, et il vaut mieux le voir que le deviner.

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


def main() -> None:
    stem = "min_T64BgKEL-Sw"
    if "--chart" in sys.argv:
        stem = sys.argv[sys.argv.index("--chart") + 1]
    p = ETAT / "charts" / f"{stem}.json"
    chart = json.loads(p.read_text(encoding="utf-8"))
    grid = chart["barGrid"]
    audio = AUD / Path(chart["audio_url"]).name
    titre = chart.get("title") or stem

    # 1. LES SECTIONS — celles de SongFormer, le détecteur en prod.
    from harmonia_min import songformer as SF
    segs = SF.detect_sections(grid, audio)

    # 2. LE BRUT — le décodage à plat, avant tout repli. La vérité terrain.
    from harmonia_min.soudure import accords_par_mesure
    brut = accords_par_mesure(chart)

    # 3. L'EMPILEMENT — les occurrences d'une lettre empilées sur le CQT, une
    #    inférence musx sur la pile, réécriture des mesures qui contribuent.
    #    C'est exactement le chemin de la prod (HARMONIA_MERGE=cqt).
    from harmonia_min import musx as _musx
    from harmonia_min.folding import fold_letter_groups
    from harmonia_min.nnls_features import extract_bothchroma
    replie = copy.deepcopy(brut)
    sections = [{"label": s["label"], "barRanges": [[s["b0"], s["b1"]]]}
                for s in segs]
    probs = _musx.frame_posteriors(audio)
    arr, times = extract_bothchroma(audio)
    cqt = _musx.song_cqt(audio)
    rapport = fold_letter_groups(
        sections, replie, grid, probs, int(chart.get("bpb") or 4),
        arr=arr, times=times, combine="cqt", cqt=cqt,
        loop=os.environ.get("HARMONIA_FOLD_LOOP", "occurrence"))

    # 4. LA PAGE.
    lettres = sorted({s["label"] for s in segs})
    blocs = []
    n_chg = 0
    for L in [x for x in dict.fromkeys(s["label"] for s in segs)]:
        occ = [(s["b0"], s["b1"]) for s in segs if s["label"] == L]
        rep = rapport.get(L) or {}
        coul = LET[lettres.index(L) % len(LET)]
        if rep.get("reason"):
            blocs.append(
                f'<section><h2><b style="background:{coul}">{html.escape(L)}'
                f'</b> — {len(occ)} passage(s)</h2>'
                f'<p class="refus">Pas d\'empilement : {html.escape(rep["reason"])}.'
                f' Chaque passage garde le décodage de sa première écoute.</p>'
                f'</section>')
            continue
        P = int(rep["period"])
        # positions -> quelles mesures, occurrence par occurrence
        lignes = []
        for oi, (b0, b1) in enumerate(occ):
            cells = []
            for k in range(P):
                b = b0 + k
                if b > b1:
                    cells.append('<td class="vide"></td>')
                    continue
                chg = b in (rep.get("changed") or [])
                var = b in (rep.get("variants") or [])
                n_chg += 1 if chg else 0
                cls = "chg" if chg else ("var" if var else "")
                cells.append(
                    f'<td class="{cls}" data-t="{grid[b]:.2f}">'
                    f'{html.escape(mesure_txt(brut[b]))}'
                    + ('<i title="variante : écartée de la pile">△</i>' if var else "")
                    + '</td>')
            lignes.append(
                f'<tr><th>passage {oi + 1}<span>mes. {b0 + 1}</span></th>'
                + "".join(cells) + "</tr>")
        # la ligne du bas : ce que musx entend sur la pile
        bas = []
        for k in range(P):
            b = occ[0][0] + k
            saute = k in (rep.get("cv_skip") or []) or k in (rep.get("pos_skip") or [])
            bas.append(
                f'<td class="pile{" saute" if saute else ""}" '
                f'data-t="{grid[b]:.2f}">'
                f'{html.escape(mesure_txt(replie[b]) if not saute else "—")}</td>')
        obs = rep.get("n_obs") or []
        lignes.append(
            '<tr class="basse"><th>musx sur la pile'
            f'<span>{len(occ)} passages empilés</span></th>' + "".join(bas) + "</tr>")
        entete = "".join(f'<th class="pos">{k + 1}</th>' for k in range(P))
        blocs.append(
            f'<section><h2><b style="background:{coul}">{html.escape(L)}</b>'
            f' — {len(occ)} passage(s), boucle de {P} mesure(s)</h2>'
            f'<div class="tw"><table><tr><th></th>{entete}</tr>'
            + "".join(lignes) + '</table></div>'
            + (f'<p class="note">Positions écartées (les passages y sont trop '
               f'différents pour être moyennés) : '
               f'{", ".join(str(k + 1) for k in sorted(set((rep.get("cv_skip") or []) + (rep.get("pos_skip") or []))))}.</p>'
               if (rep.get("cv_skip") or rep.get("pos_skip")) else "")
            + '</section>')

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
h2{{font-size:14px;margin:0 0 9px;font-weight:600}}
h2 b{{color:#fff;border-radius:4px;padding:1px 7px;margin-right:6px}}
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
td.pile.saute{{background:#f7f4ec;color:var(--doux);font-weight:400}}
.refus,.note{{margin:0;font-size:12.5px;color:var(--doux)}}
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
<b>△</b> = passage écarté de la pile, trop différent des autres.</p>
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
    print(f"{n_chg} mesure(s) réécrites par l'empilement")
    print(f"  http://100.89.209.63:7772/plots/{SORTIE.name}")


if __name__ == "__main__":
    main()
