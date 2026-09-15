#!/usr/bin/env python3
"""scripts/page_concordance_barre.py — empiler le début, pas la cadence.

Louis, 2026-08-19 :

    « quand on empile, souvent les 2 dernières barres sont différentes, auquel
      cas on n'empile que le début, à prendre en compte tout le temps car c'est
      un classique.. il faut une notion de concordance par barre quand on
      empile pour voir lesquels on empile lesquels on garde comme différents »

CE QUE LA PAGE MONTRE. Pour chaque lettre jouée plusieurs fois : une ligne par
passage, une colonne par mesure de la boucle, et en bas la pile — avec, mesure
par mesure, si elle a été empilée ou gardée à part.

Deux règles côte à côte :

  * **par lettre** (la prod aujourd'hui) — une seule mesure discordante et
    TOUTE la lettre est refusée, y compris les mesures qui concordaient ;
  * **par mesure** (`HARMONIA_FOLD_GATE=bar`) — la mesure discordante est seule
    exclue, chaque passage y garde son propre accord, le reste garde sa pile.

Les mesures gardées à part sont marquées. Chaque case se joue d'un tap : c'est
à l'oreille qu'on juge si la mesure exclue devait l'être.

    .venv/bin/python scripts/page_concordance_barre.py [--morceaux N]
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
SORTIE = REPO / "docs" / "plots" / "concordance_barre.html"

NOTES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
LET = ["#3f6f8f", "#a4462b", "#3d6b47", "#8a6d1f", "#6b5b8a", "#b07a3a"]


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
    return " ".join(txt(c) for c in bar) if bar else "·"


REGLES = (
    ("prod", "la règle d'aujourd'hui", {}),
    ("barre", "par mesure", {"gate": "bar"}),
    ("transpo", "par mesure + transposition",
     {"gate": "bar", "transpose": True}),
)


def trois_regles(chart, grid, brut, probs, arr, times, cqt):
    """{clé: (rapport, mesures)} pour chacune des trois lois d'empilement."""
    from harmonia_min.folding import fold_letter_groups
    bpb = int(chart.get("bpb") or 4)
    boucle = os.environ.get("HARMONIA_FOLD_LOOP", "occurrence")
    sections = []
    for sec in chart.get("sections") or []:
        for b0, b1 in sec.get("barRanges") or []:
            sections.append({"label": str(sec.get("label") or "?"),
                             "barRanges": [[int(b0), int(b1)]]})
    sections.sort(key=lambda s: s["barRanges"][0][0])
    out = {}
    for cle, _nom, kw in REGLES:
        bars = copy.deepcopy(brut)
        rap = fold_letter_groups(sections, bars, grid, probs, bpb, arr=arr,
                                 times=times, combine="cqt", cqt=cqt,
                                 loop=boucle, **kw)
        out[cle] = (rap, bars)
    return sections, out


def bloc_lettre(L, occ, res, brut, grid, coul):
    """Le HTML d'une lettre : les passages, puis une pile par règle."""
    P = None
    for cle, _n, _k in REGLES:
        r = (res[cle][0].get(L) or {}).get("period")
        if r:
            P = int(r)
            break
    if not P or len(occ) < 2:
        return None
    gard = [(b0, b1) for b0, b1 in occ if b1 - b0 + 1 == P]
    if len(gard) < 2:
        return None

    lignes = []
    for oi, (b0, _b1) in enumerate(gard, start=1):
        cells = "".join(
            f'<td data-t="{grid[b0 + k]:.2f}">'
            f'{html.escape(mesure_txt(brut[b0 + k]))}</td>' for k in range(P))
        lignes.append(f'<tr><th>passage {oi}<span>mes. {b0 + 1}</span></th>'
                      f"{cells}</tr>")

    b0 = gard[0][0]
    notes = []
    for cle, nom, _kw in REGLES:
        rap, bars = res[cle]
        rep = rap.get(L) or {}
        pris = bool(rep.get("period")) and not rep.get("reason")
        if not pris:
            cells = f'<td class="refus" colspan="{P}">refusé — ' \
                    'chaque passage garde ce qu\'il a entendu seul</td>'
        else:
            apart = set(rep.get("pos_skip") or []) | set(rep.get("cv_skip") or [])
            cells = "".join(
                (f'<td class="pile apart" data-t="{grid[b0 + k]:.2f}">'
                 "chacun le sien</td>") if k in apart else
                (f'<td class="pile" data-t="{grid[b0 + k]:.2f}">'
                 f"{html.escape(mesure_txt(bars[b0 + k]))}</td>")
                for k in range(P))
            dt = rep.get("demiton") or {}
            if dt:
                for oi, (bb0, _bb1) in enumerate(gard, start=1):
                    r = dt.get(bb0) or dt.get(str(bb0))
                    if r:
                        notes.append(f"passage {oi} ramené de {r} demi-ton"
                                     f'{"s" if r > 1 else ""} avant la pile')
        lignes.append(f'<tr class="basse {cle}"><th>{nom}</th>{cells}</tr>')

    entete = "".join(f'<th class="pos">{k + 1}</th>' for k in range(P))
    mot = ('<p class="gagne">\u266b ' + " ; ".join(dict.fromkeys(notes))
           + ".</p>") if notes else ""
    return (f'<div class="bloc"><h3><b style="background:{coul}">'
            f'{html.escape(L)}</b> — {len(gard)} passages, boucle de {P} '
            f'mesures</h3><div class="tw"><table><tr><th></th>{entete}</tr>'
            f'{"".join(lignes)}</table></div>{mot}</div>')


def main() -> None:
    from harmonia_min import musx as _musx
    from harmonia_min.nnls_features import extract_bothchroma
    from harmonia_min.soudure import accords_par_mesure
    from harmonia_min.span_rescore import musx_cache_path

    vise = int(sys.argv[sys.argv.index("--morceaux") + 1]) \
        if "--morceaux" in sys.argv else 4
    sections_html, faits = [], 0
    for p in sorted((ETAT / "charts").glob("min_*.json")):
        if faits >= vise:
            break
        try:
            chart = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        audio = AUD / Path(chart.get("audio_url") or "x").name
        grid = chart.get("barGrid") or []
        if not audio.exists() or len(grid) < 3:
            continue
        if not musx_cache_path(audio).exists():
            continue                     # jamais recalculer un modèle ici
        try:
            brut = accords_par_mesure(chart)
            probs = _musx.frame_posteriors(audio)
            arr, times = extract_bothchroma(audio)
            cqt = _musx.song_cqt(audio)
            sections, res = trois_regles(chart, grid, brut, probs, arr,
                                         times, cqt)
        except Exception as exc:
            print(f"  passé {p.stem} : {type(exc).__name__}: {exc}")
            continue
        lettres = sorted({s["label"] for s in sections})
        blocs = []
        for L in dict.fromkeys(s["label"] for s in sections):
            occ = sorted(tuple(s["barRanges"][0]) for s in sections
                         if s["label"] == L)
            h = bloc_lettre(L, occ, res, brut, grid,
                            LET[lettres.index(L) % len(LET)])
            if h:
                blocs.append(h)
        if not blocs:
            continue
        faits += 1
        sections_html.append(
            f'<section data-audio="{chart["audio_url"]}" '
            f'data-dur="{float(grid[-1]):.2f}">'
            f'<h2>{html.escape(chart.get("title") or p.stem)}</h2>'
            f'<audio controls preload="none"></audio>'
            f'{"".join(blocs)}<div class="regle"><i></i></div></section>')
        print(f"  {chart.get('title')} — {len(blocs)} lettre(s)")

    doc = f"""<!doctype html><html lang="fr"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Empiler le début, pas la cadence</title>
<style>
:root{{--fg:#1b1b19;--fond:#fbf7ec;--trait:#ddd6c7;--doux:#6b6558}}
*{{box-sizing:border-box}}
body{{margin:0 auto;padding:18px 14px 60px;background:var(--fond);color:var(--fg);
 font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif;
 max-width:920px}}
h1{{font-size:20px;margin:0 0 6px}}
h2{{font-size:15px;margin:0 0 8px}}
h3{{font-size:13.5px;margin:0 0 8px;font-weight:600}}
h3 b{{color:#fff;border-radius:4px;padding:1px 7px;margin-right:6px}}
.chapo,.lex{{color:var(--doux);margin:0 0 13px;font-size:13.5px}}
.lex b{{color:var(--fg)}}
section{{background:#fff;border:1px solid var(--trait);border-radius:9px;
 padding:11px 13px;margin:0 0 12px}}
audio{{width:100%;height:32px;margin:0 0 10px;display:block}}
.bloc{{margin:0 0 16px}}
.bloc:last-of-type{{margin-bottom:6px}}
.tw{{overflow-x:auto;-webkit-overflow-scrolling:touch}}
table{{border-collapse:collapse;font-size:12.5px;min-width:100%}}
th,td{{border:1px solid #ece5d6;padding:5px 7px;text-align:left;white-space:nowrap}}
th{{color:var(--doux);font-weight:600;font-size:11px;text-align:right;
 background:#faf6ec}}
th span{{display:block;font-weight:400;opacity:.65;font-size:10px}}
th.pos{{text-align:center;width:1%}}
td{{cursor:pointer;font-variant-numeric:tabular-nums}}
td:active{{background:#f3ead8}}
td.refus{{background:#faf4ea;color:var(--doux);font-style:italic;
 cursor:default}}
tr.basse th{{background:#f1ece0;color:var(--fg);text-align:right}}
tr.prod th{{background:#f6efe6}}
tr.transpo th{{background:#eef3ee}}
td.pile{{background:#eef3ee;font-weight:600}}
td.pile.apart{{background:#f7f1e6;font-weight:400;font-style:italic}}
.gagne{{margin:0;font-size:12.5px;background:#f3f6f1;border-left:3px solid #3d6b47;
 padding:7px 9px;border-radius:0 5px 5px 0}}
.note{{margin:0;font-size:12.5px;color:var(--doux)}}
.regle{{position:relative;height:3px;margin-top:8px;background:#efe9db;
 border-radius:2px}}
.regle i{{position:absolute;top:-2px;width:2px;height:7px;background:#a4462b;
 border-radius:1px;left:0;opacity:0}}
</style>
<h1>Empiler le début, pas la cadence — et recaler la hauteur</h1>
<p class="chapo">Une section jouée quatre fois est la même quatre fois — sauf
sa fin, qui prépare la suite et change à chaque tour. Et quand le morceau
monte d'un ton, c'est la même section, plus haut. Trois lois d'empilement,
côte à côte, sur les mêmes passages.</p>
<p class="lex"><b>Une ligne = un passage</b>, tel que musx l'a entendu seul.
<b>Les trois lignes du bas</b> = ce que la pile écrit sous chaque loi.
<b>« chacun le sien »</b> = mesure gardée à part, les passages n'y concordent
pas. Tape une case pour l'écouter.</p>
{''.join(sections_html)}
<script>
document.querySelectorAll("section[data-audio]").forEach(function(sec){{
  var el = sec.querySelector("audio"), dur = parseFloat(sec.dataset.dur);
  var tete = sec.querySelector(".regle i");
  el.src = sec.dataset.audio;
  if (window.fetch) fetch(sec.dataset.audio)
    .then(function(r){{ return r.ok ? r.blob() : null; }})
    .then(function(b){{ if (b) el.src = URL.createObjectURL(b); }})
    .catch(function(){{}});
  sec.addEventListener("click", function(ev){{
    var c = ev.target.closest("[data-t]"); if (!c) return;
    try {{ el.currentTime = parseFloat(c.dataset.t); el.play(); }} catch(e){{}}
  }});
  /* timeupdate, jamais requestAnimationFrame (Safari iOS). */
  el.addEventListener("timeupdate", function(){{
    tete.style.left = (el.currentTime/dur*100) + "%"; tete.style.opacity = 1;
  }});
}});
</script>
</html>
"""
    SORTIE.write_text(doc, encoding="utf-8")
    print(f"\n  http://100.89.209.63:7772/plots/{SORTIE.name}")


if __name__ == "__main__":
    main()
