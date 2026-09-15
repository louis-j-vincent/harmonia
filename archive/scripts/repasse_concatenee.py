#!/usr/bin/env python3
"""scripts/repasse_concatenee.py — redécoder sur les occurrences EMPILÉES.

Louis, 2026-08-17/18 :

  « et si tu réinfères en donnant à musx les postériors agrégés ? car sur
    Don't Know Why tu devrais choper Bbmaj7 Bb7 Ebmaj7 D+ G-7 C7 F7 Bb »
  « le + confiant chez musx si besoin de trancher, mais une repasse des
    sections concaténées c'est le mieux »

La page met côte à côte, pour chaque tour d'une lettre : ce que le tour dit
TOUT SEUL, puis ce que musx dit quand on lui donne les occurrences EMPILÉES.
Deux façons d'empiler, parce qu'elles ne donnent pas la même chose :

  * **moyenne des postérieures** — on décode chaque occurrence et on moyenne
    les probabilités de musx position par position ;
  * **empilement CQT** — on moyenne les SPECTRES et on refait tourner musx
    dessus. C'est le défaut de la prod depuis le 2026-08-12, choisi à
    l'oreille.

CE QUE LA PAGE NE FAIT PAS : elle n'écrit dans aucun chart, et elle DÉSARME
les deux vetos de la prod (cohérence de lettre, CV par position) — sinon on ne
verrait pas le résultat dont on discute. Ces vetos sont justement la question
ouverte : sur Don't Know Why ils refusent d'écrire les deux positions que
Louis attend.

    .venv/bin/python scripts/repasse_concatenee.py
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harmonia_min import musx as _musx                        # noqa: E402
from harmonia_min.folding import (                            # noqa: E402
    CV_MAX, _bar_vecs, _template_chords)
from harmonia_min.nnls_features import extract_bothchroma     # noqa: E402
from harmonia_min.sections import halfbar_features            # noqa: E402
from harmonia_min.soudure import accords_par_mesure           # noqa: E402

NOTE = "C Db D Eb E F Gb G Ab A Bb B".split()
CHARTS = REPO / "harmonia_min" / "state" / "charts"
MARQUES = REPO / "harmonia_min" / "state" / "sections"
SORTIE = REPO / "docs" / "plots" / "repasse_concatenee.html"

# (chart, lettre, période du tour, ce que Louis attend — ou None)
CAS = [
    ("min_norah_jones_don_t_know_why", "Norah Jones — Don't Know Why", "A", 4,
     "Bb^7 Bb7 | Eb^7 D+ | G-7 C7 | F7 Bb"),
    ("min_X-yIEMduRXk", "Adele — Easy On Me", "A", 4, None),
    ("min_maroon_5_this_love", "Maroon 5 — This Love", "B", 4, None),
]


def nom(c: dict) -> str:
    return "N.C." if c.get("nc") else NOTE[c["root"] % 12] + (c.get("q") or "")


def mots(bar) -> str:
    return " ".join(nom(c) for c in bar) or "%"


def cas(stem: str, titre: str, lettre: str, P: int, cible: str | None) -> str:
    chart = json.loads((CHARTS / f"{stem}.json").read_text(encoding="utf-8"))
    fm = MARQUES / f"{Path(chart['audio_url']).stem}.json"
    if not fm.exists():
        return ""
    marks = json.loads(fm.read_text(encoding="utf-8"))["sections"]
    grid, bpb = chart["barGrid"], int(chart["bpb"])
    audio = REPO / "docs" / "audio" / Path(chart["audio_url"]).name
    if not audio.exists():
        return ""
    probs = _musx.frame_posteriors(audio)
    arr, times = extract_bothchroma(audio)
    cqt = _musx.song_cqt(audio)
    bars = accords_par_mesure(chart)
    Vb = _bar_vecs(halfbar_features(grid, arr, times), len(grid) - 1)

    departs = [s["b0"] + t * P for s in marks if s["label"] == lettre
               for t in range((s["b1"] - s["b0"] + 1) // P)]
    if len(departs) < 2:
        return ""
    pos_members = [[a + k for a in departs] for k in range(P)]

    def bar_probs(b):
        i = max(0, int(round(grid[b] / _musx.FRAME_DT)))
        j = min(probs[0].shape[0], int(round(grid[b + 1] / _musx.FRAME_DT)))
        return [p[i:j] for p in probs]

    def bar_cqt(b):
        i = max(0, int(round(grid[b] / _musx.FRAME_DT)))
        j = min(cqt.shape[0], int(round(grid[b + 1] / _musx.FRAME_DT)))
        return cqt[i:j]

    Lf = max(bpb, int(round(float(np.median(np.diff(grid))) / _musx.FRAME_DT)))
    agr = {}
    for combine, bc in (("moyenne des postérieures", None),
                        ("empilement CQT", bar_cqt)):
        pos = _template_chords(pos_members, bar_probs, len(probs), Lf, bpb, P,
                               combine=("cqt" if bc else "mean"), weight=None,
                               bass_mode="avg", bar_cqt=bc, check_thr=None)
        agr[combine] = [mots(pos[k]) if pos[k] else "—" for k in range(P)]

    coh = [float(np.median([Vb[a] @ Vb[b] for i, a in enumerate(g)
                            for b in g[i + 1:]])) for g in pos_members]

    # POURQUOI la prod n'écrit pas : le veto CV, position par position
    def cv(g):
        out = 0.0
        for half in (0, 1):
            mid = 0.5
            X = []
            for b in g:
                t0, t1 = grid[b], grid[b + 1]
                m = t0 + mid * (t1 - t0)
                a, z = ((t0, m), (m, t1))[half]
                sel = (times >= a) & (times < z)
                X.append(arr[sel].mean(0) if sel.any()
                         else arr[int(np.argmin(np.abs(times - 0.5 * (a + z))))])
            X = np.array(X)
            out = max(out, float(np.sqrt(X.var(0).mean())
                                 / max(X.mean(0).mean(), 1e-9)))
        return out

    cvs = [cv(g) for g in pos_members]

    lignes = "".join(
        f'<tr><td class="ou"><button data-t="{grid[a]:.2f}" '
        f'data-fin="{grid[min(a + P, len(grid) - 1)]:.2f}">▶</button>'
        f'mes. {a + 1}</td>'
        + "".join(f'<td class="c">{html.escape(mots(bars[a + k]))}</td>'
                  for k in range(P)) + "</tr>" for a in departs)

    def rang(vals, cls):
        return ("".join(f'<td class="c {cls}">{html.escape(v)}</td>'
                        for v in vals))

    cible_html = (f'<tr class="cible"><td class="ou">ta cible</td>'
                  + "".join(f'<td class="c cib">{html.escape(x.strip())}</td>'
                            for x in cible.split("|")) + "</tr>"
                  if cible else "")

    gates = "".join(
        f'<td class="g {"bad" if cvs[k] > CV_MAX else ""}">'
        f'{coh[k]:.2f}<i>CV {cvs[k]:.2f}</i></td>' for k in range(P))

    return f"""
<section data-audio="{chart['audio_url']}">
  <h3>{html.escape(titre)} — lettre {html.escape(lettre)}</h3>
  <audio controls preload="none"></audio>
  <table>
    <tr><th>chaque tour, seul</th>{''.join(f'<th>mes. {k+1}</th>' for k in range(P))}</tr>
    {lignes}
    <tr class="agr"><td class="ou"><b>agrégé</b> — moyenne des postérieures</td>
      {rang(agr['moyenne des postérieures'], 'ag')}</tr>
    <tr class="agr"><td class="ou"><b>agrégé</b> — empilement CQT <i>(défaut prod)</i></td>
      {rang(agr['empilement CQT'], 'ag')}</tr>
    {cible_html}
    <tr class="gate"><td class="ou">cohérence · CV <i>(la prod refuse au-dessus de {CV_MAX:.2f})</i></td>{gates}</tr>
  </table>
</section>"""


def main() -> None:
    corps = "".join(cas(*c) for c in CAS)
    doc = f"""<!doctype html><html lang="fr"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>La repasse sur les sections concaténées</title>
<style>
:root{{--fg:#1b1b19;--fond:#fbf7ec;--trait:#ddd6c7;--doux:#6b6558;--rouge:#a4462b;
 --vert:#3d6b47;--bleu:#3f6f8f}}
*{{box-sizing:border-box}}
body{{margin:0 auto;padding:24px 18px 80px;background:var(--fond);color:var(--fg);
 font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif;
 max-width:900px}}
h1{{font-size:24px;margin:0 0 4px}}
h3{{font-size:17px;margin:0 0 6px}}
section{{margin:0 0 38px;padding:0 0 16px;border-bottom:2px solid var(--trait)}}
.chapo{{color:var(--doux);max-width:78ch;margin:0 0 10px}}
.verdict{{background:#fff;border:1px solid var(--trait);border-left:4px solid var(--vert);
 border-radius:8px;padding:12px 14px;margin:14px 0 24px;max-width:78ch}}
audio{{width:100%;max-width:420px;height:34px;margin:4px 0 8px;display:block}}
table{{border-collapse:collapse;font-size:13.5px;width:100%}}
th{{text-align:left;font-size:11px;color:var(--doux);text-transform:uppercase;
 letter-spacing:.03em;padding:0 8px 5px 0;font-weight:600}}
td{{padding:4px 8px 4px 0;border-top:1px solid #efe9db;white-space:nowrap}}
td.ou{{color:var(--doux);font-size:12.5px}}
td.ou b{{color:var(--fg)}}
td.ou i{{font-style:normal;font-size:11px}}
td.c{{font-variant-numeric:tabular-nums}}
tr.agr td{{background:#eef3ee;border-top:1px solid #dbe5dc}}
tr.agr td.ag{{font-weight:700;color:var(--vert)}}
tr.cible td.cib{{font-weight:700;color:var(--bleu)}}
tr.gate td{{font-size:11.5px;color:var(--doux)}}
td.g i{{display:block;font-style:normal;font-size:10.5px}}
td.g.bad{{color:var(--rouge);font-weight:600}}
button{{font:inherit;font-size:12px;padding:1px 7px;margin-right:6px;
 border:1px solid var(--trait);background:#fff;border-radius:6px;cursor:pointer}}
button:hover{{background:#f2ecdd}}
</style>
<h1>La repasse sur les sections concaténées</h1>
<p class="chapo">Chaque tour décodé tout seul, puis ce que musx dit quand on lui
donne les occurrences <b>empilées</b>. Deux façons d'empiler, qui ne donnent pas
la même chose.</p>

<div class="verdict"><b>Sur Don't Know Why, l'agrégation te rend 7 de tes 8
accords</b> — Eb^7, D+, G-7, C7, F7, Bb sortent tous, alors qu'aucun tour seul
ne les a tous. Il ne manque que la moitié de mesure Bb^7 au tout début. Et la
moyenne des postérieures fait mieux que l'empilement CQT, qui est pourtant le
défaut de la prod : celui-ci écrit D7 au lieu de D+ et F7sus4 au lieu de F7.
<br><br>
<b>Mais la prod n'écrirait rien de tout ça.</b> Le veto CV refuse les deux
premières positions — exactement celles qui portent tes Bb^7 et D+.</div>
{corps}
<script>
document.querySelectorAll("section[data-audio]").forEach(function(sec){{
  var el = sec.querySelector("audio"), url = sec.dataset.audio, fin = null;
  el.src = url;
  if (window.fetch) fetch(url).then(function(r){{ return r.ok ? r.blob() : null; }})
    .then(function(b){{ if (b) el.src = URL.createObjectURL(b); }}).catch(function(){{}});
  sec.addEventListener("click", function(ev){{
    var c = ev.target.closest("[data-t]"); if (!c) return;
    fin = c.dataset.fin ? parseFloat(c.dataset.fin) : null;
    try {{ el.currentTime = parseFloat(c.dataset.t); el.play(); }} catch(e){{}}
  }});
  el.addEventListener("timeupdate", function(){{
    if (fin !== null && el.currentTime >= fin) {{ el.pause(); fin = null; }}
  }});
}});
</script>
</html>
"""
    SORTIE.write_text(doc, encoding="utf-8")
    print(f"  {SORTIE.relative_to(REPO)}")
    print(f"  http://100.89.209.63:7772/plots/{SORTIE.name}")


if __name__ == "__main__":
    main()
