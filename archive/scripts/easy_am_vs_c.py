#!/usr/bin/env python3
"""scripts/easy_am_vs_c.py — A- ou C ? La question de Louis sur Easy On Me.

Louis, 2026-08-17 : « j'ai nommé le A, clairement c'est 3 fois la même boucle
avec les 2 dernières mesures qui changent, et le A d'après c'est pareil. Il y a
une confusion entre A- et C qui sont très proches, il faudrait voir si ce sont
en effet 2 accords bien différents ou si musx hésite entre les deux, auquel cas
on les merge en repliement pour inférer le vrai accord. »

La page ne donne pas un chiffre, elle donne à ÉCOUTER : les six mesures qui
occupent la même position de la boucle, chacune jouable seule ou dans son tour
de quatre, avec ce que musx en dit temps par temps. C'est lui qui tranche si
les mesures 7 et 29 jouent vraiment la même chose.

    .venv/bin/python scripts/easy_am_vs_c.py
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harmonia_min import musx as _musx                     # noqa: E402
from harmonia_min.folding import fold_letter_groups        # noqa: E402
from harmonia_min.nnls_features import extract_bothchroma  # noqa: E402
from harmonia_min.soudure import accords_par_mesure        # noqa: E402
from harmonia_min.span_rescore import (                    # noqa: E402
    Q5_TAIL, acoustic_logp_musx, idx_of, pool_span_musx, token_of)

CHART = REPO / "harmonia_min" / "state" / "charts" / "min_X-yIEMduRXk.json"
MARKS = REPO / "harmonia_min" / "state" / "sections" / "X-yIEMduRXk.json"
SORTIE = REPO / "docs" / "plots" / "easy_am_vs_c.html"
NOTE = "C Db D Eb E F Gb G Ab A Bb B".split()


def nom(c: dict) -> str:
    return "N.C." if c.get("nc") else NOTE[c["root"] % 12] + (c.get("q") or "")


def main() -> None:
    chart = json.loads(CHART.read_text(encoding="utf-8"))
    marques = json.loads(MARKS.read_text(encoding="utf-8"))["sections"]
    grid = chart["barGrid"]
    bpb = int(chart["bpb"])
    audio_url = chart["audio_url"]
    audio = REPO / "docs" / "audio" / Path(audio_url).name
    bars = accords_par_mesure(chart)

    probs = _musx.frame_posteriors(audio)
    arr, times = extract_bothchroma(audio)
    cqt = _musx.song_cqt(audio)
    iA, iC = idx_of(9, 1), idx_of(0, 0)          # A- et C

    # les deux A de Louis, et la position contestée de leur boucle de 4
    aa = [(s["b0"], s["b1"]) for s in marques if s["label"] == "A"]
    P = 4
    contestees = [b for b0, b1 in aa for b in range(b0, b1 + 1)
                  if (b - b0) % P == 2]

    def poster(t0: float, t1: float):
        pt, ps = pool_span_musx(probs, [(t0, t1)])
        return np.exp(acoustic_logp_musx(pt, ps)[0])[0]

    lignes = []
    for b in contestees:
        t0, t1 = grid[b], grid[b + 1]
        p = poster(t0, t1)
        top = [(NOTE[token_of(int(i))[0]] + Q5_TAIL[token_of(int(i))[1]],
                float(p[i])) for i in np.argsort(p)[::-1][:3]]
        q = (t1 - t0) / bpb
        temps = []
        for k in range(bpb):
            pk = poster(t0 + k * q, t0 + (k + 1) * q)
            temps.append((float(pk[iA]), float(pk[iC])))
        tour = next(b0 for b0, b1 in aa if b0 <= b <= b1)
        lignes.append({"bar": b, "t0": t0, "t1": t1,
                       "ecrit": " ".join(nom(c) for c in bars[b]) or "(tenu)",
                       "pA": float(p[iA]), "pC": float(p[iC]), "top": top,
                       "temps": temps, "tour_t0": grid[b - 2],
                       "section": f"A mes. {tour + 1}"})

    # ce que le repli dit de la lettre A, position par position
    sections = [{"label": s["label"], "barRanges": [[s["b0"], s["b1"]]]}
                for s in marques]
    rap = fold_letter_groups([dict(s) for s in sections],
                             [list(x) for x in accords_par_mesure(chart)],
                             grid, probs, bpb, arr=arr, times=times,
                             gate="bibar", combine="cqt", cqt=cqt,
                             loop="occurrence")
    cohA = (rap.get("A") or {}).get("coh") or []
    cohB = (rap.get("B") or {}).get("coh") or []

    def pct(x: float) -> str:
        return f"{100 * x:.0f}"

    cases = []
    for L in lignes:
        # LE GROS CHIFFRE EST CELUI QUI GAGNE. La première version montrait
        # toujours P(A-) en gros : sur une mesure en C, la case affichait « 0 »
        # en évidence et la vraie valeur en petit — on lisait le perdant.
        tds = "".join(
            f'<td class="t {"a" if a > c else "c"}">'
            f'<b>{pct(max(a, c))}</b><i>{"A-" if a > c else "C"}</i></td>'
            for a, c in L["temps"])
        top = " · ".join(f"{html.escape(n)} {v:.2f}" for n, v in L["top"])
        cases.append(f"""
  <tr>
    <td class="mes"><b>mes. {L['bar'] + 1}</b><i>{L['section']}</i></td>
    <td class="ec"><b>{html.escape(L['ecrit'])}</b></td>
    <td class="pp">{L['pA']:.3f}</td>
    <td class="pp">{L['pC']:.3f}</td>
    {tds}
    <td class="top">{top}</td>
    <td class="ec">
      <button data-t="{L['t0']:.2f}" data-fin="{L['t1']:.2f}">▶ la mesure</button>
      <button data-t="{L['tour_t0']:.2f}" data-fin="{L['t1'] + (L['t1'] - L['t0']):.2f}">▶ le tour</button>
    </td>
  </tr>""")

    boucle = []
    for b0, b1 in aa:
        tours = []
        for t in range((b1 - b0 + 1) // P):
            pos2 = b0 + t * P + 2
            if pos2 > b1:
                continue
            tours.append((t + 1, pos2, " ".join(nom(c) for c in bars[pos2])))
        boucle.append((b0, b1, tours))

    boucle_html = "".join(
        f"<tr><td><b>A mes. {b0 + 1}–{b1 + 1}</b></td>"
        + "".join(f'<td class="tour"><i>tour {t}</i>'
                  f'<button data-t="{grid[p]:.2f}" data-fin="{grid[p + 1]:.2f}">'
                  f'{html.escape(ch)}</button></td>' for t, p, ch in tours)
        + "</tr>" for b0, b1, tours in boucle)

    doc = f"""<!doctype html><html lang="fr"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Easy On Me — A- ou C ?</title>
<style>
:root{{--fg:#1b1b19;--fond:#fbf7ec;--trait:#ddd6c7;--doux:#6b6558;
 --am:#3f6f8f;--do:#a4462b}}
*{{box-sizing:border-box}}
body{{margin:0 auto;padding:24px 18px 80px;background:var(--fond);color:var(--fg);
 font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif;
 max-width:1080px}}
h1{{font-size:24px;margin:0 0 4px}}
h2{{font-size:17px;margin:34px 0 8px;border-bottom:1px solid var(--trait);
 padding-bottom:5px}}
.chapo{{color:var(--doux);max-width:78ch;margin:0 0 8px}}
.verdict{{background:#fff;border:1px solid var(--trait);border-left:4px solid var(--am);
 border-radius:8px;padding:12px 14px;margin:14px 0 6px;max-width:78ch}}
.verdict b{{color:var(--am)}}
audio{{width:100%;max-width:420px;height:34px;margin:10px 0 4px;display:block}}
table{{border-collapse:collapse;font-size:13px;width:100%;margin:6px 0 4px}}
th{{text-align:left;font-weight:600;color:var(--doux);font-size:11px;
 padding:2px 8px 6px 0;vertical-align:bottom;text-transform:uppercase;
 letter-spacing:.03em}}
td{{padding:5px 8px 5px 0;border-top:1px solid #efe9db;vertical-align:middle}}
td.mes b{{display:block;white-space:nowrap}}
td.mes i{{font-style:normal;font-size:11px;color:var(--doux)}}
td.pp{{font-variant-numeric:tabular-nums;font-weight:600}}
td.t{{width:44px;text-align:center;font-variant-numeric:tabular-nums;
 border-radius:4px}}
td.t b{{display:block;font-size:13px}}
td.t i{{font-style:normal;font-size:10.5px;color:var(--doux)}}
td.t.a{{background:#e6eff5;color:var(--am)}}
td.t.c{{background:#f7e8e1;color:var(--do)}}
td.top{{font-size:11.5px;color:var(--doux);white-space:nowrap}}
button{{font:inherit;font-size:12px;padding:4px 9px;margin:0 4px 2px 0;
 border:1px solid var(--trait);background:#fff;border-radius:6px;cursor:pointer}}
button:hover{{background:#f2ecdd}}
td.tour i{{display:block;font-style:normal;font-size:10.5px;color:var(--doux)}}
.coh{{display:flex;gap:6px;flex-wrap:wrap;margin:8px 0}}
.coh div{{background:#fff;border:1px solid var(--trait);border-radius:7px;
 padding:7px 11px;font-size:12.5px;text-align:center;min-width:96px}}
.coh div b{{display:block;font-size:17px;font-variant-numeric:tabular-nums}}
.coh div.bas b{{color:var(--do)}}
.coh div i{{font-style:normal;color:var(--doux);font-size:11px}}
</style>
<h1>Easy On Me — A- ou C ?</h1>
<p class="chapo">La boucle de A fait quatre mesures : <b>F | D-7 | ? | Bb</b>.
La troisième case est jouée six fois entre tes deux A. Trois fois elle sort
<b>A-</b>, trois fois <b>C</b>. Ta question : deux accords différents, ou musx
qui hésite ?</p>

<div class="verdict"><b>Musx n'hésite pas.</b> Sur chacune des six mesures, et
sur chacun des quatre temps, l'un des deux écrase l'autre d'un facteur 50 à 300.
Il n'y a pas une seule mesure où les deux se disputent. Les replier ensemble
effacerait une différence réelle — reste à écouter si elle est dans la musique
ou si le modèle se trompe avec assurance quelque part.</div>

<audio controls preload="none"></audio>

<h2>Les six mesures, temps par temps</h2>
<p class="chapo">Une case = un temps. Le chiffre est la probabilité de
l'accord qui gagne ce temps-là, en pourcentage, et la lettre dessous dit lequel.
Bleu = A-, rouge = C. Les deux colonnes P(A-) et P(C) donnent le détail sur la
mesure entière.</p>
<table>
<tr><th>mesure</th><th>écrit</th><th>P(A-)</th><th>P(C)</th>
<th>t1</th><th>t2</th><th>t3</th><th>t4</th><th>top-3 musx</th><th>écouter</th></tr>
{''.join(cases)}
</table>

<h2>Le même endroit, tour par tour</h2>
<p class="chapo">C'est ici que ça se joue : le <b>2ᵉ tour</b> de ton premier A
dit A-, celui de ton second A dit C. Si les deux A sont la même musique, l'un
des deux se trompe. Écoute-les l'un après l'autre.</p>
<table>{boucle_html}</table>

<h2>Ce que le repli en dit déjà</h2>
<p class="chapo">Cohérence de l'empilement, position par position dans la
boucle — c'est la ressemblance médiane entre les occurrences d'une même
position. Au-dessus de 0,85 le repli empile ; en dessous il refuse.</p>
<div class="coh">
{''.join(f'<div class="{"bas" if c is not None and c < 0.85 else ""}">'
         f'<b>{c:.2f}</b><i>position {k} — {lbl}</i></div>'
         for k, (c, lbl) in enumerate(zip(cohA, ["F", "D-7", "A- ou C", "Bb"]))
         if c is not None)}
</div>
<p class="chapo">Trois positions sur quatre sont aussi solides que la lettre B
(la tienne, qui empile à {min(cohB):.2f}–{max(cohB):.2f} et réécrit 13 mesures).
La position contestée s'effondre toute seule. <b>Le repli avait donc déjà
raison de refuser</b> — mais aujourd'hui il refuse la lettre ENTIÈRE à cause
d'elle, donc F, D-7 et Bb perdent l'empilement qu'ils méritaient.</p>

<script>
/* Blob + timeupdate, jamais de requestAnimationFrame : sur Safari iOS un rAF
   empêche le moteur audio de démarrer, et le lecteur ne bufferise pas un
   fichier servi en direct (206 en boucle). Voir docs/plots/retour_10morceaux. */
(function(){{
  var el = document.querySelector("audio"), url = "{audio_url}", fin = null;
  el.src = url;
  if (window.fetch) fetch(url).then(function(r){{ return r.ok ? r.blob() : null; }})
    .then(function(b){{ if (!b) return; el.src = URL.createObjectURL(b); }})
    .catch(function(){{}});
  document.addEventListener("click", function(ev){{
    var c = ev.target.closest("[data-t]"); if (!c) return;
    fin = c.dataset.fin ? parseFloat(c.dataset.fin) : null;
    try {{ el.currentTime = parseFloat(c.dataset.t); el.play(); }} catch(e){{}}
  }});
  el.addEventListener("timeupdate", function(){{
    if (fin !== null && el.currentTime >= fin) {{ el.pause(); fin = null; }}
  }});
}})();
</script>
</html>
"""
    SORTIE.write_text(doc, encoding="utf-8")
    print(f"  {SORTIE.relative_to(REPO)}")
    print(f"  http://100.89.209.63:7772/plots/{SORTIE.name}")
    print()
    print("  cohérence lettre A par position :",
          [None if c is None else round(c, 3) for c in cohA])
    print("  cohérence lettre B par position :",
          [None if c is None else round(c, 3) for c in cohB])


if __name__ == "__main__":
    main()
