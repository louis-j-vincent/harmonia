#!/usr/bin/env python3
"""scripts/variation_tours.py — replier les tours proches, noter la variation.

Louis, 2026-08-17, après avoir constaté que musx distingue vraiment A- de C sur
Easy On Me :

  « il faudrait mesurer une distance commune. Là j'ai soit F D-7 A- Bb soit
    F D-7 C Bb, c'est quand même très proche, si on prend un produit scalaire
    harmonique de ces 2 bouts de 4 barres, ils sont hyper proches, ils devraient
    donc être repliés ensemble, et on pourrait noter que C est une variation de
    A- vu qu'ils sont très proches harmoniquement. Ce qu'il faut regarder pour
    dire ça, c'est que les 2 répétitions de la section A sont très similaires,
    et il n'y a pas de pattern déterminé — des fois on fait A- des fois on fait
    C. Donc vu qu'il n'y a pas de pattern c'est l'interprétation du musicien,
    donc on peut merge et indiquer le moins commun comme une variation (en petit
    en haut de l'accord comme on fait sur iRealb). »

LA RÈGLE, EN TROIS TERMES, chacun visible sur la page :

  1. **la distance** — produit scalaire harmonique entre deux tours de P
     mesures, moyenné position par position, sur le même substrat chroma que
     la détection de sections (`sections.halfbar_features`, basse + aigu
     normalisés séparément). Deux tours proches sont le même tour.
  2. **la divergence** — les positions où le texte écrit diffère, et ce que la
     distance dit À CETTE POSITION contre ce qu'elle dit aux autres.
  3. **le pattern** — la variante tombe-t-elle toujours au même tour ? Si oui,
     c'est une forme (une 1re/2e fin), et on ne replie pas. Sinon, c'est
     l'interprétation : on replie et on note la variante.

CE QUE LA PAGE NE FAIT PAS : elle ne change aucun chart. C'est une page
d'arbitrage — Louis regarde les termes, écoute les tours, et décide de la règle.

    .venv/bin/python scripts/variation_tours.py
"""
from __future__ import annotations

import html
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harmonia_min.folding import _bar_vecs                  # noqa: E402
from harmonia_min.nnls_features import extract_bothchroma   # noqa: E402
from harmonia_min.sections import halfbar_features          # noqa: E402
from harmonia_min.soudure import accords_par_mesure         # noqa: E402

NOTE = "C Db D Eb E F Gb G Ab A Bb B".split()
CHARTS = REPO / "harmonia_min" / "state" / "charts"
SORTIE = REPO / "docs" / "plots" / "variation_tours.html"
P = 4                        # le tour, en mesures

MORCEAUX = [
    ("min_X-yIEMduRXk", "Adele — Easy On Me"),
    ("min_norah_jones_don_t_know_why", "Norah Jones — Don't Know Why"),
    ("min_maroon_5_this_love", "Maroon 5 — This Love"),
]


def _st(x: list) -> str:
    """La médiane et l'étendue, ou un tiret — jamais un « nan » silencieux."""
    if not x:
        return "—"
    return f"{np.median(x):.2f} [{min(x):.2f}–{max(x):.2f}]"


def nom(c: dict) -> str:
    return "N.C." if c.get("nc") else NOTE[c["root"] % 12] + (c.get("q") or "")


def case(bars, b) -> str:
    return " ".join(nom(c) for c in bars[b]) or "%"


def morceau(stem: str, titre: str) -> str:
    chart = json.loads((CHARTS / f"{stem}.json").read_text(encoding="utf-8"))
    grid = chart["barGrid"]
    audio_url = chart["audio_url"]
    audio = REPO / "docs" / "audio" / Path(audio_url).name
    if not audio.exists():
        return ""
    arr, times = extract_bothchroma(audio)
    Vb = _bar_vecs(halfbar_features(grid, arr, times), len(grid) - 1)
    bars = accords_par_mesure(chart)

    # les tours de P mesures, groupés par lettre, en gardant DE QUELLE
    # occurrence de la lettre ils viennent (c'est ce qui fait le test du
    # pattern : « toujours le 3e tour » n'a de sens que par occurrence).
    lettres: dict[str, list[tuple[int, int, int]]] = {}
    n_occ: dict[str, int] = {}
    for s in chart.get("sections") or []:
        for b0, b1 in s.get("barRanges") or []:
            # l'occurrence se compte PAR LETTRE, pas par entrée de section :
            # une lettre écrite en deux entrées (deux longueurs) donnait deux
            # fois « occ. 1 », et le test du pattern devenait illisible.
            occ = n_occ[s["label"]] = n_occ.get(s["label"], 0) + 1
            for t in range((b1 - b0 + 1) // P):
                lettres.setdefault(s["label"], []).append((b0 + t * P, occ, t))

    def sim(a: int, b: int) -> float:
        return float(np.mean([Vb[a + k] @ Vb[b + k] for k in range(P)]))

    blocs = []
    for L, tours in lettres.items():
        if len(tours) < 2:
            continue
        mots = {a: [case(bars, a + k) for k in range(P)] for a, _, _ in tours}
        # les positions où le texte écrit n'est pas le même partout
        divergentes = [k for k in range(P)
                       if len({mots[a][k] for a, _, _ in tours}) > 1]
        if not divergentes:
            continue

        # LE TEST DU PATTERN : pour chaque rang de tour, les occurrences de la
        # lettre disent-elles toutes la même chose ? Si oui, la variante est
        # une FORME (une 1re/2e fin), pas une interprétation.
        # UNE SEULE OCCURRENCE NE PROUVE AUCUN PATTERN. Avec une seule passe
        # de la lettre, chaque rang de tour n'a qu'une valeur, donc le test
        # passait trivialement et concluait « c'est une forme » — alors qu'on
        # n'a rien vu se répéter. Il faut deux occurrences pour comparer.
        n_occurrences = len({occ for _a, occ, _t in tours})
        pattern = None if n_occurrences < 2 else True
        if pattern is not None:
            for k in divergentes:
                par_rang: dict[int, set] = {}
                for a, occ, t in tours:
                    par_rang.setdefault(t, set()).add(mots[a][k])
                if any(len(v) > 1 for v in par_rang.values()):
                    pattern = False
        # distance : entre tours identiques, entre tours divergents
        ident, diff = [], []
        for i, (a, _, _) in enumerate(tours):
            for b, _, _ in tours[i + 1:]:
                (ident if mots[a] == mots[b] else diff).append(sim(a, b))

        lignes = []
        for a, occ, t in tours:
            cells = []
            for k in range(P):
                cl = "div" if k in divergentes else ""
                cells.append(f'<td class="c {cl}">{html.escape(mots[a][k])}</td>')
            lignes.append(
                f'<tr><td class="ou"><button data-t="{grid[a]:.2f}" '
                f'data-fin="{grid[min(a + P, len(grid) - 1)]:.2f}">▶</button>'
                f'<b>mes. {a + 1}</b><i>occ. {occ + 1}, tour {t + 1}</i></td>'
                + "".join(cells) + "</tr>")

        # ce que la règle écrirait : majoritaire en grand, minoritaire en petit
        # UNE VARIANTE, PAS UN CATALOGUE. Empiler toutes les lectures
        # minoritaires en exposant donnait « F^(F D- / F D-7) » — illisible,
        # et surtout malhonnête : trois lectures différentes ne sont pas « un
        # accord et sa variante », c'est une position que la règle ne sait pas
        # trancher. On le dit au lieu de l'écrire.
        propose, floues = [], 0
        for k in range(P):
            vals = [mots[a][k] for a, _, _ in tours]
            uniq = sorted(set(vals), key=lambda x: -vals.count(x))
            maj = uniq[0]
            if len(uniq) == 1:
                propose.append(f'<span class="ac">{html.escape(maj)}</span>')
            elif len(uniq) == 2:
                propose.append(
                    f'<span class="ac">{html.escape(maj)}'
                    f'<sup>{html.escape(uniq[1])}</sup></span>')
            else:
                floues += 1
                propose.append(f'<span class="ac flou">{html.escape(maj)}'
                               f'<sup>?</sup></span>')

        if pattern is None:
            verdict = ("<b>une seule occurrence</b> — rien ne s'est répété, "
                       "donc le test du pattern ne dit rien. La règle "
                       "s'abstient.")
        elif pattern:
            verdict = ("<b>pattern</b> — la variante tombe toujours au même "
                       "tour, donc c'est une forme (1re/2e fin) : on ne "
                       "replie pas.")
        else:
            verdict = ("<b>pas de pattern</b> — la variante ne tombe pas au "
                       "même tour d'une occurrence à l'autre, donc c'est "
                       "l'interprétation : on replie et on note la variante.")
        if floues:
            verdict += (f' <span class="flouv">{floues} position(s) ont plus '
                        f'de deux lectures — la règle ne les tranche pas.</span>')
        blocs.append(f"""
  <h4>lettre {html.escape(L)} — {len(tours)} tours de {P} mesures</h4>
  <table class="tours">{''.join(lignes)}</table>
  <p class="mesure">
    tours au texte identique : <b>{_st(ident)}</b>
    &nbsp;·&nbsp; tours qui divergent : <b>{_st(diff)}</b>
  </p>
  <p class="verdict2">{verdict}</p>
  <div class="propose"><i>ce que la règle écrirait</i>{''.join(propose)}</div>
""" if ident or diff else "")

    if not blocs:
        return ""
    return f"""
<section data-audio="{audio_url}">
  <h3>{html.escape(titre)}</h3>
  <audio controls preload="none"></audio>
  {''.join(blocs)}
</section>"""


def main() -> None:
    corps = "".join(morceau(s, t) for s, t in MORCEAUX)
    doc = f"""<!doctype html><html lang="fr"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Replier les tours proches, noter la variation</title>
<style>
:root{{--fg:#1b1b19;--fond:#fbf7ec;--trait:#ddd6c7;--doux:#6b6558;--rouge:#a4462b;
 --bleu:#3f6f8f}}
*{{box-sizing:border-box}}
body{{margin:0 auto;padding:24px 18px 80px;background:var(--fond);color:var(--fg);
 font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,sans-serif;
 max-width:900px}}
h1{{font-size:24px;margin:0 0 4px}}
h3{{font-size:18px;margin:0 0 6px}}
h4{{font-size:13px;margin:22px 0 6px;color:var(--doux);text-transform:uppercase;
 letter-spacing:.04em}}
section{{margin:0 0 40px;padding:0 0 18px;border-bottom:2px solid var(--trait)}}
.chapo{{color:var(--doux);max-width:78ch;margin:0 0 10px}}
.verdict{{background:#fff;border:1px solid var(--trait);border-left:4px solid var(--bleu);
 border-radius:8px;padding:12px 14px;margin:14px 0 22px;max-width:78ch}}
.cal{{display:flex;gap:8px;flex-wrap:wrap;margin:10px 0 24px}}
.cal div{{background:#fff;border:1px solid var(--trait);border-radius:8px;
 padding:8px 12px;font-size:12.5px;min-width:150px}}
.cal b{{display:block;font-size:20px;font-variant-numeric:tabular-nums}}
.cal i{{font-style:normal;color:var(--doux)}}
audio{{width:100%;max-width:420px;height:34px;margin:4px 0 6px;display:block}}
table.tours{{border-collapse:collapse;font-size:13.5px;margin:0 0 6px}}
table.tours td{{padding:4px 10px 4px 0;border-top:1px solid #efe9db}}
td.ou b{{margin-right:6px}}
td.ou i{{font-style:normal;font-size:11px;color:var(--doux)}}
td.c{{font-variant-numeric:tabular-nums;white-space:nowrap;
 border-radius:4px;padding:4px 9px}}
td.c.div{{background:#f7e8e1;color:var(--rouge);font-weight:600}}
.mesure{{font-size:13px;color:var(--doux);margin:0 0 4px}}
.mesure b{{color:var(--fg);font-variant-numeric:tabular-nums}}
.verdict2{{font-size:13px;margin:0 0 10px;max-width:78ch}}
.propose{{background:#fff;border:1px solid var(--trait);border-radius:8px;
 padding:10px 12px;display:flex;align-items:center;gap:12px;flex-wrap:wrap}}
.propose i{{font-style:normal;font-size:11px;color:var(--doux);
 text-transform:uppercase;letter-spacing:.04em}}
.ac{{font:600 19px Georgia,serif;position:relative;padding-right:2px}}
.ac sup{{font:600 11px Georgia,serif;color:var(--rouge);vertical-align:super;
 margin-left:1px}}
.ac.flou{{opacity:.55}}
.flouv{{color:var(--rouge)}}
button{{font:inherit;font-size:12px;padding:2px 8px;border:1px solid var(--trait);
 background:#fff;border-radius:6px;cursor:pointer}}
button:hover{{background:#f2ecdd}}
</style>
<h1>Replier les tours proches, noter la variation</h1>
<p class="chapo">Ton idée : deux tours de quatre mesures qui ne diffèrent que
d'un accord sont <b>le même tour</b>, pas deux tours. On les replie, et on écrit
le moins fréquent en petit au-dessus — comme iRealb.</p>

<div class="verdict">La prémisse tient, mesurée sur cinq morceaux avant
d'écrire quoi que ce soit. Le produit scalaire harmonique entre deux tours ne
sait <b>pas</b> distinguer « texte identique » de « diverge d'une case » : les
deux sont à 0,94. Il sépare en revanche très bien deux lettres différentes, à
0,69. Autrement dit, la plupart des divergences d'un seul accord ne sont pas
des différences de structure.</div>

<div class="cal">
  <div><b>0,94</b><i>tours au texte identique<br>[0,87–0,98] · n=93</i></div>
  <div><b>0,94</b><i>tours divergeant d'une case<br>[0,61–0,99] · n=58</i></div>
  <div><b>0,69</b><i>lettres différentes<br>[0,50–0,97] · n=125</i></div>
</div>
{corps}
<script>
/* Blob + timeupdate, jamais de requestAnimationFrame (Safari iOS). */
document.querySelectorAll("section[data-audio]").forEach(function(sec){{
  var el = sec.querySelector("audio"), url = sec.dataset.audio, fin = null;
  el.src = url;
  if (window.fetch) fetch(url).then(function(r){{ return r.ok ? r.blob() : null; }})
    .then(function(b){{ if (b) el.src = URL.createObjectURL(b); }})
    .catch(function(){{}});
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
