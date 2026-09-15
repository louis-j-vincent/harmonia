"""Voir le score d'appariement, position par position (Louis, 2026-08-12).

« Lorsque j'identifie une première section elle est mal rematchée aux autres
endroits, montre-moi le scoring de matching pour chaque section. »

Pour chaque lettre annotée à la main, on prend sa PREMIÈRE occurrence comme
ancre — exactement ce que fait l'outil quand le doigt marque une section —
et on affiche le score de TOUTES les positions du morceau : celles que
l'outil retient, celles qu'il rejette, et pourquoi. Les vraies occurrences
(son annotation) sont marquées, donc les erreurs se voient d'un coup d'œil.

Chaque position est écoutable : c'est la seule façon de trancher « ce match
est bon » quand aucune vérité d'accords n'existe.

    python scripts/match_scoring_page.py [--songs 6]
"""
from __future__ import annotations

import html
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from harmonia_min import harmonic_sections as HS      # noqa: E402
from harmonia_min import musx as _musx                # noqa: E402
from harmonia_min import section_tool as ST           # noqa: E402
from harmonia_min import voice_sections as VS         # noqa: E402

CHARTS = REPO / "harmonia_min" / "state" / "charts"
GT = REPO / "harmonia_min" / "state" / "sections"
AUDIO = REPO / "docs" / "audio"
REPORTS = REPO / "harmonia_min" / "state" / "reports"
SKIP = {"intro", "outro"}


def analyse(stem, gt, chart):
    grid = chart["barGrid"]
    n = len(grid) - 1
    V = HS.harmonic_vectors(_musx.frame_posteriors(AUDIO / f"{stem}.m4a")[0],
                            grid)
    S = V @ V.T
    by = defaultdict(list)
    for s in gt["sections"]:
        by[s["label"]].append((s["b0"], s["b1"]))
    out = []
    for lab, occ in sorted(by.items()):
        if lab in SKIP or len(occ) < 2:
            continue
        occ.sort()
        (b0, b1), rest = occ[0], occ[1:]
        L = b1 - b0 + 1
        if L < 2 or b0 + L > n:
            continue
        ch = VS._slide(S, b0, L, n)
        sc = VS.block_score(ch, ch, b0, n, mute=None, block=L)
        claimed = np.zeros(n, bool)
        claimed[b0:b0 + L] = True
        kept = VS._peaks(sc, b0, n, ST.TOOL_THR, L, claimed, par=0)
        true_starts = {r[0] for r in rest}
        # tolérance ±1 mesure pour « c'est la même reprise »
        def is_true(c):
            return any(abs(c - t) <= 1 for t in true_starts)
        cands = []
        for c in range(0, n - L + 1):
            if abs(c - b0) < L:
                continue
            why = None
            if c not in kept:
                if sc[c] < ST.TOOL_THR:
                    why = "sous le seuil"
                elif (c - b0) % 2:
                    why = "hors grille (mesure impaire)"
                else:
                    why = "masquée par un pic plus fort"
            cands.append({"c": c, "score": round(float(sc[c]), 3),
                          "kept": c in kept, "true": is_true(c), "why": why})
        found = sum(1 for t in true_starts
                    if any(abs(k - t) <= 1 for k in kept))
        out.append({"letter": lab, "b0": b0, "L": L, "n_true": len(rest),
                    "found": found,
                    "wrong": sum(1 for c in cands if c["kept"] and not c["true"]),
                    "cands": cands})
    return out


CSS = """
*{box-sizing:border-box}
body{margin:0;padding:14px 10px 90px;background:#faf6ec;color:#2c2820;
 font:15px/1.45 system-ui,sans-serif;max-width:760px;margin-inline:auto}
h1{font-size:19px;margin:0 0 6px} h2{font-size:16px;margin:24px 0 2px}
.lede{background:#fff7df;border:1px solid #e8d9a8;border-radius:10px;
 padding:10px 12px;font-size:14px;margin:10px 0}
.verdict{background:#eef4e6;border:1px solid #c6d9ab;border-radius:10px;
 padding:10px 12px;font-size:14px;margin:10px 0}
.note{color:#8a8371;font-size:12.5px}
.strip{display:flex;align-items:flex-end;gap:2px;height:96px;margin:8px 0 4px;
 padding:4px;background:#fff;border:1px solid #e0d8c2;border-radius:8px;
 overflow-x:auto}
.b{flex:0 0 9px;border-radius:2px 2px 0 0;cursor:pointer;position:relative}
.b.kept{background:#1f8a5b}.b.no{background:#d8cfb4}
.b.miss{background:#b3261e}.b.wrong{background:#c9762b}
.b.anchor{background:#2a6fb0}
.leg{font-size:12px;color:#6b654f;margin:2px 0 10px}
.sw{display:inline-block;width:10px;height:10px;border-radius:2px;
 margin:0 4px 0 10px;vertical-align:-1px}
table{border-collapse:collapse;font-size:12.5px;width:100%;display:block;
 overflow-x:auto;margin:6px 0}
td,th{border:1px solid #ddd3b8;padding:3px 7px;white-space:nowrap;text-align:left}
th{background:#f3edda}
"""
JS = """
const au=document.getElementById('au');let stopAt=null,cur=null;
au.addEventListener('timeupdate',()=>{if(stopAt!=null&&au.currentTime>=stopAt){
 au.pause();stopAt=null;if(cur){cur.style.outline='';cur=null;}}});
function play(el,src,t0,t1){
 if(cur===el&&!au.paused){au.pause();return;}
 if(cur)cur.style.outline='';cur=el;el.style.outline='2px solid #8a2b2b';
 const go=()=>{try{au.currentTime=t0;}catch(e){}stopAt=t1;
  au.play().catch(()=>{el.style.outline='';cur=null;});};
 if(au.getAttribute('src')!==src){au.setAttribute('src',src);
  au.addEventListener('loadedmetadata',go,{once:true});au.load();}else go();}
"""


def main(argv):
    nmax = int(argv[argv.index("--songs") + 1]) if "--songs" in argv else 6
    songs = []
    for f in sorted(GT.glob("*.json")):
        gt = json.loads(f.read_text())
        stem = gt["stem"]
        cp = CHARTS / f"min_{stem}.json"
        if not cp.exists() or not (AUDIO / f"{stem}.m4a").exists():
            continue
        chart = json.loads(cp.read_text())
        if len(chart["barGrid"]) - 1 != gt.get("n"):
            continue
        res = analyse(stem, gt, chart)
        if not res:
            continue
        bad = sum(r["n_true"] - r["found"] + r["wrong"] for r in res)
        songs.append({"stem": stem, "title": chart.get("title") or stem,
                      "audio": chart.get("audio_url") or "",
                      "grid": chart["barGrid"], "res": res, "bad": bad})
    songs.sort(key=lambda s: -s["bad"])
    keep = songs[:nmax]

    tot_true = sum(r["n_true"] for s in songs for r in s["res"])
    tot_found = sum(r["found"] for s in songs for r in s["res"])
    tot_wrong = sum(r["wrong"] for s in songs for r in s["res"])
    B = ["<h1>Le score d'appariement, position par position</h1>",
         "<div class=lede>Pour chaque section que tu as annotée, on prend sa "
         "<b>première occurrence comme ancre</b> — exactement ce que fait "
         "l'outil quand ton doigt marque une section — et on affiche le score "
         "de <b>toutes</b> les positions du morceau. Chaque barre est "
         "cliquable : tu entends la position et tu juges le match toi-même."
         "</div>",
         f"<div class=verdict>Sur tes {len(songs)} morceaux annotés : "
         f"<b>{tot_found}/{tot_true}</b> reprises retrouvées, "
         f"<b>{tot_wrong}</b> fausses retenues.<br>"
         "Et sur ta question — « on favorise le scoring de longues sections "
         "même si elles ne sont pas bonnes ? » — mesuré : le score médian "
         "d'un endroit AU HASARD monte bien avec la longueur du bloc "
         "(0,27 pour 2 mesures → 0,40 pour 16), donc oui, un long bloc "
         "ressemble à tout. Mais le <b>pouvoir de séparation</b> ne change "
         "pas : remplacer la moyenne par le minimum ou le premier quartile "
         "donne exactement le même résultat (AUC 0,927 contre 0,927 et "
         "0,926). Le problème n'est donc pas la loi de score.</div>",
         "<div class=leg><span class='sw' style='background:#2a6fb0'></span>"
         "l'ancre<span class='sw' style='background:#1f8a5b'></span>retenue "
         "et juste<span class='sw' style='background:#c9762b'></span>retenue "
         "à tort<span class='sw' style='background:#b3261e'></span>vraie "
         "reprise ratée<span class='sw' style='background:#d8cfb4'></span>"
         "rejetée</div>"]
    for s in keep:
        B.append(f"<h2>{html.escape(s['title'])}</h2>")
        for r in s["res"]:
            B.append(f"<div class=note>section <b>{html.escape(r['letter'])}"
                     f"</b> · ancre mesures {r['b0'] + 1}–{r['b0'] + r['L']} "
                     f"({r['L']} mesures) · {r['found']}/{r['n_true']} "
                     f"reprises retrouvées, {r['wrong']} fausse(s)</div>")
            B.append("<div class=strip>")
            for c in r["cands"]:
                h = max(3, int(round(88 * max(0.0, min(1.0, c["score"])))))
                cls = ("kept" if c["kept"] and c["true"] else
                       "wrong" if c["kept"] else
                       "miss" if c["true"] else "no")
                g = s["grid"]
                t0 = g[c["c"]] if c["c"] < len(g) else 0
                t1 = g[min(len(g) - 1, c["c"] + r["L"])]
                tip = (f"mesure {c['c'] + 1} · score {c['score']:.2f}"
                       + (f" · {c['why']}" if c["why"] else " · retenue"))
                B.append(f"<div class='b {cls}' style='height:{h}px' "
                         f"title=\"{html.escape(tip)}\" "
                         f"onclick=\"play(this,'{s['audio']}',{t0:.2f},"
                         f"{t1:.2f})\"></div>")
            B.append("</div>")
            bad = [c for c in r["cands"]
                   if (c["kept"] and not c["true"]) or
                      (c["true"] and not c["kept"])]
            if bad:
                B.append("<table><tr><th>mesure</th><th>score</th>"
                         "<th>ce qui s'est passé</th></tr>")
                for c in bad[:10]:
                    what = ("retenue à tort" if c["kept"]
                            else f"vraie reprise RATÉE — {c['why']}")
                    B.append(f"<tr><td>{c['c'] + 1}</td>"
                             f"<td>{c['score']:.2f}</td>"
                             f"<td>{what}</td></tr>")
                B.append("</table>")
    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "match_scoring.html"
    out.write_text("<meta charset=utf-8><meta name=viewport "
                   "content='width=device-width,initial-scale=1'>"
                   "<title>Score d'appariement</title><style>" + CSS +
                   "</style><body>" + "".join(B) +
                   "<audio id=au preload=auto playsinline></audio><script>"
                   + JS + "</script>", encoding="utf-8")
    print(f"→ {out}")
    print(f"{tot_found}/{tot_true} reprises retrouvées · {tot_wrong} fausses")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
