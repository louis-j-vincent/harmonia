"""Le chart inféré depuis un tab, à côté du vrai tab — pour que Louis valide.

Louis, 2026-09-18 : « tu me proposes des charts bruts inférés guitar tabs avec
le vrai chart guitar tab à côté, et je valide. Commence par 2 chansons pour
faire un POC ».

La page montre, morceau par morceau :

  * la GRILLE INFÉRÉE — mesure par mesure, l'accord du tab que l'alignement a
    posé là, avec dessous le top-1 de musx quand les deux ne disent pas la
    même chose ; toute mesure s'écoute au doigt ;
  * le TAB D'ORIGINE, tel qu'il est écrit, sections comprises ;
  * ce que la transposition a trouvé, et de combien elle bat le deuxième
    décalage — un écart faible est un signal, pas un détail.

Les verdicts partent tout seuls vers `/api/verdicts/tabs_poc` : plus de
bouton copier, plus de collage dans le fil (Louis, 2026-09-16 : « fais en
sorte que les réponses te remontent automatiquement si tu peux »).

    python -m tools.tabs_poc                       # les deux du POC
    python -m tools.tabs_poc --chart min_xxx --requete "Artiste Titre"
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np

from harmonia.integrations.tab_align import (NOMS, aligner, compresser,
                                             meilleure_transposition, nom_q5,
                                             prior_de_grille, priors_musx,
                                             sequence_du_tab, vraisemblance)
from harmonia.settings import SETTINGS

#: le POC : deux standards pop, un tab unique et très noté chacun
POC = [("min_maroon_5_this_love", "Maroon 5 This Love"),
       ("min_bruno_mars_grenade_official_music_video", "Bruno Mars Grenade")]


def etudier(cle: str, requete: str) -> dict | None:
    from harmonia.integrations.tab_fetcher import fetch_tab_chords, search_tabs

    p = SETTINGS.charts_dir / f"{cle}.json"
    if not p.exists():
        print(f"   {cle} : pas de chart")
        return None
    chart = json.loads(p.read_text(encoding="utf-8"))
    stem = Path(chart.get("audio_url") or "").stem
    audio = SETTINGS.audio_dir / f"{stem}.m4a"
    grille = [float(t) for t in (chart.get("barGrid") or [])]
    if not audio.exists() or len(grille) < 3:
        print(f"   {cle} : pas d'audio ou grille trop courte")
        return None

    res = [r for r in search_tabs(requete, tab_types=("Chords",), max_results=6)
           if r.rating >= 4.5]
    if not res:
        print(f"   {requete} : aucun tab au-dessus de 4,5 étoiles")
        return None
    tab = fetch_tab_chords(res[0])
    if not tab:
        print(f"   {requete} : tab illisible")
        return None

    seq = compresser(sequence_du_tab(tab.raw_content))
    if not seq:
        print(f"   {requete} : aucun accord dans le tab")
        return None
    P = priors_musx(audio, grille)
    dec, scores = meilleure_transposition(seq, P)
    chemin = aligner(seq, P, dec)

    cases = [{"root": (a["root"] + dec) % 12, "q5": a["q5"],
              "texte_tab": a["texte"], "section": a["section"],
              "repetitions": a.get("repetitions", 1)} for a in seq]
    par_mesure, total = [], 0.0
    for b, i in enumerate(chemin):
        c = cases[i]
        lv = vraisemblance(P[b], (c["root"], c["q5"]))
        total += lv
        top = int(np.argmax(P[b]))
        par_mesure.append({
            "case": i, "tab": nom_q5(c["root"], c["q5"]),
            "musx": nom_q5(top // 5, top % 5),
            "p_musx": round(float(P[b][top]), 3),
            "lv": round(lv, 3), "section": c["section"],
            "t0": round(grille[b], 2), "t1": round(grille[b + 1], 2)})

    k = chart.get("key") or {}
    return {
        "cle": cle, "titre": chart.get("title") or stem,
        "audio": chart.get("audio_url") or f"/audio/{stem}.m4a",
        "tonalite": f"{NOMS[int(k.get('tonic') or 0)]} {k.get('mode') or ''}".strip(),
        "tab": {"note": round(res[0].rating, 2), "votes": res[0].votes,
                "tonalite": res[0].tonality, "url": res[0].tab_url,
                "brut": tab.raw_content},
        "decalage": dec,
        "marge": round(scores[0][0] - scores[1][0], 3),
        "scores": [(s, round(v, 3)) for v, s in scores],
        "n_cases": len(seq), "cases_utilisees": len(set(chemin)),
        "prior": {nom_q5(r_, q): round(v, 3) for (r_, q), v
                  in sorted(prior_de_grille(seq)["part"].items(),
                            key=lambda x: -x[1])},
        "mesures": par_mesure,
        "lv_moyenne": round(total / max(1, len(par_mesure)), 3),
        "accord_musx": round(
            float(np.mean([m["tab"] == m["musx"] for m in par_mesure])), 3),
    }


CSS = """
*{box-sizing:border-box}
body{margin:0 auto;padding:14px 12px 40px;background:#faf6ec;color:#2c2820;
 font:15px/1.45 system-ui,-apple-system,sans-serif;max-width:900px}
h1{font-size:19px;margin:0 0 6px}
h2{font-size:16px;margin:24px 0 4px}
.note{color:#8a8371;font-size:12.5px}
.lede{background:#fff7df;border:1px solid #e8d9a8;border-radius:10px;
 padding:10px 12px;font-size:13.5px;margin:10px 0}
.carte{border:1px solid #ddd3b8;border-radius:12px;background:#fffdf7;
 padding:11px 13px;margin:16px 0}
.chiffres{display:flex;gap:14px;flex-wrap:wrap;font-size:12.5px;color:#8a8371;
 margin:4px 0 8px}
.chiffres b{color:#2c2820}
.grille{display:grid;grid-template-columns:repeat(4,1fr);gap:3px}
.mes{position:relative;min-height:52px;border-radius:5px;padding:14px 4px 4px;
 background:#f3edda;cursor:pointer;text-align:center;
 -webkit-tap-highlight-color:transparent}
.mes.dacc{background:#e8f0e4}
.mes .no{position:absolute;top:2px;left:5px;font:600 9.5px system-ui;color:#8a8371}
.mes .t{font:700 15px ui-monospace,monospace;color:#2c2820}
.mes .m{font:600 11px ui-monospace,monospace;color:#8a2b2b;margin-top:1px}
.sec{grid-column:1/-1;font:600 10.5px system-ui;letter-spacing:.06em;
 text-transform:uppercase;color:#8a8371;margin:8px 0 1px}
pre{background:#fffdf7;border:1px solid #ddd3b8;border-radius:10px;padding:10px;
 font:12px/1.5 ui-monospace,monospace;overflow-x:auto;max-height:320px;margin:6px 0}
pre b{color:#8a2b2b;font-weight:700}
.row{display:flex;gap:6px;flex-wrap:wrap;align-items:center;margin-top:8px}
button{border:1px solid #d8cfb4;border-radius:9px;background:#fff;padding:8px 11px;
 font:600 13px system-ui;cursor:pointer;min-height:42px;color:#2c2820}
button.on{background:#2c2820;border-color:#2c2820;color:#fff}
details{margin-top:8px}summary{cursor:pointer;font:600 13px system-ui}
"""

JS = r"""
const au=document.getElementById('au');let stop=null;
au.addEventListener('timeupdate',()=>{if(stop!=null&&au.currentTime>=stop){au.pause();stop=null;}});
function jouer(src,t0,t1){
  const go=()=>{try{au.currentTime=t0;}catch(e){}stop=t1;au.play().catch(()=>{});};
  if(au.getAttribute('src')!==src){au.setAttribute('src',src);
    au.addEventListener('loadedmetadata',go,{once:true});au.load();}
  else go();
}
let V={};
fetch("/api/verdicts/tabs_poc").then(r=>r.json()).then(d=>{V=d.reponses||{};peindre();})
                               .catch(()=>peindre());
function peindre(){
  for(const c of document.querySelectorAll('.carte')){
    const v=V[c.dataset.cle];
    for(const b of c.querySelectorAll('.vd button')) b.classList.toggle('on', b.dataset.v===v);
  }
  const n=Object.keys(V).length;
  document.getElementById('n').textContent=n+" morceau"+(n>1?"x":"")+" jugé"+(n>1?"s":"");
}
function juger(cle,quoi){
  V[cle]=quoi; peindre();
  // les réponses remontent toutes seules : pas de bouton copier, pas de collage
  fetch("/api/verdicts/tabs_poc",{method:"POST",
    headers:{"Content-Type":"application/json"},
    body:JSON.stringify({reponses:V})}).catch(()=>{});
}
"""


def rendre_tab(brut: str) -> str:
    """Le tab tel qu'il est écrit, ses accords en évidence."""
    t = html.escape(brut or "")
    t = t.replace("[ch]", "<b>").replace("[/ch]", "</b>")
    t = t.replace("[tab]", "").replace("[/tab]", "")
    return t


def page(songs: list[dict]) -> str:
    B = ["<h1>Les tabs, alignés sur ce que l'audio dit</h1>",
         "<div class=lede>Le tab dit <b>quoi</b> et <b>dans quel ordre</b> ; "
         "l'audio dit <b>quand</b>. La grille ci-dessous est la suite du tab "
         "posée sur les mesures, à l'alignement qui rend l'audio le plus "
         "vraisemblable.<br>Sous chaque mesure, en rouge, ce que musx entendait "
         "tout seul — affiché <b>seulement quand il n'est pas d'accord</b>. Les "
         "mesures vertes sont celles où les deux disent la même chose.<br>"
         "Touche une mesure pour l'écouter. Tes verdicts remontent tout seuls.</div>",
         "<p class=note><span id=n>aucun morceau jugé</span></p>"]
    for s in songs:
        B.append(f"<div class=carte data-cle=\"{html.escape(s['cle'])}\">")
        B.append(f"<h2>{html.escape(s['titre'])}</h2>")
        B.append(
            "<div class=chiffres>"
            f"<span>tab <b>{s['tab']['note']}★</b> · {s['tab']['votes']} votes</span>"
            f"<span>tonalité du chart <b>{html.escape(s['tonalite'])}</b></span>"
            f"<span>transposition <b>+{s['decalage']} demi-tons</b> "
            f"(marge {s['marge']})</span>"
            f"<span>cases du tab utilisées <b>{s['cases_utilisees']}/{s['n_cases']}</b></span>"
            f"<span>d'accord avec musx <b>{s['accord_musx']:.0%}</b></span>"
            f"<span>log-vraisemblance moyenne <b>{s['lv_moyenne']}</b></span>"
            "</div>")
        B.append("<div class=grille>")
        section = object()
        for i, m in enumerate(s["mesures"]):
            if m["section"] != section:
                section = m["section"]
                if section:
                    B.append(f"<div class=sec>{html.escape(str(section))}</div>")
            dacc = " dacc" if m["tab"] == m["musx"] else ""
            B.append(
                f"<div class='mes{dacc}' onclick=\"jouer('{s['audio']}',"
                f"{m['t0']},{m['t1']})\"><div class=no>{i+1}</div>"
                f"<div class=t>{html.escape(m['tab'])}</div>"
                + ("" if dacc else
                   f"<div class=m>{html.escape(m['musx'])} {m['p_musx']}</div>")
                + "</div>")
        B.append("</div>")
        B.append("<details><summary>le tab d'origine, tel qu'il est écrit</summary>"
                 f"<pre>{rendre_tab(s['tab']['brut'])}</pre></details>")
        B.append("<details><summary>le prior de grille, et les 12 transpositions"
                 "</summary><p class=note>fréquence de chaque accord dans le tab : "
                 + html.escape(json.dumps(s["prior"], ensure_ascii=False))
                 + "<br>score par décalage : "
                 + html.escape(json.dumps(s["scores"])) + "</p></details>")
        cle = html.escape(s["cle"])
        B.append("<div class='row vd'>"
                 f"<button data-v=bon onclick=\"juger('{cle}','bon')\">"
                 "la grille est juste</button>"
                 f"<button data-v=decale onclick=\"juger('{cle}','decale')\">"
                 "bonne grille, mal placée</button>"
                 f"<button data-v=faux onclick=\"juger('{cle}','faux')\">"
                 "la grille est fausse</button>"
                 f"<button data-v='sais pas' onclick=\"juger('{cle}','sais pas')\">"
                 "je ne sais pas</button></div>")
        B.append("</div>")
    return ("<!-- tools/tabs_poc.py -->"
            "<meta charset=utf-8>"
            "<meta name=viewport content='width=device-width,initial-scale=1'>"
            "<title>Tabs alignés — POC</title><style>" + CSS + "</style><body>"
            + "".join(B) + "<audio id=au preload=auto playsinline></audio>"
            "<script>" + JS + "</script>")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--chart", default=None)
    ap.add_argument("--requete", default=None)
    ap.add_argument("--out", type=Path,
                    default=SETTINGS.reports_dir / "tabs_poc.html")
    a = ap.parse_args(argv)
    paires = [(a.chart, a.requete)] if a.chart and a.requete else POC
    songs = []
    for cle, req in paires:
        print(f"→ {req}")
        d = etudier(cle, req)
        if d:
            songs.append(d)
            print(f"   +{d['decalage']} demi-tons (marge {d['marge']}) · "
                  f"{d['cases_utilisees']}/{d['n_cases']} cases · "
                  f"{d['accord_musx']:.0%} d'accord avec musx")
    if not songs:
        print("rien à montrer")
        return 1
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(page(songs), encoding="utf-8")
    print(f"\n→ {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
