"""La page qui montre ce que « Valider les sections » complète tout seul.

Louis, 2026-09-16 : « une fois qu'on a acté les premières sections au doigt et
cliqué sur valider, les sections suivantes devraient automatiquement être
complétées en cherchant le même pattern plusieurs fois dans la chanson », puis
« rappelle-toi que la vérité terrain de l'annotation des sections est
imparfaite, donc surtout regarde si les sections découpées et détectées font
sens ».

D'où cette page : on SIMULE son geste — il marque la première occurrence de
chaque lettre, puis valide — et on montre, bloc par bloc, ce que la machine
en fait, à côté de ce que lui avait écrit. Chaque bloc s'écoute.

    python -m tools.sections_completion_page [--port 7772] [stem …]
"""
from __future__ import annotations

import argparse
import html
import json
import urllib.request
from pathlib import Path

from harmonia.settings import SETTINGS

DEFAUT = ["maroon_5_this_love", "let_it_be_remastered_2009",
          "aretha_franklin_chain_of_fools_official_lyric_video",
          "mayer_hawthorne_the_walk", "T64BgKEL-Sw"]

CSS = """
*{box-sizing:border-box}
body{margin:0;padding:14px 12px 90px;background:#faf6ec;color:#2c2820;
 font:15px/1.45 system-ui,sans-serif;max-width:860px;margin-inline:auto}
h1{font-size:19px;margin:0 0 6px} h2{font-size:16px;margin:24px 0 2px}
.lede{background:#fff7df;border:1px solid #e8d9a8;border-radius:10px;
 padding:10px 12px;font-size:14px;margin:10px 0}
.note{color:#8a8371;font-size:12.5px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:8px 0}
td,th{border:1px solid #ddd3b8;padding:4px 8px;text-align:left;white-space:nowrap}
th{background:#f3edda;font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.ok{background:#eef4e6} .ko{background:#fbeceb}
.play{cursor:pointer;border:1px solid #d8cfb4;border-radius:7px;background:#fff;
 padding:2px 7px;font:600 12px system-ui}
.play.on{background:#8a2b2b;border-color:#8a2b2b;color:#fff}
.src{font:600 10px system-ui;text-transform:uppercase;letter-spacing:.04em;color:#8a8371}
"""
JS = """
const au=document.getElementById('au');let stop=null,cur=null;
au.addEventListener('timeupdate',()=>{if(stop!=null&&au.currentTime>=stop){
 au.pause();stop=null;if(cur){cur.classList.remove('on');cur=null;}}});
function play(el,src,t0,t1){
 if(cur===el&&!au.paused){au.pause();return;}
 if(cur)cur.classList.remove('on');cur=el;el.classList.add('on');
 const go=()=>{try{au.currentTime=t0;}catch(e){}stop=t1;au.play().catch(()=>{});};
 if(au.getAttribute('src')!==src){au.setAttribute('src',src);
  au.addEventListener('loadedmetadata',go,{once:true});au.load();}else go();}
"""


def post(url, body):
    r = urllib.request.Request(url, data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(r, timeout=180) as f:
        return json.loads(f.read())


def une_chanson(stem: str, port: str) -> dict | None:
    sp = SETTINGS.sections_dir / f"{stem}.json"
    cp = SETTINGS.charts_dir / f"min_{stem}.json"
    if not sp.exists() or not cp.exists():
        return None
    d = json.loads(sp.read_text(encoding="utf-8"))
    if not d.get("validated"):
        return None
    chart = json.loads(cp.read_text(encoding="utf-8"))
    grid = chart["barGrid"]
    vt = sorted(d["sections"], key=lambda s: s["b0"])
    vues, humain, donnes = set(), [], set()
    for s in vt:
        if s["label"] in vues:
            continue
        vues.add(s["label"])
        humain.append({"label": s["label"], "mesure_debut": s["b0"] + 1,
                       "mesure_fin": s["b1"] + 1})
        donnes |= set(range(s["b0"], s["b1"] + 1))
    r = post(f"http://127.0.0.1:{port}/api/sections/inferer/min_{stem}",
             {"humain": humain})
    if r.get("error"):
        return None
    sien = {}
    for s in vt:
        for b in range(s["b0"], s["b1"] + 1):
            sien[b] = s["label"]
    blocs = []
    for x in r.get("sections") or []:
        b0, b1 = x["mesure_debut"] - 1, x["mesure_fin"] - 1
        labs = {sien.get(b) for b in range(b0, b1 + 1)} - {None}
        blocs.append({"b0": b0, "b1": b1, "machine": x["label"],
                      "source": x.get("source") or "",
                      "toi": " / ".join(sorted(labs)) if labs else "—",
                      "donne": any(b in donnes for b in range(b0, b1 + 1)),
                      "t0": grid[max(0, min(b0, len(grid) - 2))],
                      "t1": grid[max(1, min(b1 + 1, len(grid) - 1))]})
    return {"stem": stem, "titre": chart.get("title") or stem,
            "audio": chart.get("audio_url") or f"/audio/{stem}.m4a",
            "humain": humain, "blocs": blocs}


def page(songs: list[dict]) -> str:
    B = ["<h1>Ce que « Valider les sections » complète tout seul</h1>",
         "<div class=lede>On simule le geste : Louis marque la PREMIÈRE occurrence "
         "de chaque lettre (les lignes grises), puis valide. Tout le reste est "
         "trouvé par la machine — le découpage par l'algo des quatre mots, le nom "
         "par ressemblance à ses propres blocs (SSM chord-tone). Chaque bloc "
         "s'écoute.<br><b>La colonne « toi » n'est pas un juge</b> : c'est son "
         "annotation, qu'il dit lui-même imparfaite. La question est de savoir si "
         "la découpe et le nom <i>ont du sens</i>, pas s'ils tombent pile.</div>",
         "<p class=note><b>de toi</b> = sa plage, figée · <b>propage</b> = contenu "
         "identique au sien · <b>ressemble</b> = la SSM l'a rapproché d'un de ses "
         "blocs · <b>algo</b> = lettre neuve, il ne ressemble à rien de marqué.</p>"]
    for s in songs:
        B.append(f"<h2>{html.escape(s['titre'])}</h2>")
        B.append("<p class=note>donné à la machine : "
                 + " · ".join(f"{html.escape(h['label'])} mes. {h['mesure_debut']}–{h['mesure_fin']}"
                              for h in s["humain"]) + "</p>")
        B.append("<table><tr><th>mesures</th><th>toi</th><th>machine</th>"
                 "<th>d'où</th><th>écouter</th></tr>")
        for b in s["blocs"]:
            meme = b["toi"] == b["machine"] or b["donne"]
            cls = "" if b["donne"] else (" class=ok" if meme else " class=ko")
            B.append(f"<tr{cls}><td>{b['b0']+1}–{b['b1']+1}</td>"
                     f"<td>{html.escape(b['toi'])}</td>"
                     f"<td><b>{html.escape(b['machine'])}</b></td>"
                     f"<td><span class=src>{html.escape(b['source'])}</span></td>"
                     f"<td><span class=play onclick=\"play(this,'{s['audio']}',"
                     f"{b['t0']:.2f},{b['t1']:.2f})\">▶</span></td></tr>")
        B.append("</table>")
    return ("<meta charset=utf-8><meta name=viewport content='width=device-width,"
            "initial-scale=1'><title>Complétion des sections</title><style>"
            + CSS + "</style><body>" + "".join(B)
            + "<audio id=au preload=auto playsinline></audio><script>"
            + JS + "</script>")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("stems", nargs="*", default=None)
    ap.add_argument("--port", default="7772")
    a = ap.parse_args(argv)
    songs = [x for x in (une_chanson(s, a.port) for s in (a.stems or DEFAUT)) if x]
    out = SETTINGS.reports_dir / "sections_completion.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(page(songs), encoding="utf-8")
    print(f"→ {out} · {len(songs)} morceau(x)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
