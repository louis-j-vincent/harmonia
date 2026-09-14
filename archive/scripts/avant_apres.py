"""La page AVANT / APRÈS du changement de loi de repli (2026-08-12).

Louis : « montre-moi des démos de chart avant/après où il y a une vraie
différence notable ».

Deux choses, parce qu'un tableau d'accords n'est pas un chart :
  1. les charts d'AVANT (la sauvegarde `charts.bak_*`) sont republiés dans
     la bibliothèque sous la clé `old_<stem>`, titre « … — AVANT » : ils
     s'ouvrent donc dans l'app elle-même, avec sa vraie mise en page, son
     audio et son transport. `--clean` les retire ;
  2. une page `/reports/avant_apres.html` classe les morceaux par ampleur
     du changement, montre mesure par mesure ce qui a bougé (cliquable pour
     écouter), et donne les deux boutons « ouvrir le chart AVANT » et
     « ouvrir le chart MAINTENANT ».

La comparaison se fait en espace CHANSON : un chart replié écrit un bloc
une fois pour N passages, donc comparer les blocs écrits comparerait des
choses de longueurs différentes. On repasse par `barSpans`, qui donne
l'instant réel de chaque mesure de chaque passage.

    python scripts/avant_apres.py [--backup DIR] [--top 8] [--clean]
"""
from __future__ import annotations

import html
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CHARTS = REPO / "harmonia_min" / "state" / "charts"
REPORTS = REPO / "harmonia_min" / "state" / "reports"
NOTES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]


def chord_txt(c):
    if c.get("nc"):
        return "N"
    t = NOTES[c["root"]] + c["q"]
    if c.get("bass", -1) >= 0 and c["bass"] != c["root"]:
        t += "/" + NOTES[c["bass"]]
    return ("(" + t + ")") if c.get("carry") else t


def song_bars(model):
    """{mesure de chanson: texte des accords} — via les temps, pas les blocs."""
    grid = model.get("barGrid") or []
    n = max(0, len(grid) - 1)
    out = {}
    for sec in model.get("sections") or []:
        for r, bar in enumerate(sec.get("bars") or []):
            spans = (sec.get("barSpans") or [])
            cand = spans[r] if r < len(spans) else []
            for sp in (cand or []):
                if not sp:
                    continue
                lo = None
                for k in range(n):
                    if grid[k] <= sp[0] + 1e-4:
                        lo = k
                if lo is not None and lo not in out:
                    out[lo] = " ".join(chord_txt(c) for c in bar) or "·"
    return out, grid


def diff_song(old, new):
    a, grid = song_bars(old)
    b, _ = song_bars(new)
    rows = []
    for k in sorted(set(a) & set(b)):
        if a[k] != b[k]:
            rows.append((k, a[k], b[k]))
    return rows, grid


CSS = """
*{box-sizing:border-box}
body{margin:0;padding:14px 10px 90px;background:#faf6ec;color:#2c2820;
 font:15px/1.45 system-ui,sans-serif;max-width:760px;margin-inline:auto}
h1{font-size:19px;margin:0 0 6px} h2{font-size:16px;margin:26px 0 2px}
.lede{background:#fff7df;border:1px solid #e8d9a8;border-radius:10px;
 padding:10px 12px;font-size:14px;margin:10px 0}
.note{color:#8a8371;font-size:12.5px}
.row{display:flex;flex-wrap:wrap;gap:4px;margin:4px 0 10px;align-items:center}
.chip{border:1px solid #d8cfb4;border-radius:8px;background:#fff;
 padding:3px 7px;min-width:54px;text-align:center;cursor:pointer}
.chip.on{background:#8a2b2b;border-color:#8a2b2b;color:#fff}
.chip.new{background:#eef4e6;border-color:#a9c488}
.bar{font:600 11px system-ui;color:#8a8371;margin-right:4px}
.open{display:inline-flex;gap:8px;margin:6px 0 2px;flex-wrap:wrap}
.open a{text-decoration:none;font:600 13px system-ui;border-radius:10px;
 padding:9px 13px;border:1px solid #d8cfb4;background:#fff;color:#2c2820}
.open a.now{background:#8a2b2b;border-color:#8a2b2b;color:#fff}
table{border-collapse:collapse;font-size:12.5px;width:100%;display:block;
 overflow-x:auto;margin:8px 0}
td,th{border:1px solid #ddd3b8;padding:3px 7px;white-space:nowrap;text-align:left}
th{background:#f3edda}
"""
JS = """
const au=document.getElementById('au');let stopAt=null,cur=null;
au.addEventListener('timeupdate',()=>{if(stopAt!=null&&au.currentTime>=stopAt){
 au.pause();stopAt=null;if(cur){cur.classList.remove('on');cur=null;}}});
function play(el,src,t0,t1){
 if(cur===el&&!au.paused){au.pause();return;}
 if(cur)cur.classList.remove('on');cur=el;el.classList.add('on');
 const go=()=>{try{au.currentTime=t0;}catch(e){}stopAt=t1;
  au.play().catch(()=>{el.classList.remove('on');cur=null;});};
 if(au.getAttribute('src')!==src){au.setAttribute('src',src);
  au.addEventListener('loadedmetadata',go,{once:true});au.load();}else go();}
"""


def main(argv):
    backup = Path(argv[argv.index("--backup") + 1]) if "--backup" in argv else \
        CHARTS.parent / "charts.bak_20260812"
    top = int(argv[argv.index("--top") + 1]) if "--top" in argv else 8
    if "--clean" in argv:
        n = 0
        for p in CHARTS.glob("old_*.json"):
            p.unlink()
            n += 1
        print(f"{n} chart(s) « AVANT » retiré(s) de la bibliothèque")
        return 0
    if not backup.is_dir():
        print(f"sauvegarde introuvable : {backup}")
        return 2

    songs = []
    for p in sorted(CHARTS.glob("min_*.json")):
        q = backup / p.name
        if not q.exists():
            continue
        new = json.loads(p.read_text(encoding="utf-8"))
        old = json.loads(q.read_text(encoding="utf-8"))
        rows, grid = diff_song(old, new)
        if not rows:
            continue
        songs.append({"key": p.stem, "title": new.get("title") or p.stem,
                      "audio": new.get("audio_url") or "", "rows": rows,
                      "grid": grid, "old": old})
    songs.sort(key=lambda s: -len(s["rows"]))
    keep = songs[:top]

    published = 0
    for s in keep:
        stem = s["key"].removeprefix("min_")
        doc = dict(s["old"])
        doc["file"] = f"old_{stem}"
        doc["title"] = (s["title"] or stem) + " — AVANT"
        (CHARTS / f"old_{stem}.json").write_text(
            json.dumps(doc, ensure_ascii=False), encoding="utf-8")
        published += 1

    B = ["<h1>Avant / après : le repli par CQT moyenné</h1>",
         "<div class=lede>Les charts d'<b>avant</b> sont republiés dans ta "
         "bibliothèque sous le titre « … — AVANT » : les deux versions "
         "s'ouvrent donc dans l'app, avec sa vraie mise en page et son audio. "
         "Ci-dessous, mesure par mesure, ce qui a changé — clique pour "
         "entendre le passage.</div>",
         f"<p class=note>{len(songs)} morceaux ont changé ; voici les "
         f"{len(keep)} où l'écart est le plus large. "
         "<code>python scripts/avant_apres.py --clean</code> retire les "
         "entrées « AVANT » de la bibliothèque.</p>",
         "<table><tr><th>morceau</th><th>mesures changées</th></tr>"]
    for s in songs:
        B.append(f"<tr><td>{html.escape(s['title'][:36])}</td>"
                 f"<td>{len(s['rows'])}</td></tr>")
    B.append("</table>")

    for s in keep:
        stem = s["key"].removeprefix("min_")
        B.append(f"<h2>{html.escape(s['title'])}</h2>")
        B.append(f"<div class=open>"
                 f"<a href='/?open=old_{stem}'>ouvrir le chart AVANT</a>"
                 f"<a class=now href='/?open={s['key']}'>ouvrir le chart "
                 f"MAINTENANT</a></div>")
        B.append(f"<p class=note>{len(s['rows'])} mesures changent</p>")
        for bar, a, b in s["rows"][:24]:
            g = s["grid"]
            t0 = g[bar] if bar < len(g) else 0
            t1 = g[bar + 1] if bar + 1 < len(g) else t0 + 2
            B.append(f"<div class=row><span class=bar>mes. {bar + 1}</span>"
                     f"<div class=chip onclick=\"play(this,'{s['audio']}',"
                     f"{t0:.2f},{t1:.2f})\">{html.escape(a)}</div>"
                     f"<div class='chip new' onclick=\"play(this,"
                     f"'{s['audio']}',{t0:.2f},{t1:.2f})\">"
                     f"{html.escape(b)}</div></div>")
        if len(s["rows"]) > 24:
            B.append(f"<p class=note>… et {len(s['rows']) - 24} autres "
                     "mesures</p>")

    REPORTS.mkdir(parents=True, exist_ok=True)
    out = REPORTS / "avant_apres.html"
    out.write_text("<meta charset=utf-8><meta name=viewport "
                   "content='width=device-width,initial-scale=1'>"
                   "<title>Avant / après</title><style>" + CSS + "</style>"
                   "<body>" + "".join(B) +
                   "<audio id=au preload=auto playsinline></audio>"
                   "<script>" + JS + "</script>", encoding="utf-8")
    print(f"→ {out}\n{published} chart(s) « AVANT » publiés dans la "
          f"bibliothèque · {len(songs)} morceaux changés au total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
