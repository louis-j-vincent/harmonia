"""Before/after preview of the three chart fixes — WITHOUT touching prod.

"Avant" is the chart JSON the server is serving right now (produced by the
old code). "Apres" re-runs inference in memory with the fixed code and never
writes to harmonia_min/state. Both are rendered by the app's own renderer.
"""
import json
import sys

sys.path.insert(0, "/Users/vincente/Documents/Projets Perso/Code/harmonia")
from harmonia_min.minimal_view import render_minimal
from harmonia_min.pipeline import analyze

REPO = "/Users/vincente/Documents/Projets Perso/Code/harmonia"
PLOTS = f"{REPO}/docs/plots"
N = ['C', 'Db', 'D', 'Eb', 'E', 'F', 'Gb', 'G', 'Ab', 'A', 'Bb', 'B']

SONGS = [
    ("min_ben_e_king_stand_by_me_audio", "ben_e_king_stand_by_me_audio.m4a",
     "Stand By Me"),
    ("min_maroon_5_this_love", "maroon_5_this_love.m4a", "This Love"),
    ("min_aretha_franklin_chain_of_fools_official_lyric_video",
     "aretha_franklin_chain_of_fools_official_lyric_video.m4a", "Chain Of Fools"),
]


def stats(m):
    ev = [c for s in m["sections"] for occ in s["bars"] for c in occ]
    nc = sum(c.get("nc", False) for c in ev)
    real = sum(1 for c in ev if not c.get("nc") and not c.get("carry"))
    return {"n": len(ev), "nc": nc,
            "pct": round(100 * nc / max(len(ev), 1)), "real": real}


def form(m):
    return " ".join(f"{s['label']}({s['barRanges'][0][1]-s['barRanges'][0][0]+1})"
                    f"{'×' + str(s['reps']) if s['reps'] > 1 else ''}"
                    for s in m["sections"])


parts = []
for key, audio, title in SONGS:
    old = json.load(open(f"{REPO}/harmonia_min/state/charts/{key}.json"))
    new = analyze(f"{REPO}/docs/audio/{audio}",
                  title=old.get("title", title), file_key=key)
    for tag, m in (("avant", old), ("apres", new)):
        open(f"{PLOTS}/deploy_{key}_{tag}.html", "w").write(render_minimal(m))
    so, sn = stats(old), stats(new)
    parts.append(f"""
<section>
<h2>{title}</h2>
<table>
<tr><th></th><th>avant (ce que sert la prod)</th><th>après (code corrigé)</th></tr>
<tr><td>N.C. affichés</td><td class="bad">{so['pct']} %</td><td class="good">{sn['pct']} %</td></tr>
<tr><td>accords réels écrits</td><td>{so['real']}</td><td class="good">{sn['real']}</td></tr>
<tr><td>tonalité</td><td>{old.get('keyName', '?')}</td><td>{new.get('keyName', '?')}</td></tr>
<tr><td>forme</td><td class="sm">{form(old)}</td><td class="sm">{form(new)}</td></tr>
</table>
<div class="cols">
  <div><div class="lab">avant</div>
    <iframe src="deploy_{key}_avant.html" loading="lazy"></iframe></div>
  <div><div class="lab">après</div>
    <iframe src="deploy_{key}_apres.html" loading="lazy"></iframe></div>
</div>
<audio controls preload="none" src="http://100.89.209.63:7772/audio/{audio}"></audio>
</section>""")

html = f"""<!doctype html>
<html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Avant / après — les trois correctifs, avant mise en prod</title>
<style>
:root {{ color-scheme: light dark; }}
* {{ box-sizing: border-box; }}
body {{ margin:0; padding:14px 12px 40px; max-width:900px; margin-inline:auto;
  font-family:-apple-system,"Segoe UI",sans-serif; background:#1e1c1a; color:#ece8e2; }}
@media (prefers-color-scheme: light) {{ body {{ background:#f7f6f3; color:#1a1a1a; }} }}
h1 {{ font-size:1.24rem; margin:4px 0 8px; }}
h2 {{ font-size:1.1rem; margin:0 0 10px; }}
section {{ border-top:1px solid rgba(128,128,128,.3); padding-top:20px; margin-top:26px; }}
.intro {{ font-size:.87rem; line-height:1.55; }}
table {{ width:100%; border-collapse:collapse; font-size:.84rem; margin-bottom:12px; }}
th {{ text-align:left; opacity:.6; font-weight:600; padding:5px 7px; font-size:.78rem; }}
td {{ padding:6px 7px; border-top:1px solid rgba(128,128,128,.22); }}
td.sm {{ font-size:.74rem; opacity:.85; }}
.bad {{ color:#d4453a; font-weight:700; }}
.good {{ color:#3f8f5f; font-weight:700; }}
.cols {{ display:flex; gap:8px; }}
.cols > div {{ flex:1; min-width:0; }}
.lab {{ font-size:.7rem; letter-spacing:.06em; text-transform:uppercase;
  opacity:.55; margin:0 0 4px; font-weight:700; }}
iframe {{ width:100%; height:660px; border:1px solid rgba(128,128,128,.3);
  border-radius:8px; background:#e7e0d0; }}
audio {{ width:100%; margin-top:10px; }}
.box {{ margin-top:14px; padding:11px 13px; border-radius:8px; font-size:.83rem;
  line-height:1.5; background:rgba(196,138,32,.12);
  border:1px solid rgba(196,138,32,.35); }}
</style></head><body>

<h1>Avant / après — avant de toucher à la prod</h1>
<div class="intro">
À gauche, le chart que le serveur sert <b>en ce moment</b> (produit par
l'ancien code, il y a 15 h). À droite, le même morceau ré-inféré <b>en mémoire</b>
avec les trois correctifs — rien n'a été écrit dans l'état de l'app, rien n'a
été redémarré. Les deux sont rendus par le moteur de chart de l'app.
</div>
<div class="box"><b>Les trois correctifs :</b> (1) on ne propage plus les N.C.,
seulement les accords ; (2) la passe écrite pour une section est celle qui
porte le plus d'attaques réelles, plus la première venue ; (3) l'accord posé
sur le bord de la fenêtre du template n'est plus jeté pour 10 ms — c'est le Cm
du refrain de This Love.</div>
{''.join(parts)}
<div class="box" style="margin-top:26px">Si ça te va, je redémarre le serveur
et je relance l'inférence sur les 36 charts (~25 min). Sinon rien ne bouge :
la prod sert toujours l'ancien code.</div>
</body></html>"""

open(f"{PLOTS}/deploy_preview_3songs.html", "w").write(html)
print("wrote", f"{PLOTS}/deploy_preview_3songs.html")
