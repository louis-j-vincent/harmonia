"""Le tab contre nous, à jouer sur le téléphone.

    python scripts/tab_vs_us.py   ->  /reports/tab_vs_us.html

Louis, 2026-08-05 : « Montres moi les charts que tu infères sur quelques sons en
partant de guitartabs et en l'alignant, rendu harmonia pour que je puisse play
et comparer les sections », puis « donne moi une version cliquable iPhone ».

Une carte par morceau, deux gros boutons : jouer le chart tiré du tab, jouer le
nôtre. Les deux charts partagent l'audio ET la grille de mesures (les charts
`__ug` sont bâtis sur la grille d'un vrai passage de `pipeline.analyze()`), donc
la tête de lecture tombe au même endroit dans les deux et la comparaison porte
sur les sections, pas sur le calage.

Les charts `min_<stem>__ug.json` sont produits par `scripts/ug_reference_align.py`
(tabs Ultimate Guitar >= 4,7*, alignés par isolation de voix + transcription
horodatée au mot + alignement forcé sur les paroles). Cette page ne fait que les
présenter ; la page d'analyse est `/reports/ug_reference.html`.
"""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CHARTS = HERE / "harmonia_min/state/charts"
OUT = HERE / "harmonia_min/state/reports/tab_vs_us.html"

# Réserves d'alignement relevées à la main en lisant les charts — elles vivent
# ici plutôt que dans la tête de celui qui lit la page.
CAVEATS = ("Réserves d'alignement : le tab de This Love est écrit à deux accords "
           "par mesure là où nous en écrivons un ; nos 8 premières mesures de "
           "Stand By Me sont N.C. (intro de basse) alors que le tab démarre le "
           "couplet mesure 1 ; le dernier « refrain » de Be My Baby fait 44 "
           "mesures, l'alignement décroche vers la fin.")


def main():
    rows = []
    for f in sorted(glob.glob(str(CHARTS / "*__ug.json"))):
        stem = os.path.basename(f)[4:-len("__ug.json")]
        ours = CHARTS / f"min_{stem}.json"
        if not ours.exists():
            print(f"  !! pas de chart à nous pour {stem}")
            continue
        m, o = json.load(open(f)), json.load(open(ours))
        types = {s["label"] for s in m["sections"] if "hors" not in s["label"]}
        rows.append(dict(
            stem=stem, title=m["title"].replace(" — TAB", ""), nbars=m["nBars"],
            tab=" · ".join(f"{s['label']} ({len(s['bars'])})" for s in m["sections"]),
            ntab=len(types),
            us=" · ".join(f"{s['label']}×{s['reps']} ({len(s['bars'])} mes)"
                          for s in o["sections"]),
            nus=len({s["label"] for s in o["sections"]})))

    cards = ""
    for r in rows:
        same = r["ntab"] == r["nus"]
        cards += f"""<div class=song>
 <div class=hd><b>{r['title']}</b><span class=n>{r['nbars']} mesures</span></div>
 <div class="row {'ok' if same else 'gap'}">
   <span class=k>tab</span> <span class=v>{r['ntab']} types</span>
   <span class=k>nous</span> <span class=v>{r['nus']} lettres</span></div>
 <div class=det><b>tab :</b> {r['tab']}</div>
 <div class=det><b>nous :</b> {r['us']}</div>
 <div class=btns>
   <a class="b tab" href="/?open=min_{r['stem']}__ug">▶ jouer le chart du TAB</a>
   <a class="b us" href="/?open=min_{r['stem']}">▶ jouer le NÔTRE</a></div>
</div>"""

    OUT.write_text(f"""<!DOCTYPE html>
<html lang=fr><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1,viewport-fit=cover">
<title>Tab contre nous — à jouer</title><style>
*{{box-sizing:border-box;-webkit-tap-highlight-color:transparent}}
body{{margin:0;background:#e7e0d0;font:16px/1.5 -apple-system,system-ui,sans-serif;color:#1c1c1c}}
.wrap{{max-width:660px;margin:0 auto;padding:20px 14px calc(40px + env(safe-area-inset-bottom))}}
h1{{font:italic 600 24px Georgia,serif;margin:0 0 2px}}
.lede{{color:#8a8371;font-size:13px;margin-bottom:18px}}
.song{{background:#fffdf6;border:1px solid #e5dcc6;border-radius:14px;padding:13px 14px;margin-bottom:11px}}
.hd{{font:600 16px system-ui;display:flex;justify-content:space-between;align-items:baseline}}
.n{{font:500 11.5px system-ui;color:#8a8371}}
.row{{font:600 12px system-ui;margin:7px 0;padding:5px 8px;border-radius:8px}}
.row.ok{{background:#dff0e6;color:#0f5132}} .row.gap{{background:#f7e7e7;color:#8a2b2b}}
.k{{opacity:.6;margin-right:3px}} .v{{margin-right:14px}}
.det{{font:500 11.5px/1.45 system-ui;color:#8a8371;margin:3px 0}}
.det b{{color:#1c1c1c}}
.btns{{display:flex;gap:8px;margin-top:10px}}
.b{{flex:1;text-align:center;text-decoration:none;border-radius:10px;padding:12px 8px;
    font:700 13px system-ui;color:#fff}}
.b.tab{{background:#2a6fb0}} .b.us{{background:#8a2b2b}}
.b:active{{opacity:.75}}
.foot{{font-size:12px;color:#8a8371;margin-top:18px}}
a.more{{display:block;text-align:center;margin-top:14px;font:600 13px system-ui;color:#2a6fb0}}
</style></head><body><div class=wrap>
<h1>Le tab contre nous</h1>
<div class=lede>Même audio, même grille de mesures — la tête de lecture tombe au même endroit
dans les deux. Le chart bleu vient d'un tab Ultimate Guitar noté ≥ 4,7★, le rouge est ce que
nous inférons. Vert = on trouve autant de sections que le tab.
<br><br><b>Attention à ce que compare le bandeau :</b> le tab nomme les sections par la FORME
de la chanson (couplet, refrain), nous par l'HARMONIE. Ce ne sont pas la même grandeur — sur
11 morceaux, 11 paires de sections que le tab nomme différemment sont la même harmonie mesure
pour mesure. Toi-même tu as demandé 2 sections sur The Walk là où le tab en nomme 4.</div>
{cards}
<a class=more href="/reports/ug_reference.html">L'analyse complète, morceau par morceau →</a>
<div class=foot>{CAVEATS}</div>
</div></body></html>""")
    print(f"wrote {OUT.relative_to(HERE)} ({len(rows)} morceaux, "
          f"{OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
