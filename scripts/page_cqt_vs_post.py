"""docs/plots/cqt_vs_post.html — empiler des spectres ou des postérieures ?

Louis, 2026-08-19 : « même sans modulation, je pense qu'il vaudrait mieux
empiler les postérieures que les CQT non ? fais-moi un exemple pour tester ».

Le 2026-08-08 il avait tranché pour le CQT (« les CQT moyennés ça marche très
bien »). Le 2026-08-19, sur la section qui MODULE de Bora Bora, il a tranché
pour les postérieures. Cette page pose la question sur le cas restant : les
sections répétées qui ne modulent PAS.

Une ligne par loi, la même section, le même décodeur
(`folding._decode_template`), les mesures divergentes colorées, tout cliquable.
On ne garde que les sections où les deux lois écrivent quelque chose de
DIFFÉRENT — ailleurs il n'y a rien à arbitrer.
"""
import html, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harmonia_min import folding as F, musx as M    # noqa: E402
from harmonia_min.labels import to_chord            # noqa: E402
from harmonia_min.nnls_features import extract_bothchroma   # noqa: E402
from harmonia_min.sections import halfbar_features          # noqa: E402

REPO = Path(__file__).resolve().parents[1]
N = ["C", "D♭", "D", "E♭", "E", "F", "G♭", "G", "A♭", "A", "B♭", "B"]
MAX_SECTIONS = 8


def txt(chords):
    o = []
    for c in chords or []:
        if c.get("carry"):
            o.append("%")
        elif c["nc"]:
            o.append("N.C.")
        else:
            s = f"{N[c['root']]}{c['q']}"
            if c.get("bass", -1) >= 0 and c["bass"] != c["root"]:
                s += f"/{N[c['bass']]}"
            o.append(s)
    return " ".join(o) or "%"


def deux_lois(chart, stem):
    """→ [(lettre, P, occ, bars_cqt, bars_post, temps)] pour ce morceau."""
    audio = REPO / "docs" / "audio" / f"{stem}.m4a"
    if not audio.exists():
        return []
    grid, bpb = chart["barGrid"], chart["bpb"]
    cibles = [s for s in chart["sections"]
              if len(s["barRanges"]) >= 2 and len(s["bars"]) >= 4
              and len({b1 - b0 for b0, b1 in s["barRanges"]}) == 1]
    if not cibles:
        return []
    probs = M.frame_posteriors(audio)
    cqt = M.song_cqt(audio)
    Lf = max(bpb, int(round(float(np.median(np.diff(grid))) / M.FRAME_DT)))

    def bar_probs(b):
        a = max(0, int(round(grid[b] / M.FRAME_DT)))
        z = min(probs[0].shape[0], int(round(grid[b + 1] / M.FRAME_DT)))
        return [p[a:z] for p in probs]

    def bar_cqt(b):
        a = max(0, int(round(grid[b] / M.FRAME_DT)))
        z = min(cqt.shape[0], int(round(grid[b + 1] / M.FRAME_DT)))
        return cqt[a:z]

    # la modulation est le cas DÉJÀ tranché : on l'écarte de cette page
    arr, times = extract_bothchroma(audio)
    Vb = F._bar_vecs(halfbar_features(grid, arr, times), len(grid) - 1)

    out = []
    for s in cibles:
        occ = s["barRanges"]
        P = len(s["bars"])
        if any(F.decalage_semitons(Vb, occ[0][0], o[0], P)[0] for o in occ[1:]):
            continue                       # ce morceau module : hors sujet ici
        pos = [[o[0] + k for o in occ if o[0] + k <= o[1]] for k in range(P)]
        cat_c = [np.asarray(x, dtype=np.float64) for x in
                 M.posteriors_from_cqt(F._cqt_template(pos, bar_cqt, Lf, P))]
        pc_c = F._decode_template(cat_c, pos, bar_probs, Lf, bpb, P)
        pc_p = F._template_chords(pos, bar_probs, len(probs), Lf, bpb, P,
                                  combine="mean")
        if pc_c is None or pc_p is None:
            continue
        bc = [txt(pc_c[k]) for k in range(P)]
        bp = [txt(pc_p[k]) for k in range(P)]
        if bc == bp:
            continue                       # rien à arbitrer
        out.append({"lettre": s["label"], "P": P, "occ": occ,
                    "cqt": bc, "post": bp,
                    "diff": sum(1 for k in range(P) if bc[k] != bp[k]),
                    "temps": [float(grid[occ[0][0] + k]) for k in range(P)],
                    "autres": [float(grid[o[0]]) for o in occ]})
    return out


meta = json.loads((REPO / "harmonia_min/state/chart_meta.json").read_text()) \
    if (REPO / "harmonia_min/state/chart_meta.json").exists() else {}
trouvees = []
for f in sorted((REPO / "harmonia_min/state/charts").glob("min_*.json")):
    if f.stem.endswith("_pile"):
        continue
    try:
        chart = json.loads(f.read_text())
        stem = Path(chart.get("audio_url") or "").stem or f.stem.removeprefix("min_")
        secs = deux_lois(chart, stem)
    except Exception as exc:                      # noqa: BLE001
        print(f"  ({f.stem} : {type(exc).__name__} {exc})")
        continue
    for sec in secs:
        m = meta.get(f.stem) or {}
        titre = (m.get("title") or chart.get("title") or stem)
        artiste = m.get("artist") or ""
        sec.update(stem=stem, titre=(f"{artiste} — {titre}" if artiste else titre))
        trouvees.append(sec)
    if secs:
        print(f"{stem}: " + ", ".join(f"{s['lettre']}×{len(s['occ'])} "
                                      f"{s['diff']}/{s['P']} mesures diffèrent"
                                      for s in secs))

# Choix des exemples : des morceaux que Louis a DÉJÀ travaillés, pas les huit
# plus gros écarts de la bibliothèque (qui sont surtout des identifiants
# YouTube qu'il n'a jamais ouverts). Règle du projet : illustrer sur ce qu'il
# connaît, sinon il ne peut pas arbitrer.
CONNUS = ["maroon_5_this_love", "norah_jones_don_t_know_why",
          "the_ronettes_be_my_baby_music_video", "bobby_hebb_sunny_official_audio",
          "ray_charles_georgia_on_my_mind_official_video",
          "elton_john_goodbye_yellow_brick_road_lyrics",
          "maroon_5_she_will_be_loved_official_music_video",
          "michael_jackson_billie_jean_official_video", "bein_green",
          "yesterday_remastered_2009", "mayer_hawthorne_the_walk",
          "norah_jones_come_away_with_me"]
rang = {st: i for i, st in enumerate(CONNUS)}
trouvees.sort(key=lambda s: (rang.get(s["stem"], 99), -s["diff"] / s["P"]))
gardees, vus = [], set()
for s in trouvees:                       # une seule section par morceau
    if s["stem"] in vus or len(gardees) >= MAX_SECTIONS:
        continue
    gardees.append(s); vus.add(s["stem"])
print(f"\n{len(trouvees)} sections où les deux lois divergent ; "
      f"page bâtie sur {len(gardees)} morceaux connus")


def cellules(s, cle, ref):
    o = []
    for k in range(s["P"]):
        d = " d" if s[cle][k] != s[ref][k] else ""
        o.append(f'<div class="m{d}" data-t="{s["temps"][k]:.2f}">'
                 f'<span class="n">{k + 1}</span>{html.escape(s[cle][k])}</div>')
    return "".join(o)


blocs = []
for s in gardees:
    liens = " ".join(f'<button data-t="{t:.2f}">occ. {i + 1}</button>'
                     for i, t in enumerate(s["autres"]))
    blocs.append(f"""
<section>
  <h2>{html.escape(s['titre'])} <span class="sec">section {html.escape(s['lettre'])}
      · {s['P']} mesures × {len(s['occ'])} passages</span></h2>
  <audio controls preload="none" src="/audio/{s['stem']}.m4a"></audio>
  <p class="lead">Aller à : {liens} · clique une mesure pour l'entendre
     (1ᵉʳ passage). {s['diff']} mesure(s) sur {s['P']} diffèrent.</p>
  <h3>CQT empilés <small>la prod : on moyenne les spectres, puis musx écoute</small></h3>
  <div class="g">{cellules(s, 'cqt', 'post')}</div>
  <h3>postérieures empilées <small>musx écoute chaque passage, on moyenne ce qu'il a compris</small></h3>
  <div class="g">{cellules(s, 'post', 'cqt')}</div>
</section>""")

page = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Empiler des spectres ou des postérieures ?</title><style>
:root{{--pap:#f7f3ea;--ink:#1c1c1c;--f:#8a7f6d;--l:#ded5c4;--acc:#8a2b2b;--d:#f2dcc4}}
body{{margin:0 auto;padding:18px 14px 60px;background:var(--pap);color:var(--ink);
 font:15px/1.55 -apple-system,system-ui,sans-serif;max-width:900px}}
h1{{font:700 22px/1.3 Georgia,serif;margin:0 0 4px}}
h2{{font:700 17px/1.35 Georgia,serif;margin:34px 0 8px;border-top:1px solid var(--l);padding-top:18px}}
h2 .sec{{font:600 12px -apple-system,sans-serif;color:var(--f);display:block;margin-top:3px}}
h3{{font:600 13px/1.4 -apple-system,sans-serif;margin:16px 0 6px;text-transform:uppercase;
 letter-spacing:.06em;color:var(--f)}}
h3 small{{text-transform:none;letter-spacing:0;font-weight:500;display:block;margin-top:2px}}
p.lead{{color:var(--f);font-size:13.5px;margin:6px 0 10px}}
audio{{width:100%;margin:2px 0 6px}}
button{{background:#fff;border:1px solid var(--l);border-radius:6px;padding:3px 9px;
 font:600 12px -apple-system,sans-serif;color:var(--ink);margin-right:3px}}
.g{{display:grid;grid-template-columns:repeat(4,1fr);gap:3px}}
.m{{position:relative;background:#fff;border:1px solid var(--l);border-radius:6px;
 padding:12px 4px 7px;text-align:center;font:600 15px Georgia,serif;cursor:pointer;min-height:20px}}
.m .n{{position:absolute;top:2px;left:4px;font:500 8px -apple-system,sans-serif;color:#c3b8a4}}
.m.d{{background:var(--d);border-color:#e0bf95}}
.m:active{{outline:2px solid var(--acc)}}
.intro{{background:#fff;border:1px solid var(--l);border-radius:10px;padding:14px 16px;margin:14px 0}}
.intro b{{color:var(--acc)}}
@media(min-width:620px){{.g{{grid-template-columns:repeat(8,1fr)}}}}
</style></head><body>
<h1>Empiler des spectres, ou des postérieures ?</h1>
<div class="intro">
<p>Quand une section revient N fois, on additionne les N passages avant de
décider. Deux endroits possibles : <b>avant</b> le modèle — on moyenne les
spectres et musx écoute la moyenne, c'est la prod depuis le 8 août — ou
<b>après</b> — musx écoute chaque passage séparément et on moyenne ce qu'il a
compris.</p>
<p>Le cas qui module est déjà tranché (postérieures, Bora Bora, 19 août). Ici
ce sont les sections qui <b>ne modulent pas</b>, et seulement celles où les deux
lois écrivent quelque chose de différent : <b>{len(trouvees)} sections</b> dans
la bibliothèque. Ci-dessous {len(gardees)} morceaux que tu as déjà travaillés,
la section la plus contrastée de chacun.</p>
<p>Les mesures colorées sont celles qui changent. Écoute et dis laquelle a
raison.</p>
</div>
{"".join(blocs)}
<script>
document.querySelectorAll('section').forEach(sec=>{{
  const a=sec.querySelector('audio');
  const go=t=>{{a.currentTime=parseFloat(t)||0;a.play();}};
  sec.querySelectorAll('.m').forEach(m=>m.addEventListener('click',()=>go(m.dataset.t)));
  sec.querySelectorAll('button').forEach(b=>b.addEventListener('click',()=>go(b.dataset.t)));
}});
</script></body></html>"""
(REPO / "docs/plots/cqt_vs_post.html").write_text(page, encoding="utf-8")
print("→ docs/plots/cqt_vs_post.html")
