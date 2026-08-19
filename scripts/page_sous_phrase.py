"""docs/plots/sous_phrase.html — empiler la sous-phrase, ou la section entière ?

Louis, 2026-08-19 : « quand la section elle-même est faite de plusieurs
répétitions d'une sous-phrase, tu peux les empiler ceux-là aussi pour avoir
encore + d'observations ».

C'est déjà ce que `folding` essaie EN PREMIER (`section_period` cherche une
boucle interne de 2, 4 ou 8 mesures ; à défaut seulement, `loop="occurrence"`
empile les passages entre eux). Mesuré sur la bibliothèque : **90 sections sur
193** trouvent une sous-phrase, et le gain en observations est énorme — This
Love B passe de 5 à 20 observations par position, Billie Jean A de 11 à 40.

Reste la question que les chiffres ne tranchent pas : ces observations en plus
sont-elles les BONNES ? Empiler à P=2, c'est affirmer que les mesures 1, 3, 5, 7
portent le même accord — vrai sur une pompe, faux dès qu'une cadence change au
dernier tour. Cette page met les deux empilements l'un sous l'autre.
"""
import html, json, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harmonia_min import folding as F, musx as M          # noqa: E402
from harmonia_min.nnls_features import extract_bothchroma  # noqa: E402
from harmonia_min.sections import halfbar_features         # noqa: E402

REPO = Path(__file__).resolve().parents[1]
N = ["C", "D♭", "D", "E♭", "E", "F", "G♭", "G", "A♭", "A", "B♭", "B"]

CIBLES = [("maroon_5_this_love", "B"),
          ("michael_jackson_billie_jean_official_video", "A"),
          ("mayer_hawthorne_the_walk", "A"),
          ("maroon_5_she_will_be_loved_official_music_video", "A"),
          ("katy_perry_hot_n_cold_official_music_video", "A"),
          ("sam_smith_i_m_not_the_only_one_official_music_video", "A")]


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


def charge(stem):
    for f in (REPO / "harmonia_min/state/charts").glob("min_*.json"):
        ch = json.loads(f.read_text())
        if (Path(ch.get("audio_url") or "").stem or "") == stem:
            return ch, f.stem
    return None, None


meta = json.loads((REPO / "harmonia_min/state/chart_meta.json").read_text())
songs = []
for stem, lettre in CIBLES:
    ch, key = charge(stem)
    if ch is None:
        print(f"({stem}: pas de chart)")
        continue
    grid, bpb = ch["barGrid"], ch["bpb"]
    sec = next((s for s in ch["sections"] if s["label"] == lettre), None)
    if sec is None:
        continue
    occ, L = sec["barRanges"], len(sec["bars"])
    audio = REPO / "docs/audio" / f"{stem}.m4a"
    probs = M.frame_posteriors(audio)
    cqt = M.song_cqt(audio)
    arr, times = extract_bothchroma(audio)
    Vb = F._bar_vecs(halfbar_features(grid, arr, times), len(grid) - 1)
    Ps = [p for p in (F.section_period(Vb, b0, b1)[0] for b0, b1 in occ) if p]
    if not Ps:
        print(f"({stem} {lettre}: pas de sous-phrase)")
        continue
    P_int = int(np.bincount(Ps).argmax())
    Lf = max(bpb, int(round(float(np.median(np.diff(grid))) / M.FRAME_DT)))

    def bar_probs(b):
        a = max(0, int(round(grid[b] / M.FRAME_DT)))
        z = min(probs[0].shape[0], int(round(grid[b + 1] / M.FRAME_DT)))
        return [p[a:z] for p in probs]

    def bar_cqt(b):
        a = max(0, int(round(grid[b] / M.FRAME_DT)))
        z = min(cqt.shape[0], int(round(grid[b + 1] / M.FRAME_DT)))
        return cqt[a:z]

    def decode(P):
        pos = [[] for _ in range(P)]
        for b0, b1 in occ:
            for b in range(b0, b1 + 1):
                pos[(b - b0) % P].append(b)
        # loi de prod depuis le 2026-08-19 : on empile les POSTÉRIEURES.
        pc = F._template_chords(pos, bar_probs, len(probs), Lf, bpb, P,
                                combine="mean")
        return pc, (len(pos[0]) if pos else 0)

    pc_i, n_i = decode(P_int)
    pc_o, n_o = decode(L)
    if pc_i is None or pc_o is None:
        print(f"({stem} {lettre}: gabarit vide)")
        continue
    bars_i = [txt(pc_i[k % P_int]) for k in range(L)]
    bars_o = [txt(pc_o[k]) for k in range(L)]
    m = meta.get(key) or {}
    songs.append({"stem": stem, "lettre": lettre, "L": L, "P": P_int,
                  "n_i": n_i, "n_o": n_o, "occ": occ,
                  "titre": " — ".join(x for x in (m.get("artist"),
                                                  m.get("title") or stem) if x),
                  "sous": bars_i, "sect": bars_o,
                  "temps": [float(grid[occ[0][0] + k]) for k in range(L)],
                  "autres": [float(grid[o[0]]) for o in occ],
                  "diff": sum(1 for k in range(L) if bars_i[k] != bars_o[k])})
    print(f"{stem} {lettre}: sous-phrase P={P_int} ({n_i} obs/pos) contre "
          f"section P={L} ({n_o} obs/pos) — {songs[-1]['diff']}/{L} mesures diffèrent")


def cell(s, cle, ref):
    o = []
    for k in range(s["L"]):
        d = " d" if s[cle][k] != s[ref][k] else ""
        o.append(f'<div class="m{d}" data-t="{s["temps"][k]:.2f}">'
                 f'<span class="n">{k + 1}</span>{html.escape(s[cle][k])}</div>')
    return "".join(o)


blocs = []
for s in songs:
    liens = " ".join(f'<button data-t="{t:.2f}">passage {i + 1}</button>'
                     for i, t in enumerate(s["autres"]))
    blocs.append(f"""
<section>
  <h2>{html.escape(s['titre'])} <span class="sec">section {s['lettre']}
      · {s['L']} mesures × {len(s['occ'])} passages</span></h2>
  <audio controls preload="none" src="/audio/{s['stem']}.m4a"></audio>
  <p class="lead">{liens}<br>{s['diff']} mesure(s) sur {s['L']} diffèrent.</p>
  <h3>par sous-phrase <small>boucle de {s['P']} mesures → <b>{s['n_i']} observations</b>
      par position</small></h3>
  <div class="g">{cell(s, 'sous', 'sect')}</div>
  <h3>par section entière <small>{s['L']} mesures → {s['n_o']} observations
      par position</small></h3>
  <div class="g">{cell(s, 'sect', 'sous')}</div>
</section>""")

page = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Empiler la sous-phrase</title><style>
:root{{--pap:#f7f3ea;--ink:#1c1c1c;--f:#8a7f6d;--l:#ded5c4;--acc:#8a2b2b;--d:#f2dcc4}}
body{{margin:0 auto;padding:18px 14px 60px;background:var(--pap);color:var(--ink);
 font:15px/1.55 -apple-system,system-ui,sans-serif;max-width:900px}}
h1{{font:700 22px/1.3 Georgia,serif;margin:0 0 4px}}
h2{{font:700 17px/1.35 Georgia,serif;margin:34px 0 8px;border-top:1px solid var(--l);padding-top:18px}}
h2 .sec{{font:600 12px -apple-system,sans-serif;color:var(--f);display:block;margin-top:3px}}
h3{{font:600 13px/1.4 -apple-system,sans-serif;margin:16px 0 6px;text-transform:uppercase;
 letter-spacing:.06em;color:var(--f)}}
h3 small{{text-transform:none;letter-spacing:0;font-weight:500;display:block;margin-top:2px}}
h3 small b{{color:var(--acc)}}
p.lead{{color:var(--f);font-size:13.5px;margin:6px 0 10px}}
audio{{width:100%;margin:2px 0 6px}}
button{{background:#fff;border:1px solid var(--l);border-radius:6px;padding:3px 9px;
 font:600 12px -apple-system,sans-serif;color:var(--ink);margin:0 3px 3px 0}}
.g{{display:grid;grid-template-columns:repeat(4,1fr);gap:3px}}
.m{{position:relative;background:#fff;border:1px solid var(--l);border-radius:6px;
 padding:12px 4px 7px;text-align:center;font:600 15px Georgia,serif;cursor:pointer;min-height:20px}}
.m .n{{position:absolute;top:2px;left:4px;font:500 8px -apple-system,sans-serif;color:#c3b8a4}}
.m.d{{background:var(--d);border-color:#e0bf95}}
.intro{{background:#fff;border:1px solid var(--l);border-radius:10px;padding:14px 16px;margin:14px 0}}
.intro b{{color:var(--acc)}}
@media(min-width:620px){{.g{{grid-template-columns:repeat(8,1fr)}}}}
</style></head><body>
<h1>Empiler la sous-phrase</h1>
<div class="intro">
<p>Une section revient 5 fois, et à l'intérieur elle répète une pompe de
2 mesures : ça ne fait pas 5 observations par position, ça en fait
<b>20</b>. C'est déjà ce que le repli essaie en premier — mesuré sur la
bibliothèque, <b>90 sections sur 193</b> trouvent une sous-phrase.</p>
<p>Ce que les chiffres ne disent pas : si ces observations en plus sont les
BONNES. Empiler à P=2, c'est affirmer que les mesures 1, 3, 5 et 7 portent le
même accord — vrai sur une pompe, faux dès qu'une cadence change au dernier
tour. Les deux empilements sont ci-dessous ; les mesures colorées sont celles
où ils ne disent pas la même chose.</p>
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
(REPO / "docs/plots/sous_phrase.html").write_text(page, encoding="utf-8")
print("→ docs/plots/sous_phrase.html")
