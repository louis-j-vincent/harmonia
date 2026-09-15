"""docs/plots/folds.html — cinq façons de combiner les 5 réseaux musx.

La question de Louis (2026-08-19) : « comment ils marchent ces 5 folds : on
fait inférer 5 fois et on prend la moyenne, ou on fait plus malin ? ». La
réponse était : bêtement la moyenne des probabilités, flux par flux, que le
décodeur recompose ensuite en additionnant les logs — donc en traitant la
triade, la basse et les extensions comme indépendantes.

Cette page compare, sur les mêmes morceaux et le MÊME re-décodage :

  * 5 folds        la moyenne des PROBABILITÉS — ce qui tourne en prod ;
  * logits         la moyenne des LOGITS, softmax après — un produit
                   d'experts plutôt qu'un mélange, ça tranche au lieu de
                   moyenner les hésitations ;
  * vote           on décode les 5 folds SÉPARÉMENT et on vote l'accord temps
                   par temps — la seule variante qui respecte la structure de
                   l'accord au lieu de la casser en six marginales ;
  * 3 folds / 1 fold   ce qu'on gagnerait à en jeter.

Les cinq variantes sortent des MÊMES 5 passes avant : on garde les logits par
fold, tout le reste est de l'arithmétique.
"""
import html, sys, time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harmonia_min import beats as B, musx as M          # noqa: E402
from harmonia_min.labels import to_chord                # noqa: E402

SHIFT, SPEC = 18, 252
dev = M._device()
with M._InMusxDir():
    from mir.nn.train import NetworkInterface
    from chordnet_ismir_naive import ChordNet
    import chordnet_ismir_naive as _cn
    M._patch_init_hidden(_cn)
    NETS = [NetworkInterface(ChordNet(None), n, load_checkpoint=False).net.eval().to(dev)
            for n in M.MODEL_NAMES]

N = ["C", "D♭", "D", "E♭", "E", "F", "G♭", "G", "A♭", "A", "B♭", "B"]


def nom(lab):
    """`to_chord` rend déjà la qualité dans la notation iReal de l'app."""
    ch = to_chord(lab)
    if ch is None:
        return "N.C."
    s = f"{N[ch['root']]}{ch['q']}"
    if ch.get("bass", -1) >= 0 and ch["bass"] != ch["root"]:
        s += f"/{N[ch['bass']]}"
    return s


def logits_par_fold(cqt):
    """Les 6 flux AVANT softmax, pour chacun des 5 réseaux. 5 passes, une fois."""
    x = torch.tensor(np.asarray(cqt), dtype=torch.float32).to(dev)
    seq = x.shape[0]
    xt = x[:, SHIFT:SHIFT + SPEC].view(1, seq, SPEC)
    out, t0 = [], time.time()
    with torch.no_grad():
        for m in NETS:
            out.append([o.float().cpu() for o in m(xt)])
    return out, time.time() - t0


def _soft(streams):
    return [torch.softmax(s, dim=1).numpy().astype(np.float32) for s in streams]


def decode(pr, bd, bpb):
    lab, _ = M.redecode(bd["beats"], pr, downbeat_times=bd["downbeats"],
                        beats_per_bar=bpb, quarter_beats="all")
    return lab


def en_mesures(lab, bt, bpb):
    bars = {}
    for t0, _t1, l in lab:
        bars.setdefault(int(np.abs(bt - t0).argmin()) // bpb, []).append(nom(l))
    nb = (max(bars) + 1) if bars else 0
    return [" ".join(bars.get(b, ["%"])) for b in range(nb)]


def par_temps(lab, bt):
    """L'étiquette qui couvre chaque temps — l'unité sur laquelle on vote."""
    out = ["N"] * len(bt)
    for t0, t1, l in lab:
        for k in np.where((bt >= t0 - 1e-6) & (bt < t1 - 1e-6))[0]:
            out[k] = l
    return out


def vote(labs_par_fold, bt, triad_moyen, bpb):
    """Majorité temps par temps sur les 5 décodages. Égalité → la plus forte
    postérieure moyenne de triade, pas le premier fold arrivé."""
    tables = [par_temps(l, bt) for l in labs_par_fold]
    step = float(np.median(np.diff(bt)))
    gagnants = []
    for k in range(len(bt)):
        c = Counter(t[k] for t in tables)
        top = c.most_common()
        best = [lab for lab, n in top if n == top[0][1]]
        if len(best) == 1:
            gagnants.append(best[0])
        else:
            gagnants.append(max(best, key=lambda l: M.label_confidence(
                triad_moyen, float(bt[k]), float(bt[k]) + step, l)))
    # temps → segments, puis mesures
    lab = []
    k0 = 0
    for k in range(len(gagnants) + 1):
        if k == len(gagnants) or gagnants[k] != gagnants[k0]:
            t1 = float(bt[k]) if k < len(bt) else float(bt[-1]) + step
            lab.append((float(bt[k0]), t1, gagnants[k0]))
            k0 = k
    return lab


SONGS = [("maroon_5_this_love", "Maroon 5 — This Love"),
         ("ben_e_king_stand_by_me_audio", "Ben E. King — Stand By Me"),
         ("autumn_leaves", "Autumn Leaves")]

VARIANTES = [
    ("v5", "5 folds", "moyenne des probabilités — ce qui tourne en prod"),
    ("vlog", "moyenne des logits", "softmax après la moyenne : un produit d'experts"),
    ("vvote", "vote des 5 décodages", "chaque fold décode, on vote temps par temps"),
    ("v3", "3 folds", "on en jette deux"),
    ("v1", "1 fold", "5× moins cher"),
]

songs = []
for stem, titre in SONGS:
    p = Path("docs/audio") / f"{stem}.m4a"
    cqt = M.song_cqt(p)
    bd = B.track(p)
    bt = np.asarray(bd["beats"])
    bpb = int(round(np.median(np.diff(bd["downbeats"])) / np.median(np.diff(bt)))) or 4
    lg, t_pass = logits_par_fold(cqt)
    probs = [_soft(s) for s in lg]                     # softmax par fold

    res = {}
    t = time.time()
    pr5 = [np.mean([probs[f][j] for f in range(5)], axis=0) for j in range(6)]
    lab5 = decode(pr5, bd, bpb)
    res["v5"] = {"lab": lab5, "t": t_pass + time.time() - t}

    t = time.time()
    prlog = _soft([torch.stack([lg[f][j] for f in range(5)]).mean(0) for j in range(6)])
    res["vlog"] = {"lab": decode(prlog, bd, bpb), "t": t_pass + time.time() - t}

    t = time.time()
    labs = [decode(probs[f], bd, bpb) for f in range(5)]
    res["vvote"] = {"lab": vote(labs, bt, pr5[0], bpb), "t": t_pass + time.time() - t}

    t = time.time()
    res["v3"] = {"lab": decode([np.mean([probs[f][j] for f in range(3)], axis=0)
                                for j in range(6)], bd, bpb), "t": 0.6 * t_pass + time.time() - t}
    t = time.time()
    res["v1"] = {"lab": decode(probs[0], bd, bpb), "t": 0.2 * t_pass + time.time() - t}

    for k in res:
        res[k]["bars"] = en_mesures(res[k]["lab"], bt, bpb)
    nb = min(len(res[k]["bars"]) for k in res)
    for k in res:
        res[k]["same"] = round(100 * sum(1 for b in range(nb)
                                         if res[k]["bars"][b] == res["v5"]["bars"][b]) / max(nb, 1))
    songs.append({"stem": stem, "titre": titre, "nb": nb,
                  "temps": [float(bt[min(b * bpb, len(bt) - 1)]) for b in range(nb)],
                  "res": res})
    print(f"{stem}: {nb} mesures | " +
          " · ".join(f"{lbl} {res[k]['same']}% {res[k]['t']:.1f}s"
                     for k, lbl, _ in VARIANTES))


def cellules(s, key):
    out = []
    for b in range(s["nb"]):
        txt = s["res"][key]["bars"][b]
        d = " d" if (key != "v5" and txt != s["res"]["v5"]["bars"][b]) else ""
        out.append(f'<div class="m{d}" data-t="{s["temps"][b]:.2f}">'
                   f'<span class="n">{b + 1}</span>{html.escape(txt)}</div>')
    return "".join(out)


blocs = []
for s in songs:
    lignes = []
    for key, lbl, sous in VARIANTES:
        r = s["res"][key]
        cote = "référence" if key == "v5" else f"{r['same']} % identiques"
        lignes.append(f'<h3>{lbl} <small>{sous} — {cote} — {r["t"]:.1f} s</small></h3>'
                      f'<div class="g">{cellules(s, key)}</div>')
    blocs.append(f"""
<section>
  <h2>{html.escape(s['titre'])}</h2>
  <audio controls preload="none" src="/audio/{s['stem']}.m4a"></audio>
  <p class="lead">{s['nb']} mesures. Clique une mesure pour t'y placer dans l'audio.
     Les mesures <b class="dd">colorées</b> diffèrent de ce qui tourne en prod.</p>
  {"".join(lignes)}
</section>""")

page = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Comment combiner les 5 folds</title><style>
:root{{--pap:#f7f3ea;--ink:#1c1c1c;--f:#8a7f6d;--l:#ded5c4;--acc:#8a2b2b;--d:#f2dcc4}}
body{{margin:0 auto;padding:18px 14px 60px;background:var(--pap);color:var(--ink);
 font:15px/1.55 -apple-system,system-ui,sans-serif;max-width:900px}}
h1{{font:700 22px/1.3 Georgia,serif;margin:0 0 4px}}
h2{{font:700 18px/1.3 Georgia,serif;margin:34px 0 8px;border-top:1px solid var(--l);padding-top:18px}}
h3{{font:600 13px/1.4 -apple-system,sans-serif;margin:18px 0 6px;text-transform:uppercase;
 letter-spacing:.06em;color:var(--f)}}
h3 small{{text-transform:none;letter-spacing:0;font-weight:500;display:block;margin-top:2px}}
p.lead{{color:var(--f);font-size:13.5px;margin:6px 0 10px}}
audio{{width:100%;margin:2px 0 6px}}
.g{{display:grid;grid-template-columns:repeat(4,1fr);gap:3px}}
.m{{position:relative;background:#fff;border:1px solid var(--l);border-radius:6px;
 padding:12px 4px 7px;text-align:center;font:600 15px Georgia,serif;cursor:pointer;min-height:20px}}
.m .n{{position:absolute;top:2px;left:4px;font:500 8px -apple-system,sans-serif;color:#c3b8a4}}
.m.d{{background:var(--d);border-color:#e0bf95}}
.m:active{{outline:2px solid var(--acc)}}
b.dd{{background:var(--d);padding:1px 5px;border-radius:4px;font-weight:600}}
.intro{{background:#fff;border:1px solid var(--l);border-radius:10px;padding:14px 16px;margin:14px 0}}
.intro b{{color:var(--acc)}}
@media(min-width:620px){{.g{{grid-template-columns:repeat(8,1fr)}}}}
</style></head><body>
<h1>Comment combiner les 5 folds</h1>
<div class="intro">
<p>musx est livré en 5 réseaux — la même architecture entraînée cinq fois sur cinq
découpages du jeu d'entraînement. En prod on <b>moyenne leurs probabilités</b>,
flux par flux : la triade, la basse, la 7e, la 9e, la 11e, la 13e sont moyennées
séparément, et le décodeur les recompose en additionnant les logs, donc comme si
elles étaient indépendantes.</p>
<p>Moyenner <i>puis</i> multiplier n'est pas multiplier <i>puis</i> moyenner : la
moyenne des marginales peut faire gagner un accord que <b>personne n'avait classé
premier</b>. D'où les deux variantes testées ici — la moyenne des <b>logits</b>
(un produit d'experts, qui tranche au lieu de mélanger) et le <b>vote</b> sur
l'accord décodé (la seule qui ne casse jamais l'accord en six morceaux).</p>
<p>Les cinq variantes sortent des mêmes 5 passes avant : tout le reste est de
l'arithmétique. Écoute les mesures colorées.</p>
</div>
{"".join(blocs)}
<script>
document.querySelectorAll('section').forEach(sec=>{{
  const a=sec.querySelector('audio');
  sec.querySelectorAll('.m').forEach(m=>m.addEventListener('click',()=>{{
    a.currentTime=parseFloat(m.dataset.t)||0; a.play();
  }}));
}});
</script></body></html>"""
Path("docs/plots/folds.html").write_text(page, encoding="utf-8")
print("→ docs/plots/folds.html")
