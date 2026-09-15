"""docs/plots/fold_stack.html — empiler les répétitions ET les folds.

Louis, 2026-08-19 : « si on détecte 4 occurrences d'un accord, alors on replie
4 fois, donc si on fait déjà un 5-fold, ça nous ferait 4 × 5 = 20 probas à
rentrer dans le décodeur ».

Aujourd'hui on fait l'inverse : les 5 folds sont écrasés en un (moyenne des
probabilités) AVANT le modèle, et les N répétitions sont empilées SUR LE CQT.
Deux agrégations, deux étages, et la première jette de l'information avant même
que la seconde n'empile.

Cette page décode la MÊME section sous quatre lois :

  CQT moyenné      la prod : moyenne des N spectres, une inférence 5-folds ;
  moyenne des N    l'ancienne loi postérieure (folds déjà écrasés) ;
  moyenne des N×5  la pile complète de Louis, moyennée ;
  produit des N×5  la même pile, en log-probs (`logpool`) — un membre confiant
                   qui dit non a un veto, ce que la moyenne ne lui accorde pas.

Le décodage, lui, est rigoureusement le même dans les quatre cas
(`folding._decode_template`, gabarit pavé ×3).
"""
import html, json, sys, time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harmonia_min import beats as B, folding as F, musx as M   # noqa: E402

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


def probs_par_fold(cqt):
    """Les 6 flux de postérieures pour CHACUN des 5 réseaux (5 passes)."""
    x = torch.tensor(np.asarray(cqt), dtype=torch.float32).to(dev)
    xt = x[:, SHIFT:SHIFT + SPEC].view(1, x.shape[0], SPEC)
    out = []
    with torch.no_grad():
        for m in NETS:
            out.append([torch.softmax(o, dim=1).float().cpu().numpy().astype(np.float32)
                        for o in m(xt)])
    return out


def bars_texte(pos_chords, P):
    """Les P mesures du gabarit, en notation iReal, comme le chart les écrit."""
    out = []
    for k in range(P):
        mots = []
        for c in pos_chords[k] if pos_chords else []:
            if c.get("carry"):
                mots.append("%")
            elif c["nc"]:
                mots.append("N.C.")
            else:
                s = f"{N[c['root']]}{c['q']}"
                if c.get("bass", -1) >= 0 and c["bass"] != c["root"]:
                    s += f"/{N[c['bass']]}"
                mots.append(s)
        out.append(" ".join(mots) or "%")
    return out


def gabarit(members, Lf, comment):
    """members[k] = liste de blocs (chacun = 6 flux) → gabarit combiné, pavé ×3."""
    P = len(members)
    n_flux = len(members[0][0])
    tmpl = []
    for k in range(P):
        avg = [F._combine_stack(np.stack([_r(m[i], Lf) for m in members[k]]), comment)
               for i in range(n_flux)]
        tmpl.append(avg)
    return [np.concatenate([tmpl[k][i] for k in range(P)] * 3) for i in range(n_flux)]


_r = F._resample

LOIS = [
    ("cqt", "CQT moyenné", "la prod : N spectres moyennés, une inférence"),
    ("N", "moyenne des N", "les folds déjà écrasés, on moyenne les répétitions"),
    ("N5", "moyenne des N×5", "la pile complète, moyennée"),
    ("N5log", "produit des N×5", "la pile complète en log-probs — le veto du confiant"),
    ("N1", "moyenne des N, UN SEUL fold", "la répétition sans l'ensemble — 5× moins cher"),
    ("N5nt", "moyenne des N×5, SANS transposer", "la même pile en ignorant la montée d'un demi-ton"),
]

CIBLES = [("min_T64BgKEL-Sw", "T64BgKEL-Sw", "Bora Bora", "A"),
          ("min_T64BgKEL-Sw", "T64BgKEL-Sw", "Bora Bora", "B"),
          ("min_maroon_5_this_love", "maroon_5_this_love", "Maroon 5 — This Love", "B"),
          ("min_ben_e_king_stand_by_me_audio", "ben_e_king_stand_by_me_audio",
           "Ben E. King — Stand By Me", "A")]


def rot_probs(streams, r):
    """Monter les postérieures de `r` demi-tons.

    Le plan triade fait 73 = 1 (N) + 6 familles × 12 fondamentales, le plan
    basse 13 = 1 (pas de basse) + 12 hauteurs : on roule chaque bloc de 12 et
    on ne touche jamais la colonne 0. Les extensions (7e, 9e, 11e, 13e) sont
    relatives à la fondamentale — rien à tourner.
    """
    if not r:
        return streams
    out = list(streams)
    tri = np.array(streams[0], copy=True)
    tri[:, 1:] = np.roll(tri[:, 1:].reshape(tri.shape[0], -1, 12), r % 12,
                         axis=2).reshape(tri.shape[0], -1)
    out[0] = tri
    bas = np.array(streams[1], copy=True)
    bas[:, 1:] = np.roll(bas[:, 1:], r % 12, axis=1)
    out[1] = bas
    return out

songs = []
for chart_key, stem, titre, lettre in CIBLES:
    ch = json.loads(Path(f"harmonia_min/state/charts/{chart_key}.json").read_text())
    grid, bpb = ch["barGrid"], ch["bpb"]
    sec = next(s for s in ch["sections"] if s["label"] == lettre)
    occ = sec["barRanges"]
    P = len(sec["bars"])
    audio = Path("docs/audio") / f"{stem}.m4a"
    cqt = M.song_cqt(audio)
    pf = probs_par_fold(cqt)                       # 5 × 6 flux
    moy = [np.mean([pf[f][j] for f in range(5)], axis=0) for j in range(6)]
    Lf = max(bpb, int(round(float(np.median(np.diff(grid))) / M.FRAME_DT)))

    def bloc(b, p, transposer=True):
        a = max(0, int(round(grid[b] / M.FRAME_DT)))
        z = min(p[0].shape[0], int(round(grid[b + 1] / M.FRAME_DT)))
        out = [x[a:z] for x in p]
        return rot_probs(out, -par_barre.get(b, 0)) if transposer else out

    def bar_cqt(b):
        a = max(0, int(round(grid[b] / M.FRAME_DT)))
        z = min(cqt.shape[0], int(round(grid[b + 1] / M.FRAME_DT)))
        return F._cqt_transpose(cqt[a:z], -par_barre.get(b, 0))

    pos_members = [[o[0] + k for o in occ if o[0] + k <= o[1]] for k in range(P)]

    # ── LA MONTÉE D'UN DEMI-TON (Louis, 2026-08-19) ─────────────────────────
    # Sans ça, un passage transposé ne ressemble plus au premier : le gabarit
    # empile deux tonalités et musx entend une bouillie. On mesure le décalage
    # de chaque occurrence contre la première, exactement comme `folding` le
    # fait, et on ramène tout dans le ton de la référence AVANT d'empiler.
    from harmonia_min.nnls_features import extract_bothchroma
    from harmonia_min.sections import halfbar_features
    _arr, _times = extract_bothchroma(audio)
    Vb = F._bar_vecs(halfbar_features(grid, _arr, _times), len(grid) - 1)
    demi = {}
    for o in occ:
        r, sc = F.decalage_semitons(Vb, occ[0][0], o[0], P)
        demi[o[0]] = int(r)
    par_barre = {o[0] + k: demi[o[0]] for o in occ for k in range(P)}
    print(f"   décalages détectés par occurrence : {[demi[o[0]] for o in occ]} demi-ton(s)")

    res = {}
    for cle, _lbl, _s in LOIS:
        t = time.time()
        if cle == "cqt":
            cat = [np.asarray(x, dtype=np.float64) for x in
                   M.posteriors_from_cqt(F._cqt_template(pos_members, bar_cqt, Lf, P))]
        elif cle == "N":
            cat = gabarit([[bloc(b, moy) for b in pos_members[k]] for k in range(P)],
                          Lf, "mean")
        elif cle == "N1":
            cat = gabarit([[bloc(b, pf[0]) for b in pos_members[k]] for k in range(P)],
                          Lf, "mean")
        elif cle == "N5nt":
            mem = [[bloc(b, pf[f], transposer=False)
                    for b in pos_members[k] for f in range(5)] for k in range(P)]
            cat = gabarit(mem, Lf, "mean")
        else:
            mem = [[bloc(b, pf[f]) for b in pos_members[k] for f in range(5)]
                   for k in range(P)]
            cat = gabarit(mem, Lf, "mean" if cle == "N5" else "logpool")
        pc = F._decode_template([np.asarray(c, dtype=np.float64) for c in cat],
                                pos_members, None, Lf, bpb, P)
        res[cle] = {"bars": bars_texte(pc, P), "t": round(time.time() - t, 2),
                    "n": len(pos_members[0]) * (5 if cle.startswith("N5") else 1)}
    for cle in res:
        res[cle]["same"] = round(100 * sum(1 for k in range(P)
                                           if res[cle]["bars"][k] == res["cqt"]["bars"][k]) / P)
    songs.append({"demi": [demi[o[0]] for o in occ],
                  "stem": stem, "titre": titre, "lettre": lettre, "P": P,
                  "occ": occ, "res": res,
                  "temps": [float(grid[occ[0][0] + k]) for k in range(P)],
                  "autres": [[float(grid[o[0] + k]) for k in range(P)] for o in occ]})
    print(f"{stem} — {lettre} : {P} mes. × {len(occ)} occ. | " +
          " · ".join(f"{lbl} {res[c]['same']}%" for c, lbl, _ in LOIS))


def cellules(s, cle):
    out = []
    for k in range(s["P"]):
        txt = s["res"][cle]["bars"][k]
        d = " d" if (cle != "cqt" and txt != s["res"]["cqt"]["bars"][k]) else ""
        out.append(f'<div class="m{d}" data-t="{s["temps"][k]:.2f}">'
                   f'<span class="n">{k + 1}</span>{html.escape(txt)}</div>')
    return "".join(out)


blocs = []
for s in songs:
    lignes = []
    for cle, lbl, sous in LOIS:
        r = s["res"][cle]
        cote = "référence" if cle == "cqt" else f"{r['same']} % identiques"
        lignes.append(f'<h3>{lbl} <small>{sous} · {r["n"]} membres empilés — '
                      f'{cote} — {r["t"]} s</small></h3>'
                      f'<div class="g">{cellules(s, cle)}</div>')
    liens = " ".join(f'<button data-t="{t[0]:.2f}">occ. {i+1}</button>'
                     for i, t in enumerate(s["autres"]))
    blocs.append(f"""
<section>
  <h2>{html.escape(s['titre'])} <span class="sec">section {s['lettre']}</span></h2>
  <audio controls preload="none" src="/audio/{s['stem']}.m4a"></audio>
  <p class="lead">{s['P']} mesures, jouées {len(s['occ'])} fois
     (décalage détecté : {", ".join(f"+{d}" if d else "0" for d in s['demi'])} demi-ton).
     Aller à : {liens}<br>
     Clique une mesure pour l'entendre (1ʳᵉ occurrence). Les mesures
     <b class="dd">colorées</b> diffèrent de ce que la prod écrit aujourd'hui.</p>
  {"".join(lignes)}
</section>""")

page = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Empiler les répétitions ET les folds</title><style>
:root{{--pap:#f7f3ea;--ink:#1c1c1c;--f:#8a7f6d;--l:#ded5c4;--acc:#8a2b2b;--d:#f2dcc4}}
body{{margin:0 auto;padding:18px 14px 60px;background:var(--pap);color:var(--ink);
 font:15px/1.55 -apple-system,system-ui,sans-serif;max-width:900px}}
h1{{font:700 22px/1.3 Georgia,serif;margin:0 0 4px}}
h2{{font:700 18px/1.3 Georgia,serif;margin:34px 0 8px;border-top:1px solid var(--l);padding-top:18px}}
h2 .sec{{font:600 12px -apple-system,sans-serif;color:var(--f);vertical-align:middle}}
h3{{font:600 13px/1.4 -apple-system,sans-serif;margin:18px 0 6px;text-transform:uppercase;
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
b.dd{{background:var(--d);padding:1px 5px;border-radius:4px;font-weight:600}}
.intro{{background:#fff;border:1px solid var(--l);border-radius:10px;padding:14px 16px;margin:14px 0}}
.intro b{{color:var(--acc)}}
@media(min-width:620px){{.g{{grid-template-columns:repeat(8,1fr)}}}}
</style></head><body>
<h1>Empiler les répétitions <i>et</i> les folds</h1>
<div class="intro">
<p>Une section jouée 4 fois, un modèle livré en 5 réseaux : ça fait
<b>20 jeux de probabilités</b> pour la même mesure. Aujourd'hui on n'en donne
jamais 20 au décodeur — les 5 folds sont moyennés <i>avant</i> le modèle, et les
répétitions sont empilées <i>sur le spectre</i>. Deux agrégations à deux étages,
dont la première jette de l'information avant que la seconde n'empile.</p>
<p>Ici les quatre lois décodent la même section, avec exactement le même
décodeur. Celle qui m'intéresse est la dernière : avec 20 votants, le
<b>produit</b> (somme des log-probs) devient tranchant là où la moyenne reste
molle — un membre confiant qui dit non garde un veto.</p>
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
Path("docs/plots/fold_stack.html").write_text(page, encoding="utf-8")
print("→ docs/plots/fold_stack.html")
