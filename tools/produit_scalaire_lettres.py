"""Page d'illustration : quel produit scalaire sur les postérieures musx pour
les « petites lettres » (Louis, 2026-10-12, Evening Star).

    .venv/bin/python -m tools.produit_scalaire_lettres

Écrit docs/plots/produit_scalaire_lettres_2026-10-12.html. Ne touche à aucun
code de pipeline : on recalcule des similarités candidates entre bi-mesures
(4 demi-mesures par bi-mesure) et on applique la règle de lettres ACTUELLE
(`soudure._mot_lettres` : accords décodés identiques ⇒ même lettre ; le seuil
de fusion = la moins ressemblante des paires identiques du morceau) à chacune.
"""
from __future__ import annotations

import html
import json
from pathlib import Path

import numpy as np

from harmonia import cache, musx, soudure
from harmonia.musx import FRAME_DT
from harmonia.sections.similarity import chord_tone_matrix, harmonic_vectors
from harmonia.settings import SETTINGS

STEM = "CHVWPIFAT5A"
CHART = SETTINGS.repo / "state/cache/charts/min_CHVWPIFAT5A.json"
OUT = SETTINGS.repo / "docs/plots/produit_scalaire_lettres_2026-10-12.html"

NOTES = "C D♭ D E♭ E F G♭ G A♭ A B♭ B".split()
QUAL = ["", "m", "sus4", "sus2", "dim", "aug"]
SEPT = ["sans 7e", "maj7", "7 (b7)", "dim7"]

# Plans d'extension de musx — sens des colonnes LU dans le code vendu
# (third_party/musx_ismir2019/complex_chord.py : SeventhTypes, NinthTypes,
# EleventhTypes, ThirteenthTypes ; chordnet_ismir_naive.py : chord_limit
# 7e=3, 9e=3, 11e=2, 13e=2 ⇒ 4/4/3/3 colonnes, colonne 0 = « aucune »).
# Le 6 est la 13e : SuffixDecoder.ADD_NOTES '6' → ThirteenthTypes.add_13,
# 'maj6'/'min6' → add_13 ; 'b6' → add_b13. Ces plans sont relatifs à la
# fondamentale (même colonne quel que soit l'accord). Le décodeur (xhmm_ismir)
# multiplie p(triade)·p(basse)·p(7e)·p(9e)·p(11e)·p(13e) : un C9 bat un C7
# exactement quand p(9e = 9) > p(9e = aucune) sur la durée de l'accord.
PLANS = [
    ("s7", "7e", ["non", "7M", "♭7", "♭♭7"]),
    ("s9", "9e", ["non", "9", "♯9", "♭9"]),
    ("s11", "11e", ["non", "11", "♯11"]),
    ("s13", "6te / 13e", ["non", "6 (=13)", "♭6 (=♭13)"]),
]
EXT_COL = ["#8a8a90", "#2a6fdb", "#e08a1e", "#9b59b6"]   # non / 1re / 2e / 3e colonne

# Les 12 bi-mesures montrées (1re mesure, 1-indexée) et leur nom.
CELLS = [(13, "A1·1"), (15, "A1·2"), (17, "A1·3"), (19, "A1·4"),
         (21, "A2·1"), (23, "A2·2"), (25, "A2·3"), (27, "A2·4"),
         (29, "B·1"), (31, "B·2"), (33, "B·3"), (35, "B·4")]


def tname(i: int) -> str:
    if i == 0:
        return "N"
    return NOTES[(i - 1) % 12] + QUAL[(i - 1) // 12]


def unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else v


def cos(a, b):
    return float(unit(a) @ unit(b))


def bhat(a, b):
    a = np.clip(a, 0, None); b = np.clip(b, 0, None)
    a = a / max(a.sum(), 1e-12); b = b / max(b.sum(), 1e-12)
    return float(np.sqrt(a) @ np.sqrt(b))


def main():
    chart = json.loads(CHART.read_text())
    z = cache.load_npz("musx_probs", (SETTINGS.audio_dir / f"{STEM}.m4a").resolve())
    triad, bass, s7 = z["triad"], z["bass"], z["s7"]
    ext = {k: z[k] for k, _, _ in PLANS}
    grid = chart["barGrid"]
    bt = np.asarray(chart["beatTimes"])
    n = len(grid) - 1
    assert chart["bpb"] == 4

    # demi-mesures : temps 1-2 / 3-4 (on vérifie 4 temps par mesure)
    halves = []
    for b in range(n):
        t0, t1 = grid[b], grid[b + 1]
        beats = bt[(bt >= t0 - 0.05) & (bt < t1 - 0.05)]
        mid = float(beats[2]) if len(beats) == 4 else 0.5 * (t0 + t1)
        halves += [(t0, mid), (mid, t1)]

    def pool(X, t0, t1):
        a = int(round(t0 / FRAME_DT)); c = max(a + 1, int(round(t1 / FRAME_DT)))
        return X[a:c].mean(0)

    M = chord_tone_matrix(73)
    H = []  # par demi-mesure : dict de vecteurs
    for (t0, t1) in halves:
        p = pool(triad, t0, t1)
        bs = pool(bass, t0, t1)
        sv = pool(s7, t0, t1)
        root = p[1:].reshape(6, 12).sum(0)
        chroma = p @ M
        # chroma + septième : la 7e de musx posée sur la fondamentale probable
        c7 = chroma.copy()
        for k, off in ((1, 11), (2, 10), (3, 9)):
            c7 += sv[k] * np.roll(root, off)
        d = dict(tri=p, root=root, chroma=chroma, c7=c7, bass=bs[1:],
                 bass_full=bs, s7=sv)
        for k, _, _ in PLANS:
            d[k] = pool(ext[k], t0, t1)
        H.append(d)

    # ── extensions : sont-elles jamais probables ? ─────────────────────────
    def bar_of(t):
        return int(np.searchsorted(grid, t, side="right"))   # 1-indexée
    ext_stats = []
    for k, nom, cols in PLANS:
        X = ext[k]
        fr = 1 - X[:, 0]
        hv = np.array([1 - h[k][0] for h in H])
        jh = int(np.argmax(hv))
        jf = int(np.argmax(fr))
        best_col = 1 + int(np.argmax(H[jh][k][1:]))
        ext_stats.append(dict(
            k=k, nom=nom, cols=cols, mean=float(fr.mean()),
            fmax=float(fr.max()), ft=jf * FRAME_DT, fbar=bar_of(jf * FRAME_DT),
            hmax=float(hv[jh]), hbar=jh // 2 + 1, htemps="1-2" if jh % 2 == 0 else "3-4",
            hcol=cols[best_col],
            hwin=int((hv > 0.5).sum()), nh=len(H),
            ffrac=float((fr > 0.5).mean())))
    # ce que le décodeur de production écrit sur ce morceau (aucun cache lab)
    lab = musx.redecode(chart["beatTimes"],
                        [z[k] for k in ("triad", "bass", "s7", "s9", "s11", "s13")],
                        downbeat_times=grid[:-1])
    quals = [l.split(":", 1)[-1] for _, _, l in lab]
    n_ext = sum(1 for q in quals if any(s in q for s in ("9", "11", "13", "6")))
    dec = dict(n=len(lab), n_ext=n_ext, n7=sum(1 for q in quals if "7" in q))
    for s in ext_stats:
        print(f"{s['k']:4s} moy {s['mean']:.3f} max trame {s['fmax']:.3f} (mes. {s['fbar']}) "
              f"max demi-mes. {s['hmax']:.3f} (mes. {s['hbar']} t{s['htemps']}, {s['hcol']}) "
              f">0.5 sur {s['hwin']}/{s['nh']} demi-mes.")
    print("décodeur :", dec)

    # ── différences avec les voisines ──────────────────────────────────────
    def tv(a, b):
        return 0.5 * float(np.abs(a - b).sum())
    for h in range(len(H)):
        prev = H[h - 1] if h > 0 else None
        H[h]["tv_tri"] = tv(H[h]["tri"], prev["tri"]) if prev else None
        H[h]["tv_bass"] = tv(H[h]["bass_full"], prev["bass_full"]) if prev else None
        H[h]["tv_ext"] = (float(np.mean([tv(H[h][k], prev[k]) for k, _, _ in PLANS]))
                          if prev else None)

    # voisines des deux côtés (dans tout le morceau, pas coupées à la zone montrée)
    def tvx(a, b, key):
        if key == "ext":
            return float(np.mean([tv(a[k], b[k]) for k, _, _ in PLANS]))
        kk = {"tri": "tri", "bass": "bass_full"}[key]
        return tv(a[kk], b[kk])
    for h in range(len(H)):
        for key in ("tri", "bass", "ext"):
            vp = tvx(H[h], H[h - 1], key) if h > 0 else None
            vn = tvx(H[h], H[h + 1], key) if h + 1 < len(H) else None
            ex = [v for v in (vp, vn) if v is not None]
            H[h]["iso_" + key] = (vp, vn, min(ex))   # min = voisine la plus proche

    # bornes : bi-mesures depuis la mesure 1 (même grille que le mot du
    # morceau, 21 jetons ; la 43e mesure reste hors grille)
    bornes = list(range(0, n - n % 2 + 1, 2))
    J = len(bornes) - 1

    def cell_vecs(j, key):
        b0 = bornes[j]
        return [H[2 * b0 + k][key] for k in range(4)]

    def S_pos(key, f=cos):
        S = np.zeros((J, J))
        for i in range(J):
            A = cell_vecs(i, key)
            for k in range(J):
                Bv = cell_vecs(k, key)
                S[i, k] = np.mean([f(A[q], Bv[q]) for q in range(4)])
        return S

    # Aujourd'hui : vecteur chord-tone par MESURE, cosinus, moyenne sur 2 mesures
    V = harmonic_vectors(triad, grid)
    Sb = V @ V.T
    today = soudure._matrice_bimesures(Sb, bornes)

    cands = [
        ("auj", "Aujourd'hui (référence)",
         "s(X,Y) = ½ Σ<sub>m=1..2</sub> cos( P̄<sub>m</sub>(X)·T , P̄<sub>m</sub>(Y)·T )",
         "Ce que l'outil fait déjà : la postérieure triade moyennée sur chaque MESURE, "
         "projetée sur ses notes d'accord (12 hauteurs), comparée mesure à mesure.",
         today),
        ("a", "(a) triade complète",
         "s(X,Y) = ¼ Σ<sub>k=1..4</sub> cos( p<sub>k</sub>(X), p<sub>k</sub>(Y) ), p ∈ ℝ<sup>73</sup>",
         "Demi-mesure par demi-mesure, « est-ce le même accord nommé ? ». "
         "Gm et B♭ sont aussi loin que Gm et E.",
         S_pos("tri")),
        ("b", "(b) fondamentale seule",
         "idem avec r<sub>k</sub> = Σ<sub>qualités</sub> p<sub>k</sub> ∈ ℝ<sup>12</sup>",
         "On oublie majeur/mineur : seule la fondamentale compte. G7 ≡ Gm.",
         S_pos("root")),
        ("c", "(c) notes d'accord",
         "idem avec c<sub>k</sub> = p<sub>k</sub>·T ∈ ℝ<sup>12</sup> (T = matrice accord→notes)",
         "Chaque accord devient ses 3 notes, pondérées par sa probabilité : "
         "Gm (sol si♭ ré) est proche de B♭ (si♭ ré fa).",
         S_pos("chroma")),
        ("c7", "(c′) notes d'accord + 7e",
         "c′<sub>k</sub> = c<sub>k</sub> + Σ<sub>7e</sub> s7<sub>k</sub>·roll(r<sub>k</sub>, intervalle)",
         "Comme (c), en ajoutant la septième que musx entend (C7 ≠ C).",
         S_pos("c7")),
        ("d", "(d) basse",
         "idem avec β<sub>k</sub> ∈ ℝ<sup>12</sup> (tête basse de musx, sans « pas de basse »)",
         "Seule la note de basse : F/C a une basse do, F a une basse fa.",
         S_pos("bass")),
        ("e", "(e) notes d'accord + basse",
         "s = ½ s<sub>(c)</sub> + ½ s<sub>(d)</sub>",
         "Les deux à parts égales : les notes de l'accord ET ce que joue la basse.",
         0.5 * S_pos("chroma") + 0.5 * S_pos("bass")),
        ("fa", "(f) triade, Bhattacharyya",
         "s(X,Y) = ¼ Σ<sub>k</sub> Σ<sub>i</sub> √p<sub>k,i</sub>(X) · √p<sub>k,i</sub>(Y)",
         "Comme (a), mais avec √p : une hésitation à 20 % compte davantage "
         "qu'avec le cosinus, dominé par l'accord à 80 %.",
         S_pos("tri", bhat)),
        ("fe", "(f′) notes + basse, Bhattacharyya",
         "s = ½ BC<sub>(c)</sub> + ½ BC<sub>(d)</sub> (vecteurs ramenés à somme 1)",
         "Comme (e), avec √p.",
         0.5 * S_pos("chroma", bhat) + 0.5 * S_pos("bass", bhat)),
    ]

    bars = soudure.accords_par_mesure(chart)
    idx = [(m - 1) // 2 for m, _ in CELLS]   # jeton de chaque cellule
    A1, A2 = idx[0:4], idx[4:8]

    def odd_one(S, grp):
        sc = [np.mean([S[i, k] for k in grp if k != i]) for i in grp]
        return int(np.argmin(sc)), sc

    def aaba_window(S, grp):
        """L'intervalle de seuils qui donne exactement a a b a sur ces 4
        cellules (même fusion que `_mot_lettres` : lien complet, le plus
        ressemblant d'abord). None si aucun seuil ne le donne."""
        vals = sorted({float(S[i, k]) for i in grp for k in grp if i < k})
        pts = sorted(set(vals + [(x + y) / 2 for x, y in zip(vals, vals[1:])]))
        ok = [t for t in pts if fusion(S, [[j] for j in grp], t, grp) == "aaba"]
        if not ok:
            return None
        lo = max([v for v in vals if v < min(ok)], default=min(ok))
        return (lo, max(ok))

    res = []
    for key, title, formula, sentence, S in cands:
        tr = {}
        mot = soudure._mot_lettres(S, bars, bornes, tr)
        thr = tr.get("seuil")
        oA1, sc1 = odd_one(S, A1)
        oA2, sc2 = odd_one(S, A2)
        w1 = aaba_window(S, A1)
        w2 = aaba_window(S, A2)
        # le morceau entier, au seuil du milieu de la fenêtre a a b a de A1
        forced = None
        if w1 is not None:
            lo, hi = w1
            if w2 is not None and max(lo, w2[0]) < min(hi, w2[1]):
                lo, hi = max(lo, w2[0]), min(hi, w2[1])   # commun à A1 et A2
            t = 0.5 * (lo + hi)
            forced = (t, fusion(S, tr["unites"], t, list(range(J))))
        res.append(dict(key=key, title=title, formula=formula, sentence=sentence,
                        S=S, mot=mot, thr=thr, oA1=oA1, oA2=oA2, sc1=sc1, sc2=sc2,
                        w1=w1, w2=w2, forced=forced))
        fm = forced[1] if forced else ""
        print(f"{key:4s} seuil={thr:.4f} mot={mot} A1={mot[A1[0]:A1[-1]+1]} "
              f"A2={mot[A2[0]:A2[-1]+1]} B={mot[idx[8]:idx[11]+1]} "
              f"intrus A1=cell{oA1+1} A2=cell{oA2+1} fenA1={w1} fenA2={w2} "
              f"force={forced[0] if forced else None} "
              f"A1={fm[A1[0]:A1[-1]+1]} A2={fm[A2[0]:A2[-1]+1]} B={fm[idx[8]:idx[11]+1]} mot={fm}")

    for m0 in (13, 21, 29):
        for k in range(16):
            h = 2 * (m0 - 1) + k
            print("iso", m0 + k // 2, "t1-2" if k % 2 == 0 else "t3-4",
                  " ".join(f"{q}:{H[h]['iso_'+q][0]:.2f}/{H[h]['iso_'+q][1]:.2f}->{H[h]['iso_'+q][2]:.2f}"
                           for q in ("tri", "bass", "ext")), chord_text(bars, m0 - 1 + k // 2))
    write_page(chart, H, bornes, idx, A1, A2, res, halves, bars, ext_stats, dec)


# ── page ─────────────────────────────────────────────────────────────────────

def fusion(S, unites, t, order):
    """Copie de l'étape ② de `soudure._mot_lettres`, au seuil `t` imposé."""
    unites = [list(u) for u in unites]
    while True:
        best = None
        for i in range(len(unites)):
            for k in range(i + 1, len(unites)):
                v = min(float(S[x, y]) for x in unites[i] for y in unites[k])
                if v >= t and (best is None or v > best[0]):
                    best = (v, i, k)
        if best is None:
            break
        _, i, k = best
        unites[i] += unites[k]
        del unites[k]
    g = {j: gi for gi, u in enumerate(unites) for j in u}
    ren = {}
    for j in order:
        ren.setdefault(g[j], len(ren))
    return "".join("abcdefghijklmnopqrstuvwxyz"[ren[g[j]]] for j in order)


def chord_text(bars, b):
    out = []
    for c in bars[b]:
        if c.get("nc"):
            out.append("N.C."); continue
        s = NOTES[int(c["root"]) % 12] + {"-": "m", "-7": "m7", "^7": "maj7"}.get(c["q"], c["q"])
        if int(c.get("bass", -1)) >= 0:
            s += "/" + NOTES[int(c["bass"]) % 12]
        out.append(s)
    return " ".join(out) or "–"


def heat(v, vmax, hue):
    a = 0 if vmax <= 0 else min(1.0, float(v) / vmax)
    return f"background:hsla({hue},70%,45%,{a:.2f})"


def half_block(h):
    p = h["tri"]
    top = np.argsort(-p)[:5]
    rows = "".join(
        f'<div class="r"><span class="n">{html.escape(tname(i))}</span>'
        f'<span class="bar"><i style="width:{100*p[i]:.0f}%"></i></span>'
        f'<span class="v">{100*p[i]:.0f}</span></div>' for i in top)
    b = h["bass_full"]
    btop = np.argsort(-b)[:3]
    bs = " · ".join(("—" if i == 0 else NOTES[i - 1]) + f" {100*b[i]:.0f}" for i in btop)
    return (f'<div class="half">{rows}<div class="sub">basse : {bs}</div>'
            f'{ext_rows(h)}</div>')


def ext_rows(h):
    """Une barre empilée par plan d'extension : gris = « non », couleurs =
    les colonnes réelles de musx ; à droite, la plus probable des non-« non »."""
    out = []
    for k, nom, cols in PLANS:
        v = h[k]
        segs = "".join(f'<i style="width:{100*v[i]:.1f}%;background:{EXT_COL[i]}" '
                       f'title="{cols[i]} {100*v[i]:.0f} %"></i>' for i in range(len(cols)))
        j = 1 + int(np.argmax(v[1:]))
        out.append(f'<div class="r ex"><span class="n">{nom.split(" ")[0]}</span>'
                   f'<span class="stk">{segs}</span>'
                   f'<span class="v">{cols[j].split(" ")[0]}&nbsp;{100*v[j]:.0f}</span></div>')
    return "".join(out)


def delta_rows(a, b):
    """Différence p(a) − p(b), plan par plan, en barres divergentes
    (vers la droite = présent ici, absent à côté ; vers la gauche = l'inverse)."""
    rows = []
    d = a["tri"] - b["tri"]
    for i in np.argsort(-np.abs(d))[:3]:
        rows.append((tname(i), d[i], ""))
    d = a["bass_full"] - b["bass_full"]
    i = int(np.argmax(np.abs(d)))
    rows.append(("/" + ("—" if i == 0 else NOTES[i - 1]), d[i], "basse"))
    best = None
    for k, nom, cols in PLANS:
        d = a[k] - b[k]
        for i in range(1, len(cols)):
            if best is None or abs(d[i]) > abs(best[1]):
                best = (cols[i].split(" ")[0], d[i], nom)
    rows.append(best)
    out = []
    for name, v, kind in rows:
        w = min(50.0, 50 * abs(v))
        side = f"left:50%;width:{w:.1f}%" if v >= 0 else f"right:50%;width:{w:.1f}%"
        cls = "up" if v >= 0 else "dn"
        tag = f' <span class="kd">{kind}</span>' if kind else ""
        out.append(f'<div class="r dr"><span class="n">{html.escape(name)}{tag}</span>'
                   f'<span class="dv"><i class="{cls}" style="{side}"></i></span>'
                   f'<span class="v">{100*v:+.0f}</span></div>')
    return "".join(out)


def strip_svg(H, b0, nb, bars, grid):
    """Bande « où ça change » : pour chaque frontière entre deux demi-mesures,
    TV = ½ Σ |p(h) − p(h−1)| pour l'accord (triade 73), la basse (13) et les
    extensions (moyenne des 4 plans). 0 = identique, 1 = tout a changé."""
    nh = 2 * nb
    W, lab, rowh, gap, top = 360, 40, 26, 12, 24
    cw = (W - lab) / nh
    rows = [("accord", "tv_tri", "var(--acc)"), ("basse", "tv_bass", "#e08a1e"),
            ("ext.", "tv_ext", "#9b59b6")]
    Ht = top + len(rows) * (rowh + gap) + 6
    g = [f'<svg viewBox="0 0 {W} {Ht}" class="strip" role="img" aria-label="force des changements">']
    for c in range(0, nb, 2):   # bi-mesures
        x = lab + 2 * c * cw
        g.append(f'<rect x="{x:.1f}" y="12" width="{4*cw:.1f}" height="{Ht-18}" '
                 f'fill="{"var(--line)" if (c // 2) % 2 == 0 else "none"}" opacity=".45"/>')
    for m in range(nb):
        x = lab + 2 * m * cw
        g.append(f'<text x="{x + cw:.1f}" y="9" class="sx">{b0 + m + 1}</text>')
    for r, (nom, key, col) in enumerate(rows):
        y0 = top + r * (rowh + gap)
        g.append(f'<text x="0" y="{y0 + rowh - 6}" class="sl">{nom}</text>')
        g.append(f'<line x1="{lab}" x2="{W}" y1="{y0+rowh}" y2="{y0+rowh}" stroke="var(--mut)" stroke-width=".5"/>')
        for k in range(nh):
            v = H[2 * b0 + k][key]
            if v is None:
                continue
            # la frontière h−1 | h est dessinée au bord gauche de la demi-mesure h
            x = lab + k * cw - 0.32 * cw
            hh = rowh * min(1.0, v)
            g.append(f'<rect x="{x:.1f}" y="{y0 + rowh - hh:.1f}" width="{0.64*cw:.1f}" '
                     f'height="{hh:.1f}" fill="{col}"><title>mes. {b0 + k//2 + 1} '
                     f'{"t1" if k % 2 == 0 else "t3"} : {v:.2f}</title></rect>')
            if v >= 0.35:
                g.append(f'<text x="{x + 0.32*cw:.1f}" y="{y0 + rowh - hh - 1:.1f}" class="sv">{100*v:.0f}</text>')
    g.append("</svg>")
    return "".join(g)


def iso_strip_svg(H, b0, nb, grid):
    """Bandes « isolement » : par demi-mesure h, barre pleine = min(TVprev, TVnext)
    (la voisine la plus proche), barre claire derrière = max(TVprev, TVnext).
    Les voisines viennent du morceau entier (H est global)."""
    nh = 2 * nb
    W, lab, rowh, gap, top = 360, 40, 30, 12, 24
    cw = (W - lab) / nh
    rows = [("accord", "tri", "var(--acc)"), ("basse", "bass", "#e08a1e"),
            ("ext.", "ext", "#9b59b6")]
    Ht = top + len(rows) * (rowh + gap) + 6
    g = [f'<svg viewBox="0 0 {W} {Ht}" class="strip" role="img" aria-label="isolement des demi-mesures">']
    for c in range(0, nb, 2):
        x = lab + 2 * c * cw
        g.append(f'<rect x="{x:.1f}" y="12" width="{4*cw:.1f}" height="{Ht-18}" '
                 f'fill="{"var(--line)" if (c // 2) % 2 == 0 else "none"}" opacity=".45"/>')
    for m in range(nb):
        g.append(f'<text x="{lab + 2*m*cw + cw:.1f}" y="9" class="sx">{b0 + m + 1}</text>')
    for r, (nom, key, col) in enumerate(rows):
        y0 = top + r * (rowh + gap)
        g.append(f'<text x="0" y="{y0 + rowh - 6}" class="sl">{nom}</text>')
        g.append(f'<line x1="{lab}" x2="{W}" y1="{y0+rowh}" y2="{y0+rowh}" stroke="var(--mut)" stroke-width=".5"/>')
        for k in range(nh):
            vp, vn, mn = H[2 * b0 + k]["iso_" + key]
            mx = max(v for v in (vp, vn) if v is not None)
            x = lab + k * cw + 0.18 * cw
            hh = rowh * min(1.0, mx)
            g.append(f'<rect x="{x:.1f}" y="{y0 + rowh - hh:.1f}" width="{0.64*cw:.1f}" height="{hh:.1f}" '
                     f'fill="{col}" opacity=".28"/>')
            hh = rowh * min(1.0, mn)
            g.append(f'<rect x="{x:.1f}" y="{y0 + rowh - hh:.1f}" width="{0.64*cw:.1f}" height="{hh:.1f}" '
                     f'fill="{col}"><title>mes. {b0 + k//2 + 1} {"t1-2" if k % 2 == 0 else "t3-4"} : '
                     f'avant {vp if vp is None else format(vp, ".2f")}, après {vn if vn is None else format(vn, ".2f")}, '
                     f'min {mn:.2f}</title></rect>')
            if mn >= 0.3:
                g.append(f'<text x="{x + 0.32*cw:.1f}" y="{y0 + rowh - hh - 1:.1f}" class="sv">{100*mn:.0f}</text>')
    g.append("</svg>")
    return "".join(g)


def pair_rows(h, hp, hn):
    """Pour une demi-mesure h : p(h)−p(précédente) (barre de gauche) et
    p(h)−p(suivante) (barre de droite), mêmes entrées des deux côtés :
    3 triades, la basse, l'extension, choisies par le plus grand |Δ| des deux."""
    items = []
    dp, dn = h["tri"] - hp["tri"], h["tri"] - hn["tri"]
    for i in np.argsort(-np.maximum(np.abs(dp), np.abs(dn)))[:3]:
        items.append((tname(i), dp[i], dn[i], ""))
    dp, dn = h["bass_full"] - hp["bass_full"], h["bass_full"] - hn["bass_full"]
    i = int(np.argmax(np.maximum(np.abs(dp), np.abs(dn))))
    items.append(("/" + ("—" if i == 0 else NOTES[i - 1]), dp[i], dn[i], "basse"))
    best = None
    for k, nom, cols in PLANS:
        dp, dn = h[k] - hp[k], h[k] - hn[k]
        for i in range(1, len(cols)):
            m = max(abs(dp[i]), abs(dn[i]))
            if best is None or m > best[0]:
                best = (m, cols[i].split(" ")[0], dp[i], dn[i], nom)
    items.append((best[1], best[2], best[3], best[4]))

    def half(v, left):
        w = min(50.0, 50 * abs(v))
        side = f"left:50%;width:{w:.1f}%" if v >= 0 else f"right:50%;width:{w:.1f}%"
        return f'<span class="dv"><i class="{"up" if v >= 0 else "dn"}" style="{side}"></i></span>'
    out = []
    for name, a, b, kind in items:
        tag = f' <span class="kd">{kind}</span>' if kind else ""
        out.append(f'<div class="pr"><div class="pn"><span class="n">{html.escape(name)}{tag}</span>'
                   f'<span class="v">{100*a:+.0f} | {100*b:+.0f}</span></div>'
                   f'<div class="pb">{half(a, True)}{half(b, False)}</div></div>')
    return "".join(out)


def iso_cards(H, b0, nb, bars):
    cards = []
    for k in range(2 * nb):
        h = 2 * b0 + k
        t = f'mes. {b0 + k//2 + 1} · t{"1-2" if k % 2 == 0 else "3-4"}'
        vt = H[h]["iso_tri"]
        cards.append(
            f'<div class="hcol"><div class="hl"><b>{t}</b> · {html.escape(chord_text(bars, b0 + k//2))}</div>'
            f'<div class="dh">accord : avant {100*vt[0]:.0f} · après {100*vt[1]:.0f} · <b>min {100*vt[2]:.0f}</b></div>'
            f'{pair_rows(H[h], H[h-1], H[h+1])}</div>')
    return "".join(cards)


def matrix_html(S, labels, thr, odd=None):
    h = '<table class="mx"><tr><th></th>' + "".join(f"<th>{l}</th>" for l in labels) + "</tr>"
    for i, li in enumerate(labels):
        h += f"<tr><th>{li}</th>"
        for k in range(len(labels)):
            v = S[i][k]
            cls = "ge" if (thr is not None and v >= thr and i != k) else ""
            if i == k:
                cls = "dg"
            a = max(0.0, min(1.0, (v - 0.3) / 0.7)) if i != k else 0
            h += f'<td class="{cls}" style="background:hsla(210,70%,50%,{0.55*a:.2f})">{v:.2f}</td>'
        h += "</tr>"
    return h + "</table>"


def write_page(chart, H, bornes, idx, A1, A2, res, halves, bars, ext_stats, dec):
    grid = chart["barGrid"]
    cell_html = []
    for (m, name), j in zip(CELLS, idx):
        b0 = m - 1
        t0, t1 = grid[b0], grid[b0 + 2]
        chords = f"{chord_text(bars, b0)} | {chord_text(bars, b0 + 1)}"
        hs = [H[2 * b0 + k] for k in range(4)]
        # mini-carte de chaleur : 4 demi-mesures × 12 notes d'accord
        vmax = max(float(h["chroma"].max()) for h in hs)
        hm = '<table class="hm"><tr><th></th>' + "".join(
            f"<th>{n}</th>" for n in NOTES) + "</tr>"
        for k, h in enumerate(hs):
            hm += f"<tr><th>{k+1}</th>" + "".join(
                f'<td style="{heat(v, vmax, 210)}"></td>' for v in h["chroma"]) + "</tr>"
        hm += "</table>"
        halves_html = "".join(
            f'<div class="hcol"><div class="hl">½{k+1} · mes. {m + k//2}, temps {"1-2" if k % 2 == 0 else "3-4"}</div>'
            f"{half_block(h)}</div>" for k, h in enumerate(hs))
        hidx = [2 * b0 + k for k in range(4)]
        diff_html = "".join(
            f'<div class="hcol"><div class="hl">½{k+1} · mes. {m + k//2}, t{"1-2" if k % 2 == 0 else "3-4"}</div>'
            f'<div class="dh">← moins la précédente'
            f'{" · TV " + format(H[h]["tv_tri"], ".2f") if H[h]["tv_tri"] is not None else ""}</div>'
            f'{delta_rows(H[h], H[h-1]) if h > 0 else ""}'
            f'<div class="dh">→ moins la suivante</div>'
            f'{delta_rows(H[h], H[h+1]) if h + 1 < len(H) else ""}</div>'
            for k, h in enumerate(hidx))
        diff_block = (f'<details class="dd"{" open" if name.startswith("A1") else ""}>'
                      f'<summary>Différences avec les demi-mesures voisines</summary>'
                      f'<div class="halves">{diff_html}</div></details>')
        cell_html.append(
            f'<section class="cell" id="c{m}"><div class="ch"><b>{name}</b> '
            f'<span class="ms">mes. {m}–{m+1}</span> <span class="chords">{html.escape(chords)}</span>'
            f'<button class="play" data-t0="{t0:.3f}" data-t1="{t1:.3f}">▶ écouter</button></div>'
            f'<div class="hmwrap"><div class="cap">notes d\'accord attendues, ligne = demi-mesure</div>{hm}</div>'
            f'<div class="halves">{halves_html}</div>{diff_block}</section>')

    # bandes « où ça change », une par section
    strips = []
    for nom, m0 in (("A1", 13), ("A2", 21), ("B", 29)):
        b0 = m0 - 1
        ch = " | ".join(chord_text(bars, b0 + i) for i in range(8))
        strips.append(
            f'<section class="cell"><div class="ch"><b>{nom}</b> <span class="ms">mes. {m0}–{m0+7}</span>'
            f'<button class="play" data-t0="{grid[b0]:.3f}" data-t1="{grid[b0+8]:.3f}">▶ écouter</button></div>'
            f'<div class="cap chords">{html.escape(ch)}</div>'
            f'{strip_svg(H, b0, 8, bars, grid)}</section>')
    iso = []
    for nom, m0 in (("A1", 13), ("A2", 21), ("B", 29)):
        b0 = m0 - 1
        ch = " | ".join(chord_text(bars, b0 + i) for i in range(8))
        iso.append(
            f'<section class="cell"><div class="ch"><b>{nom}</b> <span class="ms">mes. {m0}–{m0+7}</span>'
            f'<button class="play" data-t0="{grid[b0]:.3f}" data-t1="{grid[b0+8]:.3f}">▶ écouter</button></div>'
            f'<div class="cap chords">{html.escape(ch)}</div>'
            f'{iso_strip_svg(H, b0, 8, grid)}'
            f'<details class="dd"{" open" if nom == "A1" else ""}><summary>Les 16 demi-mesures, voisine d\'avant et voisine d\'après</summary>'
            f'<div class="halves">{iso_cards(H, b0, 8, bars)}</div></details></section>')
    ext_tbl = "".join(
        f"<tr><td><b>{s['nom']}</b><div class='mut'>{' / '.join(s['cols'])}</div></td>"
        f"<td>{100*s['mean']:.0f} %</td>"
        f"<td>{100*s['hmax']:.0f} % <span class='mut'>({s['hcol']}, mes. {s['hbar']} t{s['htemps']})</span></td>"
        f"<td>{100*s['fmax']:.0f} % <span class='mut'>({'mes. ' + str(s['fbar']) if s['fbar'] < len(grid) else 'après la dernière mesure'})</span></td>"
        f"<td>{s['hwin']} / {s['nh']}</td></tr>" for s in ext_stats)

    labs12 = [n for _, n in CELLS]
    labsA1 = labs12[:4]

    def sub(S, ids):
        return [[float(S[i, k]) for k in ids] for i in ids]

    cand_html = []
    table_rows = []
    for r in res:
        mot = r["mot"]
        lA1 = "".join(mot[j] for j in A1)
        lA2 = "".join(mot[j] for j in A2)
        lB = "".join(mot[j] for j in idx[8:])
        w1, w2, fo = r["w1"], r["w2"], r["forced"]

        def win(w):
            return (f'<span class="ok">oui, seuil entre {w[0]:.2f} et {w[1]:.2f}</span>'
                    if w else '<span class="ko">non, aucun seuil</span>')
        says = verdict(r, lA1, lA2, w1, w2)
        r["says"] = says
        if fo:
            fm = fo[1]
            fA1 = "".join(fm[j] for j in A1); fA2 = "".join(fm[j] for j in A2)
            fB = "".join(fm[j] for j in idx[8:])
            ftxt = f"seuil {fo[0]:.2f} : <span class='mono'>{fA1} · {fA2} · {fB}</span>"
        else:
            ftxt = "—"
        r["ftxt"] = ftxt
        table_rows.append(
            f"<tr><td>{r['title']}</td><td class='mono'>{lA1} · {lA2}</td>"
            f"<td>{r['oA1']+1} · {r['oA2']+1}</td>"
            f"<td>{'oui' if w1 else 'non'} · {'oui' if w2 else 'non'}</td><td>{ftxt}</td></tr>")
        forced_block = (f"""<div class="kv">Au seuil {fo[0]:.2f} (milieu de la fenêtre{' commune à A1 et A2' if (w2 and fo[0] >= w2[0] and fo[0] <= w2[1]) else ' de A1'}), le morceau entier s'écrit
 <b class="mono">{fo[1]}</b> — A1 <b class="mono">{fA1}</b>, A2 <b class="mono">{fA2}</b>, B <b class="mono">{fB}</b>.</div>""" if fo else "")
        cand_html.append(f"""
<section class="cand" id="k{r['key']}">
<h3>{r['title']}</h3>
<div class="formula">{r['formula']}</div>
<p>{r['sentence']}</p>
<p class="says"><b>Ce que ça dit :</b> {says}</p>
<div class="kv">Règle actuelle — seuil = paire identique la moins ressemblante du morceau : <b>{r['thr']:.3f}</b>
 → A1 <b class="mono">{lA1}</b> · A2 <b class="mono">{lA2}</b> · B <b class="mono">{lB}</b>.</div>
<div class="kv">Existe-t-il un seuil qui donne <b>a a b a</b> (même fusion que l'outil) ? A1 : {win(w1)} · A2 : {win(w2)}.</div>
{forced_block}
<div class="cap">A1, 4 × 4 — plus foncé = plus ressemblant ; encadré = au-dessus du seuil a a b a. Une cellule ne rejoint un groupe que si TOUTES ses paires passent le seuil.</div>
<div class="scroll">{matrix_html(sub(r['S'], A1), labsA1, fo[0] if fo else None)}</div>
<details><summary>Les 12 cellules (A1, A2, B)</summary>
<div class="scroll">{matrix_html(sub(r['S'], idx), labs12, fo[0] if fo else None)}</div></details>
<details><summary>Le mot du morceau entier (21 bi-mesures)</summary><p class="mono">{mot}</p></details>
</section>""")

    summary = SUMMARY(res, A1, A2, ext_stats, dec, H)
    page = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Produit scalaire des lettres</title>
<style>
:root{{--bg:#fbfaf7;--fg:#1d1d1f;--mut:#6b6b70;--line:#e3e1dc;--acc:#2a6fdb;--ok:#1d8a4e;--ko:#c0392b;--card:#fff}}
@media (prefers-color-scheme:dark){{:root{{--bg:#151517;--fg:#ececef;--mut:#9a9aa2;--line:#2c2c31;--acc:#6ea0ff;--ok:#4cc38a;--ko:#ff7a6b;--card:#1d1d21}}}}
*{{box-sizing:border-box}} html,body{{margin:0;overflow-x:hidden}}
body{{background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,system-ui,sans-serif;padding:16px;max-width:980px;margin:auto}}
h1{{font-size:20px;margin:4px 0 8px}} h2{{font-size:17px;margin:26px 0 8px}} h3{{font-size:15px;margin:0 0 6px}}
.sum{{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--acc);border-radius:10px;padding:10px 12px}}
.sum p{{margin:4px 0}} .mut{{color:var(--mut);font-size:13px}}
.mono{{font-family:ui-monospace,Menlo,monospace;letter-spacing:.08em}}
.scroll{{overflow-x:auto;max-width:100%}}
table{{border-collapse:collapse}} .tbl td,.tbl th{{border-bottom:1px solid var(--line);padding:4px 6px;font-size:13px;text-align:left}}
.cell,.cand{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px;margin:10px 0}}
.ch{{display:flex;flex-wrap:wrap;gap:6px;align-items:center;margin-bottom:6px}}
.ms{{color:var(--mut);font-size:13px}} .chords{{font-weight:600}}
.play{{margin-left:auto;font:inherit;font-size:14px;padding:6px 12px;border-radius:16px;border:1px solid var(--acc);background:transparent;color:var(--acc)}}
.play.on{{background:var(--acc);color:#fff}}
.halves{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:6px}}
@media (max-width:640px){{.halves{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
.hcol{{min-width:0;border-top:1px solid var(--line);padding-top:4px}} .hl{{font-size:11px;color:var(--mut)}}
.r{{display:grid;grid-template-columns:3.6em 1fr 2em;gap:4px;align-items:center;font-size:12px}}
.n{{font-weight:600;white-space:nowrap;overflow:hidden}} .v{{text-align:right;color:var(--mut)}}
.bar{{height:8px;background:var(--line);border-radius:4px;overflow:hidden}} .bar i{{display:block;height:100%;background:var(--acc)}}
.sub{{font-size:11px;color:var(--mut)}}
.stk{{display:flex;height:8px;border-radius:4px;overflow:hidden;background:var(--line)}} .stk i{{display:block;height:100%}}
.ex .n{{font-weight:400;color:var(--mut)}} .r.ex{{grid-template-columns:2.6em 1fr 3.4em}}
.dh{{font-size:11px;color:var(--mut);margin-top:4px}}
.r.dr{{grid-template-columns:4.4em 1fr 2.2em}} .kd{{font-weight:400;color:var(--mut);font-size:10px}}
.dv{{position:relative;height:8px;background:linear-gradient(to right,transparent calc(50% - .5px),var(--mut) calc(50% - .5px),var(--mut) calc(50% + .5px),transparent calc(50% + .5px)),var(--line);border-radius:4px}}
.dv i{{position:absolute;top:0;height:100%}} .dv i.up{{background:var(--ok)}} .dv i.dn{{background:var(--ko)}}
.pr{{margin:3px 0}} .pn{{display:flex;justify-content:space-between;font-size:11px}}
.pb{{display:grid;grid-template-columns:1fr 1fr;gap:3px}}
.dd summary{{margin:6px 0}}
.strip{{width:100%;height:auto;display:block;margin-top:4px}} .strip .sx{{font-size:9px;fill:var(--mut);text-anchor:middle}}
.strip .sl{{font-size:9px;fill:var(--mut)}} .strip .sv{{font-size:8px;fill:var(--fg);text-anchor:middle}}
.hmwrap{{margin:4px 0 8px}} .cap{{font-size:11px;color:var(--mut);margin:2px 0}}
.hm{{width:100%;table-layout:fixed}} .hm th{{font-size:10px;font-weight:400;color:var(--mut);padding:0}} .hm td{{height:14px;border:1px solid var(--bg)}}
.hm tr th:first-child{{width:1.4em}}
.formula{{font-family:ui-monospace,Menlo,monospace;font-size:12.5px;background:var(--bg);border-radius:6px;padding:6px 8px;overflow-x:auto}}
.says{{margin:6px 0}} .kv{{font-size:13px;margin:4px 0}}
.ok{{color:var(--ok);font-weight:600}} .ko{{color:var(--ko);font-weight:600}}
.mx td,.mx th{{font-size:11px;padding:2px 4px;text-align:right;font-variant-numeric:tabular-nums}}
.mx th{{color:var(--mut);font-weight:400}} .mx td.ge{{outline:2px solid var(--ok);outline-offset:-2px;font-weight:700}} .mx td.dg{{color:var(--mut)}}
details{{margin-top:6px}} summary{{font-size:13px;color:var(--acc)}}
</style></head><body>
<h1>Quel produit scalaire pour les petites lettres ? — Evening Star</h1>
<div class="sum">{summary}</div>

<h2>1. Les 6te, 9e, 11e, 13e : ce que musx en dit</h2>
<p class="mut">musx sort, à chaque instant, un vote par « étage » de l'accord, en plus de la triade et de la basse. Colonnes lues dans le code de musx (pas devinées) ; le 6 est rangé dans l'étage 13e. « Proba. d'une extension » = 1 − p(non).</p>
<div class="scroll"><table class="tbl"><tr><th>étage (colonnes)</th><th>moyenne sur le morceau</th><th>max sur une demi-mesure</th><th>max sur un instant (23 ms)</th><th>demi-mesures où l'extension l'emporte sur « non »</th></tr>
{ext_tbl}</table></div>
<p class="mut">Le décodeur de production, relancé ici sur ce morceau : {dec['n']} accords écrits, dont {dec['n7']} avec une 7e et <b>{dec['n_ext']}</b> avec 6, 9, 11 ou 13. Il écrirait C9 plutôt que C7 dès que p(9) &gt; p(non) sur la durée de l'accord ; ça n'arrive jamais ici.</p>

<h2>2. Où ça change : les différences entre demi-mesures voisines</h2>
<p class="mut"><b>Formule</b> — à chaque frontière entre la demi-mesure h−1 et h : TV = ½ Σ<sub>i</sub> |p<sub>i</sub>(h) − p<sub>i</sub>(h−1)|, calculée séparément pour l'accord (73 triades), la basse (13 notes) et les extensions (moyenne des 4 étages). 0 = même postérieure, 1 = plus rien en commun. Chaque barre est posée sur la frontière (début de la demi-mesure h) ; le nombre = % de la probabilité qui a changé. Bandes grisées = bi-mesures (les cellules des lettres).</p>
{''.join(strips)}

<h2>2 bis. Les deux voisines à la fois</h2>
<p class="mut"><b>Formule</b> — pour chaque demi-mesure h : TV<sub>avant</sub> = ½ Σ|p(h) − p(h−1)|, TV<sub>après</sub> = ½ Σ|p(h) − p(h+1)|, et <b>isolement = min(TV<sub>avant</sub>, TV<sub>après</sub>)</b>. Les voisines viennent du morceau entier (la mesure 13 regarde la 12, la 20 regarde la 21).
<b>Lecture</b> : isolement petit = elle ressemble à au moins une voisine (elle fait partie d'un accord tenu) ; isolement grand = elle ne ressemble à aucune des deux (accord d'une demi-mesure seulement, ou accident de musx). Bandes : barre pleine = isolement, barre claire = la plus éloignée des deux voisines ; le nombre = isolement en %. Même trois plans que la section 2.</p>
<p class="mut">Dans le détail de chaque demi-mesure : à gauche p(h) − p(avant), à droite p(h) − p(après), mêmes entrées des deux côtés ; vert = plus de ça ici que chez la voisine, rouge = moins. Les deux nombres sont en points de %.</p>
{''.join(iso)}
<div class="sum">{ISO_NOTE}</div>

<h2>3. Les candidats « produit scalaire » en un tableau</h2>
<p class="mut">Gardé pour information. Lettres données par la règle actuelle (accords identiques ⇒ même lettre ; deux groupes se rejoignent si toutes leurs paires dépassent la paire identique la moins ressemblante du morceau). La colonne « a a b a possible ? » visait une cible abandonnée : Louis a réentendu, A n'est pas a a b a.</p>
<div class="scroll"><table class="tbl"><tr><th>définition</th><th>lettres, règle actuelle (A1 · A2)</th><th>intrus (A1 · A2)</th><th>a a b a possible ? (A1 · A2)</th><th>avec ce seuil : A1 · A2 · B</th></tr>
{''.join(table_rows)}</table></div>

<h2>4. Les postérieures, demi-mesure par demi-mesure</h2>
<p class="mut">Une « demi-mesure » = temps 1-2 ou 3-4 (4 temps par mesure, vérifié sur la grille). Pour chacune : les 5 accords les plus probables de musx (en %), les 3 basses, puis une barre par étage — <b>7e</b> (non / 7M / ♭7 / ♭♭7), <b>9e</b> (non / 9 / ♯9 / ♭9), <b>11e</b> (non / 11 / ♯11), <b>6te</b> = 13e (non / 6 / ♭6) : gris = « non », couleur = l'extension, chiffre = la plus probable. Sous « Différences » : p(cette demi-mesure) − p(voisine), barre à droite (verte) = présent ici et pas à côté, à gauche (rouge) = l'inverse ; 3 triades, la basse, l'extension qui bougent le plus. Au-dessus, les notes d'accord attendues (Σ probabilité × notes de l'accord), une ligne par demi-mesure. ▶ joue exactement les 2 mesures.</p>
{''.join(cell_html)}

<h2>5. Les définitions candidates (information)</h2>
<p class="mut">Toutes « alignées » : on compare la demi-mesure k de X à la demi-mesure k de Y, puis on fait la moyenne des 4.</p>
{''.join(cand_html)}

<audio id="au" preload="none"></audio>
<script>
(function(){{
  var au=document.getElementById('au'), src='../audio/{STEM}.m4a', blobUrl=null, stopAt=null, cur=null;
  // iPhone : fetch()+blob (le chargeur média de la PWA ne bufferise pas), pas de boucle rAF
  fetch(src).then(function(r){{return r.blob();}}).then(function(b){{blobUrl=URL.createObjectURL(b);au.src=blobUrl;}}).catch(function(){{au.src=src;}});
  au.addEventListener('timeupdate',function(){{ if(stopAt!==null && au.currentTime>=stopAt){{au.pause();}} }});
  au.addEventListener('pause',function(){{ if(cur){{cur.classList.remove('on');cur=null;}} stopAt=null; }});
  document.querySelectorAll('.play').forEach(function(b){{
    b.addEventListener('click',function(){{
      if(cur===b){{au.pause();return;}}
      if(!au.src) au.src=src;
      if(cur) cur.classList.remove('on');
      cur=b; b.classList.add('on');
      var t0=parseFloat(b.dataset.t0); stopAt=parseFloat(b.dataset.t1);
      try{{au.currentTime=t0;}}catch(e){{}}
      var p=au.play(); if(p&&p.catch) p.catch(function(){{}});
    }});
  }});
}})();
</script>
</body></html>"""
    OUT.write_text(page)
    print("écrit", OUT)


#: Une phrase musicale par candidat (écrite après avoir lu les matrices ;
#: les chiffres de la phrase sont recalculés, pas recopiés).
NOTE = {
    "auj": "F/C et F ont les mêmes notes (fa la do) : la cellule 3 colle à la 1 ; c'est Gm, à la place de C7, qui fait de la 2 l'intruse.",
    "a": "En nommant l'accord exact, la 2 (Gm sur toute la mesure) s'écarte le plus ; la 3 reste proche de la 1 et de la 4.",
    "b": "Par la fondamentale, G7 C7 (cellule 3) et Gm7 C7 (cellule 4) sont presque la même chose : la 3 se fond dans le a.",
    "c": "En notes d'accord, F/C = F et G7 ≈ Gm : rien ne distingue la 3 ; la 2 reste la plus éloignée.",
    "c7": "Ajouter la septième rapproche tout (C7, G7, Gm7 partagent des notes) : même verdict que (c), en plus serré.",
    "d": "Seule la basse entend le do de F/C (mes. 17) : la cellule 3 descend vers do là où les autres restent sur fa.",
    "e": "Les deux tirent en sens contraire — les notes écartent la 2, la basse écarte la 3 — et se neutralisent presque.",
    "fa": "√p donne plus de poids aux hésitations, mais le classement ne change pas : la 2 reste l'intruse.",
    "fe": "Même neutralisation que (e) : la basse ne pèse pas assez contre les notes d'accord.",
}


ISO_NOTE = """<p><b>Ce que ça montre.</b> Presque toutes les demi-mesures ont un isolement d'environ 40 % (accord) : même dans un accord tenu tout au long de la mesure (F mes. 13, C7 mes. 14, Gm mes. 16), les deux moitiés ne se ressemblent qu'à 60 %. C'est le flottement de musx, et il est aussi fort que la différence entre deux vrais accords voisins (F → C7 : 61 %).</p>
<p>Donc <b>le min ne sépare pas un vrai changement du flottement</b> : une demi-mesure au bord d'un changement garde une voisine identique, et son min retombe à 40 %. Il sépare autre chose : l'accord d'une demi-mesure seulement (G7 | C7 mes. 18, Gm7 | C7 mes. 20 : 87–91 % et 75–79 %) contre l'accord franchement tenu (F/C mes. 17 : 19 % ; D7 mes. 32 : 2–6 %).</p>
<p>La basse est plus nette que l'accord : sur F/C (mes. 17, 25) son isolement tombe à 4–5 %, là où l'accord reste à 19–25 % ; c'est elle qui dit « rien n'a bougé ».</p>"""


def verdict(r, lA1, lA2, w1, w2):
    o1, o2 = r["oA1"] + 1, r["oA2"] + 1
    s = NOTE.get(r["key"], "") + f" Intrus : cellule {o1} sur A1, {o2} sur A2."
    if w1 and w2:
        s += " Un seuil donne a a b a sur les deux A."
    elif w1:
        s += " Un seuil donne a a b a sur A1, pas sur A2."
    else:
        s += " Aucun seuil ne donne a a b a."
    return s


def SUMMARY(res, A1, A2, ext_stats, dec, H):
    E = {e["k"]: e for e in ext_stats}
    nh = len(H)
    def mx(k):
        e = E[k]
        return f"{100*e['hmax']:.0f} % (mes. {e['hbar']}, temps {e['htemps']})"
    l0 = ("<p><b>Correction du 12/10 :</b> A n'est pas « a a b a » — Louis y a réentendu des "
          "différences. La page ne cherche plus le candidat qui donne a a b a ; les matrices "
          "restent en bas, pour information.</p>")
    l1 = ("<p><b>Les 6te, 9e, 11e : musx ne les entend jamais assez fort.</b> "
          f"Sur une demi-mesure, la 9e monte au plus à {mx('s9')}, la 6te à {mx('s13')}, "
          f"la 11e à {mx('s11')}. Sur aucune des {nh} demi-mesures elles ne passent devant « non ». "
          f"Ce n'est pas l'aval qui les jette : le décodeur, relancé ici, écrit <b>{dec['n_ext']}</b> "
          f"accord avec 6, 9, 11 ou 13 sur {dec['n']} — il les écrirait si musx votait plus pour "
          f"l'extension que pour « non ». Seule la 7e est vraiment là (elle l'emporte sur "
          f"{E['s7']['hwin']} demi-mesures sur {nh}).</p>")
    l2 = ("<p><b>Où ça change, sur A1 (section 2) :</b> les 4 bi-mesures démarrent toutes par un "
          "changement net au temps 1 vers F (≈ 90 % de la probabilité bouge). Elles diffèrent par la "
          "2e mesure : C7 (basse sol → do au milieu), Gm (basse sol → si♭ au milieu), G7 C7 et Gm7 C7 "
          "(un 2e changement net au temps 3). Mes. 17 est la seule mesure immobile : la basse reste "
          "sur do toute la mesure (F/C, 4 % bouge au milieu).</p>")
    thr = [r["thr"] for r in res]
    l3 = ("<p class='mut'>Rappel des candidats (section 3) : seule la tête « basse » isolait la cellule 3 ; "
          f"la règle de seuil actuelle ({min(thr):.2f}–{max(thr):.2f}) ne fusionne rien quelle que soit la "
          "définition. Un seul morceau : hypothèse, pas loi.</p>")
    return l0 + l1 + l2 + l3


if __name__ == "__main__":
    main()
