"""Expérience (Louis, 2026-10-12, Evening Star) : le décodeur nourri demi-mesure
par demi-mesure avec « postérieure + différence aux deux voisines », et la
question du diminué de la mesure 18.

    .venv/bin/python -m tools.demi_mesures_delta_decodeur

Écrit docs/plots/demi_mesures_delta_decodeur_2026-10-12.html. Ne touche à
aucun code de pipeline ni à aucune constante : on réutilise le décodeur de
production (`musx.redecode` → `XHMMDecoder.decode` vendu) en lui donnant une
suite de trames CONSTANTE PAR MORCEAUX — chaque trame d'une demi-mesure porte
la même postérieure q(h). Le décodeur fait Σ_trames log q·(produit des 6
plans), donc à l'intérieur d'une demi-mesure il ne change jamais d'accord
(aucun gain, une pénalité) : la grille effective est bien 2 cases par mesure.

Définitions (pour chaque plan : triade 73, basse 13, 7e 4, 9e 4, 11e 3, 13e 3) :
    p(h)  = moyenne des trames musx sur la demi-mesure h
    S(h)  = 2 p(h) − p(h−1) − p(h+1)        (Δavant + Δaprès)
    q_λ   = norm( max(0, p + λ S) + ε ),     λ ∈ {0, ½, 1}
    q_S⁺  = norm( max(0, S) + ε )           (= limite λ → ∞)
ε = 1e-3 : le décodeur prend des log, une case à 0 serait −∞.
"""
from __future__ import annotations

import html
import json

import numpy as np

from harmonia import cache, musx, soudure
from harmonia.musx import FRAME_DT
from harmonia.nnls_features import extract_bothchroma
from harmonia.settings import SETTINGS

STEM = "CHVWPIFAT5A"
CHART = SETTINGS.repo / "state/cache/charts/min_CHVWPIFAT5A.json"
OUT = SETTINGS.repo / "docs/plots/demi_mesures_delta_decodeur_2026-10-12.html"
PLANES = ("triad", "bass", "s7", "s9", "s11", "s13")
EPS = 1e-3

NOTES = "C D♭ D E♭ E F G♭ G A♭ A B♭ B".split()
QUAL = ["", "m", "sus4", "sus2", "dim", "aug"]   # triad_idx 0..5 ; dim = 4
VARIANTS = [("l0", "λ = 0", "q = p (postérieure seule, moyennée sur la demi-mesure)"),
            ("l05", "λ = ½", "q = norm(max(0, p + ½·S))"),
            ("l1", "λ = 1", "q = norm(max(0, p + S)) = norm(max(0, 3p − p<sub>avant</sub> − p<sub>après</sub>))"),
            ("sp", "S⁺", "q = norm(max(0, S)) — la différence seule, partie positive"),
            ("l1tb", "λ = 1, tri+basse", "comme λ = 1 sur la triade et la basse ; 7e/9e/11e/13e : p seul"),
            ("sptb", "S⁺, tri+basse", "comme S⁺ sur la triade et la basse ; 7e/9e/11e/13e : p seul")]
SECTIONS = [("A1", 13, 20), ("A2", 21, 28), ("B", 29, 36)]
# Les cellules de A selon Louis (2 mesures chacune) — référence musicale, pas vérité.
LOUIS_A = ["F", "C7", "F", "Gm", "F/C", "G7 C7", "F C", "Gm7 C7"]


def lab_name(lab: str) -> str:
    """'G:7' → 'G7', 'F:maj/5' → 'F/C', 'B:dim7' → 'B°7'."""
    if lab == "N":
        return "N.C."
    r, _, q = lab.partition(":")
    names = {"C": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3, "E": 4, "F": 5, "F#": 6,
             "Gb": 6, "G": 7, "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11}
    root = names[r]
    bass = ""
    if "/" in q:
        q, _, b = q.partition("/")
        iv = {"3": 4, "b3": 3, "5": 7, "b7": 10, "2": 2}[b]
        bass = "/" + NOTES[(root + iv) % 12]
    suf = {"maj": "", "min": "m", "7": "7", "maj7": "maj7", "min7": "m7", "dim": "°",
           "dim7": "°7", "hdim7": "ø7", "aug": "+", "sus4": "sus4", "sus2": "sus2",
           "sus4(b7)": "7sus4", "9": "9", "maj9": "maj9", "min9": "m9", "11": "11",
           "13": "13", "maj(6)": "6", "min(6)": "m6"}.get(q, q)
    return NOTES[root] + suf + bass


def chart_name(c) -> str:
    if c.get("nc"):
        return "N.C."
    s = NOTES[int(c["root"]) % 12] + {"-": "m", "-7": "m7", "^7": "maj7", "o": "°",
                                       "-7b5": "ø7", "o7": "°7"}.get(c["q"], c["q"])
    if int(c.get("bass", -1)) >= 0:
        s += "/" + NOTES[int(c["bass"]) % 12]
    return s


def main():
    chart = json.loads(CHART.read_text())
    audio = (SETTINGS.audio_dir / f"{STEM}.m4a").resolve()
    z = cache.load_npz("musx_probs", audio)
    P = [np.asarray(z[k], dtype=np.float64) for k in PLANES]
    T = P[0].shape[0]
    grid = chart["barGrid"]
    bt = np.asarray(chart["beatTimes"])
    n = len(grid) - 1
    assert chart["bpb"] == 4

    halves, beats_of = [], []
    for b in range(n):
        t0, t1 = grid[b], grid[b + 1]
        bb = bt[(bt >= t0 - 0.05) & (bt < t1 - 0.05)]
        assert len(bb) == 4, (b, bb)
        halves += [(t0, float(bb[2])), (float(bb[2]), t1)]
        beats_of.append(list(map(float, bb)) + [t1])
    H = len(halves)
    fr = [(int(round(a / FRAME_DT)), int(round(c / FRAME_DT))) for a, c in halves]
    p = [np.stack([X[a:c].mean(0) for a, c in fr]) for X in P]          # (H, d) par plan

    def S_of(X):
        S = np.zeros_like(X)
        S[1:-1] = 2 * X[1:-1] - X[:-2] - X[2:]
        S[0] = 2 * (X[0] - X[1]); S[-1] = 2 * (X[-1] - X[-2])           # bords : une voisine, comptée 2×
        return S
    S = [S_of(X) for X in p]

    def norm(Y):
        Y = np.maximum(0, Y) + EPS
        return Y / Y.sum(1, keepdims=True)
    Q = {"l0": [norm(X) for X in p],
         "l05": [norm(X + 0.5 * s) for X, s in zip(p, S)],
         "l1": [norm(X + s) for X, s in zip(p, S)],
         "sp": [norm(s) for s in S]}
    Q["l1tb"] = Q["l1"][:2] + Q["l0"][2:]
    Q["sptb"] = Q["sp"][:2] + Q["l0"][2:]

    with musx._InMusxDir():
        hmm = musx._decoder(musx.DEFAULT_PENALTY)

    def frames_of(qs):
        """Suite de trames constante par demi-mesure ; hors grille (avant la
        mesure 1, après la dernière) : les trames brutes, inchangées."""
        out = [X.copy() for X in P]
        for i, (a, c) in enumerate(fr):
            for k in range(6):
                out[k][a:c] = qs[k][i]
        return out

    def decode(plist):
        lab = musx.redecode(chart["beatTimes"], plist, downbeat_times=grid[:-1],
                            quarter_beats="all")   # comme le pipeline (SETTINGS.quarter_bar)
        mids = [(a + c) / 2 for a, c in halves]
        res = []
        for m in mids:
            res.append(next(l for t0, t1, l in lab if t0 <= m < t1))
        return res, lab

    def top3(qs, i):
        names, lp = hmm.get_chord_tag_obs([q[i:i + 1] for q in qs])
        lp = lp[0]
        w = np.exp(lp - lp.max()); w /= w.sum()
        o = np.argsort(-w)[:3]
        return [(lab_name(names[j]), float(w[j])) for j in o]

    DEC, TOP = {}, {}
    for key, *_ in VARIANTS:
        DEC[key], _ = decode(frames_of(Q[key]))
        TOP[key] = [top3(Q[key], i) for i in range(H)]
    raw, raw_lab = decode(P)            # trames brutes : le décodage de prod, avant repli
    DEC["raw"] = raw

    # texte du chart (après repli) par demi-mesure
    bars = soudure.accords_par_mesure(chart)
    chart_h = []
    for b in range(n):
        for half in (0, 1):
            lo, hi = 2 * half, 2 * half + 2
            cs = [c for c in bars[b] if lo <= int(c.get("beat", 0)) < hi]
            if not cs or int(cs[0].get("beat", 0)) != lo:
                prev = [c for c in bars[b] if int(c.get("beat", 0)) < lo]
                if prev:
                    cs = [prev[-1]] + cs
                elif not cs and b > 0 and bars[b - 1]:
                    cs = [bars[b - 1][-1]]
            chart_h.append(" → ".join(chart_name(c) for c in cs) or "–")

    # ── TRAME DE TENUE : un accord tenu sur ≥ 3 demi-mesures garde-t-il son nom ? ──
    held = []
    for i in range(1, H - 1):
        r0 = DEC["l0"]
        if r0[i - 1] == r0[i] == r0[i + 1] and r0[i] != "N":
            held.append(i)
    held_stats = {}
    for key, *_ in VARIANTS:
        keep = sum(DEC[key][i] == DEC["l0"][i] for i in held)
        held_stats[key] = (keep, len(held))
    print("demi-mesures « tenues » (même accord brut avant/ici/après) :", len(held))
    for key, *_ in VARIANTS:
        print(f"  {key:4s} garde le nom sur {held_stats[key][0]}/{held_stats[key][1]}")

    # masse de S⁺ sur les tenues : combien de « signal » reste-t-il ?
    tri_mass = {"tenue": float(np.mean([np.maximum(0, S[0][i]).sum() for i in held])),
                "toutes": float(np.mean([np.maximum(0, S[0][i]).sum() for i in range(1, H - 1)]))}
    print("masse S⁺ triade moyenne :", tri_mass)

    for sec, b0, b1 in SECTIONS:
        print("──", sec)
        for i in range(2 * (b0 - 1), 2 * b1):
            print(f"{i // 2 + 1:3d}{'ab'[i % 2]} chart={chart_h[i]:10s} brut={lab_name(raw[i]):7s}",
                  " ".join(f"{k}={lab_name(DEC[k][i]):7s}" for k, *_ in VARIANTS))

    # ── TÂCHE 2 : la mesure 18 ─────────────────────────────────────────────
    dim_cols = 1 + 4 * 12 + np.arange(12)
    m18 = {}
    for i in range(2 * 16, 2 * 19):                 # mesures 17..19
        tri = p[0][i]; bas = p[1][i]
        rank = np.argsort(-tri)
        dims = sorted(((NOTES[r] + "°", float(tri[dim_cols[r]])) for r in range(12)),
                      key=lambda x: -x[1])
        g7_rank = int(np.where(rank == 1 + 7)[0][0]) + 1        # G majeur (triade) ; 7e dans s7
        best_dim_rank = int(min(np.where(rank == c)[0][0] for c in dim_cols)) + 1
        names, lp = hmm.get_chord_tag_obs([X[i:i + 1] for X in p])
        lp = lp[0]; w = np.exp(lp - lp.max()); w /= w.sum()
        order = np.argsort(-w)
        lab_rank = {names[j]: r + 1 for r, j in enumerate(order)}
        dimlabs = sorted(((lab_name(nm), float(w[j]), lab_rank[nm]) for j, nm in enumerate(names)
                          if nm.endswith((":dim", ":dim7", ":hdim7"))), key=lambda x: -x[1])[:4]
        m18[i] = dict(
            top5=[(("N.C." if c == 0 else NOTES[(c - 1) % 12] + QUAL[(c - 1) // 12]), float(tri[c]))
                  for c in rank[:5]],
            dims=dims[:4], dim_total=float(tri[dim_cols].sum()), best_dim_rank=best_dim_rank,
            g_rank=g7_rank,
            bass=sorted(((("aucune" if k == 0 else NOTES[k - 1]), float(bas[k])) for k in range(13)),
                        key=lambda x: -x[1])[:4],
            s7=[float(x) for x in p[2][i]], s9=[float(x) for x in p[3][i]],
            dec_top=[(lab_name(names[j]), float(w[j])) for j in order[:5]],
            dec_dims=dimlabs,
            Sdim=float(S[0][i][dim_cols].sum()), Sdim_best=max(
                ((NOTES[r] + "°", float(S[0][i][dim_cols[r]])) for r in range(12)), key=lambda x: x[1]),
            s7_bb7_S=float(S[2][i][3]), s9_b9_S=float(S[3][i][3]))
        d = m18[i]
        print(f"mes {i // 2 + 1}{'ab'[i % 2]} top={d['top5'][:3]} dim tot={d['dim_total']:.4f} "
              f"best dims={d['dims'][:2]} rang dim={d['best_dim_rank']} s7={np.round(d['s7'], 3)} "
              f"s9={np.round(d['s9'], 3)} basse={d['bass'][:3]} dec={d['dec_top'][:3]} decdims={d['dec_dims'][:2]} "
              f"S dim={d['Sdim']:+.4f} {d['Sdim_best']}")

    # trame par trame dans la mesure 18 : le max de la masse « dim » et de ♭9
    a18, c18 = int(round(grid[17] / FRAME_DT)), int(round(grid[18] / FRAME_DT))
    fdim = P[0][a18:c18][:, dim_cols].sum(1)
    fb9 = P[3][a18:c18][:, 3]
    fbb7 = P[2][a18:c18][:, 3]
    print(f"mes 18 trames : max dim {fdim.max():.4f} (t={(a18 + fdim.argmax()) * FRAME_DT:.2f}s), "
          f"max ♭9 {fb9.max():.4f}, max ♭♭7 {fbb7.max():.4f}")
    frame_curve = dict(t=[float((a18 + k) * FRAME_DT) for k in range(c18 - a18)],
                       dim=fdim.tolist(), b9=fb9.tolist(), bb7=fbb7.tolist(),
                       G=P[0][a18:c18][:, 1 + 7].tolist(), C=P[0][a18:c18][:, 1 + 0].tolist(),
                       F=P[0][a18:c18][:, 1 + 5].tolist())

    # chroma NNLS (audio brut, indépendant de musx) : temps par temps, mes. 17-19
    arr, times = extract_bothchroma(audio)
    # index 0 = A dans la sortie VAMP → on décale de −3 pour avoir index 0 = C.
    # Vérifié (règle n°1) : la mesure 19 (F au chart, F à 97 % chez musx) sort
    # fa/la/do en tête et basse fa ; avec +3 elle sortait mi♭/si/sol♭.
    bassc = np.roll(arr[:, :12], -3, axis=1)
    treb = np.roll(arr[:, 12:], -3, axis=1)
    TPL = {"G7": [7, 11, 2, 5], "C7": [0, 4, 7, 10], "F": [5, 9, 0], "Gm7": [7, 10, 2, 5],
           "G7♭9": [7, 11, 2, 5, 8],
           "°7 {G♯ B D F}": [8, 11, 2, 5], "°7 {F♯ A C E♭}": [6, 9, 0, 3],
           "°7 {C♯ E G B♭}": [1, 4, 7, 10], "G° (G B♭ D♭)": [7, 10, 1], "B° (B D F)": [11, 2, 5],
           "G7♭5 (G B D♭ F)": [7, 11, 1, 5], "Gø7 (G B♭ D♭ F)": [7, 10, 1, 5]}
    def score(v, pcs):
        t = np.zeros(12); t[pcs] = 1
        nv = np.linalg.norm(v)
        return float(v @ t / (nv * np.linalg.norm(t))) if nv > 1e-9 else 0.0
    chroma_rows = []
    for b in (16, 17, 18, 25):
        bb = beats_of[b]
        for k in range(8):                         # croches : 2 par temps
            k2, hlf = divmod(k, 2)
            a = bb[k2] + (bb[k2 + 1] - bb[k2]) * hlf / 2
            e = bb[k2] + (bb[k2 + 1] - bb[k2]) * (hlf + 1) / 2
            sel = (times >= a) & (times < e)
            tv = treb[sel].mean(0); bv = bassc[sel].mean(0)
            sc = {nm: score(tv, pcs) for nm, pcs in TPL.items()}
            chroma_rows.append(dict(bar=b + 1, beat=k2 + 1, et=hlf, t0=a, t1=e,
                                    treb=(tv / max(tv.max(), 1e-9)).tolist(),
                                    bass=NOTES[int(bv.argmax())],
                                    sc=sc))
            best = sorted(sc.items(), key=lambda x: -x[1])[:3]
            print(f"chroma mes {b + 1} t{k2 + 1}{'+' if hlf else ' '}: basse={NOTES[int(bv.argmax())]} "
                  f"notes={' '.join(NOTES[j] for j in np.argsort(-tv)[:5])} meilleurs={[(a, round(s, 3)) for a, s in best]}")

    # chroma par demi-mesure (pour juger les changements proposés par la différence)
    TPL2 = {"F": [5, 9, 0], "C": [0, 4, 7], "C7": [0, 4, 7, 10], "Gm": [7, 10, 2], "Gm7": [7, 10, 2, 5],
            "B♭m": [10, 1, 5], "B♭m7": [10, 1, 5, 8], "B♭": [10, 2, 5], "G7": [7, 11, 2, 5],
            "Fmaj7": [5, 9, 0, 4], "D7": [2, 6, 9, 0], "Aø7": [9, 0, 3, 7], "C/G": [0, 4, 7],
            "Gm6": [7, 10, 2, 4], "B♭m6": [10, 1, 5, 7], "E♭": [3, 7, 10], "Dm": [2, 5, 9]}
    half_chroma = []
    for i in range(H):
        a, c = halves[i]
        sel = (times >= a) & (times < c)
        tv = treb[sel].mean(0); bv = bassc[sel].mean(0)
        half_chroma.append(dict(notes=[NOTES[j] for j in np.argsort(-tv)[:5]],
                                rel=[float(tv[j] / max(tv.max(), 1e-9)) for j in np.argsort(-tv)[:5]],
                                bass=NOTES[int(bv.argmax())],
                                sc={nm: score(tv, pcs) for nm, pcs in TPL2.items()}))
    for i in (26, 27, 28, 29, 30, 31, 36, 37, 42, 43, 46, 47, 60, 61):
        hc = half_chroma[i]
        best = sorted(hc["sc"].items(), key=lambda x: -x[1])[:4]
        print(f"½chroma {i // 2 + 1}{'ab'[i % 2]} basse={hc['bass']} notes={hc['notes']} "
              f"{[(a, round(v, 2)) for a, v in best]} | l0={lab_name(DEC['l0'][i])} l1={lab_name(DEC['l1'][i])}")
        print("     top triade p:", [(("N" if c == 0 else NOTES[(c - 1) % 12] + QUAL[(c - 1) // 12]), round(float(p[0][i][c]), 3)) for c in np.argsort(-p[0][i])[:4]],
              " q_l1:", [(("N" if c == 0 else NOTES[(c - 1) % 12] + QUAL[(c - 1) // 12]), round(float(Q['l1'][0][i][c]), 3)) for c in np.argsort(-Q['l1'][0][i])[:4]])

    # le ré♭ du 1er temps : revient-il à chaque G7 en début de demi-mesure ?
    g7_occ = []
    for i in range(0, H):
        if DEC["l0"][i] not in ("G:7", "G:maj"):
            continue
        b = i // 2; k0 = 2 * (i % 2)
        bb = beats_of[b]
        sel = (times >= bb[k0]) & (times < bb[k0 + 1])
        sel2 = (times >= bb[k0 + 1]) & (times < bb[k0 + 2])
        v1 = treb[sel].mean(0); v1 = v1 / max(v1.max(), 1e-9)
        v2 = treb[sel2].mean(0); v2 = v2 / max(v2.max(), 1e-9)
        g7_occ.append(dict(bar=b + 1, half=i % 2, t0=bb[k0], t1=bb[k0 + 2],
                           db1=float(v1[1]), d1=float(v1[2]), db2=float(v2[1]), d2=float(v2[2]),
                           ab1=float(v1[8]), bb1=float(v1[10])))
        print(f"G7 mes {b + 1}{'ab'[i % 2]} temps1 ré♭={v1[1]:.2f} ré={v1[2]:.2f} la♭={v1[8]:.2f} si♭={v1[10]:.2f} | temps2 ré♭={v2[1]:.2f} ré={v2[2]:.2f}")

    data = dict(H=H, half_chroma=half_chroma, g7_occ=g7_occ, beats_of=beats_of, halves=halves, chart_h=chart_h, DEC={k: [lab_name(x) for x in v] for k, v in DEC.items()},
                TOP=TOP, held=held, held_stats=held_stats, tri_mass=tri_mass,
                m18=m18, frame_curve=frame_curve, chroma_rows=chroma_rows, grid=grid,
                Snorm=[0.5 * float(np.abs(S[0][i]).sum()) for i in range(H)],
                n_seg={k: int(sum(1 for i in range(1, H) if DEC[k][i] != DEC[k][i - 1])) for k in DEC})
    (SETTINGS.repo / "scratchpad").mkdir(exist_ok=True)
    (SETTINGS.repo / "scratchpad/demi_mesures_delta.json").write_text(json.dumps(data, default=float))
    write_page(data)
    return data



# ── page ─────────────────────────────────────────────────────────────────────

VKEYS = ["l0", "l05", "l1", "sp", "l1tb", "sptb"]
VSHORT = {"l0": "λ0", "l05": "λ½", "l1": "λ1", "sp": "S⁺", "l1tb": "λ1 t+b", "sptb": "S⁺ t+b"}

#: Lecture musicale des divergences (écrite après avoir lu chroma + postérieures
#: imprimés par main() ; « h » = indice de demi-mesure, 0 = mes. 1 t1-2).
#: verdict : "ok" = probablement juste, "?" = à écouter, "x" = artefact probable.
NOTES_DIV = {
    26: ("?", "Basse sol sur les deux temps, mais le do est la note la plus forte, avec fa et si♭ : "
              "C7sus4/G autant que Gm7. Un ii–V (Gm7 → C7) est plausible ; à écouter."),
    42: ("?", "Même endroit dans A2, même hésitation : basse sol, do et fa forts. Les deux A disent "
              "la même chose, ce qui est un indice pour Gm7 → C7."),
    31: ("ok", "Basse si♭, et ré♭ la note la plus forte (avec fa, si♭, sol) : B♭m6 = si♭ ré♭ fa sol. "
               "Le iv mineur, couleur classique de ballade. Mêmes notes qu'un Gø7 (demi-diminué sur sol)."),
    47: ("ok", "Même B♭m6 dans A2 (basse si♭, ré♭ fort). Deux A sur deux : ce n'est pas un hasard."),
    37: ("ok", "Le chart écrit déjà F → C (do au 4e temps). La cellule de Louis est « F C » : "
               "la différence met le C sur la demi-mesure, la postérieure seule l'oubliait."),
    29: ("x", "2e moitié d'une mesure de F : rien ne change à l'oreille (basse do, fa la do). "
              "C7 vient des étages 7e/9e, pas de la triade : avec « triade+basse » seulement, ça reste F/C."),
    45: ("x", "Même chose : F tenu, la différence invente un C quand on l'applique aussi aux extensions."),
    53: ("x", "Idem (F tenu, C6/C13 inventés par les extensions)."),
    60: ("?", "Basse la, notes ré, la♭, la, sol♭ : ni do ni mi♭, donc pas d'Aø7 dans le signal. "
              "Probable artefact, peut-être une note de passage."),
}


def _chord_cell(name, ref, top=None):
    cls = "same" if name == ref else "diff"
    return f'<td class="{cls}">{html.escape(name)}</td>'


def write_page(d):
    H = d["H"]; D = d["DEC"]; TOP = d["TOP"]; grid = d["grid"]; halves = d["halves"]
    m18 = {int(k): v for k, v in d["m18"].items()}

    def hname(i):
        return f"{i // 2 + 1}<span class='mut'>·{'1-2' if i % 2 == 0 else '3-4'}</span>"

    def play(t0, t1, label="▶"):
        return f'<button class="play" data-t0="{t0:.3f}" data-t1="{t1:.3f}">{label}</button>'

    # ── tableaux par section ───────────────────────────────────────────────
    tables = []
    for sec, b0, b1 in SECTIONS:
        rows = []
        divs = []
        for i in range(2 * (b0 - 1), 2 * b1):
            ref = D["l0"][i]
            cells = "".join(_chord_cell(D[k][i], ref) for k in VKEYS)
            nd = sum(D[k][i] != ref for k in VKEYS)
            rows.append(f"<tr><td class='bn'>{hname(i)}</td><td class='ch'>{html.escape(d['chart_h'][i])}</td>{cells}</tr>")
            if nd:
                divs.append(i)
        notes = []
        for i in divs:
            t0, t1 = halves[i]
            ref = D["l0"][i]
            ks = [k for k in VKEYS if D[k][i] != ref]
            tops = "".join(
                f"<div class='tp'><b>{VSHORT[k]}</b> " + " · ".join(
                    f"{html.escape(n)} {100 * p:.0f} %" for n, p in TOP[k][i]) + "</div>"
                for k in ["l0"] + ks)
            v = NOTES_DIV.get(i)
            verdict = ""
            if v:
                tag = {"ok": "probablement juste", "?": "à écouter", "x": "artefact probable"}[v[0]]
                verdict = f"<p class='vd v{ {'ok': 'ok', '?': 'q', 'x': 'x'}[v[0]] }'><b>{tag}.</b> {v[1]}</p>"
            elif all(D[k][i] in ("Fmaj7",) or D[k][i] == ref for k in ks if k in ("l05", "l1", "l1tb")) and any(
                    D[k][i] == "Fmaj7" for k in ks):
                verdict = "<p class='vd vq'><b>à écouter.</b> F → Fmaj7 : le mi est bien présent dans le chroma ; musx hésitait déjà (7M ≈ « pas de 7e »).</p>"
            else:
                verdict = ("<p class='vd vx'><b>artefact probable.</b> Seul S⁺ appliqué aux extensions change "
                           "l'accord (9e/11e/13e ou N.C. sortis de rien).</p>")
            notes.append(f"<div class='dv'><div class='dh'><b>mes. {i // 2 + 1}, temps {'1-2' if i % 2 == 0 else '3-4'}</b>"
                         f" <span class='mut'>chart : {html.escape(d['chart_h'][i])}</span>{play(t0, t1)}</div>{tops}{verdict}</div>")
        tables.append(f"""
<h3>{sec} — mesures {b0} à {b1} {play(grid[b0 - 1], grid[b1], '▶ section')}</h3>
<table class="dec"><thead><tr><th>mes.</th><th>chart</th>{''.join(f'<th>{VSHORT[k]}</th>' for k in VKEYS)}</tr></thead>
<tbody>{''.join(rows)}</tbody></table>
<details><summary>Les {len(divs)} demi-mesures où une variante diverge de λ0 — top 3 et lecture</summary>{''.join(notes)}</details>""")

    hs = d["held_stats"]
    nseg = d["n_seg"]

    # ── mesure 18 ─────────────────────────────────────────────────────────
    def pct(x):
        return f"{100 * x:.1f} %" if x >= 0.001 else "&lt; 0,1 %"
    mrows = []
    for i in range(32, 38):
        m = m18[i]
        pn = lambda v: f"{100 * v:.1f}" if v >= 0.001 else "≈0"
        mrows.append(
            f"<tr><td class='bn'>{hname(i)}</td>"
            f"<td>{' · '.join(f'{n} {100 * p:.0f}' for n, p in m['top5'][:2])}</td>"
            f"<td>{pn(m['dim_total'])}</td>"
            f"<td>{pn(m['s7'][3])}</td><td>{pn(m['s9'][3])}</td>"
            f"<td>{' · '.join(f'{n} {100 * p:.0f}' for n, p in m['bass'][:2])}</td>"
            f"<td>{m['dec_dims'][0][0]} ({m['dec_dims'][0][2]}e)</td></tr>")

    # courbe trame par trame, mesure 18
    fc = d["frame_curve"]
    t = np.asarray(fc["t"]); W, Hh = 340, 130
    x = lambda tt: 30 + (tt - t[0]) / (t[-1] - t[0]) * (W - 40)
    y = lambda v: 10 + (1 - v) * (Hh - 30)
    def line(vals, col, dash=""):
        pts = " ".join(f"{x(a):.1f},{y(b):.1f}" for a, b in zip(t, vals))
        return f'<polyline points="{pts}" fill="none" stroke="{col}" stroke-width="1.6" {dash}/>'
    bb = d["beats_of"][17]
    beat_lines = "".join(f'<line x1="{x(b):.1f}" x2="{x(b):.1f}" y1="10" y2="{Hh - 20}" class="gl"/>'
                         f'<text x="{x(b) + 2:.1f}" y="{Hh - 8}" class="ax">t{k + 1}</text>'
                         for k, b in enumerate(bb[:4]))
    curve = f"""<svg viewBox="0 0 {W} {Hh}" class="svg" role="img" aria-label="postérieures musx trame par trame, mesure 18">
{beat_lines}
<text x="2" y="{y(1) + 4:.0f}" class="ax">100</text><text x="8" y="{y(0) + 4:.0f}" class="ax">0</text>
{line(fc['G'], 'var(--c1)')}{line(fc['C'], 'var(--c2)')}{line(np.asarray(fc['dim']) * 1, 'var(--ko)')}
{line(fc['b9'], 'var(--c3)', 'stroke-dasharray="3 2"')}
</svg>
<div class="lg"><span><i style="background:var(--c1)"></i>sol majeur</span><span><i style="background:var(--c2)"></i>do majeur</span>
<span><i style="background:var(--ko)"></i>tous les diminués (12 fondamentales)</span><span><i style="background:var(--c3)"></i>9e = ♭9</span></div>"""

    # chroma en croches, mesures 18 et 26
    def heat(bar):
        rows = [r for r in d["chroma_rows"] if r["bar"] == bar]
        cw, ch = 34, 13
        out = [f'<svg viewBox="0 0 {30 + 8 * cw} {12 * ch + 34}" class="svg" role="img" aria-label="chroma mesure {bar}">']
        for k, r in enumerate(rows):
            for pc in range(12):
                v = r["treb"][pc]
                out.append(f'<rect x="{30 + k * cw}" y="{(11 - pc) * ch}" width="{cw - 1}" height="{ch - 1}" '
                           f'fill="var(--acc)" fill-opacity="{max(0.04, v):.2f}"/>')
            out.append(f'<text x="{30 + k * cw + cw / 2}" y="{12 * ch + 12}" class="ax" text-anchor="middle">'
                       f'{"t" + str(r["beat"]) if r["et"] == 0 else "+"}</text>')
            out.append(f'<text x="{30 + k * cw + cw / 2}" y="{12 * ch + 26}" class="ax" text-anchor="middle">{r["bass"]}</text>')
        for pc in range(12):
            bold = ' font-weight="700"' if NOTES[pc] in ("D♭", "D", "B", "F", "G") else ""
            out.append(f'<text x="2" y="{(11 - pc) * ch + 10}" class="ax"{bold}>{NOTES[pc]}</text>')
        out.append("</svg>")
        return "".join(out)

    sc_rows = []
    for r in d["chroma_rows"]:
        if r["bar"] != 18:
            continue
        best = sorted(r["sc"].items(), key=lambda kv: -kv[1])[:3]
        sc_rows.append(f"<tr><td>t{r['beat']}{'+' if r['et'] else ''}</td><td>{r['bass']}</td><td>"
                       + " · ".join(f"{html.escape(n)} {v:.2f}" for n, v in best) + "</td></tr>")

    g7 = d["g7_occ"]
    g7rows = "".join(
        f"<tr><td>{o['bar']} <span class='mut'>{'1-2' if o['half'] == 0 else '3-4'}</span></td>"
        f"<td class='{'hi' if o['db1'] > 0.6 else ''}'>{o['db1']:.2f}</td><td>{o['d1']:.2f}</td>"
        f"<td>{o['db2']:.2f}</td><td>{o['d2']:.2f}</td><td>{play(o['t0'], o['t1'])}</td></tr>" for o in g7)

    a18 = m18[34]
    page = f"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Demi-mesures et différences</title>
<style>
:root{{--bg:#fbfaf7;--fg:#1d1d1f;--mut:#6b6b70;--line:#e3e1dc;--acc:#2a6fdb;--ok:#1d8a4e;--ko:#c0392b;--q:#b7791f;--card:#fff;--c1:#2a6fdb;--c2:#1d8a4e;--c3:#9b59b6;--diff:#fde7c8}}
@media (prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--bg:#151517;--fg:#ececef;--mut:#9a9aa2;--line:#2c2c31;--acc:#6ea0ff;--ok:#4cc38a;--ko:#ff7a6b;--q:#e0b050;--card:#1d1d21;--c1:#6ea0ff;--c2:#4cc38a;--c3:#c39bd3;--diff:#4a3a1c}}}}
*{{box-sizing:border-box}} html,body{{margin:0;overflow-x:hidden}}
body{{background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,system-ui,sans-serif;padding:16px;max-width:820px;margin:auto}}
h1{{font-size:20px;margin:4px 0 8px}} h2{{font-size:17px;margin:26px 0 8px}} h3{{font-size:15px;margin:18px 0 6px;display:flex;align-items:center;gap:8px;flex-wrap:wrap}}
.sum{{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--acc);border-radius:10px;padding:10px 12px}}
.sum p{{margin:5px 0}} .mut{{color:var(--mut);font-size:12px}} p{{margin:6px 0}}
.f{{font-family:ui-monospace,Menlo,monospace;font-size:12.5px;background:var(--card);border:1px solid var(--line);border-radius:6px;padding:6px 8px;margin:4px 0;overflow-wrap:anywhere}}
table{{border-collapse:collapse;width:100%}}
table.dec{{table-layout:fixed;font-size:11.5px}} .dec th,.dec td{{padding:3px 2px;border-bottom:1px solid var(--line);text-align:left;overflow-wrap:anywhere}}
.dec th{{font-weight:600;font-size:10.5px;color:var(--mut)}} .dec th:first-child{{width:13%}} .dec th:nth-child(2){{width:15%}}
td.diff{{background:var(--diff);font-weight:600}} td.ch{{color:var(--mut)}} td.bn{{white-space:nowrap}}
.tbl td,.tbl th{{border-bottom:1px solid var(--line);padding:4px 3px;font-size:12px;text-align:left;vertical-align:top}}
.tbl th{{color:var(--mut);font-weight:600;font-size:11px}} td.hi{{font-weight:700;color:var(--ko)}}
details{{margin:6px 0 4px}} summary{{cursor:pointer;color:var(--acc);font-size:13.5px}}
.dv{{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px;margin:6px 0}}
.dh{{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin-bottom:4px}} .tp{{font-size:12px}}
.vd{{font-size:13px;margin:5px 0 0}} .vok b{{color:var(--ok)}} .vq b{{color:var(--q)}} .vx b{{color:var(--ko)}}
.play{{margin-left:auto;font:inherit;font-size:13px;padding:4px 11px;border-radius:14px;border:1px solid var(--acc);background:transparent;color:var(--acc);cursor:pointer}}
.play.on{{background:var(--acc);color:#fff}}
.svg{{width:100%;height:auto;display:block}} .ax{{font-size:9px;fill:var(--mut)}} .gl{{stroke:var(--line);stroke-width:1}}
.lg{{display:flex;flex-wrap:wrap;gap:4px 12px;font-size:12px;color:var(--mut)}} .lg i{{display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:4px;vertical-align:-1px}}
.btns{{display:flex;gap:8px;flex-wrap:wrap;margin:8px 0}} .btns .play{{margin-left:0}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:10px;margin:10px 0}}
</style></head><body>
<h1>Demi-mesures : le décodeur nourri avec la différence aux voisines</h1>
<p class="mut">Evening Star (CHVWPIFAT5A), 2026-10-12. Une seule chanson : ce sont des illustrations, pas une mesure de progrès.</p>

<div class="sum">
<p><b>1.</b> Donner au décodeur « postérieure + différence aux deux voisines » change peu d'accords si on ne l'applique qu'à la triade et à la basse (λ1 t+b : {sum(D['l1tb'][i] != D['l0'][i] for i in range(H))} demi-mesures sur {H}). Ce qui change est souvent musical : un <b>B♭m6</b> (iv mineur) mesure 16 et 24 que la postérieure seule cachait sous Gm, le C de la mesure 19.</p>
<p><b>2.</b> La différence seule (S⁺) devient un détecteur de changements : elle change d'accord à {nseg['sp']} frontières sur {H - 1} et invente des 9e/11e/13e. Le piège de l'accord tenu est réel pour elle (garde le nom {hs['sp'][0]} fois sur {hs['sp'][1]}), absent pour λ1 t+b ({hs['l1tb'][0]}/{hs['l1tb'][1]}).</p>
<p><b>3.</b> Mesure 18, temps 1-2 : musx ne voit <b>aucun diminué</b> (tous les diminués réunis : {pct(a18['dim_total'])} ; sol majeur {100 * a18['top5'][0][1]:.0f} %). Mais l'audio brut a un <b>ré♭</b> comme note la plus forte au 1er temps, sur sol-si-fa : sol si ré♭ fa = <b>G7♭5</b>, qui remonte au ré au 2e temps. Même chose mesures 26 et 42. C'est sans doute ce que tu entends ; musx n'a pas de case pour une quinte diminuée sur une tierce majeure.</p>
</div>
<div class="btns">{play(grid[17], grid[18], '▶ mes. 18')}{play(grid[17], grid[17] + (grid[18] - grid[17]) / 4, '▶ mes. 18, 1er temps')}{play(grid[16], grid[19], '▶ mes. 17–19')}{play(grid[25], grid[26], '▶ mes. 26')}</div>

<h2>1. Les variantes, en une ligne chacune</h2>
<p>p(h) = la postérieure musx moyennée sur la demi-mesure h (temps 1-2 ou 3-4), pour chaque étage : triade (73 cases), basse (13), 7e, 9e, 11e, 13e.</p>
<div class="f">S(h) = 2·p(h) − p(h−1) − p(h+1)   (différence avec la voisine d'avant + avec celle d'après)</div>
{''.join(f'<div class="f"><b>{VSHORT[k]}</b> : {f}</div>' for k, _, f in VARIANTS)}
<p class="mut">norm(·) = on ajoute 0,001 partout puis on ramène à somme 1 (le décodeur prend des logarithmes, une case à zéro serait interdite). Pourquoi seulement λ = 0, ½, 1 et S⁺ : λ = 0 est la référence, λ = 1 donne à la différence le même poids qu'à la postérieure, et S⁺ est ce que devient λ quand il tend vers l'infini. Ces quatre points couvrent tout le chemin, sans chercher un réglage. Les deux variantes « t+b » ne touchent que la triade et la basse : on les a ajoutées après avoir vu S⁺ inventer des extensions.</p>
<p><b>Comment on décode.</b> On reprend tel quel le décodeur de production (le Viterbi de musx, mêmes accords permis, mêmes pénalités de changement aux temps). Il lit des trames de 23 ms ; on lui donne une suite de trames où toutes les trames d'une demi-mesure portent la même distribution q(h). Il additionne log q sur les trames, donc il ne change jamais d'accord au milieu d'une demi-mesure : la grille est bien de 2 cases par mesure. Le « top 3 » montré est le score d'une trame pour chaque accord du vocabulaire (produit des 6 étages, ramené à somme 1). Le décodeur peut garder l'accord d'avant même quand un autre est premier dans ce top 3 : changer d'accord coûte une pénalité.</p>
<p class="mut">« chart » = le texte du chart actuel (après repliement des sections), qui n'est pas une vérité. λ0 diffère du décodage brut de production sur 3 demi-mesures seulement : moyenner sur la demi-mesure ne change presque rien en soi. Une case <span style="background:var(--diff);padding:0 3px">surlignée</span> = différente de λ0.</p>

<h2>2. Ce que décode chaque variante</h2>
<p class="mut">La cellule de A selon toi (2 mesures chacune) : F | C7, F | Gm, F/C | G7 C7, F C | Gm7 C7.</p>
{''.join(tables)}

<h2>3. Le piège : un accord tenu garde-t-il son nom ?</h2>
<p>Sur un accord tenu, la différence avec les voisines est presque nulle. Si on ne gardait que la différence, l'accord perdrait son identité. On a pris les demi-mesures où λ0 écrit le même accord avant, pendant et après ({len(d['held'])} cas sur le morceau : Evening Star change beaucoup).</p>
<table class="tbl"><tr><th>variante</th><th>garde le nom</th><th>changements d'accord sur le morceau</th></tr>
{''.join(f"<tr><td>{VSHORT[k]}</td><td>{hs[k][0]} / {hs[k][1]}</td><td>{nseg[k]} / {H - 1}</td></tr>" for k in VKEYS)}
</table>
<p>S⁺ seul tombe dans le piège : mesure 27 temps 3-4, un F tenu devient C13 ; mesure 31 temps 3-4, D7 devient D6. Cinq cas, c'est peu : c'est une illustration. Surtout, il change d'accord à presque chaque frontière : c'est un détecteur de changements, plus un détecteur d'accords. Sur un accord tenu, la masse positive de S sur la triade est en moyenne {d['tri_mass']['tenue']:.2f}, contre {d['tri_mass']['toutes']:.2f} sur l'ensemble des demi-mesures : ce qui reste est du bruit, et la normalisation l'amplifie.</p>
<p>λ = ½ et λ = 1 sur triade+basse ne perdent aucun accord tenu : p garde l'identité, S ne fait que pencher vers ce qui est nouveau. λ = ½ appliqué aussi aux extensions ajoute pourtant des F/C et C/G discutables (mes. 14, 15) ; λ = 1 t+b est celle qui casse le moins de choses visibles ici.</p>

<h2>4. Mesure 18 : où est le diminué ?</h2>
<p><b>Dans musx : nulle part.</b> Le réseau a bien des cases « diminué » (triade dim, 12 fondamentales ; 7e = ♭♭7 pour le °7) et le décodeur peut écrire dim, dim7 et ø7. Ici elles sont vides :</p>
<table class="tbl"><tr><th>mes.</th><th>triade (%)</th><th>dim (%)</th><th>♭♭7 (%)</th><th>♭9 (%)</th><th>basse (%)</th><th>1er dim</th></tr>
{''.join(mrows)}</table>
<p class="mut">« dim » = les 12 diminués réunis. « 1er dim » = le premier accord diminué (dim, °7 ou ø7) dans le classement du décodeur, et son rang. Trame par trame dans la mesure 18, la masse « diminué » ne dépasse jamais {100 * max(fc['dim']):.1f} % :</p>
{curve}
<p><b>Dans l'audio brut : un ré♭ au 1er temps.</b> Chroma NNLS (indépendant de musx), une colonne par croche, la ligne du bas = la basse. Au 1er temps de la mesure 18, ré♭ est la note la plus forte, avec fa, si et sol ; au 2e temps, ré♭ disparaît et ré prend sa place.</p>
<div class="card"><b>mes. 18</b> {play(grid[17], grid[18])}{heat(18)}</div>
<table class="tbl"><tr><th>croche</th><th>basse</th><th>accords les plus proches (cosinus chroma)</th></tr>{''.join(sc_rows)}</table>
<p>Sol-si-ré♭-fa, c'est <b>G7♭5</b> : deux tritons (sol–ré♭ et si–fa), d'où une couleur « diminuée ». Ce n'est pas un vrai °7 : il faudrait un la♭ (si ré fa la♭), et le la♭ est à {g7[[o['bar'] for o in g7].index(18)]['ab1']:.2f}. Ce n'est pas non plus G° (sol si♭ ré♭) : le si♭ est faible, la tierce est majeure.</p>
<div class="card"><b>mes. 26</b> (même endroit dans A2) {play(grid[25], grid[26])}{heat(26)}</div>
<p><b>Ça revient à chaque A.</b> Ré♭ et ré (relatifs à la note la plus forte), 1er et 2e temps, pour chaque demi-mesure que λ0 lit comme G7 :</p>
<table class="tbl"><tr><th>mes.</th><th>ré♭ t1</th><th>ré t1</th><th>ré♭ t2</th><th>ré t2</th><th></th></tr>{g7rows}</table>
<p>Mesures 18, 26 et 42 (le même G7 de la cellule F/C | G7 C7) : ré♭ à 0,95–1,00 au 1er temps, puis ré au 2e. Les autres G7 n'ont pas ce ré♭ (sauf mesure 2, l'intro, et la fin de la mesure 34).</p>
<p><b>Pourquoi musx ne l'entend pas.</b> Sa triade n'a que majeur, mineur, sus4, sus2, diminué, augmenté : pas de « tierce majeure + quinte diminuée ». Sol-si-ré♭ n'a donc pas de case ; la plus proche est sol majeur (même fondamentale, même tierce). En plus ça dure un temps sur deux. Ce n'est pas le décodeur ni la suite du pipeline : la postérieure elle-même ne contient pas le diminué.</p>
<p><b>À noter :</b> le seul endroit de A où musx et le chroma voient vraiment une couleur diminuée, c'est mesures 16 et 24 temps 3-4 : B♭m6 = si♭ ré♭ fa sol = les notes de Gø7 (Gdim y est à 7 %). Si ce n'est pas le ré♭ de la mesure 18, c'est peut-être celui-là.</p>

<h2>5. Ce que ça ne résout pas</h2>
<p>Une seule chanson, et aucune vérité : les verdicts ci-dessus sont des lectures du chroma, à confirmer à l'oreille. On n'a rien branché : le pipeline et ses réglages sont intacts.</p>
<p>Le G7♭5 ne peut pas sortir de musx tel quel. Prochaine mesure la moins chère : un détecteur « quinte diminuée sur accord de dominante » au temps près, lu dans le chroma (ré♭ fort et ré faible sur un G7 de musx), compté sur la bibliothèque pour voir combien de fois il se déclenche et à quels endroits — à valider sur ces trois mesures d'abord.</p>
<p>Pour la différence aux voisines : regarder λ1 t+b sur deux autres morceaux que tu connais avant d'en tirer quoi que ce soit.</p>
<p class="mut">Script : tools/demi_mesures_delta_decodeur.py.</p>

<audio id="au" preload="none"></audio>
<script>
(function(){{
  var au=document.getElementById('au'), src='../audio/{STEM}.m4a', blobUrl=null, stopAt=null, cur=null;
  // iPhone : fetch()+blob (le chargeur média de la PWA ne bufferise pas), pas de boucle rAF
  fetch(src).then(function(r){{return r.blob();}}).then(function(b){{blobUrl=URL.createObjectURL(b);au.src=blobUrl;}}).catch(function(){{au.src=src;}});
  au.addEventListener('timeupdate',function(){{ if(stopAt!==null && au.currentTime>=stopAt){{au.pause();}} }});
  au.addEventListener('pause',function(){{ if(cur){{cur.classList.remove('on');cur=null;}} stopAt=null; }});
  document.querySelectorAll('.play').forEach(function(b){{
    b.addEventListener('click',function(ev){{
      ev.preventDefault(); ev.stopPropagation();
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


if __name__ == "__main__":
    main()
