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

from harmonia import cache, soudure
from harmonia.musx import FRAME_DT
from harmonia.sections.similarity import chord_tone_matrix, harmonic_vectors
from harmonia.settings import SETTINGS

STEM = "CHVWPIFAT5A"
CHART = SETTINGS.repo / "state/cache/charts/min_CHVWPIFAT5A.json"
OUT = SETTINGS.repo / "docs/plots/produit_scalaire_lettres_2026-10-12.html"

NOTES = "C D♭ D E♭ E F G♭ G A♭ A B♭ B".split()
QUAL = ["", "m", "sus4", "sus2", "dim", "aug"]
SEPT = ["sans 7e", "maj7", "7 (b7)", "dim7"]

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
        H.append(dict(tri=p, root=root, chroma=chroma, c7=c7, bass=bs[1:],
                      bass_full=bs, s7=sv))

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

    write_page(chart, H, bornes, idx, A1, A2, res, halves, bars)


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
    sv = h["s7"]
    k = int(np.argmax(sv))
    s7 = " · ".join(f"{SEPT[i]} {100*sv[i]:.0f}" for i in np.argsort(-sv)[:2])
    return (f'<div class="half">{rows}<div class="sub">basse : {bs}</div>'
            f'<div class="sub">7e : {s7}</div></div>')


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


def write_page(chart, H, bornes, idx, A1, A2, res, halves, bars):
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
        cell_html.append(
            f'<section class="cell" id="c{m}"><div class="ch"><b>{name}</b> '
            f'<span class="ms">mes. {m}–{m+1}</span> <span class="chords">{html.escape(chords)}</span>'
            f'<button class="play" data-t0="{t0:.3f}" data-t1="{t1:.3f}">▶ écouter</button></div>'
            f'<div class="hmwrap"><div class="cap">notes d\'accord attendues, ligne = demi-mesure</div>{hm}</div>'
            f'<div class="halves">{halves_html}</div></section>')

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

    summary = SUMMARY(res, A1, A2)
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

<h2>Les candidats en un tableau</h2>
<p class="mut">Lettres données par la règle actuelle (accords identiques ⇒ même lettre ; deux groupes se rejoignent si toutes leurs paires dépassent la paire identique la moins ressemblante du morceau). L'oreille de Louis : A1 = <b>a a b a</b>, l'intrus = cellule 3 (F/C | G7 C7).</p>
<div class="scroll"><table class="tbl"><tr><th>définition</th><th>lettres, règle actuelle (A1 · A2)</th><th>intrus (A1 · A2)</th><th>a a b a possible ? (A1 · A2)</th><th>avec ce seuil : A1 · A2 · B</th></tr>
{''.join(table_rows)}</table></div>

<h2>1. Les postérieures, demi-mesure par demi-mesure</h2>
<p class="mut">Une « demi-mesure » = temps 1-2 ou 3-4 (4 temps par mesure, vérifié sur la grille). Pour chacune : les 5 accords les plus probables de musx (en %), les 3 basses, la septième. Au-dessus, les notes d'accord attendues (Σ probabilité × notes de l'accord), une ligne par demi-mesure. ▶ joue exactement les 2 mesures.</p>
{''.join(cell_html)}

<h2>2. Les définitions candidates</h2>
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


def SUMMARY(res, A1, A2):
    ok1 = [r for r in res if r["w1"]]
    ok12 = [r for r in ok1 if r["w2"]]
    thr = [r["thr"] for r in res]
    l1 = ("<p><b>1. Seule la basse donne a a b a.</b> "
          + (", ".join(f"{r['title']} (seuil {r['w1'][0]:.2f}–{r['w1'][1]:.2f})" for r in ok1)
             if ok1 else "Aucune définition")
          + " : la cellule 3 (F/C | G7 C7) est la seule à poser do à la basse. "
          "Toutes les définitions « notes d'accord » désignent au contraire la 2 (F | Gm).</p>")
    l2 = ("<p><b>2. Cohérent sur A2 :</b> "
          + (", ".join(f"{r['title']} — {r['ftxt']}" for r in ok12) if ok12 else "aucune")
          + ".</p>")
    l3 = ("<p><b>3. Mais pas avec la règle de seuil actuelle.</b> Le seuil vient des paires identiques "
          f"({min(thr):.2f}–{max(thr):.2f} selon la définition) : il est si haut que rien ne fusionne et "
          "A1 sort toujours <span class='mono'>ghij</span>. a a b a demande un seuil vers 0.5 : il faut donc changer "
          "aussi la règle du seuil, pas seulement le produit scalaire. Un seul morceau : à vérifier ailleurs "
          "avant d'en faire une loi.</p>")
    return l1 + l2 + l3


if __name__ == "__main__":
    main()
