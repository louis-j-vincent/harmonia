"""La page des candidates de ressemblance entre accords (`harmonia.chord_similarity`).

    .venv/bin/python -m tools.similarite_accords_page
    # -> docs/plots/similarite_accords.html  (http://100.89.209.63:7772/plots/similarite_accords.html)

Trois parties, AUCUN verdict agrégé (règle de Louis, 2026-08-09 : « je ne
veux JAMAIS juste voir le résultat ») :

  1. le panneau de paires que Louis a nommées, chaque candidate côte à côte ;
  2. une carte par candidate : un accord de référence (au choix) contre toutes
     les fondamentales x qualités ;
  3. trois morceaux qu'il connaît (Lost Without U, Claustrophobic, This
     Love) : chaque comparaison de passes (une passe contre la passe de
     référence de sa lettre ; le 2e tour d'une passe de 8 contre le 1er),
     décidée par la règle actuelle (`folding.meme_moment`) et par chaque
     candidate à son seuil ; ce qui bascule est surligné, chaque passe
     s'écoute.

Les accords des passes sont ceux du décodage BRUT du morceau (`prompter`),
temps par temps, jamais ceux recopiés d'une passe sœur (mémoire
« raw decode beats template »). Lecture seule : rien dans `state/` n'est
écrit.

Une mesure « est la même » pour une candidate si TOUS ses temps le sont
(score de la mesure = le pire temps) ; deux passes sont « les mêmes » si
toutes leurs mesures le sont. Le seuil de chaque candidate est posé SANS
regarder les morceaux : la plus petite valeur qu'elle donne aux paires que
Louis a déclarées équivalentes (D-7/G7sus4, Bb/C/C7sus4, C7/C9, C/C^7, G7/G).
"""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

from harmonia import chord_similarity as cs
from harmonia.roles import meme_lecture
from harmonia.settings import SETTINGS

CHARTS = SETTINGS.repo / "state" / "cache" / "charts"
SORTIE = SETTINGS.repo / "docs" / "plots" / "similarite_accords.html"
NOMS = ["C", "D♭", "D", "E♭", "E", "F", "G♭", "G", "A♭", "A", "B♭", "B"]
TOK = {"": "", "-": "m", "7": "7", "^7": "maj7", "-7": "m7", "6": "6", "-6": "m6",
       "o": "dim", "o7": "dim7", "h7": "ø7", "+": "+", "sus4": "sus4", "sus2": "sus2",
       "7sus4": "7sus4", "9": "9", "-9": "m9", "^9": "maj9", "13": "13", "69": "6/9"}

#: Les candidates montrées, dans l'ordre des colonnes, avec une ligne d'explication.
COLS = [
    ("notes", "notes sonnées (cosinus, basse = une note de plus) — la ressemblance chord-tone d'aujourd'hui"),
    ("notes7", "la même, sans les couleurs (9, 11, 13 retirées)"),
    ("notes_basse", "notes sonnées, la basse pèse double"),
    ("tiv", "espace tonal (TIV : DFT des notes)"),
    ("musx", "appris : à quel point musx hésite entre les deux, 310 morceaux"),
    ("fonction", "contexte : même fonction tonale (T / PD / D) dans la tonalité"),
    ("notes+fonction", "moitié notes, moitié fonction"),
    ("notes+musx", "moitié notes, moitié musx"),
]


def C(r, q="", b=-1):
    return {"root": r, "q": q, "bass": b, "nc": False}


def nom(c) -> str:
    if c.get("nc"):
        return "N.C."
    s = NOMS[c["root"] % 12] + TOK.get(c.get("q") or "", c.get("q") or "")
    b = c.get("bass", -1)
    if b is not None and b >= 0 and b % 12 != c["root"] % 12:
        s += "/" + NOMS[b % 12]
    return s


# (étiquette, a, b, attendu, tonalité, note musicale)
MAJ, MIN = (0, "major"), (0, "minor")
PANEL = [
    ("D-7 / G7sus4", C(2, "-7"), C(7, "7sus4"), "proche", MAJ, "le ii sur la pédale de V (Lost Without U)"),
    ("B♭/C / C7sus4", C(10, "", 0), C(0, "7sus4"), "proche", MAJ, "Claustrophobic, règle 2302eb2"),
    ("C7 / C9", C(0, "7"), C(0, "9"), "proche", MAJ, "couleur"),
    ("C / Cmaj7", C(0), C(0, "^7"), "proche", MAJ, "couleur"),
    ("G7 / G", C(7, "7"), C(7), "proche", MAJ, "couleur (V avec ou sans 7)"),
    ("Cmaj7 / Am7", C(0, "^7"), C(9, "-7"), "?", MAJ, "tonique et son relatif"),
    ("C6 / Am7", C(0, "6"), C(9, "-7"), "?", MAJ, "mêmes quatre notes, autre basse"),
    ("G7 / D♭7", C(7, "7"), C(1, "7"), "?", MAJ, "substitution tritonique"),
    ("C / Am", C(0), C(9, "-"), "?", MAJ, "relatif mineur"),
    ("C / Em", C(0), C(4, "-"), "?", MAJ, "I / iii"),
    ("F / Dm", C(5), C(2, "-"), "?", MAJ, "IV / ii"),
    ("G7 / B°", C(7, "7"), C(11, "o"), "?", MAJ, "V7 sans fondamentale"),
    ("Dm7 / F", C(2, "-7"), C(5), "?", MAJ, "ii7 / IV"),
    ("Am / A", C(9, "-"), C(9), "?", MAJ, "même fondamentale, tierce changée"),
    ("G7 / Gm7", C(7, "7"), C(7, "-7"), "?", MAJ, "tierce changée sur un accord de 4 sons (Lost Without U, mes. 4)"),
    ("Gm7 / G7sus4", C(7, "-7"), C(7, "7sus4"), "?", MAJ, "Lost Without U, mes. 8 (known_issues : à fusionner ?)"),
    ("C7 / C7sus4", C(0, "7"), C(0, "7sus4"), "?", MAJ, "sus contre dominante (règle actuelle : même famille)"),
    ("Dm7 / G7", C(2, "-7"), C(7, "7"), "?", MAJ, "ii contre V (Lost Without U, mes. 2 contre 6)"),
    ("C / F", C(0), C(5), "loin", MAJ, "I / IV : doit rester LOIN"),
    ("C / F♯", C(0), C(6), "loin", MAJ, "triton"),
    ("G7 / C", C(7, "7"), C(0), "loin", MAJ, "V / I"),
    ("This Love : Fm / F7", C(5, "-"), C(5, "7"), "loin", MIN, "B, mes. 7 (Louis : harmoniquement différent)"),
    ("This Love : A♭ / B♭", C(8), C(10), "loin", MIN, "B, mes. 8 contre mes. 4"),
    ("This Love : G / E♭", C(7), C(3), "loin", MIN, "B, mes. 8 temps 3"),
    ("This Love : Dø7 / B♭7/A♭", C(2, "h7"), C(10, "7", 8), "?", MIN, "A, mes. 4 temps 3 (accord de passage)"),
    ("B♭ / Gm (contrôle)", C(10), C(7, "-"), "?", MIN, "règle verrouillée : B♭ plus près de Gm…"),
    ("B♭ / F (contrôle)", C(10), C(5), "?", MIN, "… que de F"),
]


def seuils() -> dict:
    """Seuil de chaque candidate = sa plus petite valeur sur les paires « proche »."""
    out = {}
    for k, _ in COLS:
        vals = [cs.CANDIDATES[k](a, b, {"cle": key}) for _, a, b, att, key, _ in PANEL if att == "proche"]
        vals = [v for v in vals if v is not None]
        out[k] = math.floor(min(vals) * 1000) / 1000 if vals else None
    out["notes7+voisins"] = out["notes7"]
    return out


def panneau(seuil) -> list[dict]:
    rows = []
    for lab, a, b, att, key, note in PANEL:
        ctx = {"cle": key}
        v = {k: cs.CANDIDATES[k](a, b, ctx) for k, _ in COLS}
        rows.append({"lab": lab, "att": att, "note": note,
                     "cle": ("C majeur" if key == MAJ else "C mineur"),
                     "actuel": bool(meme_lecture(a, b)),
                     "exact": bool(meme_lecture(a, b, grain="exact")),
                     "musx_lab": f"{cs.nom_musx(a)} · {cs.nom_musx(b)}",
                     "cran": cs.cran(a, b),
                     "v": {k: (None if x is None else round(x, 3)) for k, x in v.items()}})
    return rows


# ── cartes ──────────────────────────────────────────────────────────────────
REFS = [C(0), C(0, "-"), C(0, "7"), C(0, "^7"), C(0, "-7"), C(0, "7sus4"), C(0, "h7"), C(0, "o7"),
        C(10, "", 0)]
QB = ["", "-", "7", "^7", "-7", "6", "7sus4", "sus4", "h7", "o", "o7", "+", "9", "13"]


def cartes() -> dict:
    out = {}
    ctx = {"cle": MAJ}
    for k, _ in COLS:
        out[k] = []
        for a in REFS:
            out[k].append([[None if (x := cs.CANDIDATES[k](a, C(r, q), ctx)) is None else round(x, 3)
                            for r in range(12)] for q in QB])
    return out


# ── morceaux ────────────────────────────────────────────────────────────────
MORCEAUX = [
    ("min_0DdCoNbbRvQ", "Lost Without U", "la fin (outro) contre le A, et les tours du A (G7sus4 / Gm7)"),
    ("min_mCwI0YA4mIY", "Claustrophobic (feat. Pell)", "les fins du A (B♭/C, C7sus4, C9…)"),
    ("min_maroon_5_this_love", "This Love", "les deuxièmes moitiés : Louis veut qu'elles restent DIFFÉRENTES"),
]


def mesures(chart) -> list[list[dict]]:
    """Par mesure, l'accord sonnant à chaque temps (décodage brut)."""
    g, bt, ch = chart["barGrid"], chart["beatTimes"], chart["prompter"]["chords"]
    bpb = int(chart.get("bpb") or 4)
    out = []
    for i in range(len(g) - 1):
        t0, t1 = g[i], g[i + 1]
        temps = [t for t in bt if t0 - 0.05 <= t < t1 - 0.05][:bpb]
        while len(temps) < bpb:
            temps.append(t0 + (t1 - t0) * len(temps) / bpb)
        bar = []
        for t in temps:
            cur = None
            for c in ch:
                if c["t0"] <= t + 0.08:
                    cur = c
                else:
                    break
            if cur is None or cur["t1"] < t:
                cur = {"nc": True, "root": 0, "q": "", "bass": -1}
            bar.append({k: cur.get(k) for k in ("root", "q", "bass", "nc")})
        out.append(bar)
    return out


def cle_a(chart, t):
    for s in chart.get("keySegments") or []:
        if s["t0"] - 0.5 <= t <= s["t1"] + 0.5:
            return (s["tonic"], s["mode"])
    k = chart.get("key") or {}
    return (k.get("tonic", 0), k.get("mode", "major"))


def bar_folding(bar) -> list[dict]:
    """La mesure au format `folding` (accords + temps de départ)."""
    out = []
    for k, c in enumerate(bar):
        if out and (out[-1]["nc"], out[-1]["root"], out[-1]["q"], out[-1]["bass"]) == \
                (bool(c["nc"]), c["root"], c["q"], c["bass"]):
            continue
        out.append({**c, "nc": bool(c["nc"]), "beat": k, "t0": float(k), "t1": float(k + 1)})
    return out


def _voisins(bars_pass, i, k):
    flat = [c for b in bars_pass for c in b]
    j = i * len(bars_pass[0]) + k
    return flat[max(0, j - 1)], flat[j], flat[min(len(flat) - 1, j + 1)]


def _vec_fen(trio):
    vs = [cs.vecteur_notes(c, couleurs=False) for c in trio]
    return [0.25 * x + 0.5 * y + 0.25 * z for x, y, z in zip(*vs)]


def score_mesure(k, pa, pb, i, ctx):
    """(pire temps, (a, b) du pire temps) pour la candidate k."""
    pire, paire = 2.0, None
    for t in range(len(pa[i])):
        a, b = pa[i][t], pb[i][t]
        if k == "notes7+voisins":
            if a.get("nc") or b.get("nc"):
                s = 1.0 if a.get("nc") and b.get("nc") else 0.0
            else:
                s = cs._cos(_vec_fen(_voisins(pa, i, t)), _vec_fen(_voisins(pb, i, t)))
        else:
            s = cs.CANDIDATES[k](a, b, ctx)
            if s is None:                          # fonction indéfinie → notes
                s = cs.notes(a, b)
        if s < pire:
            pire, paire = s, (a, b)
    return pire, paire


def temps_ecrits(bar, bpb=4, avant=None) -> list[dict]:
    """Une mesure ÉCRITE (accords + `beat`) → l'accord sonnant à chaque temps.
    Une mesure vide (tenue) reprend le dernier accord d'`avant`."""
    out, cur = [], avant
    for k in range(bpb):
        for c in bar or []:
            if int(c.get("beat") or 0) <= k:
                cur = c
        cur = cur or {"nc": True, "root": 0, "q": "", "bass": -1}
        out.append({"root": cur.get("root", 0), "q": cur.get("q") or "",
                    "bass": -1 if cur.get("bass") is None else cur.get("bass"),
                    "nc": bool(cur.get("nc"))})
    return out


def ecrits(bars, bpb=4) -> list[list[dict]]:
    out, avant = [], None
    for b in bars:
        t = temps_ecrits(b, bpb, avant)
        avant = t[-1]
        out.append(t)
    return out


def compare(titre, pa, pb, cle, ma, mb, ta=None, tb=None, genre="") -> dict:
    """Deux passages (listes de mesures, chaque mesure = un accord par temps)."""
    # La règle ACTUELLE, appliquée TEMPS PAR TEMPS comme les candidates (pour
    # que seule la mesure d'accord diffère, pas la tolérance de structure) :
    # `roles.meme_lecture` au grain de la décision — « racine » pour une fin
    # (`_ireal_endings` : une vraie fin change de fondamentale), « famille »
    # pour une boucle (`meme_harmonie`) et un passage brut (`meme_moment`,
    # sans sa tolérance d'accord de passage).
    grain = "racine" if genre == "fin" else "famille"

    def actuel(x, y):
        return all(meme_lecture(u, v, grain=grain) for u, v in zip(x, y))
    n = min(len(pa), len(pb))
    pa, pb = pa[:n], pb[:n]
    ctx = {"cle": cle}
    rows = []
    for i in range(n):
        r = {"ma": ma[i], "mb": mb[i],
             "a": " ".join(nom(c) for c in bar_folding(pa[i])),
             "b": " ".join(nom(c) for c in bar_folding(pb[i])),
             "actuel": bool(actuel(pa[i], pb[i])), "s": {}, "pire": {},
             "cran": max(cs.cran(u, v) for u, v in zip(pa[i], pb[i]))}
        for k in [c for c, _ in COLS] + ["notes7+voisins"]:
            s, paire = score_mesure(k, pa, pb, i, ctx)
            r["s"][k] = round(s, 3)
            r["pire"][k] = f"{nom(paire[0])} / {nom(paire[1])}" if paire else ""
        rows.append(r)
    return {"titre": titre, "genre": genre, "rows": rows,
            "cle": f"{NOMS[cle[0] % 12]} {'majeur' if cle[1] == 'major' else 'mineur'}",
            "ta": ta, "tb": tb}


def comparaisons(stem):
    chart = json.loads((CHARTS / f"{stem}.json").read_text())
    bpb = int(chart.get("bpb") or 4)
    M = mesures(chart)
    g = chart["barGrid"]
    audio = (chart.get("audio_url") or "").lstrip("/")
    comps = []

    def t_of(rg):
        return [g[rg[0]], g[min(rg[-1] + 1, len(g) - 1)]]

    for s in chart["sections"]:
        rgs = [list(range(r[0], r[1] + 1)) for r in sorted(s["barRanges"])]
        cle = cle_a(chart, s["spans"][0][0] if s.get("spans") else 0.0)
        lab = s["label"]
        W = ecrits(s.get("bars") or [], bpb)
        first = rgs[0] if rgs else None
        # 1. LA BOUCLE : le 2e tour du bloc écrit contre le 1er
        if len(W) >= 8 and len(W) % 2 == 0:
            h = len(W) // 2
            rep = next((r for r in rgs if len(r) == len(W)), None)
            comps.append(compare(
                f"{lab} écrit : mes. {h+1}–{len(W)} contre mes. 1–{h}", W[h:], W[:h], cle,
                list(range(h + 1, len(W) + 1)), list(range(1, h + 1)),
                t_of(rep[h:]) if rep else None, t_of(rep[:h]) if rep else None, "boucle"))
        # 2. LES FINS : chaque fin contre la plus jouée
        e = s.get("endings")
        if e:
            vs = e["variants"]
            main_v = max(vs, key=lambda v: (len(v["passes"]), -v["passes"][0]))
            Wm = ecrits(main_v["bars"], bpb)
            tail = e["tail"]
            for v in vs:
                if v is main_v:
                    continue
                pv, pm = v["passes"][0], main_v["passes"][0]
                rv = rgs[pv] if pv < len(rgs) else None
                rm = rgs[pm] if pm < len(rgs) else None
                comps.append(compare(
                    f"{lab} fin des passes {[p+1 for p in v['passes']]} contre fin des passes {[p+1 for p in main_v['passes']]}",
                    ecrits(v["bars"], bpb), Wm, cle,
                    [len(W) - tail + 1 + i for i in range(tail)], [len(W) - tail + 1 + i for i in range(tail)],
                    t_of(rv[-tail:]) if rv else None, t_of(rm[-tail:]) if rm else None, "fin"))
        # 3. LE DÉCODAGE BRUT : chaque passe contre la passe de référence
        if len(rgs) >= 2:
            lens = [len(r) for r in rgs]
            modal = max(set(lens), key=lambda L: (lens.count(L), L))
            ref = next(r for r in rgs if len(r) == modal)
            for r in rgs:
                if r is ref or len(r) < 2:
                    continue
                pa, pb = [M[i] for i in r], [M[i] for i in ref]
                if all(c["nc"] for b in pa for c in b):
                    continue
                comps.append(compare(
                    f"{lab} mes. {r[0]+1}–{r[-1]+1} contre {lab} mes. {ref[0]+1}–{ref[-1]+1}",
                    pa, pb, cle, [i + 1 for i in r], [i + 1 for i in ref], t_of(r), t_of(ref), "brut"))
    if stem == "min_0DdCoNbbRvQ":
        secs = {s["label"]: s for s in chart["sections"]}
        A, out = secs["A"], secs["outro"]
        WA, WO = ecrits(A["bars"], bpb), ecrits(out["bars"], bpb)
        ro = list(range(out["barRanges"][0][0], out["barRanges"][0][1] + 1))
        rA = sorted(A["barRanges"])[0]
        rA = list(range(rA[0], rA[1] + 1))
        comps.insert(0, compare(
            "outro mes. 1–4 (G7sus4 | G7 | C | C Gm7) contre le 2e tour du A écrit (mes. 5–8)",
            WO[:4], WA[4:8], cle_a(chart, g[ro[0]]), [1, 2, 3, 4], [5, 6, 7, 8],
            t_of(ro[:4]), t_of(rA[4:8]), "fin"))
    return {"stem": stem, "audio": audio, "comps": comps}


# ── page ────────────────────────────────────────────────────────────────────

def main() -> None:
    S = seuils()
    data = {"cols": [{"k": k, "d": d} for k, d in COLS], "seuils": S,
            "panel": panneau(S), "cartes": cartes(),
            "refs": [nom(c) for c in REFS], "qb": [TOK.get(q, q) or "maj" for q in QB],
            "roots": NOMS,
            "morceaux": [{"titre": t, "pourquoi": w, **comparaisons(s)} for s, t, w in MORCEAUX],
            "musx_diag": cs.table_musx()["diagnostics"], "version": cs.VERSION}
    page = PAGE.replace("__DATA__", json.dumps(data, ensure_ascii=False))
    SORTIE.write_text(page)
    print(f"-> {SORTIE}\n   http://100.89.209.63:7772/plots/{SORTIE.name}")


PAGE = r"""<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Ressemblance entre accords</title>
<style>
:root{--bg:#fbfaf7;--fg:#1d1c1a;--faint:#77736b;--rule:#e2ded5;--card:#fff;--hi:#fff1c2;--hi2:#c0392b;
--ok:#2e7d32;--no:#9e9e9e;--acc:#3b5bdb}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#17171a;--fg:#ecebe8;--faint:#9a978f;--rule:#2f2f34;--card:#1f1f23;--hi:#4a3d12;--hi2:#ff8a75;--ok:#7bd88f;--no:#77736b;--acc:#8ea2ff}}
:root[data-theme="dark"]{--bg:#17171a;--fg:#ecebe8;--faint:#9a978f;--rule:#2f2f34;--card:#1f1f23;--hi:#4a3d12;--hi2:#ff8a75;--ok:#7bd88f;--no:#77736b;--acc:#8ea2ff}
body{background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,system-ui,sans-serif;margin:0;padding:0 16px 60px}
.w{max-width:1100px;margin:0 auto}
h1{font-size:22px;margin:22px 0 6px}h2{font-size:18px;margin:30px 0 6px}h3{font-size:16px;margin:18px 0 4px}
p.l,p.m,.lesson{overflow-wrap:anywhere}
p.l{color:var(--faint);margin:4px 0 10px;max-width:760px}
.sc{overflow-x:auto;-webkit-overflow-scrolling:touch}
table{border-collapse:collapse;font-variant-numeric:tabular-nums}
td,th{border-bottom:1px solid var(--rule);padding:4px 6px;text-align:right;white-space:nowrap;font-size:13px}
th{font-weight:600;color:var(--faint);vertical-align:bottom}
td.t,th.t{text-align:left}
.cmp td.t:first-child{white-space:normal;min-width:150px;max-width:240px}
.att-proche{color:var(--ok)}.att-loin{color:var(--hi2)}
.y{font-weight:700}.flip{background:var(--hi)}
.m{color:var(--faint);font-size:12px}
.cm td{width:30px;height:24px;padding:0;text-align:center;font-size:10px;border:1px solid var(--bg)}
.bar{display:flex;gap:8px;flex-wrap:wrap;align-items:center;margin:8px 0}
button{font:inherit;font-size:13px;padding:4px 10px;border:1px solid var(--rule);background:var(--card);color:var(--fg);border-radius:6px}
button[aria-pressed="true"]{border-color:var(--acc);color:var(--acc)}
button.joue{background:var(--acc);color:var(--bg)}
details{margin:6px 0 10px}summary{cursor:pointer}
.box{background:var(--card);border:1px solid var(--rule);border-radius:8px;padding:10px 12px;margin:10px 0}
.lesson{border-left:3px solid var(--acc);padding:2px 10px;margin:8px 0}
</style></head><body><div class="w">
<h1>Ressemblance entre deux accords : les candidates</h1>
<p class="l">Une seule mesure s(a, b | contexte) entre 0 et 1, au lieu des équivalences écrites une à une.
Rien n'est branché dans l'app. Seuil de chaque candidate = sa plus petite valeur sur les 5 paires que tu as déclarées
équivalentes (vert ci-dessous), posé <b>sans regarder les morceaux</b>.</p>

<h2>1. Les paires</h2>
<p class="l">Gras = « même accord » au seuil de la colonne. « actuel » = la règle d'aujourd'hui
(<code>roles.meme_lecture</code>, même fondamentale + même famille, ou B♭/C ≡ C7sus4).
Les étiquettes musx sont celles vers lesquelles chaque accord est projeté pour lire la table apprise.</p>
<div class="sc"><table id="panel"></table></div>
<div class="lesson" id="lecon1"></div>

<h2>2. Les cartes</h2>
<p class="l">Un accord de référence contre chaque fondamentale (colonnes) × qualité (lignes). Plus foncé = plus proche.
Contexte des cartes : C majeur.</p>
<div class="bar" id="refbar"></div>
<div class="bar" id="colbar"></div>
<div class="sc"><table class="cm" id="carte"></table></div>
<p class="m" id="cartenote"></p>

<h2>3. Sur trois morceaux</h2>
<p class="l">Chaque ligne = deux passages comparés mesure à mesure, temps par temps. ✓ = « les mêmes » : une mesure compte si
tous ses temps passent le seuil. « actuel » = la règle d'aujourd'hui au grain de cette décision (fin : même fondamentale ; boucle et brut :
même fondamentale + famille), appliquée temps par temps elle aussi. Fond jaune = la candidate dit autre chose que la règle actuelle.
<b>cran</b> = le plus petit cran du curseur « Section merging » (proposition, sur notes7) où les deux fusionneraient :
0 mêmes notes · 1 ≥ 0,72 (défaut proposé) · 2 ≥ 0,62 · 3 ≥ 0,49 · 4 tout. ▶ 1 / ▶ 2 jouent les deux passages.</p>
<div id="morceaux"></div>
<p class="m" id="pied"></p>
</div>
<script>
const D = __DATA__;
const cols = D.cols.map(c=>c.k);
const f = v => v==null ? '—' : v.toFixed(2);
const same = (k,v) => v!=null && D.seuils[k]!=null && v >= D.seuils[k]-1e-9;
// ── 1. panneau
(function(){
  let h = '<tr><th class="t">paire</th><th class="t">tonalité</th><th>actuel</th>' +
    cols.map(k=>`<th title="${D.cols.find(c=>c.k==k).d}">${k}<br><span class="m">seuil ${f(D.seuils[k])}</span></th>`).join('') +
    '<th>cran</th><th class="t">musx lit</th><th class="t">note</th></tr>';
  for(const r of D.panel){
    h += `<tr><td class="t att-${r.att}">${r.lab}</td><td class="t m">${r.cle}</td><td>${r.actuel?'<b>oui</b>':'non'}</td>` +
      cols.map(k=>{const v=r.v[k]; const s=same(k,v); const fl=(s!==r.actuel)&&v!=null;
        return `<td class="${s?'y':''} ${fl?'flip':''}">${f(v)}</td>`}).join('') +
      `<td><b>${r.cran}</b></td><td class="t m">${r.musx_lab}</td><td class="t m">${r.note}</td></tr>`;
  }
  document.getElementById('panel').innerHTML = h;
  document.getElementById('lecon1').innerHTML =
   '<b>Ce que ça montre.</b> « notes » et « notes7 » sont les seules à mettre les 5 paires vertes nettement au-dessus de toutes les rouges (0,75 contre 0,58 pour Fm/F7). '+
   '« notes_basse » casse ton cas de départ : D-7/G7sus4 change de basse, il tombe à 0,57, sous Fm/F7 (0,77). '+
   '« tiv » sépare aussi, mais de 0,07 seulement, et rapproche C de F (voisins de quinte : il mesure la TONALITÉ, pas l\'accord). '+
   '« musx » n\'hésite presque jamais entre deux fondamentales : D-7/G7sus4 vaut 0,03 (4× C/F, mais quasi nul) ; ses vraies hésitations sont les couleurs (C7/C9 0,54) et les accords de même basse. '+
   '« fonction » est grossière : ii et V7sus4 n\'ont pas la même fonction de manuel (0,5), C/Em = 1 mais C/Am = 0,5. '+
   'Le point faible de notes7 : tout couple de 4 notes qui en partage 3 vaut 0,75 — G7 ≈ Gm7 autant que D-7 ≈ G7sus4.';
})();
// ── 2. cartes
let refI = 0, colK = 'notes7';
function carte(){
  const M = D.cartes[colK][refI];
  let h = '<tr><th></th>' + D.roots.map(r=>`<th>${r}</th>`).join('') + '</tr>';
  M.forEach((row,qi)=>{
    h += `<tr><th class="t">${D.qb[qi]}</th>` + row.map((v,r)=>{
      if(v==null) return '<td class="m">·</td>';
      const s = same(colK,v);
      return `<td style="background:color-mix(in srgb, var(--acc) ${Math.round(Math.max(0,v)*100)}%, var(--bg));color:${v>0.55?'var(--bg)':'var(--fg)'};${s?'outline:2px solid var(--fg);outline-offset:-2px':''}" title="${D.roots[r]}${D.qb[qi]}: ${f(v)}">${v>=0.995?'1':v.toFixed(2).slice(1)}</td>`}).join('') + '</tr>';
  });
  document.getElementById('carte').innerHTML = h;
  document.getElementById('cartenote').textContent = `Référence ${D.refs[refI]}, candidate « ${colK} » (${D.cols.find(c=>c.k==colK).d}). Cadre = au-dessus du seuil ${f(D.seuils[colK])}.`;
  document.querySelectorAll('#refbar button').forEach((b,i)=>b.setAttribute('aria-pressed', i==refI));
  document.querySelectorAll('#colbar button').forEach(b=>b.setAttribute('aria-pressed', b.dataset.k==colK));
}
document.getElementById('refbar').innerHTML = '<span class="m">référence</span>' + D.refs.map((r,i)=>`<button onclick="refI=${i};carte()">${r}</button>`).join('');
document.getElementById('colbar').innerHTML = '<span class="m">candidate</span>' + cols.map(k=>`<button data-k="${k}" onclick="colK='${k}';carte()">${k}</button>`).join('');
carte();
// ── 3. morceaux
const scols = cols.concat(['notes7+voisins']);
const LECONS = {
 'min_0DdCoNbbRvQ': "Aucune décision ne bascule. notes7 change deux mesures (G7sus4 ≈ Dm7 au début de l'outro, G7sus4 ≈ Gm7 en mes. 8 du A), mais ce qui bloque est ailleurs : la mes. 2 du A (Dm7 tenu contre Dm7 | G7 — un décalage de rythme harmonique, 0,50) et la mes. 4 de l'outro (le C tenu de la cadence finale contre Fmaj7, 0,58). Les deux fusionneraient au cran 3.",
 'min_mCwI0YA4mIY': "Aucune décision ne bascule avec notes7. La fin C9 F7 rejoint C7sus4 Fmaj7 (0,75 ; déjà vrai aujourd'hui), les fins B♭m7… et B♭7 Fmaj7 restent de vraies fins (0,50). « notes » avec couleurs séparerait C9 de C7sus4 (0,67) : c'est pour ça qu'on retire 9/11/13. Dans le décodage brut, E♭7sus4 ≈ B♭m7 (le même cas que D-7/G7sus4, en E♭) devient « même accord ».",
 'min_maroon_5_this_love': "Les deuxièmes moitiés de B restent DIFFÉRENTES pour toutes les candidates (A♭ contre B♭ = 0, G contre E♭ = 0,33), même au cran 3 ; A et bridge aussi. Une seule bascule : l'intro, dont la mes. 4 (D° puis Fm/A♭) et la mes. 8 (Dø7) ont presque les mêmes notes (0,87) — notes7 fermerait la boucle de l'intro (4 mesures ×2). À toi de dire si c'est la même chose.",
};
const GENRES = [['fin','Les fins : une fin contre la plus jouée (et l\'outro contre le A)', true],
                ['boucle','La boucle : le 2e tour du bloc écrit contre le 1er', true],
                ['brut','Le décodage brut : chaque passe contre la passe de référence (accords non repliés)', false]];
const btn = (m,t,lab) => t ? `<button onclick="joue('${m.audio}',${t[0]},${t[1]},this)">▶ ${lab}</button>` : '';
function ligne(m,c){
  const dec = k => c.rows.every(r => r.s[k] >= D.seuils[k]-1e-9);
  const act = c.rows.every(r=>r.actuel);
  let h = `<tr><td class="t">${c.titre} <span class="m">(${c.cle})</span></td><td>${btn(m,c.ta,'1')} ${btn(m,c.tb,'2')}</td>` +
    `<td>${act?'✓':'✗'}</td>` + scols.map(k=>{const d=dec(k);return `<td class="${d!==act?'flip':''}">${d?'✓':'✗'}</td>`}).join('') +
    `<td><b>${Math.max(...c.rows.map(r=>r.cran))}</b></td></tr>`;
  h += `<tr><td colspan="${4+scols.length}" class="t"><details><summary class="m">mesure par mesure</summary><table><tr><th>mes.</th><th class="t">passage 1</th><th class="t">passage 2</th><th>actuel</th>` +
    scols.map(k=>`<th>${k}</th>`).join('') + '<th>cran</th></tr>' +
    c.rows.map(r=>`<tr><td class="m">${r.ma}/${r.mb}</td><td class="t">${r.a}</td><td class="t">${r.b}</td><td>${r.actuel?'✓':'✗'}</td>` +
      scols.map(k=>{const s=r.s[k]>=D.seuils[k]-1e-9; return `<td class="${s?'y':''} ${s!==r.actuel?'flip':''}" title="pire temps : ${r.pire[k]}">${f(r.s[k])}</td>`}).join('') + `<td>${r.cran}</td></tr>`).join('') +
    '</table></details></td></tr>';
  return h;
}
let html = '';
for(const m of D.morceaux){
  html += `<h3>${m.titre}</h3><p class="l">${m.pourquoi}</p>`;
  for(const [g, titre, ouvert] of GENRES){
    const cs_ = m.comps.filter(c=>c.genre===g);
    if(!cs_.length) continue;
    const nflip = cs_.filter(c=>{const act=c.rows.every(r=>r.actuel); return c.rows.every(r=>r.s['notes7']>=D.seuils['notes7']-1e-9)!==act}).length;
    html += `<details ${ouvert?'open':''}><summary><b>${titre}</b> <span class="m">— ${cs_.length} comparaison(s), ${nflip} bascule(nt) avec notes7</span></summary>` +
      `<div class="sc"><table class="cmp"><tr><th class="t">comparaison</th><th></th><th>actuel</th>` + scols.map(k=>`<th>${k}</th>`).join('') + '<th>cran</th></tr>' +
      cs_.map(c=>ligne(m,c)).join('') + '</table></div></details>';
  }
  html += `<div class="lesson"><b>Ce que ça montre.</b> ${LECONS[m.stem]||''}</div>`;
}
document.getElementById('morceaux').innerHTML = html;
document.getElementById('pied').textContent = `harmonia.chord_similarity v${D.version} · table musx : ${D.musx_diag.n_songs} morceaux, ${D.musx_diag.frames.all} images, fidélité moitié/moitié r = ${D.musx_diag.split_half_r}, GuitarSet contre le reste r = ${D.musx_diag.guitarset_vs_other_r}. Seuils : ${Object.entries(D.seuils).map(([k,v])=>k+' '+v).join(', ')}`;
// ── audio (fetch+blob, pas de rAF : pièges iOS du projet)
const cache = {}; let cour = null;
async function el(src){ if(cache[src]) return cache[src]; const a = new Audio();
  try{ const r = await fetch('/' + src); a.src = URL.createObjectURL(await r.blob()); }catch(e){ a.src = '/' + src; }
  a.preload='auto'; cache[src]=a; return a; }
function coupe(){ if(!cour) return; try{cour.a.pause();}catch(e){} cour.a.removeEventListener('timeupdate', cour.stop); cour.b.classList.remove('joue'); cour=null; }
async function joue(src,t0,t1,btn){ const re = cour && cour.b===btn; coupe(); if(re) return;
  const a = await el(src); const stop = ()=>{ if(a.currentTime>=t1) coupe(); };
  a.addEventListener('timeupdate', stop); try{ a.currentTime=t0; }catch(e){}
  btn.classList.add('joue'); cour={a,b:btn,stop}; a.play().catch(()=>{btn.classList.remove('joue'); cour=null;}); }
</script></body></html>
"""

if __name__ == "__main__":
    main()
