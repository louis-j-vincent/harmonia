"""Le banc d'essai des règles de sections : ses annotations, notre découpage.

    python scripts/section_bench.py                 # la règle vivante
    python scripts/section_bench.py --cache         # remplit le cache, puis sort

Ce fichier existe pour une seule raison : rendre une itération INSTANTANÉE.
Calculer nos sections sur un morceau demande musx (les postérieures d'accords),
demucs (la séparation de voix) et pyin (le suivi de hauteur) — de l'ordre de la
minute. Tester une règle sur les dix-sept morceaux annotés coûterait donc un
quart d'heure par essai, ce qui interdit toute recherche.

On met donc en cache ce que les règles NE changent PAS — la grille de mesures,
les vecteurs harmoniques, la matrice du chant, les mesures muettes, les notes —
et une règle candidate n'est plus qu'une fonction de ces entrées vers une liste
de sections. Un essai complet passe de quinze minutes à moins d'une seconde.

Le score est `section_metric.compare`, vérifié par `tests/test_section_metric.py`.
"""
from __future__ import annotations

import copy
import json
import os
import pickle
import sys
from pathlib import Path

import numpy as np

HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "scripts"))

from section_metric import compare                                  # noqa: E402

TRUTH_DIR = HERE / "docs/ground_truth/sections"
CACHE = HERE / "data/cache/section_bench"

# Easy : annoté sur 76 mesures, la grille en fait 70 depuis la réparation des
# temps insérés du 2026-08-07 (`beats.drop_inserted_beats`). L'annotation n'est
# plus posée sur la même grille, et elle n'avait jamais été validée. Écartée
# jusqu'à ce que Louis la refasse.
STALE = {"the_commodores_easy_1977"}


def truth(stem=None):
    """Les annotations validées de Louis, sur la grille où il les a posées."""
    out = {}
    for p in sorted(TRUTH_DIR.glob("*.json")):
        d = json.loads(p.read_text())
        if d["stem"] in STALE or not d.get("sections"):
            continue
        if stem and d["stem"] != stem:
            continue
        out[d["stem"]] = d
    return out


# ── le cache ────────────────────────────────────────────────────────────────

def features(stem, rebuild=False):
    """Tout ce qu'une règle de sections peut consommer, calculé une seule fois.

    grid   les temps de début de chaque mesure (n+1 valeurs)
    V      un vecteur de douze hauteurs par mesure (l'harmonie)
    S      V @ V.T, la matrice de similarité harmonique
    M      la matrice de similarité du CHANT (mélodie, mesure à mesure)
    mute   True là où personne ne chante
    notes  [(temps, durée, midi)] de la voix
    """
    CACHE.mkdir(parents=True, exist_ok=True)
    p = CACHE / f"{stem}.pkl"
    if p.exists() and not rebuild:
        with open(p, "rb") as f:
            return pickle.load(f)

    from harmonia_min import sections as hs, musx as mx, pipeline as _pl
    import harmonia_min.harmonic_sections as HS
    import vocal_anchor as VA
    import vocal_melody as VM
    import melody_ssm as MS

    audio = HERE / f"docs/audio/{stem}.m4a"
    real, c = hs.detect_sections, {}

    def spy(g, a, t, bars=None, **k):
        c.update(grid=list(map(float, g)), bars=copy.deepcopy(bars))
        return real(g, a, t, bars, **k)

    hs.detect_sections = spy
    try:
        _pl.analyze(audio, title="x", file_key="x", audio_url="")
    finally:
        hs.detect_sections = real

    grid = c["grid"]
    n = len(grid) - 1
    V = HS.harmonic_vectors(mx.frame_posteriors(audio)[0], grid)
    voc = VA.separate_vocals(audio)
    tt, ff, vv, rr = VM.track_f0(voc)
    notes, _ = VM.melody_notes(tt, ff, vv, rr)
    notes, _ = VM.clean(notes)
    M, mute = MS.melody_bars(notes, grid, n)
    d = {"stem": stem, "grid": grid, "n": n, "V": V, "S": V @ V.T,
         "M": M, "mute": np.asarray(mute, bool), "notes": notes,
         "bars": c.get("bars")}
    with open(p, "wb") as f:
        pickle.dump(d, f)
    return d


# ── les règles candidates ───────────────────────────────────────────────────

def rule_live(F):
    """La règle telle qu'elle tourne en prod (`HARMONIA_SECTIONS=voice`)."""
    from harmonia_min import voice_sections as VS
    n = F["n"]
    start = VS.sung_start(F["notes"], F["grid"], F["mute"])
    claimed = np.zeros(n, bool)
    runs = VS._pass(F["S"], F["M"], F["mute"], n, start, VS.BLOCK, VS.THR8, claimed)
    runs += VS._pass(F["S"], F["M"], F["mute"], n, start, VS.FILL, VS.THR4, claimed)
    return assemble(runs, n, start)


def assemble(runs, n, start):
    """runs -> sections contiguës, comme `voice_sections.detect_sections`.

    Doit rester le MIROIR de la mise en forme livrée, y compris pour la
    séparation des occurrences adjacentes : sans ça le banc mesurerait une
    version qui n'existe nulle part.
    """
    owner, occid, k = np.full(n, -1), np.full(n, -1), 0
    for i, r in enumerate(runs):
        for c in [r["b0"]] + r["occ"]:
            owner[c:min(n, c + r["block"])] = i
            occid[c:min(n, c + r["block"])] = k
            k += 1
    out, b = [], 0
    if start > 0:
        out.append({"b0": 0, "b1": start - 1, "label": "intro"})
        b = start
    while b < n:
        z = b
        while z + 1 < n and owner[z + 1] == owner[b] and occid[z + 1] == occid[b]:
            z += 1
        out.append({"b0": b, "b1": z, "label": owner[b]})
        b = z + 1
    ren, k = {}, 0
    for s in out:
        if isinstance(s["label"], (int, np.integer)):
            if s["label"] not in ren:
                ren[s["label"]] = chr(ord("A") + k)
                k += 1
            s["label"] = ren[s["label"]]
    return out


# ── la mesure ───────────────────────────────────────────────────────────────

def bench(rule, stems=None, quiet=False, ref=None):
    """Une règle contre toute la vérité. Renvoie {stem: résultat} + la moyenne."""
    T = truth()
    stems = stems or sorted(T)
    out = {}
    for st in stems:
        F = features(st)
        n = min(F["n"], T[st]["n"])
        pred = rule(F)
        r = compare(pred, T[st]["sections"], n)
        r["pred"] = pred
        out[st] = r
    m = float(np.mean([r["score"] for r in out.values()])) if out else 0.0
    if not quiet:
        print(f"{'morceau':46} {'score':>6} {'spans':>6} {'lettres':>7} "
              f"{'sect':>5}" + (f" {'Δ':>6}" if ref else ""))
        for st in stems:
            r = out[st]
            d = f" {r['score'] - ref[st]['score']:+6.3f}" if ref else ""
            print(f"{st[:44]:46} {r['score']:>6.3f} {r['spans']:>6.3f} "
                  f"{r['letters']:>7.3f} {r['n_pred']:>2d}/{r['n_ref']:<2d}{d}")
        line = f"{'MOYENNE':46} {m:>6.3f}"
        if ref:
            line += f" {'':>6} {'':>7} {'':>5} " \
                    f"{m - float(np.mean([ref[s]['score'] for s in stems])):+6.3f}"
        print(line)
    return out, m


def main():
    if "--cache" in sys.argv:
        for st in sorted(truth()):
            F = features(st)
            print(f"  {st[:46]:48} n={F['n']}")
        return
    bench(rule_live)


if __name__ == "__main__":
    main()
