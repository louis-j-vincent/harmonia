"""Les zones de tonalité qu'on MONTRE sur le chart (Louis, 2026-10-13).

`harmonic_key.analyze_harmony` découpe le morceau en `keySegments`
({t0, t1, tonic, mode, x3, …}). Ce découpage est fait pour colorier les
fonctions, pas pour écrire une modulation : il peut contenir un segment de
deux mesures ou un passage au relatif. Ce module en tire les ZONES que le
chart écrit (une pastille « → A♭ majeur » au début de chacune, et
l'orthographe des accords ♭/♯ selon la zone).

La règle, une phrase : un segment ouvre une nouvelle zone s'il CHANGE LA
GAMME (l'armure, c.-à-d. la majeure relative : sol majeur → mi mineur n'en
change pas) et s'il dure au moins une phrase (`PHRASE_BARS` = 4 mesures de
la grille du chart lui-même). Sinon il est rendu à la zone qui le précède
(au premier segment : à la zone qui le suit). Ce n'est pas un seuil réglé :
4 mesures est la plus petite phrase de la musique qu'on écrit ici, la même
unité que la règle des sections (« une section = une mini-boucle ≥ 4
mesures »).

Ce que ça NE résout PAS :
  * `x3` (la part de tierce majeure, `mode_audit`) dit le MODE d'un
    segment, pas la confiance du segment ; il n'est donc pas utilisé ici.
    Un segment long mais faux (emprunt au IV mineur tenu 8 mesures, p. ex.)
    passe la règle — voir la liste du corpus dans `docs/known_issues.md`.
  * Les couleurs de fonction et le Transposer ne suivent pas les zones
    (choix de Louis : seule l'orthographe change).
  * Les zones sont en SECONDES : sur un chart replié, une mesure écrite une
    fois mais jouée dans deux tonalités prend celle de son premier passage.
"""
from __future__ import annotations

from bisect import bisect_left

PHRASE_BARS = 4


def _signature(tonic: int, mode: str) -> int:
    """L'armure : la tonique de la majeure relative."""
    return int(tonic) % 12 if mode != "minor" else (int(tonic) + 3) % 12


def _bar_starts(bar_grid, beat_times, bpb) -> list[float]:
    if bar_grid:
        return sorted(float(t) for t in bar_grid)
    if beat_times:
        return [float(t) for t in list(beat_times)[::max(1, int(bpb or 4))]]
    return []


def _n_bars(starts: list[float], t0: float, t1: float) -> int:
    return bisect_left(starts, t1) - bisect_left(starts, t0)


def key_zones(key_segments, bar_grid, *, beat_times=None, bpb=4) -> list[dict]:
    """[{t0, t1, tonic, mode, bars}] — zones contiguës couvrant le morceau.

    Une seule zone = le morceau ne change pas de gamme (aucune pastille)."""
    segs = sorted((s for s in (key_segments or []) if s), key=lambda s: s["t0"])
    if not segs:
        return []
    starts = _bar_starts(bar_grid, beat_times, bpb)
    song_t0, song_t1 = float(segs[0]["t0"]), float(segs[-1]["t1"])

    kept = [s for s in segs
            if _n_bars(starts, float(s["t0"]), float(s["t1"])) >= PHRASE_BARS]
    if not kept:
        kept = [max(segs, key=lambda s: s["t1"] - s["t0"])]

    zones: list[dict] = []
    for s in kept:
        if zones and _signature(zones[-1]["tonic"], zones[-1]["mode"]) == \
                _signature(s["tonic"], s["mode"]):
            continue                      # même gamme : la zone continue
        zones.append({"t0": float(s["t0"]), "tonic": int(s["tonic"]) % 12,
                      "mode": "minor" if s["mode"] == "minor" else "major"})
    zones[0]["t0"] = song_t0
    for a, b in zip(zones, zones[1:]):
        a["t1"] = b["t0"]
    zones[-1]["t1"] = song_t1
    for z in zones:
        z["bars"] = _n_bars(starts, z["t0"], z["t1"])
    return zones


def key_zones_for_model(model: dict) -> list[dict]:
    return key_zones(model.get("keySegments"), model.get("barGrid"),
                     beat_times=model.get("beatTimes"), bpb=model.get("bpb") or 4)
