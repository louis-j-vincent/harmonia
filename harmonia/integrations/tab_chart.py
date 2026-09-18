"""Un tab posé sur un morceau déjà analysé — comme source de chart, et comme
deuxième avis sur nos accords.

Louis, 2026-09-18 : « branche-nous ça comme façon alternative de choper des
charts, et tu vas t'en servir pour flagger si on a fait des détections
d'accords douteux ».

CE QUI A CHANGÉ. L'import de tablature avait été abandonné au refactor, et la
raison était bonne : « une tablature n'a ni mesures ni temps, en faire un
chart demande l'alignement audio, pas construit » (`routes/irealb.py`).
L'alignement existe maintenant (`tab_align.poser_tout`), donc la raison est
tombée.

LES DEUX USAGES PARTENT DU MÊME CALCUL et c'est voulu — une seule façon de
poser un tab sur un morceau, pas deux qui divergeront.

  `poser()`   le tab placé sur la ligne du temps du chart ;
  `doutes()`  là où le tab n'est pas d'accord avec ce qu'on a écrit ;
  `chart()`   un chart complet dont les accords viennent du tab, et les
              sections du rasoir d'Occam (`tab_structure`).

CE QUE ÇA NE FAIT PAS.

  * Ça ne remplace jamais un chart existant. `chart()` rend un objet ; c'est
    l'appelant qui décide où l'écrire, et la route l'écrit sous une AUTRE clé.
  * Ça ne juge pas qui a raison. Un doute est un désaccord entre deux sources,
    pas une erreur démontrée — le tab est plus pauvre que nos charts
    (2026-08-05), il se trompe aussi. C'est l'oreille de Louis qui tranche ;
    la marque sert à lui dire où écouter.
  * Un accord CONFIRMÉ à la main n'est jamais marqué. Ce qu'il a validé est la
    vérité terrain, point.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from harmonia.integrations import tab_structure as TS
from harmonia.integrations.tab_align import (NOMS, compresser, lire_accord,
                                             meilleure_transposition, nom_q5,
                                             poser_tout, priors_musx,
                                             sequence_du_tab)
from harmonia.settings import SETTINGS

log = logging.getLogger("harmonia.tab_chart")

#: en dessous, on ne se sert pas du tab — un tab mal noté n'est pas un avis
NOTE_MINI = 4.5

#: du vocabulaire du chart vers celui d'un tab, pour ne lire les accords qu'à
#: un seul endroit
VERS_TAB = {"": "", "-": "m", "-7": "m7", "o": "dim", "h7": "m7b5",
            "^7": "maj7", "^": "maj7", "7": "7", "+": "", "sus": "sus4"}


def lire_du_chart(ch: dict) -> tuple | None:
    """Un accord du chart → `(root, q5, basse)`, ou None s'il ne dit rien."""
    if ch.get("nc"):
        return None
    lu = lire_accord(NOMS[int(ch.get("root", 0)) % 12]
                     + VERS_TAB.get(str(ch.get("q") or ""),
                                    str(ch.get("q") or "")))
    if lu is None:
        return None
    b = ch.get("bass", -1)
    return (lu["root"], lu["q5"],
            None if b is None or int(b) < 0 else int(b) % 12)


def texte_du_chart(ch: dict) -> str:
    if ch.get("nc"):
        return "N.C."
    s = NOMS[int(ch.get("root", 0)) % 12] + str(ch.get("q") or "")
    b = ch.get("bass", -1)
    if b is not None and int(b) >= 0 and int(b) != int(ch.get("root", -1)):
        s += "/" + NOMS[int(b) % 12]
    return s


def poser(chart: dict, requete: str, url: str | None = None) -> dict | None:
    """Le tab le mieux noté, posé sur la ligne du temps de ce chart.

    Rend None — et DIT pourquoi dans le journal — si rien n'est utilisable :
    pas d'audio, pas de grille, aucun tab au-dessus de `NOTE_MINI`, ou un tab
    qui a plus d'accords que le morceau n'a de temps.
    """
    from harmonia.integrations.tab_fetcher import fetch_tab_chords, search_tabs

    stem = Path(chart.get("audio_url") or "").stem
    audio = SETTINGS.audio_dir / f"{stem}.m4a"
    grille = [float(t) for t in (chart.get("barGrid") or [])]
    temps = [float(t) for t in (chart.get("beatTimes") or [])]
    if not audio.exists() or len(grille) < 3 or len(temps) < 8:
        log.info("tab : %s n'a pas d'audio ou pas de grille", stem)
        return None

    if url:
        trouve = next((r for r in search_tabs(requete, tab_types=("Chords",),
                                              max_results=12)
                       if r.tab_url == url), None)
        res = [trouve] if trouve else []
    else:
        res = [r for r in search_tabs(requete, tab_types=("Chords",),
                                      max_results=6) if r.rating >= NOTE_MINI]
    if not res or res[0] is None:
        log.info("tab : rien au-dessus de %.1f étoiles pour %r",
                 NOTE_MINI, requete)
        return None
    brut = fetch_tab_chords(res[0])
    seq = compresser(sequence_du_tab(brut.raw_content))
    if not seq:
        log.info("tab : aucun accord lisible dans %s", res[0].tab_url)
        return None

    P = priors_musx(audio, grille)
    dec, scores = meilleure_transposition(seq, P)
    Pt = priors_musx(audio, temps)
    bpb = int(chart.get("bpb") or 4)
    origine = int(np.argmin([abs(t - grille[0]) for t in temps]))
    chemin = poser_tout(seq, Pt, dec, bpb=bpb, origine=origine)
    if chemin is None:
        log.info("tab : %d accords pour %d temps sur %s — infaisable",
                 len(seq), len(temps) - 1, stem)
        return None

    segments = []
    for t, i in enumerate(chemin):
        if segments and segments[-1]["i"] == i:
            segments[-1]["t1"] = temps[min(t + 1, len(temps) - 1)]
            continue
        c = seq[i]
        b = None if c.get("bass") is None else (c["bass"] + dec) % 12
        segments.append({
            "i": i, "t0": temps[t], "t1": temps[min(t + 1, len(temps) - 1)],
            "root": (c["root"] + dec) % 12, "q5": c["q5"], "bass": b,
            "texte": nom_q5((c["root"] + dec) % 12, c["q5"], b),
            "section": c.get("section") or ""})

    mots = TS.mots_par_mesure(segments, grille, temps)
    return {"segments": segments, "seq": seq, "decalage": dec,
            "marge": round(scores[0][0] - scores[1][0], 3),
            "grille": grille, "temps": temps, "bpb": bpb,
            "forme": TS.forme(mots), "mots": mots,
            "tab": {"titre": res[0].song_name, "artiste": res[0].artist_name,
                    "note": round(res[0].rating, 2), "votes": res[0].votes,
                    "url": res[0].tab_url}}


def _a_l_instant(segments: list[dict], t: float) -> dict | None:
    for g in segments:
        if g["t0"] - 1e-6 <= t < g["t1"] - 1e-6:
            return g
    return None


def doutes(chart: dict, pose: dict, part_mini: float = 0.5) -> list[dict]:
    """Les accords ÉCRITS du chart sur lesquels le tab n'est pas d'accord.

    Un accord écrit est joué plusieurs fois (le chart est replié). On regarde
    chacun de ses passages et on ne marque que si le tab conteste au moins
    `part_mini` d'entre eux : contester un passage sur quatre, c'est du bruit
    d'alignement, pas un désaccord d'accord.

    Deux gravités, et la différence compte :

      « fondamentale » — pas la même fondamentale, ou pas la même basse. La
        cible de ce projet est la basse qui SONNE (2026-07-16), donc une basse
        différente est un vrai désaccord, pas un détail ;
      « couleur »      — même fondamentale et même basse, autre famille. Le
        tab écrit souvent `D°` là où on écrit `Dø` ; c'est de l'orthographe et
        ça vaut moins qu'un point d'exclamation.
    """
    segs, temps = pose["segments"], pose["temps"]
    out = []
    for sec in chart.get("sections") or []:
        spans = sec.get("barSpans") or []
        for k, bar in enumerate(sec.get("bars") or []):
            passages = spans[k] if k < len(spans) else []
            for j, ch in enumerate(bar):
                if ch.get("confirmed") or ch.get("nc"):
                    continue          # sa validation est la vérité terrain
                notre = lire_du_chart(ch)
                if notre is None:
                    continue
                vus = conteste = 0
                propose: dict[str, int] = {}
                grave = False
                for a0, a1 in passages:
                    duree = max(1e-6, a1 - a0)
                    d0 = a0 + duree * (float(ch.get("beat") or 0)
                                       / max(1, int(chart.get("bpb") or 4)))
                    d1 = (a0 + duree * (float(bar[j + 1].get("beat") or 0)
                                        / max(1, int(chart.get("bpb") or 4)))
                          if j + 1 < len(bar) else a1)
                    dedans = [t for t in temps if d0 - 1e-6 <= t < d1 - 1e-6]
                    if not dedans:
                        continue
                    g = _a_l_instant(segs, dedans[0])
                    if g is None:
                        continue
                    vus += 1
                    bn = notre[2] if notre[2] is not None else notre[0]
                    bt = g["bass"] if g["bass"] is not None else g["root"]
                    if notre[0] != g["root"] or bn % 12 != bt % 12:
                        conteste += 1
                        grave = True
                        propose[g["texte"]] = propose.get(g["texte"], 0) + 1
                    elif notre[1] != g["q5"]:
                        conteste += 1
                        propose[g["texte"]] = propose.get(g["texte"], 0) + 1
                if vus and conteste / vus >= part_mini:
                    out.append({
                        "bar": ch.get("bar"), "beat": ch.get("beat"),
                        "notre": texte_du_chart(ch),
                        "tab": max(propose, key=propose.get) if propose else "",
                        "gravite": "fondamentale" if grave else "couleur",
                        "passages": vus, "contestes": conteste})
    return out


#: de nos cinq familles vers le vocabulaire qu'un chart écrit
VERS_CHART = {0: "", 1: "-", 2: "7", 3: "h7", 4: "o"}


def chart(source: dict, pose: dict, titre: str | None = None,
          stem: str | None = None) -> dict:
    """Un chart complet dont les ACCORDS viennent du tab.

    Tout ce qui vient de l'audio est repris tel quel du chart source — la
    grille de mesures, les temps, la tonalité, le point de départ. Rien de
    tout ça ne dépend de qui fournit les accords, et le recalculer donnerait
    deux vérités là où il en faut une.

    Les SECTIONS viennent du rasoir d'Occam (`tab_structure.forme`) : chaque
    lettre est une section, ses reprises sont ses occurrences. C'est
    exactement la forme de repli qu'un chart attend — une section écrite une
    fois, jouée n fois — donc rien à replier après coup.

    LA CONFIANCE de chaque accord est la masse que musx lui donne en moyenne
    sur les temps qu'il couvre. Ce n'est pas « le tab a raison à 0,8 » : c'est
    « l'audio soutient cet accord-là à 0,8 ». Un accord que le tab impose
    contre l'audio sort bas, et se voit.
    """
    grille, temps = pose["grille"], pose["temps"]
    bpb = pose["bpb"]
    segs, forme = pose["segments"], pose["forme"]

    # la confiance : ce que musx donne à cet accord, en moyenne, sur sa durée
    audio = SETTINGS.audio_dir / f"{Path(source.get('audio_url') or '').stem}.m4a"
    try:
        Pt = priors_musx(audio, temps)
    except Exception:                                      # noqa: BLE001
        log.warning("tab : postérieures illisibles, confiance à 0.5")
        Pt = None
    for g in segs:
        if Pt is None:
            g["c"] = 0.5
            continue
        idx = g["root"] * 5 + g["q5"]
        vals = [float(Pt[k][idx]) for k, t in enumerate(temps[:Pt.shape[0]])
                if g["t0"] - 1e-6 <= t < g["t1"] - 1e-6]
        g["c"] = round(float(np.mean(vals)), 3) if vals else 0.5

    def accords_de(b: int) -> list[dict]:
        """Les accords à écrire dans la mesure `b` — ceux qui y commencent, et
        celui qui la tient si aucun n'y commence."""
        a0, a1 = grille[b], grille[b + 1]
        dans = [g for g in segs if a0 - 1e-6 <= g["t0"] < a1 - 1e-6]
        if not dans:
            tenu = _a_l_instant(segs, a0)
            if tenu is None:
                return []
            dans = [{**tenu, "t0": a0, "porte": True}]
        out = []
        for k, g in enumerate(dans):
            # le temps dans la mesure, en nombre de temps
            beat = sum(1 for t in temps if a0 - 1e-6 <= t < g["t0"] - 1e-6)
            out.append({
                "root": g["root"], "q": VERS_CHART.get(g["q5"], ""),
                "bass": -1 if g["bass"] is None else int(g["bass"]),
                "nc": False, "carry": bool(g.get("porte")),
                "beat": int(beat), "bar": b, "c": g.get("c", 0.5),
                "t0": round(max(a0, g["t0"]), 3),
                "t1": round(min(a1, g["t1"]), 3) if k + 1 >= len(dans)
                      else round(dans[k + 1]["t0"], 3),
                "colour": "natural", "n": 0, "source": "tab"})
        return out

    # une section par lettre, ses reprises en occurrences
    par_lettre: dict[str, list[tuple]] = {}
    for a, b, lettre in forme["sections"]:
        par_lettre.setdefault(lettre, []).append((a, b))
    sections = []
    for lettre, occs in par_lettre.items():
        longueur = occs[0][1] - occs[0][0]
        bars, barspans = [], []
        for k in range(longueur):
            bars.append(accords_de(occs[0][0] + k))
            barspans.append([[round(grille[a + k], 3),
                              round(grille[a + k + 1], 3)]
                             for a, b in occs if a + k + 1 < len(grille)])
        sections.append({
            "id": lettre, "label": lettre, "tag": None, "reps": len(occs),
            "barRanges": [[a, b - 1] for a, b in occs],
            "spans": [[round(grille[a], 3),
                       round(grille[min(b, len(grille) - 1)], 3)]
                      for a, b in occs],
            "bars": bars, "barSpans": barspans})
    # dans l'ordre où le morceau les joue
    sections.sort(key=lambda s: s["barRanges"][0][0])

    t = pose["tab"]
    return {
        "file": stem or f"tab_{Path(source.get('audio_url') or '').stem}",
        "title": titre or (source.get("title") or t["titre"]),
        "audio_url": source.get("audio_url"), "video_id": source.get("video_id"),
        "barGrid": grille, "beatTimes": temps, "bpb": bpb,
        "bar1": source.get("bar1"), "nBars": len(grille) - 1,
        "key": source.get("key"), "keyName": source.get("keyName"),
        "keySegments": source.get("keySegments"),
        "fold": None, "form": None, "prompter": None,
        "sections": sections,
        "meta": {**(source.get("meta") or {}), "engine": "tab",
                 "raw": False, "pending": False,
                 "tab": t, "transposition": pose["decalage"],
                 "marge_transposition": pose["marge"],
                 "occam": {"mesures_ecrites": forme["cout"][0],
                           "sections": forme["cout"][1],
                           "posees": forme["cout"][2],
                           "forme": TS.mot(forme["sections"])}},
    }
