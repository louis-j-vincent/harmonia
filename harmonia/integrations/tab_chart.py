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


def _voix_de(chart: dict):
    """La voix isolée de ce morceau, si elle existe — sinon None.

    Elle sert au quatrième critère (le chant départage deux tabs à égalité sur
    les accords) ; son absence ne bloque rien.
    """
    from harmonia.settings import SETTINGS
    stem = Path(str(chart.get("audio_url") or "")).stem
    if not stem:
        return None
    v = SETTINGS.data_cache / "stems" / "htdemucs" / stem / "vocals.wav"
    return v if v.exists() else None


def poser(chart: dict, requete: str, url: str | None = None, *,
          artiste: str = "") -> dict | None:
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
        # URL EXPLICITE = CHOIX DE LOUIS. On ne la discute pas : il a vu le tab
        # et l'a désigné. La porte à quatre critères ne s'applique qu'à ce que
        # NOUS choisissons à sa place.
        trouve = next((r for r in search_tabs(requete, tab_types=("Chords",),
                                              max_results=12)
                       if r.tab_url == url), None)
        res = [trouve] if trouve else []
        if not res or res[0] is None:
            log.info("tab : l'URL demandée est introuvable pour %r", requete)
            return None
        brut_txt = fetch_tab_chords(res[0]).raw_content
    else:
        # LA MÊME PORTE QUE PARTOUT AILLEURS (2026-10-02). `poser` cherchait
        # par TITRE SEUL et ne filtrait que sur la note — il pouvait donc
        # aligner la grille d'une AUTRE chanson, et `tab-doutes` contestait
        # alors nos accords au nom d'un morceau sans rapport. Titre, artiste,
        # accords à transposition près, puis le chant pour départager.
        choix = tab_du_morceau(chart, requete, artiste, voix=_voix_de(chart))
        if choix is None:
            log.info("tab : aucun tab ne correspond vraiment à %r de %r",
                     requete, artiste or "?")
            return None
        res = [choix["tab"]]
        brut_txt = choix["brut"]
    # on ne manipule que le TEXTE du tab : reconstruire l'objet du fetcher
    # cassait dès qu'il gagnait un champ (vécu deux fois)
    seq = compresser(sequence_du_tab(brut_txt))
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
    # LE TEXTE DU TAB VOYAGE AVEC LA POSE. Sans lui, un appelant qui veut les
    # paroles refait une recherche et peut tomber sur un AUTRE tab que celui
    # sur lequel on vient de se caler (c'était le cas de tools/tab_paroles_page).
    return {"brut": brut_txt,
            "segments": segments, "seq": seq, "decalage": dec,
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


# ── LA PORTE À TROIS CRITÈRES ───────────────────────────────────────────────
#: part du vocabulaire du tab qu'on doit retrouver dans le nôtre, à la
#: meilleure transposition, pour croire que c'est le même morceau
ACCORD_MINI = 0.50


def _normal(x: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]", "", (x or "").lower())


def meme_identite(titre: str, artiste: str, t) -> bool:
    """Le titre ET l'artiste de ce tab concordent-ils avec les nôtres ?"""
    tt, ta = _normal(t.song_name), _normal(t.artist_name)
    nt, na = _normal(titre), _normal(artiste)
    if na and not (na in ta or ta in na):
        return False
    return bool(nt) and (nt in tt or tt in nt)


def _vocab_tab(brut: str) -> set:
    import re

    from harmonia.integrations.tab_align import lire_accord
    out = set()
    for tok in re.findall(r"\[ch\](.*?)\[/ch\]", brut or ""):
        a = lire_accord(tok.strip())
        if a:
            out.add((a["root"], a["q5"]))
    return out


def _vocab_chart(chart: dict) -> set:
    out = set()
    for sec in chart.get("sections") or []:
        for b in sec.get("bars") or []:
            for c in b:
                t = lire_du_chart(c)
                if t:
                    out.add((t[0], t[1]))
    return out


def concordance(chart: dict, brut: str) -> tuple[float, int]:
    """(part, transposition) — combien du vocabulaire du tab on retrouve.

    On essaie les douze transpositions et on garde la meilleure : un tab écrit
    au capo, ou dans une autre tonalité, reste le même morceau.
    """
    nous, leur = _vocab_chart(chart), _vocab_tab(brut)
    if not nous or not leur:
        return 0.0, 0
    best, dec = 0.0, 0
    for d in range(12):
        t = {((r + d) % 12, q) for (r, q) in leur}
        s = len(t & nous) / len(t)
        if s > best:
            best, dec = s, d
    return best, dec


def score_des_paroles(voix, brut: str) -> float | None:
    """À quel point le chant entendu colle aux paroles de ce tab (0 à 1).

    Idée de Louis, 2026-10-01 : « on a un détecteur cheap de paroles qui prend
    comme prior les paroles du tab ; celui-là devrait pouvoir donner un score
    de à quel point les paroles entendues collent avec le prior ». L'aligneur
    forcé rend déjà un score par mot ; on en prend la médiane.

    CE QUE ÇA NE PEUT PAS FAIRE, et c'est mesuré : servir de seuil absolu. Sur
    huit morceaux, le bon tab et un tab intrus se CHEVAUCHENT —

        bon tab   min 0,130 · médiane 0,377 · max 0,694
        intrus    min 0,041 · médiane 0,140 · max 0,332

    le pire bon (Chain of Fools, un gospel très mélismatique où l'aligneur
    peine même avec le bon texte) tombe SOUS le meilleur intrus (un tab court
    plaqué sur un morceau long, qui trouve des correspondances fortuites). Le
    score dépend donc autant de la difficulté du chant et de la longueur du
    texte que de la justesse du tab.

    CE QUE ÇA FAIT TRÈS BIEN : départager deux candidats SUR LE MÊME AUDIO. Là,
    le bon l'emporte 8 fois sur 8, de ×1,1 à ×4,7. C'est un classement, pas une
    note — comme le seuil de rappel et les accords de passage, le signal est
    relatif à son morceau.

    Rend None si la voix isolée manque ou si rien ne s'aligne. L'appel coûte
    une dizaine de secondes : il n'a lieu qu'en cas d'ambiguïté réelle.
    """
    import statistics as st

    from harmonia.integrations import paroles_align as PA
    try:
        mots = PA.mots_a_aligner(PA.lignes_de_paroles(brut))
        if not mots:
            return None
        al = PA.aligner(voix, mots)
    except Exception:                                        # noqa: BLE001
        log.exception("score_des_paroles : alignement impossible")
        return None
    sc = [m["score"] for m in al if m.get("score") is not None]
    return float(st.median(sc)) if sc else None


def _cle_choix(titre: str, artiste: str) -> str:
    import hashlib
    return hashlib.sha1(f"{titre}|{artiste}".encode()).hexdigest()[:16]


def choix_en_cache(titre: str, artiste: str) -> dict | None:
    """Le tab déjà choisi pour ce morceau, ou None.

    LE CHOIX NE CHANGE PAS D'UNE OUVERTURE À L'AUTRE, et il coûte cher : la
    porte lit jusqu'à huit tabs sur le réseau et, en cas d'égalité sur les
    accords, aligne le chant sur deux ou trois d'entre eux (10 à 20 s chacun).
    Sans ce cache la page mettait 22 à 56 s à s'ouvrir — mesuré. On le garde
    donc sur disque, et la page retombe à sa seconde habituelle.

    Effacer `state/cache/tabs/` suffit à tout refaire décider.
    """
    from harmonia.settings import SETTINGS
    f = SETTINGS.cache_dir / "tabs" / f"{_cle_choix(titre, artiste)}.json"
    if not f.exists():
        return None
    try:
        import json as _j
        return _j.loads(f.read_text())
    except Exception:                                        # noqa: BLE001
        log.exception("choix_en_cache : %s illisible", f)
        return None


def _garder_choix(titre: str, artiste: str, d: dict) -> None:
    from harmonia.settings import SETTINGS
    try:
        import json as _j
        dossier = SETTINGS.cache_dir / "tabs"
        dossier.mkdir(parents=True, exist_ok=True)
        (dossier / f"{_cle_choix(titre, artiste)}.json").write_text(
            _j.dumps(d, ensure_ascii=False))
    except Exception:                                        # noqa: BLE001
        log.exception("_garder_choix : écriture impossible")


def tab_du_morceau(chart: dict, titre: str, artiste: str = "", *,
                   accord_mini: float = ACCORD_MINI, voix=None,
                   cache: bool = True):
    """Le meilleur tab qui soit VRAIMENT ce morceau, ou None.

    Louis, 2026-09-30 : « il faut que ça matche sur titre + artiste + accords
    (à transpo près), sinon c'est pas la peine de proposer ».

    LES TROIS CRITÈRES, ET LEUR ORDRE COMPTE. Mesuré sur les 22 morceaux à
    sections faites main, la recherche par TITRE SEUL — ce que tout le code
    faisait — prend le mauvais artiste 8 fois. Exemples relevés :

        « Bora Bora » (Henri Salvador)  ->  « Electric Love » de BØRNS
        « Bein Green »                  ->  « Tired Of Being Alone » d'Al Green
        « The Walk » (Mayer Hawthorne)  ->  « The Walk » de Paramore

    ET LES ACCORDS SEULS NE SUFFIRAIENT PAS : sur « The Lazy Song », le tab de
    Bob Marley concorde avec nos accords à 1,00 quand celui de Bruno Mars — le
    bon — est à 0,86. Deux chansons à trois accords se ressemblent
    parfaitement. L'identité passe donc AVANT, et les accords ne font que
    confirmer.

    LE SEUIL, mesuré. Une fois l'identité imposée, les vrais tabs concordent à
    1,00 (8 morceaux) ou entre 0,60 et 0,89 (9 morceaux) ; un seul tombe à 0,32
    (Every Breath You Take, dont le tab décrit un autre arrangement). On coupe
    donc à 0,50, entre les deux populations.

    RIEN PLUTÔT QUE N'IMPORTE QUOI : sans tab qui passe les trois critères, on
    rend None, et l'appelant ne propose pas de feuille. C'est la consigne.
    """
    from harmonia.integrations.tab_fetcher import fetch_tab_chords, search_tabs
    if cache:
        garde = choix_en_cache(titre, artiste)
        if garde is not None:
            if not garde.get("url"):
                log.info("tab_du_morceau : %r de %r — aucun tab (en cache)",
                         titre, artiste or "?")
                return None
            from harmonia.integrations.tab_fetcher import TabResult
            # on rejoue le résultat TEL QUEL : reconstruire une poignée de
            # champs cassait dès que la classe en gagnait un (vécu)
            t = TabResult(**garde["tab"])
            try:
                brut = fetch_tab_chords(t).raw_content
            except Exception:                                # noqa: BLE001
                brut = None
            if brut:
                log.info("tab_du_morceau : %r de %r (en cache) — accords "
                         "%.0f %%", t.song_name, t.artist_name,
                         100 * garde.get("accords", 0))
                return {"tab": t, "brut": brut,
                        "accords": garde.get("accords", 0),
                        "decalage": garde.get("decalage", 0),
                        "paroles": garde.get("paroles")}
    try:
        res = search_tabs(titre, artiste, tab_types=("Chords",), max_results=8)
    except Exception:                                        # noqa: BLE001
        log.exception("tab_du_morceau : recherche impossible pour %r", titre)
        return None
    cands = [t for t in res if meme_identite(titre, artiste, t)]
    if not cands:
        log.info("tab_du_morceau : aucun tab dont l'identité soit %r de %r",
                 titre, artiste or "?")
        if cache:
            _garder_choix(titre, artiste, {"url": None})
        return None
    passent = []
    for t in cands:
        try:
            brut = fetch_tab_chords(t).raw_content
        except Exception:                                    # noqa: BLE001
            continue
        part, dec = concordance(chart, brut)
        if part >= accord_mini:
            passent.append({"tab": t, "brut": brut, "accords": part,
                            "decalage": dec})
        else:
            log.info("tab_du_morceau : %r de %r ÉCARTÉ — accords %.0f %% "
                     "sous le seuil de %.0f %%", t.song_name, t.artist_name,
                     100 * part, 100 * accord_mini)
    if not passent:
        if cache:
            _garder_choix(titre, artiste, {"url": None})
        return None

    # LE CHANT DÉPARTAGE, quand et seulement quand plusieurs tabs passent les
    # deux premières portes. Le score d'alignement ne sait pas dire « celui-ci
    # est bon » dans l'absolu (voir `score_des_paroles`), mais il sait dire
    # lequel de deux candidats colle le mieux à CE chant-ci. Un alignement
    # coûte une dizaine de secondes : sans ambiguïté, on ne le paie pas.
    passent.sort(key=lambda c: -c["accords"])
    # ON NE PAIE LA VOIX QUE SUR UNE VRAIE ÉGALITÉ. Un alignement coûte 10 à
    # 20 s ; le tester sur les 5 à 7 candidats d'un morceau faisait grimper la
    # page à 84–157 s, soit une page cassée. Deux bornes, mesurées sur ce
    # symptôme : on ne départage que les candidats à MOINS DE 10 POINTS
    # d'accords du meilleur, et au plus TROIS. Quand un tab domine nettement
    # sur les accords, la voix n'a rien à arbitrer.
    tetes = [c for c in passent
             if c["accords"] >= passent[0]["accords"] - 0.10][:3]
    if len(tetes) > 1 and voix is not None and Path(str(voix)).exists():
        for c in tetes:
            c["paroles"] = score_des_paroles(voix, c["brut"])
        connus = [c for c in tetes if c.get("paroles") is not None]
        if connus:
            connus.sort(key=lambda c: -c["paroles"])
            for c in connus:
                log.info("tab_du_morceau : %r de %r — accords %.0f %%, "
                         "paroles %.3f", c["tab"].song_name,
                         c["tab"].artist_name, 100 * c["accords"], c["paroles"])
            passent = connus + [c for c in passent if c not in connus]

    best = passent[0]
    log.info("tab_du_morceau : retenu %r de %r — accords %.0f %% "
             "(transposition %+d)%s", best["tab"].song_name,
             best["tab"].artist_name, 100 * best["accords"], best["decalage"],
             ("" if best.get("paroles") is None
              else ", paroles %.3f sur %d candidat(s)"
              % (best["paroles"], len(passent))))
    if cache:
        import dataclasses
        _garder_choix(titre, artiste, {
            "url": best["tab"].tab_url,
            "tab": dataclasses.asdict(best["tab"]),
            "accords": best["accords"], "decalage": best["decalage"],
            "paroles": best.get("paroles")})
    return best
