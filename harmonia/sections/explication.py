"""« Pourquoi ? » — le calcul de « Valider les sections », étape par étape.

Louis, 2026-10-11 : « sur l'outil d'annotation de sections, je veux une
interface complète qui explique comment les autres sections sont détectées,
je veux comprendre tous les mécanismes derrière chaque étape ». Il a précisé :
calculé EN DIRECT sur le morceau et les briques qu'il a tracées (jamais un
exemple figé), DANS l'app, et le calcul montré TEL QU'IL EST, défauts compris
— c'est un outil de débogage, pas une vitrine.

`expliquer` rejoue LE MÊME chemin que la route `sections_inferer`
(`occurrences_de_tous` → `remplir_par_algo` → `reponse_inferer`), avec le
paramètre `trace` de chaque fonction pour récupérer les intermédiaires
qu'elles ne rendent pas. Puis il fait À BLANC ce que le client et
`/api/soudure/valider` font ensuite pour écrire le chart (`combler_trous`,
`scinder_multiples`, `refold`, `sections_et_fusions`) — sans rien écrire.
`tests/test_explication_sections.py` vérifie, sur tous les morceaux annotés à
la main, que les sections finales sont exactement celles de Valider.

CE QUE ÇA NE FAIT PAS. `combler_trous` est une copie Python de
`comblerTrous` (sections_editor.js) : le client est en JavaScript, il n'y a
pas de code commun à appeler — une divergence future entre les deux ne
serait pas détectée par le test. Les raisons données pour chaque pic écarté
par les suggestions pâles (`find_repeats`) sont reconstituées à partir de sa
courbe et de ses règles (`similarity._peaks`), pas tracées de l'intérieur.
"""
from __future__ import annotations

import copy
import logging
import re
import time
from pathlib import Path

import numpy as np

logger = logging.getLogger("harmonia.sections.explication")

_NOMS = ["C", "D♭", "D", "E♭", "E", "F", "G♭", "G", "A♭", "A", "B♭", "B"]


def _nom_accord(c: dict) -> str:
    if c.get("nc"):
        return "N.C."
    q = str(c.get("q") or "").replace("h7", "ø7").replace("h", "ø").replace("^", "Δ")
    s = _NOMS[int(c.get("root", 0)) % 12] + q
    b = c.get("bass")
    if b is not None and int(b) >= 0 and int(b) % 12 != int(c.get("root", 0)) % 12:
        s += "/" + _NOMS[int(b) % 12]
    return s


def noms_par_mesure(bars: list) -> list[str]:
    """« F-7 B♭7 » — les accords de chaque mesure, sans répéter un accord tenu."""
    out = []
    for bar in bars:
        noms: list[str] = []
        for c in bar:
            n = _nom_accord(c)
            if not noms or noms[-1] != n:
                noms.append(n)
        out.append(" ".join(noms) if noms else "·")
    return out


def reponse_inferer(occ, blocs, queues_proposees, perdus, diagnostics) -> dict:
    """Le corps JSON de `/api/sections/inferer`, à partir de ce que rendent
    `occurrences_de_tous` et `remplir_par_algo`. UNE seule fonction pour la
    route et pour l'explication : sinon l'écran « Pourquoi ? » pourrait
    expliquer autre chose que ce que Valider écrit."""
    tracees = {(o["m0"], o["m1"]) for o in occ if o["trace"]}
    # Un candidat EN SUSPENS tranché par la cohérence du morceau (règle 5b,
    # 2026-09-30) sort « ressemble » : une proposition, en pointillés, que
    # l'écran ne renvoie PAS comme contrainte — elle est donc retranchée à
    # chaque « Valider », avec les briques que Louis a ajoutées entre-temps.
    devinees = {(o["m0"], o["m1"]) for o in occ if o.get("suspens")}
    # `rot` par plage, pour que le rendu puisse dire « A, monté d'un ton »
    rotations = {(o["m0"], o["m1"]): o.get("rot", 0) for o in occ}
    out = []
    for b in blocs:
        if b["source"] == "brique":
            source = ("humain" if (b["m0"], b["m1"]) in tracees
                      else "ressemble" if (b["m0"], b["m1"]) in devinees
                      else "propage")
        else:
            source = "algo"
        out.append({"label": b["label"],
                    "mesure_debut": b["m0"] + 1, "mesure_fin": b["m1"] + 1,
                    "source": source, "reste": False, "type": "",
                    "rot": rotations.get((b["m0"], b["m1"]), 0)})
    return {"sections": out,
            "queues_proposees": [
                {"label": q["label"], "mesure_debut": q["m0"] + 1,
                 "mesure_fin": q["m1"] + 1, "raison": q["raison"]}
                for q in queues_proposees],
            "ecartes": [{"label": t.get("label"), "raison": t["raison"]}
                        for t in perdus],
            "abstentions": [d for d in diagnostics
                            if d.get("raison") == "sans_contraste"]}


def combler_trous(sections: list[dict], n_bars: int) -> list[dict]:
    """Copie Python de `comblerTrous` (harmonia/static/screens/
    sections_editor.js) — ce que le client envoie à `/api/soudure/valider`.
    Les mesures que personne ne réclame : « intro » en tête (sauf si la
    section suivante est déjà une intro), « queue » ailleurs, suffixées
    quand un groupe (label, longueur) est déjà pris."""
    n = max(0, int(n_bars or 0))
    vrais = []
    for x in sections:
        if not x or not x.get("label") or x.get("label") == "?":
            continue
        d = max(1, int(x.get("mesure_debut") or 0))
        f = min(n, int(x.get("mesure_fin") or 0)) if n else int(x.get("mesure_fin") or 0)
        if f >= d:
            vrais.append({"label": str(x["label"]), "mesure_debut": d, "mesure_fin": f})
    vrais.sort(key=lambda x: x["mesure_debut"])
    if not n:
        return vrais
    pris = {f"{x['label']}|{x['mesure_fin'] - x['mesure_debut'] + 1}" for x in vrais}

    def nom(d, f, suivant=None):
        base = "intro" if (d == 1 and suivant != "intro") else "queue"
        lg, lab, k = f - d + 1, None, 1
        lab = base
        while f"{lab}|{lg}" in pris:
            k += 1
            lab = f"{base} {k}"
        pris.add(f"{lab}|{lg}")
        return lab
    out, cur = [], 1
    for s in vrais:
        if s["mesure_debut"] > cur:
            out.append({"label": nom(cur, s["mesure_debut"] - 1, s["label"]),
                        "mesure_debut": cur, "mesure_fin": s["mesure_debut"] - 1})
        out.append(s)
        cur = max(cur, s["mesure_fin"] + 1)
    if cur <= n:
        out.append({"label": nom(cur, n), "mesure_debut": cur, "mesure_fin": n})
    return out


def _j(x, nd: int = 4):
    """Rendre sérialisable (numpy, tuples, ensembles) et arrondir."""
    if isinstance(x, dict):
        return {str(k): _j(v, nd) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_j(v, nd) for v in x]
    if isinstance(x, (set, frozenset)):
        return [_j(v, nd) for v in sorted(x)]
    if isinstance(x, np.ndarray):
        return _j(x.tolist(), nd)
    if isinstance(x, (bool, np.bool_)):
        return bool(x)
    if isinstance(x, (int, np.integer)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return round(float(x), nd)
    return x


def _pales(chart, audio, gardes, claimed) -> list[dict]:
    """Ce que les suggestions pâles (`/api/section-repeats` → `find_repeats`)
    proposent pour chaque brique, avec ce que le client en garde
    (`proposerReprises` : même tonalité seulement, cellules recollées à la
    longueur tracée, rien sur une mesure déjà à lui) et pourquoi chaque pic de
    sa courbe a été écarté."""
    from scipy.signal import find_peaks

    from harmonia import musx as _musx
    from harmonia import section_tool as st
    from harmonia.sections import similarity as VS
    grid = chart.get("barGrid") or []
    n = len(grid) - 1
    triad = _musx.frame_posteriors(audio)[0]
    out = []
    for b0, b1, label in gardes:
        try:
            r = st.find_repeats(grid, triad, b0, b1, claimed_bars=sorted(claimed))
        except Exception as exc:                             # noqa: BLE001
            out.append({"label": label, "m0": b0, "m1": b1, "erreur": str(exc)})
            continue
        L, cell = b1 - b0 + 1, max(1, int(r.get("cell_bars") or (b1 - b0 + 1)))
        trouvees = sorted((o for o in r["occurrences"]
                           if str(o["source"]).startswith("found") and not o["rot"]),
                          key=lambda o: o["b0"])
        items = []
        if cell < L:
            need = round(L / cell)
            i = 0
            while i < len(trouvees):
                k = 1
                while (k < need and i + k < len(trouvees)
                       and trouvees[i + k]["b0"] == trouvees[i]["b0"] + k * cell):
                    k += 1
                items.append({"b0": trouvees[i]["b0"], "b1": trouvees[i + k - 1]["b1"]})
                i += k
        else:
            items = [{"b0": o["b0"], "b1": o["b1"]} for o in trouvees]
        items = [it for it in items
                 if not any(b in claimed for b in range(it["b0"], it["b1"] + 1))]
        # pourquoi chaque pic de la courbe est (ou n'est pas) gardé
        sc = np.asarray(r["curve"], float)
        par = 0 if r.get("parity") == "paire" else 1
        thr = float(r["thr"]) + (VS.ODD_BONUS if par else 0.0)
        idx, _ = find_peaks(sc, height=thr, distance=VS.UNIT)
        gardes_starts = {o["b0"] for o in r["occurrences"]
                         if str(o["source"]).startswith("found") and not o["rot"]}
        pics = []
        for p in (int(x) for x in idx):
            if (p - b0) % VS.UNIT != par:
                raison = "parite"
            elif abs(p - b0) < cell:
                raison = "chevauche_la_brique"
            elif p + cell > n:
                raison = "deborde_la_fin"
            elif any(b in claimed for b in range(p, min(n, p + cell))) or b0 <= p <= b1:
                raison = "deja_pris"
            elif p in gardes_starts:
                raison = None
            else:
                raison = "voisin_plus_fort"
            pics.append({"p": p, "score": round(float(sc[p]), 3), "raison": raison})
        out.append({"label": label, "m0": b0, "m1": b1, "cell_bars": cell,
                    "seuil": thr, "parite": r.get("parity"),
                    "inondation": r.get("flood_fallback"),
                    "courbe": r["curve"], "pics": pics,
                    "occurrences": r["occurrences"], "proposees": items})
    return out


def expliquer(chart: dict, humain: list[dict], audio_dir) -> dict:
    """Tout ce que l'écran « Pourquoi ? » dessine, pour CE morceau et CES
    traits — voir la docstring du module. Ne touche à aucun fichier."""
    from harmonia import musx as _musx
    from harmonia.refold import refold
    from harmonia.sections import similarity as VS
    from harmonia.sections.simulation import (PLANCHER_REPRISE, PROEMINENCE_MIN,
                                              occurrences_de_tous,
                                              remplir_par_algo)
    from harmonia.folding import STACK_COHERENCE
    from harmonia.soudure import (accords_par_mesure, scinder_multiples,
                                  sections_et_fusions)
    t_debut = time.perf_counter()
    grid = chart.get("barGrid") or []
    n = len(grid) - 1
    stem = Path(chart.get("audio_url") or "").stem
    audio = Path(audio_dir) / f"{stem}.m4a"

    # ── le calcul de Valider, avec ses traces ─────────────────────────────
    tr_occ: dict = {}
    occ, perdus, diagnostics = occurrences_de_tous(chart, humain, audio_dir,
                                                   trace=tr_occ)
    queues_proposees: list = []
    tr_algo: dict = {}
    blocs = remplir_par_algo(chart, occ, audio_dir, propositions=queues_proposees,
                             trace=tr_algo)
    reponse = reponse_inferer(occ, blocs, queues_proposees, perdus, diagnostics)
    t_valider = time.perf_counter()

    # ── l'écriture du chart, À BLANC ──────────────────────────────────────
    pour = combler_trous(reponse["sections"], n)
    scindees = scinder_multiples(copy.deepcopy(pour))
    bars, rap = refold(copy.deepcopy(chart), scindees, audio_dir)
    neuves, _fusions = sections_et_fusions(copy.deepcopy(chart), scindees, bars=bars,
                                           fold_report=rap.get("rapport"))
    t_ecrire = time.perf_counter()

    # ── ce que la machine entend ──────────────────────────────────────────
    triad = _musx.frame_posteriors(audio)[0]
    V = VS.harmonic_vectors(triad, grid)
    S = V @ V.T
    accords = noms_par_mesure(accords_par_mesure(chart))

    briques = []
    for tb, (b0, b1, label), info in zip(tr_occ.get("briques", []),
                                          tr_occ.get("gardes", []), diagnostics):
        d = tb.get("detail") or {}
        briques.append({
            "label": label, "m0": b0, "m1": b1, "info": info,
            "boucle": {k: d.get(k) for k in ("boucle", "suite", "sous_periode", "tau")},
            "periode": (tb.get("boucle") or {}).get("periode", []),
            "sous_periode": (tb.get("boucle") or {}).get("sous_periode", []),
            "mode": d.get("mode"), "courbe": d.get("courbe", []),
            "pics": d.get("pics", []), "reference": d.get("reference"),
            "egalite": d.get("egalite"), "suspens": d.get("suspens", []),
            "seuil": d.get("seuil"), "mediane": d.get("mediane"), "sigma": d.get("sigma"),
            "comps": tb.get("comps", []), "fins": tb.get("fins", []),
            "transposees": tb.get("transposees", []),
            "reprises": tb.get("reprises", [])})

    mot = tr_algo.get("mot") or {}
    B = mot.get("B")
    hist = None
    if B is not None:
        J = B.shape[0]
        off = B[~np.eye(J, dtype=bool)]
        h, edges = np.histogram(off, bins=40, range=(min(0.0, float(off.min())), 1.0))
        hist = {"comptes": h.tolist(), "bords": edges.tolist()}

    # ── les cinq décisions, en chiffres ───────────────────────────────────
    pales = []
    try:
        claimed = set()
        for b0, b1, _l in tr_occ.get("gardes", []):
            claimed.update(range(b0, b1 + 1))
        pales = _pales(chart, audio, tr_occ.get("gardes", []), claimed)
    except Exception:                                        # noqa: BLE001
        logger.exception("explication : suggestions pâles indisponibles")
    t_fin = time.perf_counter()

    out = {
        "n": n, "titre": chart.get("title") or stem, "barGrid": grid,
        "accords": accords, "humain": humain,
        "constantes": {"plancher": PLANCHER_REPRISE, "proeminence": PROEMINENCE_MIN,
                       "coherence_pile": STACK_COHERENCE},
        "nettoyage": {"gardes": tr_occ.get("gardes", []), "ecartes": perdus},
        "V": np.round(V, 2), "S": np.round(S, 3),
        "briques": briques,
        "arbitrage": {"verdicts": tr_occ.get("verdicts", []),
                      "tenues": tr_occ.get("tenues", []),
                      "retenues": tr_occ.get("retenues", []),
                      "retours": tr_occ.get("retours", []),
                      "en_suspens": tr_occ.get("en_suspens", []),
                      "occurrences": occ},
        "mot": {"bornes": tr_algo.get("bornes"), "lettres": tr_algo.get("mot_lettres"),
                "source": tr_algo.get("source"), "B": None if B is None else np.round(B, 3),
                "seuil": mot.get("seuil"), "germes": mot.get("germes", []),
                "histogramme": hist},
        "soudures": {"depart": tr_algo.get("depart"), "geles": tr_algo.get("geles"),
                     "par_jeton": tr_algo.get("par_jeton"),
                     "cible": tr_algo.get("cible"), "couts": tr_algo.get("couts"),
                     "essais": [{k: e[k] for k in ("cible", "cout", "steps", "nom")}
                                for e in tr_algo.get("essais", [])],
                     "nommes": tr_algo.get("nommes"), "restes": tr_algo.get("restes")},
        "nommage": {"prefixe": tr_algo.get("prefixe"),
                    "intro_avale": tr_algo.get("intro_avale", []),
                    "tete": tr_algo.get("tete"),
                    "a_deja_une_intro": tr_algo.get("a_deja_une_intro"),
                    "avant_queue": tr_algo.get("avant_queue"),
                    "queue_opt": tr_algo.get("queue_opt", []),
                    "apres_queue": tr_algo.get("apres_queue"),
                    "dernier_tour": tr_algo.get("dernier_tour"),
                    "final": tr_algo.get("final")},
        "valider": reponse,
        "ecriture": {"combler": pour, "scindees": scindees,
                     "empile": {k: v for k, v in rap.items() if k != "rapport"},
                     "rapport": rap.get("rapport") or {},
                     # la cohérence de pile d'une lettre REFUSÉE n'est dite
                     # que dans sa raison (« … pairwise 0.83 ») : on la sort
                     "coh_min": {L: float(m.group(1)) for L, r in
                                 (rap.get("rapport") or {}).items()
                                 for m in [re.search(r"pairwise ([0-9.]+)",
                                                     str(r.get("reason") or ""))]
                                 if m},
                     "sections_chart": [{"label": s.get("label"),
                                         "barRanges": s.get("barRanges")}
                                        for s in (neuves or [])]},
        "pales": pales,
        "durees_ms": {"valider": round(1000 * (t_valider - t_debut)),
                      "ecriture": round(1000 * (t_ecrire - t_valider)),
                      "total": round(1000 * (t_fin - t_debut))},
    }
    return _j(out)
