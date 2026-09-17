"""Re-dériver les accords empilés depuis UN découpage.

Louis, 2026-08-17, en deux temps.

    « la vérité terrain doit toujours être le chart brut, les sections sont des
      modalités d'affichage par dessus »
    « mais lorsqu'on renomme les sections, on retourne sur le brut et donc pas
      les accords renommés, qui eux sont la CONSÉQUENCE des sections »

LES TROIS COUCHES, dans cet ordre et jamais dans l'autre :

  1. **le chart brut** — `prompter.chords`, le décodage à plat capturé dans
     `pipeline.prompter_chords` AVANT que les sections et le repli n'existent.
     C'est la vérité terrain. Elle ne dépend d'aucune hypothèse de structure.
  2. **les sections** — une hypothèse : « ces passages sont le même passage ».
  3. **les accords empilés** — la CONSÉQUENCE de 2 : on empile les occurrences
     d'une même lettre, on re-décode le gabarit, on réécrit les mesures qui
     contribuent (`folding.fold_letter_groups`). Mesuré sur 20 charts, ça
     change **42 accords sur 797, soit 5,3 %** — ce n'est donc pas un
     affichage, et on ne fait pas semblant du contraire.

CE QUE ÇA IMPOSE. Une conséquence ne survit pas à la disparition de sa cause.
Quand Louis retrace ses sections, les accords empilés du chart d'avant ont été
dérivés d'un AUTRE découpage : les garder, c'est faire dire à sa nouvelle
structure ce qu'a dit l'ancienne. Les jeter, c'est lui rendre un chart moins
bon que celui qu'il avait. La seule réponse juste est de **repartir du brut et
de refaire l'empilement avec SES sections** — c'est ce que fait ce module.

CE QUE ÇA NE RÉSOUT PAS. Le repli garde le droit de grouper des occurrences de
longueurs différentes (108 passages sur 552 du corpus, 19,6 %) ; c'est la règle
« under-fold, never over-fold » encore en dette, voir docs/known_issues.md
2026-08-17. Ici, ça n'abîme plus les accords — on part du brut, qui les a tous —
mais la carte des sections reste optimiste.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def _sections_min(secs: list[dict], n_bars: int) -> list[dict]:
    """[{label, barRanges:[[b0,b1]]}] — une entrée PAR OCCURRENCE.

    `fold_letter_groups` lit `s["barRanges"][0]` et regroupe par `label` :
    c'est lui qui rassemble les occurrences d'une lettre, pas l'appelant.
    """
    out = []
    for s in secs:
        try:
            b0 = max(0, int(s["mesure_debut"]) - 1)
            b1 = min(n_bars - 1, int(s["mesure_fin"]) - 1)
        except (KeyError, TypeError, ValueError):
            continue
        if b1 < b0:
            continue
        out.append({"label": str(s.get("label") or "?"),
                    "barRanges": [[b0, b1]]})
    return out


def refold(chart: dict, secs: list[dict], audio_dir) -> tuple[list, dict]:
    """(bars, rapport) — les mesures du morceau, empilées selon `secs`.

    `bars` part TOUJOURS du brut (`soudure.accords_par_mesure`) puis reçoit
    l'empilement en place. Le rapport dit ce qui s'est passé, y compris quand
    rien ne s'est passé : `{"ok": False, "raison": …}` — jamais un repli muet,
    sinon on croit avoir empilé alors qu'on rend le brut.

    Même loi que la pipeline, la seule qui existe (`folding.fold_letter_groups`,
    refactor décision 4) : réécrire un chart depuis l'outil et le ré-analyser
    ne peuvent plus donner deux résultats différents, il n'y a plus de
    variable d'environnement à faire diverger.
    """
    from harmonia.soudure import accords_par_mesure
    bars = accords_par_mesure(chart)
    grid = chart.get("barGrid") or []
    n_bars = chart.get("nBars") or (len(grid) - 1)
    bpb = int(chart.get("bpb") or 4)
    sections = _sections_min(secs, n_bars)
    if not sections or len(grid) < 2:
        return bars, {"ok": False, "raison": "pas de sections exploitables"}

    nom = Path(chart.get("audio_url") or "").name
    if not nom:
        return bars, {"ok": False, "raison": "chart sans audio"}
    audio = Path(audio_dir) / nom
    from harmonia import cache
    if not cache.exists("musx_probs", audio):
        # Le brut est rendu tel quel, et on le DIT : c'est la différence entre
        # « on n'a pas pu empiler » et « il n'y avait rien à empiler ».
        # `cache.exists` (pas `musx_cache_path(...).exists()`, sprint 15) :
        # ce dernier ne teste que la clé NEUVE (`<stem>__<taille>`), et tous
        # les caches musx de la bibliothèque vivent encore sous la clé
        # historique dans `data/cache/` (non renommée, sprint 21) — le
        # mauvais test faisait chuter ce chemin en silence sur 46/46 morceaux
        # (rapport d'or : refold "réussissait" en apparence, en fait dégradé).
        return bars, {"ok": False,
                      "raison": "pas de cache musx pour ce morceau"}

    try:
        from harmonia import musx as _musx
        from harmonia.folding import fold_letter_groups
        from harmonia.nnls_features import extract_bothchroma
        probs = _musx.frame_posteriors(audio)          # cache
        arr, times = extract_bothchroma(audio)         # cache
        rapport = fold_letter_groups(
            sections, bars, grid, probs, bpb, arr=arr, times=times)
    except Exception as exc:                    # jamais un chart cassé pour ça
        logger.exception("refold: empilement impossible")
        return bars, {"ok": False, "raison": f"{type(exc).__name__}: {exc}"}

    # LES SUGGESTIONS SE RECALCULENT ICI, comme dans la pipeline (Louis,
    # 2026-09-17 : « est ce que […] la suggestion des top 5 accords dans le
    # compass est bien implémentée ? j'ai l'impression que ca a été retiré ? »).
    #
    # `accords_par_mesure` recolle la métadonnée de l'accord ÉCRIT qui occupe
    # le même temps — et après un repli, la plupart des mesures n'en trouvent
    # aucun : sur This Love, 12 accords sur 39 gardaient leurs candidats, les
    # 27 autres ouvraient l'éditeur sur rien. Recoller vaut mieux que perdre,
    # mais recalculer vaut mieux que recoller : les postérieures musx et la
    # chroma sont DÉJÀ chargées trois lignes plus haut (et en cache), donc
    # c'est une lecture en mémoire, pas une extraction.
    from harmonia.span_rescore import bass_suggestions, musx_suggestions
    plat = [c for bar in bars for c in bar]
    musx_suggestions(probs, plat)
    bass_suggestions(arr, times, plat)

    # le compte de répétitions se recalcule sur les accords empilés, comme
    # dans la pipeline — sinon « joué 7 fois » parle des accords d'avant.
    from collections import Counter
    fam: Counter = Counter()
    for bar in bars:
        for c in bar:
            if not c.get("nc") and not c.get("carry"):
                fam[(c["root"], (c.get("q") or "")[:1])] += 1
    for bar in bars:
        for c in bar:
            c["n"] = 0 if c.get("nc") else fam[(c["root"], (c.get("q") or "")[:1])]

    change = sorted({b for r in rapport.values()
                     for b in (r.get("changed") or [])})
    lettres = {L: r.get("period") for L, r in rapport.items()}
    return bars, {"ok": True, "mesures_reecrites": change,
                  "n_reecrites": len(change), "lettres": lettres,
                  "rapport": rapport}
