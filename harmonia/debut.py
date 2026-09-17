"""Où commence vraiment un morceau — la première note de basse, calée sur la grille.

Le problème, posé par Louis le 2026-09-16 sur Sam Smith : le fichier audio
porte 40 secondes d'intro parlée du clip, Beat This! pose donc sa première
battue à 37 s, en plein fondu, et la mesure 1 du chart tombe une mesure avant
que le groupe n'entre. Tout est décalé derrière — en particulier les sections
qu'il trace au doigt, qui se calent sur la mesure 1.

Le traqueur de battues n'a aucun moyen de savoir qu'un fondu n'est pas un
début : il entend un pouls, il le suit. Il faut donc un autre indice.

LA RÈGLE, ET COMMENT ELLE A ÉTÉ CHOISIE
---------------------------------------
Louis a tranché à l'oreille le vrai début de 42 morceaux de sa bibliothèque
(`state/human/debuts.json`), et a proposé trois pistes. Mesurées contre ses
42 réponses, en comptant juste quand on désigne LA MÊME ligne de mesure que
lui :

    le traqueur seul (mesure 1 = 1re ligne)        28/42
    la marche d'énergie à la ligne de mesure       27/42
    le 1er accord non-N.C. de musx                 32/42
    le 1er son audible du fichier                  31/42
    **la 1re note de basse**                       **35/42**

Et douze combinaisons (consensus, médiane, garde-fous mutuels) : aucune ne
dépasse la basse seule. Les meilleures l'égalent — « la plus précoce de basse
et accord », « médiane de son, basse et accord » — sans rien apporter. On garde
donc la plus simple.

CE QUE ÇA NE RÉSOUT PAS
-----------------------
Sept morceaux sur 42 restent faux, et leur mécanisme est musical, pas
technique. Louis l'avait annoncé : « souvent la première note de basse, mais
pas toujours ».

* **Le morceau ouvre sur la batterie.** Billie Jean (quatre mesures de caisse
  claire) et Be My Baby (le break de batterie le plus connu du monde) ont leur
  basse en retard d'une à deux mesures. C'est la piste que Louis a nommée en
  premier et qui n'est PAS implémentée ici : un détecteur de transitoires
  aigus, indépendant de l'harmonie.
* **La basse joue déjà pendant l'intro.** Urdlvw0SSEc (9 mesures d'avance) et
  fd02pGJx0s0 : la basse entre avant le début que Louis entend.
* **La grille elle-même est fausse.** h_D3VFfhvs4 a des trous de 12 et 42 s
  dans son `barGrid` ; aucune ligne n'y est bonne, et caler sur la plus proche
  ne peut pas le rattraper.

Et cinq des 42 réponses de Louis ne tombent sur AUCUNE ligne de la grille
(Stand By Me à deux tiers de mesure de la plus proche) : là, ce n'est pas le
choix de la ligne qui est en cause mais la phase du traqueur, un autre
problème.

Ce module ne décide rien tout seul : une marque posée à la main
(`state/human/marks/`) reste souveraine — c'est Louis qui définit où commence
la chanson.
"""
from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import numpy as np

log = logging.getLogger("harmonia.debut")

#: résolution de l'enveloppe d'énergie, en secondes
PAS_ENERGIE = 0.25
#: part de l'énergie médiane au-dessus de laquelle le fichier « fait un son »
PART_SON = 0.10
#: durée pendant laquelle ce son doit tenir, en pas
TENUE_SON = 5
#: masse minimale de la tête de basse de musx pour dire « il y a une basse »
SEUIL_BASSE = 0.25
#: nombre de trames musx consécutives à ce niveau (6 × 23,22 ms ≈ 0,14 s)
TENUE_BASSE = 6
#: en deçà de cette fraction de mesure, c'est la ligne de la grille qui gagne
PART_CALAGE = 0.25


def enveloppe(audio: Path, fenetre: float = 90.0) -> list[float] | None:
    """L'énergie du début du fichier, par tranches de `PAS_ENERGIE`.

    Mono 8 kHz : on ne cherche pas une hauteur, seulement s'il y a du son.
    """
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(audio), "-t", str(fenetre),
         "-ac", "1", "-ar", "8000", "-f", "f32le", "-"],
        capture_output=True).stdout
    if not raw:
        return None
    x = np.frombuffer(raw, dtype=np.float32)
    n = int(PAS_ENERGIE * 8000)
    if len(x) < n * 4:
        return None
    return [float(np.sqrt(np.mean(x[i:i + n] ** 2)))
            for i in range(0, len(x) - n, n)]


def premier_son(env: list[float]) -> float:
    """Le premier instant où le fichier fait un son, et s'y tient.

    Sert de plancher à la recherche de basse : rien de musical ne commence
    dans le silence, et un modèle qui l'affirme se trompe. Sans ce plancher,
    la tête de basse de musx annonçait une basse à 0,00 s sur Hot N Cold, dont
    les quatre premières secondes sont muettes.
    """
    e = np.asarray(env, dtype=float)
    if not len(e):
        return 0.0
    seuil = PART_SON * float(np.median(e))
    for i in range(len(e) - TENUE_SON):
        if bool(np.all(e[i:i + TENUE_SON] > seuil)):
            return float(i * PAS_ENERGIE)
    return 0.0


def premiere_basse(probs_basse, apres: float = 0.0) -> float | None:
    """Le premier instant où la tête BASSE de musx sort de « rien » et y reste.

    `probs_basse` est `musx.frame_posteriors(audio)[1]` : (trames, 13), dont la
    colonne 0 est « pas de basse ». On cherche où son complément passe
    `SEUIL_BASSE` et s'y tient `TENUE_BASSE` trames, en partant de `apres`.

    Les deux réglages sont mesurés, pas choisis. Contre huit débuts connus,
    l'erreur médiane vaut 0,31 s à (0,25 ; 6 trames) contre 0,95 s à
    (0,50 ; 6) — l'ancien réglage arrivait 1,6 s trop tard sur Sam Smith et
    8,8 s trop tard sur Stand By Me. Un seuil de 0,35 fait mieux sur la
    médiane mais rate l'entrée de Stand By Me de 6,5 s : on préfère le réglage
    qui ne casse aucun cas franc.
    """
    from harmonia.musx import FRAME_DT
    son = 1.0 - np.asarray(probs_basse)[:, 0]
    debut = max(0, int(apres / FRAME_DT))
    for i in range(debut, len(son) - TENUE_BASSE):
        if bool(np.all(son[i:i + TENUE_BASSE] > SEUIL_BASSE)):
            return float(i * FRAME_DT)
    return None


def cale_sur_grille(t: float | None, grille) -> tuple[float | None, bool]:
    """La détection, ramenée sur la ligne de mesure quand elle en est proche.

    Louis, 2026-09-17 : « il faut vraiment que tu utilises le grid de la
    grille pour t'aiguiller aussi, en cas de doute c'est lui qui tranche ». Un
    seuil franchi sur une postérieure a la précision d'une trame ; une ligne de
    mesure vient du traqueur de battues, qui est meilleur que ça. Quand les
    deux se disputent à moins de `PART_CALAGE` mesure, la ligne gagne.

    Rend `(temps, posé_sur_une_ligne)`. Au-delà on NE cale PAS et on le dit :
    écraser l'un par l'autre effacerait le désaccord, qui est précisément le
    signal qu'un des deux se trompe.
    """
    if t is None:
        return None, False
    g = np.asarray(grille, dtype=float)
    if len(g) < 2:
        return t, False
    i = int(np.abs(g - t).argmin())
    mesure = float(np.median(np.diff(g))) if len(g) > 2 else float(g[1] - g[0])
    if mesure > 0 and abs(float(g[i]) - t) <= PART_CALAGE * mesure:
        return float(g[i]), True
    return t, False


def debut_du_morceau(audio: Path, grille, probs_basse=None) -> dict:
    """Où commence le morceau : `{t, mesure, sur_une_ligne, son}`.

    `t` est en secondes, `mesure` l'index de la ligne de `grille` retenue.
    `sur_une_ligne` dit si la détection est tombée assez près d'une ligne pour
    que la grille tranche ; quand c'est faux, `t` est la détection brute et la
    grille et le détecteur ne sont pas d'accord — un cas à montrer, pas à
    arrondir.

    Rend `{"t": None}` si la basse ne se déclare jamais : un silence n'a pas
    de début, et inventer la ligne 0 serait un repli muet.
    """
    env = enveloppe(Path(audio))
    son = premier_son(env) if env else 0.0
    if probs_basse is None:
        from harmonia import musx as _musx
        probs_basse = _musx.frame_posteriors(Path(audio))[1]
    brut = premiere_basse(probs_basse, apres=son)
    if brut is None:
        log.info("debut: aucune basse déclarée dans %s", Path(audio).name)
        return {"t": None, "mesure": None, "sur_une_ligne": False, "son": son}
    t, sur = cale_sur_grille(brut, grille)
    g = np.asarray(grille, dtype=float)
    return {"t": t, "son": son, "sur_une_ligne": sur,
            "mesure": (int(np.abs(g - t).argmin()) if len(g) else None)}
