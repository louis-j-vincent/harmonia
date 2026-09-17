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
    la 1re note de basse                           35/42
    **la 1re note de basse APRÈS le dernier trou**  **37/40**

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
#: sous cette masse de basse, musx n'entend aucune harmonie — c'est un « trou »
SEUIL_TROU = 0.15
#: durée minimale d'un trou pour qu'il sépare deux musiques, en secondes
DUREE_TROU = 3.0
#: sous cette part de la mélodie médiane du morceau, il n'y a pas de mélodie
PART_MELODIE = 0.60
#: au-dessus de cette cosinus entre l'avant et l'après, c'est la MÊME matière
#: — donc une simple respiration, pas une frontière entre deux musiques
SEUIL_COHERENCE = 0.85
#: de combien de secondes de part et d'autre on compare les deux matières
MARGE_COHERENCE = 10.0


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


def melodie(chroma, times, liss: float = 1.0) -> np.ndarray:
    """La présence d'une VRAIE mélodie, image par image.

    Louis, 2026-09-17 : « pour détecter un trou, c'est pas seulement la ligne
    de basse, c'est aussi pas de mélodie ».

    `chroma` est le bothchroma NNLS 24 dimensions (`nnls_features`), dont la
    moitié haute (colonnes 12 à 23) est ce qui sonne au-dessus de la basse. Une
    mélodie y concentre l'énergie sur peu de notes ; la parole, une salle, un
    souffle l'étalent sur les douze. On mesure donc la NETTETÉ — un moins
    l'entropie du profil — pondérée par l'énergie.

    Les deux facteurs sont nécessaires : l'énergie seule ne distingue rien (sur
    Sam Smith elle vaut autant sur les 40 s de dialogue que sur le morceau), et
    la netteté seule explose dans le silence, où le chroma devient dégénéré et
    se concentre par hasard sur une note (Urdlvw0SSEc marque 0,92 en plein
    trou). Leur produit sépare : sur Sam Smith 0,35 à 0,46 fois la médiane
    pendant le dialogue, 1,19 fois dès l'entrée du groupe ; Let It Be, qui n'a
    pas d'intro, ne descend jamais sous 0,91.
    """
    g = np.asarray(chroma, dtype=float)[:, 12:]
    e = g.sum(1)
    s = e.copy()
    s[s <= 1e-9] = 1.0
    p = g / s[:, None]
    nettete = 1.0 - (-(p * np.log(p + 1e-12)).sum(1) / np.log(12))
    ref = float(np.median(e)) or 1.0
    m = nettete * np.minimum(1.0, e / ref)
    pas = float(times[1] - times[0]) if len(times) > 1 else liss
    k = max(1, int(liss / pas))
    return np.convolve(m, np.ones(k) / k, mode="same")


def _profil(chroma, times, t0: float, t1: float):
    """Le profil mélodique moyen de [t0, t1), normalisé — ou None si vide."""
    sel = (np.asarray(times) >= t0) & (np.asarray(times) < t1)
    if not sel.any():
        return None
    v = np.asarray(chroma, dtype=float)[sel, 12:].mean(0)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else None


def _sans_basse(probs_basse, apres: float, fenetre: float) -> list[tuple]:
    """Les (début, fin) des passages où musx n'entend aucune basse assez
    longtemps. Ce sont les CANDIDATS ; deux autres conditions les filtrent."""
    from harmonia.musx import FRAME_DT
    pb = np.asarray(probs_basse)
    n = min(len(pb), int(fenetre / FRAME_DT))
    k = int(DUREE_TROU / FRAME_DT)
    if n <= k:
        return []
    creux = (1.0 - pb[:n, 0]) < SEUIL_TROU
    out, i = [], max(0, int(apres / FRAME_DT))
    while i < n - k:
        if bool(np.all(creux[i:i + k])):
            j = i
            while j < n and creux[j]:
                j += 1
            out.append((i * FRAME_DT, j * FRAME_DT))
            i = j
        else:
            i += 1
    return out


def dernier_trou(probs_basse, chroma=None, times=None, apres: float = 0.0,
                 fenetre: float = 90.0) -> float | None:
    """La fin du DERNIER trou avant `fenetre` — la règle de Louis, à trois
    conditions, TOUTES nécessaires.

    Louis, 2026-09-17 : « pour détecter un trou, c'est pas seulement la ligne
    de basse, c'est aussi pas de mélodie, et EN PLUS c'est une incohérence
    mélodique en pattern entre le début de la chanson et le trou. Il faut
    toutes ces règles-là. »

    1. **Pas de basse** — la tête de basse de musx reste sous `SEUIL_TROU`
       pendant au moins `DUREE_TROU`.
    2. **Pas de mélodie** — la mélodie (voir `melodie`) y reste sous
       `PART_MELODIE` fois la médiane du morceau. Sans cette condition, un
       passage où la basse se tait mais où le chant continue passerait pour
       une frontière de fichier.
    3. **Incohérence de matière** — ce qui joue AVANT le trou ne ressemble pas
       à ce qui joue APRÈS : cosinus des deux profils mélodiques sous
       `SEUIL_COHERENCE`, mesuré sur `MARGE_COHERENCE` secondes de part et
       d'autre. C'est la condition que Louis exige explicitement, « sinon ça
       peut juste être une pause dans la musique ».

    Ce qui joue avant le dernier trou survivant n'est pas le morceau : c'est le
    préambule du clip, qui peut très bien être de la vraie musique — d'où
    l'échec des détecteurs d'intensité, qui l'entendaient à juste titre.

    Sans `chroma`, seule la première condition s'applique et on le DIT dans le
    journal : c'est un repli dégradé, jamais silencieux.

    MESURÉ sur les 40 morceaux à grille exploitable, contre les réponses de
    Louis. La 1re note de basse seule désigne la bonne mesure 35 fois ; après
    le dernier trou, 37 fois, sans jamais casser un cas déjà juste. Les deux
    gagnés sont ceux qu'il avait expliqués : Urdlvw0SSEc (22,77 s) et
    fd02pGJx0s0 (9,98 s). Les trois conditions donnent le MÊME score que la
    seule condition de basse, mais en ne déclarant que 6 trous au lieu de 9 —
    trois faux trous en moins sur des morceaux où ça ne se voyait pas encore.
    Les deux seuils sont au genou de leur courbe : à 0,50 de mélodie on perd un
    gain, à 0,70 de cohérence aussi.

    CE QUE ÇA NE RÉSOUT PAS : Be My Baby ouvre sur un break de batterie sans
    harmonie, donc sans trou à trouver ; Smooth Criminal trouve son trou (fini
    à 66,6 s) mais tombe une mesure trop loin, sur une grille elle-même trouée ;
    Chain of Fools rend une détection qui ne tombe sur aucune ligne de mesure.

    ESSAYÉ ET REJETÉ avant d'en arriver là, chacun mesuré : la récurrence
    harmonique en veto (0 gagné 0 perdu), la récurrence rythmique (+1, −10), et
    le changement de tempo, invisible parce que Beat This! impose un tempo
    unique au fichier entier. Voir `docs/debut_pourquoi_2026-09-17.md`.
    """
    candidats = _sans_basse(probs_basse, apres, fenetre)
    if not candidats:
        return None
    if chroma is None or times is None:
        log.info("debut: pas de chroma, le trou n'est jugé que sur la basse "
                 "(2 des 3 conditions de Louis manquent)")
        return candidats[-1][1]
    mel = melodie(chroma, times)
    med = float(np.median(mel)) or 1.0
    t_arr = np.asarray(times)
    garde = None
    for t0, t1 in candidats:
        dedans = (t_arr >= t0) & (t_arr < t1)
        if not dedans.any():
            continue
        if float(mel[dedans].mean()) >= PART_MELODIE * med:
            continue                                  # 2/ il y a une mélodie
        avant = _profil(chroma, times, max(0.0, t0 - MARGE_COHERENCE), t0)
        apres_ = _profil(chroma, times, t1, t1 + MARGE_COHERENCE)
        if avant is None or apres_ is None:
            continue              # rien avant : le silence de tête, pas un trou
        if float(avant @ apres_) >= SEUIL_COHERENCE:
            continue                       # 3/ même matière : une respiration
        garde = t1
    return garde


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


def arbitrage(stem: str) -> str | None:
    """Ce que Louis a tranché pour ce morceau — « confirmé bon », « déplacé »,
    ou None s'il ne s'est pas prononcé.

    Lu dans `state/human/debuts.json`, la vérité terrain qu'il a produite le
    2026-09-17 sur 42 morceaux. Un « confirmé bon » dit que la mesure 1 du
    traqueur est la bonne : le détecteur doit alors se taire, même s'il pense
    autrement. Sans cette lecture, la règle déplacerait Be My Baby, dont il a
    dit qu'il était juste — un « c'est bon » est une décision de sa part
    exactement comme une marque posée à la main, et la contredire en silence
    serait pire que de ne rien détecter.

    Rend None sur n'importe quelle erreur de lecture : l'absence d'arbitrage
    est le cas normal (5 morceaux de la bibliothèque, et tous les nouveaux).
    """
    from harmonia.settings import SETTINGS
    f = SETTINGS.repo / "state" / "human" / "debuts.json"
    try:
        import json
        return ((json.loads(f.read_text(encoding="utf-8")).get("debuts") or {})
                .get(stem, {}).get("source"))
    except (OSError, ValueError, AttributeError):
        return None


def debut_du_morceau(audio: Path, grille, probs_basse=None,
                     chroma=None, times=None) -> dict:
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
    # Ce qui joue avant le dernier trou d'harmonie n'est pas le morceau : on
    # ne cherche la première basse qu'APRÈS (voir `dernier_trou`).
    if chroma is None:
        try:
            from harmonia.nnls_features import extract_bothchroma
            chroma, times = extract_bothchroma(Path(audio))
        except Exception as exc:                          # noqa: BLE001
            # Jamais muet : sans chroma il ne reste qu'une des trois conditions
            # de Louis, et le trou détecté est moins sûr.
            log.warning("debut: chroma NNLS indisponible pour %s (%s) — le "
                        "trou ne sera jugé que sur la basse", Path(audio).name, exc)
            chroma = times = None
    trou = dernier_trou(probs_basse, chroma, times, apres=son)
    brut = premiere_basse(probs_basse, apres=trou if trou is not None else son)
    if brut is None:
        log.info("debut: aucune basse déclarée dans %s", Path(audio).name)
        return {"t": None, "mesure": None, "sur_une_ligne": False, "son": son}
    t, sur = cale_sur_grille(brut, grille)
    g = np.asarray(grille, dtype=float)
    return {"t": t, "son": son, "trou": trou, "sur_une_ligne": sur,
            "mesure": (int(np.abs(g - t).argmin()) if len(g) else None)}


# ── les quatre pistes, pour regarder POURQUOI on se trompe ───────────────────
# Louis, 2026-09-17 : « montre-moi ceux où on se trompe [...] et mets sur
# chacun en dessous les métriques dont tu te sers (intensité de la basse, de
# la batterie, des accords, + une autre métrique de bruit ambiant parce que
# des fois dans l'intro il y a ça) ».
#
# Une seule de ces quatre courbes sert la règle d'aujourd'hui (la basse). Les
# trois autres sont là pour qu'il puisse dire ce qui manque — en particulier
# la BATTERIE, qu'il a nommée en premier et qui est le seul indice ne
# partageant rien avec musx, donc le seul capable de rattraper un morceau qui
# ouvre sur un break de caisse claire (Billie Jean, Be My Baby).

#: résolution des courbes affichées, en secondes
PAS_PISTE = 0.1
#: au-dessus de cette fréquence, on regarde les transitoires de la batterie
HZ_BATTERIE = 4000.0


def _spectre(audio: Path, fenetre: float, sr: int = 22050,
             n: int = 1024, saut: int = 512):
    """(|X|, fréquences, pas) — un spectrogramme d'amplitude, sans dépendance.

    Lu par ffmpeg en mono, fenêtré Hann. Rien ici ne mérite une bibliothèque
    de plus : deux transformées et une moyenne.
    """
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(audio), "-t", str(fenetre),
         "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"],
        capture_output=True).stdout
    x = np.frombuffer(raw, dtype=np.float32)
    if len(x) < n * 2:
        return None, None, None
    fen = np.hanning(n).astype(np.float32)
    trames = [np.abs(np.fft.rfft(x[i:i + n] * fen))
              for i in range(0, len(x) - n, saut)]
    return np.asarray(trames), np.fft.rfftfreq(n, 1.0 / sr), saut / sr


def _vers_pas(v, pas_source: float, pas_cible: float = PAS_PISTE) -> list[float]:
    """Ramène une courbe à `pas_cible`, en prenant le MAXIMUM de chaque tranche.

    Le maximum et pas la moyenne : une attaque de batterie dure moins qu'un
    dixième de seconde, une moyenne l'effacerait.

    Le découpage se fait sur le TEMPS, pas sur un nombre entier de trames.
    Grouper 4 trames musx (4 × 23,22 ms = 92,9 ms) en les appelant « 100 ms »
    décalait la courbe de 7 % — trois secondes à 42 s, de quoi montrer une
    entrée de basse une mesure à côté. C'est exactement le genre d'erreur
    d'unité qui produit des chiffres plausibles et faux (règle 1 du CLAUDE.md).
    """
    v = np.asarray(v, dtype=float)
    if not len(v):
        return []
    duree = len(v) * pas_source
    n = max(1, int(duree / pas_cible))
    bords = (np.arange(n + 1) * pas_cible / pas_source).astype(int)
    bords = np.clip(bords, 0, len(v))
    return [float(v[a:b].max()) if b > a else float(v[min(a, len(v) - 1)])
            for a, b in zip(bords, bords[1:])]


def _normalise(v) -> list[float]:
    """0 à 1 sur le 95e centile, pour que la courbe se lise malgré un pic."""
    a = np.asarray(v, dtype=float)
    if not len(a):
        return []
    haut = float(np.percentile(a, 95)) or float(a.max()) or 1.0
    return [round(float(min(1.0, x / haut)), 3) for x in a]


def batterie(audio: Path, fenetre: float = 90.0) -> list[float]:
    """L'intensité des transitoires AIGUS — la piste « batterie » de Louis.

    Le flux spectral demi-redressé au-dessus de `HZ_BATTERIE` : ce qui monte
    brusquement dans l'aigu, c'est-à-dire une caisse claire, un charley, un
    coup de crash. Volontairement aveugle à l'harmonie, puisque c'est
    exactement ce qui manque à la basse et aux accords, tous deux tirés de
    musx sur le même audio.

    CE QUE ÇA NE RÉSOUT PAS : ça ne distingue pas une batterie d'un autre
    transitoire aigu (applaudissements, claquement de porte, sifflante de la
    voix parlée). Sur une intro de clip bavarde, la parole en produit — c'est
    pour ça que la courbe de bruit ambiant l'accompagne.
    """
    X, f, pas = _spectre(Path(audio), fenetre)
    if X is None:
        return []
    haut = X[:, f >= HZ_BATTERIE]
    flux = np.maximum(0.0, np.diff(haut, axis=0)).sum(axis=1)
    return _normalise(_vers_pas(np.concatenate([[0.0], flux]), pas))


def bruit_ambiant(audio: Path, fenetre: float = 90.0) -> list[float]:
    """La platitude spectrale — haute pour du bruit, basse pour du son tenu.

    Le rapport entre moyenne géométrique et moyenne arithmétique du spectre.
    Une note de basse ou un accord concentrent leur énergie sur des
    harmoniques : platitude basse. Une salle, une foule, un souffle, une voix
    parlée s'étalent : platitude haute. C'est la courbe qui doit expliquer les
    intros de clip, où il se passe du son sans qu'il se passe de la musique.
    """
    X, _f, pas = _spectre(Path(audio), fenetre)
    if X is None:
        return []
    P = X ** 2 + 1e-12
    plat = np.exp(np.mean(np.log(P), axis=1)) / np.mean(P, axis=1)
    return _normalise(_vers_pas(plat, pas))


def pistes(audio: Path, probs=None, fenetre: float = 90.0) -> dict:
    """Les quatre courbes, au même pas, pour une carte de diagnostic.

    `{basse, accords, batterie, bruit}` — chacune de 0 à 1, une valeur tous
    les `PAS_PISTE`. Seule `basse` est lue par `debut_du_morceau` ; les autres
    servent à voir ce qui lui manque.
    """
    from harmonia import musx as _musx
    from harmonia.musx import FRAME_DT
    audio = Path(audio)
    if probs is None:
        probs = _musx.frame_posteriors(audio)
    n = int(fenetre / FRAME_DT)
    return {
        "basse": _normalise(_vers_pas(1.0 - probs[1][:n, 0], FRAME_DT)),
        "accords": _normalise(_vers_pas(1.0 - probs[0][:n, 0], FRAME_DT)),
        "batterie": batterie(audio, fenetre),
        "bruit": bruit_ambiant(audio, fenetre),
        "pas": PAS_PISTE,
    }
