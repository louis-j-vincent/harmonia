"""harmonia_min/beats.py — Beat This! beats + downbeats. The ONLY beat tracker.

Re-extracted 2026-07-30 from chord_pipeline_v1._get_beatthis (~15 lines) +
harmonia/serving/audio.py::_beats_and_downbeats (the m4a lesson), for the
minimal rebuild. Two hard-won rules are load-bearing here:

1. **librosa is banned.** It locks 2x tempo octaves (validated 2026-07-21:
   tempo-octave 65% vs Beat This! 78% on POP909; song 002 doubled to ~129 BPM
   against three agreeing GT annotations at ~64). The old code silently fell
   back to librosa whenever Beat This! threw — which was EVERY m4a on this box
   (see 2), so the fallback was the code path. Here a failure raises.

2. **Beat This! must get a wav.** It reads audio via torchaudio / soundfile /
   madmom and on this box all three refuse .m4a (no torchcodec; libsndfile has
   no AAC; madmom is py3.12-broken). Discovered 2026-07-30 after the librosa
   fallback had silently served every song. So: try the file as-is, and on any
   decode error transcode to a temp wav with ffmpeg and retry. Only if THAT
   fails do we raise.

Results are disk-cached to harmonia_min/state/beats/<stem>.json (stem-keyed,
same rationale as the musx cache: fresh downloads of the same video get new
mtimes, the stem is the stable key).
"""
from __future__ import annotations

import json
import logging
import subprocess
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent / "state" / "beats"

_f2b = None  # loaded once per process (~2 s model load)


def _get_beatthis():
    global _f2b
    if _f2b is None:
        from beat_this.inference import File2Beats
        _f2b = File2Beats(device="cpu", dbn=False)
        logger.info("beats: loaded Beat This! (MIT) beat+downbeat tracker")
    return _f2b


class BeatTrackingError(RuntimeError):
    """Beat This! could not produce a usable beat grid. No fallback exists —
    librosa is banned (2x tempo-octave lock), so the caller must surface this."""


# ── grid guard (Louis, 2026-08-05) ──────────────────────────────────────────
# « utilise ça comme garde-fou et tu refuses, signales LOUDLY une chanson dont
# la grille échoue pour mauvais calage rythmique. »
#
# The whole pipeline downstream assumes FOUR beats per bar: `pipeline.analyze`
# builds bars as `off + b*bpb` over beat INDICES with bpb=4. When the tracker
# does not actually deliver that, the "bars" are not musical bars, and every
# similarity computed on them compares misaligned material — Georgia On My Mind
# has 5.5% of its SSM above 0.80 and Billie Jean 56.7%, and neither number says
# anything about repetition (docs/known_issues.md, 2026-08-05).
#
# The measure is the beat count BETWEEN consecutive downbeats: a healthy 4/4
# track gives 4,4,4,4… Its mode is the detected metre and the share of bars
# hitting that mode is the grid's self-consistency. Measured over the 82 cached
# tracks: median consistency 1.000, and the songs that fail are exactly the ones
# whose charts were wrong — Georgia 0.45 with mode 2, blue_bossa 0.53 mode 2,
# Close to You 0.64 — while This Love, Sunny and Don't Know Why sit at 1.00 and
# Billie Jean at 0.94.
#
# A cruder ratio (len(beats)/len(downbeats)) was tried first and rejected: its
# corpus median is 3.82, so it would have flagged half of a healthy corpus.
GRID_MIN_CONSISTENCY = 0.80   # raw-consistency threshold: DIAGNOSTIC ONLY since
                              # 2026-08-07, no longer the accept/refuse test.
GRID_MIN_BARS = 30            # below this the statistic is noise (GuitarSet
                              # excerpts run 10-16 bars) — don't judge.

# ── 2026-08-07: the guard was refusing the MEASURE, not the music ────────────
# Louis hit "only 60% of bars actually hold 4 beats" on song after song. Over
# the 63-song library the raw guard refused 19; characterising all 19 (never a
# single song — rule #5) showed the mode-of-gaps statistic conflates two very
# different things:
#
#   * the tracker MARKS TOO MANY DOWNBEATS. Beat This! also flags beat 3 as a
#     bar start, so one true 4-beat bar is reported as 2 + 2. The gap histogram
#     is then bimodal — Close to You is {4:63, 2:35}, Land of 1000 Dances is
#     exactly {4:64, 2:40} and nothing else — and "consistency" collapses to
#     ~0.6 while the beats themselves are metronomic (local IBI steadiness
#     1.00). Once more than half the bars get split the MODE flips to 2, which
#     the old guard then read as the half-tempo octave error. It is not:
#     Georgia's beats sit at 65 BPM, which is Georgia's real tempo.
#   * the beats really are unreliable (insertions/deletions scattered through
#     the song).
#
# Two hypotheses were tested and REFUTED, recorded so nobody redoes them:
#   (c) argmin ambiguity — dead. Beat This!'s downbeats are an exact subset of
#       its beats (measured: 100% of downbeats sit within 1e-6 s of a beat on
#       all 63 songs), so no downbeat ever falls half-way between two beats.
#   (d) one local metric irregularity dragging a global percentage down — dead.
#       The off-grid bars are DISPERSED, not clustered: Chiquitita has 28
#       separate uncovered runs, Autumn Leaves 26, and on every low-coverage
#       song the three longest runs together cover under 20% of the song.
#
# So the measure is replaced by one that cannot be fooled by extra downbeats:
# tile the beat axis with exact `metre`-beat bars using ONLY beat indices the
# tracker already called downbeats. Coverage = share of the span so tiled.
#   * It can only DELETE the tracker's downbeats, never invent one, so it can
#     never invent a bar phase — the Georgia trap cannot be rebuilt this way.
#   * A genuine half-tempo lock stays refused — but NOT by coverage, see
#     GRID_MIN_DIRECT below. Metre 2 remains refused; it just stops being
#     diagnosed on songs that are not in 2.
#   * On the 44 songs the old guard accepted, the repaired grid reproduces the
#     pipeline's (bpb, phase) EXACTLY — 44/44, measured. Zero regression.
GRID_MIN_COVERAGE = 0.85      # healthy songs bottom out at 0.87 (Kermit) and
                              # the best unrepairable one reaches 0.81
                              # (Chiquitita) — 0.85 sits in that gap.
GRID_METRES = (4, 3, 6)       # L'ORDRE COMPTE : 4 gagne les égalités, donc il
                              # reste en tête. 6 ajouté le 2026-08-07 (6/8,
                              # validé sur deux morceaux du corpus, voir
                              # `check_grid`). 2 reste refusé — c'est la
                              # signature du verrou d'octave demi-tempo — et
                              # 5/7 aussi : aucun morceau du corpus ne les a
                              # exercés (règle #4, ce reste n'est PAS résolu ici).
# The tiling must be ANCHORED by the tracker, never invented. A downbeat track
# that is uniformly every 2 beats — the true half-tempo octave lock — tiles
# perfectly at 4 and would sail through on coverage alone (found while writing
# the red test for it, not after shipping). So the chosen metre must also
# appear directly in the tracker's own bar gaps. Corpus range: 0.00 for a pure
# 2-lock, 0.05 for A-DuOmA75lI, then a jump to 0.19 (Nina Simone) for the
# lowest song we want to accept. The margin above 0.15 is thin — one song — so
# this threshold is the first thing to re-measure if a 4/4 song is ever refused
# with "beats per bar" in the message.
GRID_MIN_DIRECT = 0.15


def grid_quality(beats, downbeats) -> dict:
    """{metre, consistency, n_bars} — how well the downbeats tile the beats.

    RAW statistic, kept for diagnostics only. It reads an over-marked downbeat
    track as a broken grid; `repair_grid` is what decides. See the block
    comment above.
    """
    import numpy as np
    from collections import Counter
    b = np.asarray(beats, float)
    db = np.asarray(downbeats, float)
    if len(b) < 8 or len(db) < 4:
        return {"metre": None, "consistency": 0.0, "n_bars": 0}
    idx = [int(np.argmin(np.abs(b - t))) for t in db]
    gaps = [j - i for i, j in zip(idx, idx[1:]) if j > i]
    if len(gaps) < 3:
        return {"metre": None, "consistency": 0.0, "n_bars": len(gaps)}
    metre, cnt = Counter(gaps).most_common(1)[0]
    return {"metre": int(metre), "consistency": cnt / len(gaps),
            "n_bars": len(gaps)}


MAX_BRIDGE = 4     # au-delà, ce n'est plus une ligne oubliée, c'est une rupture


def _tile(idx: list[int], metre: int, max_bridge: int = MAX_BRIDGE):
    """Plus long pavage en mesures de `metre` temps, à partir des downbeats.

    Renvoie (couverture, indices de début de mesure, nombre de mesures).

    ON ENJAMBE LES LIGNES OUBLIÉES (2026-08-07). La version d'origine n'acceptait
    une mesure que si le traceur avait marqué SES DEUX BOUTS : une seule ligne
    manquante cassait la chaîne et faisait perdre tout ce qui suivait. Mesuré,
    c'était le vrai blocage, pas le choix de la métrique — Piano Man est lu en 3
    avec un ancrage de 0.85, donc sans la moindre ambiguïté, et se faisait refuser
    à 78 % de couverture contre un plancher à 85 %.

    On autorise donc à relier deux downbeats séparés par k×metre temps, en
    comptant k mesures et en insérant les k-1 lignes intermédiaires. Ce qu'on
    insère n'est jamais inventé : les positions tombent sur de VRAIS temps, et
    les deux extrémités sont de VRAIS downbeats du traceur. On ne crée donc
    aucune phase — le piège de Georgia reste inatteignable par ce chemin, et le
    garde `direct` le couvre de toute façon.

    `max_bridge` borne l'enjambée : au-delà de quatre mesures d'affilée sans
    aucune marque, ce n'est plus un oubli du traceur mais une rupture, et la
    prétendre couverte serait mentir.
    """
    idx = sorted({int(v) for v in idx})
    if len(idx) < 2:
        return 0.0, [], 0
    at = {v: k for k, v in enumerate(idx)}
    n = len(idx)
    best = [0] * n
    back: list[tuple[int, int, int] | None] = [None] * n
    for j in range(n):
        cur, bk = (best[j - 1], (0, j - 1, 0)) if j else (0, None)
        for k in range(1, max_bridge + 1):
            p = at.get(idx[j] - k * metre)
            if p is not None and best[p] + k > cur:
                cur, bk = best[p] + k, (1, p, k)
        best[j], back[j] = cur, bk
    starts: list[int] = []
    j: int | None = n - 1
    while j is not None and j >= 0:
        if back[j] is None:
            break
        kind, k, nbars = back[j]
        if kind == 1:
            starts += [idx[k] + i * metre for i in range(nbars + 1)]
        j = k
    span = idx[-1] - idx[0]
    cov = min(metre * best[-1] / span, 1.0) if span else 0.0
    return cov, sorted(set(starts)), best[-1]


DUP_TOL = 0.25     # en deca, deux temps ne peuvent pas etre deux temps

RIGIDE_ECART = 0.01   # trois zones d'accord a 1 % : le morceau ne derive pas
RIGIDE_ZONE = 24      # battues par zone (debut, milieu, fin)
RIGIDE_COLLE = 0.25   # 95 % des battues a moins d un quart de temps d une case :
                      # les morceaux acceptes sont a 13 % au pire, les refuses
                      # a 44 % au mieux — le seuil se pose dans ce fosse


def _periode_locale(t):
    """La periode d'un paquet de battues consecutives : la pente de t vs rang."""
    import numpy as np
    i = np.arange(len(t))
    A = np.vstack([i, np.ones_like(i, dtype=float)]).T
    P, phi = np.linalg.lstsq(A, np.asarray(t, float), rcond=None)[0]
    return float(P), float(phi)


BPM_FEN = 16          # battues par fenetre glissante
BPM_TOL = 0.15        # une battue "tombe juste" a 15 % du temps pres
BPM_MIN_INLIERS = 0.6 # sous ce taux, on n'a pas trouve de tempo stable


def bpm_rigide(beats, fen=BPM_FEN, tol=BPM_TOL, iters=6):
    """Le tempo du morceau, pris dans son COEUR et non sur ses bords.

    Louis, 2026-08-18, apres avoir regarde les six morceaux refuses : « il n'y a
    pas de drift notable, par contre souvent des intros sans tempo, donc il faut
    ajuster le tempo par rapport au milieu du morceau peut-etre. Dans tous les
    cas trouve-moi un algo rigide pour inferer le bpm. »

    Il a raison sur le diagnostic : ces morceaux ne ralentissent pas, ils
    DECROCHENT par endroits. Stand By Me lit 92,7 / 119,6 / 120,0 BPM sur ses
    trois zones — deux d'accord, et c'est l'intro qui manque des temps. Smooth
    Criminal lit 19,9 sur sa premiere zone parce qu'elle tombe dans un trou.
    Tester trois points fixes etait donc trop fragile : une seule mauvaise zone
    suffisait a refuser un morceau metronomique.

    L'ALGORITHME, en trois temps, aucun seuil devine :

      1. **fenetres glissantes** — la periode locale de chaque fenetre de `fen`
         battues consecutives (pente de t vs rang). Une fenetre posee dans une
         intro sans tempo ou dans un trou donne une valeur aberrante ; il y en a
         des dizaines d'autres.
      2. **la mediane des fenetres** — elle ignore les aberrantes par
         construction, sans avoir a les reconnaitre. C'est le « milieu du
         morceau » qu'il demande, au sens statistique plutot que temporel : le
         tempo majoritaire, d'ou qu'il vienne.
      3. **raffinement sur les seules battues qui tombent juste** — on cale une
         grille a cette periode, on garde les battues a moins de `tol` d'une
         case (les inliers), on refait la regression sur elles, on recommence.
         Les battues de l'intro sans tempo ne votent pas.

    Rend {periode, bpm, phase, inliers, ecart_median} ou None si moins de
    `BPM_MIN_INLIERS` des battues tombent juste — auquel cas le morceau n'a
    vraiment pas UN tempo, et il faut le laisser tranquille.

    Ce que ca ne resout pas : un morceau qui change VRAIMENT de tempo en cours
    de route (deux tempos legitimes) rend ici le majoritaire, et les inliers
    chutent — c'est ce que `inliers` sert a dire.
    """
    import numpy as np
    b = np.asarray([float(t) for t in beats], float)
    if len(b) < 3 * fen:
        return None
    # 1. les periodes locales, fenetre par fenetre
    locales = []
    for i in range(0, len(b) - fen + 1, max(1, fen // 2)):
        P, _phi = _periode_locale(b[i:i + fen])
        if P > 0:
            locales.append(P)
    if not locales:
        return None
    # 2. la mediane : les fenetres aberrantes ne pesent pas
    P0 = float(np.median(locales))
    # 2 bis. VERROUILLAGE DE PHASE. La mediane des fenetres donne le bon tempo a
    # 1 % pres — et 1 % d'erreur sur la periode fait QUATRE temps d'ecart au
    # bout de quatre cents battues. On affine donc en balayant finement autour
    # d'elle et en gardant la periode qui aligne le mieux TOUTES les battues :
    # R(P) = |moyenne des exp(2i.pi.t/P)| vaut 1 quand chaque battue tombe sur
    # une case, 0 quand elles sont dispersees. C'est ce que la regression sur
    # les inliers ne pouvait pas trouver seule : partie d'une periode a 1 %
    # pres, elle ne voyait d'inliers que dans la region ou la grille tombait
    # juste par hasard, et s'y enfermait (Goodbye Yellow Brick Road : 24 %
    # d'inliers, ecart median 31 %, alors que ses fenetres locales s'accordent
    # a 0,5 %).
    grille_P = P0 * (1.0 + np.linspace(-0.03, 0.03, 4001))
    R = np.abs(np.exp(2j * np.pi * b[:, None] / grille_P[None, :]).mean(axis=0))
    P = float(grille_P[int(np.argmax(R))])
    # LA PHASE VIENT DU MILIEU, pas du premier temps (Louis : « souvent des
    # intros sans tempo, donc il faut ajuster le tempo par rapport au milieu du
    # morceau »). Ancree sur b[0], une intro decalee entrainait tout le
    # raffinement avec elle : Goodbye Yellow Brick Road, Georgia, Sunrise et
    # If I Ain't Got You etaient refuses pour cette seule raison.
    ang = 2 * np.pi * (b % P) / P
    phi = float(np.angle(np.exp(1j * ang).mean()) % (2 * np.pi)) / (2 * np.pi) * P
    # 3. raffinement sur les inliers
    for _ in range(iters):
        k = np.round((b - phi) / P)
        ecart = np.abs(b - (phi + k * P)) / P
        dedans = ecart <= tol
        if dedans.sum() < 2 * fen:
            break
        A = np.vstack([k[dedans], np.ones(int(dedans.sum()))]).T
        Pn, phin = np.linalg.lstsq(A, b[dedans], rcond=None)[0]
        if Pn <= 0 or abs(Pn - P) / P > 0.25:
            break                       # garde-fou : pas de saut d'octave
        P, phi = float(Pn), float(phin)
    k = np.round((b - phi) / P)
    ecart = np.abs(b - (phi + k * P)) / P
    dedans = ecart <= tol
    part = float(dedans.mean())
    if part < BPM_MIN_INLIERS:
        return None
    return {"periode": round(P, 5), "bpm": round(60.0 / P, 2),
            "phase": round(phi % P, 5), "inliers": round(part, 3),
            "ecart_median": round(float(np.median(ecart)), 4)}


def grille_rigide(beats, downbeats, ecart_max=RIGIDE_ECART, zone=RIGIDE_ZONE):
    """Une seule grille continue quand le morceau ne derive PAS.

    Louis, 2026-08-18, apres avoir vu les doublons de Beat This! sur la regle de
    temps : « la grille reste rigide en dehors, donc tu mesures ce doublon-la
    qui ne sert a rien. Soit il y a un drift, on le prend en compte ; soit il
    n'y a pas de drift, et a ce moment-la c'est des doublons mal detectes par
    BTS. Regarde la grille rigide des premieres mesures, du milieu et des
    dernieres : si elles ont toutes le meme BPM, tu relies tout ensemble et tu
    fais une longue grille continue. »

    C'est exactement ce que fait cette fonction. On mesure la periode sur trois
    paquets de `zone` battues — au debut, au milieu, a la fin. Si les trois
    tombent d'accord a `ecart_max` pres, le morceau est metronomique : on rend
    une grille `phi + k*P` continue du debut a la fin, et les doublons
    disparaissent par CONSTRUCTION, sans avoir a les chasser un par un.

    Sinon on rend `None` : le morceau derive pour de vrai (une ballade rubato
    ne se rigidifie pas), et l'appelant garde la grille du traqueur.

    MESURE sur les 84 morceaux du disque qui ont assez de battues : 25 (30 %)
    passent le test. Parmi les 18 qui portent des doublons, 4 passent — dont
    Another Day (0,60 % d'ecart entre zones, 15 doublons), le morceau qui a
    motive la regle. Les 14 autres derivent franchement : Chiquitita 15 %,
    At Last 65 %, Georgia On My Mind 86 %.

    La phase vient de TOUTES les battues (moyenne circulaire a la periode
    trouvee), pas du premier paquet : une phase prise au debut se paie sur trois
    minutes. Les downbeats sont recales sur la case la plus proche, pour que les
    mesures ne bougent pas.

    Ce que ca ne resout pas : un morceau qui derive garde ses doublons, traites
    localement par `drop_duplicate_beats`. Et le test ne voit pas un changement
    de tempo qui reviendrait a son point de depart.
    """
    import numpy as np
    # LES DOUBLONS D'ABORD. Ils se concentrent la ou le traqueur peine — donc
    # souvent dans une seule zone — et ils y ecrasent la periode locale : sur
    # Another Day la zone de fin mesurait 0,2933 s au lieu de 0,4442, soit 38 %
    # d'ecart entre zones, et le test refusait un morceau parfaitement
    # metronomique. Nettoyes, les trois zones tombent a 0,60 %.
    propres, _ = drop_duplicate_beats(beats, downbeats or [])
    b = np.asarray([float(t) for t in propres], float)
    if len(b) < 3 * zone:
        return None
    n = len(b)
    paquets = (b[:zone], b[n // 2 - zone // 2:n // 2 + zone // 2], b[-zone:])
    Ps = [_periode_locale(z)[0] for z in paquets]
    if min(Ps) <= 0:
        return None
    if (max(Ps) - min(Ps)) / float(np.mean(Ps)) > ecart_max:
        return None                      # le morceau derive : on ne rigidifie pas
    P = float(np.mean(Ps))
    ang = 2 * np.pi * (b % P) / P
    phi = float(np.angle(np.exp(1j * ang).mean()) % (2 * np.pi)) / (2 * np.pi) * P
    k0 = int(np.floor((b[0] - phi) / P + 0.5))
    k1 = int(np.floor((b[-1] - phi) / P + 0.5))
    grille = [round(phi + k * P, 4) for k in range(k0, k1 + 1)]
    if not grille:
        return None
    g = np.asarray(grille)
    # ELLE DOIT COLLER A CE QU'ELLE REMPLACE. Trois zones d'accord sur le BPM ne
    # suffisent pas : un morceau peut battre au meme tempo au debut, au milieu
    # et a la fin en ayant glisse de phase entre les deux. On verifie donc que
    # les battues du traqueur tombent VRAIMENT sur les cases — sans cette garde,
    # blue_bossa_150bpm et XpqqjU7u5Yc passaient avec 5 % de leurs battues a
    # pres d'un demi-temps de la case la plus proche, c'est-a-dire une grille
    # rigide qui deplace la musique au lieu de la decrire.
    ecarts = np.array([np.min(np.abs(g - t)) for t in b]) / P
    if float(np.percentile(ecarts, 95)) > RIGIDE_COLLE:
        return None
    dbs = [round(float(g[int(np.argmin(np.abs(g - float(t))))]), 4)
           for t in downbeats or []]
    # une case ne porte qu'un downbeat, et l'ordre est garde
    vus, dbs_p = set(), []
    for t in dbs:
        if t not in vus:
            vus.add(t); dbs_p.append(t)
    return {"beats": grille, "downbeats": dbs_p, "periode": round(P, 5),
            "ecart_zones": round((max(Ps) - min(Ps)) / float(np.mean(Ps)), 5),
            "colle_p95": round(float(np.percentile(ecarts, 95)), 4)}



def drop_duplicate_beats(beats, downbeats, tol=DUP_TOL):
    """Retire les temps JUMEAUX — deux marques pour un seul temps.

    Louis, 2026-08-18, sur Another Day de Jamie Lidell : « les dernieres
    sections devraient toutes etre un A et ils decalent, comment ca se fait ? ».
    Mesure : le morceau tient 136,4 BPM du debut a la fin, mais Beat This! pose
    **15 temps de trop apres 149,5 s**, chacun a exactement 0,080 s du
    precedent — 0,18 fois le temps median. Onze mesures sur les vingt-cinq de
    la derniere section font alors 20 a 50 % de moins que les autres, dont
    quatre font une DEMI-mesure, et la boucle de quatre accords glisse d'une
    mesure toutes les huit. Les accords etaient justes ; c'est la grille qui
    avait bouge.

    Ce n'est PAS le doublement de tempo que traite `drop_inserted_beats` : un
    temps insere tombe a la MOITIE du temps (0,5), un jumeau tombe a 0,18. Sur
    les 145 morceaux du disque, les intervalles sous 0,62 fois la mediane se
    groupent en trois paquets — 0,10 a 0,20 (les jumeaux), un pic a 0,35 (un
    tiers de temps, un autre phenomene qu'on ne touche pas) et 0,50 (l'octave,
    deja traitee). Le seuil se pose donc a 0,25, entre le premier paquet et le
    deuxieme.

    ON GARDE CELUI QUI TOMBE LE MIEUX : des deux jumeaux, on conserve celui le
    plus proche de la position attendue (le temps precedent conserve + la
    periode mediane), pas systematiquement le premier ni le second. C'est ce
    qui evite d'introduire un decalage la ou on venait d'en retirer un.

    **On ne fait que supprimer, jamais inserer** — meme doctrine que
    `repair_grid` et `drop_inserted_beats`, donc aucune phase n'est inventee.
    """
    import numpy as np
    if len(beats) < 8:
        return beats, downbeats
    b = [float(t) for t in beats]
    med = float(np.median(np.diff(b)))
    if med <= 0:
        return beats, downbeats
    out = [b[0]]
    i = 1
    while i < len(b):
        if b[i] - out[-1] >= tol * med:
            out.append(b[i])
            i += 1
            continue
        # jumeaux : on garde celui qui tombe le plus pres de l'attendu
        attendu = out[-1] + med
        jumeaux = [out[-1], b[i]]
        garde = min(jumeaux, key=lambda t: abs(t - attendu))
        out[-1] = garde
        i += 1
    if len(out) == len(b):
        return beats, downbeats
    keep = set(out)
    return out, [t for t in downbeats if float(t) in keep]


HALF_TOL = 0.18    # « la moitie d'un temps », a 18 % pres


def drop_inserted_beats(beats, downbeats, tol=HALF_TOL):
    """Retire les temps que le traceur a INSERES en verrouillant le double tempo.

    Louis, 2026-08-08, sur Easy des Commodores : « je crois qu'il y a un léger
    décalage sur nos beats, ou c'est leur musique qui est trop funky ». Ni l'un
    ni l'autre — mesuré, la distribution des intervalles entre temps y est
    BIMODALE ET EXACTE :

        0.5 x le temps median :  50 intervalles
        1.0 x le temps median : 250 intervalles
        et rien entre les deux

    Le traceur double donc le tempo par endroits, sur onze zones couvrant 22 s
    des 250 s du morceau, et toujours PAR PAIRES — un temps inséré entre deux
    vrais. C'est l'erreur d'octave classique, mais locale au lieu d'être globale,
    et invisible dans toutes nos vérifications précédentes parce que la métrique
    et la couverture restaient bonnes : les mesures étaient simplement trop
    courtes par endroits.

    La réparation est déterministe parce que la période globale ne prête pas à
    discussion (250 intervalles sur 302 au même écart) : deux intervalles
    consécutifs valant chacun la moitié de la médiane se recollent en un seul
    temps. **On ne fait que supprimer, jamais insérer** — même doctrine que
    `repair_grid`, donc aucune phase n'est inventée.

    Mesuré sur les 66 morceaux du disque : 59 intouchés, 7 modifiés (Easy,
    Georgia, Yam-B, Chiquitita, Autumn Leaves, Jorja, A-Du), et AUCUN ne change
    de verdict d'acceptation. Sur Easy l'irrégularité des temps passe de 18.9 %
    à 1.9 % et la couverture de 0.95 à 0.99.

    Ce que ça ne résout pas (règle #4) : les morceaux dont les temps sont
    irréguliers SANS cette signature exacte — Alessi reste à 36 %, Bein Green à
    29 %. Ceux-là ont un autre problème, que cette fonction ne voit pas.
    """
    import numpy as np
    if len(beats) < 8:
        return beats, downbeats
    b = [float(t) for t in beats]
    med = float(np.median(np.diff(b)))
    if med <= 0:
        return beats, downbeats
    out, i = [b[0]], 0
    while i < len(b) - 1:
        d1 = b[i + 1] - b[i]
        d2 = b[i + 2] - b[i + 1] if i + 2 < len(b) else None
        if (d2 is not None and abs(d1 / med - .5) < tol
                and abs(d2 / med - .5) < tol):
            out.append(b[i + 2]); i += 2        # le temps du milieu est insere
        else:
            out.append(b[i + 1]); i += 1
    if len(out) == len(b):
        return beats, downbeats
    keep = set(out)
    return out, [t for t in downbeats if float(t) in keep]


def repair_grid(beats, downbeats, metres: tuple = GRID_METRES) -> dict:
    """Drop the tracker's mid-bar downbeats and report what is left.

    Returns {metre, coverage, n_bars, downbeats, kept, raw_metre,
    raw_consistency}. `downbeats` is the repaired list of bar-start TIMES — a
    subset of the input, in seconds.

    What this does NOT solve (rule #4): it never adds a bar start, so a bar
    line the tracker missed entirely stays missing, and the uncovered stretches
    are reported through `coverage` rather than repaired. It also says nothing
    about slow tempo drift — the bars are indexed on real beat times, so drift
    is carried by the beats, but a tracker that drifts off the music is
    invisible here.
    """
    import numpy as np
    b = np.asarray(beats, float)
    db = np.asarray(downbeats, float)
    raw = grid_quality(b, db)
    out = {"metre": None, "coverage": 0.0, "n_bars": raw["n_bars"],
           "downbeats": [], "kept": 0.0, "direct": 0.0,
           "raw_metre": raw["metre"], "raw_consistency": raw["consistency"],
           "consistency": raw["consistency"]}
    if len(b) < 8 or len(db) < 4:
        return out
    idx = [int(np.argmin(np.abs(b - t))) for t in db]
    gaps = [j - i for i, j in zip(idx, idx[1:]) if j > i]
    # LE CHOIX DE LA MÉTRIQUE. La couverture seule ne suffit PAS, et le test
    # rouge l'a montré en ajoutant 6 le 2026-08-07 : une valse se pave
    # parfaitement en 6 (deux mesures de trois), donc « à égalité, la plus
    # grande » transformait toutes les valses en 6/8. C'est le piège de
    # Georgia un cran plus haut, exactement ce que le commentaire d'origine
    # annonçait.
    #
    # On départage donc par `direct` — la part de mesures que le TRACEUR
    # marque lui-même à cette longueur — qui est la doctrine déjà écrite
    # au-dessus : le pavage doit être ancré, jamais inventé. Sur une valse
    # direct(3)=1.00 contre direct(6)=0.00 ; sur l'Alicia Keys en 6/8
    # direct(6)=0.84 contre direct(3)=0.03. La couverture ne les sépare pas,
    # celui-ci les sépare franchement.
    def _direct(m):
        return (sum(1 for g in gaps if g == m) / len(gaps)) if gaps else 0.0

    # L'ordre des deux critères compte, et le second test rouge l'a montré :
    # filtrer d'abord par la couverture laissait une grille en 3 abîmée se faire
    # lire en 6 (couverture meilleure, `direct` nul) puis refuser avec le mauvais
    # message. On ne RETIENT donc que les métriques que le traceur marque
    # vraiment, et on maximise la couverture parmi celles-là. Si aucune ne passe
    # — le verrou d'octave demi-tempo — on retombe sur la meilleure couverture
    # pour que le garde puisse refuser en le disant correctement.
    scored = []
    for m in metres:
        cov, starts, nb = _tile(idx, m)
        scored.append((cov, _direct(m), m, starts, nb))
    # …et parmi les métriques ancrées on prend LA MIEUX ANCRÉE, pas la mieux
    # couverte. Troisième piège trouvé en mesurant, pas en relisant : la
    # couverture favorise mécaniquement la grande métrique — une mesure plus
    # longue se pave plus facilement — donc départager par elle relisait une
    # valse en 6/8. Ju8Hr50Ckwk : écarts {3: 84, 6: 19}, direct 0.76 contre
    # 0.17, mais couverture 0.67 contre 0.96. `direct` les sépare dans le bon
    # sens, la couverture dans le mauvais.
    anchored = [x for x in scored if x[1] >= GRID_MIN_DIRECT]
    if anchored:
        cov, direct, metre, starts, nb = max(anchored,
                                             key=lambda x: (x[1], x[2], x[0]))
    else:
        # Rien d'ancré : le traceur ne marque aucune mesure d'une longueur
        # admise. Sur un morceau assez long le garde refusera ; sur un morceau
        # trop court pour être jugé il passe, et on ne veut pas qu'il passe en
        # ANNONÇANT 6/8 sur la foi de la seule couverture. On retombe donc sur
        # les métriques historiquement validées.
        fallback = [x for x in scored if x[2] in (4, 3)] or scored
        cov, direct, metre, starts, nb = max(fallback,
                                             key=lambda x: (x[0], x[2]))
    out.update(metre=metre, coverage=cov, n_bars=nb, direct=direct,
               downbeats=[round(float(b[i]), 4) for i in starts],
               kept=len(starts) / len(idx) if idx else 0.0)
    return out


def check_grid(beats, downbeats, name: str, bpb: int = 4,
               allowed: tuple = GRID_METRES) -> dict:
    """Raise BeatTrackingError unless the grid carries a legitimate metre.

    Refuses loudly rather than producing a chart built on bars that are not
    bars — a wrong chart is worse than no chart, and this failure was
    previously invisible.

    2026-08-07 (Louis, Alicia Keys refusée en 6 : « je sais que ça va ») — une
    métrique de 6 est un 6/8, pas une erreur. Les deux morceaux du corpus qui la
    montrent sont sans ambiguïté : l'histogramme des écarts entre downbeats est
    écrasé par 6 (93 sur 110, et 48 sur 64), le pavage en 6 couvre 0.97 et 0.85
    avec un `direct` de 0.84 et 0.73, tandis que 2, 3 et 4 plafonnent tous à 0.06.
    Ce n'est donc pas un 3 doublé : le traceur marque LUI-MÊME des mesures de six
    temps, ce qu'un 3/4 dont on aurait raté une mesure sur deux ne ferait pas.
    Le garde `direct` reste ce qui protège du piège d'octave, exactement comme
    pour 4 — on n'ajoute aucun mécanisme, on ouvre une valeur validée.

    2026-08-07 (Louis: « les tiers de barre doivent pouvoir s'afficher ») —
    a detected metre of 3 is a WALTZ, not an error: it is accepted alongside
    4. 5/6/7 stay refused: no corpus song has exercised them, and letting an
    unvalidated metre through would silently rebuild the georgia trap one
    number higher (rule #4: that remainder is NOT solved here).

    2026-08-07, second pass — the verdict is now taken on the REPAIRED grid
    (`repair_grid`), because the raw statistic was refusing the measure and
    not the music: 13 of the 19 refusals were songs whose beats are
    metronomic and whose only fault is that Beat This! also marks beat 3 as a
    bar start. Metre 2 is still refused, but now only when the song really is
    in 2 — not merely because more than half its bars got split in two. NOTE
    the mechanism, because the obvious guess is wrong: a downbeat every 2
    beats tiles PERFECTLY at 4 (coverage 1.0), so coverage does not catch the
    octave error at all. What catches it is `direct` — zero bars of 4 beats
    are visible in the tracker's own gaps, so the 4 would be invented.

    Returns the repaired grid; the caller MUST use ``result["downbeats"]``
    downstream, not the tracker's raw list, or the repair buys nothing.
    """
    r = repair_grid(beats, downbeats, metres=tuple(allowed))
    raw_bars = grid_quality(beats, downbeats)["n_bars"]
    if raw_bars < GRID_MIN_BARS:
        return r                       # too short to judge; let it through
    if r["direct"] < GRID_MIN_DIRECT:
        # No bar of the chosen metre is actually visible in the tracker's own
        # downbeats — the tiling would be inventing it. This is what a true
        # half-tempo octave lock looks like (every downbeat exactly 2 beats
        # apart: it tiles at 4 for free, and every "bar" would be two).
        raise BeatTrackingError(
            f"{name}: the beat tracker reports {r['raw_metre']} beats per bar "
            f"— not a metre this chart can carry (allowed: "
            f"{'/'.join(map(str, allowed))}) — the bar grid would be wrong "
            f"for the whole song. (only {r['direct']:.0%} of its bars are "
            f"{r['metre']} beats long, needs {GRID_MIN_DIRECT:.0%})")
    if r["coverage"] >= GRID_MIN_COVERAGE:
        return r
    raise BeatTrackingError(
        f"{name}: only {r['coverage']:.0%} of the song can be laid out in "
        f"{r['metre']}-beat bars (needs {GRID_MIN_COVERAGE:.0%}) over "
        f"{raw_bars} bars — the rhythm is too loose or rubato for a "
        f"fixed bar grid, so the chart would be built on bars that are "
        f"not bars.")


def _clean(d: dict) -> dict:
    """Le cache garde la sortie BRUTE du traceur ; la réparation s'applique à la
    lecture.

    Placée d'abord à l'écriture, elle ne servait à rien : `track` rend le JSON
    stocké avant d'y arriver, donc les 66 morceaux déjà en cache — c'est-à-dire
    tous — gardaient leurs temps doublés. Ici elle s'applique aux deux chemins,
    et le cache reste comparable à ce que Beat This! a réellement produit.
    """
    import numpy as np
    brut = d.get("beats", [])
    # LES JUMEAUX D'ABORD. Une paire de temps a 0,08 s l'un de l'autre fausse la
    # mediane que `drop_inserted_beats` utilise pour reconnaitre un demi-temps ;
    # on nettoie donc les doublons avant de chercher l'octave.
    b, db = drop_duplicate_beats(brut, d.get("downbeats", []))
    b, db = drop_inserted_beats(b, db)
    # PUIS LA GRILLE RIGIDE (Louis, 2026-08-18 : « donne-le en entree de la
    # grille en prod »). Quand `bpm_rigide` trouve un tempo qui explique le
    # morceau, on pose SA grille : un temps toutes les P secondes, du debut a la
    # fin. Les trous que le traqueur laisse dans une intro ou un pont sont alors
    # combles, et les temps qu'il pose de travers sont remis en place — c'est ce
    # que Louis demande depuis qu'il a vu la grille glisser sous Another Day.
    rig = _poser_grille_rigide(b, db)
    if rig is not None:
        b, db = rig
    elif len(b) == len(brut):
        return d
    return {**d, "beats": b, "downbeats": db,
            "bpm": round(60.0 / float(np.median(np.diff(b))), 2)}


#: Sous ce taux de battues qui tombent juste, on garde la grille du traqueur :
#: le morceau n'a pas UN tempo, et le rigidifier deplacerait la musique.
GRILLE_MIN_INLIERS = 0.85


def _poser_grille_rigide(beats, downbeats):
    """(beats, downbeats) reposes sur le tempo rigide, ou None.

    On ne rigidifie que si `bpm_rigide` trouve un tempo ET qu'au moins
    GRILLE_MIN_INLIERS des battues y tombent deja : au-dessous, la grille
    deplacerait trop de temps pour qu'on puisse dire qu'elle les DECRIT.
    Les downbeats sont recales sur la case la plus proche, pour que les mesures
    ne changent pas de place.
    """
    import numpy as np
    r = bpm_rigide(beats)
    if r is None or r["inliers"] < GRILLE_MIN_INLIERS:
        return None
    b = np.asarray([float(t) for t in beats], float)
    P, phi = r["periode"], r["phase"]
    k0 = int(np.floor((b[0] - phi) / P + 0.5))
    k1 = int(np.floor((b[-1] - phi) / P + 0.5))
    if k1 - k0 + 1 < 8:
        return None
    g = np.array([phi + k * P for k in range(k0, k1 + 1)])
    dbs, vus = [], set()
    for t in downbeats or []:
        c = round(float(g[int(np.argmin(np.abs(g - float(t))))]), 4)
        if c not in vus:
            vus.add(c); dbs.append(c)
    return [round(float(x), 4) for x in g], dbs


def track(audio_path: str | Path, *, use_cache: bool = True) -> dict:
    """Beats + downbeats for one audio file.

    Returns {"beats": [s...], "downbeats": [s...], "bpm": float}.
    Raises BeatTrackingError instead of ever falling back to another tracker.
    """
    audio_path = Path(audio_path)
    cache = CACHE_DIR / f"{audio_path.stem}.json"
    if use_cache and cache.exists():
        try:
            d = json.loads(cache.read_text(encoding="utf-8"))
            if len(d.get("beats", [])) >= 4:
                return _clean(d)
        except ValueError:
            pass

    f2b = _get_beatthis()
    try:
        bts, dbs = f2b(str(audio_path))
    except Exception:
        # Almost always "cannot decode m4a" — transcode and retry (lesson 2).
        with tempfile.TemporaryDirectory() as td:
            wav = f"{td}/a.wav"
            try:
                subprocess.run(
                    ["ffmpeg", "-v", "error", "-y", "-i", str(audio_path),
                     "-ac", "1", "-ar", "22050", wav],
                    check=True, timeout=300)
                bts, dbs = f2b(wav)
            except Exception as exc:
                raise BeatTrackingError(
                    f"Beat This! failed on {audio_path.name} even after ffmpeg "
                    f"transcode ({exc}). No librosa fallback — fix the input."
                ) from exc

    beats = [round(float(t), 4) for t in bts]
    downbeats = [round(float(t), 4) for t in dbs]
    if len(beats) < 4:
        raise BeatTrackingError(
            f"Beat This! returned only {len(beats)} beats for {audio_path.name}")
    import numpy as np
    bpm = round(60.0 / float(np.median(np.diff(beats))), 2)
    out = {"beats": beats, "downbeats": downbeats, "bpm": bpm}
    if use_cache:
        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(out), encoding="utf-8")
    return _clean(out)
