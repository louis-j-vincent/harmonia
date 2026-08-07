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
                return d
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
    return out
