"""harmonia_min/pipeline.py — the thin orchestration. Audio in → ChartModel out.

    audio → Beat This! beats/downbeats → musx frame posteriors →
    beat-grid re-decode (boundaries ON our beats) → key from decoded chords →
    bars on REAL downbeats → ChartModel dict for app_shell.html

Deliberate divergences from the old pipeline (feat/minimal-pipeline, 2026-07-31):

* **The bar grid is Beat This!'s real downbeats**, not a synthetic constant-
  tempo lattice and not chord-onset anchors interpolated per bar
  (known_issues "BOTH CLOCKS ARE SYNTHETIC"). Synthetic bar lines exist only
  where the tracker gave none (leading pickup / trailing bars), extrapolated
  at the median bar length.
* **No repeat folding, no section detection**: one section spanning the whole
  song, every bar written out. Sections/folding are a later milestone, scoped
  with Louis.
* **No post-passes**: no Occam, no vocabulary fold, no section arbiter. What
  the re-decode says is what renders.
* **Confidence is acoustic**: mean musx triad posterior of the decoded label
  over the segment's frames — not the old repetition-count heuristic.

ChartModel contract (mirrors chart_model.py's docstring + what loadModel in
app_shell.html actually reads): {file,title,video_id,audio_url,key,keyName,
bpb,nBars,barGrid,beatTimes,form,sections:[{id,label,tag,reps,spans,barRanges,
bars,barSpans}]}, Bar=[Chord×0..bpb] (granularity unrestricted 2026-08-07;
typesetting already handles crammed 3-4 chord bars),
Chord={root,q,c,bass,nc,bar,beat,t0,t1}.
barSpans[r]=[[t0,t1]] is the playhead's map (server-built, one pass each).

DEUX RENDUS, UN SEUL PIPELINE (2026-08-07). `analyze_steps()` rend d'abord un
chart BRUT — mesures, accords, audio, une section « A » — puis reprend et rend
le chart RAFFINÉ (sections, repli, analyse harmonique). `analyze()` reste le
raccourci qui ne renvoie que le second. Voir la docstring d'`analyze_steps`
pour les mesures qui ont dicté l'endroit de la coupure.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

from harmonia_min import beats as _beats
from harmonia_min import musx as _musx
from harmonia_min.key_profiles import infer_key
from harmonia_min.labels import chord_pcs, to_chord

logger = logging.getLogger(__name__)

# The triad-plane family table now lives in musx.TRIAD_FAMILY, next to the
# posteriors it indexes and shared with folding. Alias kept for callers.
_TRIAD_FAMILY = _musx.TRIAD_FAMILY


def _segment_confidence(triad: np.ndarray, t0: float, t1: float,
                        label: str) -> float:
    """Mean triad-plane posterior of the decoded label over its own frames.

    Delegates to `musx.label_confidence` so that folding — which computes the
    same thing on its AVERAGED template — cannot drift onto a different scale.
    See that function for why this matters (measured: 2 points of AUC).
    """
    return _musx.label_confidence(triad, t0, t1, label)


# ── bar layout by BEAT-INDEX arithmetic (the live app's method) ──────────────
# First harmonia_min version assigned chords to bars by raw time containment
# (which bar's [t0,t1) contains the onset). Louis rejected that chart
# (2026-07-31, This Love): redecode boundaries sit on beats only up to the
# 23.22 ms musx frame grid, so a chord changing ON a bar line lands a few ms
# either side and gets the WRONG bar — every bar-opening chord rendered as the
# tail of the previous bar plus a held "%". The live app never does time
# containment: chords carry a detected-beat INDEX and bar/beat are integer
# arithmetic (scripts/render_youtube_chart.py::chart_to_interactive_inputs).
# This is the minimal version of exactly that.

# Harmonic bar-phase re-anchor thresholds — copied from the live app
# (render_youtube_chart.py, justified there on This Love: 75/120 chords on
# "beat 3", 1/120 on beat 0). If ≥55% of chords agree on ONE non-zero
# beat-in-bar residue and ≤15% sit on beat 0, the chords out-vote the
# tracker's downbeat phase: on a chart, the chord changes ARE the bar lines.
PHASE_CONSENSUS_MIN = 0.55
PHASE_BEAT0_MAX = 0.15


def _phase_correction(residues: list[int], bpb: int) -> int:
    """Signed beat shift to add to the bar-phase offset (0 = no-op)."""
    if not residues or bpb <= 1:
        return 0
    from collections import Counter
    res = Counter(r % bpb for r in residues)
    n = sum(res.values())
    modal, cnt = max(res.items(), key=lambda kv: kv[1])
    if modal == 0 or cnt / n < PHASE_CONSENSUS_MIN or res.get(0, 0) / n > PHASE_BEAT0_MAX:
        return 0
    return modal - bpb if modal > bpb / 2 else modal


def _bar_time(bt: np.ndarray, beat_idx: int, step: float) -> float:
    """Time of (possibly out-of-range) beat index, extrapolating at the edges."""
    n = len(bt)
    if beat_idx < 0:
        return float(bt[0] + beat_idx * step)
    if beat_idx >= n:
        return float(bt[-1] + (beat_idx - (n - 1)) * step)
    return float(bt[beat_idx])


def prompter_chords(segments, triad: np.ndarray) -> list[dict]:
    """Flat {t0,t1,root,q,bass,nc,c} list for the scrolling-prompter view.

    Taken from the beat-grid re-decode BEFORE sections, folding and the
    template re-decode touch anything (Louis, 2026-08-05: the prompter renders
    what the chord detection heard, structure analysis stops mattering here).
    """
    out = []
    for t0, t1, lab in segments:
        ch = to_chord(lab)
        out.append({
            "t0": round(float(t0), 3), "t1": round(float(t1), 3),
            "root": 0 if ch is None else ch["root"],
            "q": "" if ch is None else ch["q"],
            "bass": -1 if ch is None else ch["bass"],
            "nc": ch is None,
            "c": round(_musx.label_confidence(triad, t0, t1, lab), 3),
        })
    return out


def _one_section(bars: list, grid: list, n_bars: int) -> list[dict]:
    """UNE section « A » couvrant tout le morceau, chaque mesure écrite.

    C'est la forme du chart BRUT : celle que `HARMONIA_RAW_CHART=1` produit et
    celle que `analyze_steps` rend au premier yield. Une seule définition, pour
    que les deux ne puissent pas diverger.
    """
    return [{
        "id": "S0", "label": "A", "tag": "", "reps": 1,
        "spans": [[grid[0], grid[n_bars]]],
        "barRanges": [[0, n_bars - 1]],
        "bars": bars,
        "barSpans": [[[grid[b], grid[b + 1]]] for b in range(n_bars)],
    }]


def _force_bar1_sections(segs: list[dict], bar1_bar: int) -> list[dict]:
    """The user's bar-1 mark is a HARD section boundary (Set bar 1 tool).

    Louis, 2026-08-08, on Sam Smith: after realigning, « la section A
    commence quand même en décalé » — the mark moved the bar PHASE but
    detect_sections still placed the first letter boundary wherever it
    liked. So: everything before the marked bar becomes one 'intro'
    section (chords and all); a segment straddling the mark is cut at it;
    letters re-assign in order of first appearance after the mark, so the
    marked bar always READS as A. Named sections keep their names.
    """
    out = []
    for sg in segs:
        if sg["b1"] < bar1_bar:
            continue
        s = dict(sg)
        if s["b0"] < bar1_bar:
            s["b0"] = bar1_bar
        out.append(s)
    # After the mark NOTHING may be called intro — the intro is by definition
    # what precedes bar 1. An intro-named tail MERGES FORWARD into the section
    # that follows it: the mark is the start of A, so the bars between the
    # mark and the detector's own first boundary belong to that first section.
    # (2026-08-09, D-major chart report: the voice entered ONE bar after the
    # mark; the previous rule — rename the tail to a letter — minted a 1-bar
    # "A" and pushed the real 8-bar structure to B/C/D. Renaming was the
    # Sam Smith fix; merging solves both charts.)
    merged: list[dict] = []
    for s in out:
        if merged and str(merged[-1].get("label") or "").lower().startswith("intro"):
            merged[-1] = {**s, "b0": merged[-1]["b0"]}
        else:
            merged.append(s)
    if merged and str(merged[-1].get("label") or "").lower().startswith("intro"):
        merged[-1]["label"] = "A"    # the whole post-mark region: it IS the form
    out = merged
    # letters re-assign in first-appearance order so the marked bar reads as
    # A; other named sections (outro, bridge…) keep their names.
    mapping: dict[str, str] = {}
    for s in out:
        lab = str(s.get("label") or "")
        if len(lab) == 1 and lab.isalpha() and lab.isupper():
            if lab not in mapping:
                mapping[lab] = chr(ord("A") + len(mapping))
            s["label"] = mapping[lab]
    if bar1_bar > 0:
        out.insert(0, {"b0": 0, "b1": bar1_bar - 1, "label": "intro"})
    return out


def _draft_key(bars: list) -> tuple[dict, str]:
    """Clé PROVISOIRE du chart brut, lue sur les accords déjà décodés.

    Un histogramme de hauteurs pondéré par la durée de chaque accord, passé au
    profil de clé (`key_profiles.infer_key`) : quelques millisecondes, aucune
    lecture d'audio. Ce n'est PAS l'analyse harmonique de l'étape 8 — elle
    donne une clé par segment, les couleurs et les alternatives, et elle écrase
    celle-ci quand le raffinement arrive. Le chart brut en a besoin parce que
    l'app orthographie ses accords à partir de `key` (setSpelling) et planterait
    sans.
    """
    chroma = np.zeros(12, dtype=float)
    for bar in bars:
        for c in bar:
            if c["nc"] or c.get("carry"):
                continue
            w = max(0.05, float(c["t1"]) - float(c["t0"]))
            for pc in chord_pcs(c["root"], c["q"]):
                chroma[pc] += w
    if chroma.sum() <= 0:
        return {"tonic": 0, "mode": "major"}, "C major"
    post = infer_key(chroma)
    return {"tonic": int(post.tonic), "mode": post.mode}, post.key_name


def analyze(audio_path, *, title: str = "", file_key: str = "",
            audio_url: str = "", progress=None, bar1_time=None) -> dict:
    """Full thin pipeline for one audio file → final ChartModel dict.

    Thin wrapper over `analyze_steps`: it drains the generator and returns the
    last (refined) model. Same signature and same result as before the
    2026-08-07 split, so every non-streaming caller is untouched.

    `bar1_time` (2026-08-13): without it here, every NON-streaming caller —
    `scripts/rebake_library.py` first among them — silently handed the chart
    back to the beat tracker's phase and wiped the user's "Set bar 1". A
    re-anchor you cannot re-apply is not an anchor.
    """
    model = None
    for _, model in analyze_steps(audio_path, title=title, file_key=file_key,
                                  audio_url=audio_url, progress=progress,
                                  bar1_time=bar1_time):
        pass
    return model


def _run_beats(audio_path, report):
    """Battues + garde-fou de grille → dict, avec `grid` en plus.

    Extrait d'`analyze_steps` le 2026-08-18 pour pouvoir tourner pendant que
    les postérieures musx se calculent dans un autre thread. Le corps est
    inchangé ; seuls les deux `report()` restent à leur place d'origine, donc
    l'écran de chargement voit exactement la même séquence qu'avant.
    """
    # 1 ── beats (hard error if Beat This! fails; librosa is banned)
    report(1, phase="listening")
    bd = _beats.track(audio_path)
    beat_times, downbeats = bd["beats"], bd["downbeats"]
    # Refuse LOUDLY on a grid that cannot carry a 4-beat bar (Louis,
    # 2026-08-05). Everything below indexes bars as `off + b*bpb` over beat
    # indices, so a tracker reporting 2 beats per bar, or only half its bars
    # holding 4, silently yields a chart whose "bars" are not bars — Georgia On
    # My Mind (metre 2, consistency 45%) is the case that exposed it. Raising
    # here reaches the analysing screen through _run_job's error path.
    grid = _beats.check_grid(beat_times, downbeats, Path(audio_path).name)
    # USE THE REPAIRED DOWNBEATS (2026-08-07). Beat This! also marks beat 3 as
    # a bar start on a third of the library; every line below reads the
    # downbeat list — `bpb` is the MEDIAN downbeat gap and `off` the MODAL
    # residue, so a bimodal {4,2} gap histogram makes both wrong (Georgia read
    # bpb=2, Sade/Chiquitita/Jorja/Nina too, and Yam-B read bpb=3 for a 4/4
    # song). check_grid returns the mid-bar marks removed; keeping the raw list
    # here would leave the guard measuring one grid and the chart built on
    # another. Verified: on the 44 songs the old guard accepted this changes
    # (bpb, off) for exactly 0 of them.
    if grid.get("downbeats"):
        downbeats = grid["downbeats"]
    logger.info("beats: grid metre %s, coverage %.0f%% over %d bars "
                "(raw metre %s, raw consistency %.0f%%, kept %.0f%% of the "
                "tracker's downbeats)",
                grid.get("metre"), 100 * grid.get("coverage", 0),
                grid.get("n_bars", 0), grid.get("raw_metre"),
                100 * grid.get("raw_consistency", 0), 100 * grid.get("kept", 0))
    report(2, tempo_bpm=bd["bpm"], phase="decoding",
           # 6 temps par mesure, c'est un 6/8 : à ces tempos (188 temps/min sur
           # l'Alicia Keys) le temps EST la croche. Un 6/4 en pop n'existe
           # pratiquement pas, et écrire « 6/4 » induirait en erreur.
           time_signature=("6/8" if grid.get("metre") == 6
                           else f"{grid.get('metre') or 4}/4"))

    return {**bd, "grid": grid}


def analyze_steps(audio_path, *, title: str = "", file_key: str = "",
                  audio_url: str = "", progress=None, bar1_time=None):
    """Générateur : `("raw", modèle)` puis `("final", modèle)`.

    Louis, 2026-08-07 : « il faut arriver au chart brut le plus rapidement
    possible … le reste (gammes, harmonies, sections) en background une fois
    que le chart est accessible ».

    CE QUI JUSTIFIE LA COUPURE, mesuré (cache chaud, `HARMONIA_SECTIONS=voice`,
    4 morceaux de 68 à 224 s) : la détection de sections pèse **96 à 98 %** du
    temps total (18 s / 18,5 · 27 s / 27,9 · 47 s / 48,7 · 54 s / 55,9). Tout
    ce dont un chart jouable a besoin — battues, posteriors musx, re-décodage,
    disposition en mesures — tient sous 2 s une fois les caches chauds. Il n'y
    avait donc rien à choisir : on rend le chart avant les sections.

    UN SEUL CORPS DE FONCTION, pas deux pipelines. Le brut n'est pas une copie
    allégée : c'est le même code, arrêté plus tôt, qui reprend là où il s'est
    interrompu. `bars` est muté sur place par le repli et par l'analyse
    harmonique ; l'appelant doit donc SÉRIALISER le modèle brut avant de
    redemander le suivant (ce que fait `server._run_job`, de façon synchrone
    dans le même thread).

    CE QUE LE BRUT N'A PAS : les sections (une seule, « A », tout le morceau),
    le repli des répétitions, les couleurs harmoniques, les alternatives
    d'accords, `keySegments`. Sa `key` est l'approximation de `_draft_key`, pas
    le verdict de `harmonic_key`.

    ``progress(stage:int, **fields)`` alimente /api/job :
      2 battues · 3 posteriors · 4 accords décodés · 5 chart brut prêt ·
      6 raffinement terminé.

    ``phase`` (UI refresh 2026-08-08, §5) : le même canal porte la phase que
    l'écran de chargement affiche — "listening" (battues) → "decoding" (les
    posteriors musx, l'attente longue) → "sections" (detect_sections, émis ici
    juste avant l'appel ; jamais sous HARMONIA_RAW_CHART=1) ; "raw" et "done"
    sont posés par server._run_job aux deux yields, avec le modèle brut.
    """
    def report(stage, **kw):
        if progress:
            progress(stage, **kw)

    # LES DEUX ÉTAPES LOURDES EN MÊME TEMPS (2026-08-18). Sur un morceau neuf
    # de 9 min : battues 11,8 s (Beat This!, CPU — mesuré 9x PLUS LENT sur MPS,
    # il y reste), puis CQT + 5 réseaux 8,9 s (les réseaux sur MPS depuis le
    # même jour). Elles ne dépendent pas l'une de l'autre : seul le re-décodage
    # a besoin des deux. Enchaînées, c'est la somme ; lancées ensemble, c'est
    # le max, et les deux occupent des unités différentes.
    #
    # `wait=True` n'est pas décoratif : `musx._InMusxDir` fait un `os.chdir`
    # visible par TOUT le processus. Un thread orphelin qui survit à une erreur
    # de battues laisserait le serveur Flask avec le répertoire courant du
    # clone musx. Pour la même raison le chemin est résolu ici une fois pour
    # toutes — plus rien en dessous ne dépend du répertoire courant.
    from concurrent.futures import ThreadPoolExecutor
    audio_path = Path(audio_path).resolve()
    _pool = ThreadPoolExecutor(max_workers=1)
    _probs_fut = _pool.submit(_musx.frame_posteriors, audio_path)
    try:
        bd = _run_beats(audio_path, report)
        probs = _probs_fut.result()
    finally:
        _pool.shutdown(wait=True)
    triad = probs[0]
    report(3, n_frames=int(triad.shape[0]))
    beat_times, downbeats = bd["beats"], bd["downbeats"]
    grid = bd["grid"]
    if grid.get("downbeats"):
        downbeats = grid["downbeats"]

    # 3 ── beat-grid re-decode: boundaries land exactly on our beats.
    # downbeat_times wired IN (2026-07-31, Louis's This Love report): at
    # phrase turns the frame evidence goes ambiguous for ~a beat and a FLAT
    # penalty is indifferent between last-beat and next-downbeat — two chords
    # landed one beat early (116.6s, 172.1s). Downbeat-graded costs (change on
    # downbeat cheap, elsewhere expensive) resolve the ambiguity the way a
    # lead sheet writes it. Old accuracy study: −0.17 pp (a wash) on label
    # overlap; placement is what the chart lives on.
    _bpb_early = int(round(np.median(np.diff(downbeats)) /
                           np.median(np.diff(beat_times)))) if len(downbeats) >= 3 else 4
    _bpb_early = _bpb_early if 2 <= _bpb_early <= 7 else 4
    # NO granularity restriction (Louis, 2026-08-07: « on ne met plus de
    # restrictions sur la granularité ») — every beat may carry a chord
    # change, at the decoder's own graded cost (downbeat 15 / mid-bar 45 /
    # other beat 100). Covers quarter-bar in 4/4 and third-of-bar in 3/4
    # alike. Kill-switch: HARMONIA_QUARTER_BAR=off restores the 2026-08-01
    # half-bar-only decode.
    _qb_env = os.environ.get("HARMONIA_QUARTER_BAR", "").strip().lower()
    _quarter = None if _qb_env in ("off", "0", "false") else "all"
    if bar1_time is not None and len(beat_times):
        # Set bar 1, follow-up (Louis, 2026-08-09: « l'accord devrait
        # commencer au début de la barre ») : le DÉCODAGE aussi doit préférer
        # les barres de l'utilisateur. Les coûts gradués du re-decode
        # (downbeat 15 / mi-mesure 45 / autre temps 100) pointaient encore
        # sur la phase du tracker, donc le premier changement d'accord
        # s'accrochait à l'ancienne barre — D au temps 2 de la mesure
        # marquée sur gbO7qQliXT8 : la phase d'AFFICHAGE était corrigée,
        # celle des accords non.
        _bt_tmp = np.asarray(beat_times, dtype=float)
        _k1 = int(np.abs(_bt_tmp - float(bar1_time)).argmin())
        downbeats = [float(t) for t in _bt_tmp[_k1 % _bpb_early::_bpb_early]]
        logger.info("pipeline: bar1 mark re-phases the decode downbeats "
                    "(phase %d, %d downbeats)", _k1 % _bpb_early,
                    len(downbeats))
    segments, latency = _musx.redecode(beat_times, probs,
                                       downbeat_times=downbeats,
                                       beats_per_bar=_bpb_early,
                                       quarter_beats=_quarter)
    report(4, draft_chords=[s for _, _, s in segments if s != "N"])

    # (stage 4 removed 2026-08-01 — the audit found the chord-tone KS key was
    # dead code: stage 8's harmonic-key verdict unconditionally overwrites it,
    # and report(2) fired twice with two different key names)

    # 5 ── bar layout by beat-index arithmetic (the live app's method — see
    # the block comment above _phase_correction)
    bt_arr = np.asarray(beat_times, dtype=float)
    # drop a SHORT leading N (leading silence, not music); a long leading N
    # (Stand By Me's bass-riff intro) stays and gets its N.C. bar
    if segments and segments[0][2] == "N" and \
            (segments[0][1] - segments[0][0]) < 2.0 * float(np.median(np.diff(bt_arr))):
        segments = segments[1:]
    step = float(np.median(np.diff(bt_arr)))
    beats_per_bar = int(round(np.median(np.diff(downbeats)) / step)) \
        if len(downbeats) >= 3 else 4
    bpb = beats_per_bar if 2 <= beats_per_bar <= 7 else 4

    # every chord onset → nearest detected beat INDEX (boundaries already
    # land on beats up to the 23 ms musx frame grid, so nearest is exact)
    seg_bidx = [int(np.abs(bt_arr - t0).argmin()) for t0, _, _ in segments]

    # bar phase: the tracker's downbeats vote first (modal beat-index residue)…
    if len(downbeats) >= 3:
        db_idx = [int(np.abs(bt_arr - t).argmin()) for t in downbeats]
        from collections import Counter
        off = Counter(i % bpb for i in db_idx).most_common(1)[0][0]
    else:
        off = 0
    # …then the chords may out-vote them (harmonic re-anchor, live thresholds)
    chord_residues = [bi - off for bi, (_, _, lab) in zip(seg_bidx, segments)
                      if lab != "N"]
    corr = _phase_correction(chord_residues, bpb)
    if corr:
        logger.info("pipeline: harmonic re-anchor fired, shifting bar phase "
                    "by %+d beat(s)", corr)
        off += corr
    _bar1_k = None
    if bar1_time is not None:
        # Set bar 1 (Louis, 2026-08-08): the user's own marked downbeat
        # OUT-VOTES both the tracker's phase and the harmonic re-anchor.
        # Snapped to the nearest tracked beat; the PHASE is taken (k mod bpb)
        # so the music before the mark stays as leading bars — nothing is
        # cut — and the marked bar itself becomes a HARD section boundary
        # further down (_force_bar1_sections), so the first section really
        # starts on the mark (same-night report on Sam Smith: the phase moved
        # but A still began elsewhere).
        _bar1_k = int(np.abs(bt_arr - float(bar1_time)).argmin())
        off = _bar1_k % bpb
        logger.info("pipeline: bar1 override at %.3fs -> beat %d, phase %d",
                    float(bar1_time), _bar1_k, off)

    # bar/beat per segment: pure integer arithmetic, no time containment
    last_beat = int(np.abs(bt_arr - max(t1 for _, t1, _ in segments)).argmin()) \
        if segments else len(bt_arr) - 1
    n_bars = max(1, -((off - last_beat) // bpb))     # ceil((last_beat-off)/bpb)
    # the marked beat's BAR index — (k - off) is a multiple of bpb by
    # construction (off = k mod bpb)
    bar1_bar = None if _bar1_k is None else \
        max(0, min(n_bars - 1, (_bar1_k - off) // bpb))
    bars: list[list[dict]] = [[] for _ in range(n_bars)]
    n_dropped = 0
    for _si, ((t0, t1, lab), bi) in enumerate(zip(segments, seg_bidx)):
        ch = to_chord(lab)
        eff = bi - off
        b = max(0, eff // bpb)                       # pickups clamp into bar 0
        if b >= n_bars:
            continue
        entry = {
            "root": 0 if ch is None else ch["root"],
            "q": "" if ch is None else ch["q"],
            "bass": -1 if ch is None else ch["bass"],
            "nc": ch is None,
            "c": round(_segment_confidence(triad, t0, t1, lab), 3),
            # pickups (eff<0) clamp into bar 0 AND display at beat 0 — the
            # modulo residue is meaningless once the bar is clamped, and the
            # beat now drives the visual quarter position
            "bar": b, "beat": (eff % bpb) if eff >= 0 else 0,
            "pickup": eff < 0,
            "t0": round(float(t0), 3), "t1": round(float(t1), 3),
        }
        # An N.C. tail crossing the barline belongs to the chord's bar: a
        # chart writes the harmony from the barline, not from the moment the
        # band comes in (Louis, 2026-08-09: « l'accord devrait commencer au
        # début de la barre » — D entered on beat 2 of his marked bar 1, the
        # two beats before it being the intro silence's tail). Only when the
        # N.C. STARTED in an earlier bar — a stop inside the bar stays
        # written where it happens, N.C. intro bars stay N.C. (Stand By Me).
        if (not entry["nc"] and entry["beat"] > 0 and not entry["pickup"]
                and _si > 0 and segments[_si - 1][2] == "N"
                and (seg_bidx[_si - 1] - off) // bpb < b):
            entry["beat"] = 0
            entry["t0"] = round(_bar_time(bt_arr, off + b * bpb, step), 3)
        bars[b].append(entry)
    # Q4 (Louis, 2026-08-01): when an N.C. and a chord land on the SAME
    # (bar, beat) slot — leading silence snapped onto a real onset, or a
    # trailing N.C. on the last detected beat — merge by DROPPING the N.C.
    for b in range(n_bars):
        slots = {}
        for c in bars[b]:
            slots.setdefault(c["beat"], []).append(c)
        for beat, group in slots.items():
            if len(group) > 1 and any(c["nc"] for c in group)                     and any(not c["nc"] for c in group):
                for c in [c for c in group if c["nc"]]:
                    bars[b].remove(c)

    # Overflow (> one chord per beat slot) is only possible in bar 0, where
    # pickups are CLAMPED in — an artifact of our own transformation. Splitter
    # lesson (2026-07-31, docs/postmusx_segment_loss.md): a floor/filter may
    # only ever discard artifacts its own cut created, NEVER received content.
    # So: shed clamped pickups (shortest first); if a bar still overflows,
    # something upstream broke its one-onset-per-beat invariant — fail loudly.
    for b in range(n_bars):
        if len(bars[b]) <= bpb:
            continue
        picks = sorted((c for c in bars[b] if c.get("pickup")),
                       key=lambda c: c["t1"] - c["t0"])
        while len(bars[b]) > bpb and picks:
            bars[b].remove(picks.pop(0))
            n_dropped += 1
        if len(bars[b]) > bpb:
            raise RuntimeError(
                f"bar {b} holds {len(bars[b])} chords for {bpb} beat slots "
                "with no pickups to shed — beat-aligned re-decode invariant "
                "broken upstream, refusing to silently drop real chords")
    if n_dropped:
        logger.info("pipeline: shed %d clamped pickup chords from bar 0",
                    n_dropped)

    # Louis's bar rule (2026-07-31): a bar lists ALL chords sounding in it.
    # If a bar's first onset is mid-bar, the chord carried over the bar line
    # is written again at beat 0 ("| C | C G |", never "| C | G |").
    # NO "%" any more (Louis, 2026-07-31 evening): a bar with no onset WRITES
    # its sounding chord too (carry-marked, so repetition counts ignore it) —
    # simile marks come back later as a pure rendering overlay. Killing the
    # empty-bar state removes a whole class of held-bar special cases.
    # …but ONLY a real chord is carried, never an N.C. (Louis, 2026-08-10:
    # « on propage les accords, pas les NC ! »). An N.C. means "I recognised
    # nothing HERE", not "nothing sounds for the next 25 seconds" — carrying
    # it turns one uncertain bar into a wall of silence. Measured before the
    # fix: Stand By Me displayed 93% N.C. against 8% in the raw detection,
    # because a single N.C. at bar 0 was copied over the eleven onset-free
    # bars that followed (docs/known_issues.md, 2026-08-09). A bar left empty
    # renders blank — honest — instead of asserting a silence nobody heard.
    prev = None
    for b in range(n_bars):
        first = bars[b][0] if bars[b] else None
        if first is None and prev is not None and not prev["nc"]:
            bars[b].append({**prev, "carry": True, "bar": b, "beat": 0,
                            "t0": round(_bar_time(bt_arr, off + b * bpb, step), 3),
                            "t1": round(_bar_time(bt_arr, off + (b + 1) * bpb, step), 3)})
        elif first is not None and first["beat"] > 0 and prev is not None \
                and not prev["nc"]:
            bars[b].insert(0, {**prev, "carry": True, "bar": b, "beat": 0,
                               "t0": round(_bar_time(bt_arr, off + b * bpb, step), 3),
                               "t1": first["t0"]})
        if bars[b]:
            prev = bars[b][-1]

    # repetition count n (feeds the UI's "played N times" wording)
    from collections import Counter
    fam = Counter()
    for bar in bars:
        for c in bar:
            if not c["nc"] and not c.get("carry"):
                fam[(c["root"], c["q"][:1])] += 1
    for bar in bars:
        for c in bar:
            c["n"] = 0 if c["nc"] else fam[(c["root"], c["q"][:1])]

    # 7 ── ChartModel sections: detected boundaries + A/B/C labels, UNFOLDED
    # (reps=1 each — the label strip milestone; folding is scoped separately).
    # Bar b's time span = the REAL beat times at its boundary indices
    # (extrapolated by the median beat only off the tracked range).
    grid = [round(_bar_time(bt_arr, off + b * bpb, step), 4)
            for b in range(n_bars + 1)]
    # Captured HERE — before detect_sections and before folding's template
    # re-decode rewrites bar chords — for the scrolling-prompter view.
    prompter = {"chords": prompter_chords(segments, triad)}

    def _model(sections, fold_report, key, key_name, key_segments, *,
               pending=()) -> dict:
        """Le ChartModel, monté une seule fois pour le brut et pour le final."""
        return {
            "file": file_key, "title": title or "Untitled", "video_id": "",
            "audio_url": audio_url,
            "key": key, "keyName": key_name, "keySegments": key_segments,
            "bpb": bpb, "nBars": n_bars,
            "barGrid": grid, "beatTimes": beat_times,
            # La marque de Louis, EN SECONDES, telle qu'il l'a posée : c'est
            # la seule forme qui survit à un re-calcul, puisque la pipeline la
            # re-cale elle-même sur le temps le plus proche. Sans ce champ,
            # toute ré-inférence (rebake compris) rendait la phase au tracker
            # et effaçait le recalage sans rien dire (2026-08-13).
            "bar1": None if bar1_time is None else round(float(bar1_time), 3),
            "form": None,
            "fold": fold_report,
            "sections": sections,
            "prompter": prompter,
            "meta": {"bpm": bd["bpm"], "musx_latency_ms": round(latency * 1000),
                     "n_segments": len(segments), "engine": "harmonia_min",
                     "raw": bool(pending), "pending": list(pending)},
        }

    # ── PREMIER RENDU : LE CHART BRUT ───────────────────────────────────────
    # Tout ce qu'il faut pour AFFICHER et JOUER : la grille de mesures, les
    # accords, l'audio. `meta.pending` dit à l'app ce qui manque encore.
    report(5, n_bars=n_bars)
    _rk, _rkn = _draft_key(bars)
    yield "raw", _model(_one_section(bars, grid, n_bars),
                        {"raw_chart": True}, _rk, _rkn, None,
                        pending=("sections", "key"))

    # ── À PARTIR D'ICI : LE RAFFINEMENT ─────────────────────────────────────
    # HARMONIA_RAW_CHART=1 (Louis, 2026-08-07: « le chart brut barre à
    # barre ») — skip section detection AND both folds: one section, every
    # bar written out with the first-pass decode, exactly the milestone-1
    # shape this file's docstring describes. The caller may re-split the
    # single section (e.g. an intro found by the voice rule). L'étape 8
    # (analyse harmonique) tourne quand même sous ce drapeau, comme avant le
    # découpage — c'est ce que lit scratchpad/raw_app_charts.py.
    if os.environ.get("HARMONIA_RAW_CHART") == "1":
        # rebâtie plutôt que réutilisée : celle du yield ci-dessus a déjà été
        # sérialisée par l'appelant, et `bars` va être muté juste après.
        sections = _one_section(bars, grid, n_bars)
        fold_report = {"raw_chart": True}
    else:
        # phase "sections" fires strictly AFTER the raw yield above and never
        # under HARMONIA_RAW_CHART=1 — the loading screen's second segment.
        report(5, phase="sections")
        from harmonia_min.sections import detect_sections
        from harmonia_min.nnls_features import extract_bothchroma as _ebc
        _arr, _times = _ebc(audio_path)
        sections = []
        # `triad` selects the shipped HARMONIC detector (repetition dictionary
        # on the musx chord posteriors); without it the old chroma detector
        # runs.
        # `audio_path` ne sert qu'au mode `voice` (HARMONIA_SECTIONS=voice),
        # qui a besoin de la piste vocale ; les deux autres l'ignorent.
        if bar1_bar:
            # Louis, 2026-08-08 (Sam Smith report): the mark is the SOURCE OF
            # TRUTH for the start of A — so don't trim boundaries computed on
            # the old origin, RE-RUN the detection on the post-mark region
            # only, its 2-bar blocks anchored on the mark. Everything before
            # the mark is the intro by definition; _force_bar1_sections then
            # guarantees the marked bar reads as a letter (never "intro").
            # form_start=0: the sub-detection anchors its block lattice ON the
            # mark instead of re-deriving a sung start inside the slice —
            # without it the voice detector re-created a post-mark intro
            # (2026-08-09: 1-bar A on the D-major chart).
            sub = list(detect_sections(grid[bar1_bar:], _arr, _times,
                                       bars[bar1_bar:],
                                       triad=triad, audio=audio_path,
                                       form_start=0))
            segs = _force_bar1_sections(
                [{**sg, "b0": sg["b0"] + bar1_bar, "b1": sg["b1"] + bar1_bar}
                 for sg in sub], bar1_bar)
            logger.info("pipeline: bar1 mark at bar %d — sections re-detected "
                        "from the mark (%d sections incl. intro)",
                        bar1_bar, len(segs))
        else:
            segs = list(detect_sections(grid, _arr, _times, bars,
                                        triad=triad, audio=audio_path))
        for si, sg in enumerate(segs):
            b0, b1 = sg["b0"], sg["b1"]
            sections.append({
                "id": f"S{si}", "label": sg["label"], "tag": "", "reps": 1,
                "spans": [[grid[b0], grid[b1 + 1]]],
                "barRanges": [[b0, b1]],
                "bars": bars[b0:b1 + 1],
                "barSpans": [[[grid[b], grid[b + 1]]]
                             for b in range(b0, b1 + 1)],
            })
        # 7b ── REPLI phase 1 (Louis, 2026-07-31): detect each section's
        # internal loop, stack same-position bars across all occurrences of a
        # letter, average their musx posteriors, decode the template a second
        # time and write its chords back on every contributing bar (variants
        # excluded — they keep the first-pass decode). Display folding comes
        # later.
        from harmonia_min.folding import fold_letter_groups
        # HARMONIA_MERGE=cqt (Louis, 2026-08-08 : « les CQT moyennés ça marche
        # très bien ») : les répétitions sont empilées AVANT le modèle, dans
        # le domaine du spectre, et musx re-tourne sur ce spectre moyen — au
        # lieu de moyenner ses probabilités de sortie. Coûte le CQT du morceau
        # (~3 s, mis en cache) plus une inférence par lettre (~2 s).
        # HARMONIA_MERGE_CHECK=<seuil> arme en plus le contrôle d'adhésion.
        # DÉFAUT DEPUIS LE 2026-08-12 : Louis a écouté les deux pages
        # (/reports/merge_cqt.html et /reports/merge_occurrences.html) et a
        # tranché — « validé partout c'est top avec la règle CQT moyenné ».
        # HARMONIA_MERGE=mean|off revient à la moyenne de postérieures.
        _merge = os.environ.get("HARMONIA_MERGE", "cqt").strip().lower()
        _mchk = os.environ.get("HARMONIA_MERGE_CHECK", "").strip()
        _cqt = None
        if _merge == "cqt":
            try:
                _cqt = _musx.song_cqt(audio_path)
            except Exception:
                logger.exception("HARMONIA_MERGE=cqt : CQT indisponible, on "
                                 "retombe sur la moyenne de postérieures")
                _merge = ""
        # `loop="occurrence"` : quand une section n'a pas de boucle interne
        # mais revient N fois à longueur égale, la section elle-même devient
        # la période. Sans ça, le repli refusait 61 lettres sur 162 — le cas
        # le plus fréquent du corpus, et celui que Louis décrivait au départ.
        # HARMONIA_FOLD_LOOP=internal revient au comportement d'avant.
        _loop = os.environ.get("HARMONIA_FOLD_LOOP", "occurrence").strip().lower()
        fold_report = fold_letter_groups(
            sections, bars, grid, probs, bpb, arr=_arr, times=_times,
            combine=("cqt" if _merge == "cqt" else "mean"),
            cqt=_cqt, loop=_loop,
            check_thr=(float(_mchk) if _mchk and _merge == "cqt" else None))
        # repetition counts recomputed on the folded chords
        from collections import Counter as _C2
        fam2 = _C2()
        for bar in bars:
            for c in bar:
                if not c["nc"] and not c.get("carry"):
                    fam2[(c["root"], c["q"][:1])] += 1
        for bar in bars:
            for c in bar:
                c["n"] = 0 if c["nc"] else fam2[(c["root"], c["q"][:1])]

        # 7c ── DISPLAY fold: repeated same-length sections written once ×N,
        # divergent tails as endings (the UI's 1./2. brackets)
        from harmonia_min.folding import minimal_fold
        sections = minimal_fold(sections, bars, grid, fold_report)

    # 8 ── harmonic key analysis (harmonic_key.py: tonic track → mode →
    # colours → feedback). FAILS LOUDLY on any error — no silent fallback.
    # Splitter lesson #2 (2026-07-31, docs/known_issues.md): the old
    # pipeline's Occam gate silently disabled itself when musx_redecode
    # failed, and two "same config" runs differed by 6 chords. A stage that
    # fails must fail where everyone can see it (same doctrine as beats.py).
    flat = [c for bar in bars for c in bar]
    from harmonia_min.harmonic_key import analyze_harmony
    from harmonia_min.nnls_features import extract_bothchroma
    arr, times = extract_bothchroma(audio_path)
    H = analyze_harmony(arr, times, flat)
    for i, c in enumerate(flat):
        c["colour"] = H["colours"][i]
        if i in H["inflections"]:
            c["inflect"] = H["inflections"][i]
        if i in H["challenges"]:
            # flag ONLY: the detector has measured signal (28% of flags land
            # on a real error, minimal_pipeline_log 2026-07-31) but its NNLS
            # chroma-scored alts were refuted at the premise check (median
            # musx posterior 0.037, 0% add a note) — sug now comes from musx
            # below, for every chord (Louis, 2026-08-07: the annotation
            # editor must show the chords musx predicted).
            c["flag"] = H["challenges"][i]["kind"]
    from harmonia_min.span_rescore import musx_suggestions
    musx_suggestions(probs, flat)
    key_segments = H["segments"]
    main = max(H["segments"], key=lambda s: s["t1"] - s["t0"])
    key = {"tonic": main["tonic"], "mode": main["mode"]}
    maj = main["tonic"] if main["mode"] == "major" else (main["tonic"] + 3) % 12
    names = ("C C# D Eb E F F# G G# A Bb B" if maj in (7, 2, 9, 4, 11)
             else "C Db D Eb E F Gb G Ab A Bb B").split()
    key_name = f"{names[main['tonic']]} {main['mode']}"

    model = _model(sections, fold_report, key, key_name, key_segments)
    # ── chord-LM second opinion (OFF by default) ────────────────────────────
    # Attaches PROPOSALS only — never rewrites a chord. Enabled with
    # HARMONIA_CHORD_LM_SUGGEST=1 (same env-flag precedent as
    # HARMONIA_SECTION_MODE / HARMONIA_VOCAB_FOLD).
    #
    # Default OFF on measured evidence, not caution: end-to-end on GuitarSet —
    # the only corpus here with verified GT shipped with its audio — the
    # two-sided rule scored net 0 (11 fixed, 11 broken) and an LM-only gate
    # scored net -18. That corpus cannot settle the question either way (median
    # excerpt 12 bars, no song form, and the LM measures 60.6% at 4-bar context
    # vs 76.7% whole-song), but nothing yet shows a gain, so nothing is applied.
    if os.environ.get("HARMONIA_CHORD_LM_SUGGEST") == "1":
        try:
            from harmonia_min.chord_lm.intervene import propose
            model["lmSuggestions"] = [s.__dict__ for s in propose(model)]
        except Exception as e:                      # never break a chart for this
            logger.warning("chord-LM suggestions skipped: %s: %s",
                           type(e).__name__, e)
            model["lmSuggestions"] = []

    report(6, key_name=key_name,
           final_chords=[s for _, _, s in segments if s != "N"],
           n_sections=len(sections))
    yield "final", model
