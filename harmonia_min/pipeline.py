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
            audio_url: str = "", progress=None) -> dict:
    """Full thin pipeline for one audio file → final ChartModel dict.

    Thin wrapper over `analyze_steps`: it drains the generator and returns the
    last (refined) model. Same signature and same result as before the
    2026-08-07 split, so every non-streaming caller is untouched.
    """
    model = None
    for _, model in analyze_steps(audio_path, title=title, file_key=file_key,
                                  audio_url=audio_url, progress=progress):
        pass
    return model


def analyze_steps(audio_path, *, title: str = "", file_key: str = "",
                  audio_url: str = "", progress=None):
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
    """
    def report(stage, **kw):
        if progress:
            progress(stage, **kw)

    # 1 ── beats (hard error if Beat This! fails; librosa is banned)
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
    report(2, tempo_bpm=bd["bpm"],
           # 6 temps par mesure, c'est un 6/8 : à ces tempos (188 temps/min sur
           # l'Alicia Keys) le temps EST la croche. Un 6/4 en pop n'existe
           # pratiquement pas, et écrire « 6/4 » induirait en erreur.
           time_signature=("6/8" if grid.get("metre") == 6
                           else f"{grid.get('metre') or 4}/4"))

    # 2 ── musx frame posteriors (cache-hit for library songs; ~minutes fresh)
    probs = _musx.frame_posteriors(audio_path)
    triad = probs[0]
    report(3, n_frames=int(triad.shape[0]))

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

    # bar/beat per segment: pure integer arithmetic, no time containment
    last_beat = int(np.abs(bt_arr - max(t1 for _, t1, _ in segments)).argmin()) \
        if segments else len(bt_arr) - 1
    n_bars = max(1, -((off - last_beat) // bpb))     # ceil((last_beat-off)/bpb)
    bars: list[list[dict]] = [[] for _ in range(n_bars)]
    n_dropped = 0
    for (t0, t1, lab), bi in zip(segments, seg_bidx):
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
    prev = None
    for b in range(n_bars):
        first = bars[b][0] if bars[b] else None
        if first is None and prev is not None:
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
        from harmonia_min.sections import detect_sections
        from harmonia_min.nnls_features import extract_bothchroma as _ebc
        _arr, _times = _ebc(audio_path)
        sections = []
        # `triad` selects the shipped HARMONIC detector (repetition dictionary
        # on the musx chord posteriors); without it the old chroma detector
        # runs.
        # `audio_path` ne sert qu'au mode `voice` (HARMONIA_SECTIONS=voice),
        # qui a besoin de la piste vocale ; les deux autres l'ignorent.
        for si, sg in enumerate(detect_sections(grid, _arr, _times, bars,
                                                triad=triad,
                                                audio=audio_path)):
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
        fold_report = fold_letter_groups(sections, bars, grid, probs, bpb,
                                         arr=_arr, times=_times)
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
