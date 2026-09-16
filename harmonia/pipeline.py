"""harmonia/pipeline.py — l'orchestration mince. Audio en entrée → ChartModel en sortie.

Enchaînement des étapes, tel qu'il est aujourd'hui (refactor 2026-09-14,
sprint 10) :

    chart de tête (aperçu, best-effort)
    → battues (harmonia.beats) ∥ postériorités musx (harmonia.musx)
    → re-décodage sur la grille de battues (harmonia.musx.redecode)
    → disposition en mesures (harmonia.bars.layout_bars)
    → clé provisoire + suggestions musx
    → **yield "raw"** — le chart brut, une section, jouable
    → sections (harmonia.sections.detect_sections)
    → repli d'occurrences (harmonia.folding.fold_letter_groups)
    → repli d'affichage (harmonia.folding.minimal_fold)
    → analyse harmonique (harmonia.harmonic_key.analyze_harmony)
    → suggestions musx (re-calculées sur les accords repliés)
    → **yield "final"** — le chart raffiné

`analyze()` reste le raccourci non-streaming qui ne renvoie que le second
modèle. Voir la docstring d'`analyze_steps` pour les mesures qui ont dicté
l'endroit de la coupure entre brut et raffiné.

ChartModel contract (mirrors chart_model.py's docstring + what loadModel in
app_shell.html actually reads): {file,title,video_id,audio_url,key,keyName,
bpb,nBars,barGrid,beatTimes,form,sections:[{id,label,tag,reps,spans,barRanges,
bars,barSpans}]}, Bar=[Chord×0..bpb] (granularity unrestricted 2026-08-07;
typesetting already handles crammed 3-4 chord bars),
Chord={root,q,c,bass,nc,bar,beat,t0,t1}. Optional per-chord fields: `sug`
([{root,q,c}, …], the model's own ranked candidates on THIS chord's span —
travels from the raw yield, see `musx_suggestions` below); `var`
({root,q,bass,c,n}, 2026-09-16, `harmonia.folding._bar_variant`) — the
"optional chord, small above" a folded position writes when ≥1 of its
stacked passes' own first-pass decode heard a real onset (confidence ≥
`folding.VAR_MIN_CONF`) that the averaged consensus smoothed away (Easy On
Me B: "F" written, "D-7" heard as a passing chord on 3 of 6 stacked bars) —
`n` counts how many stacked passes support it. `sug` is the model's ranking
of alternates for the WRITTEN chord's own span (within-bar uncertainty);
`var` is a DIFFERENT chord a minority of OTHER passes actually played at
this position (cross-pass disagreement) — distinct semantics, distinct
fields, on purpose. Rendered by `harmonia/static/screens/chart.js` (the
`ch.var` glyph, small, above the main chord — a field the renderer already
expected since 2026-08-17 for "the other reading of this cell on a folded
chart", never fed by the server until this date).
barSpans[r]=[[t0,t1]] is the playhead's map (server-built, one pass each).

Ce que ce module ne fait PAS : aucun algorithme n'est ici — détection de
battues, décodage d'accords, disposition en mesures, détection de sections,
repli, analyse harmonique vivent chacun dans leur propre module, et ce
fichier ne fait que les appeler dans l'ordre. Il ne fait pas non plus d'I/O
au-delà des caches que CES modules possèdent déjà (`harmonia.cache`) — pas de
chemin construit ici, pas de fichier ouvert directement.
"""
from __future__ import annotations

import logging
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from harmonia import beats as _beats
from harmonia import musx as _musx
from harmonia.bars import layout_bars
from harmonia.key_profiles import infer_key
from harmonia.labels import chord_pcs, to_chord
from harmonia.settings import SETTINGS

logger = logging.getLogger(__name__)

# The triad-plane family table now lives in musx.TRIAD_FAMILY, next to the
# posteriors it indexes and shared with folding. Alias kept for callers.
_TRIAD_FAMILY = _musx.TRIAD_FAMILY


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

    C'est la forme du chart BRUT, celle que `analyze_steps` rend au premier
    yield ("raw") — la seule définition, appelée depuis ce seul point, pour
    qu'aucune deuxième copie ne puisse diverger.
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
    # `head_s=0` : le chart de tête ne sert qu'à l'écran de chargement. Un
    # appelant non-streaming (rebake_library sur 80 morceaux) paierait 2 s par
    # morceau pour un modèle qu'il jette.
    for _, model in analyze_steps(audio_path, title=title, file_key=file_key,
                                  audio_url=audio_url, progress=progress,
                                  bar1_time=bar1_time, head_s=0.0):
        pass
    return model


#: Durée du « chart de tête » : les premières secondes du morceau, analysées
#: seules pour que Louis ait des accords sous les yeux pendant que la passe
#: complète tourne. 45 s parce que c'est la plus courte fenêtre mesurée
#: (2026-08-18, deux morceaux) qui rende EXACTEMENT les mêmes battues que le
#: morceau entier — écart max 0 à 20 ms, même phase de mesure.
HEAD_SECONDS = 45.0

#: En dessous, un chart de tête n'a plus de sens : la passe complète arrive
#: quasiment en même temps.
HEAD_MIN_SONG = 90.0

_HEAD_DIR = Path(tempfile.gettempdir()) / "harmonia_head"


def duration_seconds(audio_path: Path) -> float:
    """Durée en secondes, par ffprobe — ÉCHOUE FORT si ffprobe échoue.

    Remplace `_duree`, qui rendait 0.0 sur toute erreur sans logger (errors
    not to carry #1 : un silence qui aurait pu faire lire « ce morceau dure
    0 s » sans que personne ne le sache). Ici un ffprobe qui échoue lève une
    `RuntimeError` avec son message d'erreur — c'est `analyze_steps` (via
    `head_s`) et son appelant qui décident quoi en faire, jamais ce module en
    silence.

    Ce que cette fonction ne résout PAS : le job serveur qui lance l'analyse
    calcule déjà sa propre durée par ffprobe pour sa fiche, et n'appelle pas
    encore `analyze_steps(duration_s=...)` avec cette valeur — la durée reste
    donc calculée deux fois jusqu'au sprint 12, qui câble ce paramètre depuis
    le job (voir `analyze_steps`).
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(audio_path)],
            capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"ffprobe n'a pas pu tourner sur {audio_path}: "
                           f"{exc}") from exc
    if out.returncode != 0 or not out.stdout.strip():
        raise RuntimeError(f"ffprobe a échoué sur {audio_path}: "
                           f"{out.stderr.strip() or '(pas de sortie)'}")
    try:
        return float(out.stdout.strip())
    except ValueError as exc:
        raise RuntimeError(f"ffprobe a rendu une durée non numérique pour "
                           f"{audio_path}: {out.stdout!r}") from exc


def _head_chart(audio_path: Path, *, title, file_key, audio_url, bar1_time):
    """Le chart des ~45 premières secondes, ou None. JAMAIS fatal.

    Pourquoi ça marche sans rien recalculer différemment : la tête est un
    PRÉFIXE du morceau, donc toutes les secondes qu'elle produit (mesures,
    spans, tête de lecture) sont déjà les bonnes dans le référentiel du
    fichier complet. On lui passe l'audio_url du morceau entier et le chart
    est jouable tel quel.

    Ce qu'il coûte, mesuré le 2026-08-18 : battues 0,9 s + CQT 0,4 s +
    réseaux 0,3 s + re-décodage 0,2 s ≈ 2 s, contre 7 à 12 s pour la passe
    complète.

    CE QU'IL N'EST PAS : la vérité. Les postérieures d'une tranche ne sont pas
    celles du morceau entier — le CNN normalise sur la fenêtre qu'on lui donne
    (InstanceNorm) et le LSTM est bidirectionnel. Mesuré sur une tranche de 8
    mesures : 94 % des accords sont déjà les bons, 6 % changeront. C'est un
    APERÇU, remplacé quelques secondes plus tard par le chart brut.

    Le nom du fichier temporaire est stable (un par morceau) pour que les
    caches (battues, CQT, postérieures) le reconnaissent d'une analyse à
    l'autre.
    """
    try:
        _HEAD_DIR.mkdir(parents=True, exist_ok=True)
        wav = _HEAD_DIR / f"{audio_path.stem}__head{int(HEAD_SECONDS)}.wav"
        if not wav.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-t",
                            str(HEAD_SECONDS), "-i", str(audio_path),
                            "-ac", "1", "-ar", "22050", str(wav)],
                           check=True, timeout=120)
        gen = analyze_steps(wav, title=title, file_key=file_key,
                            audio_url=audio_url, head_s=0.0,
                            bar1_time=(bar1_time if bar1_time is not None
                                       and float(bar1_time) < HEAD_SECONDS
                                       else None))
        try:
            for kind, model in gen:
                if kind == "raw":
                    return model
        finally:
            gen.close()
    except Exception as exc:                      # noqa: BLE001 — best effort
        # Bruyant, jamais silencieux : un aperçu qui manque est une information
        # (souvent le garde-fou de grille sur une intro rubato), pas un détail.
        logger.warning("chart de tête indisponible pour %s : %s",
                       audio_path.name, exc)
    return None


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
                  audio_url: str = "", progress=None, bar1_time=None,
                  head_s: float = HEAD_SECONDS,
                  duration_s: float | None = None):
    """Générateur : `("raw", modèle)` puis `("final", modèle)`.

    `duration_s` : la durée du morceau, si l'appelant l'a déjà (ffprobe coûte
    un aller-retour disque). Le job serveur calcule cette durée pour sa fiche
    mais ne la transmet pas encore ici — câblage prévu au sprint 12 ; en
    attendant, laisser `None` fait recalculer la durée avec
    `duration_seconds()` (qui échoue fort plutôt que de rendre 0).

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
    juste avant l'appel) ; "raw" et "done" sont posés par server._run_job aux
    deux yields, avec le modèle brut.

    Un appelant qui ne veut QUE le chart brut arrête simplement de consommer
    le générateur après le yield "raw" — il n'y a plus de drapeau qui coupe
    la suite à l'intérieur de cette fonction (le kill-switch
    `HARMONIA_RAW_CHART` a été retiré au refactor, décision 4 : un seul chart
    raffiné, pas deux formes à maintenir).
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

    # ── LE CHART DE TÊTE, avant tout le reste (2026-08-18) ──────────────────
    # Louis : « j'aimerais speedup l'arrivée des premiers accords ». Les
    # premières mesures sortent en ~2 s au lieu de 7 à 12. `head_s=0` coupe la
    # récursion : c'est le MÊME générateur qui analyse la tête.
    if head_s and (duration_s if duration_s is not None
                  else duration_seconds(audio_path)) > HEAD_MIN_SONG:
        _head = _head_chart(audio_path, title=title, file_key=file_key,
                            audio_url=audio_url, bar1_time=bar1_time)
        if _head is not None:
            yield "head", _head

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
    # alike. `SETTINGS.quarter_bar` is a constant (refactor decision 4,
    # 2026-09-14) — the 2026-08-01 half-bar-only decode is no longer a
    # runtime kill-switch (`HARMONIA_QUARTER_BAR=off`), it would take a
    # commit to `settings.py`.
    _quarter = "all" if SETTINGS.quarter_bar else None
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
    segments = _musx.redecode(beat_times, probs, downbeat_times=downbeats,
                              beats_per_bar=_bpb_early, quarter_beats=_quarter)
    report(4, draft_chords=[s for _, _, s in segments if s != "N"])

    # (stage 4 removed 2026-08-01 — the audit found the chord-tone KS key was
    # dead code: stage 8's harmonic-key verdict unconditionally overwrites it,
    # and report(2) fired twice with two different key names)

    # 5 ── bar layout by beat-index arithmetic (harmonia/bars.py — extracted
    # 2026-09-14, sprint 6 of the refactor; see that module's docstring for
    # the rules it encodes and their dates). `segments` is REBOUND to what
    # `lay` actually laid out (a short leading N may have been dropped) —
    # `prompter_chords` and the final chord count below must read the same
    # list, not the pre-layout one.
    lay = layout_bars(segments, beat_times, downbeats, triad,
                      bar1_time=bar1_time)
    segments = lay.segments
    bpb, n_bars, bar1_bar, grid, bars = \
        lay.bpb, lay.n_bars, lay.bar1_bar, lay.grid, lay.bars
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
            "meta": {"bpm": bd["bpm"], "n_segments": len(segments),
                     # Littéral gardé tel quel : le rapport d'or compare le
                     # JSON octet par octet, et bascule sur "harmonia" au
                     # sprint 22 (suppression de harmonia_min) — pas avant.
                     "engine": "harmonia_min",
                     "raw": bool(pending), "pending": list(pending)},
        }

    # ── PREMIER RENDU : LE CHART BRUT ───────────────────────────────────────
    # Tout ce qu'il faut pour AFFICHER et JOUER : la grille de mesures, les
    # accords, l'audio. `meta.pending` dit à l'app ce qui manque encore.
    report(5, n_bars=n_bars)
    # LE CLASSEMENT DU MODÈLE VOYAGE AVEC LE CHART BRUT (2026-08-20).
    # Louis, sur Bora Bora : « je veux annotate le F dans la section A, et il
    # me dit *it was written without the model's ranking* ». Le chart en
    # question est un chart BRUT (`meta.raw`) dont les sections ont ensuite été
    # posées à l'outil de soudure : les candidats (`sug`) n'étaient calculés
    # qu'à l'étape finale, donc tout chart qui n'atteint jamais cette étape —
    # raffinement interrompu, découpage validé à la main sur le brut — arrivait
    # dans l'éditeur d'annotation sans rien à proposer.
    # C'est un pooling de postérieures déjà en mémoire, pas une inférence :
    # quelques dizaines de millisecondes, mesurées, avant le premier rendu.
    from harmonia.span_rescore import musx_suggestions
    musx_suggestions(probs, [c for bar in bars for c in bar
                             if not c.get("carry")])
    _rk, _rkn = _draft_key(bars)
    yield "raw", _model(_one_section(bars, grid, n_bars),
                        {"raw_chart": True}, _rk, _rkn, None,
                        pending=("sections", "key"))

    # ── À PARTIR D'ICI : LE RAFFINEMENT ─────────────────────────────────────
    # Un chart BRUT est déjà sorti au yield ci-dessus ; un appelant qui n'en
    # veut que ça arrête de consommer le générateur — il n'y a plus de
    # drapeau qui coupe la suite ici (voir la docstring de la fonction).
    # phase "sections" fires strictly AFTER the raw yield above — the loading
    # screen's second segment.
    report(5, phase="sections")
    from harmonia.sections import detect_sections
    from harmonia.nnls_features import extract_bothchroma as _ebc
    # `_arr`/`_times` no longer feed detection itself (songformer only
    # needs the audio file) — they still feed `fold_letter_groups` below,
    # which stacks occurrences on the raw NNLS substrate.
    _arr, _times = _ebc(audio_path)
    sections = []
    if bar1_bar:
        # Louis, 2026-08-08 (Sam Smith report): the mark is the SOURCE OF
        # TRUTH for the start of A — so don't trim boundaries computed on
        # the old origin, RE-RUN the detection on the post-mark region
        # only, its 2-bar blocks anchored on the mark. Everything before
        # the mark is the intro by definition; _force_bar1_sections then
        # guarantees the marked bar reads as a letter (never "intro").
        # form_start=0: songformer skips its own intro search on this
        # slice, since the mark already says where the form starts —
        # without it, treating this as mid-song re-created a spurious
        # post-mark intro (2026-08-09: 1-bar A on the D-major chart).
        sub = list(detect_sections(grid[bar1_bar:], audio_path,
                                   form_start=0))
        segs = _force_bar1_sections(
            [{**sg, "b0": sg["b0"] + bar1_bar, "b1": sg["b1"] + bar1_bar}
             for sg in sub], bar1_bar)
        logger.info("pipeline: bar1 mark at bar %d — sections re-detected "
                    "from the mark (%d sections incl. intro)",
                    bar1_bar, len(segs))
    else:
        segs = list(detect_sections(grid, audio_path))
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
    from harmonia.folding import fold_letter_groups
    # OÙ ON ADDITIONNE LES RÉPÉTITIONS : après le modèle, sur les
    # postérieures. musx écoute CHAQUE passage séparément et on moyenne ce
    # qu'il a compris (Louis, 2026-08-19, après /plots/cqt_vs_post.html :
    # « je préfère les postérieures empilées c'est + propre » — renverse
    # son arbitrage du 2026-08-12 pour la moyenne de spectres CQT). Seule
    # loi de production depuis le refactor (décision 4, 2026-09-14) : plus
    # d'alternative CQT, de veto mesure-par-mesure ni de transposition à
    # activer par variable d'environnement — voir `harmonia.folding` pour
    # ce qui a été supprimé et pourquoi. `merge_letters` (Louis,
    # 2026-09-14, Sunny Afternoon : « la section A et C sont les mêmes »)
    # reste le seul réglage explicite, recherche de qualité de section en
    # cours, défaut off.
    fold_report = fold_letter_groups(
        sections, bars, grid, probs, bpb, arr=_arr, times=_times,
        merge_letters=SETTINGS.merge_letters)
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
    from harmonia.folding import minimal_fold
    sections = minimal_fold(sections, bars, grid, fold_report)

    # 8 ── harmonic key analysis (harmonic_key.py: tonic track → mode →
    # colours → feedback). FAILS LOUDLY on any error — no silent fallback.
    # Splitter lesson #2 (2026-07-31, docs/known_issues.md): the old
    # pipeline's Occam gate silently disabled itself when musx_redecode
    # failed, and two "same config" runs differed by 6 chords. A stage that
    # fails must fail where everyone can see it (same doctrine as beats.py).
    flat = [c for bar in bars for c in bar]
    from harmonia.harmonic_key import analyze_harmony
    from harmonia.nnls_features import extract_bothchroma
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
    from harmonia.span_rescore import musx_suggestions
    musx_suggestions(probs, flat)
    key_segments = H["segments"]
    main = max(H["segments"], key=lambda s: s["t1"] - s["t0"])
    key = {"tonic": main["tonic"], "mode": main["mode"]}
    maj = main["tonic"] if main["mode"] == "major" else (main["tonic"] + 3) % 12
    names = ("C C# D Eb E F F# G G# A Bb B" if maj in (7, 2, 9, 4, 11)
             else "C Db D Eb E F Gb G Ab A Bb B").split()
    key_name = f"{names[main['tonic']]} {main['mode']}"

    model = _model(sections, fold_report, key, key_name, key_segments)
    # Le chord-LM (second avis, HARMONIA_CHORD_LM_SUGGEST) a été RETIRÉ au
    # refactor (plan 2026-09-14, décision 3) : sur GuitarSet — le seul corpus
    # ici avec de la vérité terrain vérifiée livrée avec son audio — la règle
    # à deux côtés faisait net 0 (11 réparés, 11 cassés) et le filtre LM seul
    # net -18. `lmSuggestions` n'apparaît donc plus jamais dans le modèle (il
    # n'apparaissait déjà par défaut chez personne, le drapeau étant OFF).

    report(6, key_name=key_name,
           final_chords=[s for _, _, s in segments if s != "N"],
           n_sections=len(sections))
    yield "final", model
