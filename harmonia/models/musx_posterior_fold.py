"""musx_posterior_fold.py — DEFAULT-OFF brick: TWO-PASS music-x-lab decode with a
vocabulary fold on **its own frame posteriors**.

WHY THIS EXISTS (and why the previous fold was a dead end)
---------------------------------------------------------
``harmonia.models.periodicity.fold_by_vocabulary`` already averages repeated
observations across the occurrences of each learned section, on the **NNLS-24**
per-beat frame (flag ``HARMONIA_VOCAB_FOLD=1``).  Measured 2026-07-30
(``docs/research_sessions/vocabulary_fold_2026-07-30.md``):

    pure-NNLS config      +15.92 pp partial-credit / +13.72 pp strict
    SHIPPED config        **+0.00 pp on every metric**

The shipped config takes segmentation, root, quality *and* bass from
music-x-lab, so the NNLS observation the fold denoises is barely read.  Folding
**downstream** of music-x-lab is arithmetically pointless.  This module moves the
fold **upstream of music-x-lab's own decoder**:

    pass 1   redecode(beat grid, RAW posteriors)      -> chords
    vocab    chords -> rigid_grid_for -> apply_rigid_grid -> vocab_sections
    fold     average the posteriors across occurrences, ON THE FRAME GRID
    pass 2   redecode(beat grid, FOLDED posteriors)   -> final chords

Louis's design, in his words: *"Once we have gotten the sections CLEANLY, by the
chart path (decode fully first, then build vocab from finished chords, then get
sections with the a prioris, after chords have been detected once by musx) — fold
and pass AGAIN to musx."*

Pass 2 is **cheap**: ``frame_posteriors`` is cached stem-keyed to
``data/cache/musx_probs``, so the second pass pays only the Viterbi decode, not a
second neural inference.  The vendored ``chord_recognition.py`` is never touched
(it only accepts an audio path and emits hard labels — which is exactly why
folding *its output* was never an option).

WHY THE VOCABULARY IS BETTER HERE THAN IN THE NNLS FOLD
------------------------------------------------------
The NNLS fold builds its vocabulary from ``_provisional_chords``: per-beat root
argmax, runs coalesced, **quality blank** (a G and a G7 read as identical).  That
chain is too noisy on This Love for ``rigid_grid_for``/``vocab_sections`` to
agree, so the fold DEFERS there.  Here the vocabulary is built from pass 1's
music-x-lab labels, which carry real qualities and beat-snapped,
latency-corrected onsets — the "chart path" vocabulary.

TWO LOAD-BEARING DECISIONS
--------------------------
1. **The bass stream is NOT folded.**  Each occurrence has its own inversion;
   averaging destroys exactly what the bass target asks for (the project's root
   target *is* the sounding bass pitch class, see
   ``harmonia.data.corpus_schema.sounding_bass_pc``).  Folding the bass cost
   **3.0 pp of bass accuracy** in the NNLS experiment.  Only
   :data:`FOLDED_STREAMS` (``triad, s7, s9, s11, s13``) are folded.
2. **A disagreement guard.**  Folding *amplifies* a grouping error: if pass 1
   mis-groups two different sections, the mean is a *confidently wrong* chord for
   every bar in the group.  A slot is folded only if its occurrences agree — mean
   pairwise cosine of the 73-d triad posterior >= :data:`DEFAULT_AGREE_MIN`.
   Disagreement is evidence the spans are not the same music.

WHAT THIS DOES **NOT** SOLVE (CLAUDE.md rule #4)
------------------------------------------------
* It inherits the bar grid wholesale.  ``docs/known_issues.md`` OPEN #1: the bar
  is metrically 2x wrong on ~1/3 of the corpus and OPEN #2 says
  ``rigid_grid_for`` never actually defers, so a wrong octave folds genuinely
  different music together.  The guard is the only defence and it is per-slot,
  not per-item.
* It is an **unweighted mean** over occurrences.  A masked or badly-mixed pass
  counts as much as a clean one; median / trimmed mean / reliability weighting is
  the obvious next lever and is not implemented.
* The guard gates on the **triad** stream only, and applies that one decision to
  all folded streams — deliberately, so the streams stay mutually consistent, but
  it means a slot whose triads agree and whose 7ths disagree still folds.
* Pass 2 re-selects the latency from the folded posteriors.  Nothing forces it to
  agree with pass 1's choice.
* No third pass.  The vocabulary is never re-derived from pass 2's chords.

Flag: ``HARMONIA_MUSX_FOLD=1``.  Default OFF ⇒ the wiring in
``harmonia.stages.chord_head`` is byte-identical to the one-pass shipped decode.
"""
from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

from harmonia.models.musx_redecode import FRAME_DT  # noqa: E402  (re-exported)

__all__ = ["enabled", "chords_from_labels", "vocab_from_chords",
           "fold_frame_posteriors", "two_pass_redecode",
           "FRAME_DT", "FOLDED_STREAMS", "DEFAULT_AGREE_MIN"]

FOLD_ENV = "HARMONIA_MUSX_FOLD"

#: Which of ``frame_posteriors``' six streams get folded.
#: ``[0]=triad(73) [1]=bass(13) [2]=s7(4) [3]=s9(4) [4]=s11(3) [5]=s13(3)``.
#: **1 (bass) is deliberately absent** — see the module docstring.
FOLDED_STREAMS: tuple[int, ...] = (0, 2, 3, 4, 5)

#: Minimum mean-pairwise-cosine agreement (on the triad stream) for a slot to be
#: folded.  **CALIBRATED end-to-end**, not read off a histogram — and the two
#: disagreed, which is the interesting part (2026-07-30,
#: ``scratchpad/musx_fold_calibrate.py`` + ``scratchpad/musx_fold_ab.py``):
#:
#: * The per-slot statistic over 18 052 multi-occurrence slots (7 frozen songs +
#:   This Love) is bimodal — median 0.904, a distinct low mode below 0.05 — and
#:   its histogram TROUGH is at **0.25–0.30**.
#: * But the trough is NOT the best operating point.  On the 7-song Brick-0
#:   benchmark, shipped config: ``0.30`` gives **+1.20 pp** partial-credit while
#:   ``0.60`` gives **+2.12 pp**.  0.60 wins on every metric.
#:
#: Why: the statistic scales with the number of occurrences.  Three occurrences
#: of which one is mis-decoded score ~1/3 and are blocked at 0.60 (correctly —
#: with n=3 one bad pass is a third of the evidence), while eight occurrences
#: with one bad pass score ~0.75 and still fold.  The guard is therefore strict
#: when there is little evidence and tolerant when there is a lot.
#:
#: Firing rate at 0.60 (fraction of multi-occurrence slots refused): stand_by_me
#: 2 %, backing-track 2 %, bein_green 16 %, every_breath 24 %, close_to_you 36 %,
#: blue_bossa 37 %, georgia 64 %.  Env override:
#: ``HARMONIA_MUSX_FOLD_AGREE``.  n=7 songs, i.e. a hypothesis (CLAUDE.md #5).
DEFAULT_AGREE_MIN = 0.60

# music-x-lab Harte quality -> iReal token, i.e. the composition of
# ``musx_bass._MUSX_Q_TO_SEV`` with ``scripts/render_youtube_chart.py``'s
# ``_QUALITY_TO_IREAL``.  Composed here rather than imported because the second
# table lives in a script, not an importable module.  ``section_vocab`` /
# ``local_key.chord_pcs`` read iReal spellings ("-" minor, "^7" maj7, "o" dim,
# "h7" half-dim), NOT Harte ones — passing "maj7" through would be read as
# min-maj7 by ``chord_pcs`` and silently corrupt the SSM.
_SEV_TO_IREAL = {
    "maj": "", "min": "-", "dim": "o", "aug": "+",
    "sus2": "sus2", "sus4": "sus", "7sus4": "7sus",
    "maj7": "^7", "min7": "-7", "7": "7", "dim7": "o7", "hdim7": "h7",
    "9": "9", "min9": "-9", "maj9": "^9", "dom11": "11", "dom13": "13",
}


def enabled() -> bool:
    """Is the two-pass posterior fold switched on?  Default **ON** since
    2026-07-30; ``HARMONIA_MUSX_FOLD=0`` is the kill switch.

    Flipped on the strength of Brick-0, all 7 verified songs, SHIPPED config,
    independently reproduced twice (agent + orchestrator, identical to 4 dp):

        partial  0.695 -> 0.716   (+2.12 pp)
        strict   0.518 -> 0.532   (+1.43 pp)
        root     0.751 -> 0.768   (+1.69 pp)
        majmin   0.732 -> 0.754   (+2.19 pp)
        bass     0.758 -> 0.774   (+1.58 pp)

    Leave-one-song-out is still positive (+1.87 partial / +1.17 strict), which is
    the honest generalisation estimate rather than the in-sample figure, and no
    song regresses on partial credit.  Per-metric per-song regressions DO exist
    and are not hidden: close_to_you strict -1.1 pp, blue_bossa_backing sevenths
    -3.1 pp.  n=7 songs is a hypothesis, not a law (CLAUDE.md rule #5).

    Follows the ``HARMONIA_MUSX_REDECODE`` / ``HARMONIA_VOCAB_FOLD`` precedent.
    """
    return os.environ.get(FOLD_ENV, "1").strip().lower() in (
        "1", "on", "true", "yes")


def agree_min_from_env() -> float:
    """The guard threshold, env-switchable (``HARMONIA_MUSX_FOLD_AGREE``).

    Exists so the operating point can be swept without a code change — the
    threshold is a single calibrated number on n=8 songs, i.e. a hypothesis
    (CLAUDE.md rule #5).  ``0`` disables the guard entirely (fold every
    multi-occurrence slot), ``1`` blocks everything but exact agreement.
    """
    try:
        return float(os.environ.get("HARMONIA_MUSX_FOLD_AGREE",
                                    DEFAULT_AGREE_MIN))
    except ValueError:
        return DEFAULT_AGREE_MIN


# ── pass 1's chords, in the shape the chart path reads ───────────────────────

def chords_from_labels(
    labels: "list[tuple[float, float, str]]",
) -> list[dict]:
    """music-x-lab ``.lab``-style spans -> chord dicts for the chart path.

    ``rigid_grid_for`` wants ``root``/``q``/``t0``/``t1`` and skips anything with
    ``nc``; ``section_vocab.build_slots`` wants ``q == "N"`` or ``nc`` to mean
    *the decoder does not know here* (it must inherit NOTHING — forward-filling a
    no-chord slot from the previous bar is what minted a phantom section on This
    Love).  So an ``N``/``X`` span is emitted explicitly rather than dropped.
    """
    from harmonia.models.musx_bass import _parse_root, quality_sev_of_label

    out: list[dict] = []
    for t0, t1, lab in labels:
        sev = quality_sev_of_label(lab)
        root = None
        if sev is not None:
            root = _parse_root(str(lab).split("/", 1)[0].split(":", 1)[0])
        if sev is None or root is None:
            out.append({"root": -1, "q": "N", "nc": True,
                        "t0": float(t0), "t1": float(t1)})
            continue
        out.append({"root": int(root), "q": _SEV_TO_IREAL.get(sev, ""),
                    "nc": False, "t0": float(t0), "t1": float(t1)})
    return out


def vocab_from_chords(
    chords: list[dict], *, tonic_pc: int = 0, beats_per_bar: int = 4,
) -> "tuple[list[dict], list[float], int] | None":
    """``(sections, bar_bounds_sec, n_bars)``, or ``None`` to decline.

    The chart path, unchanged: ``rigid_grid_for`` (period+phase from the chord
    onset TIMES) -> ``apply_rigid_grid`` (re-bin onto rigid bars, dropping the
    pre-grid pickup) -> ``vocab_sections`` (learn the repeating items and cover
    the song).  Declines whenever any link has nothing confident to say — too few
    chords, no grid, < 8 bars, a single-loop or through-composed song.  A no-op is
    always safe; a wrong grouping averages genuinely different music.

    ``tonic_pc=0`` is harmless even off C: the tonic only makes the vocabulary's
    root indices key-relative, and neither the chord-tone SSM nor the fold
    depends on that offset.
    """
    from harmonia.models.rigid_grid import apply_rigid_grid, rigid_grid_for
    from harmonia.models.section_vocab import vocab_sections

    real = [c for c in chords if not c.get("nc")]
    if len(real) < 8:
        return None
    grid = rigid_grid_for(chords, tonic_pc=tonic_pc, beats_per_bar=beats_per_bar)
    if grid is None:
        return None
    regridded, n_bars = apply_rigid_grid(
        chords, grid, beats_per_bar=beats_per_bar, drop_before_grid=True)
    if n_bars < 8:
        return None
    bars: list[list[dict]] = [[] for _ in range(n_bars)]
    for c in regridded:
        b = int(c.get("bar", 0))
        if 0 <= b < n_bars:
            bars[b].append(c)
    sections = vocab_sections(bars, n_bars, tonic_pc=tonic_pc, bpb=beats_per_bar)
    if not sections:
        return None
    return sections, list(grid), n_bars


# ── the fold, on the FRAME grid ──────────────────────────────────────────────

def _bar_frame_index(
    bar_bounds_sec: "list[float] | np.ndarray", n_frame: int,
) -> "tuple[list[np.ndarray], int]":
    """``(frames_of_bar, n_bars)`` — which frame indices each bar owns, by TIME.

    Frame ``f`` covers ``[f * FRAME_DT, (f+1) * FRAME_DT)``, so bar ``b`` owns
    ``round(bounds[b] / FRAME_DT) .. round(bounds[b+1] / FRAME_DT)``.  Reads no
    ``beat`` label from anywhere, for the same reason
    ``periodicity._bar_beat_index`` does not: ``rigid_grid`` backs every bar edge
    off 0.15 bar and never compensates it in the ``beat`` it writes
    (``known_issues.md`` OPEN #3).  That backoff is a **constant** phase shift —
    it moves which frames every bar owns identically, so slot alignment between
    occurrences is untouched.

    Frames before the first edge (a pickup) and after the last belong to no bar.
    This is the *easy* half of what the beat version had to solve: frames are
    uniform, so no bar can "miss" one and there is no ragged-count problem beyond
    a +-1 rounding difference in the last slot of a bar.
    """
    bounds = np.asarray(sorted(float(x) for x in bar_bounds_sec), dtype=np.float64)
    n_bars = max(len(bounds) - 1, 0)
    if n_bars == 0 or n_frame <= 0:
        return [], 0
    edge = np.clip(np.round(bounds / FRAME_DT).astype(int), 0, int(n_frame))
    return [np.arange(edge[b], edge[b + 1]) for b in range(n_bars)], n_bars


def _slot_map(
    sections: "list[dict]", frames_of_bar: "list[np.ndarray]", n_bars: int,
) -> "dict[tuple[str, int, int], list[int]]":
    """``(item label, bar WITHIN the item, ordinal frame within that bar)`` ->
    absolute frame indices.  Only COMPLETE occurrences fold: a section shorter
    than its own item (``vocab_sections`` reports ``reps = max(1, ...)``, so a
    1-bar remainder of a 4-bar item still says ``reps=1``) contributes nothing and
    its frames pass through raw rather than being folded against a truncated
    pattern.
    """
    slots: dict[tuple[str, int, int], list[int]] = {}
    for sec in sections or []:
        try:
            b0, b1 = int(sec["bar0"]), int(sec["bar1"])
            d = int(sec.get("d_bars") or 0)
        except (KeyError, TypeError, ValueError):
            continue
        if d < 1:
            continue
        b0, b1 = max(b0, 0), min(b1, n_bars)
        label = str(sec.get("label", ""))
        for r in range((b1 - b0) // d):
            for k in range(d):
                for j, f in enumerate(frames_of_bar[b0 + r * d + k]):
                    slots.setdefault((label, k, j), []).append(int(f))
    return slots


def _agreement(rows: np.ndarray) -> np.ndarray:
    """Mean pairwise cosine within each group of rows.  ``rows`` is
    ``(n_slot, n_occ, k)``; returns ``(n_slot,)``.

    Closed form: with unit-normalised rows ``u_i``,
    ``sum_{i != j} cos(u_i, u_j) = ||sum_i u_i||^2 - n``, so the mean pairwise
    cosine is ``(||sum u||^2 - n) / (n (n-1))``.  Three occurrences, two of which
    are the same one-hot chord and one a different one, score exactly 1/3.
    """
    n = rows.shape[1]
    if n < 2:
        return np.ones(rows.shape[0])
    u = rows.astype(np.float64)
    nrm = np.linalg.norm(u, axis=2, keepdims=True)
    u = u / np.clip(nrm, 1e-12, None)
    s = u.sum(1)
    return ((s * s).sum(1) - n) / (n * (n - 1))


def fold_frame_posteriors(
    probs: "list[np.ndarray]",
    sections: "list[dict] | None",
    bar_bounds_sec: "list[float] | np.ndarray",
    *,
    agree_min: float = DEFAULT_AGREE_MIN,
    report: bool = False,
) -> "tuple[list[np.ndarray], dict]":
    """Average music-x-lab's frame posteriors across the occurrences of each
    learned vocabulary item.  Returns ``(folded_probs, stats)``.

    ``probs`` is exactly what ``musx_redecode.frame_posteriors`` returns:
    ``[triad(73), bass(13), s7(4), s9(4), s11(3), s13(3)]``, each
    ``(n_frame, k)`` on the 23.22 ms grid.  Copies are returned; the inputs are
    never mutated, and every value is read from the **immutable input**, so
    overlapping sections cannot compound through an already-folded value (the
    order-independence rule ``user_constraints.pool_beat_evidence`` learned the
    hard way).

    A slot is keyed ``(item label, bar within the item, ordinal frame within that
    bar)``.  Per-BAR keying (rather than offset-from-occurrence-start) costs
    nothing on a rigid grid — ``rigid_grid._build_bounds`` emits exactly uniform
    bars — and keeps a +-1-frame rounding difference contained inside the one bar
    where it happened instead of shifting every later slot of that occurrence.

    Folded: :data:`FOLDED_STREAMS`.  **Never the bass.**  Each folded row is
    renormalised to sum to 1 (a mean of distributions already does, but the
    vendored decoder takes ``log`` of these values, so the invariant is pinned
    rather than assumed).

    ``agree_min`` is the disagreement guard (module docstring).  ``report=True``
    additionally returns the per-slot agreement values under ``"agree"``, for
    calibration.

    Deferrals — inputs returned unchanged: no sections, no bars, an item with a
    single occurrence, empty posteriors.
    """
    folded = [np.array(p, copy=True) for p in probs]
    stats: dict = {"n_frame": int(probs[0].shape[0]) if probs else 0,
                   "n_slots": 0, "n_slots_folded": 0, "n_slots_guarded": 0,
                   "frames_folded": 0, "agree_min": float(agree_min)}
    if not probs or stats["n_frame"] == 0 or not sections:
        return folded, stats

    frames_of_bar, n_bars = _bar_frame_index(bar_bounds_sec, stats["n_frame"])
    if n_bars == 0:
        return folded, stats
    slots = _slot_map(sections, frames_of_bar, n_bars)
    stats["n_slots"] = sum(1 for v in slots.values() if len(v) > 1)
    if not slots:
        return folded, stats

    # Group the slots by occurrence count so the whole fold is a handful of
    # vectorised operations instead of a per-slot Python loop.
    by_n: dict[int, list[list[int]]] = {}
    for idxs in slots.values():
        if len(idxs) > 1:
            by_n.setdefault(len(idxs), []).append(idxs)

    agree_all: list[np.ndarray] = []
    for n_occ, group in sorted(by_n.items()):
        idx = np.asarray(group, dtype=int)                    # (n_slot, n_occ)
        agree = _agreement(probs[0][idx])
        agree_all.append(agree)
        keep = agree >= float(agree_min)
        stats["n_slots_guarded"] += int((~keep).sum())
        stats["n_slots_folded"] += int(keep.sum())
        stats["frames_folded"] += int(keep.sum()) * n_occ
        if not keep.any():
            continue
        sel = idx[keep]                                       # (m, n_occ)
        for s in FOLDED_STREAMS:
            if s >= len(probs):
                continue
            mean = np.asarray(probs[s], dtype=np.float64)[sel].mean(1)
            mean /= np.clip(mean.sum(1, keepdims=True), 1e-12, None)
            folded[s][sel] = mean[:, None, :].astype(folded[s].dtype)
    if report and agree_all:
        stats["agree"] = np.concatenate(agree_all)
    return folded, stats


# ── the two-pass decode ──────────────────────────────────────────────────────

def two_pass_redecode(
    audio_path, beat_times, *,
    penalty: "float | None" = None,
    latency_grid=None,
    downbeat_times=None,
    beats_per_bar: int = 4,
    tonic_pc: int = 0,
    agree_min: "float | None" = None,
    probs: "list[np.ndarray] | None" = None,
) -> "tuple[list[tuple[float, float, str]], float, dict]":
    """``(labels, latency, stats)`` — decode, learn the vocabulary, fold, decode
    again.

    Pass 1 is *bit-for-bit* the shipped one-pass call
    (``musx_redecode.redecode`` on the raw posteriors with the same penalty and
    latency grid), so when the vocabulary declines or the guard blocks every slot
    this returns pass 1's own answer and the caller sees no change at all.

    The posteriors are loaded once (cached, stem-keyed), so pass 2 costs one extra
    Viterbi decode per candidate latency — no second neural inference.
    """
    from harmonia.models import musx_redecode as mxr

    if agree_min is None:
        agree_min = agree_min_from_env()
    kw = dict(downbeat_times=downbeat_times)
    if penalty is not None:
        kw["penalty"] = float(penalty)
    if latency_grid is not None:
        kw["latency_grid"] = latency_grid

    if probs is None:
        probs = mxr.frame_posteriors(audio_path)
    lab1, lat1 = mxr.redecode(beat_times, probs, **kw)
    stats: dict = {"pass1_segments": len(lab1), "pass1_latency": float(lat1),
                   "deferred": True, "reason": "", "form": "", "n_items": 0,
                   "n_bars": 0, "bar_s": 0.0}

    # Anything from here on is an ENHANCEMENT of a decode we already have.  A
    # failure must therefore cost the fold, never the re-decode: without this the
    # caller's single except-block would swallow pass 1 too and fall all the way
    # back to the NNLS root-change segmentation + raw .lab (CLAUDE.md #6 — a
    # component swap must not silently change more than its target).
    try:
        v = vocab_from_chords(chords_from_labels(lab1), tonic_pc=tonic_pc,
                              beats_per_bar=beats_per_bar)
    except Exception as exc:                                       # noqa: BLE001
        logger.warning("musx_posterior_fold: vocabulary failed (%s); keeping "
                       "pass-1 labels", exc)
        stats["reason"] = f"vocabulary raised {type(exc).__name__}"
        return lab1, lat1, stats
    if v is None:
        stats["reason"] = "no confident grid/vocabulary"
        return lab1, lat1, stats
    sections, grid, n_bars = v

    try:
        folded, fstats = fold_frame_posteriors(
            probs, sections, grid, agree_min=agree_min)
    except Exception as exc:                                       # noqa: BLE001
        logger.warning("musx_posterior_fold: fold failed (%s); keeping pass-1 "
                       "labels", exc)
        stats["reason"] = f"fold raised {type(exc).__name__}"
        return lab1, lat1, stats
    stats.update(fstats)
    if fstats["n_slots_folded"] == 0:
        stats["deferred"] = True
        stats["reason"] = "disagreement guard blocked every slot"
        return lab1, lat1, stats

    from harmonia.models.section_vocab import form_string
    stats.update({
        "deferred": False,
        "form": form_string(sections),
        "n_items": len({s["label"] for s in sections}),
        "n_bars": int(n_bars),
        "bar_s": float(np.median(np.diff(np.asarray(grid, dtype=float)))),
    })

    try:
        lab2, lat2 = mxr.redecode(beat_times, folded, **kw)
    except Exception as exc:                                       # noqa: BLE001
        logger.warning("musx_posterior_fold: pass 2 failed (%s); keeping pass-1 "
                       "labels", exc)
        stats.update({"deferred": True,
                      "reason": f"pass 2 raised {type(exc).__name__}"})
        return lab1, lat1, stats
    if not lab2:                       # a degenerate decode is worse than none
        stats.update({"deferred": True, "reason": "pass 2 returned no segments"})
        return lab1, lat1, stats
    stats.update({"pass2_segments": len(lab2), "pass2_latency": float(lat2),
                  "changed_segments": int(lab1 != lab2)})
    logger.info(
        "musx_posterior_fold: %s (%d items, %d bars @ %.3fs) — folded %d/%d "
        "slots (%d guarded, %.0f%% fired), %d/%d frames; segments %d -> %d, "
        "latency %.0f -> %.0f ms",
        stats["form"], stats["n_items"], n_bars, stats["bar_s"],
        fstats["n_slots_folded"], fstats["n_slots"], fstats["n_slots_guarded"],
        100.0 * fstats["n_slots_guarded"] / max(fstats["n_slots"], 1),
        fstats["frames_folded"], fstats["n_frame"], len(lab1), len(lab2),
        lat1 * 1000, lat2 * 1000)
    return lab2, lat2, stats
