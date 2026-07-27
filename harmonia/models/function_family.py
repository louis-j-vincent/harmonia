"""function_family.py — DEFAULT-OFF brick: fix the maj↔dom FUNCTION error by
combining music-x-lab's seventh-degree head with a parameter-free harmonic
context (does this root resolve down a perfect fifth?).

WHY THIS EXISTS (measured 2026-07-27, `docs/research_sessions/fn_function_family_2026-07-27.html`)
--------------------------------------------------------------------------------------------------
The 2026-07-27 taxonomy showed segmentation is largely solved musically
(tone-weighted coverage 0.9032) and re-scoped the residue to harmonic FUNCTION
errors.  A full duration-weighted inventory of partial-credit error on the seven
frozen songs (`scratchpad/fn_inventory.py`, `fn_inventory2.py`) splits it as:

    ROOT_WRONG    367.0 s  62.8 % of error   (22.30 % of scoreable time)
    QUALITY_ONLY  156.9 s  26.9 % of error   ( 9.53 % of scoreable time)
    NC_MISMATCH    60.1 s  10.3 % of error   ( 3.65 % of scoreable time)

and inside QUALITY_ONLY (root already right, family wrong) the single largest
named confusion is **maj↔dom at the same root**:

    dom -> maj   49.3 s     maj -> dom   40.1 s     = 89.4 s = 5.44 % of scoreable

It beats every root-level function confusion, including i↔V (58.8 s, 3.58 %), and
unlike i↔V it sits on ALREADY-PURE slices (62 % / 85 % of its time is in
segments that are ≥80 % one GT chord) — so no boundary work can reach it, and
none is needed.  Musically it is the function error: the b7 is exactly what
turns a I/IV into a V, which is why ``accuracy_score._FAMILY`` deliberately does
NOT forgive ``7→maj``.

THE TWO WITNESSES, AND WHY NEITHER WORKS ALONE
-----------------------------------------------
* ACOUSTIC — music-x-lab's network emits a dedicated 4-way seventh-degree
  posterior ``s7`` = {none, add_7, add_b7, add_bb7} at 43.07 fps
  (``musx_redecode.frame_posteriors``; ``complex_chord.SeventhTypes``).  The
  shipped path never reads it, but the vendored Viterbi that produces the ``.lab``
  already consumed it, so it is **not an independent witness**: over 1036 s of
  1070 s of maj/dom chart time it simply agrees with the printed family.  Alone
  it is worth +0.78 pp partial at the degenerate 0.5 tie and **exactly 0.00 pp at
  any real confidence gate**.
* CONTEXT — "the root resolves down a perfect fifth to the next chart chord".
  Premise passes hard: P(GT family = dom | resolves) = 0.785 vs 0.072 when it does
  not, an **odds lift of 47×** (51× on the repaired reference), consistent on 6 of
  7 songs (the 7th, every_breath_you_take, simply contains no dominants at all).
  But as a hard rewrite rule it does NOT pay — the printed family is already right
  86.6 % of the time, so overriding it costs −0.83 pp (maj→dom) and buys +1.13 pp
  (dom→maj) with per-song regressions of −8 to −18 pp.

So the context is used as a **weak tie-breaker on the acoustic log-odds**, never
as a rule:

    z          = log P_s7(add_b7) − log P_s7(none or add_7)   (segment mean)
    z         += w_ctx · (+1 if the root resolves down a fifth else −1)
    P(dom)     = sigmoid(z)
    flip maj→dom when P(dom) ≥ 0.5;  flip dom→maj when P(dom) ≤ 0.5

``w_ctx = 0.5`` gives the context a total swing of 1.0 nat ≈ **2.7:1 odds** —
deliberately far weaker than the measured 47×, because its job is only to break
ties the acoustic head cannot.  That is the whole brick: one knob, no training.

MEASURED (7 frozen songs, both chart configs × both references, w_ctx = 0.5)
-----------------------------------------------------------------------------
Scored with a mask-aware scorer (`scratchpad/fn_brick_eval.py`) against BOTH the
raw frozen reference and the 2026-07-27 repair overlay at level L4 (EXCISE ∪
IMPLIED ∪ WRONG masked out):

    charts segment_source="nnls" (the 0.6419 taxonomy baseline)
        raw  partial 0.6451 -> 0.6692  (+2.41 pp)   strict +2.21   7ths +2.29
        L4   partial 0.6805 -> 0.7077  (+2.72 pp)   strict +2.50   7ths +2.59
    charts segment_source="musx_redecode" (the CURRENT production default)
        raw  partial 0.6698 -> 0.6967  (+2.69 pp)   strict +2.57   7ths +2.57
        L4   partial 0.7114 -> 0.7426  (+3.12 pp)   strict +2.99   7ths +2.99

**Zero per-song regressions in all four combinations.**  Per song (shipped
charts, partial): blue_bossa_backing +10.33, blue_bossa +2.18,
georgia_on_my_mind +1.25, the other four exactly 0.00.  It touches 73.3 s over 34
segments = 3.9 % of chart time.  LOSO on the single knob (pick ``w_ctx`` on six
songs, score the seventh) gives +2.50 pp raw / +3.12 pp L4 — the knob generalises.
The zero-regression plateau is ``w_ctx`` ∈ [0.2, 0.6] on the shipped charts and
[0.3, 0.5] on the nnls charts; 0.5 is the intersection's upper end.

WHAT THIS DOES *NOT* FIX (CLAUDE.md #4)
----------------------------------------
* **Root errors — 62.8 % of all error time — are untouched.**  ``mirex_root`` and
  ``bass_root`` move by exactly +0.00 pp, by construction: the brick only ever
  rewrites the quality token, never the root or the bass.  The root bottleneck is
  upstream (backlog P1) and this brick does not address it.
* **The other same-root family confusions are untouched**: min↔maj (16.8 s),
  min↔dom (26.5 s), hdim↔min (17.5 s).  hdim↔min is the b5 analogue and would need
  the ``triad`` head, not ``s7``; it was not screened here.
* **i↔V (58.8 s) is untouched and is NOT reachable this way.**  78 % of its time is
  in slices that are less than 80 % pure — it is entangled with segmentation, not a
  clean identity error, and the mission hypothesis that it is a pure function error
  is not supported by the inventory.
* **The root-level residue is a long tail**: 163 distinct (GT→predicted) root-wrong
  pairs on 7 songs; 46–56 of them are needed to cover 80 % of the time.  No
  targeted discriminator can address that shape.
* Validated on 7 songs, 4 of which carry the maj/dom confusion at all.  Repo rule
  #5: treat as a hypothesis until a broader jazz corpus confirms it.  In
  particular ``every_breath_you_take`` contains no dominants, so it can only ever
  score 0.00 here — the brick is a no-op on purely diatonic pop.
* The context witness reads the NEXT CHART SYMBOL, so it inherits any root error
  in that symbol.  It also uses one symbol of lookahead, which is fine for an
  offline chart pass but is NOT causal — do not reuse it in a streaming decoder
  without re-deriving it.

INTEGRATION HOOK (default OFF; NOT wired — wiring touches the concurrently-owned
``chord_pipeline_v1.py``, so it is the orchestrator's call)
--------------------------------------------------------------------------------
In ``_infer_nnls24``, immediately AFTER the chart's chord list is final and
BEFORE any rendering::

    from harmonia.models import function_family
    chords = function_family.apply_to_audio(chords, audio_path)

``apply_to_audio`` returns the SAME list object untouched when the brick is
disabled or when the music-x-lab posteriors are unavailable, so the OFF path is
an exact no-op and the ON path degrades to a no-op rather than raising.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)

# music-x-lab frame grid (== musx_redecode.FRAME_DT, restated so this module can
# be reasoned about on its own; the unit test pins them equal — CLAUDE.md #1).
FRAME_DT = 512 / 22050.0            # 0.023219954648526078 s, 43.07 fps

# complex_chord.SeventhTypes order of the vendored `s7` head.
S7_NONE, S7_ADD7, S7_ADDB7, S7_ADDBB7 = 0, 1, 2, 3

ENV_ENABLE = "HARMONIA_FUNCTION_FAMILY"
ENV_W = "HARMONIA_FUNCTION_FAMILY_W"

DEFAULT_W_CTX = 0.5

_NAME_TO_PC = {
    "C": 0, "B#": 0, "C#": 1, "Db": 1, "D": 2, "D#": 3, "Eb": 3,
    "E": 4, "Fb": 4, "E#": 5, "F": 5, "F#": 6, "Gb": 6, "G": 7,
    "G#": 8, "Ab": 8, "A": 9, "A#": 10, "Bb": 10, "B": 11, "Cb": 11,
}

# Quality tokens this brick recognises as the maj family / the dom family.  It is
# deliberately narrower than accuracy_score.chord_family: only tokens whose ONLY
# difference is the seventh degree may be rewritten, so a 6/9, sus or altered
# spelling is left strictly alone.
_MAJ_TOKENS = {"", "maj", "major", "M", "maj7", "M7", "maj9", "maj13"}
_DOM_TOKENS = {"7", "9", "13", "dom", "dom7", "dom9", "dom13"}

# maj token -> the dom token with the same extension, and back.  Anything not in
# these maps is never rewritten.
_MAJ_TO_DOM = {"": "7", "maj": "7", "major": "7", "M": "7",
               "maj7": "7", "M7": "7", "maj9": "9", "maj13": "13"}
_DOM_TO_MAJ = {"7": "maj7", "dom": "maj7", "dom7": "maj7",
               "9": "maj9", "dom9": "maj9", "13": "maj13", "dom13": "maj13"}


def enabled() -> bool:
    """True iff ``HARMONIA_FUNCTION_FAMILY=1``.  Default OFF (CLAUDE.md #6)."""
    return os.environ.get(ENV_ENABLE, "0") == "1"


def weight() -> float:
    """Context weight ``w_ctx`` in nats, from ``HARMONIA_FUNCTION_FAMILY_W``."""
    try:
        return float(os.environ.get(ENV_W, DEFAULT_W_CTX))
    except ValueError:
        logger.warning("%s is not a float — using %.2f", ENV_W, DEFAULT_W_CTX)
        return DEFAULT_W_CTX


# ---------------------------------------------------------------------------
# label helpers
# ---------------------------------------------------------------------------

def _split(label: str) -> tuple[str, str, str]:
    """label -> (root_text, quality, bass_suffix).  Bass is preserved verbatim."""
    bass = ""
    base = label
    if "/" in label:
        base, b = label.split("/", 1)
        bass = "/" + b
    if ":" in base:
        r, q = base.split(":", 1)
    else:
        r, q = base, ""
    return r, q, bass


def _root_pc(label: str) -> int | None:
    if not label or label.strip() in ("N", "X", "NC", "N.C."):
        return None
    r, _q, _b = _split(label)
    return _NAME_TO_PC.get(r)


def _family(quality: str) -> str | None:
    if quality in _MAJ_TOKENS:
        return "maj"
    if quality in _DOM_TOKENS:
        return "dom"
    return None


# ---------------------------------------------------------------------------
# the two witnesses
# ---------------------------------------------------------------------------

def s7_log_odds(s7: np.ndarray, t0: float, t1: float,
                frame_dt: float = FRAME_DT) -> float:
    """log P(add_b7) − log P(none or add_7), averaged over ``[t0, t1)``.

    The mean is taken over the POSTERIOR, then the log — averaging probabilities
    over the segment and taking one log-odds is a far lower-variance estimator of
    "does this segment carry a b7" than a point sample at the midpoint, which is
    what ``musx_bass.root_quality_per_segment`` does.
    """
    n = s7.shape[0]
    a = max(0, min(int(round(t0 / frame_dt)), n - 1))
    b = max(a + 1, min(int(round(t1 / frame_dt)), n))
    p = s7[a:b].mean(axis=0)
    p_dom = float(p[S7_ADDB7]) + 1e-9
    p_maj = float(p[S7_NONE]) + float(p[S7_ADD7]) + 1e-9
    return float(np.log(p_dom / p_maj))


def resolves_down_a_fifth(roots: list[int | None], i: int) -> bool:
    """Does chord ``i``'s root sit a perfect fifth ABOVE the next DIFFERENT root?

    Repeats of the same root are skipped (a chord restated across a bar line is
    the same harmony, not a resolution).  Returns False at the end of the chart
    and wherever a root is missing (N spans).
    """
    mine = roots[i]
    if mine is None:
        return False
    for j in range(i + 1, len(roots)):
        if roots[j] is not None and roots[j] != mine:
            return (mine - roots[j]) % 12 == 7
    return False


# ---------------------------------------------------------------------------
# the brick
# ---------------------------------------------------------------------------

def apply(chords: list[dict], s7: np.ndarray | None, *,
          w_ctx: float | None = None, frame_dt: float = FRAME_DT,
          force: bool = False) -> list[dict]:
    """Rewrite maj↔dom quality tokens in ``chords`` from the two witnesses.

    ``chords`` is the ``ChordChart.chords`` shape: dicts carrying ``label``,
    ``start_s``, ``end_s`` (other keys are copied through untouched).  ``s7`` is
    the ``(n_frame, 4)`` seventh-degree posterior from
    ``musx_redecode.frame_posteriors``.

    Returns **the identical list object** when the brick is disabled, when
    ``s7`` is None, or when nothing would change — so an OFF run is a provable
    no-op, not a rebuilt-but-equal list.  ``force=True`` bypasses the env gate
    (used by the tests and by the offline evaluation harness).
    """
    if not (force or enabled()):
        return chords
    if s7 is None or len(chords) == 0:
        return chords
    w = weight() if w_ctx is None else float(w_ctx)

    roots = [_root_pc(c.get("label", "")) for c in chords]
    out: list[dict] = []
    changed = 0
    for i, c in enumerate(chords):
        label = c.get("label", "")
        r, q, bass = _split(label)
        fam = _family(q) if _root_pc(label) is not None else None
        if fam is None:
            out.append(c)
            continue
        z = s7_log_odds(s7, float(c["start_s"]), float(c["end_s"]), frame_dt)
        z += w * (1.0 if resolves_down_a_fifth(roots, i) else -1.0)
        p_dom = 1.0 / (1.0 + np.exp(-z))
        new_q = None
        if fam == "maj" and p_dom >= 0.5:
            new_q = _MAJ_TO_DOM.get(q)
        elif fam == "dom" and p_dom <= 0.5:
            new_q = _DOM_TO_MAJ.get(q)
        if new_q is None or new_q == q:
            out.append(c)
            continue
        d = dict(c)
        d["label"] = f"{r}:{new_q}{bass}"
        d["function_family_flip"] = f"{q}->{new_q}"
        out.append(d)
        changed += 1
    if changed == 0:
        return chords
    logger.info("function_family: rewrote %d/%d segments (w_ctx=%.2f)",
                changed, len(chords), w)
    return out


def apply_to_audio(chords: list[dict], audio_path: Path | str, **kw) -> list[dict]:
    """``apply`` with the ``s7`` head fetched (cached) for ``audio_path``.

    Degrades to an exact no-op — never raises — if music-x-lab is unavailable,
    matching ``musx_bass``'s cache-and-degrade-gracefully contract.
    """
    if not (kw.get("force") or enabled()):
        return chords
    try:
        from harmonia.models.musx_redecode import frame_posteriors
        s7 = frame_posteriors(audio_path)[2]
    except Exception:                                    # noqa: BLE001
        logger.warning("function_family: music-x-lab posteriors unavailable — "
                       "brick is a no-op for %s", audio_path, exc_info=True)
        return chords
    return apply(chords, s7, **kw)
