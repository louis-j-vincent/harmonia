"""seventh_upgrade.py — DEFAULT-OFF brick: recover the 7th on minor triads.

WHY (measured 2026-07-26, `docs/research_sessions/confusion_and_temporal_support_2026-07-26.md`)
------------------------------------------------------------------------------------------------
On the 7 hand-verified frozen Brick-0 songs the weakest headline metric is
``mirex_sevenths`` (0.5173).  A duration-weighted quality confusion matrix over
the ROOT-CORRECT spans (1218.5 s) shows the loss is not spread out: **61.1% of it
is a 7th that the chart omits**, and the single biggest cell is
``GT min7 -> predicted min`` (129.2 s = 35.6% of all quality loss).  7th
*addition* is only 10.5%, so the error is strongly one-directional.

The shipped path takes root+quality from music-x-lab sampled at ONE instant (the
midpoint of each NNLS root-change segment); this brick is the counterpart that
re-decides ONE bit — "is there a 7th on this minor triad?" — from the NNLS
bothchroma pooled over the chord's **whole span**, i.e. from a genuinely
different (and much longer) temporal support than the label it corrects.

FEATURE
-------
For a chord predicted ``X:min`` over ``[t0, t1)``:

    tre       = L2-normalised, C-framed treble half of the mean NNLS bothchroma
                over [t0, t1)
    b7, M7    = tre[(root+10)%12], tre[(root+11)%12]
    baseline  = mean of the 7 bins that are neither chord tones nor 7ths
    F         = max(b7, M7) / baseline
    F_rel     = F / median(F over every triad-predicted chord IN THIS SONG)

and the chord is upgraded to ``X:min7`` when ``F_rel >= theta``.

The **per-song median normalisation is load-bearing**: an absolute threshold on
``F`` does not transfer between recordings (leave-one-song-out: −0.36pp with an
absolute threshold vs **+3.22pp** with the relative one), because chroma
"noisiness" is a property of the mix, not of the harmony.

MEASURED (7 frozen songs, on/off, same base, `accuracy_score.score_timeline`)
-----------------------------------------------------------------------------
theta=1.5:  sevenths **0.5173 -> 0.5432 (+2.59pp)**, strict +0.75pp,
            root / majmin / partial / bass **unchanged by construction**;
            per song: backing +3.6, every_breath +15.1, close_to_you +0.7,
            georgia −0.1, bein_green / blue_bossa / stand_by_me 0.0.
theta sweep 1.0…2.0: +3.67 … +2.10pp (broad plateau, no spike).
Leave-one-song-out on theta: **+3.22pp**.

WHAT THIS DOES **NOT** SOLVE (CLAUDE.md #4)
-------------------------------------------
* **Nothing on the major side.** The same rule applied to maj triads
  (maj->7 / maj->maj7, choosing by whether b7 or M7 is larger) measured
  **−0.50pp sevenths** and was deliberately dropped: separating maj / maj7 / 7
  is a 3-way decision on two weak, mutually confusable bins, unlike the single
  unambiguous b7 above a minor triad.
* **Nothing for the maj7<->7 confusion** (34.7 s, 9.6% of quality loss).
* **Nothing for root, bass or timing** — the chord's root, its sounding bass and
  its span are untouched, so ``mirex_root``/``bass_root``/``partial_credit``
  cannot move (verified: 0.0000 delta on all three).
* **It is not repertoire-neutral.** It assumes the target vocabulary *wants* the
  7th (jazz/iReal-style charts).  On a strictly triadic pop chart it can only
  hurt: at theta=1.0 stand_by_me loses 2.5pp; at theta>=1.1 it is a no-op there.
  Screen a triadic-repertoire gate before enabling on non-jazz material.
* **Not streaming-safe:** the per-song median needs the whole song's chart.
* The 7th evidence it keys on is weak in absolute terms — on the omitted-7th
  spans the b7 sits at 0.168 (normalised) vs 0.292 where the model already gets
  the 7th right and 0.087 on true triads, i.e. AUC ≈ 0.56 against true triads.
  Some GT 7ths are simply not sounding in the recording; this brick cannot and
  should not recover those.

USAGE (nothing in the pipeline calls this — default OFF, no wiring)
-------------------------------------------------------------------
    from harmonia.models.seventh_upgrade import upgrade_minor_sevenths
    chords, n = upgrade_minor_sevenths(chart.chords, arr, times)   # env-gated

``HARMONIA_SEVENTH_UPGRADE`` = ``"off"`` (default) returns the input list
unchanged (a new list of the same dicts, no mutation); ``"on"`` applies the
rule.  Pass ``enabled=True/False`` to bypass the env check in tests.
"""
from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

ENV_FLAG = "HARMONIA_SEVENTH_UPGRADE"
DEFAULT_THETA = 1.5
_ROLL_TO_C = 9  # nnls_features convention: bothchroma index 0 = A


def _parse(label: str) -> tuple[int | None, str, str]:
    """(root_pc, quality, bass_suffix) for a Harmonia/Harte chord label."""
    from harmonia.eval.accuracy_score import _NAME_TO_PC

    if not label:
        return None, "", ""
    s = label.strip()
    base, _, bass = s.partition("/")
    if ":" in base:
        root_str, quality = base.split(":", 1)
    else:  # colon-less spelling ("Cmaj7")
        if len(base) >= 2 and base[1] in "#b" and base[:2] in _NAME_TO_PC:
            root_str, quality = base[:2], base[2:]
        else:
            root_str, quality = base[:1], base[1:]
    pc = _NAME_TO_PC.get(root_str.strip())
    return pc, quality.strip(), bass


def _relabel_min7(label: str) -> str:
    """``A:min`` -> ``A:min7``; ``A:min/C`` -> ``A:min7/C``; anything else unchanged."""
    base, sep, bass = label.partition("/")
    if ":" in base:
        root, quality = base.split(":", 1)
        if quality.strip() not in ("min", "m", "-"):
            return label
        return f"{root}:min7{sep}{bass}"
    return label


def seventh_evidence(arr: np.ndarray, times: np.ndarray,
                     t0: float, t1: float, root_pc: int, is_minor: bool
                     ) -> float | None:
    """``F`` = max(b7, maj7) energy over the non-chord-tone baseline, on [t0,t1).

    ``arr``/``times`` are the raw NNLS ``bothchroma`` matrix and frame times as
    returned by ``nnls_features.extract_bothchroma`` (index 0 = A, 24 dims).
    Returns None when the span holds no frame.
    """
    m = (times >= t0) & (times < t1)
    if not m.any():
        return None
    seg = np.asarray(arr)[m].mean(0)
    tre = np.roll(seg[12:], _ROLL_TO_C)
    n = float(np.linalg.norm(tre))
    if n < 1e-9:
        return None
    tre = tre / n
    r = int(root_pc) % 12
    b7, maj7 = float(tre[(r + 10) % 12]), float(tre[(r + 11) % 12])
    third = (r + 3) % 12 if is_minor else (r + 4) % 12
    skip = {r, third, (r + 7) % 12, (r + 10) % 12, (r + 11) % 12}
    baseline = float(np.mean([tre[i] for i in range(12) if i not in skip]))
    return max(b7, maj7) / max(baseline, 1e-6)


def upgrade_minor_sevenths(chords: list[dict], arr: np.ndarray, times: np.ndarray,
                           *, theta: float = DEFAULT_THETA,
                           enabled: bool | None = None) -> tuple[list[dict], int]:
    """Upgrade ``X:min`` -> ``X:min7`` where the span's own chroma carries a 7th.

    Returns ``(chords, n_upgraded)``.  OFF (the default) is an exact no-op: the
    same dicts in a new list, nothing mutated, ``n_upgraded == 0``.
    """
    if enabled is None:
        enabled = os.environ.get(ENV_FLAG, "off").strip().lower() in ("1", "on", "true")
    if not enabled or not chords:
        return list(chords), 0

    # 1st pass: the song's own scale for "how much 7th energy is a lot here".
    feats: list[float | None] = []
    for c in chords:
        pc, quality, _bass = _parse(c.get("label", ""))
        if pc is None or quality not in ("maj", "min", "m", "-", ""):
            feats.append(None)
            continue
        feats.append(seventh_evidence(arr, times, float(c["start_s"]),
                                      float(c["end_s"]), pc,
                                      is_minor=quality in ("min", "m", "-")))
    vals = [f for f in feats if f is not None]
    if not vals:
        return list(chords), 0
    med = float(np.median(vals)) or 1.0

    out: list[dict] = []
    n = 0
    for c, f in zip(chords, feats):
        pc, quality, _bass = _parse(c.get("label", ""))
        if (f is not None and quality in ("min", "m", "-") and f / med >= theta):
            d = dict(c)
            d["label"] = _relabel_min7(c["label"])
            if d["label"] != c["label"]:
                n += 1
                out.append(d)
                continue
        out.append(c)
    if n:
        logger.info("seventh_upgrade: %d minor triad(s) -> min7 "
                    "(theta=%.2f, song median F=%.3f)", n, theta, med)
    return out, n
