"""Lock propagation for harmonia_min — the span-lattice re-score, routed (P2).

``span_rescore`` + ``chord_context_prior`` were in the package but reached by
no route ("unrouted milestone-2 bricks"). This module is the thin seam
between the shell's wire shape and those bricks; the server just calls
:func:`rescore`.

Wire shape, read off the shell (``buildContextRescoreRequest`` / ``applyResp``),
not off a doc:

    request   {chords:[{t0,t1,root,q}], confirms:[{t0,t1,root,q}]}
    response  {key, tempo_bpm, n_changed,
               diff:[{index,start_s,end_s,old_label,new_label,
                      old_confidence,new_confidence}],
               chords:[{t0,t1,root,q}]}

Labels on the wire are Harte, because the shell parses them with
``parseLabel`` (``HARTE_Q``). Confirms are HARD evidence: a confirmed span is
written straight back and never rescored away.

Premise check that justified building this (handoff rule #2, run before a
line of it existed): on the 6 saved charts, one deliberately-wrong lock
propagated to 4 neighbours on This Love and 1 on Norah Jones, while zero
locks changed nothing anywhere — structurally, since ``differential_rescore``
runs the identical lattice twice and only keeps lock-attributable moves.

What this does NOT solve
------------------------
* **The five-family bottleneck.** The lattice decodes QUAL5
  (maj/min/dom/hdim/dim), so a changed span loses its seventh: a rescored
  ``C-7`` comes back ``C:min`` → the shell renders ``Cm``. Unchanged spans are
  never in the diff and keep their full quality, so this only bites where the
  model actually moved — but it is a real downgrade at exactly those spans.
* **Boundaries never move** (by design): a lock cannot split or merge a span,
  only relabel it. Merges still need the old full re-decode path.
* Nothing here re-runs musx: a cold ``musx_probs`` cache degrades to the
  NNLS-24 heads (``span_rescore.compute_acoustic_logp``), which is a weaker
  acoustic backend, silently from the user's point of view.
* No genre conditioning — the pooled prior table is used for every song.
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# QUAL5 index -> Harte quality the shell's HARTE_Q maps back cleanly
Q5_HARTE = {0: "maj", 1: "min", 2: "7", 3: "hdim7", 4: "dim"}
NOTE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def ireal_q_to_q5(q: str | None) -> int:
    """iReal-ish tail -> QUAL5 index. Byte-identical to the mapping
    ``/api/reinfer`` already trusts (harmonia/serving/analysis.py)."""
    if not q:
        return 0
    if q.startswith("-7b5") or q.startswith("h"):
        return 3
    if q.startswith("-") or q.startswith("m"):
        return 1
    if q.startswith("o") or q.startswith("dim"):
        return 4
    if q.startswith("^") or "maj7" in q or "M7" in q:
        return 0
    if any(t in q for t in ("7", "9", "13", "alt")):
        return 2
    return 0


def harte(root_pc: int, q5: int) -> str:
    return f"{NOTE_NAMES[int(root_pc) % 12]}:{Q5_HARTE.get(int(q5), 'maj')}"


def _spans(chords: list[dict]):
    """(spans, displayed) from the wire chords, dropping degenerate ones."""
    spans, displayed, kept = [], [], []
    for c in chords:
        try:
            t0, t1 = float(c["t0"]), float(c["t1"])
            root = int(c["root"]) % 12
        except (KeyError, TypeError, ValueError):
            continue
        if t1 <= t0:
            continue
        spans.append((t0, t1))
        displayed.append((root, ireal_q_to_q5(c.get("q"))))
        kept.append(c)
    return spans, displayed, kept


def _locks_from_confirms(spans, confirms):
    """Map each confirm onto the span it overlaps most (the shell sends the
    same {t0,t1} it displays, but never rely on exact float equality)."""
    locks: list[tuple[int, int] | None] = [None] * len(spans)
    for cf in confirms or []:
        try:
            c0, c1 = float(cf["t0"]), float(cf["t1"])
            token = (int(cf["root"]) % 12, ireal_q_to_q5(cf.get("q")))
        except (KeyError, TypeError, ValueError):
            continue
        best, best_ov = -1, 0.0
        for i, (t0, t1) in enumerate(spans):
            ov = min(t1, c1) - max(t0, c0)
            if ov > best_ov:
                best, best_ov = i, ov
        if best >= 0:
            locks[best] = token
    return locks


def rescore(model: dict, chords: list[dict], confirms: list[dict],
            *, lam: float = 1.0, k: int = 6, delta: float = 0.0,
            margin_gate: float = 0.0) -> dict:
    """Run the differential lattice re-score and build the shell's response."""
    from harmonia_min import span_rescore as sr

    spans, displayed, kept = _spans(chords)
    out = {"key": model.get("keyName") or model.get("key"),
           "tempo_bpm": (model.get("meta") or {}).get("tempo_bpm"),
           "n_changed": 0, "diff": [], "chords": []}
    if len(spans) < 2:
        out["chords"] = [{"t0": c["t0"], "t1": c["t1"], "root": c["root"],
                          "q": c.get("q", "")} for c in kept]
        return out

    locks = _locks_from_confirms(spans, confirms)
    stem = Path(model.get("audio_url", "")).stem or model.get("file", "")
    audio = Path(__file__).resolve().parent.parent / "docs" / "audio" / f"{stem}.m4a"
    ac = sr.compute_acoustic_logp(audio, spans, fallback_audio=audio)
    scorer = sr.load_context_scorer()
    final, changed, _margins = sr.differential_rescore(
        ac["logp"], displayed, locks, scorer,
        lam=lam, K=k, delta=delta, margin_gate=margin_gate)

    diff = []
    result_chords = []
    for i, (c, (t0, t1)) in enumerate(zip(kept, spans)):
        root, q5 = final[i]
        if locks[i] is not None:            # hard evidence, verbatim
            root, q5 = locks[i]
            q_out = c.get("q", "")
            for cf in confirms or []:
                if abs(float(cf["t0"]) - t0) < 0.25:
                    q_out = cf.get("q", q_out)
                    break
        else:
            q_out = c.get("q", "") if not changed[i] else _q_tail(q5)
        result_chords.append({"t0": t0, "t1": t1, "root": root, "q": q_out})
        if changed[i] and locks[i] is None:
            diff.append({
                "index": i, "start_s": t0, "end_s": t1,
                "old_label": harte(*displayed[i]),
                "new_label": harte(root, q5),
                "old_confidence": c.get("c"),
                "new_confidence": 1.0 if locks[i] is not None else None,
            })
    out["diff"] = diff
    out["n_changed"] = len(diff)
    out["chords"] = result_chords
    logger.info("context_rescore: %d span(s), %d lock(s), %d propagated "
                "(backend=%s cache_hit=%s)",
                len(spans), sum(x is not None for x in locks), len(diff),
                ac["backend"], ac["cache_hit"])
    return out


def _q_tail(q5: int) -> str:
    """QUAL5 -> the iReal tail the shell renders (the seventh is gone: see
    the five-family bottleneck in the module docstring)."""
    return {0: "", 1: "-", 2: "7", 3: "-7b5", 4: "o"}.get(int(q5), "")
