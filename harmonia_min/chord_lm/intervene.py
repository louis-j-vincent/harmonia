"""harmonia_min/chord_lm/intervene.py — let the chord LM speak where we are unsure.

The decision, in one line:

    propose the LM's chord at a half-bar when the LM is confident AND the
    pipeline is not, and never otherwise.

Both halves are required. Measured 2026-08-01: on a chart that is ALREADY right,
applying the LM wherever it is confident (p>=0.9) breaks 368 more chords than it
fixes — it disagrees with 20.5% of correct slots and that false-alarm floor does
not go away. A one-sided gate is a regression generator.

The pipeline's side needs `c` to be *meaningful*, not calibrated. Measured on
GuitarSet (verified GT shipped with the audio): AUC 0.8175 at separating its own
right roots from its wrong ones, which clears the AUC>=0.70 bar that makes a
two-sided rule worth more than an LM-only threshold.

Deliberately NOT an auto-apply. `propose()` returns suggestions with a rank-3
shortlist; the caller decides whether to surface them. The truth is in the LM's
top 3 for 92.8% of slots but its top 1 for only 79.5%, so a shortlist recovers
most of what a silent override throws away — and nothing regresses without a
human click.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .from_app import app_chart_to_grid
from .grid import expand_repeats
from .vocab import PC_NAMES, REP, VOCAB_SIZE, is_chord, split_chord, token_name

# Tuned on GuitarSet with real pipeline confidence + verified GT
# (scripts/chord_lm_prod_eval.py). Two-sided: the LM must be at least this sure,
# and the pipeline at most this sure, before we say anything at all.
LM_MIN = float(os.environ.get("HARMONIA_LM_MIN", "0.80"))
PIPE_MAX = float(os.environ.get("HARMONIA_LM_PIPE_MAX", "0.60"))
TOP_K = 3

_MODEL = None


def model_path() -> Path:
    return Path(os.environ.get("HARMONIA_CHORD_LM",
                               "data/models/chord_lm_rope_masked.pt"))


def load_model(device: str = "cpu"):
    """Cached load. Returns None if the checkpoint is absent — the LM is an
    optional refinement and its absence must never break a chart."""
    global _MODEL
    if _MODEL is None:
        p = model_path()
        if not p.exists():
            return None
        import torch  # noqa: F401  (imported lazily: the app boots without it)
        from .model import ChordLM
        _MODEL = ChordLM.from_checkpoint(p, device)
    return _MODEL


@dataclass
class Suggestion:
    bar: int
    slot: int
    t0: float
    shown: str            # what the chart says now
    proposed: str         # the LM's top-1
    lm_conf: float
    pipe_conf: float
    alternatives: list[tuple[str, float]]   # top-K, including the top-1


def _slot_confidence(chart: dict, slots_per_bar: int) -> np.ndarray:
    """Pipeline confidence per half-bar slot: the `c` of the chord sounding there."""
    bpb = int(chart.get("bpb", 4) or 4)
    out: list[float] = []
    sections = sorted(chart.get("sections", []),
                      key=lambda s: (s.get("barRanges") or [[10 ** 9]])[0][0])
    carry = 0.5
    for sec in sections:
        for bar in sec.get("bars", []):
            events = sorted(((float(ch.get("beat", 0)), float(ch.get("c", 0.5)))
                             for ch in bar), key=lambda e: e[0])
            for s in range(slots_per_bar):
                start = s * bpb / slots_per_bar
                at = [c for b, c in events if b <= start]
                if at:
                    carry = at[-1]
                out.append(carry)
            if events:
                carry = events[-1][1]
    return np.asarray(out, dtype=np.float32)


def cloze_probs(net, tokens: list[int], device: str = "cpu",
                batch: int = 128) -> np.ndarray:
    """(T, V) — each slot's distribution computed with that slot masked out."""
    import torch
    import torch.nn.functional as F
    from .vocab import MASK
    T = len(tokens)
    base = torch.tensor(tokens, dtype=torch.long)
    out = np.zeros((T, VOCAB_SIZE), dtype=np.float32)
    with torch.no_grad():
        for s0 in range(0, T, batch):
            pos = list(range(s0, min(s0 + batch, T)))
            x = base.unsqueeze(0).repeat(len(pos), 1).clone()
            for r, p in enumerate(pos):
                x[r, p] = MASK
            x = x.to(device)
            slot_ix = (torch.arange(T, device=device) % net.cfg.slots_per_bar
                       ).expand(len(pos), T)
            lp = F.log_softmax(net(x, slot_ix), -1)
            rows = torch.arange(len(pos), device=device)
            out[s0:s0 + len(pos)] = lp[rows, torch.tensor(pos, device=device)
                                       ].exp().cpu().numpy()
    return out


def _name(tok: int | None, prev_abs: int | None) -> str:
    """Display name, resolving REP against the chord it stands for."""
    if tok is None:
        return "?"
    if tok == REP:
        return f"% ({token_name(prev_abs)})" if prev_abs is not None else "%"
    if is_chord(tok):
        root, fam = split_chord(tok)
        return f"{PC_NAMES[root]}:{fam}"
    return token_name(tok)


def propose(chart: dict, *, device: str = "cpu", lm_min: float = LM_MIN,
            pipe_max: float = PIPE_MAX, top_k: int = TOP_K,
            slots_per_bar: int = 2) -> list[Suggestion]:
    """Half-bars where the LM is confident and the pipeline is not.

    Returns [] rather than raising if the model is missing, torch is absent, or
    the chart is too short to give the LM any context — this is a refinement,
    never a dependency.
    """
    net = load_model(device)
    if net is None:
        return []
    g = app_chart_to_grid(chart, slots_per_bar=slots_per_bar)
    if len(g.tokens) < 8:
        return []
    pipe = _slot_confidence(chart, slots_per_bar)
    n = min(len(pipe), len(g.tokens))
    probs = cloze_probs(net, g.tokens[:n], device)
    order = np.argsort(-probs, axis=1)
    absolute = expand_repeats(g.tokens[:n])

    bpb = int(chart.get("bpb", 4) or 4)
    grid = chart.get("barGrid") or []
    out: list[Suggestion] = []
    for i in range(n):
        top = int(order[i, 0])
        conf = float(probs[i, top])
        if top == g.tokens[i] or conf < lm_min or pipe[i] > pipe_max:
            continue
        bar, slot = divmod(i, slots_per_bar)
        t0 = 0.0
        if bar + 1 < len(grid):
            bw = grid[bar + 1] - grid[bar]
            t0 = float(grid[bar] + slot * bw / slots_per_bar)
        prev_abs = absolute[i - 1] if i > 0 else None
        out.append(Suggestion(
            bar=bar, slot=slot, t0=round(t0, 2),
            shown=_name(g.tokens[i], prev_abs),
            proposed=_name(top, prev_abs),
            lm_conf=round(conf, 3), pipe_conf=round(float(pipe[i]), 3),
            alternatives=[(_name(int(order[i, k]), prev_abs),
                           round(float(probs[i, int(order[i, k])]), 3))
                          for k in range(top_k)],
        ))
    return out
