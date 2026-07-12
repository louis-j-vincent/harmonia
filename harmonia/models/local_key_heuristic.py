"""harmonia/models/local_key_heuristic.py — zero-parameter local-key baseline (#23).

Ports the app's *client-side* key-tracking heuristic (``chart_interactive.py``'s
``continuity()`` / ``coreTones()``) to the server and evaluates it against the
same iReal section-key oracle used to train :class:`LocalKeyGRU`
(``local_key_data.oracle_section_key``).

The heuristic itself lives in the theory layer as
:func:`harmonia.theory.local_key.continuity_scale_track_v2` — a
harmonic-minor-aware successor to the JS ``continuity()``: hold the current
diatonic collection until a chord's tones leave it (accepting the harmonic-/
melodic-minor colours of the collection's relative minor, so a minor key's own
V7 or i6 does not read as a modulation — the #23 root cause), then jump to the
nearest collection on the circle of fifths that fits, breaking ties by a
2-chord lookahead. This
module runs that per-chord tracker over each song and reduces it to one key per
section (duration-weighted vote) so it can be scored against the per-section
oracle, exactly on the GRU's validation split.

Why this matters: the heuristic has **zero learned parameters**. If it lands
within a few points of the GRU, the GRU is not buying much over a rules-based
tracker that already ships in the browser (see docs/known_issues.md #23).
"""
from __future__ import annotations

import json
from pathlib import Path

from ..theory.local_key import continuity_scale_track_v2, parse_token
from .local_key_data import (
    key_to_idx,
    oracle_section_key,
    parse_global_key,
    section_instances,
    token_to_q5,
)

__all__ = ["section_pred_from_track", "build_heuristic_examples", "evaluate_heuristic"]


def section_pred_from_track(
    section_scales: list[dict], durations: list[float]
) -> int:
    """Reduce a section's per-chord continuity scales to a single 0..23 key idx.

    Duration-weighted vote over ``key_to_idx(tonic, mode)`` — the key the section
    spends the most time in under the continuity tracker. A short/no-duration
    chord still gets a small floor weight (0.25), mirroring ``section_chroma``.
    """
    votes: dict[int, float] = {}
    for sc, w in zip(section_scales, durations):
        idx = key_to_idx(sc["tonic"], sc["mode"])
        votes[idx] = votes.get(idx, 0.0) + max(float(w), 0.25)
    if not votes:
        return 0
    # tie-break toward the lower key index for determinism
    return max(votes, key=lambda k: (votes[k], -k))


def build_heuristic_examples(
    db_path: Path,
    margin: float = 6.0,
    corpora: set[str] | None = None,
) -> list[dict]:
    """One prediction per section instance, aligned 1:1 with
    :func:`local_key_data.build_examples`.

    For each song the continuity tracker is run over the *whole* chord sequence
    (concatenated section by section, in bar order) seeded on the song's global
    key, then each section's slice is reduced to one key. The section filter and
    oracle-label/``modulated`` computation are identical to ``build_examples``,
    and ``song_idx`` is the raw line index, so ``split_examples`` yields exactly
    the GRU's validation split.

    Each example: ``{pred, y, y_global, modulated, corpus, song_idx, label}``.
    """
    out: list[dict] = []
    for song_idx, line in enumerate(open(db_path)):
        rec = json.loads(line)
        if corpora is not None and rec["corpus"] not in corpora:
            continue
        gk = parse_global_key(rec["key"])
        if gk is None:
            continue
        global_idx = key_to_idx(*gk)
        home_mode = "minor" if gk[1] == "minor" else "major"

        secs = section_instances(rec)
        all_tokens = [tok for sec in secs for tok, _ in sec["tokens"]]
        if not all_tokens:
            continue
        track = continuity_scale_track_v2(all_tokens, home_tonic=gk[0], home_mode=home_mode)

        pos = 0
        for sec in secs:
            n = len(sec["tokens"])
            sec_scales = track[pos:pos + n]
            durs = [d for _, d in sec["tokens"]]
            pos += n

            # identical section filter to build_examples: need >=1 usable q5 token
            seq = [
                (parse_token(tok)[0] % 12, q)
                for tok, _ in sec["tokens"]
                if (q := token_to_q5(tok)) is not None
            ]
            if not seq:
                continue

            y, modulated = oracle_section_key(sec["tokens"], global_idx, margin)
            pred = section_pred_from_track(sec_scales, durs)
            out.append({
                "pred": pred,
                "y": y,
                "y_global": global_idx,
                "modulated": modulated,
                "corpus": rec["corpus"],
                "song_idx": song_idx,
                "label": sec["label"],
            })
    return out


def evaluate_heuristic(examples: list[dict]) -> dict:
    """Accuracy + modulated-subset recall of the heuristic on ``examples``.

    Returns ``{acc, mod_acc, nonmod_acc, n, n_mod, n_nonmod}``. ``mod_acc`` is
    the recall on sections the oracle marked as modulated — the transferable
    capability the global-key baseline scores 0% on by construction.
    """
    correct = mod_c = nonmod_c = 0
    n = mod_t = nonmod_t = 0
    for e in examples:
        ok = int(e["pred"] == e["y"])
        correct += ok
        n += 1
        if e["modulated"]:
            mod_c += ok
            mod_t += 1
        else:
            nonmod_c += ok
            nonmod_t += 1
    return {
        "acc": correct / max(n, 1),
        "mod_acc": mod_c / max(mod_t, 1),
        "nonmod_acc": nonmod_c / max(nonmod_t, 1),
        "n": n,
        "n_mod": mod_t,
        "n_nonmod": nonmod_t,
    }
