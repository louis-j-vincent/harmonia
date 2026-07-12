"""harmonia/models/local_key_seq_data.py — per-CHORD local-key distillation set.

Companion to :mod:`local_key_data` (which labels one key per *section* from the
duration-weighted **oracle**). This module builds a *per-chord* dataset whose
target is the key the **rule-based heuristic**
(:func:`theory.local_key.continuity_scale_track_v2`) assigns to each chord — the
teacher the user explicitly chose over the oracle (docs/known_issues.md #20/#23).

Rationale (user decision, expert musician): on the disagreement cases examined
together (Criss Cross, Dear Old Stockholm, A Beautiful Friendship) the heuristic
tracks *what an improviser would actually play on this exact chord* better than
the section oracle, even if it is noisier locally. We therefore distil the
heuristic — NOT the oracle — into a sequence model, giving that model wider
context than the heuristic's 2-chord lookahead so it can read a secondary-
dominant chain (Em7 A7 D7 G7#5) as one coherent gesture rather than 3–4 keys
flickering past, while still reacting to genuine collection changes (Gm7 in C →
F major; then Eb → Bb major).

Granularity is **per chord across the whole song** (not per section): the
heuristic is run once over the full bar-ordered token stream (seeded on the
song's global key, exactly as :mod:`local_key_heuristic` does), and every chord
position becomes a training target. Section boundaries are no longer the unit of
prediction.

    input   : per-chord (root_pc ∈ 0..11, q5 ∈ 0..4)          — as ProgressionEncoder
    target  : per-chord key idx ∈ 0..23  (12 major + 12 minor) — heuristic label

Deliberately symbolic (chart-only); the teacher is a deterministic function of
the clean chords.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..theory.local_key import continuity_scale_track_v2, parse_token
from .local_key_data import (
    DEFAULT_DB,
    JAZZ_CORPORA,
    POP_CORPORA,
    key_to_idx,
    parse_global_key,
    section_instances,
    token_to_q5,
)

__all__ = [
    "build_seq_examples",
    "split_seq_examples",
    "heuristic_track_for_tokens",
    "tokens_to_rel_example",
    "rel_to_abs_key",
    "collection_of",
    "count_collection_changes",
    "DEFAULT_DB",
    "POP_CORPORA",
    "JAZZ_CORPORA",
]


# ── relative-to-global encoding (transpose-equivariant BY CONSTRUCTION) ─────────
# Both the input roots and the target keys are expressed RELATIVE to the song's
# global tonic, so transposing a whole song leaves the (input, target) pair bit-
# for-bit identical → the model learns each harmonic motif once and generalises
# across all 12 keys for free (iRealb has only ~1856 songs; absolute encoding
# would fragment them ~150/key). Reconstruct an absolute key at inference by
# adding the known global tonic back (:func:`rel_to_abs_key`).
def rel_to_abs_key(rel_idx: int, global_tonic: int) -> int:
    """Relative key idx (delta from global tonic + 12*mode) → absolute 0..23."""
    return (rel_idx % 12 + global_tonic) % 12 + (rel_idx // 12) * 12


def tokens_to_rel_example(
    tokens: list[str], global_tonic: int, global_mode: str
) -> tuple[list[tuple[int, int]], list[int]]:
    """(tokens, global key) → (seq_rel, y_rel), both relative to ``global_tonic``.

    ``seq_rel`` is ``[(root_rel, q5)]`` with ``root_rel = (root - global_tonic) %
    12``; ``y_rel`` is the heuristic's per-chord key expressed as
    ``(local_tonic - global_tonic) % 12 + 12*mode``. Positions whose token has no
    q5 mapping are dropped from both, keeping the two index-aligned.
    """
    track = continuity_scale_track_v2(tokens, home_tonic=global_tonic,
                                      home_mode=global_mode)
    seq: list[tuple[int, int]] = []
    ys: list[int] = []
    for tok, sc in zip(tokens, track):
        q5 = token_to_q5(tok)
        if q5 is None:
            continue
        root_rel = (parse_token(tok)[0] - global_tonic) % 12
        mode_off = 0 if sc["mode"] == "major" else 12
        y_rel = (sc["tonic"] - global_tonic) % 12 + mode_off
        seq.append((root_rel, q5))
        ys.append(y_rel)
    return seq, ys


def collection_of(key_idx: int) -> int:
    """Diatonic *collection* (0..11) of a 0..23 key idx.

    A major key and its relative minor share the same 7 pitch classes, so they
    are the same collection (C major == A minor == collection 0). Churn is
    measured at collection level: a C-major↔A-minor label flip is NOT a change
    of scale, only a change of tonal centre within the same notes.
    """
    tonic, mode = key_idx % 12, key_idx // 12
    return tonic if mode == 0 else (tonic + 3) % 12


def count_collection_changes(keys: list[int]) -> int:
    """Number of collection changes across a per-chord key-idx sequence."""
    colls = [collection_of(k) for k in keys]
    return sum(1 for a, b in zip(colls, colls[1:]) if a != b)


def heuristic_track_for_tokens(
    tokens: list[str], home_tonic: int = 0, home_mode: str = "major"
) -> list[int]:
    """Per-chord heuristic key idx (0..23) for a raw token stream.

    Thin wrapper over :func:`continuity_scale_track_v2` that reduces each
    ``{tonic, mode}`` to the 0..23 index used everywhere else. Seeded on the
    supplied home key, matching :func:`local_key_heuristic.build_heuristic_examples`.
    """
    track = continuity_scale_track_v2(tokens, home_tonic=home_tonic, home_mode=home_mode)
    return [key_to_idx(t["tonic"], t["mode"]) for t in track]


def build_seq_examples(
    db_path: Path = DEFAULT_DB,
    corpora: set[str] | None = None,
) -> list[dict]:
    """One example per *song*: the full per-chord (seq, target) sequence, encoded
    **relative to the song's global tonic** (transpose-equivariant by construction).

    For each song the heuristic is run over the whole bar-ordered token stream
    (seeded on the global key); every chord with a usable q5 quality becomes one
    (input, target) step via :func:`tokens_to_rel_example`. Both roots and key
    targets are relative to the global tonic, so the example is identical under
    any transposition of the song.

    Each example: ``{seq: [(root_rel, q5)], y: [rel_key_idx], global_tonic,
    global_idx, corpus, song_idx, title}``. Songs with < 2 usable chords are
    skipped (nothing to smooth).
    """
    out: list[dict] = []
    for song_idx, line in enumerate(open(db_path)):
        rec = json.loads(line)
        if corpora is not None and rec["corpus"] not in corpora:
            continue
        gk = parse_global_key(rec["key"])
        if gk is None:
            continue
        global_tonic, global_mode = gk

        secs = section_instances(rec)
        all_tokens = [tok for sec in secs for tok, _ in sec["tokens"]]
        if not all_tokens:
            continue

        seq, ys = tokens_to_rel_example(all_tokens, global_tonic, global_mode)
        if len(seq) < 2:
            continue
        out.append({
            "seq": seq,
            "y": ys,
            "global_tonic": global_tonic,
            "global_idx": key_to_idx(global_tonic, global_mode),
            "corpus": rec["corpus"],
            "song_idx": song_idx,
            "title": rec.get("title", ""),
        })
    return out


def split_seq_examples(
    examples: list[dict], val_every: int = 5
) -> tuple[list[dict], list[dict]]:
    """Deterministic train/val split by *song* (1-in-5 to val), matching the
    project's other models (``local_key_data.split_examples`` /
    ``progression_encoder.split_sequences``). Song identity is ``song_idx`` (the
    raw db line index), so the split is stable across corpora filters."""
    train = [e for e in examples if e["song_idx"] % val_every != 0]
    val = [e for e in examples if e["song_idx"] % val_every == 0]
    return train, val
