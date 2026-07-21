"""Single source of truth for training-corpus `match`-quality labels and I/O.

## Why this module exists

`docs/refactoring_suggestions.md` §2a: the `match`/quality field on every
training corpus record was a free-string convention with no enum, no
constant, no validator. Grepping actual values written to corpora found:

    114  "exact"
     93  "family"
     45  "none"
     15  "mismatch"
      2  "billboard_gt"   <- scratchpad/build_billboard_pilot.py, build_billboard_60.py

Downstream trainers filtered on hardcoded literals like
`match == "exact"` (e.g. `scripts/train_real_audio_final.py:179`). A corpus
tagged `"billboard_gt"` — a value only two scratch builders ever emitted —
was therefore silently filtered to **zero rows** by any such gate, because
no trainer's literal recognized it. That silent-all-rows-dropped failure is
the bug this module exists to make structurally impossible.

## The `billboard_gt` decision

`billboard_gt` is aliased to `MatchQuality.EXACT`, not promoted to its own
level. Reasoning: `billboard_gt` records come from Billboard's hand-annotated
ground-truth chord charts matched directly to audio by duration — zero
inference/heuristic steps between label and audio, which is the same trust
tier as `"exact"` (an exact string/timing match from the source corpus's own
matching heuristic). Both represent "no reason to distrust this label."
`"family"` and below involve an actual quality-collapse or fuzzy-match step
that `"exact"`/`"billboard_gt"` don't.

**This is a judgment call, not a measurement — override it here if you
disagree.** If you'd rather keep Billboard ground truth distinguishable from
heuristic-exact YouTube matches (e.g. to weight them differently in a loss),
add a real `MatchQuality.BILLBOARD_GT` level instead of aliasing, and update
`_ALIASES` below accordingly. Nothing downstream depends on the alias yet
(see "What this module does NOT do").

## What this module does NOT do (yet)

- It is not wired into any corpus builder or trainer. Existing scripts
  (`scripts/train_real_audio_final.py`, `scripts/train_yt_exact_matches.py`,
  `scripts/train_yt_real_audio.py`, `scratchpad/build_billboard_*.py`) still
  use their own literal `match == "exact"` checks and their own ad hoc
  `np.savez` calls. Swapping each one over to `filter_by_match`/`save_corpus`/
  `load_corpus` is deliberately deferred to a follow-up change per song, one
  trainer at a time, so each swap can be verified independently (same row
  count in vs. out, unless the old count was itself the bug).
- It does not validate feature array dtypes/shapes beyond what
  `save_corpus`/`load_corpus` naturally preserve via `np.savez`/`np.load`
  (e.g. it won't catch a `feat48` array that's secretly `(N, 47)`). That's a
  separate, not-yet-built validation layer.
- It does not know about `qualities`/`quality_idx` semantics (the 7-way
  maj/min/dom/hdim/dim/aug/sus quality vocabulary) beyond passing them
  through as opaque arrays — that vocabulary lives in
  `harmonia/data/billboard_translator.py`.
"""

from __future__ import annotations

from enum import IntEnum
from pathlib import Path
from typing import Any, Iterable

import numpy as np


class MatchQuality(IntEnum):
    """Trust tiers for a corpus record's chord-label match, worst to best.

    Ordering (low to high) lets `filter_by_match(records, minimum=...)` do a
    simple `>=` comparison. Only levels actually observed in existing
    corpora are represented — see module docstring for the grep that
    produced this list. `"billboard_gt"` is intentionally NOT its own level;
    it is aliased to EXACT (see module docstring for the reasoning).
    """

    NONE = 0
    MISMATCH = 1
    FAMILY = 2
    EXACT = 3


# String values as they actually appear in `match` arrays on disk, mapped to
# the enum level they represent. `billboard_gt` is the alias described in the
# module docstring.
_STRING_TO_LEVEL: dict[str, MatchQuality] = {
    "none": MatchQuality.NONE,
    "mismatch": MatchQuality.MISMATCH,
    "family": MatchQuality.FAMILY,
    "exact": MatchQuality.EXACT,
    "billboard_gt": MatchQuality.EXACT,  # aliased, see module docstring
}


class UnknownMatchValueError(ValueError):
    """Raised when a corpus contains a `match` string this module doesn't recognize.

    This is the direct fix for the §2a bug: previously an unrecognized value
    (e.g. `"billboard_gt"` against a trainer hardcoded to `match == "exact"`)
    was silently filtered to zero rows instead of raising. Any new match
    string must be explicitly added to `_STRING_TO_LEVEL` above before a
    corpus using it can be loaded — that is the point.
    """


def match_level(value: str) -> MatchQuality:
    """Map a raw `match` string (as stored in a corpus) to a `MatchQuality`.

    Raises `UnknownMatchValueError` on any value not in `_STRING_TO_LEVEL`,
    rather than silently dropping it.
    """
    try:
        return _STRING_TO_LEVEL[value]
    except KeyError as e:
        raise UnknownMatchValueError(
            f"Unrecognized match value {value!r}. Known values: "
            f"{sorted(_STRING_TO_LEVEL)}. If this is a genuinely new match "
            f"category, add it to MatchQuality/_STRING_TO_LEVEL in "
            f"harmonia/data/corpus_schema.py — do not filter it out silently."
        ) from e


def filter_by_match(
    match: Iterable[str], minimum: MatchQuality = MatchQuality.EXACT
) -> np.ndarray:
    """Return a boolean mask selecting records whose match quality >= `minimum`.

    Safe replacement for ad hoc `match == "exact"` literals scattered across
    trainers. Raises `UnknownMatchValueError` (via `match_level`) on any
    unrecognized string in `match` rather than silently excluding it.

    Example:
        mask = filter_by_match(corpus["match"], minimum=MatchQuality.EXACT)
        kept = {k: v[mask] for k, v in corpus.items() if hasattr(v, "__len__")}
    """
    match = np.asarray(list(match))
    levels = np.array([match_level(m) for m in match])
    return levels >= minimum


# Keys every corpus written by `save_corpus` is expected to carry. Grounded
# in the two existing corpora inspected while writing this module:
# data/cache/yt_corpus/corpus_50.npz and data/cache/billboard_bp48_pilot.npz
# (read-only inspection; neither file was modified). Both share this core
# set; corpus_50.npz additionally carries feat12_cqt/feat12_cqt_abs, which
# save_corpus/load_corpus pass through as optional extra keys rather than
# baking into the required set (no premature generality).
REQUIRED_KEYS = (
    "feat48",       # (N, 48) float32 — BP48 pitch-class features, relative-rooted
    "feat48_abs",   # (N, 48) float32 — same features, absolute-pitch-rooted
    "root",         # (N,) int32 — chord root pitch class 0-11
    "quality_idx",  # (N,) int32 — index into `qualities`
    "quality",      # (N,) str — quality name, e.g. "maj"/"dom"
    "labels",       # (N,) str — original chord label string
    "match",        # (N,) str — match-quality tag, see MatchQuality
    "t0",           # (N,) float64 — segment start time (s)
    "t1",           # (N,) float64 — segment end time (s)
    "song_id",      # (N,) str — source song/track identifier
    "qualities",    # (Q,) str — the quality vocabulary, e.g. 7-way maj/min/dom/...
)


def save_corpus(path: str | Path, **arrays: np.ndarray) -> None:
    """Write a training corpus to `path` as an .npz.

    Warns (does not raise) if any of `REQUIRED_KEYS` is missing, since some
    legitimate corpora (e.g. pre-match-quality experiments) may omit a key
    like `match`. Raises `UnknownMatchValueError` up front if a `match`
    array is present and contains a value `load_corpus` would later reject
    — better to fail at write time than at the next load.
    """
    missing = [k for k in REQUIRED_KEYS if k not in arrays]
    if missing:
        import warnings

        warnings.warn(
            f"save_corpus: missing expected keys {missing} (writing anyway); "
            f"see REQUIRED_KEYS in harmonia/data/corpus_schema.py",
            stacklevel=2,
        )
    if "match" in arrays:
        for m in arrays["match"]:
            match_level(str(m))  # raises UnknownMatchValueError if bad
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


# ── Sounding-bass pitch-class resolver ───────────────────────────────────────
#
# Added 2026-07-16 for the deliberate project-wide target redefinition from
# FUNCTIONAL ROOT to SOUNDING BASS PITCH CLASS (CLAUDE.md rule #3: "ground
# truth is a measurement"). For `C:maj/D` the old target was `C` (functional
# root); the new target is the sounding bass pitch class `D`.
#
# Harte `/bass` annotations appear in TWO conventions across sources in this
# repo; this resolver handles both:
#
#   * Numeric SCALE-DEGREE relative to the root, e.g. `/3`, `/b7`, `/5`.
#     Resolved as `bass_pc = (root_pc + degree_semitone_offset) % 12`.
#     **This is the ONLY form present in RWC-Popular** — verified 2026-07-16:
#     all 1633/1633 inverted labels use degree tokens (`/3` 707, `/5` 491,
#     `/b3` 108, `/b7` 95, `/2` 175, `/4` 38, `/7` 14, plus `/6 /b6 /b5`);
#     none is a note-letter (docs/known_issues.md, "VERIFIED NON-BUG").
#   * Literal NOTE-LETTER bass, e.g. `/D`, `/Bb`, `/F#`. Resolved directly via
#     the note-name→pitch-class map, INDEPENDENT of the root. (Present in some
#     Billboard/other Harte exports; kept here so the resolver is source-agnostic.)
#   * No slash (root-position chord): `bass_pc == root_pc`. No change from the
#     old functional-root behaviour for the ~87.6% of RWC chords in root position.
#
# Sources that STRUCTURALLY CANNOT supply this target:
#   * POP909 discards `/bass` inversions entirely at parse time (CLAUDE.md).
#     It can only ever report `bass_pc == root_pc`, so under the new target it
#     is DEGRADED/EXCLUDED, not silently mishandled. Callers using POP909 for
#     bass-target work must flag it, not treat its root as a sounding bass.
#
# Disambiguation of the `/tail` token: strip leading accidentals (`#`/`b`);
# if the next character is a DIGIT it is a scale degree, if it is an uppercase
# `A`–`G` it is a note letter. This is unambiguous because Harte degree tokens
# are `[b#]*<digit>` (`b7`) and note tokens are `<A-G>[b#]*` (`Bb`).

_NOTE_NAME_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# Scale-degree token -> semitones above the chord root. Matches the Harte
# reference and `scripts/build_jaah_corpus.py:_DEG_SEMI`. Do NOT hand-edit
# without checking both against the spec (CLAUDE.md rule #1).
_BASS_DEGREE_SEMI = {
    "1": 0, "b2": 1, "2": 2, "#2": 3, "b3": 3, "3": 4, "4": 5, "#4": 6,
    "b5": 6, "5": 7, "#5": 8, "b6": 8, "6": 9, "bb7": 9, "b7": 10, "7": 11,
    "b9": 1, "9": 2, "#9": 3, "11": 5, "#11": 6, "b13": 8, "13": 9, "#13": 10,
}


def _note_letter_pc(tok: str) -> int | None:
    """Resolve an absolute Harte note token (`D`, `Bb`, `F#`) -> pitch class."""
    if not tok or tok[0] not in _NOTE_NAME_PC:
        return None
    pc = _NOTE_NAME_PC[tok[0]]
    for c in tok[1:]:
        if c == "#":
            pc += 1
        elif c == "b":
            pc -= 1
        else:
            return None  # trailing non-accidental -> not a bare note token
    return pc % 12


def sounding_bass_pc(label: str, root_pc: int | None) -> int | None:
    """Resolve the SOUNDING BASS pitch class (0–11) of a Harte chord label.

    This is the NEW project-wide prediction target (2026-07-16), replacing the
    functional root. See the module comment above for the full convention.

    Parameters
    ----------
    label : str
        A Harte label, e.g. ``"C:maj"``, ``"C:maj/D"``, ``"Eb:7/3"``, ``"N"``.
    root_pc : int | None
        The FUNCTIONAL root pitch class already parsed for this label (the
        thing stored in `corpus["root"]`). Required to resolve scale-degree
        bass tokens; may be ``None`` for no-chord labels.

    Returns
    -------
    int | None
        The sounding bass pitch class in 0–11, or ``None`` for `N`/`X`/empty
        or when the bass token is unrecognized AND no valid root is available.

    Semantics
    ---------
    * No slash  -> returns ``root_pc`` (root-position chord).
    * `/<degree>` (e.g. `/3`, `/b7`) -> ``(root_pc + offset) % 12``.
    * `/<note>`  (e.g. `/D`, `/Bb`)  -> absolute note pc, root-independent.
    * Unrecognized bass token -> falls back to ``root_pc`` (never invents a pc).

    Notes
    -----
    What this does NOT solve (CLAUDE.md rule #4): it trusts the *label's* bass
    annotation. It does not verify that the annotated bass is the acoustically
    sounding one, and it cannot recover a bass that the source (e.g. POP909)
    discarded before this point.
    """
    if label is None:
        return root_pc
    lab = label.strip()
    if lab in ("N", "X", ""):
        return None
    if "/" not in lab:
        return None if root_pc is None else root_pc % 12
    tail = lab.split("/", 1)[1].strip()
    if not tail:
        return None if root_pc is None else root_pc % 12
    # Disambiguate: skip leading accidentals, inspect first "real" char.
    i = 0
    while i < len(tail) and tail[i] in "#b":
        i += 1
    if i < len(tail) and tail[i].isdigit():
        # scale-degree token, relative to root
        off = _BASS_DEGREE_SEMI.get(tail)
        if off is None or root_pc is None:
            return None if root_pc is None else root_pc % 12
        return (root_pc + off) % 12
    # otherwise treat as an absolute note-letter bass
    pc = _note_letter_pc(tail)
    if pc is None:
        return None if root_pc is None else root_pc % 12
    return pc


def load_corpus(path: str | Path) -> dict[str, np.ndarray]:
    """Read a corpus .npz written by `save_corpus` (or any np.savez).

    **Critical property:** if a `match` array is present, every value in it
    is validated against `MatchQuality` via `match_level`. An unrecognized
    value raises `UnknownMatchValueError` immediately — it is NOT silently
    filtered out. This is the direct fix for the §2a bug (a corpus tagged
    `"billboard_gt"` used to vanish under any `match == "exact"` gate with
    no error at all).
    """
    path = Path(path)
    with np.load(path, allow_pickle=False) as npz:
        data: dict[str, Any] = {k: npz[k] for k in npz.files}
    if "match" in data:
        for m in data["match"]:
            match_level(str(m))  # raises UnknownMatchValueError if bad
    return data
