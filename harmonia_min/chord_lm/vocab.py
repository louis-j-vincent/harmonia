"""harmonia_min/chord_lm/vocab.py — the chord-LM token vocabulary.

One token per *metrical slot* (half-bar by default). A token is either

    a chord   = (root pitch-class 0-11) x (family, 7 of them)
    NC        = explicit no-chord / silence
    REP       = "same sounding chord as the previous slot"

plus the usual PAD / BOS / EOS / MASK machinery.

Why families and not full qualities
-----------------------------------
The brief (Louis, 2026-08-01): "l'accord avec la qualité famille, donc pas de
septième sauf la dominante qui qualifie une famille en elle-même". A 7th is an
*acoustic* question — whether the player voiced the 7th — not a grammatical one.
Whether a chord is dominant, on the other hand, IS grammatical: it is what makes
a V a V. So maj7/maj6/maj9/6-9 all collapse to `maj`, min7/min9/min6 to `min`,
but every dominant alteration stays `dom`.

Seven families, with their share of the 142,373 chord symbols in
data/ireal/*.txt (2,401 tunes, measured 2026-08-01):

    maj  32.6%   min  27.9%   dom  32.5%
    hdim  2.5%   sus   2.4%   dim   1.6%   aug 0.2%

`hdim` (m7b5) is kept separate from `dim` because it is functionally a *minor*
ii chord heading to a dominant, not a diminished passing chord — the grammar
around it is completely different. `aug` is rare (0.2%) but costs 12 tokens.

Why REP
-------
On a fixed metrical grid, ~70% of slots repeat the previous chord. Spelling that
out as the chord token again would let the model score high by copying, and
would hide the question we actually care about: *does the harmony change here?*
REP splits the problem in two — harmonic rhythm (change vs hold) and identity
(which chord) — and makes both separately measurable. See `metrics` in
scripts/chord_lm_eval.py.

REP always refers to the last *sounding* chord, skipping over other REPs but NOT
over NC: a chord after an NC is written in full.

Roots are FUNCTIONAL, not sounding-bass
---------------------------------------
The rest of the project targets the sounding bass pitch class (see
harmonia.data.corpus_schema.sounding_bass_pc, CLAUDE.md error-pattern #3). This
module deliberately does NOT: a grammar model needs `D-7 G7 C^7` to look like
ii-V-I, and `D-7/A` must be the same token as `D-7`. Slash basses are dropped
here. Anything wiring the LM to the live pipeline has to convert.
"""
from __future__ import annotations

import re

# ── families ────────────────────────────────────────────────────────────────
FAMILIES = ["maj", "min", "dom", "dim", "hdim", "sus", "aug"]
FAM_IDX = {f: i for i, f in enumerate(FAMILIES)}
N_FAM = len(FAMILIES)

PC_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
_NOTE_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# ── token ids ───────────────────────────────────────────────────────────────
# 0..83  chords, id = 7*root + family
N_CHORD = 12 * N_FAM  # 84
NC = N_CHORD          # 84
REP = N_CHORD + 1     # 85
PAD = N_CHORD + 2     # 86
BOS = N_CHORD + 3     # 87
EOS = N_CHORD + 4     # 88
MASK = N_CHORD + 5    # 89
VOCAB_SIZE = N_CHORD + 6  # 90

SPECIALS = {NC: "N.C.", REP: "%", PAD: "<pad>", BOS: "<s>", EOS: "</s>",
            MASK: "<mask>"}


def chord_id(root_pc: int, family: str) -> int:
    """(root pitch-class, family name) -> token id."""
    return (root_pc % 12) * N_FAM + FAM_IDX[family]


def is_chord(tok: int) -> bool:
    return 0 <= tok < N_CHORD


def split_chord(tok: int) -> tuple[int, str]:
    """token id -> (root pitch-class, family name). Raises on a special token."""
    if not is_chord(tok):
        raise ValueError(f"not a chord token: {tok} ({SPECIALS.get(tok, '?')})")
    return tok // N_FAM, FAMILIES[tok % N_FAM]


def token_name(tok: int) -> str:
    """Human-readable, e.g. 12 -> 'Db:maj', 85 -> '%'."""
    if tok in SPECIALS:
        return SPECIALS[tok]
    root, fam = split_chord(tok)
    return f"{PC_NAMES[root]}:{fam}"


def transpose(tok: int, semitones: int) -> int:
    """Transpose a chord token; specials pass through unchanged."""
    if not is_chord(tok):
        return tok
    root, fam = split_chord(tok)
    return chord_id(root + semitones, fam)


def parse_root(sym: str) -> int:
    """'Bb' -> 10, 'F#' -> 6, 'C' -> 0."""
    pc = _NOTE_PC[sym[0].upper()]
    for ch in sym[1:]:
        if ch in "#s":
            pc += 1
        elif ch in "b!":
            pc -= 1
    return pc % 12


# ── iReal quality tail -> family ────────────────────────────────────────────
# EXHAUSTIVE over every tail observed in data/ireal/*.txt (72 distinct, 2026-08-01)
# plus the iReal tails harmonia_min.labels.to_chord emits, so the same table
# serves both the training corpus and live pipeline output.
#
# Rules applied, in the brief's terms:
#   * any dominant colour (b9 #9 #11 b13 #5 b5 13 9 alt) -> dom
#   * any major colour (^ 6 69 add9 2 M7) -> maj, INCLUDING ^7#5 (a maj7#5 is a
#     lydian-augmented major sound, not an augmented triad heading anywhere)
#   * a bare `5` power chord -> maj (its third is unstated, and maj is the prior)
#   * `sus` beats the dominant colour it carries: 7sus/9sus/13sus -> sus, because
#     the suspension is what constrains the grammar (it resolves, or it doesn't)
#   * `o` -> dim, `h` -> hdim, `-7b5` -> hdim (alias), `o^7` -> dim
#   * `+` (augmented triad) -> aug; `-#5` (minor #5) -> min
_QUALITY_FAMILY: dict[str, str] = {}


def _reg(fam: str, *tails: str) -> None:
    for t in tails:
        _QUALITY_FAMILY[t] = fam


_reg("maj", "", "^", "^7", "^9", "^13", "6", "69", "2", "add9", "5",
     "^7#11", "^9#11", "^7#5", "M7", "M7*", "maj", "maj7", "maj9", "6/9")
_reg("min", "-", "-7", "-9", "-6", "-11", "-69", "-^7", "-^9", "-b6",
     "-add9", "-#5", "min", "min7", "min9", "m6", "minmaj7", "-^")
_reg("dom", "7", "9", "11", "13", "7b9", "7#9", "7b5", "7#5", "7#11", "7b13",
     "7alt", "7at", "9#5", "9b5", "9#11", "13b9", "13#9", "13#11",
     "7b9#5", "7b9b5", "7b9b13", "7b9#9", "7b9#11", "7#9#5", "7#9b5", "7#9#11",
     "dom7", "dom7alt", "aug7")
_reg("dim", "o", "o7", "o^7", "dim", "dim7")
_reg("hdim", "h", "h7", "h9", "-7b5", "m7b5", "hdim7")
_reg("sus", "sus", "sus4", "sus2", "7sus", "7sus4", "9sus", "13sus",
     "7b9sus", "7b13sus", "7susadd3", "11sus", "sus4(b7)")
_reg("aug", "+", "aug", "augmaj7")

# tails iReal writes truncated in the wild (the parser can clip a trailing 's'
# when it strips size markers) — normalised before lookup
_TAIL_ALIASES = {"su": "sus", "7su": "7sus", "9su": "9sus", "13su": "13sus",
                 "7b9su": "7b9sus", "7b13su": "7b13sus"}


def quality_to_family(tail: str) -> str | None:
    """iReal quality tail -> family name, or None if unmappable.

    Returns None rather than guessing: callers count the misses (a silent
    default-to-maj is exactly the bug CLAUDE.md/labels.py warn about).
    """
    tail = tail.strip()
    tail = _TAIL_ALIASES.get(tail, tail)
    return _QUALITY_FAMILY.get(tail)


_CHORD_RE = re.compile(r"^([A-G][b#]?)(.*)$")
# iReal decoration that is not part of the harmony: repeat-ending numbers,
# segno/coda/fermata letters, and the s/l chord-size prefixes.
_STRIP_PREFIX = re.compile(r"^[sl]+")
_STRIP_TRAIL = re.compile(r"[UQSrxz,\s]+$")


def parse_ireal_token(token: str) -> int | None:
    """One iReal chord token (e.g. 'Eh7', 'D-7/A', 'n', 'sC7') -> token id.

    Returns NC for explicit no-chord, None if the token cannot be mapped
    (caller decides: skip the slot, or reject the song).
    """
    tok = token.strip()
    tok = re.sub(r"N\d", "", tok)
    tok = _STRIP_PREFIX.sub("", tok)
    tok = _STRIP_TRAIL.sub("", tok)
    if not tok:
        return None
    if tok[0] == "n":            # explicit N.C.
        return NC
    if tok[0] == "p":            # iReal's own "repeat previous chord" glyph
        return REP
    if tok.startswith("W"):
        # bass-note-only symbol ("W/Ab"): no quality is stated. Treat as maj on
        # that bass — the same choice tune_to_mma makes (power chord).
        m = re.match(r"W/([A-G][b#]?)", tok)
        return chord_id(parse_root(m.group(1)), "maj") if m else None
    m = _CHORD_RE.match(tok)
    if m is None:
        return None
    root_s, qual = m.groups()
    qual = qual.split("/")[0]        # drop slash bass: functional root only
    qual = qual.replace("W", "").strip()
    fam = quality_to_family(qual)
    if fam is None:
        return None
    return chord_id(parse_root(root_s), fam)


def parse_app_chord(root: int, tail: str) -> int | None:
    """harmonia_min.labels.to_chord output -> token id.

    `to_chord` returns {root, q, bass}; `q` is an iReal tail from the same
    alphabet, so this is just the family lookup. The `bass` field is ignored on
    purpose (see the module docstring: functional root, not sounding bass).
    """
    fam = quality_to_family(tail)
    return None if fam is None else chord_id(root, fam)
