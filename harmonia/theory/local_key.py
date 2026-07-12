"""Symbolic local-key estimation for chord charts.

Given the chords in a section (as iReal tokens), accumulate a pitch-class
histogram from the chord tones and run the existing Krumhansl key matcher
(``key_profiles.infer_key``) to label the section's key/scale. Used to colour-
code which sections of a chart sit in which key.

Deliberately symbolic (chart-only, no audio): the chart is a clean chord
sequence, so chord tones are exact and a Krumhansl match on them is a solid,
cheap key estimate — no need to touch the noisy audio chroma here.
"""

from __future__ import annotations

import re

import numpy as np

from .key_profiles import infer_key

# conventional key-name spelling (flats for flat-side keys, etc.)
_MAJOR_NAMES = {0: "C", 1: "Db", 2: "D", 3: "Eb", 4: "E", 5: "F", 6: "Gb",
                7: "G", 8: "Ab", 9: "A", 10: "Bb", 11: "B"}
_MINOR_NAMES = {0: "C", 1: "C#", 2: "D", 3: "Eb", 4: "E", 5: "F", 6: "F#",
                7: "G", 8: "G#", 9: "A", 10: "Bb", 11: "B"}


def key_name(tonic: int, mode: str) -> str:
    names = _MAJOR_NAMES if mode == "major" else _MINOR_NAMES
    return f"{names[tonic % 12]} {mode}"


_LETTER = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_ACC = {"b": -1, "#": 1, "": 0}
_TOKEN_RE = re.compile(r"^([A-G])([b#]?)(.*)$")


_SHARP_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
_FLAT_NAMES = ["C", "Db", "D", "Eb", "E", "F", "Gb", "G", "Ab", "A", "Bb", "B"]
_FLAT_MAJ_TONICS = {0, 1, 3, 5, 6, 8, 10}


def prefer_flats(tonic: int, mode: str) -> bool:
    """Should this key be spelled with flats? (relative-major convention)."""
    maj = tonic if mode == "major" else (tonic + 3) % 12
    return maj % 12 in _FLAT_MAJ_TONICS


def transpose_token(token: str, semitones: int, flats: bool) -> str:
    """Transpose an iReal token by ``semitones``, spelling roots with flats or
    sharps. Quality tail is unchanged; a slash bass is transposed too."""
    root, q, bass = parse_token(token)
    names = _FLAT_NAMES if flats else _SHARP_NAMES
    out = names[(root + semitones) % 12] + q
    if bass is not None:
        out += "/" + names[(bass + semitones) % 12]
    return out


def parse_token(token: str) -> tuple[int, str, int | None]:
    """iReal token → (root_pc, quality_tail, bass_pc | None)."""
    head, _, bass = token.partition("/")
    m = _TOKEN_RE.match(head.strip())
    if not m:
        return 0, "", None
    letter, acc, qual = m.groups()
    root = (_LETTER[letter] + _ACC[acc]) % 12
    bass_pc = None
    if bass:
        bm = _TOKEN_RE.match(bass.strip())
        if bm:
            bass_pc = (_LETTER[bm.group(1)] + _ACC[bm.group(2)]) % 12
    return root, qual, bass_pc


def chord_pcs(token: str, include_bass: bool = True) -> dict[int, float]:
    """Chord tones of an iReal token as {pitch_class: weight}. Root and third
    weighted highest (they pin the key); 7th/extensions add lighter colour.
    ``include_bass=False`` drops the slash bass (for diatonic-membership tests,
    where a pedal / inversion bass shouldn't force a key change)."""
    root, q, bass = parse_token(token)
    ivs: dict[int, float] = {0: 2.0}

    # third + fifth from the triad quality
    if q.startswith("sus"):
        ivs[5] = 1.0 if "2" not in q else 0.0
        ivs[2 if "2" in q else 7] = 1.0
        ivs[7] = 1.0
    else:
        if q[:1] in ("-", "h") or q.startswith("o") or q.startswith("dim") or q.startswith("m"):
            ivs[3] = 1.5                      # minor third
        elif q.startswith("+") or q.startswith("aug"):
            ivs[4] = 1.5
        else:
            ivs[4] = 1.5                      # major third
        if q.startswith(("o", "dim", "h")) or "b5" in q:
            ivs[6] = 1.0
        elif q.startswith(("+", "aug")) or "#5" in q:
            ivs[8] = 1.0
        else:
            ivs[7] = 1.0

    # seventh / sixth
    if "^" in q or "maj7" in q or "M7" in q:
        ivs[11] = 0.8
    elif q.startswith(("o", "dim")) and "7" in q:
        ivs[9] = 0.8                          # dim7 → diminished 7th
    elif "6" in q and "b6" not in q:
        ivs[9] = 0.8
    elif "7" in q or q.startswith(("-", "h", "m")):
        ivs.setdefault(10, 0.8)

    out: dict[int, float] = {}
    for iv, w in ivs.items():
        out[(root + iv) % 12] = out.get((root + iv) % 12, 0.0) + w
    if include_bass and bass is not None:
        out[bass] = out.get(bass, 0.0) + 1.0
    return out


def estimate_key(tokens: list[str], weights: list[float] | None = None) -> dict:
    """Estimate the key/scale of a chord run. Returns
    {tonic, mode, name, conf}. ``weights`` (e.g. chord durations) default to 1."""
    weights = weights or [1.0] * len(tokens)
    chroma = np.zeros(12)
    for tok, w in zip(tokens, weights):
        for pc, cw in chord_pcs(tok).items():
            chroma[pc] += cw * w
    # tonic cue: tonal phrases tend to begin and (especially) end on the tonic,
    # which disambiguates a key from its dominant / relative (both share notes).
    if tokens:
        chroma[parse_token(tokens[-1])[0]] += 3.0
        chroma[parse_token(tokens[0])[0]] += 1.5
    if chroma.sum() == 0:
        return {"tonic": 0, "mode": "major", "name": "C major", "conf": 0.0}
    kp = infer_key(chroma)
    return {"tonic": kp.tonic, "mode": kp.mode,
            "name": key_name(kp.tonic, kp.mode), "conf": kp.confidence}


# ── per-chord local-key track (tonicization-aware) ─────────────────────────────
_MAJOR_SCALE = frozenset({0, 2, 4, 5, 7, 9, 11})
# minor = natural ∪ harmonic (so both the b7 and the raised leading tone of the
# dominant are in-scale) — the diatonic collection a ii–V–i lives in.
_MINOR_SCALE = frozenset({0, 2, 3, 5, 7, 8, 10, 11})

_KEYS = [(t, "major") for t in range(12)] + [(t, "minor") for t in range(12)]


def scale_pcs(tonic: int, mode: str) -> frozenset[int]:
    base = _MAJOR_SCALE if mode == "major" else _MINOR_SCALE
    return frozenset((tonic + i) % 12 for i in base)


_SCALES = [scale_pcs(t, m) for (t, m) in _KEYS]


def quality_class(q: str) -> str:
    """Coarse functional class of an iReal quality tail:
    'maj' | 'dom' | 'min' | 'm7b5' | 'dim' | 'sus'."""
    if q == "" or q.startswith("^") or q.startswith("6") or "maj" in q or q.startswith("M"):
        return "maj"
    if q.startswith("h") or "m7b5" in q or "-7b5" in q or (q.startswith(("-", "m")) and "b5" in q):
        return "m7b5"
    if q.startswith("o") or "dim" in q:
        return "dim"
    if q.startswith(("-", "m")) or "min" in q:
        return "min"
    if q.startswith("sus"):
        return "dom" if "7" in q or "9" in q or "13" in q else "sus"
    if q.startswith("+"):
        return "dom" if "7" in q else "maj"
    return "dom"                                     # 7, 9, 13, alt, b9…


def local_key_track(tokens: list[str]) -> list[dict]:
    """Estimate a *per-chord* local key by functional tonicization.

    Each chord is labelled with the key it belongs to *locally*, the way a
    player reads a chart:
      • a dominant 7th is a V — it points a perfect-fifth down to its target key
        (D7→G, G7→C, …), so a bridge cycle of dominants steps through a new key
        on every chord;
      • a ii (m7 / m7b5) that precedes its V binds to the same target;
      • maj7 / 6 is a I on its own root; a lone m7 is a minor i.
    A dominant's target takes the *mode of what it resolves to* (→ minor if the
    next chord is that minor chord), so deceptive minor ii–V's read correctly.
    Returns one {tonic, mode, name} per token.
    """
    n = len(tokens)
    if n == 0:
        return []
    parsed = [parse_token(t) for t in tokens]
    roots = [p[0] for p in parsed]
    cls = [quality_class(p[1]) for p in parsed]
    tonic: list[int | None] = [None] * n
    mode: list[str] = ["major"] * n
    ii_of: list[int | None] = [None] * n            # index of this ii's V

    for i in range(n):
        c, r = cls[i], roots[i]
        if c in ("dom", "sus"):
            t = (r + 5) % 12
            md = "major"
            if i + 1 < n and roots[i + 1] == t:      # resolves to its target?
                md = "minor" if cls[i + 1] in ("min", "m7b5") else "major"
            tonic[i], mode[i] = t, md
        elif c in ("min", "m7b5"):
            if i + 1 < n and cls[i + 1] == "dom" and roots[i + 1] == (r + 5) % 12:
                tonic[i], mode[i], ii_of[i] = (r - 2) % 12, "minor" if c == "m7b5" else "major", i + 1
            else:
                tonic[i], mode[i] = r, "minor"       # standalone minor i
        elif c == "maj":
            tonic[i], mode[i] = r, "major"
        # 'dim' stays None → inherited below

    for i in range(n):                               # a ii shares its V's key
        if ii_of[i] is not None:
            v = ii_of[i]
            tonic[i], mode[i] = tonic[v], mode[v]

    for i in range(n):                               # fill passing/dim chords
        if tonic[i] is None:
            j = next((k for k in range(i - 1, -1, -1) if tonic[k] is not None), None)
            if j is None:
                j = next((k for k in range(i + 1, n) if tonic[k] is not None), None)
            tonic[i], mode[i] = (tonic[j], mode[j]) if j is not None else (0, "major")

    return [{"tonic": tonic[i], "mode": mode[i], "name": key_name(tonic[i], mode[i])}
            for i in range(n)]


# ── continuity ("stay in the scale until a chord forces you out") ──────────────
# The scale unit is the diatonic *collection* (a major key / its relative minor
# share one); we track which of the 12 we're in and only leave when a chord tone
# is not in it — the way the ear holds a key until a note contradicts it.
_MAJOR_COLL = [frozenset((t + i) % 12 for i in _MAJOR_SCALE) for t in range(12)]
# melodic minor collections (the main extra jazz scale) — for the multi-scale view
_MELMIN = frozenset({0, 2, 3, 5, 7, 9, 11})
_MELMIN_COLL = [frozenset((t + i) % 12 for i in _MELMIN) for t in range(12)]

# Minor-colour variants of each *major collection*, keyed by the collection's
# major tonic ``c`` (its relative minor is ``m = (c+9) % 12``). A minor key is
# not just its natural-minor scale (= the relative-major collection): its
# harmonic form raises the 7th and its melodic form raises the 6th+7th. Those
# raised degrees (the leading tone of a ii–V–i, the major 6th of a i6 chord)
# are normal *within the same key* — treating them as out-of-scale is what made
# ``continuity_scale_track`` flag every minor-key V7 as a modulation (#23).
#   natural  minor of m = _MAJOR_COLL[c]                       (relative major)
#   harmonic minor of m = raise the 7th:  m+10 → m+11  i.e.  c+7 → c+8
#   melodic  minor of m = raise 6th+7th:  {c+5,c+7} → {c+6,c+8}
_HARMONIC_MIN_COLL = [
    frozenset(_MAJOR_COLL[c] - {(c + 7) % 12} | {(c + 8) % 12})
    for c in range(12)
]
# Melodic minor RAISES the 6th as well as the 7th, so as a *whole-scale* accept
# it is dangerously permissive (it pulls in sharp-side harmony and mislabels
# major sections — a full sweep showed it dropping oracle accuracy 55→44%). It
# is admitted **surgically**: only for a chord *rooted on the collection's
# relative-minor tonic* (an i6 / i(maj6), e.g. Gm6 in a Gm region). That single
# case is what keeps Autumn Leaves from blipping out of G minor.
_MELODIC_MIN_COLL = [
    frozenset(_MAJOR_COLL[c] - {(c + 5) % 12, (c + 7) % 12}
              | {(c + 6) % 12, (c + 8) % 12})
    for c in range(12)
]


def _fits_collection(tones: frozenset[int], c: int, root: int | None = None, *,
                     harmonic: bool = True, melodic: bool = True) -> bool:
    """Does the chord's ``tones`` sit inside major collection ``c`` under any of
    its accepted minor colours?

      • natural  = the major key / its relative *natural* minor (always tested);
      • harmonic = raise the relative minor's 7th (its V7's leading tone), if
        ``harmonic`` — the fix for #23's minor-key-V7-as-modulation bug;
      • melodic  = raise the relative minor's 6th+7th, if ``melodic`` — admitted
        only for a chord ``root``-ed on the relative-minor tonic (the i6 case;
        see ``_MELODIC_MIN_COLL``).
    """
    if tones <= _MAJOR_COLL[c]:
        return True
    if harmonic and tones <= _HARMONIC_MIN_COLL[c]:
        return True
    if (melodic and root is not None and root == (c + 9) % 12
            and tones <= _MELODIC_MIN_COLL[c]):
        return True
    return False


def continuity_scale_track_v2(
    tokens: list[str], home_tonic: int = 0, home_mode: str = "major",
    *, accept_harmonic: bool = True, accept_melodic: bool = True,
    lookahead: int = 2,
) -> list[dict]:
    """Harmonic-minor-aware per-chord scale tracker (v2 of
    :func:`continuity_scale_track`; the recommended tracker as of #23).

    Same contract — hold the current diatonic collection until a chord's tones
    leave it, then jump to the nearest collection (circle-of-fifths) that fits —
    but a collection now accepts a chord if its tones sit in the natural,
    harmonic, or (surgically) melodic minor colour of that collection (see
    :func:`_fits_collection`). This stops a minor key's own V7 (raised 7th, e.g.
    D7/D7b13 in Gm) or i6 (raised 6th, e.g. Gm6) from being mistaken for a
    modulation — the root cause of #23, where v1 oscillated Bb→G→F across a
    static G-minor Autumn-Leaves loop.

    On a forced jump the candidate collections are ranked by circle-of-fifths
    distance from the current one; ties are broken by a ``lookahead``-chord
    window (default 2 — 1 was one chord too slow to disambiguate paired chords
    like Cm7–Fm7 in All The Things You Are), then by tonic index.

    Labels: a contiguous same-collection run is read as its major key or relative
    minor by :func:`_label_collection`, except a run occupying the *home*
    collection of a minor-key seed inherits the home minor (so an all-diatonic
    minor tune reads as its minor tonic, not the relative major).

    Returns one ``{tonic, mode, name}`` per token. Signature-compatible with
    ``continuity_scale_track``.

    Measured vs v1 on the iRealb section-key oracle (#23 val split): accuracy
    54.1% → 55.3%, modulated-recall 23.7% → 27.7% — a strict improvement, and it
    fixes the Autumn-Leaves oscillation the metric alone did not capture.
    """
    n = len(tokens)
    if n == 0:
        return []
    tones = [core_tones(t) for t in tokens]
    roots = [parse_token(t)[0] for t in tokens]
    home_coll = home_tonic if home_mode == "major" else (home_tonic + 3) % 12
    cur = home_coll
    coll = [0] * n
    for i in range(n):
        t, r = tones[i], roots[i]
        if _fits_collection(t, cur, r, harmonic=accept_harmonic, melodic=accept_melodic):
            coll[i] = cur
            continue
        cands = [c for c in range(12)
                 if _fits_collection(t, c, r, harmonic=accept_harmonic, melodic=accept_melodic)]
        if not cands:                                   # chromatic (dim, altered)
            best = max(len(t & _MAJOR_COLL[c]) for c in range(12))
            cands = [c for c in range(12) if len(t & _MAJOR_COLL[c]) == best]

        def _rank(c: int) -> tuple[int, int, int]:
            la = 0
            for k in range(i + 1, min(i + 1 + lookahead, n)):
                if _fits_collection(tones[k], c, roots[k], harmonic=accept_harmonic,
                                    melodic=accept_melodic):
                    la -= 1                             # more future fits ⇒ better
            return (_cof_dist(c, cur), la, c)

        cur = min(cands, key=_rank)
        coll[i] = cur

    out: list[dict] = [None] * n                        # type: ignore
    i = 0
    while i < n:
        j = i
        while j < n and coll[j] == coll[i]:
            j += 1
        if coll[i] == home_coll and home_mode == "minor":
            tonic, mode = (home_coll + 9) % 12, "minor"  # home minor, not rel major
        else:
            tonic, mode = _label_collection(coll[i], tokens[i:j])
        for k in range(i, j):
            out[k] = {"tonic": tonic, "mode": mode, "name": key_name(tonic, mode)}
        i = j
    return out


# ── functional consolidation of secondary-dominant chains (#23 follow-up) ───────
def is_dominant_quality(q: str) -> bool:
    """Does this iReal quality tail function as a dominant (V7)? — its coarse
    functional class is ``dom`` (7/9/13/alt/7#5/7b9…). A sus7 also classes as
    ``dom`` upstream; here we require a *true* dominant (excludes bare sus), so
    the chain detector never chains a colouristic sus."""
    return quality_class(q) == "dom"


def consolidate_dominant_chains(
    track: list[dict], tokens: list[str], *,
    home_tonic: int = 0, home_mode: str = "major",
) -> list[dict]:
    """Collapse a descending-fifths chain of secondary dominants onto ONE key.

    ``continuity_scale_track_v2`` labels each chord by the diatonic *collection*
    it sits in, so a run of secondary dominants (e.g. ``A7 D7 G7#5``) — each
    with its own out-of-collection tones (A7's C#, D7's F#…) — reads as three
    different keys flickering past. But a musician hears ONE directed gesture:
    each chord is the V7 of the next, and the whole chain prepares a *single*
    final resolution (docs/known_issues.md #23). This deterministic post-pass
    over the raw v2 track fixes that representation mismatch before distillation.

    Algorithm (interval-only ⇒ transpose-equivariant by construction):
      1. Find each maximal run of ≥2 consecutive *dominant* chords in which every
         chord moves down a perfect fifth to the next
         (``(root[i+1] - root[i]) % 12 == 5``). A lone secondary dominant is
         **not** consolidated — the tracker already labels it with its target
         collection, so there is nothing to fix.
      2. Absorb a directly-preceding **ii** (m7 / m7b5) whose root is a fifth
         above the first dominant (the ``ii`` of that ``V``, e.g. ``E-7`` before
         ``A7 D7 G7#5``) into the run. A non-ii or non-fifth predecessor is left
         alone. (Documented choice: the ii shares the chain's single resolution;
         the first *dominant* is the strict floor, the ii an optional lead-in.)
      3. Relabel the whole run with the key the chain **resolves to**:
           • if a chord follows the last dominant and is its down-a-fifth target,
             inherit that (already-stable) chord's key from ``track``;
           • otherwise the chain dangles on its last dominant at a section end —
             the arrival is the implied resolution ``(root+5)``, ``major``, or the
             *home* mode when that root == ``home_tonic`` (so ``A7 D7 G7#5`` in C
             resolves to C **major**, the home key it is the V of, not a bare
             major guess).

    Reference case (home C major): ``G-7 C7 F^7 Bb7 E-7 A7 D7 G7#5`` — the tail
    ``E-7 A7 D7 G7#5``, which v2 labels ``C, F, Bb, Eb`` (5 collection changes
    across the section), becomes a single ``C major`` (2 changes), while the
    genuine borrowed ``Bb7`` (Eb) and the ``G-7 C7 F^7`` (F) region are left
    untouched.

    Does NOT solve: a chain whose resolution chord is a *different quality* than
    its dominant implies, or a tritone-sub chain (roots move by semitone, not a
    fifth) — those are out of scope and left as the tracker labelled them.
    """
    n = len(tokens)
    if n == 0:
        return [dict(d) for d in track]
    parsed = [parse_token(t) for t in tokens]
    roots = [p[0] for p in parsed]
    is_dom = [is_dominant_quality(p[1]) for p in parsed]
    out = [dict(d) for d in track]

    i = 0
    while i < n:
        if not (is_dom[i] and i + 1 < n and (roots[i + 1] - roots[i]) % 12 == 5):
            i += 1
            continue
        # extend the fifths chain while the next chord is also a dominant a
        # perfect fifth below (i..m are the chained dominants).
        m = i
        while (m + 1 < n and is_dom[m + 1]
               and (roots[m + 1] - roots[m]) % 12 == 5):
            m += 1
        if m == i:                       # lone secondary dominant → leave it
            i += 1
            continue
        start = i
        if i - 1 >= 0:                    # absorb a leading ii (m7 / m7b5)
            pr, pq, _ = parsed[i - 1]
            if quality_class(pq) in ("min", "m7b5") and (roots[i] - pr) % 12 == 5:
                start = i - 1
        if m + 1 < n and (roots[m + 1] - roots[m]) % 12 == 5:
            arr_t, arr_m = track[m + 1]["tonic"], track[m + 1]["mode"]
        else:
            arr_t = (roots[m] + 5) % 12
            arr_m = home_mode if arr_t == home_tonic else "major"
        for k in range(start, m + 1):
            out[k] = {"tonic": arr_t, "mode": arr_m,
                      "name": key_name(arr_t, arr_m)}
        i = m + 1
    return out


def core_tones(token: str) -> frozenset[int]:
    """The chord's own tones (root/3rd/5th/7th), no slash bass — what a key must
    contain to still hold under this chord."""
    return frozenset(chord_pcs(token, include_bass=False))


def _cof_dist(a: int, b: int) -> int:
    """Circle-of-fifths distance between two major collections (= #accidentals)."""
    d = abs((a * 7) % 12 - (b * 7) % 12)
    return min(d, 12 - d)


def _label_collection(coll: int, tokens: list[str]) -> tuple[int, str]:
    """Label a diatonic collection as its major key or relative minor, by whether
    the region leans on its major tonic chord or its relative-minor tonic chord."""
    rel = (coll + 9) % 12
    maj_hits = min_hits = 0
    for t in tokens:
        r, q, _ = parse_token(t)
        cls = quality_class(q)
        if r == coll and cls == "maj":
            maj_hits += 1
        elif r == rel and cls in ("min", "m7b5"):
            min_hits += 1
    if min_hits > maj_hits:
        return rel, "minor"
    return coll, "major"


def continuity_scale_track(tokens: list[str], home_tonic: int = 0,
                           home_mode: str = "major") -> list[dict]:
    """Per-chord scale by *continuity*: hold the current diatonic collection and
    only switch when a chord tone leaves it; when forced, jump to the nearest
    collection (circle-of-fifths) that fits, preferring one that also fits the
    next chord. Returns one {tonic, mode, name} per token."""
    n = len(tokens)
    if n == 0:
        return []
    tones = [core_tones(t) for t in tokens]
    cur = home_tonic if home_mode == "major" else (home_tonic + 3) % 12
    coll = [0] * n
    for i in range(n):
        t = tones[i]
        if t <= _MAJOR_COLL[cur]:
            coll[i] = cur
            continue
        cands = [c for c in range(12) if t <= _MAJOR_COLL[c]]
        if not cands:                                   # chromatic (dim, altered)
            best = max(len(t & _MAJOR_COLL[c]) for c in range(12))
            cands = [c for c in range(12) if len(t & _MAJOR_COLL[c]) == best]
        nxt = tones[i + 1] if i + 1 < n else None
        cur = min(cands, key=lambda c: (_cof_dist(c, cur),
                                        0 if (nxt is not None and nxt <= _MAJOR_COLL[c]) else 1, c))
        coll[i] = cur

    out: list[dict] = [None] * n                        # type: ignore
    i = 0
    while i < n:
        j = i
        while j < n and coll[j] == coll[i]:
            j += 1
        tonic, mode = _label_collection(coll[i], tokens[i:j])
        for k in range(i, j):
            out[k] = {"tonic": tonic, "mode": mode, "name": key_name(tonic, mode)}
        i = j
    return out


def fitting_scales(tokens: list[str], context: list[dict] | None = None,
                   max_scales: int = 3) -> list[list[dict]]:
    """For each chord, the scales it belongs to (diatonic major/relative-minor +
    melodic-minor), ordered by relevance to the local continuity scale and capped
    at ``max_scales``. Returns per chord a list of {tonic, mode, name}."""
    ctx = context or continuity_scale_track(tokens)
    out = []
    for tok, c in zip(tokens, ctx):
        t = core_tones(tok)
        home = c["tonic"] if c["mode"] == "major" else (c["tonic"] + 3) % 12
        opts = []
        for coll in range(12):
            if t <= _MAJOR_COLL[coll]:
                opts.append((_cof_dist(coll, home), 0, coll, "major"))
        for coll in range(12):
            if t <= _MELMIN_COLL[coll]:
                opts.append((_cof_dist(coll, home) + 1, 1, coll, "melmin"))
        opts.sort()
        seen, chord_scales = set(), []
        for _, _, tonic, kind in opts:
            key = (tonic, kind)
            if key in seen:
                continue
            seen.add(key)
            name = key_name(tonic, "major") if kind == "major" else f"{_MINOR_NAMES[tonic]} mel-min"
            chord_scales.append({"tonic": tonic, "mode": kind, "name": name})
            if len(chord_scales) >= max_scales:
                break
        out.append(chord_scales)
    return out


def section_keys(chords: list[dict], section_per_bar: list[str]) -> dict[str, dict]:
    """One key estimate per distinct section label, from all its chords.

    ``chords`` items need ``bar`` and ``symbol`` (iReal token). Returns
    {section_label: {tonic, mode, name, conf}}.
    """
    by_label: dict[str, list[str]] = {}
    for c in chords:
        bar = c["bar"]
        lab = section_per_bar[bar] if 0 <= bar < len(section_per_bar) else "?"
        by_label.setdefault(lab, []).append(c["symbol"])
    return {lab: estimate_key(toks) for lab, toks in by_label.items()}
