"""harmonia/models/chord_context_prior.py — Phase 1 of the chord-context-prior
design (docs/design_chord_context_prior.md, branch feat/chord-context-prior).

Target-relative infilling model: given a chord locked/known BEFORE a position
and one AFTER it, score every (root, quality) candidate for the position in
between.  This is deliberately NOT the section_structure.py trigram (which
predicts the NEXT chord from the two PRECEDING ones, deltas anchored on the
middle known chord).  Here both neighbours are known and the deltas are
anchored on the CANDIDATE itself (Louis's 2026-07-30 directive) so the same
table fires identically in every key with no 12x transpose augmentation:

    key(prev, candidate, next) = (q_c, Δp, q_p, Δn, q_n)
    Δp = (root_p - root_c) % 12,  Δn = (root_n - root_c) % 12

Worked example (design doc): Dm7 G7 Cmaj, scoring candidate G7 (root 7, dom)
with prev Dm7 (root 2, min) and next Cmaj (root 0, maj):
    Δp = (2-7)%12 = 7, q_p = min; Δn = (0-7)%12 = 5, q_n = maj; q_c = dom.

Quality is coarsened to the 5-class functional family used across the project
(harmonia.models.progression_encoder.QUAL5: maj/min/dom/hdim/dim).

Phase 1 scope: corpus assembly + count tables + the two cheap evals (symbolic
infilling recall, theory sanity panel).  NO serving/UI, NO span-lattice
decode — that is Phase 2+ per the design doc's integration section.

What this module does NOT solve
--------------------------------
  * No root PRIOR beyond what the training corpora attest — an utterly novel
    (Δp,Δn,q_p,q_n) combination gets its mass from backoff/root-evidence
    weighting (see score_candidates), not from music theory encoded by hand.
  * Corpus is pooled (jazz-heavy accomp_db + pop-heavy POP909/ChoCo); a
    genre-conditioned table is explicitly left for later (design doc).
  * This does not integrate with the live decode path in any way — see the
    design doc's "Integration" section for the Phase-2 span-lattice plan.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

# Inlined 2026-07-31 (progression_encoder / chord_vocabulary are on the
# minimal rebuild's do-not-import list); byte-identical to the originals.
QUAL5 = ["maj", "min", "dom", "hdim", "dim"]
QUAL5_IDX = {q: i for i, q in enumerate(QUAL5)}

# ── Q8: the extended quality vocabulary (2026-08-06) ────────────────────────
#
# WHY. `/api/context_rescore` decoded only QUAL5, so any span the re-score
# moved came back stripped of its seventh: a `C-7` returned `C:min` and the
# shell rendered `Cm`. Sevenths are not decoration in this repertoire.
#
# WHY EIGHT AND NOT EIGHTEEN. Per-class counts over the pooled corpus, taken
# from the RAW quality strings (not through any index mapping — see the note on
# _load_accomp_db below for why that distinction cost a day):
#
#     maj 54.5%   min 20.8%   dom7 8.2%   min7 7.6%
#     sus  4.9%   maj7 3.2%   dim 0.65%   hdim7 0.28%
#
# A "Q16" arm that also splits 6ths and 9ths adds eight classes of which seven
# sit under 0.5% (maj6 1.27%, dom9 0.51%, min9 0.47%, maj9 0.34%, min6 0.23%,
# aug 0.20%, dom13 0.10%, minmaj7 0.05%). A trigram table cannot estimate those,
# and they would dilute the classes that carry the repertoire. Q8 is also
# exactly enough for the user-visible goal: maj7/min7/dom7 ARE the sevenths.
#
# dim7 is folded into dim on purpose: 0.41% + 0.24% does not buy two classes,
# and QUAL5 already treats them as one. hdim7 stays despite 0.28% because it
# already exists today and is functionally distinct (the ii-half-diminished of
# a minor ii-V) — folding it would be a regression against what ships.
QUAL8 = ["maj", "maj7", "min", "min7", "dom7", "sus", "dim", "hdim7"]
QUAL8_IDX = {q: i for i, q in enumerate(QUAL8)}

#: Q8 -> QUAL5, the backoff step. `sus` folds to `maj` because that is the
#: doubly-confirmed project convention (chord_pipeline_v1._HARTE_TO_Q5NAME and
#: progression_encoder.FINE_TO_QUAL5 independently agree).
QUAL8_TO_QUAL5 = {"maj": "maj", "maj7": "maj", "sus": "maj",
                  "min": "min", "min7": "min",
                  "dom7": "dom", "dim": "dim", "hdim7": "hdim"}
Q8_TO_Q5_IDX = np.array([QUAL5_IDX[QUAL8_TO_QUAL5[q]] for q in QUAL8],
                        dtype=np.int64)
SEMITONE_NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]

REPO = Path(__file__).resolve().parent.parent

N_Q5 = 5
N_ROOT = 12
N_CAND = N_ROOT * N_Q5  # 60 — the QUAL5 shape, kept for back-compat


def n_qual() -> int:
    """Classes in the ACTIVE vocabulary. Every table shape derives from this.

    Hard-coding 5 is how a vocabulary switch half-lands: the counts move to 8
    classes while the arrays stay 5 wide, and numpy either broadcasts or throws
    somewhere far from the cause.
    """
    return len(VOCABULARIES[_VOCAB_NAME][2])


def n_cand() -> int:
    return N_ROOT * n_qual()

_BACKOFF_K = 5.0  # Witten-Bell pseudo-count; same constant/style as
                  # harmonia/models/section_structure.py::_BACKOFF_K.

# ── quality vocabulary: Harte-format token -> q5 family ────────────────────
#
# aug -> maj and sus2/sus4 -> maj are an existing, DOUBLY-confirmed project
# convention: harmonia/models/chord_pipeline_v1.py::_HARTE_TO_Q5NAME (the live
# path's family/seventh classifier) and
# harmonia/models/progression_encoder.py::FINE_TO_QUAL5 (the fine MMA-bucket
# table) independently agree on both folds. Reused here verbatim, extended
# with tokens that appear in POP909/ChoCo but not in either existing table
# (9th/11th/13th extensions fold to their underlying 7th-chord family — the
# project's quality vocabulary stops at 5 functional families, so an
# extension carries no information beyond its parent).
HARTE_QUAL_TO_QUAL5: dict[str, str] = {
    "maj": "maj", "maj7": "maj", "maj6": "maj", "maj9": "maj", "maj13": "maj",
    "aug": "maj", "augmaj7": "maj", "sus2": "maj", "sus4": "maj",
    "min": "min", "min7": "min", "min6": "min", "min9": "min", "min11": "min",
    "min13": "min", "minmaj7": "min",
    "7": "dom", "9": "dom", "11": "dom", "13": "dom", "aug7": "dom",
    "hdim7": "hdim",
    "dim": "dim", "dim7": "dim",
}

#: Same domain as HARTE_QUAL_TO_QUAL5 — every key present in one is present in
#: the other, asserted below — only the target granularity changes. Extensions
#: still fold to their parent seventh (a maj9 is a maj7 with a colour tone);
#: only the seventh itself is now preserved.
HARTE_QUAL_TO_QUAL8: dict[str, str] = {
    "maj": "maj", "maj6": "maj", "aug": "maj",
    "maj7": "maj7", "maj9": "maj7", "maj13": "maj7", "augmaj7": "maj7",
    "sus2": "sus", "sus4": "sus",
    "min": "min", "min6": "min",
    "min7": "min7", "min9": "min7", "min11": "min7", "min13": "min7",
    "minmaj7": "min7",
    "7": "dom7", "9": "dom7", "11": "dom7", "13": "dom7", "aug7": "dom7",
    "hdim7": "hdim7",
    "dim": "dim", "dim7": "dim",
}
assert set(HARTE_QUAL_TO_QUAL8) == set(HARTE_QUAL_TO_QUAL5), (
    "the two quality tables must accept exactly the same labels, or the Q8 arm "
    "silently trains on a different corpus than the QUAL5 arm it is compared to")
assert all(QUAL8_TO_QUAL5[v] == HARTE_QUAL_TO_QUAL5[k]
           for k, v in HARTE_QUAL_TO_QUAL8.items()), (
    "Q8 must refine QUAL5, never disagree with it: backing off from a Q8 class "
    "has to land on the QUAL5 class the same label would have had")
# Deliberately UNMAPPED (dropped as unparseable -> breaks the sequence, counted
# in the per-corpus dropped-label stats, never silently folded):
#   "5"   power chord (root+5th, no 3rd)      -- major/minor genuinely undetermined
#   "1"   bare root / unison, no 3rd/5th at all
#   Harte explicit interval-list chords, e.g. "C:(3,5,b7,b9)" -- no shorthand
#   quality name to look up.
# No existing project precedent forces a fold for these (unlike aug/sus), and
# silently defaulting either to "maj" would inject a false label into the
# count tables — exactly the silent-calibration-bug pattern CLAUDE.md warns
# about (verified: an earlier draft of parse_harte_lite here DID silently
# fold interval-list chords to "maj" via an empty-string default; caught by
# re-checking ChoCo's dropped-label counts against raw source before this
# module's counts were built — see the report for both counts).

NOTE_TO_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}

# ── the one switch every loader must obey ───────────────────────────────────
#
# THE BUG THIS EXISTS TO PREVENT (found 2026-08-06, cost the whole vocabulary
# screen). The corpus has three loaders. Two went through `parse_harte_lite`;
# `_load_accomp_db` did not — it called `progression_encoder.fine_to_q5`, a
# hard-wired QUAL5 map. The screen that chose the vocabulary swapped
# `parse_harte_lite` to change the class alphabet, so 26.8% of the pooled
# corpus (92,268 tokens — jazz1460 + pop400 + blues50, the MOST seventh-rich
# source) kept emitting QUAL5 indices into a stream the other two were filling
# with Q8/Q16 indices. Two incompatible index spaces, pooled, silently: index 2
# meant "dom" from one loader and "dim" from another. Nothing raised, and the
# resulting per-class table read 8.4% augmented chords.
#
# The vocabulary is therefore module state, every loader reads it, and
# `load_all_corpus_sequences` asserts afterwards that no token escaped its
# range. That assertion is the actual protection — the fix without it would
# just be the same bug waiting for the next loader.
VOCABULARIES = {
    "q5": (HARTE_QUAL_TO_QUAL5, QUAL5_IDX, QUAL5),
    "q8": (HARTE_QUAL_TO_QUAL8, QUAL8_IDX, QUAL8),
}
_VOCAB_NAME = os.environ.get("HARMONIA_CHORD_VOCAB", "q8")
_ACTIVE_VOCAB = VOCABULARIES[_VOCAB_NAME][:2]


def active_vocab() -> tuple[str, list[str]]:
    """(name, class list) currently in force — for reports and assertions."""
    return _VOCAB_NAME, VOCABULARIES[_VOCAB_NAME][2]


def set_vocab(name: str) -> None:
    """Switch the quality alphabet for EVERY loader at once.

    Never monkeypatch `parse_harte_lite` to do this: `_load_accomp_db` does not
    call it, and that is exactly how the screen came to compare two arms that
    were sharing a quarter of their tokens.
    """
    global _VOCAB_NAME, _ACTIVE_VOCAB
    if name not in VOCABULARIES:
        raise ValueError(f"unknown vocabulary {name!r}; have {sorted(VOCABULARIES)}")
    _VOCAB_NAME = name
    _ACTIVE_VOCAB = VOCABULARIES[name][:2]


#: accomp_db speaks MMA "fine" bucket names, not Harte. Same target classes.
FINE_TO_QUAL8: dict[str, str] = {
    "maj": "maj", "6": "maj", "aug": "maj",
    "maj7": "maj7", "augmaj7": "maj7",
    "sus2": "sus", "sus4": "sus", "7sus4": "sus",
    "min": "min", "m6": "min",
    "min7": "min7", "minmaj7": "min7",
    "dom7": "dom7", "dom7alt": "dom7", "aug7": "dom7",
    "m7b5": "hdim7",
    "dim": "dim", "dim7": "dim",
}


def fine_to_active(qual: str) -> int | None:
    """MMA fine bucket -> index in the ACTIVE vocabulary, or None if unmapped."""
    if _VOCAB_NAME == "q8":
        name = FINE_TO_QUAL8.get(qual)
        return None if name is None else QUAL8_IDX[name]
    from harmonia.models.progression_encoder import fine_to_q5
    return fine_to_q5(qual)

_DISPLAY_SUFFIX = {"maj": "maj", "min": "m7", "dom": "7", "hdim": "m7b5", "dim": "dim7"}


def parse_harte_lite(label: str) -> tuple[int, int] | None:
    """Minimal Harte-ish chord label -> (root_pc, qual5_idx), or None.

    None covers no-chord ("N"), ambiguous ("X"), and any quality token not in
    HARTE_QUAL_TO_QUAL5 — callers MUST treat None as a sequence break, not a
    skip (see module docstring / corpus-assembly section).

    Deliberately NOT reusing harmonia.data.pop909_parser.parse_harte_label: it
    silently defaults an unrecognised quality to ChordQuality.MAJOR ("Fall
    back ... defaulting to maj"), which is exactly the kind of silent
    calibration bug this project has been burned by before (CLAUDE.md rule 1).
    Verified against raw source lines for POP909 and ChoCo (this report's
    verification tables) before use.

    Bass/inversion notation (``/3``, ``/5``, ``/b7``, absolute-note bass) is
    ignored — this model only ever reasons about the chord ROOT, per the
    design doc's target-relative representation.
    """
    label = label.strip()
    if label in ("N", "n", "NC", "X", "x", "?", ""):
        return None
    pc = NOTE_TO_PC.get(label[0].upper())
    if pc is None:
        return None
    i = 1
    while i < len(label) and label[i] in "#b":
        pc += 1 if label[i] == "#" else -1
        i += 1
    rest = label[i:]
    if rest.startswith(":"):
        qual_str = rest[1:].split("/")[0].split("(")[0]
        if qual_str == "":
            return None  # colon present but quality unresolvable (e.g. interval list)
    else:
        qual_str = "maj"  # Harte convention: quality omitted entirely -> major triad
    table, idx = _ACTIVE_VOCAB
    name = table.get(qual_str)
    if name is None:
        return None
    return pc % 12, idx[name]


def chord_name(root_pc: int, q5_idx: int) -> str:
    """Human-readable label for a (root, q5) token, e.g. (7, 2) -> 'G7'.

    One canonical 7th-chord spelling per family (maj/m7/7/m7b5/dim7) — the
    model only knows 5 functional families, not the triad-vs-7th distinction,
    so there is no principled way to pick a triad spelling for some families
    and not others; this is a display convention, documented once here.
    """
    return f"{SEMITONE_NAMES[root_pc % 12]}{_DISPLAY_SUFFIX[active_vocab()[1][q5_idx]]}"


def cand_index(root_pc: int, q5_idx: int) -> int:
    return root_pc * n_qual() + q5_idx


def cand_decode(idx: int) -> tuple[int, int]:
    return divmod(idx, n_qual())


def trigram_key(
    prev: tuple[int, int], cand: tuple[int, int], nxt: tuple[int, int]
) -> tuple[int, int, int, int, int]:
    """(q_c, Δp, q_p, Δn, q_n) for prev=(root_p,q_p), cand=(root_c,q_c), nxt=(root_n,q_n).

    Transposition-invariant by construction: transposing every root in
    (prev, cand, nxt) by the same amount leaves Δp/Δn (differences) and the
    qualities unchanged.
    """
    root_p, q_p = prev
    root_c, q_c = cand
    root_n, q_n = nxt
    dp = (root_p - root_c) % 12
    dn = (root_n - root_c) % 12
    return q_c, dp, q_p, dn, q_n


def _dedupe(seq: list[tuple[int, int]]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for tok in seq:
        if not out or out[-1] != tok:
            out.append(tok)
    return out


# ── corpus assembly ─────────────────────────────────────────────────────────
# Global song ids are prefixed by source ("accomp:...", "pop909:...",
# "choco:<partition>:...") so ids never collide across corpora and the
# eval script can split its accuracy report by family from the id alone.

def _segments_from_labels(
    labels: list[str], corpus_tag: str, dropped: Counter
) -> list[list[tuple[int, int]]]:
    """Split a raw per-song label stream into break-respecting segments.

    A "break" (N / X / unmapped label) ends the current segment WITHOUT
    bridging across it — the next segment starts fresh at the next parseable
    label.  Each returned segment has consecutive-identical tokens collapsed
    (change-to-change transitions only).
    """
    segments: list[list[tuple[int, int]]] = []
    cur: list[tuple[int, int]] = []
    for lab in labels:
        parsed = parse_harte_lite(lab)
        if parsed is None:
            if lab in ("N", "n", "NC", ""):
                dropped[f"{corpus_tag}:N"] += 1
            elif lab in ("X", "x", "?"):
                dropped[f"{corpus_tag}:X"] += 1
            else:
                dropped[f"{corpus_tag}:unmapped:{lab}"] += 1
            if cur:
                segments.append(_dedupe(cur))
                cur = []
            continue
        cur.append(parsed)
    if cur:
        segments.append(_dedupe(cur))
    return segments


def _load_accomp_db(
    db_path: Path, corpora: tuple[str, ...] = ("jazz1460", "pop400", "blues50")
) -> tuple[dict[str, list[list[tuple[int, int]]]], dict]:
    """accomp_db/db.jsonl -> {song_id: [segment, ...]}, stats.

    Reuses scripts.analyze_accomp_emission.song_chord_spans + progression_
    encoder.fine_to_q5 (verified against raw 'mma' chord-timeline tokens for
    one song per corpus in this report before trusting bulk counts). MMA
    accompaniment charts have no "N"/no-chord concept, so each song is one
    segment — EXCEPT the 2 songs (of 1856) whose 'unmapped_tokens' field is
    non-empty: song_chord_spans silently skips (bridges across) an unparseable
    MMA token rather than breaking the sequence there, so those 2 songs are
    dropped whole rather than relying on that bridging behaviour.
    """
    sys.path.insert(0, str(REPO / "scripts"))
    from analyze_accomp_emission import song_chord_spans  # noqa: E402

    songs: dict[str, list[list[tuple[int, int]]]] = {}
    stats = {
        "n_records": 0, "n_kept_songs": 0, "n_dropped_short": 0,
        "n_skipped_unmapped_token_songs": 0, "n_tokens": 0,
        "dropped_labels": Counter(),
    }
    for line in open(db_path):
        rec = json.loads(line)
        if rec.get("corpus") not in corpora:
            continue
        stats["n_records"] += 1
        if rec.get("unmapped_tokens"):
            stats["n_skipped_unmapped_token_songs"] += 1
            for tok in rec["unmapped_tokens"]:
                stats["dropped_labels"][f"accomp:unmapped_mma:{tok}"] += 1
            continue
        seq: list[tuple[int, int]] = []
        for _t0, _t1, root, qual in song_chord_spans(rec):
            q = fine_to_active(qual)     # NOT fine_to_q5: see set_vocab's note
            if q is None:
                stats["dropped_labels"][f"accomp:unmapped_fine:{qual}"] += 1
                continue
            seq.append((root % 12, q))
        deduped = _dedupe(seq)
        if len(deduped) < 4:
            stats["n_dropped_short"] += 1
            continue
        songs[f"accomp:{rec['song_id']}"] = [deduped]
        stats["n_kept_songs"] += 1
        stats["n_tokens"] += len(deduped)
    return songs, stats


def _load_pop909(pop909_dir: Path) -> tuple[dict[str, list[list[tuple[int, int]]]], dict]:
    """POP909 chord_midi.txt files -> {song_id: [segment, ...]}, stats."""
    songs: dict[str, list[list[tuple[int, int]]]] = {}
    stats = {
        "n_records": 0, "n_kept_songs": 0, "n_dropped_short": 0,
        "n_tokens": 0, "dropped_labels": Counter(),
    }
    for song_dir in sorted(p for p in Path(pop909_dir).iterdir() if p.is_dir()):
        chord_path = song_dir / "chord_midi.txt"
        if not chord_path.exists():
            continue
        stats["n_records"] += 1
        labels = []
        for line in open(chord_path):
            parts = line.strip().split()
            if len(parts) < 3:
                continue
            labels.append(parts[2])
        segments = _segments_from_labels(labels, "pop909", stats["dropped_labels"])
        total = sum(len(s) for s in segments)
        if total < 4:
            stats["n_dropped_short"] += 1
            continue
        songs[f"pop909:{song_dir.name}"] = segments
        stats["n_kept_songs"] += 1
        stats["n_tokens"] += total
    return songs, stats


CHOCO_PARTITIONS = (
    "billboard", "isophonics", "jaah", "robbie-williams", "rwc-pop", "uspop2002",
)


def _load_choco(
    choco_root: Path, partitions: tuple[str, ...] = CHOCO_PARTITIONS
) -> tuple[dict[str, list[list[tuple[int, int]]]], dict]:
    """ChoCo JAMS 'chord' namespace -> {song_id: [segment, ...]}, stats."""
    songs: dict[str, list[list[tuple[int, int]]]] = {}
    stats = {
        "n_records": 0, "n_kept_songs": 0, "n_dropped_short": 0,
        "n_unreadable": 0, "n_no_chord_ns": 0, "n_tokens": 0,
        "dropped_labels": Counter(), "per_partition": {},
    }
    for part in partitions:
        jdir = Path(choco_root) / part / "choco" / "jams"
        if not jdir.exists():
            stats["per_partition"][part] = "missing"
            continue
        n_kept = 0
        for jp in sorted(jdir.glob("*.jams")):
            stats["n_records"] += 1
            try:
                j = json.loads(jp.read_text())
            except Exception:
                stats["n_unreadable"] += 1
                continue
            ch = [a for a in j.get("annotations", []) if a.get("namespace") == "chord"]
            if not ch or not ch[0].get("data"):
                stats["n_no_chord_ns"] += 1
                continue
            labels = [d["value"] for d in sorted(ch[0]["data"], key=lambda d: d["time"])]
            segments = _segments_from_labels(labels, f"choco:{part}", stats["dropped_labels"])
            total = sum(len(s) for s in segments)
            if total < 4:
                stats["n_dropped_short"] += 1
                continue
            songs[f"choco:{part}:{jp.stem}"] = segments
            stats["n_kept_songs"] += 1
            stats["n_tokens"] += total
            n_kept += 1
        stats["per_partition"][part] = n_kept
    return songs, stats


#: corpus-arm name -> which of the three loaders feed it (genre-arm experiment,
#: docs/context_prior_phase1_results.md "Genre-arm experiment"). jazz = accomp_db
#: only (the jazz-standard/accompaniment corpus); pop = POP909 + ChoCo (pop909 is
#: MIDI-derived pop, choco's kept partitions are billboard/isophonics/rwc-pop/
#: uspop2002/robbie-williams pop-rock plus jaah, which is jazz-labelled but small
#: enough that mislabelling it "pop" here does not materially change the arm —
#: see the genre-arm doc section for the exact per-partition token counts).
CORPUS_SOURCES: dict[str, tuple[str, ...]] = {
    "pooled": ("accomp_db", "pop909", "choco"),
    "jazz": ("accomp_db",),
    "pop": ("pop909", "choco"),
}


def load_all_corpus_sequences(
    accomp_db_path: str | Path | None = None,
    pop909_dir: str | Path | None = None,
    choco_dir: str | Path | None = None,
    sources: tuple[str, ...] = ("accomp_db", "pop909", "choco"),
) -> tuple[dict[str, list[list[tuple[int, int]]]], dict]:
    """Corpora pooled (or a subset via ``sources``), keyed by a global
    (source-prefixed) song id.

    ``sources`` restricts which of the three loaders run — e.g. ``("accomp_db",)``
    for a jazz-only table, ``("pop909", "choco")`` for a pop-only table (see
    ``CORPUS_SOURCES``, used by ``build_context_prior(corpus=...)``). Missing
    requested sources are reported (not silently skipped) via
    stats[...]["error"]; sources not requested are simply absent from stats.
    """
    accomp_db_path = Path(accomp_db_path) if accomp_db_path else REPO / "data/accomp_db/db.jsonl"
    pop909_dir = Path(pop909_dir) if pop909_dir else REPO / "data/pop909/POP909"
    choco_dir = Path(choco_dir) if choco_dir else REPO / "data/cache/choco/partitions"

    songs: dict[str, list[list[tuple[int, int]]]] = {}
    stats: dict = {}

    if "accomp_db" in sources:
        if accomp_db_path.exists():
            s, st = _load_accomp_db(accomp_db_path)
            songs.update(s)
            stats["accomp_db"] = st
        else:
            stats["accomp_db"] = {"error": f"missing: {accomp_db_path}"}

    if "pop909" in sources:
        if pop909_dir.exists():
            s, st = _load_pop909(pop909_dir)
            songs.update(s)
            stats["pop909"] = st
        else:
            stats["pop909"] = {"error": f"missing: {pop909_dir}"}

    if "choco" in sources:
        if choco_dir.exists():
            s, st = _load_choco(choco_dir)
            songs.update(s)
            stats["choco"] = st
        else:
            stats["choco"] = {"error": f"missing: {choco_dir}"}

    # THE GUARD. Every loader must have emitted indices in the ACTIVE
    # vocabulary's range. Without this, a loader that quietly keeps its own
    # class alphabet pools incompatible indices into one table and nothing
    # complains — which is precisely what `_load_accomp_db` did for the whole
    # first vocabulary screen (26.8% of tokens, silently QUAL5, while the rest
    # of the stream had moved to Q8/Q16). Per-source, so the message names the
    # culprit instead of just failing.
    name, classes = active_vocab()
    n_classes = len(classes)
    per_source_max: dict[str, int] = {}
    for sid, seqs in songs.items():
        src = sid.split(":", 1)[0]
        for seq in seqs:
            for _root, q in seq:
                if q > per_source_max.get(src, -1):
                    per_source_max[src] = q
    bad = {s: m for s, m in per_source_max.items() if m >= n_classes}
    if bad:
        raise AssertionError(
            f"vocabulary {name!r} has {n_classes} classes but these loaders "
            f"emitted out-of-range quality indices: {bad}")
    stats["vocab"] = {"name": name, "n_classes": n_classes,
                      "max_index_per_source": per_source_max}
    return songs, stats


def _is_heldout(song_id: str, denom: int = 10) -> bool:
    """Deterministic 1-in-``denom`` held-out split by stable hash of song id."""
    h = int(hashlib.sha1(song_id.encode()).hexdigest(), 16)
    return h % denom == 0


def _stats_jsonable(stats: dict) -> dict:
    """Recursively convert Counters to plain dicts (json.dumps-safe), and
    trim each dropped_labels Counter to its 40 most common entries (full
    per-song detail is not needed downstream, and some corpora have long
    tails of one-off unmapped labels)."""
    out = {}
    for k, v in stats.items():
        if isinstance(v, Counter):
            out[k] = dict(v.most_common(40))
        elif isinstance(v, dict):
            out[k] = _stats_jsonable(v)
        else:
            out[k] = v
    return out


class ContextPriorModel(dict):
    """A fitted chord-context-prior table: dict-like (unchanged) PLUS the
    object-style API the design doc's "Phase 2 wiring contract" specifies.

    This IS a dict subclass, not a wrapper around one — every existing
    ``model["tri"]`` / ``model.get("heldout_ids")`` / ``"stats" in model``
    access (build_context_prior's own return, this module's tests, the Phase
    1 eval scripts, the genre-arm eval script) keeps working with ZERO
    changes, because that access already goes straight to dict methods.

    Added 2026-07-30 (genre-arm session) because ``harmonia/models/
    span_rescore.py`` (the Phase 2 lattice-rescore scaffold, merged onto this
    branch concurrently) calls ``context_scorer.score_candidates(prev, next)``
    as a bound method on whatever ``load_context_prior()`` returns — see its
    ``load_context_scorer()`` / ``_ctx_logp()``. A raw dict has no such
    method, so span_rescore's defensive try/except was silently degrading to
    its uniform (lam-inert) stub even though Phase 1 had already landed. This
    class is the minimal fix: bind the module-level ``score_candidates`` to
    ``self`` as the model.
    """

    def score_candidates(
        self, prev_token: tuple[int, int] | None, next_token: tuple[int, int] | None,
    ) -> np.ndarray:
        """(60,) normalized array — see module-level ``score_candidates``."""
        return score_candidates(prev_token, next_token, model=self)


# ── count tables ─────────────────────────────────────────────────────────────

def build_context_prior(
    cache_path: str | Path | None = None,
    accomp_db_path: str | Path | None = None,
    pop909_dir: str | Path | None = None,
    choco_dir: str | Path | None = None,
    heldout_denom: int = 10,
    corpus: str = "pooled",
) -> ContextPriorModel:
    """Build the target-relative trigram/bigram/unigram count tables.

    ``corpus`` selects which sources feed the table (genre-arm experiment,
    docs/context_prior_phase1_results.md "Genre-arm experiment"):
        "pooled" (default, unchanged Phase-1 behaviour) — accomp_db + POP909 + ChoCo.
        "jazz"   — accomp_db only.
        "pop"    — POP909 + ChoCo only.
    Held-out split is computed AFTER the source filter, over whatever songs
    that corpus subset contains — so "jazz" held-out songs are a subset of
    accomp_db, never touched by pop training counts and vice versa.

    Returns a dict with:
        tri     : (5,12,5,12,5) raw counts, key (q_c,Δp,q_p,Δn,q_n).
        prevbi  : (5,12,5) raw counts, key (q_c,Δp,q_p) — directed prev-only bigram.
        nextbi  : (5,12,5) raw counts, key (q_c,Δn,q_n) — directed next-only bigram.
        uni     : (5,) raw counts over q_c.
        train_ids / heldout_ids : sorted lists of global song ids.
        corpus  : the corpus arm this table was built from.
        stats   : per-corpus load/parse/drop counters (JSON-safe).

    Held-out songs never contribute a single count (split happens before any
    table is touched). Persists to ``cache_path`` (npz) when given.
    """
    if corpus not in CORPUS_SOURCES:
        raise ValueError(f"corpus must be one of {sorted(CORPUS_SOURCES)}, got {corpus!r}")
    sources = CORPUS_SOURCES[corpus]
    songs, load_stats = load_all_corpus_sequences(
        accomp_db_path, pop909_dir, choco_dir, sources=sources
    )
    train_ids = sorted(sid for sid in songs if not _is_heldout(sid, heldout_denom))
    heldout_ids = sorted(sid for sid in songs if _is_heldout(sid, heldout_denom))

    NQ = n_qual()
    tri = np.zeros((NQ, N_ROOT, NQ, N_ROOT, NQ), dtype=np.float64)
    prevbi = np.zeros((NQ, N_ROOT, NQ), dtype=np.float64)
    nextbi = np.zeros((NQ, N_ROOT, NQ), dtype=np.float64)
    uni = np.zeros(NQ, dtype=np.float64)

    for sid in train_ids:
        for seg in songs[sid]:
            for tok in seg:
                uni[tok[1]] += 1
            for i in range(len(seg) - 1):
                a, b = seg[i], seg[i + 1]
                dp = (a[0] - b[0]) % 12
                prevbi[b[1], dp, a[1]] += 1
                dn = (b[0] - a[0]) % 12
                nextbi[a[1], dn, b[1]] += 1
            for i in range(len(seg) - 2):
                p, c, n = seg[i], seg[i + 1], seg[i + 2]
                qc, dp, qp, dn, qn = trigram_key(p, c, n)
                tri[qc, dp, qp, dn, qn] += 1

    model = ContextPriorModel({
        "tri": tri.astype(np.float32),
        "prevbi": prevbi.astype(np.float32),
        "nextbi": nextbi.astype(np.float32),
        "uni": uni.astype(np.float32),
        "train_ids": train_ids,
        "heldout_ids": heldout_ids,
        "corpus": corpus,
        "stats": load_stats,
    })

    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            tri=model["tri"], prevbi=model["prevbi"], nextbi=model["nextbi"], uni=model["uni"],
            train_ids=np.array(train_ids, dtype=object),
            heldout_ids=np.array(heldout_ids, dtype=object),
            corpus=np.array(corpus),
            stats_json=np.array(json.dumps(_stats_jsonable(load_stats))),
        )
    return model


def _default_cache_path_for(corpus: str) -> Path:
    """data/cache/chord_context_prior.npz for the pooled arm (Phase-1 path,
    unchanged so existing callers/caches keep working); data/cache/
    chord_context_prior_{corpus}.npz for the jazz/pop genre arms."""
    if corpus == "pooled":
        return REPO / "data" / "cache" / "chord_context_prior.npz"
    return REPO / "data" / "cache" / f"chord_context_prior_{corpus}.npz"


def load_context_prior(
    cache_path: str | Path | None = None, rebuild: bool = False, corpus: str = "pooled",
) -> ContextPriorModel:
    """Load the cached count-table model, building + caching it if missing.

    Mirrors harmonia/models/section_structure.py::load_progression_model.
    Zero-arg call is UNCHANGED from Phase 1's dict-shaped contents: still
    returns the pooled table from data/cache/chord_context_prior.npz, still
    indexable exactly like the old dict (``model["tri"]`` etc). The RETURN
    TYPE changed 2026-07-30 (genre-arm session) from a plain dict to
    ``ContextPriorModel`` (a dict subclass) so callers using the object-style
    API from the design doc's "Phase 2 wiring contract" — ``model.
    score_candidates(prev, next)`` — get a real bound method instead of an
    AttributeError (span_rescore.py's scaffold depends on this). Pass
    ``corpus="jazz"`` or ``corpus="pop"`` for a genre-restricted table, cached
    at a distinct path (see ``_default_cache_path_for``); an explicit
    ``cache_path`` always wins over the corpus-derived default.
    """
    if corpus not in CORPUS_SOURCES:
        raise ValueError(f"corpus must be one of {sorted(CORPUS_SOURCES)}, got {corpus!r}")
    cache_path = Path(cache_path) if cache_path else _default_cache_path_for(corpus)
    if cache_path.exists() and not rebuild:
        try:
            z = np.load(cache_path, allow_pickle=True)
            if all(k in z.files for k in ("tri", "prevbi", "nextbi", "uni")):
                return ContextPriorModel({
                    "tri": z["tri"].astype(np.float64),
                    "prevbi": z["prevbi"].astype(np.float64),
                    "nextbi": z["nextbi"].astype(np.float64),
                    "uni": z["uni"].astype(np.float64),
                    "train_ids": list(z["train_ids"].tolist()) if "train_ids" in z.files else [],
                    "heldout_ids": list(z["heldout_ids"].tolist()) if "heldout_ids" in z.files else [],
                    "corpus": str(z["corpus"]) if "corpus" in z.files else corpus,
                    "stats": json.loads(str(z["stats_json"])) if "stats_json" in z.files else {},
                })
        except Exception:
            pass
    return build_context_prior(cache_path=cache_path, corpus=corpus)


# ── scoring ──────────────────────────────────────────────────────────────────

_default_model: dict | None = None


def _get_default_model() -> dict:
    global _default_model
    if _default_model is None:
        _default_model = load_context_prior()
    return _default_model


def score_candidates(
    prev_token: tuple[int, int] | None,
    next_token: tuple[int, int] | None,
    model: dict | None = None,
) -> np.ndarray:
    """P(candidate | prev, next) over the 60-candidate space (12 roots x 5 quals).

    ``prev_token``/``next_token`` are (root_pc, qual5_idx) or None (missing
    neighbour — falls back to the single available directed bigram; both None
    falls back to the flat unigram spread uniformly over roots).

    Smoothing (Witten-Bell-style, matching section_structure.py's style):
      1. Per candidate ROOT r, the prev-only and next-only directed bigrams
         are each independently backed off to the quality unigram:
         ``p = λ·MLE + (1-λ)·unigram``, ``λ = c/(c+K)``.
      2. The two backed-off bigram estimates are combined by GEOMETRIC MEAN
         (renormalised), not arithmetic mean.  Justification: prev-context and
         next-context are two separate, roughly-conditionally-independent
         observations of the SAME target quality, so combining them is a
         product-of-experts / naive-Bayes fusion (AND-like: both neighbours
         must agree to push mass onto a candidate) — exactly what a ii-V-I
         needs, since neither neighbour alone implies "V7", only their
         conjunction does. Arithmetic mean is the right tool one level up
         (step 3: blending trigram and backoff, two estimates of the SAME
         conditioning event at different orders), which is why that step
         below stays linear, matching section_structure.py's precedent.
      3. The trigram MLE for that root (when available) is linearly
         interpolated with the combined-bigram estimate from step 2.
      4. The per-root quality distribution from step 3 is weighted by that
         root's total evidence (trigram context count + both bigram context
         counts, +epsilon so a totally unseen root still gets a small,
         non-zero, numerically-safe share) before the full (root, quality)
         array is normalised to sum to 1. Steps 1-3 alone would give every
         root — seen or not — equal total mass (each root's distribution
         over the 5 qualities sums to 1 by construction), which would erase
         all root-level information the tables actually contain; step 4
         restores it. This is a documented heuristic (not a claimed-exact
         Bayesian posterior) that composes a smoothed quality-shape with a
         raw-evidence root-plausibility weight.
    """
    model = model if model is not None else _get_default_model()
    tri, prevbi, nextbi, uni = model["tri"], model["prevbi"], model["nextbi"], model["uni"]
    uni_total = uni.sum()
    NQ = uni.shape[0]
    uni_p = uni / uni_total if uni_total > 0 else np.full(NQ, 1.0 / NQ)
    K = _BACKOFF_K

    scores = np.zeros((N_ROOT, NQ), dtype=np.float64)

    for r in range(N_ROOT):
        p_prev = c_p = None
        if prev_token is not None:
            root_p, q_p = prev_token
            dp = (root_p - r) % 12
            raw_p = prevbi[:, dp, q_p].astype(np.float64)
            c_p = float(raw_p.sum())
            ml_p = raw_p / c_p if c_p > 0 else uni_p
            lam_p = c_p / (c_p + K)
            p_prev = lam_p * ml_p + (1 - lam_p) * uni_p

        p_next = c_n = None
        if next_token is not None:
            root_n, q_n = next_token
            dn = (root_n - r) % 12
            raw_n = nextbi[:, dn, q_n].astype(np.float64)
            c_n = float(raw_n.sum())
            ml_n = raw_n / c_n if c_n > 0 else uni_p
            lam_n = c_n / (c_n + K)
            p_next = lam_n * ml_n + (1 - lam_n) * uni_p

        if p_prev is not None and p_next is not None:
            p_bi = np.sqrt(p_prev * p_next)
            p_bi = p_bi / p_bi.sum()
            root_evidence = c_p + c_n
        elif p_prev is not None:
            p_bi, root_evidence = p_prev, c_p
        elif p_next is not None:
            p_bi, root_evidence = p_next, c_n
        else:
            p_bi, root_evidence = uni_p, 0.0

        if prev_token is not None and next_token is not None:
            raw_t = tri[:, dp, q_p, dn, q_n].astype(np.float64)
            c_t = float(raw_t.sum())
            ml_t = raw_t / c_t if c_t > 0 else p_bi
            lam_t = c_t / (c_t + K)
            p_ctx = lam_t * ml_t + (1 - lam_t) * p_bi
            root_evidence += c_t
        else:
            p_ctx = p_bi

        scores[r, :] = (root_evidence + 1e-6) * p_ctx

    flat = scores.reshape(-1)  # cand_index(r, qi) == r*N_Q5+qi matches this order
    total = flat.sum()
    return flat / total if total > 0 else np.full(flat.size, 1.0 / flat.size)


def top_candidates(
    prev_token: tuple[int, int] | None,
    next_token: tuple[int, int] | None,
    k: int = 8,
    model: dict | None = None,
) -> list[dict]:
    """Top-``k`` candidates as human-readable dicts, sorted by probability."""
    probs = score_candidates(prev_token, next_token, model=model)
    order = np.argsort(probs)[::-1][:k]
    out = []
    for idx in order:
        root, qi = cand_decode(int(idx))
        out.append({
            "root": root, "q5": active_vocab()[1][qi], "name": chord_name(root, qi),
            "prob": round(float(probs[idx]), 4),
        })
    return out
