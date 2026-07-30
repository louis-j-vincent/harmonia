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
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from harmonia.models.progression_encoder import QUAL5, QUAL5_IDX
from harmonia.theory.chord_vocabulary import SEMITONE_NAMES

REPO = Path(__file__).resolve().parent.parent.parent

N_Q5 = 5
N_ROOT = 12
N_CAND = N_ROOT * N_Q5  # 60 = 12 roots x 5 quality families

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
    q5name = HARTE_QUAL_TO_QUAL5.get(qual_str)
    if q5name is None:
        return None
    return pc % 12, QUAL5_IDX[q5name]


def chord_name(root_pc: int, q5_idx: int) -> str:
    """Human-readable label for a (root, q5) token, e.g. (7, 2) -> 'G7'.

    One canonical 7th-chord spelling per family (maj/m7/7/m7b5/dim7) — the
    model only knows 5 functional families, not the triad-vs-7th distinction,
    so there is no principled way to pick a triad spelling for some families
    and not others; this is a display convention, documented once here.
    """
    return f"{SEMITONE_NAMES[root_pc % 12]}{_DISPLAY_SUFFIX[QUAL5[q5_idx]]}"


def cand_index(root_pc: int, q5_idx: int) -> int:
    return root_pc * N_Q5 + q5_idx


def cand_decode(idx: int) -> tuple[int, int]:
    return divmod(idx, N_Q5)


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
    from harmonia.models.progression_encoder import fine_to_q5

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
            q5 = fine_to_q5(qual)
            if q5 is None:
                stats["dropped_labels"][f"accomp:unmapped_fine:{qual}"] += 1
                continue
            seq.append((root % 12, q5))
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


def load_all_corpus_sequences(
    accomp_db_path: str | Path | None = None,
    pop909_dir: str | Path | None = None,
    choco_dir: str | Path | None = None,
) -> tuple[dict[str, list[list[tuple[int, int]]]], dict]:
    """All three corpora pooled, keyed by a global (source-prefixed) song id.

    Missing sources are reported (not silently skipped) via stats[...]["error"].
    """
    accomp_db_path = Path(accomp_db_path) if accomp_db_path else REPO / "data/accomp_db/db.jsonl"
    pop909_dir = Path(pop909_dir) if pop909_dir else REPO / "data/pop909/POP909"
    choco_dir = Path(choco_dir) if choco_dir else REPO / "data/cache/choco/partitions"

    songs: dict[str, list[list[tuple[int, int]]]] = {}
    stats: dict = {}

    if accomp_db_path.exists():
        s, st = _load_accomp_db(accomp_db_path)
        songs.update(s)
        stats["accomp_db"] = st
    else:
        stats["accomp_db"] = {"error": f"missing: {accomp_db_path}"}

    if pop909_dir.exists():
        s, st = _load_pop909(pop909_dir)
        songs.update(s)
        stats["pop909"] = st
    else:
        stats["pop909"] = {"error": f"missing: {pop909_dir}"}

    if choco_dir.exists():
        s, st = _load_choco(choco_dir)
        songs.update(s)
        stats["choco"] = st
    else:
        stats["choco"] = {"error": f"missing: {choco_dir}"}

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


# ── count tables ─────────────────────────────────────────────────────────────

def build_context_prior(
    cache_path: str | Path | None = None,
    accomp_db_path: str | Path | None = None,
    pop909_dir: str | Path | None = None,
    choco_dir: str | Path | None = None,
    heldout_denom: int = 10,
) -> dict:
    """Build the target-relative trigram/bigram/unigram count tables.

    Returns a dict with:
        tri     : (5,12,5,12,5) raw counts, key (q_c,Δp,q_p,Δn,q_n).
        prevbi  : (5,12,5) raw counts, key (q_c,Δp,q_p) — directed prev-only bigram.
        nextbi  : (5,12,5) raw counts, key (q_c,Δn,q_n) — directed next-only bigram.
        uni     : (5,) raw counts over q_c.
        train_ids / heldout_ids : sorted lists of global song ids.
        stats   : per-corpus load/parse/drop counters (JSON-safe).

    Held-out songs never contribute a single count (split happens before any
    table is touched). Persists to ``cache_path`` (npz) when given.
    """
    songs, load_stats = load_all_corpus_sequences(accomp_db_path, pop909_dir, choco_dir)
    train_ids = sorted(sid for sid in songs if not _is_heldout(sid, heldout_denom))
    heldout_ids = sorted(sid for sid in songs if _is_heldout(sid, heldout_denom))

    tri = np.zeros((N_Q5, N_ROOT, N_Q5, N_ROOT, N_Q5), dtype=np.float64)
    prevbi = np.zeros((N_Q5, N_ROOT, N_Q5), dtype=np.float64)
    nextbi = np.zeros((N_Q5, N_ROOT, N_Q5), dtype=np.float64)
    uni = np.zeros(N_Q5, dtype=np.float64)

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

    model = {
        "tri": tri.astype(np.float32),
        "prevbi": prevbi.astype(np.float32),
        "nextbi": nextbi.astype(np.float32),
        "uni": uni.astype(np.float32),
        "train_ids": train_ids,
        "heldout_ids": heldout_ids,
        "stats": load_stats,
    }

    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            cache_path,
            tri=model["tri"], prevbi=model["prevbi"], nextbi=model["nextbi"], uni=model["uni"],
            train_ids=np.array(train_ids, dtype=object),
            heldout_ids=np.array(heldout_ids, dtype=object),
            stats_json=np.array(json.dumps(_stats_jsonable(load_stats))),
        )
    return model


def load_context_prior(cache_path: str | Path | None = None, rebuild: bool = False) -> dict:
    """Load the cached count-table model, building + caching it if missing.

    Mirrors harmonia/models/section_structure.py::load_progression_model.
    Defaults the cache to data/cache/chord_context_prior.npz.
    """
    cache_path = Path(cache_path) if cache_path else REPO / "data" / "cache" / "chord_context_prior.npz"
    if cache_path.exists() and not rebuild:
        try:
            z = np.load(cache_path, allow_pickle=True)
            if all(k in z.files for k in ("tri", "prevbi", "nextbi", "uni")):
                return {
                    "tri": z["tri"].astype(np.float64),
                    "prevbi": z["prevbi"].astype(np.float64),
                    "nextbi": z["nextbi"].astype(np.float64),
                    "uni": z["uni"].astype(np.float64),
                    "train_ids": list(z["train_ids"].tolist()) if "train_ids" in z.files else [],
                    "heldout_ids": list(z["heldout_ids"].tolist()) if "heldout_ids" in z.files else [],
                    "stats": json.loads(str(z["stats_json"])) if "stats_json" in z.files else {},
                }
        except Exception:
            pass
    return build_context_prior(cache_path=cache_path)


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
    uni_p = uni / uni_total if uni_total > 0 else np.full(N_Q5, 1.0 / N_Q5)
    K = _BACKOFF_K

    scores = np.zeros((N_ROOT, N_Q5), dtype=np.float64)

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
    return flat / total if total > 0 else np.full(N_CAND, 1.0 / N_CAND)


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
            "root": root, "q5": QUAL5[qi], "name": chord_name(root, qi),
            "prob": round(float(probs[idx]), 4),
        })
    return out
