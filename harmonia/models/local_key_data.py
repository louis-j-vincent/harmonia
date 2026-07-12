"""harmonia/models/local_key_data.py — symbolic section-key dataset (issue #20/#23).

Builds a per-section local-key dataset from the iReal corpus (``data/accomp_db/
db.jsonl``). Each *section instance* (a contiguous run of bars sharing one
section label, e.g. the whole ``A`` of an AABA head) becomes one example:

    (chord_sequence: list[(root_pc, qual5_idx)], oracle_key_label: int in 0..23)

The oracle label is produced by a **rules-based** symbolic key estimator (no
learning): a duration-weighted chord-tone chroma of the section is matched
against the 24 Krumhansl major/minor profiles (``theory.key_profiles.infer_key``),
with a **margin gate against the song's global key** so a section is only marked
as *modulated* when its chords are decisively better explained by another key
than by the global one. This encodes the musician's default — hold the home key
until the harmony forces a change — and avoids the noisy per-window ``infer_key``
that made the diatonic prior net-neutral (issue #20).

Deliberately symbolic (chart-only): iReal charts are clean chord sequences, so
chord tones are exact; the noisy-audio-chroma problem is a *phase-2* concern
(MMA renders / YouTube), not handled here.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from ..theory.key_profiles import infer_key
from ..theory.local_key import chord_pcs, parse_token
from .progression_encoder import QUAL5_IDX, fine_to_q5

# ── quality mapping (iReal token -> 5-class family) ────────────────────────────
# We reuse progression_encoder's FINE_TO_QUAL5, but that keys on the *fine* MMA
# quality bucket. The db carries the raw iReal token; local_key.quality_class
# already maps an iReal quality tail to a coarse functional class, and it maps
# 1:1 onto the q5 vocabulary used everywhere else.
from ..theory.local_key import quality_class  # noqa: E402

_QCLASS_TO_Q5 = {
    "maj": "maj", "min": "min", "dom": "dom", "sus": "dom",
    "m7b5": "hdim", "dim": "dim",
}


def token_to_q5(token: str) -> int | None:
    """iReal token -> 5-class family index (maj/min/dom/hdim/dim), or None."""
    _root, qual, _bass = parse_token(token)
    fam = _QCLASS_TO_Q5.get(quality_class(qual))
    return QUAL5_IDX[fam] if fam is not None else None


# ── global-key parsing (db 'key' field, e.g. 'C', 'Bb', 'E-', 'F#-') ───────────
_LETTER = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}
_KEY_RE = re.compile(r"^([A-G])([b#]?)(-?)$")


def parse_global_key(key: str) -> tuple[int, str] | None:
    """db 'key' string -> (tonic_pc, mode). '-' suffix = minor. None if unparsable."""
    m = _KEY_RE.match(key.strip())
    if not m:
        return None
    letter, acc, minus = m.groups()
    pc = (_LETTER[letter] + (1 if acc == "#" else -1 if acc == "b" else 0)) % 12
    return pc, ("minor" if minus else "major")


def key_to_idx(tonic: int, mode: str) -> int:
    """(tonic_pc, mode) -> 0..23 index (0-11 major, 12-23 minor)."""
    return tonic % 12 + (12 if mode == "minor" else 0)


def idx_to_key(idx: int) -> tuple[int, str]:
    return idx % 12, ("minor" if idx >= 12 else "major")


# ── section instances (contiguous same-label bar runs) ─────────────────────────
def section_instances(rec: dict) -> list[dict]:
    """Split a db record into contiguous-section-label instances.

    Returns a list of {label, bar_lo, bar_hi (inclusive, 1-indexed),
    tokens: [(ireal_token, duration_beats)]}. Duration is beats-to-next-chord,
    matching scripts.analyze_accomp_emission.song_chord_spans.
    """
    spb = rec["section_per_bar"]
    bpb = rec["beats_per_bar"]
    n_beats = rec["n_bars"] * bpb

    # contiguous runs of identical label -> (label, bar_lo, bar_hi) 1-indexed
    runs: list[tuple[str, int, int]] = []
    for i, lab in enumerate(spb):
        bar = i + 1
        if runs and runs[-1][0] == lab and runs[-1][2] == bar - 1:
            runs[-1] = (lab, runs[-1][1], bar)
        else:
            runs.append((lab, bar, bar))

    # chord events sorted on an absolute beat axis, with durations
    slots = sorted(
        ((ev["bar"] - 1) * bpb + ev["beat"], ev["bar"], ev.get("ireal", ""))
        for ev in rec["chord_timeline"]
    )
    events = []  # (bar, ireal_token, duration_beats)
    for i, (beat, bar, tok) in enumerate(slots):
        end = slots[i + 1][0] if i + 1 < len(slots) else n_beats
        dur = max(end - beat, 0.0)
        events.append((bar, tok, dur))

    out = []
    for lab, lo, hi in runs:
        toks = [(tok, dur) for bar, tok, dur in events if lo <= bar <= hi]
        if toks:
            out.append({"label": lab, "bar_lo": lo, "bar_hi": hi, "tokens": toks})
    return out


# ── rules-based oracle section-key labeler ─────────────────────────────────────
def section_chroma(tokens: list[tuple[str, float]]) -> np.ndarray:
    """Duration-weighted chord-tone chroma of a section, with first/last tonic
    cues (tonal phrases lean on the tonic). Mirrors local_key.estimate_key."""
    chroma = np.zeros(12)
    toks = [t for t, _ in tokens]
    for tok, w in tokens:
        for pc, cw in chord_pcs(tok).items():
            chroma[pc] += cw * max(w, 0.25)
    if toks:
        chroma[parse_token(toks[-1])[0]] += 3.0
        chroma[parse_token(toks[0])[0]] += 1.5
    return chroma


def oracle_section_key(
    tokens: list[tuple[str, float]],
    global_idx: int,
    margin: float = 6.0,
) -> tuple[int, bool]:
    """Rules-based local key for one section.

    Returns (key_idx, modulated). The section is labelled as *modulated* (best
    key != global) only when the best key's log-likelihood beats the global
    key's by >= ``margin`` nats; otherwise it inherits the global key. This is
    the "hold the home key until forced out" gate — conservative by design so
    the modulation rate is a lower bound, not inflated by short ambiguous runs.
    """
    chroma = section_chroma(tokens)
    if chroma.sum() == 0:
        return global_idx, False
    kp = infer_key(chroma)
    best = int(np.argmax(kp.log_probs))
    if best == global_idx:
        return global_idx, False
    if kp.log_probs[best] - kp.log_probs[global_idx] >= margin:
        return best, True
    return global_idx, False


# ── dataset assembly ───────────────────────────────────────────────────────────
POP_CORPORA = {"pop400", "blues50"}
JAZZ_CORPORA = {"jazz1460"}


def build_examples(
    db_path: Path,
    margin: float = 6.0,
    corpora: set[str] | None = None,
) -> list[dict]:
    """One example per section instance across the corpus.

    Each example: {seq: [(root_pc, q5)], y: oracle_key_idx, y_global: global_idx,
    modulated: bool, corpus, song_idx, label}. Sections whose chords yield no
    usable q5 tokens are skipped.
    """
    examples: list[dict] = []
    for song_idx, line in enumerate(open(db_path)):
        rec = json.loads(line)
        if corpora is not None and rec["corpus"] not in corpora:
            continue
        gk = parse_global_key(rec["key"])
        if gk is None:
            continue
        global_idx = key_to_idx(*gk)
        for sec in section_instances(rec):
            seq: list[tuple[int, int]] = []
            for tok, _dur in sec["tokens"]:
                q5 = token_to_q5(tok)
                if q5 is None:
                    continue
                seq.append((parse_token(tok)[0] % 12, q5))
            if not seq:
                continue
            y, modulated = oracle_section_key(sec["tokens"], global_idx, margin)
            examples.append({
                "seq": seq,
                "y": y,
                "y_global": global_idx,
                "modulated": modulated,
                "corpus": rec["corpus"],
                "song_idx": song_idx,
                "label": sec["label"],
            })
    return examples


def split_examples(
    examples: list[dict], val_every: int = 5
) -> tuple[list[dict], list[dict]]:
    """Deterministic train/val split by *song* (no section leakage across split),
    matching progression_encoder's every-5th-song convention."""
    train = [e for e in examples if e["song_idx"] % val_every != 0]
    val = [e for e in examples if e["song_idx"] % val_every == 0]
    return train, val


DEFAULT_DB = Path(__file__).resolve().parents[2] / "data" / "accomp_db" / "db.jsonl"
