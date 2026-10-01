"""Rebuild the musx-learned chord-similarity table (`harmonia/assets/chord_similarity_musx.json`).

    .venv/bin/python -m tools.chord_similarity_musx

What it measures: how often musx itself hesitates between two chords. For
every cached song (`data/cache/musx_probs/*.npz`, one file per audio, the
`__head45` previews skipped), the six musx heads are combined exactly as the
vendored decoder combines them (`XHMMDecoder.get_chord_tag_obs`, the
301-label "submission" vocabulary: N + 25 qualities x 12 roots) and
normalised per frame into a label posterior p_t (temperature 1 — the
decoder's own scale, no knob). Frames where N carries more than half the
mass are dropped. Then

    M[i, j] = sum_t p_t[i] p_t[j]          (mass musx puts on BOTH at once)

is pooled over the 12 transpositions BY INTERVAL — M_pool[qa, qb, d] =
sum_r M[(r, qa), (r + d, qb)] — and normalised like a cosine:

    S[qa, qb, d] = M_pool[qa, qb, d] / sqrt(D[qa] D[qb]),  D[q] = sum_r M[(r,q),(r,q)]

so S = 1 for a label with itself and S = 0 for two labels musx never
entertains at the same instant. Pooling numerators and denominators (not
averaging 12 cosines) weights each root by how often it was actually heard.

Diagnostics stored with the table (split-half reliability: songs split by
parity of their sorted index; library-only vs GuitarSet-only) say whether a
value is a property of musx or noise of one corpus.

Deterministic: sorted file list, fixed frame stride, no randomness.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np

from harmonia import musx
from harmonia.settings import SETTINGS

#: Version of the table's DEFINITION (bump if anything below changes).
VERSION = 1
#: Keep one frame in STRIDE (43 fps -> ~10.8 fps): chords last >= 1 beat.
STRIDE = 4
#: A frame counts only if musx puts at most this much mass on N.
MAX_N = 0.5
QUALITIES = ["maj", "min", "7", "maj7", "min7", "dim", "dim7", "hdim7", "aug",
             "sus2", "sus4", "sus4(b7)", "9", "min9", "maj9", "11", "13",
             "maj/3", "maj/5", "maj/b7", "maj/2", "min/b3", "min/5", "min/b7",
             "min/2"]
NOTES = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
OUT = Path(__file__).resolve().parents[1] / "harmonia" / "assets" / "chord_similarity_musx.json"
_HEADS = ("triad", "bass", "s7", "s9", "s11", "s13")


def corpus_files() -> list[Path]:
    """One npz per audio: previews (`__head45`) skipped; for a stem cached
    twice (`stem.npz` and `stem__<size>.npz`) the shortest name wins."""
    d = SETTINGS.data_cache / "musx_probs"
    by_stem: dict[str, Path] = {}
    for f in sorted(d.glob("*.npz")):
        if "__head" in f.name:
            continue
        stem = f.stem.split("__")[0]
        if stem not in by_stem or len(f.name) < len(by_stem[stem].name):
            by_stem[stem] = f
    return [by_stem[s] for s in sorted(by_stem)]


def is_guitarset(f: Path) -> bool:
    return f.name[:3] in {f"{i:02d}_" for i in range(6)} and "_comp_" in f.name


def _label_index(names: list[str]) -> np.ndarray:
    """(25, 12) -> column in the decoder's label list (-1 if absent)."""
    pc = {n: i for i, n in enumerate(NOTES)}
    pc.update({"Db": 1, "D#": 3, "Gb": 6, "G#": 8, "A#": 10})
    ix = -np.ones((len(QUALITIES), 12), dtype=int)
    for j, name in enumerate(names):
        if name == "N":
            continue
        r, _, q = name.partition(":")
        if q in QUALITIES:
            ix[QUALITIES.index(q), pc[r]] = j
    return ix


def accumulate(files: list[Path]) -> tuple[dict, list[str]]:
    """Per-file M (301x301) summed into groups: all / even / odd / guitarset / other."""
    groups = {g: None for g in ("all", "even", "odd", "guitarset", "other")}
    frames = {g: 0 for g in groups}
    names = None
    with musx._InMusxDir():
        from extractors.xhmm_ismir import XHMMDecoder
        dec = XHMMDecoder(template_file="data/submission_chord_list.txt")
        for k, f in enumerate(files):
            z = np.load(f)
            pl = [z[h][::STRIDE].astype(np.float64) for h in _HEADS]
            nm, lp = dec.get_chord_tag_obs(pl)
            if names is None:
                names = nm
            assert nm == names
            lp = lp - lp.max(1, keepdims=True)
            p = np.exp(lp)
            p /= p.sum(1, keepdims=True)
            p = p[p[:, 0] <= MAX_N]
            m = p.T @ p
            for g in ("all", "even" if k % 2 == 0 else "odd",
                      "guitarset" if is_guitarset(f) else "other"):
                groups[g] = m if groups[g] is None else groups[g] + m
                frames[g] += len(p)
            if k % 25 == 0:
                print(f"  {k}/{len(files)} {f.name}", file=sys.stderr)
    return {"M": groups, "frames": frames}, names


def pooled_similarity(M: np.ndarray, ix: np.ndarray) -> np.ndarray:
    """S[qa, qb, d] (25, 25, 12), see module docstring."""
    nq = len(QUALITIES)
    num = np.zeros((nq, nq, 12))
    diag = np.zeros(nq)
    for qa in range(nq):
        for r in range(12):
            i = ix[qa, r]
            if i < 0:
                continue
            diag[qa] += M[i, i]
            for qb in range(nq):
                for d in range(12):
                    j = ix[qb, (r + d) % 12]
                    if j >= 0:
                        num[qa, qb, d] += M[i, j]
    den = np.sqrt(np.outer(diag, diag))[:, :, None]
    with np.errstate(invalid="ignore", divide="ignore"):
        S = np.where(den > 0, num / den, np.nan)
    return S


def main() -> None:
    files = corpus_files()
    print(f"{len(files)} songs", file=sys.stderr)
    acc, names = accumulate(files)
    ix = _label_index(names)
    tables = {g: pooled_similarity(m, ix) for g, m in acc["M"].items()}
    S = tables["all"]

    def corr(a, b):
        mask = np.isfinite(a) & np.isfinite(b)
        off = mask.copy()
        for q in range(len(QUALITIES)):
            off[q, q, 0] = False
        return float(np.corrcoef(a[off], b[off])[0, 1])

    diag = {
        "split_half_r": round(corr(tables["even"], tables["odd"]), 4),
        "guitarset_vs_other_r": round(corr(tables["guitarset"], tables["other"]), 4),
        "frames": acc["frames"],
        "n_songs": len(files),
        "n_guitarset": sum(is_guitarset(f) for f in files),
        "max_abs_split_diff": round(float(np.nanmax(np.abs(tables["even"] - tables["odd"]))), 4),
    }
    corpus_hash = hashlib.sha1("\n".join(f.name for f in files).encode()).hexdigest()[:12]
    out = {
        "version": VERSION,
        "built": date.today().isoformat(),
        "definition": "S[qa][qb][d] = pooled cosine of musx label posteriors, "
                      "label b rooted d semitones above label a; see tools/chord_similarity_musx.py",
        "stride": STRIDE, "max_n": MAX_N,
        "corpus_sha1": corpus_hash,
        "qualities": QUALITIES,
        "diagnostics": diag,
        "S": np.round(np.nan_to_num(S, nan=0.0), 4).tolist(),
        "S_guitarset": np.round(np.nan_to_num(tables["guitarset"], nan=0.0), 3).tolist(),
        "S_other": np.round(np.nan_to_num(tables["other"], nan=0.0), 3).tolist(),
    }
    OUT.write_text(json.dumps(out, separators=(",", ":")))
    print(json.dumps(diag, indent=1))
    print(f"-> {OUT} ({OUT.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
