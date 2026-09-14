"""Shared helpers for Mission-3 confidence calibration (fused score + ECE).

This module backs ``scripts/calibrate_quality.py`` (fit) and
``scripts/eval_calibration.py`` (measure). Both operate on the same primitive:
per-output-chord ``(score, correct)`` pairs harvested from the REAL production
pipeline (``infer_chords_v1``), so nothing here forks the model.

The fused score (Mission 3, issue #29)
--------------------------------------
The displayed confidence must fold BOTH heads' uncertainty:

    fused_raw = confidence_raw * root_conf
              = max_quality_softmax * root_posterior@label_root

- ``confidence_raw`` is the QUALITY head's max soft-prob (``conf`` in the
  ChordChart; the joint marginal on the joint path).
- ``root_conf`` is the span-mean beat-sequence root posterior AT the label's
  root pc (``_span_root_conf``). It is the "honest" root-side confidence: a
  confidently-wrong root shrinks the fused score even when the quality head is
  sure. Falls back to ``confidence_raw`` alone when the root model is off.

The OLD ``real`` map (issue #26/#29) fit ``confidence_raw`` alone — root-blind.
Mission 3 refits on ``fused_raw`` so the root-blindness fix (#26) also holds on
the real-audio path.

Ground-truth targets
--------------------
``correct`` = predicted (root pc AND q5 family maj/min/dom/hdim/dim) both match
the GT chord at the output chord's midpoint. This is "the chord the app shows is
right at the granularity it shows it".

Benchmark schema (Mission 1 output — not yet built)
---------------------------------------------------
``load_benchmark(path)`` reads the non-circular real-audio benchmark. Expected
JSON (``data/real_audio_benchmark/aligned_chords_per_song.json``), tolerant to a
few shapes::

    {"songs": [
       {"song_id": "...", "audio_path": "data/cache/.../x.m4a",
        "chords": [{"start_s": 5.0, "end_s": 6.5, "label": "G:min7"}, ...]},
       ...]}

Each GT chord needs an audio-time span (``start_s``/``end_s``, aliases
``t0``/``t1``) and one of: explicit ``root_pc`` (0-11) + ``q5`` (family name);
or a ``label`` parsed by :func:`parse_gt_label` (Harte ``root:sev`` or a common
chord symbol like ``Gm7``/``C7``/``Fmaj7``). Until Mission 1 emits this file the
loaders raise a clear "waiting on Mission 1" error.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.models import chord_pipeline_v1 as P  # noqa: E402

NOTE_TO_PC = {n: i for i, n in enumerate(P.NOTE)}
# accidental spellings the pipeline's NOTE table (sharps only) does not list
_FLAT_TO_PC = {"Db": 1, "Eb": 3, "Gb": 6, "Ab": 8, "Bb": 10,
               "Cb": 11, "Fb": 4, "E#": 5, "B#": 0}

BENCH_DEFAULT = REPO / "data" / "real_audio_benchmark" / "aligned_chords_per_song.json"


# ── GT label parsing ──────────────────────────────────────────────────────────
def _root_to_pc(root: str) -> int | None:
    if root in NOTE_TO_PC:
        return NOTE_TO_PC[root]
    return _FLAT_TO_PC.get(root)


_SYMBOL_RE = re.compile(r"^([A-G][#b]?)(.*)$")


def parse_gt_label(label: str) -> tuple[int, str] | None:
    """(root_pc, q5_family) from a GT chord label, or None if unparseable.

    Handles Harte ``root:sev`` (e.g. ``G:min7``) and plain jazz symbols
    (``Gm7``, ``C7``, ``Fmaj7``, ``Bhalfdim``…). q5 family ∈ maj/min/dom/hdim/dim.
    """
    if not label or label in ("N", "NC", "N.C."):
        return None
    if ":" in label:  # Harte
        root, sev = label.split(":", 1)
        pc = _root_to_pc(root)
        q5i = P._harte_to_q5idx(sev)
        if pc is None or q5i is None:
            return None
        return pc, P._Q5_NAMES[q5i]
    m = _SYMBOL_RE.match(label.strip())
    if not m:
        return None
    pc = _root_to_pc(m.group(1))
    if pc is None:
        return None
    q = m.group(2).lower().replace("-", "min").replace("Δ", "maj")
    # order matters: check the more specific families first
    if any(t in q for t in ("dim7", "°7", "dim", "°")) and "half" not in q:
        fam = "dim"
    elif any(t in q for t in ("m7b5", "min7b5", "halfdim", "ø", "hdim")):
        fam = "hdim"
    elif q.startswith("maj") or "maj7" in q or "major" in q or q in ("", "6", "add9"):
        fam = "maj"
    elif q.startswith("min") or q.startswith("m") or "minor" in q:
        fam = "min"
    elif q.startswith("7") or "9" in q or "13" in q or "11" in q or "dom" in q:
        fam = "dom"
    elif q.startswith("sus") or q.startswith("aug") or q.startswith("+"):
        return None  # sus/aug are not in q5 — skip (matches corpus handling)
    else:
        fam = "maj"
    return pc, fam


# ── benchmark loader ──────────────────────────────────────────────────────────
class MissingBenchmark(FileNotFoundError):
    pass


def load_benchmark(path: Path | None = None) -> list[dict]:
    """Return ``[{song_id, audio_path (Path), spans: [(t0,t1,root_pc,q5)]}]``.

    Raises :class:`MissingBenchmark` with a Mission-1 pointer if the file is
    absent — this is the expected pre-Mission-1 state.
    """
    path = Path(path) if path is not None else BENCH_DEFAULT
    if not path.exists():
        raise MissingBenchmark(
            f"real-audio benchmark not found at {path}. This is Mission 1's "
            f"deliverable (data/real_audio_benchmark/). Run Mission 1 first, or "
            f"pass --benchmark to point at the aligned-chords JSON.")
    raw = json.loads(path.read_text())
    songs = raw["songs"] if isinstance(raw, dict) and "songs" in raw else raw
    if isinstance(songs, dict):  # keyed by song_id
        songs = [{"song_id": k, **v} for k, v in songs.items()]
    out = []
    for s in songs:
        ap = s.get("audio_path") or s.get("audio")
        if ap is None:
            continue
        ap = Path(ap)
        if not ap.is_absolute():
            ap = REPO / ap
        spans = []
        for c in s.get("chords", []):
            t0 = c.get("start_s", c.get("t0"))
            t1 = c.get("end_s", c.get("t1"))
            if t0 is None or t1 is None or t1 <= t0:
                continue
            if "root_pc" in c and "q5" in c:
                gt = (int(c["root_pc"]), str(c["q5"]))
            else:
                gt = parse_gt_label(c.get("label", ""))
            if gt is None:
                continue
            spans.append((float(t0), float(t1), gt[0], gt[1]))
        if spans:
            out.append({"song_id": s.get("song_id", ap.stem),
                        "audio_path": ap, "spans": spans})
    if not out:
        raise MissingBenchmark(
            f"benchmark at {path} parsed 0 usable songs — check schema "
            f"(need audio_path + chords[start_s,end_s,label|root_pc+q5]).")
    return out


# ── per-chord score collection from the production pipeline ────────────────────
def collect_pairs(chart, spans) -> list[tuple[float, float, bool]]:
    """(confidence_raw, fused_raw, correct) per output chord of ``chart``.

    ``fused_raw = confidence_raw * root_conf`` (root_conf→1 fallback when the
    root model is off). ``correct`` = pred root pc AND q5 family match GT at the
    chord midpoint. ``spans`` = GT ``[(t0,t1,root_pc,q5)]``.
    """
    rows = []
    for c in chart.chords:
        label = c["label"]
        if ":" not in label:
            continue
        mid = 0.5 * (c["start_s"] + c["end_s"])
        gt = next(((r, q) for t0, t1, r, q in spans if t0 <= mid < t1), None)
        if gt is None:
            continue
        pred_pc = NOTE_TO_PC.get(label.split(":", 1)[0])
        q5i = P._harte_to_q5idx(label.split(":", 1)[1])
        pred_fam = P._Q5_NAMES[q5i] if q5i is not None else None
        if pred_pc is None or pred_fam is None:
            continue
        raw = float(c["confidence_raw"])
        rc = c.get("root_conf")
        fused = raw * float(rc) if rc is not None else raw
        correct = (pred_pc == gt[0]) and (pred_fam == gt[1])
        rows.append((raw, fused, correct))
    return rows


def collect_benchmark(songs, *, cache_dir=None, audio_domain="real",
                      progress=True) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run the pipeline on benchmark ``songs`` → (conf_raw, fused_raw, correct).

    Uses the retrained quality head automatically (the pipeline loads
    ``ctx_v2.npz`` / ``ctx_v3.npz`` from disk — Mission 2 replaces those files;
    no code change needed here). ``audio_domain`` selects only the display map,
    not the labels, so the harvested RAW scores are calibration-map-independent.
    """
    import tempfile

    tmpdir = None
    if cache_dir is None:
        tmpdir = tempfile.TemporaryDirectory()
        cache_dir = Path(tmpdir.name)
    raws, fused, corr = [], [], []
    try:
        for i, s in enumerate(songs):
            ap = s["audio_path"]
            if not ap.exists():
                if progress:
                    print(f"  [{i+1}/{len(songs)}] {s['song_id']}: audio missing "
                          f"({ap}) — skipped", flush=True)
                continue
            if progress:
                print(f"  [{i+1}/{len(songs)}] {s['song_id']}", flush=True)
            chart = P.infer_chords_v1(ap, cache_dir=Path(cache_dir),
                                      audio_domain=audio_domain)
            for r, f, c in collect_pairs(chart, s["spans"]):
                raws.append(r)
                fused.append(f)
                corr.append(c)
    finally:
        if tmpdir is not None:
            tmpdir.cleanup()
    return (np.asarray(raws, float), np.asarray(fused, float),
            np.asarray(corr, bool))


# ── ECE / reliability ─────────────────────────────────────────────────────────
def ece_bins(conf: np.ndarray, correct: np.ndarray, n_bins: int = 10,
             min_count: int = 10):
    """(rows, ECE). Rows = (lo, hi, mean_conf, mean_acc, n) per populated bin.

    Same binning convention as scripts/plot_calibration.py and the existing
    fit_confidence_calibration*.py so numbers are comparable across the project.
    """
    conf = np.asarray(conf, float)
    correct = np.asarray(correct, float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows, ece = [], 0.0
    n = len(conf)
    for i in range(n_bins):
        hi_cmp = (conf < edges[i + 1]) if i < n_bins - 1 else (conf <= 1.0)
        m = (conf >= edges[i]) & hi_cmp
        if m.sum() >= min_count:
            rows.append((edges[i], edges[i + 1], float(conf[m].mean()),
                         float(correct[m].mean()), int(m.sum())))
            ece += m.sum() / n * abs(conf[m].mean() - correct[m].mean())
    return rows, ece


def report_reliability(name: str, conf: np.ndarray, correct: np.ndarray,
                       n_bins: int = 10, min_count: int = 10) -> float:
    rows, ece = ece_bins(conf, correct, n_bins, min_count)
    print(f"\n  {name}:  ECE = {ece:.4f}  (n={len(conf)}, "
          f"mean conf={np.mean(conf):.3f}, acc={np.mean(correct):.3f})")
    for lo, hi, c, a, k in rows:
        print(f"    [{lo:.1f},{hi:.1f})  conf={c:.3f}  acc={a:.3f}  n={k}")
    return ece
