#!/usr/bin/env python3
"""Mission 4 — automatic section-merge detection (repeat detection + scoring).

Issue #28 (measured 2026-07-13) showed that pooling Basic-Pitch evidence across
repeats of the SAME chord within a song lifts the quality head's accuracy
43.8 → 53.8% (+10pp) on REAL audio — the first real-audio confirmation of the
Mission-3 pooled-emission claim.  The pooling mechanism already exists
(``harmonia.models.user_constraints.SectionMerge`` → ``pool_beat_evidence``,
wired into ``chord_pipeline_v1.infer_chords_v1(user_constraints=...)``) but only
fires on an explicit USER assertion.

This module fires it AUTOMATICALLY, but only where a repeat is confidently
detected — never a blind average (blind averaging was Gen-1 "Candidate C" and it
hurt; see docs/known_issues.md).  A candidate merge is emitted only if BOTH:

  * **structural confidence** — the two spans are the same SECTION (equal musical
    length AND their decoded chord sequences agree), and
  * **acoustic agreement** — the two spans SOUND the same (their per-beat chroma
    is highly correlated),

clear ``threshold`` (default 0.75).  A candidate that overlaps a span the user
has already asserted a merge on is dropped (user assertion is stronger).

The output is a list of merge dicts ``{"spans": [[t0,t1],[t0,t1]], ...}`` ready
to hand straight to ``infer_chords_v1(user_constraints={"merges": ...})``.

Pure over a ChordChart + (optional) audio — no model state — so the scoring
logic is cheap to unit-test in isolation (see ``_pair_scores``).

CLI:
  .venv/bin/python scripts/detect_auto_merges.py CHART.json --audio SONG.wav
  .venv/bin/python scripts/detect_auto_merges.py --self-test
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.models.chord_pipeline_v1 import NOTE, _harte_to_q5idx  # noqa: E402


# ── chart → per-beat symbolic chord lookup ────────────────────────────────────

def _label_to_root_q5(label: str) -> tuple[int, int] | None:
    """(root_pc, q5idx) of a chart label like ``'Bb:min7'``; None if unparseable
    or a no-chord ('N'/'X')."""
    if not label or ":" not in label:
        return None
    name, sev = label.split(":", 1)
    try:
        r = NOTE.index(name)
    except ValueError:
        return None
    qi = _harte_to_q5idx(sev)
    return (r, qi) if qi is not None else None


def _chord_at(chords: list[dict], t: float) -> tuple[int, int] | None:
    """(root, q5) predicted at time ``t`` (seconds) from a chart's chord list."""
    lab = None
    for c in chords:
        if c["start_s"] <= t < c["end_s"]:
            lab = c["label"]
            break
        if c["start_s"] <= t:
            lab = c["label"]
    return _label_to_root_q5(lab) if lab else None


# ── structural confidence (do the two spans carry the same progression?) ──────

def _structural_confidence(
    chords: list[dict], span_a: tuple[float, float], span_b: tuple[float, float],
    n_samples: int,
) -> float:
    """Fraction of aligned sample points where the two spans' decoded chords
    match on (root, q5).

    Both spans are sampled at ``n_samples`` evenly-spaced RELATIVE offsets (so a
    tempo-identical repeat aligns slot-for-slot even if the two spans are not the
    exact same number of seconds).  A sample where either side has no parseable
    chord counts as a mismatch (conservative — an unknown span should not be
    merged).  Returns 0.0 for a degenerate (zero-length) span.
    """
    a0, a1 = span_a
    b0, b1 = span_b
    if a1 <= a0 or b1 <= b0 or n_samples < 1:
        return 0.0
    match = 0
    for k in range(n_samples):
        frac = (k + 0.5) / n_samples
        ga = _chord_at(chords, a0 + frac * (a1 - a0))
        gb = _chord_at(chords, b0 + frac * (b1 - b0))
        if ga is not None and gb is not None and ga == gb:
            match += 1
    return match / n_samples


# ── acoustic agreement (do the two spans sound the same?) ─────────────────────

def _pool_chroma_span(chroma: np.ndarray, times: np.ndarray,
                      t0: float, t1: float) -> np.ndarray | None:
    """Mean chroma (12,) over frames in [t0, t1); None if the window is empty."""
    m = (times >= t0) & (times < t1)
    if m.sum() == 0:
        return None
    return chroma[m].mean(axis=0)


def _acoustic_agreement(
    chroma: np.ndarray, times: np.ndarray,
    span_a: tuple[float, float], span_b: tuple[float, float],
    n_slots: int,
) -> float:
    """Mean per-slot cosine similarity of the two spans' chroma profiles.

    Each span is sliced into ``n_slots`` equal RELATIVE windows; the two spans'
    slot-aligned mean-chroma vectors are compared by Pearson-style (mean-removed)
    cosine — the same geometry the DTW aligner uses, which cancels the full-mix
    DC chroma floor.  Returns the mean similarity over valid slots (in [0,1],
    negatives clipped to 0), or 0.0 if no slot is comparable.
    """
    a0, a1 = span_a
    b0, b1 = span_b
    if a1 <= a0 or b1 <= b0 or n_slots < 1:
        return 0.0
    sims: list[float] = []
    for k in range(n_slots):
        fa0, fa1 = a0 + k / n_slots * (a1 - a0), a0 + (k + 1) / n_slots * (a1 - a0)
        fb0, fb1 = b0 + k / n_slots * (b1 - b0), b0 + (k + 1) / n_slots * (b1 - b0)
        va = _pool_chroma_span(chroma, times, fa0, fa1)
        vb = _pool_chroma_span(chroma, times, fb0, fb1)
        if va is None or vb is None:
            continue
        va = va - va.mean()
        vb = vb - vb.mean()
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na < 1e-9 or nb < 1e-9:
            continue
        sims.append(float(va @ vb / (na * nb)))
    if not sims:
        return 0.0
    return float(np.clip(np.mean(sims), 0.0, 1.0))


def _load_audio_chroma(audio_path: Path, sr: int = 22050,
                       hop: int = 1024) -> tuple[np.ndarray, np.ndarray]:
    """(chroma[M,12], frame_times[M]) CQT chromagram of an audio file."""
    import librosa
    y, _ = librosa.load(audio_path, sr=sr)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop).T
    t = librosa.frames_to_time(np.arange(chroma.shape[0]), sr=sr, hop_length=hop)
    return chroma.astype(np.float32), t.astype(np.float32)


# ── candidate detection ───────────────────────────────────────────────────────

@dataclass
class MergeCandidate:
    span_a: tuple[float, float]
    span_b: tuple[float, float]
    label: str                 # section label the pair shares (e.g. "A")
    struct_conf: float
    acoustic_conf: float
    fired: bool                # both thresholds cleared and not user-owned

    def as_merge(self) -> dict:
        return {"spans": [list(self.span_a), list(self.span_b)],
                "struct_conf": round(self.struct_conf, 4),
                "acoustic_conf": round(self.acoustic_conf, 4)}


def _spans_overlap(a: tuple[float, float], b: tuple[float, float]) -> bool:
    return a[0] < b[1] and b[0] < a[1]


def _pair_scores(
    chords: list[dict], sec_a: dict, sec_b: dict,
    chroma: np.ndarray | None, times: np.ndarray | None,
) -> tuple[float, float]:
    """(structural_conf, acoustic_conf) for a candidate section pair.

    Sample count = 4·bars (one slot per beat in 4/4), floored at 4.  Acoustic
    conf is 0.0 (never fires alone) when no chroma is supplied.
    """
    span_a = (sec_a["start_s"], sec_a["end_s"])
    span_b = (sec_b["start_s"], sec_b["end_s"])
    n = max(4, 4 * int(sec_a.get("n_bars", 1)))
    struct = _structural_confidence(chords, span_a, span_b, n)
    acoustic = (0.0 if chroma is None
                else _acoustic_agreement(chroma, times, span_a, span_b, n))
    return struct, acoustic


def detect_auto_merges(
    chart,
    *,
    audio_path: Path | None = None,
    chroma: np.ndarray | None = None,
    chroma_times: np.ndarray | None = None,
    struct_threshold: float = 0.75,
    acoustic_threshold: float = 0.75,
    user_merges: list | None = None,
    require_equal_bars: bool = True,
) -> list[MergeCandidate]:
    """Detect section pairs safe to auto-merge (pool) on a decoded ChordChart.

    Gating (all must hold for ``fired=True``):
      1. two sections carry the SAME label (same-section by the SSM labeller),
      2. equal musical length (equal ``n_bars`` — ``pool_beat_evidence`` rejects
         unequal beat counts, so this is required for the merge to take effect),
      3. structural_conf > ``struct_threshold`` (decoded progressions agree),
      4. acoustic_conf > ``acoustic_threshold`` (chroma sounds the same), and
      5. neither span overlaps a span in ``user_merges`` (user assertion wins).

    Acoustic evidence comes from ``chroma``/``chroma_times`` if supplied, else it
    is computed from ``audio_path`` (a CQT chromagram); with neither, acoustic
    conf is 0 and nothing fires (fail-safe: no evidence ⇒ no auto-merge).

    Returns ALL evaluated pairs as ``MergeCandidate`` (with ``fired`` flags) so a
    caller/eval can inspect near-misses; ``[c.as_merge() for c in ... if
    c.fired]`` is the list to feed to ``infer_chords_v1``.
    """
    sections = list(getattr(chart, "sections", []) or [])
    chords = list(getattr(chart, "chords", []) or [])
    if chroma is None and audio_path is not None:
        chroma, chroma_times = _load_audio_chroma(Path(audio_path))

    # user-asserted merge spans (any overlap ⇒ user owns it)
    user_spans: list[tuple[float, float]] = []
    for m in (user_merges or []):
        spans = m.get("spans", []) if isinstance(m, dict) else getattr(m, "spans", [])
        for s in spans:
            user_spans.append((float(s[0]), float(s[1])))

    # group section indices by label
    by_label: dict[str, list[int]] = {}
    for i, sec in enumerate(sections):
        by_label.setdefault(sec.get("label", "A"), []).append(i)

    out: list[MergeCandidate] = []
    for label, idxs in by_label.items():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                sec_a, sec_b = sections[idxs[a]], sections[idxs[b]]
                if require_equal_bars and sec_a.get("n_bars") != sec_b.get("n_bars"):
                    continue
                struct, acoustic = _pair_scores(
                    chords, sec_a, sec_b, chroma, chroma_times)
                span_a = (sec_a["start_s"], sec_a["end_s"])
                span_b = (sec_b["start_s"], sec_b["end_s"])
                user_owned = any(
                    _spans_overlap(span_a, us) or _spans_overlap(span_b, us)
                    for us in user_spans)
                fired = (struct > struct_threshold
                         and acoustic > acoustic_threshold
                         and not user_owned)
                out.append(MergeCandidate(
                    span_a=span_a, span_b=span_b, label=label,
                    struct_conf=struct, acoustic_conf=acoustic, fired=fired))
    return out


def fired_merges(candidates: list[MergeCandidate]) -> list[dict]:
    """Extract the ``infer_chords_v1``-ready merge dicts from fired candidates."""
    return [c.as_merge() for c in candidates if c.fired]


# ── self-test (no audio, no model) ────────────────────────────────────────────

def _self_test() -> int:
    """Exercise the pure scoring/gating logic on a synthetic AABA chart."""
    from harmonia.pipeline import ChordChart

    # AABA: A = |C:maj|G:maj| repeated, B = |D:min|E:min|.  Two A sections at
    # 0-4s and 8-12s should merge; the B section (4-8s) should not pair with A.
    def bar(t, lab):
        return {"label": lab, "start_s": t, "end_s": t + 2.0,
                "duration_beats": 4, "confidence": 0.9}

    chords = [
        bar(0, "C:maj"), bar(2, "G:maj"),          # A1
        bar(4, "D:min"), bar(6, "E:min"),          # B
        bar(8, "C:maj"), bar(10, "G:maj"),         # A2
    ]
    sections = [
        {"start_s": 0.0, "end_s": 4.0, "n_bars": 2, "label": "A"},
        {"start_s": 4.0, "end_s": 8.0, "n_bars": 2, "label": "B"},
        {"start_s": 8.0, "end_s": 12.0, "n_bars": 2, "label": "A"},
    ]
    chart = ChordChart(
        source_path="synthetic", duration_s=12.0, tempo_bpm=120.0,
        time_signature="4/4", global_key="C", global_key_confidence=0.9,
        style="test", modulations=[], chords=chords, segments=[],
        sections=sections)

    # Fake chroma: identical for the two A spans, different for B.
    sr_t = np.arange(0.0, 12.0, 0.05)
    chroma = np.zeros((len(sr_t), 12), dtype=np.float32)
    for i, t in enumerate(sr_t):
        pcs = {0, 4, 7} if (t < 4 or t >= 8) else {2, 5, 9}  # C-ish vs D-ish
        for pc in pcs:
            chroma[i, pc] = 1.0

    ok = True

    # 1. With matching chroma, the two A sections fire.
    cands = detect_auto_merges(chart, chroma=chroma, chroma_times=sr_t)
    aa = [c for c in cands if c.label == "A"]
    assert len(aa) == 1, f"expected 1 A-pair, got {len(aa)}"
    c = aa[0]
    print(f"  A-A pair: struct={c.struct_conf:.2f} acoustic={c.acoustic_conf:.2f} "
          f"fired={c.fired}")
    ok &= c.struct_conf == 1.0 and c.acoustic_conf > 0.9 and c.fired
    assert len(fired_merges(cands)) == 1

    # 2. Structural mismatch (edit A2's chords) must block the merge.
    chart.chords[4]["label"] = "F:maj"
    chart.chords[5]["label"] = "A:min"
    cands2 = detect_auto_merges(chart, chroma=chroma, chroma_times=sr_t)
    c2 = [c for c in cands2 if c.label == "A"][0]
    print(f"  struct-mismatch: struct={c2.struct_conf:.2f} fired={c2.fired}")
    ok &= c2.struct_conf < 0.75 and not c2.fired
    chart.chords[4]["label"] = "C:maj"
    chart.chords[5]["label"] = "G:maj"

    # 3. No chroma / no audio ⇒ acoustic 0 ⇒ never fires (fail-safe).
    cands3 = detect_auto_merges(chart)
    ok &= all(not c.fired for c in cands3)
    print(f"  no-audio fail-safe: fired={sum(c.fired for c in cands3)} (want 0)")

    # 4. User assertion over span A1 blocks the auto-merge on that pair.
    cands4 = detect_auto_merges(
        chart, chroma=chroma, chroma_times=sr_t,
        user_merges=[{"spans": [[0.0, 4.0], [8.0, 12.0]]}])
    c4 = [c for c in cands4 if c.label == "A"][0]
    ok &= not c4.fired
    print(f"  user-owned block: fired={c4.fired} (want False)")

    print("SELF-TEST:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("chart", nargs="?", help="ChordChart JSON (from save_json)")
    ap.add_argument("--audio", type=Path, help="audio file for acoustic agreement")
    ap.add_argument("--struct-threshold", type=float, default=0.75)
    ap.add_argument("--acoustic-threshold", type=float, default=0.75)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test or not args.chart:
        return _self_test()

    from harmonia.pipeline import ChordChart
    data = json.loads(Path(args.chart).read_text())
    chart = ChordChart(
        source_path=data.get("source", args.chart),
        duration_s=data.get("duration_s", 0.0),
        tempo_bpm=data.get("tempo_bpm", 120.0),
        time_signature=data.get("time_signature", "4/4"),
        global_key=data.get("global_key", "C"),
        global_key_confidence=data.get("global_key_confidence", 0.0),
        style=data.get("style", "v1"), modulations=data.get("modulations", []),
        chords=data.get("chords", []), segments=data.get("segments", []),
        sections=data.get("sections", []))

    cands = detect_auto_merges(
        chart, audio_path=args.audio,
        struct_threshold=args.struct_threshold,
        acoustic_threshold=args.acoustic_threshold)
    print(f"{len(chart.sections)} sections, {len(cands)} same-label equal-length "
          f"pairs evaluated:")
    for c in cands:
        flag = "FIRE" if c.fired else "skip"
        print(f"  [{flag}] {c.label}: {c.span_a[0]:.1f}-{c.span_a[1]:.1f}s "
              f"<-> {c.span_b[0]:.1f}-{c.span_b[1]:.1f}s  "
              f"struct={c.struct_conf:.2f} acoustic={c.acoustic_conf:.2f}")
    merges = fired_merges(cands)
    print(f"\n{len(merges)} auto-merge(s) would fire:")
    print(json.dumps(merges, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
