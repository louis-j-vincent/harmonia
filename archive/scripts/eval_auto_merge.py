#!/usr/bin/env python3
"""Mission 4 — measure whether AUTOMATIC section-merge helps (issue #28 wiring).

Runs the pipeline twice per song — base (no constraints) and auto-merged
(``detect_auto_merges`` → ``infer_chords_v1(user_constraints={"merges": ...})``)
— and reports majmin / 7ths accuracy before/after, per song and in aggregate.
Expected lift +5..+10pp (CLAUDE.md / issue #28), but NEVER assumed: the whole
point is to measure it, and to report which songs benefited and which did not.

Two data paths:

  * ``--benchmark`` (default): the Mission-1 real-audio benchmark
    (``data/real_audio_benchmark/*.json``).  This is the honest, non-circular
    target.  **Gated on Mission 1**: if the benchmark is not yet built the script
    prints a clear "not ready" message and exits 2 (see docs/mission_1_status.md).

  * ``--synth-fallback``: MMA-rendered jazz1460 songs with db.jsonl GT (the same
    harness as scripts/eval_user_merge.py).  Circular-free by construction (GT is
    the render's own chart) but NOT real audio — use only to smoke-test the
    detect→merge→score plumbing before the real benchmark lands.

Usage:
  # once Mission 1 is done:
  .venv/bin/python scripts/eval_auto_merge.py
  # to exercise the plumbing now:
  .venv/bin/python scripts/eval_auto_merge.py --synth-fallback --start 20 --n 20
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from harmonia.models import chord_pipeline_v1 as P  # noqa: E402
from detect_auto_merges import (  # noqa: E402
    detect_auto_merges, fired_merges, _load_audio_chroma,
    _label_to_root_q5,
)

NOTE = P.NOTE
BENCHMARK_DIR = REPO / "data" / "real_audio_benchmark"

# GT MMA quality -> q5 idx (maj/min/dom/hdim/dim); shared taxonomy.
MMA_TO_Q5IDX = {
    "maj": 0, "maj7": 0, "6": 0, "aug": 0, "augmaj7": 0, "sus2": 0, "sus4": 0,
    "min": 1, "min7": 1, "m6": 1, "minmaj7": 1,
    "dom7": 2, "dom7alt": 2, "7": 2, "9": 2, "aug7": 2, "7sus4": 2,
    "hdim7": 3, "m7b5": 3, "dim": 4, "dim7": 4,
}
Q5_MAJMIN = {0: "maj", 1: "min", 2: "maj", 3: "other", 4: "other"}


# ── scoring (shared by both data paths) ───────────────────────────────────────

def _pred_at(chart, t):
    return _chart_pred_at(chart.chords, t)


def _chart_pred_at(chords, t):
    lab = None
    for c in chords:
        if c["start_s"] <= t < c["end_s"]:
            lab = c["label"]; break
        if c["start_s"] <= t:
            lab = c["label"]
    return _label_to_root_q5(lab) if lab else None


def score_chart(chords, gt_spans, step: float = 0.05) -> tuple[int, int, int, int]:
    """(root_ok, majmin_ok, sevenths_ok, n) over frames whose GT is maj/min/dom.

    Frames scored only where GT q5 is defined and in the maj/min/dom families
    (the majmin-eval convention of eval_user_merge.py).  ``sevenths_ok`` = exact
    (root AND q5) match.
    """
    rt = mm = sv = n = 0
    if not gt_spans:
        return (0, 0, 0, 0)
    t0, t1 = gt_spans[0][0], gt_spans[-1][1]
    t = t0
    while t < t1:
        g = None
        for a, b, r, q in gt_spans:
            if a <= t < b:
                g = (r % 12, q); break
        if g is not None and g[1] is not None and Q5_MAJMIN.get(g[1]) in ("maj", "min"):
            n += 1
            p = _chart_pred_at(chords, t)
            if p and p[0] == g[0]:
                rt += 1
                if Q5_MAJMIN.get(p[1]) == Q5_MAJMIN[g[1]]:
                    mm += 1
                if p[1] == g[1]:
                    sv += 1
        t += step
    return (rt, mm, sv, n)


# ── data loaders ──────────────────────────────────────────────────────────────

def load_synth_songs(start: int, n: int):
    """MMA-rendered jazz1460 fallback: yields (song_id, wav_path, gt_spans)."""
    sys.path.insert(0, str(REPO / "scripts"))
    from analyze_accomp_emission import song_chord_spans
    from build_audio_chord_features import BUCKET_FAMILY
    from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig

    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2 = renderer._find_soundfont("MuseScore_General.sf2")
    db = REPO / "data" / "accomp_db" / "db.jsonl"
    recs = [json.loads(l) for l in open(db)]
    jz = [r for r in recs if r.get("corpus") == "jazz1460"
          and r["beats_per_bar"] == 4 and (REPO / r["midi_path"]).exists()]
    held = jz[start:start + n]

    for rec in held:
        spans = [(t0, t1, r % 12, MMA_TO_Q5IDX.get(q))
                 for t0, t1, r, q in song_chord_spans(rec)
                 if t1 > t0 and q in BUCKET_FAMILY]
        if not spans:
            continue
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
            tmp = Path(wf.name)
        renderer.render(REPO / rec["midi_path"], tmp, RenderConfig(soundfont_path=sf2))
        yield rec["song_id"], tmp, spans, "synth"


def _benchmark_gt_spans(record: dict):
    """Convert one Mission-1 benchmark record's aligned chords to (t0,t1,root,q5).

    Tolerant to the still-settling Mission-1 schema: each aligned chord needs a
    time span (``t0``/``t0_beat``/``start_s`` … ``t1``/``t1_beat``/``end_s``) and
    a ``label``.  q5 is parsed from the label the same way the pipeline does, so
    GT and prediction share one taxonomy.  Chords that don't parse are dropped.
    """
    out = []
    for c in record.get("aligned_chords", record.get("chords", [])):
        t0 = c.get("t0", c.get("t0_beat", c.get("start_s")))
        t1 = c.get("t1", c.get("t1_beat", c.get("end_s")))
        lab = c.get("label", "")
        if t0 is None or t1 is None or t1 <= t0:
            continue
        rq = _label_to_root_q5(lab if ":" in lab else _ireal_to_harte(lab))
        if rq is None:
            continue
        out.append((float(t0), float(t1), rq[0], rq[1]))
    return out


def _ireal_to_harte(label: str) -> str:
    """Best-effort iReal chord label -> 'Root:quality' Harte-ish for parsing.

    Only needs enough to land in the q5 taxonomy (maj/min/dom/hdim/dim).  Falls
    back to the raw label so ``_label_to_root_q5`` can drop it if unrecognised.
    """
    from harmonia.data.ireal_corpus import chord_root_pc
    r = chord_root_pc(label)
    if r is None:
        return label
    q = label[1:]
    if q[:1] in ("#", "b"):
        q = q[1:]
    q = q.split("/")[0]
    if q.startswith("m7b5") or q.startswith("h"):
        sev = "min7b5"
    elif q.startswith("dim") or q.startswith("o"):
        sev = "dim"
    elif q.startswith("m") and not q.startswith("maj") and not q.startswith("M"):
        sev = "min7" if "7" in q else "min"
    elif any(x in q for x in ("maj7", "M7", "Maj7")):
        sev = "maj7"
    elif "7" in q or "9" in q or "13" in q:
        sev = "7"
    else:
        sev = "maj"
    return f"{NOTE[r % 12]}:{sev}"


def load_benchmark_songs():
    """Yields (song_id, audio_path, gt_spans, 'real') from the Mission-1 set.

    Exits(2) with guidance if the benchmark is not yet built (Mission-1 gate).
    """
    records = sorted(BENCHMARK_DIR.glob("*.json"))
    # PROTOCOL.md / design docs are not song records
    records = [p for p in records if p.name not in ("manifest.json",)]
    if not records:
        print("\n" + "=" * 72)
        print("  Mission 1 benchmark NOT READY — auto-merge eval is gated on it.")
        print("  Expected per-song records in", BENCHMARK_DIR)
        print("  (see docs/mission_1_status.md — protocol done, build pending).")
        print("  Run with --synth-fallback to exercise the plumbing meanwhile.")
        print("=" * 72)
        raise SystemExit(2)

    for p in records:
        rec = json.loads(p.read_text())
        audio = rec.get("audio_path")
        if not audio:
            continue
        audio = Path(audio)
        if not audio.is_absolute():
            audio = REPO / audio
        if not audio.exists():
            print(f"  [skip] {p.name}: audio missing ({audio})")
            continue
        spans = _benchmark_gt_spans(rec)
        if not spans:
            print(f"  [skip] {p.name}: no parseable GT chords")
            continue
        yield rec.get("song_id", p.stem), audio, spans, "real"


# ── driver ────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synth-fallback", action="store_true",
                    help="use MMA-rendered jazz1460 instead of the Mission-1 set")
    ap.add_argument("--start", type=int, default=20, help="synth: first jazz idx")
    ap.add_argument("--n", type=int, default=20, help="synth: number of songs")
    ap.add_argument("--struct-threshold", type=float, default=0.75)
    ap.add_argument("--acoustic-threshold", type=float, default=0.75)
    ap.add_argument("--transition-weight", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0, help="cap songs (0 = all)")
    args = ap.parse_args()

    domain = "synth" if args.synth_fallback else "real"
    if args.synth_fallback:
        songs = load_synth_songs(args.start, args.n)
    else:
        songs = load_benchmark_songs()

    agg = {"base": [0, 0, 0, 0], "merged": [0, 0, 0, 0]}  # rt, mm, sv, n
    per_song = []
    n_done = 0

    with tempfile.TemporaryDirectory() as cd:
        cache = Path(cd)
        for song_id, audio, gt_spans, dom in songs:
            tmp_wav = audio if dom == "real" else audio
            try:
                base = P.infer_chords_v1(
                    audio, cache_dir=cache, audio_domain=domain,
                    joint_transition_weight=args.transition_weight)
                chroma, ctimes = _load_audio_chroma(audio)
                cands = detect_auto_merges(
                    base, chroma=chroma, chroma_times=ctimes,
                    struct_threshold=args.struct_threshold,
                    acoustic_threshold=args.acoustic_threshold)
                merges = fired_merges(cands)
                if merges:
                    merged = P.infer_chords_v1(
                        audio, cache_dir=cache, audio_domain=domain,
                        joint_transition_weight=args.transition_weight,
                        user_constraints={"merges": merges})
                else:
                    merged = base  # nothing fired ⇒ identical decode

                bs = score_chart(base.chords, gt_spans)
                ms = score_chart(merged.chords, gt_spans)
                if bs[3] == 0:
                    continue
                for k, s in zip(("base", "merged"), (bs, ms)):
                    for j in range(4):
                        agg[k][j] += s[j]
                n_done += 1
                d_mm = (ms[1] - bs[1]) / bs[3]
                d_sv = (ms[2] - bs[2]) / bs[3]
                per_song.append((song_id, len(merges), bs, ms, d_mm, d_sv))
                print(f"  {song_id}: {len(merges)} merge(s)  "
                      f"mm {bs[1]/bs[3]:.1%}->{ms[1]/ms[3]:.1%} ({d_mm:+.1%})  "
                      f"7th {bs[2]/bs[3]:.1%}->{ms[2]/ms[3]:.1%} ({d_sv:+.1%})",
                      flush=True)
            finally:
                if dom == "synth":
                    Path(tmp_wav).unlink(missing_ok=True)
            if args.limit and n_done >= args.limit:
                break

    # ── report ────────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"AUTO-MERGE EVAL — {domain} ({n_done} songs, "
          f"struct>{args.struct_threshold} acoustic>{args.acoustic_threshold})")
    print("=" * 72)
    tot = agg["base"][3]
    if not tot:
        print("  no scorable frames")
        return 1
    n_fired = sum(1 for r in per_song if r[1] > 0)
    n_up = sum(1 for r in per_song if r[4] > 1e-6)
    n_down = sum(1 for r in per_song if r[4] < -1e-6)
    print(f"  songs with ≥1 auto-merge fired: {n_fired}/{n_done}")
    print(f"  majmin: base {agg['base'][1]/tot:.1%}  merged {agg['merged'][1]/tot:.1%}"
          f"  Δ {(agg['merged'][1]-agg['base'][1])/tot:+.1%}")
    print(f"  7ths  : base {agg['base'][2]/tot:.1%}  merged {agg['merged'][2]/tot:.1%}"
          f"  Δ {(agg['merged'][2]-agg['base'][2])/tot:+.1%}")
    print(f"  root  : base {agg['base'][0]/tot:.1%}  merged {agg['merged'][0]/tot:.1%}"
          f"  Δ {(agg['merged'][0]-agg['base'][0])/tot:+.1%}")
    print(f"  per-song majmin: {n_up} improved, {n_down} regressed, "
          f"{n_done - n_up - n_down} unchanged")
    if n_down:
        print("  REGRESSIONS (auto-merge should never hurt a confident merge):")
        for sid, nm, bs, ms, dmm, dsv in per_song:
            if dmm < -1e-6:
                print(f"    {sid}: {nm} merge(s)  Δmm {dmm:+.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
