"""validate_pool_effect.py — 2026-07-18 chord-robustness reframe, Step 4.

End-to-end validation of pool_beat_evidence's real effect on a real song:
runs infer_chords_v1 TWICE (same config, same cache_dir so stage-1 activations
are reused — mirrors exactly what /api/reinfer does) — once unconstrained,
once with a real candidate bar-merge pair from bar_merge_candidates.py — and
reports the confidence/label diff for the merged spans AND their immediate
neighbors (the user's specific question: does pooling propagate beyond the
merged bars themselves).
"""
from __future__ import annotations
import sys, subprocess as sp, tempfile, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmonia.models.chord_pipeline_v1 import infer_chords_v1
from bar_merge_candidates import candidate_groups

REPO = Path(__file__).resolve().parent.parent


def chord_at(chords, t):
    for c in chords:
        if c["start_s"] <= t < c["end_s"]:
            return c
    return None


def main(song_key, audio_name, pair_rank=0):
    audio = REPO / "docs" / "audio" / audio_name
    groups, meta = candidate_groups(audio)
    if not groups:
        print("no candidates found"); return
    cand = groups[pair_rank]
    print(f"=== {song_key} === candidate pair bars={cand['bars']} "
          f"sim={cand['confidence']:.4f} spans={cand['spans']}")

    tmp_dir = Path(tempfile.mkdtemp(prefix="pool_validate_"))
    wav = tmp_dir / "a.wav"
    sp.run(["ffmpeg", "-y", "-i", str(audio), "-ac", "1", "-ar", "22050", str(wav)],
           check=True, capture_output=True, timeout=120)

    print("running UNCONSTRAINED decode...")
    base = infer_chords_v1(wav, cache_dir=tmp_dir, joint_transition_weight=0.0)

    print("running POOLED (merge) decode...")
    constraints = {"confirms": [], "merges": [{"spans": cand["spans"]}]}
    pooled = infer_chords_v1(wav, cache_dir=tmp_dir, joint_transition_weight=0.0,
                              user_constraints=constraints)

    t0a, t1a = cand["spans"][0]
    t0b, t1b = cand["spans"][1]
    mid_a, mid_b = 0.5 * (t0a + t1a), 0.5 * (t0b + t1b)

    def report(label, t, base_chords, pooled_chords):
        b = chord_at(base_chords, t)
        p = chord_at(pooled_chords, t)
        print(f"  {label} (t={t:.2f}s):")
        print(f"    base:   {b['label'] if b else None}  conf={b.get('confidence') if b else None}")
        print(f"    pooled: {p['label'] if p else None}  conf={p.get('confidence') if p else None}")

    print("\n--- merged bars themselves ---")
    report("bar_i", mid_a, base.chords, pooled.chords)
    report("bar_j", mid_b, base.chords, pooled.chords)

    print("\n--- immediate neighbors (propagation check) ---")
    # neighbor = 2s before/after each merged span
    for label, t in [("before_i", t0a - 1.0), ("after_i", t1a + 1.0),
                      ("before_j", t0b - 1.0), ("after_j", t1b + 1.0)]:
        if t < 0:
            continue
        report(label, t, base.chords, pooled.chords)

    n_changed = sum(1 for bc, pc in zip(
        [c for c in base.chords if c["end_s"] > c["start_s"]],
        [c for c in pooled.chords if c["end_s"] > c["start_s"]])
        if bc["label"] != pc["label"])
    print(f"\ntotal label changes across whole song: {n_changed} / "
          f"{len(base.chords)} chords")

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("song_key")
    ap.add_argument("audio_name")
    ap.add_argument("--rank", type=int, default=0)
    args = ap.parse_args()
    main(args.song_key, args.audio_name, args.rank)
