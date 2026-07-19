"""realaudio_threshold_check.py — 2026-07-18 (overnight autonomous call,
follow-up to the auto-apply measurement's regression finding).

**Why this exists**: task 3's real-audio auto-apply measurement found 162/267
(61%) of touched bars had confidence go DOWN and 97/267 (36%) flip label
after auto-applying tau_auto=0.96 "never-a-false-positive" merges — directly
contradicting that threshold's corpus-scale derivation
(`scratchpad/tau_auto_search.py`, CP-upper-95% 0.86%-1.54% error across 5
folds). Root-caused via 2 hand-inspected examples (abba bars 32/64, aretha
bars 13/17): in BOTH cases, the two bars judged sim>=0.96 by real-audio
rawchroma `bt_concat` cosine similarity were ALREADY decoded to genuinely
different chords (different root entirely for aretha 13/17: E:maj7 vs
C:maj7) by the model's own UNCONSTRAINED baseline — i.e. real-audio
`bt_concat` similarity at 0.96-0.98 does not reliably indicate same-chord-
identity at all. Hypothesis: `tau_auto` was calibrated on
`tau_auto_search.py`'s SYMBOLIC iReal proxy features (clean root-one-hot /
chord-tone-binary vectors built directly from ground-truth MMA chord
symbols — a "closest available proxy" because iReal has no audio, per that
script's own docstring) and then ported UNCHANGED as a literal similarity
threshold onto `bar_merge_candidates.py`'s real-audio `rawchroma.bt_concat`
feature (continuous, noisy audio-chroma-derived vectors) — a feature-space
mismatch, exactly CLAUDE.md rule #6 ("a component swap changes more than
the target metric").

**This script measures it properly, corpus-scale (all 3 real songs'
full candidate census, not 2 examples)**: uses the model's own UNCONSTRAINED
baseline decode as noisy pseudo-ground-truth (the best available proxy —
these 3 songs don't have hand-verified root/quality ground truth at bar
grain) to check, for every candidate pair regardless of tier, whether
real-audio bt_concat similarity actually predicts baseline-label agreement
(coarse root+quality-family match). This is explicitly a NOISY estimate
(the pseudo-GT is model output, not truth — errors in the baseline decode
itself will bias this both ways), stated up front, not glossed over.
"""
from __future__ import annotations
import sys, json, tempfile, shutil
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from auto_apply_merges import SONGS, AUDIO_DIR, transcode, chord_at

OUT_DIR = Path(__file__).resolve().parent


def label_bucket(label):
    """Coarse (root_pc, quality_family) bucket from a billboard-style label
    string ('C#:maj7', 'D:min', 'N', ...) -- mirrors harmonia_server.py's
    own `_ireal_q_to_q5` bucket definition (maj/min/dom/hdim/dim) so this
    is directly comparable to the rest of the project's conventions."""
    if not label or label in ("N", "X"):
        return None
    NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    if ":" in label:
        root, qual = label.split(":", 1)
    else:
        root, qual = label, "maj"
    try:
        pc = NOTE.index(root)
    except ValueError:
        return None
    q = qual.lower()
    if "hdim" in q or "m7b5" in q:
        fam = 3
    elif q.startswith("dim") or q == "o":
        fam = 4
    elif q.startswith("min") or q.startswith("m") and not q.startswith("maj"):
        fam = 1
    elif "maj" in q:
        fam = 0
    elif any(t in q for t in ("7", "9", "13", "alt")):
        fam = 2
    else:
        fam = 0
    return (pc, fam)


def get_baseline_chords(slug):
    """Direct pipeline call, unconstrained -- same code path as
    auto_apply_merges.py, cached to a JSON sidecar so repeated runs of this
    script don't re-decode audio every time."""
    cache_path = OUT_DIR / f"baseline_chords_{slug}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text())
    from harmonia.models.chord_pipeline_v1 import infer_chords_v1
    audio_path = AUDIO_DIR / SONGS[slug]["audio_name"]
    tmp = Path(tempfile.mkdtemp(prefix="harmonia_baseline_"))
    try:
        wav = tmp / "a.wav"
        transcode(audio_path, wav)
        base = infer_chords_v1(wav, cache_dir=tmp, joint_transition_weight=0.0)
        base_ch = [c for c in base.chords if c["end_s"] > c["start_s"]]
        cache_path.write_text(json.dumps(base_ch))
        return base_ch
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    all_rows = []  # (sim, agree, song)
    per_song_summary = {}
    for slug in SONGS:
        census = json.loads((OUT_DIR / f"bar_merge_full_census_{slug}.json").read_text())
        base_ch = get_baseline_chords(slug)
        rows = []
        for c in census["candidates"]:
            (t0a, t1a), (t0b, t1b) = c["spans"]
            ca = chord_at(base_ch, 0.5 * (t0a + t1a))
            cb = chord_at(base_ch, 0.5 * (t0b + t1b))
            if ca is None or cb is None:
                continue
            ba, bb = label_bucket(ca["label"]), label_bucket(cb["label"])
            if ba is None or bb is None:
                continue
            agree = int(ba == bb)
            rows.append((c["confidence"], agree, c["tier"]))
            all_rows.append((c["confidence"], agree, slug))
        sims = np.array([r[0] for r in rows])
        agrees = np.array([r[1] for r in rows])
        tau_auto_mask = sims >= 0.96
        tau_suggest_mask = (sims >= 0.80) & (sims < 0.96)
        n_auto = int(tau_auto_mask.sum())
        n_suggest = int(tau_suggest_mask.sum())
        auto_agree_rate = float(agrees[tau_auto_mask].mean()) if n_auto else None
        suggest_agree_rate = float(agrees[tau_suggest_mask].mean()) if n_suggest else None
        print(f"\n=== {slug} ===  n_pairs_checked={len(rows)}")
        print(f"  tau_auto (>=0.96) tier: n={n_auto}  baseline-label-agreement rate={auto_agree_rate}")
        print(f"  tau_suggest (0.80-0.96) tier: n={n_suggest}  baseline-label-agreement rate={suggest_agree_rate}")
        per_song_summary[slug] = {
            "n_pairs_checked": len(rows), "n_auto": n_auto, "n_suggest": n_suggest,
            "auto_agree_rate": auto_agree_rate, "suggest_agree_rate": suggest_agree_rate,
        }

    sims_all = np.array([r[0] for r in all_rows])
    agrees_all = np.array([r[1] for r in all_rows])
    print("\n=== POOLED across all 3 songs, dense threshold sweep ===")
    print("  (agreement rate = fraction of pairs at sim>=tau whose model-baseline")
    print("   root+quality-family labels MATCH -- noisy pseudo-precision estimate)")
    sweep = []
    for tau in [0.99, 0.98, 0.97, 0.96, 0.95, 0.93, 0.90, 0.85, 0.80]:
        mask = sims_all >= tau
        n = int(mask.sum())
        agree_rate = float(agrees_all[mask].mean()) if n else None
        sweep.append({"tau": tau, "n": n, "agree_rate": agree_rate})
        print(f"  tau={tau:.2f}  n={n:5d}  agree_rate={agree_rate}")

    out = {"per_song": per_song_summary, "pooled_sweep": sweep,
           "n_total_pairs_checked": len(all_rows),
           "pooled_overall_agree_rate": float(agrees_all.mean()) if len(agrees_all) else None}
    (OUT_DIR / "realaudio_threshold_check_results.json").write_text(json.dumps(out, indent=2))
    print("\nwrote realaudio_threshold_check_results.json")


if __name__ == "__main__":
    main()
