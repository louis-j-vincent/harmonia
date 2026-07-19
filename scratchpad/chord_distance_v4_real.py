"""chord_distance_v4_real.py — Task 1 of the 2026-07-18 Call 3 brief: V4, the
4th user-proposed chord-similarity variant — raw REAL per-bar chroma/root-
softmax dot product, ZERO idealized triad templates, ZERO training (not even
the small BiLSTM). Position-aligned dot product, same block_sim()/predict_
union() logic as chord_distance_eval.py's V1/V2/V3 (reused verbatim, not
reimplemented, so the "position-aligned not pool-then-dot" fix from that file
carries over automatically) — just swap the per-bar vector SOURCE from an
idealized chord_vector_binary() template to real_root_proba.py's live
per-bar 12-d root softmax from the actual nnls24 pipeline.

On clean iReal symbolic data V4 necessarily reduces to V1 (no real chroma
exists for symbolic chord charts) — not tested here, already noted as
expected in docs/known_issues.md. V4's real test is QUALITATIVE, on the same
3 real songs as Call 1/2 (autumn_leaves, abba_chiquitita,
aretha_chain_of_fools), reusing real_root_proba.py's extraction and
symstruct_proba.py's estimate_tonic_pc() heuristic key-norm exactly as
run_real_structure.py (the learned-encoder real-audio script) does, for a
fair side-by-side.

tau: no GT exists for real audio, so there is no way to tune tau on this
data. Reuses V1's val-selected tau*=0.84 (size=8, clean iReal corpus) as a
documented, principled default — NOT re-tuned per-song, flagged explicitly
as a heuristic carried over, same spirit as run_real_structure.py reusing
its own training-time tau_star.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from real_root_proba import per_bar_root_proba
from symstruct_proba import estimate_tonic_pc
from chord_distance_eval import nuclear_spans, block_sim

V1_TAU_STAR = 0.84  # reused from chord_distance_eval.py size=8 val selection

REPO = Path(__file__).resolve().parent.parent
AUDIO_DIR = REPO / "docs" / "audio"
OUT = Path(__file__).resolve().parent / "chord_distance_v4_real_results.json"

SONGS = [
    ("autumn_leaves", AUDIO_DIR / "autumn_leaves.m4a"),
    ("abba_chiquitita", AUDIO_DIR / "abba_chiquitita_official_lyric_video.m4a"),
    ("aretha_chain_of_fools", AUDIO_DIR / "aretha_franklin_chain_of_fools_official_lyric_video.m4a"),
]


def rotate12(bar_proba12, shift):
    """Rigid whole-song key-norm rotation of a (n,12) array by `shift`
    (matches symstruct_proba.rotate13's semantics but on the plain 12-d
    slice used here — no NC dim to carry since V4 has no idealized 'no
    chord' template)."""
    return np.roll(bar_proba12, shift, axis=1)


def predict_union_v4(bar_vecs12, n, size, tau):
    """Same union-find logic as chord_distance_eval.predict_union, just
    inlined here since that one is typed against the SCHEMES dict form —
    bar_vecs12 is a plain (n,12) array, one row per bar."""
    spans = nuclear_spans(n, size)
    block_bars = [bar_vecs12[s:e] for (s, e) in spans]
    m = len(spans)
    parent = list(range(m))
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for i in range(m):
        for j in range(i + 1, m):
            if block_sim(block_bars[i], block_bars[j]) >= tau:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    remap = {}; lab = ["A"] * n
    for k, (s, e) in enumerate(spans):
        r = find(k)
        if r not in remap: remap[r] = len(remap)
        for t in range(s, e):
            lab[t] = "S%d" % remap[r]
    return lab


def main():
    results = {}
    for name, path in SONGS:
        if not path.exists():
            print("MISSING audio for %s (%s) — skipping" % (name, path))
            continue
        print("\n=== %s (V4 raw chroma dot product) ===" % name)
        bar_proba, bar_times, tempo = per_bar_root_proba(path)
        bar12 = bar_proba[:, :12].astype(np.float64)
        tonic = estimate_tonic_pc(bar_proba)
        shift = (-tonic) % 12
        bar12_kn = rotate12(bar12, shift)
        labels = predict_union_v4(bar12_kn, len(bar12_kn), size=8, tau=V1_TAU_STAR)
        n_sections = len(set(labels))
        print("  tempo=%.1f bpm  n_bars=%d  est_tonic_pc=%d  n_sections=%d  tau=%.2f"
              % (tempo, len(bar12), tonic, n_sections, V1_TAU_STAR))
        runs = []
        cur = labels[0]; start = 0
        for i in range(1, len(labels) + 1):
            if i == len(labels) or labels[i] != cur:
                runs.append({"label": cur, "bar_start": start, "bar_end": i})
                if i < len(labels):
                    cur = labels[i]; start = i
        print("  %d runs: %s" % (len(runs), " ".join(
            "%s(%d-%d)" % (r["label"], r["bar_start"], r["bar_end"]) for r in runs)))

        # Tau-sensitivity diagnostic (root-cause step, doctrine Phase 3.1):
        # is a degenerate result a tau-miscalibration artifact or a real
        # geometry failure? Sweep tau on THIS song only (still zero training,
        # this is not a fit — just a sensitivity probe for the report).
        tau_sweep = {}
        for tau in (0.55, 0.65, 0.70, 0.75, 0.78, 0.84):
            lab_t = predict_union_v4(bar12_kn, len(bar12_kn), size=8, tau=tau)
            tau_sweep[str(tau)] = len(set(lab_t))
        print("  tau-sweep n_sections:", tau_sweep)

        results[name] = {
            "tempo_bpm": tempo, "n_bars": len(bar12), "est_tonic_pc": tonic,
            "n_sections": n_sections, "tau": V1_TAU_STAR, "runs": runs,
            "labels": labels, "tau_sensitivity_n_sections": tau_sweep,
        }
    OUT.write_text(json.dumps(results, indent=1))
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
