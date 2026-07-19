"""bar_distance_matrix.py — Task 3 of the 2026-07-18 Call 3 brief: build a
full bar-to-bar self-similarity matrix (SSM) for a song — EVERY bar against
EVERY other bar (n_bars x n_bars), not just within nuclear blocks.

Two examples, per the brief:
  1. Clean iReal song WITH GT section labels, legible form: "All Of Me"
     (jazz1460.txt), 32 bars, AABC (['A']*8+['B']*8+['A']*8+['C']*8) — a
     textbook easy-to-eyeball form. Scheme: V1 binary chord-tone membership
     (chord_distance.chord_vector_binary), the most interpretable of the 3
     validated hand-crafted schemes (task 1-2 found V1/V2 statistically tied,
     V3 reliably ~0.004-0.005 lower across 7 seeds — V1 is the right default
     per the brief's own guidance when scores are close).
  2. Real audio song using real per-bar root-softmax (V4,
     chord_distance_v4_real.py): "aretha_chain_of_fools" (83 bars, small
     enough for a readable matrix, and Task 1 already characterized its
     structure qualitatively: dominant S0 75/83 bars + one 8-bar bridge).

Bar-to-bar similarity = plain cosine of the two bars' 12-d vectors (single
bar, not a block — this is the atomic version of block_sim's position-
aligned dot product with L=1).
"""
from __future__ import annotations
import sys, io, json
from pathlib import Path
from contextlib import redirect_stdout
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from harmonia.data.ireal_corpus import load_playlist, tune_to_mma, chord_root_pc
from chord_distance import chord_vector_binary, cosine
from symstruct import qbucket
from real_root_proba import per_bar_root_proba
from symstruct_proba import estimate_tonic_pc
from chord_distance_v4_real import rotate12

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent


def build_ireal_matrix(playlist_file, title, key_shift=True):
    buf = io.StringIO()
    with redirect_stdout(buf):
        tunes = load_playlist(REPO / "data" / "ireal" / playlist_file)
    tune = next(t for t in tunes if t.title == title)
    mc = tune_to_mma(tune)
    shift = 0
    if key_shift and mc.key:
        pc = chord_root_pc(mc.key.rstrip("-"))
        shift = (-pc % 12) if pc is not None else 0

    bar_vecs = []
    labels = []
    for bar_no, section, slots in mc.timeline:
        accum = None
        for (_, _, mma) in slots:
            pc = chord_root_pc(mma)
            if pc is None:
                continue
            rpc = (pc + shift) % 12
            q = qbucket(mma)
            v = chord_vector_binary(rpc, q)
            accum = v if accum is None else accum + v
        bar_vecs.append(accum if accum is not None else np.zeros(12))
        labels.append(section)

    n = len(bar_vecs)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim[i, j] = cosine(bar_vecs[i], bar_vecs[j])

    return {
        "song": title, "source": "ireal_clean_symbolic", "scheme": "V1_binary",
        "n_bars": n, "labels": labels, "key": mc.key, "key_shift_applied": shift,
        "similarity_matrix": sim.tolist(),
    }


def build_real_audio_matrix(name, audio_path):
    bar_proba, bar_times, tempo = per_bar_root_proba(audio_path)
    bar12 = bar_proba[:, :12].astype(np.float64)
    tonic = estimate_tonic_pc(bar_proba)
    shift = (-tonic) % 12
    bar12_kn = rotate12(bar12, shift)

    n = len(bar12_kn)
    sim = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sim[i, j] = cosine(bar12_kn[i], bar12_kn[j])

    return {
        "song": name, "source": "real_audio_V4_chroma", "scheme": "V4_raw_chroma",
        "n_bars": n, "tempo_bpm": tempo, "est_tonic_pc": tonic,
        "bar_times_s": [round(float(t), 3) for t in bar_times],
        "similarity_matrix": sim.tolist(),
    }


def sanity_check(mat, label_key="labels"):
    sim = np.array(mat["similarity_matrix"])
    n = mat["n_bars"]
    print("  song=%s  shape=%s  scheme=%s" % (mat["song"], sim.shape, mat["scheme"]))
    diag = np.diag(sim)
    print("  diagonal: min=%.4f max=%.4f mean=%.4f (expect ~1.0)" %
          (diag.min(), diag.max(), diag.mean()))
    if label_key in mat:
        labels = mat[label_key]
        # Group into contiguous RUNS (a "section occurrence"), then compare
        # POSITION-MATCHED bars across two occurrences of the same label
        # (bar k-within-run vs bar k-within-run of another run) — comparing
        # arbitrary same-label bars regardless of phrase position is not a
        # meaningful "known repeat" check (e.g. bar 0 of an A-section vs the
        # LAST bar of a different A-section are different phrase positions,
        # not a repeat of each other).
        runs = []
        cur = labels[0]; start = 0
        for i in range(1, len(labels) + 1):
            if i == len(labels) or labels[i] != cur:
                runs.append((cur, start, i))
                if i < len(labels):
                    cur = labels[i]; start = i
        from collections import defaultdict
        runs_by_label = defaultdict(list)
        for (l, s, e) in runs:
            runs_by_label[l].append((s, e))
        for l, occs in runs_by_label.items():
            if len(occs) >= 2:
                (s1, e1), (s2, e2) = occs[0], occs[1]
                k = min(e1 - s1, e2 - s2) // 2  # a middle bar within the run
                i, j = s1 + k, s2 + k
                print("  known-repeat check (position-matched): bar %d (%s run1) "
                      "vs bar %d (%s run2) sim=%.3f" % (i, l, j, l, sim[i, j]))
        # cross-label sample (different section labels, should generally score lower)
        labs = list(runs_by_label.keys())
        if len(labs) >= 2:
            i = runs_by_label[labs[0]][0][0]
            j = runs_by_label[labs[1]][0][0]
            print("  cross-label check: bar %d (%s) vs bar %d (%s) sim=%.3f"
                  % (i, labs[0], j, labs[1], sim[i, j]))


def main():
    print("=== Building iReal clean-symbolic matrix: 'All Of Me' (AABC, 32 bars) ===")
    m1 = build_ireal_matrix("jazz1460.txt", "All Of Me")
    sanity_check(m1)
    out1 = OUT_DIR / "bar_distance_matrix_all_of_me.json"
    out1.write_text(json.dumps(m1))
    print("  wrote", out1)

    print("\n=== Building real-audio V4 matrix: 'aretha_chain_of_fools' (83 bars) ===")
    audio = REPO / "docs" / "audio" / "aretha_franklin_chain_of_fools_official_lyric_video.m4a"
    m2 = build_real_audio_matrix("aretha_chain_of_fools", audio)
    sanity_check(m2)
    out2 = OUT_DIR / "bar_distance_matrix_aretha_chain_of_fools.json"
    out2.write_text(json.dumps(m2))
    print("  wrote", out2)


if __name__ == "__main__":
    main()
