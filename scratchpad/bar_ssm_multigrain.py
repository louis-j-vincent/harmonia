"""bar_ssm_multigrain.py — same two songs as bar_distance_matrix.py, but the
self-similarity matrix at 5 granularities: 1/2/4/8/16-bar blocks. Each block's
vector is the CONCATENATION of its member bars (position-aligned, per the
fixed pool-then-dot bug from earlier tonight — see chord_distance_eval.py's
block_sim docstring), not a pooled sum. At size=1 this reduces exactly to
bar_distance_matrix.py's per-bar matrix (sanity-checked below).
"""
from __future__ import annotations
import sys, io, json, collections
from pathlib import Path
from contextlib import redirect_stdout
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np

from harmonia.data.ireal_corpus import load_playlist, tune_to_mma, chord_root_pc
from chord_distance import chord_vector_binary
from chord_distance_eval import nuclear_spans, block_sim
from symstruct import qbucket
from real_root_proba import per_bar_root_proba
from symstruct_proba import estimate_tonic_pc
from chord_distance_v4_real import rotate12

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = Path(__file__).resolve().parent
SIZES = [1, 2, 4, 8, 16]


def multigrain_matrices(bar_vecs, labels=None):
    n = len(bar_vecs)
    out = {}
    for size in SIZES:
        if n < size:
            continue
        spans = nuclear_spans(n, size)
        block_bars = [bar_vecs[s:e] for (s, e) in spans]
        m = len(spans)
        sim = np.zeros((m, m))
        for i in range(m):
            for j in range(m):
                sim[i, j] = block_sim(block_bars[i], block_bars[j])
        entry = {"n_blocks": m, "spans": spans, "similarity_matrix": sim.tolist()}
        if labels is not None:
            block_labels = []
            for (s, e) in spans:
                cnt = collections.Counter(labels[s:e])
                block_labels.append(cnt.most_common(1)[0][0])
            entry["labels"] = block_labels
        out[str(size)] = entry
    return out


def ireal_bar_vecs(playlist_file, title, key_shift=True):
    buf = io.StringIO()
    with redirect_stdout(buf):
        tunes = load_playlist(REPO / "data" / "ireal" / playlist_file)
    tune = next(t for t in tunes if t.title == title)
    mc = tune_to_mma(tune)
    shift = 0
    if key_shift and mc.key:
        pc = chord_root_pc(mc.key.rstrip("-"))
        shift = (-pc % 12) if pc is not None else 0
    bar_vecs, labels = [], []
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
    return bar_vecs, labels, mc.key


def real_audio_bar_vecs(audio_path):
    bar_proba, bar_times, tempo = per_bar_root_proba(audio_path)
    bar12 = bar_proba[:, :12].astype(np.float64)
    tonic = estimate_tonic_pc(bar_proba)
    shift = (-tonic) % 12
    bar12_kn = rotate12(bar12, shift)
    return [row for row in bar12_kn], tempo, tonic


def sanity_check_size1(bar_vecs, multigrain, label):
    """size=1 must reduce EXACTLY to a plain per-bar cosine matrix (already
    validated in bar_distance_matrix.py) — cross-check before trusting the
    coarser grains."""
    from chord_distance import cosine
    n = len(bar_vecs)
    ref = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            ref[i, j] = cosine(bar_vecs[i], bar_vecs[j])
    got = np.array(multigrain["1"]["similarity_matrix"])
    diff = np.abs(ref - got).max()
    print("  [%s] size=1 vs plain-cosine max abs diff: %.6f (expect ~0)" % (label, diff))
    assert diff < 1e-6, "size=1 multigrain matrix does not match plain per-bar cosine"


def main():
    print("=== All Of Me (iReal, AABC) — multigrain ===")
    bar_vecs, labels, key = ireal_bar_vecs("jazz1460.txt", "All Of Me")
    mg1 = multigrain_matrices(bar_vecs, labels)
    sanity_check_size1(bar_vecs, mg1, "All Of Me")
    for size in SIZES:
        if str(size) in mg1:
            print("  size=%2d -> %d blocks" % (size, mg1[str(size)]["n_blocks"]))
    out1 = {"song": "All Of Me", "source": "ireal_clean_symbolic",
            "scheme": "V1_binary", "n_bars": len(bar_vecs), "key": key,
            "grains": mg1}
    (OUT_DIR / "bar_ssm_multigrain_all_of_me.json").write_text(json.dumps(out1))
    print("  wrote bar_ssm_multigrain_all_of_me.json")

    print("\n=== aretha_chain_of_fools (real audio) — multigrain ===")
    audio = REPO / "docs" / "audio" / "aretha_franklin_chain_of_fools_official_lyric_video.m4a"
    bar_vecs2, tempo, tonic = real_audio_bar_vecs(audio)
    mg2 = multigrain_matrices(bar_vecs2, labels=None)
    sanity_check_size1(bar_vecs2, mg2, "aretha_chain_of_fools")
    for size in SIZES:
        if str(size) in mg2:
            print("  size=%2d -> %d blocks" % (size, mg2[str(size)]["n_blocks"]))
    out2 = {"song": "aretha_chain_of_fools", "source": "real_audio_V4_chroma",
            "scheme": "V4_raw_chroma", "n_bars": len(bar_vecs2),
            "tempo_bpm": tempo, "est_tonic_pc": tonic, "grains": mg2}
    (OUT_DIR / "bar_ssm_multigrain_aretha_chain_of_fools.json").write_text(json.dumps(out2))
    print("  wrote bar_ssm_multigrain_aretha_chain_of_fools.json")


if __name__ == "__main__":
    main()
