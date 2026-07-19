"""symstruct.py — symbolic structure model from iReal chord sequences (Idea #2).

Builds (per-bar chord sequence, per-bar section-label) pairs for the whole iReal
corpus, then evaluates a chord-sequence-ONLY structure predictor against the
iReal *A/*B/*C form GT with V-measure (mir_eval.segment.nce) — the same metric
tonight's audio baselines used (symbolic V_F 0.23, librosa V_F 0.40 on 3 songs).

No audio anywhere: input = per-bar chord symbols, target = section labels.
"""
from __future__ import annotations
import sys, io, json, collections, random
from pathlib import Path
from contextlib import redirect_stdout
import numpy as np

from harmonia.data.ireal_corpus import load_playlist, tune_to_mma, chord_root_pc

FILES = ["jazz1460", "pop400", "blues50", "brazilian220",
         "country", "dixieland1", "latin_salsa50"]

# coarse quality bucket from an MMA chord tail
def qbucket(mma: str) -> int:
    if not mma or mma == "z":
        return -1
    tail = mma[1:]
    if tail and tail[0] in "#b":
        tail = tail[1:]
    if tail.startswith("m") and not tail.startswith("maj") and not tail.startswith("M"):
        return 1  # minor family
    if tail.startswith("M") or tail.startswith("maj") or tail.startswith("6") or tail == "":
        return 0  # major family
    if tail.startswith("dim") or tail.startswith("o"):
        return 3
    if tail.startswith("aug") or tail.startswith("+"):
        return 4
    if tail.startswith("sus"):
        return 5
    # 7,9,11,13 dominant-ish
    return 2


NQ = 6


def bar_features(mma_chart) -> tuple[np.ndarray, list[str]]:
    """One feature row per bar; returns (feat[n_bars, 12+NQ], section_per_bar)."""
    rows = []
    labels = []
    for bar_no, section, slots in mma_chart.timeline:
        vec = np.zeros(12 + NQ, dtype=np.float32)
        for (_, _, mma) in slots:
            pc = chord_root_pc(mma)
            if pc is None:
                continue
            vec[pc] += 1.0
            q = qbucket(mma)
            if q >= 0:
                vec[12 + q] += 1.0
        rows.append(vec)
        labels.append(section)
    if not rows:
        return np.zeros((0, 12 + NQ)), []
    feat = np.stack(rows)
    n = np.linalg.norm(feat, axis=1, keepdims=True)
    feat = feat / np.clip(n, 1e-9, None)
    return feat, labels


def bar_ssm(feat: np.ndarray) -> np.ndarray:
    if len(feat) == 0:
        return np.zeros((0, 0))
    return np.clip(feat @ feat.T, 0.0, 1.0)


def load_corpus(verbose=False):
    """Return list of dicts: {file,title,feat,labels,form,key}."""
    out = []
    for f in FILES:
        try:
            buf = io.StringIO()
            with redirect_stdout(buf):
                tunes = load_playlist(Path("data/ireal/%s.txt" % f))
        except Exception:
            continue
        for t in tunes:
            try:
                mc = tune_to_mma(t)
            except Exception:
                continue
            feat, labels = bar_features(mc)
            if len(labels) < 8:
                continue
            out.append({"file": f, "title": mc.title, "feat": feat,
                        "labels": labels, "form": mc.form,
                        "key": mc.key})
    return out


# ── V-measure via mir_eval on per-bar label arrays ────────────────────────────
def vmeasure(ref_labels: list[str], est_labels: list[str]) -> tuple[float, float, float]:
    """Return (V_F, S_over, S_under). Bars are unit-length frames."""
    import mir_eval
    n = len(ref_labels)
    intervals = np.array([[i, i + 1] for i in range(n)], dtype=float)
    s_o, s_u, vf = mir_eval.segment.nce(intervals, [str(x) for x in ref_labels],
                                        intervals, [str(x) for x in est_labels],
                                        frame_size=1.0, beta=1.0, marginal=False)
    return vf, s_o, s_u


# ── Predictor 1: unsupervised symbolic SSM ────────────────────────────────────
def predict_ssm(feat, *, form_lengths=(4, 8, 16, 32), rep_floor=0.25,
                merge_threshold=0.60, sim_threshold=0.70, min_section_bars=4):
    from harmonia.models.section_structure import (
        detect_section_boundaries, label_sections)
    ssm = bar_ssm(feat)
    n = len(feat)
    if n < 4:
        return ["A"] * n
    bounds = detect_section_boundaries(
        ssm, beats_per_bar=1, form_lengths=form_lengths,
        merge_threshold=merge_threshold, rep_floor=rep_floor,
        min_section_bars=min_section_bars)
    grid = [0] + [b for b in bounds if 0 < b < n] + [n]
    grid = sorted(set(grid))
    seclabels = label_sections(ssm, grid, sim_threshold=sim_threshold)
    per_bar = []
    for i in range(len(grid) - 1):
        per_bar += [seclabels[i]] * (grid[i + 1] - grid[i])
    if len(per_bar) < n:
        per_bar += [seclabels[-1] if seclabels else "A"] * (n - len(per_bar))
    return per_bar[:n]


# ── Predictor 2: transposition-invariant block matching ───────────────────────
def _bar_sig(feat_row: np.ndarray) -> tuple:
    """Hashable (dominant-root, quality) signature of one bar's feature row."""
    roots = feat_row[:12]
    if roots.sum() <= 0:
        return (-1, -1)
    r = int(np.argmax(roots))
    q = int(np.argmax(feat_row[12:])) if feat_row[12:].sum() > 0 else -1
    return (r, q)


def _block_sim(sig_a, sig_b) -> float:
    """Transposition-invariant similarity of two equal-length bar-sig lists.

    Try every root offset; score = fraction of bars whose (root-rel, qual) match.
    """
    n = min(len(sig_a), len(sig_b))
    if n == 0:
        return 0.0
    a = sig_a[:n]
    b = sig_b[:n]
    best = 0.0
    for off in range(12):
        m = 0
        for (ra, qa), (rb, qb) in zip(a, b):
            if ra < 0 and rb < 0:
                m += 1
            elif ra >= 0 and rb >= 0 and ((rb - ra) % 12) == off and qa == qb:
                m += 1
        best = max(best, m / n)
    return best


def predict_blockmatch(feat, *, base_bars=8, sim_threshold=0.75):
    """Segment into fixed base_bars blocks, cluster by transposition-invariant
    chord-signature match, merge adjacent same-letter blocks. Pure symbolic."""
    n = len(feat)
    if n < base_bars:
        return ["A"] * n
    sigs = [_bar_sig(feat[i]) for i in range(n)]
    # blocks (last one may be short)
    blocks = []
    i = 0
    while i < n:
        j = min(i + base_bars, n)
        blocks.append((i, j))
        i = j
    # fold a too-short tail into previous block
    if len(blocks) >= 2 and (blocks[-1][1] - blocks[-1][0]) < base_bars // 2:
        s, e = blocks.pop()
        blocks[-1] = (blocks[-1][0], e)
    block_sigs = [sigs[s:e] for s, e in blocks]
    # greedy clustering
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    labels = []
    reps = []  # (letter, sig)
    for bs in block_sigs:
        assigned = None
        for let, rsig in reps:
            if _block_sim(bs, rsig) >= sim_threshold:
                assigned = let
                break
        if assigned is None:
            assigned = letters[len(reps) % 26]
            reps.append((assigned, bs))
        labels.append(assigned)
    # expand to per-bar
    per_bar = []
    for (s, e), lab in zip(blocks, labels):
        per_bar += [lab] * (e - s)
    return per_bar[:n]


# ── Baselines for context ─────────────────────────────────────────────────────
def predict_allone(feat):
    return ["A"] * len(feat)


def predict_fixed8(feat):
    """Uniform 8-bar grid, each block a new letter (no repeat detection)."""
    n = len(feat)
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    return [letters[(i // 8) % 26] for i in range(n)]


if __name__ == "__main__":
    random.seed(0)
    print("loading corpus...", file=sys.stderr)
    corpus = load_corpus()
    # keep only multi-section tunes (real form GT)
    multi = [c for c in corpus if len(set(c["labels"])) >= 2]
    print("corpus: %d tunes total, %d multi-section (>=2 labels)" %
          (len(corpus), len(multi)), file=sys.stderr)

    # song-level held-out split (80/20) for honesty even though SSM is untrained
    random.shuffle(multi)
    ntest = len(multi) // 5
    test = multi[:ntest]
    train = multi[ntest:]

    methods = {"allone": predict_allone, "fixed8": predict_fixed8,
               "ssm": predict_ssm,
               "block8": lambda f: predict_blockmatch(f, base_bars=8),
               "block4": lambda f: predict_blockmatch(f, base_bars=4)}
    for split_name, split in [("TEST", test), ("ALL-multi", multi)]:
        print("\n==== %s (%d tunes) ====" % (split_name, len(split)))
        for mname, fn in methods.items():
            vs = []
            for c in split:
                est = fn(c["feat"])
                vf, so, su = vmeasure(c["labels"], est)
                vs.append(vf)
            vs = np.array(vs)
            print("  %-8s V_F mean=%.3f median=%.3f  (n=%d)" %
                  (mname, vs.mean(), np.median(vs), len(vs)))
