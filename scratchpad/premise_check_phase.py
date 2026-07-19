"""Premise check: is the per-bar V_F loss dominated by GRID PHASE misalignment
(fixed 8-bar blocks starting at bar 0 straddle real boundaries) rather than by
genuine content-matching failure?

Test: for each song, try all 8 possible phase offsets (0..7) for the 8-bar grid,
re-cluster and re-score at EACH phase, and compare:
  - phase-0 (current, fixed) per-bar V_F  -- what's been reported all along
  - best-of-8-phases per-bar V_F  -- oracle upper bound if phase were free
If the oracle-phase number closes most of the phase-0 vs block-level (0.732) gap,
phase misalignment (not matching quality) is the dominant loss source.
"""
import sys, collections
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from symstruct import load_corpus, vmeasure, _bar_sig, _block_sim

corpus = load_corpus()
multi = [c for c in corpus if len(set(c["labels"])) >= 2]

size = 8
letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def blockmatch_phased(feat, phase, base_bars=8, sim_threshold=0.75):
    n = len(feat)
    sigs = [_bar_sig(feat[i]) for i in range(n)]
    blocks = []
    i = phase % base_bars
    if i > 0:
        blocks.append((0, min(i, n)))
    while i < n:
        j = min(i + base_bars, n)
        blocks.append((i, j))
        i = j
    blocks = [(s, e) for (s, e) in blocks if e > s]
    block_sigs = [sigs[s:e] for s, e in blocks]
    reps = []
    labels = []
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
    per_bar = []
    for (s, e), lab in zip(blocks, labels):
        per_bar += [lab] * (e - s)
    return per_bar[:n]


phase0_vfs = []
oracle_vfs = []
per_song = []
for c in multi:
    feat, gt = c["feat"], c["labels"]
    n = len(feat)
    if n < size:
        continue
    vfs_by_phase = []
    for phase in range(size):
        pred = blockmatch_phased(feat, phase, base_bars=size)
        vf = vmeasure(gt, pred)[0]
        vfs_by_phase.append(vf)
    phase0_vfs.append(vfs_by_phase[0])
    oracle_vfs.append(max(vfs_by_phase))
    per_song.append((c["title"], vfs_by_phase[0], max(vfs_by_phase),
                     int(np.argmax(vfs_by_phase))))

phase0_vfs = np.array(phase0_vfs)
oracle_vfs = np.array(oracle_vfs)
print("n songs:", len(phase0_vfs))
print("phase-0 (current, fixed grid)   per-bar V_F mean=%.3f median=%.3f" %
      (phase0_vfs.mean(), np.median(phase0_vfs)))
print("best-of-8-phases (oracle phase) per-bar V_F mean=%.3f median=%.3f" %
      (oracle_vfs.mean(), np.median(oracle_vfs)))
print("mean per-song gain from phase correction alone: %.3f" %
      (oracle_vfs - phase0_vfs).mean())
print("fraction of songs where oracle phase != 0: %.1f%%" %
      (100.0 * np.mean([p[3] != 0 for p in per_song])))

per_song.sort(key=lambda x: -(x[2] - x[1]))
print("\nbiggest phase-correction winners:")
for title, p0, best, ph in per_song[:8]:
    print("  %-35s phase0=%.3f best=%.3f (phase=%d) gain=%.3f" %
          (title[:35], p0, best, ph, best - p0))
