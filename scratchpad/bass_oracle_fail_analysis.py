"""STEP 4 — Structure of the ORACLE-FAILURE chords (where argmax, full24-head,
bass12-head AND pYIN are ALL wrong). This is the residual that caps the 0.95 target.
Cheap, no audio. Answers: are these short spans? concentrated songs? specific
qualities/pitch-class intervals? — i.e. would a better bass signal (real Demucs)
plausibly recover them, or are they a structural ceiling?
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scratchpad")); sys.path.insert(0, str(REPO / "scripts"))
from harmonia.data.corpus_schema import load_corpus, sounding_bass_pc
from multihead_training import train_clf, predict_proba
from rwc_nnls_multihead_cv import song_split

nn = load_corpus(REPO / "data/cache/rwc/rwc_nnls24.npz")
bp = load_corpus(REPO / "data/cache/rwc/rwc_bp48_fixed.npz")
py = np.load(REPO / "scratchpad/pyin_bass_cache.npz", allow_pickle=True)
nn24 = nn["nnls24"].astype(np.float32); keep = np.abs(nn24).sum(1) > 0
nn24 = nn24[keep]; roots = bp["root"].astype(np.int64)[keep] % 12
sid = bp["song_id"][keep]; labels = bp["labels"][keep]
t0 = bp["t0"][keep].astype(float); t1 = bp["t1"][keep].astype(float); dur = t1 - t0
quals = bp["quality_idx"].astype(np.int64)[keep]
QUAL = ['maj', 'min', 'dom', 'hdim', 'dim', 'aug', 'sus']
py_bass = py["bass_pc"][keep].astype(np.int64)
_gbl = [sounding_bass_pc(str(labels[i]), int(roots[i])) for i in range(len(roots))]
gb = np.array([(-1 if v is None else v % 12) for v in _gbl], dtype=np.int64)
argmax = nn24[:, :12].argmax(1)
inv = gb != roots
has_py = py_bass >= 0

# gather pooled test preds (5 seeds) for argmax/headf/headb/pyin
allfail = np.zeros(len(gb), bool)   # OR over seeds: was this row an oracle-fail when in test
intest = np.zeros(len(gb), bool)
for seed in range(5):
    tr, va, te = song_split(sid, seed)
    mf = train_clf(nn24[tr], gb[tr], nn24[va], gb[va], 24, 12, hid=(128, 64), epochs=50)
    mb = train_clf(nn24[tr][:, :12], gb[tr], nn24[va][:, :12], gb[va], 12, 12, hid=(128, 64), epochs=50)
    pf = predict_proba(mf, nn24).argmax(1); pb = predict_proba(mb, nn24[:, :12]).argmax(1)
    correct = (argmax == gb) | (pf == gb) | (pb == gb)
    correct = correct | (has_py & (py_bass == gb))   # pYIN helps only where covered
    fail = te & ~correct
    allfail |= fail; intest |= te

F = allfail  # oracle-fail rows (pooled)
print(f"oracle-fail rows (pooled test): {F.sum()} / {intest.sum()} test = {F.mean()/max(intest.mean(),1e-9):.3f} frac")
print(f"\n-- inversion split of failures --")
print(f"   inversions: {(F&inv).sum()} ({(F&inv).sum()/F.sum():.2%} of fails); rootpos: {(F&~inv).sum()}")
print(f"   (base rates: inv={inv[intest].mean():.3%})")

print(f"\n-- duration (fails vs all test) --")
qs = [0, 25, 50, 75, 100]
print(f"   fail dur pctiles {np.round(np.percentile(dur[F], qs),2)}")
print(f"   test dur pctiles {np.round(np.percentile(dur[intest], qs),2)}")
short = dur < np.percentile(dur[intest], 25)
print(f"   fail rate in shortest-quartile spans: {(F&short).sum()/max((intest&short).sum(),1):.3f} "
      f"vs longest-quartile: {(F&(dur>np.percentile(dur[intest],75))).sum()/max((intest&(dur>np.percentile(dur[intest],75))).sum(),1):.3f}")

print(f"\n-- error interval (argmax_pred - true_bass) mod 12, on fails --")
di = (argmax[F] - gb[F]) % 12
for k in range(12):
    c = (di == k).sum()
    if c: print(f"   +{k:2d} semitones: {c:4d} ({c/F.sum():.2%})")

print(f"\n-- quality of failing chords vs test --")
for q in range(7):
    fq = (F & (quals == q)).sum(); tq = (intest & (quals == q)).sum()
    if tq: print(f"   {QUAL[q]:5s} fail-rate={fq/tq:.3f}  (n_test={tq})")

print(f"\n-- song concentration --")
songs = np.unique(sid[intest])
frates = []
for s in songs:
    m = intest & (sid == s)
    if m.sum() >= 20: frates.append((F & m).sum()/m.sum())
frates = np.array(sorted(frates, reverse=True))
print(f"   per-song fail-rate: median={np.median(frates):.3f}  top-5={np.round(frates[:5],2)}  "
      f"bottom-5={np.round(frates[-5:],2)}")
print(f"   songs with >40% fail-rate: {(frates>0.40).sum()}/{len(frates)}")
