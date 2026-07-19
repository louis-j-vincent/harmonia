"""Evaluate REAL Demucs-pYIN bass vs 400Hz-PROXY pYIN vs NNLS estimators, on the
songs Demucs has processed. Headline: does the cleaner bass stem recover more of the
HARD RESIDUAL (chords where all 3 NNLS estimators fail)? All numbers from completed runs.
"""
import sys, numpy as np
from pathlib import Path
sys.path.insert(0, '.'); sys.path.insert(0, 'scratchpad'); sys.path.insert(0, 'scripts')
from harmonia.data.corpus_schema import load_corpus, sounding_bass_pc
from multihead_training import train_clf, predict_proba
from rwc_nnls_multihead_cv import song_split

nn = load_corpus('data/cache/rwc/rwc_nnls24.npz'); bp = load_corpus('data/cache/rwc/rwc_bp48_fixed.npz')
proxy = np.load('scratchpad/pyin_bass_cache.npz', allow_pickle=True)
dem = np.load('scratchpad/demucs_bass_cache.npz', allow_pickle=True)
dem_songs = set(dem['songs_done'].tolist())
nn24 = nn['nnls24'].astype('float32'); keep = np.abs(nn24).sum(1) > 0
nn24 = nn24[keep]; roots = bp['root'].astype('int64')[keep] % 12
sid = bp['song_id'][keep]; labels = bp['labels'][keep]
_gbl = [sounding_bass_pc(str(labels[i]), int(roots[i])) for i in range(len(roots))]
gb = np.array([(-1 if v is None else v % 12) for v in _gbl], dtype=np.int64)
argmax = nn24[:, :12].argmax(1)
proxy_bass = proxy['bass_pc'][keep].astype('int64')
dem_bass = dem['bass_pc'][keep].astype('int64')
in_dem = np.isin(sid, list(dem_songs)) & (dem_bass >= 0)
inv = gb != roots
print(f"Demucs-covered songs: {len(dem_songs)}; chords with Demucs bass: {in_dem.sum()}", flush=True)

# trained heads pooled over test folds, restricted to Demucs-covered rows
hardmask = np.zeros(len(gb), bool); tested = np.zeros(len(gb), bool)
headf_correct = np.zeros(len(gb), bool)
for seed in range(5):
    tr, va, te = song_split(sid, seed)
    mf = train_clf(nn24[tr], gb[tr], nn24[va], gb[va], 24, 12, hid=(128, 64), epochs=50)
    mb = train_clf(nn24[tr][:, :12], gb[tr], nn24[va][:, :12], gb[va], 12, 12, hid=(128, 64), epochs=50)
    pf = predict_proba(mf, nn24).argmax(1); pb = predict_proba(mb, nn24[:, :12]).argmax(1)
    m = te & in_dem
    any3 = (argmax == gb) | (pf == gb) | (pb == gb)
    hardmask |= (m & ~any3); tested |= m; headf_correct |= (m & (pf == gb))

cov = tested  # Demucs+test rows
def acc(pred, mask): return float((pred[mask] == gb[mask]).mean()) if mask.sum() else float('nan')
print(f"\nOn Demucs+test rows (n={cov.sum()}, {int((cov&inv).sum())} inv):")
for lbl, pred in [("NNLS argmax", argmax), ("proxy-pYIN", proxy_bass), ("Demucs-pYIN", dem_bass)]:
    print(f"   {lbl:14s} all={acc(pred,cov):.3f}  inv={acc(pred,cov&inv):.3f}")
H = hardmask
print(f"\nHARD RESIDUAL (all 3 NNLS estimators fail): n={H.sum()} ({H.sum()/max(cov.sum(),1):.3f} of covered)")
print(f"   proxy-pYIN correct on hard: {acc(proxy_bass,H):.3f}")
print(f"   Demucs-pYIN correct on hard: {acc(dem_bass,H):.3f}")
# combined ceiling: any of {argmax,headf,headb, X}
def ceil(x):
    c = np.zeros(cov.sum(), bool)
    # reconstruct any3 correctness on cov via headf_correct + argmax; approximate with stored
    base = (argmax[cov] == gb[cov]) | headf_correct[cov]
    return float((base | (x[cov] == gb[cov])).mean())
print(f"\nCeiling {{argmax,headf}}+X on covered rows:")
print(f"   + proxy-pYIN:  {ceil(proxy_bass):.3f}")
print(f"   + Demucs-pYIN: {ceil(dem_bass):.3f}")
