"""2026-07-16 sounding-bass target redefinition — headline eval + renorm calibration.

Task 3: old-target (functional root) vs new-target (sounding bass pc) headline
        root-accuracy, multi-seed, with inverted-chord-subset breakdown.
Task 4: root-anchored renormalization -> predict bass pc, WITH calibration (ECE)
        and post-shift-back predicted-class histogram (degeneracy guard).

All bass targets are re-derived from the NEW resolver
`harmonia.data.corpus_schema.sounding_bass_pc` (NOT the old argmax / legacy
bass field), per coordinator guard-rail.
"""
from __future__ import annotations
import sys
from pathlib import Path
from collections import Counter
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from harmonia.data.corpus_schema import sounding_bass_pc

d = np.load(REPO / "data/cache/rwc/rwc_bp48_fixed.npz", allow_pickle=True)
labels = d["labels"]
root = d["root"].astype(int) % 12
song = d["song_id"]
fabs = d["feat48_abs"].astype(np.float32)   # [ch_on, ch_nt, bass, treble]
N = len(labels)

# ── NEW target from the resolver ─────────────────────────────────────────────
bass_true = np.array([sounding_bass_pc(l, int(root[i])) for i, l in enumerate(labels)])
# resolver returns None only for N/X — none expected among the parsed corpus rows
assert not any(b is None for b in bass_true), "unexpected None bass target"
bass_true = bass_true.astype(int)

# cross-check against the prior ad-hoc sounding_bass used to build the baseline
DEG = {'1':0,'b2':1,'2':2,'b3':3,'3':4,'4':5,'#4':6,'b5':6,'5':7,'#5':8,'b6':8,
       '6':9,'bb7':9,'b7':10,'7':11,'#7':0,'9':2,'b9':1,'#9':3,'11':5,'#11':6,'13':9,'b13':8}
def _old_sb(lab, r):
    if '/' in lab:
        inv = lab.split('/')[1].strip()
        if inv in DEG:
            return (r + DEG[inv]) % 12
    return r % 12
old_sb = np.array([_old_sb(l, int(root[i])) for i, l in enumerate(labels)])
print(f"[cross-check] new resolver == prior ad-hoc sounding_bass on all {N}: "
      f"{(bass_true == old_sb).all()} (mismatches={int((bass_true!=old_sb).sum())})")

inv = np.array(['/' in l for l in labels])
print(f"[corpus] N={N}  inverted(slash)={inv.sum()} ({inv.mean()*100:.2f}%)  "
      f"root-position={N-inv.sum()}")
print(f"[corpus] on inversions: bass_pc != root for {(bass_true[inv]!=root[inv]).mean()*100:.1f}%")

songs = np.unique(song)


def ece(conf, correct, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    e, M = 0.0, len(conf)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum() == 0:
            continue
        e += (m.sum() / M) * abs(correct[m].mean() - conf[m].mean())
    return e


def cv_target(y, X_fn, relative_anchor=None, seeds=5, hidden=(64, 32)):
    """Train MLP to predict y from X_fn(mask). If relative_anchor given, y is
    encoded RELATIVE to the anchor, predicted in the rotated frame, then
    SHIFTED BACK by +anchor before scoring (degeneracy guard reports hist)."""
    accs, inv_accs, eces = [], [], []
    pred_hist = Counter()
    for s in range(seeds):
        rng = np.random.RandomState(s)
        sh = songs.copy(); rng.shuffle(sh)
        cut = int(0.8 * len(sh)); trs = set(sh[:cut])
        trm = np.array([x in trs for x in song]); tem = ~trm
        Xtr = X_fn(trm); Xte = X_fn(tem)
        if relative_anchor is not None:
            atr, ate = relative_anchor[trm], relative_anchor[tem]
            ytr = (y[trm] - atr) % 12
        else:
            ytr = y[trm]
        sc = StandardScaler().fit(Xtr)
        clf = MLPClassifier(hidden, max_iter=300, random_state=s, early_stopping=True)
        clf.fit(sc.transform(Xtr), ytr)
        proba = clf.predict_proba(sc.transform(Xte))
        pred_rel = clf.classes_[proba.argmax(1)]
        if relative_anchor is not None:
            pred = (pred_rel + ate) % 12          # <-- SHIFT BACK
        else:
            pred = pred_rel
        conf = proba.max(1)
        correct = (pred == y[tem])
        accs.append(correct.mean())
        inv_accs.append(correct[inv[tem]].mean())
        eces.append(ece(conf, correct))
        pred_hist.update(pred.tolist())
    return (np.mean(accs), np.std(accs), np.mean(inv_accs),
            np.mean(eces), pred_hist)


# ══ TASK 3: old (functional root) vs new (sounding bass) headline ════════════
print("\n=== TASK 3: headline accuracy, functional-root vs sounding-bass target ===")
print("(pooled 48-d abs features, MLP(64,32), 5-seed song-grouped 80/20 CV)\n")

for name, y in [("OLD target: functional ROOT", root),
                ("NEW target: sounding BASS pc", bass_true)]:
    acc, sd, inv_acc, e, hist = cv_target(y, lambda m: fabs[m])
    print(f"{name:32s} acc={acc:.4f}±{sd:.4f}  inv-subset={inv_acc:.4f}  ECE={e:.4f}")

# same-model cross-scoring: a ROOT-trained model scored against the NEW target
print("\n-- same root-trained predictions, scored against BOTH label defs --")
acc_r, sd_r, _, _, _ = cv_target(root, lambda m: fabs[m])
# reuse: train on root, score vs bass (mismatched)
def cross_score(seeds=5):
    on_root, on_bass, on_root_inv, on_bass_inv = [], [], [], []
    for s in range(seeds):
        rng = np.random.RandomState(s); sh = songs.copy(); rng.shuffle(sh)
        cut = int(0.8*len(sh)); trs = set(sh[:cut])
        trm = np.array([x in trs for x in song]); tem = ~trm
        sc = StandardScaler().fit(fabs[trm])
        clf = MLPClassifier((64,32), max_iter=300, random_state=s, early_stopping=True)
        clf.fit(sc.transform(fabs[trm]), root[trm])
        pred = clf.predict(sc.transform(fabs[tem]))
        on_root.append((pred==root[tem]).mean()); on_bass.append((pred==bass_true[tem]).mean())
        im = inv[tem]
        on_root_inv.append((pred[im]==root[tem][im]).mean())
        on_bass_inv.append((pred[im]==bass_true[tem][im]).mean())
    return (np.mean(on_root),np.mean(on_bass),np.mean(on_root_inv),np.mean(on_bass_inv))
r_on_r, r_on_b, r_on_r_inv, r_on_b_inv = cross_score()
print(f"root-trained model  vs OLD(root): all={r_on_r:.4f} inv={r_on_r_inv:.4f}")
print(f"root-trained model  vs NEW(bass): all={r_on_b:.4f} inv={r_on_b_inv:.4f}  "
      f"(drop on inversions = model predicts functional root, not sounding bass)")


# ══ TASK 4: root-anchored renorm -> predict bass, calibrated ═════════════════
print("\n=== TASK 4: root-anchored chroma renorm -> predict SOUNDING BASS (calibrated) ===")
print("Anchor = GT functional root (candidate). Rotate a 12-d chroma block so anchor")
print("sits at index 0; target = bass RELATIVE to anchor (= inversion degree); shift back.\n")

def roll_rows(X, shifts):
    out = np.empty_like(X)
    for i in range(len(X)):
        out[i] = np.roll(X[i], -int(shifts[i]))
    return out

# candidate blocks: ch_on 0:12, ch_nt 12:24, bass 24:36, treble 36:48
BLOCKS = {"ch_on": (0,12), "ch_nt": (12,24), "treble": (36,48)}
bass_abs = fabs[:, 24:36]

print("BASELINE (absolute, non-renormalized):")
acc, sd, inv_acc, e, hist = cv_target(bass_true, lambda m: bass_abs[m])
top = hist.most_common(3); tot=sum(hist.values())
print(f"  A  abs pooled bass-12         acc={acc:.4f}±{sd:.4f} inv={inv_acc:.4f} "
      f"ECE={e:.4f}  pred top3={top} frac_C={hist.get(0,0)/tot:.3f}")

print("\nROOT-ANCHORED RENORM (rotate block by -root, predict bass-rel, shift back):")
for bname, (a, b) in BLOCKS.items():
    block = fabs[:, a:b]
    def Xfn(m, block=block):
        return roll_rows(block[m], root[m])
    acc, sd, inv_acc, e, hist = cv_target(bass_true, Xfn, relative_anchor=root)
    tot = sum(hist.values()); top = hist.most_common(3)
    print(f"  root-anchored {bname:7s} acc={acc:.4f}±{sd:.4f} inv={inv_acc:.4f} "
          f"ECE={e:.4f}  pred top3={top} frac_C={hist.get(0,0)/tot:.3f}")

# also: renorm ALL blocks by root, concat (48-d rotated)
def Xfn_all(m):
    blocks = [roll_rows(fabs[m][:, i:i+12], root[m]) for i in (0,12,24,36)]
    return np.hstack(blocks)
acc, sd, inv_acc, e, hist = cv_target(bass_true, Xfn_all, relative_anchor=root)
tot=sum(hist.values()); top=hist.most_common(3)
print(f"  root-anchored full48  acc={acc:.4f}±{sd:.4f} inv={inv_acc:.4f} "
      f"ECE={e:.4f}  pred top3={top} frac_C={hist.get(0,0)/tot:.3f}")

print("\nTRUE bass-pc distribution (base rates):",
      {k: round(v/N,3) for k,v in sorted(Counter(bass_true.tolist()).items())})
