"""Adjudicate: is the oracle-bass quality gap a TRAINING/CAPACITY artifact or STRUCTURAL?

Prior result (scratchpad/oracle_bass_family.py, MLP(64,32), early_stopping, max_iter=300):
  ORACLE-BASS strict 0.709 ; ORACLE-ROOT strict 0.767 ; gap -5.8pp, -24pp on inversions.
Prior interpretation: structural (root-relative quality => bass anchor needs inversion-
dependent templates).  User pushes back: it's undertraining/capacity.

This script rules out "undertrained" rigorously:
  STEP 0  reproduce baseline (64,32).
  STEP 1  capacity + training length sweep on the SAME features.
  STEP 2  DECISIVE test: rotation augmentation. The root- and bass-anchored feature
          vectors of the SAME chord are cyclic ROTATIONS of one another (same 4x12
          shape, shifted by the inversion degree). Rotation is a bijection => NO
          information is lost by bass-anchoring; the only cost is that an MLP is not
          rotation-invariant and bass-anchoring spreads each quality's canonical shape
          across inversion-dependent offsets. If we train with random cyclic rotation
          augmentation (same roll applied to all 4 blocks), a rotation-ROBUST MLP should
          let ORACLE-BASS converge toward ORACLE-ROOT. If the gap CLOSES -> artifact
          (user right). If it PERSISTS -> structural (prior agent right).
  STEP 3  auxiliary bass->root interval feature: give the bass model the info it "loses".
"""
import numpy as np, time
from collections import Counter
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

d = np.load('data/cache/rwc/rwc_bp48_fixed.npz', allow_pickle=True)
labels = d['labels']; root = d['root'].astype(int); song = d['song_id']
qidx = d['quality_idx'].astype(int); QVOCAB = [str(q) for q in d['qualities']]
fabs = d['feat48_abs'].astype(np.float32)
frel = d['feat48'].astype(np.float32)
N = len(labels); songs = np.unique(song)

DEG = {'1':0,'b2':1,'2':2,'b3':3,'3':4,'4':5,'#4':6,'b5':6,'5':7,'#5':8,'b6':8,
       '6':9,'bb7':9,'b7':10,'7':11,'#7':0,'9':2,'b9':1,'#9':3,'11':5,'#11':6,'13':9,'b13':8}
def sounding_bass(lab, r):
    if '/' in lab:
        inv = lab.split('/')[1].strip()
        if inv in DEG: return (r + DEG[inv]) % 12
    return r % 12
bass_true = np.array([sounding_bass(l, root[i]) for i, l in enumerate(labels)])
inv_mask = np.array(['/' in l and l.split('/')[1].strip() in DEG for l in labels])
inv_deg  = np.array([DEG.get(l.split('/')[1].strip(),0) if '/' in l else 0 for l in labels])

THIRD = {'maj':0,'dom':0,'aug':0,'min':1,'hdim':1,'dim':1,'sus':2}
third_of_qidx = np.array([THIRD[QVOCAB[q]] for q in range(len(QVOCAB))])

def roll_all_blocks(F, shifts):
    out = np.empty_like(F)
    for i in range(len(F)):
        s = int(shifts[i])
        for b in range(4):
            out[i, b*12:(b+1)*12] = np.roll(F[i, b*12:(b+1)*12], -s)
    return out

def roll_uniform(Xrows, s):
    # roll every 12-block of a batch by the same integer s (vectorized)
    out = np.empty_like(Xrows)
    for b in range(4):
        out[:, b*12:(b+1)*12] = np.roll(Xrows[:, b*12:(b+1)*12], -s, axis=1)
    return out

X_oracle = roll_all_blocks(fabs, bass_true)
X_root   = frel

# sanity: on root-position chords, oracle-bass == oracle-root
rp = ~inv_mask
assert np.allclose(X_oracle[rp], X_root[rp], atol=1e-5), "root-position frames should match"
print(f"N={N} songs={len(songs)} inv={inv_mask.mean():.3f} | root-pos frames identical: OK")

def split(seed):
    rng = np.random.RandomState(seed); sh = songs.copy(); rng.shuffle(sh)
    cut = int(0.8*len(sh)); trs = set(sh[:cut])
    trm = np.array([x in trs for x in song]); return trm, ~trm

def evaluate(X, hidden=(64,32), seeds=5, max_iter=300, early=True, alpha=1e-4,
             augment=0, lr_init=1e-3, Xaug_source=None, tag=''):
    """augment=k: for each train row add k random cyclic-rotated copies (same roll on all blocks)."""
    strict, inv_strict, bal = [], [], []
    for s in range(seeds):
        trm, tem = split(s)
        Xtr, ytr = X[trm], qidx[trm]
        if augment > 0:
            src = X if Xaug_source is None else Xaug_source
            base = src[trm]
            extra_X, extra_y = [], []
            rng = np.random.RandomState(1000+s)
            for _ in range(augment):
                shifts = rng.randint(0,12,size=base.shape[0])
                # per-row uniform roll
                rolled = np.empty_like(base)
                for sh_v in range(12):
                    m = shifts==sh_v
                    if m.any(): rolled[m] = roll_uniform(base[m], sh_v)
                extra_X.append(rolled); extra_y.append(qidx[trm])
            Xtr = np.vstack([Xtr]+extra_X); ytr = np.concatenate([ytr]+extra_y)
        sc = StandardScaler().fit(Xtr)
        clf = MLPClassifier(hidden, max_iter=max_iter, random_state=s,
                            early_stopping=early, alpha=alpha,
                            learning_rate_init=lr_init, n_iter_no_change=15)
        clf.fit(sc.transform(Xtr), ytr)
        pred = clf.predict(sc.transform(X[tem])); yte = qidx[tem]
        strict.append((pred==yte).mean())
        bal.append(balanced_accuracy_score(yte,pred))
        im = inv_mask[tem]
        inv_strict.append((pred[im]==yte[im]).mean() if im.sum() else np.nan)
    return (np.mean(strict),np.std(strict),np.mean(inv_strict),np.std(inv_strict),
            np.mean(bal), clf.n_iter_)

def show(tag, r):
    print(f"{tag:44s} strict {r[0]:.4f}±{r[1]:.3f}  inv {r[2]:.4f}±{r[3]:.3f}  "
          f"bal {r[4]:.3f}  iters~{r[5]}")

print("\n=== STEP 0: reproduce baseline (64,32, early_stop, max_iter=300) ===")
show("ORACLE-ROOT  (64,32)  baseline", evaluate(X_root))
show("ORACLE-BASS  (64,32)  baseline", evaluate(X_oracle))


print("\n=== STEP 1: capacity sweep (rule out too-small model) ===", flush=True)
for hid in [(128,64),(256,128,64)]:
    show(f"ORACLE-BASS {str(hid):14s} early_stop", evaluate(X_oracle, hidden=hid))
    show(f"ORACLE-ROOT {str(hid):14s} early_stop", evaluate(X_root,   hidden=hid))

print("\n=== STEP 1b: convergence check (undertrained? train long, no early stop) ===", flush=True)
show("ORACLE-BASS (256,128,64) noES iter=600 a=3e-4", evaluate(X_oracle, hidden=(256,128,64), early=False, max_iter=600, alpha=3e-4))
show("ORACLE-ROOT (256,128,64) noES iter=600 a=3e-4", evaluate(X_root,   hidden=(256,128,64), early=False, max_iter=600, alpha=3e-4))

print("\n=== STEP 3: auxiliary bass->root interval one-hot (give back 'lost' info) ===", flush=True)
invdeg_oh = np.zeros((N,12), np.float32); invdeg_oh[np.arange(N), inv_deg] = 1.0
X_oracle_aux = np.hstack([X_oracle, invdeg_oh]).astype(np.float32)
show("ORACLE-BASS + inv-degree one-hot (256,128,64)", evaluate(X_oracle_aux, hidden=(256,128,64), early=False, max_iter=600, alpha=3e-4))

print("\n=== STEP 4: data-scaling curve (is root's edge scale-invariant?) ===", flush=True)
def evaluate_frac(X, frac, hidden=(128,64), seeds=5):
    strict, inv_s = [], []
    for s in range(seeds):
        trm, tem = split(s)
        idx = np.where(trm)[0]
        rng = np.random.RandomState(500+s); rng.shuffle(idx)
        keep = idx[:int(frac*len(idx))]
        sc = StandardScaler().fit(X[keep])
        clf = MLPClassifier(hidden, max_iter=300, random_state=s, early_stopping=True, n_iter_no_change=15)
        clf.fit(sc.transform(X[keep]), qidx[keep])
        pred = clf.predict(sc.transform(X[tem])); yte = qidx[tem]
        strict.append((pred==yte).mean())
        im = inv_mask[tem]; inv_s.append((pred[im]==yte[im]).mean() if im.sum() else np.nan)
    return np.mean(strict), np.mean(inv_s)
print(f"{'frac':>6s} {'ROOT strict':>12s} {'BASS strict':>12s} {'gap':>7s} | {'ROOT inv':>9s} {'BASS inv':>9s}")
for frac in [0.25, 0.5, 1.0]:
    rr = evaluate_frac(X_root, frac); bb = evaluate_frac(X_oracle, frac)
    print(f"{frac:6.2f} {rr[0]:12.4f} {bb[0]:12.4f} {rr[0]-bb[0]:+7.4f} | {rr[1]:9.4f} {bb[1]:9.4f}", flush=True)

print("\n=== STEP 2: DECISIVE rotation-augmentation (structural vs artifact) ===", flush=True)
print("root/bass frames of a chord are cyclic ROTATIONS of each other (bijection => no", flush=True)
print("info lost). If gap is only 'MLP not rotation-invariant', rot-aug closes it.", flush=True)
for k in [4]:
    show(f"ORACLE-ROOT aug x{k} (256,128,64)", evaluate(X_root,   hidden=(256,128,64), early=False, max_iter=300, augment=k))
    show(f"ORACLE-BASS aug x{k} (256,128,64)", evaluate(X_oracle, hidden=(256,128,64), early=False, max_iter=300, augment=k))
print("DONE", flush=True)
