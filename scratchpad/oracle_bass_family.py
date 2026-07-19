"""Oracle-bass-anchored chord FAMILY/QUALITY ceiling experiment.

Question: if bass detection were perfectly solved (oracle), how well can chord
quality be predicted from bass-anchored chroma alone? Isolates "bass is hard"
from "family is hard even with bass known".

Corpus: data/cache/rwc/rwc_bp48_fixed.npz (13204 chords / 100 songs).
  feat48_abs = [ch_on, ch_nt, bass, treble], each 12-d L2-normed, ABSOLUTE pitch.
  feat48     = same, but already rotated to the FUNCTIONAL ROOT (index 0).
  quality    = 7-way vocab [maj,min,dom,hdim,dim,aug,sus] = the family target.

Anchorings compared (target = quality, MLP(64,32), 5-seed song-grouped 80/20):
  ORACLE-BASS : rotate all 4 blocks of feat48_abs so SOUNDING-BASS pc -> idx 0.
                (ceiling: bass-detection error removed)
  ARGMAX-BASS : rotate all 4 blocks so pooled-bass-chroma argmax -> idx 0.
                (realistic proxy for today's bass detector, ~57% correct)
  ORACLE-ROOT : feat48 as-is (rotated to functional root). Non-bass-anchored
                baseline = the established family-classifier input.
  ABSOLUTE    : feat48_abs, no rotation (reference floor).

Scoring: strict 7-way accuracy + partial-credit "third-family" accuracy
  (collapse to major-3rd {maj,dom,aug} / minor-3rd {min,hdim,dim} / sus),
  i.e. MIREX majmin-level credit; plus balanced accuracy (class imbalance:
  always-maj floor = 56.4%).
"""
import numpy as np
from collections import Counter
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

d = np.load('data/cache/rwc/rwc_bp48_fixed.npz', allow_pickle=True)
labels = d['labels']; root = d['root'].astype(int); song = d['song_id']
qidx = d['quality_idx'].astype(int); QVOCAB = list(d['qualities'])
fabs = d['feat48_abs'].astype(np.float32)   # absolute
frel = d['feat48'].astype(np.float32)       # root-anchored
N = len(labels); songs = np.unique(song)

# --- sounding-bass pc from label (root + inversion degree) ---
DEG = {'1':0,'b2':1,'2':2,'b3':3,'3':4,'4':5,'#4':6,'b5':6,'5':7,'#5':8,'b6':8,
       '6':9,'bb7':9,'b7':10,'7':11,'#7':0,'9':2,'b9':1,'#9':3,'11':5,'#11':6,'13':9,'b13':8}
def sounding_bass(lab, r):
    if '/' in lab:
        inv = lab.split('/')[1].strip()
        if inv in DEG:
            return (r + DEG[inv]) % 12
    return r % 12
bass_true = np.array([sounding_bass(l, root[i]) for i, l in enumerate(labels)])
inv_mask = np.array(['/' in l and l.split('/')[1].strip() in DEG for l in labels])
bass_argmax = fabs[:, 24:36].argmax(1)

# --- third-family collapse for partial credit ---
# maj-3rd: maj,dom,aug | min-3rd: min,hdim,dim | sus: sus
THIRD = {'maj': 0, 'dom': 0, 'aug': 0, 'min': 1, 'hdim': 1, 'dim': 1, 'sus': 2}
third_of_qidx = np.array([THIRD[QVOCAB[q]] for q in range(len(QVOCAB))])

def roll_all_blocks(F, shifts):
    out = np.empty_like(F)
    for i in range(len(F)):
        s = shifts[i]
        for b in range(4):
            out[i, b*12:(b+1)*12] = np.roll(F[i, b*12:(b+1)*12], -s)
    return out

# Precompute rotated feature matrices once (rotation independent of split)
X_oracle = roll_all_blocks(fabs, bass_true)
X_argmax = roll_all_blocks(fabs, bass_argmax)
X_root   = frel                      # already root-anchored
X_abs    = fabs
CONFIGS = {
    'ORACLE-BASS': X_oracle,
    'ARGMAX-BASS': X_argmax,
    'ORACLE-ROOT (baseline)': X_root,
    'ABSOLUTE (floor)': X_abs,
}

def cv(X, y, seeds=5, hidden=(64, 32)):
    strict, partial, bal = [], [], []
    strict_inv = []
    cm = np.zeros((7, 7), int)
    for s in range(seeds):
        rng = np.random.RandomState(s); sh = songs.copy(); rng.shuffle(sh)
        cut = int(0.8 * len(sh)); trs = set(sh[:cut])
        trm = np.array([x in trs for x in song]); tem = ~trm
        sc = StandardScaler().fit(X[trm])
        clf = MLPClassifier(hidden, max_iter=300, random_state=s, early_stopping=True)
        clf.fit(sc.transform(X[trm]), y[trm])
        pred = clf.predict(sc.transform(X[tem]))
        yte = y[tem]
        strict.append((pred == yte).mean())
        partial.append((third_of_qidx[pred] == third_of_qidx[yte]).mean())
        bal.append(balanced_accuracy_score(yte, pred))
        invm = inv_mask[tem]
        if invm.sum() > 0:
            strict_inv.append((pred[invm] == yte[invm]).mean())
        for t, pr in zip(yte, pred):
            cm[t, pr] += 1
    return (np.mean(strict), np.std(strict), np.mean(partial), np.std(partial),
            np.mean(bal), np.std(bal), np.mean(strict_inv), np.std(strict_inv), cm)

print(f"N={N} chords, {len(songs)} songs, inversions={inv_mask.sum()} "
      f"({100*inv_mask.mean():.1f}%)")
print(f"quality dist: {Counter(QVOCAB[q] for q in qidx).most_common()}")
print(f"always-maj floor = {(qidx==QVOCAB.index('maj')).mean():.4f}")
print(f"bass-argmax==sounding-bass: {(bass_argmax==bass_true).mean():.4f}\n")

print(f"{'config':26s} {'strict':>14s} {'3rd-family':>14s} {'bal-acc':>14s} {'inv-strict':>13s}")
results = {}
for name, X in CONFIGS.items():
    r = cv(X, qidx)
    results[name] = r
    print(f"{name:26s} {r[0]:.4f}±{r[1]:.3f}  {r[2]:.4f}±{r[3]:.3f}  "
          f"{r[4]:.4f}±{r[5]:.3f}  {r[6]:.4f}±{r[7]:.3f}")

# deltas
ob = results['ORACLE-BASS']; ab = results['ARGMAX-BASS']; rt = results['ORACLE-ROOT (baseline)']
print("\n=== DELTAS (strict 7-way) ===")
print(f"oracle-bass - argmax-bass  = {ob[0]-ab[0]:+.4f}  (cost of imperfect bass detection)")
print(f"oracle-bass - oracle-root  = {ob[0]-rt[0]:+.4f}  (bass-anchor vs root-anchor, both oracle)")
print(f"  [inversions only] oracle-bass {ob[6]:.4f} vs oracle-root {rt[6]:.4f}  = {ob[6]-rt[6]:+.4f}")

print("\n=== ORACLE-BASS confusion (rows=true, cols=pred) ===")
print("        " + " ".join(f"{q:>5s}" for q in QVOCAB))
cm = ob[8]
for i, q in enumerate(QVOCAB):
    tot = cm[i].sum()
    rec = cm[i, i] / tot if tot else 0
    print(f"{q:>6s}  " + " ".join(f"{cm[i,j]:5d}" for j in range(7)) + f"   recall={rec:.3f} (n={tot})")
