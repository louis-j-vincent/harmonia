"""Head-to-head: train a root classifier on NNLS 24-dim vs BP48 feat48_abs,
identical blocks, GROUPED (by song) 5-fold CV so no song leaks across folds.
Reports root accuracy for each feature set with the same model."""
import sys
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
z = np.load(REPO/"scratchpad/nnls_feats.npz", allow_pickle=True)
sid=z["sid"]; root=z["root"]; nn=z["nnls24"]; bp=z["feat48a"]
groups=sid
print(f"{len(root)} blocks, {len(np.unique(sid))} songs, {len(np.unique(root))} roots")

def cv(X, clf_factory):
    gkf=GroupKFold(n_splits=5); accs=[]
    for tr,te in gkf.split(X,root,groups):
        clf=clf_factory(); clf.fit(X[tr],root[tr])
        accs.append((clf.predict(X[te])==root[te]).mean())
    return np.mean(accs), np.std(accs)

lr=lambda: LogisticRegression(max_iter=2000, C=1.0)
mlp=lambda: MLPClassifier(hidden_layer_sizes=(96,), max_iter=400, alpha=1e-3)

for name,X in [("NNLS-24",nn),("BP48-48",bp)]:
    a,s=cv(X,lr); print(f"{name}  LR   root acc {a:.3f} +/- {s:.3f}")
for name,X in [("NNLS-24",nn),("BP48-48",bp)]:
    a,s=cv(X,mlp); print(f"{name}  MLP  root acc {a:.3f} +/- {s:.3f}")

# argmax proxies (no training)
print(f"argmax NNLS bass  {(nn[:,:12].argmax(1)==root).mean():.3f}")
print(f"argmax BP48 bass  {(bp[:,24:36].argmax(1)==root).mean():.3f}")
