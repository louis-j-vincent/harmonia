"""Head-to-head trained root head on REAL-plugin NNLS-24 vs BP48-48, identical
blocks, GroupKFold(5) by song. Also aggregates muddiness + untrained argmax."""
import json
from pathlib import Path
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
z = np.load(REPO / "scratchpad/nnls_real_feats.npz", allow_pickle=True)
sid = z["sid"]; root = z["root"]; nn = z["nnls24"]; bp = z["feat48a"]
print(f"{len(root)} blocks, {len(np.unique(sid))} songs, {len(np.unique(root))} roots")


def cv(X, factory):
    gkf = GroupKFold(n_splits=5); accs = []
    for tr, te in gkf.split(X, root, sid):
        clf = factory(); clf.fit(X[tr], root[tr])
        accs.append((clf.predict(X[te]) == root[te]).mean())
    return np.mean(accs), np.std(accs)


lr = lambda: LogisticRegression(max_iter=2000, C=1.0)
mlp = lambda: MLPClassifier(hidden_layer_sizes=(96,), max_iter=400, alpha=1e-3)

print("\n== trained root head (grouped 5-fold) ==")
for name, X in [("NNLS-24", nn), ("BP48-48", bp)]:
    a, s = cv(X, lr); print(f"{name}  LR   root acc {a:.3f} +/- {s:.3f}")
for name, X in [("NNLS-24", nn), ("BP48-48", bp)]:
    a, s = cv(X, mlp); print(f"{name}  MLP  root acc {a:.3f} +/- {s:.3f}")

print("\n== untrained argmax->root ==")
print(f"NNLS bass  {(nn[:,:12].argmax(1)==root).mean():.3f}")
print(f"NNLS treb  {(nn[:,12:].argmax(1)==root).mean():.3f}")
print(f"BP48 bass  {(bp[:,24:36].argmax(1)==root).mean():.3f}")
print(f"BP48 note  {(bp[:,12:24].argmax(1)==root).mean():.3f}")

mud = json.load(open(REPO / "scratchpad/nnls_real_muddiness.json"))
w = np.array([m["n"] for m in mud], float)
def wm(k): return float(np.average([m[k] for m in mud], weights=w))
print("\n== muddiness (block-weighted over %d songs) ==" % len(mud))
print(f"peak/mean:  NNLS-24 {wm('nnls_pm24'):.2f}  vs BP48-48 {wm('bp48_pm48'):.2f}")
print(f"peak/mean:  NNLS-treb12 {wm('nnls_pm_treb12'):.2f}  vs BP48-note12 {wm('bp48_pm_note12'):.2f}")
print(f"norm-ent:   NNLS-24 {wm('nnls_ent24'):.3f}  vs BP48-48 {wm('bp48_ent48'):.3f}")
nnwins = sum(1 for m in mud if m['nnls_pm24'] > m['bp48_pm48'])
print(f"NNLS peak/mean wins {nnwins}/{len(mud)} songs")
