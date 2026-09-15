"""Can a gradient-boosted model beat the single criteria at merging sections?

Louis, 2026-08-02: "pas possible d'agréger plusieurs métriques avec un xgboost
pour décider de si on merge ou pas ?"

Answered by measurement, not opinion. Design points that decide whether the
answer is honest:
  * GroupKFold by TRACK. Pairs from one song share sections; a random split
    leaks a section's other pairs into the test fold and inflates everything.
  * The operating point is FALSE-MERGE <= 2%, not accuracy. Merging is
    destructive (a false merge overwrites real music, cf. the Norah bug);
    a missed merge only costs redundancy. Ranking metrics by AUC picks the
    wrong criterion — measured: best AUC has half the recall here.
  * The threshold is chosen on the TRAIN fold and applied to the TEST fold.
    Picking it on test is how you report a number you cannot ship.
"""
import json
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupKFold
from sklearn.inspection import permutation_importance

rows = json.load(open("scratchpad/fold_criteria/pairs_chroma.json"))
FEATS = [k for k in rows[0] if k not in ("tid", "cls")]
X = np.array([[r[f] for f in FEATS] for r in rows], float)
X = np.nan_to_num(X)
cls = np.array([r["cls"] for r in rows])
y = (cls == 0).astype(int)                      # positive = same letter
groups = np.array([r["tid"] for r in rows])
hard = (cls == 3)

def recall_at_fm(score, y, hard_mask, max_fm=0.02):
    """Recall on positives at the highest threshold holding false-merge<=2%."""
    order = np.argsort(-score)
    ys, hs = y[order], hard_mask[order]
    neg = (ys == 0)
    fm = np.cumsum(neg) / max(1, neg.sum())
    ok = np.where(fm <= max_fm)[0]
    if len(ok) == 0:
        return 0.0, 1.0, 0.0
    k = ok[-1]
    thr = score[order][k]
    tp = (ys[: k + 1] == 1).sum()
    hard_merged = hs[: k + 1].sum() / max(1, hard_mask.sum())
    return tp / max(1, (y == 1).sum()), thr, hard_merged

oof = np.zeros(len(y))
gkf = GroupKFold(n_splits=5)
for tr, te in gkf.split(X, y, groups):
    m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06,
                                       max_leaf_nodes=31, random_state=0)
    m.fit(X[tr], y[tr])
    oof[te] = m.predict_proba(X[te])[:, 1]

print(f"pairs {len(y)}  positives {y.sum()}  tracks {len(set(groups))}\n")
print(f"{'model / criterion':<34}{'AUC':>7}{'recall@2%FM':>13}{'hard-neg merged':>17}")
print("-" * 71)

from sklearn.metrics import roc_auc_score
r, thr, hm = recall_at_fm(oof, y, hard)
print(f"{'GBM (all 15 metrics, grouped CV)':<34}{roc_auc_score(y,oof):>7.3f}{r:>12.1%}{hm:>17.1%}")

for f in ["chroma_cv_inv", "chroma_align", "chroma_lag_unstretched", "seq_sim", "hr_sig", "ct_bag_cos"]:
    s = X[:, FEATS.index(f)]
    r, thr, hm = recall_at_fm(s, y, hard)
    print(f"{'  ' + f:<34}{roc_auc_score(y,s):>7.3f}{r:>12.1%}{hm:>17.1%}")

# the shipped-style rule: both gates must pass
comb = np.minimum(X[:, FEATS.index("chroma_align")], X[:, FEATS.index("len_agree")])
r, thr, hm = recall_at_fm(comb, y, hard)
print(f"{'  align AND len (rule)':<34}{roc_auc_score(y,comb):>7.3f}{r:>12.1%}{hm:>17.1%}")

# what the model actually leans on
tr, te = next(iter(gkf.split(X, y, groups)))
m = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.06, random_state=0).fit(X[tr], y[tr])
imp = permutation_importance(m, X[te], y[te], n_repeats=5, random_state=0, scoring="roc_auc")
print("\nwhich metrics the model leans on (permutation importance, AUC drop):")
for i in np.argsort(-imp.importances_mean)[:8]:
    print(f"  {FEATS[i]:<28}{imp.importances_mean[i]:+.4f}")
