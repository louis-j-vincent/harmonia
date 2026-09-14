import sys, pickle
from pathlib import Path
import numpy as np
SD=Path("/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-Code-harmonia/8a011198-4935-4f2e-a73e-da83232ee2cd/scratchpad")
data=pickle.load(open(SD/"secfeat.pkl","rb"))
FEAT=["chord_recur","phrase_restart","drum_fill","energy_nov","timbre_nov","nc_adj","phrase_pos","harm_rhythm"]
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

def f1_at(scores, labels, bars, gt_bars, k):
    # take top-k scoring candidates as predicted boundaries; F1 vs GT (±1 bar tol)
    idx=np.argsort(scores)[::-1][:k]
    pred=set(bars[i] for i in idx)
    tp=sum(1 for p in pred if any(abs(p-g)<=1 for g in gt_bars))
    fp=len(pred)-tp
    matched=set(g for g in gt_bars if any(abs(p-g)<=1 for p in pred))
    fn=len(gt_bars)-len(matched)
    prec=tp/max(tp+fp,1); rec=len(matched)/max(len(gt_bars),1)
    return 2*prec*rec/max(prec+rec,1e-9)

# LOSO CV
allX=np.array([r[0] for d in data for r in d["rows"]])
scaler=StandardScaler().fit(allX)
loso_f1=[]; base_f1=[]
for held in data:
    tr=[d for d in data if d["name"]!=held["name"]]
    Xtr=scaler.transform(np.array([r[0] for d in tr for r in d["rows"]]))
    ytr=np.array([r[1] for d in tr for r in d["rows"]])
    clf=LogisticRegression(class_weight="balanced",max_iter=1000).fit(Xtr,ytr)
    Xh=scaler.transform(np.array([r[0] for r in held["rows"]]))
    bars=[r[2] for r in held["rows"]]; labels=[r[1] for r in held["rows"]]
    scores=clf.decision_function(Xh)
    k=len(held["gt_bars"])
    loso_f1.append(f1_at(scores,labels,bars,held["gt_bars"],k))
    # baseline: phrase-position only (predict at 8-bar multiples nearest GT count)
    phrase_score=np.array([-abs(r[0][6]) for r in held["rows"]])  # phrase_pos feature
    base_f1.append(f1_at(phrase_score,labels,bars,held["gt_bars"],k))
    print(f"  {held['name']}: model F1 {loso_f1[-1]:.2f}  phrase-only F1 {base_f1[-1]:.2f}")

print(f"\nLOSO mean F1: model {np.mean(loso_f1):.3f}  vs phrase-position baseline {np.mean(base_f1):.3f}")
# full-data logistic for feature importances
X=scaler.transform(allX); y=np.array([r[1] for d in data for r in d["rows"]])
clf=LogisticRegression(class_weight="balanced",max_iter=1000).fit(X,y)
print("\nFeature weights (standardized logistic, + = predicts boundary):")
for name,w in sorted(zip(FEAT,clf.coef_[0]),key=lambda x:-abs(x[1])):
    print(f"  {name:12s} {w:+.3f}")
# univariate correlation with label
print("\nUnivariate corr(feature, is-boundary):")
for i,name in enumerate(FEAT):
    c=np.corrcoef(allX[:,i],y)[0,1]
    print(f"  {name:12s} {c:+.3f}")
