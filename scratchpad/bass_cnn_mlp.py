import numpy as np
from collections import Counter
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

p = np.load('scratchpad/bass_cnn_prep.npz', allow_pickle=True)
bass_abs=p['bass_abs']; fabs=p['fabs']; y=p['bass_true']; song=p['song']
N=len(y); songs=np.unique(song)

def roll_rows(X, shifts):
    out=np.empty_like(X)
    for i in range(len(X)):
        out[i]=np.roll(X[i], -shifts[i])
    return out

def cv(make_X_y, seeds=5, hidden=(64,32), predist=False):
    accs=[]; alldist=Counter()
    for s in range(seeds):
        rng=np.random.RandomState(s); sh=songs.copy(); rng.shuffle(sh)
        cut=int(0.8*len(sh)); trs=set(sh[:cut]); tes=set(sh[cut:])
        trm=np.array([x in trs for x in song]); tem=~trm
        Xtr,ytr,_,_=make_X_y(trm)
        Xte,yte,argm_te,_=make_X_y(tem)
        sc=StandardScaler().fit(Xtr)
        clf=MLPClassifier(hidden,max_iter=300,random_state=s,early_stopping=True)
        clf.fit(sc.transform(Xtr),ytr)
        pred=clf.predict(sc.transform(Xte))
        if predist:   # pred is RELATIVE -> shift back to absolute
            pred_abs=(pred+argm_te)%12
        else:
            pred_abs=pred
        acc=(pred_abs==y[tem]).mean(); accs.append(acc)
        alldist.update(pred_abs.tolist())
    return np.mean(accs),np.std(accs),alldist

# ---- Config A: absolute pooled MLP, bass-12 only ----
def A(mask):
    X=bass_abs[mask]; return X,y[mask],p['argm'][mask],None
# ---- Config B: argmax-renorm, bass-12, relative target + shift back ----
def B(mask):
    Xa=bass_abs[mask]; am=Xa.argmax(1)
    Xr=roll_rows(Xa,am); yr=(y[mask]-am)%12
    return Xr,yr,am,None
# ---- Config C: absolute pooled MLP, full 48 ----
def C(mask):
    X=fabs[mask]; return X,y[mask],p['argm'][mask],None
# ---- Config D: argmax(bass)-renorm applied to ALL 4 blocks, relative target ----
def D(mask):
    Fa=fabs[mask].copy(); ba=Fa[:,24:36]; am=ba.argmax(1)
    blocks=[Fa[:,0:12],Fa[:,12:24],Fa[:,24:36],Fa[:,36:48]]
    rot=np.hstack([roll_rows(b,am) for b in blocks])
    yr=(y[mask]-am)%12
    return rot,yr,am,None

for name,fn,pd_ in [("A abs pooled bass12",A,False),
                    ("B RENORM bass12",B,True),
                    ("C abs pooled full48",C,False),
                    ("D RENORM full48",D,True)]:
    m,sd,dist=cv(fn,predist=pd_)
    tot=sum(dist.values()); top=dist.most_common(3)
    frac0=dist.get(0,0)/tot
    print(f"{name:24s} acc={m:.4f}±{sd:.4f}  pred-dist top3={top} frac_C={frac0:.3f}")

# true label distribution for reference
td=Counter(y.tolist()); tot=sum(td.values())
print("TRUE bass dist frac per class (should be ~uniform-ish):",
      {k:round(v/tot,3) for k,v in sorted(td.items())})
