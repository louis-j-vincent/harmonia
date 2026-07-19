import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
p=np.load('scratchpad/bass_cnn_prep.npz',allow_pickle=True)
d=np.load('data/cache/rwc/rwc_bp48_fixed.npz',allow_pickle=True)
bass_abs=p['bass_abs']; y=p['bass_true']; song=p['song']; labels=d['labels']
DEG={'1':0,'b2':1,'2':2,'b3':3,'3':4,'4':5,'#4':6,'b5':6,'5':7,'#5':8,'b6':8,'6':9,'bb7':9,'b7':10,'7':11}
inv=np.array(['/' in l and l.split('/')[1].strip() in DEG for l in labels])
songs=np.unique(song)
def rollr(X,s):
    o=np.empty_like(X)
    for i in range(len(X)): o[i]=np.roll(X[i],-s[i])
    return o
Aacc=[];Bacc=[];Ainv=[];Binv=[]
for s in range(5):
    rng=np.random.RandomState(s); sh=songs.copy(); rng.shuffle(sh)
    trs=set(sh[:int(0.8*len(sh))]); trm=np.array([x in trs for x in song]); tem=~trm
    # A absolute
    sc=StandardScaler().fit(bass_abs[trm])
    cA=MLPClassifier((64,32),max_iter=300,random_state=s,early_stopping=True).fit(sc.transform(bass_abs[trm]),y[trm])
    pA=cA.predict(sc.transform(bass_abs[tem]))
    # B renorm
    amtr=bass_abs[trm].argmax(1); Xtr=rollr(bass_abs[trm],amtr); ytr=(y[trm]-amtr)%12
    amte=bass_abs[tem].argmax(1); Xte=rollr(bass_abs[tem],amte)
    scB=StandardScaler().fit(Xtr)
    cB=MLPClassifier((64,32),max_iter=300,random_state=s,early_stopping=True).fit(scB.transform(Xtr),ytr)
    pB=(cB.predict(scB.transform(Xte))+amte)%12
    ie=inv[tem]
    Aacc.append((pA==y[tem]).mean()); Bacc.append((pB==y[tem]).mean())
    Ainv.append((pA[ie]==y[tem][ie]).mean()); Binv.append((pB[ie]==y[tem][ie]).mean())
print("ALL     A(abs)=%.4f  B(renorm)=%.4f"%(np.mean(Aacc),np.mean(Bacc)))
print("INVonly A(abs)=%.4f  B(renorm)=%.4f  (n_inv/test~%d)"%(np.mean(Ainv),np.mean(Binv),int(inv.sum()*0.2)))
