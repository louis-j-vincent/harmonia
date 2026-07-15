"""Architecture sweep for the quality head (mission A-D): MLP vs CNN-1D vs LSTM
over the [3-before, target, 3-after] chroma sequence. Oracle-frame (root-relative);
reports test balanced acc + dom recall. Also re-saves the winning marginalized head."""
import numpy as np, torch, torch.nn as nn
from multihead_training import (load, song_split, neighbor, rotate_by_root,
                                train_clf, predict_proba, balanced_recall, ROOT, MLP)

D=load(); tr,va,te=song_split(D['sid'])
roots=D['roots']; quals=D['quals']; sid=D['sid']; bass,treb=D['bass'],D['treb']
K=5; cnt=np.bincount(quals,minlength=K)
cw=(cnt.sum()/(K*np.maximum(cnt,1))).astype(np.float32); cw[2]*=1.8

# Build a 7-step sequence of root-relative 24-d chroma, each step rotated by the
# TARGET root (transposition-invariant window). Steps: -3..+3, boundary-zeroed.
def seq_tensor():
    steps=[]
    for o in (-3,-2,-1,0,1,2,3):
        b=neighbor(bass,sid,o); t=neighbor(treb,sid,o)
        # rotate each step by the TARGET root
        br=np.empty_like(b); tr_=np.empty_like(t)
        for r in range(12):
            m=roots==r
            if m.any():
                br[m]=np.roll(b[m],-r,axis=1); tr_[m]=np.roll(t[m],-r,axis=1)
        steps.append(np.concatenate([br,tr_],1))       # (N,24)
    return np.stack(steps,1)                             # (N,7,24)

S=seq_tensor()  # (N,7,24)
cwt=torch.tensor(cw)

def fit_eval(model, Xtr,ytr,Xva,yva,Xte,yte,epochs=50,lr=1e-3,bs=512):
    opt=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-4)
    Xtr=torch.tensor(Xtr,dtype=torch.float32); ytr=torch.tensor(ytr)
    Xva=torch.tensor(Xva,dtype=torch.float32); yva=torch.tensor(yva)
    best=(1e9,None); n=len(Xtr)
    for ep in range(epochs):
        model.train(); perm=torch.randperm(n)
        for i in range(0,n,bs):
            b=perm[i:i+bs]; opt.zero_grad()
            loss=nn.functional.cross_entropy(model(Xtr[b]),ytr[b],weight=cwt)
            loss.backward(); opt.step()
        model.eval()
        with torch.no_grad(): vl=nn.functional.cross_entropy(model(Xva),yva,weight=cwt).item()
        if vl<best[0]: best=(vl,{k:v.clone() for k,v in model.state_dict().items()})
    model.load_state_dict(best[1]); model.eval()
    with torch.no_grad():
        pr=torch.softmax(model(torch.tensor(Xte,dtype=torch.float32)),1).argmax(1).numpy()
    return balanced_recall(pr,yte,K)

class CNN1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.c=nn.Sequential(nn.Conv1d(24,64,3,padding=1),nn.BatchNorm1d(64),nn.ReLU(),
                             nn.Conv1d(64,64,3,padding=1),nn.BatchNorm1d(64),nn.ReLU())
        self.head=nn.Sequential(nn.Linear(64,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,K))
    def forward(self,x):            # x:(B,7,24)
        h=self.c(x.transpose(1,2))  # (B,64,7)
        return self.head(h.mean(2))

class LSTMq(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn=nn.LSTM(24,64,batch_first=True,bidirectional=True)
        self.head=nn.Sequential(nn.Linear(128,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,K))
    def forward(self,x):
        o,_=self.rnn(x); return self.head(o[:,3,:])  # center (target) step

print(f"{'arch':10s} {'bal':>5s} {'dom':>5s}  rec[maj min dom hdim dim]")
# MLP on flattened target-only (24d) for reference
mlp=MLP(24,K,(128,64))
rec=fit_eval(mlp,S[tr][:,3,:],quals[tr],S[va][:,3,:],quals[va],S[te][:,3,:],quals[te])
print(f"{'MLP_tgt':10s} {np.nanmean(rec):.3f} {rec[2]:.3f}  {np.round(rec,2)}")
for name,mk in [('CNN1D',CNN1D),('LSTM',LSTMq)]:
    rec=fit_eval(mk(),S[tr],quals[tr],S[va],quals[va],S[te],quals[te])
    print(f"{name:10s} {np.nanmean(rec):.3f} {rec[2]:.3f}  {np.round(rec,2)}")
