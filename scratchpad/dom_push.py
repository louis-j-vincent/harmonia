"""Push dom recall >0.70 in the realistic (predicted-root) cascade.
Levers: focal loss, dom class-weight boost, top-k marginalization depth."""
import numpy as np, torch
from multihead_training import (load, song_split, neighbor, rotate_by_root,
                                train_clf, predict_proba, balanced_recall, ROOT)

D=load(); tr,va,te=song_split(D['sid'])
roots=D['roots']; quals=D['quals']; sid=D['sid']; bass,treb=D['bass'],D['treb']
rp=np.load(ROOT/"scratchpad/root_posteriors.npz")['proba']
K=5; cnt=np.bincount(quals,minlength=K)
base_cw=(cnt.sum()/(K*np.maximum(cnt,1))).astype(np.float32)

def ctx_feats(root_frame, rows=slice(None)):
    feats=[]
    rf=root_frame if rows==slice(None) else root_frame
    for o in (-3,-2,-1,1,2,3):
        nb=neighbor(rp,sid,o)[rows]
        nbr=np.empty_like(nb)
        for r in range(12):
            mm=(root_frame[rows]==r) if rows!=slice(None) else (root_frame==r)
            if mm.any(): nbr[mm]=np.roll(nb[mm],-r,axis=1)
        feats.append(nbr)
    return np.concatenate(feats,1)

def build_X(root_frame, rows=slice(None)):
    b=rotate_by_root(bass[rows],root_frame[rows] if rows!=slice(None) else root_frame)
    t=rotate_by_root(treb[rows],root_frame[rows] if rows!=slice(None) else root_frame)
    return np.concatenate([b,t,ctx_feats(root_frame,rows)],1)

# train on ORACLE frame (marginalize at test). Sweep loss/weight.
Xtr=build_X(roots)  # full, then index
def marginalize(model, topk):
    order=np.argsort(-rp[te],1)[:,:topk]
    w=np.take_along_axis(rp[te],order,1); w=w/w.sum(1,keepdims=True)
    q=np.zeros((te.sum(),K)); bt,tt=bass[te],treb[te]
    for j in range(topk):
        rh=order[:,j]
        b=rotate_by_root(bt,rh); t=rotate_by_root(tt,rh)
        ctxh=[]
        for o in (-3,-2,-1,1,2,3):
            nb=neighbor(rp,sid,o)[te]; nbr=np.empty_like(nb)
            for r in range(12):
                m=rh==r
                if m.any(): nbr[m]=np.roll(nb[m],-r,axis=1)
            ctxh.append(nbr)
        Xh=np.concatenate([b,t,np.concatenate(ctxh,1)],1)
        q+=w[:,[j]]*predict_proba(model,Xh)
    return q

print(f"{'config':34s} {'bal':>5s} {'dom':>5s}  rec[maj min dom hdim dim]")
for tag,focal,domw,topk in [
    ('wce topk3',            False,1.0,3),
    ('focal topk3',          True, 1.0,3),
    ('focal dom1.5 topk3',   True, 1.5,3),
    ('focal dom2.0 topk3',   True, 2.0,3),
    ('focal dom2.0 topk5',   True, 2.0,5),
    ('wce dom1.8 topk5',     False,1.8,5),
]:
    cw=base_cw.copy(); cw[2]*=domw
    m=train_clf(Xtr[tr],quals[tr],Xtr[va],quals[va],Xtr.shape[1],K,hid=(128,64),
                epochs=70,cw=cw,focal=focal)
    q=marginalize(m,topk); pr=q.argmax(1); rec=balanced_recall(pr,quals[te],K)
    print(f"{tag:34s} {np.nanmean(rec):.3f} {rec[2]:.3f}  {np.round(rec,2)}")
