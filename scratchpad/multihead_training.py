"""
Structured multi-head chord recognition — root head (bass-anchored) + quality
head (root-relative + learned trigram context) + optional 7th head.

Mission: Chord_AI-style structured decomposition on Billboard NNLS features.
Corpus: data/cache/bass_root_features.npz  (24-d = bass-chroma[0:12] + treble[12:24],
        A-referenced -> rolled by 9 to C-frame; roots 0..11 C-ref; quals maj/min/dom/hdim/dim)
Rows are sequential within song_id (verified prev_root==roots[i-1] exactly).

Run stages via CLI arg: root | quality | seventh | all
"""
import sys, json, numpy as np, torch, torch.nn as nn
from pathlib import Path

torch.manual_seed(0); np.random.seed(0)
ROOT = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
DEV = "cpu"
QUALS = ['maj','min','dom','hdim','dim']

# ---------- data ----------
def load():
    d = np.load(ROOT/"data/cache/bass_root_features.npz", allow_pickle=True)
    f = d['feats'].astype(np.float32)
    bass = np.roll(f[:,:12], 9, axis=1)      # -> C-frame
    treb = np.roll(f[:,12:], 9, axis=1)
    # L2 normalize each band (per frame) — keeps shape, kills loudness
    def l2(x): return x/ (np.linalg.norm(x,axis=1,keepdims=True)+1e-8)
    bass_n, treb_n = l2(bass), l2(treb)
    roots = d['roots'].astype(np.int64)
    quals = d['quals'].astype(np.int64)
    sid = d['song_id']
    return dict(bass=bass_n, treb=treb_n, bass_raw=bass, treb_raw=treb,
                roots=roots, quals=quals, sid=sid)

def song_split(sid, seed=42):
    songs = np.unique(sid); rng=np.random.RandomState(seed); rng.shuffle(songs)
    n=len(songs); tr=songs[:int(.8*n)]; va=songs[int(.8*n):int(.9*n)]; te=songs[int(.9*n):]
    idx = {s:i for i,s in enumerate(songs)}
    m = lambda ss: np.isin(sid, ss)
    return m(tr), m(va), m(te)

def neighbor(X, sid, off):
    """Shift X by `off` rows within-song; zero at song boundaries. off<0 = past, >0 = future."""
    out = np.zeros_like(X)
    n=len(X)
    if off<0:
        out[-off:] = X[:n+off]
        bad = sid[-off:]!=sid[:n+off]
        out[-off:][bad]=0
    elif off>0:
        out[:n-off] = X[off:]
        bad = sid[:n-off]!=sid[off:]
        out[:n-off][bad]=0
    return out

def p45_share(pred, true):
    err = pred!=true
    if err.sum()==0: return 0.0,0.0
    dp = (pred[err]-true[err])%12
    return err.mean(), np.isin(dp,[5,7]).mean()

def balanced_recall(pred, true, K):
    rec=[]
    for c in range(K):
        m=true==c
        rec.append((pred[m]==c).mean() if m.sum() else float('nan'))
    return np.array(rec)

# ---------- models ----------
class MLP(nn.Module):
    def __init__(self, din, dout, hid=(128,64), p=0.2):
        super().__init__()
        layers=[]; d=din
        for h in hid:
            layers += [nn.Linear(d,h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]; d=h
        layers += [nn.Linear(d,dout)]
        self.net=nn.Sequential(*layers)
    def forward(self,x): return self.net(x)

def train_clf(Xtr,ytr,Xva,yva,din,dout,hid=(128,64),epochs=60,wd=1e-4,cw=None,
              focal=False,lr=1e-3,bs=512,verbose=False):
    m=MLP(din,dout,hid).to(DEV)
    opt=torch.optim.Adam(m.parameters(),lr=lr,weight_decay=wd)
    Xtr=torch.tensor(Xtr,dtype=torch.float32); ytr=torch.tensor(ytr)
    Xva=torch.tensor(Xva,dtype=torch.float32); yva=torch.tensor(yva)
    if cw is not None: cw=torch.tensor(cw,dtype=torch.float32)
    best=(1e9,None,-1)
    n=len(Xtr)
    for ep in range(epochs):
        m.train(); perm=torch.randperm(n)
        for i in range(0,n,bs):
            b=perm[i:i+bs]; opt.zero_grad()
            logit=m(Xtr[b])
            if focal:
                ce=nn.functional.cross_entropy(logit,ytr[b],weight=cw,reduction='none')
                pt=torch.exp(-ce); loss=((1-pt)**2*ce).mean()
            else:
                loss=nn.functional.cross_entropy(logit,ytr[b],weight=cw)
            loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():
            vl=nn.functional.cross_entropy(m(Xva),yva,weight=cw).item()
        if vl<best[0]:
            best=(vl,{k:v.clone() for k,v in m.state_dict().items()},ep)
    m.load_state_dict(best[1]); m.eval()
    return m

def predict_proba(m,X):
    with torch.no_grad():
        return torch.softmax(m(torch.tensor(X,dtype=torch.float32)),1).numpy()

# ================= ROOT HEAD =================
def run_root(D, tr,va,te):
    print("\n===== ROOT HEAD =====")
    roots=D['roots']; sid=D['sid']
    bass,treb=D['bass'],D['treb']
    # feature variants
    ctx_bass = np.concatenate([neighbor(bass,sid,o) for o in (-1,1)],axis=1)
    ctx_treb = np.concatenate([neighbor(treb,sid,o) for o in (-1,1)],axis=1)
    variants = {
        'treb_only'      : treb,
        'bass_only'      : bass,
        'bass+treb'      : np.concatenate([bass,treb],1),
        'bass+treb+ctx1' : np.concatenate([bass,treb,ctx_bass,ctx_treb],1),
    }
    results={}; best=None
    for name,X in variants.items():
        m=train_clf(X[tr],roots[tr],X[va],roots[va],X.shape[1],12,hid=(128,64),epochs=50)
        pr=predict_proba(m,X[te]).argmax(1)
        acc=(pr==roots[te]).mean(); e,p45=p45_share(pr,roots[te])
        results[name]=dict(acc=float(acc),p45=float(p45))
        print(f"  {name:16s} root_acc={acc:.3f}  P4/P5_share={p45:.3f}")
        if best is None or acc>best[1]:
            best=(name,acc,m,X)
    # save best root head + full-corpus root posteriors (for quality context)
    name,acc,m,X = best
    full_proba = predict_proba(m,X)
    np.savez(ROOT/"scratchpad/root_posteriors.npz", proba=full_proba, feat_variant=name)
    torch.save({'state':m.state_dict(),'variant':name,'din':X.shape[1]},
               ROOT/"data/models/root_head_multihead_v1.pt")
    print(f"  -> best={name} acc={acc:.3f}; saved root_head_multihead_v1.pt")
    return results, full_proba, X, name

# ================= QUALITY HEAD =================
def rotate_by_root(chroma, root):
    """root-relative: put `root` at index 0. rotated[k]=chroma[(root+k)%12]."""
    out=np.empty_like(chroma)
    for r in range(12):
        m=root==r
        if m.any(): out[m]=np.roll(chroma[m],-r,axis=1)
    return out

def run_quality(D, tr,va,te, root_proba=None):
    print("\n===== QUALITY HEAD =====")
    roots=D['roots']; quals=D['quals']; sid=D['sid']
    bass,treb=D['bass'],D['treb']
    K=5
    cnt=np.bincount(quals,minlength=K); cw=(cnt.sum()/(K*np.maximum(cnt,1))).astype(np.float32)

    # ---- ablation A: raw absolute chroma, no rotation, no context (baseline) ----
    Xabs=np.concatenate([bass,treb],1)
    # ---- oracle root-relative rotation ----
    br=rotate_by_root(bass,roots); tr_=rotate_by_root(treb,roots)
    Xrr=np.concatenate([br,tr_],1)
    # ---- learned trigram context: neighbor ROOT-posterior distributions ----
    # (P(root|context) as 12-dim, from root head). Rotate neighbor dists into the
    #  target's root frame so context is transposition-invariant.
    def ctx_feats(root_frame):
        feats=[]
        for o in (-3,-2,-1,1,2,3):
            nb=neighbor(root_proba,sid,o)      # 12-dim posterior of neighbor
            # rotate neighbor dist relative to target root_frame
            nbr=np.empty_like(nb)
            for r in range(12):
                mm=root_frame==r
                if mm.any(): nbr[mm]=np.roll(nb[mm],-r,axis=1)
            feats.append(nbr)
        return np.concatenate(feats,1)          # 72-dim

    configs={}
    configs['abs_noctx']       = (Xabs, quals)
    configs['rootrel_oracle']  = (Xrr, quals)
    if root_proba is not None:
        ctx_oracle=ctx_feats(roots)
        configs['rootrel+ctx_oracle'] = (np.concatenate([Xrr,ctx_oracle],1), quals)

    res={}
    for name,(X,y) in configs.items():
        for loss_name,foc in [('wce',False)]:
            m=train_clf(X[tr],y[tr],X[va],y[va],X.shape[1],K,hid=(128,64),epochs=60,cw=cw,focal=foc)
            pr=predict_proba(m,X[te]).argmax(1)
            rec=balanced_recall(pr,y[te],K)
            bal=np.nanmean(rec)
            res[name]=dict(bal=float(bal),rec=[float(x) for x in rec])
            print(f"  {name:22s} bal_acc={bal:.3f}  rec[maj/min/dom/hdim/dim]={np.round(rec,2)}  dom={rec[2]:.3f}")

    # ---- CASCADE + MARGINALIZATION over root uncertainty ----
    if root_proba is not None:
        print("  --- cascade (predicted root) variants ---")
        pred_root=root_proba.argmax(1)
        # hard cascade: rotate by argmax predicted root
        Xrr_hard=np.concatenate([rotate_by_root(bass,pred_root),rotate_by_root(treb,pred_root),
                                 ctx_feats(pred_root)],1)
        m=train_clf(Xrr_hard[tr],quals[tr],Xrr_hard[va],quals[va],Xrr_hard.shape[1],K,
                    hid=(128,64),epochs=60,cw=cw)
        pr=predict_proba(m,Xrr_hard[te]).argmax(1); rec=balanced_recall(pr,quals[te],K)
        print(f"  {'cascade_hard_argmax':22s} bal_acc={np.nanmean(rec):.3f}  dom={rec[2]:.3f} rec={np.round(rec,2)}")
        res['cascade_hard']=dict(bal=float(np.nanmean(rec)),rec=[float(x) for x in rec])

        # MARGINALIZE: train the oracle-frame model, at test sum quality posterior
        # over top-k root hypotheses weighted by root posterior.
        # Use configs['rootrel+ctx_oracle'] model.
        Xmodel,_=configs['rootrel+ctx_oracle']
        mm=train_clf(Xmodel[tr],quals[tr],Xmodel[va],quals[va],Xmodel.shape[1],K,
                     hid=(128,64),epochs=60,cw=cw)
        # marginalize on test set
        topk=3
        order=np.argsort(-root_proba[te],1)[:,:topk]
        w=np.take_along_axis(root_proba[te],order,1); w=w/w.sum(1,keepdims=True)
        qpost=np.zeros((te.sum(),K))
        bass_te,treb_te=bass[te],treb[te]
        rp_te=root_proba  # need ctx per hypothesis
        for j in range(topk):
            rh=order[:,j]
            br_h=rotate_by_root(bass_te,rh); tr_h=rotate_by_root(treb_te,rh)
            # ctx rotated by hypothesis root — recompute ctx just for test rows
            ctxh=[]
            for o in (-3,-2,-1,1,2,3):
                nb=neighbor(root_proba,sid,o)[te]
                nbr=np.empty_like(nb)
                for r in range(12):
                    msk=rh==r
                    if msk.any(): nbr[msk]=np.roll(nb[msk],-r,axis=1)
                ctxh.append(nbr)
            Xh=np.concatenate([br_h,tr_h,np.concatenate(ctxh,1)],1)
            qpost += w[:,[j]]*predict_proba(mm,Xh)
        prm=qpost.argmax(1); recm=balanced_recall(prm,quals[te],K)
        print(f"  {'cascade_MARGINALIZED':22s} bal_acc={np.nanmean(recm):.3f}  dom={recm[2]:.3f} rec={np.round(recm,2)}")
        res['cascade_marginalized']=dict(bal=float(np.nanmean(recm)),rec=[float(x) for x in recm])
        torch.save({'state':mm.state_dict(),'din':Xmodel.shape[1]},
                   ROOT/"data/models/quality_head_trigram_v1.pt")
    return res

# ================= 7TH HEAD =================
def run_seventh(D, tr,va,te):
    print("\n===== 7TH HEAD (factored: base3 + has7th) =====")
    quals=D['quals']; roots=D['roots']; bass,treb=D['bass'],D['treb']
    # map 5-way -> base3 (maj/min/other) + has7th
    #   maj->base maj,no7 ; min->base min,no7 ; dom->base maj,7 ; hdim->base min,7(ish) ; dim->other
    base=np.array([0,1,0,1,2])[quals]   # 0 maj-fam,1 min-fam,2 other(dim)
    has7=np.array([0,0,1,1,0])[quals]    # dom & hdim carry a 7th
    Xrr=np.concatenate([rotate_by_root(bass,roots),rotate_by_root(treb,roots)],1)
    # base3
    cb=np.bincount(base,minlength=3); cwb=(cb.sum()/(3*np.maximum(cb,1))).astype(np.float32)
    mb=train_clf(Xrr[tr],base[tr],Xrr[va],base[va],Xrr.shape[1],3,cw=cwb,epochs=50)
    pb=predict_proba(mb,Xrr[te]).argmax(1); recb=balanced_recall(pb,base[te],3)
    print(f"  base3 bal={np.nanmean(recb):.3f} rec[maj/min/oth]={np.round(recb,2)}")
    # has7 binary
    c7=np.bincount(has7,minlength=2); cw7=(c7.sum()/(2*np.maximum(c7,1))).astype(np.float32)
    m7=train_clf(Xrr[tr],has7[tr],Xrr[va],has7[va],Xrr.shape[1],2,cw=cw7,epochs=50)
    p7=predict_proba(m7,Xrr[te]).argmax(1); rec7=balanced_recall(p7,has7[te],2)
    print(f"  has7  bal={np.nanmean(rec7):.3f} rec[no7/7]={np.round(rec7,2)}")
    # reassemble to 5-way vs flat comparison: dom = base maj & has7
    # compare dom recall of factored vs flat
    dom_true=quals[te]==2
    dom_fac=(pb==0)&(p7==1)
    dom_rec_fac=(dom_fac[dom_true]).mean()
    print(f"  factored dom recall (base=maj & has7)={dom_rec_fac:.3f}")
    torch.save({'base':mb.state_dict(),'has7':m7.state_dict(),'din':Xrr.shape[1]},
               ROOT/"data/models/seventh_head_v1.pt")
    return dict(base3=float(np.nanmean(recb)),has7=float(np.nanmean(rec7)),dom_fac=float(dom_rec_fac))

if __name__=="__main__":
    stage=sys.argv[1] if len(sys.argv)>1 else "all"
    D=load()
    tr,va,te=song_split(D['sid'])
    print(f"split: train={tr.sum()} val={va.sum()} test={te.sum()} | quals dist {np.bincount(D['quals'])}")
    out={}
    if stage in ("root","all"):
        rr, full_proba, Xbest, vname = run_root(D,tr,va,te)
        out['root']=rr
    else:
        full_proba=np.load(ROOT/"scratchpad/root_posteriors.npz")['proba']
    if stage in ("quality","all"):
        out['quality']=run_quality(D,tr,va,te,root_proba=full_proba)
    if stage in ("seventh","all"):
        out['seventh']=run_seventh(D,tr,va,te)
    json.dump(out, open(ROOT/"scratchpad/multihead_results.json","w"), indent=2)
    print("\nsaved scratchpad/multihead_results.json")
