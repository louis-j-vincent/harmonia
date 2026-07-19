"""Cascade analysis: our model's root/quality accuracy on the maj/min majority
vs the non-maj/min hard residual. Lean 2-seed song-strat CV, CPU."""
import sys, numpy as np; sys.path.insert(0,'.'); sys.path.insert(0,'scripts')
from harmonia.data.corpus_schema import MatchQuality, filter_by_match, load_corpus
from train_real_audio_final import _train_head, QUALITIES
d=load_corpus('data/cache/rwc/rwc_bp48.npz')
keep=filter_by_match(d['match'], minimum=MatchQuality.EXACT)
feat=d['feat48_abs'][keep]; featrel=d['feat48'][keep]
root=d['root'].astype(int)[keep]; qi=d['quality_idx'].astype(int)[keep]
song=d['song_id'][keep]; q=d['quality'][keep]
majmin = np.isin(q, ['maj','min'])   # family membership per chord
songs=sorted(set(song.tolist()))
def _lg(X,m,mu,sd):
    import torch
    with torch.no_grad(): return m(torch.tensor(((X-mu)/sd).astype(np.float32))).argmax(1).cpu().numpy()
RP=[]; QP=[]; MM=[]
for seed in range(2):
    rng=np.random.RandomState(seed); sh=list(songs); rng.shuffle(sh)
    test=set(sh[:max(1,round(0.2*len(sh)))])
    tr=np.array([s not in test for s in song]); te=~tr
    rm,rmu,rsd=_train_head(feat[tr],root[tr],12,epochs=30,lr=3e-4,batch=512,device='cpu',head_name='root')
    qm,qmu,qsd=_train_head(featrel[tr],qi[tr],7,epochs=30,lr=3e-4,batch=512,device='cpu',head_name='qual')
    rp=_lg(feat[te],rm,rmu,rsd); qp=_lg(featrel[te],qm,qmu,qsd)
    RP.append((rp==root[te])); QP.append((qp==qi[te])); MM.append(majmin[te])
    # also majmin-collapsed quality correctness (both maj-fam or both min-fam etc)
r=np.concatenate(RP); qok=np.concatenate(QP); mm=np.concatenate(MM)
print(f"pooled test n={len(r)}  maj/min-family {mm.mean():.3f}")
for lbl,mask in [("maj/min subset",mm),("residual (non maj/min)",~mm)]:
    print(f"\n{lbl}: n={mask.sum()}")
    print(f"   ROOT acc:            {r[mask].mean():.3f}")
    print(f"   QUALITY(7-way) acc:  {qok[mask].mean():.3f}")
    print(f"   JOINT root&quality:  {(r&qok)[mask].mean():.3f}")
