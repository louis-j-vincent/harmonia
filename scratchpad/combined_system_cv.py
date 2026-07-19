"""COMBINED SYSTEM end-to-end eval on RWC: NNLS-24 (root+quality) + pYIN (bass/
inversion), vs every individual baseline. Reuses the exact NNLS full-recipe harness
(scripts/rwc_nnls_multihead_cv.py functions) so root/quality match the 0.789/0.693
baseline. pYIN bass from scratchpad/pyin_bass_cache.npz (global-row-aligned).

Deployable cascade setting (predicted root, not oracle). Multi-seed song-grouped CV;
combined bass/inversion/full-chord metrics computed on the pYIN-covered TEST chords,
pooled across seeds. Root/quality heads train on ALL 100 songs.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO/"scratchpad")); sys.path.insert(0, str(REPO/"scripts"))
from harmonia.data.corpus_schema import load_corpus
from multihead_training import train_clf, predict_proba, rotate_by_root, balanced_recall
from rwc_nnls_multihead_cv import song_split, ctx_feats
QUAL = ['maj','min','dom','hdim','dim','aug','sus']; KQ=7
BS={"b2":1,"2":2,"b3":3,"3":4,"4":5,"b5":6,"5":7,"b6":8,"6":9,"b7":10,"7":11,"b9":1,"9":2}

def gt_bass(lab, root):
    lab=str(lab).strip()
    if "/" not in lab or lab in ("N","X",""): return root%12, 0
    b=lab.split("/",1)[1].strip()
    if b not in BS: return root%12, 0
    return (root+BS[b])%12, 1

def main():
    import argparse; ap=argparse.ArgumentParser(); ap.add_argument("--seeds",type=int,default=3); a=ap.parse_args()
    nn=load_corpus(REPO/"data/cache/rwc/rwc_nnls24.npz"); bp=load_corpus(REPO/"data/cache/rwc/rwc_bp48_fixed.npz")
    py=np.load(REPO/"scratchpad/pyin_bass_cache.npz",allow_pickle=True)
    nn24=nn["nnls24"].astype(np.float32); filled=np.abs(nn24).sum(1)>0
    keep=filled
    nn24=nn24[keep]
    f48a=bp["feat48_abs"].astype(np.float32)[keep]; f48rr=bp["feat48"].astype(np.float32)[keep]
    roots=bp["root"].astype(np.int64)[keep]%12; quals=bp["quality_idx"].astype(np.int64)[keep]
    sid=bp["song_id"][keep]; labels=bp["labels"][keep]
    # GT sounding-bass + inversion
    gb=np.zeros(len(roots),int); ginv=np.zeros(len(roots),int)
    for i in range(len(roots)):
        gb[i],ginv[i]=gt_bass(labels[i],roots[i])
    # pYIN bass (global-row aligned) restricted to keep
    py_bass=py["bass_pc"][keep]; py_conf=py["conf"][keep]; py_rel=py["reliable"][keep]
    bp48_bass=f48a[:,24:36].argmax(1)     # BP48 bass-block argmax baseline
    nn_bass_arg=nn24[:,:12].argmax(1)     # NNLS bass-half argmax baseline
    has_py=py_bass>=0
    cached_songs=set(py["songs_done"].tolist())
    print(f"rows={len(roots)}, songs={len(np.unique(sid))}, pYIN-cached songs={len(cached_songs)}, "
          f"chords with pYIN bass={has_py.sum()}",flush=True)

    # pooled test collectors (deployable/predicted-root)
    P={k:[] for k in ["gb","ginv","root_ok","qual_ok","py_bass","bp48_bass","nn_bass",
                      "pred_root","py_rel","py_conf"]}
    for seed in range(a.seeds):
        tr,va,te=song_split(sid,seed)
        cnt=np.bincount(quals,minlength=KQ); cw=(cnt.sum()/(KQ*np.maximum(cnt,1))).astype(np.float32)
        # root head (NNLS)
        rm=train_clf(nn24[tr],roots[tr],nn24[va],roots[va],nn24.shape[1],12,hid=(128,64),epochs=50)
        proba=predict_proba(rm,nn24); pred_root=proba.argmax(1)
        # quality head (NNLS cascade: rotate by predicted root + trigram ctx)
        nb,ntr=nn24[:,:12],nn24[:,12:]
        Xc=np.concatenate([rotate_by_root(nb,pred_root),rotate_by_root(ntr,pred_root),
                           ctx_feats(proba,sid,pred_root)],1)
        qm=train_clf(Xc[tr],quals[tr],Xc[va],quals[va],Xc.shape[1],KQ,hid=(128,64),epochs=60,cw=cw)
        pred_q=predict_proba(qm,Xc).argmax(1)
        # collect TEST chords that have pYIN bass
        m=te&has_py
        P["gb"]+=gb[m].tolist(); P["ginv"]+=ginv[m].tolist()
        P["root_ok"]+=(pred_root[m]==roots[m]).tolist(); P["qual_ok"]+=(pred_q[m]==quals[m]).tolist()
        P["py_bass"]+=py_bass[m].tolist(); P["bp48_bass"]+=bp48_bass[m].tolist(); P["nn_bass"]+=nn_bass_arg[m].tolist()
        P["pred_root"]+=pred_root[m].tolist(); P["py_rel"]+=py_rel[m].tolist(); P["py_conf"]+=py_conf[m].tolist()
        print(f"[seed {seed}] test∩pYIN n={m.sum()}  root={np.mean((pred_root[m]==roots[m])):.3f} "
              f"qual_acc={np.mean((pred_q[m]==quals[m])):.3f}",flush=True)

    A={k:np.array(v) for k,v in P.items()}
    n=len(A["gb"]); inv=A["ginv"]==1
    print(f"\n{'='*70}\nCOMBINED SYSTEM — deployable (predicted root), pooled {n} pYIN-covered test chords")
    print(f"  ({inv.sum()} inversions, {(~inv).sum()} root-position)\n")
    def acc(pred,gt,mask=None):
        mask=np.ones(n,bool) if mask is None else mask
        return (pred[mask]==gt[mask]).mean() if mask.sum() else float('nan')
    # BASS
    print("BASS (sounding-bass pc) — pYIN vs baselines:")
    for lbl,pred in [("pYIN",A["py_bass"]),("BP48 bass-argmax",A["bp48_bass"]),("NNLS bass-argmax",A["nn_bass"])]:
        print(f"   {lbl:18s} all={acc(pred,A['gb']):.3f}  inv={acc(pred,A['gb'],inv):.3f}  rootpos={acc(pred,A['gb'],~inv):.3f}")
    # ROOT / QUALITY (NNLS)
    print(f"\nROOT (NNLS-24, predicted):  {A['root_ok'].mean():.3f}")
    print(f"QUALITY (NNLS-24, cascade): acc={A['qual_ok'].mean():.3f}")
    # ENSEMBLE bass: NNLS-bass primary; when pYIN agrees -> high confidence
    ens_bass=A["nn_bass"].copy()  # NNLS-bass is the strongest single estimator
    agree=A["py_bass"]==A["nn_bass"]
    print(f"\nENSEMBLE bass (NNLS-bass primary): all={acc(ens_bass,A['gb']):.3f}  inv={acc(ens_bass,A['gb'],inv):.3f}")
    print(f"   pYIN&NNLS agree on {agree.mean():.3f} of chords; where they AGREE bass-acc={acc(A['nn_bass'],A['gb'],agree):.3f}; where they DISAGREE={acc(A['nn_bass'],A['gb'],~agree):.3f}")
    # INVERSION DETECTION
    print("\nINVERSION detection (inversions are {:.1%} of chords):".format(inv.mean()))
    def pr(pinv,label):
        tp=((pinv==1)&inv).sum(); fp=((pinv==1)&~inv).sum(); fn=((pinv==0)&inv).sum()
        print(f"   {label:42s} precision={tp/max(tp+fp,1):.3f} recall={tp/max(tp+fn,1):.3f}")
    pr((A["nn_bass"]!=A["pred_root"]).astype(int),"NNLS-bass != root")
    pr((A["py_bass"]!=A["pred_root"]).astype(int),"pYIN-bass != root")
    pr((A["bp48_bass"]!=A["pred_root"]).astype(int),"BP48-bass != root (baseline ~0.20)")
    # ENSEMBLE inversion: both estimators disagree with root AND agree with each other
    ens_pinv=((A["nn_bass"]!=A["pred_root"])&(A["py_bass"]!=A["pred_root"])&(A["nn_bass"]==A["py_bass"])).astype(int)
    pr(ens_pinv,"ENSEMBLE (NNLS&pYIN agree, both != root)")
    # END-TO-END full-chord accuracy: root & quality & bass all correct
    print("\nEND-TO-END full-chord (root & quality & sounding-bass all correct):")
    rq=(A["root_ok"]==1)&(A["qual_ok"]==1)
    print(f"   root&quality only (no bass): {rq.mean():.3f}")
    print(f"   + NNLS-bass  (BEST):         {(rq&(A['nn_bass']==A['gb'])).mean():.3f}")
    print(f"   + pYIN-bass:                 {(rq&(A['py_bass']==A['gb'])).mean():.3f}")
    print(f"   + ensemble-bass:             {(rq&(ens_bass==A['gb'])).mean():.3f}")
    print(f"   + BP48-bass (baseline):      {(rq&(A['bp48_bass']==A['gb'])).mean():.3f}")
    # fallback analysis: unreliable pYIN spans
    unrel=A["py_rel"]==0
    print(f"\npYIN fallback (unreliable spans, vfrac<0.30): {unrel.mean():.3f} of chords")
    if unrel.sum(): print(f"   bass-acc on reliable={acc(A['py_bass'],A['gb'],~unrel):.3f}  unreliable={acc(A['py_bass'],A['gb'],unrel):.3f}")

if __name__=="__main__": main()
