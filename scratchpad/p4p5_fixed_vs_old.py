"""Matched P4/P5 root-error diagnostic: OLD (bleeding) vs FIXED (frame-clip) RWC.

Runs the IDENTICAL procedure on both corpora so the only difference is the
feature pooling:
  - same song-stratified split (train_jaah_cv seed 0, test_frac 0.2)
  - train root head on train (roll augment), predict on held-out test
  - report held-out root acc, P4/P5 share of errors, and the contamination
    probe: does the TRUE root's third out-energize the WRONG root's third in
    the held-out P4/P5-error features? (chance 0.5; >0.5 = true root actually
    present, i.e. the error was contamination-driven, now recoverable)
"""
import sys
from pathlib import Path
import numpy as np

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))
from harmonia.data.corpus_schema import load_corpus
from train_real_audio_final import _train_head, _augment_root_by_roll

THIRD = {"maj":4,"dom":4,"aug":4,"sus":5,"min":3,"hdim":3,"dim":3}
def blk(f,name): o={"onset":0,"note":12,"bass":24,"treble":36}[name]; return f[o:o+12]
def energy(f,name,pc): return blk(f,name)[pc%12]

def run(corpus_path, seed=0, test_frac=0.2, device="cpu"):
    d = load_corpus(corpus_path)
    feat = d["feat48_abs"].astype(np.float32)
    roots = d["root"].astype(int); song = d["song_id"].astype(str)
    q = d["quality"].astype(str)
    songs = sorted(set(song.tolist()))
    rng = np.random.RandomState(seed); rng.shuffle(songs)
    n_test = max(1, int(round(test_frac*len(songs))))
    test = set(songs[:n_test])
    tr = np.array([s not in test for s in song]); te = ~tr
    Xtr, ytr = feat[tr], roots[tr]
    Xtr, ytr = _augment_root_by_roll(Xtr, ytr)
    rm, rmean, rstd = _train_head(Xtr, ytr, 12, epochs=60, lr=3e-4, batch=64,
                                  device=device, head_name="root")
    import torch
    with torch.no_grad():
        Xte = ((feat[te]-rmean)/rstd).astype(np.float32)
        pred = rm(torch.tensor(Xte)).argmax(1).numpy()
    gt = roots[te]; qte = q[te]; fte = feat[te]
    err = pred != gt
    iv = (pred-gt)%12; is_p4p5 = np.isin(iv,[5,7])
    acc = (pred==gt).mean()
    n_err=err.sum(); n_p4p5=(err&is_p4p5).sum()
    # contamination probe on held-out P4/P5 errors
    idx = np.where(err&is_p4p5)[0]; wins=0
    for i in idx:
        f=fte[i]; t3=THIRD.get(qte[i],4)
        et=energy(f,"note",gt[i]+t3); ew=energy(f,"note",pred[i]+t3)
        if et>ew: wins+=1
    probe = wins/max(len(idx),1)
    return dict(n_test=int(te.sum()), acc=acc, n_err=int(n_err),
                n_p4p5=int(n_p4p5), p4p5_share=n_p4p5/max(n_err,1),
                probe=probe, n_probe=len(idx))

if __name__ == "__main__":
    for tag, name in [("OLD  ", "rwc_bp48.npz"), ("FIXED", "rwc_bp48_fixed.npz")]:
        p = REPO/"data/cache/rwc"/name
        if not p.exists():
            print(f"{tag}: {name} missing, skip"); continue
        r = run(p)
        print(f"{tag} | root_acc={r['acc']:.3f} | held-out n={r['n_test']} "
              f"err={r['n_err']} p4p5={r['n_p4p5']} "
              f"p4p5_share={r['p4p5_share']:.3f} | "
              f"3rd-probe(true>wrong)={r['probe']:.3f} (n={r['n_probe']}, chance .500)")
