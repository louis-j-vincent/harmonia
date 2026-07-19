"""Multi-seed comparison: unweighted vs class-weighted root head training,
same protocol as train_nnls24_heads.py (song-level held-out split), 5 seeds
(matching the project's own "RWC 5-seed CV" convention).

Screens the premise from docs/adversarial_review_2026_07_17.md's "Bias check"
before any retrain touches the shipped nnls24_heads.npz.
"""
import numpy as np
from multihead_training import train_clf, predict_proba

NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
SEEDS = [0, 1, 2, 3, 4]

d = np.load("data/cache/rwc/rwc_nnls24.npz", allow_pickle=True)
nn24 = d["nnls24"].astype(np.float32)
roots = d["root"].astype(np.int64) % 12
sid = d["song_id"]


def song_split(seed, test_frac=0.2):
    songs = np.unique(sid)
    rng = np.random.RandomState(seed)
    rng.shuffle(songs)
    nte = max(1, int(round(test_frac * len(songs))))
    te = np.isin(sid, songs[:nte])
    tr_all = ~te
    rng2 = np.random.RandomState(seed + 100)
    tr_idx = np.where(tr_all)[0]
    va_pick = rng2.choice(tr_idx, size=max(1, len(tr_idx) // 8), replace=False)
    va = np.zeros(len(nn24), bool)
    va[va_pick] = True
    tr = tr_all & ~va
    return tr, va, te


def run(seed, cw):
    tr, va, te = song_split(seed)
    rm = train_clf(nn24[tr], roots[tr], nn24[va], roots[va], 24, 12,
                    hid=(128, 64), epochs=50, cw=cw)
    proba = predict_proba(rm, nn24)
    pred = proba.argmax(1)
    r_te, p_te = roots[te], pred[te]
    acc = (r_te == p_te).mean()
    per_class = np.full(12, np.nan)
    for pc in range(12):
        m = r_te == pc
        if m.sum():
            per_class[pc] = (p_te[m] == pc).mean()
    return acc, per_class


accs_uw, accs_cw = [], []
pc_uw, pc_cw = [], []
for seed in SEEDS:
    tr, va, te = song_split(seed)
    cnt = np.bincount(roots[tr], minlength=12)
    cw = (cnt.sum() / (12 * np.maximum(cnt, 1))).astype(np.float32)

    a_uw, p_uw = run(seed, None)
    a_cw, p_cw = run(seed, cw)
    accs_uw.append(a_uw); accs_cw.append(a_cw)
    pc_uw.append(p_uw); pc_cw.append(p_cw)
    print(f"seed={seed}  unweighted={a_uw*100:.2f}%  weighted={a_cw*100:.2f}%  "
          f"delta={100*(a_cw-a_uw):+.2f}pp")

pc_uw = np.array(pc_uw)  # (5, 12)
pc_cw = np.array(pc_cw)

print()
print(f"MEAN aggregate acc: unweighted={100*np.mean(accs_uw):.2f}% "
      f"(+/-{100*np.std(accs_uw):.2f})  "
      f"weighted={100*np.mean(accs_cw):.2f}% (+/-{100*np.std(accs_cw):.2f})")
print()
print(f"{'note':4s} {'uw_mean':>8s} {'cw_mean':>8s} {'delta':>7s} {'uw_std':>7s} {'cw_std':>7s}")
for pc in range(12):
    uw_m, cw_m = np.nanmean(pc_uw[:, pc]), np.nanmean(pc_cw[:, pc])
    uw_s, cw_s = np.nanstd(pc_uw[:, pc]), np.nanstd(pc_cw[:, pc])
    print(f"{NOTE[pc]:4s} {uw_m*100:8.1f} {cw_m*100:8.1f} {100*(cw_m-uw_m):7.1f} "
          f"{uw_s*100:7.1f} {cw_s*100:7.1f}")

rare = [1, 6, 8, 3, 11]  # C#, F#, G#, D#, B (the 5 rarest by training freq)
rare_uw = np.nanmean(pc_uw[:, rare])
rare_cw = np.nanmean(pc_cw[:, rare])
print()
print(f"5 rarest classes mean recall: unweighted={rare_uw*100:.1f}%  "
      f"weighted={rare_cw*100:.1f}%  delta={100*(rare_cw-rare_uw):+.1f}pp")
