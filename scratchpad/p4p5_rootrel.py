"""P4/P5 root disambiguation: EXPLICIT root-relative-normalized views for BOTH
candidates (user refinement 2026-07-16). Extends scratchpad/p4p5_rwc.py.

Prior rawfull(96d) ALREADY concatenated [rollA, rollB] (two rotated views) and
scored ~0.58 pooled CV. This script adds the genuinely-untested constructions
the user asked for:
  (concatAB)  reproduce the two-rotated-views baseline (=prior rawfull)
  (diffAB)    View A - View B, 48d elementwise difference
  (norm_concat) each rotated view divided by its own index-0 root energy, concat
  (tmplscore) score each rotated view against canonical maj/min/dom triads AT
              index 0 (root-position shape), per block; features = both views'
              scores + differences/ratios
  (tmpl+diff) tmplscore concatenated with diffAB
All on clean RWC; same binary "pick the true root from 2 candidates" task,
symmetric (chance=0.5), GroupKFold by song, plus a dedicated held-out song split.
"""
import numpy as np, torch, json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
rng = np.random.default_rng(0)

d = np.load(REPO/"data/cache/rwc/rwc_bp48.npz", allow_pickle=True)      # READ ONLY
ck = torch.load(REPO/"data/models/_eval_only_rwc_bp48_boundary_check.pt",
                map_location="cpu", weights_only=False)
rm = ck["root_model"]; rm.eval()
feat = d["feat48_abs"].astype(np.float32)
Xn = ((feat - ck["root_mean"]) / ck["root_std"]).astype(np.float32)
with torch.no_grad():
    pred = rm(torch.tensor(Xn)).argmax(1).numpy()
gt   = d["root"].astype(int)
song = d["song_id"].astype(str)

err = (pred != gt)
iv  = (pred - gt) % 12
p4p5_err = err & np.isin(iv, [5, 7])
correct = (pred == gt)
nctrl = p4p5_err.sum()
ci = np.where(correct)[0]
ctrl = rng.choice(ci, size=min(nctrl, len(ci)), replace=False)

cases = []
for i in np.where(p4p5_err)[0]: cases.append((i, gt[i], pred[i], True))
for i in ctrl:
    dist = (gt[i] + (7 if rng.random() < 0.5 else 5)) % 12
    cases.append((i, gt[i], dist, False))
N = len(cases)
print(f"N cases = {N}  (P4/P5 errors {int(p4p5_err.sum())} + controls {len(ctrl)}), chance=0.500")

def roll_view(f, r):
    out = np.empty(48, np.float32)
    for oo in (0,12,24,36): out[oo:oo+12] = np.roll(f[oo:oo+12], -r)
    return out
def blkv(v, name):  # accessor into a 48-view
    o = {"onset":0,"note":12,"bass":24,"treble":36}[name]; return v[o:o+12]

# canonical root-position templates at index 0
T_MAJ = np.zeros(12); T_MAJ[[0,4,7]] = 1
T_MIN = np.zeros(12); T_MIN[[0,3,7]] = 1
T_DOM = np.zeros(12); T_DOM[[0,4,7,10]] = 1
def tmpl_feats(v):
    # for note & bass block: dot with maj/min/dom, plus root(0) and fifth(7) energy
    out = []
    for name in ("note","bass","treble"):
        b = blkv(v, name); nb = b/(np.linalg.norm(b)+1e-9)
        out += [b@T_MAJ, b@T_MIN, b@T_DOM,           # dot
                nb@T_MAJ, nb@T_MIN, nb@T_DOM,         # cosine-ish
                b[0], b[7], b[4], b[3]]               # root/fifth/M3/m3 energy
    return np.array(out, np.float32)

labels = np.zeros(N, dtype=int)
grp = song[np.array([c[0] for c in cases])]
is_err = np.array([c[3] for c in cases])
F_concat, F_diff, F_norm, F_tmpl, F_tmpldiff = [], [], [], [], []
for k,(i, tr, dr, ise) in enumerate(cases):
    f = feat[i]
    swap = rng.random() < 0.5
    a, b = (dr, tr) if swap else (tr, dr)
    labels[k] = 1 if swap else 0
    vA, vB = roll_view(f, a), roll_view(f, b)
    F_concat.append(np.concatenate([vA, vB]))
    F_diff.append(vA - vB)
    # normalize each view by its own note-block root(0) energy ("first root note")
    nA = vA / (blkv(vA,"note")[0] + 1e-6); nB = vB / (blkv(vB,"note")[0] + 1e-6)
    F_norm.append(np.concatenate([nA, nB]))
    tA, tB = tmpl_feats(vA), tmpl_feats(vB)
    F_tmpl.append(np.concatenate([tA, tB, tA - tB]))
    F_tmpldiff.append(np.concatenate([tA, tB, tA - tB, vA - vB]))
F_concat=np.array(F_concat);F_diff=np.array(F_diff);F_norm=np.array(F_norm)
F_tmpl=np.array(F_tmpl);F_tmpldiff=np.array(F_tmpldiff)
y = labels

def lr(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, C=1.0))
def mlp(): return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,16),max_iter=1200,alpha=1e-2,random_state=0))

def cv_eval(X, y, groups, model_fn, err_mask):
    gkf = GroupKFold(n_splits=5); oof = np.zeros(len(y))
    for tr_i, te_i in gkf.split(X, y, groups):
        m = model_fn(); m.fit(X[tr_i], y[tr_i]); oof[te_i] = m.predict(X[te_i])
    return (oof==y).mean(), (oof[err_mask]==y[err_mask]).mean(), (oof[~err_mask]==y[~err_mask]).mean()

sets = {"concatAB(96d)":F_concat,"diffAB(48d)":F_diff,"norm_concat(96d)":F_norm,
        "tmplscore(90d)":F_tmpl,"tmpl+diff(138d)":F_tmpldiff}
print(f"\n=== Pooled GroupKFold CV (chance=0.500; prior rawfull=0.58) ===")
print(f"{'featureset':17s} {'model':4s} | pooled | on-err | on-ctrl")
for name, X in sets.items():
    for mname, fn in [("LR",lr),("MLP",mlp)]:
        a,ae,ac = cv_eval(X, y, grp, fn, is_err)
        print(f"{name:17s} {mname:4s} |  {a:.3f} | {ae:.3f}  | {ac:.3f}")

# ---- dedicated held-out song split (task point 4): train 70% songs, test 30% ----
print(f"\n=== Dedicated held-out song split (70/30 by song, 5 seeds) ===")
print(f"{'featureset':17s} {'model':4s} | mean test acc (+/-std)")
for name, X in sets.items():
    for mname, fn in [("LR",lr),("MLP",mlp)]:
        accs = []
        for s in range(5):
            gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=s)
            tr_i, te_i = next(gss.split(X, y, grp))
            m = fn(); m.fit(X[tr_i], y[tr_i]); accs.append((m.predict(X[te_i])==y[te_i]).mean())
        accs = np.array(accs)
        print(f"{name:17s} {mname:4s} | {accs.mean():.3f} +/- {accs.std():.3f}")
