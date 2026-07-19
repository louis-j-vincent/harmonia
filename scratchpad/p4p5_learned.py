"""P4/P5 disambiguation: can a LEARNED chroma classifier pick the true root
from the {true, confusable-fifth} candidate pair, where the hand-built
template failed (49.8%, chance)?

Binary task: given two candidate roots a fifth/fourth apart (exactly one is
the GT root), pick the true one. Symmetric framing (candidate order randomized)
=> chance is 50% by construction. No leak of "which one the model predicted".

Feature sets:
  third   : maj3/min3 energies of both candidates (note chroma)
  fifth   : fifth (r+7) energies of both candidates -- the diagnostic one is
            the WRONG root's own fifth (F# for a B-vs-E confusion), absent if
            the true chord is E. Both-candidate framing captures it naturally.
  combo   : root + maj3 + min3 + fifth of both candidates, note+bass blocks
  rawfull : full 48-dim chroma rolled into EACH candidate frame (96-dim),
            unconstrained learned model.

Groups = song_id (GroupKFold) to avoid within-song leakage.
"""
import numpy as np, torch
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
rng = np.random.default_rng(0)

d = np.load(REPO/"data/cache/billboard_bp48_60_fixed_beatgrid.npz", allow_pickle=True)
ck = torch.load(REPO/"data/models/billboard_bp48_60_rollaug_v1.pt", map_location="cpu", weights_only=False)
rm = ck["root_model"]; rm.eval()
feat = d["feat48_abs"].astype(np.float32)          # (N,48) blocks: onset,note,bass,treble
Xn = ((feat - ck["root_mean"]) / ck["root_std"]).astype(np.float32)
with torch.no_grad():
    pred = rm(torch.tensor(Xn)).argmax(1).numpy()
gt = d["root"].astype(int)
song = d["song_id"]
q = d["quality"]

err = pred != gt
iv = (pred - gt) % 12
p4p5_err = err & np.isin(iv, [5, 7])               # error cases: distractor = pred

# control: correctly-classified chords, distractor = a random P4/P5 neighbor
correct = pred == gt
nctrl = p4p5_err.sum()
ci = np.where(correct)[0]
ctrl = rng.choice(ci, size=nctrl, replace=False)

# assemble case list: (idx, true_root, distractor_root, is_error)
cases = []
for i in np.where(p4p5_err)[0]:
    cases.append((i, gt[i], pred[i], True))
for i in ctrl:
    dist = (gt[i] + (7 if rng.random() < 0.5 else 5)) % 12
    cases.append((i, gt[i], dist, False))

N = len(cases)
print(f"N cases = {N}  (P4/P5 errors {p4p5_err.sum()} + controls {nctrl})")

# block accessors
def blk(f, name):
    o = {"onset":0, "note":12, "bass":24, "treble":36}[name]
    return f[o:o+12]

def energy(f, name, pc):
    return blk(f, name)[pc % 12]

# build feature matrices; candidate order randomized -> label = position of true root
labels = np.zeros(N, dtype=int)
grp = np.array([c[0] for c in cases])
grp = song[grp]
is_err = np.array([c[3] for c in cases])

F_third, F_fifth, F_combo, F_raw = [], [], [], []
for (i, tr, dr, ise) in cases:
    f = feat[i]
    swap = rng.random() < 0.5           # randomize which slot holds true root
    a, b = (dr, tr) if swap else (tr, dr)
    lab = 1 if swap else 0              # position of TRUE root
    F_third.append([energy(f,"note",a+4), energy(f,"note",a+3),
                    energy(f,"note",b+4), energy(f,"note",b+3)])
    F_fifth.append([energy(f,"note",a+7), energy(f,"bass",a+7),
                    energy(f,"note",b+7), energy(f,"bass",b+7)])
    F_combo.append([
        energy(f,"note",a),   energy(f,"bass",a),
        energy(f,"note",a+4), energy(f,"note",a+3), energy(f,"note",a+7), energy(f,"bass",a+7),
        energy(f,"note",b),   energy(f,"bass",b),
        energy(f,"note",b+4), energy(f,"note",b+3), energy(f,"note",b+7), energy(f,"bass",b+7),
    ])
    # full 48 rolled into each candidate frame (roll each 12-block by -root)
    def roll_all(f, r):
        out = np.empty(48, np.float32)
        for oo in (0,12,24,36):
            out[oo:oo+12] = np.roll(f[oo:oo+12], -r)
        return out
    F_raw.append(np.concatenate([roll_all(f,a), roll_all(f,b)]))
    labels[len(F_third)-1] = lab

F_third = np.array(F_third); F_fifth=np.array(F_fifth)
F_combo = np.array(F_combo); F_raw=np.array(F_raw)
y = labels

def cv_eval(X, y, groups, model_fn, err_mask):
    gkf = GroupKFold(n_splits=5)
    oof = np.zeros(len(y));
    for tr_i, te_i in gkf.split(X, y, groups):
        m = model_fn()
        m.fit(X[tr_i], y[tr_i])
        oof[te_i] = m.predict(X[te_i])
    acc = (oof == y).mean()
    acc_err = (oof[err_mask] == y[err_mask]).mean()
    acc_ctrl = (oof[~err_mask] == y[~err_mask]).mean()
    return acc, acc_err, acc_ctrl

def lr(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
def mlp(): return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(16,),
             max_iter=800, alpha=1e-2, random_state=0))

sets = {"third(4d)":F_third, "fifth(4d)":F_fifth, "combo(12d)":F_combo, "rawfull(96d)":F_raw}
print(f"\n{'featureset':14s} {'model':4s} | held-out acc | on-errors | on-controls")
print("-"*62)
for name, X in sets.items():
    for mname, fn in [("LR",lr),("MLP",mlp)]:
        a,ae,ac = cv_eval(X, y, grp, fn, is_err)
        print(f"{name:14s} {mname:4s} |    {a:.3f}    |   {ae:.3f}   |   {ac:.3f}")
print("\nchance baseline = 0.500 (symmetric two-candidate task)")
print("prior hand-template third-presence probe on errors = 0.498")

# ---- ERRORS-ONLY CV: is there ANY learnable signal within the hard cases? ----
print("\n=== CV trained+tested ONLY on the P4/P5 error cases ===")
ei = np.where(is_err)[0]
ge = grp[ei]; ye = y[ei]
print(f"{'featureset':14s} {'model':4s} | held-out acc (errors only)")
print("-"*44)
for name, X in sets.items():
    Xe = X[ei]
    for mname, fn in [("LR",lr),("MLP",mlp)]:
        gkf = GroupKFold(n_splits=5)
        oof = np.zeros(len(ye))
        for tr_i, te_i in gkf.split(Xe, ye, ge):
            m = fn(); m.fit(Xe[tr_i], ye[tr_i]); oof[te_i]=m.predict(Xe[te_i])
        print(f"{name:14s} {mname:4s} |    {(oof==ye).mean():.3f}")
print("(>0.55 held-out = recoverable signal the model missed; ~0.50 = dead end)")
