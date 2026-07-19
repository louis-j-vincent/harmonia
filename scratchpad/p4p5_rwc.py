"""P4/P5 root-confusion diagnostic on RWC (clean, user-verified alignment).

Re-run of the Billboard P4/P5 "acoustic illusion" pipeline on RWC to test
whether that conclusion was partly a Billboard GT-misalignment artifact.

Reuses:
  - RWC root model from data/models/_eval_only_rwc_bp48_boundary_check.pt
    (read-only; produced by parallel boundary-check agent; root_acc 0.630 held-out)
  - feature block layout identical to billboard: feat48_abs = [onset,note,bass,treble]
  - methodology of scratchpad/p4p5_learned.py + the template_screen third-presence probe
"""
import numpy as np, torch, json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
rng = np.random.default_rng(0)

d = np.load(REPO/"data/cache/rwc/rwc_bp48.npz", allow_pickle=True)
ck = torch.load(REPO/"data/models/_eval_only_rwc_bp48_boundary_check.pt",
                map_location="cpu", weights_only=False)
split = json.load(open(REPO/"data/models/_eval_only_rwc_boundary_check_split.json"))
val_songs = set(split["val_songs"])

rm = ck["root_model"]; rm.eval()
feat = d["feat48_abs"].astype(np.float32)
Xn = ((feat - ck["root_mean"]) / ck["root_std"]).astype(np.float32)
with torch.no_grad():
    pred = rm(torch.tensor(Xn)).argmax(1).numpy()
gt   = d["root"].astype(int)
song = d["song_id"].astype(str)
q    = d["quality"].astype(str)

heldout = np.array([s in val_songs for s in song])

# ---------- 1. Held-out error composition ----------
err = (pred != gt)
iv  = (pred - gt) % 12
is_p4p5 = np.isin(iv, [5, 7])

ho = heldout
n_ho = ho.sum()
acc_ho = (pred[ho] == gt[ho]).mean()
n_err_ho = (err & ho).sum()
n_p4p5_ho = (err & ho & is_p4p5).sum()
print(f"held-out val songs: {len(val_songs)}, records: {n_ho}")
print(f"held-out root acc: {acc_ho:.3f}   (checkpoint reported 0.630)")
print(f"held-out errors: {n_err_ho}   P4/P5 errors: {n_p4p5_ho}"
      f"   P4/P5 share of errors: {n_p4p5_ho/max(n_err_ho,1):.3f}")

# full-corpus figure too (for the learned classifier N)
n_err_all = err.sum(); n_p4p5_all = (err & is_p4p5).sum()
print(f"full-corpus P4/P5 share of errors: {n_p4p5_all/n_err_all:.3f}"
      f"  (n_err={n_err_all}, n_p4p5={n_p4p5_all})")

# ---------- block accessors ----------
def blk(f, name):
    o = {"onset":0, "note":12, "bass":24, "treble":36}[name]
    return f[o:o+12]
def energy(f, name, pc):
    return blk(f, name)[pc % 12]

THIRD = {"maj":4,"dom":4,"aug":4,"sus":5,"min":3,"hdim":3,"dim":3}  # GT-quality 3rd
def third_int(quality):
    return THIRD.get(quality, 4)

# ---------- 2/3. Third-presence + fifth-presence probe (held-out P4/P5 errors) ----------
def probe(mask, use_bass_too=False):
    idx = np.where(mask)[0]
    tot = len(idx)
    n_third_true_wins = 0
    n_fifth_wins = 0
    for i in idx:
        f = feat[i]; tr = gt[i]; dr = pred[i]
        t3 = third_int(q[i])
        # third of TRUE root vs third of WRONG root (note block)
        et = energy(f,"note",tr+t3)
        ew = energy(f,"note",dr+t3)   # wrong root's own third, same quality assumption
        if use_bass_too:
            et += energy(f,"bass",tr+t3); ew += energy(f,"bass",dr+t3)
        if et > ew: n_third_true_wins += 1
        # wrong-root PC (the fifth of true) vs true-root PC
        e_wrongpc = energy(f,"note",dr) + (energy(f,"bass",dr) if use_bass_too else 0)
        e_truepc  = energy(f,"note",tr) + (energy(f,"bass",tr) if use_bass_too else 0)
        if e_wrongpc > e_truepc: n_fifth_wins += 1
    return tot, n_third_true_wins/tot, n_fifth_wins/tot

for tag, m in [("HELD-OUT", err & ho & is_p4p5), ("FULL-CORPUS", err & is_p4p5)]:
    tot, thw, fw = probe(m)
    tot_b, thwb, fwb = probe(m, use_bass_too=True)
    print(f"\n[{tag} P4/P5 errors, n={tot}]")
    print(f"  true-root 3rd MORE present than wrong-root 3rd (note):      {thw:.3f}")
    print(f"  true-root 3rd MORE present than wrong-root 3rd (note+bass): {thwb:.3f}")
    print(f"  wrong-root PC (=true's 5th) MORE present than true-root PC: {fw:.3f}  (BB=0.756)")
print("  [chance = 0.500; Billboard held-out third-probe = 0.498]")

# ---------- 2a. Hand-built template screen (dot AND cosine), GT quality ----------
def triad(root, quality):
    v = np.zeros(12); t3 = third_int(quality)
    for pc in (root, root+t3, root+7): v[pc % 12] = 1.0
    return v
def tmpl_screen(mask, rep, score):
    idx = np.where(mask)[0]; wins = 0
    for i in idx:
        c = blk(feat[i], rep)
        tt = triad(gt[i], q[i]); tw = triad(pred[i], q[i])
        if score == "dot":
            st, sw = c@tt, c@tw
        else:  # cosine
            st = c@tt/(np.linalg.norm(c)*np.linalg.norm(tt)+1e-9)
            sw = c@tw/(np.linalg.norm(c)*np.linalg.norm(tw)+1e-9)
        if st > sw: wins += 1
    return wins/len(idx)
print("\n=== Hand-template screen: fraction TRUE root template scores > WRONG (GT quality) ===")
m_ho = err & ho & is_p4p5
for rep in ["onset","note","bass","treble"]:
    row = " ".join(f"{score}={tmpl_screen(m_ho,rep,score):.3f}" for score in ["dot","cosine"])
    print(f"  {rep:7s} {row}")
print("  [Billboard: true root wins only 31-35% across all reps/scores]")

# ---------- 2b. Learned CV classifier (exact p4p5_learned methodology) ----------
p4p5_err = err & is_p4p5
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
labels = np.zeros(N, dtype=int)
grp = song[np.array([c[0] for c in cases])]
is_err = np.array([c[3] for c in cases])
F_third, F_fifth, F_combo, F_raw = [], [], [], []
for k,(i, tr, dr, ise) in enumerate(cases):
    f = feat[i]
    swap = rng.random() < 0.5
    a, b = (dr, tr) if swap else (tr, dr)
    labels[k] = 1 if swap else 0
    F_third.append([energy(f,"note",a+4), energy(f,"note",a+3),
                    energy(f,"note",b+4), energy(f,"note",b+3)])
    F_fifth.append([energy(f,"note",a+7), energy(f,"bass",a+7),
                    energy(f,"note",b+7), energy(f,"bass",b+7)])
    F_combo.append([energy(f,"note",a),energy(f,"bass",a),energy(f,"note",a+4),
                    energy(f,"note",a+3),energy(f,"note",a+7),energy(f,"bass",a+7),
                    energy(f,"note",b),energy(f,"bass",b),energy(f,"note",b+4),
                    energy(f,"note",b+3),energy(f,"note",b+7),energy(f,"bass",b+7)])
    def roll_all(f, r):
        out = np.empty(48, np.float32)
        for oo in (0,12,24,36): out[oo:oo+12] = np.roll(f[oo:oo+12], -r)
        return out
    F_raw.append(np.concatenate([roll_all(f,a), roll_all(f,b)]))
F_third=np.array(F_third);F_fifth=np.array(F_fifth);F_combo=np.array(F_combo);F_raw=np.array(F_raw)
y = labels
def cv_eval(X, y, groups, model_fn, err_mask):
    gkf = GroupKFold(n_splits=5); oof = np.zeros(len(y))
    for tr_i, te_i in gkf.split(X, y, groups):
        m = model_fn(); m.fit(X[tr_i], y[tr_i]); oof[te_i] = m.predict(X[te_i])
    return (oof==y).mean(), (oof[err_mask]==y[err_mask]).mean(), (oof[~err_mask]==y[~err_mask]).mean()
def lr(): return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, C=1.0))
def mlp(): return make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(16,),max_iter=800,alpha=1e-2,random_state=0))
sets = {"third(4d)":F_third,"fifth(4d)":F_fifth,"combo(12d)":F_combo,"rawfull(96d)":F_raw}
print(f"\n=== Learned CV classifier (N={N}, chance=0.500), GroupKFold by song ===")
print(f"{'featureset':14s} {'model':4s} | held-out | on-err | on-ctrl")
for name, X in sets.items():
    for mname, fn in [("LR",lr),("MLP",mlp)]:
        a,ae,ac = cv_eval(X, y, grp, fn, is_err)
        print(f"{name:14s} {mname:4s} |  {a:.3f}   | {ae:.3f}  | {ac:.3f}")
# errors-only
print("\n=== errors-only CV (>0.55 = recoverable signal; ~0.50 dead end) ===")
ei = np.where(is_err)[0]; ge = grp[ei]; ye = y[ei]
for name, X in sets.items():
    Xe = X[ei]
    for mname, fn in [("LR",lr),("MLP",mlp)]:
        gkf = GroupKFold(n_splits=5); oof = np.zeros(len(ye))
        for tr_i, te_i in gkf.split(Xe, ye, ge):
            m = fn(); m.fit(Xe[tr_i], ye[tr_i]); oof[te_i]=m.predict(Xe[te_i])
        print(f"{name:14s} {mname:4s} |  {(oof==ye).mean():.3f}")
