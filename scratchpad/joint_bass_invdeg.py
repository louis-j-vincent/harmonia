"""Joint bass-pc + inversion-degree prediction, and closing the loop to quality.

Builds on the oracle-bass adjudication (scratchpad/oracle_bass_adjudicate.py,
known_issues 2026-07-16): feeding the TRUE inversion-degree one-hot to a
bass-anchored quality classifier closes the root/bass gap (0.560->0.879 on
inversions at the (256,128,64) config). That feature was ORACLE. This script
builds a model that PREDICTS the inversion-degree (jointly with bass) and asks
how much of the oracle gain survives with a realistic predicted degree.

Inversion-degree here = semitone offset of the sounding bass above the
functional root, (sounding_bass_pc - root) % 12 in 0..11. deg 0 = root position.
This 12-way semitone offset is exactly the one-hot the oracle experiment fed to
the quality classifier (rotation-regime indicator), so we predict it directly
and also report a collapsed {root/3rd/5th/7th/other} view for interpretability.

Parts
-----
1. torch models on feat48_abs (48-d beat chroma, 4 octave blocks x 12):
     ST-bass    trunk + absolute-bass-pc head            (the 0.654 baseline)
     ST-degree  trunk + degree head (single-task)
     MT         shared trunk + BOTH heads (joint training)
     SEQ        bass head, then degree head on [chroma (+) predicted-bass one-hot]
   Global-transpose roll augmentation (roll chroma by k: bass += k, degree
   invariant, root += k). Reports bass-pc acc (ST vs MT) and degree acc
   (ST vs MT vs SEQ), 5-seed song-grouped 80/20 CV.
2. Degree accuracy on its own: raw 12-way + balanced (collapsed) + inversions,
   split by whether the abs-bass prediction was correct (error correlation).
3. Quality closing-loop (sklearn, exact oracle setup, bass-anchored on TRUE
   bass): no-degree  vs  PREDICTED-degree one-hot  vs  ORACLE-degree one-hot,
   all at (256,128,64) noES 600it a=3e-4. Predicted degree is fully out-of-fold
   (two swapped fits per seed so both train & test quality folds use OOF preds).

Read-only on the corpus npz and corpus_schema. No commits.
"""
from __future__ import annotations
import json, sys, time
from pathlib import Path
from collections import Counter
import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from harmonia.data.corpus_schema import sounding_bass_pc

import torch
import torch.nn as nn
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score

import os
DEV = "mps" if torch.backends.mps.is_available() else "cpu"
SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 5
NSHIFT = int(os.environ.get("NSHIFT", "12"))   # transpose-aug copies (12=full)
EPOCHS = int(os.environ.get("EPOCHS", "60"))
QITER = int(os.environ.get("QITER", "600"))    # sklearn quality MLP max_iter
rng_global = np.random.RandomState(0)

# ---------------------------------------------------------------- data
d = np.load(REPO / "data/cache/rwc/rwc_bp48_fixed.npz", allow_pickle=True)
labels = [str(x) for x in d["labels"]]
root = d["root"].astype(int)
qidx = d["quality_idx"].astype(int)
song = d["song_id"]
fabs = d["feat48_abs"].astype(np.float32)   # absolute chroma
N = len(labels)
songs = np.unique(song)

bass_true = np.array([int(sounding_bass_pc(l, int(root[i]))) for i, l in enumerate(labels)])
degree = (bass_true - root) % 12             # 12-way semitone offset, 0 = root pos
inv_mask = degree != 0

# collapsed degree classes for balanced reporting
def collapse(deg):
    if deg == 0: return 0            # root
    if deg in (3, 4): return 1       # 3rd
    if deg == 7: return 2            # 5th
    if deg in (10, 11): return 3     # 7th
    return 4                         # other
deg_coll = np.array([collapse(x) for x in degree])
COLL_NAMES = ["root", "3rd", "5th", "7th", "other"]

print(f"N={N} songs={len(songs)} inversions={inv_mask.mean():.3f} dev={DEV} seeds={SEEDS}")
print("degree dist:", {int(k): int(v) for k, v in sorted(Counter(degree).items())})


def split(seed):
    sh = songs.copy(); np.random.RandomState(seed).shuffle(sh)
    cut = int(0.8 * len(sh)); trs = set(sh[:cut].tolist())
    trm = np.array([s in trs for s in song])
    return trm, ~trm


# ---------------------------------------------------------------- roll aug
def roll_all(F, k):
    """Roll every 12-block of F (n,48) by k (global transpose up by k)."""
    n = F.shape[0]
    return np.roll(F.reshape(n, 4, 12), shift=k, axis=2).reshape(n, 48)


def make_aug(F, bass, deg, n_shifts=NSHIFT):
    """12x transpose aug. bass -> (bass+k)%12, degree invariant."""
    Xs, bs, ds = [], [], []
    for k in range(n_shifts):
        Xs.append(roll_all(F, k)); bs.append((bass + k) % 12); ds.append(deg)
    return (np.concatenate(Xs).astype(np.float32),
            np.concatenate(bs), np.concatenate(ds))


# ---------------------------------------------------------------- torch model
class MultiHead(nn.Module):
    def __init__(self, d_in, hidden, heads):
        super().__init__()
        layers = []; prev = d_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(0.1)]; prev = h
        self.trunk = nn.Sequential(*layers)
        self.heads = nn.ModuleDict({k: nn.Linear(prev, n) for k, n in heads.items()})

    def forward(self, x):
        z = self.trunk(x)
        return {k: h(z) for k, h in self.heads.items()}


def class_weights(y, k=12, cap=15.0):
    """Inverse-frequency class weights, normalized to mean 1, clipped at cap.
    Counters the 87.6% root-position (deg 0) majority collapse."""
    cnt = np.bincount(y, minlength=k).astype(float)
    w = np.where(cnt > 0, 1.0 / cnt, 0.0)
    w = w / w[w > 0].mean()
    w = np.clip(w, 0.0, cap)
    return torch.tensor(w, device=DEV, dtype=torch.float32)


def train_model(Xtr, targets, d_in, heads, epochs=EPOCHS, lr=1e-3, batch=256, seed=0,
                weighted=()):  # weighted: head names to apply inverse-freq class weights
    """targets: dict head_name -> np int array aligned to Xtr."""
    torch.manual_seed(seed)
    mean = Xtr.mean(0, keepdims=True); std = Xtr.std(0, keepdims=True) + 1e-6
    Xn = ((Xtr - mean) / std).astype(np.float32)
    xt = torch.tensor(Xn, device=DEV)
    yts = {k: torch.tensor(v, device=DEV, dtype=torch.long) for k, v in targets.items()}
    model = MultiHead(d_in, [128, 64], heads).to(DEV)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    lossfs = {k: nn.CrossEntropyLoss(weight=class_weights(targets[k], heads[k])
                                     if k in weighted else None) for k in heads}
    n = len(Xn)
    for ep in range(epochs):
        perm = torch.randperm(n, device=DEV)
        model.train()
        for i in range(0, n, batch):
            idx = perm[i:i+batch]
            out = model(xt[idx])
            loss = sum(lossfs[k](out[k], yts[k][idx]) for k in heads)
            opt.zero_grad(); loss.backward(); opt.step()
    return model, mean, std


def predict(model, X, mean, std, head):
    model.eval()
    Xn = ((X - mean) / std).astype(np.float32)
    with torch.no_grad():
        out = model(torch.tensor(Xn, device=DEV))[head].cpu().numpy()
    return out


# ---------------------------------------------------------------- Part 1 & 2
records = []          # per-seed metrics
oof_deg_pred = {}     # seed -> full-length predicted degree (OOF)
oof_bass_pred = {}    # seed -> full-length predicted abs bass (OOF, ST-bass)

for seed in range(SEEDS):
    t0 = time.time()
    trm, tem = split(seed)
    Ftr, Fte = fabs[trm], fabs[tem]
    bass_tr, bass_te = bass_true[trm], bass_true[tem]
    deg_tr, deg_te = degree[trm], degree[tem]

    Xaug, baug, daug = make_aug(Ftr, bass_tr, deg_tr)

    # ST-bass
    m_b, mu, sd = train_model(Xaug, {"bass": baug}, 48, {"bass": 12}, seed=seed)
    bass_pred = predict(m_b, Fte, mu, sd, "bass").argmax(1)
    bass_acc = (bass_pred == bass_te).mean()

    # ST-degree (class-weighted to counter root-position majority collapse)
    m_d, mu2, sd2 = train_model(Xaug, {"deg": daug}, 48, {"deg": 12}, seed=seed,
                                weighted=("deg",))
    stdeg_pred = predict(m_d, Fte, mu2, sd2, "deg").argmax(1)
    stdeg_acc = (stdeg_pred == deg_te).mean()

    # MT joint (bass unweighted, degree class-weighted)
    m_mt, mu3, sd3 = train_model(Xaug, {"bass": baug, "deg": daug}, 48,
                                 {"bass": 12, "deg": 12}, seed=seed, weighted=("deg",))
    mt_bass_pred = predict(m_mt, Fte, mu3, sd3, "bass").argmax(1)
    mt_deg_pred = predict(m_mt, Fte, mu3, sd3, "deg").argmax(1)
    mt_bass_acc = (mt_bass_pred == bass_te).mean()
    mt_deg_acc = (mt_deg_pred == deg_te).mean()

    # SEQ: degree from [chroma (+) predicted-bass one-hot]
    # bass preds for train (in-sample from m_b) & test, one-hot appended.
    def bass_oh(F, model, mu, sd):
        p = predict(model, F, mu, sd, "bass").argmax(1)
        oh = np.zeros((len(F), 12), np.float32); oh[np.arange(len(F)), p] = 1
        return oh, p
    # need aug versions consistent: recompute bass one-hot on augmented train
    btr_oh, _ = bass_oh(Ftr, m_b, mu, sd)
    # augment [chroma+onehot]: roll chroma, roll bass one-hot by same k, deg invariant
    Xseq_parts, dseq_parts = [], []
    for k in range(NSHIFT):
        Xr = roll_all(Ftr, k)
        ohr = np.roll(btr_oh, k, axis=1)
        Xseq_parts.append(np.concatenate([Xr, ohr], axis=1)); dseq_parts.append(deg_tr)
    Xseq = np.concatenate(Xseq_parts).astype(np.float32)
    dseq = np.concatenate(dseq_parts)
    m_seq, mus, sds = train_model(Xseq, {"deg": dseq}, 60, {"deg": 12}, seed=seed,
                                  weighted=("deg",))
    bte_oh, _ = bass_oh(Fte, m_b, mu, sd)
    Xseq_te = np.concatenate([Fte, bte_oh], axis=1).astype(np.float32)
    seq_deg_pred = predict(m_seq, Xseq_te, mus, sds, "deg").argmax(1)
    seq_deg_acc = (seq_deg_pred == deg_te).mean()

    # error-correlation: degree acc split by whether abs-bass pred correct (MT deg, ST bass)
    bcorr = bass_pred == bass_te
    def sub_acc(mask):
        return (mt_deg_pred[mask] == deg_te[mask]).mean() if mask.sum() else float("nan")
    deg_acc_bcorr = sub_acc(bcorr)
    deg_acc_bwrong = sub_acc(~bcorr)
    inv_te = inv_mask[tem]
    # inversions-only degree acc
    def inv_acc(pred):
        return (pred[inv_te] == deg_te[inv_te]).mean() if inv_te.sum() else float("nan")

    # balanced acc on collapsed classes (MT degree)
    coll_te = deg_coll[tem]
    mt_coll_pred = np.array([collapse(x) for x in mt_deg_pred])
    bal_coll = balanced_accuracy_score(coll_te, mt_coll_pred)

    rec = dict(seed=seed,
               bass_st=float(bass_acc), bass_mt=float(mt_bass_acc),
               deg_st=float(stdeg_acc), deg_mt=float(mt_deg_acc), deg_seq=float(seq_deg_acc),
               deg_st_inv=float(inv_acc(stdeg_pred)), deg_mt_inv=float(inv_acc(mt_deg_pred)),
               deg_seq_inv=float(inv_acc(seq_deg_pred)),
               deg_mt_bal_coll=float(bal_coll),
               deg_acc_bass_correct=float(deg_acc_bcorr), deg_acc_bass_wrong=float(deg_acc_bwrong),
               frac_bass_correct=float(bcorr.mean()),
               n_inv_te=int(inv_te.sum()))
    records.append(rec)
    print(f"seed {seed}: bass ST={bass_acc:.3f} MT={mt_bass_acc:.3f} | "
          f"deg ST={stdeg_acc:.3f} MT={mt_deg_acc:.3f} SEQ={seq_deg_acc:.3f} | "
          f"deg-inv MT={inv_acc(mt_deg_pred):.3f} | "
          f"deg|bassOK={deg_acc_bcorr:.3f} deg|bassBAD={deg_acc_bwrong:.3f} "
          f"({time.time()-t0:.0f}s)", flush=True)

    # ---- OOF predicted degree for ALL rows (this seed): swap-fit ----
    # model above trained on train->predicts test. Now train on test->predict train.
    Xaug_te, baug_te, daug_te = make_aug(Fte, bass_te, deg_te)
    m_mt2, mu4, sd4 = train_model(Xaug_te, {"bass": baug_te, "deg": daug_te}, 48,
                                  {"bass": 12, "deg": 12}, seed=seed + 100, weighted=("deg",))
    deg_pred_full = np.empty(N, int)
    deg_pred_full[tem] = mt_deg_pred
    deg_pred_full[trm] = predict(m_mt2, Ftr, mu4, sd4, "deg").argmax(1)
    oof_deg_pred[seed] = deg_pred_full
    bass_pred_full = np.empty(N, int)
    bass_pred_full[tem] = bass_pred
    bass_pred_full[trm] = predict(m_mt2, Ftr, mu4, sd4, "bass").argmax(1)
    oof_bass_pred[seed] = bass_pred_full


# ---------------------------------------------------------------- Part 3 quality
# exact oracle setup: bass-anchor on TRUE bass, sklearn MLP (256,128,64) noES 600it
def roll_blocks_by(F, shifts):
    out = np.empty_like(F)
    for i in range(len(F)):
        s = int(shifts[i])
        for b in range(4):
            out[i, b*12:(b+1)*12] = np.roll(F[i, b*12:(b+1)*12], -s)
    return out

X_oracle = roll_blocks_by(fabs, bass_true)   # bass-anchored frame (TRUE bass)


def onehot(v, k=12):
    oh = np.zeros((len(v), k), np.float32); oh[np.arange(len(v)), v] = 1; return oh


def quality_eval(degree_feature):
    """degree_feature: None | 'oracle' | dict seed->full pred degree.
    Returns (pooled mean/std, inv mean/std) over seeds."""
    pooled, inv = [], []
    for seed in range(SEEDS):
        trm, tem = split(seed)
        if degree_feature is None:
            X = X_oracle
        elif degree_feature == "oracle":
            X = np.hstack([X_oracle, onehot(degree)])
        else:
            X = np.hstack([X_oracle, onehot(degree_feature[seed])])
        sc = StandardScaler().fit(X[trm])
        clf = MLPClassifier((256, 128, 64), max_iter=QITER, random_state=seed,
                            early_stopping=False, alpha=3e-4, learning_rate_init=1e-3,
                            n_iter_no_change=15)
        clf.fit(sc.transform(X[trm]), qidx[trm])
        pred = clf.predict(sc.transform(X[tem])); yte = qidx[tem]
        pooled.append((pred == yte).mean())
        im = inv_mask[tem]
        inv.append((pred[im] == yte[im]).mean() if im.sum() else np.nan)
    return (float(np.mean(pooled)), float(np.std(pooled)),
            float(np.mean(inv)), float(np.std(inv)))

print("\n=== Part 3: bass-anchored quality (256,128,64 noES 600it) ===", flush=True)
q_none = quality_eval(None)
print(f"  no degree        pooled {q_none[0]:.3f}±{q_none[1]:.3f}  inv {q_none[2]:.3f}±{q_none[3]:.3f}", flush=True)
q_pred = quality_eval(oof_deg_pred)
print(f"  PREDICTED degree pooled {q_pred[0]:.3f}±{q_pred[1]:.3f}  inv {q_pred[2]:.3f}±{q_pred[3]:.3f}", flush=True)
q_orac = quality_eval("oracle")
print(f"  oracle degree    pooled {q_orac[0]:.3f}±{q_orac[1]:.3f}  inv {q_orac[2]:.3f}±{q_orac[3]:.3f}", flush=True)


def ms(key):
    v = np.array([r[key] for r in records]); return float(v.mean()), float(v.std())

summary = {
    "config": dict(N=N, songs=len(songs), inv_frac=float(inv_mask.mean()), seeds=SEEDS),
    "bass_st": ms("bass_st"), "bass_mt": ms("bass_mt"),
    "deg_st": ms("deg_st"), "deg_mt": ms("deg_mt"), "deg_seq": ms("deg_seq"),
    "deg_st_inv": ms("deg_st_inv"), "deg_mt_inv": ms("deg_mt_inv"), "deg_seq_inv": ms("deg_seq_inv"),
    "deg_mt_bal_coll": ms("deg_mt_bal_coll"),
    "deg_acc_bass_correct": ms("deg_acc_bass_correct"),
    "deg_acc_bass_wrong": ms("deg_acc_bass_wrong"),
    "frac_bass_correct": ms("frac_bass_correct"),
    "quality_no_degree": q_none, "quality_pred_degree": q_pred, "quality_oracle_degree": q_orac,
}
(REPO / "scratchpad/joint_bass_invdeg_result.json").write_text(
    json.dumps({"summary": summary, "records": records}, indent=2))

print("\n" + "=" * 70)
print("SUMMARY (mean±std over seeds)")
print(f"  bass-pc acc      ST={ms('bass_st')[0]:.3f}  MT={ms('bass_mt')[0]:.3f}")
print(f"  degree acc (all) ST={ms('deg_st')[0]:.3f}  MT={ms('deg_mt')[0]:.3f}  SEQ={ms('deg_seq')[0]:.3f}")
print(f"  degree acc (inv) ST={ms('deg_st_inv')[0]:.3f}  MT={ms('deg_mt_inv')[0]:.3f}  SEQ={ms('deg_seq_inv')[0]:.3f}")
print(f"  degree bal-acc (collapsed) MT={ms('deg_mt_bal_coll')[0]:.3f}")
print(f"  degree acc | bass correct={ms('deg_acc_bass_correct')[0]:.3f}  | bass wrong={ms('deg_acc_bass_wrong')[0]:.3f}")
print(f"  QUALITY on inversions: none={q_none[2]:.3f}  PRED={q_pred[2]:.3f}  oracle={q_orac[2]:.3f}")
print("wrote scratchpad/joint_bass_invdeg_result.json")
