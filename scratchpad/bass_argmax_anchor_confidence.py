"""Argmax-BASS-anchored bass-pc classifier + CALIBRATED CONFIDENCE (2026-07-16).

Refined re-test of the previously-REJECTED argmax-renorm, per user's two asks:
  (1) anchor = argmax of the BASS-12 chroma block (feat48_abs[:,24:36]) -- the
      exact same feature as oracle_bass_family.py's bass_argmax. Verified.
  (2) add a calibrated confidence signal (max-softmax + temperature scaling);
      test whether confidence is LOWER on inversions, report ECE split by
      root-pos/inversion, and an accuracy-vs-coverage (selective prediction) curve.

Target = SOUNDING BASS pc (resolver). 5 seeds, song-grouped 80/20 CV.
Compared configs (same feature = bass-12 block, matching known_issues baseline A):
  ABSOLUTE  : MLP on bass_block, predict bass_true directly (baseline A, 0.654/0.539).
  ARGMAX-ANCHORED : rotate bass_block by -argmax, predict (bass-argmax)%12, shift back.
Also full-48-d variants for completeness.
"""
from __future__ import annotations
import sys
from pathlib import Path
from collections import Counter
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from harmonia.data.corpus_schema import sounding_bass_pc

d = np.load(REPO / "data/cache/rwc/rwc_bp48_fixed.npz", allow_pickle=True)
labels = d["labels"]; root = d["root"].astype(int) % 12; song = d["song_id"]
fabs = d["feat48_abs"].astype(np.float32)
N = len(labels)
bass_true = np.array([sounding_bass_pc(l, int(root[i])) for i, l in enumerate(labels)]).astype(int)
inv = np.array(["/" in l for l in labels])
bass_block = fabs[:, 24:36]
anchor_pc = bass_block.argmax(1)
songs = np.unique(song)


def roll_rows(X, sh):
    out = np.empty_like(X)
    for i in range(len(X)):
        out[i] = np.roll(X[i], -int(sh[i]))
    return out


def softmax_T(logits, T):
    z = logits / T
    z = z - z.max(1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(1, keepdims=True)


def fit_temperature(proba, y_idx):
    """Fit scalar T>0 minimising NLL. proba are softmax probs -> recover pseudo-logits=log p."""
    logits = np.log(np.clip(proba, 1e-8, 1))
    n = len(y_idx)

    def nll(T):
        p = softmax_T(logits, T)
        return -np.log(np.clip(p[np.arange(n), y_idx], 1e-12, 1)).mean()
    r = minimize_scalar(nll, bounds=(0.3, 5.0), method="bounded")
    return r.x, logits


def ece(conf, correct, n_bins=15):
    bins = np.linspace(0, 1, n_bins + 1)
    e, M = 0.0, len(conf)
    for i in range(n_bins):
        m = (conf > bins[i]) & (conf <= bins[i + 1])
        if m.sum():
            e += (m.sum() / M) * abs(correct[m].mean() - conf[m].mean())
    return e


def run(feat_fn, anchored, seeds=5):
    """feat_fn(mask, anchor_for_mask)-> X. If anchored, target encoded rel anchor."""
    accs, inv_accs, rp_accs = [], [], []
    ece_all, ece_rp, ece_inv = [], [], []
    conf_rp_mean, conf_inv_mean = [], []
    # pooled arrays for coverage curve (concat over seeds)
    all_conf, all_correct, all_inv = [], [], []
    for s in range(seeds):
        rng = np.random.RandomState(s); sh = songs.copy(); rng.shuffle(sh)
        cut = int(0.8 * len(sh))
        trs = set(sh[:cut]); cals = set(sh[cut - max(1, (len(sh)-cut)//1):cut])  # small calib slice
        # use last 20% of TRAIN songs as calibration for temperature
        cal_songs = set(sh[int(0.6*len(sh)):cut])
        core_songs = set(sh[:int(0.6*len(sh))])
        trm = np.array([x in trs for x in song]); tem = ~trm
        calm = np.array([x in cal_songs for x in song])
        corem = np.array([x in core_songs for x in song])

        Xtr = feat_fn(trm, anchor_pc[trm]); Xte = feat_fn(tem, anchor_pc[tem])
        Xcore = feat_fn(corem, anchor_pc[corem]); Xcal = feat_fn(calm, anchor_pc[calm])
        if anchored:
            ytr = (bass_true[trm] - anchor_pc[trm]) % 12
            ycore = (bass_true[corem] - anchor_pc[corem]) % 12
            ycal = (bass_true[calm] - anchor_pc[calm]) % 12
        else:
            ytr = bass_true[trm]; ycore = bass_true[corem]; ycal = bass_true[calm]

        sc = StandardScaler().fit(Xtr)
        clf = MLPClassifier((64, 32), max_iter=300, random_state=s, early_stopping=True)
        clf.fit(sc.transform(Xtr), ytr)

        # temperature on calibration split (fit a fresh model on core for T, but
        # simpler: fit T using the main clf's proba on the held-out cal songs)
        pcal = clf.predict_proba(sc.transform(Xcal))
        classes = clf.classes_
        y2i = {c: i for i, c in enumerate(classes)}
        ycal_i = np.array([y2i.get(v, 0) for v in ycal])
        T, _ = fit_temperature(pcal, ycal_i)

        proba = clf.predict_proba(sc.transform(Xte))
        logits = np.log(np.clip(proba, 1e-8, 1))
        proba_T = softmax_T(logits, T)
        pred_rel = classes[proba_T.argmax(1)]
        conf = proba_T.max(1)
        if anchored:
            pred = (pred_rel + anchor_pc[tem]) % 12
        else:
            pred = pred_rel
        yte = bass_true[tem]; correct = (pred == yte); im = inv[tem]
        accs.append(correct.mean())
        inv_accs.append(correct[im].mean()); rp_accs.append(correct[~im].mean())
        ece_all.append(ece(conf, correct))
        ece_rp.append(ece(conf[~im], correct[~im])); ece_inv.append(ece(conf[im], correct[im]))
        conf_rp_mean.append(conf[~im].mean()); conf_inv_mean.append(conf[im].mean())
        all_conf.append(conf); all_correct.append(correct); all_inv.append(im)
    A = lambda x: (np.mean(x), np.std(x))
    return dict(acc=A(accs), inv=A(inv_accs), rp=A(rp_accs), ece=A(ece_all),
                ece_rp=A(ece_rp), ece_inv=A(ece_inv),
                conf_rp=A(conf_rp_mean), conf_inv=A(conf_inv_mean),
                conf=np.concatenate(all_conf), correct=np.concatenate(all_correct),
                invmask=np.concatenate(all_inv))


def fabs_block(m, a): return bass_block[m]
def fabs_block_anch(m, a): return roll_rows(bass_block[m], a)
def full48(m, a): return fabs[m]
def full48_anch(m, a):
    return np.hstack([roll_rows(fabs[m][:, i:i+12], a) for i in (0, 12, 24, 36)])


print(f"N={N}  inversions={inv.sum()} ({100*inv.mean():.1f}%)  "
      f"anchor==sounding-bass={ (anchor_pc==bass_true).mean():.4f}\n")

CONFIGS = [
    ("ABSOLUTE bass-12 (baseline A)", fabs_block, False),
    ("ARGMAX-ANCHORED bass-12", fabs_block_anch, True),
    ("ABSOLUTE full-48", full48, False),
    ("ARGMAX-ANCHORED full-48", full48_anch, True),
]
res = {}
print(f"{'config':32s} {'acc':>13s} {'root-pos':>13s} {'inversion':>13s} "
      f"{'ECE':>7s} {'ECErp':>7s} {'ECEinv':>7s} {'conf-rp':>8s} {'conf-inv':>8s}")
for name, fn, anch in CONFIGS:
    r = run(fn, anch)
    res[name] = r
    print(f"{name:32s} {r['acc'][0]:.3f}±{r['acc'][1]:.3f} {r['rp'][0]:.3f}±{r['rp'][1]:.3f} "
          f"{r['inv'][0]:.3f}±{r['inv'][1]:.3f} {r['ece'][0]:.3f}  {r['ece_rp'][0]:.3f}  "
          f"{r['ece_inv'][0]:.3f}  {r['conf_rp'][0]:.3f}   {r['conf_inv'][0]:.3f}")

print("\n=== CONFIDENCE hypothesis: is confidence LOWER on inversions? ===")
for name in [c[0] for c in CONFIGS]:
    r = res[name]
    dlt = r['conf_rp'][0] - r['conf_inv'][0]
    print(f"  {name:32s} conf(root-pos)={r['conf_rp'][0]:.3f}  conf(inv)={r['conf_inv'][0]:.3f}  "
          f"delta={dlt:+.3f}")

print("\n=== ACCURACY vs COVERAGE (selective prediction, pooled over seeds) ===")
print("Answer only the top-c fraction by confidence; report accuracy on that subset.")
for name in ["ABSOLUTE bass-12 (baseline A)", "ARGMAX-ANCHORED bass-12"]:
    r = res[name]
    conf = r['conf']; correct = r['correct']
    order = np.argsort(-conf)
    print(f"\n  [{name}]  full-coverage acc = {correct.mean():.3f}")
    print(f"  {'coverage':>9s} {'acc':>7s} {'thresh':>7s}")
    im = res[name]['invmask']
    print(f"  {'coverage':>9s} {'acc':>7s} {'thresh':>7s} {'acc-rp':>7s} {'acc-inv':>7s} {'inv-kept%':>9s}")
    n_inv_tot = im.sum()
    for cov in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1):
        k = max(1, int(cov * len(conf)))
        idx = order[:k]
        km = np.zeros(len(conf), bool); km[idx] = True
        rp_ans = km & ~im; iv_ans = km & im
        acc_rp = correct[rp_ans].mean() if rp_ans.sum() else float('nan')
        acc_iv = correct[iv_ans].mean() if iv_ans.sum() else float('nan')
        inv_kept = iv_ans.sum() / n_inv_tot
        print(f"  {cov*100:7.0f}% {correct[idx].mean():7.3f} {conf[idx].min():7.3f} "
              f"{acc_rp:7.3f} {acc_iv:7.3f} {inv_kept*100:8.1f}%")
