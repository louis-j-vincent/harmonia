"""FINAL ensemble — proper OUT-OF-FOLD stacked meta over {NNLS heads, NNLS argmax,
music-x-lab} for sounding-bass pc on RWC (100 songs). Leak-free: for each outer test
fold, the NNLS heads' predictions on the meta-training songs come from an INNER k-fold
(OOF), so the meta trains on ~80 songs of honest head preds + argmax + musx, then is
applied to the outer-test songs (heads retrained on all 80 outer-train songs).

Compares: best single estimator (music-x-lab), naive-val stacker, OOF stacker, and a
musx-primary confidence gate. Also reports the oracle ceiling. All numbers completed runs.
"""
import sys, json, numpy as np
from pathlib import Path
sys.path.insert(0, '.'); sys.path.insert(0, 'scratchpad'); sys.path.insert(0, 'scripts')
from harmonia.data.corpus_schema import load_corpus, sounding_bass_pc
from scripts.build_jaah_corpus import parse_jaah as parse_harte
from multihead_training import train_clf, predict_proba
from rwc_nnls_multihead_cv import song_split
from sklearn.linear_model import LogisticRegression

SCRATCH = Path("/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-Code-harmonia/a6a4757d-a729-450a-a5be-b9970a43c412/scratchpad")
MUSX_OUT = SCRATCH / "musx_out"

def load_lab(p):
    out = []
    if p.exists():
        for line in p.read_text().splitlines():
            q = line.split()
            if len(q) >= 3:
                try: out.append((float(q[0]), float(q[1]), q[2]))
                except ValueError: pass
    return out
def lab_at(iv, t):
    for a, b, l in iv:
        if a <= t < b: return l
    return None
def oh(x, k=12):
    o = np.zeros((len(x), k), np.float32); m = x >= 0; o[np.arange(len(x))[m], x[m]] = 1; return o

nn = load_corpus('data/cache/rwc/rwc_nnls24.npz'); bp = load_corpus('data/cache/rwc/rwc_bp48_fixed.npz')
nn24 = nn['nnls24'].astype('float32'); keep = np.abs(nn24).sum(1) > 0
nn24 = nn24[keep]; roots = bp['root'].astype('int64')[keep] % 12
sid = bp['song_id'][keep]; labels = bp['labels'][keep]
t0 = bp['t0'][keep].astype(float); t1 = bp['t1'][keep].astype(float)
_g = [sounding_bass_pc(str(labels[i]), int(roots[i])) for i in range(len(roots))]
gb = np.array([(-1 if v is None else v % 12) for v in _g], dtype=np.int64)
argmax = nn24[:, :12].argmax(1); inv = gb != roots

musx_bass = np.full(len(gb), -1, int)
for i in range(1, 101):
    rid = f"RWC_P{i:03d}"; iv = load_lab(MUSX_OUT / f"{rid}.lab")
    if not iv: continue
    for gi in np.where(sid == "rwc_" + rid)[0]:
        pl = lab_at(iv, 0.5*(t0[gi]+t1[gi]))
        if pl is None: continue
        pr, pf, _ = parse_harte(pl)
        if pr is None: continue
        b = sounding_bass_pc(pl, pr)
        if b is not None: musx_bass[gi] = b % 12
cov = musx_bass >= 0
print(f"musx-covered rows: {cov.sum()}", flush=True)

def train_heads(mask_tr, mask_va):
    mf = train_clf(nn24[mask_tr], gb[mask_tr], nn24[mask_va], gb[mask_va], 24, 12, hid=(128,64), epochs=50)
    mb = train_clf(nn24[mask_tr][:, :12], gb[mask_tr], nn24[mask_va][:, :12], gb[mask_va], 12, 12, hid=(128,64), epochs=50)
    return mf, mb

def feats(mask, pf_p, pb_p):
    return np.concatenate([pf_p[mask], pb_p[mask], oh(argmax[mask]), oh(musx_bass[mask])], 1)

pool = {k: [] for k in ["gb","inv","musx","argmax","headf","stack_oof","gate"]}
songs = np.unique(sid)
for seed in range(5):
    tr, va, te = song_split(sid, seed)
    outer_tr = (tr | va) & cov            # meta-train pool (non-test songs)
    te_c = te & cov
    # heads trained on all outer-train, predict outer-test
    mf, mb = train_heads(tr & cov, va & cov)
    pf_full = predict_proba(mf, nn24); pb_full = predict_proba(mb, nn24[:, :12])
    # INNER OOF: produce head preds on outer-train songs, leak-free
    otr_songs = np.unique(sid[outer_tr]); rng = np.random.RandomState(100+seed); rng.shuffle(otr_songs)
    folds = np.array_split(otr_songs, 4)
    pf_oof = np.zeros((len(gb), 12), np.float32); pb_oof = np.zeros((len(gb), 12), np.float32)
    for k in range(4):
        hold = np.isin(sid, folds[k]) & outer_tr
        fit = outer_tr & ~hold
        # small val slice from fit for early stop
        fs = np.unique(sid[fit]); rng.shuffle(fs); vs = set(fs[:max(1,len(fs)//9)])
        vamask = np.isin(sid, list(vs)) & fit; trmask = fit & ~vamask
        imf, imb = train_heads(trmask, vamask)
        pf_oof[hold] = predict_proba(imf, nn24[hold]); pb_oof[hold] = predict_proba(imb, nn24[hold][:, :12])
    # meta trains on outer-train OOF feats, applied to outer-test full-head feats
    metaO = LogisticRegression(max_iter=4000, C=0.5)
    metaO.fit(feats(outer_tr, pf_oof, pb_oof), gb[outer_tr])
    stack_oof = metaO.predict(feats(te_c, pf_full, pb_full))
    # musx-primary confidence gate: override musx with headf-argmax consensus only where they agree with each other and disagree with musx
    hf = pf_full.argmax(1)
    gate = musx_bass[te_c].copy()
    consensus = (hf[te_c] == argmax[te_c]) & (hf[te_c] != musx_bass[te_c])
    gate[consensus] = hf[te_c][consensus]
    for k, v in [("gb", gb[te_c]), ("inv", inv[te_c]), ("musx", musx_bass[te_c]),
                 ("argmax", argmax[te_c]), ("headf", hf[te_c]), ("stack_oof", stack_oof), ("gate", gate)]:
        pool[k] += list(v)
    print(f"[seed {seed}] te={te_c.sum()} musx={np.mean(musx_bass[te_c]==gb[te_c]):.3f} "
          f"stack_oof={np.mean(stack_oof==gb[te_c]):.3f} gate={np.mean(gate==gb[te_c]):.3f}", flush=True)

A = {k: np.array(v) for k, v in pool.items()}
GT = A["gb"]; INV = A["inv"] == 1; n = len(GT)
def acc(p, m): return float((p[m] == GT[m]).mean()) if m.sum() else float('nan')
allm = np.ones(n, bool)
print(f"\n{'='*60}\nFINAL ensemble, 5-seed song-grouped CV, {n} musx-covered test chords ({INV.sum()} inv)")
res = {}
for lbl, p in [("music-x-lab (best single)", A["musx"]), ("NNLS argmax", A["argmax"]),
               ("NNLS head full24", A["headf"]), ("OOF STACK {heads,argmax,musx}", A["stack_oof"]),
               ("musx-primary gate", A["gate"])]:
    print(f"   {lbl:32s} all={acc(p,allm):.3f}  inv={acc(p,INV):.3f}")
    res[lbl] = [acc(p, allm), acc(p, INV)]
def oracle(ests):
    c = np.zeros(n, bool)
    for e in ests: c |= (e == GT)
    return float(c.mean()), float((np.zeros(INV.sum(), bool) | np.logical_or.reduce([e[INV]==GT[INV] for e in ests])).mean())
oc = oracle([A["argmax"], A["headf"], A["musx"]])
print(f"   {'ORACLE {argmax,headf,musx}':32s} all={oc[0]:.3f}  inv={oc[1]:.3f}")
res["oracle"] = list(oc)
json.dump({"n": n, "n_inv": int(INV.sum()), "results": res},
          open('scratchpad/bass_oof_stack_result.json', 'w'), indent=2)
print("saved scratchpad/bass_oof_stack_result.json")
