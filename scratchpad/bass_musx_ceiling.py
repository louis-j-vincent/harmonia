"""Fold music-x-lab (ISMIR2019 Chord Structure Decomposition) bass predictions into
the oracle ceiling + stacker, re-scored at OUR corpus chord spans (rwc_bp48_fixed
t0/t1), so it is row-aligned with {argmax, trained NNLS heads, pYIN}. musx covers
RWC_P001..P018 only. All numbers from completed runs.
"""
import sys, numpy as np
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
    if not p.exists(): return out
    for line in p.read_text().splitlines():
        q = line.split()
        if len(q) < 3: continue
        try: out.append((float(q[0]), float(q[1]), q[2]))
        except ValueError: pass
    return out

def lab_at(iv, t):
    for a, b, l in iv:
        if a <= t < b: return l
    return None

nn = load_corpus('data/cache/rwc/rwc_nnls24.npz'); bp = load_corpus('data/cache/rwc/rwc_bp48_fixed.npz')
proxy = np.load('scratchpad/pyin_bass_cache.npz', allow_pickle=True)
nn24 = nn['nnls24'].astype('float32'); keep = np.abs(nn24).sum(1) > 0
nn24 = nn24[keep]; roots = bp['root'].astype('int64')[keep] % 12
sid = bp['song_id'][keep]; labels = bp['labels'][keep]
t0 = bp['t0'][keep].astype(float); t1 = bp['t1'][keep].astype(float)
_gbl = [sounding_bass_pc(str(labels[i]), int(roots[i])) for i in range(len(roots))]
gb = np.array([(-1 if v is None else v % 12) for v in _gbl], dtype=np.int64)
argmax = nn24[:, :12].argmax(1)
proxy_bass = proxy['bass_pc'][keep].astype('int64')
inv = gb != roots

# musx bass pc per row (18 songs), aligned at OUR span midpoints
musx_bass = np.full(len(gb), -1, int)
musx_songs = [f"RWC_P{i:03d}" for i in range(1, 101)]
for rid in musx_songs:
    iv = load_lab(MUSX_OUT / f"{rid}.lab")
    if not iv: continue
    sfull = "rwc_" + rid
    idx = np.where(sid == sfull)[0]
    for gi in idx:
        pl = lab_at(iv, 0.5*(t0[gi]+t1[gi]))
        if pl is None: continue
        pr, pf, _ = parse_harte(pl)
        if pr is None: continue
        b = sounding_bass_pc(pl, pr)
        if b is not None: musx_bass[gi] = b % 12
in_musx = musx_bass >= 0
print(f"musx-covered rows: {in_musx.sum()} ({int((in_musx&inv).sum())} inversions)", flush=True)

# pooled test over 5 seeds, restricted to musx-covered rows
pooled = {k: [] for k in ["gb","inv","argmax","headf","headb","pyin","musx","stack"]}
for seed in range(5):
    tr, va, te = song_split(sid, seed)
    mf = train_clf(nn24[tr], gb[tr], nn24[va], gb[va], 24, 12, hid=(128,64), epochs=50)
    mb = train_clf(nn24[tr][:,:12], gb[tr], nn24[va][:,:12], gb[va], 12, 12, hid=(128,64), epochs=50)
    pf_p = predict_proba(mf, nn24); pb_p = predict_proba(mb, nn24[:,:12])
    pf = pf_p.argmax(1); pb = pb_p.argmax(1)
    def oh(x):
        o = np.zeros((len(x),12),np.float32); m=x>=0; o[np.arange(len(x))[m],x[m]]=1; return o
    # stacker incl musx: meta trained on VAL (leak-free), features head probas + argmax/pyin/musx one-hots
    def MX(mask):
        return np.concatenate([pf_p[mask], pb_p[mask], oh(argmax[mask]),
                               oh(np.where(proxy_bass[mask]>=0,proxy_bass[mask],-1)),
                               oh(np.where(musx_bass[mask]>=0,musx_bass[mask],-1))],1)
    m_te = te & in_musx; m_va = va & in_musx
    meta = LogisticRegression(max_iter=3000, C=1.0); meta.fit(MX(m_va), gb[m_va])
    stack = meta.predict(MX(m_te))
    for k,v in [("gb",gb[m_te]),("inv",inv[m_te]),("argmax",argmax[m_te]),("headf",pf[m_te]),
                ("headb",pb[m_te]),("pyin",proxy_bass[m_te]),("musx",musx_bass[m_te]),("stack",stack)]:
        pooled[k]+=list(v)

A = {k:np.array(v) for k,v in pooled.items()}
GT=A["gb"]; INV=A["inv"]==1; n=len(GT)
def acc(p,m): return float((p[m]==GT[m]).mean()) if m.sum() else float('nan')
allm=np.ones(n,bool)
print(f"\nPooled musx-covered TEST rows: {n} ({INV.sum()} inv)")
print("standalone estimators (all / inv):")
for lbl,p in [("NNLS argmax",A["argmax"]),("head full24",A["headf"]),("head bass12",A["headb"]),
              ("proxy-pYIN",A["pyin"]),("music-x-lab",A["musx"]),("STACK(+musx)",A["stack"])]:
    print(f"   {lbl:14s} all={acc(p,allm):.3f}  inv={acc(p,INV):.3f}")
def oracle(ests,m):
    c=np.zeros(m.sum(),bool)
    for e in ests: c|=(e[m]==GT[m])
    return float(c.mean())
base=[A["argmax"],A["headf"],A["headb"]]
print(f"\nORACLE best-of-N ceiling (all / inv):")
print(f"   {{argmax,headf,headb}}          all={oracle(base,allm):.3f}  inv={oracle(base,INV):.3f}")
print(f"   {{+pyin}}                        all={oracle(base+[A['pyin']],allm):.3f}  inv={oracle(base+[A['pyin']],INV):.3f}")
print(f"   {{+musx}}                        all={oracle(base+[A['musx']],allm):.3f}  inv={oracle(base+[A['musx']],INV):.3f}")
print(f"   {{+pyin+musx}}                   all={oracle(base+[A['pyin'],A['musx']],allm):.3f}  inv={oracle(base+[A['pyin'],A['musx']],INV):.3f}")
