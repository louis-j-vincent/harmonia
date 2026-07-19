"""GT-NOISE INVESTIGATION. Find chords where the 4 INDEPENDENT bass estimators
{NNLS argmax, trained NNLS-24 head, pYIN/low-pass proxy, music-x-lab} AGREE WITH
EACH OTHER but DISAGREE with the RWC ground-truth sounding bass. Consensus among
4 independently-built systems on a *wrong* answer is much stronger evidence of a
GT error than of a genuinely hard chord (which would scatter the estimators).

For each consensus-vs-GT row we also read the NNLS BASS CHROMA (first 12 bins =
per-pitch-class bass energy) to spectrally check: does the audio bass energy back
the estimator consensus over the GT label? That is the inspectable evidence.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
SCRATCH = Path("/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-Code-harmonia/a6a4757d-a729-450a-a5be-b9970a43c412/scratchpad")
MUSX_OUT = SCRATCH / "musx_out"
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scratchpad")); sys.path.insert(0, str(REPO / "scripts"))
from harmonia.data.corpus_schema import load_corpus, sounding_bass_pc
from scripts.build_jaah_corpus import parse_jaah as parse_harte
from multihead_training import train_clf, predict_proba
from rwc_nnls_multihead_cv import song_split

PC = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
QUAL = ['maj', 'min', 'dom', 'hdim', 'dim', 'aug', 'sus']

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

nn = load_corpus(REPO / 'data/cache/rwc/rwc_nnls24.npz')
bp = load_corpus(REPO / 'data/cache/rwc/rwc_bp48_fixed.npz')
proxy = np.load(REPO / 'scratchpad/pyin_bass_cache.npz', allow_pickle=True)
nn24 = nn['nnls24'].astype('float32'); keep = np.abs(nn24).sum(1) > 0
nn24 = nn24[keep]; roots = bp['root'].astype('int64')[keep] % 12
sid = bp['song_id'][keep]; labels = bp['labels'][keep]
t0 = bp['t0'][keep].astype(float); t1 = bp['t1'][keep].astype(float); dur = t1 - t0
quals = bp['quality_idx'].astype('int64')[keep]
_gbl = [sounding_bass_pc(str(labels[i]), int(roots[i])) for i in range(len(roots))]
gb = np.array([(-1 if v is None else v % 12) for v in _gbl], dtype=np.int64)
argmax = nn24[:, :12].argmax(1)
proxy_bass = proxy['bass_pc'][keep].astype('int64')
inv = gb != roots
bass_chroma = nn24[:, :12]  # per-pc bass energy

# music-x-lab bass pc per row (all 100 songs)
musx_bass = np.full(len(gb), -1, int)
for i in range(1, 101):
    rid = f"RWC_P{i:03d}"
    iv = load_lab(MUSX_OUT / f"{rid}.lab")
    if not iv: continue
    idx = np.where(sid == "rwc_" + rid)[0]
    for gi in idx:
        pl = lab_at(iv, 0.5*(t0[gi]+t1[gi]))
        if pl is None: continue
        pr, pf, _ = parse_harte(pl)
        if pr is None: continue
        b = sounding_bass_pc(pl, pr)
        if b is not None: musx_bass[gi] = b % 12

# pooled test-fold head preds (first seed each row is in test); leak-free enough for agreement analysis
head_pred = np.full(len(gb), -1, int)
for seed in range(5):
    tr, va, te = song_split(sid, seed)
    mf = train_clf(nn24[tr], gb[tr], nn24[va], gb[va], 24, 12, hid=(128, 64), epochs=50)
    pf = predict_proba(mf, nn24).argmax(1)
    new = te & (head_pred < 0)
    head_pred[new] = pf[new]

# valid rows: GT defined, all 4 estimators defined, head predicted (i.e. row appeared in some test fold)
valid = (gb >= 0) & (proxy_bass >= 0) & (musx_bass >= 0) & (head_pred >= 0)
print(f"valid rows (all 4 defined, GT defined): {valid.sum()}")

# 4-way consensus
consensus_val = argmax.copy()  # placeholder
agree4 = valid & (argmax == head_pred) & (argmax == proxy_bass) & (argmax == musx_bass)
cons_wrong = agree4 & (argmax != gb)
cons_right = agree4 & (argmax == gb)
print(f"\n4-way agreement rows: {agree4.sum()}  ({agree4.sum()/valid.sum():.1%} of valid)")
print(f"   4 agree & CORRECT: {cons_right.sum()}  ({cons_right.sum()/agree4.sum():.1%})  <- when all 4 agree they're usually right")
print(f"   4 agree & WRONG (consensus-vs-GT): {cons_wrong.sum()}  <- GT-error candidates")

# relaxed 3-way among INDEPENDENT signals: NNLS(argmax) + pyin + musx (drop head, correlated w/ argmax)
agree3 = valid & (argmax == proxy_bass) & (argmax == musx_bass)
cons3_wrong = agree3 & (argmax != gb)
print(f"\n3-way independent {{argmax,pyin,musx}} agree: {agree3.sum()};  agree & WRONG: {cons3_wrong.sum()}")

# For each consensus-wrong row: bass chroma evidence
def chroma_evidence(i, pred):
    bc = bass_chroma[i]
    order = np.argsort(bc)[::-1]
    g = gb[i]; e_pred = bc[pred]; e_gt = bc[g]
    rank_gt = int(np.where(order == g)[0][0]) + 1
    rank_pred = int(np.where(order == pred)[0][0]) + 1
    return e_pred, e_gt, rank_pred, rank_gt, bc

rows = np.where(cons_wrong)[0]
print(f"\n{'='*90}\nALL 4-WAY CONSENSUS-vs-GT ROWS (n={len(rows)})\n{'='*90}")
records = []
interval_ct = {}
for i in rows:
    pred = int(argmax[i])
    e_pred, e_gt, rank_pred, rank_gt, bc = chroma_evidence(i, pred)
    ivl = (pred - gb[i]) % 12
    interval_ct[ivl] = interval_ct.get(ivl, 0) + 1
    rec = dict(song=str(sid[i]).replace('rwc_',''), t0=round(float(t0[i]),1), t1=round(float(t1[i]),1),
               dur=round(float(dur[i]),1), label=str(labels[i]), gt_bass=PC[gb[i]],
               consensus=PC[pred], qual=QUAL[quals[i]], is_inv=bool(inv[i]),
               interval=ivl, e_pred=round(float(e_pred),2), e_gt=round(float(e_gt),2),
               rank_pred=rank_pred, rank_gt=rank_gt)
    records.append(rec)

# sort by strength of chroma evidence for GT-error: high e_pred, low e_gt, pred rank 1
records.sort(key=lambda r: (r['rank_gt'], -(r['e_pred']-r['e_gt'])))
for r in records:
    flag = ''
    if r['rank_pred'] == 1 and r['e_pred'] > r['e_gt'] + 0.15:
        flag = ' <<< bass energy strongly backs consensus (GT-WRONG candidate)'
    elif r['e_gt'] >= r['e_pred']:
        flag = ' (GT pc has >= bass energy: GT plausibly correct/ambiguous)'
    print(f"{r['song']:9s} {r['t0']:6.1f}-{r['t1']:5.1f}s dur={r['dur']:4.1f} "
          f"lab={r['label']:10s} GT_bass={r['gt_bass']:2s}(E={r['e_gt']:.2f},rk{r['rank_gt']}) "
          f"consensus={r['consensus']:2s}(E={r['e_pred']:.2f},rk{r['rank_pred']}) "
          f"iv=+{r['interval']:2d} {'INV' if r['is_inv'] else 'root'}{flag}")

print(f"\n-- interval of consensus vs GT (pred-gt mod12) --")
for k in sorted(interval_ct):
    print(f"   +{k:2d} semitones: {interval_ct[k]}  ({PC[k]} rel)")

# auto-tally by chroma evidence
gt_wrong = sum(1 for r in records if r['rank_pred'] == 1 and r['e_pred'] > r['e_gt'] + 0.15)
gt_ok = sum(1 for r in records if r['e_gt'] >= r['e_pred'])
incon = len(records) - gt_wrong - gt_ok
print(f"\nAUTO-TALLY (chroma-based heuristic, n={len(records)}):")
print(f"   GT-WRONG candidate (consensus dominates bass chroma): {gt_wrong}")
print(f"   GT-plausible (GT pc >= consensus bass energy):        {gt_ok}")
print(f"   inconclusive (consensus higher but not decisive):      {incon}")

def _clean(o):
    if isinstance(o, dict): return {k:_clean(v) for k,v in o.items()}
    if isinstance(o, (list, tuple)): return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)): return int(o)
    if isinstance(o, (np.floating,)): return float(o)
    if isinstance(o, (np.bool_,)): return bool(o)
    return o
json.dump(_clean(records), open(SCRATCH / 'bass_gt_noise_records.json','w'), indent=1)
print(f"\nsaved {len(records)} records -> {SCRATCH/'bass_gt_noise_records.json'}")
