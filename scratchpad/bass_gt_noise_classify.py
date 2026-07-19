"""Follow-up: musically classify the 54 four-way-consensus-vs-GT rows, and quantify
how much removing consensus-GT-errors moves the oracle ceiling. Key skeptic check:
the 5th-in-bass cases could be a harmonic confound (3rd harmonic of the root folds
onto the 5th in bass chroma) -- but pYIN (an f0 tracker, not a chroma-peak) and musx
(trained bass slot) BOTH agreeing on the 5th is evidence of a real played note, not
an overtone. We separate 3rd-in-bass / 5th-in-bass / other-chord-tone / non-chord.
"""
from __future__ import annotations
import sys
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

def load_lab(p):
    out=[]
    if not p.exists(): return out
    for line in p.read_text().splitlines():
        q=line.split()
        if len(q)<3: continue
        try: out.append((float(q[0]),float(q[1]),q[2]))
        except ValueError: pass
    return out
def lab_at(iv,t):
    for a,b,l in iv:
        if a<=t<b: return l
    return None

nn=load_corpus(REPO/'data/cache/rwc/rwc_nnls24.npz'); bp=load_corpus(REPO/'data/cache/rwc/rwc_bp48_fixed.npz')
proxy=np.load(REPO/'scratchpad/pyin_bass_cache.npz',allow_pickle=True)
nn24=nn['nnls24'].astype('float32'); keep=np.abs(nn24).sum(1)>0
nn24=nn24[keep]; roots=bp['root'].astype('int64')[keep]%12
sid=bp['song_id'][keep]; labels=bp['labels'][keep]
t0=bp['t0'][keep].astype(float); t1=bp['t1'][keep].astype(float)
_gbl=[sounding_bass_pc(str(labels[i]),int(roots[i])) for i in range(len(roots))]
gb=np.array([(-1 if v is None else v%12) for v in _gbl],dtype=np.int64)
argmax=nn24[:,:12].argmax(1); proxy_bass=proxy['bass_pc'][keep].astype('int64')

musx_bass=np.full(len(gb),-1,int)
for i in range(1,101):
    rid=f"RWC_P{i:03d}"; iv=load_lab(MUSX_OUT/f"{rid}.lab")
    if not iv: continue
    for gi in np.where(sid=="rwc_"+rid)[0]:
        pl=lab_at(iv,0.5*(t0[gi]+t1[gi]))
        if pl is None: continue
        pr,pf,_=parse_harte(pl)
        if pr is None: continue
        b=sounding_bass_pc(pl,pr)
        if b is not None: musx_bass[gi]=b%12

head_pred=np.full(len(gb),-1,int)
for seed in range(5):
    tr,va,te=song_split(sid,seed)
    mf=train_clf(nn24[tr],gb[tr],nn24[va],gb[va],24,12,hid=(128,64),epochs=50)
    pf=predict_proba(mf,nn24).argmax(1); new=te&(head_pred<0); head_pred[new]=pf[new]

valid=(gb>=0)&(proxy_bass>=0)&(musx_bass>=0)&(head_pred>=0)
agree4=valid&(argmax==head_pred)&(argmax==proxy_bass)&(argmax==musx_bass)
cons_wrong=agree4&(argmax!=gb)

# --- musical classification: relation of consensus pc to the LABELED chord root ---
rows=np.where(cons_wrong)[0]
cats={'5th-in-bass':0,'3rd-in-bass':0,'b7-in-bass':0,'other-chord-tone':0,'non-chord-tone':0,'label-already-inv':0}
detail=[]
for i in rows:
    r=int(roots[i]); c=int(argmax[i]); ivl=(c-r)%12  # consensus relative to labeled ROOT (not gt bass)
    lab=str(labels[i]); is_labinv = '/' in lab
    # relation to root
    if ivl==7: cat='5th-in-bass'
    elif ivl==4 or ivl==3: cat='3rd-in-bass'
    elif ivl==10 or ivl==11: cat='b7-in-bass'
    elif ivl in (0,2,5,9): cat='other-chord-tone'  # incl root, 9th, 4th, 6th (context)
    else: cat='non-chord-tone'
    cats[cat]+=1
    if is_labinv: cats['label-already-inv']+=1
    detail.append((PC[r],lab,PC[gb[i]],PC[c],ivl,is_labinv,cat))
print("Consensus PC relative to the LABELED CHORD ROOT (n=%d):"%len(rows))
for k,v in cats.items():
    print(f"   {k:20s}: {v}")

# per-song concentration
from collections import Counter
songct=Counter(str(sid[i]).replace('rwc_','') for i in rows)
print("\nper-song counts (concentration => per-song systematic issue, not random noise):")
for s,c in songct.most_common(): print(f"   {s}: {c}")

# --- impact on oracle ceiling ---
# oracle over the 4 estimators (+head) on valid rows; then treat consensus-wrong as GT-error (model correct)
ests=[argmax,head_pred,proxy_bass,musx_bass]
oracle_correct=np.zeros(valid.sum(),bool)
vidx=np.where(valid)[0]
for e in ests: oracle_correct|=(e[vidx]==gb[vidx])
base=oracle_correct.mean()
# consensus-wrong rows are oracle-FAILS by definition (all agree on !=gt). Reclassify as correct:
cw_in_valid=cons_wrong[vidx]
adj=(oracle_correct|cw_in_valid).mean()
print(f"\nORACLE ceiling on valid rows: {base:.4f}")
print(f"   after treating {cw_in_valid.sum()} consensus-vs-GT rows as GT-errors (model right): {adj:.4f}")
print(f"   => consensus-GT-errors alone close {(adj-base)*100:.2f}pp of the ceiling gap")

# how many oracle-fails are consensus (all agree) vs scatter?
ofail=~oracle_correct
print(f"\noracle-fail rows on valid: {ofail.sum()}  ({ofail.mean():.1%})")
print(f"   of which 4-way-consensus (strong GT-error candidates): {cw_in_valid.sum()} ({cw_in_valid.sum()/max(ofail.sum(),1):.1%})")
print(f"   of which estimators SCATTER (genuinely hard): {ofail.sum()-cw_in_valid.sum()} ({(ofail.sum()-cw_in_valid.sum())/max(ofail.sum(),1):.1%})")
