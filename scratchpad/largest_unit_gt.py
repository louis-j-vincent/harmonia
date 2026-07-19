"""Flag 2 concept test on GT (decode-free): find the LARGEST repeating unit L
(bar-multiples 4/8/16) via lag-recurrence, segment at L-boundaries, cluster
L-blocks by ORDERED content into section letters. Test on Let It Be + Billie Jean.
"""
import sys; sys.path.insert(0, "/Users/vincente/Documents/Projets Perso/Code/harmonia")
from pathlib import Path
import numpy as np
from harmonia.data.ireal_corpus import load_playlist, sectionized_measures, split_chords
_L={'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}
NOTE=['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
def rpc(t):
    t=t.strip()
    if not t or t[0] not in _L: return None
    pc=_L[t[0]]; i=1
    while i<len(t) and t[i] in '#b': pc+=1 if t[i]=='#' else -1; i+=1
    return pc%12
def bar_roots(t):
    out=[]
    for _l,meas in sectionized_measures(t):
        r=None
        for tok in split_chords(meas):
            s=tok.strip()
            if s and s[0] not in 'npW': r=rpc(s)
            if r is not None: break
        out.append(r if r is not None else -1)
    return out

def largest_unit_sections(R, cands=(16,8), rec_min=0.55, match=0.7):
    m=len(R)
    # largest L (bar multiple) whose L-shifted recurrence is strong
    bestL=None
    for L in cands:
        if 2*L>m: continue
        rec=np.mean([R[b]==R[b-L] for b in range(L,m)])
        if rec>=rec_min:
            bestL=L; break
    if bestL is None:
        # fall back to a single block
        return None, [(0,m)]
    L=bestL
    # segment into L-blocks (last partial block kept)
    blocks=[(i,min(i+L,m)) for i in range(0,m,L)]
    # cluster by ordered content (sequence near-equality)
    def sig(bl): return tuple(R[bl[0]:bl[1]])
    def near(a,b):
        if len(a)!=len(b): return False
        if not a: return True
        return sum(1 for x,y in zip(a,b) if x==y)/len(a)>=match
    clusters=[]  # (repr_sig, letter, members)
    labels=[]
    for bl in blocks:
        s=sig(bl); c=next((c for c in clusters if near(c[0],s)),None)
        if c is None: c=[s,len(clusters),[]]; clusters.append(c)
        c[2].append(bl); labels.append(c[1])
    return L, list(zip(blocks,labels))

POP={t.title.lower():t for t in load_playlist(Path('data/ireal/pop400.txt'))}
JAZZ={t.title.lower():t for t in load_playlist(Path('data/ireal/jazz1460.txt'))}
for title,corp in [('let it be',POP),('billie jean',POP),("bein' green",JAZZ)]:
    t=next((v for k,v in corp.items() if title in k),None)
    if not t: print(title,'not found'); continue
    R=bar_roots(t); m=len(R)
    L,segs=largest_unit_sections(R)
    seq=''.join(NOTE[r][0] if r>=0 else '.' for r in R)
    print(f'\n{t.title}: {m} bars, seq={seq[:56]}')
    print(f'  largest unit L={L}')
    # collapse consecutive same-letter into A xN
    from itertools import groupby
    letters=[lab for _,lab in segs]
    runs=[(k,len(list(g))) for k,g in groupby(letters)]
    print(f'  section letters (blocks): {[chr(65+l) for l in letters]}')
    print(f'  collapsed: {[(chr(65+k)+" x"+str(n)) for k,n in runs]}')
