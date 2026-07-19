import sys; sys.path.insert(0,'/Users/vincente/Documents/Projets Perso/Code/harmonia')
from pathlib import Path
from collections import Counter
import numpy as np, random
from harmonia.data.ireal_corpus import load_playlist, sectionized_measures, split_chords
from harmonia.models.chord_pipeline_v1 import occam_compress_bars, detect_loop_pattern, NOTE
_L={'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}
def rpc(t):
    t=t.strip()
    if not t or t[0] not in _L: return None
    pc=_L[t[0]]; i=1
    while i<len(t) and t[i] in '#b': pc+=1 if t[i]=='#' else -1; i+=1
    return pc%12
def barroots(t):
    out=[]
    for _l,meas in sectionized_measures(t):
        r=None
        for tok in split_chords(meas):
            s=tok.strip()
            if s and s[0] not in 'npW': r=rpc(s)
            if r is not None: break
        if r is not None: out.append(r)
    return out
tunes=load_playlist(Path('data/ireal/pop400.txt'))
applied=[]
for t in tunes:
    roots=barroots(t); m=len(roots)
    if m<8: continue
    post=np.full((m,12),1e-3)
    for i,r in enumerate(roots): post[i,r]=1.0
    loop=detect_loop_pattern(roots,post,list(range(m)))
    if loop: applied.append((t.title,loop[0],[NOTE[r] for r in loop[1]]))
print(f'compression rate: {len(applied)}/{sum(1 for t in tunes if len(barroots(t))>=8)} charts read as a loop')
random.seed(3)
print('--- 5 random loop reads (GT symbolic) ---')
for title,P,pat in random.sample(applied,5):
    print(f'  {title}: period {P} pattern {pat}')
from collections import Counter as C2
print('period distribution:', dict(sorted(C2(a[1] for a in applied).items())))
