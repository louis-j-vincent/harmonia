"""La règle de Louis : repartir de la PREMIERE mesure non couverte, et avancer.

    python scripts/anchored_dictionary.py

Louis, 2026-08-05, en rejetant la recherche actuelle (« pas du tout la bonne
idée !! ») : « Il faut repartir du premier temps pas couvert par les blocs et
leurs répétitions déjà dans le dictionnaire et faire le même exercice, et ainsi
de suite jusqu'à ce qu'on couvre toutes les sections. Et si on ne trouve rien de
récurrent on continue d'avancer jusqu'à ce qu'on trouve quelque chose qui le
soit. »

MESURÉ, et le résultat est en deux temps :

* **Sa règle trouve bien la cellule qu'on ratait.** Sur Grenade elle sort
  `Dm Bb | F C` avec 12 occurrences — exactement le deuxième motif qu'il
  pointait, que la recherche actuelle noyait dans un bloc de 18 mesures. Sur
  This Love elle sort `Cm Fm | Bb Eb` avec 15 occurrences.
* **Mais l'assemblage en sections casse.** This Love passe de 3 lettres à 4,
  The Walk de 2 à 1, Norah reste à 2. La raison n'est pas la règle : c'est que
  `sections_from` fabrique une section par chaîne d'occurrences consécutives
  d'UNE entrée. Ça marche quand une entrée EST une section ; ça ne marche plus
  quand une entrée est une CELLULE de 2 mesures répétée douze fois.

Conclusion : les deux étages doivent changer ensemble. Avec sa règle, une
section n'est plus la répétition d'une cellule — c'est la répétition d'une
SÉQUENCE de cellules. Ce script garde la mesure telle quelle pour que la
comparaison reste vérifiable.
"""

import sys, copy, numpy as np
sys.path[:0]=[".","scripts"]
import logging; logging.disable(logging.INFO)
from pathlib import Path
from harmonia_min import sections as hs, musx as mx
import harmonia_min.harmonic_sections as HS
NAMES="C Db D Eb E F Gb G Ab A Bb B".split()

def load(stem):
    from harmonia_min import pipeline as _pl
    real=hs.detect_sections; c={}
    def spy(g,a,t,bars=None,**k):
        c.update(grid=g,bars=copy.deepcopy(bars)); return real(g,a,t,bars,**k)
    hs.detect_sections=spy
    try: _pl.analyze(Path(f"docs/audio/{stem}.m4a"),title="x",file_key="x",audio_url="")
    finally: hs.detect_sections=real
    grid=c["grid"]; n=len(grid)-1
    V=HS.harmonic_vectors(mx.frame_posteriors(Path(f"docs/audio/{stem}.m4a"))[0],grid)
    return V@V.T, n, c["bars"]

def anchored_dictionary(S, n, max_entries=8):
    """Louis, 2026-08-05 : « repartir du premier temps pas couvert par les blocs
    et leurs répétitions déjà dans le dictionnaire et faire le même exercice, et
    ainsi de suite ; et si on ne trouve rien de récurrent on continue d'avancer
    jusqu'à ce qu'on trouve quelque chose qui le soit »."""
    off=HS.off_diagonal(S)
    strong=float(np.quantile(off,HS.PHASE_QUANTILE)); cont=float(np.quantile(off,HS.CONT_QUANTILE))
    entries=[]; claimed=np.zeros(n,bool); cursor=0
    while cursor < n and len(entries) < max_entries:
        if claimed[cursor]: cursor+=1; continue
        b0=cursor
        best=None                                   # (run, lag)
        for d in range(HS.LAG_MIN, min(HS.LAG_MAX, n-b0-1)+1):
            r=0
            while (b0+r+d < n and not claimed[b0+r]
                   and (S[b0+r,b0+r+d] >= strong or (r>0 and S[b0+r,b0+r+d] >= cont))):
                r+=1
            if r >= HS.LAG_MIN and (best is None or r > best[0]):
                best=(r,d)                          # tie-break: smallest d wins
        if best is None:
            cursor += 1                             # rien de récurrent ici, on avance
            continue
        run,lag=best
        L=int(min(run,lag))
        curve=HS.slide(S,L,b0)
        cand=sorted((int(o) for o in HS.peaks(curve,L,b0)), key=lambda o:-curve[o])
        occ,taken=[],claimed.copy()
        for o in cand:
            if taken[o:min(n,o+L)].any(): continue
            occ.append(o); taken[o:min(n,o+L)]=True
        occ.sort()
        if not occ:
            cursor += 1
            continue
        entries.append({"L":L,"b0":b0,"curve":curve,"occ":occ,"lag":lag,"run":run})
        claimed[b0:min(n,b0+L)]=True
        for o in occ: claimed[o:min(n,o+L)]=True
        cursor = b0+1
    return entries, claimed

TARGET={"mayer_hawthorne_the_walk":2,"norah_jones_don_t_know_why":2,"maroon_5_this_love":3}
STEMS=["bruno_mars_grenade_official_music_video","let_it_be_remastered_2009",
       "maroon_5_this_love","norah_jones_don_t_know_why","mayer_hawthorne_the_walk"]
for stem in STEMS:
    S,n,bars=load(stem)
    old,_=HS.build_dictionary(S,n)
    new,cl=anchored_dictionary(S,n)
    print(f"\n### {stem[:40]}  ({n} mesures)")
    print("  ACTUEL (plus longue suite libre) :")
    for i,e in enumerate(old):
        print(f"     e{i+1}: motif {e['L']:>2} mes, depart {e['b0']+1:>3}, "
              f"suite {e.get('run','?')} a distance {e.get('lag','?')}, occ {[o+1 for o in e['occ']]}")
    print("  TA REGLE (repartir de la 1re mesure libre) :")
    for i,e in enumerate(new):
        cell=" | ".join(" ".join("N.C." if x.get("nc") else NAMES[x["root"]]+x["q"] for x in bars[e['b0']+k]) or "-"
                        for k in range(min(e["L"],6)))
        print(f"     e{i+1}: motif {e['L']:>2} mes, depart {e['b0']+1:>3}, "
              f"suite {e['run']} a distance {e['lag']}, occ {[o+1 for o in e['occ']]}")
        print(f"          {cell}")
    print(f"     couverture : {int(cl.sum())}/{n} mesures")
    # sections + lettres
    for lbl,ent in (("ACTUEL",old),("TA REGLE",new)):
        secs=HS.sections_from(S,n,ent)
        k=len({s['letter'] for s in secs})
        tg=TARGET.get(stem)
        print(f"  -> {lbl:<9} {k} lettres" + (f" (cible {tg}) {'OK' if k==tg else 'RATE'}" if tg else "")
              + " : " + " ".join(f"{s['letter']}[{s['b0']+1}-{s['b1']+1}]" for s in secs))
