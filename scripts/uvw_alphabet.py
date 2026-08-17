"""L'etage du bas : l'alphabet DUR (soudure) contre l'alphabet MOU (NMF).

Question : deux bi-mesures que Louis met dans la MEME section recoivent-elles
la meme lettre ? Et deux bi-mesures de sections DIFFERENTES en recoivent-elles
des differentes ? C'est tout ce que l'etage du bas doit faire ; le reste est
l'affaire de l'etage de liaison.

Aucune pipeline complete ici : on mesure l'alphabet, pas le decoupage.
"""
import sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import NMF
HERE = Path('/Users/vincente/Documents/Projets Perso/Code/harmonia')
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE/'scripts'))
import section_bench as SB
from harmonia_min import voice_sections as VS

SEUIL_DUR = 0.93          # celui de soudure.grille_et_mot
M_MOU = 8

def bimesures(V12, n):
    return [(b, min(b+2, n)) for b in range(0, n-1, 2)]

def dur(S, bi):
    """L'alphabet par egalite : la 1re bi-mesure assez proche gagne."""
    lab, k = [], 0
    for i, (a, _b) in enumerate(bi):
        t = None
        for j in range(i):
            c, d = bi[j][0], a
            if float(np.mean([S[c, d], S[c+1, min(d+1, len(S)-1)]])) >= SEUIL_DUR:
                t = lab[j]; break
        if t is None: t = k; k += 1
        lab.append(t)
    return np.array(lab)

def mou(V12, bi, m=M_MOU):
    X = np.array([np.concatenate([V12[a], V12[min(a+1, b-1)]]) for a, b in bi])
    X = np.clip(X, 0, None)
    mm = min(m, len(X)-1)
    f = NMF(n_components=mm, init='nndsvda', max_iter=800, random_state=0)
    return np.argmax(f.fit_transform(X), axis=1)

T = SB.truth()
res = {"dur": [], "mou": []}
for st in sorted(T):
    F = SB.features(st)
    n = min(F["n"], T[st]["n"])
    sec = {}
    for s in T[st]["sections"]:
        for b in range(s["b0"], min(s["b1"], n-1)+1): sec[b] = s["label"]
    bi = bimesures(F["V"], n)
    bi = [(a, b) for a, b in bi if a in sec and b-1 in sec and sec[a] == sec[b-1]]
    if len(bi) < 6: continue
    vrai = np.array([sec[a] for a, _ in bi])
    for nom, lab in (("dur", dur(F["S"], bi)), ("mou", mou(F["V"], bi))):
        meme_sec, meme_let = [], []
        for i in range(len(bi)):
            for j in range(i+1, len(bi)):
                meme_sec.append(vrai[i] == vrai[j]); meme_let.append(lab[i] == lab[j])
        meme_sec = np.array(meme_sec); meme_let = np.array(meme_let)
        tp = float((meme_sec & meme_let).sum()); fp = float((~meme_sec & meme_let).sum())
        fn = float((meme_sec & ~meme_let).sum())
        p = tp/max(tp+fp,1); r = tp/max(tp+fn,1)
        res[nom].append((p, r, 0 if p+r==0 else 2*p*r/(p+r), len(set(lab))))

print(f"{'alphabet':10}{'precision':>11}{'rappel':>9}{'F1':>8}{'lettres':>9}   (18 morceaux, paires de bi-mesures)")
for nom in ("dur", "mou"):
    a = np.array([x[:3] for x in res[nom]]); L = np.mean([x[3] for x in res[nom]])
    print(f"{nom:10}{a[:,0].mean():>11.3f}{a[:,1].mean():>9.3f}{a[:,2].mean():>8.3f}{L:>9.1f}")
