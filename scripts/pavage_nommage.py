"""Le pavage donne les FRONTIERES. Est-ce qu'il nomme bien ?

L'affectation dure « le motif le plus proche gagne » jette tout ce qui n'est
pas l'argmax. On regarde si le PROFIL complet (la ressemblance a TOUS les
motifs) separe ce que l'argmax confond.
"""
import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-Code-harmonia/f1d059d0-fd31-4a5f-bb74-c856f6498a5d/scratchpad')
import numpy as np
from pavage import X, n, FRONT, pave, seg

def sim(M, b, L=4):
    a, v = M.ravel(), seg(b, L)
    if a.sum() <= 0 or v.sum() <= 0: return 0.0
    return float(a @ v / max(np.linalg.norm(a)*np.linalg.norm(v), 1e-9))

def apprends(k=6, q=0.03, seed=4, iters=15):
    rng = np.random.default_rng(seed)
    motifs = [X[d:d+4].copy() for d in rng.choice(np.arange(0, n-4), size=k, replace=False)]
    lon = [4]*k
    for _ in range(iters):
        t = pave(motifs, lon, q=q)[0]
        for j in range(k):
            pris = [X[a:a+4] for a, _b, jj in t if jj == j and a+4 <= n]
            if pris: motifs[j] = np.mean(pris, axis=0)
    return pave(motifs, lon, q=q)[0], motifs

t, M = apprends()
tuiles = [(a, b, j) for a, b, j in t if j is not None]
VRAI = {4:"couplet",8:"couplet",12:"refrain",16:"refrain",20:"tag",
        26:"couplet",30:"couplet",34:"montee",38:"montee",
        42:"pont",46:"pont",50:"refrain",54:"refrain",58:"tag"}

print("tuile        argmax   profil complet (ressemblance a chaque motif)     ce que c'est")
prof = {}
for a, b, j in tuiles:
    p = np.array([sim(M[x], a) for x in range(len(M))])
    prof[a] = p
    nom = VRAI.get(a, "?")
    print(f"mes.{a+1:3}-{b:<3}  motif {j+1}   " +
          " ".join(f"{v:.2f}" for v in p) + f"   {nom}")

print("\nCE QUE L'ARGMAX CONFOND :")
from collections import defaultdict
g = defaultdict(list)
for a, b, j in tuiles: g[j].append(VRAI.get(a, "?"))
for j, noms in sorted(g.items()):
    if len(set(noms)) > 1:
        print(f"   motif {j+1} recoit : {sorted(set(noms))}  <-- musiques differentes, meme lettre")

print("\nLE PROFIL COMPLET LES SEPARE-T-IL ?")
for j, noms in sorted(g.items()):
    if len(set(noms)) <= 1: continue
    par = defaultdict(list)
    for a, b, jj in tuiles:
        if jj == j: par[VRAI.get(a, "?")].append(prof[a])
    cles = sorted(par)
    for i in range(len(cles)):
        for k2 in range(i+1, len(cles)):
            A = np.mean(par[cles[i]], axis=0); B = np.mean(par[cles[k2]], axis=0)
            c = float(A @ B / max(np.linalg.norm(A)*np.linalg.norm(B), 1e-9))
            print(f"   motif {j+1} : « {cles[i]} » contre « {cles[k2]} »  "
                  f"profils a {c:.3f} l'un de l'autre "
                  f"{'-> SEPARABLES' if c < 0.99 else '-> identiques, meme le profil ne voit rien'}")
