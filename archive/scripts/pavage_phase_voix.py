"""La VOIX pose la phase que l'harmonie ne peut pas poser.

Le pavage garde son cout harmonique, plus un bonus quand une section DEMARRE
la ou la voix repart apres un silence. C'est le seul signal mesure independant
de l'harmonie (docs/voix_indice.md, branche feat/voix : 0,50 de rappel sur les
frontieres que l'harmonie ne voit pas, contre 0,07 pour l'harmonie).
"""
import sys, pickle
sys.path.insert(0, '/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-Code-harmonia/f1d059d0-fd31-4a5f-bb74-c856f6498a5d/scratchpad')
import numpy as np
from pavage import X, n, FRONT, seg

P = pickle.load(open('/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-Code-harmonia/f1d059d0-fd31-4a5f-bb74-c856f6498a5d/scratchpad/easy.pkl','rb'))
chante = P['chante'][:n]
pas = np.median(np.diff(P['grid']))

# depart de phrase : la voix se tait (ou presque) avant, et repart ici
depart = np.zeros(n)
for b in range(1, n):
    if chante[b] > 0.25*pas and chante[b-1] <= 0.40*pas:
        depart[b] = 1.0
depart[0] = 1.0
print("mesures ou la voix repart :", [b+1 for b in np.where(depart > 0)[0]])
print("vraies frontieres          :", [f+1 for f in FRONT])

def cout(M, b, L):
    a, v = M.ravel(), seg(b, L)
    if a.sum() <= 0 or v.sum() <= 0: return 1.0
    return 1.0 - float(a @ v / max(np.linalg.norm(a)*np.linalg.norm(v), 1e-9))

def pave(motifs, lon, q, lam):
    INF = 1e9; dp = np.full(n+1, INF); dp[0] = 0.0; back = [None]*(n+1)
    for b in range(n):
        if dp[b] >= INF: continue
        if dp[b] + q < dp[b+1]: dp[b+1] = dp[b] + q; back[b+1] = (b, None)
        for j, (M, L) in enumerate(zip(motifs, lon)):
            z = min(b+L, n)
            c = dp[b] + cout(M, b, L) - lam*depart[b]      # <- le bonus de phase
            if c < dp[z]: dp[z] = c; back[z] = (b, j)
    out, b = [], n
    while b > 0:
        p, j = back[b]; out.append((p, b, j)); b = p
    return list(reversed(out))

def fus(t):
    o = []
    for a, b, j in t:
        if o and o[-1][2] == j and o[-1][1] == a and j is not None: o[-1] = (o[-1][0], b, j)
        else: o.append([a, b, j])
    return o

def essai(k, q, lam, seed, iters=15):
    rng = np.random.default_rng(seed)
    motifs = [X[d:d+4].copy() for d in rng.choice(np.arange(0, n-4), size=k, replace=False)]
    lon = [4]*k
    for _ in range(iters):
        t = pave(motifs, lon, q, lam)
        for j in range(k):
            pris = [X[a:a+4] for a, _b, jj in t if jj == j and a+4 <= n]
            if pris: motifs[j] = np.mean(pris, axis=0)
    f = [(a,b,j) for a,b,j in fus(pave(motifs, lon, q, lam)) if j is not None]
    ok = sum(1 for a,_,_ in f if any(abs(a-x) <= 1 for x in FRONT))
    exact = sum(1 for a,_,_ in f if a in FRONT)
    okf = sum(1 for x in FRONT if any(abs(a-x) <= 1 for a,_,_ in f))
    return f, ok/max(len(f),1), exact/max(len(f),1), okf/len(FRONT)

print(f"\n{'lambda':>8}{'sections':>10}{'a +-1':>8}{'EXACT':>8}{'rappel':>8}")
for lam in (0.0, 0.05, 0.15, 0.3, 0.6, 1.0):
    best = None
    for k in (4, 5, 6):
        for q in (0.03, 0.06, 0.12):
            for s in range(3):
                f, p, e, r = essai(k, q, lam, s)
                sc = e + 0.3*r
                if best is None or sc > best[0]: best = (sc, f, p, e, r)
    _sc, f, p, e, r = best
    print(f"{lam:>8.2f}{len(f):>10}{p:>8.2f}{e:>8.2f}{r:>8.2f}")
    if lam == 0.3:
        for a, b, j in f:
            print(f"      mes.{a+1:3}-{b:<3} " + ("<-- EXACT" if a in FRONT else ""))
