"""Le VRAI pavage coute-t-il moins cher que celui qu'on trouve ?

Si oui  -> on rate l'optimum, c'est un probleme d'optimisation.
Si non  -> l'harmonie PREFERE le decalage, et aucun reglage ne sauvera ca.
"""
import sys
sys.path.insert(0, '/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-Code-harmonia/f1d059d0-fd31-4a5f-bb74-c856f6498a5d/scratchpad')
import numpy as np
from pavage import X, n, FRONT

# la forme VRAIE, en tuiles de 4 (0-indexe) : intro 0-3, puis les sections
VRAI = [(0,4), (4,8), (8,12), (12,16), (16,20), (20,24),
        (26,30), (30,34), (34,38), (38,42), (42,46), (46,50),
        (50,54), (54,58), (58,62)]
TROUVE = [(3,7), (7,11), (11,15), (15,19), (19,23), (25,29), (29,33),
          (34,38), (38,42), (42,46), (46,50), (51,55), (55,59), (59,63)]

def cellule(a, L=4):
    z = min(a+L, n); v = X[a:z].ravel()
    if z-a < L: v = np.concatenate([v, np.zeros((L-(z-a))*12)])
    return v

def cout_pavage(tuiles, k=6, iters=20):
    """k-means sur CES tuiles-la : le meilleur cout atteignable avec ce
    decoupage. C'est la comparaison honnete -- chaque decoupage a droit a
    ses meilleurs motifs."""
    V = np.array([cellule(a) for a, _b in tuiles])
    rng = np.random.default_rng(0)
    C = V[rng.choice(len(V), size=min(k, len(V)), replace=False)]
    for _ in range(iters):
        sim = V @ C.T / np.maximum(np.linalg.norm(V,axis=1)[:,None]*np.linalg.norm(C,axis=1)[None,:], 1e-9)
        aff = np.argmax(sim, axis=1)
        for j in range(len(C)):
            m = aff == j
            if m.any(): C[j] = V[m].mean(0)
    sim = V @ C.T / np.maximum(np.linalg.norm(V,axis=1)[:,None]*np.linalg.norm(C,axis=1)[None,:], 1e-9)
    return float(np.mean(1 - sim.max(axis=1)))

cv, ct = cout_pavage(VRAI), cout_pavage(TROUVE)
print(f"cout du VRAI decoupage    : {cv:.4f}  ({len(VRAI)} tuiles)")
print(f"cout de celui TROUVE      : {ct:.4f}  ({len(TROUVE)} tuiles)")
print(f"-> {'le vrai est MEILLEUR : probleme d optimisation' if cv < ct else 'le vrai est PIRE : l harmonie prefere le decalage'}")

print("\nPourquoi : l'intro et le couplet ont-ils les memes accords ?")
for d in range(4):
    print(f"   mesure {d+1} de l'intro vs mesure {d+1} du couplet : "
          f"{float(X[d] @ X[d+4] / max(np.linalg.norm(X[d])*np.linalg.norm(X[d+4]),1e-9)):.3f}")
print("\nUne boucle de 4 mesures a QUATRE rotations. L'harmonie seule ne peut")
print("pas dire laquelle est la bonne -- elle donne la periode, pas la phase.")
