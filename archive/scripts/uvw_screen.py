"""X ~= U V W : trois etages, grain bi-mesure, et V decide de la LIAISON.

  X  (n_bi x 24)   une bi-mesure = 2 mesures d'accords mis bout a bout
  W  (m x 24)      l'alphabet des bi-mesures        <- « le mot » de la soudure
  H  (n_bi x m)    quelle bi-mesure est quel motif
  P  (n_bi-1 x 2m) TOUTES les paires adjacentes, encodees par fente
  V  (k x 2m)      la grammaire : quelles bi-mesures se soudent en 4 mesures
  U  (n_bi-1 x k)  ou chaque soudure se joue

Le test qui compte : la liaison (donc le DECOUPAGE) sort-elle du calcul, ou
faut-il encore la donner ? On part d'une grille de 2 mesures RIGIDE depuis la
mesure 0 -- aucune couture fournie.
"""
import json, sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import NMF
HERE = Path('/Users/vincente/Documents/Projets Perso/Code/harmonia')
sys.path.insert(0, str(HERE))
from harmonia_min import harmonic_sections as HS, musx as MX

chart = json.load(open(HERE/'harmonia_min/state/charts/min_X-yIEMduRXk.json'))
grid, n = chart['barGrid'], chart['nBars']
X1 = np.clip(HS.harmonic_vectors(MX.frame_posteriors(HERE/'docs/audio/X-yIEMduRXk.m4a')[0], grid), 0, None)

# --- bi-mesures : grille RIGIDE de 2 mesures, aucune couture donnee
bi = [(b, min(b+2, n)) for b in range(0, n-1, 2)]
X = np.array([np.concatenate([X1[a], X1[min(a+1, b-1)]]) for a, b in bi])
n_bi = len(X)
print(f"X : {X.shape}  ({n_bi} bi-mesures, grille rigide de 2 mesures)")

# --- etage 1 : l'alphabet des bi-mesures
m = 8
n1 = NMF(n_components=m, init='nndsvda', max_iter=800, random_state=0)
H = n1.fit_transform(X); W = n1.components_
Hn = H / np.clip(H.sum(1, keepdims=True), 1e-9, None)
print(f"W : {W.shape}   residu etage 1 : {np.linalg.norm(X - H@W)/np.linalg.norm(X):.3f}")

# --- etage 2 : toutes les paires adjacentes, encodage par FENTE (l'ordre compte)
P = np.array([np.concatenate([Hn[i], Hn[i+1]]) for i in range(n_bi-1)])
k = 6
n2 = NMF(n_components=k, init='nndsvda', max_iter=800, random_state=0)
U = n2.fit_transform(P); V = n2.components_
Un = U / np.clip(U.sum(1, keepdims=True), 1e-9, None)
lien = Un.max(1)                       # « ces deux bi-mesures forment-elles un bloc ? »
res_p = np.array([np.linalg.norm(P[i] - U[i]@V)/max(np.linalg.norm(P[i]),1e-9)
                  for i in range(len(P))])
print(f"V : {V.shape}   residu etage 2 : {np.linalg.norm(P - U@V)/np.linalg.norm(P):.3f}")

# --- la liaison : on prend les meilleures soudures sans chevauchement
force = lien * (1 - res_p)
pris, blocs = set(), []
for i in np.argsort(-force):
    if i in pris or (i+1) in pris: continue
    pris.add(i); pris.add(i+1)
    blocs.append((int(i), float(force[i])))
queues = [i for i in range(n_bi) if i not in pris]
blocs.sort()

print(f"\nLIAISON TROUVEE : {len(blocs)} blocs de 4 mesures, {len(queues)} queue(s)")
for i, f in blocs:
    print(f"   mes.{bi[i][0]+1:3}-{bi[i+1][1]:<3}  force {f:.3f}  lettre {int(np.argmax(Un[i]))}")
print("   QUEUES :", [f"mes.{bi[i][0]+1}-{bi[i][1]}" for i in queues])
print("\n   (la liaison de 2 mesures du morceau est mes. 25-26 ; l'accord final mes. 63)")
