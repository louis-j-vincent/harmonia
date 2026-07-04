"""Check 1: is the emission matrix's own template geometry the binding constraint?

Two questions, no audio needed:
(a) Row-to-row cosine similarity of build_emission_matrix() rows — which
    (same-root, different-quality) pairs are nearly indistinguishable?
(b) Sharper: for an IDEAL observation of chord X (the template itself as the
    observation), what is the margin between X's own emission score and the
    best competitor's? If the margin is ~0 or negative for pairs like
    maj vs maj7, no audio improvement can ever separate them.
"""
import numpy as np
from harmonia.models.chord_hmm import build_emission_matrix
from harmonia.theory.chord_vocabulary import build_index

E = build_emission_matrix(max_phase=1)          # (181, 88), rows L1-normalized
idx_to_chord, _ = build_index(1)
C = E.shape[0]

# (a) cosine similarity, same-root pairs (root 0, all 15 qualities)
En = E / np.linalg.norm(E, axis=1, keepdims=True)
S = En @ En.T

quals = [q for (r, q) in idx_to_chord if r == 0]
idxs = [i for i, (r, q) in enumerate(idx_to_chord) if r == 0]
print("=== (a) same-root cosine similarity, top confusable pairs (root C) ===")
pairs = []
for a in range(len(idxs)):
    for b in range(a + 1, len(idxs)):
        pairs.append((S[idxs[a], idxs[b]], quals[a].value, quals[b].value))
pairs.sort(reverse=True)
for s, qa, qb in pairs[:15]:
    print(f"  {qa:8s} vs {qb:8s}  cos={s:.4f}")

# cross-root: most similar different-root pair overall
print("\n=== (a2) most similar cross-root pairs ===")
best = []
for i in range(C):
    for j in range(i + 1, C):
        ri, qi = idx_to_chord[i]
        rj, qj = idx_to_chord[j]
        if ri == rj or ri < 0 or rj < 0:
            continue
        best.append((S[i, j], f"{ri}:{qi.value}", f"{rj}:{qj.value}"))
best.sort(reverse=True)
for s, a, b in best[:10]:
    print(f"  {a:10s} vs {b:10s}  cos={s:.4f}")

# (b) ideal-observation margin test: observation = chord's own emission row
print("\n=== (b) ideal-observation identification: obs = own template row ===")
scores = E @ E.T   # scores[x, c] = emission score of chord c given ideal obs of x
wrong = 0
zero_margin = []
for x in range(C):
    rx, qx = idx_to_chord[x]
    if rx < 0:
        continue
    order = np.argsort(scores[x])[::-1]
    top = order[0]
    margin = (scores[x, x] - max(scores[x, c] for c in range(C) if c != x)) / scores[x, x]
    runner = max((c for c in range(C) if c != x), key=lambda c: scores[x, c])
    rr, qr = idx_to_chord[runner]
    if top != x:
        wrong += 1
        rt, qt = idx_to_chord[top]
        print(f"  MISID: ideal {rx}:{qx.value} -> argmax {rt}:{qt.value} (margin {margin:+.4f})")
    elif margin < 0.02:
        zero_margin.append((margin, f"{rx}:{qx.value}", f"{rr}:{qr.value}"))
print(f"  misidentified under IDEAL observation: {wrong}/180")
zero_margin.sort()
print("  thinnest margins (<2%), own vs runner-up:")
for m, a, b in zero_margin[:15]:
    print(f"    {a:10s} runner-up {b:10s} margin {m:+.4%}")
