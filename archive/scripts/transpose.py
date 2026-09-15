"""L'escalier : une reprise transposée reste une reprise.

Louis, 2026-08-07 :

  « Sur Sunny on ne fait que monter d'un demi-ton à chaque fois, donc
    essentiellement c'est les mêmes sections qui se répètent mais avec modulation
    à chaque fois. Il faudrait un cas spécial pour ça — répétition avec
    modulation : si le pattern se répète en décalant tout de la modulation, c'est
    confirmé. » Puis : « transposition sur la voix aussi du coup », et « garde
    l'escalier, il est très important ».

POURQUOI LA MATRICE NE PEUT PAS LE VOIR. Chaque mesure est un vecteur de douze
classes de hauteur. Transposer d'un demi-ton, c'est faire TOURNER ce vecteur
d'un cran ; le cosinus entre l'original et sa version tournée s'effondre. Deux
passages musicalement identiques deviennent donc étrangers l'un à l'autre.
Mesuré sur Sunny, le bloc de la mesure 5 contre ses reprises :

    mesures 17, 21    0.967  0.943   (même ton)
    mesures 33, 37    0.158  0.159   <- invisibles
    mesures 49, 53    0.099  0.126   <- invisibles
    mesures 65, 69    0.454  0.449

En autorisant la rotation, les mêmes reprises donnent 0.93, 0.95, 0.94, 0.97,
0.92, 0.94 — et le décalage gagnant monte de un demi-ton à chaque fois. L'escalier
est là, il suffisait de le chercher.

LES DEUX GARDE-FOUS, et ils sont nécessaires. Douze rotations, c'est douze fois
plus d'occasions de correspondre par hasard : dans le tableau ci-dessus, la
mesure 29, qui n'est pas une reprise, passe de 0.256 à 0.628.

  1. **Un seul demi-ton pour tout le bloc**, c'est la règle de Louis. Une
     coïncidence n'a aucune raison de tenir sur huit mesures d'affilée.
  2. **Le même demi-ton pour l'harmonie ET pour le chant.** Un chanteur qui
     module monte avec l'orchestre. Un faux positif devrait donc mentir sur les
     deux voies en même temps, avec la même valeur.

Un troisième garde-fou est laissé à l'appelant : `MARGIN`, la prime qu'un accord
transposé doit rendre par rapport à l'accord non transposé pour être préféré. À
zéro on prend le meilleur des douze ; au-dessus, on ne quitte le ton d'origine
que si ça vaut vraiment le coup.
"""
from __future__ import annotations

import numpy as np

MARGIN = 0.05      # ce qu'une transposition doit gagner pour valoir le détour
W_SAME = 0.50      # dans le ton d'origine : les deux voies à égalité
W_TRANS = 0.80     # EN TRANSPOSITION : l'harmonie mène. Louis, 2026-08-07, a
                   # validé la raison — « donner plus de poids à l'harmonie
                   # quand une transposition est en jeu, bonne idée ». Une
                   # modulation est d'abord un fait harmonique, et le chant y
                   # est un témoin bien plus faible : sur Sunny les reprises
                   # transposées scorent 0.93–0.97 sur l'harmonie seule et
                   # retombent à 0.57–0.76 dès qu'on moyenne à parts égales.
                   # Le chant ne disparaît pas — il lui reste un cinquième, donc
                   # il peut encore contredire une transposition inventée.


def rot_stack(V):
    """S[k, i, j] = ressemblance de la mesure i avec la mesure j transposée de k."""
    V = np.asarray(V, float)
    return np.stack([V @ np.roll(V, k, axis=1).T for k in range(12)])


def slide(Sh, Sm, b0, L, n, w=W_SAME, w_trans=W_TRANS, margin=MARGIN, unit=1):
    """Le bloc [b0, b0+L) glissé, avec droit de transposer — un seul k par bloc.

    Renvoie trois tableaux de longueur n : le score combiné, le score de chaque
    voie confondu (pour l'affichage), et le demi-ton retenu à chaque position.
    Le demi-ton est signé dans [-5, 6] : c'est la lecture musicale (monter de
    onze demi-tons, c'est descendre d'un).
    """
    score = np.zeros(n)
    semis = np.zeros(n, int)
    for c in range(0, n - L + 1, unit):
        best_k, best_v = 0, -1.0
        for k in range(12):
            h = float(np.mean([Sh[k, b0 + i, c + i] for i in range(L)]))
            m = float(np.mean([Sm[k, b0 + i, c + i] for i in range(L)]))
            wk = w if k == 0 else w_trans
            v = wk * h + (1 - wk) * m
            if k and v < best_v + margin:      # le ton d'origine garde la main
                continue
            if v > best_v:
                best_k, best_v = k, v
        score[c] = best_v
        semis[c] = best_k if best_k <= 6 else best_k - 12
    return score, semis


def staircase(semis, occ):
    """Les demi-tons aux endroits retenus — l'escalier, à afficher tel quel."""
    return [int(semis[p]) for p in occ]


if __name__ == "__main__":        # la vérification de la page, pas un test
    import os
    import sys
    from pathlib import Path
    HERE = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE / "scripts"))
    from pattern_lanes import load_vectors
    import vocal_anchor as VA, blocks8 as B8, melody_ssm as MS, vocal_melody as VM

    for st in (sys.argv[1:] or ["bobby_hebb_sunny_official_audio"]):
        Vh, S, n, grid = load_vectors(st)
        voc = VA.separate_vocals(HERE / f"docs/audio/{st}.m4a")
        tt, ff, vv, rr = VM.track_f0(voc)
        notes, _ = VM.melody_notes(tt, ff, vv, rr)
        notes, _ = VM.clean(notes)
        Vm, mute = MS.melody_vectors(notes, grid, n)
        onset, *_ = B8.sing_onset(voc)
        b0 = MS.voice_start(VA.bar_of(grid, onset), n) or 0
        Sh, Sm = rot_stack(Vh), rot_stack(Vm)
        sc, se = slide(Sh, Sm, b0, 8, n, unit=2)
        sc0, _ = slide(Sh, Sm[0:1].repeat(12, 0) * 0 + Sm, b0, 8, n, unit=2, margin=9)
        print(f"\n{st} · ancre mesure {b0+1}")
        print("  mesure   sans   avec   demi-tons")
        for c in range(b0 + 8, n - 7, 4):
            print(f"   {c+1:>4}   {sc0[c]:.3f}  {sc[c]:.3f}   {se[c]:+d}"
                  + ("   <<<" if sc[c] > .85 else ""))
