"""harmonia/sections — les briques de CALCUL des sections, pas de détecteur.

Il n'y a plus de détection automatique de sections : le chart sort avec la
forme du brut (`pipeline._one_section`) et les sections viennent de l'outil
d'annotation. Le dernier détecteur, SongFormer, est supprimé depuis le
2026-09-30 (`git log -- harmonia/sections/songformer.py`).

Ce qui vit ici :
  * `similarity` — la matrice de ressemblance chord-tone partagée (outil
    sections, Soudure, `/ssm`, `folding`) ;
  * `simulation` — la recherche des reprises d'une brique tracée à la main ;
  * `nouveaute` — des coupures proposées pour les trous non tracés.
"""
