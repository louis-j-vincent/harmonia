"""Pont vers `harmonia.folding` (refactor, sprint 7). Disparaît avec harmonia_min.

Les noms privés sont listés parce que `import *` les saute : à régénérer
quand un consommateur en importe un nouveau (ça s'est produit le 2026-09-14
avec `_ireal_cascade`, ajouté par une session concurrente)."""
from harmonia.folding import *  # noqa: F401,F403
from harmonia.folding import _bar_vecs, _ireal_cascade, _ireal_endings, _template_chords, _write_position  # noqa: F401
