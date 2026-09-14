"""Une seule convention pour les caches de calcul par morceau.

Quatre étages gardent un résultat par morceau : battues, postérieures musx,
chroma NNLS, sections songformer. Dans harmonia_min chacun construit son
chemin lui-même, avec trois conventions (nom seul ; nom + extension ; dossier
propre ou dossier partagé sous `data/cache`). Ici un seul endroit décide.

La clé reste POUR L'INSTANT celle de harmonia_min (le nom du fichier audio,
sans extension — avec extension pour songformer), pour que le rapport d'or
compare des choses identiques. Le sprint 15 passe à `<nom>__<taille>` et
renomme les caches en place, sans les dupliquer (le disque est plein à 96 %).

Ce que ce module ne fait PAS : calculer. Un cache absent, c'est à l'étage de
décider s'il calcule (chemin froid du serveur) ou s'il refuse (rapport d'or).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from harmonia.settings import SETTINGS, Settings


def _stem(audio: Path) -> str:
    return Path(audio).stem


def _name(audio: Path) -> str:
    return Path(audio).name


#: kind → (dossier, fonction de clé, extension)
KINDS = {
    "beats":      (lambda s: s.state_dir / "beats",       _stem, ".json"),
    "songformer": (lambda s: s.state_dir / "songformer",  _name, ".json"),
    "musx_probs": (lambda s: s.data_cache / "musx_probs", _stem, ".npz"),
    "musx_cqt":   (lambda s: s.data_cache / "musx_cqt",   _stem, ".npz"),
    "nnls":       (lambda s: s.data_cache / "nnls_infer", _stem, ".npz"),
}


def path(kind: str, audio: Path, settings: Settings = SETTINGS) -> Path:
    folder, key, ext = KINDS[kind]
    return folder(settings) / (key(audio) + ext)


def load_json(kind: str, audio: Path) -> dict | None:
    p = path(kind, audio)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(kind: str, audio: Path, obj: dict) -> Path:
    p = path(kind, audio)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return p


def load_npz(kind: str, audio: Path) -> dict[str, np.ndarray] | None:
    p = path(kind, audio)
    if not p.exists():
        return None
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def save_npz(kind: str, audio: Path, **arrays: np.ndarray) -> Path:
    p = path(kind, audio)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **arrays)
    return p
