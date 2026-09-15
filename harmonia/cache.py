"""Une seule convention pour les caches de calcul par morceau.

Quatre étages gardent un résultat par morceau : battues, postérieures musx,
chroma NNLS, sections songformer. Dans harmonia_min chacun construit son
chemin lui-même, avec trois conventions (nom seul ; nom + extension ; dossier
propre ou dossier partagé sous `data/cache`). Ici un seul endroit décide.

LA CLÉ (sprint 15) est `<stem>__<taille en octets>` pour LES QUATRE genres —
ce que songformer faisait déjà en substance (son propre champ `taille` dans
le JSON), généralisé et mis dans le nom de fichier : un ré-encodage du même
morceau, même stem, change de taille et ne réutilise donc jamais le cache
d'un autre fichier.

CE QUI NE BOUGE PAS ENCORE : `data/cache/{musx_probs,musx_cqt,nnls_infer}`
est un lien symbolique vers le cache PARTAGÉ du serveur vivant — le disque
est plein à 96 %, et renommer 12 Go de caches en place n'est pas ce sprint
(prévu à la bascule prod, sprint 21). Ces trois genres ÉCRIVENT donc sous la
clé neuve mais LISENT d'abord la clé neuve puis, si absente, l'ANCIENNE clé
(le stem seul, ou pour songformer le nom de fichier complet) — une ligne
INFO par lecture historique, pour qu'on voie le jour où plus rien ne la
déclenche. `beats` et `songformer`, eux, ont changé de DOSSIER aussi (ils
vivent maintenant sous `state/cache/`, suivi par personne) : c'est
`tools/migrate_state.py` qui les y recopie déjà sous la clé neuve, donc leur
repli historique ne sert qu'avant la migration ou dans un test.

Ce que ce module ne fait PAS : calculer. Un cache absent, c'est à l'étage de
décider s'il calcule (chemin froid du serveur) ou s'il refuse (rapport d'or).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from harmonia.settings import SETTINGS, Settings

logger = logging.getLogger(__name__)


def _stem(audio: Path) -> str:
    return Path(audio).stem


def _name(audio: Path) -> str:
    return Path(audio).name


def _new_key(audio: Path) -> str:
    audio = Path(audio)
    return f"{audio.stem}__{audio.stat().st_size}"


#: kind → (dossier, fonction de clé HISTORIQUE, extension)
KINDS = {
    "beats":      (lambda s: s.beats_dir,                 _stem, ".json"),
    "songformer": (lambda s: s.songformer_dir,            _name, ".json"),
    "musx_probs": (lambda s: s.data_cache / "musx_probs", _stem, ".npz"),
    "musx_cqt":   (lambda s: s.data_cache / "musx_cqt",   _stem, ".npz"),
    "nnls":       (lambda s: s.data_cache / "nnls_infer", _stem, ".npz"),
}


def folder(kind: str, settings: Settings = SETTINGS) -> Path:
    """Le dossier d'un cache — pour les ponts et les outils qui listent ou vident."""
    return KINDS[kind][0](settings)


def path(kind: str, audio: Path, settings: Settings = SETTINGS) -> Path:
    """Le chemin sous la clé NEUVE (`<stem>__<taille>`) — celui des écritures."""
    _, _, ext = KINDS[kind]
    return folder(kind, settings) / (_new_key(audio) + ext)


def _legacy_path(kind: str, audio: Path, settings: Settings = SETTINGS) -> Path:
    """Le chemin sous la clé HISTORIQUE — repli de lecture seulement."""
    _, legacy_key, ext = KINDS[kind]
    return folder(kind, settings) / (legacy_key(audio) + ext)


def exists(kind: str, audio: Path, settings: Settings = SETTINGS) -> bool:
    """Vrai si un cache est chaud, sous la clé neuve OU l'historique."""
    if path(kind, audio, settings).exists():
        return True
    return _legacy_path(kind, audio, settings).exists()


def _resolve_read(kind: str, audio: Path, settings: Settings) -> Path | None:
    p = path(kind, audio, settings)
    if p.exists():
        return p
    legacy = _legacy_path(kind, audio, settings)
    if legacy.exists():
        logger.info("cache %s : clé historique %s (clé neuve %s absente)",
                    kind, legacy.name, p.name)
        return legacy
    return None


def load_json(kind: str, audio: Path) -> dict | None:
    p = _resolve_read(kind, audio, SETTINGS)
    if p is None:
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def save_json(kind: str, audio: Path, obj: dict) -> Path:
    p = path(kind, audio)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False), encoding="utf-8")
    return p


def load_npz(kind: str, audio: Path) -> dict[str, np.ndarray] | None:
    p = _resolve_read(kind, audio, SETTINGS)
    if p is None:
        return None
    with np.load(p) as z:
        return {k: z[k] for k in z.files}


def save_npz(kind: str, audio: Path, **arrays: np.ndarray) -> Path:
    p = path(kind, audio)
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(p, **arrays)
    return p
