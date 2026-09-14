#!/usr/bin/env python3
"""scripts/songformer_run.py — faire tourner SongFormer sur les morceaux annotés.

Louis, 2026-08-18 : « lance le 1 » — faire tourner un modèle pré-entraîné de
structure sur ses morceaux et arbitrer À L'OREILLE, avant d'écrire une ligne de
code de structure de plus.

SongFormer (ASLP-lab, octobre 2025) sort intro/couplet/refrain/pont directement
depuis l'audio, sans rien entraîner. On ne lui donne rien d'autre que le wav.

    .venv/bin/python scripts/songformer_run.py [--un <stem>]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SORTIE = REPO / "docs" / "research_sessions" / "songformer.json"
ETAT = REPO / "harmonia_min" / "state"
AUD = REPO / "docs" / "audio"


def morceaux():
    """Les morceaux annotés à la main dont on a l'audio ici."""
    out, vus = [], set()
    for d in ("sections", "sections_draft"):
        for f in sorted((ETAT / d).glob("*.json")):
            if f.stem in vus:
                continue
            p = ETAT / "charts" / f"min_{f.stem}.json"
            if not p.exists():
                continue
            m = json.loads(p.read_text(encoding="utf-8"))
            a = AUD / Path(m.get("audio_url") or "x").name
            if not a.exists():
                continue
            vus.add(f.stem)
            out.append((f.stem, a, m.get("title") or f.stem))
    return out


def main() -> None:
    import importlib.util

    import torch
    from huggingface_hub import snapshot_download
    from safetensors.torch import load_file

    local = snapshot_download(
        repo_id="ASLP-lab/SongFormer", repo_type="model",
        ignore_patterns=["SongFormer.pt", "figs/*"])
    sys.path.insert(0, local)
    os.environ["SONGFORMER_LOCAL_DIR"] = local
    print(f"modele dans {local}")

    # PAS `AutoModel.from_pretrained`. transformers 5.x construit sous le
    # device « meta » (des tenseurs sans memoire, materialises apres coup) et
    # le MuQ interne de SongFormer y melange des tenseurs cpu et meta :
    # « Tensor on device cpu is not on the expected device meta! » a la
    # construction, avec ou sans low_cpu_mem_usage. On importe donc la classe
    # du depot et on la construit nous-memes, hors de tout contexte meta.
    spec = importlib.util.spec_from_file_location(
        "msf", os.path.join(local, "modeling_songformer.py"))
    msf = importlib.util.module_from_spec(spec)
    sys.modules["msf"] = msf
    spec.loader.exec_module(msf)
    cfg = msf.SongFormerConfig.from_pretrained(local)
    with torch.device("cpu"):
        mdl = msf.SongFormerModel(cfg)
    # `model.safetensors` porte TOUT (muq + musicfm + la tete) ;
    # `SongFormer.safetensors` n'est que la copie EMA de la tete, 62 tenseurs.
    miss, unexp = mdl.load_state_dict(
        load_file(os.path.join(local, "model.safetensors")), strict=False)
    print(f"poids : {len(miss)} manquants, {len(unexp)} inattendus")
    dev = "cpu"                    # les noyaux de MuQ ne sont pas tous sur MPS
    mdl.to(dev).eval()
    print(f"charge sur {dev}")

    seul = None
    if "--un" in sys.argv:
        seul = sys.argv[sys.argv.index("--un") + 1]

    res = {}
    if SORTIE.exists():
        res = json.loads(SORTIE.read_text(encoding="utf-8"))
    for stem, audio, titre in morceaux():
        if seul and stem != seul:
            continue
        if stem in res:
            print(f"  deja fait : {stem}")
            continue
        try:
            with torch.no_grad():
                out = mdl(str(audio))
        except Exception as exc:
            print(f"  RATE {stem}: {type(exc).__name__}: {str(exc)[:120]}")
            continue
        res[stem] = {"titre": titre, "brut": _serialisable(out)}
        SORTIE.parent.mkdir(parents=True, exist_ok=True)
        SORTIE.write_text(json.dumps(res, ensure_ascii=False), encoding="utf-8")
        print(f"  ok {stem}: {str(_serialisable(out))[:150]}")


def _serialisable(o):
    import numpy as np
    if isinstance(o, dict):
        return {k: _serialisable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_serialisable(x) for x in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if hasattr(o, "tolist"):
        return o.tolist()
    return o


if __name__ == "__main__":
    main()
