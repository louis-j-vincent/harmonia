"""T5-prep — rejoue le pipeline Mac complet avec un levier de vitesse changé.

Usage (venv de PROD, `.venv`, celui qui a torch + beat_this) :

    .venv/bin/python mobile/tools/fastmode/run_lever.py LEVIER OUT.json

LEVIER ∈ {baseline, small0, fold1, fold3}. `baseline` = final0 + 5 folds
(ce que le serveur :7772 rend aujourd'hui). `small0` change SEULEMENT le
checkpoint beat_this. `fold1`/`fold3` changent SEULEMENT le nombre de folds
musx moyennés (1 ou 3 au lieu de 5).

SANDBOX DISQUE (obligatoire, lu dans harmonia/settings.py et harmonia/cache.py
avant d'écrire une ligne de code) :
- `HARMONIA_STATE_DIR` est posé AVANT tout import de `harmonia` → `beats_dir`,
  `songformer_dir`, `charts_dir`, `reports_dir`, `human_dir` partent tous dans
  un dossier scratch, jamais dans `state/` du worktree (partagé, tracké git).
- `harmonia.cache.KINDS["musx_probs"|"musx_cqt"|"nnls"]` sont monkey-patchés
  APRÈS import pour rediriger loin de `data/cache` (lien symbolique vers le
  cache PARTAGÉ du serveur :7772 vivant — une écriture sous une clé stem-only
  avec un modèle différent l'aurait silencieusement corrompu pour la prod).
Rien dans `harmonia/` n'est modifié sur disque : tout ça est du monkeypatch
en mémoire, dans CE script, comme fait T1c pour son propre parity harness.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

LEVER = sys.argv[1]
OUT = Path(sys.argv[2])
assert LEVER in ("baseline", "small0", "fold1", "fold3"), LEVER

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

# Un cache PAR LEVIER (namespacé par LEVER) : la clé de cache d'harmonia est
# stem+taille, PAS le modèle — deux leviers différents sur le même morceau
# écriraient/liraient le MÊME fichier de cache si le dossier était partagé.
# Vu en pratique sur ce script (this_love/stand_by_me sont revenus en 0.2s
# à la 2e invocation avant ce correctif : cache réutilisé à tort).
SCRATCH = Path("/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-Code-harmonia"
                "/b9d38c1f-e014-4053-a5a0-04b56f362161/scratchpad/fastmode_state") / LEVER
os.environ["HARMONIA_STATE_DIR"] = str(SCRATCH / "state")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import torch
torch.set_num_threads(1)

import harmonia.cache as cache_mod
import harmonia.beats as beats_mod
import harmonia.musx as musx_mod
from harmonia.pipeline import analyze
from harmonia.backends import BACKENDS

# ── sandbox musx_probs / musx_cqt / nnls away from the shared data/cache ───
_priv = SCRATCH / "data_cache"
cache_mod.KINDS = {
    **cache_mod.KINDS,
    "musx_probs": (lambda s: _priv / "musx_probs", cache_mod._stem, ".npz"),
    "musx_cqt":   (lambda s: _priv / "musx_cqt",   cache_mod._stem, ".npz"),
    "nnls":       (lambda s: _priv / "nnls_infer", cache_mod._stem, ".npz"),
}

SONGS = {
    "maroon_5_this_love": Path("/Users/vincente/harmonia/data/cache/tune_lock_propagation/wav/maroon_5_this_love.wav"),
    "ben_e_king_stand_by_me_audio": Path("/Users/vincente/harmonia/data/cache/tune_lock_propagation/wav/ben_e_king_stand_by_me_audio.wav"),
    # Pas dans le worktree (docs/audio/ n'y est pas checked out) — lecture
    # seule depuis l'arbre partagé, même fichier que celui utilisé par T1c
    # (mobile/tools/parity_beats.py).
    "autumn_leaves": Path("/Users/vincente/Documents/Projets Perso/Code/harmonia/docs/audio/autumn_leaves.m4a"),
}

# ── apply the lever ─────────────────────────────────────────────────────────
if LEVER == "small0":
    from beat_this.inference import File2Beats
    beats_mod._f2b = File2Beats(device="cpu", dbn=False, checkpoint_path="small0")
elif LEVER in ("fold1", "fold3"):
    n = 1 if LEVER == "fold1" else 3
    musx_mod.MODEL_NAMES = musx_mod.MODEL_NAMES[:n]

results = {}
for stem, path in SONGS.items():
    assert path.exists(), path
    BACKENDS.reset_timings()
    t0 = time.perf_counter()
    model = analyze(str(path), title=stem, file_key=stem)
    total = time.perf_counter() - t0
    results[stem] = {
        "lever": LEVER,
        "total_s": round(total, 3),
        "backend_timings": dict(BACKENDS.timings),
        "nBars": model.get("nBars"),
        "n_beats": len(model.get("beatTimes") or []),
        "bpb": model.get("bpb"),
        "barGrid": model.get("barGrid"),
        "sections": [
            {"bars": sec.get("bars"), "barSpans": sec.get("barSpans"),
             "label": sec.get("label")}
            for sec in (model.get("sections") or [])
        ],
    }
    print(f"{LEVER} {stem}: total {total:.1f}s  timings={BACKENDS.timings}", flush=True)

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(results), encoding="utf-8")
print(f"wrote {OUT}")
