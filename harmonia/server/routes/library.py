"""La bibliothèque : liste des charts, le ChartModel d'un fichier (avec
overlay des annotations), les métadonnées éditables (artiste/titre),
les dossiers, et la suppression d'un chart.

Porté verbatim depuis `harmonia_min/server.py` (sprints 11-14). `CHARTS_DIR`,
`META_PATH` et `_load_chart_meta` viennent de `jobs.py` — c'est là que le
sidecar `chart_meta.json` est aussi écrit (autosave artiste/titre à la fin
d'un job), donc une seule définition plutôt que deux qui pourraient diverger.

Ce que ce module ne fait PAS : écrire un chart (voir `jobs.py`) ; savoir ce
qu'un chart contient au-delà de ce qu'il faut pour la vignette de
bibliothèque (voir `harmonia/chart_model.py`, à venir).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from flask import Blueprint, jsonify, request

from harmonia_min import annotations, titles as _titles
from harmonia.server.jobs import CHARTS_DIR, META_PATH, _load_chart_meta
from harmonia.server.routes.annotate import fill_empty_bars
from harmonia.server.routes.sections import _safe_stem
from harmonia.settings import SETTINGS

log = logging.getLogger("harmonia.server.routes.library")

bp = Blueprint("library", __name__)

FOLDERS_PATH = SETTINGS.folders_path


def _pretty_title(stem: str) -> str:
    """Un stem de fichier rendu lisible, mentions de production enlevées.
    Délègue à `harmonia_min.titles` — même règle partout, testée là-bas."""
    return _titles.pretty_from_slug(stem)


def _capabilities() -> list[str]:
    """What this server can actually do right now.

    The shell decides what to show from this list; we only state facts.
    "reinfer" (chord-propagation) was retired 2026-08-20 (Louis: "cette
    fonction est deprecated, enlève-la") and is no longer advertised.
    """
    return ["annotations"]


@bp.get("/api/library")
def library():
    meta = _load_chart_meta()
    charts = []
    for p in sorted(CHARTS_DIR.glob("*.json")):
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        mm = meta.get(p.stem) or {}
        charts.append({
            "file": p.stem,
            "title": mm.get("title") or m.get("title") or _pretty_title(p.stem),
            "artist": mm.get("artist") or "",
            "key": m.get("key") or {"tonic": 0, "mode": "major"},
            "bars": m.get("nBars") or 0,
            "hasAudio": bool(m.get("audio_url")),
            "mtime": p.stat().st_mtime,
        })
    return jsonify({"charts": charts, "capabilities": _capabilities()})


@bp.get("/api/chart-model/<file>")
def chart_model(file):
    p = CHARTS_DIR / f"{Path(file).stem}.json"
    if not p.exists():
        return jsonify({"error": "no such chart"}), 404
    # Rehydrate server-side: the shell POSTs annotations but never GETs
    # them, so the overlay has to happen here or every lock dies on reload.
    model = json.loads(p.read_text(encoding="utf-8"))
    ann = annotations.load_annotation(file)
    model = annotations.overlay(model, ann)
    # overlay() can insert a new chord into a bar that already has one (it
    # clones that chord as a template); it silently drops a fix that targets
    # a bar with NONE (a fully-held "%" bar) — fill_empty_bars covers just
    # that case (see its docstring).
    return jsonify(fill_empty_bars(model, ann))


# ── artiste / titre éditables (delta2 §8) ────────────────────────────────────
# « le titre YouTube ment souvent »: le client peut poser artist/title par
# chart. Sidecar unique (state/chart_meta.json), dernier écrit gagne, ressert
# dans /api/library — jamais écrit dans le chart lui-même (le stem reste la
# clé, le modèle reste ce que le pipeline a produit).

@bp.post("/api/chart-meta/<file>")
def chart_meta(file):
    stem = _safe_stem(Path(file).stem)
    if not stem:
        return jsonify({"error": "bad stem"}), 400
    doc = request.get_json(silent=True) or {}
    meta = _load_chart_meta()
    meta[stem] = {"artist": (doc.get("artist") or "").strip()[:120],
                  "title": (doc.get("title") or "").strip()[:200]}
    try:
        META_PATH.parent.mkdir(parents=True, exist_ok=True)
        META_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    except OSError as exc:
        log.warning("chart-meta save failed for %s: %s", stem, exc)
        return jsonify({"error": "could not persist"}), 500
    return jsonify({"ok": True, **meta[stem]})


# ── dossiers (delta2 §8) ─────────────────────────────────────────────────────
# Le client garde localStorage comme source de vérité et POSTe en
# write-through; le serveur n'en fait (pour l'instant) qu'une copie de
# sauvegarde — {"order": [...], "of": {"<file>": "<dossier>"}}.

@bp.post("/api/folders")
def save_folders():
    doc = request.get_json(silent=True) or {}
    payload = {"order": [str(x)[:80] for x in (doc.get("order") or [])][:200],
               "of": {str(k)[:200]: str(v)[:80]
                      for k, v in (doc.get("of") or {}).items()}}
    try:
        FOLDERS_PATH.parent.mkdir(parents=True, exist_ok=True)
        FOLDERS_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=1))
    except (OSError, TypeError) as exc:
        log.warning("folders save failed: %s", exc)
        return jsonify({"error": "could not persist"}), 500
    return jsonify({"ok": True, "folders": len(payload["order"])})


@bp.delete("/api/chart/<file>")
def delete_chart(file):
    p = CHARTS_DIR / f"{Path(file).stem}.json"
    if p.exists():
        p.unlink()
    annotations.delete_annotation(file)
    return jsonify({"ok": True})


# Ce que ce module ne fait PAS : la recherche (locale ou YouTube — voir
# `analyze.py`/`youtube.py`) ; les sections ou l'annotation des accords (voir
# `sections.py`/`annotate.py`).
