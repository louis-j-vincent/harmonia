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

from harmonia import annotations, titles as _titles
from harmonia.server.jobs import CHARTS_DIR, META_PATH, _load_chart_meta
from harmonia.server.routes.annotate import fill_empty_bars
from harmonia.server.routes.sections import _safe_stem
from harmonia.settings import SETTINGS

log = logging.getLogger("harmonia.server.routes.library")

bp = Blueprint("library", __name__)

FOLDERS_PATH = SETTINGS.folders_path


def _pretty_title(stem: str) -> str:
    """Un stem de fichier rendu lisible, mentions de production enlevées.
    Délègue à `harmonia.titles` — même règle partout, testée là-bas."""
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

@bp.get("/api/folders")
def read_folders():
    """La copie serveur des dossiers — pour que le téléphone la retrouve.

    Le `localStorage` reste la source de vérité côté client (voir
    `library.js`), mais il est PAR APPAREIL : un classement fait sur le Mac
    n'existait pas sur l'iPhone, et un navigateur vidé le perdait. L'écran
    fusionne maintenant cette copie pour les charts qu'il ne classe pas
    lui-même — il ne l'écrase jamais avec, donc un classement local gagne
    toujours sur la copie.
    """
    try:
        return jsonify(json.loads(FOLDERS_PATH.read_text(encoding="utf-8")))
    except (OSError, ValueError):
        return jsonify({"order": [], "of": {}})


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
    """Retirer un morceau de la bibliothèque — sans jamais perdre son travail.

    Louis, 2026-09-17 : « je veux une option pour facilement supprimer une
    chanson si j'en veux plus ».

    CE QUI EST DÉPLACÉ, PAS DÉTRUIT. Tout ce qu'il a fait à la main part dans
    `state/human/corbeille/<date>_<stem>/` — le chart, les accords confirmés,
    le découpage des sections (vérité ET brouillon), la marque de mesure 1 ou
    de tempo, et sa ligne de `chart_meta`. Ce dossier est SUIVI PAR GIT, comme
    le reste de `state/human/` : une annotation faite à la main a déjà été
    perdue une fois pour avoir vécu dans un dossier ignoré (2026-08-12), et
    c'est cette séparation qui est le correctif. Supprimer un morceau ne doit
    pas rouvrir ce trou.

    CE QUI EST VRAIMENT EFFACÉ : l'audio et les caches régénérables (battues,
    songformer, postérieures musx, CQT, NNLS) — environ 5 Mo par morceau,
    mesuré. Ils se refabriquent à partir du fichier, et le chart mis de côté
    garde l'identifiant de la vidéo pour le retélécharger.

    CE QUE ÇA NE FAIT PAS : remettre le morceau. La restauration est un geste
    manuel (recopier le dossier de corbeille), pas un bouton — l'écran, lui,
    offre déjà cinq secondes d'annulation avant même que cette route ne parte.
    """
    import shutil
    from datetime import datetime

    from harmonia import cache
    stem_chart = Path(file).stem
    p = CHARTS_DIR / f"{stem_chart}.json"
    chart = {}
    if p.exists():
        try:
            chart = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            chart = {}
    stem = Path(chart.get("audio_url") or "").stem or \
        stem_chart.removeprefix("min_")

    corbeille = (SETTINGS.human_dir / "corbeille" /
                 f"{datetime.now():%Y-%m-%d_%H%M%S}_{stem}")
    corbeille.mkdir(parents=True, exist_ok=True)
    garde = []
    for src, nom in ((p, "chart.json"),
                     (annotations.chemin_annotation(stem_chart),
                      "annotations.json"),
                     (SETTINGS.sections_dir / f"{stem}.json", "sections.json"),
                     (SETTINGS.sections_draft_dir / f"{stem}.json",
                      "sections_brouillon.json"),
                     (SETTINGS.marks_dir / f"{stem}.json", "marque.json")):
        if src.exists():
            shutil.move(str(src), str(corbeille / nom))
            garde.append(nom)

    # sa ligne de chart_meta part avec le reste
    meta_f = SETTINGS.chart_meta_path
    try:
        meta = json.loads(meta_f.read_text(encoding="utf-8"))
        if stem_chart in meta:
            (corbeille / "chart_meta.json").write_text(
                json.dumps({stem_chart: meta.pop(stem_chart)}, indent=1,
                           ensure_ascii=False), encoding="utf-8")
            meta_f.write_text(json.dumps(meta, indent=1, ensure_ascii=False),
                              encoding="utf-8")
            garde.append("chart_meta.json")
    except (OSError, ValueError):
        log.warning("suppression %s : chart_meta illisible, laissé tel quel", file)

    # l'audio et les caches, eux, s'en vont pour de bon — ils se refabriquent
    audio = SETTINGS.audio_dir / f"{stem}.m4a"
    libere = 0
    if audio.exists():
        for kind in cache.KINDS:
            try:
                c = cache.path(kind, audio)
            except OSError:
                continue
            if c.exists():
                libere += c.stat().st_size
                c.unlink()
        libere += audio.stat().st_size
        audio.unlink()

    log.info("suppression %s : %d fichier(s) mis de côté dans %s, %.1f Mo "
             "libérés", file, len(garde), corbeille.name, libere / 1e6)
    return jsonify({"ok": True, "corbeille": corbeille.name,
                    "garde": garde, "libere_mo": round(libere / 1e6, 1)})


# Ce que ce module ne fait PAS : la recherche (locale ou YouTube — voir
# `analyze.py`/`youtube.py`) ; les sections ou l'annotation des accords (voir
# `sections.py`/`annotate.py`).
