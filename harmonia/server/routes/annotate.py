"""Les corrections de Louis sur les accords : persister, relire, et les deux
routes retirées qui doivent répondre 410 plutôt que 404.

Porté verbatim depuis `harmonia_min/server.py` (sprints 11-14).

Ce que ce module ne fait PAS : appliquer les corrections au ChartModel servi
(voir `library.chart_model`, qui appelle `annotations.overlay`) ; savoir ce
qu'un chart contient (`harmonia.annotations` seul connaît le schéma du
sidecar).
"""
from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

from harmonia import annotations

log = logging.getLogger("harmonia.server.routes.annotate")

bp = Blueprint("annotate", __name__)


@bp.post("/api/annotations/<file>")
def save_annotations(file):
    """Persist confirmed chords + merges. Body is the shell's whole doc
    (last-write-wins); the response must be JSON — the shell calls
    r.json() on it."""
    doc = request.get_json(silent=True) or {}
    try:
        saved = annotations.save_annotation(file, {
            "annotator": doc.get("annotator", ""),
            "chords": doc.get("chords", []),
            "merges": doc.get("merges", []),
        })
    except (OSError, TypeError, ValueError) as exc:
        # Never fail silently: the shell swallows errors, so the log is the
        # only place a lost lock can surface.
        log.warning("annotation save failed for %s: %s", file, exc)
        return jsonify({"error": "could not persist annotations"}), 500
    log.info("annotations %s: %d chord(s), %d merge(s)",
             file, len(saved["chords"]), len(saved["merges"]))
    return jsonify(saved)


@bp.get("/api/annotations/<file>")
def get_annotations(file):
    return jsonify(annotations.load_annotation(file))


@bp.route("/api/context_rescore/<file>", methods=["POST"])
@bp.route("/api/reinfer/<file>", methods=["POST"])
def propagation_retiree(file):
    """RETIRÉE (Louis, 2026-08-20 : « quand on annote un nouvel accord, ça se
    propage sur les accords suivants, mais cette fonction est deprecated,
    enlève-la »).

    On garde la route pour DIRE que c'est parti, au lieu d'un 404 que le shell
    avalerait en silence. Le chemin « merge » (`/api/reinfer/`) passait par ici
    lui aussi et n'y recevait déjà qu'une réponse sans effet : il obtient
    maintenant un refus lisible plutôt qu'un faux succès de vingt secondes.
    """
    return jsonify({"error": "La propagation d'un accord sur ses voisins a "
                             "été retirée. Corrige l'accord à la main : il "
                             "sera gardé tel quel."}), 410


# Ce que ce module ne fait PAS : les sections annotées à la main (voir
# `sections.py`) — un sidecar différent, un schéma différent.
