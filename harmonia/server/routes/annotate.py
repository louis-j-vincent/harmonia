"""Les corrections de Louis sur les accords : persister, relire, et les deux
routes retirées qui doivent répondre 410 plutôt que 404.

Porté verbatim depuis `harmonia_min/server.py` (sprints 11-14).

Ce que ce module ne fait PAS : appliquer les corrections au ChartModel servi
(voir `library.chart_model`, qui appelle `annotations.overlay` PUIS
`fill_empty_bars` ci-dessous) ; savoir ce qu'un chart contient
(`harmonia.annotations` seul connaît le schéma du sidecar).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from flask import Blueprint, jsonify, request

from harmonia import annotations
from harmonia.server.jobs import CHARTS_DIR
from harmonia.settings import SETTINGS

log = logging.getLogger("harmonia.server.routes.annotate")

bp = Blueprint("annotate", __name__)

AUDIO_DIR = SETTINGS.audio_dir


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
    # LE REGISTRE DES CORRECTIONS (Louis, 2026-09-18 : « récupère
    # automatiquement toutes les différences d'annotation et persiste-les,
    # afin qu'un agent dédié puisse apprendre de mes corrections »). On compare
    # ce qu'il vient d'écrire au chart CUIT — celui qui porte encore ce que la
    # machine pensait — et on ajoute une ligne par écart.
    #
    # Le registre ne doit JAMAIS faire échouer une sauvegarde : son annotation
    # est le travail, la ligne de registre n'est qu'une trace.
    try:
        from harmonia import corrections
        p = CHARTS_DIR / f"{Path(file).stem}.json"
        if p.exists():
            corrections.note_corrections_accords(
                Path(file).stem, json.loads(p.read_text(encoding="utf-8")),
                saved.get("chords") or [])
    except Exception:                                        # noqa: BLE001
        log.exception("registre des corrections : accords non notés pour %s",
                      file)
    return jsonify(saved)


@bp.get("/api/annotations/<file>")
def get_annotations(file):
    return jsonify(annotations.load_annotation(file))


@bp.post("/api/chord-candidates/<file>")
def chord_candidates(file):
    """Les candidats d'un créneau qu'on vient d'AJOUTER, classés par le delta.

    Louis, 2026-09-16 : « dans l'option editing quand on clique pour créer un
    nouvel accord, dans les suggestions on prend celles du delta prior ».

    Un accord qu'on invente n'a pas de `sug` : ce créneau-là n'existait pas
    quand le chart a été cuit, personne n'a classé ses candidats. On les
    calcule ici, sur les postérieures musx déjà en cache, et on les classe par
    ce qu'ils GAGNENT sur le créneau précédent — voir
    `span_rescore.delta_candidates` pour le pourquoi (la pédale) et pour ce
    que ça ne résout pas (c'est une hypothèse mesurée sur 2 accords).

    Corps : `{"t0","t1"}` du nouveau créneau, `{"prev_t0","prev_t1"}` de
    l'accord qui précède (absents = premier accord du morceau, on rend alors
    le classement brut en le disant). Le client les connaît mieux que nous :
    c'est lui qui vient de poser le brouillon.
    """
    from harmonia import musx as _musx
    from harmonia.span_rescore import delta_candidates
    p = CHARTS_DIR / f"{Path(file).stem}.json"
    if not p.exists():
        return jsonify({"error": "no such chart"}), 404
    d = request.get_json(silent=True) or {}
    try:
        span = (float(d["t0"]), float(d["t1"]))
    except (KeyError, TypeError, ValueError):
        return jsonify({"error": "t0 et t1 requis"}), 400
    prev = None
    if d.get("prev_t0") is not None and d.get("prev_t1") is not None:
        try:
            prev = (float(d["prev_t0"]), float(d["prev_t1"]))
        except (TypeError, ValueError):
            prev = None
    chart = json.loads(p.read_text(encoding="utf-8"))
    stem = Path(chart.get("audio_url") or "").stem
    audio = AUDIO_DIR / f"{stem}.m4a"
    if not stem or not audio.exists():
        # Pas de repli muet : sans audio il n'y a rien à classer, et l'éditeur
        # doit pouvoir dire « pas de classement » plutôt qu'en inventer un.
        return jsonify({"error": "pas d'audio pour ce chart", "sug": []}), 404
    try:
        probs = _musx.frame_posteriors(audio)      # cache disque, pas d'inférence
    except Exception as exc:                        # noqa: BLE001
        log.warning("candidats: postérieures indisponibles pour %s (%s)", stem, exc)
        return jsonify({"error": "postérieures musx indisponibles", "sug": []}), 503
    return jsonify(delta_candidates(probs, span, prev))


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


# ── ajouter un accord dans une mesure ENTIÈREMENT tenue ──────────────────────
# L'écran Annotate (2026-09-16, Louis : « quand je clique sur un endroit vide
# de la barre je peux ajouter un accord dessus ») pose un accord sur n'importe
# quel temps d'une mesure, y compris une mesure sans aucun accord (le « % »
# tenu du chart). `annotations.overlay` sait déjà insérer un accord dans une
# mesure qui en a AU MOINS UN (son mécanisme de repli — voir
# `harmonia/static/screens/annotate.js::splitBar`, la même idée côté client —
# clone un accord-hôte de la même mesure) : ci-dessous complète le seul cas
# qu'il laisse tomber en silence, une mesure `[]`, où aucun hôte n'existe à
# cloner.

def _fixkey(bar, beat) -> tuple:
    """Même clé que `harmonia.annotations._key`, dupliquée ici plutôt
    qu'importée : c'est un symbole privé d'un autre module, pas une
    convention partagée — mieux vaut la refaire à l'identique que de
    dépendre d'un détail interne qui peut changer sans préavis."""
    try:
        return int(bar), round(float(beat or 0), 3)
    except (TypeError, ValueError):
        return bar, beat


def _locate_empty_prefix_bar(model: dict, bar_no: int):
    """La liste d'accords (encore vide) de la mesure `bar_no`, ou None.

    Seules les mesures du CORPS d'une section suivent l'arithmétique simple
    `barRanges[0][0] + position` (vérifié sur le chart réel
    `min_Ju8Hr50Ckwk.json` : la section « outro », une seule mesure `[]`,
    `barRanges=[[128,128]]`). Une mesure vide DANS une fin alternative
    (« 1st/2nd ending ») est numérotée par un calcul différent
    (`chart.js::passSpan`) et n'est pas couverte ici — plus rare, jamais
    rencontrée sur la bibliothèque au 2026-09-16 ; `overlay` l'ignorait déjà
    en silence, le comportement ne régresse pas.
    """
    for sec in model.get("sections", []) or []:
        ranges = sec.get("barRanges") or []
        bars = sec.get("bars") or []
        if not ranges or not bars:
            continue
        endings = sec.get("endings") or {}
        tail = int(endings.get("tail") or 0)
        prefix_len = len(bars) - tail
        pos = bar_no - ranges[0][0]
        if 0 <= pos < prefix_len and bars[pos] == []:
            return bars[pos]
    return None


def fill_empty_bars(model: dict, ann: dict) -> dict:
    """Après `overlay(model, ann)` : insère les corrections qu'il a dû
    abandonner faute d'hôte (mesure `bars == []`). Mute `model` en place et
    le retourne, comme `overlay`.
    """
    fixes = [f for f in (ann.get("chords") or []) if f.get("root") is not None]
    if not fixes:
        return model
    present = set()
    for sec in model.get("sections", []) or []:
        for bar in sec.get("bars", []) or []:
            for ch in bar:
                present.add(_fixkey(ch.get("bar"), ch.get("beat")))
    for fix in fixes:
        key = _fixkey(fix.get("bar"), fix.get("beat"))
        if key in present:
            continue  # overlay() l'a déjà appliquée ou synthétisée
        try:
            bar_no = int(fix.get("bar"))
        except (TypeError, ValueError):
            continue
        target = _locate_empty_prefix_bar(model, bar_no)
        if target is None:
            continue
        bass = fix.get("bass", -1)
        target.append({
            "root": int(fix["root"]) % 12, "q": fix.get("q", ""),
            "bass": int(bass) if bass is not None and int(bass) >= 0 else -1,
            "nc": False, "c": 1.0, "confirmed": True, "n": 0, "sug": None,
            "bar": bar_no, "beat": fix.get("beat", 0),
            "t0": fix.get("t0"), "t1": fix.get("t1"),
        })
        present.add(key)
    return model


# Ce que ce module ne fait PAS : les sections annotées à la main (voir
# `sections.py`) — un sidecar différent, un schéma différent.
