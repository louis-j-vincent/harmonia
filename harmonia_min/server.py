"""harmonia_min/server.py — minimal Flask server for the app shell. Port 7772.

NEVER port 7771 — that is the old production server, owned by other sessions.

Serves exactly what app_shell.html's milestone-1 path needs:

    GET  /                        the app shell (copied verbatim)
    GET  /api/library             chart list  {charts:[{file,title,key,bars,hasAudio,mtime}]}
    GET  /api/chart-model/<file>  a ChartModel JSON from state/charts/, with
                                  the annotation sidecar overlaid server-side
    POST /api/annotations/<file>  persist confirmed chords + merges (sidecar)
    GET  /api/annotations/<file>  read the sidecar back (the shell never does)
    POST /api/analyze {url}       resolve → background pipeline run → {job_id}
    GET  /api/job/<id>            job record (stage/tempo/key/…/status/url).
                                  `chart_url` apparaît dès que le chart BRUT
                                  est écrit, `status` encore "running" ;
                                  `status=done` quand le raffinement (sections,
                                  clé) a réécrit le MÊME fichier.
    POST /api/yt-search {q}       matches LOCAL docs/audio stems (id "local:<stem>"),
                                  so the library flow works fully offline; real
                                  YouTube URLs pasted into the box still analyze
                                  via yt-dlp when it is installed
    GET  /audio/<name>            the local audio the charts play
    DELETE /api/chart/<file>      remove a chart from the library

Everything else the shell may call (annotations, reinfer, billboard, jam,
irealb, section-merge) returns 404/501 — the UI catches and degrades; those
surfaces are later milestones.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

from harmonia_min import annotations, titles as _titles
from harmonia_min.pipeline import analyze_steps

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("harmonia_min.server")

PKG = Path(__file__).resolve().parent
REPO = PKG.parent
AUDIO_DIR = REPO / "docs" / "audio"
CHARTS_DIR = PKG / "state" / "charts"
# Le port par défaut reste 7772 (l'app vivante). HARMONIA_MIN_PORT permet
# de lancer une SECONDE instance depuis un worktree — indispensable pour
# essayer une UI sans écraser celle qu'une autre session sert.
PORT = int(os.environ.get("HARMONIA_MIN_PORT", "7772"))

app = Flask(__name__)
_jobs: dict[str, dict] = {}


def _pretty_title(stem: str) -> str:
    """Un stem de fichier rendu lisible, mentions de production enlevées.
    Délègue à `harmonia_min.titles` — même règle partout, testée là-bas."""
    return _titles.pretty_from_slug(stem)


# ── app shell + audio ────────────────────────────────────────────────────────

@app.get("/")
def index():
    return send_file(PKG / "app_shell.html")


@app.get("/audio/<path:name>")
def audio(name):
    # Python's mimetypes guesses .m4a → "audio/mp4a-latm", which iOS's media
    # player does not treat as a playable container: in the installed (PWA
    # standalone) app the <audio> stalls at HAVE_METADATA with an empty
    # buffer forever (Louis's iPhone, 2026-08-02). Safari-in-browser is
    # lenient, the standalone player is not. The old :7771 app serves
    # "audio/mp4" and plays fine — do the same.
    mt = "audio/mp4" if name.lower().endswith((".m4a", ".mp4")) else None
    resp = send_from_directory(AUDIO_DIR, name, mimetype=mt)
    # 2026-08-02 iPhone-stall triage: record exactly what byte windows the
    # phone asks for and what we answer — werkzeug's access log only shows
    # "206", which cannot distinguish a healthy chunked playback from the
    # retry storm we are chasing.
    log.info("AUDIO %s range=%r -> %s len=%s",
             request.remote_addr, request.headers.get("Range"),
             resp.headers.get("Content-Range"),
             resp.headers.get("Content-Length"))
    # Second half of the same iPhone stall (fix ported from the old app,
    # harmonia/serving/api.py serve_audio): the shell's <audio> is
    # crossOrigin="anonymous", and iOS validates EVERY 206 Range response
    # for Access-Control-Allow-Origin — even same-origin. Without it WebKit
    # silently discards the data: server logs show a storm of 206s while
    # `buffered` stays empty. Desktop browsers are lenient, which masks it.
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


@app.get("/pwa/<path:name>")
def pwa(name):
    """PWA manifest + icons (reused verbatim from the old app, docs/pwa/)."""
    return send_from_directory(REPO / "docs" / "pwa", name)


@app.get("/reports/<path:name>")
def reports(name):
    """Pipeline explainer reports (state/reports/*.html)."""
    return send_from_directory(PKG / "state" / "reports", name)


@app.get("/plots/<path:name>")
def plots(name):
    """Diagnostic / listening pages (docs/plots/*.html), reachable over
    Tailscale (Louis, 2026-08-07: file:// links don't work for him — pages
    must live on http://100.89…:7772). Their relative ../audio/<stem>.m4a
    references resolve to the /audio route above, same Range/CORS handling."""
    return send_from_directory(REPO / "docs" / "plots", name)


@app.get("/soudure/<file>")
def soudure(file):
    """Le jeu de soudure sur UNE chanson de la bibliothèque (Louis,
    2026-08-13 : « mets le moi comme une option sur chaque chanson »).

    La page est `docs/plots/soudure.html`, autonome et inchangée : on lui pose
    simplement son `window.SONG` devant, comme le fait déjà
    `scripts/soudure_pages.py` pour les morceaux du banc. Un seul fichier, une
    seule page — pas de copie du moteur ici.
    """
    from harmonia_min.soudure import song_du_chart
    p = CHARTS_DIR / f"{Path(file).stem}.json"
    if not p.exists():
        return jsonify({"error": "no such chart"}), 404
    chart = json.loads(p.read_text(encoding="utf-8"))
    song = song_du_chart(chart, audio_dir=AUDIO_DIR)
    if not song:
        return ("<!doctype html><meta charset=utf-8><div style=\"font:16px "
                "-apple-system,system-ui,sans-serif;max-width:26rem;"
                "margin:22vh auto;padding:0 1.5rem;color:#1c1c1c\">"
                "<p>Ce chart n’a pas assez de mesures pour faire une bande.</p>"
                f"<a href=\"/?open={file}\" style=\"color:#8a2b2b\">"
                "Retour au chart</a></div>"), 404
    gabarit = (REPO / "docs" / "plots" / "soudure.html").read_text(encoding="utf-8")
    tete = ("<script>window.SONG = "
            + json.dumps(song, ensure_ascii=False, separators=(",", ":"))
            + ";</script>\n")
    page = gabarit.replace("<body>", "<body>\n" + tete, 1)
    page = page.replace("<title>Soudure</title>",
                        f"<title>Soudure — {song['titre']}</title>", 1)
    resp = app.make_response(page)
    resp.headers["Content-Type"] = "text/html; charset=utf-8"
    # Le gabarit change quand on corrige la page : pas de cache, sinon Safari
    # ressert une version périmée (déjà payé le 2026-08-13 sur le son).
    resp.headers["Cache-Control"] = "no-cache"
    return resp


@app.get("/min/<file>")
def minimal(file):
    """The minimalist representation (Louis, 2026-08-02): one block per
    letter, chronological timeline chips (tap = jump playback there)."""
    from harmonia_min.minimal_view import render_minimal
    p = CHARTS_DIR / f"{Path(file).stem}.json"
    if not p.exists():
        return jsonify({"error": "no such chart"}), 404
    return render_minimal(json.loads(p.read_text(encoding="utf-8")))


# ── library ──────────────────────────────────────────────────────────────────

@app.get("/debug/section-merge-game")
def section_merge_game():
    """A full-page dead end in milestone 1: the library links here, and a bare
    Flask 404 leaves you stranded with no way back (P3). Say so, and offer the
    door."""
    return (
        "<!doctype html><meta charset=utf-8>"
        "<meta name=viewport content='width=device-width,initial-scale=1'>"
        "<title>Section cleanup — not in this build</title>"
        "<div style=\"font:16px/1.6 -apple-system,system-ui,sans-serif;"
        "max-width:30rem;margin:22vh auto;padding:0 1.5rem;color:#2b2b2b\">"
        "<h1 style='font-size:1.15rem;margin:0 0 .6rem'>Section cleanup isn’t "
        "in this build</h1>"
        "<p style='margin:0 0 1.4rem;color:#6b6b6b'>The ear-training game for "
        "section merges lives in the older app. Everything else here works "
        "normally.</p>"
        "<a href='/' style=\"display:inline-block;background:#2b2b2b;color:#fff;"
        "text-decoration:none;border-radius:10px;padding:.6rem 1.1rem;"
        "font-weight:600\">Back to the library</a></div>"
    ), 404


@app.get("/api/section-merge-verdict")
def section_merge_verdict():
    """The library calls this on every page load to show a label count; a 404
    made two failed requests per load. Nothing has been judged in this build —
    say zero, truthfully, and the shell renders its default subtitle."""
    return jsonify({"total": 0, "merge": 0, "keep": 0})


def _capabilities() -> list[str]:
    """What this server can actually do right now (P3).

    The shell decides what to show; we only state facts. Reported as a
    capability only if the route exists AND its dependency is really there —
    "reinfer" is gated on the trained prior table, because without it
    span_rescore silently falls back to a uniform scorer that can never
    change an argmax (a button that looks alive and does nothing).
    """
    caps = ["annotations"]
    try:
        from harmonia_min import span_rescore as sr
        scorer = sr.load_context_scorer()
        if type(scorer).__name__ != "_UniformContextScorer":
            caps.append("reinfer")
    except Exception:  # noqa: BLE001 — a missing brick is a missing capability
        log.warning("capabilities: context scorer unavailable", exc_info=True)
    return caps


# ── artiste / titre éditables (delta2 §8) ────────────────────────────────────
# « le titre YouTube ment souvent »: le client peut poser artist/title par
# chart. Sidecar unique (state/chart_meta.json), dernier écrit gagne, ressert
# dans /api/library — jamais écrit dans le chart lui-même (le stem reste la
# clé, le modèle reste ce que le pipeline a produit).

META_PATH = PKG / "state" / "chart_meta.json"


def _load_chart_meta() -> dict:
    try:
        return json.loads(META_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


@app.post("/api/chart-meta/<file>")
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

FOLDERS_PATH = PKG / "state" / "folders.json"


@app.post("/api/folders")
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


@app.get("/api/library")
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


@app.get("/api/chart-model/<file>")
def chart_model(file):
    p = CHARTS_DIR / f"{Path(file).stem}.json"
    if not p.exists():
        return jsonify({"error": "no such chart"}), 404
    # Rehydrate server-side: the shell POSTs annotations but never GETs
    # them, so the overlay has to happen here or every lock dies on reload.
    model = json.loads(p.read_text(encoding="utf-8"))
    return jsonify(annotations.overlay(model, annotations.load_annotation(file)))


@app.post("/api/annotations/<file>")
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


@app.get("/api/annotations/<file>")
def get_annotations(file):
    return jsonify(annotations.load_annotation(file))


# ── sections annotées à la main (2026-08-07) ────────────────────────────────
# Louis annote lui-même les sections dans /reports/annotate.html pour donner une
# vraie vérité terrain, et `/api/annotations/<file>` ne sait pas les porter : il
# ne garde que `chords` et `merges` et jette tout le reste. D'où une route à
# part, volontairement bête — un fichier JSON par morceau, dernier écrit gagne,
# aucune interprétation côté serveur. Le stem est nettoyé avant de toucher au
# disque : il vient d'une page web.

SECTIONS_DIR = PKG / "state" / "sections"
#: Les gestes de l'outil du chart (brouillons) — séparés des 18
#: annotations faites à la main, qui sont la vérité terrain du projet.
SECTIONS_DRAFT_DIR = PKG / "state" / "sections_draft"


def _safe_stem(stem: str) -> str:
    return "".join(c for c in stem if c.isalnum() or c in "._-")[:120]


@app.post("/api/sections/<stem>")
def save_sections(stem):
    stem = _safe_stem(stem)
    if not stem:
        return jsonify({"error": "bad stem"}), 400
    doc = request.get_json(silent=True) or {}
    secs = doc.get("sections", [])
    if not isinstance(secs, list) or not all(isinstance(x, dict) for x in secs):
        # Il écrivait le fichier PUIS plantait en comptant : la vérité terrain
        # se retrouvait invalide sur disque avec un 500 côté client.
        return jsonify({"error": "sections doit être une liste d'objets"}), 400
    try:
        SECTIONS_DIR.mkdir(parents=True, exist_ok=True)
        (SECTIONS_DIR / f"{stem}.json").write_text(
            json.dumps({"stem": stem,
                        "n": doc.get("n"),
                        "validated": bool(doc.get("validated")),
                        "sections": doc.get("sections", [])},
                       ensure_ascii=False, indent=1))
    except (OSError, TypeError, ValueError) as exc:
        log.warning("sections save failed for %s: %s", stem, exc)
        return jsonify({"error": "could not persist sections"}), 500
    log.info("sections %s: %d", stem, len(doc.get("sections", [])))
    return jsonify({"ok": True, "stem": stem,
                    "count": len(doc.get("sections", []))})


@app.get("/api/sections/<stem>")
def get_sections(stem):
    p = SECTIONS_DIR / f"{_safe_stem(stem)}.json"
    if not p.exists():
        return jsonify({"stem": stem, "sections": []})
    try:
        return jsonify(json.loads(p.read_text()))
    except (OSError, ValueError):
        return jsonify({"stem": stem, "sections": []})


@app.get("/api/sections")
def all_sections():
    """Tout d'un coup — c'est ce que lisent les scripts d'analyse."""
    out = {}
    for p in sorted(SECTIONS_DIR.glob("*.json")) if SECTIONS_DIR.exists() else []:
        try:
            out[p.stem] = json.loads(p.read_text())
        except (OSError, ValueError):
            continue
    return jsonify(out)


# ── l'outil sections du chart (Louis, 2026-08-09) ───────────────────────────

@app.post("/api/section-repeats/<file>")
def api_section_repeats(file):
    """« Je viens de passer le doigt sur ces mesures : où ça se rejoue ? »

    Corps {b0, b1, claimed:[mesures déjà prises par les lettres validées],
    thr?, melody?} → la réponse de `section_tool.find_repeats`.

    Le stem de l'audio est relu dans le chart plutôt que reçu du client :
    c'est la même règle que /api/bar1, et ça évite qu'une page fabrique un
    chemin. Les postérieures musx sont en cache disque (clé = stem), donc
    l'appel tient largement dans un geste — la mélodie, elle, coûterait une
    séparation de voix et reste sur demande explicite.
    """
    p = CHARTS_DIR / f"{Path(file).stem}.json"
    if not p.exists():
        return jsonify({"error": "no such chart"}), 404
    body = request.get_json(silent=True) or {}
    if body.get("b0") is None or body.get("b1") is None:
        return jsonify({"error": "b0/b1 required"}), 400
    # Valider AVANT d'appeler le moteur : un type faux y levait une exception
    # rendue en 500 (page HTML), et le shell fait `r.json()` dessus — il
    # cassait sans rien afficher.
    try:
        b0 = int(body["b0"])
        b1 = int(body["b1"])
    except (TypeError, ValueError, OverflowError):
        return jsonify({"error": "b0/b1 doivent être des entiers"}), 400
    thr = body.get("thr")
    if thr is not None:
        try:
            thr = float(thr)
        except (TypeError, ValueError):
            return jsonify({"error": "thr doit être un nombre"}), 400
        if not 0.0 <= thr <= 1.0:
            return jsonify({"error": "thr doit être entre 0 et 1"}), 400
    claimed = body.get("claimed") or []
    if not isinstance(claimed, list):
        return jsonify({"error": "claimed doit être une liste de mesures"}), 400
    try:
        claimed = [int(x) for x in claimed]
    except (TypeError, ValueError):
        return jsonify({"error": "claimed doit contenir des entiers"}), 400
    model = json.loads(p.read_text(encoding="utf-8"))
    stem = Path(model.get("audio_url") or "").stem or \
        Path(file).stem.removeprefix("min_")
    audio = AUDIO_DIR / f"{stem}.m4a"
    try:
        from harmonia_min import musx as _musx
        from harmonia_min import section_tool as st
        triad = _musx.frame_posteriors(audio)[0]
        out = st.find_repeats(
            model["barGrid"], triad, b0, b1,
            audio=audio if body.get("melody") else None,
            melody=bool(body.get("melody")),
            thr=thr, claimed_bars=claimed)
    except Exception as exc:  # noqa: BLE001 — l'UI affiche l'erreur
        log.exception("section-repeats failed for %s", file)
        return jsonify({"error": f"repeat search failed: {exc}"}), 500
    out["file"] = Path(file).stem
    out["stem"] = stem
    out["n_bars"] = len(model["barGrid"]) - 1
    return jsonify(out)


def _sections_known(stem: str, model: dict) -> dict:
    """Ce qu'on sait déjà des sections de ce morceau, par ordre de confiance.

    Louis, 2026-08-09 : « même quand les sections sont écrites, on devrait
    pouvoir les modifier dans le même outil, et il devrait aussi être
    présent pour les chansons déjà annotées. » L'outil doit donc OUVRIR sur
    l'existant, jamais sur une page blanche — sinon modifier une annotation
    veut dire la refaire.

    Ordre : la vérité écrite à la main d'abord (`state/sections/`), puis le
    brouillon en cours (`sections_draft/`), puis, à défaut, ce que le
    détecteur a trouvé et que le chart affiche. Le dernier est le seul qui
    ne vient pas de lui : il est étiqueté comme tel pour que l'UI le dise.
    """
    # Le PLUS RÉCENT des deux gagne, pas la vérité par principe : l'outil
    # sauvegarde un brouillon à chaque geste, et donner systématiquement la
    # priorité au fichier validé faisait disparaître tout le travail de
    # correction dès qu'on refermait l'outil sans enregistrer (audit
    # 2026-08-10). La réponse dit toujours d'où ça vient.
    found = []
    for d, src in ((SECTIONS_DIR, "truth"), (SECTIONS_DRAFT_DIR, "draft")):
        f = d / f"{stem}.json"
        if not f.exists():
            continue
        try:
            doc = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        secs = [x for x in (doc.get("sections") or [])
                if isinstance(x, dict) and "b0" in x and "b1" in x]
        if secs:
            found.append((f.stat().st_mtime, src, secs,
                          bool(doc.get("validated"))))
    if found:
        found.sort(reverse=True)                      # le plus récent d'abord
        _, src, secs, validated = found[0]
        return {"source": src, "sections": secs, "validated": validated}
    # à défaut : les sections du chart lui-même, une entrée par passage
    out = []
    for s in model.get("sections", []):
        for rng in s.get("barRanges", []) or []:
            if len(rng) == 2:
                out.append({"label": s.get("label") or "?",
                            "b0": int(rng[0]), "b1": int(rng[1])})
    out.sort(key=lambda s: s["b0"])
    return {"source": "chart", "sections": out, "validated": False}


@app.get("/api/section-marks/<file>")
def api_section_marks_get(file):
    """Ce que l'outil doit afficher à l'ouverture (voir `_sections_known`)."""
    p = CHARTS_DIR / f"{Path(file).stem}.json"
    if not p.exists():
        return jsonify({"error": "no such chart"}), 404
    model = json.loads(p.read_text(encoding="utf-8"))
    stem = _safe_stem(Path(model.get("audio_url") or "").stem or
                      Path(file).stem.removeprefix("min_"))
    known = _sections_known(stem, model)
    known.update({"stem": stem, "n": len(model["barGrid"]) - 1})
    return jsonify(known)


@app.post("/api/section-marks/<file>")
def api_section_marks(file):
    """Les marques validées → le morceau écrit par sections.

    Corps {marks:[{label, occurrences:[{b0,b1}]}], validated?} → la liste
    de sections, ET son écriture dans `state/sections/<stem>.json`, le
    fichier que Louis remplit déjà à la main dans /reports/annotate.html.
    Même schéma, même endpoint de lecture, mêmes scripts de mesure en aval :
    l'outil du chart et la page d'annotation écrivent au même endroit,
    sinon deux vérités terrain divergentes coexisteraient.
    """
    p = CHARTS_DIR / f"{Path(file).stem}.json"
    if not p.exists():
        return jsonify({"error": "no such chart"}), 404
    model = json.loads(p.read_text(encoding="utf-8"))
    stem = _safe_stem(Path(model.get("audio_url") or "").stem or
                      Path(file).stem.removeprefix("min_"))
    n_bars = len(model["barGrid"]) - 1
    body = request.get_json(silent=True)
    if body is None or not isinstance(body, dict):
        # Un corps illisible valait « aucune marque » : la sauvegarde
        # automatique remplaçait alors le brouillon en cours par une liste
        # vide, donc effaçait le travail (audit 2026-08-10).
        return jsonify({"error": "corps JSON invalide"}), 400
    marks = body.get("marks")
    if marks is not None and not isinstance(marks, list):
        return jsonify({"error": "marks doit être une liste"}), 400
    from harmonia_min import section_tool as st
    sections = st.sections_from_marks(marks or [], n_bars)
    keep = [{"label": s["label"], "b0": s["b0"], "b1": s["b1"]}
            for s in sections if not s.get("pending")]
    # UN BROUILLON N'ÉCRASE PAS UNE VÉRITÉ TERRAIN. `state/sections/` porte
    # les 18 annotations faites à la main par Louis — la seule référence de
    # sections du projet, ce que lisent section_bench et section_metric.
    # L'outil sauvegarde à chaque geste ; sans séparation, le premier essai
    # remplaçait ses 7 sections de Bein Green par 23 cellules de 2 mesures
    # (constaté en test). Les gestes vont donc dans `sections_draft/`, et
    # seul un `validated: true` explicite touche la vraie annotation — en
    # gardant d'abord une copie `.bak` de ce qui était là.
    validated = bool(body.get("validated"))
    target_dir = SECTIONS_DIR if validated else SECTIONS_DRAFT_DIR
    doc = {"stem": stem, "n": n_bars, "validated": validated,
           "sections": keep}
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
        dest = target_dir / f"{stem}.json"
        bak = target_dir / f"{stem}.json.bak"
        if validated and dest.exists() and not bak.exists():
            # PREMIÈRE sauvegarde seulement : le .bak doit garder l'annotation
            # d'ORIGINE. L'écraser à chaque fois faisait qu'un deuxième
            # enregistrement détruisait définitivement le travail à la main.
            bak.write_text(dest.read_text(encoding="utf-8"), encoding="utf-8")
        dest.write_text(json.dumps(doc, ensure_ascii=False, indent=1))
    except (OSError, TypeError, ValueError) as exc:
        log.warning("section marks save failed for %s: %s", stem, exc)
        return jsonify({"error": "could not persist sections"}), 500
    log.info("outil sections %s: %d marque(s) → %d section(s) écrite(s) dans "
             "%s", stem, len(marks or []), len(keep), target_dir.name)
    return jsonify({"ok": True, "stem": stem, "n": n_bars,
                    "sections": sections, "written": len(keep),
                    "draft": not validated})


@app.route("/api/context_rescore/<file>", methods=["POST"])
@app.route("/api/reinfer/<file>", methods=["POST"])
def context_rescore(file):
    """Lock propagation: re-score the spans AROUND a locked chord.

    Boundaries never move and confirmed spans are hard evidence — see
    harmonia_min/context_rescore.py for the wire shape and the non-solves
    (notably: a changed span loses its seventh, QUAL5 only).

    /api/reinfer/ is aliased here on purpose: the shell's merge path still
    posts there, and a merge has no lock-lattice equivalent yet, so it gets
    the (correct) no-op answer instead of a 404 it silently swallows.
    """
    p = CHARTS_DIR / f"{Path(file).stem}.json"
    if not p.exists():
        return jsonify({"error": "no such chart"}), 404
    body = request.get_json(silent=True) or {}
    model = json.loads(p.read_text(encoding="utf-8"))
    try:
        from harmonia_min import context_rescore as cr
        out = cr.rescore(model, body.get("chords") or [],
                         body.get("confirms") or [])
    except Exception as exc:  # noqa: BLE001 — the shell swallows errors
        log.exception("context_rescore failed for %s", file)
        return jsonify({"error": f"rescore failed: {exc}"}), 500
    return jsonify(out)


@app.delete("/api/chart/<file>")
def delete_chart(file):
    p = CHARTS_DIR / f"{Path(file).stem}.json"
    if p.exists():
        p.unlink()
    annotations.delete_annotation(file)
    return jsonify({"ok": True})


# ── local search (the offline library flow) ─────────────────────────────────

SEARCH_PER_PAGE = 12


def _local_matches(words):
    out = []
    for p in sorted(AUDIO_DIR.glob("*.m4a")):
        hay = p.stem.lower()
        if words and all(w in hay for w in words):
            out.append({"id": f"local:{p.stem}",
                        "title": _pretty_title(p.stem),
                        "uploader": "déjà téléchargé", "duration": None,
                        "thumb": "", "local": True})
    return out


def _youtube_search(q, page, per):
    """Real YouTube search via the yt_dlp PYTHON api (the CLI shells out and
    costs ~1 s extra per call). `extract_flat` skips per-video extraction, so a
    page comes back in about a second.

    yt-dlp has no cursor: `ytsearchN:` always returns the first N. Paging is
    therefore "ask for (page+1)*per and drop what we already showed" — the
    flat call is cheap enough that this is fine at the depths a human scrolls.
    """
    want = (page + 1) * per
    opts = {"quiet": True, "no_warnings": True, "skip_download": True,
            "extract_flat": "in_playlist"}
    import yt_dlp
    with yt_dlp.YoutubeDL(opts) as y:
        info = y.extract_info(f"ytsearch{want}:{q}", download=False)
    entries = (info or {}).get("entries") or []
    out = []
    for e in entries[page * per:]:
        vid = e.get("id")
        if not vid:
            continue
        thumbs = e.get("thumbnails") or []
        out.append({
            "id": vid,
            "title": e.get("title") or vid,
            "uploader": e.get("uploader") or e.get("channel") or "YouTube",
            "duration": e.get("duration"),
            "thumb": (thumbs[0] or {}).get("url", "") if thumbs else "",
            "local": False,
        })
    return out, len(entries) >= want


@app.post("/api/yt-search")
def yt_search():
    """Local library FIRST (instant, offline, already downloaded), then real
    YouTube results underneath.

    Was local-only — which is why Louis could not find any new song (2026-08-04:
    "je ne peux pas trouver de nouveaux morceaux sur youtube, comme si on ne
    pouvait accéder qu'aux morceaux cachés"). That was milestone-1 scope, not a
    bug, but it made the search box look broken. Paged so the shell can scroll
    on indefinitely instead of stopping at whatever the library happened to hold.
    """
    body = request.get_json(silent=True) or {}
    # `q` et `page` viennent d'une page web : un type faux plantait la route
    # en 500 (audit 2026-08-10).
    q = str(body.get("q") or "").strip()
    try:
        page = max(0, int(body.get("page") or 0))
    except (TypeError, ValueError):
        page = 0
    if not q:
        return jsonify({"results": [], "hasMore": False, "page": page})
    words = [w for w in re.split(r"\W+", q.lower()) if w]
    results = _local_matches(words) if page == 0 else []
    has_more = False
    try:
        yt, has_more = _youtube_search(q, page, SEARCH_PER_PAGE)
        results += yt
    except Exception:  # noqa: BLE001 — offline or yt-dlp missing: local still works
        log.warning("yt-search: YouTube unreachable, serving local only",
                    exc_info=True)
    return jsonify({"results": results, "hasMore": has_more, "page": page})


# ── tablatures (delta2 §9): façade HTTP sur harmonia.tab_fetcher ─────────────
# La recherche est un vrai passage sur Ultimate Guitar (curl_cffi); si la
# dépendance ou le réseau manquent, on renvoie une liste vide — l'onglet reste
# visible et l'état « rien trouvé » du shell fait le travail (choix delta2).
# L'IMPORT, lui, n'est PAS une façade: une tablature n'a ni mesures ni temps,
# en faire un chart demande l'alignement audio (tab_aligner sert à annoter un
# chart existant, pas à en créer). Tant que ce n'est pas construit, la route
# répond honnêtement — le shell affiche le message tel quel.

@app.post("/api/tab-search")
def tab_search():
    body = request.get_json(silent=True) or {}
    q = (body.get("q") or "").strip()
    if not q:
        return jsonify({"results": []})
    try:
        from harmonia.tab_fetcher import search_tabs
        found = search_tabs(q, tab_types=("Chords",), max_results=12)
    except Exception:  # noqa: BLE001 — curl_cffi absent / UG down: liste vide
        log.warning("tab-search failed for %r", q, exc_info=True)
        return jsonify({"results": []})
    return jsonify({"results": [{
        "id": r.id,
        "title": r.song_name,
        "artist": r.artist_name,
        "kind": (r.tab_type or "").lower() == "chords" and "chords" or r.tab_type,
        "rating": round(r.rating, 1) if r.rating else None,
        "url": r.tab_url,
    } for r in found]})


@app.post("/api/tab-import")
def tab_import():
    # 200 exprès: le shell lit {error} et l'affiche tel quel — un 501 partirait
    # dans son catch générique et le message honnête se perdrait.
    return jsonify({"error": "l'import de tablatures n'est pas encore branché "
                             "— une tab n'a pas de mesures, il faut l'aligner "
                             "sur l'audio d'abord"})


# ── analyze + jobs ───────────────────────────────────────────────────────────

def _ytdlp_bin():
    """The yt-dlp sitting next to THIS interpreter: the venv's bin/ is only on
    PATH when the server was launched from an activated shell, and a plain
    `python -m harmonia_min.server` otherwise reports "yt-dlp not installed"
    while it is right there."""
    import shutil
    import sys
    cand = Path(sys.executable).parent / "yt-dlp"
    return str(cand) if cand.exists() else shutil.which("yt-dlp")


def _video_meta(ytdlp: str, url: str) -> tuple[str, str]:
    """(artiste, titre) d'une vidéo YouTube → ("", "") si on n'a rien pu lire.

    Louis, 2026-08-09 : « on a des codes à la place ». La cause était ici :
    quand yt-dlp télécharge, le fichier prend le nom de l'IDENTIFIANT de la
    vidéo, et le titre affiché était ce stem passé en capitales de titre —
    `J36Z7Anhvom`. Rien n'a jamais lu les métadonnées. Un appel de plus, deux
    secondes, et on a le vrai nom.

    Volontairement non fatal : l'analyse d'un morceau ne doit pas échouer parce
    qu'un champ de métadonnée manque. Le repli reste le stem, comme avant.
    """
    import subprocess
    try:
        r = subprocess.run(
            [ytdlp, "--skip-download", "--no-warnings", "--no-playlist",
             "--print", "%(title)s\t%(artist)s\t%(track)s\t%(uploader)s", url],
            capture_output=True, text=True, timeout=90)
        line = next((x for x in r.stdout.splitlines() if x.strip()), "")
        if not line:
            return "", ""
        f = (line.split("\t") + [""] * 4)[:4]
        return _titles.split(f[0], artist=f[1], track=f[2], uploader=f[3])
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("metadata lookup failed for %s: %s", url, exc)
        return "", ""


#: Les clients d'API YouTube essayés, dans l'ordre. yt-dlp en choisit un tout
#: seul ; quand celui-là se fait jeter, la commande entière échoue alors que
#: le suivant aurait marché. Constaté le 2026-08-10 (Louis) : « android vr »
#: a rendu `HTTP Error 403: Forbidden`, et exactement la même URL est passée
#: à la reprise, sans rien changer. C'est intermittent et côté YouTube — donc
#: ça se réessaie, ça ne se diagnostique pas.
YTDLP_CLIENTS = (None, "web_safari", "android", "ios", "tv")


def _download_audio(ytdlp: str, url: str, out: Path) -> None:
    """Télécharge l'audio, en réessayant avec un autre client YouTube.

    Lève une RuntimeError au message LISIBLE : l'échec précédent remontait
    jusqu'à l'écran de Louis sous la forme d'un `CalledProcessError` avec la
    ligne de commande complète, qui ne dit pas ce qui s'est passé ni quoi
    faire. La vraie cause (« 403 Forbidden ») était, elle, uniquement dans
    les logs du serveur.
    """
    import subprocess          # comme partout ailleurs dans ce fichier
    errors = []
    for client in YTDLP_CLIENTS:
        cmd = [ytdlp, "-f", "bestaudio[ext=m4a]/bestaudio",
               "--extract-audio", "--audio-format", "m4a",
               "--retries", "5", "--fragment-retries", "5",
               "-o", str(out)]
        if client:
            cmd += ["--extractor-args", f"youtube:player_client={client}"]
        cmd.append(url)
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode == 0 and out.exists():
            if client:
                log.info("yt-dlp: réussi avec le client %s", client)
            return
        tail = (r.stderr or r.stdout or "").strip().splitlines()
        msg = tail[-1] if tail else f"code {r.returncode}"
        errors.append(f"{client or 'défaut'}: {msg}")
        log.warning("yt-dlp: client %s a échoué — %s", client or "défaut", msg)
        for junk in AUDIO_DIR.glob(out.name + "*.part"):
            junk.unlink(missing_ok=True)     # sinon la reprise repart de rien
    joined = " | ".join(errors)
    if "403" in joined or "Forbidden" in joined:
        raise RuntimeError("YouTube a refusé le téléchargement (403) sur tous "
                           "les clients essayés. C'est passager : réessaie "
                           "dans un moment.")
    if "Private video" in joined or "Sign in" in joined:
        raise RuntimeError("Cette vidéo demande une connexion (privée ou "
                           "restreinte) — elle ne peut pas être téléchargée.")
    if "Video unavailable" in joined:
        raise RuntimeError("Cette vidéo n'est pas disponible.")
    raise RuntimeError(f"Le téléchargement a échoué. Détail : {joined}")


def _resolve_audio(url: str) -> tuple[Path, str, str]:
    """analyze URL → (audio path, title, artist). Three forms:
    'local:<stem>' (from our own search results, possibly wrapped in a
    youtube.com/watch?v= prefix by the untouched UI), a local stem typed
    directly, or a real YouTube URL (yt-dlp, if installed).

    L'artiste peut être vide : un slug de fichier a perdu le tiret qui séparait
    l'artiste du titre, et inventer la coupe serait pire que le champ vide (voir
    `harmonia_min.titles`). `scripts/backfill_titles.py` rattrape ces cas-là en
    retrouvant la vidéo d'origine.
    """
    m = re.search(r"local:([\w\-]+)", url)
    if m:
        p = AUDIO_DIR / f"{m.group(1)}.m4a"
        if p.exists():
            return p, _titles.pretty_from_slug(p.stem), ""
        raise FileNotFoundError(f"no local audio {m.group(1)}")
    stem = url.strip().strip("/")
    p = AUDIO_DIR / f"{Path(stem).stem}.m4a"
    if p.exists():
        return p, _titles.pretty_from_slug(p.stem), ""
    if re.search(r"youtu\.?be", url):
        import subprocess
        ytdlp = _ytdlp_bin()
        if not ytdlp:
            raise RuntimeError("yt-dlp not installed — paste a library song "
                               "name or install yt-dlp for YouTube links")
        vid = re.search(r"(?:v=|youtu\.be/)([\w\-]{6,})", url)
        out = AUDIO_DIR / (f"{vid.group(1)}.m4a" if vid else "download.m4a")
        if not out.exists():
            # `ytdlp`, pas "yt-dlp" : le chemin résolu juste au-dessus n'était
            # pas utilisé ici, ce qui annulait la raison d'être de _ytdlp_bin.
            _download_audio(ytdlp, url, out)
        artist, title = _video_meta(ytdlp, url)
        return out, (title or _titles.pretty_from_slug(out.stem)), artist
    raise FileNotFoundError(f"could not resolve {url!r} to audio")


def _run_job(job_id: str, url: str, bar1_time=None):
    """Le chart BRUT est publié dès qu'il existe ; le raffinement continue après.

    Louis, 2026-08-07 : « dès que le chart brut est dispo tu l'affiches direct,
    et le reste (gammes, harmonies, sections) tu le fais en background ».

    Le job expose donc deux jalons plutôt qu'un :

        chart_url  écrit au premier rendu, `status` restant "running" ;
                   l'app bascule dessus et laisse le sondage tourner.
        status     passe à "done" quand le modèle raffiné a ÉCRASÉ le même
                   fichier — même `file_key`, donc rien ne se dédouble.

    `refining` liste ce qui manque encore (l'app l'affiche discrètement dans le
    chart) et `refined_at` change de valeur exactement une fois, ce qui donne à
    l'app un front sur lequel recharger.

    Les deux écritures sont sérialisées ici, dans le thread du job, entre deux
    reprises du générateur : `bars` est muté sur place par le repli et par
    l'analyse harmonique, donc le brut DOIT être en JSON avant que la suite ne
    tourne (voir pipeline.analyze_steps).
    """
    job = _jobs[job_id]

    def progress(stage, **kw):
        job["stage"] = max(job.get("stage", 0), stage)
        job.update(kw)

    try:
        audio_path, title, artist = _resolve_audio(url)
        job["title"] = job.get("title") or title
        job["artist"] = artist
        import subprocess
        dur = float(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(audio_path)]).strip())
        job.update(stage=1, duration_s=dur)
        file_key = f"min_{audio_path.stem}"
        CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        dest = CHARTS_DIR / f"{file_key}.json"
        # Le sidecar est ce que /api/library ressert : c'est là que l'artiste
        # doit atterrir pour être VU. On n'écrase jamais une saisie de Louis —
        # l'éditeur artiste/titre de l'app écrit dans le même fichier.
        if artist or title:
            _meta = _load_chart_meta()
            if not (_meta.get(file_key) or {}).get("title"):
                _meta[file_key] = {"artist": artist, "title": title}
                try:
                    META_PATH.parent.mkdir(parents=True, exist_ok=True)
                    META_PATH.write_text(
                        json.dumps(_meta, ensure_ascii=False, indent=1),
                        encoding="utf-8")
                except OSError as exc:
                    log.warning("chart-meta autosave failed: %s", exc)
        t0 = time.time()
        for kind, model in analyze_steps(
                audio_path, title=job["title"], file_key=file_key,
                audio_url=f"/audio/{audio_path.name}", progress=progress,
                bar1_time=bar1_time):
            dest.write_text(json.dumps(model), encoding="utf-8")
            if kind == "raw":
                # UI refresh 2026-08-08 (§5): the raw ChartModel rides the job
                # itself — the loading screen renders it through the SAME
                # loadModel()/buildIReal() path the final chart uses, which is
                # what keeps the brut→final hand-off from moving a single
                # cell. phase="raw" therefore only ever appears with a
                # non-None raw_model (acceptance #1).
                n_chords = sum(1 for sec in model["sections"]
                               for bar in sec["bars"] for c in bar
                               if not c["nc"] and not c.get("carry"))
                job.update(chart_url=f"/chart/{file_key}",
                           refining=list(model["meta"].get("pending") or []),
                           raw_s=round(time.time() - t0, 2),
                           phase="raw", raw_model=model,
                           n_bars=model["nBars"], n_chords=n_chords)
                log.info("job %s: CHART BRUT en %.1f s → %s (%d mesures) ; "
                         "raffinement en cours",
                         job_id, time.time() - t0, file_key, model["nBars"])
            else:
                job.update(status="done", url=f"/chart/{file_key}",
                           chart_url=f"/chart/{file_key}", refining=[],
                           refined_at=round(time.time() - t0, 2), stage=6,
                           phase="done",
                           sections_found=len(model["sections"]))
                log.info("job %s done en %.1f s → %s (%d sections)",
                         job_id, time.time() - t0, file_key,
                         len(model["sections"]))
    except Exception as exc:
        log.exception("job %s failed", job_id)
        if job.get("chart_url"):
            # Le chart brut est DÉJÀ publié et l'app le montre peut-être déjà :
            # le passer en "error" effacerait un chart qui marche. On dit la
            # vérité — terminé, mais sans raffinement — et la trace complète
            # est dans le log ci-dessus (pas de repli silencieux).
            job.update(status="done", url=job["chart_url"], refining=[],
                       refine_error=str(exc), stage=6,
                       phase="done", sections_found=1)
        else:
            job.update(status="error", error=str(exc))


@app.post("/api/analyze")
def api_analyze():
    url = (request.get_json(silent=True) or {}).get("url", "")
    if not url:
        return jsonify({"error": "no url"})
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"status": "running", "stage": 0, "created": time.time(),
                     "title": ""}
    threading.Thread(target=_run_job, args=(job_id, url), daemon=True).start()
    return jsonify({"job_id": job_id})


@app.post("/api/bar1/<file>")
def api_bar1(file):
    """Set bar 1 (the in-app tool, 2026-08-08): re-lay the chart with the
    user's own bar-1 mark.

    Body {t}: the second the user put under the marker. The pipeline snaps it
    to the nearest tracked beat and takes that beat's PHASE — nothing before
    the mark is cut. Returns {job_id}: the same job machinery, so the shell
    shows the two-phase loading screen and the same file_key is rewritten.
    """
    t = (request.get_json(silent=True) or {}).get("t")
    if t is None:
        return jsonify({"error": "no t"})
    p = CHARTS_DIR / f"{Path(file).stem}.json"
    if not p.exists():
        return jsonify({"error": "no such chart"}), 404
    model = json.loads(p.read_text(encoding="utf-8"))
    stem = Path(model.get("audio_url") or "").stem or \
        Path(file).stem.removeprefix("min_")
    job_id = uuid.uuid4().hex[:12]
    _jobs[job_id] = {"status": "running", "stage": 0, "created": time.time(),
                     "title": model.get("title") or ""}
    threading.Thread(target=_run_job, args=(job_id, stem),
                     kwargs={"bar1_time": float(t)}, daemon=True).start()
    return jsonify({"job_id": job_id})


@app.get("/api/job/<job_id>")
def api_job(job_id):
    job = _jobs.get(job_id)
    if job is None:
        return jsonify({"status": "error", "error": "unknown job"}), 404
    return jsonify(job)


# ── on-device audio triage page (2026-08-02 iPhone stall) ───────────────────

AUDIOTEST_HTML = """<!DOCTYPE html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Test audio</title><style>
body{font-family:-apple-system,sans-serif;background:#1c1c1c;color:#eee;margin:0;padding:16px;line-height:1.5}
button{display:block;width:100%;margin:8px 0;padding:14px;border:none;border-radius:10px;font:600 15px -apple-system;background:#8a2b2b;color:#fff}
pre{background:#111;border-radius:10px;padding:10px;font:11px/1.5 Menlo,monospace;white-space:pre-wrap;min-height:120px}
audio{width:100%;margin:8px 0}</style></head><body>
<h3>Test audio — 4 variantes</h3>
<p style="font-size:13px;color:#aaa">Host: <b id="host"></b> · lance 1 puis 2 puis 3 puis 4, envoie la capture (bouton Copier).</p>
<button onclick="t1()">1 — Réseau brut (fetch 512 Ko)</button>
<button onclick="t2()">2 — Bip WebAudio (session audio)</button>
<div id="nat"></div>
<button onclick="t3()">3 — Lecteur natif &lt;audio controls&gt;</button>
<button onclick="t4()">4 — Comme l'app (new Audio + load/play)</button>
<button style="background:#444" onclick="navigator.clipboard.writeText(L.textContent)">Copier le rapport</button>
<pre id="log"></pre>
<script>
const SRC="/audio/maroon_5_this_love.m4a";
const L=document.getElementById("log");
document.getElementById("host").textContent=location.host;
const say=m=>{L.textContent+=(performance.now()/1000).toFixed(1)+"s  "+m+"\\n";};
say("UA: "+navigator.userAgent);
say("standalone: "+(navigator.standalone===true));
async function t1(){
  say("T1 fetch 512Ko…"); const st=performance.now(); let got=0;
  try{
    const ctl=new AbortController(); setTimeout(()=>ctl.abort(),15000);
    const r=await fetch(SRC,{headers:{Range:"bytes=0-524287"},signal:ctl.signal});
    say("T1 status "+r.status+" type "+r.headers.get("content-type"));
    const rd=r.body.getReader();
    for(;;){const{done,value}=await rd.read(); if(done)break; got+=value.length;}
    say("T1 OK: "+got+" octets en "+((performance.now()-st)/1000).toFixed(2)+"s");
  }catch(e){ say("T1 ECHEC après "+got+" octets: "+e); }
}
function t2(){
  try{
    const c=new (window.AudioContext||window.webkitAudioContext)();
    c.resume(); const o=c.createOscillator(),g=c.createGain();
    g.gain.value=.2; o.frequency.value=440; o.connect(g); g.connect(c.destination);
    o.start(); o.stop(c.currentTime+0.6);
    say("T2 bip lancé (tu dois l'entendre) state="+c.state);
  }catch(e){ say("T2 ECHEC: "+e); }
}
function t3(){
  const d=document.getElementById("nat"); d.innerHTML="";
  const a=document.createElement("audio");
  a.controls=true; a.preload="metadata"; a.src=SRC; a.setAttribute("playsinline","");
  d.appendChild(a); say("T3 lecteur natif affiché — tape SON play, attends 3 s");
  ["playing","waiting","stalled","suspend","canplay","error"].forEach(ev=>
    a.addEventListener(ev,()=>say("T3 "+ev+" ct="+a.currentTime.toFixed(2)+" buf="+(a.buffered.length?a.buffered.end(0).toFixed(1):"0"))));
}
function t4(){
  const a=new Audio(); a.preload="auto";
  a.playsInline=true; a.setAttribute("playsinline","");
  a.src=SRC; a.load();
  a.play().then(()=>say("T4 play() accepté")).catch(e=>say("T4 play() refusé: "+e));
  setTimeout(()=>say("T4 après 3s: ct="+a.currentTime.toFixed(2)+" rs="+a.readyState+" ns="+a.networkState+" buf="+(a.buffered.length?a.buffered.end(0).toFixed(1):"0")),3000);
}
</script></body></html>"""


@app.get("/audiotest")
def audiotest():
    """One-round-trip on-device triage: raw network throughput (T1) vs audio
    session (T2) vs native media stack (T3) vs the shell's JS pattern (T4).
    Splits Tailscale-path problems from standalone-PWA problems from our own
    player code — the 2026-08-02 iPhone stall reproduces on none of our
    desktop browsers, so the device itself has to tell us which layer dies."""
    return AUDIOTEST_HTML


# ── everything else: honest 404s the UI degrades on ─────────────────────────

@app.route("/api/<path:rest>", methods=["GET", "POST", "DELETE"])
def api_unimplemented(rest):
    return jsonify({"error": f"/api/{rest} is not part of harmonia_min "
                             "milestone 1"}), 404


def main():
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)
    log.info("harmonia_min server on http://localhost:%d (charts: %s)",
             PORT, CHARTS_DIR)
    app.run(host="0.0.0.0", port=PORT, debug=False, use_reloader=False)


if __name__ == "__main__":
    main()
