"""Lancer une analyse (YouTube / fichier local / micro), suivre son job, et
la recherche qui alimente la boîte "coller un lien ou chercher un titre".

Porté verbatim depuis `harmonia_min/server.py` (sprints 11-14). `_jobs` et
`_run_job` viennent de `jobs.py` ; `_youtube_search`/`_local_matches` de
`youtube.py`.

Ce que ce module ne fait PAS : télécharger l'audio lui-même (voir
`jobs._resolve_audio` → `youtube._download_audio`) ; savoir ce qu'un job
affiche à l'écran (c'est `app_shell.html`).
"""
from __future__ import annotations

import logging
import re
import subprocess
import time
from pathlib import Path

from flask import Blueprint, jsonify, request

from harmonia.server import youtube
from harmonia.server.jobs import CHARTS_DIR, _jobs, start_job
from harmonia.settings import SETTINGS

log = logging.getLogger("harmonia.server.routes.analyze")

bp = Blueprint("analyze", __name__)

AUDIO_DIR = SETTINGS.audio_dir


@bp.post("/api/analyze")
def api_analyze():
    url = (request.get_json(silent=True) or {}).get("url", "")
    if not url:
        return jsonify({"error": "no url"})
    job_id = start_job(url)
    return jsonify({"job_id": job_id})


@bp.post("/api/bar1/<file>")
def api_bar1(file):
    """Set bar 1 (the in-app tool, 2026-08-08): re-lay the chart with the
    user's own bar-1 mark.

    Body {t}: the second the user put under the marker. The pipeline snaps it
    to the nearest tracked beat and takes that beat's PHASE — nothing before
    the mark is cut. Returns {job_id}: the same job machinery, so the shell
    shows the two-phase loading screen and the same file_key is rewritten.
    """
    import json
    t = (request.get_json(silent=True) or {}).get("t")
    if t is None:
        return jsonify({"error": "no t"})
    p = CHARTS_DIR / f"{Path(file).stem}.json"
    if not p.exists():
        return jsonify({"error": "no such chart"}), 404
    model = json.loads(p.read_text(encoding="utf-8"))
    stem = Path(model.get("audio_url") or "").stem or \
        Path(file).stem.removeprefix("min_")
    job_id = start_job(stem, title=model.get("title") or "", bar1_time=float(t))
    return jsonify({"job_id": job_id})


@bp.get("/api/job/<job_id>")
def api_job(job_id):
    job = _jobs.get(job_id)
    if job is None:
        return jsonify({"status": "error", "error": "unknown job"}), 404
    return jsonify(job)


def _ffmpeg(src: Path, dest: Path, args: list[str], *, timeout: int) -> bool:
    """Transcode, ou False (jamais d'exception) — le format que MediaRecorder
    produit dépend du navigateur : webm/opus sur Chrome, mp4/aac sur Safari."""
    try:
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src),
                        *args, str(dest)],
                       check=True, capture_output=True, timeout=timeout)
        return dest.exists() and dest.stat().st_size > 0
    except (OSError, subprocess.SubprocessError) as exc:
        log.warning("ffmpeg %s → %s a échoué : %s", src.name, dest.name, exc)
        return False


@bp.post("/api/record-analyze")
def api_record_analyze():
    """Un enregistrement micro entre dans la bibliothèque comme un morceau.

    Le fichier atterrit dans docs/audio/ sous un stem UNIQUE et repart dans le
    MÊME `_run_job` que YouTube : même écran de chargement, même chart, et
    l'audio est rejouable sous la tête de lecture (`/audio/<stem>.m4a`).

    Le stem unique n'est pas cosmétique : battues, CQT et postérieures musx
    sont TOUS cachés par stem de fichier. Deux enregistrements sous le même nom
    et le second se verrait servir les accords du premier (mesuré en juillet
    sur l'ancienne app — un clip de 45 s avait tronqué l'analyse d'un morceau
    de 283 s).
    """
    f = request.files.get("audio")
    if f is None:
        return jsonify({"error": "Aucun audio reçu"}), 400
    stem = f"rec_{int(time.time() * 1000)}"
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    raw = AUDIO_DIR / f"{stem}.upload"
    f.save(raw)
    dest = AUDIO_DIR / f"{stem}.m4a"
    ok = _ffmpeg(raw, dest, ["-ac", "1", "-ar", "44100", "-c:a", "aac",
                             "-b:a", "128k"], timeout=120)
    raw.unlink(missing_ok=True)
    if not ok:
        return jsonify({"error": "Enregistrement illisible (transcodage "
                                 "impossible)"}), 400

    title = (request.form.get("title") or "").strip() or \
        "Enregistrement du " + time.strftime("%d/%m à %H:%M")
    job_id = start_job(stem, title=title)
    return jsonify({"job_id": job_id})


@bp.post("/api/yt-search")
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
    results = youtube._local_matches(words) if page == 0 else []
    has_more = False
    try:
        yt, has_more = youtube._youtube_search(q, page, youtube.SEARCH_PER_PAGE)
        results += yt
    except Exception:  # noqa: BLE001 — offline or yt-dlp missing: local still works
        log.warning("yt-search: YouTube unreachable, serving local only",
                    exc_info=True)
    return jsonify({"results": results, "hasMore": has_more, "page": page})


# Ce que ce module ne fait PAS : les routes de section/annotation/jam/iReal —
# voir les fichiers de `routes/` du même nom.
