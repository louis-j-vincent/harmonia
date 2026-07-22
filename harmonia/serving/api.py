"""harmonia/serving/api.py — SAFE-GET serving routes as a Flask Blueprint (Phase 6e).

First batch of the serving-layer route extraction: read-only routes whose every
dependency already lives in an extracted leaf module (``config`` paths,
``render._chart_model_for``) or is a self-contained file-serve. Moved VERBATIM
out of ``scripts/harmonia_server.py`` — same bodies, same guards, same
docstrings; only the decorator changed from ``@app.route`` to ``@api.route``.

Behavior-preservation contract (this is the delicate one — blueprint
registration, endpoint names, url_for):

  * The blueprint is registered in ``scripts/harmonia_server.py`` with
    ``app.register_blueprint(api, name="")``. The EMPTY name is load-bearing:
    Flask computes each endpoint as
    ``f"{name_prefix}.{name}.{endpoint}".lstrip(".")`` (flask/sansio/
    blueprints.py), so an empty name yields the BARE endpoint — ``serve_audio``,
    ``api_library``, … — exactly as the original ``@app.route`` produced. The
    resulting ``app.url_map`` is therefore byte-for-byte identical (same rules,
    same endpoints, same methods), and any ``url_for("serve_audio")`` still
    resolves. A NON-empty name would namespace them to ``api.serve_audio`` and
    silently break both the url_map identity and url_for.
  * ``_chart_model_for`` is imported from ``harmonia.serving.render`` (the SAME
    live object the server uses); it still lazily binds its remaining
    server-owned deps at call time, unchanged.

Routes that LOOK movable but were deliberately LEFT in the server this round:
``/gt-chart`` (needs the server-owned ``_PWA_HEAD`` constant plus the
``_inject_overlay`` / ``_inject_back_button`` helpers, none of which are
extracted yet).
"""

from __future__ import annotations

import logging
import re
from urllib.parse import quote

from flask import Blueprint, Response, jsonify, redirect, send_from_directory

from harmonia.serving.config import AUDIO_DIR, PLOTS_DIR, PWA_DIR
from harmonia.serving.render import _chart_model_for

log = logging.getLogger(__name__)

api = Blueprint("api", __name__)


@api.route("/audio/<path:filename>")
def serve_audio(filename):
    """Serve downloaded song audio for the docked player. conditional=True
    (Flask's default) handles Range requests, which iOS Safari needs to
    seek in an <audio> element without redownloading the whole file."""
    p = AUDIO_DIR / filename
    if not p.exists() or p.parent != AUDIO_DIR:
        return "Not found", 404
    # Python's mimetypes module guesses "audio/mp4a-latm" for .m4a on some
    # systems — force the standard type iOS Safari expects for AAC/m4a.
    mimetype = "audio/mp4" if p.suffix == ".m4a" else None
    resp = send_from_directory(AUDIO_DIR, filename, conditional=True, mimetype=mimetype)
    # iOS Safari puts <audio crossorigin="anonymous"> media into CORS mode and
    # validates EVERY Range (206) response for an Access-Control-Allow-Origin
    # header — even same-origin. send_from_directory doesn't add it, so WebKit
    # silently taints the resource and playback (native Play button) does
    # nothing. Desktop Chromium is lenient about same-origin and plays anyway,
    # which masked this. Emit ACAO so any crossOrigin media element works.
    resp.headers["Access-Control-Allow-Origin"] = "*"
    return resp


@api.route("/pwa/<path:filename>")
def serve_pwa_asset(filename):
    """Serve the PWA manifest and home-screen icons."""
    p = PWA_DIR / filename
    if not p.exists() or p.parent != PWA_DIR:
        return "Not found", 404
    mimetype = "application/manifest+json" if p.suffix == ".json" else None
    return Response(p.read_bytes(), mimetype=mimetype)


@api.route("/api/library")
def api_library():
    """Every chart we have, as library cards (title, key, bars, has-audio)."""
    from harmonia.output.chart_model import chart_summary

    charts = []
    for p in sorted(PLOTS_DIR.glob("inferred_*.html")):
        try:
            c = chart_summary(_chart_model_for(p.name, include_gt=False))
            c["mtime"] = p.stat().st_mtime  # lets the client offer a Recent sort too
            charts.append(c)
        except (OSError, ValueError, KeyError) as e:
            log.warning("Skipping %s in library: %s", p.name, e)
    # newest first — the chart you just analysed should be at the top
    charts.sort(key=lambda c: c["mtime"], reverse=True)
    return jsonify(charts=charts)


@api.route("/api/chart-model/<filename>")
def api_chart_model(filename):
    """The one clean shape the app UI consumes — see chart_model.to_chart_model."""
    p = PLOTS_DIR / filename
    if not p.exists() or p.suffix != ".html" or p.parent != PLOTS_DIR:
        return jsonify(error="Not found"), 404
    try:
        return jsonify(_chart_model_for(filename))
    except (OSError, ValueError, KeyError) as e:
        log.exception("chart-model failed for %s", filename)
        return jsonify(error=str(e)), 500


@api.route("/api/irealb-export/<filename>")
def api_irealb_export(filename):
    """Export a chart to an irealb:// URL (Share button, 2026-07-20) — the
    reverse of the existing iReal IMPORT path. iReal Pro opens this link
    directly (or it can be pasted into the app)."""
    p = PLOTS_DIR / filename
    if not p.exists() or p.suffix != ".html" or p.parent != PLOTS_DIR:
        return jsonify(error="Not found"), 404
    try:
        from harmonia.irealb_export import chart_model_to_irealb_url
        model = _chart_model_for(filename, include_gt=False)
        url = chart_model_to_irealb_url(model)
        return jsonify(url=url)
    except Exception as e:  # noqa: BLE001 — export is best-effort, never 500 the chart
        log.exception("irealb export failed for %s", filename)
        return jsonify(error=str(e)), 500


@api.route("/chart/<filename>")
def serve_chart(filename):
    """Serve a chart HTML file with the YouTube overlay injected.

    2026-07-17: also re-derives the embedded ``const P = {...};`` payload
    under any saved bar-1 offset before serving. This route reads the baked
    HTML straight off disk (``p.read_text()``) — it is a SEPARATE code path
    from `_chart_model_for`/`/api/chart-model/<file>` (used by the SPA at
    "/"), which already applied `_apply_bar1_offset_to_payload`. Bug: a user
    who saves a bar-1 offset via /bar1-offset-fix and then lands here (its
    own "← chart" link points at this exact route) saw the ORIGINAL,
    un-shifted chart — the offset was real and persisted, but this route
    never read it. Mirrors the same re-derivation `_chart_model_for` does,
    but as a text substitution on the embedded payload rather than a
    structured API response, since this route serves the standalone
    self-contained chart page (chart_interactive.py's own client JS reads
    `P` directly, not `/api/chart-model`).

    2026-07-18: user complaint — following a `/chart/<file>` link (e.g. from
    the align tool, a swipe-nav, or a shared link) lands on this route's own
    baked-HTML rendering, which is a structurally separate, plainer UI than
    the SPA's own chart view at "/" (`API.build`/`openChart` via
    `/api/chart-model`) even though both show the same "Read/Analyse/
    Annotate" control — two divergent code paths for the same content (see
    docs/known_issues.md's two prior "old UI" entries, both patches WITHIN
    this route rather than a structural fix). Durable fix: this route now
    redirects to `/?open=<file>`, a new deep-link param `app_shell.html`'s
    `API.build` reads to call `openChart(file)` directly instead of landing
    on the library first — so any old-style `/chart/<file>` link now opens
    straight into the polished SPA showing that exact song.

    2026-07-18, later same day: the redirect briefly carried a content-check
    EXCEPTION for `inferred_autumn_leaves.html`, the one baked chart with a
    bar-merge-suggestions overlay string-patched directly into its static
    HTML (`#suggest-mode-btn`/`.sugg-badge`) that hadn't been ported to the
    SPA's own JS-driven chart renderer yet. User complaint #2 ("Toujours
    l'interface dégueu", still landing on the ugly chooser page for THIS
    song) made clear that exception was no longer acceptable — the user
    wants no ugly page anywhere, full stop. Fix: the bar-merge-suggestions
    overlay is now ALSO implemented natively in `app_shell.html` (see its
    own "BAR-MERGE SUGGESTIONS" comment block — `S.suggMode`/
    `suggestToggleBtn`/`paintSuggestions`/`openSuggestionSheet`, same
    `/api/bar-merge-candidates/<file>` + `/api/reinfer/<file>` endpoints,
    same preview-only semantics), so every song's suggestions are reachable
    from the SPA now, not just this one manually-patched static file. The
    exception is gone: this route redirects UNCONDITIONALLY (after the
    existence check below) — no content sniffing, no per-file exemption.
    The baked-HTML rendering pipeline that used to run below this point
    (bar-1 offset re-derivation, PWA head injection, YouTube-overlay
    injection, swipe-nav) is now dead for every file and has been removed;
    if a future feature needs the standalone baked-HTML path again, restore
    it from git history (this docstring's prior revision) rather than
    re-adding a content-sniffed exception here."""
    p = PLOTS_DIR / filename
    if not p.exists() or not p.suffix == ".html":
        return "Not found", 404
    return redirect(f"/?open={quote(filename)}", code=302)


@api.route("/diagnostics/<filename>")
def serve_diagnostic(filename):
    """Serve diagnostic HTML files from docs/plots/."""
    # Sanitize filename to prevent directory traversal
    filename = re.sub(r"[^A-Za-z0-9_\-.]", "", filename)
    p = PLOTS_DIR / filename
    if not p.exists() or not p.suffix == ".html":
        return f"<p>Diagnostic {filename} not found</p>", 404
    return send_from_directory(PLOTS_DIR, filename, mimetype="text/html")
