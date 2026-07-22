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

from harmonia.serving.config import AUDIO_DIR, PLOTS_DIR, PWA_DIR, REPO
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


# ---------------------------------------------------------------------------
# Batch 2 (Phase 6c, debug-route group): the read-only /debug/* file-serve
# routes. Each serves a pre-built, self-contained static HTML file straight off
# disk (scratchpad/*.html) — no server-side templating (except the single
# text-substitution in /debug/bar-merge-game), so the served page can't drift
# from what was actually reviewed. Every one depends only on a dedicated,
# pure module-level Path constant (moved alongside it here, built from
# ``REPO`` out of harmonia.serving.config) plus flask.Response. Moved VERBATIM
# out of scripts/harmonia_server.py: same bodies, same guards, same docstrings;
# only ``@app.route`` -> ``@api.route``.
#
# LEFT in the server this round (not clean file-serves): ``/debug/structure``
# (needs the logic helper ``_seg_divs_from_runs`` + ``_STRUCT_COLORS`` + json,
# not a pure constant); ``/debug/section-align`` (runs ffmpeg + a LIVE
# ireal->audio alignment, and depends on the server-owned ``lookup_slug``
# helper — not read-only/deterministic); ``/sw.js`` (needs ``_APP_SHELL``,
# a constant shared with the ``/`` index route that stays in the server).
# ---------------------------------------------------------------------------

# NEW debug routes (2026-07-18, chord-distance work — same authorization as
# /debug/structure above). These serve pre-built, self-contained static HTML
# files straight off disk — no server-side templating, so they can't drift
# from what was actually reviewed. Nothing else touched.
_METRIC_ARTIFACT_HTML = REPO / "scratchpad" / "structure_metric_artifact.html"
_SSM_VIZ_HTML = REPO / "scratchpad" / "bar_ssm_viz.html"


@api.route("/debug/metric-artifact")
def debug_metric_artifact():
    """V-measure block-level-vs-per-bar granularity artifact chart (4 iReal
    songs) — see docs/known_issues.md 'CORRECTION: the 0.732 clean-GT oracle'."""
    if not _METRIC_ARTIFACT_HTML.exists():
        return Response("not generated yet", status=404)
    return Response(_METRIC_ARTIFACT_HTML.read_bytes(), mimetype="text/html")


@api.route("/debug/ssm")
def debug_ssm():
    """Bar-to-bar chord-tone-distance self-similarity matrices (one clean
    iReal chart with GT, one real-audio song with raw NNLS chroma) — see
    docs/known_issues.md 'Hand-crafted CHORD-TONE-DISTANCE similarity'."""
    if not _SSM_VIZ_HTML.exists():
        return Response("not generated yet", status=404)
    return Response(_SSM_VIZ_HTML.read_bytes(), mimetype="text/html")


_SSM_MULTIGRAIN_HTML = REPO / "scratchpad" / "bar_ssm_multigrain_viz.html"


@api.route("/debug/ssm-multigrain")
def debug_ssm_multigrain():
    """Same two songs as /debug/ssm, but self-similarity at 5 granularities
    (1/2/4/8/16-bar blocks) side by side per song."""
    if not _SSM_MULTIGRAIN_HTML.exists():
        return Response("not generated yet", status=404)
    return Response(_SSM_MULTIGRAIN_HTML.read_bytes(), mimetype="text/html")


_DUAL_MATRIX_HTML = REPO / "scratchpad" / "dual_matrix_viz.html"


@api.route("/debug/dual-matrix")
def debug_dual_matrix():
    """Audio vs structural (decoded-chord) similarity matrices side by side
    at 8-bar grain, for the 3 real songs, plus the inferred section labels
    from clustering both together — see docs/known_issues.md ★ STRUCTURE /
    SEGMENTATION, 2026-07-18, the section-repeat-ranking diagnostic."""
    if not _DUAL_MATRIX_HTML.exists():
        return Response("not generated yet", status=404)
    return Response(_DUAL_MATRIX_HTML.read_bytes(), mimetype="text/html")


_CRITERIA_VIZ_HTML = REPO / "scratchpad" / "criteria_viz.html"


@api.route("/debug/criteria")
def debug_criteria():
    """Side-by-side comparison of 3 candidate section-matching criteria
    (all built on the Mantel-validated dual-matrix), at k=3/4/5, with the
    <=5-distinct-sections rule and the block0/block1 sanity check per
    criterion/song — see docs/known_issues.md ★ STRUCTURE / SEGMENTATION,
    2026-07-18, "no more than 4-5 sections" constraint work."""
    if not _CRITERIA_VIZ_HTML.exists():
        return Response("not generated yet", status=404)
    return Response(_CRITERIA_VIZ_HTML.read_bytes(), mimetype="text/html")


_K_PRIOR_VIZ_HTML = REPO / "scratchpad" / "k_prior_viz.html"


@api.route("/debug/k-prior")
def debug_k_prior():
    """Learned prior P(k|song_length_bars) from the full 1992-tune iReal
    corpus, combined with the silhouette clustering-quality signal into a
    principled k-selection rule — corpus-scale validation + the 3 real
    songs' chosen k, plotted against the corpus scatter. See
    docs/known_issues.md ★ STRUCTURE / SEGMENTATION, 2026-07-18."""
    if not _K_PRIOR_VIZ_HTML.exists():
        return Response("not generated yet", status=404)
    return Response(_K_PRIOR_VIZ_HTML.read_bytes(), mimetype="text/html")


_BARGRID_PLAYER_HTML = REPO / "scratchpad" / "bargrid_debug_player.html"


@api.route("/debug/bargrid-player")
def debug_bargrid_player():
    """Real waveform + the exact beat_grid()-derived bar timestamps the
    production chart uses, with a synced audio playhead and click-to-seek —
    built so the user can personally listen through a song end-to-end and
    verify by ear/eye whether the bar lines actually land on the downbeats,
    per the 2026-07-19 'la derivation des barres n'est pas du tout bonne'
    report even under fairly constant tempo. See docs/known_issues.md
    ★ CHART / BAR-GRID."""
    if not _BARGRID_PLAYER_HTML.exists():
        return Response("not generated yet", status=404)
    return Response(_BARGRID_PLAYER_HTML.read_bytes(), mimetype="text/html")


_SECTION_SUGGESTION_PROTOTYPE_HTML = REPO / "scratchpad" / "section_suggestion_prototype.html"


@api.route("/debug/section-suggestions")
def debug_section_suggestions():
    """PROTOTYPE ONLY, not wired into the live app — proposed human-confirm
    UI for section-STRUCTURE suggestions (root vs chord-tone disagreements),
    2026-07-21. Deliberately a different visual pattern from the existing
    gold bar-merge cell outlines / violet section-repeat badges in
    app_shell.html: this is a relabel-a-stretch decision, not a span-merge
    one, so it's two stacked timeline "ribbons" (current vs suggested) with
    the disagreement bracketed and the specific chord evidence shown. Real
    candidates from scratchpad/section_structure_candidates.py's corpus
    scan, not invented data. See docs/known_issues.md ★ STRUCTURE."""
    if not _SECTION_SUGGESTION_PROTOTYPE_HTML.exists():
        return Response("not generated yet", status=404)
    return Response(_SECTION_SUGGESTION_PROTOTYPE_HTML.read_bytes(), mimetype="text/html")


_REAL_TRANSFER_HTML = REPO / "scratchpad" / "real_transfer_viz.html"
_GRID_ALIGN_HTML = REPO / "scratchpad" / "grid_align_debug.html"


@api.route("/debug/merge-criterion")
def debug_merge_criterion():
    """2026-07-18 overnight continuation: Steps 2/3/4's bar-merge criterion,
    intro detector, and section detector (all trained/validated on clean
    iReal, see docs/known_issues.md "Step 2"/"Step 3"/"Step 4" entries)
    transferred to the 3 real-audio songs with NO ground truth — qualitative
    human-inspection page only, per the brief's explicit "human validation"
    requirement for the real-audio transfer step. Pre-built static HTML,
    same off-disk-serving pattern as /debug/ssm and /debug/metric-artifact
    above (can't drift from what was actually reviewed)."""
    if not _REAL_TRANSFER_HTML.exists():
        return Response("not generated yet", status=404)
    return Response(_REAL_TRANSFER_HTML.read_bytes(), mimetype="text/html")


@api.route("/debug/grid-align")
def debug_grid_align():
    """Interactive audio+waveform grid-alignment diagnostic (2026-07-20).

    Lets the user personally listen through any cached song with FOUR
    overlaid hypotheses (raw detected beats / stock uniform grid / bestfit
    grid / the chart actually shown today) and a synced, click-to-seek
    playhead — so "the grid still looks wrong" can be confirmed or refuted
    by eye+ear on the real audio, not by a self-reported offline metric."""
    if not _GRID_ALIGN_HTML.exists():
        return Response("grid-align page not found", status=404)
    return Response(_GRID_ALIGN_HTML.read_bytes(), mimetype="text/html")


_BAR_MERGE_GAME_HTML = REPO / "scratchpad" / "bar_merge_game.html"
_BAR_MERGE_GAME_DATA = REPO / "scratchpad" / "bar_merge_game_data.json"


@api.route("/debug/bar-merge-game")
def debug_bar_merge_game():
    """2026-07-18 chord-robustness reframe: interactive "pairs game" for
    confirming candidate bar-merges (from scratchpad/bar_merge_candidates.py,
    threshold+pairs on the untrained 1-bar raw-chroma SSM — see
    docs/known_issues.md "REFRAME: bar-merge SSM pooling") and POSTing
    confirmed spans to the EXISTING /api/reinfer/<filename> merge-pooling
    endpoint (harmonia.models.user_constraints.pool_beat_evidence).
    Separate new debug route per the user's explicit instruction NOT to
    edit chart_interactive.py's existing manual merge UI for this — same
    self-contained-HTML-off-disk pattern as every other /debug/* route
    tonight, candidate data precomputed (scratchpad/bar_merge_game_data.json,
    build via scratchpad/bar_merge_candidates.py) and templated in once at
    request time so the served page can't drift from what was reviewed."""
    if not _BAR_MERGE_GAME_HTML.exists() or not _BAR_MERGE_GAME_DATA.exists():
        return Response("not generated yet — run scratchpad/bar_merge_candidates.py "
                        "and rebuild bar_merge_game_data.json", status=404)
    html = _BAR_MERGE_GAME_HTML.read_text()
    data = _BAR_MERGE_GAME_DATA.read_text()  # already valid JSON text
    html = html.replace("__CANDIDATE_DATA__", data)
    return Response(html, mimetype="text/html")
