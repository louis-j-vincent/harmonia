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

Later rounds moved the two read-only page routes whose deps became fully
extracted: ``/gt-chart`` (``_PWA_HEAD`` / ``_inject_overlay`` /
``_inject_back_button`` now live in ``render``; ``_yt_video_ids`` /
``_yt_audio_meta`` in ``state``; ``lookup_slug`` in ``cache``) and
``/demo/progressive-analysis`` (config ``REPO`` + a self-contained file-serve).

Routes that LOOK movable but were deliberately LEFT in the server: ``/gt-align``
(still needs the server-owned ``_waveform_peaks`` audio-envelope helper, which
is not an extracted module and is out of the data-loader scope).
"""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from urllib.parse import quote

from flask import Blueprint, Response, jsonify, redirect, request, send_from_directory

from harmonia.serving.cache import lookup_slug
from harmonia.serving.config import AUDIO_DIR, PLOTS_DIR, PWA_DIR, REPO
from harmonia.serving.render import (
    _chart_model_for,
    _PWA_HEAD,
    _inject_back_button,
    _inject_overlay,
)
from harmonia.serving.loaders import _annot_path
from harmonia.serving.state import (
    _remember_annotation,
    _remember_video_id,
    _save_section_labels,
    _training_log_dir,
    _YT_AUDIO_FILE,
    _YT_IDS_FILE,
    _yt_audio_meta,
    _yt_video_ids,
)

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


# ---------------------------------------------------------------------------
# Read-only page routes (serving refactor, loaders round). Both are now fully
# unblocked — every dependency lives in an extracted leaf module: config
# (REPO / PLOTS_DIR / AUDIO_DIR), state (_yt_video_ids / _yt_audio_meta),
# render (_PWA_HEAD / _inject_overlay / _inject_back_button), cache
# (lookup_slug). Moved VERBATIM out of scripts/harmonia_server.py: same bodies,
# same guards, same docstrings; only ``@app.route`` -> ``@api.route``.
# ---------------------------------------------------------------------------


@api.route("/demo/progressive-analysis")
def demo_progressive_analysis():
    """Standalone, self-contained mockup of the proposed progressive-analysis
    screen (draft NNLS chords filling in, then corrected by music-x-lab) —
    for viewing on a real phone over the VPN. Not part of the app; served
    straight from scratchpad, no build step. Remove once the design is
    settled or ported into app_shell.html for real."""
    p = REPO / "scratchpad" / "progressive_analysis_demo.html"
    if not p.exists():
        return jsonify(error="demo file not found"), 404
    return Response(p.read_text(encoding="utf-8"), mimetype="text/html")


@api.route("/gt-chart")
def gt_chart():
    """Serve iReal ground-truth chart with YouTube video sync.

    ?song=<slug>  →  displays irealb_<slug>.html (ground truth) with YouTube/audio playback
    """
    slug = lookup_slug(request.args.get("song") or "autumn_leaves")
    filename = f"irealb_{slug}.html"
    p = PLOTS_DIR / filename

    if not p.exists():
        return f"<p>No iReal chart for {slug}</p>", 404

    content = p.read_text(encoding="utf-8")
    content = content.replace("</head>", _PWA_HEAD + "</head>", 1)

    # Inject YouTube video ID if available
    vid = _yt_video_ids.get(f"inferred_{slug}.html", "")
    if vid:
        content = content.replace(
            "</head>",
            f'<script>window.YT_VIDEO_ID="{vid}"; window.PAGE_TITLE="GT: {slug}";</script></head>',
            1,
        )

    # Inject audio metadata
    audio_meta = _yt_audio_meta.get(f"inferred_{slug}.html")
    if audio_meta and (AUDIO_DIR / Path(audio_meta["audio"]).name).exists():
        content = content.replace(
            "</head>",
            '<script>window.HARM_AUDIO_URL=' + json.dumps(audio_meta["audio"])
            + ';window.HARM_THUMB_URL=' + json.dumps(audio_meta.get("thumb", ""))
            + ';</script></head>',
            1,
        )

    # Add banner: "This is ground truth (iReal), not model inference"
    banner = '''<div style="position:fixed;top:0;right:0;background:#00c9a7;color:#0e1116;padding:8px 12px;font-size:11px;font-weight:700;z-index:100;border-radius:0 0 0 6px;">🎼 GROUND TRUTH (iReal)</div>'''
    content = content.replace("<body>", "<body>" + banner, 1)

    return Response(_inject_back_button(_inject_overlay(content)), mimetype="text/html")


# ---------------------------------------------------------------------------
# Mutating routes (serving refactor, mutating-routes round). MOVED VERBATIM out
# of scripts/harmonia_server.py -- function bodies byte-identical to HEAD, the
# ONLY change per route being the @app.route -> @api.route decorator. Gated by
# move-identity + url_map identity (they mutate state / hit the network, so
# they can't be byte-diffed over HTTP). Deps are all extracted leaves
# (config/state/loaders) or lazy in-body harmonia imports; no server-owned or
# grid/offset helper is dragged. SKIPPED this round: POST /api/gt-offset
# (_save_gt_offset clears the billboard GT cache), POST /api/bar1-offset
# (_bar1_offset_bounds is an offset helper, off-limits). tab-align and
# irealb-search reach the network in-body, so they were proven by move-identity
# only, not smoke-run.
# ---------------------------------------------------------------------------


@api.route("/api/chart/<filename>", methods=["DELETE"])
def api_delete_chart(filename):
    """Remove a chart from the library (UX audit 2026-07-20: there was no way
    to delete or reorganize charts once analysed). Removes the chart HTML plus
    its registry entries (video-id link, retained-audio link, annotation
    sidecar) so nothing dangling is left behind; the underlying downloaded
    audio in docs/audio/ is NOT deleted (it may be cheap to keep and re-used
    if the same video is analysed again)."""
    p = PLOTS_DIR / filename
    if p.suffix != ".html" or p.parent != PLOTS_DIR or not p.name.startswith("inferred_"):
        return jsonify(error="Not found"), 404
    if not p.exists():
        return jsonify(error="Not found"), 404
    try:
        p.unlink()
    except OSError as e:
        return jsonify(error=str(e)), 500
    if filename in _yt_video_ids:
        del _yt_video_ids[filename]
        try:
            _YT_IDS_FILE.write_text(json.dumps(_yt_video_ids), encoding="utf-8")
        except OSError:
            log.warning("Could not persist YouTube video ids after deleting %s", filename)
    if filename in _yt_audio_meta:
        del _yt_audio_meta[filename]
        try:
            _YT_AUDIO_FILE.write_text(json.dumps(_yt_audio_meta), encoding="utf-8")
        except OSError:
            log.warning("Could not persist audio registry after deleting %s", filename)
    _annot_path(filename).unlink(missing_ok=True)
    return jsonify(ok=True)

@api.route("/api/section-labels/<filename>", methods=["POST"])
def api_section_labels_save(filename):
    """Persist hand-drawn section labels. Body: {"labels": {"<bar>": "<label>"}}.
    Last-write-wins: the client posts the whole current map on every change,
    mirroring /api/annotations. Keys must be int-like bar indices; blank values
    drop that bar's label. Render-only — no re-inference in the request path."""
    data = request.get_json(force=True, silent=True) or {}
    labels = data.get("labels", {})
    if not isinstance(labels, dict):
        return jsonify(error="labels must be an object {bar: label}"), 400
    clean: dict[str, str] = {}
    for k, v in labels.items():
        try:
            bar = int(k)
        except (TypeError, ValueError):
            continue
        if bar < 0 or v is None:
            continue
        text = str(v).strip()[:24]
        if text:
            clean[str(bar)] = text
    saved = _save_section_labels(filename, clean)
    return jsonify(ok=True, filename=filename, **saved)

@api.route("/api/annotations/<filename>", methods=["POST"])
def post_annotations(filename):
    """Persist the annotation sidecar. The client posts the whole current
    doc (annotator name + chords + merges) on every change — last-write-
    wins, no merge/conflict logic (single annotator per song, decided).
    Dumb on purpose: no re-inference in the request path."""
    data = request.get_json(silent=True) or {}
    doc = {
        "annotator": data.get("annotator", ""),
        "chords": data.get("chords", []),
        "merges": data.get("merges", []),
    }
    saved = _remember_annotation(filename, doc)
    return jsonify(saved)

@api.route("/api/correction-log/<song>", methods=["POST"])
def api_correction_log(song):
    """Persist ONE human-correction record as training data.

    The annotator POSTs one of these per corrected chord after a save: the
    model's original prediction, the human fix, the /api/reinfer diff it
    produced, and a small benefit analysis (see the task schema). We write one
    JSON file per correction to data/training_logs/<song>/<ts>_<user>_bar<N>.json
    so the corpus can be swept offline (~600 labelled model errors over 20 songs).

    This route is deliberately forgiving: the annotation itself already saved via
    /api/annotations, so logging is pure bonus. On any error we return a JSON
    error the client logs to console and ignores — the save flow never blocks."""
    data = request.get_json(silent=True) or {}
    if not data:
        return jsonify(error="empty correction payload"), 400

    d = _training_log_dir(song)
    try:
        d.mkdir(parents=True, exist_ok=True)   # recursive: fixes the perms/first-run case
    except OSError as e:
        log.warning("correction-log: cannot create %s (%s)", d, e)
        return jsonify(error=f"mkdir failed: {e}"), 500

    # Canonicalise the fields the schema promises even if the client omitted them.
    ts = data.get("timestamp") or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    data["timestamp"] = ts
    data["song"] = lookup_slug(song or "") or "unknown"

    # Filename: <timestamp>_<username>_bar<corrected_bar>.json. Colons are illegal
    # on some filesystems, so ISO 8601's are swapped for dashes.
    fs_ts = ts.replace(":", "-")
    user = re.sub(r"[^A-Za-z0-9_-]", "", (data.get("human_session") or "anon"))[:32] or "anon"
    bar = (data.get("original_prediction") or {}).get("bar", "x")
    stem = f"{fs_ts}_{user}_bar{bar}"
    path = d / f"{stem}.json"
    if path.exists():
        # Timestamp collision (two corrections in the same second) — microsecond suffix.
        stem = f"{stem}_{int(time.time() * 1e6) % 1_000_000:06d}"
        path = d / f"{stem}.json"

    try:
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except OSError as e:
        log.warning("correction-log: write failed %s (%s)", path, e)
        return jsonify(error=f"write failed: {e}"), 500

    log.info("correction-log %s: wrote %s (self_corrected=%s, propagation=%s)",
             song, path.name,
             (data.get("benefit") or {}).get("self_corrected"),
             (data.get("benefit") or {}).get("propagation_count"))
    return jsonify(ok=True, file=path.name, path=str(path))

@api.route("/api/tab-align", methods=["POST"])
def api_tab_align():
    """Fetch a UG tab, align it to a chart payload, return per-chord annotations.

    Body: {
        tab_url, song_name, artist_name, rating, votes, tonality,
        chart_chords: [ {root, lv: {seventh: {q, c}}} ]   ← P.chords from the viewer
    }
    Returns: {
        transpose_semitones, dtw_cost,
        annotations: [ {chord_idx, tab_chord, match, tab_conf_boost} ]
    }
    """
    data = request.get_json(silent=True) or {}
    tab_url     = (data.get("tab_url") or "").strip()
    song_name   = (data.get("song_name") or "").strip()
    artist_name = (data.get("artist_name") or "").strip()
    rating      = float(data.get("rating") or 0)
    votes       = int(data.get("votes") or 0)
    tonality    = (data.get("tonality") or "").strip()
    chart_chords = data.get("chart_chords") or []

    if not tab_url:
        return jsonify(error="No tab_url provided"), 400
    if not chart_chords:
        return jsonify(error="No chart_chords provided"), 400

    try:
        from harmonia.tab_fetcher import TabResult, fetch_tab_chords
        from harmonia.tab_aligner import align_tab_to_chart

        stub = TabResult(id=0, song_name=song_name, artist_name=artist_name,
                         tab_type="Chords", rating=rating, votes=votes,
                         tonality=tonality, difficulty="", tab_url=tab_url, score=0)
        tab = fetch_tab_chords(stub)
        if tab is None:
            return jsonify(error="Could not fetch tab content"), 502

        result = align_tab_to_chart(
            chart_chords, tab.chords, tab_rating=rating, tab_votes=votes
        )
    except ImportError as e:
        return jsonify(error=str(e)), 500
    except Exception as e:
        log.exception("tab-align failed")
        return jsonify(error=str(e)), 500

    return jsonify(
        transpose_semitones=result.transpose_semitones,
        dtw_cost=result.dtw_cost,
        annotations=[
            {
                "chord_idx":      a.chord_idx,
                "tab_chord":      a.tab_chord,
                "match":          a.match,
                "tab_conf_boost": a.tab_conf_boost,
            }
            for a in result.annotations
        ],
    )

@api.route("/api/irealb-compare", methods=["POST"])
def api_irealb_compare():
    """iReal Pro grid with inferred chords overlaid in each cell.

    The iReal chart is rendered in its standard bar-grid form.
    Each cell shows the GT chord (large) and the inferred chord (small, below),
    colored by match quality.  Both are in the same key — iReal labels are
    transposed to match the inferred chart's key.

    Body: {irealb_url, p_chords, bpm (opt), video_id (opt)}
    Returns: {url: "/chart/compare_<slug>.html", ...stats}
    """
    import json as _json, re as _re, urllib.parse as _up
    data       = request.get_json(silent=True) or {}
    irealb_url = (data.get("irealb_url") or "").strip()
    p_chords   = data.get("p_chords") or []
    bpm        = data.get("bpm")
    bpm        = float(bpm) if bpm else None
    vid        = (data.get("video_id") or "").strip()

    if not irealb_url or not p_chords:
        return jsonify(error="irealb_url and p_chords required"), 400

    try:
        from pyRealParser import Tune
        from harmonia.data.ireal_corpus import tune_to_mma
        from harmonia.irealb_aligner import align_irealb_to_inferred
        from harmonia.irealb_fetcher import _esc

        decoded = _up.unquote(irealb_url)
        tunes   = Tune.parse_ireal_url(decoded)
        if not tunes:
            return jsonify(error="No tunes found"), 400
        tune = tunes[0]
        mma  = tune_to_mma(tune, tempo=int(bpm) if bpm else None)
        result = align_irealb_to_inferred(mma, p_chords, bpm_override=bpm)

        # ── label helpers ──────────────────────────────────────────────
        flat_names = ["C","D♭","D","E♭","E","F","G♭","G","A♭","A","B♭","B"]
        QUAL_LABEL = {
            "min7":"m7","min":"m","dom7":"7","maj7":"maj7","maj":"",
            "hdim7":"ø7","dim7":"°7","aug":"+","minmaj7":"mM7","sus4":"sus4",
        }
        def inf_label_from_pc(pc, q):
            if q.startswith(":"): q = q[1:]
            if pc < 0: return "N"
            return flat_names[pc % 12] + QUAL_LABEL.get(q, q)

        # Build a fast lookup: given audio time t → inferred label
        # (the inferred chord whose [t0,t1) contains t)
        inf_sorted = []
        for c in p_chords:
            t0 = float(c.get("t0") or 0)
            t1 = float(c.get("t1") or t0 + 0.5)
            pc = c.get("root", -1)
            q  = c.get("lv", {}).get("seventh", {}).get("q", "")
            inf_sorted.append((t0, t1, inf_label_from_pc(pc, q)))

        def inferred_at(t):
            for t0, t1, lbl in inf_sorted:
                if t0 <= t < t1:
                    return lbl
            return ""

        # ── group iReal chords by bar ──────────────────────────────────
        bars = {}   # bar_no → list of result.chords entries
        for c in result.chords:
            b = c["bar"]
            bars.setdefault(b, []).append(c)
        bar_nos = sorted(bars)

        # ── determine section breaks ───────────────────────────────────
        bar_section = {}
        prev_sec = ""
        for b in bar_nos:
            sec = bars[b][0]["section"]
            bar_section[b] = sec if sec != prev_sec else ""
            prev_sec = sec

        # ── build grid HTML ───────────────────────────────────────────
        MATCH_BG  = {"exact":  "rgba(34,197,94,.18)",
                     "family": "rgba(245,158,11,.18)",
                     "mismatch":"rgba(239,68,68,.18)",
                     "gap":    "rgba(148,163,184,.12)"}
        MATCH_BDR = {"exact":  "#16a34a", "family": "#d97706",
                     "mismatch":"#dc2626", "gap":    "#94a3b8"}

        grid_html = ""
        bars_per_row = 4
        for row_start in range(0, len(bar_nos), bars_per_row):
            row_bars = bar_nos[row_start : row_start + bars_per_row]

            # Section label for this row
            sec = bar_section.get(row_bars[0], "")
            sec_html = (f'<div class="sec-lbl"><span>{_esc(sec)}</span></div>'
                        if sec else '<div class="sec-lbl"></div>')

            row_html = f'<div class="ir-row">{sec_html}'
            for b in row_bars:
                chords_in_bar = bars[b]
                # bar number label
                row_html += f'<div class="ir-bar" data-bar="{b}">'
                row_html += f'<span class="bar-no">{b+1}</span>'
                for c in chords_in_bar:
                    t0 = c["t0"]
                    t1 = c["t1"]
                    m  = c["match"]
                    bg  = MATCH_BG[m]
                    bdr = MATCH_BDR[m]
                    # find what the model inferred at the midpoint of this chord
                    inf_lbl = inferred_at((t0 + t1) / 2) if t0 is not None else ""
                    cell_id = f'chord-{result.chords.index(c)}'
                    row_html += (
                        f'<div class="ir-cell m-{m}" id="{cell_id}" '
                        f'data-t0="{t0}" data-t1="{t1}" '
                        f'style="background:{bg};border-color:{bdr};" '
                        f'title="{_esc(c["label"])} vs {_esc(inf_lbl)} [{m}] {t0:.1f}s">'
                        f'<span class="gt">{_esc(c["label"])}</span>'
                        f'<span class="inf-lbl">{_esc(inf_lbl)}</span>'
                        f'</div>'
                    )
                row_html += '</div>'
            # pad empty bars
            for _ in range(bars_per_row - len(row_bars)):
                row_html += '<div class="ir-bar ir-empty"></div>'
            row_html += '</div>'
            grid_html += row_html

        # ── stats banner ───────────────────────────────────────────────
        ts = mma.time_signature or (4, 4)
        stats_html = (
            f'<span class="si">Key {_esc(mma.key or "?")}  ·  {mma.tempo} BPM  ·  {ts[0]}/{ts[1]}</span>'
            f'<span class="si">transpose +{result.transpose_semitones} st</span>'
            f'<span class="si">{result.n_repeats}× form</span>'
            f'<span class="sok">■ exact {result.exact_frac:.0%}</span>'
            f'<span class="sfam">■ family {result.family_frac:.0%}</span>'
            f'<span class="smiss">■ mismatch {result.mismatch_frac:.0%}</span>'
        )

        # ── legend ────────────────────────────────────────────────────
        legend_html = (
            '<div class="legend">'
            '<span class="leg-item"><span class="leg-swatch" style="background:rgba(34,197,94,.25);border-color:#16a34a"></span>exact root + quality family</span>'
            '<span class="leg-item"><span class="leg-swatch" style="background:rgba(245,158,11,.25);border-color:#d97706"></span>family match (root ok, quality differs)</span>'
            '<span class="leg-item"><span class="leg-swatch" style="background:rgba(239,68,68,.25);border-color:#dc2626"></span>root mismatch</span>'
            '<span class="leg-lbl" style="font-size:11px;color:#4a4636;font-family:system-ui,sans-serif">'
            'Each cell: <b>GT</b> above · <i>inferred</i> below</span>'
            '</div>'
        )

        # ── YouTube ────────────────────────────────────────────────────
        p_json = _json.dumps({"chords": result.chords, "tempo": mma.tempo})
        yt_dock = yt_script = ""
        if vid:
            yt_dock = (
                '<div id="yt-dock">'
                '<div id="yt-pw"><div id="yt-player"></div></div>'
                '<div id="yt-ctrl">'
                '<button onclick="ytToggle()">▶ / ⏸</button>'
                '<span id="yt-t">0:00</span>'
                '</div></div>'
            )
            yt_script = f"""
let _yp=null,_yr=false,_rf=null;
function ytToggle(){{if(!_yr)return;const s=_yp.getPlayerState();if(s===1)_yp.pauseVideo();else _yp.playVideo();}}
function onYouTubeIframeAPIReady(){{
  _yp=new YT.Player('yt-player',{{videoId:{_json.dumps(vid)},
    playerVars:{{origin:window.location.origin,controls:1}},
    events:{{onReady:()=>{{_yr=true;go();}},onStateChange:e=>{{if(e.data===1)go();else stop();}}}}
  }});
}}
function go(){{stop();_rf=requestAnimationFrame(loop);}}
function stop(){{if(_rf){{cancelAnimationFrame(_rf);_rf=null;}}}}
function loop(){{
  if(!_yr)return;
  const t=_yp.getCurrentTime();
  document.getElementById('yt-t').textContent=Math.floor(t/60)+':'+String(Math.floor(t%60)).padStart(2,'0');
  highlightAt(t);
  _rf=requestAnimationFrame(loop);
}}
const _s=document.createElement('script');
_s.src='https://www.youtube.com/iframe_api';document.head.appendChild(_s);
"""

        html = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>{_esc(tune.title)} — GT vs Inferred</title>
<style>
:root{{--paper:#f7f3e9;--ink:#1c1c1c;--rule:#b9b09a;--acc:#8a2b2b;--faint:#8a8371;}}
*{{box-sizing:border-box;margin:0;padding:0;}}
body{{background:var(--paper);color:var(--ink);font-family:Georgia,serif;padding-bottom:160px;}}
.page{{max-width:960px;margin:0 auto;padding:22px 22px 0;}}
h1{{font-size:22px;margin-bottom:4px;}}
.meta{{font-size:12px;color:var(--faint);font-style:italic;margin-bottom:14px;}}
/* stats */
.stats{{display:flex;gap:10px;flex-wrap:wrap;font-family:system-ui,sans-serif;
        font-size:12px;margin-bottom:14px;padding:7px 12px;
        background:#efe9d9;border-radius:7px;align-items:center;}}
.sok{{color:#166534;font-weight:600;}}.sfam{{color:#92400e;font-weight:600;}}
.smiss{{color:#991b1b;font-weight:600;}}.si{{color:#4a4636;}}
/* legend */
.legend{{display:flex;gap:14px;flex-wrap:wrap;align-items:center;margin-bottom:16px;
         font-family:system-ui,sans-serif;font-size:11px;color:#4a4636;}}
.leg-item{{display:flex;align-items:center;gap:5px;}}
.leg-swatch{{display:inline-block;width:14px;height:14px;border:1.5px solid;border-radius:3px;}}
/* grid */
.ir-grid{{display:flex;flex-direction:column;gap:3px;
          border-top:2.5px solid var(--acc);border-bottom:2.5px solid var(--acc);padding:8px 0;}}
.ir-row{{display:grid;grid-template-columns:26px repeat(4,1fr);gap:3px;}}
.sec-lbl{{display:flex;align-items:flex-start;justify-content:center;padding-top:6px;}}
.sec-lbl span{{font-family:system-ui,sans-serif;font-size:10px;font-weight:700;
   color:var(--acc);border:1.5px solid var(--acc);border-radius:3px;padding:1px 3px;}}
.ir-bar{{border:1px solid var(--rule);border-radius:3px;background:#fff;
          min-height:62px;position:relative;display:flex;flex-wrap:wrap;
          align-items:stretch;overflow:hidden;}}
.ir-empty{{border:1px dashed #e0d8c0;background:transparent;}}
.bar-no{{position:absolute;top:2px;left:3px;font-family:system-ui,sans-serif;
          font-size:8px;color:#b0a89a;line-height:1;}}
/* chord cells */
.ir-cell{{flex:1 1 40%;display:flex;flex-direction:column;align-items:center;
           justify-content:center;border:1.5px solid transparent;border-radius:2px;
           padding:14px 2px 4px;min-width:30px;cursor:default;transition:filter .08s;}}
.ir-cell+.ir-cell{{border-left:1px solid var(--rule);}}
.gt{{font-family:'Menlo','Courier New',monospace;font-size:13px;font-weight:600;
     color:var(--ink);line-height:1.2;}}
.inf-lbl{{font-family:'Menlo','Courier New',monospace;font-size:10px;font-weight:400;
           color:var(--faint);line-height:1.2;margin-top:2px;font-style:italic;}}
.ir-cell.now{{filter:brightness(1.06);outline:2px solid #555;outline-offset:1px;}}
/* youtube */
#yt-dock{{position:fixed;bottom:0;left:0;right:0;background:#111;
           display:flex;align-items:center;gap:12px;padding:8px 20px;z-index:200;}}
#yt-pw{{width:200px;height:113px;flex:0 0 200px;}}
#yt-player{{width:200px;height:113px;}}
#yt-ctrl{{color:#fff;font-family:system-ui,sans-serif;font-size:13px;
           display:flex;align-items:center;gap:10px;}}
#yt-ctrl button{{background:#333;color:#fff;border:none;border-radius:4px;
                  padding:4px 12px;cursor:pointer;}}
@media(prefers-color-scheme:dark){{
  :root{{--paper:#1c1a16;--ink:#f0ebe0;--rule:#3a3628;}}
  .ir-bar{{background:#252218;}}
  .stats{{background:#272420;}}
  .inf-lbl{{color:#8a8073;}}
}}
</style>
</head><body>
<div class="page">
  <h1>{_esc(tune.title)}</h1>
  <p class="meta">{_esc(tune.composer or "")} — GT chord chart with inferred chords overlaid</p>
  <div class="stats">{stats_html}</div>
  {legend_html}
  <div class="ir-grid">{grid_html}</div>
</div>
{yt_dock}
<script>
const P={p_json};
function highlightAt(t){{
  document.querySelectorAll('.ir-cell.now').forEach(e=>e.classList.remove('now'));
  document.querySelectorAll('.ir-cell').forEach(e=>{{
    const t0=parseFloat(e.dataset.t0),t1=parseFloat(e.dataset.t1);
    if(t0!=null&&!isNaN(t0)&&t>=t0&&t<t1) e.classList.add('now');
  }});
}}
{yt_script}
</script>
</body></html>"""

    except Exception as e:
        log.exception("irealb-compare failed")
        return jsonify(error=str(e)), 500

    try:
        slug_raw = _up.unquote(irealb_url).split("=")[0].replace("irealb://", "")
        slug = re.sub(r"[^a-z0-9]+", "_", slug_raw.lower()).strip("_") or "irealb"
    except Exception:
        slug = "irealb"

    out = PLOTS_DIR / f"compare_{slug[:60]}.html"
    out.write_text(html, encoding="utf-8")
    if vid:
        _remember_video_id(out.name, vid)

    return jsonify(
        url=f"/chart/{out.name}",
        transpose_semitones=result.transpose_semitones,
        dtw_cost=result.dtw_cost,
        n_repeats=result.n_repeats,
        exact_frac=result.exact_frac,
        family_frac=result.family_frac,
        mismatch_frac=result.mismatch_frac,
    )

@api.route("/api/irealb-search", methods=["POST"])
def api_irealb_search():
    """Search iReal Pro community for songs.

    Body: {title, artist}
    Returns: {results: [{title, composer, key, style, time_sig, irealb_url}]}
    """
    data   = request.get_json(silent=True) or {}
    title  = (data.get("title")  or "").strip()
    artist = (data.get("artist") or "").strip()
    if not title:
        return jsonify(error="No title provided"), 400
    query = f"{title} {artist}".strip()
    try:
        from harmonia.irealb_fetcher import search_community
        results = search_community(query)
    except Exception as e:
        log.exception("irealb-search failed")
        return jsonify(error=str(e)), 500
    return jsonify(results=results)
