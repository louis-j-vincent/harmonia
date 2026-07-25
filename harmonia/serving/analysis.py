"""harmonia/serving/analysis.py — analysis-pipeline leaves for the serving layer
(Phase 6c, final route cluster).

Home of the ``_run_analysis`` background worker and the small helpers the three
analysis routes (POST /api/analyze, /api/record-analyze, /api/reinfer — all in
harmonia.serving.api) depend on. MOVED VERBATIM out of scripts/harmonia_server.py:
the function bodies are byte-identical to their prior server definitions; only
their home module changed. Nothing here mutates module-level state of its own —
every stateful touch goes through an already-extracted leaf:

  * runtime  — the ``jobs`` registry (+ its lock) and the read-only ``_ANALYZE_*``
    env config are the SAME live objects the server exposes; ``runtime.ARGS`` is
    read live (it is reassigned once at startup by ``main()``), exactly as the
    server read it.
  * state    — disk-backed registries / offset stores (_yt_audio_meta,
    _load_bar1_offsets, _save_bar1_offset, _remember_video_id/_remember_audio/
    _remember_ireal_url).
  * config   — filesystem paths (PLOTS_DIR / AUDIO_DIR / PITCH_CACHE_DIR).

This is a leaf: it imports only config/state/runtime (themselves leaves) plus
stdlib; the heavy pipeline modules (chord_pipeline_v1, render_youtube_chart,
chart_interactive, stage1_pitch, irealb_fetcher, yt_dlp, soundfile) are pulled
LAZILY in-body exactly where the original code pulled them, so importing this
module stays cheap and creates no import cycle (harmonia.serving.api imports
these names; nothing here imports api or the server).
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
import threading
from pathlib import Path

from harmonia.serving.config import AUDIO_DIR, PITCH_CACHE_DIR, PLOTS_DIR
import harmonia.serving.runtime as runtime
from harmonia.serving.runtime import (
    jobs as _jobs,
    jobs_lock as _jobs_lock,
    _ANALYZE_FEATURE_FRONTEND,
    _ANALYZE_BASS_FRONTEND,
    _ANALYZE_QUALITY_FRONTEND,
    _ANALYZE_SEGMENT_SOURCE,
    _ANALYZE_BEAT_PERIOD_MODE,
)
from harmonia.serving.state import (
    _load_bar1_offsets,
    _remember_audio,
    _remember_ireal_url,
    _remember_video_id,
    _save_bar1_offset,
    _yt_audio_meta,
)

log = logging.getLogger(__name__)


def _extract_video_id(url: str) -> str:
    """Extract YouTube video ID from a URL. Returns '' if not found."""
    m = re.search(r'(?:v=|youtu\.be/)([A-Za-z0-9_-]{11})', url)
    return m.group(1) if m else ""


def _chart_audio_path(filename: str) -> Path | None:
    """Locate the cached local audio for a chart, or None."""
    meta = _yt_audio_meta.get(filename)
    if not meta:
        return None
    p = AUDIO_DIR / Path(meta["audio"]).name
    return p if p.exists() else None


def _chord_at(chords: list[dict], t: float) -> dict | None:
    for c in chords:
        if c["start_s"] <= t < c["end_s"]:
            return c
    return None


# iReal quality tail (as stored in the annotation sidecar's `q`) → the model's
# 5-way q5 family index (maj/min/dom/hdim/dim). Mirrors the chart's qualBucket().
def _ireal_q_to_q5(q: str | None) -> int:
    if not q:
        return 0
    if q.startswith("-7b5") or q.startswith("h"):
        return 3                                    # half-diminished
    if q.startswith("-") or q.startswith("m"):
        return 1                                    # minor (any)
    if q.startswith("o") or q.startswith("dim"):
        return 4                                    # diminished
    if q.startswith("^") or "maj7" in q or "M7" in q:
        return 0                                    # major (with maj7)
    if any(t in q for t in ("7", "9", "13", "alt")):
        return 2                                    # dominant
    return 0                                        # plain major / 6


def _run_analysis(job_id: str, url: str, seg_source_override: str | None = None,
                   local_audio_path: "Path | None" = None,
                   local_title: str | None = None) -> None:
    """``local_audio_path`` (2026-07-20, mic recording): analyze an already-
    local audio file instead of downloading one — skips the yt-dlp block
    entirely, everything downstream (inference, baking, audio transcode,
    library registration) is unchanged since it already only depends on
    having *some* ``audio_path`` + ``video_title``, not on the source being
    YouTube. ``url`` is then a synthetic, non-YouTube string purely for
    ``_extract_video_id`` (which safely returns "" for it, same as any other
    non-YouTube URL) — no video id / thumbnail gets attached."""
    def update(status, message="", **kw):
        with _jobs_lock:
            _jobs[job_id].update(status=status, message=message, **kw)

    # `stage` + `results` drive the app's Analysing screen. They are the real
    # steps, not a scripted animation: stage 1 is one call into infer_chords_v1
    # (Basic Pitch → beats → sections → key → chord HMM), which reports nothing
    # intermediate, so it stays lit for as long as the decode actually takes and
    # its result chip is filled from what the decode returned.
    results = (["preparing your recording"] if local_audio_path is not None
               else ["downloading from YouTube"]) + [
               "notes, beats, sections, key, chords", "laying out the lead sheet"]

    def stage(i: int, result: str | None = None, **kw):
        if result is not None and i > 0:
            results[i - 1] = result
        update("running", "", stage=i, results=list(results), **kw)

    # Progressive-analysis wiring (2026-07-20): infer_chords_v1's progress_cb
    # fires mid-decode, well before the job would otherwise next touch the job
    # record — see docs/inference_pipeline_timing_and_animation_scope.md. Each
    # event just merges its fields into the job dict; the client
    # (app_shell.html) computes its own bar grid from tempo+chords rather than
    # requiring the full chart_model pipeline mid-flight, so this preview never
    # needs sections. Best-effort by construction (infer_chords_v1 already
    # swallows progress_cb exceptions) — a failure here can't break analysis.
    def on_progress(kind: str, data: dict) -> None:
        field = {
            "beats": lambda: {"tempo_bpm": data["tempo_bpm"],
                               "time_signature": data["time_signature"]},
            "key": lambda: {"key_name": data["key"]},
            "draft": lambda: {"draft_chords": data["chords"]},
            # "fold"/"n_folds" present = an intermediate music-x-lab ensemble
            # fold (2026-07-20, fold-by-fold reveal); absent = the true final
            # result. Always set both so the client can tell them apart.
            "chords": lambda: {"final_chords": data["chords"],
                                "musx_fold": data.get("fold"),
                                "musx_n_folds": data.get("n_folds")},
            "sections": lambda: {"n_sections": data["n_sections"]},
        }.get(kind)
        if field is None:
            return
        with _jobs_lock:
            _jobs[job_id].update(field())

    tmp_dir = Path(tempfile.mkdtemp(prefix="harmonia_yt_"))
    _bg_owns_tmp = False  # set True once the post-"done" finalizer thread (which
                          # reads audio_path from tmp_dir) takes over cleanup.
    try:
        stage(0)

        if local_audio_path is not None:
            audio_path = local_audio_path
            video_title = local_title or "Recording"
            duration = 0
            try:
                import soundfile as _sf
                _info = _sf.info(str(audio_path))
                duration = int(_info.frames / _info.samplerate)
            except Exception:
                duration = 0
        else:
            # Download via yt-dlp Python API (no subprocess, no shell quoting issues)
            try:
                import yt_dlp
            except ImportError:
                update("error", error="yt-dlp not installed in venv")
                return

            audio_path: Path | None = None
            video_title = url

            ydl_opts = {
                "format": "bestaudio/best",
                "outtmpl": str(tmp_dir / "%(id)s.%(ext)s"),
                # unchanged from before — the inference pipeline's audio loader
                # expects whatever this has always produced. A separate m4a copy
                # is transcoded below, only for browser playback.
                "postprocessors": [{"key": "FFmpegExtractAudio", "preferredcodec": "best"}],
                "quiet": True,
                "no_warnings": True,
            }

            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                video_title = info.get("title", url)
                # Find downloaded file
                audio_exts = {".opus", ".m4a", ".mp3", ".webm", ".ogg", ".flac", ".wav", ".aac"}
                files = sorted(tmp_dir.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True)
                for f in files:
                    if f.suffix in audio_exts:
                        audio_path = f
                        break

            if audio_path is None:
                update("error", error="yt-dlp did not produce an audio file")
                return

            duration = 0
            try:
                duration = int(info.get("duration") or 0)
            except (TypeError, ValueError):
                duration = 0
        stage(1, result=(f"{duration // 60}:{duration % 60:02d} of audio" if duration
                         else "audio fetched"), title=video_title, duration_s=duration)

        # Acoustic backend (2026-07-15): prefer the Billboard real-audio
        # checkpoint (data/models/billboard_bp48_60_rollaug_v1.pt — 58 real
        # Billboard songs, zero alignment error, root 54.3%/quality balanced
        # ~19.7% held-out) for freshly-analyzed real audio, since this is the
        # only model in the repo actually validated on real (non-synthetic)
        # recordings — see docs/known_issues.md 2026-07-15 "Billboard root
        # accuracy campaign". Falls back to the Gen-2 ensemble (infer_chords_v1,
        # tuned on POP909/jazz1460) if the checkpoint is missing so this route
        # never hard-fails.
        from harmonia.models.chord_pipeline_v1 import (
            infer_chords_billboard_v1, infer_chords_v1,
        )

        pipeline_chart = None
        backend_used = None

        # New default (2026-07-17 deploy): NNLS-24 features + music-x-lab routed
        # bass. Additive + reversible — controlled by _ANALYZE_FEATURE_FRONTEND
        # (env HARMONIA_ANALYZE_FRONTEND). Any failure here falls through to the
        # exact prior Billboard→infer_chords_v1 chain below, so this can never
        # hard-break analysis for users.
        if _ANALYZE_FEATURE_FRONTEND == "nnls24":
            # Per-request A/B override (2026-07-17) beats the env-var default
            # for this one job only; server-wide default is untouched.
            segment_source_used = seg_source_override or _ANALYZE_SEGMENT_SOURCE
            try:
                pipeline_chart = infer_chords_v1(
                    audio_path,
                    cache_dir=Path(runtime.ARGS.cache_dir),
                    feature_frontend="nnls24",
                    bass_frontend=_ANALYZE_BASS_FRONTEND,
                    quality_frontend=_ANALYZE_QUALITY_FRONTEND,
                    segment_source=segment_source_used,
                    beat_period_mode=_ANALYZE_BEAT_PERIOD_MODE,
                    progress_cb=on_progress,
                )
                backend_used = (f"infer_chords_v1(nnls24, bass={_ANALYZE_BASS_FRONTEND}, "
                                f"quality={_ANALYZE_QUALITY_FRONTEND}, "
                                f"seg={segment_source_used}"
                                f"{' [override]' if seg_source_override else ''})")
            except Exception as e:  # noqa: BLE001 — defensive: never break analyze
                log.warning("analysis %s: nnls24 front-end failed (%s) — falling "
                            "back to Billboard/BP48 chain", job_id, e)
                pipeline_chart = None

        # Prior behaviour (unchanged) — also the fallback if nnls24 is disabled or
        # raised above: Billboard real-audio checkpoint, then Gen-2 ensemble.
        if pipeline_chart is None:
            try:
                pipeline_chart = infer_chords_billboard_v1(
                    audio_path, cache_dir=Path(runtime.ARGS.cache_dir),
                )
                backend_used = "billboard_bp48_60_rollaug_v1"
            except RuntimeError as e:
                log.warning("billboard backend unavailable (%s) — falling back to infer_chords_v1", e)
                pipeline_chart = infer_chords_v1(
                    audio_path,
                    cache_dir=Path(runtime.ARGS.cache_dir),
                )
                backend_used = "infer_chords_v1 (fallback)"
        log.info("analysis %s: acoustic backend = %s", job_id, backend_used)

        stage(2, result=(f"{pipeline_chart.global_key} · {pipeline_chart.tempo_bpm:.0f} bpm"
                         f" · {len(pipeline_chart.chords)} chords"))

        from scripts.render_youtube_chart import chart_to_interactive_inputs
        from harmonia.output.chart_interactive import render_interactive

        source_desc = f"inferred from YouTube · {url}"
        slug = re.sub(r"[^a-z0-9]+", "_", video_title.lower()).strip("_") or "yt"
        slug = slug[:60]
        bar1_offset = _load_bar1_offsets().get(slug, {}).get("offset_beats", 0)
        # The bar-locked section pass OWNS the bar-1 phase: it detects the intro
        # and locks every section boundary to a 4-bar phrase on the beat-0 grid.
        # A saved bar1 offset (typically set on an earlier acoustic chart, and
        # commonly a non-multiple of 4) would shift that grid and knock the
        # phrase-locked boundaries off the 4-bar lattice (user report 2026-07-19:
        # offset=9 dropped the intro + first A). So when barlocked is active and
        # actually produced an intro/section structure, ignore the stale offset.
        if (os.environ.get("HARMONIA_SECTION_MODE", "barlocked") == "barlocked"
                and getattr(pipeline_chart, "sections", None)):
            # barlocked OWNS the bar-1 phase.  With HARMONIA_GRID_ANCHOR=structure
            # it computed a structure-anchored downbeat phase; use THAT as the
            # renderer's bar1_offset so chords and sections share the anchored grid
            # (else use 0 — the beat-0 grid barlocked pooled on).  Either way a
            # stale saved offset must NOT survive to the serve path.
            anchor = int(getattr(pipeline_chart, "grid_anchor_beats", 0) or 0)
            if anchor != bar1_offset:
                log.warning("analysis %s: barlocked owns bar-1 phase — baking "
                            "bar1_offset=%d (structure anchor), was saved %d",
                            job_id, anchor, bar1_offset)
            bar1_offset = anchor
            # The chart is BAKED with this offset; the serve path
            # (_apply_bar1_offset_to_payload) must NOT re-apply it — persist 0 so
            # it does not double-shift the already-anchored chart.
            _save_bar1_offset(slug, 0)
        chart_obj, chord_dicts = chart_to_interactive_inputs(
            pipeline_chart, video_title, source_desc, bar1_offset_beats=bar1_offset,
        )

        out = PLOTS_DIR / f"inferred_{slug}.html"
        render_interactive(chart_obj, chord_dicts, out, bars_per_row=4,
                           sections=pipeline_chart.sections)

        vid = _extract_video_id(url)
        if vid:
            _remember_video_id(out.name, vid)

        # Keep the audio we already downloaded instead of throwing it away —
        # the docked player plays this back locally rather than re-embedding
        # a YouTube iframe. Transcode to AAC/m4a: whatever format the
        # inference pipeline's loader wants (the download above, unchanged)
        # isn't necessarily one iOS Safari's <audio> can play natively.
        try:
            audio_dest = AUDIO_DIR / f"{slug[:60]}.m4a"
            subprocess.run(
                ["ffmpeg", "-y", "-i", str(audio_path), "-vn",
                 "-acodec", "aac", "-b:a", "128k",
                 # +faststart moves the moov atom to the FRONT of the file.
                 # Without it ffmpeg writes moov after mdat (audio data first,
                 # index last) — iOS Safari's <audio> then gets duration from
                 # a metadata probe but never manages to actually fetch the
                 # sample data via Range requests: readyState stalls at
                 # HAVE_METADATA, networkState goes idle, buffered stays
                 # empty, forever (confirmed on-device 2026-07-20 — this is
                 # THE bug behind "button toggles, no sound, time stuck at
                 # 0:00"). Desktop/Chromium players are lenient about moov
                 # position and played the same file fine, which masked this.
                 "-movflags", "+faststart", str(audio_dest)],
                check=True, capture_output=True, timeout=120,
            )
            thumb_url = f"https://i.ytimg.com/vi/{vid}/mqdefault.jpg" if vid else ""
            _remember_audio(out.name, f"/audio/{audio_dest.name}", thumb_url)
        except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as e:
            log.warning("Could not persist/transcode audio for %s: %s", out.name, e)

        # Mark the job done NOW — the chart the user is waiting on is fully
        # rendered and the playback audio is transcoded. Everything below
        # (Basic Pitch persist + iReal fetch) produces *annotator-facing*
        # sidecar artifacts that the freshly-opened chart never reads, so it
        # is moved to a background thread AFTER "done" (2026-07-17 perf fix,
        # docs/known_issues.md). Previously the Basic Pitch persist ran an
        # ~11–18 s COLD Basic Pitch pass here, inside "Drawing the chart", on
        # every analyze: its "cache hit" comment was WRONG — the live nnls24
        # front-end (_infer_nnls24) never calls PitchExtractor, and the
        # extract is temp-path-keyed so it never warms — pure user-visible
        # wait for a .npz that only a future annotator-reload surface (arch-
        # extensions §13) will read.
        results[2] = f"{chart_obj.n_bars} bars"
        update("done", url=f"/chart/{out.name}", stage=3,
               results=list(results), title=video_title)

        # Background post-processing. Owns tmp_dir cleanup (Basic Pitch reads
        # audio_path, which lives inside tmp_dir) so the outer finally must NOT
        # rmtree once this thread is launched. Fully guarded: any failure here
        # is logged and swallowed — the user already has their chart, so a
        # background hiccup must never surface as a user-facing error.
        def _finalize_bg(job_id=job_id, tmp_dir=tmp_dir, audio_path=audio_path,
                         slug=slug, out=out, video_title=video_title,
                         pipeline_chart=pipeline_chart):
            try:
                # Persist chroma/pitch activations, addressable by slug — for a
                # later "re-score bars against pooled chroma" annotator surface
                # (arch-extensions §13). This is a full Basic Pitch run (the
                # nnls24 path never populated PitchExtractor's cache), hence why
                # it belongs off the hot path.
                try:
                    from harmonia.models.stage1_pitch import PitchExtractor
                    activations = PitchExtractor(cache_dir=Path(runtime.ARGS.cache_dir)).extract(audio_path)
                    activations.save(PITCH_CACHE_DIR / f"{slug[:60]}.npz")
                except Exception as e:  # noqa: BLE001 — best-effort, never user-facing
                    log.warning("bg: could not persist pitch/chroma cache for %s: %s", out.name, e)

                # Fetch iReal chart from community if available, so the annotator
                # tool (which needs docs/plots/irealb_<slug>.html) doesn't 404.
                try:
                    from harmonia.irealb_fetcher import search_community, render_irealb_chart
                    results_ir = search_community(video_title, max_results=1)
                    if results_ir:
                        irealb_url = results_ir[0]["irealb_url"]
                        html_ir = render_irealb_chart(irealb_url, chart_offset_s=0.0,
                                                      tempo_override=int(round(pipeline_chart.tempo_bpm)))
                        ir_out = PLOTS_DIR / f"irealb_{slug[:60]}.html"
                        ir_out.write_text(html_ir, encoding="utf-8")
                        _remember_ireal_url(out.name, irealb_url)
                        log.info("bg: saved iReal chart for %s (%s)", out.name, irealb_url)
                except Exception as e:  # noqa: BLE001 — best-effort, never user-facing
                    log.warning("bg: could not fetch/render iReal chart for %s: %s", video_title, e)
            finally:
                shutil.rmtree(tmp_dir, ignore_errors=True)

        threading.Thread(target=_finalize_bg, daemon=True,
                          name=f"harmonia-finalize-{job_id}").start()
        _bg_owns_tmp = True

    except Exception as e:
        log.exception("Analysis failed for %s", url)
        update("error", error=str(e))
    finally:
        # The background finalizer owns tmp_dir once launched; only clean up
        # here if we never got that far (early error, or nnls24 disabled path).
        if not _bg_owns_tmp:
            shutil.rmtree(tmp_dir, ignore_errors=True)
