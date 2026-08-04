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
    GET  /api/job/<id>            job record (stage/tempo/key/…/status/url)
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
import re
import threading
import time
import uuid
from pathlib import Path

from flask import Flask, jsonify, request, send_file, send_from_directory

from harmonia_min import annotations
from harmonia_min.pipeline import analyze

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("harmonia_min.server")

PKG = Path(__file__).resolve().parent
REPO = PKG.parent
AUDIO_DIR = REPO / "docs" / "audio"
CHARTS_DIR = PKG / "state" / "charts"
PORT = 7772

app = Flask(__name__)
_jobs: dict[str, dict] = {}


def _pretty_title(stem: str) -> str:
    return stem.replace("_", " ").strip().title()


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


@app.get("/api/library")
def library():
    charts = []
    for p in sorted(CHARTS_DIR.glob("*.json")):
        try:
            m = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        charts.append({
            "file": p.stem, "title": m.get("title") or _pretty_title(p.stem),
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
    q = (body.get("q") or "").strip()
    page = max(0, int(body.get("page") or 0))
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


# ── analyze + jobs ───────────────────────────────────────────────────────────

def _resolve_audio(url: str) -> tuple[Path, str]:
    """analyze URL → (audio path, title). Three forms:
    'local:<stem>' (from our own search results, possibly wrapped in a
    youtube.com/watch?v= prefix by the untouched UI), a local stem typed
    directly, or a real YouTube URL (yt-dlp, if installed)."""
    m = re.search(r"local:([\w\-]+)", url)
    if m:
        p = AUDIO_DIR / f"{m.group(1)}.m4a"
        if p.exists():
            return p, _pretty_title(p.stem)
        raise FileNotFoundError(f"no local audio {m.group(1)}")
    stem = url.strip().strip("/")
    p = AUDIO_DIR / f"{Path(stem).stem}.m4a"
    if p.exists():
        return p, _pretty_title(p.stem)
    if re.search(r"youtu\.?be", url):
        import shutil
        import subprocess
        import sys
        # Prefer the yt-dlp sitting next to THIS interpreter: the venv's
        # bin/ is only on PATH when the server was launched from an activated
        # shell, and a plain `python -m harmonia_min.server` otherwise reports
        # "yt-dlp not installed" while it is right there.
        _cand = Path(sys.executable).parent / "yt-dlp"
        ytdlp = str(_cand) if _cand.exists() else shutil.which("yt-dlp")
        if not ytdlp:
            raise RuntimeError("yt-dlp not installed — paste a library song "
                               "name or install yt-dlp for YouTube links")
        vid = re.search(r"(?:v=|youtu\.be/)([\w\-]{6,})", url)
        out = AUDIO_DIR / (f"{vid.group(1)}.m4a" if vid else "download.m4a")
        if not out.exists():
            subprocess.run(
                ["yt-dlp", "-f", "bestaudio[ext=m4a]/bestaudio",
                 "--extract-audio", "--audio-format", "m4a",
                 "-o", str(out), url], check=True, timeout=600)
        return out, _pretty_title(out.stem)
    raise FileNotFoundError(f"could not resolve {url!r} to audio")


def _run_job(job_id: str, url: str):
    job = _jobs[job_id]

    def progress(stage, **kw):
        job["stage"] = max(job.get("stage", 0), stage)
        job.update(kw)

    try:
        audio_path, title = _resolve_audio(url)
        job["title"] = job.get("title") or title
        import subprocess
        dur = float(subprocess.check_output(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "csv=p=0", str(audio_path)]).strip())
        job.update(stage=1, duration_s=dur)
        file_key = f"min_{audio_path.stem}"
        model = analyze(audio_path, title=job["title"], file_key=file_key,
                        audio_url=f"/audio/{audio_path.name}",
                        progress=progress)
        CHARTS_DIR.mkdir(parents=True, exist_ok=True)
        (CHARTS_DIR / f"{file_key}.json").write_text(
            json.dumps(model), encoding="utf-8")
        job.update(status="done", url=f"/chart/{file_key}", stage=5)
        log.info("job %s done → %s", job_id, file_key)
    except Exception as exc:
        log.exception("job %s failed", job_id)
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
