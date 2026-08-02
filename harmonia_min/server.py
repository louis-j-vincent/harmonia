"""harmonia_min/server.py — minimal Flask server for the app shell. Port 7772.

NEVER port 7771 — that is the old production server, owned by other sessions.

Serves exactly what app_shell.html's milestone-1 path needs:

    GET  /                        the app shell (copied verbatim)
    GET  /api/library             chart list  {charts:[{file,title,key,bars,hasAudio,mtime}]}
    GET  /api/chart-model/<file>  a ChartModel JSON from state/charts/
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
    return jsonify({"charts": charts})


@app.get("/api/chart-model/<file>")
def chart_model(file):
    p = CHARTS_DIR / f"{Path(file).stem}.json"
    if not p.exists():
        return jsonify({"error": "no such chart"}), 404
    return send_file(p, mimetype="application/json")


@app.delete("/api/chart/<file>")
def delete_chart(file):
    p = CHARTS_DIR / f"{Path(file).stem}.json"
    if p.exists():
        p.unlink()
    return jsonify({"ok": True})


# ── local search (the offline library flow) ─────────────────────────────────

@app.post("/api/yt-search")
def yt_search():
    q = (request.get_json(silent=True) or {}).get("q", "").lower()
    words = [w for w in re.split(r"\W+", q) if w]
    results = []
    for p in sorted(AUDIO_DIR.glob("*.m4a")):
        hay = p.stem.lower()
        if words and all(w in hay for w in words):
            results.append({"id": f"local:{p.stem}",
                            "title": _pretty_title(p.stem),
                            "uploader": "local audio", "duration": None,
                            "thumb": ""})
    return jsonify({"results": results[:12]})


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
        if not shutil.which("yt-dlp"):
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
