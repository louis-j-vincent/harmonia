"""`create_app()` construit l'app Flask : les routes statiques (shell, audio,
pwa, reports, plots, la page de triage audio) et l'enregistrement de chaque
groupe de routes API (`routes/`). `app = create_app()` au niveau module, pour
que `flask run` / les tests / `__main__.py` trouvent tous le même objet.

Porté depuis `harmonia_min/server.py` (sprints 11-14). Le shell est servi
depuis `harmonia/static/index.html` (sprint 16) ; tout chemin ici part de
`SETTINGS.repo`, jamais du dossier de CE fichier-ci.

Ce que ce module ne fait PAS : construire les routes elles-mêmes (voir
`routes/*.py`) ; lire une variable d'environnement (tout passe par
`SETTINGS`) ; savoir résoudre une URL d'analyse (`jobs.py`) ou parler à
yt-dlp (`youtube.py`).
"""
from __future__ import annotations

import logging

from flask import Flask, jsonify, request, send_file, send_from_directory

from harmonia.server.jobs import CHARTS_DIR
from harmonia.settings import SETTINGS

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s")
log = logging.getLogger("harmonia.server.app")

AUDIO_DIR = SETTINGS.audio_dir


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


def create_app() -> Flask:
    # static_folder=None: Flask's own auto-registered "/static/<path:filename>"
    # route (rooted at harmonia/server/static/, which doesn't exist) would
    # otherwise shadow the explicit /static route below (rooted at
    # harmonia/static/, the sprint-16 frontend split's actual home) — same
    # URL pattern, first-registered wins, 404 either way without this.
    app = Flask(__name__, static_folder=None)
    CHARTS_DIR.mkdir(parents=True, exist_ok=True)

    @app.get("/")
    def index():
        # Sprint 16 : le shell est le host statique
        # `harmonia/static/index.html` (le même <head>/<style>/markup que
        # l'ancien `harmonia_min/app_shell.html`, moins le <script> inline —
        # voir /static/main.js). Le fichier d'origine est parti avec
        # `harmonia_min` au sprint 22 ; le tag `pre-refactor-2026-09-14` le
        # garde pour comparaison.
        return send_file(SETTINGS.repo / "harmonia" / "static" / "index.html")

    @app.get("/static/<path:name>")
    def static_files(name):
        """Les modules ES du shell (main.js, state.js, ui/, screens/…),
        servis tels quels — pas de bundler, pas d'étape de build (plan
        « Frontend split design »)."""
        return send_from_directory(SETTINGS.repo / "harmonia" / "static", name)

    @app.get("/audio/<path:name>")
    def audio(name):
        # Python's mimetypes guesses .m4a → "audio/mp4a-latm", which iOS's
        # media player does not treat as a playable container: in the
        # installed (PWA standalone) app the <audio> stalls at HAVE_METADATA
        # with an empty buffer forever (Louis's iPhone, 2026-08-02).
        # Safari-in-browser is lenient, the standalone player is not. The old
        # :7771 app serves "audio/mp4" and plays fine — do the same.
        mt = "audio/mp4" if name.lower().endswith((".m4a", ".mp4")) else None
        resp = send_from_directory(AUDIO_DIR, name, mimetype=mt)
        # 2026-08-02 iPhone-stall triage: record exactly what byte windows
        # the phone asks for and what we answer — werkzeug's access log only
        # shows "206", which cannot distinguish a healthy chunked playback
        # from the retry storm we are chasing.
        log.info("AUDIO %s range=%r -> %s len=%s",
                 request.remote_addr, request.headers.get("Range"),
                 resp.headers.get("Content-Range"),
                 resp.headers.get("Content-Length"))
        # Second half of the same iPhone stall (fix ported from the old app,
        # harmonia/serving/api.py serve_audio): the shell's <audio> is
        # crossOrigin="anonymous", and iOS validates EVERY 206 Range response
        # for Access-Control-Allow-Origin — even same-origin. Without it
        # WebKit silently discards the data: server logs show a storm of
        # 206s while `buffered` stays empty. Desktop browsers are lenient,
        # which masks it.
        resp.headers["Access-Control-Allow-Origin"] = "*"
        return resp

    @app.get("/pwa/<path:name>")
    def pwa(name):
        """PWA manifest + icons (reused verbatim from the old app, docs/pwa/)."""
        return send_from_directory(SETTINGS.repo / "docs" / "pwa", name)

    @app.get("/reports/<path:name>")
    def reports(name):
        """Pipeline explainer reports (state/reports/*.html)."""
        return send_from_directory(SETTINGS.reports_dir, name)

    @app.get("/plots/<path:name>")
    def plots(name):
        """Diagnostic / listening pages (docs/plots/*.html), reachable over
        Tailscale (Louis, 2026-08-07: file:// links don't work for him —
        pages must live on http://100.89…:7772). Their relative
        ../audio/<stem>.m4a references resolve to the /audio route above,
        same Range/CORS handling."""
        return send_from_directory(SETTINGS.repo / "docs" / "plots", name)

    @app.get("/audiotest")
    def audiotest():
        """One-round-trip on-device triage: raw network throughput (T1) vs
        audio session (T2) vs native media stack (T3) vs the shell's JS
        pattern (T4). Splits Tailscale-path problems from standalone-PWA
        problems from our own player code — the 2026-08-02 iPhone stall
        reproduces on none of our desktop browsers, so the device itself has
        to tell us which layer dies."""
        return AUDIOTEST_HTML

    from harmonia.server.routes import (analyze, annotate, irealb, jam, library,
                                        sections, verdicts)

    app.register_blueprint(library.bp)
    app.register_blueprint(analyze.bp)
    app.register_blueprint(annotate.bp)
    app.register_blueprint(sections.bp)
    app.register_blueprint(jam.bp)
    app.register_blueprint(irealb.bp)
    app.register_blueprint(verdicts.bp)

    # ── everything else: honest 404s the UI degrades on ─────────────────────
    @app.route("/api/<path:rest>", methods=["GET", "POST", "DELETE"])
    def api_unimplemented(rest):
        return jsonify({"error": f"/api/{rest} is not part of harmonia's "
                                 "served API"}), 404

    return app


app = create_app()
