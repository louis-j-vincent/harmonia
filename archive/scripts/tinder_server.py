#!/usr/bin/env python3
"""Serve the pred-vs-GT Tinder page AND persist verdicts/corrections durably.

The static page (scripts/build_pred_tinder.py) POSTs every decision here; we
keep them in a single JSON ledger in the repo so corrections survive a browser
cache-clear, are shared across devices (iPhone + Mac hit the same ledger over
Tailscale), and can be analysed offline (scripts/analyze_tinder_ledger.py).

Endpoints:
    GET  /                 -> the generated HTML page
    GET  /api/state        -> {id: record} of everything saved so far
    POST /api/verdict      -> upsert one record (keyed by its "id")
    POST /api/reset        -> clear the ledger (page's "Reset all")

Bind to 127.0.0.1 and expose over HTTPS with:
    tailscale serve --bg --https=443 http://127.0.0.1:8891

Usage:
    .venv/bin/python scripts/tinder_server.py --port 8891
"""
from __future__ import annotations

import argparse
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RS = REPO / "docs" / "research_sessions"

# Two independent benchmarks share one server, each with its own page + ledger.
# api base -> (page file, ledger file). "/api" = brick0 (frozen GT),
# "/api/jaah" = JAAH real-jazz absolute-timestamp GT.
BENCHES = {
    "/api": (RS / "pred_tinder_2026-07-27.html", RS / "tinder_ledger.json"),
    "/api/jaah": (RS / "jaah_tinder_2026-07-27.html", RS / "jaah_tinder_ledger.json"),
    "/api/guitarset": (RS / "guitarset_tinder_2026-07-28.html",
                       RS / "guitarset_tinder_ledger.json"),
    "/api/choco_isophonics": (RS / "choco_isophonics_tinder_2026-07-28.html",
                              RS / "choco_isophonics_tinder_ledger.json"),
    "/api/choco_billboard": (RS / "choco_billboard_tinder_2026-07-28.html",
                             RS / "choco_billboard_tinder_ledger.json"),
}
# GET path -> page file
PAGES = {
    "/": BENCHES["/api"][0],
    "/jaah": BENCHES["/api/jaah"][0],
    "/guitarset": BENCHES["/api/guitarset"][0],
    "/choco_isophonics": BENCHES["/api/choco_isophonics"][0],
    "/choco_billboard": BENCHES["/api/choco_billboard"][0],
}

_lock = threading.Lock()


def load_ledger(path: Path) -> dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def save_ledger(path: Path, d: dict) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(d, indent=1, ensure_ascii=False))
    tmp.replace(path)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _bench_for(self, path: str):
        """Return the ledger file for an /api[...] path — longest prefix wins so
        /api/jaah/* routes to the JAAH ledger, not the brick0 one."""
        base = max((b for b in BENCHES if path.startswith(b)),
                   key=len, default=None)
        return BENCHES[base][1] if base else None

    def do_GET(self):
        p = self.path.split("?", 1)[0].rstrip("/") or "/"
        if p in PAGES:
            page = PAGES[p]
            if not page.exists():
                self._send(404, b"page not built", "text/plain")
                return
            self._send(200, page.read_bytes(), "text/html; charset=utf-8")
        elif p.endswith("/state"):
            ledger = self._bench_for(p)
            with _lock:
                self._send(200, json.dumps(load_ledger(ledger) if ledger else {}).encode())
        else:
            self._send(404, b"{}")

    def do_POST(self):
        p = self.path.split("?", 1)[0]
        n = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(n) if n else b"{}"
        ledger = self._bench_for(p)
        if ledger is None:
            self._send(404, b"{}")
            return
        if p.endswith("/verdict"):
            try:
                rec = json.loads(raw)
                rid = rec["id"]
            except Exception:
                self._send(400, b'{"ok":false}')
                return
            with _lock:
                d = load_ledger(ledger)
                d[rid] = rec
                save_ledger(ledger, d)
            self._send(200, b'{"ok":true}')
        elif p.endswith("/reset"):
            with _lock:
                save_ledger(ledger, {})
            self._send(200, b'{"ok":true}')
        else:
            self._send(404, b"{}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8891)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args(argv)
    RS.mkdir(parents=True, exist_ok=True)
    srv = ThreadingHTTPServer((args.host, args.port), Handler)
    counts = {b: len(load_ledger(BENCHES[b][1])) for b in BENCHES}
    print(f"tinder server on http://{args.host}:{args.port}  ledgers: "
          + ", ".join(f"{b}={n}" for b, n in counts.items()), flush=True)
    srv.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
