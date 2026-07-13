#!/usr/bin/env python
"""Screenshot a URL at a real iPhone viewport (390×844 @2x), via CDP.

The recipe every doc in this repo repeats —

    google-chrome --headless --screenshot=x.png --window-size=390,844 URL

— does NOT work on macOS: Chrome clamps its window to a 500px minimum width,
renders the page at 500 CSS px, then scales the image down to 390. The result
looks like a phone screenshot but is not one, and a layout that overflows at
390 will look fine in it. (Confirmed by reading window.innerWidth inside the
page: 500.) So drive Chrome's DevTools Protocol and set the device metrics
explicitly, which is what mobile emulation in DevTools actually does.

Usage:
    phone_screenshot.py URL OUT.png [--wait 3] [--width 390] [--height 844]
    phone_screenshot.py URL OUT.png --eval "document.title"   # also print JS result
    phone_screenshot.py URL OUT.png --click "text=Annotate"   # tap a button first
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import json
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


async def shoot(url: str, out: Path, *, width: int, height: int, wait: float,
                js: str | None, click: str | None, scale: int, settle: float = 1.0) -> str | None:
    import websockets

    port = _free_port()
    profile = Path(tempfile.mkdtemp(prefix="harmonia_shot_"))
    proc = subprocess.Popen(
        [CHROME, "--headless=new", f"--remote-debugging-port={port}",
         f"--user-data-dir={profile}", "--no-first-run", "--disable-gpu",
         "--hide-scrollbars", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        ws_url = None
        for _ in range(60):
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/json/list", timeout=1) as r:
                    tabs = json.loads(r.read())
                page = next((t for t in tabs if t["type"] == "page"), None)
                if page:
                    ws_url = page["webSocketDebuggerUrl"]
                    break
            except Exception:
                time.sleep(0.15)
        if not ws_url:
            raise RuntimeError("Chrome DevTools never came up")

        async with websockets.connect(ws_url, max_size=64 * 1024 * 1024) as ws:
            n = 0

            async def send(method, params=None):
                nonlocal n
                n += 1
                await ws.send(json.dumps({"id": n, "method": method, "params": params or {}}))
                while True:
                    msg = json.loads(await ws.recv())
                    if msg.get("id") == n:
                        if "error" in msg:
                            raise RuntimeError(f"{method}: {msg['error']}")
                        return msg.get("result", {})

            await send("Page.enable")
            # A JS exception leaves a half-built page that still screenshots
            # fine — surface it instead of letting the picture lie.
            await send("Runtime.enable")
            # The whole point: a real 390-CSS-px mobile viewport, @2x, touch on.
            await send("Emulation.setDeviceMetricsOverride", {
                "width": width, "height": height, "deviceScaleFactor": scale,
                "mobile": True, "screenWidth": width, "screenHeight": height,
            })
            await send("Emulation.setTouchEmulationEnabled", {"enabled": True, "maxTouchPoints": 5})
            await send("Page.navigate", {"url": url})
            await asyncio.sleep(wait)

            if click:
                sel = click[5:] if click.startswith("text=") else click
                by_text = click.startswith("text=")
                expr = (
                    "(()=>{const t=%s;const els=[...document.querySelectorAll('button,a')];"
                    # exact label first (a segmented-control tab), then the first
                    # clickable whose label merely contains it (a list card)
                    "let el=els.find(e=>e.textContent.trim()===t&&e.offsetParent!==null);"
                    "if(!el) el=els.find(e=>e.textContent.includes(t)&&e.offsetParent!==null);"
                    "if(el){el.click();return 'clicked '+t;} return 'NOT FOUND: '+t;})()"
                    % json.dumps(sel)
                ) if by_text else f"(()=>{{const e=document.querySelector({json.dumps(sel)}); if(e){{e.click(); return 'clicked';}} return 'NOT FOUND';}})()"
                r = await send("Runtime.evaluate", {"expression": expr, "returnByValue": True})
                print("click:", r.get("result", {}).get("value"))
                await asyncio.sleep(1.2)

            js_out = None
            if js:
                # awaitPromise: an --eval that opens an animated sheet must be
                # able to `await` before measuring, or it reads coordinates
                # mid-transition and reports a layout bug that doesn't exist.
                r = await send("Runtime.evaluate",
                               {"expression": js, "returnByValue": True, "awaitPromise": True})
                if "exceptionDetails" in r:
                    exc = r["exceptionDetails"]
                    js_out = "JS EXCEPTION: " + (exc.get("exception", {}).get("description")
                                                 or exc.get("text", "?"))
                else:
                    js_out = r.get("result", {}).get("value")
                # let whatever the JS triggered finish animating — a screenshot
                # taken mid-transition shows a half-faded sheet and reads as a
                # layout bug that isn't there
                await asyncio.sleep(settle)

            shot = await send("Page.captureScreenshot", {"format": "png", "captureBeyondViewport": False})
            out.write_bytes(base64.b64decode(shot["data"]))
            return js_out
    finally:
        proc.terminate()
        shutil.rmtree(profile, ignore_errors=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("url")
    ap.add_argument("out", type=Path)
    ap.add_argument("--width", type=int, default=390)
    ap.add_argument("--height", type=int, default=844)
    ap.add_argument("--scale", type=int, default=2)
    ap.add_argument("--wait", type=float, default=3.0)
    ap.add_argument("--eval", dest="js", default=None, help="JS to run before the shot; its value is printed")
    ap.add_argument("--click", default=None, help='CSS selector, or "text=Annotate" to tap by label')
    ap.add_argument("--settle", type=float, default=1.0, help="seconds to wait after --eval, for animations")
    a = ap.parse_args()
    val = asyncio.run(shoot(a.url, a.out, width=a.width, height=a.height, wait=a.wait,
                            js=a.js, click=a.click, scale=a.scale, settle=a.settle))
    if val is not None:
        print("eval:", val)
    print(f"wrote {a.out} ({a.width}×{a.height} @{a.scale}x)")


if __name__ == "__main__":
    main()
