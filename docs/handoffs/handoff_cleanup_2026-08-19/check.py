#!/usr/bin/env python
"""Open mock_live.html at a phone viewport, run a JS probe, shoot a screenshot.

Usage: check.py OUT.png [--query "?open=autumn_leaves"] [--js FILE] [--w 375] [--h 667]
"""
import argparse, asyncio, json, sys
from pathlib import Path
from playwright.async_api import async_playwright

HERE = Path(__file__).resolve().parent

async def main(a):
    async with async_playwright() as pw:
        br = await pw.chromium.launch()
        pg = await br.new_page(viewport={"width": a.w, "height": a.h},
                               device_scale_factor=2, is_mobile=True, has_touch=True)
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: errs.append("console." + m.type + ": " + m.text)
              if m.type == "error" else None)
        await pg.goto("file://" + str(HERE / "mock_live.html") + a.query)
        await pg.wait_for_timeout(a.wait)
        if a.js:
            out = await pg.evaluate(Path(a.js).read_text())
            print(json.dumps(out, indent=1, ensure_ascii=False))
        if a.out:
            await pg.screenshot(path=a.out, full_page=False)
            print("shot ->", a.out, file=sys.stderr)
        if errs:
            print("PAGE ERRORS:", *errs, sep="\n  ", file=sys.stderr)
        await br.close()

ap = argparse.ArgumentParser()
ap.add_argument("out", nargs="?")
ap.add_argument("--query", default="?open=autumn_leaves")
ap.add_argument("--js"); ap.add_argument("--w", type=int, default=375)
ap.add_argument("--h", type=int, default=667); ap.add_argument("--wait", type=int, default=900)
asyncio.run(main(ap.parse_args()))
