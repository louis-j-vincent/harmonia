#!/usr/bin/env python
"""Ouvre une page du serveur à un viewport de téléphone, sonde en JS, capture.

Descendant direct de handoff_cleanup/check.py (2026-08-19), qui ouvrait une
maquette locale ; ici c'est le VRAI serveur (par défaut celui du worktree, sur
:7773 — jamais :7772, qui est l'app vivante de Louis).

    python -m tools.render_check out.png --url http://127.0.0.1:7773/ --query "?open=min_xxx"
    python -m tools.render_check --url http://127.0.0.1:7773/ --js probe.js

Ce qu'il ne vérifie PAS : le rendu sur l'iPhone lui-même (Safari, pas
Chromium) — pour ça, le lien Tailscale, à l'œil.
"""
import argparse
import asyncio
import json
import sys
from pathlib import Path

from playwright.async_api import async_playwright


async def run(a):
    async with async_playwright() as pw:
        br = await pw.chromium.launch()
        pg = await br.new_page(viewport={"width": a.w, "height": a.h},
                               device_scale_factor=2, is_mobile=True,
                               has_touch=True)
        errs = []
        pg.on("pageerror", lambda e: errs.append(str(e)))
        pg.on("console", lambda m: errs.append("console." + m.type + ": " + m.text)
              if m.type == "error" else None)
        await pg.goto(a.url.rstrip("/") + "/" + a.query.lstrip("/"))
        await pg.wait_for_timeout(a.wait)
        if a.js:
            out = await pg.evaluate(Path(a.js).read_text())
            print(json.dumps(out, indent=1, ensure_ascii=False))
        overflow = await pg.evaluate(
            "document.documentElement.scrollWidth > document.documentElement.clientWidth")
        if a.out:
            await pg.screenshot(path=a.out, full_page=False)
            print("shot ->", a.out, file=sys.stderr)
        if overflow:
            errs.append("débordement horizontal (scrollWidth > clientWidth)")
        if errs:
            print("PAGE ERRORS:", *errs, sep="\n  ", file=sys.stderr)
        await br.close()
        return 1 if errs else 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("out", nargs="?")
    ap.add_argument("--url", default="http://127.0.0.1:7773/")
    ap.add_argument("--query", default="")
    ap.add_argument("--js")
    ap.add_argument("--w", type=int, default=375)
    ap.add_argument("--h", type=int, default=667)
    ap.add_argument("--wait", type=int, default=2500)
    return asyncio.run(run(ap.parse_args(argv)))


if __name__ == "__main__":
    raise SystemExit(main())
