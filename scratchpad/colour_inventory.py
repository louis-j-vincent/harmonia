"""Inventory: which baked charts are minor-key, real-audio candidates for
generalising the scale-colour tracker beyond This Love?

Criteria: docs/plots/inferred_<slug>.html payload has home.mode == "minor",
docs/audio/<slug>.m4a exists, data/cache/nnls_infer/<slug>.npz exists.

Usage: .venv/bin/python scratchpad/colour_inventory.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def _load_payload(html_path: Path) -> dict | None:
    txt = html_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"const\s+P\s*=\s*", txt)
    if not m:
        return None
    i = txt.index("{", m.end())
    depth, j, instr, esc = 0, i, False, False
    while j < len(txt):
        c = txt[j]
        if instr:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                instr = False
        else:
            if c == '"':
                instr = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    break
        j += 1
    try:
        return json.loads(txt[i:j + 1])
    except Exception:
        return None


def main() -> None:
    rows = []
    for p in sorted((REPO / "docs/plots").glob("inferred_*.html")):
        P = _load_payload(p)
        if not P:
            continue
        slug = p.stem.replace("inferred_", "")
        home = P.get("home") or {}
        audio = (REPO / "docs/audio" / f"{slug}.m4a").exists()
        nnls = (REPO / "data/cache/nnls_infer" / f"{slug}.npz").exists()
        rows.append({
            "slug": slug,
            "keyName": P.get("keyName", "?"),
            "tonic": home.get("tonic"),
            "mode": home.get("mode"),
            "bpb": P.get("bpb"),
            "nBars": P.get("nBars"),
            "nChords": len(P.get("chords") or []),
            "audio": audio,
            "nnls": nnls,
        })

    print(f"{len(rows)} baked charts total\n")
    print("MINOR + real audio + nnls cache:")
    hdr = f"{'slug':55s} {'keyName':10s} {'tonic':>5s} {'bpb':>3s} {'nBars':>5s} {'nCh':>4s}"
    print(hdr)
    for r in rows:
        if r["mode"] == "minor" and r["audio"] and r["nnls"]:
            print(f"{r['slug']:55s} {r['keyName']:10s} {str(r['tonic']):>5s} "
                  f"{str(r['bpb']):>3s} {str(r['nBars']):>5s} {r['nChords']:>4d}")

    print("\nMINOR but missing audio or nnls:")
    for r in rows:
        if r["mode"] == "minor" and not (r["audio"] and r["nnls"]):
            print(f"{r['slug']:55s} {r['keyName']:10s} audio={r['audio']} nnls={r['nnls']}")

    print("\nAll modes summary:")
    from collections import Counter
    print(Counter((r["mode"], r["audio"] and r["nnls"]) for r in rows))


if __name__ == "__main__":
    main()
