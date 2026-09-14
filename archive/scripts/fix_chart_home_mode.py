#!/usr/bin/env python
"""Repair `home.mode` in already-baked charts (known_issues #30).

`_parse_home_key` tested for "m" before "maj", so pipeline_v1's global_key
format ("G# major") parsed as MINOR. Every real-audio chart was rendered with
`P.home.mode = "minor"` regardless of its actual mode, and the chart's own JS
derives its relative-major reference from that field —

    const maj = h.mode === "major" ? h.tonic : mod(h.tonic + 3, 12);

— so the function/scale colouring on those charts was keyed to a tonic three
semitones off. The parser is fixed, but a chart's payload is baked at render
time and re-rendering means re-running inference. The raw key string survives
in the chart's subhead ("Key G# major"), so the mode can simply be recomputed
and patched in place. Idempotent; --dry-run to preview.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.output.chart_interactive import _parse_home_key  # noqa: E402
from harmonia.output.chart_model import _KEYNAME_RE, _PAYLOAD_RE  # noqa: E402


def fix(path: Path, dry_run: bool) -> str | None:
    text = path.read_text(encoding="utf-8")
    m = _PAYLOAD_RE.search(text)
    if not m:
        return None
    payload = json.loads(m.group(1))
    km = _KEYNAME_RE.search(text)
    if not km:
        return None
    key_name = km.group(1).strip()
    tonic, mode = _parse_home_key(key_name)
    old = payload.get("home", {})
    if old.get("tonic") == tonic and old.get("mode") == mode and payload.get("keyName") == key_name:
        return None
    payload["home"] = {"tonic": tonic, "mode": mode}
    payload["keyName"] = key_name
    if not dry_run:
        patched = text[:m.start(1)] + json.dumps(payload) + text[m.end(1):]
        path.write_text(patched, encoding="utf-8")
    return f'{key_name}: {old.get("tonic")}/{old.get("mode")} → {tonic}/{mode}'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    plots = REPO / "docs" / "plots"
    n = 0
    for p in sorted(plots.glob("inferred_*.html")):
        r = fix(p, a.dry_run)
        if r:
            n += 1
            print(f"  {p.name[:52]:54s} {r}")
    print(f"\n{n} chart(s) {'would be' if a.dry_run else ''} patched")


if __name__ == "__main__":
    main()
