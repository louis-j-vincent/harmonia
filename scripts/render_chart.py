"""Render an iReal-Pro-style chord chart for a song in the accompaniment DB.

Usage:
    .venv/bin/python scripts/render_chart.py --title "Autumn Leaves"
    .venv/bin/python scripts/render_chart.py --title "All The Things" -o out.png

→ writes docs/plots/chart_<slug>.png by default.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.output.chart_render import Chart, render_chart  # noqa: E402

DB = REPO / "data" / "accomp_db" / "db.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default="Autumn Leaves")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--cols", type=int, default=4, help="bars per row")
    ap.add_argument("--mma", action="store_true", help="use MMA tokens, not iReal")
    args = ap.parse_args()

    records = [json.loads(l) for l in open(DB)]
    cand = [r for r in records if args.title.lower() in r["title"].lower()]
    if not cand:
        sys.exit(f"no song matching {args.title!r}")
    rec = cand[0]

    chart = Chart.from_db_record(rec, use_ireal=not args.mma)
    slug = re.sub(r"[^a-z0-9]+", "_", rec["title"].lower()).strip("_")
    out = Path(args.out) if args.out else REPO / "docs" / "plots" / f"chart_{slug}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    render_chart(chart, out, bars_per_row=args.cols)
    print(f"{rec['title']}  ({rec['form']}, {rec['n_bars']} bars) → {out}")


if __name__ == "__main__":
    main()
