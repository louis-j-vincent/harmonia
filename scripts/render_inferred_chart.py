"""Render an iReal-Pro-style chart of chords INFERRED from audio.

Runs the full Harmonia inference (demo_infer_song.infer_song) on one song, then
draws the predicted chart in the clean chart_render style — each chord placed at
the confident depth of the chord tree and tinted by the model's confidence
(muted red = unsure → near-black = sure). The header notes family accuracy so
the inferred sheet can be eyeballed against the ground-truth chart.

Usage:
    .venv/bin/python scripts/render_inferred_chart.py --title "Autumn Leaves"
    .venv/bin/python scripts/render_inferred_chart.py --title "All The Things" --phone

→ writes docs/plots/inferred_<slug>.png
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, to_hex  # noqa: E402

from demo_infer_song import infer_song  # noqa: E402
from harmonia.output.chart_render import BarChord, Chart, render_chart  # noqa: E402
from harmonia.output.chart_interactive import render_interactive  # noqa: E402

# certainty scale: red (unsure) → amber → green (sure)
_CMAP = LinearSegmentedColormap.from_list("conf", ["#c0392b", "#d99a1c", "#e0c31a", "#3a8a3a"])


def _conf_colour(cf: float) -> str:
    return to_hex(_CMAP(float(np.clip((cf - 0.35) / 0.6, 0, 1))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--title", default="Autumn Leaves")
    ap.add_argument("--conf", type=float, default=0.6, help="confidence to descend the tree")
    ap.add_argument("--phone", action="store_true", help="infer from a grubby phone recording")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--cols", type=int, default=4)
    args = ap.parse_args()

    rec, chords, _bpb, stats = infer_song(args.title, conf_thresh=args.conf, phone=args.phone)

    chart = Chart.from_db_record(rec)
    chart.chords = [
        BarChord(bar=c["bar"], beat=c["beat"], symbol=c["pred_ireal"],
                 colour=_conf_colour(c["conf"]))
        for c in chords
    ]
    src = "phone recording" if args.phone else "clean audio"
    chart.style = f"inferred from {src}  ·  family {stats['fam_acc']:.0%}"

    slug = re.sub(r"[^a-z0-9]+", "_", rec["title"].lower()).strip("_")
    tag = "_phone" if args.phone else ""
    out = Path(args.out) if args.out else REPO / "docs" / "plots" / f"inferred_{slug}{tag}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    render_chart(chart, out, bars_per_row=args.cols,
                 caption="chord colour = model certainty (red = unsure → green = sure); "
                         "only descends to 7th / exact quality where confident")
    print(f"→ {out}")

    # interactive twin: choose level + colour scale live in the browser
    # (chart.style already carries "inferred from … · family …", so no meta_line)
    html_out = out.with_suffix(".html")
    render_interactive(chart, chords, html_out, bars_per_row=args.cols)
    print(f"→ {html_out}")


if __name__ == "__main__":
    main()
