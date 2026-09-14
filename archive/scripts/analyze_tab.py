"""
analyze_tab.py — parse a Guitar Pro file and print the chord timeline.

Usage:
    .venv/bin/python scripts/analyze_tab.py MySong.gp5
    .venv/bin/python scripts/analyze_tab.py MySong.gp5 --json output.json
    .venv/bin/python scripts/analyze_tab.py MySong.gp5 --unknowns

Options:
    --json PATH     write events to a JSON file (compatible with ChordChart format)
    --unknowns      also print chord names that couldn't be normalised
    --verbose       show debug logging (tempo map, track details)
    --raw           show the raw chord name column (default: shown)
    --no-raw        hide the raw chord name column (cleaner for long charts)
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path

# Allow running from repo root without installing the package
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.tab_parser import parse_guitar_pro, print_chord_timeline


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Parse a Guitar Pro file and print the chord timeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("file", metavar="FILE.gp[x|5|4|3]",
                    help="Guitar Pro file to parse")
    ap.add_argument("--json", metavar="PATH",
                    help="write chord events as JSON (ChordChart-compatible)")
    ap.add_argument("--unknowns", action="store_true",
                    help="print a list of chord names that could not be normalised")
    ap.add_argument("--verbose", "-v", action="store_true",
                    help="enable debug logging")
    ap.add_argument("--no-raw", action="store_true",
                    help="hide the raw chord name column in the output")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    path = Path(args.file)
    if not path.exists():
        sys.exit(f"Error: file not found: {path}")

    # Capture warnings so we can optionally report unknowns
    unknown_names: list[str] = []
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            events = parse_guitar_pro(path)
        except ImportError as e:
            sys.exit(str(e))
        except Exception as e:
            sys.exit(f"Error parsing {path.name}: {e}")

    for w in caught:
        if "unrecognised chord quality" in str(w.message) or "cannot parse" in str(w.message):
            unknown_names.append(str(w.message))
        else:
            # Re-emit non-parser warnings normally
            warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)

    if not events:
        print(f"No chord annotations found in {path.name}")
        sys.exit(0)

    # ── Print chord timeline ──────────────────────────────────────────────
    print(f"\n{'━'*60}")
    print(f"  {path.name}")
    duration = events[-1]["end_s"] if events else 0.0
    print(f"  {len(events)} chord events  •  duration ≈ {duration:.1f}s")
    print(f"{'━'*60}")

    if args.no_raw:
        print(f"  {'#':>4}  {'CHORD':<12} {'START':>7}  {'END':>7}  {'MEAS':>4}")
        print(f"  {'─'*42}")
        for i, ev in enumerate(events, 1):
            print(
                f"  {i:>4}  {ev['label']:<12} {ev['start_s']:>7.2f}  "
                f"{ev['end_s']:>7.2f}  {ev['measure']:>4}"
            )
    else:
        print(f"  {'#':>4}  {'CHORD':<12} {'START':>7}  {'END':>7}  {'MEAS':>4}  RAW NAME")
        print(f"  {'─'*58}")
        for i, ev in enumerate(events, 1):
            print(
                f"  {i:>4}  {ev['label']:<12} {ev['start_s']:>7.2f}  "
                f"{ev['end_s']:>7.2f}  {ev['measure']:>4}  {ev.get('raw_name', '')}"
            )

    print(f"{'━'*60}\n")

    # ── Unknown chords ────────────────────────────────────────────────────
    if unknown_names:
        if args.unknowns:
            print(f"Chord names not fully normalised ({len(unknown_names)}):")
            for u in unknown_names:
                print(f"  {u}")
            print()
        else:
            print(
                f"  Note: {len(unknown_names)} chord name(s) could not be fully normalised "
                f"(run with --unknowns to see them).\n"
            )

    # ── JSON output ───────────────────────────────────────────────────────
    if args.json:
        out_path = Path(args.json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        # Emit in a ChordChart-compatible format
        output = {
            "source": str(path.resolve()),
            "format": "guitar_pro_tab",
            "n_events": len(events),
            "duration_s": events[-1]["end_s"] if events else 0.0,
            "chords": [
                {
                    "label": ev["label"],
                    "start_s": ev["start_s"],
                    "end_s": ev["end_s"],
                    "raw_name": ev.get("raw_name", ""),
                    "measure": ev.get("measure", -1),
                    "track": ev.get("track", -1),
                }
                for ev in events
            ],
        }
        out_path.write_text(json.dumps(output, indent=2))
        print(f"Chord events written to {out_path}")


if __name__ == "__main__":
    main()
