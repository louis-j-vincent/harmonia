#!/usr/bin/env python3
"""Detect GT annotation errors in iRealb charts.

Primary pattern: the same chord appears at multiple section boundaries with
INCONSISTENT doubling — e.g., single G-6 at A boundaries but double G-6 at
C boundaries. This causes structural misalignment throughout the song.

Example: Autumn Leaves bug — G-6 is doubled at C boundaries (correct) but
single at A boundaries (incorrect). Fix: make all G-6 boundaries consistent
(all double, to match C sections).
"""

import json
import sys
from pathlib import Path
from collections import defaultdict


def find_boundary_chord_structure(chords):
    """Map each boundary chord to its occurrence pattern.

    Returns:
    - boundary_chords: (section_label, section_run_num) → {"label": X, "is_double": bool}
    - boundary_by_chord: chord_label → [(section, run_num, is_double), ...]
    """
    if not chords:
        return {}, {}

    # Find section runs
    runs = []
    current_section = None
    start_idx = 0
    for i, chord in enumerate(chords):
        section = chord.get("section")
        if section != current_section:
            if current_section is not None:
                runs.append((current_section, start_idx, i - 1))
            current_section = section
            start_idx = i
    if current_section is not None:
        runs.append((current_section, start_idx, len(chords) - 1))

    # For each section, record its boundary chord
    boundary_chords = {}
    boundary_by_chord = defaultdict(list)

    for section_label, start, end in runs:
        last_chord = chords[end]
        boundary_label = last_chord.get("label")

        # Check if DOUBLE
        is_double = False
        if end > start:
            prev_chord = chords[end - 1]
            if (prev_chord.get("section") == section_label and
                prev_chord.get("label") == boundary_label and
                prev_chord.get("bar") != last_chord.get("bar")):
                is_double = True

        run_num = len([r for r in runs if r[0] == section_label and r[2] <= end])

        boundary_chords[(section_label, run_num)] = {
            "label": boundary_label,
            "is_double": is_double,
            "chord_idx": end,
            "start_idx": start,
        }
        boundary_by_chord[boundary_label].append((section_label, run_num, is_double))

    return boundary_chords, boundary_by_chord


def detect_cross_section_inconsistencies(chords):
    """Find chords that appear at different section boundaries with inconsistent doubling.

    Returns list of issues.
    """
    boundary_chords, boundary_by_chord = find_boundary_chord_structure(chords)
    issues = []

    # For each chord that appears at multiple section boundaries
    for chord_label, occurrences in boundary_by_chord.items():
        # Group by section
        by_section = defaultdict(list)
        for section, run_num, is_double in occurrences:
            by_section[section].append(is_double)

        if len(by_section) < 2:
            continue  # Only care if it appears at multiple section boundaries

        # Check for doubling inconsistency across sections
        all_doubles = all(all(d for d in doubles) for doubles in by_section.values())
        all_singles = all(all(not d for d in doubles) for doubles in by_section.values())

        if not (all_doubles or all_singles):
            # INCONSISTENT across sections
            patterns = {
                section: ("all double" if all(d for d in doubles) else "all single")
                for section, doubles in by_section.items()
            }

            # Recommend which pattern to use (use the one that appears more)
            double_sections = [s for s, p in patterns.items() if p == "all double"]
            single_sections = [s for s, p in patterns.items() if p == "all single"]
            recommend = "all double" if len(double_sections) >= len(single_sections) else "all single"

            issues.append({
                "type": "cross_section_chord_doubling_inconsistency",
                "chord": chord_label,
                "message": (
                    f"Chord '{chord_label}' has inconsistent doubling across section boundaries: "
                    f"{patterns}. All occurrences should be consistent. "
                    f"Recommend: {recommend}"
                ),
                "patterns": patterns,
                "recommend": recommend,
                "occurrences": [(s, r, d) for s, r, d in occurrences],
                "boundary_chords": boundary_chords,
            })

    return issues


def main():
    if len(sys.argv) < 2:
        print("Usage: python detect_ireal_gt_errors.py <annotation.json> [<annotation2.json> ...]")
        sys.exit(1)

    all_issues = []

    for path_str in sys.argv[1:]:
        path = Path(path_str)
        print(f"\n{'='*70}")
        print(f"Analyzing: {path.name}")
        print(f"{'='*70}")

        try:
            with open(path) as f:
                data = json.load(f)

            issues = detect_cross_section_inconsistencies(data.get("chords", []))

            if not issues:
                print("✓ No cross-section chord inconsistencies detected")
            else:
                for issue in issues:
                    all_issues.append((path.name, issue))
                    print(f"\n⚠️  {issue['message']}")
                    print("\n  Pattern:")
                    for section, pattern in issue["patterns"].items():
                        print(f"    Section {section}: {pattern}")
                    print(f"\n  Recommendation: Make all '{issue['chord']}' boundaries {issue['recommend']}")

        except Exception as e:
            print(f"Error: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    if all_issues:
        print(f"\n\n{'='*70}")
        print(f"SUMMARY: Found {len(all_issues)} issue(s) across {len(set(f[0] for f in all_issues))} file(s)")
        print(f"{'='*70}")
        for filename, issue in all_issues:
            print(f"\n{filename}:")
            print(f"  Chord '{issue['chord']}': {issue['message']}")


if __name__ == "__main__":
    main()
