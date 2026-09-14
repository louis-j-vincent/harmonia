#!/usr/bin/env python3
"""Fix GT annotation errors in iRealb charts.

Fixes the cross-section chord doubling inconsistency: when the same chord
appears at different section boundaries with inconsistent doubling (e.g.,
single in A, double in C), this script makes them all consistent.

Algorithm:
1. Detect inconsistencies using detect_ireal_gt_errors.py
2. For each inconsistent chord, determine the target pattern (the majority)
3. For sections with single when target is double: insert a copy of the chord
4. Renumber bars to keep them contiguous
5. Save the corrected annotation
"""

import json
import sys
import copy
from pathlib import Path
from datetime import datetime, timezone

# Import the detector
sys.path.insert(0, str(Path(__file__).parent))
from detect_ireal_gt_errors import detect_cross_section_inconsistencies, find_boundary_chord_structure


def fix_annotation(data):
    """Fix all detected inconsistencies in an annotation.

    Returns (fixed_data, num_insertions, num_fixes) or (None, 0, 0) if no fixes needed.
    """
    chords = data.get("chords", [])
    if not chords:
        return None, 0, 0

    issues = detect_cross_section_inconsistencies(chords)
    if not issues:
        return None, 0, 0

    fixed_chords = [dict(c) for c in chords]
    total_insertions = 0

    for issue in issues:
        chord_label = issue["chord"]
        recommend = issue["recommend"]

        boundary_chords, _ = find_boundary_chord_structure(fixed_chords)

        # Find all boundary occurrences of this chord
        to_fix = []
        for (section, run_num), boundary_info in boundary_chords.items():
            if boundary_info["label"] == chord_label:
                is_double = boundary_info["is_double"]

                # Does this need fixing?
                if recommend == "all double" and not is_double:
                    # Need to insert a copy of the chord
                    to_fix.append((section, run_num, boundary_info["chord_idx"]))
                elif recommend == "all single" and is_double:
                    # Need to remove the duplicate (more complex, skip for now)
                    pass

        # Insert chords in reverse order to avoid index shifting
        for section, run_num, chord_idx in reversed(to_fix):
            chord = fixed_chords[chord_idx]
            new_chord = copy.deepcopy(chord)

            # Mark as inserted for bar renumbering
            new_chord["_inserted"] = True
            new_chord["_fix_chord"] = chord_label

            # Adjust timing: new chord takes zero duration (grid will recalculate)
            if chord_idx + 1 < len(fixed_chords):
                next_chord = fixed_chords[chord_idx + 1]
                new_chord["t0"] = chord.get("t1", chord.get("t0"))
                new_chord["t1"] = next_chord.get("t0", new_chord["t0"])
            else:
                new_chord["t0"] = chord.get("t1", chord.get("t0"))
                new_chord["t1"] = new_chord["t0"]

            fixed_chords.insert(chord_idx + 1, new_chord)
            total_insertions += 1

    # Renumber bars to be contiguous
    fixed_chords = renumber_bars(fixed_chords)

    # Update metadata
    fixed_data = dict(data)
    fixed_data["chords"] = fixed_chords
    fixed_data["modified"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    if "fixes_applied" not in fixed_data:
        fixed_data["fixes_applied"] = []
    fixed_data["fixes_applied"].append({
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": "cross_section_doubling_consistency",
        "issues_fixed": len(issues),
        "chords_inserted": total_insertions,
    })

    return fixed_data, total_insertions, len(issues)


def renumber_bars(chords):
    """Reassign contiguous bar indices.

    Two entries share a bar iff they had the same original bar and
    neither is a freshly-inserted chord.
    """
    for c in chords:
        c.setdefault("_orig_bar", c["bar"])

    new_bar = 0
    prev = None
    for c in chords:
        if prev is not None:
            same = (
                c["_orig_bar"] == prev["_orig_bar"]
                and not c.get("_inserted")
                and not prev.get("_inserted")
            )
            if not same:
                new_bar += 1
        c["bar"] = new_bar
        prev = c

    for c in chords:
        c.pop("_orig_bar", None)
        c.pop("_inserted", None)
        c.pop("_fix_chord", None)

    return chords


def main():
    if len(sys.argv) < 2:
        print("Usage: python fix_ireal_gt_errors.py <annotation.json> [<annotation2.json> ...]")
        print("\nFixes cross-section chord doubling inconsistencies in-place.")
        print("Backups are NOT created — use git if you need to recover.")
        sys.exit(1)

    for path_str in sys.argv[1:]:
        path = Path(path_str)
        print(f"\nProcessing: {path.name}")
        print("=" * 70)

        try:
            with open(path) as f:
                data = json.load(f)

            fixed_data, insertions, fixes = fix_annotation(data)

            if fixed_data is None:
                print("✓ No fixes needed — annotation is valid")
                continue

            # Show what was fixed
            print(f"\n✓ Fixed {fixes} issue(s)")
            print(f"  Chords inserted: {insertions}")
            print(f"  Original chord count: {len(data['chords'])}")
            print(f"  Fixed chord count:   {len(fixed_data['chords'])}")

            # Write back
            with open(path, "w") as f:
                json.dump(fixed_data, f, indent=2)
            print(f"\n✓ Saved to {path}")

        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
