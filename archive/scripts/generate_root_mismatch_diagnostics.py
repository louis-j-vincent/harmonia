#!/usr/bin/env python3
"""
Generate root label mismatch diagnostics for Autumn Leaves.

This script compares Phase 1 GT, Phase 2 GT, and inferred roots across all bars
and produces a detailed diagnostic JSON file.

Usage:
    python scripts/generate_root_mismatch_diagnostics.py
"""

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def compute_root_offset(root1: str, root2: str) -> int:
    """
    Compute semitone offset from root1 to root2.
    Returns semitone offset (0-11), or None if roots cannot be parsed.
    """
    note_to_semitone = {
        'C': 0, 'D': 2, 'E': 4, 'F': 5, 'G': 7, 'A': 9, 'B': 11,
    }

    # Extract base note (first character)
    try:
        if not root1 or not root2:
            return None

        base1 = root1[0].upper()
        base2 = root2[0].upper()

        if base1 not in note_to_semitone or base2 not in note_to_semitone:
            return None

        offset = note_to_semitone[base2] - note_to_semitone[base1]
        # Normalize to 0-11
        return offset % 12 if offset != 0 else 0

    except (IndexError, KeyError):
        return None


def extract_chord_root(chord_label: str) -> str:
    """Extract root note from chord label."""
    if not chord_label:
        return None

    # Handle flat/sharp notation
    if len(chord_label) > 1 and chord_label[1] in '#b':
        return chord_label[:2]
    return chord_label[0]


def generate_root_mismatch_diagnostics(song_name: str = "autumn_leaves"):
    """
    Generate root mismatch diagnostic data.
    """
    diagnostics = []

    try:
        # Load iRealb annotations (Phase 1 GT)
        irealb_file = PROJECT_ROOT / "docs" / "plots" / "annotations" / f"irealb_{song_name}.html.json"
        phase1_chords = []
        phase1_source = str(irealb_file) if irealb_file.exists() else "unknown"

        if irealb_file.exists():
            with open(irealb_file) as f:
                data = json.load(f)
                if isinstance(data, dict) and 'chords' in data:
                    phase1_chords = data['chords']
                elif isinstance(data, list):
                    phase1_chords = data

        # Load or generate inferred annotations
        # Try to load from existing inferred HTML JSON
        inferred_file = PROJECT_ROOT / "docs" / "plots" / "annotations" / f"inferred_{song_name}.html.json"
        inferred_chords = []
        inferred_source = str(inferred_file) if inferred_file.exists() else "inferred"

        if inferred_file.exists():
            with open(inferred_file) as f:
                data = json.load(f)
                if isinstance(data, dict) and 'chords' in data:
                    inferred_chords = data['chords']
                elif isinstance(data, list):
                    inferred_chords = data

        # Create bar-by-bar comparison
        max_bar = max(
            [c.get('bar', 0) for c in phase1_chords] +
            [c.get('bar', 0) for c in inferred_chords]
        )

        for bar_num in range(max_bar + 1):
            # Get chords for this bar
            phase1_bar = next((c for c in phase1_chords if c.get('bar') == bar_num), None)
            inferred_bar = next((c for c in inferred_chords if c.get('bar') == bar_num), None)

            if not phase1_bar and not inferred_bar:
                continue

            # Extract roots
            phase1_root = extract_chord_root(phase1_bar['label']) if phase1_bar else None
            inferred_root = extract_chord_root(inferred_bar['label']) if inferred_bar else None

            # Compute offset
            offset = compute_root_offset(phase1_root, inferred_root) if phase1_root and inferred_root else None

            # Determine match status
            match = phase1_root == inferred_root if phase1_root and inferred_root else None

            # Safely get section
            section = None
            if inferred_bar and 'section' in inferred_bar:
                section = inferred_bar.get('section')
            elif phase1_bar and 'section' in phase1_bar:
                section = phase1_bar.get('section')

            diagnostic = {
                'bar': bar_num,
                'time': inferred_bar['t0'] if inferred_bar else (phase1_bar['t0'] if phase1_bar else None),
                'section': section,
                'gt1_root': phase1_root,
                'gt1_source': phase1_source,
                'gt2_root': None,  # Could load another GT if available
                'gt2_source': None,
                'inferred_root': inferred_root,
                'match': match,
                'offset': offset,
                'phase1_full_chord': phase1_bar['label'] if phase1_bar else None,
                'inferred_full_chord': inferred_bar['label'] if inferred_bar else None,
            }

            diagnostics.append(diagnostic)

        # Compute statistics
        matches = sum(1 for d in diagnostics if d['match'] is True)
        total = sum(1 for d in diagnostics if d['match'] is not None)
        accuracy = (matches / total * 100) if total > 0 else 0

        # Count offset patterns
        offset_counts = {}
        for d in diagnostics:
            if d['offset'] is not None:
                offset_counts[d['offset']] = offset_counts.get(d['offset'], 0) + 1

        # Create output
        output = {
            'song': song_name,
            'timestamp': __import__('datetime').datetime.now().isoformat(),
            'summary': {
                'total_bars': len(diagnostics),
                'bars_with_match_status': total,
                'matched_bars': matches,
                'accuracy_percent': round(accuracy, 1),
                'offset_distribution': offset_counts,
                'has_systematic_offset': len(offset_counts) > 0 and max(offset_counts.keys(), key=lambda x: offset_counts[x]) != 0
            },
            'diagnostics': diagnostics
        }

        return output

    except Exception as e:
        print(f"Error generating root mismatch diagnostics: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        return None


def main():
    output = generate_root_mismatch_diagnostics()

    if output:
        output_file = PROJECT_ROOT / "docs" / "plots" / "autumn_leaves_root_mismatch.json"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)

        print(f"✓ Root mismatch diagnostics written to {output_file}")
        print(f"  - Total bars: {output['summary']['total_bars']}")
        print(f"  - Accuracy: {output['summary']['accuracy_percent']}%")
        if output['summary']['has_systematic_offset']:
            print(f"  - Systematic offset detected: {output['summary']['offset_distribution']}")
    else:
        print("✗ Failed to generate diagnostics", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
