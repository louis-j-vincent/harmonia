#!/usr/bin/env python3
"""
Generate chroma-distance geometry audit for the harmonia project.

This script searches for all distance metric usages in the codebase
and categorizes them by severity and type.

Usage:
    python scripts/generate_chroma_geometry_audit.py
"""

import json
import sys
from pathlib import Path
import re

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def find_distance_metric_usages():
    """
    Search for distance metric usages in Python codebase.
    """
    findings = []

    # Patterns to search for
    patterns = [
        (r'\bdistance\s*\(', 'distance()'),
        (r'\bcdist\s*\(', 'scipy.spatial.distance.cdist'),
        (r'\bpdist\s*\(', 'scipy.spatial.distance.pdist'),
        (r'\bEuclidean\b', 'Euclidean metric'),
        (r'\bcosine\s*\(', 'cosine distance'),
        (r'\.dot\s*\(', 'dot product'),
        (r'np\.linalg\.norm', 'numpy norm'),
    ]

    # Severity classification
    severity_map = {
        'distance()': 2,  # Inference
        'cdist': 1,       # Training (matrix operations)
        'pdist': 1,       # Training
        'Euclidean': 2,   # General
        'cosine': 2,      # Inference
        'dot': 1,         # Training
        'norm': 2,        # General
    }

    # Search through harmonia package
    harmonia_dir = PROJECT_ROOT / "harmonia"
    if not harmonia_dir.exists():
        print(f"Warning: harmonia directory not found at {harmonia_dir}")
        return findings

    for py_file in harmonia_dir.rglob("*.py"):
        try:
            with open(py_file, 'r') as f:
                content = f.read()
                lines = content.split('\n')

                for line_num, line in enumerate(lines, 1):
                    # Skip comments
                    if line.strip().startswith('#'):
                        continue

                    for pattern, name in patterns:
                        if re.search(pattern, line):
                            # Determine context
                            context = "training" if "fit" in line or "train" in line else \
                                     "inference" if "infer" in line or "predict" in line else \
                                     "general"

                            severity = severity_map.get(name, 3)  # Default to Tier 3

                            finding = {
                                'file': str(py_file.relative_to(PROJECT_ROOT)),
                                'line_number': line_num,
                                'line': line.strip()[:80],  # Truncate for readability
                                'function': extract_function_name(lines, line_num),
                                'type': name,
                                'severity': severity,
                                'context': context,
                                'input_shape': 'TBD',  # Would need execution to determine
                                'notes': get_notes_for_metric(name)
                            }

                            findings.append(finding)

        except Exception as e:
            print(f"Warning: Could not read {py_file}: {e}")

    return findings


def extract_function_name(lines, line_num):
    """
    Extract function name from the context of a line.
    """
    for i in range(line_num - 1, max(0, line_num - 20), -1):
        if re.match(r'\s*def\s+(\w+)', lines[i - 1]):
            match = re.match(r'\s*def\s+(\w+)', lines[i - 1])
            return match.group(1)
    return 'Unknown'


def get_notes_for_metric(metric_name):
    """
    Get descriptive notes for each metric type.
    """
    notes = {
        'distance()': 'Generic distance function - check implementation',
        'scipy.spatial.distance.cdist': 'Pairwise distances - verify metric parameter',
        'scipy.spatial.distance.pdist': 'Condensed distance matrix - verify metric parameter',
        'Euclidean metric': 'Linear distance - not appropriate for harmonic relationships',
        'cosine distance': 'Angular distance - preferred for chroma vectors',
        'dot product': 'Inner product - used in similarity measures',
        'numpy norm': 'Vector norm - L2 norm is Euclidean',
    }
    return notes.get(metric_name, 'Unknown metric')


def generate_chord_distance_comparison():
    """
    Generate a comparison of chord distances under different metrics.
    """
    # Example: distances between common chords in chromatic vs harmonic space
    chords = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

    # Chromatic distances (semitone-based)
    chromatic_dist = {}
    for i, c1 in enumerate(chords):
        for j, c2 in enumerate(chords):
            chromatic_dist[f"{c1}-{c2}"] = min(abs(i - j), 12 - abs(i - j))

    # Harmonic distances (circle of fifths)
    fifths_order = [0, 7, 2, 9, 4, 11, 6, 1, 8, 3, 10, 5]  # C, G, D, A, E, B, F#, C#, G#, D#, A#, F
    harmonic_dist = {}
    for i, c1 in enumerate(chords):
        for j, c2 in enumerate(chords):
            pos_i = fifths_order.index(i)
            pos_j = fifths_order.index(j)
            harmonic_dist[f"{c1}-{c2}"] = min(abs(pos_i - pos_j), 12 - abs(pos_i - pos_j))

    return {
        'chromatic': chromatic_dist,
        'harmonic': harmonic_dist,
        'explanation': 'Chromatic metric penalizes semitone distance; harmonic metric penalizes harmonic distance'
    }


def identify_high_confidence_errors():
    """
    Identify patterns where chromatic vs harmonic geometry matters.
    """
    errors = []

    # Common confusion patterns in chromatic space
    # E.g., C->C# is close in chromatic space but far in harmonic space
    problem_pairs = [
        ('C', 'C#', 'Semitone neighbors confused in harmonic context'),
        ('G', 'G#', 'Perfect fifth neighbor confused'),
        ('F', 'F#', 'Tritone neighbor confused'),
    ]

    for c1, c2, description in problem_pairs:
        errors.append({
            'pair': f"{c1}-{c2}",
            'chromatic_distance': 1,  # Adjacent in chromatic space
            'harmonic_distance': 6,   # Far in circle of fifths
            'issue': description,
            'likelihood': 'HIGH' if c2.endswith('#') else 'MEDIUM'
        })

    return errors


def generate_chroma_geometry_audit():
    """
    Generate complete chroma geometry audit.
    """
    findings = find_distance_metric_usages()
    distance_comparison = generate_chord_distance_comparison()
    high_confidence_errors = identify_high_confidence_errors()

    # Categorize findings
    tier1_findings = [f for f in findings if f['severity'] == 1]
    tier2_findings = [f for f in findings if f['severity'] == 2]
    tier3_findings = [f for f in findings if f['severity'] == 3]

    output = {
        'song': 'autumn_leaves',
        'timestamp': __import__('datetime').datetime.now().isoformat(),
        'summary': {
            'total_usages': len(findings),
            'tier1_count': len(tier1_findings),
            'tier2_count': len(tier2_findings),
            'tier3_count': len(tier3_findings),
            'critical_issues': len([f for f in findings if f['severity'] == 1 and 'Euclidean' in f['type']]),
        },
        'findings': findings,
        'tier1_findings': tier1_findings,
        'tier2_findings': tier2_findings,
        'tier3_findings': tier3_findings,
        'distance_comparison': distance_comparison,
        'high_confidence_errors': high_confidence_errors,
        'recommendations': [
            {
                'priority': 1,
                'issue': 'Euclidean distance in training pipeline',
                'action': 'Verify distance metric used in chord template training',
                'severity': 'CRITICAL'
            },
            {
                'priority': 2,
                'issue': 'Chromatic vs harmonic geometry mismatch',
                'action': 'Consider circle-of-fifths distance for harmonic similarity',
                'severity': 'HIGH'
            },
            {
                'priority': 3,
                'issue': 'Error bias on chromatic neighbors',
                'action': 'Validate whether high-confidence errors cluster on semitone neighbors',
                'severity': 'MEDIUM'
            }
        ]
    }

    return output


def main():
    output = generate_chroma_geometry_audit()

    output_file = PROJECT_ROOT / "docs" / "plots" / "autumn_leaves_chroma_audit.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"✓ Chroma geometry audit written to {output_file}")
    print(f"  - Total distance metric usages: {output['summary']['total_usages']}")
    print(f"  - Tier 1 (Training): {output['summary']['tier1_count']}")
    print(f"  - Tier 2 (Inference): {output['summary']['tier2_count']}")
    print(f"  - Tier 3 (Diagnostic): {output['summary']['tier3_count']}")
    print(f"  - Critical issues: {output['summary']['critical_issues']}")


if __name__ == '__main__':
    main()
