#!/usr/bin/env python3
"""
Master script to generate all Autumn Leaves diagnostics.

Runs:
1. Root mismatch analysis
2. Semi-Markov configuration check
3. Chroma geometry audit

Usage:
    python scripts/generate_all_diagnostics.py
"""

import subprocess
import sys
from pathlib import Path
import json

PROJECT_ROOT = Path(__file__).parent.parent


def run_diagnostic_script(script_path, description):
    """Run a diagnostic script and report results."""
    print(f"\n{'=' * 60}")
    print(f"Running: {description}")
    print(f"{'=' * 60}\n")

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            timeout=60
        )

        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)

        if result.returncode != 0:
            print(f"⚠️ Script exited with code {result.returncode}")
            return False

        return True

    except subprocess.TimeoutExpired:
        print(f"✗ Script timed out after 60 seconds")
        return False
    except Exception as e:
        print(f"✗ Error running script: {e}")
        return False


def generate_diagnostic_manifest():
    """
    Create a manifest of all generated diagnostic files.
    """
    manifest = {
        'generated_at': __import__('datetime').datetime.now().isoformat(),
        'files': [],
        'dashboard_url': 'docs/plots/autumn_leaves_complete_diagnostics.html'
    }

    # Check which diagnostic files exist
    diagnostic_files = [
        'autumn_leaves_root_mismatch.json',
        'autumn_leaves_semi_markov.json',
        'autumn_leaves_chroma_audit.json',
    ]

    plots_dir = PROJECT_ROOT / "docs" / "plots"

    for filename in diagnostic_files:
        filepath = plots_dir / filename
        if filepath.exists():
            manifest['files'].append({
                'name': filename,
                'size_kb': filepath.stat().st_size / 1024,
                'path': f"docs/plots/{filename}"
            })

    return manifest


def main():
    print("\n" + "=" * 60)
    print("AUTUMN LEAVES COMPLETE DIAGNOSTICS GENERATOR")
    print("=" * 60)

    scripts = [
        (PROJECT_ROOT / "scripts" / "generate_root_mismatch_diagnostics.py", "Root Label Mismatch Analysis"),
        (PROJECT_ROOT / "scripts" / "generate_semi_markov_diagnostics.py", "Semi-Markov Configuration Check"),
        (PROJECT_ROOT / "scripts" / "generate_chroma_geometry_audit.py", "Chroma Geometry Audit"),
    ]

    results = {}
    for script_path, description in scripts:
        if not script_path.exists():
            print(f"\n✗ Script not found: {script_path}")
            results[description] = False
            continue

        results[description] = run_diagnostic_script(script_path, description)

    # Generate manifest
    manifest = generate_diagnostic_manifest()
    manifest_file = PROJECT_ROOT / "docs" / "plots" / ".diagnostics_manifest.json"
    with open(manifest_file, 'w') as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 60)
    print("DIAGNOSTICS SUMMARY")
    print("=" * 60)

    all_success = True
    for description, success in results.items():
        status = "✓" if success else "✗"
        print(f"{status} {description}")
        if not success:
            all_success = False

    print(f"\n📊 Dashboard: {manifest['dashboard_url']}")
    print(f"📁 Data files: {len(manifest['files'])} generated")

    for file_info in manifest['files']:
        print(f"   - {file_info['name']} ({file_info['size_kb']:.1f} KB)")

    if all_success and len(manifest['files']) > 0:
        print("\n✓ All diagnostics completed successfully!")
        print("  Open docs/plots/autumn_leaves_complete_diagnostics.html in browser")
        return 0
    else:
        print("\n⚠️ Some diagnostics failed or no data files generated")
        return 1


if __name__ == '__main__':
    sys.exit(main())
