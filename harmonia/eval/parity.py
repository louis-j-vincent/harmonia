"""harmonia/eval/parity.py — parity gate harness for the rewrite.

Captures and diffs ChordChart outputs (old vs new) with field-specific tolerances.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class DiffResult:
    """Single field diff between old and new ChordChart."""
    field: str
    old_value: Any
    new_value: Any
    match: bool
    tolerance: float | None = None
    error: float | None = None
    details: str = ""


@dataclass
class ChartDiff:
    """Full diff between two ChordChart JSONs."""
    name: str
    song_id: str
    matches: bool
    differences: list[DiffResult]

    def report(self, verbose=False) -> str:
        """Render a human-readable diff report."""
        status = "✓ MATCH" if self.matches else "✗ DIFFER"
        lines = [f"{status}: {self.name} (song={self.song_id})"]

        if self.differences:
            for diff in self.differences:
                if not diff.match or verbose:
                    lines.append(
                        f"  {diff.field}: {diff.old_value} → {diff.new_value} "
                        f"(tol={diff.tolerance}, error={diff.error})"
                    )

        return "\n".join(lines)


def load_chart_json(path: Path) -> dict:
    """Load a ChordChart JSON file."""
    with open(path) as f:
        return json.load(f)


def save_chart_json(chart: dict, path: Path) -> None:
    """Save a ChordChart JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(chart, f, indent=2)


def diff_charts(
    old_chart: dict,
    new_chart: dict,
    name: str = "chart",
    float_tol: float = 1e-4,
    label_exact: bool = True,
) -> ChartDiff:
    """Compare two ChordChart dicts field-by-field.

    Args:
        old_chart: old JSON dict
        new_chart: new JSON dict
        name: song name for reporting
        float_tol: tolerance for floating-point fields (tempo, confidence, etc.)
        label_exact: if True, chord labels must match exactly; if False, allow label order variance

    Returns:
        ChartDiff with all field-level diffs
    """
    diffs = []

    # Top-level scalar fields (exact match for most, tolerance for floats)
    scalar_fields = {
        "source_path": (str, True),
        "duration_s": (float, False),
        "tempo_bpm": (float, False),
        "time_signature": (str, True),
        "global_key": (str, True),
        "global_key_confidence": (float, False),
        "style": (str, True),
    }

    for field, (expected_type, exact) in scalar_fields.items():
        old_val = old_chart.get(field)
        new_val = new_chart.get(field)

        if exact:
            match = old_val == new_val
            error = None
        else:  # float, with tolerance
            match = (old_val is not None and new_val is not None and
                    abs(float(old_val) - float(new_val)) < float_tol)
            error = (abs(float(old_val) - float(new_val))
                    if old_val is not None and new_val is not None else None)

        if not match:
            diffs.append(DiffResult(
                field=field,
                old_value=old_val,
                new_value=new_val,
                match=match,
                tolerance=float_tol if not exact else None,
                error=error,
            ))

    # Chord sequence (critical): compare label-for-label
    old_chords = old_chart.get("chords", [])
    new_chords = new_chart.get("chords", [])

    if len(old_chords) != len(new_chords):
        diffs.append(DiffResult(
            field="chords.length",
            old_value=len(old_chords),
            new_value=len(new_chords),
            match=False,
        ))

    for i, (old_ch, new_ch) in enumerate(zip(old_chords, new_chords)):
        # Exact label match
        if old_ch.get("label") != new_ch.get("label"):
            diffs.append(DiffResult(
                field=f"chords[{i}].label",
                old_value=old_ch.get("label"),
                new_value=new_ch.get("label"),
                match=False,
            ))

        # Time boundaries: tight tolerance (50ms)
        for bound in ["start_s", "end_s"]:
            old_b = old_ch.get(bound)
            new_b = new_ch.get(bound)
            error = abs(float(old_b) - float(new_b)) if old_b is not None and new_b is not None else None
            match = error is not None and error < 0.050
            if not match:
                diffs.append(DiffResult(
                    field=f"chords[{i}].{bound}",
                    old_value=old_b,
                    new_value=new_b,
                    match=match,
                    tolerance=0.050,
                    error=error,
                ))

        # Confidence scores: loose tolerance (0.01)
        for conf_field in ["confidence", "confidence_raw", "root_conf"]:
            old_c = old_ch.get(conf_field)
            new_c = new_ch.get(conf_field)
            if old_c is not None and new_c is not None:
                error = abs(float(old_c) - float(new_c))
                match = error < 0.01
                if not match:
                    diffs.append(DiffResult(
                        field=f"chords[{i}].{conf_field}",
                        old_value=old_c,
                        new_value=new_c,
                        match=match,
                        tolerance=0.01,
                        error=error,
                    ))

    # Segments (structure boundaries)
    old_segs = old_chart.get("segments", [])
    new_segs = new_chart.get("segments", [])

    if len(old_segs) != len(new_segs):
        diffs.append(DiffResult(
            field="segments.length",
            old_value=len(old_segs),
            new_value=len(new_segs),
            match=False,
        ))

    for i, (old_s, new_s) in enumerate(zip(old_segs, new_segs)):
        # Segment boundaries with 50ms tolerance
        for bound in ["start_s", "end_s"]:
            old_b = old_s.get(bound)
            new_b = new_s.get(bound)
            error = abs(float(old_b) - float(new_b)) if old_b is not None and new_b is not None else None
            match = error is not None and error < 0.050
            if not match:
                diffs.append(DiffResult(
                    field=f"segments[{i}].{bound}",
                    old_value=old_b,
                    new_value=new_b,
                    match=match,
                    tolerance=0.050,
                    error=error,
                ))

    song_id = old_chart.get("source_path", "unknown").split("/")[-1]
    return ChartDiff(
        name=name,
        song_id=song_id,
        matches=len(diffs) == 0,
        differences=diffs,
    )
