"""High-precision chord-training-dataset harvesting pipeline.

Turns (iReal chart, YouTube audio) pairs into CLEAN (audio-segment -> chord)
training rows via the existing Brick-0 aligner + a strict 3-way confidence gate.
Precision >> recall: emit only spans the alignment is confident about; route
confident-but-chart!=recording spans to a substitution/review queue; drop the
rest.

Public API
----------
* :func:`~harmonia.dataset.ingest.add_song` — one call to grow the dataset.
* :func:`~harmonia.dataset.harvest.harvest_song` — align + gate one song.
* :func:`~harmonia.dataset.gate.gate_segment` — the pure, audio-free 3-way gate.
* :func:`~harmonia.dataset.harvest.write_manifests` — emit the JSONL manifests.

See ``harmonia/dataset/README.md`` for the manifest schemas and the gate's
calibrated operating point.
"""
from __future__ import annotations

from .gate import (Bucket, GateConfig, GateDecision, SegmentSignals,
                   gate_segment)
from .harvest import (BeatLock, HarvestResult, build_segment_signals,
                      compute_beat_lock, harvest_song, run_aligner,
                      write_manifests)
from .ingest import YouTubeFetcher, add_song, resolve_chart, slugify

__all__ = [
    "Bucket", "GateConfig", "GateDecision", "SegmentSignals", "gate_segment",
    "BeatLock", "HarvestResult", "build_segment_signals", "compute_beat_lock",
    "harvest_song", "run_aligner", "write_manifests",
    "YouTubeFetcher", "add_song", "resolve_chart", "slugify",
]
