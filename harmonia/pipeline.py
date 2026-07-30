"""
Shared pipeline data types: the `ChordChart` output and the `PipelineConfig`
description of a run.

There is NO pipeline class in this module. The one inference pipeline is
`harmonia.models.chord_pipeline_v1.infer_chords_v1`, which returns the
`ChordChart` defined here.

Usage:
    from harmonia.models.chord_pipeline_v1 import infer_chords_v1
    chart = infer_chords_v1(Path("my_song.wav"))
    chart.print()
    chart.save_json("output.json")

History (2026-07-30): this file used to also hold `HarmoniaPipeline`, a
SECOND, Gen-1 audio→chart pipeline (Basic Pitch → Segmenter → infer_key →
`chord_hmm.ChordInferrer`). Nothing shipped ran it and no live metric measured
it, so work briefed against "the pipeline" kept landing in the copy nobody
scored. It was deleted; see docs/known_issues.md (2026-07-30). Do not
reintroduce a second entry point here — add a config flag to
`infer_chords_v1` instead.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Chart output
# ---------------------------------------------------------------------------

@dataclass
class ChordChart:
    """The final output: a complete chord chart for a piece of audio."""
    source_path: str
    duration_s: float
    tempo_bpm: float
    time_signature: str
    global_key: str
    global_key_confidence: float
    style: str
    modulations: list[dict]             # [{beat, time_s, key}]
    chords: list[dict]                  # [{label, start_s, end_s, duration_beats, conf}]
    segments: list[dict]                # [{start_s, end_s, key, n_beats}]
    # Section-structure boundaries (issue #22): [{start_s, end_s, n_bars}] spans
    # of AABA/A-B sections inferred from the symbolic chord SSM.  Optional/new —
    # empty on pipelines that do not run section detection.
    sections: list[dict] = field(default_factory=list)
    # Structure-anchored downbeat phase (beats), 0 = grid starts at t=0.  Set by
    # the opt-in HARMONIA_GRID_ANCHOR=structure path (chord_pipeline_v1) so the
    # renderer's bar-1 offset matches the grid barlocked pooled on.  0 otherwise.
    grid_anchor_beats: int = 0
    # Real DETECTED beat times (seconds), non-uniform — the display layer snaps
    # chord onset times to these so the playhead tracks the audio through tempo
    # rubato that the uniform bestfit decode grid cannot absorb (user report
    # 2026-07-20 "vérifie l'alignement audio"). Decode/bar-layout stay on the
    # uniform grid; only the displayed t0/t1 are snapped. Empty → no snapping.
    beat_times: list[float] = field(default_factory=list)
    # Coarse whole-track RMS-energy envelope for section-arbiter energy
    # confirmation (2026-07-21): {"hop_s": float, "rms": [float, ...]}, a
    # downsampled |y| RMS at a fixed hop from t=0. The renderer pools this into
    # a per-DISPLAY-bar scalar (it owns the display grid) which the ChartModel's
    # section clusterer uses to re-allow / block harmony merges (see
    # section_arbiter.energy_zscores). ``None`` on symbolic/no-audio charts →
    # section clustering degrades to harmony+veto only. Not persisted by
    # save_json (in-memory, consumed at render time; only the pooled per-bar
    # scalars reach the baked payload).
    energy_env: "dict | None" = None

    def print(self) -> None:
        """Pretty-print the chord chart to stdout."""
        print(f"\n{'━'*60}")
        print(f"  {Path(self.source_path).name}")
        print(f"  Key: {self.global_key}  Tempo: {self.tempo_bpm:.0f} BPM  "
              f"Time: {self.time_signature}")
        print(f"  Style: {self.style}  Duration: {self.duration_s:.1f}s")
        if self.modulations:
            mods = ", ".join(f"{m['key']} at {m['time_s']:.1f}s"
                             for m in self.modulations)
            print(f"  Modulations: {mods}")
        print(f"{'━'*60}")
        print(f"  {'CHORD':<10} {'START':>6}  {'END':>6}  {'BEATS':>5}  {'CONF':>5}")
        print(f"  {'─'*46}")
        for ch in self.chords:
            print(f"  {ch['label']:<10} {ch['start_s']:>6.2f}  "
                  f"{ch['end_s']:>6.2f}  {ch['duration_beats']:>5}  "
                  f"{ch['confidence']:>4.0%}")
        print(f"{'━'*60}\n")

    def save_json(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "source": self.source_path,
            "duration_s": self.duration_s,
            "tempo_bpm": self.tempo_bpm,
            "time_signature": self.time_signature,
            "global_key": self.global_key,
            "global_key_confidence": self.global_key_confidence,
            "style": self.style,
            "modulations": self.modulations,
            "segments": self.segments,
            "sections": self.sections,
            "chords": self.chords,
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info(f"Saved chart → {path}")


# ---------------------------------------------------------------------------
# Phase 5: PipelineConfig — serializable, replaces ~30 kwarg signature
# ---------------------------------------------------------------------------

from dataclasses import asdict
from typing import Literal


@dataclass
class PipelineConfig:
    """Configuration for the chord inference pipeline (Phase 5 PORT).

    Replaces the ~30 boolean kwargs in infer_chords_v1() signature.
    This config is serializable to JSON, so a run is reproducible
    from the config file alone, not from remembering which flags were on.
    """

    # Feature extraction frontend
    feature_frontend: Literal["nnls24", "bp48", "musx"] = "nnls24"
    bass_frontend: Literal["musx", "nnls24"] = "musx"
    quality_frontend: Literal["musx", "nnls24"] = "musx"

    # Beat tracking
    beat_period_mode: Literal["bestfit", "librosa"] = "bestfit"
    beat_backend: Literal["beatthis", "librosa"] = "beatthis"

    # Segmentation / boundary detection
    # DEFAULT FLIPPED 2026-07-27 (see harmonia/serving/runtime.py and
    # harmonia/stages/chord_head.py): "musx_redecode" = beat-aware,
    # latency-compensated re-decode of music-x-lab's frame posteriors.
    # Rollback: HARMONIA_ANALYZE_SEGSOURCE=nnls / HARMONIA_MUSX_REDECODE=0.
    segment_source: Literal["musx_redecode", "nnls", "musx"] = "musx_redecode"
    theta_novelty: float = 0.08  # chroma novelty threshold
    cell_size_beats: int = 2  # segmentation cell size

    # Chord quality inference
    seventh_gate: float = 0.0  # confidence gate for seventh detection
    use_context_classifier: bool = True  # use context-blend dual-head model
    context_classifier_variant: Literal["684d", "801d_two_pass"] = "684d"

    # Prior terms (off by default, opt-in per use case)
    use_diatonic_prior: bool = False
    diatonic_boost: float = 4.0
    threshold_chromatic: float = 0.80

    use_progression_prior: bool = False
    progression_weight: float = 2.0

    use_local_key_prior: bool = False
    local_key_weight: float = 4.0
    local_key_threshold_chromatic: float = 0.80

    # Joint decoding (Viterbi-style)
    use_joint_decode: bool = True
    joint_K: int = 3  # beam width
    joint_transition_weight: float = 0.0
    joint_fusion_iters: int = 1

    # Semi-Markov duration modeling
    use_semi_markov: bool = True
    semi_markov_dur_weight: float = 0.25

    # Post-processing
    use_phase_correction: bool = True  # beat-grid phase recovery via harmony
    occam_postpass: bool = False  # post-hoc simplification (debugging)

    # Caching
    cache_dir: Path | None = None

    def to_dict(self) -> dict:
        """Serialize config to dict (JSON-safe)."""
        d = asdict(self)
        if self.cache_dir is not None:
            d["cache_dir"] = str(self.cache_dir)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineConfig":
        """Deserialize config from dict."""
        d = dict(d)
        if d.get("cache_dir"):
            d["cache_dir"] = Path(d["cache_dir"])
        return cls(**d)

    @classmethod
    def live_defaults(cls) -> "PipelineConfig":
        """Return the current live production defaults.

        These match the values hardcoded in scripts/harmonia_server.py
        and are the parity baseline for Phase 0.
        """
        return cls(
            feature_frontend="nnls24",
            bass_frontend="musx",
            quality_frontend="musx",
            beat_period_mode="bestfit",
            segment_source="musx_redecode",
            use_context_classifier=True,
            context_classifier_variant="684d",
            # All priors off (default)
            use_diatonic_prior=False,
            use_progression_prior=False,
            use_local_key_prior=False,
            # Joint decode on
            use_joint_decode=True,
            # Semi-Markov on
            use_semi_markov=True,
            # Phase correction on
            use_phase_correction=True,
        )
