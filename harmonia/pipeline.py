"""
End-to-end Harmonia inference pipeline.

Chains all stages:
    audio → pitch → beats → segments → key per segment → chords per segment
         → unified chord chart

Usage:
    from harmonia.pipeline import HarmoniaPipeline
    pipeline = HarmoniaPipeline()
    chart = pipeline.run("my_song.wav")
    chart.print()
    chart.save_json("output.json")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from harmonia.models.chord_hmm import ChordEvent, ChordInferrer
from harmonia.models.rhythm import RhythmAnalyser
from harmonia.models.stage1_pitch import PitchExtractor
from harmonia.models.structure import Segmenter
from harmonia.theory.jazz_priors import infer_style_posteriors
from harmonia.theory.key_profiles import KeyPosterior, detect_modulations, infer_key

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
            "chords": self.chords,
        }
        path.write_text(json.dumps(data, indent=2))
        logger.info(f"Saved chart → {path}")


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class HarmoniaPipeline:
    """
    Full inference pipeline: audio → chord chart.

    Args:
        max_phase:        chord vocabulary phase (1=triads+7ths, 2=+9ths...)
        cache_dir:        directory for caching Basic Pitch activations.
        prefer_madmom:    use madmom for beat tracking (falls back to librosa).
        kernel_size:      structural segmentation kernel size in beats.
        min_segment_beats: minimum segment length in beats.
        diatonic_boost:   how strongly to prefer diatonic chords (higher = stricter).
    """

    def __init__(
        self,
        max_phase: int = 1,
        cache_dir: Path | None = None,
        prefer_madmom: bool = True,
        kernel_size: int = 8,
        min_segment_beats: int = 8,
        diatonic_boost: float = 3.0,
        self_transition_boost: float = 2.0,
        no_chord_self_transition_boost: float = 0.5,
        normalize_emission: bool = False,
        compress_emission: str | None = None,
        onset_percentile: float | None = None,
        duration_prior: dict[str, np.ndarray] | None = None,
        use_periodicity: bool = False,
        periodicity_weight: float = 1.0,
        periodicity_top_k: int = 3,
        key_prior_per_beat: bool = True,
        key_prior_weight: float = 0.2,
        emission_scoring: str = "dot",
        progression_prior_weight: float = 0.0,
    ):
        """
        emission_scoring: see ChordInferrer — "dot" (default) or "cosine".
        duration_prior: see ChordInferrer — pass a fitted prior (e.g. from
            harmonia.theory.duration_prior.fit_duration_prior()) to switch to
            explicit-duration decoding.
        use_periodicity: if True, detect repeated song structure (see
            harmonia.models.periodicity) once per track and fold it into
            each segment's chord emission as extra weighted evidence.
        key_prior_per_beat: see ChordInferrer — apply the key-conditioned
            diatonic-quality prior at every beat instead of only the first
            beat of each segment.
        key_prior_weight: default 0.2 (a light nudge). The key predicts the
            third (major vs minor) — the note audio hears worst — but must not
            override it: at weight 1.0 it regressed song 001 (majmin 34%→22%),
            while at 0.2 it lifts the 5-song mean majmin 26.5%→32.6% and root
            35.7%→37.1% with no song-001 regression. See
            docs/bayesian_family_combination_2026-07-04.md (measured audio:key
            trust ratio ~5:1) and the --sweep-key-prior harness.
        """
        self.max_phase = max_phase
        self.pitch_extractor = PitchExtractor(cache_dir=cache_dir)
        self.rhythm_analyser = RhythmAnalyser(prefer_madmom=prefer_madmom)
        self.onset_percentile = onset_percentile
        self.use_periodicity = use_periodicity
        self.periodicity_top_k = periodicity_top_k
        self.segmenter = Segmenter(
            kernel_size=kernel_size,
            min_segment_beats=min_segment_beats,
        )
        self.chord_inferrer = ChordInferrer(
            max_phase=max_phase,
            diatonic_boost=diatonic_boost,
            self_transition_boost=self_transition_boost,
            compress_emission=compress_emission,
            no_chord_self_transition_boost=no_chord_self_transition_boost,
            normalize_emission=normalize_emission,
            duration_prior=duration_prior,
            periodicity_weight=periodicity_weight,
            key_prior_per_beat=key_prior_per_beat,
            key_prior_weight=key_prior_weight,
            emission_scoring=emission_scoring,
            progression_prior_weight=progression_prior_weight,
        )

    def run(self, audio_path: str | Path) -> ChordChart:
        """
        Run the full pipeline on an audio file.

        Returns:
            ChordChart with complete chord sequence.
        """
        audio_path = Path(audio_path)
        logger.info(f"Harmonia pipeline: {audio_path.name}")

        # ── Stage 1: Pitch extraction ──────────────────────────────────────
        logger.info("[1/5] Pitch extraction (Basic Pitch)...")
        activations = self.pitch_extractor.extract(
            audio_path, onset_percentile=self.onset_percentile
        )
        logger.info(f"  {activations.n_frames} frames, {activations.duration_s:.1f}s")

        # ── Stage 2: Beat tracking ─────────────────────────────────────────
        logger.info("[2/5] Beat tracking...")
        beat_grid = self.rhythm_analyser.analyse(audio_path)
        logger.info(
            f"  {beat_grid.n_beats} beats @ {beat_grid.tempo_bpm:.1f} BPM "
            f"({beat_grid.time_signature.value}, {beat_grid.backend})"
        )

        # Quantise frame-level activations → beat-level.
        # Use onset_probs (sparse attack signal) not note_probs (flat sustain).
        beat_probs = beat_grid.quantise_frames(
            activations.frame_times, activations.onset_probs
        )

        # ── Stage 2b: Style inference from tempo ──────────────────────────
        style_posteriors = infer_style_posteriors(beat_grid.tempo_bpm)
        style = max(style_posteriors, key=style_posteriors.get)
        logger.info(f"  Inferred style: {style} ({style_posteriors[style]:.2f})")

        # ── Stage 2c: Periodicity — detect repeated song structure once,
        # over the full track (a structural segment alone is usually too
        # short to see an 8-bar loop reliably) ────────────────────────────
        folded_full: dict[int, np.ndarray] = {}
        period_weights: dict[int, float] = {}
        if self.use_periodicity:
            from harmonia.models.periodicity import fold_beat_probs, score_periods

            periods = score_periods(
                beat_probs, beats_per_bar=beat_grid.beats_per_bar(),
                top_k=self.periodicity_top_k,
            )
            total_score = sum(periods.values()) or 1.0
            period_weights = {L: s / total_score for L, s in periods.items()}
            folded_full = {L: fold_beat_probs(beat_probs, L) for L in periods}
            logger.info(f"  Periodicity: {periods}")

        # ── Stage 3: Structural segmentation ──────────────────────────────
        logger.info("[3/5] Structural segmentation...")
        segments = self.segmenter.segment(beat_probs, beat_grid.beat_times)
        logger.info(f"  {len(segments)} segments detected")

        # ── Stage 4: Key inference per segment ────────────────────────────
        logger.info("[4/5] Key inference...")
        key_posteriors: list[KeyPosterior] = []
        for seg in segments:
            kp = infer_key(seg.chroma)
            key_posteriors.append(kp)
            logger.debug(
                f"  Segment [{seg.start_time_s:.1f}s–{seg.end_time_s:.1f}s]: "
                f"{kp.key_name} ({kp.confidence:.2f})"
            )

        # Global key: infer from full-track chroma
        global_chroma = activations.chroma()
        global_key = infer_key(global_chroma)

        # Modulation detection
        modulation_beats = detect_modulations(key_posteriors)
        modulations = []
        for seg_idx in modulation_beats:
            seg = segments[seg_idx]
            modulations.append({
                "beat": seg.start_beat,
                "time_s": round(seg.start_time_s, 3),
                "key": key_posteriors[seg_idx].key_name,
            })

        # ── Stage 5: Chord inference per segment ──────────────────────────
        logger.info("[5/5] Chord inference (Bayesian HMM)...")
        all_events: list[ChordEvent] = []
        for seg, key in zip(segments, key_posteriors):
            seg_beat_times = beat_grid.beat_times[
                seg.start_beat: min(seg.end_beat, len(beat_grid.beat_times))
            ]
            # The authoritative end time for this segment's last chord event
            # is the start of the *next* beat after it — read that from the
            # full beat grid (not just this segment's slice) so consecutive
            # segments' chord events don't overlap. Only the final segment of
            # the whole track has no such beat to read.
            if seg.end_beat < len(beat_grid.beat_times):
                segment_end_time_s = float(beat_grid.beat_times[seg.end_beat])
            else:
                segment_end_time_s = None

            # Slice this segment's view out of each full-track folded array
            # using the same absolute beat range as seg.beat_probs, so slot
            # alignment (position mod period) stays correct across segments.
            folded_views = [
                (folded_full[L][seg.start_beat: min(seg.end_beat, len(beat_probs))],
                 period_weights[L])
                for L in folded_full
            ] or None

            events = self.chord_inferrer.infer(
                beat_probs=seg.beat_probs,
                beat_times=seg_beat_times,
                key=key,
                style=style,
                segment_end_time_s=segment_end_time_s,
                folded_views=folded_views,
            )
            # Offset beat indices to global position
            for ev in events:
                ev.start_beat += seg.start_beat
                ev.end_beat += seg.start_beat
            all_events.extend(events)

        logger.info(f"  {len(all_events)} chord events")

        # ── Build output ───────────────────────────────────────────────────
        return ChordChart(
            source_path=str(audio_path),
            duration_s=activations.duration_s,
            tempo_bpm=round(beat_grid.tempo_bpm, 1),
            time_signature=beat_grid.time_signature.value,
            global_key=global_key.key_name,
            global_key_confidence=round(global_key.confidence, 4),
            style=style,
            modulations=modulations,
            segments=[
                {
                    "start_s": round(seg.start_time_s, 3),
                    "end_s": round(seg.end_time_s, 3),
                    "key": kp.key_name,
                    "n_beats": seg.n_beats,
                }
                for seg, kp in zip(segments, key_posteriors)
            ],
            chords=[
                {
                    "label": ev.label,
                    "start_s": round(ev.start_time_s, 3),
                    "end_s": round(ev.end_time_s, 3),
                    "duration_beats": ev.duration_beats,
                    "confidence": round(ev.confidence, 4),
                }
                for ev in all_events
            ],
        )
