"""
MIDI → Audio renderer using FluidSynth.

Renders a MIDI file to audio using one or more soundfonts, producing
diverse timbral augmentations from a single MIDI source. Since the MIDI
encodes ground-truth notes and timing, every rendered variant has free labels.

This is the core of the data generation pipeline:
  1. Record piano → export MIDI (ground truth)
  2. Render MIDI through N soundfonts → N training examples
  3. Optionally layer tracks (drums, bass from MIDI → add on top of piano)
  4. Apply acoustic augmentations (room IR, compression, EQ variation)
"""

from __future__ import annotations

import logging
import random
import subprocess
import tempfile
from pathlib import Path
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

# Default sample rate for all rendered audio
DEFAULT_SR = 44100


# ---------------------------------------------------------------------------
# Soundfont registry
# ---------------------------------------------------------------------------

# Map from instrument category to soundfont filename.
# Users should place .sf2 files in data/soundfonts/.
# Free high-quality soundfonts:
#   - Salamander Grand Piano: https://freepats.zenvoid.org/Piano/
#   - GeneralUser GS: https://schristiancollins.com/generaluser.php
#   - FluidR3_GM: ships with many Linux distros

SOUNDFONT_CATEGORIES: dict[str, list[str]] = {
    "piano":    ["Salamander.sf2", "FluidR3_GM.sf2"],
    "electric": ["GeneralUser.sf2"],
    "organ":    ["GeneralUser.sf2"],
    "strings":  ["FluidR3_GM.sf2"],
    "guitar":   ["GeneralUser.sf2"],
    "full_gm":  ["FluidR3_GM.sf2", "GeneralUser.sf2"],
}

# GM program numbers for quick program changes
GM_PROGRAMS: dict[str, int] = {
    "grand_piano": 0,
    "bright_piano": 1,
    "electric_grand": 2,
    "honky_tonk": 3,
    "electric_piano_1": 4,
    "electric_piano_2": 5,
    "harpsichord": 6,
    "organ_1": 16,
    "organ_2": 17,
    "organ_3": 18,
    "nylon_guitar": 24,
    "steel_guitar": 25,
    "jazz_guitar": 26,
    "violin": 40,
    "viola": 41,
    "cello": 42,
    "string_ensemble": 48,
    "synth_strings": 50,
    "choir": 52,
}


# ---------------------------------------------------------------------------
# Render config
# ---------------------------------------------------------------------------

@dataclass
class RenderConfig:
    """Configuration for a single MIDI → audio render pass."""
    soundfont_path: Path
    program: int = 0          # GM program number (0 = grand piano)
    gain: float = 0.8         # FluidSynth gain (0.0–1.0)
    sample_rate: int = DEFAULT_SR
    reverb: bool = True
    chorus: bool = False
    reverb_room: float = 0.4   # 0–1
    reverb_damp: float = 0.3
    reverb_width: float = 0.5
    reverb_level: float = 0.4


# ---------------------------------------------------------------------------
# Renderer
# ---------------------------------------------------------------------------

class MIDIRenderer:
    """
    Renders MIDI files to audio using FluidSynth.

    Requires FluidSynth to be installed:
        macOS:   brew install fluidsynth
        Ubuntu:  apt install fluidsynth
        pip:     pip install pyfluidsynth  (Python bindings, optional)
    """

    def __init__(self, soundfont_dir: Path):
        self.soundfont_dir = Path(soundfont_dir)
        self._check_fluidsynth()

    def _check_fluidsynth(self) -> None:
        try:
            subprocess.run(
                ["fluidsynth", "--version"],
                capture_output=True, check=True,
            )
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError(
                "FluidSynth not found. Install with: brew install fluidsynth  "
                "(macOS) or apt install fluidsynth (Linux)"
            )

    def _find_soundfont(self, filename: str) -> Path:
        path = self.soundfont_dir / filename
        if not path.exists():
            raise FileNotFoundError(
                f"Soundfont not found: {path}\n"
                f"Place .sf2 files in {self.soundfont_dir}/"
            )
        return path

    def render(
        self,
        midi_path: Path,
        output_path: Path,
        config: RenderConfig,
    ) -> Path:
        """
        Render a MIDI file to a WAV file using FluidSynth.

        Returns:
            Path to the rendered WAV file.
        """
        midi_path = Path(midi_path)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            "fluidsynth",
            "-ni",                          # no interactive mode
            "-g", str(config.gain),         # gain
            "-r", str(config.sample_rate),  # sample rate
            "-F", str(output_path),         # output file
        ]

        cmd += ["-o", f"synth.reverb.active={'1' if config.reverb else '0'}"]
        if config.reverb:
            cmd += [
                "-o", f"synth.reverb.room-size={config.reverb_room}",
                "-o", f"synth.reverb.damp={config.reverb_damp}",
                "-o", f"synth.reverb.width={config.reverb_width}",
                "-o", f"synth.reverb.level={config.reverb_level}",
            ]
        cmd += ["-o", f"synth.chorus.active={'1' if config.chorus else '0'}"]
        cmd += [str(config.soundfont_path), str(midi_path)]

        logger.debug(f"FluidSynth: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise RuntimeError(
                f"FluidSynth failed on {midi_path.name}:\n{result.stderr}"
            )

        return output_path

    def render_variants(
        self,
        midi_path: Path,
        output_dir: Path,
        n_variants: int = 5,
        programs: list[int] | None = None,
        soundfonts: list[str] | None = None,
        randomise_reverb: bool = True,
    ) -> list[Path]:
        """
        Render N timbral variants of a MIDI file.

        Each variant uses a (potentially different) soundfont + program +
        reverb setting, maximising acoustic diversity from a single ground-truth
        MIDI source.

        Args:
            midi_path:       source MIDI file.
            output_dir:      directory to write WAV files.
            n_variants:      number of distinct renders to produce.
            programs:        GM program numbers to cycle through.
                             Defaults to a curated piano/keyboard set.
            soundfonts:      soundfont filenames to use. Must exist in soundfont_dir.
            randomise_reverb: if True, randomise reverb parameters per variant.

        Returns:
            List of paths to rendered WAV files.
        """
        if programs is None:
            programs = [
                GM_PROGRAMS["grand_piano"],
                GM_PROGRAMS["electric_piano_1"],
                GM_PROGRAMS["electric_piano_2"],
                GM_PROGRAMS["harpsichord"],
                GM_PROGRAMS["organ_1"],
            ]

        if soundfonts is None:
            soundfonts = [sf for sfs in SOUNDFONT_CATEGORIES.values() for sf in sfs]
            # Filter to those that actually exist
            soundfonts = [
                sf for sf in soundfonts
                if (self.soundfont_dir / sf).exists()
            ]

        if not soundfonts:
            raise FileNotFoundError(
                f"No soundfonts found in {self.soundfont_dir}. "
                "Download .sf2 files and place them there."
            )

        output_paths: list[Path] = []
        midi_stem = Path(midi_path).stem

        for i in range(n_variants):
            sf = soundfonts[i % len(soundfonts)]
            prog = programs[i % len(programs)]

            if randomise_reverb:
                reverb_room = random.uniform(0.1, 0.9)
                reverb_damp = random.uniform(0.1, 0.7)
                reverb_level = random.uniform(0.2, 0.6)
            else:
                reverb_room, reverb_damp, reverb_level = 0.4, 0.3, 0.4

            config = RenderConfig(
                soundfont_path=self._find_soundfont(sf),
                program=prog,
                gain=random.uniform(0.6, 0.9),
                reverb=True,
                reverb_room=reverb_room,
                reverb_damp=reverb_damp,
                reverb_level=reverb_level,
            )

            out = output_dir / f"{midi_stem}_v{i:03d}_prog{prog}.wav"
            try:
                self.render(midi_path, out, config)
                output_paths.append(out)
                logger.info(f"  Rendered variant {i+1}/{n_variants}: {out.name}")
            except Exception as e:
                logger.error(f"  Variant {i+1} failed: {e}")

        return output_paths
