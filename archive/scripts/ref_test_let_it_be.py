"""Reference test for the #20 'A major where La mineur is expected' bug.

Synthesises the Let-It-Be / Georgia-On-My-Mind progression C-G-Am-F (a diatonic
I-V-vi-IV in C major), renders it, and runs infer_chords_v1 under both ctx
classifier variants. The load-bearing check: is the Am bar called MINOR (correct
— the vi of C) or MAJOR (the bug the key-relative feature is meant to fix)?

Usage:
    .venv/bin/python scripts/ref_test_let_it_be.py
"""
from __future__ import annotations

import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np  # noqa: F401
import pretty_midi

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig
from harmonia.models import chord_pipeline_v1 as P

# C major diatonic I-V-vi-IV, root + triad, one chord per bar
PROG = [
    ("C", 60, [60, 64, 67]),   # C major
    ("G", 55, [55, 59, 62]),   # G major
    ("Am", 57, [57, 60, 64]),  # A minor  <-- the chord under test
    ("F", 53, [53, 57, 60]),   # F major
]
BPM = 100
BEATS_PER_BAR = 4
N_LOOPS = 4


def build_midi() -> pretty_midi.PrettyMIDI:
    pm = pretty_midi.PrettyMIDI(initial_tempo=BPM)
    piano = pretty_midi.Instrument(program=0, name="piano")
    bass = pretty_midi.Instrument(program=32, name="bass")
    spb = 60.0 / BPM
    bar = spb * BEATS_PER_BAR
    t = 0.0
    for _ in range(N_LOOPS):
        for _name, root, triad in PROG:
            for p in triad:
                piano.notes.append(pretty_midi.Note(90, p, t, t + bar))
            bass.notes.append(pretty_midi.Note(95, root - 12, t, t + bar))
            t += bar
    pm.instruments += [piano, bass]
    return pm


def run(variant: str, wav: Path):
    # production defaults: progression prior ON (the shared realization path for
    # the ctx family decision), the morning LocalKeySeqGRU reranker OFF.
    chart = P.infer_chords_v1(
        wav, ctx_classifier_variant=variant,
        use_progression_prior=True, use_local_key_prior=False,
    )
    return [c["label"] for c in chart.chords]


def main():
    pm = build_midi()
    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2 = renderer._find_soundfont("MuseScore_General.sf2")
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as mf, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
        pm.write(mf.name)
        renderer.render(Path(mf.name), Path(wf.name), RenderConfig(soundfont_path=sf2))
        wav = Path(wf.name)
        print("progression: C - G - Am - F  (x%d), %d BPM" % (N_LOOPS, BPM))
        for variant in ("684d", "801d_two_pass"):
            labels = run(variant, wav)
            # find the label(s) rooted on A (pc 9) — the chord under test
            a_labels = [lab for lab in labels if lab.split(":")[0] in ("A",)]
            verdict = "?"
            if a_labels:
                quals = [lab.split(":", 1)[1] for lab in a_labels]
                is_min = all(q.startswith("min") for q in quals)
                is_maj = any(q in ("maj", "maj7", "7") for q in quals)
                verdict = ("MINOR ✓ (bug fixed)" if is_min
                           else "MAJOR ✗ (bug present)" if is_maj else "mixed")
            print(f"\n[{variant}]  A-rooted chords: {a_labels}  -> {verdict}")
            print(f"  full: {labels}")
        Path(mf.name).unlink(missing_ok=True)
        wav.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
