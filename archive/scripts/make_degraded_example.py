"""Render one song clean + hard-degraded so the strength is audible. Writes both WAVs
to demo_audio/ for A/B listening. Usage: .venv/bin/python scripts/make_degraded_example.py"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

from build_accomp_audio_hard import strong_nonuniform_degrade, time_varying_degrade  # noqa: E402
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig  # noqa: E402

OUT = REPO / "demo_audio"; OUT.mkdir(exist_ok=True)


def main():
    recs = [json.loads(l) for l in open(REPO / "data/accomp_db/db.jsonl")]
    rec = next(r for r in recs if r["corpus"] == "jazz1460" and r["beats_per_bar"] == 4
               and (REPO / r["midi_path"]).exists() and len(set(r["section_per_bar"])) > 1)
    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2 = renderer._find_soundfont("MuseScore_General.sf2")
    tmp = OUT / "_clean.wav"
    renderer.render(REPO / rec["midi_path"], tmp, RenderConfig(soundfont_path=sf2))
    y, sr = sf.read(tmp); y = (y.mean(1) if y.ndim > 1 else y).astype("float32")
    rng = np.random.default_rng(7)

    sf.write(OUT / "example_clean.wav", y, sr)
    sf.write(OUT / "example_degraded_medium.wav", time_varying_degrade(y.copy(), sr, rng), sr)
    sf.write(OUT / "example_degraded_hard.wav", strong_nonuniform_degrade(y.copy(), sr, rng), sr)
    tmp.unlink(missing_ok=True)
    print(f"'{rec['title']}' ({rec['tempo']} BPM, {rec['form']}) → {OUT}/")
    for f in ("example_clean.wav", "example_degraded_medium.wav", "example_degraded_hard.wav"):
        print(f"    {f}")


if __name__ == "__main__":
    main()
