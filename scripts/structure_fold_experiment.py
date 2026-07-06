"""Does the STRUCTURE prior work if we generate synthetic data with the property that
matters — INDEPENDENT repeats? The fold died before because each section-repeat was the
same MIDI (correlated Basic Pitch errors → pooling averages nothing). Here we render each
occurrence with an independent voicing (vary_voicings) + independent noise, so the same
chord has a different audio surface each repeat → independent errors → poolable.

Test: for each "slot" (same section-label + same position within the section, i.e. beats
that carry the SAME chord across repeats), compare root accuracy of
  single   one occurrence's evidence
  pooled   audio evidence summed across all occurrences of the slot (the fold)
If pooled >> single, the structure prior is unblocked by the right synthetic data.

Usage: .venv/bin/python scripts/structure_fold_experiment.py --n-songs 12 [--degrade]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO)); sys.path.insert(0, str(REPO / "scripts"))

from analyze_accomp_emission import parse_chord, song_chord_spans  # noqa: E402
from build_accomp_audio_hard import time_varying_degrade, vary_voicings  # noqa: E402
from build_audio_chord_features import BUCKET_FAMILY  # noqa: E402
from root_model_experiment import TEMPLATES, chroma88  # noqa: E402
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig  # noqa: E402
from harmonia.models.stage1_pitch import PitchExtractor  # noqa: E402
from harmonic_rhythm_probe import pool_beats  # noqa: E402

DB = REPO / "data" / "accomp_db" / "db.jsonl"


def predict_root(on, nt, M):
    oc = chroma88(on)
    f = np.concatenate([oc, chroma88(nt), chroma88(on, 0, 52), chroma88(on, 60, 200)])
    if len(M["mean"]) == 60:
        f = np.concatenate([f, [max(oc @ t for r2, t in TEMPLATES if r2 == r) for r in range(12)]])
    z = (f - M["mean"]) / M["scale"]
    return int(M["classes"][np.argmax(z @ M["coef"].T + M["intercept"])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-songs", type=int, default=12)
    ap.add_argument("--degrade", action="store_true")
    ap.add_argument("--no-vary", action="store_true",
                    help="control: correlated repeats (same MIDI, no voicing variation)")
    args = ap.parse_args()
    M = dict(np.load(REPO / "harmonia/models/root_model.npz"))

    recs = [json.loads(l) for l in open(DB)]
    songs = [r for r in recs if r["corpus"] == "jazz1460" and r["beats_per_bar"] == 4
             and (REPO / r["midi_path"]).exists() and len(set(r["section_per_bar"])) > 1]
    songs = songs[:: max(len(songs) // args.n_songs, 1)][: args.n_songs]
    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2 = renderer._find_soundfont("MuseScore_General.sf2")
    ex = PitchExtractor(cache_dir=None)
    rng = np.random.default_rng(5)

    single_hit = single_tot = pool_hit = pool_tot = 0
    for rec in songs:
        spb = 60.0 / rec["tempo"]; bpb = rec["beats_per_bar"]; nb = rec["n_bars"]
        n_beats = nb * bpb; sv = rec["section_per_bar"]
        pm = pretty_midi.PrettyMIDI(str(REPO / rec["midi_path"]))
        if not args.no_vary:
            pm = vary_voicings(pm, sv, spb, bpb, rng)      # INDEPENDENT repeats
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as mf:
            mid = Path(mf.name)
        pm.write(str(mid))
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
            tmp = Path(wf.name)
        try:
            renderer.render(mid, tmp, RenderConfig(soundfont_path=sf2))
            y, sr = sf.read(tmp); y = (y.mean(1) if y.ndim > 1 else y).astype("float32")
            if args.degrade:
                y = time_varying_degrade(y, sr, rng); sf.write(tmp, y, sr)
            acts = ex.extract(tmp, use_cache=False)
        finally:
            mid.unlink(missing_ok=True); tmp.unlink(missing_ok=True)
        bt = np.arange(n_beats + 1) * spb
        onb = pool_beats(acts.frame_times, acts.onset_probs, bt)
        ntb = pool_beats(acts.frame_times, acts.note_probs, bt)

        # section-relative position of each bar (for slot alignment)
        sec_start = {}; i = 0
        while i < nb:
            j = i
            while j < nb and sv[j] == sv[i]:
                j += 1
            for b in range(i, j):
                sec_start[b] = i
            i = j

        # GT root per beat
        def gtroot(t):
            for t0, t1, root, _q in song_chord_spans(rec):
                if t0 <= t < t1:
                    return root % 12
            return None

        # group beats into slots: (section label, bar-within-section, beat-in-bar)
        slots = defaultdict(list)
        for b in range(n_beats):
            bar = b // bpb
            slots[(sv[bar], bar - sec_start[bar], b % bpb)].append(b)

        for beats in slots.values():
            if len(beats) < 2:
                continue
            g = gtroot((beats[0] + 0.5) * spb)
            if g is None:
                continue
            for b in beats:                                # single-instance accuracy
                single_tot += 1
                single_hit += int(predict_root(onb[b], ntb[b], M) == g)
            on_p = sum(onb[b] for b in beats); nt_p = sum(ntb[b] for b in beats)  # fold
            pool_tot += 1
            pool_hit += int(predict_root(on_p, nt_p, M) == g)

    cond = "DEGRADED" if args.degrade else "clean"
    print(f"\n=== structure fold with INDEPENDENT repeats, {len(songs)} {cond} songs ===")
    print(f"  repeated slots: {pool_tot}, instances: {single_tot} "
          f"(avg {single_tot / max(pool_tot,1):.1f} repeats/slot)")
    print(f"  single-instance root acc : {single_hit / max(single_tot,1):.1%}")
    print(f"  pooled-across-repeats acc: {pool_hit / max(pool_tot,1):.1%}")
    print("\n  pooled >> single => independent repeats make the structure prior work.")


if __name__ == "__main__":
    main()
