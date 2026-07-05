"""Generate HARD, realistic audio: multi-instrument stems mixed with varying
balance (weak accompaniment vs loud drums / lead melody), complexity, and noise.

The clean pilot had complete comping voicings front-and-centre — that's why the
audio model looked so strong and the priors looked useless. Real recordings bury
the comping under drums and a lead voice, and drop chord tones. This renders the
same charts but:
  - splits the MMA accompaniment into stems (bass / chords / drums);
  - synthesizes a monophonic LEAD melody from the chord tones (+ occasional
    passing tones) on a lead instrument — the "voix de tête";
  - mixes the stems with a randomly drawn BALANCE scenario (e.g. drums-loud,
    melody-masks, sparse-quiet-comp) so the accompaniment is often weak;
  - adds pink noise at a random SNR; varies soundfont, reverb, transpose.

Every render logs all its parameters (scenario, per-stem gains, SNR, groove, …)
to manifest_hard.jsonl so results can be sliced by difficulty for the blog.

Usage: .venv/bin/python scripts/build_accomp_audio_hard.py --n-songs 60 --variants 2
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pretty_midi
import soundfile as sf

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig  # noqa: E402

DB = REPO / "data" / "accomp_db" / "db.jsonl"
AUDIO = REPO / "data" / "accomp_db" / "audio_hard"
SOUNDFONTS = ["MuseScore_General.sf2", "GeneralUser.sf2"]
LEAD_PROGRAMS = [73, 56, 65, 71]  # flute, trumpet, alto sax, clarinet

# balance scenarios: per-stem linear gains (chords, bass, drums, melody)
SCENARIOS = {
    "balanced":          dict(chords=1.0, bass=0.9, drums=0.7, melody=0.0),
    "drums_loud":        dict(chords=0.55, bass=0.8, drums=1.3, melody=0.0),
    "melody_masks":      dict(chords=0.5, bass=0.7, drums=0.6, melody=1.2),
    "sparse_quiet_comp": dict(chords=0.35, bass=0.7, drums=0.7, melody=1.0),
    "full_band":         dict(chords=0.8, bass=0.9, drums=1.0, melody=0.9),
}


def vary_voicings(pm, section_per_bar, spb, bpb, rng):
    """Give each OCCURRENCE of a section an independent voicing of the SAME chords,
    so repeats differ (dropped/added octave tones, velocity, micro-timing) while the
    harmony is unchanged — the variation that makes structure-folding useful."""
    import copy
    pm = copy.deepcopy(pm)
    # section-run time ranges (each occurrence gets its own random draws)
    runs = []
    i = 0
    while i < len(section_per_bar):
        j = i
        while j < len(section_per_bar) and section_per_bar[j] == section_per_bar[i]:
            j += 1
        runs.append((i * bpb * spb, j * bpb * spb))
        i = j
    for inst in pm.instruments:
        if inst.is_drum or "bass" in inst.name.lower():
            continue  # vary the comping voicing only; bass/drums stay
        new_notes = []
        for t0, t1 in runs:
            local = [n for n in inst.notes if t0 <= n.start < t1]
            r = np.random.default_rng(rng.integers(0, 2**31))  # independent per occurrence
            for n in local:
                if r.random() < 0.18:            # drop a voice (incomplete voicing)
                    continue
                pitch = n.pitch
                if r.random() < 0.30:            # octave-shift a voice (same pitch class)
                    pitch = int(np.clip(pitch + 12 * r.choice([-1, 1]), 24, 96))
                vel = int(np.clip(n.velocity * r.uniform(0.8, 1.12), 20, 127))
                jit = float(r.uniform(-0.012, 0.012))
                new_notes.append(pretty_midi.Note(velocity=vel, pitch=pitch,
                                                  start=max(0, n.start + jit),
                                                  end=n.end + jit))
        # keep notes outside any run unchanged
        covered = [n for n in inst.notes if any(t0 <= n.start < t1 for t0, t1 in runs)]
        new_notes += [n for n in inst.notes if n not in covered]
        inst.notes = new_notes
    return pm


def stem_midi(pm, keep_pred, program=None):
    out = pretty_midi.PrettyMIDI()
    for inst in pm.instruments:
        if keep_pred(inst):
            ni = pretty_midi.Instrument(program=program if program is not None else inst.program,
                                        is_drum=inst.is_drum, name=inst.name)
            ni.notes = list(inst.notes)
            out.instruments.append(ni)
    return out if out.instruments else None


def make_melody(pm, program, rng):
    """Monophonic lead from the comping's top voice, octave-lifted, + passing tones."""
    chord_notes = [n for i in pm.instruments if not i.is_drum and "bass" not in i.name.lower()
                   for n in i.notes]
    if not chord_notes:
        return None
    end = max(n.end for n in chord_notes)
    mel = pretty_midi.Instrument(program=program, name="Melody")
    t = 0.0
    beat = 0.5  # approx; melody rhythm is illustrative
    while t < end:
        active = [n.pitch for n in chord_notes if n.start <= t < n.end]
        if active:
            pitch = max(active)
            while pitch < 72:
                pitch += 12
            if rng.random() < 0.3:                    # passing / non-chord tone
                pitch += int(rng.choice([-2, -1, 1, 2]))
            pitch = int(np.clip(pitch, 60, 88))
            mel.notes.append(pretty_midi.Note(velocity=110, pitch=pitch,
                                              start=t, end=t + beat * 0.9))
        t += beat
    return mel if mel.notes else None


def render_to_array(renderer, pm, sf_name, reverb):
    with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as mf, \
         tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
        pm.write(mf.name)
        cfg = RenderConfig(soundfont_path=renderer._find_soundfont(sf_name), reverb=reverb)
        renderer.render(Path(mf.name), Path(wf.name), cfg)
        audio, sr = sf.read(wf.name)
    Path(mf.name).unlink(missing_ok=True); Path(wf.name).unlink(missing_ok=True)
    if audio.ndim > 1:
        audio = audio.mean(1)
    return audio.astype(np.float32), sr


def pink(n, rng):
    w = rng.standard_normal(n)
    s = np.fft.rfft(w); f = np.fft.rfftfreq(n); f[0] = f[1]
    p = np.fft.irfft(s / np.sqrt(f), n)
    return (p / (p.std() + 1e-9)).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-songs", type=int, default=60)
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--corpus", default="jazz1460")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--vary-voicings", action="store_true",
                    help="give each section occurrence its own voicing (for structure folding)")
    ap.add_argument("--out-suffix", default="",
                    help="append to manifest name, e.g. '_pop' or '_varied'")
    args = ap.parse_args()

    records = [json.loads(l) for l in open(DB)]
    songs = [r for r in records if r["corpus"] == args.corpus and r["beats_per_bar"] == 4]
    songs.sort(key=lambda r: (r["groove"], r["song_id"]))
    stride = max(len(songs) // args.n_songs, 1)
    songs = songs[::stride][: args.n_songs]

    rng = np.random.default_rng(args.seed)
    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    AUDIO.mkdir(parents=True, exist_ok=True)
    n = 0
    manifest_path = AUDIO / f"manifest_hard{args.out_suffix}.jsonl"
    with open(manifest_path, "w") as mf:
        for rec in songs:
            src = REPO / rec["midi_path"]
            if not src.exists():
                continue
            pm_base = pretty_midi.PrettyMIDI(str(src))
            spb0 = 60.0 / rec["tempo"]
            for vi in range(args.variants):
                pm = (vary_voicings(pm_base, rec["section_per_bar"], spb0,
                                    rec["beats_per_bar"], rng)
                      if args.vary_voicings else pm_base)
                scen = str(rng.choice(list(SCENARIOS)))
                gains = {k: v * float(rng.uniform(0.85, 1.15)) for k, v in SCENARIOS[scen].items()}
                sf_name = SOUNDFONTS[int(rng.integers(0, len(SOUNDFONTS)))]
                reverb = bool(rng.integers(0, 2))
                snr_opts = [None, 15.0, 8.0, 4.0]
                snr = snr_opts[int(rng.choice(len(snr_opts), p=[.25, .3, .3, .15]))]
                lead_prog = int(rng.choice(LEAD_PROGRAMS))

                stems = {
                    "chords": stem_midi(pm, lambda i: (not i.is_drum) and "bass" not in i.name.lower()),
                    "bass": stem_midi(pm, lambda i: "bass" in i.name.lower()),
                    "drums": stem_midi(pm, lambda i: i.is_drum),
                }
                if gains["melody"] > 0.01:
                    stems["melody"] = pretty_midi.PrettyMIDI()
                    m = make_melody(pm, lead_prog, rng)
                    if m:
                        stems["melody"].instruments.append(m)
                    else:
                        stems["melody"] = None

                waves = {}
                sr = 44100
                for name, s in stems.items():
                    if s is None or not s.instruments:
                        continue
                    waves[name], sr = render_to_array(renderer, s, sf_name, reverb)
                if not waves:
                    continue
                L = max(len(w) for w in waves.values())
                mix = np.zeros(L, dtype=np.float32)
                stems_present = []
                for name, w in waves.items():
                    g = gains.get(name, 0.0)
                    if g <= 0.01:
                        continue
                    mix[:len(w)] += g * w
                    stems_present.append(name)
                # noise
                if snr is not None:
                    sig = float(np.mean(mix ** 2)) + 1e-12
                    mix = mix + pink(L, rng) * np.sqrt(sig / (10 ** (snr / 10)))
                peak = np.abs(mix).max()
                if peak > 0.99:
                    mix *= 0.99 / peak

                out = AUDIO / f"{rec['song_id']}{args.out_suffix}_h{vi}_{scen}.wav"
                sf.write(out, mix, sr)
                mf.write(json.dumps({
                    "song_id": rec["song_id"], "wav": str(out.relative_to(REPO)),
                    "scenario": scen, "gains": {k: round(v, 2) for k, v in gains.items()},
                    "stems": stems_present, "soundfont": sf_name, "reverb": reverb,
                    "snr_db": snr, "lead_program": lead_prog, "transpose": 0,
                    "tempo": rec["tempo"], "beats_per_bar": rec["beats_per_bar"],
                    "n_bars": rec["n_bars"], "groove": rec["groove"],
                    "midi_path": rec["midi_path"],
                }) + "\n")
                n += 1
                if n % 20 == 0:
                    print(f"  … {n} hard renders")
    print(f"Done: {n} hard renders → {AUDIO}")


if __name__ == "__main__":
    main()
