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
    """Give each OCCURRENCE of a section an independent audio surface for the SAME chords:
    octave-shifts (30% per non-bass note), velocity swings (±25%), and micro-timing jitter
    (±15ms). All pitch classes are preserved — chroma is unchanged. Independence is in the
    waveform (different BP onset errors per repeat), not in the harmony. This is what makes
    structure-folding useful: pooling across repeats averages out independent BP errors."""
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
        if inst.is_drum:
            # humanise drums per occurrence (velocity/timing) so the mix differs
            for t0, t1 in runs:
                rr = np.random.default_rng(rng.integers(0, 2**31))
                for n in inst.notes:
                    if t0 <= n.start < t1:
                        n.velocity = int(np.clip(n.velocity * rr.uniform(0.8, 1.15), 20, 127))
                        n.start = max(0, n.start + float(rr.uniform(-0.01, 0.01)))
            continue
        is_bass = "bass" in inst.name.lower()
        new_notes = []
        for t0, t1 in runs:
            local = [n for n in inst.notes if t0 <= n.start < t1]
            r = np.random.default_rng(rng.integers(0, 2**31))  # independent per occurrence
            # Vary the audio SURFACE only — all pitch classes kept (chroma unchanged).
            # Independence comes from register (octave shifts), dynamics (velocity), and
            # micro-timing — so Basic Pitch sees a different waveform but the same harmony.
            # No notes are dropped; no pitch classes are omitted.
            for n in local:
                pitch = n.pitch
                if not is_bass and r.random() < 0.30:   # octave-shift a non-bass voice
                    pitch = int(np.clip(pitch + 12 * r.choice([-1, 1]), 24, 96))
                vel = int(np.clip(n.velocity * r.uniform(0.75, 1.15), 20, 127))
                jit = float(r.uniform(-0.015, 0.015))
                new_notes.append(pretty_midi.Note(velocity=vel, pitch=pitch,
                                                  start=max(0, n.start + jit),
                                                  end=max(n.start + jit + 0.05, n.end + jit)))
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


def time_varying_degrade(x, sr, rng, phone=True):
    """Degrade the audio NON-UNIFORMLY over time (drifting gain, noise level, and
    muffling) so each repeat of a section is corrupted differently — the condition
    structure-folding needs. Optionally with the phone band-limit/hum on top."""
    n = len(x)
    if phone:
        # static band-limit + hum/hiss + gentle clip (the "phone" character)
        X = np.fft.rfft(x); f = np.fft.rfftfreq(n, 1 / sr)
        hp = 1 / (1 + (150 / np.maximum(f, 1)) ** 4)
        lp = 1 / (1 + (np.maximum(f, 1) / 6000) ** 4)
        x = np.fft.irfft(X * hp * lp, n).astype(np.float32)
        tt = np.arange(n) / sr
        x = x + (0.004 * np.sin(2 * np.pi * 50 * tt)).astype(np.float32)
    # smooth drifting control curves (mic distance / handling / background)
    K = max(5, n // (sr * 2))
    ctrl = np.linspace(0, n - 1, K)
    idx = np.arange(n)
    gain = np.interp(idx, ctrl, rng.uniform(0.5, 1.1, K)).astype(np.float32)
    snr = np.interp(idx, ctrl, rng.uniform(3.0, 20.0, K)).astype(np.float32)
    p = float(np.mean(x ** 2)) + 1e-9
    noise = pink(n, rng) * np.sqrt(p / (10 ** (snr / 10)))
    y = x * gain + noise
    if phone:
        g = rng.uniform(1.5, 3.0)
        y = np.tanh(g * y) / np.tanh(g)
    peak = np.abs(y).max()
    return (y * 0.99 / peak).astype(np.float32) if peak > 0.99 else y.astype(np.float32)


def strong_nonuniform_degrade(x, sr, rng):
    """Hard, VERY non-uniform grubby-recording chain — DISTORTION, not just noise:
    wow/flutter pitch warp, drifting muffle, tremolo, roomy comb smear, near-silent
    dropouts, wide gain/SNR swings, mains hum, asymmetric overdrive, hard-clip + bitcrush.
    Some regions stay intelligible, others are buried/warped — the weak-evidence
    condition. Pitch warp + saturation specifically corrupt the chord-relevant content."""
    from scipy.signal import istft, stft
    n = len(x); t = np.arange(n); tt = t / sr
    # A) wow & flutter: time-varying pitch via fractional resampling (smears harmony)
    depth = sr * float(rng.uniform(0.0008, 0.0022))
    warp = t + depth * (np.sin(2 * np.pi * rng.uniform(0.3, 1.0) * tt)
                        + 0.4 * np.sin(2 * np.pi * rng.uniform(6, 11) * tt))
    x = np.interp(warp, t, x).astype(np.float32)
    # B) phone-ish high-pass
    X = np.fft.rfft(x); f0 = np.fft.rfftfreq(n, 1 / sr)
    x = np.fft.irfft(X * (1 / (1 + (120 / np.maximum(f0, 1)) ** 4)), n).astype(np.float32)
    # C) drifting muffle: per-STFT-frame low-pass cutoff wandering 1.2–7 kHz
    f, _, Z = stft(x, fs=sr, nperseg=1024, noverlap=768)
    nf = Z.shape[1]
    knots = np.linspace(0, nf - 1, max(6, nf // 40))
    cutoff = np.interp(np.arange(nf), knots, rng.uniform(1200, 7000, len(knots)))
    Z = Z * (1.0 / (1 + (f[:, None] / cutoff[None, :]) ** 6))
    _, xm = istft(Z, fs=sr, nperseg=1024, noverlap=768)
    x = (xm[:n] if len(xm) >= n else np.pad(xm, (0, n - len(xm)))).astype(np.float32)
    # D) tremolo (amplitude wobble) + E) roomy comb / early reflections
    x = x * (1 + float(rng.uniform(0.15, 0.4)) * np.sin(2 * np.pi * rng.uniform(3, 7) * tt)).astype(np.float32)
    xc = x.copy()
    for d_s, g in [(rng.uniform(0.015, 0.03), rng.uniform(0.2, 0.4)),
                   (rng.uniform(0.03, 0.055), rng.uniform(0.1, 0.25))]:
        d = int(d_s * sr)
        if d < n:
            x[d:] += float(g) * xc[:-d]
    # F) very non-uniform drifting gain + SNR + near-silent dropouts + hum
    K = max(8, n // (sr // 2)); ctrl = np.linspace(0, n - 1, K)
    gain = np.interp(t, ctrl, rng.uniform(0.25, 1.15, K)).astype(np.float32)
    for _ in range(int(rng.integers(3, 6))):
        st = int(rng.integers(0, max(1, n - sr))); ln = int(rng.integers(sr // 8, sr // 2))
        gain[st:st + ln] *= float(rng.uniform(0.05, 0.25))
    snr = np.interp(t, ctrl, rng.uniform(-2.0, 14.0, K)).astype(np.float32)
    p = float(np.mean(x ** 2)) + 1e-9
    noise = pink(n, rng) * np.sqrt(p / (10 ** (snr / 10)))
    y = x * gain + noise + (0.006 * np.sin(2 * np.pi * 50 * tt)).astype(np.float32)
    # G) asymmetric overdrive (adds even + odd harmonics) then H) hard-clip + bitcrush
    g = float(rng.uniform(2.5, 5.0))
    y = np.tanh(g * y) / np.tanh(g); y = y + 0.08 * y ** 2
    thr = float(rng.uniform(0.6, 0.9)); y = np.clip(y, -thr, thr) / thr
    levels = int(rng.integers(32, 128)); y = np.round(y * levels) / levels
    peak = float(np.abs(y).max())
    return (y * 0.99 / peak).astype(np.float32) if peak > 0 else y.astype(np.float32)


def phone_degrade(x, sr, rng):
    """Make it sound like a cheap phone recording: band-limit (kill sub-bass and
    high treble), add mains hum + hiss, soft-clip (mic/AGC distortion), light
    bit-crush. Returns the grubbier signal."""
    n = len(x)
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(n, 1 / sr)
    # band-pass ~ phone mic: roll off below 150 Hz and above 6 kHz
    hp = 1 / (1 + (150 / np.maximum(f, 1)) ** 4)
    lp = 1 / (1 + (np.maximum(f, 1) / 6000) ** 4)
    x = np.fft.irfft(X * hp * lp, n).astype(np.float32)
    # mains hum (50 Hz + harmonic) + broadband hiss
    t = np.arange(n) / sr
    hum = 0.004 * (np.sin(2 * np.pi * 50 * t) + 0.4 * np.sin(2 * np.pi * 100 * t))
    x = x + hum.astype(np.float32) + (rng.standard_normal(n).astype(np.float32) * 0.006)
    # soft clip (AGC / cheap preamp) then light bit-crush
    g = rng.uniform(1.5, 3.0)
    x = np.tanh(g * x) / np.tanh(g)
    levels = rng.integers(64, 256)
    x = np.round(x * levels) / levels
    peak = np.abs(x).max()
    return (x * 0.99 / peak).astype(np.float32) if peak > 0.99 else x


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
    ap.add_argument("--full-diversity", action="store_true",
                    help="max diversity: force full band + varying melody every render")
    ap.add_argument("--phone", action="store_true",
                    help="simulate a grubby phone recording (band-limit, hum, hiss, clip, crush)")
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
                scen = "full_band" if args.full_diversity else str(rng.choice(list(SCENARIOS)))
                gains = {k: v * float(rng.uniform(0.8, 1.2)) for k, v in SCENARIOS[scen].items()}
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
                if args.phone:
                    mix = time_varying_degrade(mix, sr, rng)   # non-uniform in time
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
