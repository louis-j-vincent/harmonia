"""Does isolating a stem help the task that stem is most informative for?

The user's question, made measurable. For each song we render isolated stems from
the MIDI (free, perfect separation) and ask:

  STRUCTURE (drums)   per-bar drum fingerprint → self-similarity → section boundaries.
                      Score novelty peaks vs the GT section_per_bar boundaries.
                      Compare stems: drums vs harmonic (chords) vs full mix.
  ROOT (bass)         per-beat root from an isolated bass stem vs from the low
                      register of the full mix (what pipeline_v0 does today).
                      Score vs the GT chord root.

Everything permissive-licensed; on real audio the stems come from HDemucs
(torchaudio, MIT). Disk-safe: one WAV at a time, deleted after use.

Usage: .venv/bin/python scripts/stem_benefit.py --n-songs 8 [--degrade]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import warnings
from pathlib import Path

import librosa
import numpy as np
import pretty_midi
import soundfile as sf

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from build_accomp_audio_hard import time_varying_degrade  # noqa: E402
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig  # noqa: E402
from harmonia.models.stage1_pitch import PitchExtractor  # noqa: E402

DB = REPO / "data" / "accomp_db" / "db.jsonl"
NOTE_TO_PC = {"C": 0, "D": 2, "E": 4, "F": 5, "G": 7, "A": 9, "B": 11}


# ---------- stems ---------------------------------------------------------
def stem_midi(src, out, keep_drums, keep_harmonic, keep_bass_only=False):
    pm = pretty_midi.PrettyMIDI(str(src))
    def keep(i):
        if i.is_drum:
            return keep_drums
        is_bass = "bass" in (i.name or "").lower() or i.program in range(32, 40)
        if keep_bass_only:
            return is_bass
        return keep_harmonic
    pm.instruments = [i for i in pm.instruments if keep(i)]
    pm.write(str(out))


# ---------- structure via self-similarity ---------------------------------
def per_bar_drum_feat(y, sr, n_bars, spb, bpb):
    """Per-bar rhythmic fingerprint: 3-band onset envelope resampled to 16 bins/bar."""
    bands = librosa.onset.onset_strength_multi(
        y=y, sr=sr, channels=[0, 16, 48, 128])            # low/mid/high mel bands
    hop_t = librosa.frames_to_time(np.arange(bands.shape[1]), sr=sr)
    feats = []
    for b in range(n_bars):
        t0, t1 = b * bpb * spb, (b + 1) * bpb * spb
        m = (hop_t >= t0) & (hop_t < t1)
        if m.sum() < 2:
            feats.append(np.zeros(3 * 16)); continue
        seg = bands[:, m]                                  # (3, frames_in_bar)
        rs = np.stack([np.interp(np.linspace(0, 1, 16), np.linspace(0, 1, seg.shape[1]), seg[c])
                       for c in range(3)]).ravel()
        feats.append(rs)
    F = np.array(feats)
    n = np.linalg.norm(F, axis=1, keepdims=True)
    return F / (n + 1e-9)


def per_bar_chroma(y, sr, n_bars, spb, bpb):
    ch = librosa.feature.chroma_cqt(y=y, sr=sr)
    ct = librosa.frames_to_time(np.arange(ch.shape[1]), sr=sr)
    feats = []
    for b in range(n_bars):
        t0, t1 = b * bpb * spb, (b + 1) * bpb * spb
        m = (ct >= t0) & (ct < t1)
        v = ch[:, m].mean(1) if m.any() else np.zeros(12)
        feats.append(v)
    F = np.array(feats); n = np.linalg.norm(F, axis=1, keepdims=True)
    return F / (n + 1e-9)


def _checker(L):
    g = np.outer(*[np.exp(-0.5 * (np.linspace(-2, 2, 2 * L)) ** 2)] * 2)
    k = np.ones((2 * L, 2 * L))
    k[:L, L:] = k[L:, :L] = -1
    return k * g


def novelty(F, L=4):
    S = F @ F.T
    n = len(S); k = _checker(L); P = np.pad(S, L)
    return np.array([(k * P[i:i + 2 * L, i:i + 2 * L]).sum() for i in range(n)])


def boundary_f(nov, gt_bounds, k, tol=1):
    """Take top-k novelty peaks (oracle count) → F-measure vs GT boundaries, ±tol bars."""
    order = [i for i in np.argsort(-nov) if 0 < i < len(nov)]
    chosen, est = [], []
    for i in order:
        if all(abs(i - c) > 1 for c in chosen):
            chosen.append(i); est.append(i)
        if len(est) >= k:
            break
    hits = sum(any(abs(e - g) <= tol for e in est) for g in gt_bounds)
    p = hits / (len(est) + 1e-9); r = hits / (len(gt_bounds) + 1e-9)
    return 2 * p * r / (p + r + 1e-9)


# ---------- root via bass -------------------------------------------------
def beat_roots(acts, beat_times, lo, hi):
    roots = []
    for b in range(len(beat_times) - 1):
        m = (acts.frame_times >= beat_times[b]) & (acts.frame_times < beat_times[b + 1])
        if not m.any():
            roots.append(-1); continue
        v = acts.note_probs[m].mean(0)
        c = np.zeros(12)
        for kk in range(88):
            if lo <= 21 + kk < hi:
                c[(21 + kk) % 12] += v[kk]
        roots.append(int(c.argmax()) if c.sum() > 1e-6 else -1)
    return roots


def gt_root_at(rec, t):
    root = None
    for ev in rec["chord_timeline"]:
        if ev["time"] <= t + 1e-6:
            s = ev["ireal"] or ev["mma"]
            if s and s[0] in NOTE_TO_PC:
                pc = NOTE_TO_PC[s[0]]
                if len(s) > 1 and s[1] == "#": pc = (pc + 1) % 12
                elif len(s) > 1 and s[1] == "b": pc = (pc - 1) % 12
                root = pc
        else:
            break
    return root


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-songs", type=int, default=8)
    ap.add_argument("--degrade", action="store_true")
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(DB)]
    songs = [r for r in recs if r["corpus"] == "jazz1460" and r["beats_per_bar"] == 4
             and (REPO / r["midi_path"]).exists() and len(set(r["section_per_bar"])) > 1]
    songs = songs[:: max(len(songs) // args.n_songs, 1)][: args.n_songs]

    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2 = renderer._find_soundfont("MuseScore_General.sf2")
    ex = PitchExtractor(cache_dir=None)
    rng = np.random.default_rng(3)

    struct = {"drums": [], "harmonic": [], "full": []}
    root = {"bass_stem": [], "full_lowreg": []}

    def render(midi, degrade):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
            tmp = Path(wf.name)
        renderer.render(midi, tmp, RenderConfig(soundfont_path=sf2))
        y, sr = sf.read(tmp); y = (y.mean(1) if y.ndim > 1 else y).astype("float32")
        if degrade:
            y = time_varying_degrade(y, sr, rng)
        return y, sr, tmp

    def filt(src, **kw):
        with tempfile.NamedTemporaryFile(suffix=".mid", delete=False) as mf:
            mid = Path(mf.name)
        stem_midi(src, mid, **kw)
        return mid

    for rec in songs:
        src = REPO / rec["midi_path"]
        spb = 60.0 / rec["tempo"]; bpb = rec["beats_per_bar"]; nb = rec["n_bars"]
        spb_v = rec["section_per_bar"]
        gt_bounds = [b for b in range(1, nb) if spb_v[b] != spb_v[b - 1]]
        beat_times = np.arange(nb * bpb + 1) * spb

        # ---- STRUCTURE: drums vs harmonic vs full ----
        for name, kw in [("drums", dict(keep_drums=True, keep_harmonic=False)),
                         ("harmonic", dict(keep_drums=False, keep_harmonic=True)),
                         ("full", dict(keep_drums=True, keep_harmonic=True))]:
            mid = filt(src, **kw)
            try:
                y, sr, tmp = render(mid, args.degrade)
            finally:
                mid.unlink(missing_ok=True)
            try:
                F = (per_bar_drum_feat if name == "drums" else per_bar_chroma)(y, sr, nb, spb, bpb)
                struct[name].append(boundary_f(novelty(F), gt_bounds, k=len(gt_bounds)))
            finally:
                tmp.unlink(missing_ok=True)

        # ---- ROOT: isolated bass stem vs low register of full mix ----
        gt = [gt_root_at(rec, beat_times[b]) for b in range(nb * bpb)]
        mid = filt(src, keep_drums=False, keep_harmonic=False, keep_bass_only=True)
        try:
            y, sr, tmp = render(mid, args.degrade)
        finally:
            mid.unlink(missing_ok=True)
        try:
            acts = ex.extract(tmp, use_cache=False)
            est_bass = beat_roots(acts, beat_times, 0, 200)      # whole bass stem
        finally:
            tmp.unlink(missing_ok=True)
        y, sr, tmp = render(filt(src, keep_drums=True, keep_harmonic=True), args.degrade)
        try:
            acts = ex.extract(tmp, use_cache=False)
            est_full = beat_roots(acts, beat_times, 0, 52)       # low register of full mix
        finally:
            tmp.unlink(missing_ok=True)
        ok = [(g, eb, ef) for g, eb, ef in zip(gt, est_bass, est_full) if g is not None]
        if ok:
            root["bass_stem"].append(np.mean([eb == g for g, eb, _ in ok]))
            root["full_lowreg"].append(np.mean([ef == g for g, _, ef in ok]))

    cond = "DEGRADED" if args.degrade else "clean"
    print(f"\n=== stem×task benefit, {len(struct['full'])} {cond} songs ===\n")
    print("STRUCTURE  (section-boundary F, top-k novelty peaks vs GT sections):")
    for k in ("drums", "harmonic", "full"):
        print(f"    {k:<9} {np.mean(struct[k]):.3f}")
    print("\nROOT  (per-beat root accuracy vs GT chord root):")
    for k in ("bass_stem", "full_lowreg"):
        print(f"    {k:<11} {np.mean(root[k]):.3f}")
    print("\nIsolation helps task X iff the isolated stem beats the full mix for X.")


if __name__ == "__main__":
    main()
