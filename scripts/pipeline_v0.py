"""End-to-end v0: raw audio → chords, using ONLY audio (no chart timing/root).

Assembles the validated experiment bricks into a real pipeline and evaluates it
end-to-end (MIREX weighted overlap), so we finally know if we have a solid
audio→chords model — the honest test the demo dodged by using the chart's timing.

Chain (all from the audio):
  1. beat tracking            librosa (tempo + beats from the waveform)
  2. per-beat features        Basic Pitch → onset/note chroma + bass/treble split
  3. chord-change detection   bass-PC change + chroma novelty + downbeat-ish phase
  4. root per segment         bass-register chroma argmax (the two-stage design)
  5. quality per segment      trained emission model (family, given the root)
  6. report                   family label (the reliable level)
Then mir_eval root/majmin vs the ground-truth chart — with DETECTED boundaries,
not oracle ones. Disk-safe: renders each song inline, deletes the WAV.

Usage: .venv/bin/python scripts/pipeline_v0.py --n-songs 15 [--degrade]
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import warnings
from pathlib import Path

import librosa
import mir_eval
import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from analyze_accomp_emission import parse_chord, song_chord_spans  # noqa: E402
from build_accomp_audio_hard import time_varying_degrade  # noqa: E402
from build_audio_chord_features import BUCKET_FAMILY, FAM_IDX, full_chroma, reg_chroma  # noqa: E402
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig  # noqa: E402
from harmonia.models.stage1_pitch import PitchExtractor  # noqa: E402

DB = REPO / "data" / "accomp_db" / "db.jsonl"
CLEAN_FEAT = REPO / "data" / "cache" / "audio_chord_features.npz"
FAMILIES = ["major", "minor", "diminished", "augmented", "suspended"]
NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FAM_HARTE = {"major": "maj", "minor": "min", "diminished": "dim",
             "augmented": "aug", "suspended": "sus4"}


def pool_to_beats(frame_times, probs, beat_times):
    """(n_beats, 88): mean frame activity within each beat interval."""
    out = np.zeros((len(beat_times) - 1, probs.shape[1]), dtype=np.float32)
    for b in range(len(beat_times) - 1):
        m = (frame_times >= beat_times[b]) & (frame_times < beat_times[b + 1])
        if m.any():
            out[b] = probs[m].mean(0)
    return out


def chroma_of(v88):
    c = np.zeros(12)
    for k in range(88):
        c[(k + 21) % 12] += v88[k]
    return c


def detect_changes(onset_beats, novelty_thresh=0.35):
    """Boundary before beat b if the beat's chroma / bass differ enough from the
    running segment — the chord-change detector (the missing brick)."""
    n = len(onset_beats)
    beat_chroma = np.stack([chroma_of(onset_beats[b]) for b in range(n)])
    beat_chroma /= (np.linalg.norm(beat_chroma, axis=1, keepdims=True) + 1e-9)
    bass_pc = []
    for b in range(n):
        c = np.zeros(12)
        for k in range(88):
            if 21 + k < 52:
                c[(k + 21) % 12] += onset_beats[b, k]
        bass_pc.append(int(c.argmax()) if c.sum() > 1e-6 else -1)
    bounds = [0]
    ref = beat_chroma[0]
    for b in range(1, n):
        novelty = 1 - float(ref @ beat_chroma[b])
        bass_changed = bass_pc[b] != bass_pc[b - 1] and bass_pc[b] >= 0
        if novelty > novelty_thresh or (bass_changed and novelty > novelty_thresh * 0.5):
            bounds.append(b); ref = beat_chroma[b]
        else:
            ref = 0.7 * ref + 0.3 * beat_chroma[b]   # running mean within segment
            ref /= np.linalg.norm(ref) + 1e-9
    bounds.append(n)
    return bounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-songs", type=int, default=15)
    ap.add_argument("--degrade", action="store_true")
    ap.add_argument("--novelty", type=float, default=0.35)
    args = ap.parse_args()

    # trained emission model (family), on the clean feature set
    d = np.load(CLEAN_FEAT, allow_pickle=True)
    Xc = np.hstack([d["onset"], d["note"], d["bass"], d["treble"]])
    sc = StandardScaler().fit(Xc)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xc), d["family"].astype(int))

    records = [json.loads(l) for l in open(DB)]
    songs = [r for r in records if r["corpus"] == "jazz1460" and r["beats_per_bar"] == 4
             and (REPO / r["midi_path"]).exists()]
    songs = songs[:: max(len(songs) // args.n_songs, 1)][: args.n_songs]

    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    ex = PitchExtractor(cache_dir=None)
    rng = np.random.default_rng(5)
    roots, majmins, n_seg_ratio = [], [], []
    for rec in songs:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
            tmp = Path(wf.name)
        try:
            renderer.render(REPO / rec["midi_path"], tmp,
                            RenderConfig(soundfont_path=renderer._find_soundfont("MuseScore_General.sf2")))
            y, sr = sf.read(tmp)
            y = y.mean(1) if y.ndim > 1 else y
            y = y.astype("float32")
            if args.degrade:
                y = time_varying_degrade(y, sr, rng)
                sf.write(tmp, y, sr)
            acts = ex.extract(tmp, use_cache=False)
        finally:
            tmp.unlink(missing_ok=True)

        # 1. beats from AUDIO (not the chart)
        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        if len(beat_times) < 4:
            continue
        onset_b = pool_to_beats(acts.frame_times, acts.onset_probs, beat_times)
        note_b = pool_to_beats(acts.frame_times, acts.note_probs, beat_times)
        # 2/3. detect chord-change boundaries
        bounds = detect_changes(onset_b, args.novelty)

        # 4/5. per segment: root from bass, quality from model
        est_int, est_lab = [], []
        for s, e in zip(bounds, bounds[1:]):
            seg_on = onset_b[s:e].sum(0); seg_nt = note_b[s:e].sum(0)
            if seg_on.sum() < 1e-6:
                continue
            bass = np.zeros(12)
            for k in range(88):
                if 21 + k < 52:
                    bass[(k + 21) % 12] += seg_on[k]
            root = int(bass.argmax()) if bass.sum() > 1e-6 else int(chroma_of(seg_on).argmax())
            rr = lambda c: np.roll(c, -root)
            feat = np.hstack([rr(chroma_of(seg_on)), rr(chroma_of(seg_nt)),
                              rr(_reg(seg_on, 0, 52)), rr(_reg(seg_on, 60, 200))])
            fam = FAMILIES[int(clf.predict(sc.transform(feat[None]))[0])]
            est_int.append([beat_times[s], beat_times[min(e, len(beat_times) - 1)]])
            est_lab.append(f"{NOTE[root]}:{FAM_HARTE[fam]}")

        # ground truth (family level, from the chart)
        spb = 60.0 / rec["tempo"]
        ref_int, ref_lab = [], []
        for t0, t1, r, _q in song_chord_spans(rec):
            mma = None
            for ev in rec["chord_timeline"]:
                if int(round(((ev["bar"] - 1) * rec["beats_per_bar"] + ev["beat"]))) == int(round(t0 / spb)):
                    mma = ev["mma"]; break
            p = parse_chord(mma) if mma else None
            if p is None or p[1] not in BUCKET_FAMILY:
                continue
            ref_int.append([t0, t1]); ref_lab.append(f"{NOTE[r % 12]}:{FAM_HARTE[BUCKET_FAMILY[p[1]]]}")
        if not est_int or not ref_int:
            continue
        scores = mir_eval.chord.evaluate(np.array(ref_int), ref_lab,
                                         np.array(est_int), est_lab)
        roots.append(scores["root"]); majmins.append(scores["majmin"])
        n_seg_ratio.append(len(est_int) / len(ref_int))

    cond = "DEGRADED" if args.degrade else "clean"
    print(f"\nEnd-to-end v0 on {len(roots)} {cond} jazz songs (audio→chords, DETECTED "
          f"beats+boundaries+root):")
    print(f"    root  (MIREX weighted overlap): {np.mean(roots):.1%}")
    print(f"    majmin                        : {np.mean(majmins):.1%}")
    print(f"    detected/GT segment ratio     : {np.mean(n_seg_ratio):.2f} "
          f"(1.0 = right count; <1 under-segments, >1 over-segments)")
    print("\nCompare: prod HMM ~37% root; oracle-boundary quality was 86.8% root — the gap "
          "is the\nchord-change detector's cost, now measured end-to-end.")


def _reg(v88, lo, hi):
    c = np.zeros(12)
    for k in range(88):
        if lo <= 21 + k < hi:
            c[(21 + k) % 12] += v88[k]
    return c


if __name__ == "__main__":
    main()
