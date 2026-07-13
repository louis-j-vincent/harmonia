"""debug_root_model.py — dissect beat_seq_model root accuracy on POP909.

Runs for all 5 rendered POP909 songs.  For each beat:
  - extracts 48d features (onset_full, note_full, bass, treble chroma)
  - compares: raw bass argmax, raw full-chroma argmax, beat_seq prediction, GT root
  - builds confusion matrices and per-register accuracy breakdown

Usage:
    .venv/bin/python scripts/debug_root_model.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import librosa
from harmonia.models.chord_pipeline_v1 import (
    MODELS, _BeatSeqModel, _chroma88, _pool_beats, _reg_raw,
)
from harmonia.models.stage1_pitch import PitchExtractor
from harmonia.data.pop909_parser import POP909Parser

NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
DATA_ROOT = REPO / "data"
POP909_DIR = DATA_ROOT / "pop909" / "POP909"


def tempo_grid_beats(y, sr):
    """Reproduce pipeline's tempo-grid de-jitter → uniform beat times."""
    tempo_arr, _ = librosa.beat.beat_track(y=y, sr=sr, units="time")
    tempo_bpm = float(np.atleast_1d(tempo_arr)[0])
    period = 60.0 / max(tempo_bpm, 1.0)
    # circular-mean phase from librosa raw beats
    _, raw_frames = librosa.beat.beat_track(y=y, sr=sr)
    raw_times = librosa.frames_to_time(raw_frames, sr=sr)
    if len(raw_times) > 0:
        phases = (raw_times % period) / period
        sin_m = np.sin(2 * np.pi * phases).mean()
        cos_m = np.cos(2 * np.pi * phases).mean()
        phase0 = float(np.arctan2(sin_m, cos_m) / (2 * np.pi) * period) % period
    else:
        phase0 = 0.0
    dur = librosa.get_duration(y=y, sr=sr)
    bt = np.arange(0, dur + period, period) + phase0
    bt = bt[bt < dur + 0.5 * period]
    return bt, tempo_bpm


def gt_root_per_beat(song, beat_times):
    """For each beat interval midpoint, look up GT root from POP909 chord_midi.txt."""
    gt = []
    for b in range(len(beat_times) - 1):
        t_mid = 0.5 * (beat_times[b] + beat_times[b + 1])
        ev = song.chord_at_time(t_mid)
        if ev is None:
            gt.append(None)
        else:
            root_str = ev.label.split(":")[0].split("/")[0]
            if root_str == "N":
                gt.append(None)
            else:
                try:
                    gt.append(NOTE.index(root_str) if root_str in NOTE else None)
                except ValueError:
                    gt.append(None)
    return gt


def print_confusion(conf, title="Confusion (GT row, pred col)"):
    """Print 12×12 confusion matrix with root names."""
    print(f"\n{title}  (normalised by GT row)")
    header = "     " + "".join(f"{n:>4}" for n in NOTE)
    print(header)
    for r in range(12):
        row_sum = conf[r].sum()
        if row_sum == 0:
            continue
        row_n = conf[r] / row_sum
        vals = "".join(f"{v:4.0f}" if v < 0.05 else f"\033[1m{v:4.0f}\033[0m"
                       for v in (row_n * 100))
        print(f"{NOTE[r]:>3}  {vals}")


def interval_name(delta):
    names = ["U", "m2", "M2", "m3", "M3", "4", "TT", "5", "m6", "M6", "m7", "M7"]
    return names[delta % 12]


def main():
    bsm_path = MODELS / "beat_seq_model.npz"
    if not bsm_path.exists():
        print("beat_seq_model.npz not found"); return
    bsm = _BeatSeqModel(bsm_path)
    ex = PitchExtractor(cache_dir=DATA_ROOT / "cache")
    parser = POP909Parser(POP909_DIR)

    songs = ["001", "002", "003", "004", "005"]

    # accumulators
    all_gt, all_bsm_pred, all_bass_pred, all_full_pred = [], [], [], []
    all_bass_conf, all_bsm_conf = [], []  # top-1 confidence

    for sid in songs:
        wav = DATA_ROOT / "renders" / "pop909" / sid / f"{sid}_v005_musescoregeneral.wav"
        if not wav.exists():
            print(f"skip {sid} — wav not found"); continue
        print(f"\n{'='*60}")
        print(f"Song {sid}")

        import soundfile as sf
        y, sr = sf.read(wav)
        y = (y.mean(1) if y.ndim > 1 else y).astype("float32")

        bt, tempo = tempo_grid_beats(y, sr)
        print(f"  tempo={tempo:.1f} BPM  n_beats={len(bt)-1}")

        acts = ex.extract(wav)
        onset_b = _pool_beats(acts.frame_times, acts.onset_probs, bt)  # (n_beats, 88)
        note_b  = _pool_beats(acts.frame_times, acts.note_probs,  bt)

        bsm_proba = bsm.predict_proba(onset_b, note_b)  # (n_beats, 12)
        bsm_pred  = bsm_proba.argmax(1)
        bsm_conf  = bsm_proba.max(1)

        # raw bass argmax (onset in MIDI 21-52, roughly E0-E3)
        bass_chroma = np.array([_reg_raw(onset_b[b], 0, 52) for b in range(len(onset_b))])
        full_chroma = np.array([_reg_raw(onset_b[b]) for b in range(len(onset_b))])
        bass_pred = bass_chroma.argmax(1)
        full_pred = full_chroma.argmax(1)
        bass_conf = bass_chroma / (bass_chroma.sum(1, keepdims=True) + 1e-9)

        gt_song = parser.parse_song(sid)
        gt = gt_root_per_beat(gt_song, bt)
        n = min(len(gt), len(bsm_pred))

        valid = [(i, g) for i, g in enumerate(gt[:n]) if g is not None]
        if not valid:
            print("  no GT roots"); continue

        idx_v = [i for i, _ in valid]
        gt_v  = np.array([g for _, g in valid])
        bsm_v = bsm_pred[idx_v]
        bass_v = bass_pred[idx_v]
        full_v = full_pred[idx_v]
        bsm_c_v  = bsm_conf[idx_v]
        bass_c_v = bass_conf[idx_v, bass_v]

        bsm_acc  = (bsm_v  == gt_v).mean()
        bass_acc = (bass_v == gt_v).mean()
        full_acc = (full_v == gt_v).mean()

        print(f"  n_valid_beats={len(gt_v)}")
        print(f"  beat_seq model:  {bsm_acc:.1%}   mean_conf={bsm_c_v.mean():.2f}")
        print(f"  raw bass argmax: {bass_acc:.1%}   mean_conf={bass_c_v.mean():.2f}")
        print(f"  raw full argmax: {full_acc:.1%}")

        # where does bass argmax help vs hurt vs agree with beat_seq?
        bsm_right  = bsm_v  == gt_v
        bass_right = bass_v == gt_v
        both_right = bsm_right & bass_right
        bsm_only   = bsm_right & ~bass_right
        bass_only  = ~bsm_right & bass_right
        both_wrong = ~bsm_right & ~bass_wrong if False else ~bsm_right & ~bass_right

        print(f"\n  Agreement analysis:")
        print(f"    both correct:        {both_right.sum():4d} ({both_right.mean():.1%})")
        print(f"    beat_seq only right: {bsm_only.sum():4d}  ({bsm_only.mean():.1%})  ← model beats raw bass")
        print(f"    bass only right:     {bass_only.sum():4d}  ({bass_only.mean():.1%})  ← raw bass beats model")
        print(f"    both wrong:          {both_wrong.sum():4d}  ({both_wrong.mean():.1%})")

        # error interval distribution for beat_seq
        errs = bsm_v[~bsm_right]
        gt_e = gt_v[~bsm_right]
        intervals = (errs - gt_e) % 12
        if len(intervals):
            counts = np.bincount(intervals, minlength=12)
            top = sorted(enumerate(counts), key=lambda x: -x[1])[:5]
            print(f"\n  beat_seq error intervals (semitones off GT):")
            for delta, cnt in top:
                if cnt > 0:
                    print(f"    {interval_name(delta):>3} (+{delta:2d} st): {cnt:3d} ({cnt/len(intervals):.1%})")

        # error interval distribution for bass argmax
        errs_b = bass_v[~bass_right]
        gt_eb  = gt_v[~bass_right]
        intervals_b = (errs_b - gt_eb) % 12
        if len(intervals_b):
            counts_b = np.bincount(intervals_b, minlength=12)
            top_b = sorted(enumerate(counts_b), key=lambda x: -x[1])[:5]
            print(f"\n  bass argmax error intervals:")
            for delta, cnt in top_b:
                if cnt > 0:
                    print(f"    {interval_name(delta):>3} (+{delta:2d} st): {cnt:3d} ({cnt/len(intervals_b):.1%})")

        all_gt.append(gt_v)
        all_bsm_pred.append(bsm_v)
        all_bass_pred.append(bass_v)
        all_full_pred.append(full_v)
        all_bsm_conf.append(bsm_c_v)
        all_bass_conf.append(bass_c_v)

    # ── corpus-level summary ──────────────────────────────────────────────────
    if not all_gt:
        print("no data"); return

    gt_all   = np.concatenate(all_gt)
    bsm_all  = np.concatenate(all_bsm_pred)
    bass_all = np.concatenate(all_bass_pred)
    full_all = np.concatenate(all_full_pred)
    bsm_conf_all  = np.concatenate(all_bsm_conf)
    bass_conf_all = np.concatenate(all_bass_conf)

    print(f"\n{'='*60}")
    print(f"CORPUS SUMMARY  (N={len(gt_all)} valid beats across 5 songs)")
    print(f"  beat_seq model:  {(bsm_all==gt_all).mean():.1%}  (88.3% on iReal CV)")
    print(f"  raw bass argmax: {(bass_all==gt_all).mean():.1%}")
    print(f"  raw full argmax: {(full_all==gt_all).mean():.1%}")

    bsm_right  = bsm_all  == gt_all
    bass_right = bass_all == gt_all
    bass_only  = ~bsm_right & bass_right
    bsm_only   = bsm_right & ~bass_right
    both_wrong = ~bsm_right & ~bass_right

    print(f"\n  both correct:        {(bsm_right & bass_right).mean():.1%}")
    print(f"  beat_seq only right: {bsm_only.mean():.1%}  ← model adds value over raw bass")
    print(f"  bass only right:     {bass_only.mean():.1%}  ← raw bass adds value over model")
    print(f"  both wrong:          {both_wrong.mean():.1%}")

    # confidence-stratified accuracy for beat_seq
    print(f"\n  beat_seq confidence-stratified accuracy:")
    for lo, hi in [(0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]:
        mask = (bsm_conf_all >= lo) & (bsm_conf_all < hi)
        if mask.sum() > 0:
            acc = (bsm_all[mask] == gt_all[mask]).mean()
            print(f"    conf [{lo:.1f},{hi:.1f}): n={mask.sum():4d}  acc={acc:.1%}")

    # corpus-level error intervals for beat_seq
    errs_all = bsm_all[~bsm_right]
    gt_ea    = gt_all[~bsm_right]
    if len(errs_all):
        ivs = (errs_all - gt_ea) % 12
        counts = np.bincount(ivs, minlength=12)
        print(f"\n  beat_seq corpus error intervals:")
        for delta, cnt in sorted(enumerate(counts), key=lambda x: -x[1])[:6]:
            if cnt > 0:
                print(f"    {interval_name(delta):>3} (+{delta:2d} st): {cnt:4d} ({cnt/len(errs_all):.1%})")

    # corpus-level error intervals for bass
    errs_b_all = bass_all[~bass_right]
    gt_eb_all  = gt_all[~bass_right]
    if len(errs_b_all):
        ivs_b = (errs_b_all - gt_eb_all) % 12
        counts_b = np.bincount(ivs_b, minlength=12)
        print(f"\n  bass argmax corpus error intervals:")
        for delta, cnt in sorted(enumerate(counts_b), key=lambda x: -x[1])[:6]:
            if cnt > 0:
                print(f"    {interval_name(delta):>3} (+{delta:2d} st): {cnt:4d} ({cnt/len(errs_b_all):.1%})")

    # per-key accuracy (where is the model systematically weak?)
    print(f"\n  beat_seq accuracy by GT root:")
    for r in range(12):
        mask = gt_all == r
        if mask.sum() < 5:
            continue
        acc = (bsm_all[mask] == r).mean()
        bass_a = (bass_all[mask] == r).mean()
        bar = "█" * int(acc * 20)
        print(f"    {NOTE[r]:>3}: {acc:.1%} {bar:<20}  (bass={bass_a:.1%})  n={mask.sum()}")


if __name__ == "__main__":
    main()
