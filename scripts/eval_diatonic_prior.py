"""End-to-end MIREX eval of the issue-#20 diatonic quality prior on BOTH corpora.

Renders MIDI → Basic Pitch → tempo-grid beats → v4 per-beat root → gmerge segs →
family classifier, then scores mir_eval root / majmin / sevenths against the
chart, for two variants:

    baseline  : acoustic family classifier only  (prior OFF)
    +diatonic : chord_pipeline_v1.apply_diatonic_prior gate applied per segment

The prior uses the *inferred* local key (infer_key over the segment window),
exactly as production infer_chords_v1 does — no GT-key leakage.

    jazz1460 : held-out songs (index >=70, UNSEEN by v4).  Expect ~49% diatonic
               → prior should be ~neutral (commit criterion: no >1pp majmin drop).
    POP909   : songs 001..00N.  ~93% diatonic → prior should lift majmin.

Usage:
    python scripts/eval_diatonic_prior.py --jazz-n 25 --pop-n 5
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import warnings
from pathlib import Path

import numpy as np
import soundfile as sf

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import librosa
import mir_eval

from analyze_accomp_emission import song_chord_spans
from build_audio_chord_features import BUCKET_FAMILY
from train_beat_seq_model_v3 import quality5 as jazz_quality5
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig
from harmonia.data.pop909_parser import POP909Parser
from harmonia.theory.chord_vocabulary import ChordQuality
from harmonia.models.stage1_pitch import PitchExtractor
from harmonia.models import chord_pipeline_v1 as P
from harmonia.theory.key_profiles import infer_key

DB = REPO / "data" / "accomp_db" / "db.jsonl"
POP909_DIR = REPO / "data" / "pop909" / "POP909"
NOTE = P.NOTE
Q5_HARTE = ["maj", "min", "7", "maj7", "dim"]  # major, minor, dom7, maj7, dim

# POP909 ChordQuality → Harte token (reduced to the sevenths vocabulary base)
_Q_HARTE = {
    ChordQuality.MAJOR: "maj", ChordQuality.MINOR: "min",
    ChordQuality.DIMINISHED: "dim", ChordQuality.AUGMENTED: "aug",
    ChordQuality.SUS2: "sus2", ChordQuality.SUS4: "sus4",
    ChordQuality.MAJ7: "maj7", ChordQuality.MIN7: "min7", ChordQuality.DOM7: "7",
    ChordQuality.MIN_MAJ7: "minmaj7", ChordQuality.HALF_DIM7: "hdim7",
    ChordQuality.DIM7: "dim7", ChordQuality.AUG_MAJ7: "maj7", ChordQuality.AUG7: "7",
    ChordQuality.DOM7SUS4: "sus4", ChordQuality.MAJ9: "maj7", ChordQuality.MIN9: "min7",
    ChordQuality.DOM9: "7", ChordQuality.DOM7B9: "7", ChordQuality.DOM7S9: "7",
    ChordQuality.DOM9SUS4: "sus4", ChordQuality.MAJ9S11: "maj7", ChordQuality.MIN11: "min7",
    ChordQuality.DOM7S11: "7", ChordQuality.DOM7B9S11: "7", ChordQuality.MAJ13: "maj7",
    ChordQuality.MIN13: "min7", ChordQuality.DOM13: "7", ChordQuality.DOM13B9: "7",
}


def tempo_grid(y, sr):
    tempo, bf = librosa.beat.beat_track(y=y, sr=sr)
    btl = librosa.frames_to_time(bf, sr=sr)
    period = 60.0 / float(np.atleast_1d(tempo)[0])
    ang = 2 * np.pi * (btl % period) / period
    phase = (np.angle(np.mean(np.exp(1j * ang))) % (2 * np.pi)) * period / (2 * np.pi)
    bt = np.arange(phase, len(y) / sr, period)
    return np.unique(np.concatenate([[0.0], bt, [len(y) / sr]]))


def gmerge_segs(beat_proba):
    pred = beat_proba.argmax(1)
    cuts = [0] + [b for b in range(1, len(pred)) if pred[b] != pred[b - 1]] + [len(pred)]
    return [(cuts[i], cuts[i + 1]) for i in range(len(cuts) - 1)]


def label_and_score(segs, bt, onset_b, note_b, beat_proba, fam_clf,
                    ref_int, ref_lab, *, use_prior, boost, thresh):
    n_beats = len(onset_b)
    labeled = []
    for s, e in segs:
        root = int(beat_proba[s:e].sum(0).argmax())
        seg_on = onset_b[s:e].sum(0); seg_nt = note_b[s:e].sum(0)
        seg_bs = P._reg_raw(seg_on, 0, 52); seg_tr = P._reg_raw(seg_on, 60, 200)
        _, sev_h, conf = fam_clf.predict(root, seg_on, seg_nt, seg_bs, seg_tr, 0.0)
        if use_prior:
            if (e - s) < 8:
                c = (s + e) // 2
                lo, hi = max(0, c - 16), min(n_beats, c + 16)
            else:
                lo, hi = s, e
            kp = infer_key(P._reg_raw(onset_b[lo:hi].sum(0)))
            sev_h = P.apply_diatonic_prior(
                root, sev_h, conf, kp.tonic, kp.mode, kp.confidence,
                diatonic_boost=boost, threshold_chromatic=thresh,
            )
        lab = f"{NOTE[root]}:{sev_h}"
        if labeled and labeled[-1][2] == lab:
            labeled[-1][1] = e
        else:
            labeled.append([s, e, lab])
    est_int = [[bt[s], bt[min(e, len(bt) - 1)]] for s, e, _ in labeled]
    est_lab = [lab for _, _, lab in labeled]
    keep = [(iv, lb) for iv, lb in zip(est_int, est_lab) if iv[1] > iv[0]]
    if not keep:
        return None
    ei, el = zip(*keep)
    try:
        sco = mir_eval.chord.evaluate(np.array(ref_int), ref_lab, np.array(ei), list(el))
    except ValueError:
        return None
    return sco["root"], sco["majmin"], sco["sevenths"]


def run_song(y, sr, acts, n_beats_hint, ref_int, ref_lab, v4, fam, args):
    bt = tempo_grid(y, sr)
    onset_b = P._pool_beats(acts.frame_times, acts.onset_probs, bt)
    note_b = P._pool_beats(acts.frame_times, acts.note_probs, bt)
    beat_proba = v4.predict_proba(onset_b, note_b)
    segs = gmerge_segs(beat_proba)
    out = {}
    for name, up in (("base", False), ("prior", True)):
        out[name] = label_and_score(
            segs, bt, onset_b, note_b, beat_proba, fam, ref_int, ref_lab,
            use_prior=up, boost=args.boost, thresh=args.thresh)
    return out


def eval_jazz(args, renderer, sf2, ex, v4, fam):
    recs = [json.loads(l) for l in open(DB)]
    jz = [r for r in recs if r.get("corpus") == "jazz1460"
          and r["beats_per_bar"] == 4 and (REPO / r["midi_path"]).exists()]
    held = jz[args.start:args.start + args.jazz_n]
    acc = {"base": [], "prior": []}
    for i, rec in enumerate(held):
        print(f"  jazz [{i+1}/{len(held)}] {rec['song_id']}          ", end="\r", flush=True)
        spans = [(t0, t1, r % 12, q) for t0, t1, r, q in song_chord_spans(rec)
                 if t1 > t0 and q in BUCKET_FAMILY]
        if not spans:
            continue
        ref_int = [[t0, t1] for t0, t1, _, _ in spans]
        ref_lab = []
        for _, _, r, q in spans:
            q5 = jazz_quality5(q)
            ref_lab.append(f"{NOTE[r]}:{Q5_HARTE[q5] if q5 is not None else 'maj'}")
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
            tmp = Path(wf.name)
        try:
            renderer.render(REPO / rec["midi_path"], tmp, RenderConfig(soundfont_path=sf2))
            y, sr = sf.read(tmp); y = (y.mean(1) if y.ndim > 1 else y).astype("float32")
            acts = ex.extract(tmp, use_cache=False)
        finally:
            tmp.unlink(missing_ok=True)
        res = run_song(y, sr, acts, None, ref_int, ref_lab, v4, fam, args)
        for k in acc:
            if res[k]:
                acc[k].append(res[k])
    print()
    return acc


def eval_pop(args, renderer, sf2, ex, v4, fam):
    parser = POP909Parser(POP909_DIR)
    ids = [f"{i:03d}" for i in range(1, args.pop_n + 1)]
    acc = {"base": [], "prior": []}
    for i, sid in enumerate(ids):
        song = parser.parse_song(sid)
        if song is None:
            continue
        print(f"  pop909 [{i+1}/{len(ids)}] {sid}          ", end="\r", flush=True)
        spans = []
        for ev in song.chord_events:
            if ev.root == -1 or ev.end_beat <= ev.start_beat:
                continue
            h = _Q_HARTE.get(ev.quality)
            if h is None:
                continue
            spans.append((ev.start_beat, ev.end_beat, ev.root, h))
        if not spans:
            continue
        ref_int = [[t0, t1] for t0, t1, _, _ in spans]
        ref_lab = [f"{NOTE[r]}:{h}" for _, _, r, h in spans]
        midi = POP909_DIR / sid / f"{sid}.mid"
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
            tmp = Path(wf.name)
        try:
            renderer.render(midi, tmp, RenderConfig(soundfont_path=sf2))
            y, sr = sf.read(tmp); y = (y.mean(1) if y.ndim > 1 else y).astype("float32")
            acts = ex.extract(tmp, use_cache=False)
        finally:
            tmp.unlink(missing_ok=True)
        res = run_song(y, sr, acts, None, ref_int, ref_lab, v4, fam, args)
        for k in acc:
            if res[k]:
                acc[k].append(res[k])
    print()
    return acc


def _summ(rows):
    a = np.array(rows)
    return a[:, 0].mean(), a[:, 1].mean(), a[:, 2].mean(), len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jazz-n", type=int, default=25)
    ap.add_argument("--pop-n", type=int, default=5)
    ap.add_argument("--start", type=int, default=70)
    ap.add_argument("--boost", type=float, default=4.0)
    ap.add_argument("--thresh", type=float, default=0.65)
    ap.add_argument("--skip-jazz", action="store_true")
    ap.add_argument("--skip-pop", action="store_true")
    args = ap.parse_args()

    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2 = renderer._find_soundfont("MuseScore_General.sf2")
    ex = PitchExtractor(cache_dir=None)
    v4 = P._get_beat_seq()
    fam = P._get_family_clf()
    print(f"root model: {type(v4).__name__}   boost={args.boost}  thresh={args.thresh}\n")

    results = {}
    if not args.skip_jazz:
        results["jazz1460"] = eval_jazz(args, renderer, sf2, ex, v4, fam)
    if not args.skip_pop:
        results["POP909"] = eval_pop(args, renderer, sf2, ex, v4, fam)

    print("\n=== diatonic prior — end-to-end MIREX (tempo grid, gmerge segs) ===")
    print(f"{'corpus':<10} {'variant':<10} {'root':>7} {'majmin':>7} {'7ths':>7} {'n':>4}")
    print("-" * 50)
    for corpus, acc in results.items():
        for name, tag in (("base", "baseline"), ("prior", "+diatonic")):
            if acc[name]:
                r, m, s, n = _summ(acc[name])
                print(f"{corpus:<10} {tag:<10} {r:>6.1%} {m:>7.1%} {s:>7.1%} {n:>4}")
        # delta line
        if acc["base"] and acc["prior"]:
            rb, mb, sb, _ = _summ(acc["base"])
            rp, mp, sp, _ = _summ(acc["prior"])
            print(f"{'':<10} {'Δ':<10} {rp-rb:>+6.1%} {mp-mb:>+7.1%} {sp-sb:>+7.1%}")
        print("-" * 50)


if __name__ == "__main__":
    main()
