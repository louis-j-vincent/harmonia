"""End-to-end MIREX on irealb-rendered audio — the first honest end-to-end number.

Renders held-out jazz1460 songs (index ≥70, UNSEEN by beat_seq_model_v4), runs the
real Gen-2 components (v4 per-beat root + family classifier), and scores mir_eval
root / majmin against the iReal chart.  Varies two axes:

  beat grid : exact (k·60/tempo, metronomic ceiling) vs tempo-grid (librosa detected
              tempo + circular-mean phase — the true end-to-end, beat-tracking incl.)
  segmentation : oracle (GT change beats) / gridmerge (production: 2-4 beat grid
              merged by root) / gmerge (cut wherever per-beat root argmax changes)

Root is the headline (what v4 targets); majmin uses the baseline family classifier
(ctx model would lift it further — not wired here to keep the harness clean).
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
from train_beat_seq_model_v3 import quality5
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig
from harmonia.models.stage1_pitch import PitchExtractor
from harmonia.models import chord_pipeline_v1 as P

DB = REPO / "data" / "accomp_db" / "db.jsonl"
NOTE = P.NOTE
Q5_HARTE = ["maj", "min", "7", "maj7", "dim"]  # major, minor, dom7, maj7, dim


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
    cuts = [0] + [b for b in range(1, len(pred)) if pred[b] != pred[b-1]] + [len(pred)]
    return [(cuts[i], cuts[i+1]) for i in range(len(cuts)-1)]


def gridmerge_segs(beat_proba, n_beats):
    grid = P._fit_harmonic_grid(beat_proba)
    return P._merge_grid_by_root(P._make_grid_segs(n_beats, grid), beat_proba)


def oracle_segs(change_times, bt):
    n = len(bt) - 1
    cuts = sorted(set([0] + [int(np.argmin(np.abs(bt - t))) for t in change_times] + [n]))
    return [(cuts[i], cuts[i+1]) for i in range(len(cuts)-1) if cuts[i+1] > cuts[i]]


def label_and_score(segs, bt, onset_b, note_b, beat_proba, fam_clf, ref_int, ref_lab,
                    prog_weight=None):
    # first pass: per-segment (root, sev_h, conf, q5_logprobs)
    seg_root, seg_sev, seg_conf, seg_se, seg_q5lp = [], [], [], [], []
    for s, e in segs:
        root = int(beat_proba[s:e].sum(0).argmax())
        seg_on = onset_b[s:e].sum(0); seg_nt = note_b[s:e].sum(0)
        seg_bs = P._reg_raw(seg_on, 0, 52); seg_tr = P._reg_raw(seg_on, 60, 200)
        _, sev_h, conf, q5lp = fam_clf.predict(root, seg_on, seg_nt, seg_bs, seg_tr, 0.0,
                                               return_q5proba=True)
        seg_root.append(root); seg_sev.append(sev_h)
        seg_conf.append(conf); seg_se.append((s, e)); seg_q5lp.append(q5lp)
    # optional second pass: progression-encoder quality rerank (real per-q5
    # log-probs from the family+seventh classifier heads, issue #21 fix)
    if prog_weight is not None and seg_root:
        seg_sev = P.rerank_progression_qualities(
            seg_root, seg_sev, seg_conf, weight=prog_weight, aco_logprobs=seg_q5lp
        )
    # coalesce adjacent same-label segments
    labeled = []
    for (s, e), root, sev_h in zip(seg_se, seg_root, seg_sev):
        lab = f"{NOTE[root]}:{sev_h}"
        if labeled and labeled[-1][2] == lab:
            labeled[-1][1] = e
        else:
            labeled.append([s, e, lab])
    est_int = [[bt[s], bt[min(e, len(bt)-1)]] for s, e, _ in labeled]
    est_lab = [lab for _, _, lab in labeled]
    keep = [(iv, lb) for iv, lb in zip(est_int, est_lab) if iv[1] > iv[0]]
    if not keep:
        return None
    ei, el = zip(*keep)
    try:
        sco = mir_eval.chord.evaluate(np.array(ref_int), ref_lab, np.array(ei), list(el))
    except ValueError:
        return None
    return sco["root"], sco["majmin"], sco["sevenths"], len(keep) / len(ref_int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--start", type=int, default=70, help="held-out start index (v4 trained on <70)")
    ap.add_argument("--progression-weight", type=float, default=None,
                    help="if set, apply ProgressionEncoder quality rerank at this weight")
    args = ap.parse_args()

    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2 = renderer._find_soundfont("MuseScore_General.sf2")
    ex = PitchExtractor(cache_dir=None)
    v4 = P._get_beat_seq()
    fam = P._get_family_clf()
    print(f"root model: {type(v4).__name__}")

    recs = [json.loads(l) for l in open(DB)]
    jz = [r for r in recs if r.get("corpus") == "jazz1460"
          and r["beats_per_bar"] == 4 and (REPO / r["midi_path"]).exists()]
    held = jz[args.start:args.start + args.n]
    print(f"held-out jazz songs: {len(held)} (index {args.start}..{args.start+args.n})")

    grids = ("exact", "tempo")
    segms = ("oracle", "gridmerge", "gmerge")
    R = {(g, s): [] for g in grids for s in segms}
    M = {(g, s): [] for g in grids for s in segms}
    SV = {(g, s): [] for g in grids for s in segms}
    SG = {(g, s): [] for g in grids for s in segms}

    for i, rec in enumerate(held):
        print(f"  [{i+1}/{len(held)}] {rec['song_id']}", end="\r", flush=True)
        spb = 60.0 / rec["tempo"]; n_beats = rec["n_bars"] * rec["beats_per_bar"]
        spans = [(t0, t1, r % 12, q) for t0, t1, r, q in song_chord_spans(rec)
                 if t1 > t0 and q in BUCKET_FAMILY]
        if not spans:
            continue
        ref_int = [[t0, t1] for t0, t1, _, _ in spans]
        ref_lab = []
        for _, _, r, q in spans:
            q5 = quality5(q)
            ref_lab.append(f"{NOTE[r]}:{Q5_HARTE[q5] if q5 is not None else 'maj'}")
        change_times = [t0 for t0, _, _, _ in spans[1:]]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
            tmp = Path(wf.name)
        try:
            renderer.render(REPO / rec["midi_path"], tmp, RenderConfig(soundfont_path=sf2))
            y, sr = sf.read(tmp); y = (y.mean(1) if y.ndim > 1 else y).astype("float32")
            acts = ex.extract(tmp, use_cache=False)
        finally:
            tmp.unlink(missing_ok=True)

        for g in grids:
            bt = np.arange(n_beats + 1) * spb if g == "exact" else tempo_grid(y, sr)
            onset_b = P._pool_beats(acts.frame_times, acts.onset_probs, bt)
            note_b = P._pool_beats(acts.frame_times, acts.note_probs, bt)
            beat_proba = v4.predict_proba(onset_b, note_b)
            nb = len(onset_b)
            seg_sets = {
                "oracle": oracle_segs(change_times, bt),
                "gridmerge": gridmerge_segs(beat_proba, nb),
                "gmerge": gmerge_segs(beat_proba),
            }
            for s, segs in seg_sets.items():
                res = label_and_score(segs, bt, onset_b, note_b, beat_proba, fam,
                                      ref_int, ref_lab, prog_weight=args.progression_weight)
                if res:
                    R[(g, s)].append(res[0]); M[(g, s)].append(res[1])
                    SV[(g, s)].append(res[2]); SG[(g, s)].append(res[3])

    tag = "baseline" if args.progression_weight is None else f"+encoder w={args.progression_weight}"
    print(f"\n\n=== end-to-end MIREX on irealb (held-out) — {tag} ===")
    print(f"{'grid':<7} {'segmentation':<11} {'root':>7} {'majmin':>7} {'7ths':>7} {'seg/GT':>7} {'n':>4}")
    print("-" * 56)
    for g in grids:
        for s in segms:
            k = (g, s)
            if R[k]:
                print(f"{g:<7} {s:<11} {np.mean(R[k]):>7.1%} {np.mean(M[k]):>7.1%} "
                      f"{np.mean(SV[k]):>7.1%} {np.mean(SG[k]):>7.2f} {len(R[k]):>4}")


if __name__ == "__main__":
    main()
