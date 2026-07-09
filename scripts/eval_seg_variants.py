"""Push on segmentation: test gmerge variants end-to-end on held-out irealb.

Baseline gmerge cuts wherever the per-beat root argmax changes.  Failure modes
(premise-checked): (a) spurious 1-beat flips from noisy per-beat root → over-seg;
(b) quality-only boundaries (root unchanged, ~6% of GT) it structurally misses.

Variants (tempo grid = detected beats, v4 root, baseline family labeling):
  gmerge          cut on root argmax change
  gmerge_vit      cut on VITERBI-smoothed root (self-transition boost kills flips)
  gmerge_qual     cut on root OR v3-quality argmax change
  gmerge_vit_qual both
Reported: MIREX root / majmin / 7ths + seg/GT, vs oracle ceiling.
"""
from __future__ import annotations

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
from harmonia.models import chord_pipeline_v1 as P

NOTE = P.NOTE
Q5H = ["maj", "min", "7", "maj7", "dim"]
DB = REPO / "data" / "accomp_db" / "db.jsonl"


def tempo_grid(y, sr):
    tempo, bf = librosa.beat.beat_track(y=y, sr=sr)
    btl = librosa.frames_to_time(bf, sr=sr)
    period = 60.0 / float(np.atleast_1d(tempo)[0])
    ang = 2 * np.pi * (btl % period) / period
    phase = (np.angle(np.mean(np.exp(1j * ang))) % (2 * np.pi)) * period / (2 * np.pi)
    bt = np.arange(phase, len(y) / sr, period)
    return np.unique(np.concatenate([[0.0], bt, [len(y) / sr]]))


def _vit(proba, boost):
    """Argmax root path with a self-transition log-boost (kills 1-beat flips)."""
    logE = np.log(proba + 1e-9)
    n, K = proba.shape
    dp = logE[0].copy(); bp = np.zeros((n, K), int)
    for t in range(1, n):
        trans = np.zeros((K, K)); np.fill_diagonal(trans, boost)
        sc = dp[:, None] + trans          # (K prev, K cur)
        bp[t] = sc.argmax(0)
        dp = sc.max(0) + logE[t]
    path = [int(dp.argmax())]
    for t in range(n - 1, 0, -1):
        path.append(bp[t, path[-1]])
    return np.array(path[::-1])


def segs_from_seq(*seqs):
    """Cut wherever ANY of the given per-beat sequences changes."""
    n = len(seqs[0])
    cuts = [0] + [b for b in range(1, n) if any(s[b] != s[b-1] for s in seqs)] + [n]
    return [(cuts[i], cuts[i+1]) for i in range(len(cuts)-1)]


def oracle_segs(change_times, bt):
    n = len(bt) - 1
    cuts = sorted(set([0] + [int(np.argmin(np.abs(bt - t))) for t in change_times] + [n]))
    return [(cuts[i], cuts[i+1]) for i in range(len(cuts)-1) if cuts[i+1] > cuts[i]]


def score(segs, bt, onset_b, note_b, beat_proba, fam, ref_int, ref_lab):
    labeled = []
    for s, e in segs:
        root = int(beat_proba[s:e].sum(0).argmax())
        so = onset_b[s:e].sum(0); sn = note_b[s:e].sum(0)
        _, sev_h, _ = fam.predict(root, so, sn, P._reg_raw(so, 0, 52), P._reg_raw(so, 60, 200), 0.0)
        lab = f"{NOTE[root]}:{sev_h}"
        if labeled and labeled[-1][2] == lab:
            labeled[-1][1] = e
        else:
            labeled.append([s, e, lab])
    keep = [([bt[s], bt[min(e, len(bt)-1)]], lab) for s, e, lab in labeled if bt[min(e, len(bt)-1)] > bt[s]]
    if not keep:
        return None
    ei, el = zip(*keep)
    try:
        sc = mir_eval.chord.evaluate(np.array(ref_int), ref_lab, np.array(ei), list(el))
    except ValueError:
        return None
    return sc["root"], sc["majmin"], sc["sevenths"], len(keep) / len(ref_int)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25)
    ap.add_argument("--start", type=int, default=70)
    ap.add_argument("--boost", type=float, default=2.0)
    args = ap.parse_args()

    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2 = renderer._find_soundfont("MuseScore_General.sf2")
    ex = P.PitchExtractor(cache_dir=None)
    v4 = P._get_beat_seq(); v3 = P._get_beat_seq_v3(); fam = P._get_family_clf()

    recs = [json.loads(l) for l in open(DB)]
    jz = [r for r in recs if r.get("corpus") == "jazz1460"
          and r["beats_per_bar"] == 4 and (REPO / r["midi_path"]).exists()][args.start:args.start + args.n]
    variants = ("oracle", "gmerge", "gmerge_vit", "gmerge_qual", "gmerge_vit_qual")
    R = {v: [] for v in variants}; M = {v: [] for v in variants}
    S = {v: [] for v in variants}; SG = {v: [] for v in variants}

    for i, rec in enumerate(jz):
        print(f"  [{i+1}/{len(jz)}] {rec['song_id']}", end="\r", flush=True)
        spans = [(t0, t1, r % 12, q) for t0, t1, r, q in song_chord_spans(rec)
                 if t1 > t0 and q in BUCKET_FAMILY]
        if not spans:
            continue
        ref_int = [[t0, t1] for t0, t1, _, _ in spans]
        ref_lab = [f"{NOTE[r]}:{Q5H[quality5(q)] if quality5(q) is not None else 'maj'}" for _, _, r, q in spans]
        ct = [t0 for t0, _, _, _ in spans[1:]]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
            tmp = Path(wf.name)
        try:
            renderer.render(REPO / rec["midi_path"], tmp, RenderConfig(soundfont_path=sf2))
            y, sr = sf.read(tmp); y = (y.mean(1) if y.ndim > 1 else y).astype("float32")
            acts = ex.extract(tmp, use_cache=False)
        finally:
            tmp.unlink(missing_ok=True)
        bt = tempo_grid(y, sr)
        onset_b = P._pool_beats(acts.frame_times, acts.onset_probs, bt)
        note_b = P._pool_beats(acts.frame_times, acts.note_probs, bt)
        bp = v4.predict_proba(onset_b, note_b)
        root_seq = bp.argmax(1)
        vit_seq = _vit(bp, args.boost)
        qual_seq = v3.qual_proba(onset_b, note_b).argmax(1) if v3 else np.zeros(len(bp), int)

        seg_sets = {
            "oracle": oracle_segs(ct, bt),
            "gmerge": segs_from_seq(root_seq),
            "gmerge_vit": segs_from_seq(vit_seq),
            "gmerge_qual": segs_from_seq(root_seq, qual_seq),
            "gmerge_vit_qual": segs_from_seq(vit_seq, qual_seq),
        }
        for v, segs in seg_sets.items():
            res = score(segs, bt, onset_b, note_b, bp, fam, ref_int, ref_lab)
            if res:
                R[v].append(res[0]); M[v].append(res[1]); S[v].append(res[2]); SG[v].append(res[3])

    print(f"\n\n=== segmentation variants, held-out irealb ({len(R['gmerge'])} songs, boost={args.boost}) ===")
    print(f"{'variant':<17} {'root':>7} {'majmin':>7} {'7ths':>7} {'seg/GT':>7}")
    print("-" * 48)
    for v in variants:
        print(f"{v:<17} {np.mean(R[v]):>7.1%} {np.mean(M[v]):>7.1%} {np.mean(S[v]):>7.1%} {np.mean(SG[v]):>7.2f}")


if __name__ == "__main__":
    main()
