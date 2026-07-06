"""Chord-change engine — COARSE pass (step 2, revised). Corpus analysis killed the
per-section-period idea (changes are irregular, land on every beat; 92% of sections
have no clean 1/2/4 period). But 2-beat is the best single grid (change-vs-hold AUC
0.962) so the coarse grid is a FIXED 2-beat merge + same-or-different, with a forced
boundary at each (GT) section change. The zoom step (next) must recover the ~39% of
changes that fall interior to a 2-beat block.

Scaffold: GT section_per_bar + exact beat grid (structure detection is separable).
Measured: (a) change-detection F vs GT change beats, (b) MIREX root/majmin — against
the naive per-beat baseline and the merge-at-2 oracle ceiling.

Usage: .venv/bin/python scripts/chord_change_engine.py --n-songs 15 [--degrade] [--zoom]
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
from build_audio_chord_features import BUCKET_BASE7, BUCKET_FAMILY  # noqa: E402
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig  # noqa: E402
from harmonia.models.stage1_pitch import PitchExtractor  # noqa: E402
from harmonic_rhythm_probe import gt_chord_per_beat, pool_beats  # noqa: E402
from root_model_experiment import TEMPLATES  # noqa: E402

DB = REPO / "data" / "accomp_db" / "db.jsonl"
CLEAN_FEAT = REPO / "data" / "cache" / "audio_chord_features.npz"
FAMILIES = ["major", "minor", "diminished", "augmented", "suspended"]
NOTE = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
FAM_HARTE = {"major": "maj", "minor": "min", "diminished": "dim",
             "augmented": "aug", "suspended": "sus4"}
# base-seventh bucket → mir_eval Harte quality
B7_HARTE = {"majT": "maj", "minT": "min", "dimT": "dim", "augT": "aug", "susT": "sus4",
            "maj7": "maj7", "min7": "min7", "dom7": "7", "m7b5": "hdim7", "dim7": "dim7",
            "minmaj7": "minmaj7", "7sus4": "sus4", "aug7": "aug", "augmaj7": "aug"}


def reg(v88, lo, hi):
    c = np.zeros(12)
    for k in range(88):
        if lo <= 21 + k < hi:
            c[(21 + k) % 12] += v88[k]
    return c


def reg_n(v88, lo=0, hi=200):
    c = reg(v88, lo, hi); n = np.linalg.norm(c)
    return c / n if n > 1e-9 else c


def norm_blocks(x):
    """L2-normalize each consecutive 12-dim chroma block so the family features are
    DURATION-INVARIANT — otherwise raw summed chroma scales with segment length and
    coarse segments land off the oracle-trained model's input distribution."""
    x = np.asarray(x, float)
    y = x.reshape(*x.shape[:-1], x.shape[-1] // 12, 12)
    n = np.linalg.norm(y, axis=-1, keepdims=True)
    return (y / (n + 1e-9)).reshape(x.shape)


class RootModel:
    """Trained 12-way root classifier (root_model_experiment.py --save); beats the
    bass-argmax root (~68%) that walking bass defeats, at ~93% held-out."""
    def __init__(self, path):
        d = np.load(path)
        self.mean, self.scale = d["mean"], d["scale"]
        self.coef, self.intercept, self.classes = d["coef"], d["intercept"], d["classes"]

    def predict(self, seg_on, seg_nt):
        oc = reg_n(seg_on)
        f = np.concatenate([oc, reg_n(seg_nt), reg_n(seg_on, 0, 52), reg_n(seg_on, 60, 200)])
        if len(self.mean) == 60:                        # model trained with template features
            tmpl = np.array([max(oc @ t for r2, t in TEMPLATES if r2 == r) for r in range(12)])
            f = np.concatenate([f, tmpl])
        z = (f - self.mean) / self.scale
        return int(self.classes[np.argmax(z @ self.coef.T + self.intercept)])


def feat24(on_beat):
    ch = reg(on_beat, 0, 200); ba = reg(on_beat, 0, 52)
    ch /= (np.linalg.norm(ch) + 1e-9); ba /= (np.linalg.norm(ba) + 1e-9)
    return np.concatenate([ch, ba])


def cos_d(a, b):
    return 1 - float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def coarse_segments(onset_b, note_b, sec, theta, cell=2):
    """Fixed cell-beat merge + same-or-different; forced cut at section changes.
    Returns list of (start_beat, end_beat)."""
    nb = len(onset_b)
    blocks = [(s, min(s + cell, nb)) for s in range(0, nb, cell)]
    bfeat = [feat24(onset_b[s:e].sum(0)) for s, e in blocks]
    segs = [list(blocks[0])]
    for i in range(1, len(blocks)):
        s, e = blocks[i]
        sec_change = sec[s] != sec[s - 1]
        if sec_change or cos_d(bfeat[i], bfeat[i - 1]) > theta:
            segs.append([s, e])
        else:
            segs[-1][1] = e
    return segs


def divisive_segments(onset_b, sec, split_tol, min_len=1):
    """Top-down: within each GT section, recursively split at the best boundary,
    scoring each candidate by the distance between the two POOLED halves (so every
    candidate has full-segment SNR — the clean 0.962-quality merged signal — not the
    noisy per-beat novelty). Splits down to beat resolution while the split gain
    exceeds split_tol."""
    nb = len(onset_b)
    # section spans
    spans = []; b0 = 0
    for b in range(1, nb + 1):
        if b == nb or sec[b] != sec[b0]:
            spans.append((b0, b)); b0 = b

    def split(s, e):
        if e - s < 2 * min_len:
            return [[s, e]]
        best_b, best_d = None, -1.0
        for b in range(s + min_len, e - min_len + 1):
            fa = feat24(onset_b[s:b].sum(0)); fb = feat24(onset_b[b:e].sum(0))
            d = cos_d(fa, fb)
            if d > best_d:
                best_b, best_d = b, d
        if best_b is not None and best_d > split_tol:
            return split(s, best_b) + split(best_b, e)
        return [[s, e]]

    out = []
    for s, e in spans:
        out += split(s, e)
    return out


def snap_boundaries(onset_b, segs, window=1):
    """Nudge each coarse boundary within ±window beats to the position that maximizes
    the distance between the two full POOLED segments it separates. Both sides keep
    full-segment SNR (the clean signal), so this fixes the ±1 grid-quantization
    without the per-beat noise that sank the naive zoom."""
    if len(segs) < 2:
        return segs
    segs = [list(s) for s in segs]
    for i in range(1, len(segs)):
        a, b, c = segs[i - 1][0], segs[i][0], segs[i][1]
        lo, hi = max(a + 1, b - window), min(c - 1, b + window)
        best_b, best_d = b, -1.0
        for nb_ in range(lo, hi + 1):
            fa = feat24(onset_b[a:nb_].sum(0)); fb = feat24(onset_b[nb_:c].sum(0))
            d = cos_d(fa, fb)
            if d > best_d:
                best_b, best_d = nb_, d
        segs[i - 1][1] = best_b; segs[i][0] = best_b
    return segs


def zoom_refine(onset_b, note_b, segs, snap_tol=0.10, split_tol=0.30):
    """Beat-resolution pass inside each coarse segment. Two moves:
      (1) SNAP: if the strongest interior beat-to-beat novelty sits near the segment
          start, move the boundary to that beat (fixes the ±1 grid-quantization).
      (2) SPLIT: if a strong interior novelty peak sits mid-segment, insert a
          boundary there (recovers a change the 2-beat block blurred).
    Beat feature = feat24 (chroma+bass); no per-track render needed for this pass."""
    bf = [feat24(onset_b[b]) for b in range(len(onset_b))]
    out = []
    for s, e in segs:
        # interior novelties: distance across each interior beat boundary
        cand = [(b, cos_d(bf[b], bf[b - 1])) for b in range(s + 1, e)]
        # (1) snap boundary s to the beat just before the largest early novelty
        if cand:
            b_snap, v_snap = max(cand[:2], key=lambda t: t[1]) if len(cand) >= 1 else (s, 0)
            if v_snap > snap_tol and out and b_snap - s <= 1:
                out[-1][1] = b_snap; s = b_snap
        # (2) split on a strong mid-segment peak (not adjacent to the edges)
        pieces = [s]
        for b, v in cand:
            if b - pieces[-1] >= 2 and e - b >= 2 and v > split_tol:
                pieces.append(b)
        pieces.append(e)
        for a, b in zip(pieces, pieces[1:]):
            out.append([a, b])
    return out


def label_segment(onset_b, note_b, s, e, sc, clf, root_model=None,
                  b7=None, base7_labels=None, gate=0.0):
    """Returns a Harte 'root:quality' label. With a base7 model (b7), descend to the
    SEVENTH when the model is confident (max-prob >= gate), else fall back to the
    triad/family — the project's 'report deeper only when confident' rule."""
    seg_on = onset_b[s:e].sum(0); seg_nt = note_b[s:e].sum(0)
    if root_model is not None:
        root = root_model.predict(seg_on, seg_nt)
    else:
        bass = reg(seg_on, 0, 52)
        root = int(bass.argmax()) if bass.sum() > 1e-6 else int(reg(seg_on, 0, 200).argmax())
    rr = lambda c: np.roll(c, -root)
    f = norm_blocks(np.hstack([rr(reg(seg_on, 0, 200)), rr(reg(seg_nt, 0, 200)),
                               rr(reg(seg_on, 0, 52)), rr(reg(seg_on, 60, 200))]))
    fam = FAMILIES[int(clf.predict(sc.transform(f[None]))[0])]
    qual = FAM_HARTE[fam]
    if b7 is not None:
        P = b7.predict_proba(sc.transform(f[None]))[0]
        if P.max() >= gate:
            qual = B7_HARTE[base7_labels[int(b7.classes_[P.argmax()])]]
    return root, f"{NOTE[root]}:{qual}"


def change_f(pred_bounds, gt_changes, tol=1):
    est = sorted(set(b for b in pred_bounds if b > 0))
    hits = sum(any(abs(e - g) <= tol for e in est) for g in gt_changes)
    p = hits / (len(est) + 1e-9); r = hits / (len(gt_changes) + 1e-9)
    return 2 * p * r / (p + r + 1e-9), p, r


def change_f_time(est_times, gt_times, tol):
    """Change-detection F in the time domain (for detected beats, where beat indices
    don't align to GT). tol in seconds (~1 inter-beat interval)."""
    est = sorted(t for t in est_times if t > 1e-6)
    hits = sum(any(abs(e - g) <= tol for e in est) for g in gt_times)
    p = hits / (len(est) + 1e-9); r = hits / (len(gt_times) + 1e-9)
    return 2 * p * r / (p + r + 1e-9), p, r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-songs", type=int, default=15)
    ap.add_argument("--degrade", action="store_true")
    ap.add_argument("--theta", type=float, default=None, help="fixed threshold; default sweeps")
    ap.add_argument("--zoom", action="store_true", help="apply the beat-resolution zoom pass")
    ap.add_argument("--divisive", action="store_true",
                    help="top-down pooled-halves splitter instead of the 2-beat coarse merge")
    ap.add_argument("--oracle-bounds", action="store_true",
                    help="use GT change beats as boundaries (isolates labeling from segmentation)")
    ap.add_argument("--root-model", action="store_true",
                    help="use the trained root classifier instead of bass-argmax")
    ap.add_argument("--no-structure", action="store_true",
                    help="drop the GT-structure scaffold (no forced section boundaries)")
    ap.add_argument("--librosa-beats", action="store_true",
                    help="track beats from the audio (fully standalone) instead of the GT grid")
    ap.add_argument("--tempo-grid", action="store_true",
                    help="standalone but impose a uniform grid at the detected tempo (de-jitter)")
    ap.add_argument("--parity", type=int, default=None,
                    help="eval on songs of this parity; train family model on the other "
                         "(disjoint held-out eval; pair with a root model trained --parity <other>)")
    ap.add_argument("--seventh", action="store_true",
                    help="report the SEVENTH level (base7) and score mir_eval sevenths")
    ap.add_argument("--seventh-gate", type=float, default=0.0,
                    help="descend to the seventh only when its max-prob >= this (else triad)")
    args = ap.parse_args()

    root_model = None
    if args.root_model:
        rm_path = REPO / "harmonia" / "models" / "root_model.npz"
        root_model = RootModel(rm_path)

    d = np.load(CLEAN_FEAT, allow_pickle=True)
    Xc = norm_blocks(np.hstack([d["onset"], d["note"], d["bass"], d["treble"]]))
    famy = d["family"].astype(int)
    if args.parity is not None:                     # train family on the OTHER parity
        keep = np.array([int(s.split("_")[1]) % 2 != args.parity for s in d["song"]])
        Xc, famy = Xc[keep], famy[keep]
    sc = StandardScaler().fit(Xc)
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(Xc), famy)

    b7_model = base7_labels = None
    if args.seventh:                                    # train the seventh model (base7)
        b7y = d["base7"].astype(int)
        Xb = norm_blocks(np.hstack([d["onset"], d["note"], d["bass"], d["treble"]]))
        if args.parity is not None:
            keep = np.array([int(s.split("_")[1]) % 2 != args.parity for s in d["song"]])
            Xb, b7y = Xb[keep], b7y[keep]
        b7_model = LogisticRegression(max_iter=2000).fit(sc.transform(Xb), b7y)
        base7_labels = [str(x) for x in d["base7_labels"]]

    recs = [json.loads(l) for l in open(DB)]
    songs = [r for r in recs if r["corpus"] == "jazz1460" and r["beats_per_bar"] == 4
             and (REPO / r["midi_path"]).exists() and len(set(r["section_per_bar"])) > 1]
    if args.parity is not None:
        songs = [r for r in songs if int(r["song_id"].split("_")[1]) % 2 == args.parity]
    songs = songs[:: max(len(songs) // args.n_songs, 1)][: args.n_songs]

    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    sf2 = renderer._find_soundfont("MuseScore_General.sf2")
    ex = PitchExtractor(cache_dir=None)
    rng = np.random.default_rng(3)

    cached = []                                          # per-song precomputed arrays
    for rec in songs:
        spb = 60.0 / rec["tempo"]; bpb = rec["beats_per_bar"]; nb = rec["n_bars"]
        n_beats = nb * bpb
        sec = [0] * n_beats if args.no_structure else \
            [rec["section_per_bar"][b // bpb] for b in range(n_beats)]
        # AUTHORITATIVE GT chord spans — the single source for segmentation, per-beat
        # GT, change-times AND the MIREX reference (fixes the harness GT-source mismatch,
        # known_issues #11). (t0, t1, root_pc, family).
        spans = []
        for t0, t1, r, _q in song_chord_spans(rec):
            mma = None
            for ev in rec["chord_timeline"]:
                if int(round(((ev["bar"] - 1) * bpb + ev["beat"]))) == int(round(t0 / spb)):
                    mma = ev["mma"]; break
            p = parse_chord(mma) if mma else None
            if p is None or p[1] not in BUCKET_FAMILY or t1 <= t0:
                continue
            b7q = B7_HARTE.get(BUCKET_BASE7.get(p[1], ""), FAM_HARTE[BUCKET_FAMILY[p[1]]])
            spans.append((t0, t1, r % 12, BUCKET_FAMILY[p[1]], b7q))

        def chord_at(t):
            for sp in spans:
                if sp[0] <= t < sp[1]:
                    return (sp[2], sp[3])
            return None
        gtc = [chord_at((b + 0.5) * spb) for b in range(n_beats)]   # mid-beat sample
        gt_changes = [b for b in range(1, n_beats) if gtc[b] is not None and gtc[b] != gtc[b - 1]]
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
            tmp = Path(wf.name)
        try:
            renderer.render(REPO / rec["midi_path"], tmp, RenderConfig(soundfont_path=sf2))
            y, sr = sf.read(tmp); y = (y.mean(1) if y.ndim > 1 else y).astype("float32")
            if args.degrade:
                y = time_varying_degrade(y, sr, rng); sf.write(tmp, y, sr)
            acts = ex.extract(tmp, use_cache=False)
        finally:
            tmp.unlink(missing_ok=True)
        if args.librosa_beats or args.tempo_grid:
            tempo, bf = librosa.beat.beat_track(y=y, sr=sr)
            btl = librosa.frames_to_time(bf, sr=sr)
            if args.tempo_grid:
                # detected tempo is accurate but per-beat times jitter; MMA is metronomic,
                # so impose a UNIFORM grid at the detected tempo + circular-mean phase.
                period = 60.0 / float(np.atleast_1d(tempo)[0])
                ang = 2 * np.pi * (btl % period) / period
                phase = (np.angle(np.mean(np.exp(1j * ang))) % (2 * np.pi)) * period / (2 * np.pi)
                bt = np.arange(phase, len(y) / sr, period)
            else:
                bt = btl
            bt = np.unique(np.concatenate([[0.0], bt, [len(y) / sr]]))
            sec = [0] * (len(bt) - 1)               # per-bar structure can't map to detected beats
        else:
            bt = np.arange(n_beats + 1) * spb
        gt_change_times = [sp[0] for sp in spans[1:]]             # exact span starts
        onset_b = pool_beats(acts.frame_times, acts.onset_probs, bt)
        note_b = pool_beats(acts.frame_times, acts.note_probs, bt)
        # MIREX reference from the SAME spans (aligned with the oracle segmentation);
        # seventh-level quality when --seventh, else triad/family.
        ref_int = [[sp[0], sp[1]] for sp in spans]
        ref_lab = [f"{NOTE[sp[2]]}:{sp[4] if args.seventh else FAM_HARTE[sp[3]]}" for sp in spans]
        cached.append((rec, sec, gt_changes, gt_change_times, onset_b, note_b, bt, ref_int, ref_lab))

    thetas = [args.theta] if args.theta is not None else [0.06, 0.08, 0.10, 0.12, 0.15]
    cond = "DEGRADED" if args.degrade else "clean"
    print(f"\n=== coarse chord-change engine (fixed 2-beat merge), {len(cached)} {cond} songs ===")
    print(f"{'theta':>6} {'chgF':>6} {'chgF0':>6} {'chgP':>6} {'chgR':>6} {'root':>6} {'majmin':>7} {'seg/GT':>7}"
          + ("  sevenths" if args.seventh else ""))
    for theta in thetas:
        Fs, F0s, Ps, Rs, roots, mms, ratios, sevs = [], [], [], [], [], [], [], []
        for (rec, sec, gt_changes, gt_change_times, onset_b, note_b, bt, ref_int, ref_lab) in cached:
            if args.oracle_bounds:
                bnds = sorted(set([0] + gt_changes + [len(onset_b)]))
                segs = [[s, e] for s, e in zip(bnds, bnds[1:])]
            elif args.divisive:
                segs = divisive_segments(onset_b, sec, theta)
            else:
                segs = coarse_segments(onset_b, note_b, sec, theta)
            if args.zoom:
                segs = snap_boundaries(onset_b, segs)
            # label, then COALESCE adjacent same-label segments (a repeated chord is one
            # chord): low theta favours recall, and merging identical neighbours undoes
            # the resulting over-segmentation for free (labels-over-time unchanged).
            labeled = []
            for s, e in segs:
                _, lab = label_segment(onset_b, note_b, s, e, sc, clf, root_model,
                                       b7_model, base7_labels, args.seventh_gate)
                if labeled and labeled[-1][2] == lab:
                    labeled[-1][1] = e
                else:
                    labeled.append([s, e, lab])
            est_times = [bt[s] for s, e, lab in labeled]
            tolb = float(np.median(np.diff(bt)))         # ~1 inter-beat interval
            f, p, r = change_f_time(est_times, gt_change_times, tolb)
            f0, _, _ = change_f_time(est_times, gt_change_times, tolb * 0.5)
            Fs.append(f); F0s.append(f0); Ps.append(p); Rs.append(r)
            est_int = [[bt[s], bt[min(e, len(bt) - 1)]] for s, e, lab in labeled]
            est_lab = [lab for s, e, lab in labeled]
            ei, el = zip(*[(iv, lb) for iv, lb in zip(est_int, est_lab) if iv[1] > iv[0]]) \
                if any(iv[1] > iv[0] for iv in est_int) else ((), ())
            if ei and ref_int:
                try:
                    sco = mir_eval.chord.evaluate(np.array(ref_int), ref_lab,
                                                  np.array(list(ei)), list(el))
                except ValueError:
                    continue                       # mir_eval interval edge case; skip song
                roots.append(sco["root"]); mms.append(sco["majmin"])
                sevs.append(sco["sevenths"])
                ratios.append(len(est_int) / len(ref_int))
        sev_col = f" {np.mean(sevs):7.1%}" if args.seventh else ""
        print(f"{theta:6.2f} {np.mean(Fs):6.2f} {np.mean(F0s):6.2f} {np.mean(Ps):6.2f} {np.mean(Rs):6.2f} "
              f"{np.mean(roots):6.1%} {np.mean(mms):7.1%} {np.mean(ratios):7.2f}{sev_col}")
    print("\nchgF/chgF0 = change-detection F vs GT changes at ±1 / exact beat. seg/GT: 1.0=right count.")
    if args.seventh:
        print("sevenths = MIREX seventh-level accuracy (maj/min/maj7/min7/7 vocabulary).")


if __name__ == "__main__":
    main()
