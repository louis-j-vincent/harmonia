"""exp_leakage_robust_identity.py — does a LEAKAGE-ROBUST feature recover chord
IDENTITY where per-segment mean-pool fails? (2026-07-24 research session)

QUESTION (session brief): the shipped nnls24 root loss is dominated by chord-
IDENTITY errors on correctly-TIMED spans (P4/P5 fifth confusion + 7th quality).
Five post-hoc adjudicators all failed for ONE reason: `nnls_features.pool_beats`
MEAN-POOLS the segment, so a neighbour chord's tones bleed across the boundary
and confound any note-level discriminator. The documented failure is on the
"leading micro-segments" — i.e. SHORT windows near a boundary, where a prev-chord
sustain tail is a large FRACTION of the pooled window. So the fix must change the
SIGNAL to be leakage-robust, not re-arbitrate the same leaky mean.

Three feature variants, evaluated on next-chord identity as leakage rises:
  V0  mean-pool absolute, window ANCHORED at the boundary + pre-boundary bleed
      (beat-snap register) — the leaky regime.
  V1  clip-pool absolute, window strictly [t0, t0+W)  (reuses yt_chord_corpus
      `_clip_pool` framing — removes the PRE-boundary bleed only).
  V2  root-normalised directional frame-delta  (L1-norm next-window minus
      prev-window, rolled by -prev_root; sustained/common tones cancel ->
      leakage-robust BY CONSTRUCTION). Window-INDEPENDENT.

Two swept axes:
  * contamination strength s in [0..1]  (the experiment)
  * POOL WINDOW W for V0/V1  — from micro-segment (0.2s) to full segment.
    This is the honest control: it reveals the regime where mean-pool breaks
    (short W, high s) vs where it is fine (long W). V2 does not depend on W.

DATA: POP909 MIDI-derived chroma (piano-roll -> bass/treble 12-pc at fixed fs),
so GT boundaries/labels are TRULY perfect and contamination is the only knob.
POP909 labels are FUNCTIONAL root (discard /bass) -> root+quality only.

CONTAMINATION (frame-level, shared by all variants): (i) exp-decaying PREV-chord
sustain into early frames of each segment, (ii) beat-snap neighbour bleed into
the pre-boundary frames, (iii) per-frame dropout + additive noise.

SCREEN (cheap, NO training): cosine-kNN, song-disjoint split, predict target =
(interval=(next_root-prev_root)%12, reduced quality). Multi-seed the
contamination. Report next-chord accuracy vs (W, s).

HONESTY: on CLEAN chroma with a long window the delta trivially LOSES (mean-pool
sees the full clean chord); the result that matters is WHERE mean-pool COLLAPSES
(short W + high s) and whether the delta degrades gracefully THERE. A null is a
valid, reportable outcome — do not manufacture a win.

Usage:
    .venv/bin/python scripts/exp_leakage_robust_identity.py --songs 60 --fs 50
    (writes docs/research_sessions/leakage_robust_identity_curves.npz)

READ-ONLY reuse: harmonia.data.pop909_parser, yt_chord_corpus _clip_pool
framing, harmonia.theory.chord_vocabulary. Modifies nothing.
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.data.pop909_parser import POP909Parser  # noqa: E402
from harmonia.theory.chord_vocabulary import ChordQuality, get_template  # noqa: E402

DATA_ROOT = REPO / "data"
POP909_DIR = DATA_ROOT / "pop909" / "POP909"
OUT_DIR = REPO / "docs" / "research_sessions"

# bass/treble split follows the repo convention (build_chord_change_features:
# bass = MIDI 0..52, treble = 60..). MIDI [53,59] dropped, same as seg_feature.
BASS_HI = 52
TREB_LO = 60


# ── reduced quality vocabulary ───────────────────────────────────────────────

def reduce_quality(q: ChordQuality) -> str:
    if q == ChordQuality.NO_CHORD:
        return "N"
    iv = get_template(q).intervals
    has_min3, has_maj3 = 3 in iv, 4 in iv
    has_b5, has_s5 = 6 in iv, 8 in iv
    has_min7, has_maj7 = 10 in iv, 11 in iv
    sus = (2 in iv or 5 in iv) and not (has_min3 or has_maj3)
    if sus:
        return "sus"
    if has_b5 and has_min3:
        return "dim"
    if has_s5 and has_maj3:
        return "aug"
    if has_maj3:
        return "dom7" if has_min7 else ("maj7" if has_maj7 else "maj")
    if has_min3:
        return "min7" if has_min7 else ("minmaj7" if has_maj7 else "min")
    return "other"


QUALITIES = ["maj", "min", "dom7", "min7", "maj7", "dim", "aug", "sus",
             "minmaj7", "other"]
QIDX = {q: i for i, q in enumerate(QUALITIES)}


# ── MIDI -> per-frame bass/treble chroma ─────────────────────────────────────

def song_chroma_frames(midi_path: Path, fs: int):
    """(frames (T,24)=[bass12|treble12], times (T,)) at fs Hz from piano-roll."""
    import pretty_midi
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    roll = pm.get_piano_roll(fs=fs)          # (128, T) velocity
    T = roll.shape[1]
    frames = np.zeros((T, 24), np.float32)
    for pitch in range(128):
        row = roll[pitch]
        if not row.any():
            continue
        pc = pitch % 12
        if pitch <= BASS_HI:
            frames[:, pc] += row
        elif pitch >= TREB_LO:
            frames[:, 12 + pc] += row
    return frames, np.arange(T) / float(fs)


# ── contamination (applied once per song per (s,seed); shared by all variants)─

def contaminate(frames, times, evs, s, rng, fs):
    out = frames.copy()
    if s <= 0:
        return out
    tau = max(1, int(0.12 * fs))             # sustain decay ~120 ms
    k_sus = int(0.40 * fs)                    # sustain reach ~400 ms
    k_pre = max(1, int(0.12 * fs))            # neighbour-bleed pre-window
    for i in range(1, len(evs)):
        fi = int(np.searchsorted(times, evs[i].start_beat, "left"))
        # prev-segment mean chroma = sustain source
        m = (times >= evs[i - 1].start_beat) & (times < evs[i].start_beat)
        if not m.any():
            continue
        prev24 = frames[m].mean(0)
        pmag = prev24.sum()
        if pmag <= 0:
            continue
        prof = prev24 / pmag
        for j in range(k_sus):                # (i) sustain into next
            t = fi + j
            if t >= out.shape[0]:
                break
            out[t] += s * np.exp(-j / tau) * pmag * prof
        for j in range(1, k_pre + 1):         # (ii) neighbour bleed pre-boundary
            t = fi - j
            if t < 0:
                break
            out[t] += 0.5 * s * pmag * prof
    drop = rng.random(out.shape) < (0.10 * s)  # (iii) dropout + noise
    out[drop] = 0.0
    out = out + rng.random(out.shape) * (0.15 * s) * (out.mean() + 1e-6)
    return out


# ── pooling helpers ──────────────────────────────────────────────────────────

def _l1_halves(v24):
    b, t = v24[:12].copy(), v24[12:].copy()
    nb, nt = b.sum(), t.sum()
    if nb > 1e-9:
        b /= nb
    if nt > 1e-9:
        t /= nt
    return np.concatenate([b, t])


def _sum_window(frames, times, ta, tb, lo, hi):
    a, b = max(ta, lo), min(tb, hi)
    if b <= a:
        return None
    m = (times >= a) & (times < b)
    if not m.any():
        return None
    return frames[m].sum(0)


def roll24(v24, r):
    r = int(r) % 12
    return np.concatenate([np.roll(v24[:12], -r), np.roll(v24[12:], -r)])


# ── per-song transition records ──────────────────────────────────────────────

@dataclass
class Rec:
    song: str
    interval: int
    quality: int
    prev_qual: int
    prev_root: int
    t0: float
    t1: float
    p0: float
    v2: np.ndarray            # window-independent delta feature (root-norm)


def extract_records(song, cframes, times, med_energy, win=0.15, efloor=0.02):
    evs = [e for e in song.chord_events if e.root >= 0]
    recs = []
    for i in range(1, len(evs)):
        prev, nxt = evs[i - 1], evs[i]
        if prev.label == nxt.label:
            continue
        pr = prev.root
        t0, t1, p0 = nxt.start_beat, nxt.end_beat, prev.start_beat
        nxtw = _sum_window(cframes, times, t0, t0 + win, t0, t1)
        prvw = _sum_window(cframes, times, t0 - win, t0, p0, t0)
        if nxtw is None or prvw is None:
            continue
        if nxtw.sum() < efloor * med_energy or prvw.sum() < efloor * med_energy:
            delta = np.zeros(24, np.float32)
        else:
            delta = _l1_halves(nxtw) - _l1_halves(prvw)
        recs.append(Rec(song.song_id, (nxt.root - pr) % 12,
                        QIDX[reduce_quality(nxt.quality)],
                        QIDX[reduce_quality(prev.quality)], pr,
                        t0, t1, p0, roll24(delta, pr).astype(np.float32)))
    return recs


def feat_v0(rec, cframes, times, W, pre=0.12):
    # mean-pool anchored at boundary + pre-boundary bleed window (beat-snap reg.)
    w = _sum_window(cframes, times, rec.t0 - pre, rec.t0 + W, rec.p0, rec.t1)
    if w is None:
        w = _sum_window(cframes, times, rec.t0, rec.t1, rec.t0, rec.t1)
    return _l1_halves(roll24(w, rec.prev_root)) if w is not None else None


def feat_v1(rec, cframes, times, W):
    # clip-pool strictly [t0, t0+W)  (no pre-boundary bleed)
    w = _sum_window(cframes, times, rec.t0, rec.t0 + W, rec.t0, rec.t1)
    if w is None:
        w = _sum_window(cframes, times, rec.t0, rec.t1, rec.t0, rec.t1)
    return _l1_halves(roll24(w, rec.prev_root)) if w is not None else None


# ── cosine kNN screen (no training) ──────────────────────────────────────────

def _cos_normalize(X):
    n = np.linalg.norm(X, axis=1, keepdims=True)
    return X / np.where(n > 1e-9, n, 1.0)


def knn_acc(Xtr, ytr, Xte, yte, k, n_classes):
    Xtr, Xte = _cos_normalize(Xtr), _cos_normalize(Xte)
    sims = Xte @ Xtr.T
    kk = min(k, Xtr.shape[0] - 1)
    idx = np.argpartition(-sims, kth=kk, axis=1)[:, :kk]
    correct = 0
    for i in range(Xte.shape[0]):
        votes = np.bincount(ytr[idx[i]], minlength=n_classes)
        correct += int(votes.argmax() == yte[i])
    return correct / Xte.shape[0]


# ── main sweep ───────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", type=int, default=60)
    ap.add_argument("--fs", type=int, default=50)
    ap.add_argument("--k", type=int, default=25)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--strengths", type=str, default="0,0.2,0.4,0.6,0.8,1.0")
    ap.add_argument("--windows", type=str, default="0.2,0.4,0.8,2.0")
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--out", type=str, default="leakage_robust_identity")
    args = ap.parse_args()

    strengths = [float(x) for x in args.strengths.split(",")]
    windows = [float(x) for x in args.windows.split(",")]
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Parsing POP909 (first {args.songs} usable songs)...")
    parser = POP909Parser(POP909_DIR)
    songs = []
    for d in sorted(POP909_DIR.iterdir()):
        if not (d.is_dir() and d.name.isdigit()):
            continue
        sg = parser.parse_song(d.name)
        if sg is None or len(sg.beat_times) == 0 or len(sg.chord_events) < 5:
            continue
        songs.append(sg)
        if len(songs) >= args.songs:
            break
    print(f"  {len(songs)} songs.")

    print("Building MIDI-derived chroma frames...")
    chroma = {sg.song_id: song_chroma_frames(sg.midi_path, args.fs) for sg in songs}
    total_frames = sum(f.shape[0] for f, _ in chroma.values())
    print(f"  {total_frames} frames @ {args.fs} Hz.")

    song_ids = [sg.song_id for sg in songs]
    nS, nW, nSeed = len(strengths), len(windows), args.seeds
    # V0/V1 depend on (strength, window, seed); V2 depends on (strength, seed)
    accI = {"v0": np.full((nS, nW, nSeed), np.nan),
            "v1": np.full((nS, nW, nSeed), np.nan),
            "v2": np.full((nS, nSeed), np.nan)}
    accQ = {"v0": np.full((nS, nW, nSeed), np.nan),
            "v1": np.full((nS, nW, nSeed), np.nan),
            "v2": np.full((nS, nSeed), np.nan)}
    n_trans = None

    for si, s in enumerate(strengths):
        for seed in range(nSeed):
            rng = np.random.default_rng(1000 * seed + 7)
            # contaminate + records per song
            all_recs, cf_ref = [], {}
            for sg in songs:
                f, t = chroma[sg.song_id]
                cf = contaminate(f, t, [e for e in sg.chord_events if e.root >= 0],
                                 s, rng, args.fs)
                med = np.median(cf.sum(1)) + 1e-9
                cf_ref[sg.song_id] = (cf, t)
                all_recs += extract_records(sg, cf, t, med)
            if n_trans is None:
                n_trans = len(all_recs)

            rs = np.random.default_rng(seed)
            perm = list(song_ids)
            rs.shuffle(perm)
            n_test = max(1, int(round(args.test_frac * len(perm))))
            test_songs = set(perm[:n_test])
            tr_recs = [r for r in all_recs if r.song not in test_songs]
            te_recs = [r for r in all_recs if r.song in test_songs]

            yI_tr = np.array([r.interval for r in tr_recs])
            yI_te = np.array([r.interval for r in te_recs])
            yQ_tr = np.array([r.quality for r in tr_recs])
            yQ_te = np.array([r.quality for r in te_recs])

            # V2 (window-independent)
            X2tr = np.array([r.v2 for r in tr_recs], np.float32)
            X2te = np.array([r.v2 for r in te_recs], np.float32)
            accI["v2"][si, seed] = knn_acc(X2tr, yI_tr, X2te, yI_te, args.k, 12)
            accQ["v2"][si, seed] = knn_acc(X2tr, yQ_tr, X2te, yQ_te, args.k, len(QUALITIES))

            # V0/V1 per window
            for wi, W in enumerate(windows):
                for name, fn in (("v0", feat_v0), ("v1", feat_v1)):
                    Xtr, itr = [], []
                    for j, r in enumerate(tr_recs):
                        cf, t = cf_ref[r.song]
                        fv = fn(r, cf, t, W)
                        if fv is not None:
                            Xtr.append(fv); itr.append(j)
                    Xte, ite = [], []
                    for j, r in enumerate(te_recs):
                        cf, t = cf_ref[r.song]
                        fv = fn(r, cf, t, W)
                        if fv is not None:
                            Xte.append(fv); ite.append(j)
                    Xtr = np.array(Xtr, np.float32); Xte = np.array(Xte, np.float32)
                    accI[name][si, wi, seed] = knn_acc(
                        Xtr, yI_tr[itr], Xte, yI_te[ite], args.k, 12)
                    accQ[name][si, wi, seed] = knn_acc(
                        Xtr, yQ_tr[itr], Xte, yQ_te[ite], args.k, len(QUALITIES))
            print(f"  s={s:.2f} seed={seed} "
                  f"I[v2={accI['v2'][si,seed]:.3f} "
                  f"v0@{windows[0]}={accI['v0'][si,0,seed]:.3f} "
                  f"v0@{windows[-1]}={accI['v0'][si,-1,seed]:.3f}]")

    npz = OUT_DIR / f"{args.out}_curves.npz"
    np.savez(npz,
             strengths=np.array(strengths), windows=np.array(windows),
             I_v0=accI["v0"], I_v1=accI["v1"], I_v2=accI["v2"],
             Q_v0=accQ["v0"], Q_v1=accQ["v1"], Q_v2=accQ["v2"],
             n_transitions=np.array([n_trans]), n_songs=np.array([len(songs)]),
             fs=np.array([args.fs]), k=np.array([args.k]))
    print(f"\nSaved {npz}")

    # console summary: interval accuracy, mean over seeds
    print("\n=== INTERVAL acc (mean/seeds). rows=strength, V0/V1 per window, V2 ===")
    print("W:        " + "   ".join(f"{w:.2f}" for w in windows))
    for si, s in enumerate(strengths):
        v0 = np.nanmean(accI["v0"][si], 1)
        v1 = np.nanmean(accI["v1"][si], 1)
        v2 = np.nanmean(accI["v2"][si])
        print(f"s={s:.2f} V0 " + " ".join(f"{x:.3f}" for x in v0)
              + f"  | V1 " + " ".join(f"{x:.3f}" for x in v1)
              + f"  | V2 {v2:.3f}")
    return npz


if __name__ == "__main__":
    main()
