"""exp_learned_engine_transition.py — does a SMALL LEARNED transposition-invariant
root+quality engine (baking in the human's ingredients: root/bass normalization,
per-frame intensity-normalized DELTA channel, previous-chord/transition
conditioning) BEAT the shipped musx-argmax source and/or RECOVER the +/-P5 fifth
confusion, on REAL-AUDIO-derived features?  (2026-07-24 research session.)

WHY (established, not re-derived): the shipped nnls24 pipeline lets music-x-lab
decide WHAT the chord is (root/quality/bass) and the NNLS root head decide only
WHERE.  On the 7 frozen brick-0 songs musx-argmax root == 0.7367; +/-P4/P5 fifth
confusion is ~37% of the root loss and is an OVERTONE-fifth acoustic fact.  A
prior screen THIS session found the human's directional DELTA is NULL as a
STANDALONE real-audio feature (template subtraction amplifies real-audio noise).
OPEN question here: does the delta add MARGINAL value as ONE auxiliary channel a
LEARNED model can downweight, and does a learned engine on leakage-robust
clip-pooled features beat musx on the +/-P5 segments?

DESIGN (screen-first; a null is a valid, reported outcome).
  * Common "where" grid = the music-x-lab .lab spans (musx's own segmentation).
    Validated cache-only probe: musx-argmax root on musx spans == 0.7347 pooled,
    matching the shipped nnls-span 0.7367 within 0.2pp -> the identity comparison
    on this grid is directly comparable to the shipped baseline, and holding the
    grid fixed for every engine isolates WHAT (identity) from WHERE (timing).
  * Head A (ABSOLUTE learned engine): root head on the clip-pooled absolute
    24-d C-frame; quality head cascade-rotated by the predicted root (mirrors the
    shipped NNLS24Heads cascade) -- retrained on the LEAKAGE-ROBUST clip-pool.
  * THE ABLATION (the point): train/eval Head A WITH vs WITHOUT the per-frame
    intensity-normalized DELTA channel, everything else fixed.  Isolates whether
    the delta adds anything once it is an AID, not a standalone crutch.
  * Secondary: prev-chord (quality) conditioning in the quality head; and a
    root-level sequential composition of Head-A emission with (a) a WEAK diatonic
    key tie-break and (b) a learned Head-B transition (expected to REINFORCE the
    fifth error -- reported honestly, not used as a default).

DATA.  TRAIN on POP909 real-audio-derived NNLS bothchroma (41 cached songs,
data/cache/nnls_infer/<id>.npz; functional-root GT from the POP909 parser).
EVAL on the 7 verified frozen brick-0 songs (golden/brick0/*.gt.json) via the
READ-ONLY harmonia.eval.accuracy_score engine.  CACHE-ONLY: no audio decode, no
VAMP, no beatthis -- disk-safe under concurrency.

CAVEAT carried in every conclusion: POP909 = functional root + MIDI-synth
lineage; frozen eval = real audio -> a real domain gap.  This is a small PROBE,
not a production engine.

Usage:
    .venv/bin/python scripts/exp_learned_engine_transition.py --seeds 5
    (writes docs/research_sessions/learned_engine_transition_2026-07-24.npz)

READ-ONLY reuse: harmonia.models.{nnls_features,musx_bass}, harmonia.eval.
accuracy_score, harmonia.data.pop909_parser, harmonia.theory.key_profiles,
harmonia.theory.chord_vocabulary.  Modifies nothing.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.data.pop909_parser import POP909Parser  # noqa: E402
from harmonia.eval.accuracy_score import (  # noqa: E402
    Chord, _lookup, load_frozen_gt, score_timeline,
)
from harmonia.models import musx_bass as mxb  # noqa: E402
from harmonia.theory.chord_vocabulary import ChordQuality, get_template  # noqa: E402

NN_CACHE = REPO / "data" / "cache" / "nnls_infer"
MX_CACHE = REPO / "data" / "cache" / "musx_infer"
POP909_DIR = REPO / "data" / "pop909" / "POP909"
OUT_DIR = REPO / "docs" / "research_sessions"
_ROLL_TO_C = 9  # NNLS index0=A -> roll by 9 puts C at index 0

# 7 verified frozen brick-0 songs -> wav stem (== nnls/musx cache key).
FROZEN = {
    "bein_green": "bein_green",
    "blue_bossa": "blue_bossa",
    "blue_bossa_backing": "blue_bossa_150bpm_backing_track",
    "close_to_you": "carpenters_close_to_you",
    "every_breath_you_take": "the_police_every_breath_you_take_official_music_video",
    "georgia_on_my_mind": "ray_charles_georgia_on_my_mind_official_video",
    "stand_by_me": "ben_e_king_stand_by_me_audio",
}


# ── reduced quality vocabulary (== exp_leakage_robust_identity) ───────────────

def reduce_quality(q) -> str:
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
NQ = len(QUALITIES)
# reduced quality -> scorer-recognizable Harte/pipeline quality token.
Q_TO_SCORER = {"maj": "maj", "min": "min", "dom7": "7", "min7": "min7",
               "maj7": "maj7", "dim": "dim", "aug": "aug", "sus": "sus4",
               "minmaj7": "minmaj7", "other": "maj"}
# music-x-lab sev_h token -> reduced quality (for prev-chord conditioning ctx).
MUSX_TO_REDUCED = {"maj": "maj", "min": "min", "7": "dom7", "maj7": "maj7",
                   "min7": "min7", "dim": "dim", "dim7": "dim", "hdim7": "min7",
                   "aug": "aug", "sus2": "sus", "sus4": "sus", "7sus4": "dom7",
                   "9": "dom7", "min9": "min7", "maj9": "maj7", "dom11": "dom7",
                   "dom13": "dom7"}


# ── pooling helpers (leakage-robust clip-pool, C-frame) ──────────────────────

def load_c_frames(stem: str):
    """Cached NNLS bothchroma -> (frames (T,24)=[bassC|trebC], times (T,))."""
    z = np.load(NN_CACHE / f"{stem}.npz")
    arr, times = z["arr"].astype(np.float32), z["times"]
    bass = np.roll(arr[:, :12], _ROLL_TO_C, axis=1)
    treb = np.roll(arr[:, 12:], _ROLL_TO_C, axis=1)
    return np.concatenate([bass, treb], axis=1), times


def _l2_halves(v24):
    b, t = v24[:12].copy(), v24[12:].copy()
    nb, nt = np.linalg.norm(b), np.linalg.norm(t)
    if nb > 1e-9:
        b /= nb
    if nt > 1e-9:
        t /= nt
    return np.concatenate([b, t]).astype(np.float32)


def _l1_halves(v24):
    b, t = v24[:12].copy(), v24[12:].copy()
    nb, nt = b.sum(), t.sum()
    if nb > 1e-9:
        b /= nb
    if nt > 1e-9:
        t /= nt
    return np.concatenate([b, t]).astype(np.float32)


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
    return np.concatenate([np.roll(v24[:12], -r), np.roll(v24[12:], -r)]).astype(np.float32)


def clip_pool_abs(frames, times, t0, t1):
    """L2-per-half clip-pool over [t0,t1) (nearest-frame fallback)."""
    w = _sum_window(frames, times, t0, t1, t0, t1)
    if w is None:
        j = int(np.argmin(np.abs(times - 0.5 * (t0 + t1))))
        w = frames[j]
    return _l2_halves(w)


def onset_delta_abs(frames, times, t0, t1, med, win=0.20, efloor=0.02):
    """Intensity-normalized (unit-L1, energy-gated) directional onset delta:
    L1(next-window) - L1(prev-window), C-frame absolute.  Zeros below floor / at
    the sequence edge (the human's crescendo->~0, real-change->structured)."""
    nxtw = _sum_window(frames, times, t0, t0 + win, t0, t1)
    prvw = _sum_window(frames, times, t0 - win, t0, times[0], t0)
    if nxtw is None or prvw is None:
        return np.zeros(24, np.float32)
    if nxtw.sum() < efloor * med or prvw.sum() < efloor * med:
        return np.zeros(24, np.float32)
    return (_l1_halves(nxtw) - _l1_halves(prvw)).astype(np.float32)


# ── record structs ───────────────────────────────────────────────────────────

@dataclass
class Seg:
    x_abs: np.ndarray      # (24,) L2-halves absolute C-frame
    d_abs: np.ndarray      # (24,) onset delta, C-frame absolute
    root: int              # 0-11 (GT for train; unused/-1 for eval)
    qual: int              # QIDX (GT for train)
    prev_qual: int         # QIDX of previous segment quality (NQ = start/none)
    t0: float
    t1: float


def pop909_records(n_songs: int, win: float):
    parser = POP909Parser(POP909_DIR)
    stems = sorted(p.stem for p in NN_CACHE.glob("*.npz")
                   if p.stem.isdigit() and len(p.stem) == 3)
    recs: list[Seg] = []
    used = 0
    for st in stems:
        if used >= n_songs:
            break
        sg = parser.parse_song(st)
        if sg is None or len(sg.chord_events) < 5:
            continue
        try:
            frames, times = load_c_frames(st)
        except FileNotFoundError:
            continue
        med = float(np.median(frames.sum(1)) + 1e-9)
        evs = [e for e in sg.chord_events if e.root >= 0]
        prev_q = NQ
        for e in evs:
            t0, t1 = float(e.start_beat), float(e.end_beat)
            if t1 <= t0:
                continue
            x = clip_pool_abs(frames, times, t0, t1)
            d = onset_delta_abs(frames, times, t0, t1, med, win)
            q = QIDX[reduce_quality(e.quality)]
            recs.append(Seg(x, d, int(e.root) % 12, q, prev_q, t0, t1))
            prev_q = q
        used += 1
    return recs, used


def frozen_segments(win: float):
    """Per frozen song: musx spans as the common grid, with clip-pooled feature +
    onset delta + musx root/quality context.  Returns {song_id: (gt, [Seg...],
    [musx_root...], [musx_qual_reduced...])}."""
    out = {}
    for sid, stem in FROZEN.items():
        gt = load_frozen_gt(REPO / "golden" / "brick0" / f"{sid}.gt.json")
        frames, times = load_c_frames(stem)
        med = float(np.median(frames.sum(1)) + 1e-9)
        labels = mxb._load_lab(MX_CACHE / f"{stem}_submission.lab")
        segs, mroots, mquals, prev_q = [], [], [], NQ
        for (t0, t1, lab) in labels:
            if t1 <= t0:
                continue
            x = clip_pool_abs(frames, times, t0, t1)
            d = onset_delta_abs(frames, times, t0, t1, med, win)
            body = lab.split("/", 1)[0]
            if lab.strip() in ("N", "X", ""):
                mr, mq = -1, None
            else:
                mr = mxb._parse_root(body.split(":", 1)[0])
                sev = mxb.quality_sev_of_label(lab)
                mq = MUSX_TO_REDUCED.get(sev, "maj") if sev else None
            segs.append(Seg(x, d, -1, -1, prev_q, float(t0), float(t1)))
            mroots.append(mr if mr is not None else -1)
            mquals.append(mq)
            prev_q = QIDX[mq] if mq in QIDX else NQ
        out[sid] = (gt, segs, mroots, mquals)
    return out


# ── small MLP (self-contained; == multihead_training.MLP topology) ───────────

def _make_mlp(din, dout, hid=(128, 64), p=0.2):
    import torch.nn as nn
    layers, d = [], din
    for h in hid:
        layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(p)]
        d = h
    layers += [nn.Linear(d, dout)]
    return nn.Sequential(*layers)


def _train_clf(X, y, din, dout, seed, epochs=60, wd=1e-4, cw=None):
    import torch
    import torch.nn as nn
    torch.manual_seed(seed)
    m = _make_mlp(din, dout)
    m.train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=wd)
    Xt = torch.tensor(np.asarray(X, np.float32))
    yt = torch.tensor(np.asarray(y, np.int64))
    cwt = torch.tensor(cw, dtype=torch.float32) if cw is not None else None
    n = len(Xt)
    rng = np.random.default_rng(seed)
    bs = 256
    for _ in range(epochs):
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            if len(b) < 2:
                continue
            opt.zero_grad()
            loss = nn.functional.cross_entropy(m(Xt[b]), yt[b], weight=cwt)
            loss.backward()
            opt.step()
    m.eval()
    return m


def _proba(m, X):
    import torch
    with torch.no_grad():
        return torch.softmax(m(torch.tensor(np.asarray(X, np.float32))), 1).numpy()


# ── Head A: absolute learned engine (root head + cascade quality head) ───────

class HeadA:
    def __init__(self, recs, seed, use_delta: bool, use_prevqual: bool):
        self.use_delta = use_delta
        self.use_prevqual = use_prevqual
        x = np.array([r.x_abs for r in recs], np.float32)
        d = np.array([r.d_abs for r in recs], np.float32)
        root = np.array([r.root for r in recs], np.int64)
        qual = np.array([r.qual for r in recs], np.int64)
        pq = np.array([r.prev_qual for r in recs], np.int64)

        # root head: absolute [x | d?]
        Xr = np.hstack([x, d]) if use_delta else x
        self.root_din = Xr.shape[1]
        cw_r = _balanced_weights(root, 12)
        self.root_m = _train_clf(Xr, root, self.root_din, 12, seed, cw=cw_r)

        # quality head: root-normalized [roll(x,root) | roll(d,root)? | onehot(pq)?]
        Xq = _qual_features(x, d, root, pq, use_delta, use_prevqual)
        self.qual_din = Xq.shape[1]
        cw_q = _balanced_weights(qual, NQ)
        self.qual_m = _train_clf(Xq, qual, self.qual_din, NQ, seed, cw=cw_q)

    def root_proba(self, X, D):
        Xr = np.hstack([X, D]) if self.use_delta else X
        return _proba(self.root_m, Xr)

    def quality(self, X, D, roots, prev_quals):
        Xq = _qual_features(X, D, np.asarray(roots) % 12,
                            np.asarray(prev_quals), self.use_delta,
                            self.use_prevqual)
        return _proba(self.qual_m, Xq).argmax(1)


def _qual_features(x, d, root, pq, use_delta, use_prevqual):
    n = len(x)
    xr = np.stack([roll24(x[i], root[i]) for i in range(n)])
    parts = [xr]
    if use_delta:
        parts.append(np.stack([roll24(d[i], root[i]) for i in range(n)]))
    if use_prevqual:
        oh = np.zeros((n, NQ + 1), np.float32)
        oh[np.arange(n), np.clip(pq, 0, NQ)] = 1.0
        parts.append(oh)
    return np.hstack(parts).astype(np.float32)


def _balanced_weights(y, k):
    c = np.bincount(y, minlength=k).astype(np.float64)
    w = np.where(c > 0, c.sum() / (k * np.maximum(c, 1)), 0.0)
    return (w / w[w > 0].mean()).astype(np.float32)


# ── diatonic key prior + Head-B transition (secondary sequential composition) ─

_MAJ_DEG = {0, 2, 4, 5, 7, 9, 11}
_MIN_DEG = {0, 2, 3, 5, 7, 8, 10}
_NOTE_PC = {"C": 0, "C#": 1, "DB": 1, "D": 2, "D#": 3, "EB": 3, "E": 4, "F": 5,
            "F#": 6, "GB": 6, "G": 7, "G#": 8, "AB": 8, "A": 9, "A#": 10,
            "BB": 10, "B": 11}


def diatonic_logprior(frames, times, t0s, t1s, beta=0.5):
    """Weak diatonic key tie-break: +beta log-boost for roots in the estimated
    key's diatonic set.  Key from the pooled treble chroma (infer_key)."""
    from harmonia.theory.key_profiles import infer_key
    # pool whole-song treble energy
    treb = frames[:, 12:].sum(0)
    try:
        kr = infer_key(treb)
        name = kr.key_name.split()
        tonic = _NOTE_PC.get(name[0].upper(), 0)
        deg = _MIN_DEG if (len(name) > 1 and "min" in name[1].lower()) else _MAJ_DEG
    except Exception:  # noqa: BLE001
        tonic, deg = 0, _MAJ_DEG
    diat = {(tonic + d) % 12 for d in deg}
    lp = np.array([beta if r in diat else 0.0 for r in range(12)], np.float32)
    return lp


def build_transition(recs, alpha=1.0):
    """Interval transition matrix indexed by prev quality: counts of
    (next_root - prev_root) % 12 for each prev-quality class, over adjacent
    same-song segments (adjacency inferred from the record order + prev_qual)."""
    T = np.ones((NQ + 1, 12), np.float64) * alpha
    for i in range(1, len(recs)):
        pq = recs[i].prev_qual
        if pq == NQ:  # song boundary (start marker) -> skip
            continue
        iv = (recs[i].root - recs[i - 1].root) % 12
        T[pq, iv] += 1.0
    T /= T.sum(1, keepdims=True)
    return np.log(T)


# ── scoring on the frozen set ────────────────────────────────────────────────

def build_pred_chords(segs, roots, quals):
    out = []
    for s, r, q in zip(segs, roots, quals):
        if r < 0:
            out.append(Chord(s.t0, s.t1, None, "N", None, "N"))
        else:
            tok = Q_TO_SCORER.get(QUALITIES[q], "maj")
            from harmonia.eval.accuracy_score import NOTE_SHARP
            lab = f"{NOTE_SHARP[r % 12]}:{tok}"
            out.append(Chord(s.t0, s.t1, r % 12, tok, r % 12, lab))
    return out


def score_pred(gt, pred):
    return score_timeline(gt.gt_chords, pred)


def musx_pred_chords(segs, mroots, mquals):
    out = []
    for s, r, q in zip(segs, mroots, mquals):
        if r < 0 or q is None:
            out.append(Chord(s.t0, s.t1, None, "N", None, "N"))
        else:
            tok = Q_TO_SCORER.get(q, "maj")
            from harmonia.eval.accuracy_score import NOTE_SHARP
            out.append(Chord(s.t0, s.t1, r % 12, tok, r % 12, f"{NOTE_SHARP[r % 12]}:{tok}"))
    return out


def fifth_recovery(gt, segs, mroots, a_roots):
    """Duration-weighted +/-P4/P5 confusion recovery: among sub-intervals where
    musx root is wrong-by-{5,7} vs GT (both chords), what fraction does Head A get
    the GT root EXACTLY?  Also Head-A damage on musx-correct spans."""
    # span-indexed lookups
    bps = {gt.gt_chords[0].t0, gt.gt_chords[-1].t1}
    lo, hi = min(c.t0 for c in gt.gt_chords), max(c.t1 for c in gt.gt_chords)
    for s in segs:
        if lo < s.t0 < hi:
            bps.add(s.t0)
        if lo < s.t1 < hi:
            bps.add(s.t1)
    for c in gt.gt_chords:
        if lo < c.t0 < hi:
            bps.add(c.t0)
        if lo < c.t1 < hi:
            bps.add(c.t1)
    bps = sorted(bps)

    def span_at(t):
        for i, s in enumerate(segs):
            if s.t0 <= t < s.t1:
                return i
        return -1

    conf_dur = rec_dur = 0.0
    corr_dur = dmg_dur = 0.0
    for a, b in zip(bps[:-1], bps[1:]):
        dur = b - a
        if dur <= 0:
            continue
        mid = 0.5 * (a + b)
        g = _lookup(gt.gt_chords, mid)
        si = span_at(mid)
        if si < 0 or g.is_nc:
            continue
        mr, ar = mroots[si], a_roots[si]
        if mr < 0:
            # musx no-chord over a GT chord: count as a fifth-independent miss set
            continue
        diff = (mr - g.root_pc) % 12
        if diff in (5, 7):                 # musx +/-P4/P5 confusion
            conf_dur += dur
            if ar == g.root_pc:
                rec_dur += dur
        elif mr == g.root_pc:              # musx correct
            corr_dur += dur
            if ar != g.root_pc:
                dmg_dur += dur
    return conf_dur, rec_dur, corr_dur, dmg_dur


# ── main ─────────────────────────────────────────────────────────────────────

def _pool_metrics(per_song_scores):
    keys = ("root", "majmin", "sevenths", "partial", "strict")
    num = {k: 0.0 for k in keys}
    tot = 0.0
    for res in per_song_scores:
        tot += res["_total"]
        for k in keys:
            num[k] += res["_num"][k]
    return {k: num[k] / tot if tot else 0.0 for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", type=int, default=41)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--win", type=float, default=0.20)
    ap.add_argument("--out", type=str,
                    default="learned_engine_transition_2026-07-24")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[data] POP909 records (<= {args.songs} songs, win={args.win}s)...")
    recs, used = pop909_records(args.songs, args.win)
    print(f"       {len(recs)} training segments from {used} POP909 songs.")
    print("[data] frozen-set musx-span segments...")
    fz = frozen_segments(args.win)
    logtrans = build_transition(recs)

    # precompute frozen per-song feature matrices
    fzX = {sid: np.array([s.x_abs for s in v[1]], np.float32) for sid, v in fz.items()}
    fzD = {sid: np.array([s.d_abs for s in v[1]], np.float32) for sid, v in fz.items()}
    # key logprior per song
    fzKP = {}
    for sid, stem in FROZEN.items():
        frames, times = load_c_frames(stem)
        segs = fz[sid][1]
        fzKP[sid] = diatonic_logprior(frames, times,
                                      [s.t0 for s in segs], [s.t1 for s in segs])

    # ---- musx baseline (fixed, no seed) ----
    musx_scores = []
    for sid, (gt, segs, mroots, mquals) in fz.items():
        pred = musx_pred_chords(segs, mroots, mquals)
        musx_scores.append((sid, score_pred(gt, pred), mroots))
    musx_pool = _pool_metrics([s for _, s, _ in musx_scores])

    # ---- experiments over configs x seeds ----
    # configs: (use_delta, use_prevqual, mode) mode in {greedy, key, transB}
    configs = {
        "A_nodelta":   dict(use_delta=False, use_prevqual=False),
        "A_delta":     dict(use_delta=True,  use_prevqual=False),
        "A_delta_pq":  dict(use_delta=True,  use_prevqual=True),
    }
    seeds = list(range(args.seeds))

    # storage: metrics[cfg][decode] = list over seeds of pooled dict
    metrics = {c: {d: [] for d in ("greedy", "key", "transB")} for c in configs}
    # per-song root acc, and fifth recovery aggregates, greedy decode only
    recov = {c: [] for c in configs}            # (conf,rec,corr,dmg) pooled per seed
    persong_root = {c: {sid: [] for sid in fz} for c in configs}

    for seed in seeds:
        for cname, kw in configs.items():
            head = HeadA(recs, seed, **kw)
            conf_t = rec_t = corr_t = dmg_t = 0.0
            dec_scores = {d: [] for d in ("greedy", "key", "transB")}
            for sid, (gt, segs, mroots, mquals) in fz.items():
                X, D = fzX[sid], fzD[sid]
                rp = head.root_proba(X, D)          # (n,12)
                logrp = np.log(rp + 1e-9)
                # decode variants (root)
                roots_greedy = rp.argmax(1)
                roots_key = (logrp + fzKP[sid][None, :]).argmax(1)
                roots_trans = _viterbi_root(logrp, logtrans,
                                            [s.prev_qual for s in segs], mquals)
                for dname, rts in (("greedy", roots_greedy),
                                   ("key", roots_key), ("transB", roots_trans)):
                    prevq = _rolling_prevq(mquals)
                    quals = head.quality(X, D, rts, prevq)
                    pred = build_pred_chords(segs, rts, quals)
                    res = score_pred(gt, pred)
                    dec_scores[dname].append(res)
                    if dname == "greedy":
                        persong_root[cname][sid].append(res["mirex_root"])
                        c, r, co, dg = fifth_recovery(gt, segs, mroots, list(rts))
                        conf_t += c; rec_t += r; corr_t += co; dmg_t += dg
            for dname in ("greedy", "key", "transB"):
                metrics[cname][dname].append(_pool_metrics(dec_scores[dname]))
            recov[cname].append((conf_t, rec_t, corr_t, dmg_t))
        print(f"  seed {seed} done.")

    # ---- reference: shipped nnls24 head root on the same clip-pooled features ─
    nnls_ref = _nnls24_reference(fz, fzX)

    _report(musx_pool, musx_scores, metrics, recov, persong_root, nnls_ref,
            configs, used, len(recs), args)

    # save numbers
    np.savez(
        OUT_DIR / f"{args.out}.npz",
        musx_pool=json.dumps(musx_pool),
        metrics=json.dumps({c: {d: [m for m in metrics[c][d]]
                                for d in metrics[c]} for c in metrics}),
        recov=json.dumps({c: recov[c] for c in recov}),
        persong_root=json.dumps({c: persong_root[c] for c in persong_root}),
        nnls_ref=json.dumps(nnls_ref),
        n_train=np.array([len(recs)]), n_songs=np.array([used]),
        seeds=np.array([args.seeds]),
    )
    print(f"\nSaved {OUT_DIR / (args.out + '.npz')}")


def _rolling_prevq(mquals):
    """prev-quality context per span = the previous span's musx quality idx."""
    out = []
    prev = NQ
    for q in mquals:
        out.append(prev)
        prev = QIDX[q] if q in QIDX else NQ
    return np.array(out, np.int64)


def _viterbi_root(logemit, logtrans, prev_quals, mquals):
    """Root-level Viterbi: emission = Head-A root logprob; transition indexed by
    the PREVIOUS segment's (musx) quality -> interval distribution (Head B)."""
    n = len(logemit)
    if n == 0:
        return np.zeros(0, np.int64)
    dp = logemit[0].copy()
    bp = np.zeros((n, 12), np.int64)
    prevq = _rolling_prevq(mquals)
    for i in range(1, n):
        pq = int(prevq[i]) if prevq[i] <= NQ else NQ
        tv = logtrans[min(pq, NQ)]            # (12,) interval logprob
        best = np.full(12, -1e18)
        for r in range(12):
            # score of coming from prev root p to r: interval (r-p)%12
            cand = dp + tv[(r - np.arange(12)) % 12]
            j = int(cand.argmax())
            best[r] = cand[j] + logemit[i, r]
            bp[i, r] = j
        dp = best
    out = np.zeros(n, np.int64)
    out[-1] = int(dp.argmax())
    for i in range(n - 1, 0, -1):
        out[i - 1] = bp[i, out[i]]
    return out


def _nnls24_reference(fz, fzX):
    """Score the SHIPPED nnls24 root head (heads.root_proba) on the same
    clip-pooled features -> the 'current learned engine' context number."""
    from harmonia.models import nnls_features as nf
    heads = nf.get_heads()
    if heads is None:
        return {"available": False}
    scores = []
    for sid, (gt, segs, mroots, mquals) in fz.items():
        rp = heads.root_proba(fzX[sid])
        roots = rp.argmax(1)
        # quality via shipped cascade
        qi = heads.quality_idx(fzX[sid], roots)
        quals7 = [heads.qualities[int(q)] for q in qi]
        from harmonia.eval.accuracy_score import NOTE_SHARP
        pred = []
        for s, r, q in zip(segs, roots, quals7):
            tok = {"maj": "maj", "min": "min", "dom": "7", "hdim": "hdim7",
                   "dim": "dim", "aug": "aug", "sus": "sus4"}.get(q, "maj")
            pred.append(Chord(s.t0, s.t1, int(r) % 12, tok, int(r) % 12,
                              f"{NOTE_SHARP[int(r) % 12]}:{tok}"))
        scores.append(score_pred(gt, pred))
    p = _pool_metrics(scores)
    return {"available": True, "pool": p}


def _mean_std(lst, key):
    a = np.array([m[key] for m in lst])
    return float(a.mean()), float(a.std())


def _report(musx_pool, musx_scores, metrics, recov, persong_root, nnls_ref,
            configs, used, ntrain, args):
    print("\n" + "=" * 78)
    print(f"LEARNED ENGINE vs musx-argmax — 7 frozen brick-0 songs "
          f"(train: {ntrain} POP909 segs / {used} songs, {args.seeds} seeds)")
    print("=" * 78)
    print(f"\nBASELINE  musx-argmax (musx spans):  "
          f"root={musx_pool['root']:.4f}  majmin={musx_pool['majmin']:.4f}  "
          f"7ths={musx_pool['sevenths']:.4f}  partial={musx_pool['partial']:.4f}  "
          f"strict={musx_pool['strict']:.4f}")
    if nnls_ref.get("available"):
        p = nnls_ref["pool"]
        print(f"REFERENCE shipped nnls24 head (clip-pool feats): "
              f"root={p['root']:.4f}  partial={p['partial']:.4f}  strict={p['strict']:.4f}")

    print(f"\nHEAD A (learned) — pooled root (mean+/-std over seeds), GREEDY decode:")
    for c in configs:
        mu, sd = _mean_std(metrics[c]["greedy"], "root")
        pm, ps = _mean_std(metrics[c]["greedy"], "partial")
        sm, ss = _mean_std(metrics[c]["greedy"], "strict")
        d_root = mu - musx_pool["root"]
        print(f"  {c:12} root={mu:.4f}+/-{sd:.4f} ({d_root:+.4f} vs musx)  "
              f"partial={pm:.4f}  strict={sm:.4f}")

    print(f"\nTHE DELTA ABLATION (A_delta - A_nodelta), GREEDY, pooled:")
    for key in ("root", "partial", "strict"):
        n_mu, n_sd = _mean_std(metrics["A_nodelta"]["greedy"], key)
        d_mu, d_sd = _mean_std(metrics["A_delta"]["greedy"], key)
        print(f"  {key:8}  no-delta={n_mu:.4f}  with-delta={d_mu:.4f}  "
              f"delta_effect={d_mu - n_mu:+.4f}")

    print(f"\nSEQUENTIAL COMPOSITION — pooled root by decode (A_delta config):")
    for d in ("greedy", "key", "transB"):
        mu, sd = _mean_std(metrics["A_delta"][d], "root")
        print(f"  {d:8}  root={mu:.4f}+/-{sd:.4f}")

    print(f"\n+/-P4/P5 RECOVERY (A config greedy, duration-weighted, mean over seeds):")
    for c in configs:
        arr = np.array(recov[c])  # (seeds,4): conf,rec,corr,dmg
        conf, rec, corr, dmg = arr.mean(0)
        rr = rec / conf if conf else 0.0
        dd = dmg / corr if corr else 0.0
        print(f"  {c:12} confused={conf:.0f}s recovered={rec:.0f}s ({rr:.1%})  "
              f"| musx-correct={corr:.0f}s damaged={dmg:.0f}s ({dd:.1%})")

    print(f"\nPER-SONG root (A_delta greedy, mean) vs musx:")
    for sid, mres, _ in musx_scores:
        a_mu = float(np.mean(persong_root["A_delta"][sid]))
        print(f"  {sid:22} musx={mres['mirex_root']:.3f}  A_delta={a_mu:.3f}  "
              f"({a_mu - mres['mirex_root']:+.3f})")


if __name__ == "__main__":
    main()
