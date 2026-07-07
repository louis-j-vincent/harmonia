"""Render GT chart vs blind-inferred chart side-by-side with per-chord error overlay.

For each song: two stacked chart images (GT on top, inferred below), with each
inferred chord coloured by correctness:
  green  = root + family correct
  amber  = root correct, family wrong
  red    = root wrong
  purple = no GT overlap (pure segmentation error)

A third row shows a per-chord error bar (MIREX-style weighted overlap metric).

Usage:
    .venv/bin/python scripts/compare_blind_charts.py --songs "Anthropology" "Autumn Leaves"
    .venv/bin/python scripts/compare_blind_charts.py --n-songs 4
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import re
import sys
import tempfile
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pretty_midi
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from analyze_accomp_emission import parse_chord, song_chord_spans
from build_accomp_audio_hard import (
    LEAD_PROGRAMS, SCENARIOS, SOUNDFONTS,
    make_melody, pink, render_to_array, stem_midi, time_varying_degrade,
)
from build_audio_chord_features import (
    BASE7, BASE7_IDX, BUCKET_BASE7, BUCKET_FAMILY,
    EXACT, EXACT_IDX, FAM_IDX, full_chroma,
)
from harmonia.data.midi_renderer import MIDIRenderer
from harmonia.models.chord_graph import ChordGraph
from harmonia.models.chord_scorer import chord_log_likelihood, best_hypothesis
from harmonia.models.motif import Chord as MChord, find_motifs
from harmonia.models.stage1_pitch import PitchExtractor
from harmonia.output.chart_render import (
    BarChord, Chart, render_chart,
    _barline, _section_box, _draw_chord,
    PAPER, INK, RULE, FAINT, ACCENT,
)

DB       = REPO / "data" / "accomp_db" / "db.jsonl"
MANIFEST = REPO / "data" / "accomp_db" / "audio" / "manifest.jsonl"
FEAT     = REPO / "data" / "cache" / "audio_chord_features.npz"
OUT_DIR  = REPO / "docs" / "plots"

NOTE     = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
FAMILIES = ["major", "minor", "diminished", "augmented", "suspended"]
FAM_SHORT = {"major": "maj", "minor": "min", "diminished": "dim",
             "augmented": "aug", "suspended": "sus"}

# Chord templates for root_model feature
_FAM_TONES = {"major": [0,4,7], "minor": [0,3,7], "diminished": [0,3,6],
              "augmented": [0,4,8], "suspended": [0,5,7]}
TEMPLATES = []
for _r in range(12):
    for _fam, _tones in _FAM_TONES.items():
        _t = np.zeros(12)
        for _off in _tones: _t[(_r + _off) % 12] = 1.0
        TEMPLATES.append((_r, _t / np.linalg.norm(_t)))


# ── helpers ───────────────────────────────────────────────────────────────────

def _reg(v88, lo, hi):
    c = np.zeros(12)
    for k in range(88):
        if lo <= 21 + k < hi:
            c[(21 + k) % 12] += v88[k]
    return c

def _c88(v88, lo=0, hi=200):
    c = _reg(v88, lo, hi); n = np.linalg.norm(c)
    return c / n if n > 1e-9 else c

def _unit(v):
    n = np.linalg.norm(v); return v / n if n > 1e-9 else v

def _softmax(x):
    e = np.exp(x - x.max()); return e / e.sum()

def pool_to_beats(frame_times, probs, beat_times):
    n = len(beat_times)
    out = np.zeros((n, probs.shape[1]), dtype=np.float32)
    idx = np.searchsorted(beat_times, frame_times)
    for b, p in zip(idx, probs):
        if 0 <= b < n: out[b] += p
    return out

def _root_model_pred(seg_model, son, snt):
    oc = _c88(son)
    tmpl = np.array([max(oc @ t for r2, t in TEMPLATES if r2 == r) for r in range(12)])
    feat = np.concatenate([oc, _c88(snt), _c88(son, 0, 52), _c88(son, 60, 200), tmpl])
    z = (feat - seg_model["mean"]) / (seg_model["scale"] + 1e-9)
    logits = seg_model["coef"] @ z + seg_model["intercept"]
    probs = _softmax(logits.astype(float))
    return int(seg_model["classes"][probs.argmax()]), probs


def _beat_seq_probs(beat_model, onset_b, note_b, beat_start, beat_end):
    """Pool per-beat windowed LR predictions (±w beats, 240d) over segment."""
    w = int(beat_model["window"][0])
    n = len(onset_b)
    probs = np.zeros(12)
    count = 0
    for b in range(beat_start, min(beat_end, n)):
        row = []
        for delta in range(-w, w + 1):
            bi = b + delta
            if 0 <= bi < n:
                f = np.concatenate([_c88(onset_b[bi]), _c88(note_b[bi]),
                                    _c88(onset_b[bi], 0, 52), _c88(onset_b[bi], 60, 200)])
            else:
                f = np.zeros(48)
            row.append(f)
        feat = np.concatenate(row)
        z = (feat - beat_model["mean"]) / (beat_model["scale"] + 1e-9)
        logits = beat_model["coef"] @ z + beat_model["intercept"]
        probs += _softmax(logits.astype(float))
        count += 1
    if count == 0:
        return np.ones(12) / 12
    p = probs / count
    return (p / p.sum()).tolist()


def _ireal_from_pred(root_pc: int, fam_idx: int) -> str:
    """Build an iReal-style token from predicted root + family index."""
    root = NOTE[root_pc]
    fam = FAMILIES[fam_idx]
    suffix = {"major": "^7", "minor": "-7", "diminished": "o7",
              "augmented": "+", "suspended": "sus"}[fam]
    return root.replace("b", "b") + suffix


def render_hard(midi_path, man_entry, rng):
    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    scen = str(rng.choice(list(SCENARIOS)))
    gains = {k: v * float(rng.uniform(0.8, 1.2)) for k, v in SCENARIOS[scen].items()}
    sf_name = SOUNDFONTS[int(rng.integers(0, len(SOUNDFONTS)))]
    reverb = bool(rng.integers(0, 2))
    stems = {
        "chords": stem_midi(pm, lambda i: (not i.is_drum) and "bass" not in i.name.lower()),
        "bass":   stem_midi(pm, lambda i: "bass" in i.name.lower()),
        "drums":  stem_midi(pm, lambda i: i.is_drum),
    }
    if gains.get("melody", 0) > 0.01:
        mel_pm = pretty_midi.PrettyMIDI()
        m = make_melody(pm, int(rng.choice(LEAD_PROGRAMS)), rng)
        if m: mel_pm.instruments.append(m); stems["melody"] = mel_pm
    waves, sr = {}, 44100
    for name, s in stems.items():
        if s and s.instruments:
            w, sr2 = render_to_array(renderer, s, sf_name, reverb)
            waves[name] = w; sr = sr2
    L = max(len(w) for w in waves.values())
    mix = np.zeros(L, np.float32)
    for name, w in waves.items(): mix[:len(w)] += gains.get(name, 0.5) * w
    mix = time_varying_degrade(mix, sr, rng)
    peak = np.abs(mix).max()
    if peak > 0.99: mix *= 0.99 / peak
    return mix, sr, scen, sf_name


def _chord_graph_pass(chords, onset_b, note_b, bt, bpb, graph, seg_model, sc, clf, ncl):
    """Iterative chord-graph refinement: split implausible chords, swap to neighbours.

    Modifies `chords` in place. Returns (n_splits, n_swaps).
    """
    grid_step = max(1, bpb // 2)
    n_splits = 0
    n_swaps = 0

    # Step A — split on implausibility
    new_chords = []
    for c in chords:
        chroma = np.array(c["chroma_audio"])
        _, _, top_score = best_hypothesis(chroma, TEMPLATES)
        beat_start = c["bar"] * bpb + c["beat"]
        # duration in beats
        dur_beats = round((c["t1"] - c["t0"]) / (bt[1] - bt[0])) if len(bt) > 1 else 1
        if top_score < 0.45 and dur_beats >= grid_step:
            # Split at midpoint
            mid_t = (c["t0"] + c["t1"]) / 2.0
            mid_beat = beat_start + dur_beats // 2
            # First half
            b0 = beat_start
            b1 = mid_beat
            if b1 > b0 and b0 < len(onset_b) and b1 <= len(onset_b):
                son1 = onset_b[b0:b1].sum(0)
                snt1 = note_b[b0:b1].sum(0)
                root1, rp1 = _root_model_pred(seg_model, son1, snt1)
                c1 = dict(c)
                c1["t1"] = mid_t
                c1["root"] = root1
                c1["root_probs"] = rp1.tolist()
                c1["chroma_onset"] = _c88(son1).tolist()
                c1["chroma_note"] = _c88(snt1).tolist()
                c1["chroma_bass"] = _c88(son1, 0, 52).tolist()
                c1["chroma_treble"] = _c88(son1, 60, 200).tolist()
                c1["graph_pass_action"] = "split"
                new_chords.append(c1)
            # Second half
            b0h = mid_beat
            b1h = beat_start + dur_beats
            if b1h > b0h and b0h < len(onset_b) and b1h <= len(onset_b):
                son2 = onset_b[b0h:b1h].sum(0)
                snt2 = note_b[b0h:b1h].sum(0)
                root2, rp2 = _root_model_pred(seg_model, son2, snt2)
                c2 = dict(c)
                c2["t0"] = mid_t
                c2["bar"] = b0h // bpb
                c2["beat"] = b0h % bpb
                c2["root"] = root2
                c2["root_probs"] = rp2.tolist()
                c2["chroma_onset"] = _c88(son2).tolist()
                c2["chroma_note"] = _c88(snt2).tolist()
                c2["chroma_bass"] = _c88(son2, 0, 52).tolist()
                c2["chroma_treble"] = _c88(son2, 60, 200).tolist()
                c2["graph_pass_action"] = "split"
                new_chords.append(c2)
            n_splits += 1
        else:
            c["graph_pass_action"] = "unchanged"
            new_chords.append(c)

    chords[:] = new_chords

    # Step B — iterative graph refinement (max 3 iterations)
    # Gate: only swap when current template score < implausibility_thresh
    # AND a graph neighbour scores > margin above it.
    # Conservative: the root model uses 60d features; CQT-template is 12d,
    # so only override when the mismatch is extreme.
    _IMPLAUS_THRESH = 0.35
    _SWAP_MARGIN = 0.20
    for _iteration in range(3):
        changed = False
        for c in chords:
            chroma = np.array(c["chroma_audio"])
            score_c = chord_log_likelihood(chroma, c["root"], c["pred_fam"], TEMPLATES)
            if score_c >= _IMPLAUS_THRESH:
                continue
            for r2, f2, _ in graph.neighbours(c["root"], c["pred_fam"], k=8):
                score_n = chord_log_likelihood(chroma, r2, f2, TEMPLATES)
                if score_n > score_c + _SWAP_MARGIN:
                    c["root"] = r2
                    c["pred_fam"] = f2
                    c["graph_pass_action"] = "swapped"
                    changed = True
                    n_swaps += 1
                    break
        if not changed:
            break

    # After swaps: update tmpl_score to reflect new chord
    for c in chords:
        if c.get("graph_pass_action") == "swapped":
            chroma_12_arr = np.array(c["chroma_audio"])
            fam_name = FAMILIES[c["pred_fam"]]
            fam_names = list(_FAM_TONES.keys())
            c["tmpl_score"] = round(
                next((float(chroma_12_arr @ t)
                      for ti, (r2, t) in enumerate(TEMPLATES)
                      if r2 == c["root"] and fam_names[ti % len(fam_names)] == fam_name), 0.0), 3)

    # Step C — re-run family classifier on updated roots
    if n_swaps > 0 or n_splits > 0:
        # Rebuild features with new roots and re-classify
        for c in chords:
            son = np.array(c["chroma_onset"]) * np.linalg.norm(np.array(c["chroma_onset"]))
            snt = np.array(c["chroma_note"]) * np.linalg.norm(np.array(c["chroma_note"]))
            rr = lambda v, root=c["root"]: np.roll(v, -root)
            oc = np.array(c["chroma_onset"])
            nc = np.array(c["chroma_note"])
            bc = np.array(c["chroma_bass"])
            tc = np.array(c["chroma_treble"])
            c["feat"] = np.hstack([rr(oc), rr(nc), rr(bc), rr(tc)])

        X = sc.transform(np.stack([c["feat"] for c in chords]))
        prob_fam = np.full((len(chords), ncl["fam"]), 1e-9)
        prob_fam[:, clf["fam"].classes_] = clf["fam"].predict_proba(X)
        prob_fam = prob_fam / prob_fam.sum(1, keepdims=True)

        for i, c in enumerate(chords):
            c["pred_fam"] = int(prob_fam[i].argmax())
            c["fam_conf"] = float(prob_fam[i].max())
            c["fam_probs"] = prob_fam[i].tolist()

    return n_splits, n_swaps


def infer_blind(rec, man_entry, sc, clf, ncl, rng, seg_model, beat_model):
    """Full blind inference. Returns list of chord dicts with GT comparison."""
    mix, sr, scen, sf_name = render_hard(REPO / man_entry["midi_path"], man_entry, rng)

    ex = PitchExtractor(cache_dir=REPO / "data" / "cache" / "compare_blind")
    tmp = Path(tempfile.mktemp(suffix=".wav"))
    try:
        sf.write(tmp, mix, sr)
        acts = ex.extract(tmp, use_cache=False)
    finally:
        tmp.unlink(missing_ok=True)

    # Use librosa only to find the phase of the first beat; then build a
    # perfectly regular grid at the known BPM.  This eliminates tempo-tracking
    # drift that misaligns cuts from the true bar grid.
    _, bf = librosa.beat.beat_track(y=mix.astype(float), sr=sr,
                                    bpm=float(man_entry["tempo"]), units="frames")
    raw_bt = librosa.frames_to_time(bf, sr=sr)
    if len(raw_bt) < 4:
        return None, scen, sf_name
    spb = 60.0 / man_entry["tempo"]
    t0_phase = float(raw_bt[0])
    n_beats_total = int((len(mix) / sr - t0_phase) / spb) + 1
    bt = np.array([t0_phase + i * spb for i in range(n_beats_total)])

    # CQT chroma from raw audio — LTAS-normalised so all 12 pitch classes have
    # equal long-term energy (prevents bass-register dominance in comparisons).
    y_f = mix.astype(float)
    hop = 512
    chroma_cqt_raw = librosa.feature.chroma_cqt(y=y_f, sr=sr,
                                                 bins_per_octave=36, hop_length=hop)
    # LTAS normalisation: divide each row by its mean, preserving local dynamics
    ltas = chroma_cqt_raw.mean(axis=1, keepdims=True)
    ltas = np.where(ltas < 1e-9, 1.0, ltas)
    chroma_cqt = chroma_cqt_raw / ltas          # (12, T), each PC has mean ≈ 1
    chroma_times = librosa.frames_to_time(np.arange(chroma_cqt.shape[1]), sr=sr,
                                          hop_length=hop)

    def _seg_chroma(t0, t1):
        i0 = int(np.searchsorted(chroma_times, t0))
        i1 = int(np.searchsorted(chroma_times, t1))
        if i1 <= i0: i1 = i0 + 1
        c = chroma_cqt[:, i0:i1].mean(axis=1)
        n = np.linalg.norm(c)
        return (c / n if n > 1e-9 else c).tolist()

    def _seg_chroma_2d(t0, t1, max_cols=72):
        i0 = int(np.searchsorted(chroma_times, t0))
        i1 = int(np.searchsorted(chroma_times, t1))
        if i1 <= i0: i1 = i0 + 1
        chunk = chroma_cqt[:, i0:i1].astype(float)
        N = chunk.shape[1]
        if N > max_cols:
            step = N / max_cols
            cols = [chunk[:, int(j*step):max(int(j*step)+1, int((j+1)*step))].mean(1)
                    for j in range(max_cols)]
            chunk = np.stack(cols, axis=1)
        mx = chunk.max()
        if mx > 1e-9: chunk /= mx
        return [[round(float(v), 3) for v in chunk[r]] for r in range(12)]

    bpb = man_entry["beats_per_bar"]
    onset_b = pool_to_beats(acts.frame_times, acts.onset_probs, bt)
    note_b  = pool_to_beats(acts.frame_times, acts.note_probs,  bt)

    # Segmentation: perfect beat grid, window-pooled CQT chroma comparison.
    # Compare pooled LTAS-normalised chroma of slot [b, b+step) vs [b-step, b).
    # Bass (below MIDI 52) drives cuts; CQT full-chroma is secondary trigger.
    grid_step       = max(1, bpb // 2)
    bass_nov_thresh = 0.12
    cqt_nov_thresh  = 0.15   # secondary: LTAS-normalised CQT chroma novelty
    n_beats = len(onset_b)

    def _pool_bass(b0, b1):
        chunk = onset_b[max(0, b0):min(b1, n_beats)].sum(0)
        return _unit(_reg(chunk, 0, 52))

    def _pool_cqt(b0, b1):
        t0 = float(bt[max(0, b0)])
        t1 = float(bt[min(b1, len(bt) - 1)])
        i0 = int(np.searchsorted(chroma_times, t0))
        i1 = int(np.searchsorted(chroma_times, t1))
        if i1 <= i0: i1 = i0 + 1
        c = chroma_cqt[:, i0:i1].mean(axis=1)
        n = np.linalg.norm(c); return c / n if n > 1e-9 else c

    boundaries = [0]
    for b in range(grid_step, n_beats, grid_step):
        prev_bass = _pool_bass(b - grid_step, b)
        curr_bass = _pool_bass(b, b + grid_step)
        bass_nov  = 1 - float(prev_bass @ curr_bass)
        prev_cqt  = _pool_cqt(b - grid_step, b)
        curr_cqt  = _pool_cqt(b, b + grid_step)
        cqt_nov   = 1 - float(prev_cqt @ curr_cqt)
        if bass_nov > bass_nov_thresh or cqt_nov > cqt_nov_thresh:
            boundaries.append(b)
    boundaries.append(n_beats)

    segs = []
    for i in range(len(boundaries) - 1):
        b0, b1 = boundaries[i], boundaries[i + 1]
        son = onset_b[b0:b1].sum(0)
        snt = note_b[b0:b1].sum(0)
        if son.sum() > 1e-6:
            segs.append((b0, b1, son, snt))
    if not segs:
        return None, scen, sf_name

    # Classify
    spb = 60.0 / man_entry["tempo"]
    chords = []
    for beat_start, beat_end, son, snt in segs:
        root, root_probs = _root_model_pred(seg_model, son, snt)
        bs_probs = _beat_seq_probs(beat_model, onset_b, note_b, beat_start, beat_end)
        rr = lambda c: np.roll(c, -root)
        feat = np.hstack([rr(full_chroma(son)), rr(full_chroma(snt)),
                          rr(_reg(son, 0, 52)), rr(_reg(son, 60, 200))])
        t0 = float(bt[beat_start])
        t1 = float(bt[min(beat_end, len(bt) - 1)])
        bar = beat_start // bpb
        beat_in_bar = beat_start % bpb
        bass_12 = _reg(son, 0, 52)
        # CQT chroma for template scoring
        chroma_12 = _seg_chroma(t0, t1)
        chroma_12_arr = np.array(chroma_12)
        # Top-5 template candidates (will attach inf_tmpl_score after family classification)
        tmpl_scored = []
        for ti, (r2, t) in enumerate(TEMPLATES):
            fam_i = ti % len(_FAM_TONES)
            tmpl_scored.append((r2, fam_i, float(chroma_12_arr @ t)))
        tmpl_scored.sort(key=lambda x: -x[2])
        top5 = [(NOTE[r2], FAMILIES[fi], round(s, 3)) for r2, fi, s in tmpl_scored[:5]]
        # Bass entropy (stability): argmax-PC distribution over bass frames in segment
        bass_frames = onset_b[beat_start:beat_end]
        bass_argmax_counts = np.zeros(12)
        for bf2 in bass_frames:
            bv = _reg(bf2, 0, 52)
            if bv.sum() > 1e-6:
                bass_argmax_counts[int(bv.argmax())] += 1
        bn = bass_argmax_counts.sum()
        if bn > 0:
            bp = bass_argmax_counts / bn
            bass_entropy = float(-np.sum(bp[bp > 0] * np.log2(bp[bp > 0])))
        else:
            bass_entropy = 0.0

        chords.append({"root": root, "bar": bar, "beat": beat_in_bar,
                       "t0": t0, "t1": t1, "feat": feat,
                       "chroma_audio":  chroma_12,
                       "chroma_2d":     _seg_chroma_2d(t0, t1),
                       "chroma_onset":  _c88(son).tolist(),
                       "chroma_note":   _c88(snt).tolist(),
                       "chroma_bass":   _c88(son, 0, 52).tolist(),
                       "chroma_treble": _c88(son, 60, 200).tolist(),
                       "root_probs":    root_probs.tolist(),
                       "beat_seq_probs": bs_probs,
                       "bass_root": int(bass_12.argmax()) if bass_12.sum() > 1e-6 else 0,
                       "top5_templates": top5,
                       "tmpl_scored": tmpl_scored,  # full list, resolved to tmpl_score after fam assigned
                       "bass_entropy": round(bass_entropy, 3)})

    X = sc.transform(np.stack([c["feat"] for c in chords]))
    prob = {}
    for lv in ("fam", "b7", "ex"):
        p = np.full((len(chords), ncl[lv]), 1e-9)
        p[:, clf[lv].classes_] = clf[lv].predict_proba(X)
        prob[lv] = p / p.sum(1, keepdims=True)

    for i, c in enumerate(chords):
        c["pred_fam"] = int(prob["fam"][i].argmax())
        c["fam_conf"] = float(prob["fam"][i].max())
        c["fam_probs"] = prob["fam"][i].tolist()
        c["b7_probs"]  = prob["b7"][i].tolist()
        # Template score for inferred chord (now that pred_fam is set)
        c["tmpl_score"] = round(
            next((s for r2, fi, s in c["tmpl_scored"]
                  if r2 == c["root"] and fi == c["pred_fam"]), 0.0), 3)
        del c["tmpl_scored"]  # don't serialize large list; top5 is enough

    # ── chord-graph refinement pass ──────────────────────────────────────────
    graph = ChordGraph()
    graph_splits, graph_swaps = _chord_graph_pass(
        chords, onset_b, note_b, bt, bpb, graph, seg_model, sc, clf, ncl)

    # Build GT map
    chord_at = {(e["bar"] - 1) * bpb + e["beat"]: e for e in rec["chord_timeline"]}
    gt_segs = []
    for t0, t1, root_gt, _q in song_chord_spans(rec):
        b0 = int(round(t0 / spb))
        mma = chord_at.get(b0, {}).get("mma")
        p = parse_chord(mma) if mma else None
        if p is None or p[1] not in BUCKET_FAMILY: continue
        ireal = chord_at.get(b0, {}).get("ireal", NOTE[root_gt % 12])
        gt_segs.append({"t0": t0, "t1": t1, "root": root_gt % 12,
                        "fam_i": FAM_IDX[BUCKET_FAMILY[p[1]]], "ireal": ireal,
                        "bar": b0 // bpb, "beat": b0 % bpb})

    # Match each inferred chord to GT by weighted overlap
    total_dur = sum(g["t1"] - g["t0"] for g in gt_segs) or 1.0
    weighted_root = weighted_fam = 0.0
    for c in chords:
        for g in gt_segs:
            ov = max(0, min(c["t1"], g["t1"]) - max(c["t0"], g["t0"]))
            if ov <= 0: continue
            if c["root"] == g["root"]:
                weighted_root += ov
            if c["pred_fam"] == g["fam_i"] and c["root"] == g["root"]:
                weighted_fam += ov

    # Attach GT to each chord (best overlap, for display)
    for c in chords:
        best_gt, best_ov = None, 0.0
        for g in gt_segs:
            ov = max(0, min(c["t1"], g["t1"]) - max(c["t0"], g["t0"]))
            if ov > best_ov: best_ov = ov; best_gt = g
        c["gt"] = best_gt
        c["gt_ov"] = best_ov
        c["root_ok"] = best_gt is not None and c["root"] == best_gt["root"]
        c["fam_ok"]  = best_gt is not None and c["pred_fam"] == best_gt["fam_i"] and c["root_ok"]
        c["seg_ok"]  = best_ov > 0

    return {
        "chords": chords,
        "gt_segs": gt_segs,
        "weighted_root": weighted_root / total_dur,
        "weighted_fam":  weighted_fam  / total_dur,
        "n_gt": len(gt_segs),
        "n_inf": len(chords),
        "graph_pass_splits": graph_splits,
        "graph_pass_swaps": graph_swaps,
        "scen": scen,
        "sf": sf_name,
    }, scen, sf_name


# ── chart rendering ────────────────────────────────────────────────────────────

_COL_OK    = "#1baf7a"   # root + family correct
_COL_ROOT  = "#eda100"   # root ok, family wrong
_COL_WRONG = "#e34948"   # root wrong
_COL_NONE  = "#9b59b6"   # no GT match


def _DEAD_inferred_chart(rec, man_entry, result) -> Chart:
    """Build a Chart from blind-inferred chords, coloured by correctness."""
    chords = []
    for c in result["chords"]:
        if c["fam_ok"]:     col = _COL_OK
        elif c["root_ok"]:  col = _COL_ROOT
        elif c["seg_ok"]:   col = _COL_WRONG
        else:               col = _COL_NONE
        symbol = _ireal_from_pred(c["root"], c["pred_fam"])
        chords.append(BarChord(bar=c["bar"], beat=c["beat"], symbol=symbol, colour=col))

    return Chart(
        title=f"INFERRED — {rec['title']}",
        composer=f"root acc {result['weighted_root']:.0%} · fam acc {result['weighted_fam']:.0%}",
        key=rec.get("key", ""),
        style=f"{result['scen']} · {result['sf']}",
        tempo=man_entry["tempo"],
        time_signature=tuple(rec.get("time_signature", [4, 4])),
        n_bars=rec.get("n_bars", 0),
        section_per_bar=rec.get("section_per_bar", []),
        chords=chords,
    )


def _render_error_bar(result, rec, man_entry, width_px=1800) -> bytes:
    """Render a per-GT-chord error strip as PNG bytes.

    Each GT chord is a horizontal cell. Colour = correctness of the best-matching
    inferred chord. Height encodes overlap fraction.
    """
    gt_segs  = result["gt_segs"]
    inf_chords = result["chords"]
    if not gt_segs:
        return b""

    total_t = gt_segs[-1]["t1"] - gt_segs[0]["t0"]
    dpi = 150
    fig_w = width_px / dpi
    fig_h = 1.1
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=dpi)
    fig.patch.set_facecolor("#0a0e14")
    ax.set_facecolor("#0a0e14")
    ax.set_xlim(0, total_t); ax.set_ylim(0, 1); ax.axis("off")

    t_start = gt_segs[0]["t0"]
    for g in gt_segs:
        x0 = g["t0"] - t_start; x1 = g["t1"] - t_start
        w = x1 - x0

        # Find best-matching inferred chord
        best_inf, best_ov = None, 0.0
        for c in inf_chords:
            ov = max(0, min(c["t1"], g["t1"]) - max(c["t0"], g["t0"]))
            if ov > best_ov: best_ov = ov; best_inf = c

        if best_inf is None:
            col = "#4a3aa7"
            label = "?"
            ov_frac = 0.0
        elif best_inf["fam_ok"]:
            col = _COL_OK
            label = NOTE[best_inf["root"]] + FAM_SHORT[FAMILIES[best_inf["pred_fam"]]]
            ov_frac = best_ov / max(g["t1"] - g["t0"], 1e-6)
        elif best_inf["root_ok"]:
            col = _COL_ROOT
            label = NOTE[best_inf["root"]] + FAM_SHORT[FAMILIES[best_inf["pred_fam"]]]
            ov_frac = best_ov / max(g["t1"] - g["t0"], 1e-6)
        else:
            col = _COL_WRONG
            label = NOTE[best_inf["root"]] + FAM_SHORT[FAMILIES[best_inf["pred_fam"]]]
            ov_frac = best_ov / max(g["t1"] - g["t0"], 1e-6)

        # Background (GT extent)
        ax.barh(0.5, w, left=x0, height=0.85, color=col + "44",
                edgecolor=col, linewidth=0.8, align="center")
        # Filled bar showing overlap fraction
        ax.barh(0.5, w * ov_frac, left=x0, height=0.55,
                color=col, alpha=0.85, align="center")

        # GT root label (top)
        gt_label = g["ireal"] if g.get("ireal") else NOTE[g["root"]]
        if w > total_t * 0.015:
            ax.text(x0 + w * 0.5, 0.88, gt_label,
                    ha="center", va="top", fontsize=5.5, color="#c8d0dc",
                    fontfamily="monospace")
            ax.text(x0 + w * 0.5, 0.12, label,
                    ha="center", va="bottom", fontsize=5.0, color=col,
                    fontfamily="monospace")

    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.02,
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return buf.getvalue()


def _bytes_to_b64(b: bytes) -> str:
    return base64.b64encode(b).decode()


def render_superposed(rec, man_entry, result, bars_per_row: int = 8) -> tuple[bytes, list]:
    """Single chart: GT chord (top half) and inferred chord (bottom half) in every cell.

    Returns (png_bytes, hotspots) where hotspots is a list of dicts:
      {x, y, w, h}  — fractional image coords (0–1)
      plus chord detail fields for the click modal.
    """
    import math
    from matplotlib.patches import FancyBboxPatch, Rectangle

    gt_chart = Chart.from_db_record(rec)
    n_bars = max(gt_chart.n_bars,
                 (max((c.bar for c in gt_chart.chords), default=-1) + 1))
    n_rows = max(1, math.ceil(n_bars / bars_per_row))

    fig_w    = 13.0
    row_in   = 1.35      # taller rows to fit two chord labels
    header_in = 1.4
    pad_in   = 0.4
    fig_h    = header_in + n_rows * row_in + pad_in

    header_h = header_in / fig_h
    pad_h    = pad_in / fig_h
    grid_top = 1 - header_h
    row_h    = (grid_top - pad_h) / n_rows
    left, right = 0.04, 0.96
    cell_w   = (right - left) / bars_per_row

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=PAPER)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_facecolor(PAPER); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    def y_from_top(inch):
        return 1 - inch / fig_h

    # header
    ax.text(0.5, y_from_top(0.30), rec["title"],
            fontsize=18, weight="bold", color=INK, ha="center", va="top", family="serif")
    sub = []
    if rec.get("key"):   sub.append(f"Key {rec['key']}")
    if rec.get("style"): sub.append(rec["style"])
    sub.append(f"♩ = {man_entry['tempo']}")
    ax.text(left, y_from_top(0.82), "   ".join(sub),
            fontsize=9, color=FAINT, ha="left", va="top", style="italic")
    ax.text(right, y_from_top(0.82),
            f"root {result['weighted_root']:.0%}  ·  fam {result['weighted_fam']:.0%}"
            f"  ·  {result['n_inf']} segs / {result['n_gt']} GT",
            fontsize=9, color=FAINT, ha="right", va="top", style="italic")

    # legend
    legend = [(_COL_OK, "root+fam ✓"), (_COL_ROOT, "root ✓ fam ✗"),
              (_COL_WRONG, "root ✗"),   (_COL_NONE, "no GT match"),
              ("#e34948", "▽ missed change")]
    lx = left
    for col, label in legend:
        patch = FancyBboxPatch((lx, y_from_top(1.12) - 0.013), 0.016, 0.013,
                               boxstyle="round,pad=0.001",
                               facecolor=col + "55", edgecolor=col, linewidth=0.8,
                               transform=ax.transAxes)
        ax.add_patch(patch)
        ax.text(lx + 0.020, y_from_top(1.12) - 0.006, label, fontsize=7.5,
                color=FAINT, va="center", transform=ax.transAxes)
        lx += 0.125

    # row labels
    ax.text(left - 0.025, grid_top - 0.5 * row_h / n_rows, "GT",
            fontsize=7, color=FAINT, ha="right", va="center",
            transform=ax.transAxes, style="italic")

    bpb = man_entry["beats_per_bar"]
    spb_list = rec.get("section_per_bar", [])

    def section_of(bar):
        return spb_list[bar] if 0 <= bar < len(spb_list) else ""

    gt_by_bar: dict[int, list] = {}
    for c in gt_chart.chords:
        gt_by_bar.setdefault(c.bar, []).append(c)

    inf_by_bar: dict[int, list] = {}
    for c in result["chords"]:
        if c["fam_ok"]:    col = _COL_OK
        elif c["root_ok"]: col = _COL_ROOT
        elif c["seg_ok"]:  col = _COL_WRONG
        else:              col = _COL_NONE
        inf_by_bar.setdefault(c["bar"], []).append(
            (c["beat"], _ireal_from_pred(c["root"], c["pred_fam"]), col))

    # Pre-compute inferred segment start times for missed-change detection
    spb = 60.0 / man_entry["tempo"]
    inf_t0_set = {c["t0"] for c in result["chords"]}  # set of inferred boundary times

    def _change_missed(gt_t0: float) -> bool:
        """True if no inferred segment starts within ±0.6 beats of this GT boundary."""
        tol = 0.6 * spb
        return not any(abs(t - gt_t0) < tol for t in inf_t0_set)

    gap = 0.10 * row_h
    gt_size  = 13
    inf_size = 12

    hotspots = []   # list of {x,y,w,h,chord_data} in figure-fraction coords

    for bar in range(n_bars):
        row = bar // bars_per_row
        col = bar % bars_per_row
        x0 = left + col * cell_w
        x1 = x0 + cell_w
        y_top = grid_top - row * row_h - gap
        y_bot = y_top - (row_h - 2 * gap)
        y_mid = (y_top + y_bot) / 2

        new_section = (bar == 0 or section_of(bar) != section_of(bar - 1))

        # inferred background tint (bottom half)
        inf_entries = sorted(inf_by_bar.get(bar, []), key=lambda x: x[0])
        if inf_entries:
            dom_col = inf_entries[0][2]
            ax.add_patch(Rectangle(
                (x0 + 0.001, y_bot), cell_w - 0.002, y_mid - y_bot,
                facecolor=dom_col + "28", edgecolor="none",
                transform=ax.transAxes, zorder=0))

        _barline(ax, x0, y_bot, y_top, heavy=new_section and col != 0,
                 double=new_section)

        if new_section and section_of(bar):
            _section_box(ax, x0, y_top, section_of(bar))

        # dashed divider
        ax.plot([x0 + 0.002, x1 - 0.002], [y_mid, y_mid],
                color=RULE, lw=0.7, linestyle="--",
                transform=ax.transAxes, zorder=2)

        # GT chords (top half)
        gt_cs = sorted(gt_by_bar.get(bar, []), key=lambda c: c.beat)
        for i, c in enumerate(gt_cs):
            cx = (x0 + x1) / 2 if len(gt_cs) == 1 else \
                 x0 + cell_w * (0.22 + 0.60 * (c.beat / bpb if bpb else i / len(gt_cs)))
            cy = y_mid + (y_top - y_mid) * 0.44
            _draw_chord(ax, cx, cy, c.symbol, gt_size, INK)

        # inferred chords (bottom half)
        for i, (beat, symbol, icol) in enumerate(inf_entries):
            cx = (x0 + x1) / 2 if len(inf_entries) == 1 else \
                 x0 + cell_w * (0.22 + 0.60 * (beat / bpb if bpb else i / len(inf_entries)))
            cy = y_bot + (y_mid - y_bot) * 0.48
            _draw_chord(ax, cx, cy, symbol, inf_size, icol)

        # Build hotspot for this bar's inferred chord(s) — whole bottom-half cell
        for c in result["chords"]:
            if c["bar"] != bar:
                continue
            gt = c.get("gt")
            hotspots.append({
                "x": x0, "y": y_bot, "w": cell_w, "h": y_mid - y_bot,
                # display fields
                "bar": bar + 1,
                "beat": c["beat"],
                "t0": round(c["t0"], 3),
                "t1": round(c["t1"], 3),
                "inf_root": NOTE[c["root"]],
                "inf_fam":  FAMILIES[c["pred_fam"]],
                "fam_conf": round(c["fam_conf"], 3),
                "bass_root": NOTE[c["bass_root"]],
                "root_probs":     [round(v, 3) for v in c["root_probs"]],
                "beat_seq_probs": [round(v, 3) for v in c["beat_seq_probs"]],
                "fam_probs":  [round(v, 3) for v in c["fam_probs"]],
                "chroma_audio":  [round(v, 3) for v in c["chroma_audio"]],
                "chroma_2d":     c["chroma_2d"],
                "chroma_onset":  [round(v, 3) for v in c["chroma_onset"]],
                "chroma_note":   [round(v, 3) for v in c["chroma_note"]],
                "chroma_bass":   [round(v, 3) for v in c["chroma_bass"]],
                "chroma_treble": [round(v, 3) for v in c["chroma_treble"]],
                "gt_root":  NOTE[gt["root"]] if gt else None,
                "gt_fam":   FAMILIES[gt["fam_i"]] if gt else None,
                "gt_ireal": gt["ireal"] if gt else None,
                "gt_ov":    round(c["gt_ov"], 3),
                "root_ok":  c["root_ok"],
                "fam_ok":   c["fam_ok"],
                "graph_pass_action": c.get("graph_pass_action", "unchanged"),
                "tmpl_score":    c.get("tmpl_score", 0.0),
                "top5_templates": c.get("top5_templates", []),
                "bass_entropy":  c.get("bass_entropy", 0.0),
                # model agreement: root model top vs beat_seq top
                "root_model_top": NOTE[int(np.argmax(c["root_probs"]))],
                "beat_seq_top":   NOTE[int(np.argmax(c["beat_seq_probs"]))],
            })

        # missed change markers: GT has a mid-bar chord change we didn't detect
        gt_cs_sorted = sorted(gt_by_bar.get(bar, []), key=lambda c: c.beat)
        for c in gt_cs_sorted:
            if c.beat == 0:
                continue  # downbeat change — always represented
            gt_t0 = (bar * bpb + c.beat) * spb
            if _change_missed(gt_t0):
                # draw a ▽ on the dashed divider at this beat position
                frac = c.beat / bpb if bpb else 0.5
                mx = x0 + cell_w * (0.12 + 0.76 * frac)
                ax.plot(mx, y_mid, marker="v", markersize=4.5,
                        color="#e34948", markeredgewidth=0,
                        transform=ax.transAxes, zorder=5, clip_on=False)

    # right barlines
    for row in range(n_rows):
        last_col = min(bars_per_row, n_bars - row * bars_per_row)
        x = left + last_col * cell_w
        y_top = grid_top - row * row_h - gap
        y_bot = y_top - (row_h - 2 * gap)
        is_final = row == n_rows - 1
        _barline(ax, x, y_bot, y_top, heavy=is_final, double=is_final,
                 colour=ACCENT if is_final else RULE)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, bbox_inches="tight", pad_inches=0.05,
                facecolor=PAPER)
    plt.close(fig)
    return buf.getvalue(), hotspots


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blind inference vs ground truth</title>
<style>
* { box-sizing:border-box; margin:0; padding:0; }
body { background:#0a0e14; color:#c8d0dc; font-family:system-ui,sans-serif; font-size:13px; }
.page { max-width:1500px; margin:0 auto; padding:24px; }
h1 { font-size:20px; color:#e2e8f0; margin-bottom:18px; }
.song-block { margin-bottom:40px; border:1px solid #1e2c3a; border-radius:10px;
              overflow:hidden; background:#0d1520; }
.song-header { background:#111820; padding:10px 16px; border-bottom:1px solid #1e2c3a;
               display:flex; justify-content:space-between; align-items:baseline; }
.song-header h2 { font-size:15px; color:#e2e8f0; }
.song-header .stats { font-size:11px; color:#5a6a7e; }
.song-header .stats b { color:#58d4ff; }
.panel { padding:12px 16px 8px; position:relative; }
.chart-wrap { position:relative; display:inline-block; width:100%; }
.chart-wrap img { width:100%; border-radius:5px; display:block; }
.chart-wrap canvas { position:absolute; top:0; left:0; width:100%; height:100%;
                     cursor:pointer; border-radius:5px; }

/* modal */
#modal-overlay { display:none; position:fixed; inset:0; background:rgba(0,0,0,.72);
                 z-index:1000; align-items:center; justify-content:center; }
#modal-overlay.open { display:flex; }
#modal-box { background:#111a26; border:1px solid #253447; border-radius:12px;
             padding:24px 28px; max-width:560px; width:90%; position:relative;
             color:#c8d0dc; font-size:13px; max-height:90vh; overflow-y:auto; }
#modal-close { position:absolute; top:12px; right:16px; background:none; border:none;
               color:#5a6a7e; font-size:20px; cursor:pointer; line-height:1; }
#modal-close:hover { color:#e2e8f0; }
.modal-title { font-size:17px; font-weight:700; color:#e2e8f0; margin-bottom:4px; }
.modal-sub { font-size:11px; color:#5a6a7e; margin-bottom:18px; }
.detail-grid { display:grid; grid-template-columns:1fr 1fr; gap:10px 24px;
               margin-bottom:18px; }
.detail-row { display:flex; flex-direction:column; gap:2px; }
.detail-label { font-size:10px; color:#4a5a6e; text-transform:uppercase; letter-spacing:.05em; }
.detail-value { font-size:14px; font-weight:600; color:#e2e8f0; font-family:monospace; }
.detail-value.ok    { color:#1baf7a; }
.detail-value.amber { color:#eda100; }
.detail-value.red   { color:#e34948; }
.section-label { font-size:10px; color:#4a5a6e; text-transform:uppercase;
                 letter-spacing:.06em; margin-bottom:6px; }
.chroma-bar-wrap { display:flex; gap:2px; align-items:flex-end; height:40px; margin-bottom:12px; }
.chroma-bar { flex:1; background:#1baf7a; border-radius:2px 2px 0 0; min-height:2px; }
.note-labels { display:flex; gap:2px; margin-bottom:14px; }
.note-labels span { flex:1; text-align:center; font-size:8.5px; color:#4a5a6e; }
.prob-row { display:flex; align-items:center; gap:8px; margin-bottom:5px; }
.prob-label { width:90px; font-size:12px; font-family:monospace; color:#c8d0dc;
              white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.prob-bar-bg { flex:1; height:10px; background:#1a2535; border-radius:3px; overflow:hidden; }
.prob-bar-fill { height:100%; border-radius:3px; background:#3b8de0; }
.prob-bar-fill.top { background:#58d4ff; }
.prob-val { width:40px; text-align:right; font-size:11px; color:#5a6a7e; font-family:monospace; }
.chroma2d-wrap { position:relative; margin-bottom:4px; }
.chroma2d-wrap canvas { width:100%; height:80px; display:block; border-radius:3px;
                        image-rendering:pixelated; }
.chroma2d-yticks { position:absolute; left:-26px; top:0; height:80px;
                   display:flex; flex-direction:column; justify-content:space-between; }
.chroma2d-yticks span { font-size:7.5px; color:#4a5a6e; line-height:1; }
.chroma2d-outer { position:relative; margin-left:28px; margin-bottom:12px; }
</style>
</head><body>
<div class="page">
<h1>Blind inference superposed on ground truth</h1>
<p style="font-size:11px;color:#5a6a7e;margin-bottom:20px">
Each bar cell is split: <b style="color:#c8d0dc">top = GT chord</b> ·
<b style="color:#c8d0dc">bottom = inferred chord</b> coloured by correctness.
<b style="color:#888">Click any inferred chord cell</b> for full inference detail.
</p>
TMPL_SONGS
</div>

<!-- per-song hotspot registration (runs after all functions defined) -->
<script>
document.addEventListener("DOMContentLoaded", function() {
TMPL_REGISTER_CALLS
});
</script>

<!-- shared modal -->
<div id="modal-overlay">
  <div id="modal-box">
    <button id="modal-close" onclick="closeModal()">×</button>
    <div id="modal-content"></div>
  </div>
</div>

<script>
const NOTES = ["C","Db","D","Eb","E","F","F#","G","Ab","A","Bb","B"];
const FAMILIES = ["major","minor","diminished","augmented","suspended"];
const FAM_SUFFIX = {major:"^7", minor:"-7", diminished:"o7", augmented:"+", suspended:"sus"};
const COL_OK="#1baf7a", COL_ROOT="#eda100", COL_WRONG="#e34948", COL_NONE="#9b59b6";

function closeModal() {
  document.getElementById("modal-overlay").classList.remove("open");
}
document.getElementById("modal-overlay").addEventListener("click", function(e) {
  if (e.target === this) closeModal();
});
document.addEventListener("keydown", function(e) {
  if (e.key === "Escape") closeModal();
});

function chromaHeatmap(grid, highlightPcs) {
  // grid: array of 12 rows indexed by pitch class (0=C … 11=B),
  // each row is array of T values (0-1).
  // Canvas row 0 = top of image = PC 11 (B); canvas row 11 = bottom = PC 0 (C).
  const T = grid[0].length;
  const id = "chroma2d_" + Math.random().toString(36).slice(2);
  setTimeout(() => {
    const canvas = document.getElementById(id);
    if (!canvas) return;
    canvas.width  = T;
    canvas.height = 12;
    const ctx = canvas.getContext("2d");
    const img = ctx.createImageData(T, 12);
    const ramp = v => {
      const r = Math.round(Math.min(255, v * 2 * 255));
      const g = Math.round(Math.min(255, v * 1.5 * 255));
      const b = Math.round(Math.min(255, (0.4 + v * 0.6) * 255));
      return [r, g, b];
    };
    for (let canvasRow = 0; canvasRow < 12; canvasRow++) {
      const pc = 11 - canvasRow;   // canvas row 0 → PC 11 (B), row 11 → PC 0 (C)
      const isHL = highlightPcs && highlightPcs.includes(pc);
      for (let col = 0; col < T; col++) {
        const v = grid[pc][col];
        let [r, g, b] = ramp(v);
        if (isHL) { r = Math.min(255, r + 60); g = Math.min(255, g + 20); }
        const idx = (canvasRow * T + col) * 4;
        img.data[idx]   = r;
        img.data[idx+1] = g;
        img.data[idx+2] = b;
        img.data[idx+3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, 0);
  // Y-tick labels: PC 11 (B) at top → PC 0 (C) at bottom, matching canvas order
  const PC_NAMES = ["C","Db","D","Eb","E","F","F#","G","Ab","A","Bb","B"];
  const yticks = [11,10,9,8,7,6,5,4,3,2,1,0].map(pc =>
    `<span>${PC_NAMES[pc]}</span>`).join("");
  return `<div class="chroma2d-outer">
    <div class="chroma2d-yticks">${yticks}</div>
    <div class="chroma2d-wrap"><canvas id="${id}"></canvas></div>
  </div>`;
}

function chromaBars(arr, highlightPc, color) {
  let max = Math.max(...arr, 1e-9);
  let bars = arr.map((v, i) => {
    let h = Math.round(v / max * 38);
    let c = i === highlightPc ? color : "#253447";
    return `<div class="chroma-bar" style="height:${h}px;background:${c}"></div>`;
  }).join("");
  let labels = NOTES.map(n =>
    `<span>${n}</span>`
  ).join("");
  return `<div class="chroma-bar-wrap">${bars}</div>
          <div class="note-labels">${labels}</div>`;
}

function probBars(probs, labels, topIdx) {
  return probs.map((p, i) => {
    let pct = Math.round(p * 100);
    let isBest = i === topIdx;
    return `<div class="prob-row">
      <div class="prob-label">${labels[i]}</div>
      <div class="prob-bar-bg">
        <div class="prob-bar-fill${isBest?" top":""}" style="width:${pct}%"></div>
      </div>
      <div class="prob-val">${pct}%</div>
    </div>`;
  }).join("");
}

function openModal(h) {
  const ok = h.root_ok && h.fam_ok;
  const rootOk = h.root_ok;
  const statusClass = ok ? "ok" : (rootOk ? "amber" : "red");
  const statusText = ok ? "root + family correct" : (rootOk ? "root ✓  family ✗" : "root wrong");

  const infToken = h.inf_root + FAM_SUFFIX[h.inf_fam];
  const gtToken  = h.gt_ireal || (h.gt_root ? h.gt_root + (FAM_SUFFIX[h.gt_fam]||"") : "—");
  const col = ok ? COL_OK : (rootOk ? COL_ROOT : (h.gt_root ? COL_WRONG : COL_NONE));

  const rootLabels = NOTES.map((n,i) => n);
  const famLabels  = FAMILIES.map(f => f);

  let html = `
  <div class="modal-title" style="color:${col}">${infToken}</div>
  <div class="modal-sub">Bar ${h.bar} · beat ${h.beat} · ${h.t0}s – ${h.t1}s &nbsp;·&nbsp;
    <span class="${statusClass}">${statusText}</span></div>
  <div class="detail-grid">
    <div class="detail-row">
      <div class="detail-label">GT chord</div>
      <div class="detail-value">${gtToken}</div>
    </div>
    <div class="detail-row">
      <div class="detail-label">GT overlap</div>
      <div class="detail-value">${(h.gt_ov*100).toFixed(0)}%</div>
    </div>
    <div class="detail-row">
      <div class="detail-label">Graph pass</div>
      <div class="detail-value" style="color:${h.graph_pass_action==='unchanged'?'#5a6a7e':(h.graph_pass_action==='split'?'#e0a03b':'#3be0c0')}">${h.graph_pass_action}</div>
    </div>
    <div class="detail-row">
      <div class="detail-label">Inferred root</div>
      <div class="detail-value ${statusClass}">${h.inf_root}
        <span style="font-size:10px;color:#5a6a7e">(bass root: ${h.bass_root})</span>
      </div>
    </div>
    <div class="detail-row">
      <div class="detail-label">Inferred family</div>
      <div class="detail-value ${h.fam_ok?"ok":(h.root_ok?"amber":"")}">${h.inf_fam}
        <span style="font-size:10px;color:#5a6a7e">(${(h.fam_conf*100).toFixed(0)}% conf)</span>
      </div>
    </div>
  </div>

  ${(()=>{
    const rmTop = h.root_model_top;
    const bsTop = h.beat_seq_top;
    const agree = rmTop === bsTop;
    const agreeCol = agree ? "#1baf7a" : "#e34948";
    const agreeText = agree ? `✓ agree on ${rmTop}` : `✗ disagree: root-model=${rmTop}  beat-seq=${bsTop}`;
    const tmplCol = h.tmpl_score > 0.55 ? "#1baf7a" : (h.tmpl_score > 0.35 ? "#eda100" : "#e34948");
    return `<div style="display:flex;gap:12px;margin-bottom:14px;flex-wrap:wrap">
      <div style="background:#1a2535;border-radius:6px;padding:6px 12px;font-size:11px">
        <span style="color:#4a5a6e;text-transform:uppercase;letter-spacing:.05em">Template score </span>
        <span style="color:${tmplCol};font-weight:700;font-family:monospace">${(h.tmpl_score*100).toFixed(0)}%</span>
      </div>
      <div style="background:#1a2535;border-radius:6px;padding:6px 12px;font-size:11px">
        <span style="color:#4a5a6e;text-transform:uppercase;letter-spacing:.05em">Model agreement </span>
        <span style="color:${agreeCol};font-family:monospace">${agreeText}</span>
      </div>
      <div style="background:#1a2535;border-radius:6px;padding:6px 12px;font-size:11px">
        <span style="color:#4a5a6e;text-transform:uppercase;letter-spacing:.05em">Bass entropy </span>
        <span style="color:#c8d0dc;font-family:monospace">${h.bass_entropy.toFixed(2)} bit${h.bass_entropy < 0.8 ? ' (stable)' : h.bass_entropy > 2.0 ? ' (walking)' : ''}</span>
      </div>
    </div>`;
  })()}

  <div class="section-label">Top-5 template candidates</div>
  ${(()=>{
    const infKey = h.inf_root + "|" + h.inf_fam;
    return h.top5_templates.map(([root, fam, score], i) => {
      const key = root + "|" + fam;
      const isInf = key === infKey;
      const isGT  = h.gt_root && key === (h.gt_root + "|" + h.gt_fam);
      const pct   = Math.round(score * 100);
      const col   = isInf ? "#58d4ff" : (isGT ? "#1baf7a" : "#3b8de0");
      const label = `${root}${{major:"^7",minor:"-7",diminished:"o7",augmented:"+",suspended:"sus"}[fam]}${isInf?" ← inferred":""}${isGT?" ← GT":""}`;
      return `<div class="prob-row">
        <div class="prob-label" style="width:130px;color:${col}">${label}</div>
        <div class="prob-bar-bg"><div class="prob-bar-fill${isInf?" top":""}" style="width:${pct}%;background:${col}"></div></div>
        <div class="prob-val">${pct}%</div>
      </div>`;
    }).join("");
  })()}
  <br>

  <div class="section-label">Root model (segment-level, 60d) — root probabilities</div>
  ${chromaBars(h.root_probs, NOTES.indexOf(h.inf_root), col)}
  ${probBars(h.root_probs, rootLabels, NOTES.indexOf(h.inf_root))}

  <br>
  <div class="section-label">Beat-seq model (±2 beat window, 240d × n_beats pooled) — root vote</div>
  ${(()=>{const bsIdx=h.beat_seq_probs.indexOf(Math.max(...h.beat_seq_probs));
    return chromaBars(h.beat_seq_probs, bsIdx, "#e0a03b") +
           probBars(h.beat_seq_probs, rootLabels, bsIdx);
  })()}

  <br>
  <div class="section-label">Family classifier probabilities</div>
  ${probBars(h.fam_probs, famLabels, FAMILIES.indexOf(h.inf_fam))}

  <br>
  <div class="section-label">Chroma — CQT LTAS-normalised (time × pitch class) · B at top, C at bottom</div>
  ${(()=>{
    const topPc = h.chroma_audio.indexOf(Math.max(...h.chroma_audio));
    const bsIdx = h.beat_seq_probs.indexOf(Math.max(...h.beat_seq_probs));
    return chromaHeatmap(h.chroma_2d, [topPc, bsIdx, NOTES.indexOf(h.inf_root)]);
  })()}

  <div class="section-label">Chroma — Basic Pitch onset activations (all registers)</div>
  ${chromaBars(h.chroma_onset, NOTES.indexOf(h.inf_root), "#3b8de0")}

  <div class="section-label">Chroma — bass register (&lt;52)</div>
  ${chromaBars(h.chroma_bass, NOTES.indexOf(h.bass_root), "#a65fd4")}

  <div class="section-label">Chroma — treble register (&gt;60)</div>
  ${chromaBars(h.chroma_treble, NOTES.indexOf(h.inf_root), "#e07a3b")}
  `;

  document.getElementById("modal-content").innerHTML = html;
  document.getElementById("modal-overlay").classList.add("open");
}

// per-song hotspot registration
function registerHotspots(canvasId, hotspots) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  canvas.addEventListener("click", function(e) {
    const rect = canvas.getBoundingClientRect();
    const fx = (e.clientX - rect.left) / rect.width;
    const fy = (e.clientY - rect.top)  / rect.height;
    // canvas y-origin is top; figure y-coords are bottom-up (matplotlib fraction)
    for (const h of hotspots) {
      // convert matplotlib-fraction coords (y=0 at bottom) to image-fraction (y=0 at top)
      const iy = 1 - h.y - h.h;
      if (fx >= h.x && fx <= h.x + h.w && fy >= iy && fy <= iy + h.h) {
        openModal(h);
        return;
      }
    }
  });
  // highlight on hover
  canvas.addEventListener("mousemove", function(e) {
    const rect = canvas.getBoundingClientRect();
    const fx = (e.clientX - rect.left) / rect.width;
    const fy = (e.clientY - rect.top)  / rect.height;
    let hit = false;
    for (const h of hotspots) {
      const iy = 1 - h.y - h.h;
      if (fx >= h.x && fx <= h.x + h.w && fy >= iy && fy <= iy + h.h) {
        hit = true; break;
      }
    }
    canvas.style.cursor = hit ? "pointer" : "default";
  });
}
</script>
</body></html>"""

SONG_TMPL = """<div class="song-block">
  <div class="song-header">
    <h2>TMPL_TITLE</h2>
    <div class="stats">
      <b>TMPL_N_INF</b> segs / <b>TMPL_N_GT</b> GT &nbsp;·&nbsp;
      root <b>TMPL_ROOT_ACC</b> &nbsp;·&nbsp; fam <b>TMPL_FAM_ACC</b> &nbsp;·&nbsp;
      <b>TMPL_SCEN</b>
    </div>
  </div>
  <div class="panel">
    <div class="chart-wrap">
      <img src="data:image/png;base64,TMPL_B64">
      <canvas id="TMPL_CANVAS_ID"></canvas>
    </div>
  </div>
</div>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", nargs="*", default=None)
    ap.add_argument("--n-songs", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    recs = {json.loads(l)["song_id"]: json.loads(l) for l in open(DB)}
    man: dict = {}
    for mm in map(json.loads, open(MANIFEST)):
        if mm["song_id"] not in man or mm.get("transpose", 0) == 0:
            man[mm["song_id"]] = mm

    avail = sorted([s for s in recs if s in man], key=lambda s: recs[s]["title"])
    if args.songs:
        chosen = []
        for q in args.songs:
            hit = next((s for s in avail if q.lower() in recs[s]["title"].lower()), None)
            if hit: chosen.append(hit)
            else: print(f"  '{q}' not found")
    else:
        chosen = avail[:args.n_songs]

    d = np.load(FEAT, allow_pickle=True)
    ncl = {"fam": 5, "b7": len(BASE7), "ex": len(EXACT)}
    Xall = np.hstack([d["onset"], d["note"], d["bass"], d["treble"]])
    sc   = StandardScaler().fit(Xall)
    clf  = {lv: LogisticRegression(max_iter=500, solver="lbfgs").fit(
                sc.transform(Xall), d[k].astype(int))
            for lv, k in [("fam", "family"), ("b7", "base7"), ("ex", "exact")]}
    seg_model  = {k: v for k, v in np.load(
        REPO / "harmonia" / "models" / "root_model.npz",   allow_pickle=True).items()}
    beat_model = {k: v for k, v in np.load(
        REPO / "harmonia" / "models" / "beat_seq_model.npz", allow_pickle=True).items()}

    print(f"Rendering {len(chosen)} songs...")
    master_rng = np.random.default_rng(args.seed)
    seeds = master_rng.integers(0, 2**31, size=len(chosen)).tolist()

    song_blocks = []
    register_calls = []
    for sid, seed in zip(chosen, seeds):
        rec = recs[sid]; m = man[sid]
        print(f"  {rec['title']}...", end=" ", flush=True)
        rng = np.random.default_rng(seed)

        result, scen, sf_name = infer_blind(rec, m, sc, clf, ncl, rng, seg_model, beat_model)
        if result is None:
            print("SKIP"); continue

        img_bytes, hotspots = render_superposed(rec, m, result, bars_per_row=8)
        canvas_id = f"canvas_{sid.replace('-','_')}"

        block = SONG_TMPL
        block = block.replace("TMPL_TITLE",     rec["title"])
        block = block.replace("TMPL_N_INF",     str(result["n_inf"]))
        block = block.replace("TMPL_N_GT",      str(result["n_gt"]))
        block = block.replace("TMPL_ROOT_ACC",  f"{result['weighted_root']:.0%}")
        block = block.replace("TMPL_FAM_ACC",   f"{result['weighted_fam']:.0%}")
        block = block.replace("TMPL_SCEN",      f"{scen}/{sf_name}")
        block = block.replace("TMPL_B64",       _bytes_to_b64(img_bytes))
        block = block.replace("TMPL_CANVAS_ID", canvas_id)
        song_blocks.append(block)
        register_calls.append(f'  registerHotspots({json.dumps(canvas_id)}, {json.dumps(hotspots)});')
        gps = result.get('graph_pass_splits', 0)
        gpw = result.get('graph_pass_swaps', 0)
        print(f"root {result['weighted_root']:.0%}  fam {result['weighted_fam']:.0%}"
              f"  [graph: {gps} splits, {gpw} swaps]")

    html = HTML_TEMPLATE.replace("TMPL_SONGS", "\n".join(song_blocks))
    html = html.replace("TMPL_REGISTER_CALLS", "\n".join(register_calls))
    out = Path(args.out) if args.out else OUT_DIR / "blind_chart_comparison.html"
    out.write_text(html)
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
