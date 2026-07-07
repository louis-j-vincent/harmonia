"""Diagnostic: run one song through the full blind pipeline and dump an interactive
HTML showing every failure mode side-by-side.

Four panels per inferred chord segment:
  1. Chord detection  — did we find a boundary at the right place? (over/under-segment)
  2. Root detection   — was the bass-chroma root right?
  3. Quality          — was the family/seventh correct? (top-3 probabilities)
  4. Motif grouping   — which inferred motif group does this belong to? does the
                        GT chord actually match across group members?

Usage:
    .venv/bin/python scripts/debug_blind_pipeline.py [--song "Anthropology"] [--seed 42]
    open docs/plots/blind_debug_<slug>.html
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import tempfile
import warnings
from collections import defaultdict
from pathlib import Path

warnings.filterwarnings("ignore")
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

import librosa
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
    EXACT, EXACT_IDX, FAM_IDX, full_chroma, reg_chroma,
)
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig
from harmonia.models.motif import Chord as MChord, find_motifs
from harmonia.models.stage1_pitch import PitchExtractor
from learn_stage1_mapping import pool_beats

DB       = REPO / "data" / "accomp_db" / "db.jsonl"
MANIFEST = REPO / "data" / "accomp_db" / "audio" / "manifest.jsonl"
FEAT     = REPO / "data" / "cache" / "audio_chord_features.npz"
OUT_DIR  = REPO / "docs" / "plots"

FAMILIES = ["major", "minor", "diminished", "augmented", "suspended"]
NOTE     = ["C", "Db", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]
FAM_SHORT = {"major": "maj", "minor": "min", "diminished": "dim",
             "augmented": "aug", "suspended": "sus"}

# Chord templates for root_model feature (same as root_model_experiment.py)
_FAM_TONES = {"major": [0,4,7], "minor": [0,3,7], "diminished": [0,3,6],
              "augmented": [0,4,8], "suspended": [0,5,7]}
TEMPLATES = []
for _r in range(12):
    for _fam, _tones in _FAM_TONES.items():
        _t = np.zeros(12)
        for _off in _tones:
            _t[(_r + _off) % 12] = 1.0
        TEMPLATES.append((_r, _t / np.linalg.norm(_t)))


def _reg(v88, lo, hi):
    c = np.zeros(12)
    for k in range(88):
        if lo <= 21 + k < hi:
            c[(21 + k) % 12] += v88[k]
    return c


def _chroma88(v88, lo=0, hi=200):
    c = _reg(v88, lo, hi)
    n = np.linalg.norm(c)
    return c / n if n > 1e-9 else c


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _softmax(x):
    e = np.exp(x - x.max())
    return e / e.sum()


def _load_root_models():
    """Load the two trained root models. Returns (seg_model, beat_seq_model) dicts."""
    seg_d  = np.load(REPO / "harmonia" / "models" / "root_model.npz",     allow_pickle=True)
    beat_d = np.load(REPO / "harmonia" / "models" / "beat_seq_model.npz", allow_pickle=True)
    return dict(seg_d), dict(beat_d)


def _root_model_probs(seg_model: dict, son: np.ndarray, snt: np.ndarray) -> np.ndarray:
    """60d segment-level root model → 12d softmax probability vector."""
    oc   = _chroma88(son)
    tmpl = np.array([max(oc @ t for r2, t in TEMPLATES if r2 == r) for r in range(12)])
    feat = np.concatenate([
        oc,
        _chroma88(snt),
        _chroma88(son, 0, 52),
        _chroma88(son, 60, 200),
        tmpl,
    ])                                                    # 60d
    z = (feat - seg_model["mean"]) / (seg_model["scale"] + 1e-9)
    logits = seg_model["coef"] @ z + seg_model["intercept"]
    return _softmax(logits.astype(float))                 # 12d


def _beat_seq_probs(beat_model: dict,
                    onset_b: np.ndarray, note_b: np.ndarray,
                    beat_start: int, beat_end: int) -> np.ndarray:
    """Pool per-beat beat_seq_model probs over [beat_start, beat_end).
    beat_seq_model expects 48d per-beat windowed features (w=2 → 240d input).
    Returns pooled 12d softmax probability vector for the segment."""
    w    = int(beat_model["window"][0])
    n    = len(onset_b)
    probs = np.zeros(12)
    count = 0
    for b in range(beat_start, min(beat_end, n)):
        row = []
        for delta in range(-w, w + 1):
            bi = b + delta
            if 0 <= bi < n:
                f = np.concatenate([
                    _chroma88(onset_b[bi]),
                    _chroma88(note_b[bi]),
                    _chroma88(onset_b[bi], 0, 52),
                    _chroma88(onset_b[bi], 60, 200),
                ])
            else:
                f = np.zeros(48)
            row.append(f)
        feat = np.concatenate(row)                        # 240d
        z = (feat - beat_model["mean"]) / (beat_model["scale"] + 1e-9)
        logits = beat_model["coef"] @ z + beat_model["intercept"]
        probs += _softmax(logits.astype(float))
        count += 1
    if count == 0:
        return np.ones(12) / 12
    p = probs / count
    return p / p.sum()


def pool_to_beats(frame_times, probs, beat_times):
    n = len(beat_times)
    out = np.zeros((n, probs.shape[1]), dtype=np.float32)
    idx = np.searchsorted(beat_times, frame_times)
    for b, p in zip(idx, probs):
        if 0 <= b < n:
            out[b] += p
    return out


def build_gt_map(rec, man_entry):
    spb = 60.0 / man_entry["tempo"]
    bpb = man_entry["beats_per_bar"]
    nb  = man_entry["n_bars"] * bpb
    chord_at = {(e["bar"] - 1) * bpb + e["beat"]: e["mma"]
                for e in rec["chord_timeline"]}
    segs = []
    for t0, t1, root, _q in song_chord_spans(rec):
        b0 = int(round(t0 / spb))
        b1 = min(int(round(t1 / spb)), nb)
        mma = chord_at.get(b0)
        p = parse_chord(mma) if mma else None
        if p is None or p[1] not in BUCKET_FAMILY or b1 <= b0:
            continue
        segs.append({
            "t0": t0, "t1": t1,
            "root": root % 12,
            "fam": BUCKET_FAMILY[p[1]],
            "fam_i": FAM_IDX[BUCKET_FAMILY[p[1]]],
            "b7_i": BASE7_IDX[BUCKET_BASE7[p[1]]],
            "ex_i": EXACT_IDX[p[1]],
            "label": NOTE[root % 12] + FAM_SHORT[BUCKET_FAMILY[p[1]]],
            "label_full": NOTE[root % 12] + p[1],
            "bar": b0 // bpb,
        })
    return segs


def match_gt(t0, t1, gt_segs):
    """Return the GT segment with most overlap, or None."""
    best, best_ov = None, 0.0
    for g in gt_segs:
        ov = max(0, min(t1, g["t1"]) - max(t0, g["t0"]))
        if ov > best_ov:
            best_ov = ov
            best = g
    return best, best_ov


def motif_groups(chords):
    if not chords:
        return {}
    mc = [MChord(root=c["root"] % 12,
                 qual=FAMILIES[c["pred_fam"]],
                 label=str(c["root"]),
                 bar=c.get("bar", i))
          for i, c in enumerate(chords)]
    n_bars = max(c.bar for c in mc) + 1
    avg_cpb = max(1, round(len(mc) / n_bars))
    min_len = max(1, avg_cpb * 2)
    max_len = min(min_len * 4, 32)
    try:
        motifs = find_motifs(mc, shape=True, min_len=min_len, max_len=max_len, min_count=2)
    except Exception:
        return {}
    chord_motif = {}
    for m in motifs:
        if m.length < 2:
            continue
        for occ in m.occurrences:
            for k in range(m.length):
                idx = occ + k
                if idx < len(mc):
                    chord_motif[idx] = (str(m.key), k, m.display)
    groups = defaultdict(list)
    for i, mk in chord_motif.items():
        groups[f"{mk[0]}:{mk[1]}"].append((i, mk[2]))
    return {k: v for k, v in groups.items() if len(v) >= 2}


def run_pipeline(rec, man_entry, sc, clf, ncl, rng):
    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    ex = PitchExtractor(cache_dir=REPO / "data" / "cache" / "accomp_blind_dbg")
    seg_model, beat_model = _load_root_models()
    pm_base = pretty_midi.PrettyMIDI(str(REPO / man_entry["midi_path"]))

    scen = str(rng.choice(list(SCENARIOS)))
    gains = {k: v * float(rng.uniform(0.8, 1.2)) for k, v in SCENARIOS[scen].items()}
    sf_name = SOUNDFONTS[int(rng.integers(0, len(SOUNDFONTS)))]
    reverb = bool(rng.integers(0, 2))
    lead_prog = int(rng.choice(LEAD_PROGRAMS))

    stems = {
        "chords": stem_midi(pm_base, lambda i: (not i.is_drum) and "bass" not in i.name.lower()),
        "bass":   stem_midi(pm_base, lambda i: "bass" in i.name.lower()),
        "drums":  stem_midi(pm_base, lambda i: i.is_drum),
    }
    if gains.get("melody", 0) > 0.01:
        mel_pm = pretty_midi.PrettyMIDI()
        m = make_melody(pm_base, lead_prog, rng)
        if m:
            mel_pm.instruments.append(m)
            stems["melody"] = mel_pm

    waves, sr = {}, 44100
    for name, s in stems.items():
        if s and s.instruments:
            w, sr = render_to_array(renderer, s, sf_name, reverb)
            waves[name] = w

    L = max(len(w) for w in waves.values())
    mix = np.zeros(L, dtype=np.float32)
    for name, w in waves.items():
        mix[:len(w)] += gains.get(name, 0.5) * w
    mix = time_varying_degrade(mix, sr, rng)
    peak = np.abs(mix).max()
    if peak > 0.99:
        mix *= 0.99 / peak

    # --- also render bass-only for diagnostic
    bass_only = np.zeros(L, dtype=np.float32)
    if "bass" in waves:
        bass_only[:len(waves["bass"])] = waves["bass"]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
        tmp = Path(wf.name)
    try:
        sf.write(tmp, mix, sr)
        acts = ex.extract(tmp, use_cache=False)
    finally:
        tmp.unlink(missing_ok=True)

    _, beat_frames = librosa.beat.beat_track(y=mix, sr=sr,
                                             bpm=float(man_entry["tempo"]),
                                             units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    if len(beat_times) < 4:
        return None, None

    bpb = man_entry["beats_per_bar"]
    onset_b = pool_to_beats(acts.frame_times, acts.onset_probs, beat_times)
    note_b  = pool_to_beats(acts.frame_times, acts.note_probs,  beat_times)

    segs, run_on, run_nt, run_start = [], None, None, 0
    cell, nov_thresh = max(1, bpb // 2), 0.35
    for b in range(len(onset_b)):
        if onset_b[b].sum() < 1e-6:
            continue
        if run_on is None:
            run_on, run_nt, run_start = onset_b[b].copy(), note_b[b].copy(), b
            continue
        ref_ch   = _unit(_reg(run_on, 0, 200))
        beat_ch  = _unit(_reg(onset_b[b], 0, 200))
        ref_bass  = _unit(_reg(run_on, 0, 52))
        beat_bass = _unit(_reg(onset_b[b], 0, 52))
        novelty  = 1 - float(ref_ch @ beat_ch)
        bass_nov = 1 - float(ref_bass @ beat_bass)
        changed  = (b - run_start) >= cell and (novelty > nov_thresh or bass_nov > nov_thresh)
        if changed:
            segs.append((run_start, b, run_on.copy(), run_nt.copy()))
            run_on, run_nt, run_start = onset_b[b].copy(), note_b[b].copy(), b
        else:
            run_on += onset_b[b]; run_nt += note_b[b]
    if run_on is not None:
        segs.append((run_start, len(onset_b), run_on, run_nt))

    def _norm(v):
        s = v.sum() + 1e-9
        return [float(x / s) for x in v]

    chords = []
    source_chromas = []   # per-segment source breakdown
    for beat_start, beat_end, son, snt in segs:
        # --- all the individual evidence sources ---
        bass_on  = _reg(son, 0, 52)    # low bass, onset energy   ← current root source
        mid_on   = _reg(son, 52, 72)   # mid-range onset (chords, no kick drum, less melody)
        hi_on    = _reg(son, 60, 200)  # treble onset  (melody + chords, less kick)
        full_on  = _reg(son, 0, 200)   # full onset
        bass_nt  = _reg(snt, 0, 52)    # bass, sustained (note_probs — less attack, more pitch)
        mid_nt   = _reg(snt, 52, 72)   # mid sustained
        hi_nt    = _reg(snt, 60, 200)  # treble sustained
        full_nt  = _reg(snt, 0, 200)   # full sustained
        # pitch class from full_chroma() helper (rolls 88-key into 12 PCs)
        full_on_chroma  = full_chroma(son)
        full_nt_chroma  = full_chroma(snt)
        bass_on_chroma  = np.zeros(12)
        for k in range(88):
            if 21 + k < 52:
                bass_on_chroma[(21 + k) % 12] += son[k]

        # current root decision
        root = int(bass_on.argmax()) if bass_on.sum() > 1e-6 else int(full_on.argmax())
        rr = lambda c: np.roll(c, -root)
        feat = np.hstack([
            rr(full_chroma(son)), rr(full_chroma(snt)),
            rr(_reg(son, 0, 52)), rr(_reg(son, 60, 200)),
        ])
        t_start = float(beat_times[beat_start])
        t_end   = float(beat_times[min(beat_end, len(beat_times) - 1)])
        bar = beat_start // bpb

        seg_root_probs  = _root_model_probs(seg_model, son, snt)
        beat_root_probs = _beat_seq_probs(beat_model, onset_b, note_b, beat_start, beat_end)

        source_chromas.append({
            "bass_on":   _norm(bass_on),                # low bass onset — current root source
            "mid_on":    _norm(mid_on),                 # mid-range onset (chord tones)
            "hi_on":     _norm(hi_on),                  # treble onset
            "full_on":   _norm(full_on),                # all onset
            "bass_nt":   _norm(bass_nt),                # bass sustained
            "mid_nt":    _norm(mid_nt),                 # mid sustained
            "hi_nt":     _norm(hi_nt),                  # treble sustained
            "full_nt":   _norm(full_nt),                # all sustained
            "root_model":    [float(x) for x in seg_root_probs],   # segment-level LR (60d)
            "beat_seq":      [float(x) for x in beat_root_probs],  # windowed beat-seq LR (±2 beats)
        })
        chords.append({
            "root": root, "bar": bar, "t_start": t_start, "t_end": t_end, "feat": feat,
        })

    if not chords:
        return None, None

    X = sc.transform(np.stack([c["feat"] for c in chords]))
    prob = {}
    for lv in ("fam", "b7", "ex"):
        p = np.full((len(chords), ncl[lv]), 1e-9)
        p[:, clf[lv].classes_] = clf[lv].predict_proba(X)
        prob[lv] = p / p.sum(1, keepdims=True)

    for i, c in enumerate(chords):
        c["pred_fam"] = int(prob["fam"][i].argmax())

    gt_segs = build_gt_map(rec, man_entry)
    spb_gt = 60.0 / man_entry["tempo"]

    mgroups = motif_groups(chords)
    chord_to_group = {}
    for gkey, members in mgroups.items():
        for ci, display in members:
            chord_to_group[ci] = (gkey, display)

    # Assemble per-segment diagnostic records
    records = []
    for i, c in enumerate(chords):
        gt, gt_ov = match_gt(c["t_start"], c["t_end"], gt_segs)
        fam_probs = prob["fam"][i].tolist()
        b7_probs  = prob["b7"][i].tolist()

        pred_root_label = NOTE[c["root"]]
        pred_fam_label  = FAMILIES[int(prob["fam"][i].argmax())]
        pred_fam_short  = FAM_SHORT[pred_fam_label]

        # Top-3 family with probability
        fam_top3 = sorted(enumerate(fam_probs), key=lambda x: -x[1])[:3]

        # Top-3 base7
        b7_top3 = sorted(enumerate(b7_probs), key=lambda x: -x[1])[:3]

        # ground-truth root / fam
        gt_root_label = NOTE[gt["root"]] if gt else "?"
        gt_fam_label  = FAM_SHORT[gt["fam"]] if gt else "?"
        gt_full_label = gt["label_full"] if gt else "?"

        root_ok = gt and c["root"] == gt["root"]
        fam_ok  = gt and int(prob["fam"][i].argmax()) == gt["fam_i"]

        # Motif group info
        grp = chord_to_group.get(i)
        grp_key  = grp[0] if grp else None
        grp_name = grp[1] if grp else None

        # GT consistency within the motif group: do all members have same GT chord?
        grp_gt_consistent = None
        if grp_key:
            members = mgroups[grp_key]
            member_gt_fams = []
            for ci2, _ in members:
                gt2, _ = match_gt(chords[ci2]["t_start"], chords[ci2]["t_end"], gt_segs)
                if gt2:
                    member_gt_fams.append(gt2["fam_i"])
            if member_gt_fams:
                grp_gt_consistent = len(set(member_gt_fams)) == 1

        # per-source argmax roots (what each source would guess independently)
        sc_src = source_chromas[i]
        src_roots = {name: int(np.argmax(v)) for name, v in sc_src.items()}

        records.append({
            "i": i,
            "t_start": round(c["t_start"], 2),
            "t_end": round(c["t_end"], 2),
            "bar": c["bar"],
            "dur": round(c["t_end"] - c["t_start"], 2),
            # predicted
            "pred_root": c["root"],
            "pred_root_label": pred_root_label,
            "pred_fam": int(prob["fam"][i].argmax()),
            "pred_fam_label": pred_fam_short,
            "pred_label": pred_root_label + pred_fam_short,
            "fam_conf": round(float(prob["fam"][i].max()), 3),
            "fam_top3": [(FAMILIES[k], round(v, 3)) for k, v in fam_top3],
            "b7_top3": [(BASE7[k], round(v, 3)) for k, v in b7_top3],
            # per-source chromas (for modal detail view)
            "sources": sc_src,
            "src_roots": src_roots,
            # backward compat fields
            "bass_pc": sc_src["bass_on"],
            "full_pc": sc_src["full_on"],
            # ground truth
            "gt_root": gt["root"] if gt else None,
            "gt_root_label": gt_root_label,
            "gt_fam_label": gt_fam_label,
            "gt_label": gt_full_label,
            "gt_ov": round(gt_ov, 3),
            # error flags
            "root_ok": root_ok,
            "fam_ok": fam_ok,
            "seg_ok": gt_ov > 0,
            # motif
            "grp_key": grp_key,
            "grp_name": grp_name,
            "grp_gt_consistent": grp_gt_consistent,
        })

    meta = {
        "title": rec["title"],
        "key": rec.get("key", "?"),
        "scenario": scen,
        "soundfont": sf_name,
        "reverb": reverb,
        "gains": {k: round(v, 2) for k, v in gains.items()},
        "gt_count": len(gt_segs),
        "inferred_count": len(chords),
        "n_motif_groups": len(mgroups),
        "root_acc": round(sum(r["root_ok"] for r in records if r["seg_ok"]) /
                          max(1, sum(r["seg_ok"] for r in records)), 3),
        "fam_acc": round(sum(r["fam_ok"] for r in records if r["seg_ok"]) /
                         max(1, sum(r["seg_ok"] for r in records)), 3),
    }
    return records, meta


HTML = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Blind pipeline debug — TMPL_TITLE</title>
<style>
* { box-sizing:border-box; margin:0; }
body { background:#0a0e14; color:#c8d0dc; font-family:system-ui,monospace; font-size:13px; }
.page { max-width:1440px; margin:0 auto; padding:20px 24px 60px; }
h1 { font-size:20px; color:#e2e8f0; margin-bottom:4px; }
.sub { color:#5a6a7e; font-size:12px; margin-bottom:16px; }
.meta-bar { display:flex; flex-wrap:wrap; gap:10px; margin-bottom:18px;
            background:#111820; border:1px solid #1e2c3a; border-radius:8px; padding:10px 14px; }
.meta-bar span { color:#58d4ff; font-weight:600; }
.meta-bar label { color:#5a6a7e; font-size:11px; }

/* filter strip */
.filters { display:flex; gap:10px; flex-wrap:wrap; margin-bottom:14px; align-items:center; }
.filters button { background:#111820; border:1px solid #1e2c3a; color:#8899aa;
                  padding:4px 12px; border-radius:20px; cursor:pointer; font-size:12px; }
.filters button.active, .filters button:hover { background:#162030; border-color:#58d4ff; color:#58d4ff; }
.filters label { color:#5a6a7e; font-size:12px; margin-left:8px; }

/* timeline */
.timeline { position:relative; height:32px; background:#111820; border-radius:6px;
            margin-bottom:18px; overflow:hidden; }
.tl-gt, .tl-inf { position:absolute; height:14px; border-radius:3px; opacity:.85; }
.tl-gt  { top:1px;  background:#1baf7a88; border:1px solid #1baf7a; }
.tl-inf { top:17px; border:1px solid #58d4ff; background:#58d4ff33; }

/* grid */
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(270px,1fr)); gap:8px; }
.card { background:#111820; border:1px solid #1e2c3a; border-radius:8px;
        padding:10px 12px; position:relative; cursor:pointer;
        transition:border-color .15s, box-shadow .15s; }
.card:hover { border-color:#58d4ff55; box-shadow:0 0 10px #58d4ff22; }
.card.hi    { border-color:#58d4ff; box-shadow:0 0 14px #58d4ff44; }
.card.err-root  { border-left:3px solid #e34948; }
.card.err-fam   { border-left:3px solid #eda100; }
.card.ok        { border-left:3px solid #1baf7a; }
.card.no-match  { border-left:3px solid #4a3aa7; opacity:.6; }

.card-head { display:flex; justify-content:space-between; align-items:baseline; margin-bottom:6px; }
.pred-label { font-size:18px; font-weight:700; color:#e2e8f0; letter-spacing:.5px; }
.gt-label   { font-size:12px; color:#1baf7a; }
.gt-label.wrong { color:#e34948; }
.conf-bar { height:4px; background:#1e2c3a; border-radius:2px; margin-bottom:6px; }
.conf-fill { height:100%; border-radius:2px; background:linear-gradient(90deg,#58d4ff,#1baf7a); }

.row { display:flex; justify-content:space-between; font-size:11px; color:#5a6a7e; margin-bottom:3px; }
.row .val { color:#c8d0dc; }
.row .val.ok  { color:#1baf7a; }
.row .val.bad { color:#e34948; }
.row .val.warn { color:#eda100; }

/* bass chroma mini-bar */
.chroma-wrap { display:flex; gap:1px; height:24px; margin:6px 0 4px; align-items:flex-end; }
.chroma-bar  { flex:1; border-radius:1px 1px 0 0; min-width:0; }
.note-label  { flex:1; text-align:center; font-size:8px; color:#3a4a5a; margin-top:1px; }

/* prob grid */
.prob-row { display:flex; gap:4px; margin-top:4px; flex-wrap:wrap; }
.prob-chip { background:#162030; border-radius:4px; padding:2px 6px;
             font-size:11px; color:#8899aa; white-space:nowrap; }
.prob-chip.top { color:#58d4ff; border:1px solid #58d4ff44; }
.prob-chip.correct { color:#1baf7a; border:1px solid #1baf7a44; }

/* motif badge */
.motif-badge { position:absolute; top:8px; right:10px;
               background:#162030; border:1px solid #2a3a50;
               border-radius:12px; padding:1px 8px; font-size:10px; color:#8899aa; }
.motif-badge.inconsistent { border-color:#e34948; color:#e34948; }
.motif-badge.consistent   { border-color:#1baf7a; color:#1baf7a; }

/* error reason tag */
.err-tag { display:inline-block; margin-top:5px; font-size:10px; padding:1px 7px;
           border-radius:10px; background:#1e2c3a; color:#eda100; }

/* ── modal ── */
.modal-overlay { display:none; position:fixed; inset:0; background:#000a; z-index:100;
                 align-items:center; justify-content:center; }
.modal-overlay.open { display:flex; }
.modal { background:#0d1520; border:1px solid #1e3a50; border-radius:12px;
         width:min(96vw,900px); max-height:90vh; overflow-y:auto;
         padding:22px 26px; position:relative; box-shadow:0 0 40px #000c; }
.modal-close { position:absolute; top:14px; right:18px; cursor:pointer;
               color:#5a6a7e; font-size:18px; line-height:1; }
.modal-close:hover { color:#e2e8f0; }
.modal h2 { font-size:18px; color:#e2e8f0; margin-bottom:4px; }
.modal .sub2 { color:#5a6a7e; font-size:11px; margin-bottom:18px; }

.src-grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(180px,1fr)); gap:12px; }
.src-block { background:#111820; border:1px solid #1e2c3a; border-radius:8px; padding:10px 12px; }
.src-block h4 { font-size:11px; color:#8899aa; margin-bottom:6px; letter-spacing:.5px; text-transform:uppercase; }
.src-block .argmax { font-size:13px; font-weight:700; margin-bottom:6px; }
.src-block .argmax.correct { color:#1baf7a; }
.src-block .argmax.wrong   { color:#e34948; }
.src-block .argmax.partial { color:#eda100; }

.bars-wrap { display:flex; gap:2px; height:40px; align-items:flex-end; }
.bar-pc { flex:1; border-radius:1px 1px 0 0; min-width:0; }
.bar-label-row { display:flex; gap:2px; margin-top:2px; }
.bar-pc-lbl { flex:1; text-align:center; font-size:7px; color:#3a4a5a; min-width:0; }

.section-title { font-size:12px; color:#58d4ff; font-weight:600;
                 margin:16px 0 8px; padding-bottom:4px; border-bottom:1px solid #1e2c3a; }

.root-vote { display:flex; flex-wrap:wrap; gap:6px; margin-top:4px; }
.vote-chip { padding:3px 10px; border-radius:12px; font-size:11px; background:#111820;
             border:1px solid #1e2c3a; }
.vote-chip.correct { border-color:#1baf7a; color:#1baf7a; }
.vote-chip.wrong   { border-color:#e34948; color:#e34948; }
</style>
</head><body>
<div class="page">
<h1>Blind pipeline — TMPL_TITLE</h1>
<div class="sub">key TMPL_KEY · scenario: TMPL_SCENARIO · soundfont: TMPL_SF · reverb: TMPL_REVERB</div>
<div class="meta-bar">
  <div><label>GT chords</label><br><span>TMPL_GT_COUNT</span></div>
  <div><label>Inferred segs</label><br><span>TMPL_INF_COUNT</span></div>
  <div><label>Root acc</label><br><span>TMPL_ROOT_ACC</span></div>
  <div><label>Family acc</label><br><span>TMPL_FAM_ACC</span></div>
  <div><label>Motif groups</label><br><span>TMPL_N_MOTIF_GROUPS</span></div>
  <div><label>Gains</label><br><span>TMPL_GAINS</span></div>
</div>

<div class="filters">
  <b style="color:#5a6a7e;font-size:12px">Show:</b>
  <button class="active" onclick="filter('all')">All</button>
  <button onclick="filter('root_err')">Root errors</button>
  <button onclick="filter('fam_err')">Family errors</button>
  <button onclick="filter('no_match')">No GT match</button>
  <button onclick="filter('in_motif')">In motif group</button>
  <button onclick="filter('inconsistent')">Inconsistent motif</button>
  <span style="margin-left:auto;color:#5a6a7e">hover card → highlight group</span>
</div>

<div class="timeline" id="tl"></div>
<div class="grid" id="grid"></div>
</div>

<!-- detail modal -->
<div class="modal-overlay" id="overlay" onclick="closeModal(event)">
  <div class="modal" id="modal">
    <span class="modal-close" onclick="document.getElementById('overlay').classList.remove('open')">✕</span>
    <div id="modal-body"></div>
  </div>
</div>

<script>
const DATA = TMPL_JSON_DATA;
const NOTE = ['C','Db','D','Eb','E','F','F#','G','Ab','A','Bb','B'];

function buildTimeline() {
  const tl = document.getElementById('tl');
  const W = tl.offsetWidth || 1200;
  const maxT = Math.max(...DATA.records.map(r => r.t_end), 1);

  // GT blocks
  const gtSeen = new Map();
  DATA.records.forEach(r => {
    if (!r.seg_ok || gtSeen.has(r.gt_label + r.t_start)) return;
    gtSeen.set(r.gt_label + r.t_start, true);
    const d = document.createElement('div');
    d.className = 'tl-gt';
    d.style.left  = (r.t_start / maxT * 100) + '%';
    d.style.width = ((r.t_end - r.t_start) / maxT * 100) + '%';
    d.title = r.gt_label;
    tl.appendChild(d);
  });

  // Inferred blocks
  DATA.records.forEach(r => {
    const d = document.createElement('div');
    d.className = 'tl-inf';
    d.style.left  = (r.t_start / maxT * 100) + '%';
    d.style.width = ((r.t_end - r.t_start) / maxT * 100) + '%';
    d.style.background = r.fam_ok ? '#58d4ff55' : r.root_ok ? '#eda10055' : '#e3494855';
    d.style.borderColor = r.fam_ok ? '#58d4ff' : r.root_ok ? '#eda100' : '#e34948';
    d.title = r.pred_label + ' (GT: ' + r.gt_label + ')';
    tl.appendChild(d);
  });
}

function renderGrid(records) {
  const grid = document.getElementById('grid');
  grid.innerHTML = '';
  records.forEach(r => {
    const card = document.createElement('div');
    let cls = 'card';
    if (!r.seg_ok) cls += ' no-match';
    else if (!r.root_ok) cls += ' err-root';
    else if (!r.fam_ok) cls += ' err-fam';
    else cls += ' ok';
    card.className = cls;
    card.dataset.grp = r.grp_key || '';

    // motif badge
    if (r.grp_key) {
      const b = document.createElement('div');
      b.className = 'motif-badge' + (r.grp_gt_consistent === false ? ' inconsistent' : r.grp_gt_consistent === true ? ' consistent' : '');
      b.textContent = r.grp_name || r.grp_key;
      card.appendChild(b);
    }

    const gtWrong = r.seg_ok && (r.pred_label !== r.gt_label.replace('dom7','7').replace('maj7','^7').replace('min7','m7'));
    const err_reason = !r.seg_ok ? 'no GT match' :
                       !r.root_ok ? `root: got ${r.pred_root_label}, GT ${r.gt_root_label}` :
                       !r.fam_ok  ? `qual: got ${r.pred_fam_label}, GT ${r.gt_fam_label}` : '';

    // Build bass chroma bars
    const bassMax = Math.max(...r.bass_pc, 1e-6);
    const chromaBars = r.bass_pc.map((v, k) => {
      const h = Math.round(v / bassMax * 22);
      const isRoot = k === r.pred_root;
      const isGtRoot = k === r.gt_root;
      const col = isRoot ? '#58d4ff' : isGtRoot ? '#1baf7a' : '#1e2c3a';
      return `<div class="chroma-bar" style="height:${h}px;background:${col}" title="${NOTE[k]}:${Math.round(v*100)}%"></div>`;
    }).join('');
    const noteLabels = NOTE.map((n, k) => {
      const col = k === r.pred_root ? '#58d4ff' : k === r.gt_root ? '#1baf7a' : '#3a4a5a';
      return `<div class="note-label" style="color:${col}">${n}</div>`;
    }).join('');

    // Top-3 fam probs
    const famChips = r.fam_top3.map(([fam, p], ii) => {
      const isCorrect = r.gt_fam_label && fam.startsWith(r.gt_fam_label.replace('maj','major').replace('min','minor'));
      const cls2 = ii === 0 ? 'prob-chip top' : isCorrect ? 'prob-chip correct' : 'prob-chip';
      return `<span class="${cls2}">${fam.slice(0,3)} ${(p*100).toFixed(0)}%</span>`;
    }).join('');

    card.innerHTML += `
      <div class="card-head">
        <span class="pred-label">${r.pred_label}</span>
        <span class="gt-label${!r.fam_ok && r.seg_ok ? ' wrong' : ''}">${r.seg_ok ? 'GT: '+r.gt_label : '—'}</span>
      </div>
      <div class="conf-bar"><div class="conf-fill" style="width:${r.fam_conf*100}%"></div></div>
      <div class="row"><span>t</span><span class="val">${r.t_start}s – ${r.t_end}s (${r.dur}s)</span></div>
      <div class="row"><span>bar</span><span class="val">${r.bar}</span></div>
      <div class="row"><span>root</span>
        <span class="val ${r.root_ok ? 'ok' : 'bad'}">${r.pred_root_label} ${r.root_ok ? '✓' : '✗ GT:'+r.gt_root_label}</span>
      </div>
      <div class="chroma-wrap">${chromaBars}</div>
      <div class="chroma-wrap">${noteLabels}</div>
      <div class="row"><span>family conf</span><span class="val">${(r.fam_conf*100).toFixed(0)}%</span></div>
      <div class="prob-row">${famChips}</div>
      ${err_reason ? '<div class="err-tag">⚠ ' + err_reason + '</div>' : ''}
    `;

    card.addEventListener('mouseenter', () => {
      if (!r.grp_key) return;
      document.querySelectorAll('[data-grp="' + r.grp_key + '"]').forEach(c => c.classList.add('hi'));
    });
    card.addEventListener('mouseleave', () => {
      document.querySelectorAll('.card.hi').forEach(c => c.classList.remove('hi'));
    });
    card.addEventListener('click', () => openModal(r));

    grid.appendChild(card);
  });
}

// ── modal ──────────────────────────────────────────────────────────────────
const SRC_META = {
  bass_on:    { label: 'Bass onset',      desc: 'Low register (<52 MIDI), attack energy  ← current root source (bass argmax)' },
  mid_on:     { label: 'Mid onset',       desc: 'Mid register (52–72 MIDI), onset — chord tones, less kick & melody' },
  hi_on:      { label: 'Treble onset',    desc: 'High register (60–200 MIDI), onset — melody + chords' },
  full_on:    { label: 'Full onset',      desc: 'All registers, onset energy' },
  bass_nt:    { label: 'Bass sustain',    desc: 'Low register, sustained pitch (less attack bias)' },
  mid_nt:     { label: 'Mid sustain',     desc: 'Mid register, sustained — most stable chord-tone signal' },
  hi_nt:      { label: 'Treble sustain',  desc: 'High register, sustained' },
  full_nt:    { label: 'Full sustain',    desc: 'All registers, sustained pitch' },
  root_model: { label: 'Root model (seg)',desc: 'Segment-level LR trained on 60d absolute chroma + template scores — root acc ~85% on oracle segs' },
  beat_seq:   { label: 'Beat-seq (±2)',   desc: 'Beat-sequence windowed LR (±2 beats, 240d) — root acc ~88.9% per beat; pooled over segment beats here' },
};

function chromaBars(vals, gtRoot, predRoot) {
  const mx = Math.max(...vals, 1e-6);
  const bars = vals.map((v, k) => {
    const h = Math.round(v / mx * 38);
    const col = k === predRoot ? '#58d4ff' : k === gtRoot ? '#1baf7a' : '#1e3a50';
    return `<div class="bar-pc" style="height:${h}px;background:${col}" title="${NOTE[k]}: ${(v*100).toFixed(1)}%"></div>`;
  }).join('');
  const lbls = NOTE.map((n, k) => {
    const col = k === predRoot ? '#58d4ff' : k === gtRoot ? '#1baf7a88' : '#2a3a4a';
    return `<div class="bar-pc-lbl" style="color:${col}">${n}</div>`;
  }).join('');
  return `<div class="bars-wrap">${bars}</div><div class="bar-label-row">${lbls}</div>`;
}

function openModal(r) {
  const gtRoot = r.gt_root;
  const predRoot = r.pred_root;

  // Which sources would vote for the correct root?
  const votes = Object.entries(r.src_roots).map(([name, root]) => {
    const correct = root === gtRoot;
    const cls = correct ? 'vote-chip correct' : 'vote-chip wrong';
    return `<span class="${cls}" title="${SRC_META[name]?.label}">${(SRC_META[name]?.label || name).replace(' onset','·on').replace(' sustain','·sus')}: ${NOTE[root]}</span>`;
  }).join('');

  // Source grids
  const srcBlocks = Object.entries(r.sources).map(([name, vals]) => {
    const argmax = r.src_roots[name];
    const correct = argmax === gtRoot;
    const cls = correct ? 'correct' : argmax === predRoot ? 'wrong' : 'partial';
    const note = NOTE[argmax];
    const m = SRC_META[name] || {label: name, desc: ''};
    return `<div class="src-block">
      <h4>${m.label}</h4>
      <div class="argmax ${cls}">${note}${correct ? ' ✓' : ' ✗'} <span style="font-size:10px;font-weight:400;color:#5a6a7e">${(vals[argmax]*100).toFixed(0)}%</span></div>
      ${chromaBars(vals, gtRoot, predRoot)}
      <div style="font-size:10px;color:#3a4a5a;margin-top:5px">${m.desc}</div>
    </div>`;
  }).join('');

  // Top family probs
  const famRows = r.fam_top3.map(([fam, p], ii) => {
    const col = ii === 0 ? '#58d4ff' : '#5a6a7e';
    return `<span style="color:${col}">${fam} ${(p*100).toFixed(0)}%</span>`;
  }).join(' &nbsp;·&nbsp; ');

  // b7 top3
  const b7Rows = r.b7_top3.map(([b7, p], ii) => {
    const col = ii === 0 ? '#58d4ff' : '#5a6a7e';
    return `<span style="color:${col}">${b7} ${(p*100).toFixed(0)}%</span>`;
  }).join(' &nbsp;·&nbsp; ');

  const motifLine = r.grp_key ? `
    <div class="row"><span>Motif group</span><span class="val ${r.grp_gt_consistent === false ? 'bad' : r.grp_gt_consistent ? 'ok' : ''}">${r.grp_name || r.grp_key} — GT ${r.grp_gt_consistent === false ? 'inconsistent ✗' : r.grp_gt_consistent ? 'consistent ✓' : 'unknown'}</span></div>` : '';

  document.getElementById('modal-body').innerHTML = `
    <h2>${r.pred_label} <span style="font-size:13px;color:#5a6a7e">seg ${r.i} · bar ${r.bar} · ${r.t_start}s–${r.t_end}s (${r.dur}s)</span></h2>
    <div class="sub2">GT: <b style="color:${r.root_ok && r.fam_ok ? '#1baf7a' : '#e34948'}">${r.seg_ok ? r.gt_label : '— no GT match'}</b>
      &nbsp;·&nbsp; root ${r.root_ok ? '✓' : '✗ (pred '+r.pred_root_label+', GT '+r.gt_root_label+')'}
      &nbsp;·&nbsp; family ${r.fam_ok ? '✓' : '✗ (pred '+r.pred_fam_label+', GT '+r.gt_fam_label+')'}
    </div>

    <div class="section-title">Root vote by source</div>
    <div style="font-size:11px;color:#5a6a7e;margin-bottom:8px">
      cyan bar = inferred root (${r.pred_root_label}) · green bar = GT root (${r.gt_root_label || '?'})
    </div>
    <div class="root-vote">${votes}</div>

    <div class="section-title">Pitch-class distribution per source</div>
    <div class="src-grid">${srcBlocks}</div>

    <div class="section-title">Classifier output</div>
    <div class="row" style="margin-bottom:6px"><span>Family (top 3)</span><span>${famRows}</span></div>
    <div class="row"><span>Seventh (top 3)</span><span>${b7Rows}</span></div>
    ${motifLine}
  `;
  document.getElementById('overlay').classList.add('open');
}

function closeModal(e) {
  if (e.target === document.getElementById('overlay'))
    document.getElementById('overlay').classList.remove('open');
}
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') document.getElementById('overlay').classList.remove('open');
});

let currentFilter = 'all';
function filter(f) {
  currentFilter = f;
  document.querySelectorAll('.filters button').forEach(b => b.classList.remove('active'));
  event.target.classList.add('active');
  let recs = DATA.records;
  if (f === 'root_err')     recs = recs.filter(r => r.seg_ok && !r.root_ok);
  if (f === 'fam_err')      recs = recs.filter(r => r.seg_ok && r.root_ok && !r.fam_ok);
  if (f === 'no_match')     recs = recs.filter(r => !r.seg_ok);
  if (f === 'in_motif')     recs = recs.filter(r => r.grp_key);
  if (f === 'inconsistent') recs = recs.filter(r => r.grp_gt_consistent === false);
  renderGrid(recs);
}

buildTimeline();
renderGrid(DATA.records);
</script>
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--song", default="Anthropology")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    recs = {json.loads(l)["song_id"]: json.loads(l) for l in open(DB)}
    man: dict = {}
    for mm in map(json.loads, open(MANIFEST)):
        if mm["song_id"] not in man or mm.get("transpose", 0) == 0:
            man[mm["song_id"]] = mm

    sid = next((s for s in recs if args.song.lower() in recs[s]["title"].lower()), None)
    if sid is None:
        print(f"Song '{args.song}' not found"); sys.exit(1)
    rec, m = recs[sid], man[sid]

    d = np.load(FEAT, allow_pickle=True)
    ncl = {"fam": 5, "b7": len(BASE7), "ex": len(EXACT)}
    Xall = np.hstack([d["onset"], d["note"], d["bass"], d["treble"]])
    sc   = StandardScaler().fit(Xall)
    clf  = {lv: LogisticRegression(max_iter=500, solver="lbfgs").fit(sc.transform(Xall), d[k].astype(int))
            for lv, k in [("fam", "family"), ("b7", "base7"), ("ex", "exact")]}

    print(f"Running blind pipeline on '{rec['title']}' (seed={args.seed})...")
    rng = np.random.default_rng(args.seed)
    records, meta = run_pipeline(rec, m, sc, clf, ncl, rng)
    if not records:
        print("No results"); sys.exit(1)

    print(f"  {meta['inferred_count']} inferred segments, {meta['gt_count']} GT")
    print(f"  root acc: {meta['root_acc']:.0%}  family acc: {meta['fam_acc']:.0%}")
    print(f"  motif groups: {meta['n_motif_groups']}")
    root_errs = sum(1 for r in records if r["seg_ok"] and not r["root_ok"])
    fam_errs  = sum(1 for r in records if r["seg_ok"] and r["root_ok"] and not r["fam_ok"])
    incon = sum(1 for r in records if r["grp_gt_consistent"] is False)
    print(f"  root errors: {root_errs}  family errors: {fam_errs}  inconsistent motif groups: {incon}")

    gains_str = " ".join(f"{k[0].upper()}:{v:.2f}" for k, v in meta["gains"].items())
    replacements = {
        "TMPL_TITLE": meta["title"], "TMPL_KEY": meta["key"],
        "TMPL_SCENARIO": meta["scenario"], "TMPL_SF": meta["soundfont"],
        "TMPL_REVERB": str(meta["reverb"]), "TMPL_GAINS": gains_str,
        "TMPL_GT_COUNT": str(meta["gt_count"]), "TMPL_INF_COUNT": str(meta["inferred_count"]),
        "TMPL_ROOT_ACC": f"{meta['root_acc']:.0%}", "TMPL_FAM_ACC": f"{meta['fam_acc']:.0%}",
        "TMPL_N_MOTIF_GROUPS": str(meta["n_motif_groups"]),
        "TMPL_JSON_DATA": json.dumps({"records": records, "meta": meta}),
    }
    html = HTML
    for k, v in replacements.items():
        html = html.replace(k, v)

    slug = re.sub(r"[^a-z0-9]+", "_", rec["title"].lower()).strip("_")
    out  = OUT_DIR / f"blind_debug_{slug}.html"
    out.write_text(html)
    print(f"\n→ {out}")


if __name__ == "__main__":
    main()
