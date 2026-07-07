"""Blind motif fold accuracy experiment — the honest test.

Conditions (no GT helpers of any kind):
  1. blind audio   — infer beats, chord boundaries, root, and quality from audio alone
  2. blind + motif — apply certainty-weighted motif-fold on the inferred chord stream

Reference: GT chord quality (family / seventh / exact) from the chart,
matched to inferred segments by weighted time-overlap (same as MIREX scoring,
but at all three quality levels).

Audio: MIDI rendered to WAV, then degraded with time_varying_degrade
       (time-varying pink noise, SNR 3–20 dB per block — "phone capture" regime).

Usage:
    .venv/bin/python scripts/eval_blind_motif_accuracy.py
    .venv/bin/python scripts/eval_blind_motif_accuracy.py --n-songs 20  # quick
    .venv/bin/python scripts/eval_blind_motif_accuracy.py --snr-db 8    # fixed SNR
"""
from __future__ import annotations

import argparse
import copy
import json
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
import soundfile as sf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import pretty_midi
from analyze_accomp_emission import parse_chord, song_chord_spans
from build_accomp_audio_hard import (
    make_melody, pink, render_to_array, stem_midi, time_varying_degrade,
    SCENARIOS, SOUNDFONTS, LEAD_PROGRAMS,
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

FAMILIES = ["major", "minor", "diminished", "augmented", "suspended"]
NOTE     = ["C", "C#", "D", "Eb", "E", "F", "F#", "G", "Ab", "A", "Bb", "B"]


def pool_to_beats(frame_times: np.ndarray, probs: np.ndarray,
                  beat_times: np.ndarray) -> np.ndarray:
    """Pool per-frame BP activations into per-beat bins."""
    n_beats = len(beat_times)
    result = np.zeros((n_beats, probs.shape[1]), dtype=np.float32)
    beat_idx = np.searchsorted(beat_times, frame_times)
    for b, p in zip(beat_idx, probs):
        if 0 <= b < n_beats:
            result[b] += p
    return result


def _reg(v88: np.ndarray, lo: int, hi: int) -> np.ndarray:
    c = np.zeros(12)
    for k in range(88):
        if lo <= 21 + k < hi:
            c[(21 + k) % 12] += v88[k]
    return c


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def _build_gt_quality_map(rec: dict, man_entry: dict) -> dict:
    """Map beat-index → (root, gt_fam, gt_b7, gt_ex, t_start, t_end)."""
    spb = 60.0 / man_entry["tempo"]
    bpb = man_entry["beats_per_bar"]
    nb  = man_entry["n_bars"] * bpb
    chord_at = {(e["bar"] - 1) * bpb + e["beat"]: e["mma"] for e in rec["chord_timeline"]}
    gt = {}
    for t0, t1, root, _q in song_chord_spans(rec):
        b0 = int(round(t0 / spb))
        b1 = min(int(round(t1 / spb)), nb)
        mma = chord_at.get(b0)
        p = parse_chord(mma) if mma else None
        if p is None or p[1] not in BUCKET_FAMILY or b1 <= b0:
            continue
        for b in range(b0, b1):
            gt[b] = {
                "root": root % 12,
                "gt_fam": FAM_IDX[BUCKET_FAMILY[p[1]]],
                "gt_b7":  BASE7_IDX[BUCKET_BASE7[p[1]]],
                "gt_ex":  EXACT_IDX[p[1]],
                "t0": t0, "t1": t1,
            }
    return gt


def _motif_groups(chords: list[dict]) -> dict[str, list[int]]:
    if not chords:
        return {}
    mc = [MChord(root=c["root"] % 12,
                 qual=FAMILIES[c["pred_fam"]],
                 label=str(c["root"]),
                 bar=c.get("bar", i)) for i, c in enumerate(chords)]
    n_bars = max(c.bar for c in mc) + 1
    avg_cpb = max(1, round(len(mc) / n_bars))
    min_len = max(1, avg_cpb * 2)
    max_len = min(min_len * 4, 32)
    try:
        motifs = find_motifs(mc, shape=True, min_len=min_len, max_len=max_len, min_count=2)
    except Exception:
        return {}
    chord_motif: dict[int, str] = {}
    for m in motifs:
        if m.length < 2:
            continue
        for occ_start in m.occurrences:
            for k in range(m.length):
                idx = occ_start + k
                if idx < len(mc):
                    chord_motif[idx] = f"{m.key}:{k}"
    groups: dict[str, list[int]] = defaultdict(list)
    for i, mk in chord_motif.items():
        groups[mk].append(i)
    return {k: v for k, v in groups.items() if len(v) >= 2}


def _vote(prob: dict, groups: dict[str, list[int]]) -> None:
    for lv in ("fam", "b7", "ex"):
        cert = prob[lv].max(1)
        for g in groups.values():
            g_arr = np.array(g)
            w = cert[g_arr] / (cert[g_arr].sum() + 1e-9)
            prob[lv][g_arr] = (prob[lv][g_arr] * w[:, None]).sum(0)


def _render_hard(midi_path: Path, man_entry: dict, rng,
                 snr_db: float | None) -> tuple[np.ndarray, int]:
    """Render a full hard-audio mix: multi-stem (chords + bass + drums + melody),
    random scenario, random soundfont, random SNR noise — no clean signal."""
    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    pm_base = pretty_midi.PrettyMIDI(str(midi_path))

    # Random scenario and mix parameters
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

    waves = {}
    sr = 44100
    for name, s in stems.items():
        if s is None or not s.instruments:
            continue
        w, sr = render_to_array(renderer, s, sf_name, reverb)
        waves[name] = w

    if not waves:
        raise RuntimeError("no audio stems rendered")

    L = max(len(w) for w in waves.values())
    mix = np.zeros(L, dtype=np.float32)
    for name, w in waves.items():
        g = gains.get(name, 0.5)
        mix[:len(w)] += g * w

    # Add pink noise
    if snr_db is not None:
        sig = float(np.mean(mix ** 2)) + 1e-12
        mix = mix + pink(L, rng) * np.sqrt(sig / (10 ** (snr_db / 10)))
    else:
        # time-varying degradation (SNR 3–20 dB, also applies phone-like filtering)
        mix = time_varying_degrade(mix, sr, rng)

    peak = np.abs(mix).max()
    if peak > 0.99:
        mix *= 0.99 / peak
    return mix, sr


def infer_song_blind(rec: dict, man_entry: dict, sc, clf, ncl: dict,
                     rng, snr_db: float | None) -> dict | None:
    """Run fully blind inference on a hard multi-stem degraded render."""
    ex = PitchExtractor(cache_dir=REPO / "data" / "cache" / "accomp_blind")

    y, sr = _render_hard(REPO / man_entry["midi_path"], man_entry, rng, snr_db)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
        tmp = Path(wf.name)
    try:
        sf.write(tmp, y, sr)
        acts = ex.extract(tmp, use_cache=False)
    finally:
        tmp.unlink(missing_ok=True)

    # Beat tracking — use GT tempo as starting-point hint to prevent 2× octave flip
    # (librosa halves the tempo on fast jazz, ≥180 BPM, ~20% of the corpus)
    _, beat_frames = librosa.beat.beat_track(y=y, sr=sr,
                                             bpm=float(man_entry["tempo"]),
                                             units="frames")
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    if len(beat_times) < 4:
        return None

    onset_b = pool_to_beats(acts.frame_times, acts.onset_probs, beat_times)
    note_b  = pool_to_beats(acts.frame_times, acts.note_probs,  beat_times)
    bpb = man_entry["beats_per_bar"]

    # Chord segmentation: running-segment chroma novelty + bass-PC change
    # (replicates pipeline_v0; no GT chord boundaries used)
    segs = []
    run_on = run_nt = None
    run_start = 0
    cell = max(1, bpb // 2)   # minimum segment duration (half-bar)
    nov_thresh = 0.35

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

    if not segs:
        return None

    # Classify each segment: root (bass chroma argmax) + quality (trained classifier)
    chords = []
    for beat_start, beat_end, son, snt in segs:
        bass = _reg(son, 0, 52)
        root = int(bass.argmax()) if bass.sum() > 1e-6 else int(_reg(son, 0, 200).argmax())
        rr = lambda c: np.roll(c, -root)
        feat = np.hstack([
            rr(full_chroma(son)), rr(full_chroma(snt)),
            rr(_reg(son, 0, 52)), rr(_reg(son, 60, 200)),
        ])
        t_start = float(beat_times[beat_start])
        t_end   = float(beat_times[min(beat_end, len(beat_times) - 1)])
        bar     = beat_start // bpb
        chords.append({
            "root": root, "bar": bar,
            "t_start": t_start, "t_end": t_end,
            "feat": feat,
        })

    if not chords:
        return None

    X = sc.transform(np.stack([c["feat"] for c in chords]))
    prob: dict[str, np.ndarray] = {}
    for lv in ("fam", "b7", "ex"):
        p = np.full((len(chords), ncl[lv]), 1e-9)
        p[:, clf[lv].classes_] = clf[lv].predict_proba(X)
        prob[lv] = p / p.sum(1, keepdims=True)

    # Assign pred_fam for motif detector (uses family bucket, not index)
    for i, c in enumerate(chords):
        c["pred_fam"] = int(prob["fam"][i].argmax())

    # Build GT quality map (keyed by beat index from GT beat grid)
    spb = 60.0 / man_entry["tempo"]
    gt_beat_map = _build_gt_quality_map(rec, man_entry)

    def match_to_gt(t_start: float, t_end: float) -> dict | None:
        """Find the GT chord that overlaps most with this segment."""
        # collect all GT beat-indexed chords that fall in [t_start, t_end)
        gt_b0 = int(round(t_start / spb))
        gt_b1 = int(round(t_end   / spb))
        hits: dict[tuple, list[float]] = defaultdict(list)
        for b in range(gt_b0, max(gt_b1, gt_b0 + 1)):
            if b in gt_beat_map:
                g = gt_beat_map[b]
                hits[(g["gt_fam"], g["gt_b7"], g["gt_ex"])].append(spb)
        if not hits:
            return None
        best = max(hits, key=lambda k: sum(hits[k]))
        fam, b7, ex = best
        return {"gt_fam": fam, "gt_b7": b7, "gt_ex": ex,
                "overlap": sum(hits[best]) / max(t_end - t_start, spb)}

    # Attach GT labels to inferred segments
    matched = []
    for i, c in enumerate(chords):
        gt = match_to_gt(c["t_start"], c["t_end"])
        if gt is None:
            continue
        matched.append({"chord_idx": i, **gt})

    if not matched:
        return None

    def acc(p_fam, p_b7, p_ex):
        fam = float(np.mean([p_fam[m["chord_idx"]].argmax() == m["gt_fam"] for m in matched]))
        b7  = float(np.mean([p_b7[m["chord_idx"]].argmax()  == m["gt_b7"]  for m in matched]))
        ex  = float(np.mean([p_ex[m["chord_idx"]].argmax()  == m["gt_ex"]  for m in matched]))
        return {"fam": fam, "b7": b7, "ex": ex}

    r_audio = acc(prob["fam"], prob["b7"], prob["ex"])

    # Motif fold on inferred chords
    prob_motif = copy.deepcopy(prob)
    motif_groups = _motif_groups(chords)
    _vote(prob_motif, motif_groups)
    r_motif = acc(prob_motif["fam"], prob_motif["b7"], prob_motif["ex"])

    return {
        "title": rec["title"],
        "n_gt": len(matched),
        "n_segs": len(chords),
        "n_motif_groups": len(motif_groups),
        "audio": r_audio,
        "motif": r_motif,
    }


if __name__ == "__main__":
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-songs", type=int, default=None, help="cap number of songs")
    ap.add_argument("--snr-db", type=float, default=None,
                    help="fixed SNR in dB; omit for time-varying (3–20 dB)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--workers", type=int, default=6, help="parallel workers (default 6)")
    args = ap.parse_args()

    master_rng = np.random.default_rng(args.seed)

    recs = {json.loads(l)["song_id"]: json.loads(l) for l in open(DB)}
    man: dict = {}
    for mm in map(json.loads, open(MANIFEST)):
        if mm["song_id"] not in man or mm.get("transpose", 0) == 0:
            man[mm["song_id"]] = mm

    avail = sorted([sid for sid in recs if sid in man], key=lambda s: recs[s]["title"])
    if args.n_songs:
        avail = avail[:args.n_songs]

    d = np.load(FEAT, allow_pickle=True)
    ncl = {"fam": 5, "b7": len(BASE7), "ex": len(EXACT)}

    print("Fitting global classifier...", flush=True)
    Xall = np.hstack([d["onset"], d["note"], d["bass"], d["treble"]])
    sc   = StandardScaler().fit(Xall)
    clf  = {lv: LogisticRegression(max_iter=500, solver="lbfgs").fit(
                sc.transform(Xall), d[k].astype(int))
            for lv, k in [("fam", "family"), ("b7", "base7"), ("ex", "exact")]}

    snr_label = f"{args.snr_db:.0f} dB" if args.snr_db is not None else "3–20 dB (time-varying)"
    print(f"  done. Running {len(avail)} songs @ SNR {snr_label}  ({args.workers} workers)...\n")

    counter_lock = threading.Lock()
    done_count = [0]

    def run_one(sid, seed):
        rng_local = np.random.default_rng(seed)
        rec = recs[sid]
        try:
            r = infer_song_blind(rec, man[sid], sc, clf, ncl, rng_local, args.snr_db)
        except Exception as e:
            r = None
            with counter_lock:
                done_count[0] += 1
                print(f"\n  SKIP {rec['title']}: {e}", flush=True)
            return None
        with counter_lock:
            done_count[0] += 1
            n = done_count[0]
            print(f"\r[{n:3d}/{len(avail)}] {rec['title'][:32]:32s}", end="", flush=True)
        return r

    # pre-generate per-song seeds so results are reproducible regardless of order
    seeds = master_rng.integers(0, 2**31, size=len(avail)).tolist()

    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, sid, seed): sid
                   for sid, seed in zip(avail, seeds)}
        for fut in as_completed(futures):
            r = fut.result()
            if r:
                results.append(r)

    if not results:
        print("\nNo results."); sys.exit(1)

    print(f"\n\nN = {len(results)} songs  |  SNR: {snr_label}\n")
    W = 72
    print("=" * W)
    print(f"{'Condition':18s}  {'Family':>8s}  {'Seventh':>8s}  {'Exact':>8s}  {'Segs/GT':>8s}")
    print("-" * W)
    for cond, lbl in [("audio", "Blind audio"), ("motif", "Blind + motif fold")]:
        fam = np.mean([r[cond]["fam"] for r in results])
        b7  = np.mean([r[cond]["b7"]  for r in results])
        ex  = np.mean([r[cond]["ex"]  for r in results])
        seg = np.mean([r["n_segs"] / max(r["n_gt"], 1) for r in results]) if cond == "audio" else 0
        seg_s = f"{seg:8.2f}" if cond == "audio" else "        "
        print(f"{lbl:18s}  {fam:8.1%}  {b7:8.1%}  {ex:8.1%}  {seg_s}")
    print("=" * W)

    d_fam = np.mean([r["motif"]["fam"] - r["audio"]["fam"] for r in results])
    d_b7  = np.mean([r["motif"]["b7"]  - r["audio"]["b7"]  for r in results])
    d_ex  = np.mean([r["motif"]["ex"]  - r["audio"]["ex"]  for r in results])
    print(f"\nDelta (motif fold vs blind audio):")
    print(f"  family {d_fam:+.1%}   seventh {d_b7:+.1%}   exact {d_ex:+.1%}")
    print(f"\nAvg motif groups found: {np.mean([r['n_motif_groups'] for r in results]):.1f}")
    print(f"Avg GT-matched segments: {np.mean([r['n_gt'] for r in results]):.1f}  "
          f"detected: {np.mean([r['n_segs'] for r in results]):.1f}")
