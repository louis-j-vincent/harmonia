"""2×2 ablation: isolate what drives the oracle→blind accuracy gap.

Conditions:
  audio:  clean (full-MIDI render, no degradation)
          degraded (multi-stem + time-varying phone noise, same as eval_blind)
  segs:   oracle (GT chord boundaries from db.jsonl)
          blind  (chroma-novelty detector, same as eval_blind)

Expected anchors
  clean + oracle  → ~89%  (matches existing eval_motif_fold_accuracy baseline)
  deg   + blind   → ~44–48% (matches eval_blind_motif_accuracy)

If  clean+blind ≈ deg+oracle  → problem is shared equally
If  clean+blind >> deg+oracle → segmentation is the main driver
If  deg+oracle  >> clean+blind → audio degradation is the main driver

Usage:
    .venv/bin/python scripts/ablation_gap.py                  # Anthropology only
    .venv/bin/python scripts/ablation_gap.py --n-songs 20
    .venv/bin/python scripts/ablation_gap.py --snr-db 8
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import warnings
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
    EXACT, EXACT_IDX, FAM_IDX, full_chroma,
)
from harmonia.data.midi_renderer import MIDIRenderer, RenderConfig
from harmonia.models.stage1_pitch import PitchExtractor

DB       = REPO / "data" / "accomp_db" / "db.jsonl"
MANIFEST = REPO / "data" / "accomp_db" / "audio" / "manifest.jsonl"
FEAT     = REPO / "data" / "cache" / "audio_chord_features.npz"


def _pool_to_beats(frame_times, probs, beat_times):
    n = len(beat_times)
    out = np.zeros((n, probs.shape[1]), dtype=np.float32)
    idx = np.searchsorted(beat_times, frame_times)
    for b, p in zip(idx, probs):
        if 0 <= b < n:
            out[b] += p
    return out


def _reg(v88, lo, hi):
    c = np.zeros(12)
    for k in range(88):
        if lo <= 21 + k < hi:
            c[(21 + k) % 12] += v88[k]
    return c


def _unit(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


# ── audio renderers ──────────────────────────────────────────────────────────

def _render_clean(midi_path: Path, renderer: MIDIRenderer) -> tuple[np.ndarray, int]:
    """Full-MIDI render, no stem mix, no degradation."""
    pm = pretty_midi.PrettyMIDI(str(midi_path))
    sf_name = SOUNDFONTS[0]
    w, sr = render_to_array(renderer, pm, sf_name, reverb=False)
    return w, sr


def _render_degraded(midi_path: Path, man_entry: dict,
                     rng, snr_db: float | None,
                     renderer: MIDIRenderer) -> tuple[np.ndarray, int]:
    """Multi-stem render + time-varying phone degradation (same as eval_blind)."""
    pm_base = pretty_midi.PrettyMIDI(str(midi_path))
    scen   = str(rng.choice(list(SCENARIOS)))
    gains  = {k: v * float(rng.uniform(0.8, 1.2)) for k, v in SCENARIOS[scen].items()}
    sf_name = SOUNDFONTS[int(rng.integers(0, len(SOUNDFONTS)))]
    reverb  = bool(rng.integers(0, 2))
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

    if not waves:
        raise RuntimeError("no audio stems rendered")

    L = max(len(w) for w in waves.values())
    mix = np.zeros(L, dtype=np.float32)
    for name, w in waves.items():
        mix[:len(w)] += gains.get(name, 0.5) * w

    if snr_db is not None:
        sig = float(np.mean(mix ** 2)) + 1e-12
        mix = mix + pink(L, rng) * np.sqrt(sig / (10 ** (snr_db / 10)))
    else:
        mix = time_varying_degrade(mix, sr, rng)

    peak = np.abs(mix).max()
    if peak > 0.99:
        mix *= 0.99 / peak
    return mix, sr


# ── segmenters ────────────────────────────────────────────────────────────────

def _segs_blind(onset_b: np.ndarray, note_b: np.ndarray,
                bpb: int) -> list[tuple]:
    """Chroma-novelty running-segment detector (identical to eval_blind)."""
    segs = []
    run_on = run_nt = None
    run_start = 0
    cell = max(1, bpb // 2)
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
    return segs


def _segs_oracle(onset_b: np.ndarray, note_b: np.ndarray,
                 beat_times: np.ndarray, rec: dict,
                 man_entry: dict) -> list[tuple]:
    """Pool beats into GT chord spans. Returns same (b0,b1,son,snt) tuples."""
    segs = []
    for t0, t1, _root, _q in song_chord_spans(rec):
        bi0 = int(np.searchsorted(beat_times, t0))
        bi1 = int(np.searchsorted(beat_times, t1))
        bi1 = max(bi1, bi0 + 1)
        bi1 = min(bi1, len(onset_b))
        if bi0 >= len(onset_b):
            continue
        son = onset_b[bi0:bi1].sum(0)
        snt = note_b[bi0:bi1].sum(0)
        if son.sum() < 1e-6:
            continue
        segs.append((bi0, bi1, son, snt))
    return segs


# ── shared: classify + score ─────────────────────────────────────────────────

def _classify_segs(segs, beat_times, sc, clf, ncl):
    """Root argmax + LR classify. Returns list of chord dicts + prob arrays."""
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
        chords.append({"root": root, "t_start": t_start, "t_end": t_end, "feat": feat})

    if not chords:
        return [], {}

    X = sc.transform(np.stack([c["feat"] for c in chords]))
    prob = {}
    for lv in ("fam", "b7", "ex"):
        p = np.full((len(chords), ncl[lv]), 1e-9)
        p[:, clf[lv].classes_] = clf[lv].predict_proba(X)
        prob[lv] = p / p.sum(1, keepdims=True)
    return chords, prob


def _build_gt_beat_map(rec, man_entry):
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
                "root":   root % 12,
                "gt_fam": FAM_IDX[BUCKET_FAMILY[p[1]]],
                "gt_b7":  BASE7_IDX[BUCKET_BASE7[p[1]]],
                "gt_ex":  EXACT_IDX[p[1]],
            }
    return gt


def _score(chords, prob, gt_beat_map, spb):
    """Weighted-overlap match to GT beats, compute root/fam/b7/ex accuracy."""
    from collections import defaultdict

    matched = []
    for i, c in enumerate(chords):
        gt_b0 = int(round(c["t_start"] / spb))
        gt_b1 = int(round(c["t_end"]   / spb))
        hits: dict[tuple, float] = defaultdict(float)
        for b in range(gt_b0, max(gt_b1, gt_b0 + 1)):
            if b in gt_beat_map:
                g = gt_beat_map[b]
                key = (g["root"], g["gt_fam"], g["gt_b7"], g["gt_ex"])
                hits[key] += spb
        if not hits:
            continue
        best = max(hits, key=hits.__getitem__)
        gt_root, gt_fam, gt_b7, gt_ex = best
        matched.append({
            "i": i,
            "gt_root": gt_root, "gt_fam": gt_fam, "gt_b7": gt_b7, "gt_ex": gt_ex,
            "pred_root": chords[i]["root"],
        })

    if not matched:
        return None

    idxs = [m["i"] for m in matched]
    root_ok = np.mean([prob_argmax(prob["fam"], m["i"]) >= 0
                       and chords[m["i"]]["root"] == m["gt_root"]
                       for m in matched])
    fam_ok  = np.mean([prob["fam"][m["i"]].argmax() == m["gt_fam"] for m in matched])
    b7_ok   = np.mean([prob["b7"][m["i"]].argmax()  == m["gt_b7"]  for m in matched])
    ex_ok   = np.mean([prob["ex"][m["i"]].argmax()  == m["gt_ex"]  for m in matched])
    return {"root": float(root_ok), "fam": float(fam_ok),
            "b7": float(b7_ok), "ex": float(ex_ok),
            "n_matched": len(matched), "n_segs": len(chords)}


def prob_argmax(arr, i):
    return int(arr[i].argmax())


# ── per-song runner ──────────────────────────────────────────────────────────

def run_song(rec, man_entry, sc, clf, ncl, rng, snr_db):
    renderer = MIDIRenderer(soundfont_dir=REPO / "data" / "soundfonts")
    ex = PitchExtractor(cache_dir=REPO / "data" / "cache" / "ablation_gap")
    midi_path = REPO / man_entry["midi_path"]
    spb = 60.0 / man_entry["tempo"]
    bpb = man_entry["beats_per_bar"]
    gt_beat_map = _build_gt_beat_map(rec, man_entry)

    def extract(y, sr):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wf:
            tmp = Path(wf.name)
        try:
            sf.write(tmp, y, sr)
            acts = ex.extract(tmp, use_cache=False)
        finally:
            tmp.unlink(missing_ok=True)
        _, beat_frames = librosa.beat.beat_track(y=y, sr=sr,
                                                  bpm=float(man_entry["tempo"]),
                                                  units="frames")
        beat_times = librosa.frames_to_time(beat_frames, sr=sr)
        if len(beat_times) < 4:
            return None, None
        onset_b = _pool_to_beats(acts.frame_times, acts.onset_probs, beat_times)
        note_b  = _pool_to_beats(acts.frame_times, acts.note_probs,  beat_times)
        return (onset_b, note_b, beat_times)

    results = {}

    # -- render both audio conditions --
    y_clean, sr_clean = _render_clean(midi_path, renderer)
    # use a split rng so degraded render is reproducible regardless of clean render
    y_deg, sr_deg = _render_degraded(midi_path, man_entry, rng, snr_db, renderer)

    for audio_label, y, sr in [("clean", y_clean, sr_clean),
                                ("deg",   y_deg,   sr_deg)]:
        feats = extract(y, sr)
        if feats is None:
            continue
        onset_b, note_b, beat_times = feats

        for seg_label, segs in [
            ("oracle", _segs_oracle(onset_b, note_b, beat_times, rec, man_entry)),
            ("blind",  _segs_blind(onset_b, note_b, bpb)),
        ]:
            if not segs:
                continue
            chords, prob = _classify_segs(segs, beat_times, sc, clf, ncl)
            if not chords:
                continue
            r = _score(chords, prob, gt_beat_map, spb)
            if r:
                results[f"{audio_label}_{seg_label}"] = r

    return results


# ── main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import threading
    from concurrent.futures import ThreadPoolExecutor, as_completed

    ap = argparse.ArgumentParser()
    ap.add_argument("--song",    default=None, help="single song name (substring match)")
    ap.add_argument("--n-songs", type=int, default=None)
    ap.add_argument("--snr-db",  type=float, default=None)
    ap.add_argument("--seed",    type=int, default=42)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    master_rng = np.random.default_rng(args.seed)

    recs = {json.loads(l)["song_id"]: json.loads(l) for l in open(DB)}
    man: dict = {}
    for mm in map(json.loads, open(MANIFEST)):
        if mm["song_id"] not in man or mm.get("transpose", 0) == 0:
            man[mm["song_id"]] = mm

    avail = sorted([sid for sid in recs if sid in man], key=lambda s: recs[s]["title"])

    if args.song:
        avail = [s for s in avail if args.song.lower() in recs[s]["title"].lower()]
        if not avail:
            print(f"No song matching '{args.song}'"); sys.exit(1)
    elif args.n_songs:
        avail = avail[:args.n_songs]

    d = np.load(FEAT, allow_pickle=True)
    ncl = {"fam": 5, "b7": len(BASE7), "ex": len(EXACT)}

    print("Fitting classifier...", flush=True)
    Xall = np.hstack([d["onset"], d["note"], d["bass"], d["treble"]])
    sc   = StandardScaler().fit(Xall)
    clf  = {lv: LogisticRegression(max_iter=500, solver="lbfgs").fit(
                sc.transform(Xall), d[k].astype(int))
            for lv, k in [("fam", "family"), ("b7", "base7"), ("ex", "exact")]}

    snr_label = f"{args.snr_db:.0f} dB" if args.snr_db is not None else "3–20 dB (time-varying)"
    print(f"Running {len(avail)} song(s)  SNR={snr_label}  workers={args.workers}\n")

    lock = threading.Lock()
    done = [0]
    all_results: list[dict] = []

    def run_one(sid, seed):
        rng_local = np.random.default_rng(seed)
        rec = recs[sid]
        try:
            r = run_song(rec, man[sid], sc, clf, ncl, rng_local, args.snr_db)
        except Exception as e:
            with lock:
                done[0] += 1
                print(f"\n  SKIP {rec['title']}: {e}", flush=True)
            return None
        with lock:
            done[0] += 1
            print(f"\r[{done[0]:3d}/{len(avail)}] {rec['title'][:40]:40s}", end="", flush=True)
        return r

    seeds = master_rng.integers(0, 2**31, size=len(avail)).tolist()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(run_one, sid, seed): sid
                for sid, seed in zip(avail, seeds)}
        for fut in as_completed(futs):
            r = fut.result()
            if r:
                all_results.append(r)

    if not all_results:
        print("\nNo results."); sys.exit(1)

    print(f"\n\nN = {len(all_results)} songs  |  SNR: {snr_label}\n")

    CONDITIONS = [
        ("clean_oracle", "Clean audio  + Oracle segs"),
        ("clean_blind",  "Clean audio  + Blind segs "),
        ("deg_oracle",   "Degraded     + Oracle segs"),
        ("deg_blind",    "Degraded     + Blind segs "),
    ]
    METRICS = [("root", "Root"), ("fam", "Family"), ("b7", "Seventh"), ("ex", "Exact")]

    W = 80
    print("=" * W)
    print(f"{'Condition':34s}" + "".join(f"  {m[1]:>8s}" for m in METRICS) + f"  {'N-segs':>8s}")
    print("-" * W)
    for cond, label in CONDITIONS:
        rows = [r[cond] for r in all_results if cond in r]
        if not rows:
            print(f"{label:34s}  (no data)")
            continue
        vals = {m: np.mean([r[m] for r in rows]) for m, _ in METRICS}
        n_segs = np.mean([r["n_segs"] for r in rows])
        print(f"{label:34s}" +
              "".join(f"  {vals[m]:8.1%}" for m, _ in METRICS) +
              f"  {n_segs:8.1f}")
    print("=" * W)

    # Print gap decomposition (for root accuracy)
    print("\nRoot accuracy gap decomposition:")
    rows_co = [r["clean_oracle"] for r in all_results if "clean_oracle" in r]
    rows_cb = [r["clean_blind"]  for r in all_results if "clean_blind"  in r]
    rows_do = [r["deg_oracle"]   for r in all_results if "deg_oracle"   in r]
    rows_db = [r["deg_blind"]    for r in all_results if "deg_blind"    in r]

    if rows_co and rows_db:
        total_gap = np.mean([r["root"] for r in rows_co]) - np.mean([r["root"] for r in rows_db])
        print(f"  Total gap (clean+oracle → deg+blind):   {total_gap:+.1%}")
    if rows_co and rows_cb:
        seg_cost = np.mean([r["root"] for r in rows_co]) - np.mean([r["root"] for r in rows_cb])
        print(f"  Segmentation cost (clean+oracle → blind): {seg_cost:+.1%}")
    if rows_co and rows_do:
        deg_cost = np.mean([r["root"] for r in rows_co]) - np.mean([r["root"] for r in rows_do])
        print(f"  Degradation cost  (clean+oracle → oracle): {deg_cost:+.1%}")
    if rows_cb and rows_db:
        deg_on_blind = np.mean([r["root"] for r in rows_cb]) - np.mean([r["root"] for r in rows_db])
        print(f"  Degradation cost on blind segs:            {deg_on_blind:+.1%}")
    if rows_do and rows_db:
        seg_on_deg = np.mean([r["root"] for r in rows_do]) - np.mean([r["root"] for r in rows_db])
        print(f"  Segmentation cost on degraded audio:       {seg_on_deg:+.1%}")
