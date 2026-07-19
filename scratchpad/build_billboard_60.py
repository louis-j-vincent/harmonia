"""Extend the Billboard BP48 pilot from 10 to 60 songs.

Same method as build_billboard_pilot.py: duration-matched YouTube search,
download, BP48 extraction at Billboard's own chords_full absolute timestamps
(no alignment/inference step), delete WAV immediately after extraction.

Samples 50 NEW track_ids (excluding the pilot's 10) via random.sample with a
fixed seed for reproducibility and spread across the corpus (mirdata Billboard
ids are non-sequential strings).
"""
from __future__ import annotations
import sys, json, random, subprocess, time
from pathlib import Path
import numpy as np

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO))

import mirdata
from harmonia.models.chord_pipeline_v1 import extract_beat_features
from harmonia.models.stage1_pitch import PitchExtractor
from harmonia.data.yt_chord_corpus import (
    seg_feature, seg_feature_abs, seg_feature_clipped, seg_feature_abs_clipped,
    download_audio, QUALITY_IDX,
)
from harmonia.data.corpus_schema import save_corpus, load_corpus

PILOT_IDS = {"954", "183", "44", "1111", "406", "362", "334", "217", "1104", "168"}
N_NEW = 50
SEED = 42
YTDLP = str(REPO / ".venv/bin/yt-dlp")

_NOTE_PC = {n: i for i, n in enumerate(
    ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
)}
_NOTE_PC.update({"Db":1,"Eb":3,"Gb":6,"Ab":8,"Bb":10,"Fb":4,"Cb":11,"E#":5,"B#":0})

_MAJ_TAILS = {"maj","maj6","maj7","maj9","maj(9)","6","9","add9","maj11","maj13",""}
_MIN_TAILS = {"min","min6","min7","min9","min11","min13","minmaj7",
              "min(b13)","min(9)","min(11)","min(6)"}
_DOM_TAILS = {"7","9","11","13","7(b9)","7(#9)","7(b13)","7(#11)","9(#11)","13(b9)"}
_HDIM_TAILS = {"hdim7","min7b5","dim7/b7"}
_DIM_TAILS = {"dim","dim7"}
_AUG_TAILS = {"aug","aug7"}
_SUS_TAILS = {"sus4","sus2","sus4(b7)","sus2(b7)","7sus4","sus"}


def harte_family(tail: str) -> str | None:
    tail = tail.strip()
    if tail in _MAJ_TAILS: return "maj"
    if tail in _MIN_TAILS: return "min"
    if tail in _DOM_TAILS: return "dom"
    if tail in _HDIM_TAILS: return "hdim"
    if tail in _DIM_TAILS: return "dim"
    if tail in _AUG_TAILS: return "aug"
    if tail in _SUS_TAILS: return "sus"
    if tail.startswith("min"): return "min"
    if tail.startswith("dim"): return "dim"
    if tail.startswith("aug"): return "aug"
    if tail.startswith("sus"): return "sus"
    if tail.startswith("hdim"): return "hdim"
    return None


def parse_harte(label: str):
    if label in ("N", "X"):
        return None, None
    base = label.split("/")[0]
    if ":" in base:
        root_str, tail = base.split(":", 1)
    else:
        root_str, tail = base, ""
    pc = _NOTE_PC.get(root_str)
    if pc is None:
        return None, None
    fam = harte_family(tail)
    if fam is None:
        return None, None
    return pc, fam


def yt_search(query, n=5):
    cmd = [YTDLP, f"ytsearch{n}:{query}", "--print", "%(id)s\t%(duration)s\t%(title)s",
           "--skip-download", "--no-warnings"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=90)
    lines = [l for l in r.stdout.strip().split("\n") if l.strip()]
    out = []
    for l in lines:
        parts = l.split("\t")
        if len(parts) != 3:
            continue
        vid, dur, title = parts
        try:
            dur = float(dur)
        except ValueError:
            dur = None
        out.append((vid, dur, title))
    return out, r.stderr


CACHE_DIR = REPO / "data/cache/billboard_60"
AUDIO_DIR = CACHE_DIR / "audio"
BP_CACHE = CACHE_DIR / "bp_cache"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
BP_CACHE.mkdir(parents=True, exist_ok=True)

ds = mirdata.initialize("billboard")
all_ids = sorted(set(ds.track_ids) - PILOT_IDS, key=int)

rng = random.Random(SEED)
sample_ids = rng.sample(all_ids, N_NEW)
print(f"Sampled {len(sample_ids)} new track_ids (seed={SEED}): {sample_ids}")

search_results = {}
all_records = []
skipped_songs = []
hits = 0

t_start = time.time()
for i, tid in enumerate(sample_ids):
    t = ds.track(tid)
    artist, title = t.artist, t.title
    cf = t.chords_full
    if len(cf.intervals) == 0:
        skipped_songs.append((tid, artist, title, "no chords_full annotations"))
        continue
    gt_dur = float(cf.intervals[-1][1])
    query = f"{artist} {title}"
    print(f"\n=== [{i+1}/{len(sample_ids)}] {tid}  {query}  (GT dur={gt_dur:.1f}s, chart={t.chart_date}) ===", flush=True)

    try:
        cands, err = yt_search(query)
    except Exception as e:
        print("  SEARCH FAILED:", e)
        skipped_songs.append((tid, artist, title, f"search failed: {e}"))
        search_results[tid] = {"artist": artist, "title": title, "gt_dur": gt_dur, "error": str(e)}
        continue

    best = None
    for vid, dur, ctitle in cands:
        if dur is None:
            continue
        diff = abs(dur - gt_dur)
        tol = max(0.05 * gt_dur, 5.0)
        ok = diff <= tol
        marker = "OK" if ok else "  "
        print(f"  [{marker}] {vid}  dur={dur:.1f}  diff={diff:.1f}  tol={tol:.1f}  {ctitle}")
        if ok and (best is None or diff < best[1]):
            best = (vid, diff, dur, ctitle)
    search_results[tid] = {"artist": artist, "title": title, "gt_dur": gt_dur,
                            "best": best, "candidates": cands}

    if not best:
        skipped_songs.append((tid, artist, title, "no duration match"))
        continue

    vid, diff, dur, ctitle = best
    print(f"  MATCH -> {vid} (diff={diff:.1f}s)", flush=True)

    try:
        wav = download_audio(vid, AUDIO_DIR)
    except Exception as e:
        print("  DOWNLOAD FAILED:", e)
        skipped_songs.append((tid, artist, title, f"download failed: {e}"))
        continue

    try:
        # Boundary-bleed fix (2026-07-16): pool frames clipped exactly to
        # [t0,t1) instead of whole-beat snapping (see docs/known_issues.md
        # "boundary bleed"). PitchExtractor.extract is cached.
        acts = PitchExtractor(cache_dir=BP_CACHE).extract(wav)
    except Exception as e:
        print("  FEATURE EXTRACTION FAILED:", e)
        skipped_songs.append((tid, artist, title, f"extraction failed: {e}"))
        wav.unlink(missing_ok=True)
        continue

    ft, onf, ntf = acts.frame_times, acts.onset_probs, acts.note_probs

    n_recs = 0
    for (t0, t1), label in zip(cf.intervals, cf.labels):
        pc, fam = parse_harte(label)
        if pc is None:
            continue
        feat48 = seg_feature_clipped(ft, onf, ntf, t0, t1, pc)
        feat48_abs = seg_feature_abs_clipped(ft, onf, ntf, t0, t1)
        if feat48 is None or feat48_abs is None:
            continue
        all_records.append({
            "feat48": feat48, "feat48_abs": feat48_abs,
            "root": int(pc % 12), "quality": fam, "quality_idx": QUALITY_IDX[fam],
            "t0": float(t0), "t1": float(t1), "label": label,
            "match": "billboard_gt", "song_id": f"bb_{tid}",
        })
        n_recs += 1
    print(f"  -> {n_recs} chord records  (corpus total: {len(all_records)})", flush=True)
    hits += 1

    # disk discipline: delete wav immediately
    wav.unlink(missing_ok=True)

    elapsed = time.time() - t_start
    print(f"  [elapsed {elapsed/60:.1f} min, {i+1}/{len(sample_ids)} done, hit rate so far {hits}/{i+1}]", flush=True)

print("\n\n=== BUILD SUMMARY (50-song extension) ===")
print(f"Songs attempted: {len(sample_ids)}")
print(f"Hits: {hits}/{len(sample_ids)}  ({100*hits/len(sample_ids):.1f}%)")
print(f"Skipped: {len(skipped_songs)}")
for s in skipped_songs:
    print("  ", s)
print(f"Total new records: {len(all_records)}")
print(f"Unique new songs in corpus: {len(set(r['song_id'] for r in all_records))}")

with open(REPO / "scratchpad/billboard_search_results_60.json", "w") as f:
    json.dump(search_results, f, indent=2, default=str)
with open(REPO / "scratchpad/billboard_60_skipped.json", "w") as f:
    json.dump(skipped_songs, f, indent=2, default=str)

if all_records:
    new_out = {
        "feat48": np.stack([r["feat48"] for r in all_records]),
        "feat48_abs": np.stack([r["feat48_abs"] for r in all_records]),
        "root": np.array([r["root"] for r in all_records], dtype=np.int32),
        "quality_idx": np.array([r["quality_idx"] for r in all_records], dtype=np.int32),
        "quality": np.array([r["quality"] for r in all_records]),
        "labels": np.array([r["label"] for r in all_records]),
        "match": np.array([r["match"] for r in all_records]),
        "t0": np.array([r["t0"] for r in all_records]),
        "t1": np.array([r["t1"] for r in all_records]),
        "song_id": np.array([r["song_id"] for r in all_records]),
        "qualities": np.array(["maj","min","dom","hdim","dim","aug","sus"]),
    }
    NEW_OUT = REPO / "data/cache/billboard_bp48_50new.npz"
    save_corpus(NEW_OUT, **new_out)
    print(f"\nWrote {NEW_OUT}")

    # merge with pilot
    pilot = load_corpus(REPO / "data/cache/billboard_bp48_pilot.npz")
    merged = {}
    for k in ["feat48", "feat48_abs", "root", "quality_idx", "quality", "labels",
              "match", "t0", "t1", "song_id"]:
        merged[k] = np.concatenate([pilot[k], new_out[k]], axis=0)
    merged["qualities"] = pilot["qualities"]
    MERGED_OUT = REPO / "data/cache/billboard_bp48_60.npz"
    save_corpus(MERGED_OUT, **merged)
    print(f"Wrote {MERGED_OUT}  (total records: {len(merged['root'])}, "
          f"unique songs: {len(set(merged['song_id'].tolist()))})")
else:
    print("\nNo new records extracted; merged corpus NOT written.")
