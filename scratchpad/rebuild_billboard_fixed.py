"""Rebuild the Billboard BP48 corpus after fixing extract_beat_features's
rigid constant-tempo beat grid (harmonia/models/chord_pipeline_v1.py ~L1929).

Reuses the already-matched YouTube video IDs from the pilot + 50-song
extension search results (no re-search needed) -- re-downloads audio,
re-runs extract_beat_features (now using real detected beat times instead
of a uniform arange grid), re-samples BP48 features at Billboard's own
chords_full absolute timestamps. Deletes each WAV immediately after
extraction (disk discipline -- only ~2.9GB free).
"""
from __future__ import annotations
import sys, json, time
from pathlib import Path
import numpy as np

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO))

import mirdata
from harmonia.models.chord_pipeline_v1 import extract_beat_features
from harmonia.data.yt_chord_corpus import (
    seg_feature, seg_feature_abs, download_audio, QUALITY_IDX,
)
from harmonia.data.corpus_schema import save_corpus

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


CACHE_DIR = REPO / "data/cache/billboard_60"
AUDIO_DIR = CACHE_DIR / "audio"
BP_CACHE = CACHE_DIR / "bp_cache"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
BP_CACHE.mkdir(parents=True, exist_ok=True)

pilot_matches = json.loads((REPO / "scratchpad/billboard_search_results.json").read_text())
ext_matches = json.loads((REPO / "scratchpad/billboard_search_results_60.json").read_text())
all_matches = {**pilot_matches, **ext_matches}
print(f"Total track_ids with prior search results: {len(all_matches)}")

ds = mirdata.initialize("billboard")

all_records = []
skipped_songs = []
hits = 0
attempted = 0

t_start = time.time()
items = list(all_matches.items())
for i, (tid, info) in enumerate(items):
    best = info.get("best")
    if not best:
        continue
    attempted += 1
    vid, diff, dur, ctitle = best
    print(f"\n=== [{i+1}/{len(items)}] {tid}  {info['artist']} - {info['title']} -> {vid} (diff={diff:.1f}s) ===", flush=True)

    try:
        wav = download_audio(vid, AUDIO_DIR)
    except Exception as e:
        print("  DOWNLOAD FAILED:", e)
        skipped_songs.append((tid, f"download failed: {e}"))
        continue

    try:
        bf = extract_beat_features(wav, cache_dir=BP_CACHE)
    except Exception as e:
        print("  FEATURE EXTRACTION FAILED:", e)
        skipped_songs.append((tid, f"extraction failed: {e}"))
        wav.unlink(missing_ok=True)
        continue

    t = ds.track(tid)
    cf = t.chords_full
    onset_b, note_b, beat_times = bf.onset_b, bf.note_b, bf.beat_times

    n_recs = 0
    for (t0, t1), label in zip(cf.intervals, cf.labels):
        pc, fam = parse_harte(label)
        if pc is None:
            continue
        b0 = int(np.searchsorted(beat_times, t0, side="right")) - 1
        b1 = int(np.searchsorted(beat_times, t1, side="right"))
        b0 = max(b0, 0); b1 = min(b1, len(onset_b))
        if b1 - b0 < 1:
            continue
        feat48 = seg_feature(onset_b, note_b, b0, b1, pc)
        feat48_abs = seg_feature_abs(onset_b, note_b, b0, b1)
        all_records.append({
            "feat48": feat48, "feat48_abs": feat48_abs,
            "root": int(pc % 12), "quality": fam, "quality_idx": QUALITY_IDX[fam],
            "t0": float(t0), "t1": float(t1), "label": label,
            "match": "exact", "song_id": f"bb_{tid}",
        })
        n_recs += 1
    print(f"  -> {n_recs} chord records  (corpus total: {len(all_records)})", flush=True)
    hits += 1
    wav.unlink(missing_ok=True)

    elapsed = time.time() - t_start
    print(f"  [elapsed {elapsed/60:.1f} min, {i+1}/{len(items)} done, hit rate so far {hits}/{attempted}]", flush=True)

print("\n\n=== REBUILD SUMMARY (fixed beat grid) ===")
print(f"Songs attempted (had a prior duration match): {attempted}")
print(f"Hits: {hits}/{attempted}")
print(f"Skipped: {len(skipped_songs)}")
for s in skipped_songs:
    print("  ", s)
print(f"Total records: {len(all_records)}")
print(f"Unique songs: {len(set(r['song_id'] for r in all_records))}")

if all_records:
    out = {
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
    OUT = REPO / "data/cache/billboard_bp48_60_fixed_beatgrid.npz"
    save_corpus(OUT, **out)
    print(f"\nWrote {OUT}")
