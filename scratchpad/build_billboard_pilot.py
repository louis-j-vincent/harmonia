"""Billboard BP48 pilot corpus builder.

For each Billboard track: download the matched YouTube audio, extract BP48
beat features via the production pipeline, and sample features directly at
Billboard's chords_full absolute-timestamp intervals (no alignment/inference
step -- GT timestamps are trusted directly, unlike the YouTube+iReal corpus).
"""
from __future__ import annotations
import sys, json
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
from harmonia.data.corpus_schema import save_corpus

_NOTE_PC = {n: i for i, n in enumerate(
    ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
)}
# enharmonic flats used by Harte
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
    return None  # includes bare power chords ('5'), single notes ('1'), N, X


def parse_harte(label: str):
    """Return (root_pc, family) or (None, None)."""
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


MATCHES = json.loads(open(Path(__file__).parent / "billboard_search_results.json").read())

CACHE_DIR = REPO / "data/cache/billboard_pilot"
AUDIO_DIR = CACHE_DIR / "audio"
BP_CACHE = CACHE_DIR / "bp_cache"
AUDIO_DIR.mkdir(parents=True, exist_ok=True)
BP_CACHE.mkdir(parents=True, exist_ok=True)

ds = mirdata.initialize("billboard")

all_records = []
skipped_songs = []

for tid, info in MATCHES.items():
    best = info.get("best")
    if not best:
        skipped_songs.append((tid, "no duration match"))
        continue
    vid, diff, dur, ctitle = best
    print(f"\n=== {tid}  {info['artist']} - {info['title']}  -> {vid} (diff={diff:.1f}s) ===", flush=True)

    try:
        wav = download_audio(vid, AUDIO_DIR)
    except Exception as e:
        print("  DOWNLOAD FAILED:", e)
        skipped_songs.append((tid, f"download failed: {e}"))
        continue

    try:
        # boundary-bleed fix (2026-07-16): clipped frame pooling, cached extract
        acts = PitchExtractor(cache_dir=BP_CACHE).extract(wav)
    except Exception as e:
        print("  FEATURE EXTRACTION FAILED:", e)
        skipped_songs.append((tid, f"extraction failed: {e}"))
        wav.unlink(missing_ok=True)
        continue

    t = ds.track(tid)
    cf = t.chords_full
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

    # cleanup: delete wav immediately to save disk
    wav.unlink(missing_ok=True)

print("\n\n=== BUILD SUMMARY ===")
print(f"Songs attempted: {len(MATCHES)}")
print(f"Songs skipped: {skipped_songs}")
print(f"Total records: {len(all_records)}")
print(f"Unique songs in corpus: {len(set(r['song_id'] for r in all_records))}")

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
    OUT = REPO / "data/cache/billboard_bp48_pilot.npz"
    save_corpus(OUT, **out)
    print(f"\nWrote {OUT}")
