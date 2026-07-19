"""PREPARED, NOT YET RUN (2026-07-15). Adapts rebuild_billboard_fixed.py to
apply hand-corrected GT offsets before re-extracting the Billboard BP48
corpus -- the follow-up step once the user has gone through
/billboard-gt-triage + /gt-offset-fix and saved corrections for enough
songs. Do NOT run this until there's a meaningful number of corrections in
data/cache/billboard_gt_offsets.json (currently near-empty -- this is
scaffolding, not a finished job).

Differences from rebuild_billboard_fixed.py:
  - Reads data/cache/billboard_gt_offsets.json (track_id -> {offset_s, ...}).
  - Only includes songs that HAVE a saved correction (skip_uncorrected=True
    default) -- the whole point of this pass is to not silently ship the
    known-wrong-phase songs again. Flip skip_uncorrected=False once the
    triage list is fully worked through and "corrected" == "verified 0.0
    offset is right", not "not looked at yet".
  - Shifts every (t0, t1) by that song's offset_s before the beat-time
    lookup (chords_full absolute timestamps -> beat indices), matching the
    convention corrected_time = raw_time + offset_s used by
    scripts/harmonia_server.py's _gt_chords_for_video().
  - Skips songs flagged >2s duration-mismatch ("likely wrong edit") even if
    an offset was saved for them, unless force_wrong_edit=True -- those
    need the McGill-Billboard-matching YouTube video re-sourced, not just a
    time shift (see docs/known_issues.md "DATA bug, not display bug").

Everything else (feature extraction, WAV disk discipline, corpus schema) is
unchanged from rebuild_billboard_fixed.py -- see that file for the parts
not reproduced here (parse_harte, seg_feature/_abs imports, etc).
"""
from __future__ import annotations
import sys, json
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

# Re-run scratchpad/rebuild_billboard_fixed.py's parse_harte / tail-set
# constants unchanged -- import by path rather than duplicating, since that
# file is a script (not a package) with no __init__.
sys.path.insert(0, str(REPO / "scratchpad"))
from rebuild_billboard_fixed import parse_harte  # noqa: E402

OFFSETS_FILE = REPO / "data" / "cache" / "billboard_gt_offsets.json"
SEARCH_FILES = [
    REPO / "scratchpad" / "billboard_search_results_60.json",
    REPO / "scratchpad" / "billboard_search_results.json",
]
AUDIO_DIR = REPO / "data" / "cache" / "billboard_audio_tmp"
BP_CACHE = REPO / "data" / "cache" / "billboard_bp48"
WRONG_EDIT_THRESHOLD_S = 2.0


def load_offsets() -> dict[str, dict]:
    try:
        return json.loads(OFFSETS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def load_corpus_meta() -> dict[str, dict]:
    merged: dict[str, dict] = {}
    for p in SEARCH_FILES:
        if p.exists():
            merged.update(json.loads(p.read_text(encoding="utf-8")))
    return merged


def build(skip_uncorrected: bool = True, force_wrong_edit: bool = False,
          out_path: Path | None = None) -> None:
    offsets = load_offsets()
    meta = load_corpus_meta()
    ds = mirdata.initialize("billboard")
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    all_records = []
    skipped = []
    for tid, info in meta.items():
        best = info.get("best") or []
        if not best:
            continue
        vid, diff, dur, ctitle = best[0], best[1], best[2], best[3] if len(best) > 3 else ""
        gt_dur = info.get("gt_dur")
        mismatch = abs(dur - gt_dur) if (dur is not None and gt_dur is not None) else 0.0
        if mismatch > WRONG_EDIT_THRESHOLD_S and not force_wrong_edit:
            skipped.append((tid, f"likely wrong edit, mismatch={mismatch:.1f}s -- re-source video first"))
            continue

        corr = offsets.get(tid)
        if corr is None:
            if skip_uncorrected:
                skipped.append((tid, "no saved GT-offset correction yet"))
                continue
            offset_s = 0.0
        else:
            offset_s = float(corr.get("offset_s", 0.0))

        print(f"=== {tid} {info.get('artist','')} - {info.get('title','')} "
              f"offset={offset_s:+.2f}s ===", flush=True)
        try:
            wav = download_audio(vid, AUDIO_DIR)
        except Exception as e:
            skipped.append((tid, f"download failed: {e}")); continue
        try:
            bf = extract_beat_features(wav, cache_dir=BP_CACHE)
        except Exception as e:
            skipped.append((tid, f"extraction failed: {e}")); wav.unlink(missing_ok=True); continue

        cf = ds.track(tid).chords_full
        onset_b, note_b, beat_times = bf.onset_b, bf.note_b, bf.beat_times
        for (t0, t1), label in zip(cf.intervals, cf.labels):
            t0c, t1c = float(t0) + offset_s, float(t1) + offset_s  # apply correction
            pc, fam = parse_harte(label)
            if pc is None:
                continue
            b0 = int(np.searchsorted(beat_times, t0c, side="right")) - 1
            b1 = int(np.searchsorted(beat_times, t1c, side="right"))
            b0 = max(b0, 0); b1 = min(b1, len(onset_b))
            if b1 - b0 < 1:
                continue
            feat48 = seg_feature(onset_b, note_b, b0, b1, pc)
            feat48_abs = seg_feature_abs(onset_b, note_b, b0, b1)
            all_records.append({
                "feat48": feat48, "feat48_abs": feat48_abs,
                "root": int(pc % 12), "quality": fam, "quality_idx": QUALITY_IDX[fam],
                "t0": t0c, "t1": t1c, "label": label,
                "match": "exact", "song_id": f"bb_{tid}",
                "gt_offset_applied_s": offset_s,
            })
        wav.unlink(missing_ok=True)

    print(f"\n{len(all_records)} chord records, {len(skipped)} songs skipped:")
    for tid, why in skipped:
        print(f"  {tid}: {why}")

    if out_path and all_records:
        save_corpus(all_records, out_path)
        print(f"saved -> {out_path}")


if __name__ == "__main__":
    # Default: dry-run style call with skip_uncorrected=True so this does
    # nothing destructive until corrections actually exist. Flip
    # out_path once ready.
    build(skip_uncorrected=True, force_wrong_edit=False, out_path=None)
