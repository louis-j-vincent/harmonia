"""Fetch exact [t0,t1) audio clips (ZERO padding) for the selected wrong-root
examples, same standard as docs/bleed_verification_2026_07_16: WAV output,
ffmpeg trim to exact sample boundaries, ffprobe-verified duration match to
t1-t0 within sub-millisecond precision.

Same disk discipline as scripts/build_rwc_corpus.py: one song's full WAV via
RemoteZip range-request extraction at a time, trim all its needed clips, then
delete the WAV before the next song. Read-only against the Zenodo RWC-P.zip
(same URL/mechanism the corpus builder uses) -- does not touch any other
agent's files.
"""
import json, subprocess, sys
from pathlib import Path
from collections import defaultdict

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO))
OUT_DIR = REPO / "docs/error_report_wrong_root_2026_07_16"
CLIPS_DIR = OUT_DIR / "clips"
CLIPS_DIR.mkdir(parents=True, exist_ok=True)
TMP_AUDIO = OUT_DIR / "_tmp_audio"
TMP_AUDIO.mkdir(parents=True, exist_ok=True)

from remotezip import RemoteZip

ZIP_URL = "https://zenodo.org/api/records/18656623/files/RWC-P.zip/content"

manifest = json.loads((OUT_DIR / "examples_manifest.json").read_text())
examples = manifest["examples"]

by_song = defaultdict(list)
for i, ex in enumerate(examples):
    by_song[ex["song_id"]].append((i, ex))

print(f"{len(examples)} examples across {len(by_song)} songs")

results = []
with RemoteZip(ZIP_URL) as z:
    names = {Path(info.filename).stem: info.filename for info in z.infolist()
             if info.filename.endswith(".wav")}
    for song_id, items in by_song.items():
        rwcid = song_id.replace("rwc_", "")
        zname = names.get(rwcid)
        if not zname:
            print(f"!! {rwcid} not found in zip, skipping {len(items)} examples")
            continue
        print(f"[{rwcid}] extracting WAV ({len(items)} clips needed)...", flush=True)
        z.extract(zname, path=str(TMP_AUDIO))
        wav = TMP_AUDIO / zname
        for i, ex in items:
            t0, t1 = ex["t0"], ex["t1"]
            dur = t1 - t0
            out = CLIPS_DIR / f"ex{i:02d}_{rwcid}_{t0:.3f}.wav"
            cmd = ["ffmpeg", "-y", "-ss", f"{t0:.6f}", "-i", str(wav),
                   "-t", f"{dur:.6f}", "-c:a", "pcm_s16le", str(out)]
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0 or not out.exists():
                print(f"   FAIL ex{i}: {r.stderr[-300:]}")
                continue
            results.append({"idx": i, "path": str(out.relative_to(OUT_DIR)),
                             "t0": t0, "t1": t1, "expected_dur": dur})
        wav.unlink(missing_ok=True)
        print(f"   done, cleaned up WAV", flush=True)

print(f"\nExtracted {len(results)} / {len(examples)} clips")

# verify via ffprobe
print("\nVerifying durations via ffprobe...")
max_err_ms = 0.0
for r in results:
    p = OUT_DIR / r["path"]
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries",
                           "format=duration", "-of",
                           "default=noprint_wrappers=1:nokey=1", str(p)],
                          capture_output=True, text=True)
    actual = float(out.stdout.strip())
    err_ms = abs(actual - r["expected_dur"]) * 1000
    max_err_ms = max(max_err_ms, err_ms)
    r["ffprobe_duration"] = actual
    r["duration_err_ms"] = err_ms

print(f"Max duration error across {len(results)} clips: {max_err_ms:.4f} ms")

(OUT_DIR / "clips_manifest.json").write_text(json.dumps(results, indent=1))
print(f"wrote {OUT_DIR / 'clips_manifest.json'}")

TMP_AUDIO.rmdir()
