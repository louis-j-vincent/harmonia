"""Audio excerpts for the triage page.

One mono MP3 per queued region: the region plus ~2 s of lead-in, base64 so the
page can carry it inline (the artifact CSP blocks every external request).

Every temp file is removed by its own explicit name immediately after it is
read - never by glob, never touching anything that already existed.
"""
from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
SCRATCH = Path("/private/tmp/claude-501/-Users-vincente-Documents-Projets-Perso-Code-harmonia/"
               "22c6b747-cea4-410f-85ae-f6ed58a9d2eb/scratchpad")
TMP = SCRATCH / "triage_tmp"
LEAD = 2.0
MAXLEN = 13.0
BITRATE = "40k"
FLOOR_GIB = 1.5


def free_gib() -> float:
    return shutil.disk_usage(REPO).free / 2**30


def encode(src: Path, out: Path, ss: float, dur: float):
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{ss:.3f}", "-i", str(src),
         "-t", f"{dur:.3f}", "-ac", "1", "-ar", "22050", "-c:a", "libmp3lame",
         "-b:a", BITRATE, "-f", "mp3", str(out)], check=True)


def main():
    songs = json.loads((SCRATCH / "gt_triage.json").read_text())
    want = []
    for sid, S in songs.items():
        src = REPO / S["audio_path"]
        for r in S["regions"]:
            if r["bucket"] not in ("needs-ear", "drift"):
                continue
            a = max(0.0, r["t0"] - LEAD)
            b = min(r["t1"] + 0.4, a + MAXLEN)
            want.append({"key": f"{sid}:{r['region_idx']}", "src": src,
                         "a": a, "b": b, "doubt": r["doubt"]})
    print(f"{len(want)} clips to build; disk free {free_gib():.2f} GiB")
    if free_gib() < FLOOR_GIB + 0.2:
        print("ABORT: too close to the disk floor")
        return 1
    TMP.mkdir(exist_ok=True)
    out, total = {}, 0
    for i, it in enumerate(want):
        if i % 20 == 0 and free_gib() < FLOOR_GIB + 0.1:
            print(f"ABORT at {i}: disk free {free_gib():.2f} GiB")
            break
        mp3 = TMP / f"t_{it['key'].replace(':', '_')}.mp3"
        encode(it["src"], mp3, it["a"], it["b"] - it["a"])
        raw = mp3.read_bytes()
        mp3.unlink()                       # this exact file, by name
        b64 = base64.b64encode(raw).decode("ascii")
        out[it["key"]] = {"b64": b64, "a": round(it["a"], 2), "b": round(it["b"], 2)}
        total += len(b64)
    (SCRATCH / "triage_clips.json").write_text(json.dumps(out, separators=(",", ":")))
    if TMP.exists() and not any(TMP.iterdir()):
        TMP.rmdir()
    print(f"{len(out)} clips, {total/1e6:.2f} MB base64; disk free {free_gib():.2f} GiB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
