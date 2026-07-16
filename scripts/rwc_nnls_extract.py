"""Extract REAL Mauch NNLS-Chroma VAMP (bothchroma, 24-dim bass|treble) for the
RWC-Popular corpus, aligned 1:1 with the existing BP48 blocks.

Provenance discipline (docs/known_issues.md PHASE-0 AUDIT, 2026-07-17):
  This produces the NNLS-24 features for the SAME [t0,t1) chord blocks already in
  data/cache/rwc/rwc_bp48_fixed.npz. Only the feature front-end changes (NNLS vs
  BP48); audio source (RWC-P WAVs from Zenodo), chord blocks, roots, qualities,
  song split are all IDENTICAL -> a confound-clean NNLS-vs-BP48 head-to-head on
  the same rows. feat48_abs is carried over verbatim so the paired baseline is
  trained on the exact same blocks/order.

Pooling (matches the session's bleed-fixed convention): for each block take the
mean bothchroma over VAMP frames whose centre time is in [t0,t1) (nearest-frame
fallback if the span captures none), roll -> C-first pc frame (roll by 9, index
0 = A in the plugin output), L2-normalize each 12-half, stack -> nnls24. This is
the exact representation scratchpad/nnls_real_extract.py produced for Billboard
and that multihead_training.py consumes (already C-frame, L2-per-half).

Disk discipline: RWC-P.zip is 4.07 GB, free disk < 7 GB. We NEVER download the
whole zip -- remotezip pulls one WAV at a time via HTTP range requests; each WAV
is deleted before the next. Peak transient footprint ~1 song. Resumable.

Output: data/cache/rwc/rwc_nnls24.npz  (nnls24 + all carried BP48 fields, 1:1).
"""
from __future__ import annotations
import sys, argparse, time, shutil
from pathlib import Path
import numpy as np

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO))

import librosa, vamp
from remotezip import RemoteZip
from harmonia.data.corpus_schema import load_corpus, save_corpus

ZIP_URL = "https://zenodo.org/api/records/18656623/files/RWC-P.zip/content"
CACHE = REPO / "data/cache/rwc"
AUDIO_DIR = CACHE / "audio_nnls"
BP48 = CACHE / "rwc_bp48_fixed.npz"
OUT = CACHE / "rwc_nnls24.npz"
SR = 44100


def l2(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def extract_bothchroma(wav: Path):
    y, _ = librosa.load(str(wav), sr=SR, mono=True)
    out = vamp.collect(y.astype(np.float32), SR, "nnls-chroma:nnls-chroma", output="bothchroma")
    step, arr = out["matrix"]            # arr (T,24), index 0 = A
    times = np.arange(arr.shape[0]) * float(step)
    return arr, times


def pool_block(arr, times, a, b):
    m = (times >= a) & (times < b)
    if m.sum() == 0:                     # span captured no frame centre -> nearest
        j = int(np.argmin(np.abs(times - 0.5 * (a + b))))
        m = np.zeros(len(times), bool); m[j] = True
    seg = arr[m].mean(0)                 # (24,) index 0=A
    bass = np.roll(seg[:12], 9)          # -> C-first pc frame
    treb = np.roll(seg[12:], 9)
    return np.hstack([l2(bass), l2(treb)]).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", type=int, default=0, help="limit to first N songs (0=all)")
    ap.add_argument("--floor-gb", type=float, default=2.0)
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()

    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    d = load_corpus(BP48)
    N = len(d["root"])
    sid = d["song_id"]
    # unique songs in order of first appearance -> RWCID for the zip name
    seen = []
    for s in sid.tolist():
        if s not in seen:
            seen.append(s)
    if a.songs:
        seen = seen[: a.songs]

    nnls24 = np.zeros((N, 24), np.float32)
    done_mask = np.zeros(N, bool)
    if a.resume and OUT.exists():
        prev = load_corpus(OUT)
        if len(prev.get("nnls24", [])) == N:
            nnls24 = prev["nnls24"].astype(np.float32)
            done_mask = np.abs(nnls24).sum(1) > 0
            print(f"[resume] {done_mask.sum()}/{N} rows already filled", flush=True)

    t_start = time.time()
    with RemoteZip(ZIP_URL) as z:
        names = {Path(i.filename).stem: i.filename for i in z.infolist()
                 if i.filename.endswith(".wav")}
        for k, song in enumerate(seen):
            rows = np.where(sid == song)[0]
            if done_mask[rows].all():
                print(f"[{k+1}/{len(seen)}] {song} already done, skip", flush=True); continue
            free = shutil.disk_usage(str(CACHE)).free / 1e9
            if free < a.floor_gb:
                print(f"!! disk {free:.2f}GB < floor {a.floor_gb}GB -> STOP", flush=True); break
            rwcid = song.replace("rwc_", "")           # rwc_RWC_P001 -> RWC_P001
            zname = names.get(rwcid)
            if not zname:
                print(f"[{k+1}] {song}: no wav in zip, SKIP (rows left zero)", flush=True); continue
            print(f"[{k+1}/{len(seen)}] {song} ({len(rows)} blocks, {free:.1f}GB free) extracting...",
                  flush=True)
            try:
                z.extract(zname, path=str(AUDIO_DIR)); wav = AUDIO_DIR / zname
            except Exception as e:
                print(f"   extract FAIL {e}", flush=True); continue
            try:
                arr, times = extract_bothchroma(wav)
                for r in rows:
                    nnls24[r] = pool_block(arr, times, float(d["t0"][r]), float(d["t1"][r]))
                done_mask[rows] = True
                # bass-argmax->root sanity (untrained premise check, C-frame)
                ba = nnls24[rows][:, :12].argmax(1)
                acc = float((ba == d["root"][rows].astype(int) % 12).mean())
                print(f"   +{len(rows)} rows (total {done_mask.sum()}/{N}, "
                      f"{(time.time()-t_start)/60:.1f}min) bass-argmax->root={acc:.3f}", flush=True)
            except Exception as e:
                print(f"   extract/pool FAIL {e}", flush=True)
            finally:
                wav.unlink(missing_ok=True)

            # checkpoint every song (resumable, cheap)
            save_corpus(OUT, nnls24=nnls24, feat48_abs=d["feat48_abs"], root=d["root"],
                        quality_idx=d["quality_idx"], quality=d["quality"], labels=d["labels"],
                        t0=d["t0"], t1=d["t1"], song_id=d["song_id"], qualities=d["qualities"])

    filled = int((np.abs(nnls24).sum(1) > 0).sum())
    print(f"\n=== DONE === {filled}/{N} rows filled, {len(seen)} songs attempted", flush=True)
    ba = nnls24[:, :12].argmax(1)
    m = np.abs(nnls24).sum(1) > 0
    print(f"overall bass-argmax->root (filled rows) = {(ba[m]==d['root'][m].astype(int)%12).mean():.3f}",
          flush=True)


if __name__ == "__main__":
    main()
