"""Frame-level TEMPORAL feature extraction for the bass/inversion model (RWC).

The pooled corpus (rwc_bp48_fixed.npz) stores ONE sum-pooled 48-d vector per
chord span. The bass model needs the FRAME-LEVEL sequence to learn bass-note
trajectory (walk/hold/move) -- collapsing to one snapshot throws away exactly
the inversion signal. This re-extracts BasicPitch activations (BP cache is
empty; audio was deleted per-song) via remotezip, one song at a time, and
stores per-frame 4-block chroma (48-d, ABSOLUTE, un-normalised) + frame_times +
per-chord [t0,t1,root,label]. Disk-safe: WAV + bp cache deleted per song.

Blocks per frame (match seg_feature_abs order, un-normalised so the temporal
model can normalise however it likes):
  [0:12]  full onset chroma  (_reg_raw(on))
  [12:24] full note  chroma  (_reg_raw(nt))
  [24:36] BASS onset chroma  (_reg_raw(on, 0, 52))   <- MIDI 21..51
  [36:48] treble onset chroma(_reg_raw(on, 60, 200)) <- MIDI 60..
Stored float16. Read-only on shared corpora; writes ONLY scratchpad/bass_temporal_*.
"""
from __future__ import annotations
import sys, json, time, shutil, argparse
from pathlib import Path
import numpy as np

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from harmonia.models.stage1_pitch import PitchExtractor
from scripts.build_jaah_corpus import parse_jaah as parse_harte
from scripts.build_rwc_corpus import fetch_chords, ZIP_URL
from remotezip import RemoteZip

CACHE = REPO / "data/cache/rwc"
AUDIO_DIR = CACHE / "audio"
BP_CACHE = CACHE / "bp_cache"
OUT = REPO / "scratchpad/bass_temporal"

# ---- fold matrices: (88, 12) column c gets note k if lo<=21+k<hi ----
def fold_matrix(lo, hi):
    P = np.zeros((88, 12), np.float32)
    for k in range(88):
        m = 21 + k
        if lo <= m < hi:
            P[k, m % 12] = 1.0
    return P

P_full = fold_matrix(0, 200)      # all
P_bass = fold_matrix(0, 52)       # MIDI < 52
P_treb = fold_matrix(60, 200)     # MIDI >= 60


def frame_features(on, nt):
    """on,nt: (F,88) -> (F,48) float16 [onset_full|note_full|bass_onset|treble_onset]."""
    return np.hstack([on @ P_full, nt @ P_full, on @ P_bass, on @ P_treb]).astype(np.float16)


def clean(wav):
    wav.unlink(missing_ok=True)
    if BP_CACHE.exists():
        for f in BP_CACHE.glob("*"):
            f.unlink(missing_ok=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", type=int, default=30, help="how many RWC songs (from song 1)")
    ap.add_argument("--start", type=int, default=1)
    ap.add_argument("--floor-gb", type=float, default=1.5)
    ap.add_argument("--out", type=str, default="bass_temporal_frames.npz")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    ids = [f"RWC_P{i:03d}" for i in range(a.start, a.start + a.songs)]
    ex = PitchExtractor(cache_dir=BP_CACHE)

    songs = {}   # song_id -> dict(frame_times, feats)
    chords = []  # dict rows
    log = []
    t_start = time.time()
    with RemoteZip(ZIP_URL) as z:
        names = {Path(i.filename).stem: i.filename for i in z.infolist()
                 if i.filename.endswith(".wav")}
        for k, rwcid in enumerate(ids):
            free = shutil.disk_usage(str(CACHE)).free / 1e9
            if free < a.floor_gb:
                print(f"!! disk {free:.2f}GB < floor -> STOP", flush=True); break
            rows = fetch_chords(rwcid)
            if not rows:
                log.append((rwcid, "no_chords")); continue
            zname = names.get(rwcid)
            if not zname:
                log.append((rwcid, "no_wav")); continue
            print(f"[{k+1}/{len(ids)}] {rwcid} ({len(rows)} chords, {free:.1f}GB) dl+BP...", flush=True)
            try:
                z.extract(zname, path=str(AUDIO_DIR)); wav = AUDIO_DIR / zname
            except Exception as e:
                log.append((rwcid, "extract_fail", str(e)[:60])); continue
            try:
                acts = ex.extract(wav)
                ft = acts.frame_times.astype(np.float32)
                feats = frame_features(acts.onset_probs, acts.note_probs)  # (F,48) f16
            except Exception as e:
                log.append((rwcid, "bp_fail", str(e)[:60])); clean(wav); continue
            sid = f"rwc_{rwcid}"
            songs[sid] = {"ft": ft, "feats": feats}
            nkept = 0
            for t0, t1, lab in rows:
                root, fam, _ = parse_harte(lab)
                if root is None:
                    continue
                chords.append({"song_id": sid, "t0": float(t0), "t1": float(t1),
                               "root": int(root % 12), "label": lab})
                nkept += 1
            log.append((rwcid, "ACCEPT", nkept))
            print(f"   F={len(ft)} feats={feats.shape} +{nkept} chords "
                  f"({(time.time()-t_start)/60:.1f}min)", flush=True)
            clean(wav)

    # pack: concatenate frame arrays with per-song offsets
    sids = list(songs.keys())
    off = {}; ft_all = []; fe_all = []; cur = 0
    for sid in sids:
        n = len(songs[sid]["ft"]); off[sid] = (cur, cur + n); cur += n
        ft_all.append(songs[sid]["ft"]); fe_all.append(songs[sid]["feats"])
    ft_all = np.concatenate(ft_all) if ft_all else np.zeros(0, np.float32)
    fe_all = np.concatenate(fe_all) if fe_all else np.zeros((0, 48), np.float16)

    outp = OUT / a.out
    np.savez_compressed(
        outp,
        frame_times=ft_all, frame_feats=fe_all,
        song_ids=np.array(sids),
        song_off=np.array([off[s] for s in sids], np.int64),
        c_song=np.array([c["song_id"] for c in chords]),
        c_t0=np.array([c["t0"] for c in chords], np.float64),
        c_t1=np.array([c["t1"] for c in chords], np.float64),
        c_root=np.array([c["root"] for c in chords], np.int32),
        c_label=np.array([c["label"] for c in chords]),
    )
    (OUT / (Path(a.out).stem + "_log.json")).write_text(json.dumps(log, indent=1, default=str))
    sz = outp.stat().st_size / 1e6
    acc = [l for l in log if l[1] == "ACCEPT"]
    print(f"\n=== DONE {len(acc)} songs, {len(chords)} chords, {len(ft_all)} frames, "
          f"{sz:.1f}MB -> {outp.name} ===")


if __name__ == "__main__":
    main()
