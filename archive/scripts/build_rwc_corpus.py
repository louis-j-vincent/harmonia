"""Build an RWC-Popular (100 J-pop/pop tracks) real-audio BP48 training corpus.

Why RWC is the SAFEST real-audio corpus this project has (2026-07-16):
  Unlike Billboard/JAAH — where audio had to be separately sourced from
  YouTube and duration-matched to GT timestamps (the exact failure mode that
  produced 0-6.9s per-song offsets and ~25% wrong-edit songs) — RWC ships
  BOTH the audio AND the chord annotations as a matched pair:
    * audio:  Zenodo record 18656623 (CC BY-NC 4.0), RWC-P.zip, 100 WAVs
    * chords: github.com/rwc-music/rwc-annotations, per-song CSVs (Cho-Bello),
              Harte labels + ABSOLUTE second timestamps, inversions preserved.
  Both keyed 1:1 by RWCID (RWC_P001 ... RWC_P100). There is NO "did I find
  the right recording" step — the annotations were made against exactly these
  files. Zero sourcing/alignment risk.

Disk discipline (this machine has had disk-full crises):
  RWC-P.zip is 4.07 GB but free disk is < 4 GB. We NEVER download the whole
  zip. `remotezip` uses HTTP range requests (Zenodo supports 206) to pull one
  WAV at a time; each WAV (+ its BasicPitch activation cache) is deleted before
  the next. Peak transient footprint is ~1 song. Self-throttles at --floor-gb.

Feature extraction reuses harmonia.models.chord_pipeline_v1.extract_beat_features
and harmonia.data.yt_chord_corpus.seg_feature[_abs] exactly as the JAAH/Billboard
builders do (no reimplementation). Chord->(root,family) reuses the JAAH Harte
parser verbatim (RWC labels are the same Harte dialect).

Modes:
  --pilot [N]   extract+featurize the first N songs (default 2), print
                parse coverage + record stats, then stop. Screen before build.
  --build       full corpus; writes data/cache/rwc/rwc_bp48.npz (resumable).
"""
from __future__ import annotations
import sys, csv, io, json, argparse, time, shutil
from pathlib import Path
import numpy as np
import urllib.request

REPO = Path("/Users/vincente/Documents/Projets Perso/Code/harmonia")
sys.path.insert(0, str(REPO))

from harmonia.models.chord_pipeline_v1 import extract_beat_features
from harmonia.models.stage1_pitch import PitchExtractor, BASIC_PITCH_FRAME_RATE
from harmonia.data.yt_chord_corpus import seg_feature, seg_feature_abs, QUALITY_IDX, QUALITIES
from harmonia.data.corpus_schema import save_corpus, load_corpus
# RWC chords are the same Harte dialect as JAAH -> reuse that parser verbatim.
from scripts.build_jaah_corpus import parse_jaah as parse_harte

from remotezip import RemoteZip

ZIP_URL = "https://zenodo.org/api/records/18656623/files/RWC-P.zip/content"
CHORD_BASE = ("https://raw.githubusercontent.com/rwc-music/rwc-annotations/"
              "main/01_annotations_preprocessed/chords/RWC-P")
UA = "harmonia-research/1.0 (louisjvincent@gmail.com)"
CACHE = REPO / "data/cache/rwc"
AUDIO_DIR = CACHE / "audio"
BP_CACHE = CACHE / "bp_cache"


def fetch_chords(rwcid: str):
    """Return list of (t_start, t_end, label) from the GitHub CSV, or None."""
    url = f"{CHORD_BASE}/{rwcid}.csv"
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            text = r.read().decode("utf-8")
    except Exception:
        return None
    rows = []
    rd = csv.reader(io.StringIO(text), delimiter=";")
    header = next(rd, None)
    for row in rd:
        if len(row) != 3:
            continue
        try:
            rows.append((float(row[0]), float(row[1]), row[2].strip()))
        except ValueError:
            continue
    return rows


# Minimum frames pooled per chord (~46ms at 86.13 Hz). Chords whose exact
# [t0,t1) clip captures fewer frames get a minimal symmetric expansion about
# the span midpoint to reach this floor, so no chord yields a zero/degenerate
# feature vector. See docs/known_issues.md "boundary-bleed fix" (2026-07-16).
MIN_FRAMES = 4


def build_song(rwcid: str, wav: Path, rows, unparsed: set, bleed_stats: list | None = None):
    """FIXED pooling (2026-07-16): pool BasicPitch frames CLIPPED EXACTLY to the
    GT chord span [t0,t1) at the 86.13 Hz frame level, instead of snapping to a
    beat grid and sum-pooling whole beats (which bled ~310ms of the NEXT chord
    into each feature — see docs/known_issues.md). Frame-sum-pooling is the same
    operation the old beat-pool then did over a beat range, just over the exact
    in-span frame set, so feature scale/semantics are unchanged (and each 12-dim
    block is L2-normed downstream anyway)."""
    ex = PitchExtractor(cache_dir=BP_CACHE)
    acts = ex.extract(wav)
    ft = acts.frame_times                       # (F,) frame-centre times, seconds
    on = acts.onset_probs; nt = acts.note_probs  # (F, 88)
    nfr = len(ft)
    recs = []
    for t0, t1, lab in rows:
        root, fam, _ = parse_harte(lab)
        if root is None:
            if lab not in ("N", "X", ""):
                unparsed.add(lab)
            continue
        # frames whose centre falls in [t0, t1)  -> exact clip, zero bleed
        i0 = int(np.searchsorted(ft, t0, side="left"))
        i1 = int(np.searchsorted(ft, t1, side="left"))
        if i1 - i0 < MIN_FRAMES:              # pathologically short span: floor it
            mid = 0.5 * (t0 + t1)
            c = int(np.searchsorted(ft, mid, side="left"))
            i0 = max(0, c - MIN_FRAMES // 2)
            i1 = min(nfr, i0 + MIN_FRAMES)
            i0 = max(0, i1 - MIN_FRAMES)
        i0 = max(i0, 0); i1 = min(i1, nfr)
        if i1 <= i0:
            continue
        if bleed_stats is not None and nfr:
            # measured feature-window extent vs the true span (verify_bleed style)
            w0, w1 = float(ft[i0]), float(ft[i1 - 1])
            bleed_stats.append((max(0.0, t0 - w0), max(0.0, w1 - t1)))
        seg_on = on[i0:i1].sum(0, keepdims=True)   # (1, 88)
        seg_nt = nt[i0:i1].sum(0, keepdims=True)   # (1, 88)
        recs.append({
            "feat48": seg_feature(seg_on, seg_nt, 0, 1, root),
            "feat48_abs": seg_feature_abs(seg_on, seg_nt, 0, 1),
            "root": int(root % 12), "quality": fam, "quality_idx": QUALITY_IDX[fam],
            "t0": float(t0), "t1": float(t1), "label": lab,
            "match": "exact", "song_id": f"rwc_{rwcid}",
        })
    return recs


def clean_transients(wav: Path):
    wav.unlink(missing_ok=True)
    if BP_CACHE.exists():
        for f in BP_CACHE.glob("*"):
            f.unlink(missing_ok=True)


def run(mode, n_pilot, max_songs, floor_gb, resume, out_name="rwc_bp48.npz"):
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    BP_CACHE.mkdir(parents=True, exist_ok=True)
    ids = [f"RWC_P{i:03d}" for i in range(1, 101)]
    if mode == "pilot":
        ids = ids[:n_pilot]
    elif max_songs:
        ids = ids[:max_songs]

    log_name = "build_log.json" if out_name == "rwc_bp48.npz" else f"build_log_{Path(out_name).stem}.json"
    prev_log, prev_recs = [], {}
    if resume:
        lp = CACHE / log_name
        if lp.exists():
            prev_log = json.loads(lp.read_text())
        done = {l[0] for l in prev_log if l[1] == "ACCEPT"}
        ids = [i for i in ids if i not in done]
        cp = CACHE / out_name
        if cp.exists():
            prev_recs = load_corpus(cp)
            print(f"[resume] {len(done)} songs done; {len(prev_recs['root'])} records loaded; "
                  f"{len(ids)} remaining", flush=True)

    all_recs, log, unparsed, bleed_stats = [], [], set(), []
    t_start = time.time()
    with RemoteZip(ZIP_URL) as z:
        names = {Path(i.filename).stem: i.filename for i in z.infolist()
                 if i.filename.endswith(".wav")}
        for k, rwcid in enumerate(ids):
            free = shutil.disk_usage(str(CACHE)).free / 1e9
            if free < floor_gb:
                print(f"!! disk {free:.2f}GB < floor {floor_gb}GB -> STOP", flush=True)
                break
            rows = fetch_chords(rwcid)
            if not rows:
                log.append((rwcid, "no_chords")); print(f"[{rwcid}] no chords, skip"); continue
            zname = names.get(rwcid)
            if not zname:
                log.append((rwcid, "no_wav")); print(f"[{rwcid}] no wav in zip, skip"); continue
            print(f"[{k+1}/{len(ids)}] {rwcid}  ({len(rows)} chords, {free:.1f}GB free) "
                  f"extracting WAV...", flush=True)
            try:
                z.extract(zname, path=str(AUDIO_DIR))
                wav = AUDIO_DIR / zname
            except Exception as e:
                log.append((rwcid, "extract_fail", str(e)[:80])); print("   extract FAIL"); continue
            try:
                recs = build_song(rwcid, wav, rows, unparsed, bleed_stats)
            except Exception as e:
                log.append((rwcid, "feat_fail", str(e)[:80])); clean_transients(wav)
                print(f"   feat FAIL: {e}"); continue
            all_recs += recs
            log.append((rwcid, "ACCEPT", len(recs)))
            print(f"   +{len(recs)} records (total {len(all_recs)}, "
                  f"{(time.time()-t_start)/60:.1f}min)", flush=True)
            clean_transients(wav)

    acc = [l for l in log if l[1] == "ACCEPT"]
    print(f"\n=== SUMMARY === attempted={len(ids)} accepted={len(acc)} "
          f"records={len(all_recs)}")
    if unparsed:
        print(f"UNPARSED labels ({len(unparsed)}): {sorted(unparsed)}")
    else:
        print("UNPARSED labels: none (parser covers full RWC-P vocab)")

    if bleed_stats:
        bs = np.array(bleed_stats)  # (n, 2): (pre_bleed_s, post_bleed_s)
        print(f"\n=== BLEED CHECK (fixed pooling, n={len(bs)} chords) ===")
        print(f"  PRE-bleed  (window starts before t0): mean {bs[:,0].mean()*1e3:.1f}ms "
              f"max {bs[:,0].max()*1e3:.1f}ms")
        print(f"  POST-bleed (window ends after t1):    mean {bs[:,1].mean()*1e3:.1f}ms "
              f"max {bs[:,1].max()*1e3:.1f}ms   [OLD baseline: ~310ms mean]")

    if mode == "pilot":
        if all_recs:
            roots = np.array([r["root"] for r in all_recs])
            fams = [r["quality"] for r in all_recs]
            from collections import Counter
            print("root dist:", dict(Counter(roots.tolist())))
            print("family dist:", dict(Counter(fams)))
            f48 = np.stack([r["feat48"] for r in all_recs])
            print(f"feat48 shape={f48.shape} finite={np.isfinite(f48).all()} "
                  f"norm~[{f48.min():.3f},{f48.max():.3f}]")
        return

    merged_log = prev_log + log
    (CACHE / log_name).write_text(json.dumps(merged_log, indent=2, default=str))
    if all_recs or prev_recs:
        out = {
            "feat48": np.stack([r["feat48"] for r in all_recs]) if all_recs else np.zeros((0, 48), np.float32),
            "feat48_abs": np.stack([r["feat48_abs"] for r in all_recs]) if all_recs else np.zeros((0, 48), np.float32),
            "root": np.array([r["root"] for r in all_recs], dtype=np.int32),
            "quality_idx": np.array([r["quality_idx"] for r in all_recs], dtype=np.int32),
            "quality": np.array([r["quality"] for r in all_recs]),
            "labels": np.array([r["label"] for r in all_recs]),
            "match": np.array([r["match"] for r in all_recs]),
            "t0": np.array([r["t0"] for r in all_recs]),
            "t1": np.array([r["t1"] for r in all_recs]),
            "song_id": np.array([r["song_id"] for r in all_recs]),
        }
        if prev_recs:
            for k in list(out.keys()):
                a = prev_recs[k] if k in prev_recs else np.array([])
                out[k] = np.concatenate([a, out[k]]) if (len(a) or len(out[k])) else a
        out["qualities"] = np.array(QUALITIES)
        outp = CACHE / out_name
        save_corpus(outp, **out)
        n_songs = len(set(out["song_id"].tolist()))
        print(f"\nWrote {outp} ({len(out['root'])} records, {n_songs} songs)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", nargs="?", type=int, const=2, default=None)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--max-songs", type=int, default=0)
    ap.add_argument("--floor-gb", type=float, default=2.5)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default="rwc_bp48.npz", help="output npz filename under data/cache/rwc/")
    a = ap.parse_args()
    if a.pilot is not None:
        run("pilot", a.pilot, 0, a.floor_gb, False, a.out)
    elif a.build:
        run("build", 0, a.max_songs, a.floor_gb, a.resume, a.out)
    else:
        ap.print_help()
