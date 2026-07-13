"""train_online.py — chord model training on all iReal playlists via YouTube.

Two phases:

  Phase 1 (extract):  For each song / video:
    - Look up or search YouTube video ID (vid_cache.json, persistent)
    - If feat_cache/{vid}.npz exists → skip (already done)
    - Else: download WAV → extract BP/CQT features → align iReal GT
            → save feat_cache/{vid}.npz (~70KB) → delete WAV + BP cache

  Phase 2 (train):  Load all feat_cache entries, train MLP for N epochs
    - 108-dim (BP48 + CQT12 + ABS48), 3-class quality (maj/min/dom)
    - Song-level train/val split (seed=42, 15% val)
    - Cosine LR, checkpoints to --out on val improvement

Disk footprint: ~70KB per song in feat_cache/ (WAVs and BP cache deleted immediately).

Usage:
    # Extract + train on all playlists
    .venv/bin/python scripts/train_online.py

    # Only extract (no training yet)
    .venv/bin/python scripts/train_online.py --skip-train

    # Only train from existing feat_cache
    .venv/bin/python scripts/train_online.py --skip-extract

    # Specific playlists
    .venv/bin/python scripts/train_online.py --playlists jazz1460 blues50
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

CACHE_DIR      = REPO / "data" / "cache" / "yt_corpus"
VID_CACHE_PATH = CACHE_DIR / "vid_cache.json"
FEAT_DIR       = CACHE_DIR / "feat_cache"
IREAL_DIR      = REPO / "data" / "ireal"

ALL_PLAYLISTS = ["jazz1460", "pop400", "blues50", "brazilian220",
                 "latin_salsa50", "dixieland1", "country"]

logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")


# ── song loading ──────────────────────────────────────────────────────────────

def load_all_songs(playlist_names: list[str]) -> list[dict]:
    import contextlib, io
    from pyRealParser import Tune
    import urllib.parse

    songs = []
    for name in playlist_names:
        path = IREAL_DIR / f"{name}.txt"
        if not path.exists():
            print(f"[warn] {path} not found", flush=True)
            continue
        corpus_raw = path.read_text()
        n_before = len(songs)
        for part in corpus_raw.split("==="):
            part = part.strip().lstrip("=")
            if not part:
                continue
            url = "irealb://" + part
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    tunes = Tune.parse_ireal_url(urllib.parse.unquote(url))
            except Exception:
                continue
            if not tunes:
                continue
            tune = tunes[0]
            title = getattr(tune, "title", "") or ""
            if not title or "irealb://" in title:
                continue
            songs.append({
                "title":      title,
                "composer":   getattr(tune, "composer", "") or "",
                "irealb_url": url,
                "playlist":   name,
            })
        print(f"  {name}: {len(songs) - n_before} songs", flush=True)
    return songs


# ── video ID cache ────────────────────────────────────────────────────────────

def load_vid_cache() -> dict:
    if VID_CACHE_PATH.exists():
        return json.loads(VID_CACHE_PATH.read_text())
    return {}


def save_vid_cache(cache: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    VID_CACHE_PATH.write_text(json.dumps(cache, indent=2))


BAD_VIDS_PATH = CACHE_DIR / "bad_vids.json"

def load_bad_vids() -> set[str]:
    if BAD_VIDS_PATH.exists():
        return set(json.loads(BAD_VIDS_PATH.read_text()))
    return set()

def save_bad_vids(bad: set[str]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    BAD_VIDS_PATH.write_text(json.dumps(sorted(bad), indent=2))


# Search queries tried in order; later queries target "clean head only" recordings
_SEARCH_QUERIES = [
    "{title} {composer}",            # broad first pass
    "{title} official",       # simpler arrangement
    "{title} jazz play along",             # backing track / head only
]

def get_video_ids(title: str, composer: str, cache: dict, n: int = 2,
                  bad_vids: set | None = None,
                  _ctr: list = [0]) -> list[str]:
    key = f"{title}|{composer}"
    bad = bad_vids or set()

    if key in cache:
        v = cache[key]
        ids = (v if isinstance(v, list) else [v]) if v else []
        good = [i for i in ids if i not in bad]
        if good:
            return good
        # All cached IDs were blacklisted — fall through to re-search

    ytdlp = shutil.which("yt-dlp") or str(Path(sys.executable).parent / "yt-dlp")

    # Try search queries in order; use whichever returns IDs not in bad_vids
    all_new_ids: list[str] = []
    for q_tmpl in _SEARCH_QUERIES:
        query = q_tmpl.format(title=title, composer=composer)
        cmd = [ytdlp, "--get-id", "--no-warnings", f"ytsearch{n}:{query}"]
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL, timeout=30)
            found = [i for i in out.strip().splitlines()[:n] if i not in bad]
        except Exception:
            found = []
        all_new_ids.extend(i for i in found if i not in all_new_ids)
        if all_new_ids:
            break  # found non-blacklisted videos — stop trying

    cache[key] = all_new_ids
    _ctr[0] += 1
    if _ctr[0] % 20 == 0:
        save_vid_cache(cache)
    return all_new_ids


def feat_quality(path: Path) -> tuple[float, float]:
    """Return (coverage_ratio, dur_weighted_mismatch) for a feat_cache file.
    coverage = fraction of total audio duration covered by GT bars.
    """
    try:
        d = np.load(path, allow_pickle=True)
        t0, t1 = d["t0"], d["t1"]
        match   = d["match"]
        total   = float(t1[-1]) if len(t1) else 0
        if total < 5:
            return 0.0, 1.0
        covered = float((t1 - t0).sum())
        coverage = min(covered / total, 1.0)
        dur_w    = (t1 - t0) / (t1 - t0).sum()
        mm_rate  = float(((match == "mismatch") * dur_w).sum())
        return coverage, mm_rate
    except Exception:
        return 0.0, 1.0


# Thresholds — tuned on the 936-song corpus (all_blues=0.08, whats_new=0.97)
COVERAGE_MIN  = 0.80   # skip if GT covers < 80% of audio (solos not in GT)
MISMATCH_MAX  = 0.45   # skip if > 45% of labelled duration is wrong root/quality


# ── feature extraction ────────────────────────────────────────────────────────

def extract_and_cache(video_id: str, irealb_url: str) -> bool:
    """Download WAV → extract features → save feat_cache → delete WAV + BP cache.
    Returns True if successful."""
    from harmonia.data.yt_chord_corpus import download_audio, extract_records, pack_arrays

    audio_dir = CACHE_DIR / "audio"
    bp_cache  = CACHE_DIR / "bp_cache"
    audio_dir.mkdir(parents=True, exist_ok=True)
    bp_cache.mkdir(parents=True, exist_ok=True)
    FEAT_DIR.mkdir(parents=True, exist_ok=True)

    wav = None
    try:
        wav = download_audio(video_id, audio_dir)
        records = extract_records(wav, irealb_url, cache_dir=bp_cache)
        if records:
            np.savez(FEAT_DIR / f"{video_id}.npz", **pack_arrays(records))
            return True
        return False
    except Exception as e:
        print(f"    [err] {e}", flush=True)
        return False
    finally:
        # Delete WAV immediately
        for p in audio_dir.glob(f"{video_id}.*"):
            p.unlink(missing_ok=True)
        # Delete entire BP cache (all entries stale once WAV is gone)
        for p in bp_cache.glob("*.npz"):
            p.unlink(missing_ok=True)


# ── extraction phase ──────────────────────────────────────────────────────────

def phase_purge_bad(vid_cache: dict) -> None:
    """Delete feat_cache files that fail the quality gate, blacklist their IDs,
    and clear their vid_cache entry so the next extraction tries different videos.
    """
    bad_vids = load_bad_vids()
    # Build reverse map: video_id -> cache key
    rev: dict[str, str] = {}
    for key, vids in vid_cache.items():
        for v in (vids if isinstance(vids, list) else [vids] if vids else []):
            rev[v] = key

    purged = 0
    for p in sorted(FEAT_DIR.glob("*.npz")):
        cov, mm = feat_quality(p)
        if cov < COVERAGE_MIN or mm > MISMATCH_MAX:
            vid = p.stem
            key = rev.get(vid, "?")
            print(f"  purge [{vid}] cov={cov:.2f} mm={mm:.2f}  {key}", flush=True)
            bad_vids.add(vid)
            # Remove this ID from the cache entry so the song is re-searched
            if key in vid_cache:
                existing = vid_cache[key]
                if isinstance(existing, list):
                    vid_cache[key] = [v for v in existing if v != vid]
                elif existing == vid:
                    vid_cache[key] = []
            p.unlink(missing_ok=True)
            purged += 1

    save_bad_vids(bad_vids)
    save_vid_cache(vid_cache)
    total_now = len(list(FEAT_DIR.glob("*.npz")))
    print(f"\nPurge done: {purged} files removed, "
          f"{total_now} feat_cache remaining, "
          f"{len(bad_vids)} IDs blacklisted.", flush=True)


def phase_extract(songs: list[dict], vid_cache: dict,
                  n_per_song: int = 2, max_songs: int = 0,
                  force: bool = False) -> None:
    existing = {p.stem for p in FEAT_DIR.glob("*.npz")}
    print(f"\n=== Phase 1: Extract  ({len(songs)} songs, {n_per_song} videos each, "
          f"{len(existing)} already cached) ===", flush=True)

    bad_vids = load_bad_vids()
    n_new = n_skip = n_err = n_no_yt = 0

    for i, song in enumerate(songs):
        title, composer = song["title"], song["composer"]
        vids = get_video_ids(title, composer, vid_cache, n=n_per_song,
                             bad_vids=bad_vids)
        if not vids:
            n_no_yt += 1
            continue

        for vid in vids:
            if not force and vid in existing:
                n_skip += 1
                continue

            print(f"[{i+1}/{len(songs)}] {title!r} ({vid}) ...", flush=True)
            t0 = time.time()
            ok = extract_and_cache(vid, song["irealb_url"])
            elapsed = time.time() - t0

            if ok:
                existing.add(vid)
                n_new += 1
                # Quick record count from saved file
                try:
                    d = np.load(FEAT_DIR / f"{vid}.npz")
                    n_clean = int(np.isin(d["match"], ["exact", "family"]).sum())
                    print(f"    → {len(d['feat48'])} records ({n_clean} clean) "
                          f"in {elapsed:.0f}s", flush=True)
                except Exception:
                    pass
            else:
                n_err += 1

        if (i + 1) % 50 == 0:
            save_vid_cache(vid_cache)
            print(f"  [{i+1}/{len(songs)}] new={n_new} cached={n_skip} "
                  f"no_yt={n_no_yt} err={n_err}", flush=True)

        if max_songs > 0 and (n_new + n_skip) >= max_songs:
            print(f"Reached --max-songs {max_songs}", flush=True)
            break

    save_vid_cache(vid_cache)
    total = len(list(FEAT_DIR.glob("*.npz")))
    print(f"\nExtraction done: {n_new} new, {n_skip} cached, "
          f"{n_no_yt} no-YouTube, {n_err} errors  "
          f"(total feat_cache: {total})", flush=True)


# ── training phase ────────────────────────────────────────────────────────────

def make_mlp(in_dim: int, n_classes: int, h1: int = 64, h2: int = 32):
    import torch.nn as nn
    return nn.Sequential(
        nn.Linear(in_dim, h1), nn.LayerNorm(h1), nn.GELU(), nn.Dropout(0.3),
        nn.Linear(h1, h2),    nn.LayerNorm(h2),  nn.GELU(), nn.Dropout(0.3),
        nn.Linear(h2, n_classes),
    )


def _load_feat(path: Path) -> dict | None:
    try:
        # Quality gate: skip recordings with long solos or systematic mismatch
        cov, mm = feat_quality(path)
        if cov < COVERAGE_MIN or mm > MISMATCH_MAX:
            return None
        d = np.load(path)
        match  = d.get("match", np.array(["exact"] * len(d["feat48"])))
        keep   = np.isin(match, ["exact", "family"])
        yi     = d["quality_idx"][keep].astype(np.int64)
        keep3  = yi <= 2
        if keep3.sum() < 2:
            return None
        f48   = d["feat48"][keep][keep3].astype(np.float32)
        fabs  = d["feat48_abs"][keep][keep3].astype(np.float32)
        _cqt  = d.get("feat12_cqt")
        cqt   = _cqt[keep][keep3].astype(np.float32) if _cqt is not None else np.zeros((len(f48), 12), np.float32)
        _cabs = d.get("feat12_cqt_abs")
        cabs  = _cabs[keep][keep3].astype(np.float32) if _cabs is not None else np.zeros((len(f48), 12), np.float32)
        y_r = d["root"][keep][keep3].astype(np.int64)
        y_q = yi[keep3]
        y_chord = y_r * 3 + y_q   # 36-class exact chord (root × quality)
        # Duration of each chord segment (seconds) — used as sample weight in loss and val.
        # Longer segments = more acoustic evidence = should count more.
        t0 = d.get("t0")
        t1 = d.get("t1")
        if t0 is not None and t1 is not None:
            dur = (t1 - t0)[keep][keep3].astype(np.float32)
            dur = np.clip(dur, 0.1, None)   # floor at 100ms to avoid zero weights
        else:
            dur = np.ones(len(y_q), np.float32)
        return {
            "X_q":     np.concatenate([f48, cqt, fabs], axis=1),
            "X_r":     np.concatenate([fabs, cabs],      axis=1),
            "y_q":     y_q,
            "y_r":     y_r,
            "y_chord": y_chord,
            "dur":     dur,    # segment duration in seconds
            "vid":     path.stem,
        }
    except Exception:
        return None


def phase_train(songs: list[dict], vid_cache: dict, *,
                epochs: int, lr: float, batch: int,
                h1: int, h2: int, val_frac: float, seed: int,
                out_path: Path) -> None:
    import torch, torch.nn as nn

    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"\n=== Phase 2: Train (device={device}) ===", flush=True)

    # ── collect feat_cache entries ────────────────────────────────────────────
    # Build video_id → song mapping for fast lookup
    vid_to_song: dict[str, dict] = {}
    for song in songs:
        for vid in (vid_cache.get(f"{song['title']}|{song['composer']}") or []):
            if isinstance(vid, str):
                vid_to_song[vid] = song

    # Load all feat_cache files
    print("Loading feat_cache ...", flush=True)
    all_data = []
    for p in sorted(FEAT_DIR.glob("*.npz")):
        d = _load_feat(p)
        if d is not None:
            all_data.append(d)

    if not all_data:
        print("No feat_cache entries — run extraction first.", flush=True)
        return

    all_X_q    = np.concatenate([d["X_q"]     for d in all_data])
    all_X_r    = np.concatenate([d["X_r"]     for d in all_data])
    all_y_q    = np.concatenate([d["y_q"]     for d in all_data])
    all_y_r    = np.concatenate([d["y_r"]     for d in all_data])
    all_y_chord= np.concatenate([d["y_chord"] for d in all_data])
    all_dur    = np.concatenate([d["dur"]     for d in all_data])   # segment durations
    all_sids   = np.concatenate([[d["vid"]] * len(d["y_q"]) for d in all_data])

    unique_vids = list(dict.fromkeys(all_sids.tolist()))
    print(f"Loaded {len(all_X_q)} records from {len(unique_vids)} videos", flush=True)

    from harmonia.data.yt_chord_corpus import QUALITIES
    for qi, qn in enumerate(QUALITIES[:3]):
        n = (all_y_q == qi).sum()
        print(f"  {qn}: {n} ({100*n/len(all_y_q):.1f}%)", flush=True)

    # ── train/val split on video level ────────────────────────────────────────
    rng = np.random.default_rng(seed)
    n_val = max(1, int(len(unique_vids) * val_frac))
    val_vids = set(rng.choice(unique_vids, n_val, replace=False).tolist())
    val_mask = np.array([v in val_vids for v in all_sids.tolist()], dtype=bool)
    tr_mask  = ~val_mask
    print(f"Train: {tr_mask.sum()} records ({len(unique_vids)-n_val} videos)  "
          f"Val: {val_mask.sum()} records ({n_val} videos)", flush=True)

    # ── standardize ───────────────────────────────────────────────────────────
    mean_q = all_X_q[tr_mask].mean(0).astype(np.float32)
    std_q  = (all_X_q[tr_mask].std(0) + 1e-9).astype(np.float32)
    mean_r = all_X_r[tr_mask].mean(0).astype(np.float32)
    std_r  = (all_X_r[tr_mask].std(0) + 1e-9).astype(np.float32)
    Xq = ((all_X_q - mean_q) / std_q).astype(np.float32)
    Xr = ((all_X_r - mean_r) / std_r).astype(np.float32)

    # ── duration-weighted val helper ──────────────────────────────────────────
    # Each segment weighted by duration (seconds): longer chord = more evidence.
    # Reports both unweighted (record-count) and weighted (duration) accuracy.
    val_dur   = all_dur[val_mask]
    val_dur_w = val_dur / val_dur.sum()   # normalized weights for val

    def _eval(qm, rm, cm):
        qm.eval(); rm.eval(); cm.eval()
        Xq_v = torch.tensor(Xq[val_mask], device=device)
        Xr_v = torch.tensor(Xr[val_mask], device=device)
        with torch.no_grad():
            pq = qm(Xq_v).argmax(1).cpu().numpy()
            pr = rm(Xr_v).argmax(1).cpu().numpy()
            pc = cm(Xr_v).argmax(1).cpu().numpy()
        qm.train(); rm.train(); cm.train()
        yq_v = all_y_q[val_mask]; yr_v = all_y_r[val_mask]; yc_v = all_y_chord[val_mask]
        # Duration-weighted accuracy (each correct prediction weighted by segment length)
        q_wacc = float(((pq == yq_v) * val_dur_w).sum())
        r_wacc = float(((pr == yr_v) * val_dur_w).sum())
        c_wacc = float(((pc == yc_v) * val_dur_w).sum())
        cq_wacc= float((((pc % 3) == yq_v) * val_dur_w).sum())
        cr_wacc= float((((pc //3) == yr_v) * val_dur_w).sum())
        return q_wacc, r_wacc, c_wacc, cq_wacc, cr_wacc

    def _per_class(qm, cm):
        qm.eval(); cm.eval()
        with torch.no_grad():
            pq = qm(torch.tensor(Xq[val_mask], device=device)).argmax(1).cpu().numpy()
            pc = cm(torch.tensor(Xr[val_mask], device=device)).argmax(1).cpu().numpy()
        qm.train(); cm.train()
        yq_v = all_y_q[val_mask]
        print("  quality head (duration-weighted):", flush=True)
        for qi, qn in enumerate(["maj", "min", "dom"]):
            mask = yq_v == qi
            if mask.sum():
                w = val_dur[mask] / val_dur[mask].sum()
                wacc = float(((pq[mask] == qi) * w).sum())
                print(f"    {qn}: n={mask.sum()} dur_acc={wacc:.3f}", flush=True)
        print("  chord head → quality margin (duration-weighted):", flush=True)
        for qi, qn in enumerate(["maj", "min", "dom"]):
            mask = yq_v == qi
            if mask.sum():
                w = val_dur[mask] / val_dur[mask].sum()
                wacc = float((((pc[mask] % 3) == qi) * w).sum())
                print(f"    {qn}: n={mask.sum()} dur_acc={wacc:.3f}", flush=True)

    def _run_config(h1_: int, h2_: int, lr_: float, epochs_: int, label: str) -> dict:
        """Train one hyperparameter configuration, return val metrics."""
        qm_ = make_mlp(108, 3,  h1_, h2_).to(device)
        rm_ = make_mlp(60,  12, h1_, h2_).to(device)
        cm_ = make_mlp(60,  36, h1_, h2_).to(device)

        def _w(counts, n_cls):
            w = 1.0 / (counts + 1); w = w / w.sum() * n_cls
            return torch.tensor(w, dtype=torch.float32, device=device)
        q_counts  = np.bincount(all_y_q[tr_mask],     minlength=3).astype(float)
        r_counts  = np.bincount(all_y_r[tr_mask],     minlength=12).astype(float)
        ch_counts = np.bincount(all_y_chord[tr_mask], minlength=36).astype(float)
        ql = nn.CrossEntropyLoss(weight=_w(q_counts,  3),  reduction="none")
        rl = nn.CrossEntropyLoss(weight=_w(r_counts,  12), reduction="none")
        cl = nn.CrossEntropyLoss(weight=_w(ch_counts, 36), reduction="none")

        q_opt_  = torch.optim.AdamW(qm_.parameters(), lr=lr_, weight_decay=1e-4)
        r_opt_  = torch.optim.AdamW(rm_.parameters(), lr=lr_, weight_decay=1e-4)
        ch_opt_ = torch.optim.AdamW(cm_.parameters(), lr=lr_, weight_decay=1e-4)
        for sched_cls in [torch.optim.lr_scheduler.CosineAnnealingLR]:
            q_sc  = sched_cls(q_opt_,  T_max=epochs_)
            r_sc  = sched_cls(r_opt_,  T_max=epochs_)
            ch_sc = sched_cls(ch_opt_, T_max=epochs_)

        # Duration weights for training (normalized per-sample)
        tr_dur_w = torch.tensor(
            all_dur[tr_mask] / all_dur[tr_mask].sum(), dtype=torch.float32, device=device
        )
        Xq_tr_ = torch.tensor(Xq[tr_mask],           dtype=torch.float32, device=device)
        Xr_tr_ = torch.tensor(Xr[tr_mask],           dtype=torch.float32, device=device)
        yq_tr_ = torch.tensor(all_y_q[tr_mask],      dtype=torch.long,    device=device)
        yr_tr_ = torch.tensor(all_y_r[tr_mask],      dtype=torch.long,    device=device)
        ych_tr_= torch.tensor(all_y_chord[tr_mask],  dtype=torch.long,    device=device)
        n_tr_  = len(Xq_tr_)

        # global_best tracks which config saved the file — needed to load it back
        # for _per_class with the right architecture
        best_qv = 0.0
        for ep in range(epochs_):
            qm_.train(); rm_.train(); cm_.train()
            perm = torch.randperm(n_tr_, device=device)
            for i in range(0, n_tr_, batch):
                idx = perm[i:i + batch]
                w_b = tr_dur_w[idx]; w_b = w_b / w_b.sum()  # re-normalize per batch

                q_opt_.zero_grad()
                (ql(qm_(Xq_tr_[idx]), yq_tr_[idx]) * w_b * len(idx)).sum().backward()
                q_opt_.step()

                r_opt_.zero_grad()
                (rl(rm_(Xr_tr_[idx]), yr_tr_[idx]) * w_b * len(idx)).sum().backward()
                r_opt_.step()

                ch_opt_.zero_grad()
                (cl(cm_(Xr_tr_[idx]), ych_tr_[idx]) * w_b * len(idx)).sum().backward()
                ch_opt_.step()

            q_sc.step(); r_sc.step(); ch_sc.step()

            q_v, r_v, c_v, cq_v, cr_v = _eval(qm_, rm_, cm_)
            print(f"  [{label}] ep {ep+1:2d}/{epochs_}  "
                  f"q={q_v:.3f}  r={r_v:.3f}  chord={c_v:.3f}  →q={cq_v:.3f}", flush=True)

            if q_v > best_qv:
                best_qv = q_v
                _save(out_path, qm_, rm_, cm_, mean_q, std_q, mean_r, std_r)
                print(f"    → best {q_v:.3f}, saved to {out_path}", flush=True)
                # Record which architecture is now in the file
                global_best[0] = {"h1": h1_, "h2": h2_, "q_val": best_qv, "label": label}

        return {"label": label, "h1": h1_, "h2": h2_, "lr": lr_,
                "q_val": best_qv,   # return best (not final) val for fair comparison
                "r_val": r_v, "chord_val": c_v, "chord_q": cq_v}

    # ── hyperparameter sweep ──────────────────────────────────────────────────
    # All configs share the same standardization and train/val split.
    # global_best[0] tracks which config's checkpoint is currently in out_path.
    global_best: list[dict] = [{"h1": 32, "h2": 16, "q_val": 0.0, "label": "?"}]

    SWEEP = [
        # (h1,  h2,  lr,    epochs, label)
        (32,  16,  3e-4, epochs, "tiny  32-16"),
        (64,  32,  3e-4, epochs, "small 64-32"),
        (128, 64,  3e-4, epochs, "std  128-64"),
        (64,  32,  1e-3, epochs, "small 64-32 lr=1e-3"),
    ]

    print(f"\n=== Hyperparameter sweep ({len(SWEEP)} configs) ===", flush=True)
    results = []
    for h1_, h2_, lr_, ep_, lbl in SWEEP:
        print(f"\n--- {lbl} ---", flush=True)
        res = _run_config(h1_, h2_, lr_, ep_, lbl)
        results.append(res)

    print("\n=== Sweep results (duration-weighted val accuracy, best epoch) ===", flush=True)
    print(f"  {'config':<22}  q_val   r_val  chord_val  chord→q", flush=True)
    for r in sorted(results, key=lambda x: -x["q_val"]):
        print(f"  {r['label']:<22}  {r['q_val']:.3f}   {r['r_val']:.3f}   "
              f"{r['chord_val']:.3f}      {r['chord_q']:.3f}", flush=True)

    gb = global_best[0]
    print(f"\nBest config: {gb['label']}  q_val={gb['q_val']:.3f}", flush=True)
    print(f"Model saved to {out_path}", flush=True)

    # Per-class breakdown — use the architecture that actually saved the file
    print("\nPer-class breakdown (best checkpoint):", flush=True)
    qm_best = make_mlp(108, 3, gb["h1"], gb["h2"]).to(device)
    cm_best = make_mlp(60, 36, gb["h1"], gb["h2"]).to(device)
    _load_best(out_path, qm_best, device)
    _per_class(qm_best, cm_best)
    print(f"Model: {out_path}", flush=True)


def _save(path: Path, qm, rm, cm, mq, sq, mr, sr) -> None:
    import torch
    def _st(m): return {k: v.cpu().numpy() for k, v in m.state_dict().items()}
    np.savez(path,
             qualities=np.array(["maj", "min", "dom"]),
             qual_mean=mq, qual_std=sq, root_mean=mr, root_std=sr,
             qual_state =np.array(_st(qm), dtype=object),
             root_state =np.array(_st(rm), dtype=object),
             chord_state=np.array(_st(cm), dtype=object),  # 36-class exact chord head
             context_window=np.int32(0))


def _load_best(path: Path, model, device: str) -> None:
    import torch
    d = np.load(path, allow_pickle=True)
    state = d["qual_state"].item()
    model.load_state_dict({k: torch.tensor(v, device=device) for k, v in state.items()})


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--playlists",   nargs="+", default=ALL_PLAYLISTS)
    ap.add_argument("--n-per-song",  default=2,   type=int,
                    help="YouTube recordings per song (different covers)")
    ap.add_argument("--max-songs",   default=0,   type=int)
    ap.add_argument("--skip-extract",  action="store_true")
    ap.add_argument("--skip-train",    action="store_true")
    ap.add_argument("--force-extract", action="store_true")
    ap.add_argument("--purge-bad",     action="store_true",
                    help="Delete feat_cache with coverage<0.60 or mismatch>0.45, "
                         "blacklist their IDs, clear vid_cache so they re-search")
    ap.add_argument("--epochs",  default=20,  type=int)
    ap.add_argument("--lr",      default=3e-4, type=float)
    ap.add_argument("--batch",   default=128, type=int)
    ap.add_argument("--hidden1", default=64,  type=int)
    ap.add_argument("--hidden2", default=32,  type=int)
    ap.add_argument("--val-frac", default=0.15, type=float)
    ap.add_argument("--seed",    default=42,  type=int)
    ap.add_argument("--out",     default=REPO / "harmonia/models/yt_online.npz", type=Path)
    args = ap.parse_args()

    songs     = load_all_songs(args.playlists)
    vid_cache = load_vid_cache()
    n_found   = sum(1 for v in vid_cache.values() if v)
    n_cached  = len(list(FEAT_DIR.glob("*.npz"))) if FEAT_DIR.exists() else 0
    print(f"Songs: {len(songs)}  vid_cache: {len(vid_cache)} searched ({n_found} found)  "
          f"feat_cache: {n_cached} videos", flush=True)

    if args.purge_bad:
        print("\n=== Purging low-quality feat_cache ===", flush=True)
        phase_purge_bad(vid_cache)

    if not args.skip_extract:
        phase_extract(songs, vid_cache,
                      n_per_song=args.n_per_song,
                      max_songs=args.max_songs,
                      force=args.force_extract)

    if not args.skip_train:
        phase_train(songs, vid_cache,
                    epochs=args.epochs, lr=args.lr, batch=args.batch,
                    h1=args.hidden1, h2=args.hidden2,
                    val_frac=args.val_frac, seed=args.seed,
                    out_path=args.out)


if __name__ == "__main__":
    main()
