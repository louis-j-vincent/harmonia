#!/usr/bin/env python3
"""End-to-end benchmark on ChoCo audio partitions (Isophonics / Billboard / …).

ChoCo (smashub/choco, Nature Sci Data 2023) aggregates 18 chord sources into one
JAMS/Harte corpus. Its AUDIO partitions carry absolute-timestamp chord labels
(with inversions); `choco/meta.csv` supplies performer+title, so the same
duration-match sourcing + overlap scoring the JAAH benchmark uses applies
unchanged. This grows the benchmark suite from jazz-only (JAAH) to pop/rock
(Isophonics = Beatles/Queen/MJ, Billboard 1958-91) with external comparability
(BTC/ChordFormer report on Isophonics/Billboard).

Reuses build_jaah_benchmark's scorer/cards, build_jaah_corpus's sourcing, and
the Tinder template. Video ids pinned for reproducibility.

Setup: ChoCo cloned at data/cache/choco (git clone --depth 1 …/choco.git).

Usage:
    .venv/bin/python scripts/build_choco_benchmark.py --partition isophonics --max 6
    .venv/bin/python scripts/build_choco_benchmark.py --partition billboard --max 40
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import numpy as np

from harmonia.eval.accuracy_score import run_prediction          # noqa: E402
from harmonia.data.yt_chord_corpus import download_audio         # noqa: E402
from scripts.build_jaah_corpus import source_candidate           # noqa: E402
from scripts.build_jaah_benchmark import score_and_cards         # noqa: E402
from scripts.build_pred_tinder import encode_clip, _TEMPLATE      # noqa: E402

CHOCO = REPO / "data" / "cache" / "choco" / "partitions"
RS = REPO / "docs" / "research_sessions"
AUDIO_DIR = REPO / "data" / "cache" / "choco" / "audio"

LEAD, MAXLEN, FLOOR_GIB, CAP = 1.2, 12.0, 1.5, 25
# per-partition meta.csv column names (performer, title)
META_COLS = {
    "isophonics": ("file_performer", "file_title"),
    "billboard": ("track_performer", "track_title"),
    "uspop2002": ("file_performer", "file_title"),
    "robbie-williams": ("file_performer", "file_title"),
}


def load_meta(partition: str):
    """id -> (performer, title) from choco/meta.csv (whitespace-padded header)."""
    perf_col, title_col = META_COLS[partition]
    out = {}
    with open(CHOCO / partition / "choco" / "meta.csv", newline="") as fh:
        for row in csv.DictReader(fh):
            row = {k.strip(): (v.strip() if v else v) for k, v in row.items()}
            out[row["id"]] = (row.get(perf_col, ""), row.get(title_col, ""))
    return out


def load_choco_chords(jams_path: Path):
    """[(t0, t1, harte_label)] from the ChoCo chord annotation."""
    j = json.loads(jams_path.read_text())
    ch = [a for a in j["annotations"] if a["namespace"] == "chord"]
    rows = []
    for d in (ch[0]["data"] if ch else []):
        rows.append((float(d["time"]), float(d["time"] + d["duration"]), d["value"]))
    rows.sort()
    dur = j.get("file_metadata", {}).get("duration")
    return rows, dur


def free_gib():
    return shutil.disk_usage(REPO).free / 2**30


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--partition", default="isophonics", choices=list(META_COLS))
    ap.add_argument("--max", type=int, default=6)
    ap.add_argument("--cap-cards", type=int, default=CAP)
    ap.add_argument("--seed", type=int, default=42,
                    help="shuffle-sample seed so a --max subset spans artists")
    args = ap.parse_args(argv)

    part = args.partition
    jdir = CHOCO / part / "choco" / "jams"
    if not jdir.exists():
        print(f"ChoCo partition not found: {jdir}")
        return 1
    meta = load_meta(part)
    out_html = RS / f"choco_{part}_tinder_2026-07-28.html"
    scores_path = RS / f"choco_{part}_benchmark_scores.json"
    pins_path = RS / f"choco_{part}_source_pins.json"
    pins = json.loads(pins_path.read_text()) if pins_path.exists() else {}

    ids = sorted(meta)
    if args.max and args.max < len(ids):
        import random
        random.Random(args.seed).shuffle(ids)
        ids = sorted(ids[:args.max])
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    scores, cards, excluded = [], [], []
    tmp = Path(tempfile.mkdtemp(prefix="chococlip_"))
    t_start = time.time()
    for cid in ids:
        jp = jdir / f"{cid}.jams"
        if not jp.exists():
            continue
        performer, title = meta[cid]
        rows, dur = load_choco_chords(jp)
        if not rows:
            continue
        target = dur or rows[-1][1]
        print(f"[{cid}] {performer} - {title}  (target {target:.0f}s, {len(rows)} chords)",
              flush=True)
        if free_gib() < FLOOR_GIB:
            print("!! disk floor, stopping", flush=True)
            break
        if cid in pins:
            vid = pins[cid]
            print(f"    pinned {vid}", flush=True)
        else:
            query = f"{performer} {title}".strip()
            best, cands = source_candidate(performer, title, target)
            if not best:
                print(f"    no duration match ({len(cands)} cands) -> EXCLUDE", flush=True)
                excluded.append(cid)
                continue
            vid = best[0]
            print(f"    candidate {vid} dur={best[2]:.0f} diff={best[1]:.1f}", flush=True)
            pins[cid] = vid
            pins_path.write_text(json.dumps(pins, indent=1))
        try:
            wav = download_audio(vid, AUDIO_DIR)
        except Exception as e:
            print(f"    download failed: {e}", flush=True)
            excluded.append(cid)
            continue
        try:
            t0 = time.time()
            pred = run_prediction(wav)
            n0 = len(cards)
            sc = score_and_cards(cid, rows, pred, wav, cards)
            sc["title"] = f"{performer} — {title}"
            scores.append(sc)
            song_cards = cards[n0:]
            if args.cap_cards and len(song_cards) > args.cap_cards:
                step = len(song_cards) / args.cap_cards
                cards[n0:] = [song_cards[min(len(song_cards) - 1, round(i * step))]
                              for i in range(args.cap_cards)]
            for k, c in enumerate(cards[n0:]):
                c["title"] = f"{performer} — {title}"
                a = max(0.0, c["t0"] - LEAD)
                b = min(c["t1"] + 0.4, a + MAXLEN)
                mp3 = tmp / f"{cid}_{k}.mp3"
                encode_clip(Path(wav), mp3, a, b - a)
                c["b64"] = base64.b64encode(mp3.read_bytes()).decode("ascii")
                c["region_a"], c["region_b"] = round(c["t0"] - a, 2), round(c["t1"] - a, 2)
                c["id"] = f"{cid}:{c['t0']}"
                mp3.unlink()
            print(f"    root={sc['root_acc']} family={sc['family_acc']} "
                  f"nc={sc['nc_acc']} ({len(cards)-n0} cards, {time.time()-t0:.0f}s)",
                  flush=True)
        finally:
            Path(wav).unlink(missing_ok=True)

    shutil.rmtree(tmp, ignore_errors=True)
    cards = [c for c in cards if "b64" in c]
    for c in cards:
        c.pop("wav", None)

    scores_path.write_text(json.dumps({"partition": part, "scores": scores,
                                       "excluded": excluded, "n_cards": len(cards)}, indent=1))
    html = _TEMPLATE.replace("__DATA__", json.dumps(cards, separators=(",", ":"))) \
                    .replace("__N__", str(len(cards))).replace("__LEVEL__", "family") \
                    .replace("__APIBASE__", f"/api/choco_{part}") \
                    .replace("__STOREKEY__", f"harmonia_choco_{part}_tinder")
    out_html.write_text(html)

    rr = [s["root_acc"] for s in scores if s["root_acc"] is not None]
    ff = [s["family_acc"] for s in scores if s["family_acc"] is not None]
    print(f"\n=== ChoCo/{part} end-to-end (shipped pipeline vs absolute-timestamp GT) ===")
    for s in sorted(scores, key=lambda s: -(s["root_acc"] or 0)):
        print(f"  {s['title'][:34]:<34} root={s['root_acc']:.3f} family={s['family_acc']:.3f}")
    if rr:
        print(f"  MEAN root={np.mean(rr):.3f} family={np.mean(ff):.3f} (n={len(rr)})")
    print(f"  excluded: {excluded}")
    print(f"\nwrote {out_html.relative_to(REPO)} ({len(cards)} cards), "
          f"{scores_path.relative_to(REPO)}")
    print(f"total {time.time()-t_start:.0f}s, disk {free_gib():.1f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
