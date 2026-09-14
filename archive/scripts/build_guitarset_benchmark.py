#!/usr/bin/env python3
"""End-to-end GuitarSet benchmark: score the SHIPPED pipeline on GuitarSet with
BUNDLED audio + absolute-timestamp chord GT — ZERO alignment risk (audio and GT
ship together; nothing to source, nothing to duration-match).

GuitarSet (Zenodo 3371780, CC-BY-4.0): 360 ~30 s excerpts, 6 players × 5 styles
(BN/Funk/SS/Rock/Jazz) × comp/solo. We use the **comping** takes only (the solo
takes are monophonic lead lines — no chords to detect from the mix). GT = the
`chord[1]` "performed" annotation (semi-automatic transcription, manually
verified) reduced to root + 7-family via parse_jaah (same convention as the
JAAH benchmark). Score by direct overlap; emit Tinder disagreement cards.

Reuses the JAAH benchmark's scorer/card logic and the Tinder template verbatim.
Contrast with JAAH: clean single-instrument audio + trusted alignment, so this
isolates the pipeline's chord model from every sourcing/alignment confound.

Setup: annotation.zip + audio_mono-mic.zip unzipped under
data/cache/guitarset/{annotation,audio} (this script does NOT download).

Usage:
    .venv/bin/python scripts/build_guitarset_benchmark.py            # sampled subset
    .venv/bin/python scripts/build_guitarset_benchmark.py --all      # all 180 comp
    .venv/bin/python scripts/build_guitarset_benchmark.py --max 40
"""
from __future__ import annotations

import argparse
import base64
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
from scripts.build_jaah_corpus import parse_jaah                 # noqa: E402
from scripts.build_jaah_benchmark import (                       # noqa: E402
    score_and_cards, gt_seed, pred_seed, FAMILY_TO_PICKER,
)
from scripts.build_pred_tinder import encode_clip, _TEMPLATE, PC  # noqa: E402

GS = REPO / "data" / "cache" / "guitarset"
ANN = GS / "annotation"
AUD = GS / "audio"
OUT_HTML = REPO / "docs" / "research_sessions" / "guitarset_tinder_2026-07-28.html"
SCORES = REPO / "docs" / "research_sessions" / "guitarset_benchmark_scores.json"

LEAD, MAXLEN, FLOOR_GIB, CAP = 1.2, 12.0, 1.5, 25
FAM_TOKEN = {"maj": "maj", "min": "min", "dom": "7", "hdim": "hdim7",
             "dim": "dim", "aug": "aug", "sus": "sus4"}


def clean_label(raw: str) -> str:
    """GuitarSet 'performed' labels carry rich Harte extensions/inversions
    (e.g. D#:min7(4,*5)/4) — reduce to a readable root+family token for the card
    (scoring uses parse_jaah's family regardless)."""
    r, f, _ = parse_jaah(raw)
    if r is None:
        return "N"
    return f"{PC[r]}:{FAM_TOKEN.get(f, f)}"


def load_jams_chords(jams_path: Path):
    """[(t0, t1, base_label)] from the performed chord annotation, bass dropped
    (GuitarSet notates bass as a scale degree, not a pitch class — parse_jaah
    drops it and we score root+family)."""
    j = json.loads(jams_path.read_text())
    chords = [a for a in j["annotations"] if a["namespace"] == "chord"]
    ann = chords[-1] if chords else None      # [-1] = performed (verified)
    rows = []
    for d in (ann["data"] if ann else []):
        base = d["value"].split("/")[0]
        rows.append((float(d["time"]), float(d["time"] + d["duration"]), base))
    rows.sort()
    return rows


def audio_for(stem: str) -> Path | None:
    for cand in (AUD / f"{stem}_mic.wav", AUD / f"{stem}.wav"):
        if cand.exists():
            return cand
    hits = list(AUD.glob(f"{stem}*.wav"))
    return hits[0] if hits else None


def pick_subset(comp_jams, sample: bool):
    if not sample:
        return comp_jams
    # one per (player, style) cell so the subset spans all 6 players × 5 styles
    seen, out = set(), []
    for p in comp_jams:
        stem = p.stem                      # e.g. 03_SS1-100-C#_comp
        player = stem.split("_")[0]
        style = stem.split("_")[1].split("-")[0].rstrip("0123456789")
        key = (player, style)
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def free_gib():
    return shutil.disk_usage(REPO).free / 2**30


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="all 180 comp takes")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--cap-cards", type=int, default=CAP)
    args = ap.parse_args(argv)

    if not AUD.exists():
        print(f"audio not unzipped at {AUD} — unzip audio_mono-mic.zip first.")
        return 1
    comp = sorted(ANN.glob("*_comp.jams"))
    jamses = pick_subset(comp, sample=not args.all)
    if args.max:
        jamses = jamses[:args.max]
    print(f"{len(jamses)} comp excerpts (of {len(comp)}); disk {free_gib():.1f} GiB",
          flush=True)

    scores, cards = [], []
    tmp = Path(tempfile.mkdtemp(prefix="gsclip_"))
    t_start = time.time()
    for jp in jamses:
        stem = jp.stem
        wav = audio_for(stem)
        if wav is None:
            print(f"[{stem}] no audio, skip", flush=True)
            continue
        if free_gib() < FLOOR_GIB:
            print("!! disk floor, stopping", flush=True)
            break
        raw_rows = load_jams_chords(jp)
        if not raw_rows:
            continue
        rows = [(t0, t1, clean_label(lab)) for t0, t1, lab in raw_rows]
        t0 = time.time()
        pred = run_prediction(wav)
        n0 = len(cards)
        sc = score_and_cards(stem, rows, pred, wav, cards)
        scores.append(sc)
        song_cards = cards[n0:]
        if args.cap_cards and len(song_cards) > args.cap_cards:
            step = len(song_cards) / args.cap_cards
            cards[n0:] = [song_cards[min(len(song_cards) - 1, round(i * step))]
                          for i in range(args.cap_cards)]
        for k, c in enumerate(cards[n0:]):
            c["title"] = stem
            a = max(0.0, c["t0"] - LEAD)
            b = min(c["t1"] + 0.4, a + MAXLEN)
            mp3 = tmp / f"{stem}_{k}.mp3"
            encode_clip(Path(wav), mp3, a, b - a)
            c["b64"] = base64.b64encode(mp3.read_bytes()).decode("ascii")
            c["region_a"], c["region_b"] = round(c["t0"] - a, 2), round(c["t1"] - a, 2)
            c["id"] = f"{stem}:{c['t0']}"
            mp3.unlink()
        print(f"[{stem}] root={sc['root_acc']} family={sc['family_acc']} "
              f"nc={sc['nc_acc']} ({len(cards)-n0} cards, {time.time()-t0:.0f}s)",
              flush=True)

    shutil.rmtree(tmp, ignore_errors=True)
    cards = [c for c in cards if "b64" in c]
    for c in cards:
        c.pop("wav", None)

    SCORES.write_text(json.dumps({"scores": scores, "n_cards": len(cards)}, indent=1))
    html = _TEMPLATE.replace("__DATA__", json.dumps(cards, separators=(",", ":"))) \
                    .replace("__N__", str(len(cards))).replace("__LEVEL__", "family") \
                    .replace("__APIBASE__", "/api/guitarset") \
                    .replace("__STOREKEY__", "harmonia_guitarset_tinder")
    OUT_HTML.write_text(html)

    rr = [s["root_acc"] for s in scores if s["root_acc"] is not None]
    ff = [s["family_acc"] for s in scores if s["family_acc"] is not None]
    print("\n=== GuitarSet end-to-end (shipped pipeline vs bundled GT) ===")
    for s in sorted(scores, key=lambda s: -(s["root_acc"] or 0)):
        print(f"  {s['slug']:<26} root={s['root_acc']:.3f} family={s['family_acc']:.3f} "
              f"nc={s['nc_acc'] if s['nc_acc'] is not None else '-'}")
    if rr:
        print(f"  MEAN  root={np.mean(rr):.3f}  family={np.mean(ff):.3f}  (n={len(rr)})")
    print(f"\nwrote {OUT_HTML.relative_to(REPO)} ({len(cards)} cards), "
          f"{SCORES.relative_to(REPO)}")
    print(f"total {time.time()-t_start:.0f}s, disk {free_gib():.1f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
