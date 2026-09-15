#!/usr/bin/env python3
"""End-to-end JAAH benchmark: score the SHIPPED pipeline on real jazz audio with
ABSOLUTE-TIMESTAMP ground truth — no alignment, no circularity.

JAAH (.lab) gives Harte chord labels on the audio clock. We source a
duration-matched YouTube candidate (the same gate build_jaah_corpus.py's pilot
validated), run infer_chords_v1, and score prediction-vs-GT by direct overlap —
exactly the brick0 contract, but on trustworthy external GT instead of our
hand-built frozen reference.

Scoring is at JAAH's own granularity: root pitch-class and 7-way quality FAMILY
(maj/min/dom/hdim/dim/aug/sus via parse_jaah), duration-weighted over the GT
span. Both GT and prediction pass through the SAME parse_jaah, so it's
apples-to-apples (same convention the NNLS-24 JAAH CV baselines use).

Also emits Tinder disagreement cards (same tool as brick0) into a JAAH page so
Louis can ear-check the real-jazz disagreements with the cylinder picker.

Disk discipline: the downloaded WAV is deleted right after clips are cut.

Usage:
    .venv/bin/python scripts/build_jaah_benchmark.py --songs airegin,st_thomas
    .venv/bin/python scripts/build_jaah_benchmark.py            # default set
    .venv/bin/python scripts/build_jaah_benchmark.py --max 10
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
from harmonia.data.yt_chord_corpus import download_audio         # noqa: E402
from scripts.build_jaah_corpus import (                          # noqa: E402
    source_candidate, load_lab, ann_meta, mb_length_ms, parse_jaah, LABS_DIR,
)
from scripts.build_pred_tinder import (                          # noqa: E402
    encode_clip, _TEMPLATE, PC,
)

OUT_HTML = REPO / "docs" / "research_sessions" / "jaah_tinder_2026-07-27.html"
SCORES = REPO / "docs" / "research_sessions" / "jaah_benchmark_scores.json"
PINS = REPO / "docs" / "research_sessions" / "jaah_source_pins.json"
AUDIO_DIR = REPO / "data" / "cache" / "jaah" / "audio"

LEAD, MAXLEN, FLOOR_GIB = 1.2, 12.0, 1.5

# 7-family -> (picker family_id, seventh_token, ext_token) for card seeds
FAMILY_TO_PICKER = {
    "maj": ("maj", "", ""), "min": ("min", "-", "-"), "dom": ("maj", "7", "7"),
    "hdim": ("dim", "h7", "h7"), "dim": ("dim", "o", "o"),
    "aug": ("aug", "+", "+"), "sus": ("sus", "sus4", "sus4"),
}
# vocab quality (pipeline output) -> 7-family, so pred goes through the same lens
DEFAULT_SET = [
    "airegin", "st_thomas", "boplicity", "misterioso", "daahoud",
    "blue_horizon", "west_end_blues", "lady_bird", "four_brothers",
    "lester_leaps_in", "moten_swing", "summertime",
]


def bass_pc_of(label: str, root_pc: int | None) -> int | None:
    if root_pc is None:
        return None
    if "/" in label:
        from harmonia.data.corpus_schema import sounding_bass_pc
        try:
            return sounding_bass_pc(label, root_pc)
        except Exception:
            return root_pc
    return root_pc


def gt_seed(label: str):
    root, fam, _ = parse_jaah(label)
    if root is None:
        return None
    f, s, e = FAMILY_TO_PICKER.get(fam, ("maj", "", ""))
    return {"root": root, "fam": f, "sev": s, "ext": e,
            "bass": bass_pc_of(label, root)}


def pred_seed(chord):
    if chord.is_nc:
        return None
    root, fam, _ = parse_jaah(chord.label)
    if root is None:
        return None
    f, s, e = FAMILY_TO_PICKER.get(fam, ("maj", "", ""))
    return {"root": root, "fam": f, "sev": s, "ext": e,
            "bass": chord.bass_pc if chord.bass_pc is not None else root}


def fam_of(label: str):
    r, f, _ = parse_jaah(label)
    return r, f


def score_and_cards(slug, gt_rows, pred, wav, cards_out):
    """Duration-weighted root/family score over the GT span + disagreement cards.

    gt_rows: [(t0,t1,label)]; pred: list[accuracy_score.Chord]."""
    lo = min(r[0] for r in gt_rows)
    hi = max(r[1] for r in gt_rows)
    # union grid
    pts = {lo, hi}
    for t0, t1, _ in gt_rows:
        if lo < t0 < hi:
            pts.add(t0)
        if lo < t1 < hi:
            pts.add(t1)
    for c in pred:
        if lo < c.t0 < hi:
            pts.add(c.t0)
        if lo < c.t1 < hi:
            pts.add(c.t1)
    bps = sorted(pts)

    def gt_at(t):
        for t0, t1, lab in gt_rows:
            if t0 <= t < t1:
                return lab
        return "N"

    def pred_at(t):
        for c in pred:
            if c.t0 <= t < c.t1:
                return c
        return None

    dur_chord = root_ok = fam_ok = 0.0    # over non-N GT
    nc_dur = nc_ok = 0.0                   # GT==N spans
    for a, b in zip(bps[:-1], bps[1:]):
        d = b - a
        if d <= 0:
            continue
        mid = 0.5 * (a + b)
        glab = gt_at(mid)
        gr, gf = fam_of(glab)
        pc = pred_at(mid)
        plab = pc.label if pc else "N"
        pr, pf = fam_of(plab)
        if gr is None:                    # GT no-chord
            nc_dur += d
            if pr is None:
                nc_ok += d
            continue
        dur_chord += d
        if pr is not None and gr == pr:
            root_ok += d
            if gf == pf:
                fam_ok += d

    # cards: one per GT chord row where the dominant pred disagrees (family lvl)
    for t0, t1, glab in gt_rows:
        gr, gf = fam_of(glab)
        # dominant pred over [t0,t1]
        best, bov = None, -1.0
        for c in pred:
            ov = min(t1, c.t1) - max(t0, c.t0)
            if ov > bov:
                best, bov = c, ov
        plab = best.label if best else "N"
        pr, pf = fam_of(plab)
        rm = (gr is not None and pr is not None and gr == pr) or (gr is None and pr is None)
        fm = rm and (gf == pf)
        if fm:
            continue
        cards_out.append({
            "song": slug, "title": slug.replace("_", " "),
            "t0": round(t0, 2), "t1": round(t1, 2),
            "gt": glab, "pred": plab, "root_ok": rm, "family_ok": fm,
            "gt_seed": gt_seed(glab), "pred_seed": pred_seed(best) if best else None,
            "wav": str(wav),
        })

    return {
        "slug": slug,
        "gt_span_s": round(hi - lo, 1),
        "n_gt_chords": len(gt_rows),
        "root_acc": round(root_ok / dur_chord, 4) if dur_chord else None,
        "family_acc": round(fam_ok / dur_chord, 4) if dur_chord else None,
        "nc_acc": round(nc_ok / nc_dur, 4) if nc_dur else None,
        "chord_dur_s": round(dur_chord, 1),
    }


def free_gib():
    return shutil.disk_usage(REPO).free / 2**30


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--songs", default="")
    ap.add_argument("--max", type=int, default=0)
    ap.add_argument("--all-gated", action="store_true",
                    help="use the 47 slugs already gated-in in jaah_bp48.npz")
    ap.add_argument("--cap-cards", type=int, default=25,
                    help="max disagreement cards emitted per song (sampled "
                         "evenly across the tune); scoring still uses ALL chords")
    args = ap.parse_args(argv)

    if args.all_gated:
        import numpy as _np
        corpus = REPO / "data/cache/jaah/jaah_bp48.npz"
        sids = _np.load(corpus, allow_pickle=True)["song_id"]
        slugs = sorted({s.replace("jaah_", "", 1) for s in map(str, sids)})
    else:
        slugs = ([s.strip() for s in args.songs.split(",") if s.strip()]
                 or DEFAULT_SET)
    if args.max:
        slugs = slugs[:args.max]
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)

    scores, cards = [], []
    excluded = []
    # slug -> chosen youtube id, so re-runs use the SAME audio (yt-dlp search is
    # nondeterministic; a benchmark must be reproducible). First good source is
    # pinned; delete a line to re-source that song.
    pins = json.loads(PINS.read_text()) if PINS.exists() else {}
    tmp = Path(tempfile.mkdtemp(prefix="jaahclip_"))
    t_start = time.time()
    for slug in slugs:
        lab = LABS_DIR / f"{slug}.lab"
        if not lab.exists():
            print(f"[{slug}] no .lab, skip", flush=True)
            continue
        if free_gib() < FLOOR_GIB:
            print(f"!! disk floor, stopping at {slug}", flush=True)
            break
        rows = [(t0, t1, l) for t0, t1, l in load_lab(lab)]
        try:
            artist, title, mbid, jdur = ann_meta(slug)
        except Exception as e:
            print(f"[{slug}] meta fail {e}", flush=True)
            continue
        mb_ms, _ = mb_length_ms(mbid) if mbid else (None, None)
        target = (mb_ms / 1000.0) if mb_ms else jdur
        print(f"[{slug}] {artist} - {title}  (target {target}s, {len(rows)} chords)", flush=True)
        if slug in pins:
            vid, diff = pins[slug], 0.0
            print(f"    pinned {vid}", flush=True)
        else:
            best, cands = source_candidate(artist, title, target)
            if not best:
                print(f"    no duration match ({len(cands)} cands) -> EXCLUDE", flush=True)
                excluded.append(slug)
                continue
            vid, diff = best[0], best[1]
            print(f"    candidate {vid} dur={best[2]:.0f} diff={diff:.1f}", flush=True)
            pins[slug] = vid
            PINS.write_text(json.dumps(pins, indent=1))
        try:
            wav = download_audio(vid, AUDIO_DIR)
        except Exception as e:
            print(f"    download failed: {e}", flush=True)
            excluded.append(slug)
            continue
        try:
            t0 = time.time()
            pred = run_prediction(wav)
            n0 = len(cards)
            sc = score_and_cards(slug, rows, pred, wav, cards)
            scores.append(sc)
            # cap this song's cards BEFORE encoding (sample evenly across the
            # tune) — scoring above already used every chord; cards are only for
            # ear-spot-checking, so a giant page + thousands of clips is waste.
            song_cards = cards[n0:]
            if args.cap_cards and len(song_cards) > args.cap_cards:
                step = len(song_cards) / args.cap_cards
                keep = [song_cards[min(len(song_cards) - 1, round(i * step))]
                        for i in range(args.cap_cards)]
                cards[n0:] = keep
            # cut clips for THIS song's cards now, while the wav still exists
            for k, c in enumerate(cards[n0:]):
                a = max(0.0, c["t0"] - LEAD)
                b = min(c["t1"] + 0.4, a + MAXLEN)
                mp3 = tmp / f"{slug}_{k}.mp3"
                encode_clip(Path(wav), mp3, a, b - a)
                c["b64"] = base64.b64encode(mp3.read_bytes()).decode("ascii")
                c["region_a"], c["region_b"] = round(c["t0"] - a, 2), round(c["t1"] - a, 2)
                c["id"] = f"{slug}:{c['t0']}"
                mp3.unlink()
            print(f"    root={sc['root_acc']} family={sc['family_acc']} "
                  f"nc={sc['nc_acc']}  ({len(cards)-n0} cards, "
                  f"{time.time()-t0:.0f}s)", flush=True)
        finally:
            Path(wav).unlink(missing_ok=True)

    shutil.rmtree(tmp, ignore_errors=True)
    cards = [c for c in cards if "b64" in c]
    for c in cards:
        c.pop("wav", None)

    SCORES.write_text(json.dumps({"scores": scores, "excluded": excluded,
                                  "n_cards": len(cards)}, indent=1))
    html = _TEMPLATE.replace("__DATA__", json.dumps(cards, separators=(",", ":"))) \
                    .replace("__N__", str(len(cards))).replace("__LEVEL__", "family") \
                    .replace("__APIBASE__", "/api/jaah") \
                    .replace("__STOREKEY__", "harmonia_jaah_tinder")
    OUT_HTML.write_text(html)

    # summary
    rr = [s["root_acc"] for s in scores if s["root_acc"] is not None]
    ff = [s["family_acc"] for s in scores if s["family_acc"] is not None]
    print("\n=== JAAH end-to-end (shipped pipeline vs absolute-timestamp GT) ===")
    print(f"  {'slug':<20} {'root':>6} {'family':>7} {'nc':>6} {'span':>6} cards")
    for s in scores:
        nc = f"{s['nc_acc']:.2f}" if s['nc_acc'] is not None else "  - "
        print(f"  {s['slug']:<20} {s['root_acc']:>6.3f} {s['family_acc']:>7.3f} "
              f"{nc:>6} {s['gt_span_s']:>6.0f}")
    if rr:
        print(f"  {'MEAN':<20} {np.mean(rr):>6.3f} {np.mean(ff):>7.3f}")
    print(f"  excluded (no source): {excluded}")
    print(f"\nwrote {OUT_HTML.relative_to(REPO)} ({len(cards)} cards), "
          f"{SCORES.relative_to(REPO)}")
    print(f"total {time.time()-t_start:.0f}s, disk {free_gib():.1f} GiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
