"""scripts/check_span_rescore_sanity.py — MANDATORY premise check before
building on span_rescore.py's acoustic term (CLAUDE.md process rule #2).

Takes a baked chart's OWN spans (docs/plots/inferred_*.html, the ``const P``
blob) and computes the per-span acoustic argmax from BOTH backends —
music-x-lab frame posteriors (PRIMARY, per Louis's 2026-07-30 redirect: "our
state-of-the-art chord detection is musx") and the NNLS-24 heads (FALLBACK) —
then reports agreement with the displayed root/quality for each. Expects
HIGH-BUT-NOT-PERFECT agreement (the displayed chart came from a related but
not identical scorer — musx-assisted, post-passed, Occam'd, etc.) and expects
musx agreement >= nnls agreement (the chart IS musx-derived by default).

If agreement is near-chance, STOP: the frame-time convention, the C-first vs
A-first pitch-class order, or the bass/treble half order is probably wrong —
this is this project's #1 historical bug class (CLAUDE.md rule #1).

Song selection: needs a chart with (a) a resolvable video-id stem (the ACTUAL
nnls/musx cache key — see docs/plots/.yt_video_ids.json; ``_chart_audio_path``'s
own docs/audio/<slug>.m4a file is a DIFFERENT stem and would silently miss the
cache) and (b) that stem present in data/cache/musx_probs/ AND/OR
data/cache/nnls_infer/. Only 3/35 charted songs currently have a musx_probs
cache hit (most of that cache was populated by benchmark corpora, not the live
YouTube analyze path) — this script uses those 3, plus a broader nnls-only set
for context.
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from harmonia.models import nnls_features as nf  # noqa: E402
from harmonia.models import span_rescore as sr  # noqa: E402
from harmonia.models.musx_redecode import frame_posteriors  # noqa: E402
from harmonia.serving.analysis import _ireal_q_to_q5  # noqa: E402

PLOTS_DIR = REPO / "docs" / "plots"
VIDEO_IDS_FILE = PLOTS_DIR / ".yt_video_ids.json"
MUSX_CACHE = REPO / "data" / "cache" / "musx_probs"
NNLS_CACHE = REPO / "data" / "cache" / "nnls_infer"


def load_chart(html_path: Path) -> dict:
    txt = html_path.read_text()
    m = re.search(r"const P = (\{.*?\});\s*\n", txt, re.S)
    if not m:
        raise ValueError(f"no `const P = {{...}}` blob found in {html_path}")
    return json.loads(m.group(1))


def displayed_spans(P: dict) -> tuple[list[tuple[float, float]], list[tuple[int, int]]]:
    """Chart's own (t0,t1) spans + (root, qual5) labels, no-chord bars dropped."""
    spans, labels = [], []
    for c in P["chords"]:
        if c.get("nc"):
            continue
        t0, t1 = float(c["t0"]), float(c["t1"])
        if t1 <= t0:
            continue
        q_tok = c["lv"]["seventh"]["q"]     # the fine token; "family" collapses hdim->min
        q5 = _ireal_q_to_q5(q_tok)
        spans.append((t0, t1))
        labels.append((int(c["root"]) % 12, int(q5)))
    return spans, labels


def agreement(pred: list[tuple[int, int]], gt: list[tuple[int, int]]) -> dict:
    n = len(gt)
    root_ok = sum(1 for p, g in zip(pred, gt) if p[0] == g[0])
    qual_ok = sum(1 for p, g in zip(pred, gt) if p[1] == g[1])
    joint_ok = sum(1 for p, g in zip(pred, gt) if p == g)
    return {"n": n, "root_agree": root_ok / n, "qual_agree": qual_ok / n,
           "joint_agree": joint_ok / n}


def run_musx(video_id: str, spans) -> tuple[list[tuple[int, int]], np.ndarray] | None:
    cache = MUSX_CACHE / f"{video_id}.npz"
    if not cache.exists():
        return None
    audio_stub = REPO / f"{video_id}.m4a"          # never read: cache hits by stem only
    probs = frame_posteriors(audio_stub)
    pooled_triad, pooled_s7 = sr.pool_span_musx(probs, spans)
    logp, n_mass = sr.acoustic_logp_musx(pooled_triad, pooled_s7)
    pred = [sr.token_of(int(row.argmax())) for row in logp]
    return pred, n_mass


def run_nnls(video_id: str, spans) -> list[tuple[int, int]] | None:
    cache = NNLS_CACHE / f"{video_id}.npz"
    if not cache.exists():
        return None
    heads = nf.get_heads()
    if heads is None:
        return None
    audio_stub = REPO / f"{video_id}.m4a"
    arr, times = nf.extract_bothchroma(audio_stub)
    feat24 = sr.pool_span_features(arr, times, spans)
    logp = sr.acoustic_logp_nnls(feat24, heads)
    return [sr.token_of(int(row.argmax())) for row in logp]


def main():
    video_ids = json.loads(VIDEO_IDS_FILE.read_text())
    musx_stems = {p.stem for p in MUSX_CACHE.glob("*.npz")}
    nnls_stems = {p.stem for p in NNLS_CACHE.glob("*.npz")}

    both = [(fn, vid) for fn, vid in video_ids.items() if vid in musx_stems]
    nnls_only = [(fn, vid) for fn, vid in video_ids.items()
                if vid in nnls_stems and vid not in musx_stems]

    print(f"charts with a musx_probs cache hit: {len(both)}/{len(video_ids)}")
    print(f"charts with an nnls_infer cache hit: "
         f"{sum(1 for v in video_ids.values() if v in nnls_stems)}/{len(video_ids)}")
    print()

    rows = []
    for fn, vid in both:
        html = PLOTS_DIR / fn
        if not html.exists():
            continue
        P = load_chart(html)
        spans, gt = displayed_spans(P)
        if len(spans) < 8:
            continue
        musx_res = run_musx(vid, spans)
        nnls_res = run_nnls(vid, spans)
        if musx_res is None:
            continue
        musx_pred, n_mass = musx_res
        a_musx = agreement(musx_pred, gt)
        row = {"song": fn, "n_spans": len(spans), "musx": a_musx,
              "mean_n_mass": float(np.mean(n_mass))}
        if nnls_res is not None:
            row["nnls"] = agreement(nnls_res, gt)
        rows.append(row)
        print(f"=== {fn}  ({len(spans)} spans, video_id={vid}) ===")
        print(f"  musx_probs : root={a_musx['root_agree']:.3f}  "
             f"qual={a_musx['qual_agree']:.3f}  joint={a_musx['joint_agree']:.3f}"
             f"   (mean N-mass {np.mean(n_mass):.3f})")
        if nnls_res is not None:
            a_nnls = row["nnls"]
            print(f"  nnls_heads : root={a_nnls['root_agree']:.3f}  "
                 f"qual={a_nnls['qual_agree']:.3f}  joint={a_nnls['joint_agree']:.3f}")
        else:
            print("  nnls_heads : (no nnls_infer cache for this stem)")
        print()

    # broader nnls-only context (no musx_probs cache for these stems)
    print(f"--- nnls-only context (no musx_probs cache), first 6 of "
         f"{len(nnls_only)} ---")
    for fn, vid in nnls_only[:6]:
        html = PLOTS_DIR / fn
        if not html.exists():
            continue
        P = load_chart(html)
        spans, gt = displayed_spans(P)
        if len(spans) < 8:
            continue
        nnls_res = run_nnls(vid, spans)
        if nnls_res is None:
            continue
        a = agreement(nnls_res, gt)
        print(f"  {fn:60s} n={len(spans):4d}  root={a['root_agree']:.3f}  "
             f"qual={a['qual_agree']:.3f}  joint={a['joint_agree']:.3f}")

    if rows:
        mean_musx_root = np.mean([r["musx"]["root_agree"] for r in rows])
        mean_musx_joint = np.mean([r["musx"]["joint_agree"] for r in rows])
        print(f"\n=== summary over {len(rows)} musx-cache-hit songs ===")
        print(f"mean musx root agreement:  {mean_musx_root:.3f}")
        print(f"mean musx joint agreement: {mean_musx_joint:.3f}")
        with_nnls = [r for r in rows if "nnls" in r]
        if with_nnls:
            mean_nnls_root = np.mean([r["nnls"]["root_agree"] for r in with_nnls])
            mean_nnls_joint = np.mean([r["nnls"]["joint_agree"] for r in with_nnls])
            print(f"mean nnls root agreement:  {mean_nnls_root:.3f}  "
                 f"(n={len(with_nnls)} songs with both caches)")
            print(f"mean nnls joint agreement: {mean_nnls_joint:.3f}")
            verdict = ("musx >= nnls, as expected (chart is musx-derived)"
                      if mean_musx_root >= mean_nnls_root
                      else "*** musx < nnls — UNEXPECTED, diagnose before proceeding ***")
            print(f"verdict: {verdict}")


if __name__ == "__main__":
    main()
